"""Module for concentrating runoff from different sources."""

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from geb.geb_types import ArrayFloat32, TwoDArrayFloat32
from geb.module import Module
from geb.workflows import balance_check

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


def triangular_weights(peak_hour: float, lag_time_hours: int) -> ArrayFloat32:
    """Compute triangular weights for given peak lag time.

    Args:
        peak_hour: Peak lag time in hours.
        lag_time_hours: Maximum lag time in hours.

    Returns:
        1D array of shape (lag_time_hours,) containing the weights.
    """
    weights: ArrayFloat32 = np.zeros(lag_time_hours, dtype=np.float32)
    accumulated_area_prev: float = 0.0
    normalization_factor: float = 2.0 * peak_hour**2

    for lag in range(lag_time_hours):
        hour_index = float(lag + 1)
        # Mirror index for the falling limb of the triangle
        falling_limb_index: float = 2.0 * peak_hour - hour_index

        # Area under the rising limb: (t^2) / (2 * peak^2)
        rising_area_fraction: float = (hour_index**2) / normalization_factor
        # Area under the falling limb: 1 - ((2*peak - t)^2) / (2 * peak^2)
        falling_area_fraction: float = (
            1.0 - (falling_limb_index**2) / normalization_factor
        )

        if hour_index <= peak_hour:
            accumulated_area_current: float = rising_area_fraction
        else:
            accumulated_area_current: float = falling_area_fraction

        # Clamp at 1.0 if we go beyond the base of the triangle
        if falling_limb_index <= 0.0:
            accumulated_area_current = 1.0

        weight: float = accumulated_area_current - accumulated_area_prev
        accumulated_area_prev = accumulated_area_current
        weights[lag] = float(weight)

    weights /= weights.sum()  # ensure sum is exactly 1.0
    return weights


@njit(cache=True)
def apply_triangular(
    hourly_inflow_m: TwoDArrayFloat32,  # shape (24, n_cells)
    weights: ArrayFloat32,  # shape (lag_time_hours,)
    buffer_m: TwoDArrayFloat32,  # shape (lag_time_hours, n_cells)
) -> None:
    """Apply triangular weighting to distribute hourly inflow into a buffer.

    Args:
        hourly_inflow_m: 2D array with shape (24, n_cells) containing
            the hourly inflow for the current day.
        weights: 1D array with shape (lag_time_hours,) containing the triangular weights.
        buffer_m: 2D array with shape (lag_time_hours, n_cells) representing rolling buffer to accumulate weighted inflow. This array is modified in place.

    """
    buffer_size_hours: int = len(weights)
    daily_timesteps: int = hourly_inflow_m.shape[0]  # = 24 hours

    # For each hourly timestep in the current day
    for t in range(daily_timesteps):
        inflow_at_hour_t = hourly_inflow_m[t]  # shape (n_cells,)

        # Distribute the inflow at hour t over the future lag period
        for k in range(buffer_size_hours):
            weight = weights[k]
            if weight == 0.0:
                continue

            # Calculate the target hour in the buffer
            target_hour_index = t + k
            if target_hour_index < buffer_size_hours:
                buffer_m[target_hour_index] += weight * inflow_at_hour_t


def advance_buffer(buffer_m: TwoDArrayFloat32, timesteps_to_advance_hours: int) -> None:
    """Shift buffer forward by a given number of hours, removing handled water.

    Args:
        buffer_m: The rolling water buffer.
        timesteps_to_advance_hours: Number of hours (typically 24) to advance.
    """
    buffer_capacity_hours = buffer_m.shape[0]
    if timesteps_to_advance_hours >= buffer_capacity_hours:
        buffer_m[:] = 0.0
        return

    # Shift flow forward in time
    buffer_m[:-timesteps_to_advance_hours] = buffer_m[timesteps_to_advance_hours:]
    # Reset the trailing part of the buffer
    buffer_m[-timesteps_to_advance_hours:] = 0.0


class RunoffConcentrator(Module):
    """
    Module to apply runoff concentration using triangular weighting.

    Receives sub-daily runoff (24 timesteps per day) and maintains a
    rolling buffer to carry over runoff to future days.
    """

    def __init__(
        self,
        model: GEBModel,
        hydrology: Hydrology,
        lag_time_hours: int = 48,
        runoff_peak_hour: float = 3.0,
    ) -> None:
        """Initialize the runoff concentrator model.

        Currently in development. Not finished yet, but working first version.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology module instance.
            lag_time_hours: int = 48 hours,
            runoff_peak_hour: float = 3.0 hours,

        Raises:
            ValueError: If lag_time_hours is insufficient for the given runoff_peak_hour.
        """
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        self.lag_time_hours: int = lag_time_hours
        self.runoff_peak_hour: float = runoff_peak_hour

        # Check if lag time is sufficient to contain the triangular weighting of the entire day.
        # The base of the triangle is at 2 * peak_hour. The latest runoff comes at hour 24.
        max_required_buffer_hours = 24 + int(np.ceil(2 * runoff_peak_hour))
        if lag_time_hours < max_required_buffer_hours:
            raise ValueError(
                f"Insufficient lag_time_hours ({lag_time_hours}). "
                f"For a peak of {runoff_peak_hour}h, the buffer must be at least "
                f"{max_required_buffer_hours} hours to avoid losing water at the end of the day."
            )

        if self.model.in_spinup:
            self.spinup()

    def _apply_triangular(
        self,
        hourly_inflow_m: TwoDArrayFloat32,  # shape (24, n_cells)
        weights: ArrayFloat32,  # shape (lag_time_hours,)
    ) -> None:
        """Apply triangular weighting to distribute hourly inflow into the model's buffer."""
        apply_triangular(hourly_inflow_m, weights, self.grid.var.overland_flow_buffer)

    def _advance_buffer(self, hours_to_advance: int) -> None:
        """Shift the model's water buffer forward in time."""
        advance_buffer(self.grid.var.overland_flow_buffer, hours_to_advance)

    def step(
        self,
        interflow_m: TwoDArrayFloat32,  # shape (24, n_cells)
        baseflow_m: ArrayFloat32,  # shape (n_cells,)
        runoff_m: TwoDArrayFloat32,  # shape (24, n_cells)
    ) -> TwoDArrayFloat32:
        """Concentrate runoff using triangular weighting.

        Currently being developed. For now, we only apply it to runoff and leave baseflow
        and interflow unchanged. We take the assumption that runoff is smoothed out over 6
        timesteps (= 6 hours), with the peak being at timestep 3. Further work includes
        changing the time component to also include slopes and land uses.

        Args:
            runoff_m: 2D array with shape (24, n_cells) containing sub-daily surface runoff.
            interflow_m: 2D array with shape (24, n_cells) containing sub-daily interflow.
            baseflow_m: 1D array with shape (n_cells,) containing daily baseflow.

        Returns:
            2D array with shape (24, n_cells) representing the runoff concentrated outflow.
        """
        assert (runoff_m >= 0).all()
        assert (interflow_m >= 0).all()
        assert (baseflow_m >= 0).all()

        assert interflow_m.shape[0] == 24
        assert runoff_m.shape[0] == 24
        assert interflow_m.ndim == 2
        assert baseflow_m.ndim == 1
        assert runoff_m.ndim == 2

        daily_hours: int = runoff_m.shape[0]

        if __debug__:
            buffer_prev_m: TwoDArrayFloat32 = self.grid.var.overland_flow_buffer.copy()

        # Apply triangular weighting to runoff inflow
        self._apply_triangular(runoff_m, self.grid.var.overland_flow_buffer_weights)

        # Outflow is the first 24 buffer steps (the current day)
        outflow_runoff_m: TwoDArrayFloat32 = self.grid.var.overland_flow_buffer[
            :daily_hours
        ].copy()

        baseflow_per_hour: ArrayFloat32 = (baseflow_m / daily_hours).astype(np.float32)

        # Baseflow is distributed evenly across the day
        total_outflow_m: TwoDArrayFloat32 = (
            outflow_runoff_m + baseflow_per_hour + interflow_m
        )

        # Advance buffer by one day (24 hours) to discard today's flow and prepare for tomorrow
        self._advance_buffer(daily_hours)

        if __debug__:
            inflow_per_cell_m: ArrayFloat32 = (
                runoff_m.sum(axis=0) + interflow_m.sum(axis=0) + baseflow_m
            )
            outflow_per_cell_m: ArrayFloat32 = total_outflow_m.sum(axis=0)
            storage_pre_per_cell_m: ArrayFloat32 = buffer_prev_m.sum(axis=0)
            storage_post_per_cell_m: ArrayFloat32 = (
                self.grid.var.overland_flow_buffer.sum(axis=0)
            )

            balance_check(
                name="runoff concentration daily",
                how="cellwise",
                influxes=[inflow_per_cell_m],
                outfluxes=[outflow_per_cell_m],
                prestorages=[storage_pre_per_cell_m],
                poststorages=[storage_post_per_cell_m],
                tolerance=1e-5,
                raise_on_error=False,
            )

        return total_outflow_m

    @property
    def name(self) -> str:
        """Return the name of the module.

        Returns:
            The name of the module.
        """
        return "hydrology.runoff_concentrator"

    def spinup(self) -> None:
        """Initialize variables needed for the runoff concentration model."""
        self.grid.var.overland_flow_buffer = np.tile(
            self.grid.full_compressed(0.0, dtype=np.float32), (self.lag_time_hours, 1)
        )

        self.grid.var.overland_flow_buffer_weights = triangular_weights(
            self.runoff_peak_hour, self.lag_time_hours
        )
