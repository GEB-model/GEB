"""Module for concentrating runoff from different sources."""

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from geb.geb_types import (
    ArrayFloat32,
    TwoDArrayFloat,
    TwoDArrayFloat32,
)
from geb.hydrology.landcovers import (
    FOREST,
    GRASSLAND_LIKE,
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)
from geb.module import Module
from geb.workflows import balance_check

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology

# TxDOT Hydraulic Design Manual Table 4-5: Kerby Retardance Coefficients (N)
# Dimensionless retardance coefficient for overland sheet flow:
KERBY_RETARDANCE_PAVEMENT: float = 0.02
KERBY_RETARDANCE_BARE_SOIL: float = 0.10
KERBY_RETARDANCE_CULTIVATED: float = 0.20
KERBY_RETARDANCE_PASTURE_GRASS: float = 0.40
KERBY_RETARDANCE_DECIDUOUS_FOREST: float = 0.60
KERBY_RETARDANCE_DENSE_CONIFEROUS_FOREST: float = 0.80

KERBY_RETARDANCE_BY_LAND_COVER: dict[int, float] = {
    FOREST: KERBY_RETARDANCE_DECIDUOUS_FOREST,
    GRASSLAND_LIKE: KERBY_RETARDANCE_PASTURE_GRASS,
    PADDY_IRRIGATED: KERBY_RETARDANCE_CULTIVATED,
    NON_PADDY_IRRIGATED: KERBY_RETARDANCE_CULTIVATED,
    SEALED: KERBY_RETARDANCE_PAVEMENT,
    OPEN_WATER: KERBY_RETARDANCE_PAVEMENT,
}


def triangular_weights(peak_hour: float, lag_time_hours: int) -> ArrayFloat32:
    """Compute 1D triangular weights for a given peak lag time.

    Uses a piecewise cumulative distribution function (CDF) for the standard NRCS
    asymmetric triangular unit hydrograph (NEH Part 630 Chapter 16):
    rising limb (37.5% volume) for t <= peak_hour, falling limb (62.5% volume)
    for peak_hour < t < (8/3)*peak_hour, and 1.0 for t >= (8/3)*peak_hour.

    Args:
        peak_hour: Peak lag time in hours.
        lag_time_hours: Maximum lag time in hours.

    Returns:
        1D array of shape (lag_time_hours,) containing the weights summing to 1.0.
    """
    weights: ArrayFloat32 = np.zeros(lag_time_hours, dtype=np.float32)
    if peak_hour <= 0.0:
        weights[0] = 1.0
        return weights

    accumulated_area_prev: float = 0.0
    base_time_hours: float = (8.0 / 3.0) * peak_hour
    recession_hours: float = (5.0 / 3.0) * peak_hour
    inv_peak_sq: float = 0.375 / (peak_hour * peak_hour)
    inv_rec_sq: float = 0.625 / (recession_hours * recession_hours)

    for lag in range(lag_time_hours):
        elapsed_hours: float = float(lag + 1)

        # Past base time: complete unit hydrograph volume (1.0)
        if elapsed_hours >= base_time_hours:
            accumulated_area_current: float = 1.0
        # Rising limb: 0.375 * (t / Tp)^2
        elif elapsed_hours <= peak_hour:
            accumulated_area_current = (elapsed_hours**2) * inv_peak_sq
        # Falling limb: 1 - 0.625 * ((Tb - t) / Tr)^2
        else:
            remaining_hours: float = base_time_hours - elapsed_hours
            accumulated_area_current = 1.0 - (remaining_hours**2) * inv_rec_sq

        weights[lag] = float(accumulated_area_current - accumulated_area_prev)
        accumulated_area_prev = accumulated_area_current

        # Once cumulative area reaches 1.0, remaining lags have weight 0.0
        if accumulated_area_current >= 1.0:
            break

    # Normalize weights to sum to 1.0
    weights /= weights.sum()
    return weights


@njit(fastmath=True, parallel=True)
def compute_slope_dependent_triangular_weights(
    hillslope_length_m: ArrayFloat32,
    slope_m_per_m: ArrayFloat32,
    retardance_roughness: ArrayFloat32,
    lag_time_hours: int,
    concentration_factor: float = 1.0,
    min_slope: float = 1e-4,
) -> TwoDArrayFloat32:
    """Computes cell-specific triangular unit hydrograph weights based on hillslope length, slope, and retardance roughness.

    Calculates overland time of concentration using the Kerby (1959) overland sheet flow equation (TxDOT Eq. 4-14):
        concentration_time_minutes = 1.44 * (roughness * hillslope_length)^0.467 * slope^-0.235  (minutes)
    where roughness is the Kerby retardance roughness coefficient (from TxDOT Table 4-5),
    hillslope_length is travel length in meters, and slope is topographic surface slope (m/m).
    Converts to hours:
        concentration_time_hours = concentration_time_minutes / 60.0  (hours)
    and determines the time to peak for the triangular unit hydrograph:
        peak_hour = (0.5 * D + 0.6 * concentration_time_hours) * concentration_factor  (hours)
    where D = 1.0 hour is the discrete rainfall time step duration, and 0.5 * D accounts for the
    rainfall centroid offset (SCS unit hydrograph theory).
    For each grid cell, generates a triangular weight distribution over lag_time_hours normalized to sum to 1.0.

    Args:
        hillslope_length_m: Distance from hillslope divide to stream channel per cell (meters).
        slope_m_per_m: Topographic surface slope per cell (m/m horizontal:vertical).
        retardance_roughness: Kerby retardance roughness coefficient per cell (dimensionless).
        lag_time_hours: Rolling buffer duration (hours).
        concentration_factor: Multiplier for overland concentration time (dimensionless, default 1.0).
        min_slope: Minimum physical slope threshold to prevent division by zero for flat cells (m/m, default 1e-4).

    Returns:
        2D array of triangular unit hydrograph weights of shape (lag_time_hours, num_cells).
    """
    num_cells: int = hillslope_length_m.shape[0]
    weights: TwoDArrayFloat32 = np.zeros((lag_time_hours, num_cells), dtype=np.float32)

    for cell_idx in prange(num_cells):
        slope: float = float(slope_m_per_m[cell_idx])
        if slope <= 0.0:
            slope = min_slope

        # Kerby (1959) overland sheet flow concentration time in hours:
        # Tc = (1.44 / 60) * (N * L)^0.467 * S^-0.235 (TxDOT Eq. 4-14)
        tc_hours: float = (
            (1.44 / 60.0)
            * (
                (
                    float(retardance_roughness[cell_idx])
                    * float(hillslope_length_m[cell_idx])
                )
                ** 0.467
            )
            * (slope**-0.235)
        )

        # https://www.hydrocad.net/neh/630ch16.pdf
        # Standard SCS / NRCS unit hydrograph time to peak: Tp = D / 2 + 0.6 * Tc
        # where D = 1.0 hour is the discrete rainfall step duration (centroid at D / 2 = 0.5 h),
        # and 0.6 * Tc is the watershed lag time (TL = 0.6 * Tc).
        peak_hour: float = (0.5 + 0.6 * tc_hours) * concentration_factor

        # Standard NRCS equivalent triangular unit hydrograph (NEH Part 630 Chapter 16):
        # Rising limb accounts for 37.5% (3/8) of volume over duration Tp.
        # Recession limb accounts for 62.5% (5/8) of volume over duration Tr = (5/3) * Tp.
        # Total base time Tb = Tp + Tr = (8/3) * Tp (~2.67 * Tp).
        base_time_hours: float = (8.0 / 3.0) * peak_hour
        recession_hours: float = (5.0 / 3.0) * peak_hour
        inv_peak_sq: float = 0.375 / (peak_hour * peak_hour)
        inv_rec_sq: float = 0.625 / (recession_hours * recession_hours)
        area_prev: float = 0.0

        for lag in range(lag_time_hours):
            t: float = float(lag + 1)

            if t >= base_time_hours:
                area_current: float = 1.0
            elif t <= peak_hour:
                area_current = (t * t) * inv_peak_sq
            else:
                rem: float = base_time_hours - t
                area_current = 1.0 - (rem * rem) * inv_rec_sq

            weights[lag, cell_idx] = np.float32(area_current - area_prev)
            area_prev = area_current

            if area_current >= 1.0:
                break

    return weights


@njit(fastmath=True, parallel=True)
def apply_triangular(
    hourly_inflow_m: TwoDArrayFloat,
    weights: TwoDArrayFloat,
    buffer_m: TwoDArrayFloat,
) -> None:
    """Apply triangular weighting to distribute hourly inflow into a rolling buffer.

    Args:
        hourly_inflow_m: 2D array of shape (24, num_cells) containing hourly inflow depths (meters).
        weights: 2D array of shape (lag_time_hours, num_cells) of triangular unit hydrograph weights.
        buffer_m: 2D rolling buffer array of shape (lag_time_hours, num_cells) modified in place.
    """
    buffer_size_hours: int = weights.shape[0]
    daily_timesteps: int = hourly_inflow_m.shape[0]
    num_cells: int = hourly_inflow_m.shape[1]

    for cell_idx in prange(num_cells):
        for hour in range(daily_timesteps):
            inflow = hourly_inflow_m[hour, cell_idx]
            if inflow <= 0.0:
                continue
            for lag in range(buffer_size_hours):
                weight = weights[lag, cell_idx]
                if weight <= 0.0:
                    continue
                target_hour = hour + lag
                if target_hour < buffer_size_hours:
                    buffer_m[target_hour, cell_idx] += weight * inflow


def advance_buffer(buffer_m: TwoDArrayFloat32, timesteps_to_advance_hours: int) -> None:
    """Shift buffer forward by a given number of hours, removing handled water.

    Args:
        buffer_m: The rolling water buffer.
        timesteps_to_advance_hours: Number of hours (typically 24) to advance.
    """
    buffer_capacity_hours: int = buffer_m.shape[0]
    if timesteps_to_advance_hours >= buffer_capacity_hours:
        buffer_m[:] = 0.0
        return

    # Shift flow forward in time
    buffer_m[:-timesteps_to_advance_hours] = buffer_m[timesteps_to_advance_hours:]
    # Reset the trailing part of the buffer
    buffer_m[-timesteps_to_advance_hours:] = 0.0


class RunoffConcentrator(Module):
    """Module to apply slope-dependent runoff concentration using triangular unit hydrographs.

    Receives sub-daily surface runoff (24 hourly timesteps per day) and routes it across the
    hillslope to the channel network using cell-specific triangular unit hydrographs. The time of
    concentration and peak arrival lag are physically derived from hillslope length and surface slope
    via the Kerby (1959) overland sheet flow equation, delaying and attenuating runoff before it
    enters stream routing. Maintains a rolling buffer to carry over runoff to future days.
    """

    def __init__(
        self,
        model: GEBModel,
        hydrology: Hydrology,
        lag_time_hours: int = 72,
    ) -> None:
        """Initialize the slope-dependent runoff concentrator model.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology module instance.
            lag_time_hours: Buffer capacity duration in hours (default 72).
        """
        super().__init__(model)
        self.hydrology: Hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        self.lag_time_hours: int = lag_time_hours
        self.concentration_factor: float = float(
            self.model.config["parameters"]["runoff_concentration_factor"]
        )

        if self.model.in_spinup:
            self.spinup()

    def set_weights(self) -> None:
        """Derive cell-specific Kerby retardance roughness and calculate triangular weights.

        Derives Kerby retardance roughness from HRU land use types, maps them to grid cells,
        and computes unit hydrograph weights based on terrain slope and hillslope length.
        """
        slope_m_per_m: ArrayFloat32 = self.grid.load2d(
            self.model.files["grid"]["landsurface/slope_m_per_m"]
        )
        hillslope_length_m: ArrayFloat32 = self.grid.load2d(
            self.model.files["grid"]["drainage/hillslope_length_m"]
        )

        # Derive cell-specific Kerby retardance roughness from HRU land use types
        # based on TxDOT Hydraulic Design Manual Table 4-5:
        # FOREST: 0.60 (deciduous forest / mixed woodland)
        # GRASSLAND_LIKE: 0.40 (pasture / average grass)
        # CROPLAND (paddy / non-paddy): 0.20 (cultivated row crops)
        # SEALED / WATER: 0.02 (pavement / smooth surface)
        retardance_table: ArrayFloat32 = np.array(
            [
                KERBY_RETARDANCE_DECIDUOUS_FOREST,
                KERBY_RETARDANCE_PASTURE_GRASS,
                KERBY_RETARDANCE_CULTIVATED,
                KERBY_RETARDANCE_CULTIVATED,
                KERBY_RETARDANCE_PAVEMENT,
                KERBY_RETARDANCE_PAVEMENT,
            ],
            dtype=np.float32,
        )
        retardance_hru: ArrayFloat32 = retardance_table[self.HRU.var.land_use_type]
        retardance_grid: ArrayFloat32 = self.hydrology.to_grid(
            HRU_data=retardance_hru
        ).astype(np.float32)

        self.grid.var.overland_flow_buffer_weights = (
            compute_slope_dependent_triangular_weights(
                hillslope_length_m=hillslope_length_m,
                slope_m_per_m=slope_m_per_m,
                retardance_roughness=retardance_grid,
                lag_time_hours=self.lag_time_hours,
                concentration_factor=self.concentration_factor,
            )
        )

    def _apply_triangular(
        self,
        hourly_inflow_m: TwoDArrayFloat32,
        weights: TwoDArrayFloat32,
    ) -> None:
        """Apply triangular weighting to distribute hourly inflow into the model's buffer."""
        apply_triangular(hourly_inflow_m, weights, self.grid.var.overland_flow_buffer)

    def _advance_buffer(self, hours_to_advance: int) -> None:
        """Shift the model's water buffer forward in time."""
        advance_buffer(self.grid.var.overland_flow_buffer, hours_to_advance)

    def step(
        self,
        interflow_m: TwoDArrayFloat32,
        baseflow_m: ArrayFloat32,
        runoff_m: TwoDArrayFloat32,
    ) -> TwoDArrayFloat32:
        """Concentrate surface runoff using slope-dependent triangular weighting.

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

        # Apply cell-specific slope-dependent triangular weighting to runoff inflow
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
        self.set_weights()
