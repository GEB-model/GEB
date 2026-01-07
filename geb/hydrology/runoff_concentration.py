"""Module for concentrating runoff from different sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geb.module import Module
from geb.types import ArrayFloat32, ArrayFloat64, TwoDArrayFloat64
from geb.workflows import balance_check

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


class RunoffConcentrator(Module):
    """
    Module to apply runoff concentration using triangular weighting.

    Receives sub-daily runoff (24 timesteps per day) and maintains a
    rolling buffer to carry over runoff to future days.
    """

    overland_runoff_storage_end_m3: np.float64

    def __init__(
        self,
        model: GEBModel,
        hydrology: Hydrology,
        lagtime: int = 48,
        runoff_peak: float = 3.0,
    ) -> None:
        """Initialize the runoff concentrator model.

        Currently in development. Not finished yet, but working first version.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology module instance.
            lagtime: int = 48 hours,
            runoff_peak: float = 3.0 hours,
        """
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        self.lagtime: int = lagtime
        self.runoff_peak: float = runoff_peak
        self.overland_runoff_storage_end_m3 = np.float64(0.0)

        # precompute triangular weights
        self.weights_runoff = self._triangular_weights(runoff_peak)

        if self.model.in_spinup:
            self.spinup()

    def _triangular_weights(self, peak: float) -> ArrayFloat64:
        """Compute triangular weights for given peak lag time.

        Returns:
            1D array of shape (lagtime,) containing the weights.
        """
        weights: ArrayFloat64 = np.zeros(self.lagtime, dtype=np.float64)
        areaFractionOld: float = 0.0
        div: float = 2.0 * peak**2

        for lag in range(self.lagtime):
            lag1 = float(lag + 1)
            lag1alt: float = 2.0 * peak - lag1
            area: float = (lag1**2) / div
            areaAlt: float = 1.0 - (lag1alt**2) / div

            if lag1 <= peak:
                areaFractionSum: float = area
            else:
                areaFractionSum: float = areaAlt

            if lag1alt <= 0.0:
                areaFractionSum = 1.0

            areaFraction: float = areaFractionSum - areaFractionOld
            areaFractionOld: float = areaFractionSum
            weights[lag] = areaFraction

        weights /= weights.sum()  # normalize
        return weights

    def _apply_triangular(
        self,
        flow: TwoDArrayFloat64,  # shape (24, n_cells)
        weights: ArrayFloat64,  # shape (lagtime,)
    ) -> None:
        """Apply triangular weighting to the flow and update the buffer."""
        lag: int = len(weights)
        n_steps: int = flow.shape[0]  # = 24

        # For each hourly timestep
        for t in range(n_steps):
            ft = flow[t]  # shape (n_cells,)

            # Route contribution to buffer positions t+k
            for k in range(lag):
                w = weights[k]
                if w == 0.0:
                    continue

                idx = t + k
                if idx < lag:
                    self.grid.var.buffer[idx] += w * ft
                # else: future contribution beyond lagtime is dropped

    def _advance_buffer(self, n_steps: int) -> None:
        """Shift buffer forward by n_steps."""
        if n_steps >= self.lagtime:
            self.grid.var.buffer[:] = 0.0
            return

        # Shift forward
        self.grid.var.buffer[:-n_steps] = self.grid.var.buffer[n_steps:]
        self.grid.var.buffer[-n_steps:] = 0.0

    def step(
        self,
        interflow: TwoDArrayFloat64,  # shape (24, n_cells)
        baseflow: ArrayFloat32,  # shape (n_cells,)
        runoff: TwoDArrayFloat64,  # shape (24, n_cells)
    ) -> TwoDArrayFloat64:
        """Concentrate runoff using triangular weighting.

        Currently being developed. For now, we only apply it to runoff and leave baseflow
        and interflow unchanged. We take the assumption that runoff is smoothed out over 6
        timesteps (= 6 hours), with the peak being at timestep 3. Further work includes
        changing the time component to also include slopes and land uses.

        Args:
            runoff: 2D array with shape (24, n_cells) containing sub-daily surface runoff.
            interflow: 2D array with shape (24, n_cells) containing sub-daily interflow.
            baseflow: 1D array with shape (n_cells,) containing daily baseflow.

        Returns:
            2D array with shape (24, n_cells) representing the runoff concentrated outflow.
        """
        assert (runoff >= 0).all()
        assert (interflow >= 0).all()
        assert (baseflow >= 0).all()

        assert interflow.shape[0] == 24
        assert runoff.shape[0] == 24
        assert interflow.ndim == 2
        assert baseflow.ndim == 1
        assert runoff.ndim == 2

        n_steps: int
        n_cells: int
        n_steps, n_cells = runoff.shape

        # Advance buffer by one day (24 hours)
        self._advance_buffer(n_steps)
        overland_runoff_storage_start_m: TwoDArrayFloat64 = (
            self.grid.var.buffer.copy().astype(np.float64)
        )
        overland_runoff_storage_start_m3: float = (
            overland_runoff_storage_start_m * self.grid.var.cell_area
        ).sum()

        # Baseflow is distributed evenly across 24 substeps
        baseflow_per_step: ArrayFloat64 = (baseflow / n_steps).astype(np.float64)
        baseflow_array: TwoDArrayFloat64 = np.broadcast_to(
            baseflow_per_step, (n_steps, n_cells)
        )  # Create array that matches the shape of the runoff

        # Apply triangular weighting to (for now only) runoff
        self._apply_triangular(runoff, self.weights_runoff)

        outflow_runoff_m: TwoDArrayFloat64 = (
            self.grid.var.buffer[:n_steps].copy()
        )  # Outflow is only the first 24 buffer steps which equals 24 hourly outflows
        total_outflow_m: TwoDArrayFloat64 = (
            outflow_runoff_m + baseflow_array + interflow
        )  # Get total outflow (including baseflow and interflow which did not change)
        overland_runoff_storage_end_m: TwoDArrayFloat64 = (
            self.grid.var.buffer[n_steps:].copy().astype(np.float64)
        )  # Everything that is stored for the next day
        self.overland_runoff_storage_end_m3 = (
            overland_runoff_storage_end_m * self.grid.var.cell_area
        ).sum()

        outflow_m3: float = (
            (outflow_runoff_m * self.grid.var.cell_area).sum()
            + (interflow * self.grid.var.cell_area).sum()
            + (baseflow_array * self.grid.var.cell_area).sum()
        )

        inflow_m3: float = (
            (runoff * self.grid.var.cell_area).sum()
            + (interflow * self.grid.var.cell_area).sum()
            + (baseflow_array * self.grid.var.cell_area).sum()
        )

        if __debug__:
            balance_check(
                name="RunoffConcentrator daily water balance",
                how="sum",
                influxes=[inflow_m3],
                outfluxes=[outflow_m3],
                prestorages=[overland_runoff_storage_start_m3],
                poststorages=[self.overland_runoff_storage_end_m3],
                tolerance=1,
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
        """Initialize variables needed for the runoff concentration model.

        Returns:
            None
        """
        self.grid.var.buffer = np.stack(
            [
                self.grid.full_compressed(0.0, dtype=np.float64)
                for _ in range(self.lagtime)
            ],
            axis=0,
        ).astype(np.float64)

        self.weights_runoff: ArrayFloat64 = self._triangular_weights(self.runoff_peak)

        return None
