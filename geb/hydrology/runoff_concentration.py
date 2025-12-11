"""Module for concentrating runoff from different sources."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from geb.module import Module
from geb.workflows import balance_check

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


class RunoffConcentrator(Module):
    """
    Stateful runoff concentrator using triangular weighting.

    Receives sub-daily runoff (24 timesteps per day) and maintains a
    rolling buffer to carry over runoff to future days.
    """

    def __init__(
        self,
        model,
        hydrology,
        lagtime: int = 48,
        runoff_peak: float = 3.0,
        interflow_peak: float = 4.0,
        baseflow_peak: float = 5.0,
    ) -> None:
        """Initialize the runoff concentrator model.

        Currently in super alpha stage and uses a lot of constants and assumptions. Should NOT
        be used for science yet.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology module instance.
            lagtime: int = 24,
            runoff_peak: float = 3.0,
            interflow_peak: float = 4.0,
            baseflow_peak: float = 5.0,
        """
        super().__init__(model)
        self.hydrology = hydrology

        if self.model.in_spinup:
            self.spinup()

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid
        self.lagtime = max(lagtime, 48)

        # precompute triangular weights
        self.weights_runoff = self._triangular_weights(runoff_peak)
        self.weights_interflow = self._triangular_weights(interflow_peak)
        self.weights_baseflow = self._triangular_weights(baseflow_peak)

        # rolling buffer: shape [lagtime, n_cells]
        self.buffer = None

    def _triangular_weights(self, peak: float) -> npt.NDArray[np.float64]:
        weights = np.zeros(self.lagtime, dtype=np.float64)
        areaFractionOld = 0.0
        div = 2.0 * peak**2

        for lag in range(self.lagtime):
            lag1 = float(lag + 1)
            lag1alt = 2.0 * peak - lag1
            area = (lag1**2) / div
            areaAlt = 1.0 - (lag1alt**2) / div

            if lag1 <= peak:
                areaFractionSum = area
            else:
                areaFractionSum = areaAlt

            if lag1alt <= 0.0:
                areaFractionSum = 1.0

            areaFraction = areaFractionSum - areaFractionOld
            areaFractionOld = areaFractionSum
            weights[lag] = areaFraction

        weights /= weights.sum()  # normalize
        return weights

    def _apply_triangular(
        self,
        flow: npt.NDArray[np.float64],  # shape (24, n_cells)
        weights: npt.NDArray[np.float64],  # shape (lagtime,)
    ) -> None:
        lag = len(weights)
        n_steps = flow.shape[0]  # = 24

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
                    self.buffer[idx] += w * ft
                # else: future contribution beyond lagtime is dropped

    def _init_buffer(self, n_cells: int) -> None:
        self.n_cells = n_cells
        self.lagtime = 48  # 2 days of hourly storage
        self.buffer = np.zeros((self.lagtime, n_cells), dtype=np.float64)
        self.runoff_peak = 3.0
        self.interflow_peak: float = 4.0
        self.baseflow_peak: float = 5.0
        self.weights_runoff = self._triangular_weights(self.runoff_peak)
        self.weights_interflow = self._triangular_weights(self.interflow_peak)
        self.weights_baseflow = self._triangular_weights(self.baseflow_peak)

    def _advance_buffer(self, n_steps: int) -> None:
        """Shift buffer forward by n_steps (24 hours)."""
        if n_steps >= self.lagtime:
            self.buffer[:] = 0.0
            return

        # Shift forward
        self.buffer[:-n_steps] = self.buffer[n_steps:]
        self.buffer[-n_steps:] = 0.0

    def step(
        self,
        interflow: npt.NDArray[np.float32],  # shape (24, n_cells)
        baseflow: npt.NDArray[np.float32],  # shape (n_cells,)
        runoff: npt.NDArray[np.float32],  # shape (24, n_cells)
    ) -> npt.NDArray[np.float32]:
        """Route daily runoff, interflow and baseflow through the triangular lag buffer.

        Args:
            runoff: 2D array with shape (24, n_cells) containing sub-daily surface runoff.
            interflow: 2D array with shape (24, n_cells) containing sub-daily interflow.
            baseflow: 1D array with shape (n_cells,) containing daily baseflow.

        Returns:
            2D array with shape (24, n_cells) representing the routed hourly outflow for the day.
        """
        n_steps, n_cells = runoff.shape  # n_steps = 24

        if self.buffer is None:
            self._init_buffer(n_cells)

        # Advance buffer by one day (24 hours)
        self._advance_buffer(n_steps)
        storage_start = self.buffer.copy().astype(np.float64)

        # Baseflow is distributed evenly across 24 substeps
        baseflow_per_step = (baseflow / n_steps).astype(np.float32)
        baseflow_series = np.broadcast_to(baseflow_per_step, (n_steps, n_cells))

        # Apply triangular routing for each flow component
        self._apply_triangular(runoff, self.weights_runoff)
        # self._apply_triangular(interflow, self.weights_interflow)
        # self._apply_triangular(baseflow_series, self.weights_baseflow)

        # Outflow is only the first 24 buffer steps which equals 24 hourly outflows
        outflow = self.buffer[:n_steps].copy()  # shape (24, n_cells)
        storage_end = self.buffer[n_steps:].copy().astype(np.float64)
        print(self.grid.compressed_size / 3)
        balance_check(
            name="RunoffConcentrator daily water balance",
            how="sum",
            influxes=[runoff, interflow, baseflow_series],
            outfluxes=[outflow.astype(np.float64)],
            prestorages=[storage_start],
            poststorages=[storage_end],
            tolerance=1e-6,
            raise_on_error=False,
        )

        return outflow

    @property
    def name(self) -> str:
        """Return the name of the module.

        Returns:
            The name of the module.
        """
        return "hydrology.runoff_concentrator"

    def spinup(self) -> None:
        """Initialize variables needed for the hillslope erosion model.

        Currently this is in alpha stage and uses a lot of constants and assumptions. Should NOT
        be use for science yet.
        Returns:
            None
        """
        return None


# def concentrate_runoff(
#     interflow: npt.NDArray[np.float32],
#     baseflow: npt.NDArray[np.float32],
#     runoff: npt.NDArray[np.float32],
# ) -> npt.NDArray[np.float32]:
#     """Combines all sources of runoff.

#     Args:
#         interflow: The interflow [m] for the cell.
#         baseflow: The baseflow [m] for the cell.
#         runoff: The surface runoff [m] for the cell.

#     Returns:
#         The total runoff [m] for the cell.
#     """
#     assert (runoff >= 0).all()
#     assert (interflow >= 0).all()
#     assert (baseflow >= 0).all()

#     assert interflow.shape[0] == 24
#     assert runoff.shape[0] == 24

#     assert interflow.ndim == 2
#     assert baseflow.ndim == 1
#     assert runoff.ndim == 2

#     baseflow_per_timestep = baseflow / np.float32(24)

#     return interflow + baseflow_per_timestep + runoff
