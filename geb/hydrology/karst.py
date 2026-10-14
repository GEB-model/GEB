"""Store karst recharge before it reaches groundwater."""

from typing import TYPE_CHECKING

import numpy as np

from geb.geb_types import ArrayFloat32, ArrayFloat64, TwoDArrayFloat32
from geb.module import Module

from .landcovers import SEALED

if TYPE_CHECKING:
    from geb.hydrology import Hydrology
    from geb.model import GEBModel


class Karst(Module):
    """Load karst coverage and release stored water gradually."""

    def __init__(self, model: GEBModel, hydrology: Hydrology) -> None:
        """Load coverage and initialize storage during spinup.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology module sharing the grid and saved storage.

        Raises:
            ValueError: If coverage is missing or contains values outside 0–1.
        """
        super().__init__(model, create_var=False)
        self.hydrology = hydrology
        self.grid = hydrology.grid
        if "groundwater/karst_fraction" not in model.files["grid"]:
            raise ValueError(
                "Karst inputs are missing. Run setup_karst and rerun spinup."
            )
        self.fraction: ArrayFloat32 = self.grid.load2d(
            model.files["grid"]["groundwater/karst_fraction"]
        )
        if (
            not np.isfinite(self.fraction).all()
            or ((self.fraction < 0) | (self.fraction > 1)).any()
        ):
            raise ValueError(
                "Karst map values must be between 0 and 1, with no NaNs or infinities."
            )
        if model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Return the module name.

        Returns:
            The name used by GEB to identify the module.
        """
        return "hydrology.karst"

    def spinup(self) -> None:
        """Start spinup with empty storage (m)."""
        # Keep the existing grid variable so checkpoints and reports stay compatible.
        self.grid.var.karst_storage_m = np.zeros_like(self.fraction, dtype=np.float64)

    @property
    def storage_m(self) -> ArrayFloat64:
        """Read saved karst storage.

        Returns:
            Stored water depth (m), per grid cell.

        Raises:
            ValueError: If karst storage is missing from the checkpoint.
        """
        if not hasattr(self.grid.var, "karst_storage_m"):
            raise ValueError(
                "Karst storage is missing. Rerun spinup to initialize karst storage."
            )
        return self.grid.var.karst_storage_m

    def capture(
        self,
        runoff_m: TwoDArrayFloat32,
        interflow_m: TwoDArrayFloat32,
        soil_recharge_m: ArrayFloat32,
    ) -> tuple[ArrayFloat32, ArrayFloat32]:
        """Divert the karst share of land flows, updating the input arrays in place.

        Args:
            runoff_m: Hourly runoff depths (m), per HRU.
            interflow_m: Hourly interflow depths (m), per HRU.
            soil_recharge_m: Daily soil drainage depth (m), per HRU.

        Returns:
            Captured runoff plus interflow, and captured soil drainage (m),
            both per HRU.
        """
        # Sealed land and open water do not recharge karst.
        land_fraction: ArrayFloat32 = self.hydrology.to_HRU(data=self.fraction) * (
            self.hydrology.HRU.var.land_use_type < SEALED
        )
        capture_fraction: ArrayFloat32 = land_fraction * np.float32(
            self.model.config["hydrology"]["karst"]["capture_fraction"]
        )
        captured_runoff_m: ArrayFloat32 = (
            runoff_m.sum(axis=0) + interflow_m.sum(axis=0)
        ) * capture_fraction
        captured_soil_m: ArrayFloat32 = soil_recharge_m * land_fraction
        # Remove captured water from its original route to avoid counting it twice.
        runoff_m *= np.float32(1.0) - capture_fraction
        interflow_m *= np.float32(1.0) - capture_fraction
        soil_recharge_m -= captured_soil_m
        return captured_runoff_m, captured_soil_m

    def step(self, inflow_m: ArrayFloat64) -> ArrayFloat64:
        """Delay karst recharge with a standard linear reservoir.

        Notes:
            Storage and outflow follow S = K * Q, where K is release_time_days.
            We solve dS/dt = inflow - S/K over one day, assuming inflow
            arrives evenly. Released water goes to MODFLOW.
            See the MIKE SHE explanation of linear reservoirs:
            https://mikeshe-mike-platform-prod.eu.mike-cloud.com/2026/WM/SZ/LinearReservoir/

        Args:
            inflow_m: Water entering karst over the day (m), per grid cell.

        Returns:
            Water released to groundwater over the day (m), per grid cell.
        """
        storage_m: ArrayFloat64 = self.storage_m
        release_time_days: float = self.model.config["hydrology"]["karst"][
            "release_time_days"
        ]
        # Release a fraction of the stored water each day: 1 - exp(-1 / release time).
        # With a release time of 10 days, about 9.5% leaves the existing store every day.
        release_fraction: np.float64 = -np.expm1(-1.0 / release_time_days)

        # Add release from existing storage and from today's inflow.
        # New water arrives evenly during the day, so it has less time to drain.
        recharge_m: ArrayFloat64 = storage_m * release_fraction + inflow_m * (
            1.0 - release_time_days * release_fraction
        )

        # Keep the rest: old storage + water entering - water sent to MODFLOW.
        storage_m += inflow_m - recharge_m
        return recharge_m
