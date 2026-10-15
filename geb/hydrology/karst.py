"""Store karst recharge before it reaches groundwater."""

from typing import TYPE_CHECKING

import numpy as np

from geb.config_schema import KarstConfig
from geb.geb_types import ArrayFloat32, ArrayFloat64, TwoDArrayFloat32
from geb.module import Module

from .landcovers import OPEN_WATER, SEALED

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
        """
        super().__init__(model, create_var=False)
        # parse_config validates a temporary model without adding schema defaults.
        karst_config: KarstConfig = KarstConfig.model_validate(
            model.config["hydrology"].get("karst", {})
        )
        model.config["hydrology"]["karst"] = karst_config.model_dump()
        self.hydrology = hydrology
        self.grid = hydrology.grid
        self.karst_fraction: ArrayFloat32 = self.grid.load2d(
            model.files["grid"]["groundwater/karst_fraction"]
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
        self.grid.var.karst_storage_m = np.zeros_like(
            self.karst_fraction, dtype=np.float64
        )

    @property
    def storage_m(self) -> ArrayFloat64:
        """Read saved karst storage.

        Returns:
            Stored water depth (m), per grid cell.
        """
        return self.grid.var.karst_storage_m

    def step(
        self,
        runoff_m: TwoDArrayFloat32,
        interflow_m: TwoDArrayFloat32,
        soil_recharge_m: ArrayFloat32,
    ) -> tuple[ArrayFloat64, ArrayFloat32, ArrayFloat32]:
        """Capture land flows and release delayed recharge to groundwater.

        Notes:
            Updates the input arrays in place to remove captured water. Sealed
            land and open water contribute no karst recharge. The store follows
            dS/dt = inflow - S/K, with constant inflow over the daily step.
            It has no maximum capacity: it delays recharge without representing
            saturation or overflow. A capacity would require an overflow route.
            K is the release time in days; after K dry days, exp(-1) of the
            initial storage remains (about 36.8%).

        Args:
            runoff_m: Hourly runoff depths (m), per HRU.
            interflow_m: Hourly interflow depths (m), per HRU.
            soil_recharge_m: Daily soil drainage depth (m), per HRU.

        Returns:
            Daily recharge released to groundwater (m), per grid cell; captured
            runoff plus interflow (m), per HRU; and captured soil drainage (m),
            per HRU.
        """
        non_sealed_land_fraction: ArrayFloat32 = self.hydrology.to_HRU(
            data=self.karst_fraction
        ) * (
            (self.hydrology.HRU.var.land_use_type != SEALED)
            & (self.hydrology.HRU.var.land_use_type != OPEN_WATER)
        )
        capture_fraction: ArrayFloat32 = non_sealed_land_fraction * np.float32(
            self.model.config["hydrology"]["karst"]["capture_fraction"]
        )
        captured_runoff_m: ArrayFloat32 = (
            runoff_m.sum(axis=0) + interflow_m.sum(axis=0)
        ) * capture_fraction
        captured_soil_m: ArrayFloat32 = soil_recharge_m * non_sealed_land_fraction
        # Remove captured water from its original route to avoid counting it twice.
        runoff_m *= np.float32(1.0) - capture_fraction
        interflow_m *= np.float32(1.0) - capture_fraction
        soil_recharge_m -= captured_soil_m
        inflow_m: ArrayFloat64 = self.hydrology.to_grid(
            HRU_data=captured_runoff_m + captured_soil_m
        ).astype(np.float64)
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

        # Retain water that was captured but has not yet drained to groundwater.
        storage_m += inflow_m - recharge_m
        return recharge_m, captured_runoff_m, captured_soil_m
