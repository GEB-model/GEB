"""Store karst recharge before it reaches groundwater."""

from typing import TYPE_CHECKING

import numpy as np

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
    ) -> tuple[ArrayFloat64, ArrayFloat32, ArrayFloat32, ArrayFloat64]:
        """Capture land flows, release delayed recharge, and handle capacity overflow.

        Notes:
            Updates the input arrays in place to remove captured water. Sealed
            land and open water contribute no karst recharge. Hillslope runoff
            and interflow are captured up to max_runoff_capture_m_per_day, with
            excess overflowing to surface stream routes. The store follows
            dS/dt = inflow - S/K, with constant inflow over the daily step.
            When max_storage_m is configured, any stored water exceeding the
            capacity ceiling overflows to surface discharge. K is the release
            time in days; after K dry days, exp(-1) of the initial storage remains
            (about 36.8%).

        Args:
            runoff_m: Hourly runoff depths (m), per HRU.
            interflow_m: Hourly interflow depths (m), per HRU.
            soil_recharge_m: Daily soil drainage depth (m), per HRU.

        Returns:
            Daily recharge released to groundwater (m), per grid cell; captured
            runoff plus interflow (m), per HRU; captured soil drainage (m),
            per HRU; and storage overflow released to discharge (m), per grid cell.
        """
        # In karst terrain, surface runoff and interflow generated on hillslopes drain towards
        # dry valleys, sinkholes, and permeable karst depressions on non-sealed land.
        karst_presence: ArrayFloat32 = (
            self.hydrology.to_HRU(data=self.karst_fraction) > np.float32(0.0)
        ).astype(np.float32) * (
            (self.hydrology.HRU.var.land_use_type != SEALED)
            & (self.hydrology.HRU.var.land_use_type != OPEN_WATER)
        )
        total_surface_flow_m: ArrayFloat32 = runoff_m.sum(axis=0) + interflow_m.sum(
            axis=0
        )
        base_capture_fraction: np.float32 = np.float32(
            self.model.config["hydrology"]["karst"]["capture_fraction"]
        )
        max_capture_m: float | None = self.model.config["hydrology"]["karst"][
            "max_runoff_capture_m_per_day"
        ]
        if max_capture_m is not None:
            # Swallow up to max_runoff_capture_m_per_day into karst features.
            potential_capture_m: ArrayFloat32 = (
                np.minimum(
                    total_surface_flow_m * base_capture_fraction,
                    np.float32(max_capture_m),
                )
                * karst_presence
            )
            # Effective capture fraction scales with the swallowing capacity ceiling.
            surface_capture_fraction: ArrayFloat32 = np.divide(
                potential_capture_m,
                total_surface_flow_m,
                out=np.zeros_like(potential_capture_m),
                where=total_surface_flow_m > np.float32(0.0),
            )
        else:
            surface_capture_fraction = karst_presence * base_capture_fraction

        captured_runoff_m: ArrayFloat32 = (
            total_surface_flow_m * surface_capture_fraction
        )

        # Deep soil drainage percolates directly into the underlying bedrock. Where soluble
        # karst formations underlie the soil (karst_fraction > 0), unsealed soil recharge drains
        # completely into the karst aquifer system rather than being artificially restricted by surface outcrop fractions.
        captured_soil_m: ArrayFloat32 = soil_recharge_m * karst_presence

        # Remove captured water from its original route to avoid counting it twice.
        runoff_m *= np.float32(1.0) - surface_capture_fraction
        interflow_m *= np.float32(1.0) - surface_capture_fraction
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

        # If a storage capacity ceiling is configured, release excess water as overflow to river discharge.
        max_storage_m: float | None = self.model.config["hydrology"]["karst"][
            "max_storage_m"
        ]
        if max_storage_m is not None:
            overflow_m: ArrayFloat64 = np.maximum(
                storage_m - np.float64(max_storage_m), 0.0
            )
            storage_m -= overflow_m
        else:
            overflow_m = np.zeros_like(storage_m)

        return recharge_m, captured_runoff_m, captured_soil_m, overflow_m
