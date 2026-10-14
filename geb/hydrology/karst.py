"""Store karst recharge before it reaches groundwater."""

from typing import TYPE_CHECKING

import numpy as np

from geb.geb_types import ArrayFloat32, ArrayFloat64
from geb.module import Module

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

    def step(self, inflow_m: ArrayFloat64) -> ArrayFloat64:
        """Store daily inflow and return recharge to MODFLOW.

        Args:
            inflow_m: Water entering karst over the day (m), per grid cell.

        Returns:
            Water released to groundwater over the day (m), per grid cell.

        Raises:
            ValueError: If karst storage is missing from the checkpoint.
        """
        if not hasattr(self.grid.var, "karst_storage_m"):
            raise ValueError(
                "Karst storage is missing. Rerun spinup to initialize karst storage."
            )
        release_time_days: float = self.model.config["hydrology"]["karst"][
            "release_time_days"
        ]
        # Spread inflow evenly over the day; exact integration conserves water.
        release_fraction: np.float64 = -np.expm1(-1.0 / release_time_days)
        recharge_m: ArrayFloat64 = (
            self.grid.var.karst_storage_m * release_fraction
            + inflow_m * (1.0 - release_time_days * release_fraction)
        )
        self.grid.var.karst_storage_m += inflow_m - recharge_m
        return recharge_m
