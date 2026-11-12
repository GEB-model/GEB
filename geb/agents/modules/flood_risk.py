"""This module contains the FloodRiskModule class, which is responsible for loading and managing flood risk data for the households in the model. It loads building, road, and rail geometries, as well as damage curves and maximum damage values for different asset types. It also loads flood maps for different return periods to be used in flood risk calculations."""

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from geb.hydrology.landcovers import FOREST
from geb.workflows.io import read_geom, read_params, read_table, read_zarr
from geb.workflows.raster import sample_from_map

from ...workflows.damage_scanner import VectorScanner, VectorScannerMultiCurves
from ..workflows.helpers import from_landuse_raster_to_polygon

if TYPE_CHECKING:
    from geb.agents.households import Households
    from geb.model import GEBModel


class FloodRiskModule:
    """Module responsible for loading and managing flood risk data for the households in the model."""

    def __init__(self, model: GEBModel, households: Households) -> None:
        """Initialize the FloodRiskModule with the model and households, and load all necessary data.

        Args:
            model (GEBModel): The main model instance containing configuration and file paths.
            households (Households): The households agent instance where the loaded data will be stored.
        """
        self.model = model
        self.households = households
        self._flood_maps_utm_cache: dict = {}     
        self._exposure_cache: dict = {}           
        self._building_geometries_cache = None    
        self.load_damage_curves()
        self.alter_damage_curves_for_flood_proofed_buildings()
        self.load_max_damage_values()
        self.load_flood_maps()

    def load_flood_maps(self) -> None:
        """Load flood maps for different return periods. This might be quite ineffecient for RAM, but faster then loading them each timestep for now."""
        self.households.return_periods = np.array(
            self.model.config["hazards"]["floods"]["return_periods"]
        )

        flood_maps = {}
        for return_period in self.households.return_periods:
            file_path = (
                self.model.output_folder.parent
                / self.model.config["general"]["spinup_name"]
                / "flood_maps"
                / f"{return_period}.zarr"
            )
            flood_maps[return_period] = read_zarr(file_path)
        self.households.flood_maps = flood_maps

    def load_max_damage_values(self) -> None:
        """Load maximum damage values from model files and store them in the model variables."""
        # Load maximum damages
        if (
            "damage_model/flood/residential/structure/maximum_damage"
            in self.households.model.files["dict"]
        ):
            self.households.var.max_dam_buildings_structure = float(
                read_params(
                    self.households.model.files["dict"][
                        "damage_model/flood/residential/structure/maximum_damage"
                    ]
                )["maximum_damage"]
            )
            self.households.buildings["maximum_damage_m2"] = (
                self.households.var.max_dam_buildings_structure
            )
        if (
            "damage_model/flood/residential/content/maximum_damage"
            in self.households.model.files["dict"]
        ):
            max_dam_buildings_content = read_params(
                self.households.model.files["dict"][
                    "damage_model/flood/residential/content/maximum_damage"
                ]
            )
            self.households.var.max_dam_buildings_content = float(
                max_dam_buildings_content["maximum_damage"]
            )

        if (
            "damage_model/flood/rail/main/maximum_damage"
            in self.households.model.files["dict"]
        ):
            self.households.var.max_dam_rail = float(
                read_params(
                    self.households.model.files["dict"][
                        "damage_model/flood/rail/main/maximum_damage"
                    ]
                )["maximum_damage"]
            )
            self.households.rail["maximum_damage_m"] = self.households.var.max_dam_rail

        max_dam_road_m: dict[str, float] = {}
        road_types = [
            (
                "residential",
                "damage_model/flood/road/residential/maximum_damage",
            ),
            (
                "unclassified",
                "damage_model/flood/road/unclassified/maximum_damage",
            ),
            ("tertiary", "damage_model/flood/road/tertiary/maximum_damage"),
            ("primary", "damage_model/flood/road/primary/maximum_damage"),
            (
                "primary_link",
                "damage_model/flood/road/primary_link/maximum_damage",
            ),
            ("secondary", "damage_model/flood/road/secondary/maximum_damage"),
            (
                "secondary_link",
                "damage_model/flood/road/secondary_link/maximum_damage",
            ),
            ("motorway", "damage_model/flood/road/motorway/maximum_damage"),
            (
                "motorway_link",
                "damage_model/flood/road/motorway_link/maximum_damage",
            ),
            ("trunk", "damage_model/flood/road/trunk/maximum_damage"),
            ("trunk_link", "damage_model/flood/road/trunk_link/maximum_damage"),
        ]

        for road_type, path in road_types:
            if path in self.households.model.files["dict"]:
                max_dam_road_m[road_type] = read_params(
                    self.households.model.files["dict"][path]
                )["maximum_damage"]

        if not max_dam_road_m:
            print(
                "Warning: No maximum damage values found for roads. Skipping loading maximum damage for roads."
            )
        else:
            self.households.roads["maximum_damage_m"] = self.households.roads[
                "object_type"
            ].map(max_dam_road_m)

        if (
            "damage_model/flood/land_use/forest/maximum_damage"
            in self.households.model.files["dict"]
        ):
            self.households.var.max_dam_forest_m2 = float(
                read_params(
                    self.households.model.files["dict"][
                        "damage_model/flood/land_use/forest/maximum_damage"
                    ]
                )["maximum_damage"]
            )

        if (
            "damage_model/flood/land_use/agriculture/maximum_damage"
            in self.households.model.files["dict"]
        ):
            self.households.var.max_dam_agriculture_m2 = float(
                read_params(
                    self.households.model.files["dict"][
                        "damage_model/flood/land_use/agriculture/maximum_damage"
                    ]
                )["maximum_damage"]
            )

    def load_damage_curves(self) -> None:
        """Load global damage curves from model files and store them in the model variables."""
        self.households.buildings_structure_curve = read_table(
            self.households.model.files["table"][
                "damage_model/flood/residential/structure/curve"
            ]
        )
        self.households.buildings_structure_curve.set_index("depth", inplace=True)

        # now do the same for the content curve. Since there are no content curves in the global model, we use the structural curve again.
        if (
            "damage_model/flood/residential/content/curve"
            not in self.households.model.files["table"]
        ):
            self.households.buildings_content_curve = (
                self.households.buildings_structure_curve.copy()
            )
        else:
            self.households.buildings_content_curve = read_table(
                self.households.model.files["table"][
                    "damage_model/flood/residential/content/curve"
                ]
            )
            self.households.buildings_content_curve.set_index("depth", inplace=True)

        """Load damage curves from model files and store them in the model variables."""
        # Load vulnerability curves [look into these curves, some only max out at 0.5 damage ratio]
        road_curves = []
        road_types = [
            ("residential", "damage_model/flood/road/residential/curve"),
            ("unclassified", "damage_model/flood/road/unclassified/curve"),
            ("tertiary", "damage_model/flood/road/tertiary/curve"),
            ("tertiary_link", "damage_model/flood/road/tertiary_link/curve"),
            ("primary", "damage_model/flood/road/primary/curve"),
            ("primary_link", "damage_model/flood/road/primary_link/curve"),
            ("secondary", "damage_model/flood/road/secondary/curve"),
            ("secondary_link", "damage_model/flood/road/secondary_link/curve"),
            ("motorway", "damage_model/flood/road/motorway/curve"),
            ("motorway_link", "damage_model/flood/road/motorway_link/curve"),
            ("trunk", "damage_model/flood/road/trunk/curve"),
            ("trunk_link", "damage_model/flood/road/trunk_link/curve"),
        ]

        for road_type, path in road_types:
            if path not in self.households.model.files["table"]:
                continue
            df = read_table(self.households.model.files["table"][path])
            df = df.rename(columns={"damage_ratio": road_type})

            road_curves.append(df[[road_type]])

        if road_curves:
            depth_column: pd.DataFrame = df[["depth"]]

            self.households.var.road_curves = pd.concat(
                [depth_column] + road_curves, axis=1
            )
            self.households.var.road_curves.set_index("depth", inplace=True)

        if (
            "damage_model/flood/land_use/forest/curve"
            in self.households.model.files["table"]
        ):
            self.households.var.forest_curve = read_table(
                self.households.model.files["table"][
                    "damage_model/flood/land_use/forest/curve"
                ]
            )
            self.households.var.forest_curve.set_index("depth", inplace=True)
            self.households.var.forest_curve = self.households.var.forest_curve.rename(
                columns={"damage_ratio": "forest"}
            )
        if (
            "damage_model/flood/land_use/agriculture/curve"
            in self.households.model.files["table"]
        ):
            self.households.var.agriculture_curve = read_table(
                self.households.model.files["table"][
                    "damage_model/flood/land_use/agriculture/curve"
                ]
            )
            self.households.var.agriculture_curve.set_index("depth", inplace=True)
            self.households.var.agriculture_curve = (
                self.households.var.agriculture_curve.rename(
                    columns={"damage_ratio": "agriculture"}
                )
            )

        if "damage_model/flood/rail/main/curve" in self.households.model.files["table"]:
            self.households.var.rail_curve = read_table(
                self.households.model.files["table"][
                    "damage_model/flood/rail/main/curve"
                ]
            )
            self.households.var.rail_curve.set_index("depth", inplace=True)
            self.households.var.rail_curve = self.households.var.rail_curve.rename(
                columns={"damage_ratio": "rail"}
            )

    @staticmethod
    def apply_dry_flood_proofing_to_curve(
        unprotected_curve: pd.Series,
        *,
        reduction_factor: float = 0.15,
        threshold_depth_m: float = 1.0,
    ) -> pd.Series:
        """Apply a dry flood-proofing reduction to a depth-damage curve.

        Dry flood-proofing is modeled as a constant multiplicative reduction in damage
        ratios up to a threshold depth, and no reduction above that threshold.

        Args:
            unprotected_curve: Baseline depth-damage curve (damage ratio, unitless) indexed
                by inundation depth (meters).
            reduction_factor: Multiplicative factor applied at depths <= threshold (unitless).
                For an 85% reduction, use 0.15.
            threshold_depth_m: Maximum protected depth (meters). Depths above this are unchanged.

        Returns:
            Flood-proofed depth-damage curve (damage ratio, unitless), indexed by depth (meters).

        Raises:
            ValueError: If reduction_factor is outside [0, 1] or threshold_depth_m is negative.
        """
        if not (0.0 <= reduction_factor <= 1.0):
            raise ValueError("reduction_factor must be in [0, 1].")
        if threshold_depth_m < 0:
            raise ValueError("threshold_depth_m must be >= 0.")

        curve: pd.Series = unprotected_curve.astype(float).copy()
        depth_index: np.ndarray = np.asarray(curve.index, dtype=float)
        protected_mask: np.ndarray = (depth_index >= 0.0) & (depth_index <= threshold_depth_m)
        curve.iloc[np.where(protected_mask)[0]] *= reduction_factor
        return curve

    def alter_damage_curves_for_flood_proofed_buildings(self) -> None:
        """Alter the global damage curves for flood-proofed buildings by applying a reduction factor to the unprotected building curves."""
        # insert a row with depth of 1.01m and damage ratio corresponding to the damage ratio at 1m depth modeling dry flood proofing until 1m depth.
        self.households.buildings_structure_curve.loc[1.01] = (
            self.households.buildings_structure_curve.loc[1]
        )
        self.households.buildings_structure_curve = (
            self.households.buildings_structure_curve.sort_index()
        )
        # also do this for content curves
        self.households.buildings_content_curve.loc[1.01] = (
            self.households.buildings_content_curve.loc[1]
        )
        self.households.buildings_content_curve = (
            self.households.buildings_content_curve.sort_index()
        )

        # sanity check
        assert self.households.buildings_structure_curve.index.equals(
            self.households.buildings_content_curve.index
        )

        self.households.buildings_structure_curve["building_unprotected"] = (
            self.households.buildings_structure_curve["damage_ratio"]
        )
        self.households.buildings_content_curve["building_unprotected"] = (
            self.households.buildings_content_curve["damage_ratio"]
        )

        # Dry flood-proofing: reduce damages by 85% up to 1.0 m, and keep full
        # damages above 1.0 m (overtopping). The inserted 1.01 m row creates a
        # step change between 1.0 m and 1.01 m.
        self.households.buildings_structure_curve["building_flood_proofed"] = (
            self.apply_dry_flood_proofing_to_curve(
                self.households.buildings_structure_curve["building_unprotected"],
                reduction_factor=0.15,
                threshold_depth_m=1.0,
            )
        )
        self.households.buildings_content_curve["building_flood_proofed"] = (
            self.apply_dry_flood_proofing_to_curve(
                self.households.buildings_content_curve["building_unprotected"],
                reduction_factor=0.15,
                threshold_depth_m=1.0,
            )
        )

        # TODO: Need to adjust the vulnerability curves
        # create another column (curve) in the buildings structure curve for
        # protected buildings with sandbags
        self.households.buildings_structure_curve["building_with_sandbags"] = (
            self.households.buildings_structure_curve["building_unprotected"] * 0.85
        )

        # create another column (curve) in the buildings structure curve for
        # protected buildings with elevated possessions -- no effect on structure
        self.households.buildings_structure_curve["building_elevated_possessions"] = (
            self.households.buildings_structure_curve["building_unprotected"]
        )

        # create another column (curve) in the buildings structure curve for
        # protected buildings with both sandbags and elevated possessions -- only sandbags have an effect on structure
        self.households.buildings_structure_curve["building_all_forecast_based"] = (
            self.households.buildings_structure_curve["building_with_sandbags"]
        )

        # create another column (curve) in the buildings content curve for protected buildings
        self.households.buildings_content_curve["building_protected"] = (
            self.households.buildings_content_curve["building_unprotected"] * 0.7
        )

        # TODO: need to adjust the vulnerability curves
        # create another column (curve) in the buildings content curve for
        # protected buildings with sandbags
        self.households.buildings_content_curve["building_with_sandbags"] = (
            self.households.buildings_content_curve["building_unprotected"] * 0.85
        )

        # create another column (curve) in the buildings content curve for
        # protected buildings with elevated possessions
        self.households.buildings_content_curve["building_elevated_possessions"] = (
            self.households.buildings_content_curve["building_unprotected"] * 0.85
        )

        # create another column (curve) in the buildings content curve for
        # protected buildings with both sandbags and elevated possessions
        self.households.buildings_content_curve["building_all_forecast_based"] = (
            self.households.buildings_content_curve["building_unprotected"] * 0.85
        )

# CBA disabled; retained for possible future use.
#     @staticmethod
#     def compute_cba_metrics_for_return_period(
#         *,
#         damages_baseline_usd_event: np.ndarray,
#         damages_flood_proofed_usd_event: np.ndarray,
#         adapted_mask: np.ndarray,
#         adaptation_costs_usd_upfront: np.ndarray,
#         return_period_years: float,
#         loan_duration_years: int,
#     ) -> tuple[pd.DataFrame, pd.DataFrame]:
#         """Compute cost-benefit analysis metrics for a single return period.
#
#         Notes:
#             - Damages are interpreted as event losses (USD, nominal) for the given return period.
#             - "With measure" damages apply flood-proofed damages only to households where
#               `adapted_mask` is True.
#             - Annualized benefit is computed as `avoided_damage_event * (1 / return_period_years)`.
#             - Annualized cost is computed as `upfront_cost / loan_duration_years`.
#
#         Args:
#             damages_baseline_usd_event: Baseline event damages per household (USD, nominal).
#             damages_flood_proofed_usd_event: Flood-proofed event damages per household (USD, nominal).
#             adapted_mask: Boolean array indicating which households are flood-proofed.
#             adaptation_costs_usd_upfront: Upfront adaptation cost per household (USD, nominal).
#             return_period_years: Return period of the event (years).
#             loan_duration_years: Duration used to annualize costs (years).
#
#         Returns:
#             Tuple of (household-level dataframe, single-row summary dataframe).
#
#         Raises:
#             ValueError: If input arrays do not have the same length, or if parameters are invalid.
#         """
#         if return_period_years <= 0:
#             raise ValueError("return_period_years must be > 0.")
#         if loan_duration_years <= 0:
#             raise ValueError("loan_duration_years must be > 0.")
#
#         baseline: np.ndarray = np.asarray(damages_baseline_usd_event, dtype=np.float64).reshape(
#             -1
#         )
#         flood_proofed: np.ndarray = np.asarray(
#             damages_flood_proofed_usd_event, dtype=np.float64
#         ).reshape(-1)
#         adapted: np.ndarray = np.asarray(adapted_mask, dtype=bool).reshape(-1)
#         costs_upfront: np.ndarray = np.asarray(
#             adaptation_costs_usd_upfront, dtype=np.float64
#         ).reshape(-1)
#
#         n: int = int(baseline.size)
#         if not (
#             flood_proofed.size == n
#             and adapted.size == n
#             and costs_upfront.size == n
#         ):
#             raise ValueError("All input arrays must have the same length.")
#
#         with_measure: np.ndarray = baseline.copy()
#         with_measure[adapted] = flood_proofed[adapted]
#
#         avoided_event: np.ndarray = baseline - with_measure
#         aep_per_year: float = 1.0 / float(return_period_years)
#         avoided_annual: np.ndarray = avoided_event * aep_per_year
#
#         costs_selected_upfront: np.ndarray = np.where(adapted, costs_upfront, 0.0)
#         costs_selected_annual: np.ndarray = costs_selected_upfront / float(
#             loan_duration_years
#         )
#
#         net_benefit_event: np.ndarray = avoided_event - costs_selected_upfront
#         net_benefit_annual: np.ndarray = avoided_annual - costs_selected_annual
#
#         # Household-level BCRs: undefined when cost is zero.
#         bcr_event: np.ndarray = np.full(n, np.nan, dtype=np.float64)
#         bcr_annual: np.ndarray = np.full(n, np.nan, dtype=np.float64)
#         positive_cost_mask: np.ndarray = costs_selected_upfront > 0
#         bcr_event[positive_cost_mask] = (
#             avoided_event[positive_cost_mask]
#             / costs_selected_upfront[positive_cost_mask]
#         )
#         bcr_annual[positive_cost_mask] = (
#             avoided_annual[positive_cost_mask]
#             / costs_selected_annual[positive_cost_mask]
#         )
#
#         household_df: pd.DataFrame = pd.DataFrame(
#             {
#                 "household_id": np.arange(n, dtype=np.int32),
#                 "adapted": adapted,
#                 "damage_baseline_usd_event": baseline,
#                 "damage_with_measure_usd_event": with_measure,
#                 "avoided_damage_usd_event": avoided_event,
#                 "benefit_usd_annual": avoided_annual,
#                 "cost_usd_upfront": costs_selected_upfront,
#                 "cost_usd_annual": costs_selected_annual,
#                 "net_benefit_usd_event": net_benefit_event,
#                 "net_benefit_usd_annual": net_benefit_annual,
#                 "bcr_event": bcr_event,
#                 "bcr_annual": bcr_annual,
#             }
#         )
#
#         total_benefit_event: float = float(np.nansum(avoided_event))
#         total_cost_upfront: float = float(np.nansum(costs_selected_upfront))
#         total_benefit_annual: float = float(np.nansum(avoided_annual))
#         total_cost_annual: float = float(np.nansum(costs_selected_annual))
#
#         summary_df: pd.DataFrame = pd.DataFrame(
#             {
#                 "return_period_years": [float(return_period_years)],
#                 "aep_per_year": [aep_per_year],
#                 "n_households": [n],
#                 "n_adapted": [int(adapted.sum())],
#                 "damage_baseline_usd_event_total": [float(np.nansum(baseline))],
#                 "damage_with_measure_usd_event_total": [float(np.nansum(with_measure))],
#                 "avoided_damage_usd_event_total": [total_benefit_event],
#                 "benefit_usd_annual_total": [total_benefit_annual],
#                 "cost_usd_upfront_total": [total_cost_upfront],
#                 "cost_usd_annual_total": [total_cost_annual],
#                 "net_benefit_usd_event_total": [
#                     total_benefit_event - total_cost_upfront
#                 ],
#                 "net_benefit_usd_annual_total": [
#                     total_benefit_annual - total_cost_annual
#                 ],
#                 "bcr_event_total": [
#                     (total_benefit_event / total_cost_upfront)
#                     if total_cost_upfront > 0
#                     else np.nan
#                 ],
#                 "bcr_annual_total": [
#                     (total_benefit_annual / total_cost_annual)
#                     if total_cost_annual > 0
#                     else np.nan
#                 ],
#             }
#         )
#
#         return household_df, summary_df
#
#     def export_cba_for_return_period(
#         self,
#         *,
#         return_period_years: float = 100.0,
#         loan_duration_years: int = 20,
#         adapted_mask: np.ndarray | None = None,
#         export_household_level: bool = True,
#     ) -> dict[str, Path]:
#         """Export avoided damage, net benefit, and BCR for a single return period.
#
#         The export is based on the current model state:
#         - Baseline damages come from the unprotected curve.
#         - Flood-proofed damages come from the flood-proofed curve.
#         - Which households are considered flood-proofed is controlled by `adapted_mask`
#           (defaults to the model's current `households.var.adapted`).
#
#         Args:
#             return_period_years: Return period of interest (years), e.g. 100 for 1/100 AEP.
#             loan_duration_years: Duration used to annualize adaptation costs (years).
#             adapted_mask: Optional boolean mask (length = number of households) to define
#                 which households are treated as flood-proofed. If None, uses the model state.
#             export_household_level: If True, also exports a household-level parquet table.
#
#         Returns:
#             Dictionary with exported file paths.
#
#         Raises:
#             ValueError: If the return period is not available or if inputs are invalid.
#         """
#         return_periods_arr: np.ndarray = np.asarray(
#             self.households.return_periods, dtype=float
#         )
#         match_idx_arr: np.ndarray = np.where(
#             return_periods_arr == float(return_period_years)
#         )[0]
#         if match_idx_arr.size == 0:
#             available: str = ", ".join(map(str, return_periods_arr.tolist()))
#             raise ValueError(
#                 f"Return period {return_period_years} not found. Available return periods: {available}"
#             )
#         rp_index: int = int(match_idx_arr[0])
#
#         damages_do_not_adapt, damages_adapt = self.calculate_building_flood_damages(
#             verbose=False,
#             export_building_damages=False,
#             dynamic=True,
#         )
#
#         n_households: int = int(self.households.n)
#         if adapted_mask is None:
#             adapted_mask_arr: np.ndarray = (
#                 np.asarray(self.households.var.adapted.data[:n_households]).reshape(-1)
#                 == 1
#             )
#         else:
#             adapted_mask_arr = np.asarray(adapted_mask, dtype=bool).reshape(-1)
#             if adapted_mask_arr.size != n_households:
#                 raise ValueError(
#                     "adapted_mask must have length equal to the number of households."
#                 )
#
#         adaptation_costs_usd_upfront: np.ndarray = np.asarray(
#             self.households.var.adaptation_costs.data[:n_households], dtype=np.float64
#         ).reshape(-1)
#
#         household_df, summary_df = self.compute_cba_metrics_for_return_period(
#             damages_baseline_usd_event=damages_do_not_adapt[rp_index, :],
#             damages_flood_proofed_usd_event=damages_adapt[rp_index, :],
#             adapted_mask=adapted_mask_arr,
#             adaptation_costs_usd_upfront=adaptation_costs_usd_upfront,
#             return_period_years=float(return_period_years),
#             loan_duration_years=int(loan_duration_years),
#         )
#
#         out_dir: Path = self.households.model.output_folder / "cba"
#         out_dir.mkdir(parents=True, exist_ok=True)
#         year: int = int(self.households.model.current_time.year)
#
#         summary_path: Path = (
#             out_dir / f"cba_summary_rp{return_period_years:g}_{year}.csv"
#         )
#         summary_df.to_csv(summary_path, index=False)
#
#         exported: dict[str, Path] = {"summary_csv": summary_path}
#         if export_household_level:
#             hh_path: Path = (
#                 out_dir / f"cba_households_rp{return_period_years:g}_{year}.parquet"
#             )
#             household_df.to_parquet(hh_path, index=False)
#             exported["households_parquet"] = hh_path
#
#         return exported
#
    def calculate_building_flood_damages(
        self,
        verbose: bool = True,
        export_building_damages: bool = False,
        dynamic: bool = True,
        building_damage_cache_dir: Path | str | None = None, #export (save file) addition
    ) -> tuple[np.ndarray, np.ndarray]:
        """This function calculates the flood damages for the households in the model.

        It iterates over the return periods and calculates the damages for each household
        based on the flood maps and the building footprints.

        Args:
            verbose: Verbosity flag.
            export_building_damages: Whether to export the building damages to parquet files.
            dynamic: Whether to calculate damages dynamically based on the current flood maps in the model (as opposed to using flood maps at t=0).
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the damage arrays for unprotected and protected buildings.
        Raises:
            RuntimeError: If the damage arrays do not match the expected shape based on return periods and number of households.
        """
        if (
            not dynamic
            and hasattr(self, "damages_do_not_adapt")
            and hasattr(self, "damages_adapt")
        ):
            expected_shape = (
                self.households.return_periods.size,
                self.households.n,
            )
            if (
                self.damages_do_not_adapt.shape != expected_shape
                or self.damages_adapt.shape != expected_shape
            ):
                raise RuntimeError(
                    "Damages array shape does not match the expected shape based on return periods and number of households. "
                    "If household relocation is modeled, damages must be calculated dynamically. "
                    f"Expected {expected_shape}, got do_not_adapt={self.damages_do_not_adapt.shape}, adapt={self.damages_adapt.shape}."
                )
            return self.damages_do_not_adapt, self.damages_adapt

        #inundadtion variable
        n_return_periods = self.households.return_periods.size
        n_households = self.households.n
        #end inundation variable part 1

        damages_do_not_adapt = np.zeros(
            (self.households.return_periods.size, self.households.n), np.float32
        )
        damages_adapt = np.zeros(
            (self.households.return_periods.size, self.households.n), np.float32
        )

        #inundation variable part 2
        inundation = np.full(
            (n_return_periods, n_households),
            np.nan,
            dtype=np.float32,
        )
        #end inundation variable part 2

        #LECZ version
        # lecz_mask = self.households.var.in_lecz.data ==1
        # #Buildings belonging to LECZ
        # lecz_building_ids = np.unique(
        #     self.households.var.building_id_of_household[lecz_mask]
        # )
        # #Agent dataframe for LECZ
        # agent_df = pd.DataFrame(
        #     {
        #         "building_id_of_household":
        #             self.households.var.building_id_of_household[lecz_mask]
        #     }
        # )

        lecz_mask = self.households.var.in_lecz.data[: self.households.n] == 1
        lecz_building_ids = np.unique(
            self.households.var.building_id_of_household.data[: self.households.n][lecz_mask]
        )
        agent_df = pd.DataFrame(
            { 
                "building_id_of_household":
                    self.households.var.building_id_of_household.data[: self.households.n][lecz_mask]
            }
        )

        if building_damage_cache_dir is not None and not export_building_damages:
            cache_dir = Path(building_damage_cache_dir)

            for i, return_period in enumerate(self.households.return_periods):
                cache_file = cache_dir / f"flood_rp{int(return_period)}.parquet"
                if not cache_file.is_file():
                    raise FileNotFoundError(
                        f"Precomputed flood damage file not found: {cache_file}"
                    )

                cached = pd.read_parquet(cache_file)
                required = {"id", "damages", "damages_flood_proofed", "inundation"}
                missing = required.difference(cached.columns)
                if missing:
                    raise ValueError(
                        f"{cache_file} is missing columns: {sorted(missing)}"
                    )

                unprotected, adapted = self.households.assign_damages_to_agents(
                    agent_df,
                    cached[["id", "damages", "damages_flood_proofed"]],
                )
                damages_do_not_adapt[i, lecz_mask] = unprotected
                damages_adapt[i, lecz_mask] = adapted

                inundation_by_building = cached[["id", "inundation"]].rename(
                    columns={"id": "building_id"}
                )
                household_inundation = (
                    agent_df.merge(
                        inundation_by_building,
                        left_on="building_id_of_household",
                        right_on="building_id",
                        how="left",
                    )["inundation"].to_numpy(dtype=np.float32)
                )
                inundation[i, lecz_mask] = household_inundation

            if not dynamic:
                self.damages_do_not_adapt = damages_do_not_adapt
                self.damages_adapt = damages_adapt

            self.inundation = inundation
            return damages_do_not_adapt, damages_adapt



        buildings = self.households.buildings[
            self.households.buildings["flooded"]
            & (self.households.buildings["id"].isin(lecz_building_ids))
        ].copy()

        # # DataFrame mapping buildings to household agents all households
        # agent_df = pd.DataFrame(
        #     {"building_id_of_household": self.households.var.building_id_of_household}
        # )

        # # # subset building to those exposed to flooding
        # buildings = self.households.buildings[
        #     self.households.buildings["flooded"]
        # ].copy()
        #LECZ version
        # buildings = self.households.buildings[(
        #     self.households.buildings["flooded"])
        #     & (self.households.buildings["id"].isin(lecz_building_ids))
        #     ].copy()
        
        flooded_building_ids = np.array(buildings["id"])

        # Cache building geometries — they don't change between years.
        # Invalidate if the flooded building set changes (e.g. after a new flood event).
        current_flooded_key = tuple(sorted(flooded_building_ids))
        if (
            self._building_geometries_cache is None
            or self._building_geometries_cache["key"] != current_flooded_key
        ):
            building_geometries = read_geom(
                self.households.model.files["geom"]["assets/open_building_map"],
                filters=[("id", "in", flooded_building_ids)],
            )
            building_geometries = building_geometries.merge(
                buildings[["id", "object_type"]],
                on="id",
                how="left",
            )
            utm_flood_map = self.reproject_to_utm(
                self.households.flood_maps[self.households.return_periods[0]]
            )
            building_geometries = building_geometries.to_crs(utm_flood_map.rio.crs)
            self._building_geometries_cache = {"key": current_flooded_key, "data": building_geometries}
            # Invalidate exposure cache when building set changes
            self._exposure_cache.clear()
        else:
            building_geometries = self._building_geometries_cache["data"]

        # Damage curves are the same for every return period — build once.
        multi_curves = {
            "damages_structure": self.households.buildings_structure_curve[
                "building_unprotected"
            ],
            "damages_content": self.households.buildings_content_curve[
                "building_unprotected"
            ],
            "damages_structure_flood_proofed": self.households.buildings_structure_curve[
                "building_flood_proofed"
            ],
            "damages_content_flood_proofed": self.households.buildings_content_curve[
                "building_flood_proofed"
            ],
        }

        # Rename cost columns once before the loop — column names never change
        # between return periods, so doing this N_rp times was wasteful.
        building_features = building_geometries.rename(
            columns={
                "COST_STRUCTURAL_USD_SQM": "maximum_damage_structure",
                "COST_CONTENTS_USD_SQM": "maximum_damage_content",
            }
        )

        for i, return_period in enumerate(self.households.return_periods):
            # Cache reprojected flood maps — they never change.
            if return_period not in self._flood_maps_utm_cache:
                self._flood_maps_utm_cache[return_period] = self.reproject_to_utm(
                    self.households.flood_maps[return_period]
                )
            flood_map: xr.DataArray = self._flood_maps_utm_cache[return_period]

            damage_buildings: pd.DataFrame = VectorScannerMultiCurves(
                features=building_features,
                hazard=flood_map,
                multi_curves=multi_curves,
                exposure_cache=self._exposure_cache,
                cache_key=return_period,
            )

            # sum structure and content damages
            damage_buildings["damages"] = (
                damage_buildings["damages_structure"]
                + damage_buildings["damages_content"]
            )
            damage_buildings["damages_flood_proofed"] = (
                damage_buildings["damages_structure_flood_proofed"]
                + damage_buildings["damages_content_flood_proofed"]
            )
            #inundation variable part 3
            centroids = building_geometries.geometry.centroid

            x_coords = xr.DataArray(
                centroids.x.to_numpy(),
                dims="building",
            )
            y_coords = xr.DataArray(
                centroids.y.to_numpy(),
                dims="building",
            )

            building_inundation = (
                flood_map
                .sel(
                    x=x_coords,
                    y=y_coords,
                    method="nearest",
                )
                .to_numpy()
                .astype(np.float32)
            )

            # building_inundation = np.asarray(
            #     building_inundation
            # ).reshape(-1)
            if building_inundation.shape[0] != len(building_geometries):
                raise RuntimeError(
                    "Number of inundation values does not match the number "
                    f"of buildings. Got {building_inundation.shape[0]}"
                    f"inundation values for {len(building_geometries)} buildings."
                )

            inundation_by_building = pd.DataFrame(
                {
                    "building_id": building_geometries["id"].to_numpy(),
                    "inundation": building_inundation,
                }
            )
            inundation_household_lecz = (
                agent_df
                .merge(
                    inundation_by_building,
                    left_on="building_id_of_household",
                    right_on="building_id",
                    how="left",
                )["inundation"]
                .to_numpy(dtype=np.float32)
            )

            #without inundation variable
            # if export_building_damages:
            #     fn_for_export = self.households.model.output_folder / "building_damages"
            #     fn_for_export.mkdir(parents=True, exist_ok=True)
            #     pd.concat([building_geometries, damage_buildings], axis=1).to_parquet(
            #         fn_for_export
            #         / f"building_damages_rp{return_period}_{self.households.model.current_time.year}.parquet"
            #     )

            if inundation_household_lecz.shape != (lecz_mask.sum(),):
                raise RuntimeError(
                    "Household inundation array has an unexpected shape. "
                    f"Expected {(lecz_mask.sum(),)}, "
                    f"got {inundation_household_lecz.shape}"
                )

            inundation[i, lecz_mask] = inundation_household_lecz

            # #with inundation variable (without saving file)
            # if export_building_damages:

            #     fn_for_export = (
            #         self.households.model.output_folder
            #         / "building_damages"
            #     )

            #     fn_for_export.mkdir(
            #         parents=True,
            #         exist_ok=True,
            #     )

            #     export_df = building_geometries.copy()
            #     for column in damage_buildings.columns:
            #         export_df[column] = damage_buildings[column].to_numpy()

            #     export_df["inundation"] = building_inundation

            #     export_df.to_parquet(
            #         fn_for_export
            #         / f"building_damages_rp{return_period}_{self.households.model.current_time.year}.parquet"
            #     )

            #export (save file) addition
            if export_building_damages:
                if building_damage_cache_dir is None:
                    raise ValueError("building_damage_cache_dir is required for export")

                cache_dir = Path(building_damage_cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)

                export_df = pd.DataFrame({
                    "id": building_geometries["id"].to_numpy(),
                    "damages": damage_buildings["damages"].to_numpy(),
                    "damages_flood_proofed": damage_buildings[
                        "damages_flood_proofed"
                    ].to_numpy(),
                    "inundation": building_inundation,
                })
                export_df.to_parquet(
                    cache_dir / f"flood_rp{int(return_period)}.parquet",
                    index=False,
                )

            # Build output directly — no copy/concat needed since damage_buildings
            # is already aligned with building_geometries by index.
            out = pd.DataFrame({
                "id": building_geometries["id"].values,
                "damages": damage_buildings["damages"].values,
                "damages_flood_proofed": damage_buildings["damages_flood_proofed"].values,
            })
            # damages_do_not_adapt[i], damages_adapt[i] = (
            #     self.households.assign_damages_to_agents(agent_df, out)
            # )

            #LECZ version
            # damages only for LECZ households
            damages_do_not_adapt_lecz, damages_adapt_lecz = (
                self.households.assign_damages_to_agents(agent_df, out)
            )

            # # insert into full household arrays
            damages_do_not_adapt[i, lecz_mask] = damages_do_not_adapt_lecz
            damages_adapt[i, lecz_mask] = damages_adapt_lecz

            # inundation_household_lecz, _ = (
            #     self.households.assign_damages_to_agents(
            #         agent_df,
            #         out.rename(
            #             columns={
            #                 "inundation": "damages"
            #             }
            #         ),
            #     )
            #)

            # inundation[i, lecz_mask] = inundation_household_lecz
            flooded_households = np.sum(inundation_household_lecz > 0)

            households_05_to_1m = np.sum(
                (inundation_household_lecz >= 0.5)
                & (inundation_household_lecz < 1.0)
            )

            fraction_05_to_1m = households_05_to_1m / flooded_households


            if verbose:
                print(
                    f"Damages rp{return_period}: {round(damages_do_not_adapt[i].sum() / 1e6)} million"
                )
                print(
                    f"Damages adapt rp{return_period}: {round(damages_adapt[i].sum() / 1e6)} million"
                )

                print(
                    f"Fraction of flooded households with inundation >=0.5 m and <1.0 m: "
                    f"{fraction_05_to_1m:.2%} "
                    f"({households_05_to_1m:,} / {flooded_households:,})"
                )
        
        if not dynamic:
            self.damages_do_not_adapt = damages_do_not_adapt
            self.damages_adapt = damages_adapt
        
        self.inundation = inundation
        return damages_do_not_adapt, damages_adapt

    @staticmethod
    def reproject_to_utm(hazard:xr.DataArray) -> xr.DataArray:
        """Reproject the hazard map to a metric (UTM) CRS if it is in a geographic CRS."""
        if not hazard.rio.crs.is_geographic:
            return hazard
        bounds = hazard.rio.bounds()
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        utm_zone = int((center_lon + 180)/6) + 1
        utm_epsg = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
        return hazard.rio.reproject(f"EPSG:{utm_epsg}")
    
    def calculate_ead(
        self,
        damages_do_not_adapt: np.ndarray,
        damages_adapt: np.ndarray,
        adapted: np.ndarray,
    ) -> np.ndarray:
        """Calculate the Expected Annual Damages (EAD) based on the damages for different return periods.

        Args:
            damages_do_not_adapt: A multi-dimensional numpy array containing damages for different return periods and agents.
            damages_adapt: A multi-dimensional numpy array containing adapted damages for different return periods and agents.
            adapted: A boolean numpy array indicating which agents have adapted.
        Returns:
            A 1D numpy array containing the EAD for each agent.
        """
        # Copy baseline damages
        all_damages = damages_do_not_adapt.copy()

        # Replace adapted households with adapted damages
        adapted_mask = adapted.astype(bool)
        all_damages[:, adapted_mask] = damages_adapt[:, adapted_mask]
        # Sort probabilities in ascending order for integration
        probabilities = 1 / self.households.return_periods
        sort_idx = np.argsort(probabilities)

        prob_sorted = probabilities[sort_idx]
        damages_sorted = all_damages[sort_idx, :]

        # Calculate Expected Annual Damage (EAD)
        ead_usd_per_year = np.trapezoid(y=damages_sorted, x=prob_sorted, axis=0)

        return ead_usd_per_year

    def flood(self, flood_depth: xr.DataArray) -> float:
        """This function computes the damages for the assets and land use types in the model.

        Args:
            flood_depth: The flood map containing water levels for the flood event [m].

        Returns:
            The total flood damages for the event for all assets and land use types.

        """
        if self.model.config["hazards"]["floods"]["damage_model"] == "global":
            raise NotImplementedError(
                "The flood function is not implemented for the global damage model yet."
            )

        flood_depth: xr.DataArray = self.reproject_to_utm(flood_depth.compute())

        # subset building to those exposed to flooding
        buildings_centroids = gpd.GeoDataFrame(
            self.households.buildings,
            geometry=gpd.points_from_xy(
                self.households.buildings["x"], self.households.buildings["y"]
            ),
            crs="EPSG:4326",
        )

        # reproject centroids to the flood raster CRS 
        buildings_centroids = buildings_centroids.to_crs(flood_depth.rio.crs)

        # extract centroid coordinates in raster CRS
        x_coords = buildings_centroids.geometry.x.values
        y_coords = buildings_centroids.geometry.y.values

        # sample raster at building centroids using nearest-neighbour interpolation
        x_dim = flood_depth.rio.x_dim
        y_dim = flood_depth.rio.y_dim
        sampled_depths = flood_depth.interp(
            {x_dim: ("points", x_coords), y_dim: ("points", y_coords)},
            method="nearest",
        )

        # attach sampled depths to building points; buildings with NaN depth are not flooded
        building_points_with_depth = buildings_centroids.copy()
        building_points_with_depth["depth"] = sampled_depths.values
        flooded_building_ids = building_points_with_depth[
            ~building_points_with_depth["depth"].isna()
        ]["id"].unique()

        building_geometries = read_geom(
            self.households.model.files["geom"]["assets/open_building_map"],
            filters=[("id", "in", flooded_building_ids)],
        )

        # merge geometry into buildings dataframe
        buildings = self.households.buildings.merge(
            building_geometries[["id", "geometry"]],
            on="id",
            how="left",
        )

        # convert to GeoDataFrame
        buildings = gpd.GeoDataFrame(
            buildings, geometry="geometry", crs=building_geometries.crs
        )

        # reproject
        buildings = buildings.to_crs(flood_depth.rio.crs)

        household_points: gpd.GeoDataFrame = (
            self.households.var.household_points.copy().to_crs(flood_depth.rio.crs)
        )

        if self.households.model.config["agent_settings"]["households"][
            "warning_response"
        ]:
            # make sure household points and actions taken have the same length
            assert len(household_points) == self.households.var.actions_taken.shape[0]

            # add columns for protective actions
            household_points["sandbags"] = False
            household_points["elevated_possessions"] = False

            # mark households that took protective actions
            household_points.loc[
                np.asarray(self.households.var.actions_taken)[:, 0] == 1,
                "elevated_possessions",
            ] = True
            household_points.loc[
                np.asarray(self.households.var.actions_taken)[:, 1] == 1, "sandbags"
            ] = True

            # spatial join to get household attributes to buildings
            buildings: gpd.GeoDataFrame = gpd.sjoin_nearest(
                buildings, household_points, how="left", exclusive=True
            )

            # Assign object types for buildings based on protective measures taken
            buildings["object_type"] = "building_unprotected"  # reset
            buildings.loc[buildings["elevated_possessions"], "object_type"] = (
                "building_elevated_possessions"
            )
            buildings.loc[buildings["sandbags"], "object_type"] = (
                "building_with_sandbags"
            )
            buildings.loc[
                buildings["elevated_possessions"] & buildings["sandbags"], "object_type"
            ] = "building_all_forecast_based"
            # TODO: need to move the update of the actions takens by households to outside the flood function

            # Save the buildings with actions taken
            output_path = (
                self.households.model.output_folder
                / "action_maps"
                / "buildings_with_protective_measures.geoparquet"
            )
            # Ensure the action_maps directory exists before writing the file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            buildings.to_parquet(output_path)

            # Assign object types for buildings centroid based on protective measures taken
            buildings_centroid = household_points.to_crs(flood_depth.rio.crs)
            buildings_centroid["object_type"] = np.select(
                [
                    (
                        buildings_centroid["elevated_possessions"]
                        & buildings_centroid["sandbags"]
                    ),
                    buildings_centroid["elevated_possessions"],
                    buildings_centroid["sandbags"],
                ],
                [
                    "building_all_forecast_based",
                    "building_elevated_possessions",
                    "building_with_sandbags",
                ],
                default="building_unprotected",
            )
            buildings_centroid["maximum_damage"] = (
                self.households.var.max_dam_buildings_content
            )

        if self.households.config["adapt"]:
            household_points["building_id"] = (
                self.households.var.building_id_of_household
            )  # first assign building id to household points gdf
            household_points: gpd.GeoDataFrame = household_points.merge(
                buildings[["id", "flood_proofed"]],
                left_on="building_id",
                right_on="id",
                how="left",
            )  # now merge to get flood proofed status  # ty:ignore[invalid-assignment]

            buildings_centroid = household_points.to_crs(flood_depth.rio.crs)

            buildings_centroid["maximum_damage"] = (
                self.households.var.max_dam_buildings_content
            )

            buildings["object_type"] = np.where(
                buildings["flood_proofed"],
                "building_flood_proofed",
                "building_unprotected",
            )

            buildings_centroid["object_type"] = np.where(
                buildings_centroid["flood_proofed"],
                "building_protected",
                "building_unprotected",
            )

        else:
            household_points["protect_building"] = False

            buildings: gpd.GeoDataFrame = gpd.sjoin_nearest(
                buildings, household_points, how="left", exclusive=True
            )

            buildings["object_type"] = "building_unprotected"

            # Right now there is no condition to make the households protect their buildings outside of the warning response
            buildings.loc[buildings["protect_building"], "object_type"] = (
                "building_protected"
            )

            buildings_centroid = household_points.to_crs(flood_depth.rio.crs)
            buildings_centroid["object_type"] = buildings_centroid[
                "protect_building"
            ].apply(lambda x: "building_protected" if x else "building_unprotected")
            buildings_centroid["maximum_damage"] = (
                self.households.var.max_dam_buildings_content
            )

        # Create the folder to save damage maps if it doesn't exist
        damage_folder: Path = self.households.model.output_folder / "damage_maps"
        damage_folder.mkdir(parents=True, exist_ok=True)

        damages_buildings_content = VectorScanner(
            features=buildings_centroid,
            hazard=flood_depth,
            vulnerability_curves=self.households.buildings_content_curve,
        )

        total_damages_content = damages_buildings_content.sum()

        # save it to a gpkg file
        gdf_content = buildings_centroid.copy()
        gdf_content["damage"] = damages_buildings_content
        category_name: str = "buildings_content"
        filename: str = f"damage_map_{category_name}.gpkg"
        gdf_content.to_file(damage_folder / filename, driver="GPKG")

        print(f"damages to building content are: {total_damages_content}")

        # Compute damages for buildings structure
        damages_buildings_structure: pd.Series = VectorScanner(
            features=buildings.rename(columns={"maximum_damage_m2": "maximum_damage"}),  # ty:ignore[invalid-argument-type]
            hazard=flood_depth,
            vulnerability_curves=self.households.buildings_structure_curve,
        )

        total_damage_structure = damages_buildings_structure.sum()

        print(f"damages to building structure are: {total_damage_structure}")

        # save it to a gpkg file
        gdf_structure = buildings.copy()
        gdf_structure["damage"] = damages_buildings_structure
        category_name: str = "buildings_structure"
        filename: str = f"damage_map_{category_name}.gpkg"
        gdf_structure.to_file(damage_folder / filename, driver="GPKG")

        print(
            f"Total damages to buildings are: {total_damages_content + total_damage_structure}"
        )

        agriculture = from_landuse_raster_to_polygon(
            self.households.HRU.decompress(self.households.HRU.var.land_owners != -1),
            self.households.HRU.transform,
            self.households.model.crs,
        )
        agriculture["object_type"] = "agriculture"
        agriculture["maximum_damage"] = self.households.var.max_dam_agriculture_m2

        agriculture = agriculture.to_crs(flood_depth.rio.crs)

        damages_agriculture = VectorScanner(
            features=agriculture,
            hazard=flood_depth,
            vulnerability_curves=self.households.var.agriculture_curve,
        )
        total_damages_agriculture = damages_agriculture.sum()
        print(f"damages to agriculture are: {total_damages_agriculture}")

        # Load landuse and make turn into polygons
        forest = from_landuse_raster_to_polygon(
            self.households.HRU.decompress(
                self.households.HRU.var.land_use_type == FOREST
            ),
            self.households.HRU.transform,
            self.households.model.crs,
        )
        forest["object_type"] = "forest"
        forest["maximum_damage"] = self.households.var.max_dam_forest_m2

        forest = forest.to_crs(flood_depth.rio.crs)

        damages_forest = VectorScanner(
            features=forest,
            hazard=flood_depth,
            vulnerability_curves=self.households.var.forest_curve,
        )
        total_damages_forest = damages_forest.sum()
        print(f"damages to forest are: {total_damages_forest}")

        roads = self.households.roads.to_crs(flood_depth.rio.crs)
        damages_roads = VectorScanner(
            features=roads.rename(columns={"maximum_damage_m": "maximum_damage"}),  # ty:ignore[invalid-argument-type]
            hazard=flood_depth,
            vulnerability_curves=self.households.var.road_curves,
        )
        total_damages_roads = damages_roads.sum()
        print(f"damages to roads are: {total_damages_roads} ")

        rail = self.households.rail.to_crs(flood_depth.rio.crs)
        damages_rail = VectorScanner(
            features=rail.rename(columns={"maximum_damage_m": "maximum_damage"}),  # ty:ignore[invalid-argument-type]
            hazard=flood_depth,
            vulnerability_curves=self.households.var.rail_curve,
        )
        total_damages_rail = damages_rail.sum()
        print(f"damages to rail are: {total_damages_rail}")

        total_flood_damages = (
            total_damage_structure
            + total_damages_content
            + total_damages_roads
            + total_damages_rail
            + total_damages_forest
            + total_damages_agriculture
        )
        print(f"the total flood damages are: {total_flood_damages}")

        return total_flood_damages

    # def return_period_flood(self, flood_protection_standard: int = 10) -> np.ndarray:
    #     """Simulate a flood event based on return periods and determine which households are flooded.

    #     Returns:
    #         Array of indices of flooded households.
    #     """
    #     # draw a single random number
    #     p_random = np.random.random()
    #     # Work with a locally sorted copy of return periods to ensure correct event selection
    #     return_periods_arr = np.asarray(self.households.return_periods, dtype=float)
    #     sort_idx = np.argsort(return_periods_arr)  # ascending order
    #     sorted_return_periods = return_periods_arr[sort_idx]
    #     probabilities = 1.0 / sorted_return_periods

    #     if p_random >= probabilities.max() or p_random >= 1 / flood_protection_standard:
    #         return np.array([], dtype=int)

    #     # find the event corresponding to the random draw
    #     event_idx = np.searchsorted(probabilities[::-1], p_random)
    #     event_idx = len(probabilities) - 1 - event_idx
    #     event = sorted_return_periods[event_idx]
    #     self.model.logger.info(
    #         "Return period flood event: %s years (p=%.4f, random draw=%.4f)",
    #         event,
    #         probabilities[event_idx],
    #         p_random,
    #     )

    #     # get the flood map for this event
    #     flood_map: xr.DataArray = self.households.flood_maps[event]

    #     # cache household coordinates in flood_map CRS (Nx2 numpy array)
    #     if not hasattr(self, "_household_xy"):
    #         import pyproj

    #         x, y = (
    #             np.array(self.households.buildings.x),
    #             np.array(self.households.buildings.y),
    #         )
    #         transformer = pyproj.Transformer.from_crs(
    #             "EPSG:4326", flood_map.rio.crs, always_xy=True
    #         )
    #         self._building_xy = np.array(transformer.transform(x, y)).T

    #     # sample flood map using clipped coordinates
    #     sampled_values = sample_from_map(
    #         array=flood_map.values,
    #         coords=self._building_xy,
    #         gt=flood_map.rio.transform(recalc=True).to_gdal(),
    #         out_of_bounds_value=np.nan,
    #     )
    #     # Use the same minimum flood depth threshold (0.05 m) as elsewhere in the model
    #     minimum_flood_depth_m = 0.05
    #     # np.where will return indices of flooded households relative to the original household array
    #     flooded_building_indices = np.where(sampled_values > minimum_flood_depth_m)[0]

    #     # get building IDs of flooded buildings
    #     flooded_building_ids = self.households.buildings.loc[
    #         flooded_building_indices, "id"
    #     ].values.astype(int)

    #     # get indices of households located in flooded buildings
    #     flooded_household_indices = np.where(
    #         np.isin(
    #             self.households.var.building_id_of_household.data, flooded_building_ids
    #         )
    #     )[0]

    #     return flooded_household_indices
    def return_period_flood(self, flood_protection_standard: int = 10) -> np.ndarray:
        """Simulate a flood event and return flooded household indices."""
        forced_date = pd.Timestamp("2021-01-01")
        current_time = pd.Timestamp(self.model.current_time)

        if current_time != forced_date:
            return np.array([], dtype=int)

        event = 100
        self.model.logger.info("Forcing return period flood event: %s years", event)
        
        # Shared logic: run for both the forced event and random events.
        flood_map: xr.DataArray = self.households.flood_maps[event]

        if not hasattr(self, "_building_xy"):
            import pyproj

            x = np.array(self.households.buildings.x)
            y = np.array(self.households.buildings.y)
            transformer = pyproj.Transformer.from_crs(
                "EPSG:4326", flood_map.rio.crs, always_xy=True
            )
            self._building_xy = np.array(transformer.transform(x, y)).T

        sampled_values = sample_from_map(
            array=flood_map.values,
            coords=self._building_xy,
            gt=flood_map.rio.transform(recalc=True).to_gdal(),
            out_of_bounds_value=np.nan,
        )

        minimum_flood_depth_m = 0.05
        flooded_building_indices = np.where(sampled_values > minimum_flood_depth_m)[0]

        flooded_building_ids = self.households.buildings.loc[
            flooded_building_indices, "id"
        ].values.astype(int)

        flooded_household_indices = np.where(
            np.isin(
                self.households.var.building_id_of_household.data,
                flooded_building_ids,
            )
        )[0]

        return flooded_household_indices