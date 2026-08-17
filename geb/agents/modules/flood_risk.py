"""This module contains the FloodRiskModule class, which is responsible for loading and managing flood risk data for the households in the model. It loads building, road, and rail geometries, as well as damage curves and maximum damage values for different asset types. It also loads flood maps for different return periods to be used in flood risk calculations."""

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from geb.hydrology.landcovers import FOREST
from geb.workflows.io import read_geom, read_params, read_table, read_zarr
from geb.workflows.raster import coords_to_pixels

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
        self.load_damage_curves()
        self.alter_damage_curves_based_on_actions()
        self.load_max_damage_values()
        if (
            self.model.config["hazards"]["floods"]["flood_risk"]
            or self.model.config["agent_settings"]["households"]["adapt"]
        ):
            self.load_return_period_flood_maps()
            self.load_flood_protection_standard()
            self.flood_in_last_year = False

    def load_flood_protection_standard(self) -> None:
        """Load flood protection standards for each subbasin.

        Raises:
            ValueError: If the flood protection standard mode is not 'manual' or 'auto'.
        """
        mode = self.model.config["hazards"]["floods"]["flood_protection_standard"][
            "mode"
        ]
        self.flood_protection_standard_subbasins = {}
        if mode == "manual":
            manual_value = self.model.config["hazards"]["floods"][
                "flood_protection_standard"
            ]["manual_value"]
            if manual_value is None:
                raise ValueError(
                    "Flood protection standard mode is 'manual' but 'manual_value' is null."
                )

            supported_return_periods = self.model.config["hazards"]["floods"][
                "return_periods"
            ]
            if manual_value not in supported_return_periods:
                raise ValueError(
                    f"Manual flood protection standard {manual_value} is not in hazards.floods.return_periods {supported_return_periods}."
                )
            self.model.logger.info(
                f"Flood protection standard set to {manual_value} years for all subbasins."
            )

            for comid in np.unique(self.households.buildings["COMID"]):
                self.flood_protection_standard_subbasins[comid] = manual_value
            return
        elif mode == "auto":
            self.model.logger.info(
                "Flood protection standard set to 'auto'. Flood protection standards will be automatically determined."
            )
            flood_protection_standards = pd.read_parquet(
                self.model.files["table"][
                    "flood_protection_standards/flood_protection_standards"
                ]
            )

            for comid in np.unique(self.households.buildings["COMID"]):
                if comid not in flood_protection_standards.index:
                    flood_protection_standard = flood_protection_standards[
                        "flood_protection_standard"
                    ].mode()[0]
                    self.model.logger.warning(
                        f"COMID {comid} not found in flood protection standards table. Using mode flood protection standard {flood_protection_standard}."
                    )

                else:
                    flood_protection_standard = flood_protection_standards.loc[
                        comid, "flood_protection_standard"
                    ]
                if flood_protection_standard == 0:
                    self.model.logger.warning(
                        f"COMID {comid} has a flood protection standard of 0. Value might be missing from flood protection standards table."
                    )
                    flood_protection_standard = 2
                # truncate to closest return period if not in return periods
                if flood_protection_standard not in self.households.return_periods:
                    closest_return_period = min(
                        self.households.return_periods,
                        key=lambda x: abs(x - flood_protection_standard),
                    )
                    flood_protection_standard = closest_return_period
                self.flood_protection_standard_subbasins[comid] = (
                    flood_protection_standard
                )
        else:
            raise ValueError(f"Invalid flood protection standard mode: {mode}")

    def load_return_period_flood_maps(self) -> None:
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

    def alter_damage_curves_based_on_actions(self) -> None:
        """Alter the global damage curves for flood-proofed buildings by applying a reduction factor to the unprotected building curves."""
        damage_reduction_over_leadtime = self.households.model.config["agent_settings"][
            "households"
        ]["warning_system"]["damage_reduction_over_leadtime"]
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

        # create another column (curve) in the buildings structure curve for flood-proofed buildings
        self.households.buildings_structure_curve["building_flood_proofed"] = (
            self.households.buildings_structure_curve["damage_ratio"]
        )
        self.households.buildings_structure_curve.loc[
            0:1, "building_flood_proofed"
        ] *= 0.15

        # create another column (curve) in the buildings content curve for flood-proofed buildings
        self.households.buildings_content_curve["building_flood_proofed"] = (
            self.households.buildings_content_curve["damage_ratio"]
        )

        self.households.buildings_content_curve.loc[0:1, "building_flood_proofed"] *= (
            0.15
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

        # create another column (curve) in the buildings structure curve for flood-proofed buildings
        self.households.buildings_structure_curve["building_flood_proofed"] = (
            self.households.buildings_structure_curve["building_unprotected"] * 0.85
        )
        self.households.buildings_structure_curve.loc[0:1, "building_flood_proofed"] = (
            0.0
        )

        # create another column (curve) in the buildings content curve for protected buildings
        self.households.buildings_content_curve["building_protected"] = (
            self.households.buildings_content_curve["building_unprotected"] * 0.7
        )
        # create another column (curve) in the buildings content curve for flood-proofed buildings
        self.households.buildings_content_curve["building_flood_proofed"] = (
            self.households.buildings_content_curve["building_unprotected"] * 0.85
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

        if damage_reduction_over_leadtime:
            # create timing-based structure curves for elevated possessions - no effect on structure
            self.households.buildings_structure_curve[
                "building_elevated_possessions_early"
            ] = self.households.buildings_structure_curve["building_unprotected"]
            self.households.buildings_structure_curve[
                "building_elevated_possessions_medium"
            ] = self.households.buildings_structure_curve["building_unprotected"]
            self.households.buildings_structure_curve[
                "building_elevated_possessions_late"
            ] = self.households.buildings_structure_curve["building_unprotected"]
            # create timing-based damage curves for elevated possessions
            # Early action (>48h lead time): 20% damage (80% reduction)
            self.households.buildings_content_curve[
                "building_elevated_possessions_early"
            ] = self.households.buildings_content_curve["building_unprotected"] * 0.20

            # Medium action (24-48h lead time): 80% damage (20% reduction)
            self.households.buildings_content_curve[
                "building_elevated_possessions_medium"
            ] = self.households.buildings_content_curve["building_unprotected"] * 0.80

            # Late action (<24h lead time): 90% damage (10% reduction)
            self.households.buildings_content_curve[
                "building_elevated_possessions_late"
            ] = self.households.buildings_content_curve["building_unprotected"] * 0.90

    def calculate_building_flood_damages(
        self,
        verbose: bool = False,
        export_building_damages: bool = False,
        dynamic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """This function calculates the flood damages for the households in the model.

        It iterates over the return periods and calculates the damages for each household
        based on the flood maps and the building footprints.

        Args:
            verbose: Verbosity flag.
            export_building_damages: Whether to export the building damages to parquet files.
            dynamic: Whether to calculate building damages dynamically based on the current flood maps in the model (as opposed to using flood maps at t=0).
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the damage arrays for unprotected and protected buildings.
        """
        # create a pandas data array for assigning damage to the agents:
        agent_df = pd.DataFrame(
            {"building_id_of_household": self.households.var.building_id_of_household}
        )

        # initiate the damage arrays for unprotected and protected buildings
        damages_do_not_adapt = np.zeros(
            (self.households.return_periods.size, self.households.n), np.float32
        )
        damages_adapt = np.zeros(
            (self.households.return_periods.size, self.households.n), np.float32
        )

        # initiate the dictionary containing the damages for each return period for each building
        # if not dynamic:
        if not dynamic and not hasattr(self, "_building_damages_all_return_periods"):
            self._building_damages_all_return_periods = {}
        elif not dynamic and self._building_damages_all_return_periods:
            for i, return_period in enumerate(self.households.return_periods):
                building_multicurve = self._building_damages_all_return_periods[
                    return_period
                ]
                damages_do_not_adapt[i], damages_adapt[i] = (
                    self.households.assign_damages_to_agents(
                        agent_df,
                        building_multicurve,
                    )
                )
                if export_building_damages:
                    fn_for_export = (
                        self.households.model.output_folder / "building_damages"
                    )
                    fn_for_export.mkdir(parents=True, exist_ok=True)
                    building_multicurve.to_parquet(
                        self.households.model.output_folder
                        / "building_damages"
                        / f"building_damages_rp{return_period}_{self.households.model.current_time.year}.parquet"
                    )

                if verbose:
                    print(
                        f"Damages rp{return_period}: {round(damages_do_not_adapt[i].sum() / 1e6)} million"
                    )
                    print(
                        f"Damages adapt rp{return_period}: {round(damages_adapt[i].sum() / 1e6)} million"
                    )
            # set attributes
            self._damages_do_not_adapt = damages_do_not_adapt
            self._damages_adapt = damages_adapt
            # return early
            return self.damages_do_not_adapt, self.damages_adapt
        # create a dictionary of multi_curves for the VectorScannerMultiCurves
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

        # subset building to those exposed to flooding
        buildings = self.households.buildings[
            self.households.buildings["flooded"]
        ].copy()
        flooded_building_ids = np.array(buildings["id"])
        building_geometries = read_geom(
            self.households.model.files["geom"]["assets/open_building_map"],
            filters=[("id", "in", flooded_building_ids)],
        )

        building_geometries = building_geometries.merge(
            buildings[["id", "object_type"]],
            on="id",
            how="left",
        )

        for i, return_period in enumerate(self.households.return_periods):
            flood_map: xr.DataArray = self.households.flood_maps[return_period]

            building_multicurve = building_geometries.copy()

            # Ensure building geometries are in the same CRS as the flood map, as the
            # damage scanner assumes aligned CRSs between vector and raster data.
            flood_crs = flood_map.rio.crs
            if building_multicurve.crs is not None and flood_crs is not None:
                if building_multicurve.crs != flood_crs:
                    building_multicurve = building_multicurve.to_crs(flood_crs)

            building_multicurve_renamed: gpd.GeoDataFrame = building_multicurve.rename(
                columns={
                    "COST_STRUCTURAL_USD_SQM": "maximum_damage_structure",
                    "COST_CONTENTS_USD_SQM": "maximum_damage_content",
                }
            )  # ty:ignore[invalid-assignment]
            damage_buildings: pd.DataFrame = VectorScannerMultiCurves(
                features=building_multicurve_renamed,
                hazard=flood_map,
                multi_curves=multi_curves,
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
            # concatenate damages to building_multicurve
            building_multicurve = pd.concat(
                [building_multicurve, damage_buildings], axis=1
            )
            building_multicurve = building_multicurve[
                ["id", "damages", "damages_flood_proofed"]
            ]
            building_multicurve["damages_t0"] = building_multicurve["damages"].copy()
            building_multicurve["damages_flood_proofed_t0"] = building_multicurve[
                "damages_flood_proofed"
            ].copy()
            if not dynamic:
                self._building_damages_all_return_periods[return_period] = (
                    building_multicurve
                )

            # merged["damage"] is aligned with agents
            damages_do_not_adapt[i], damages_adapt[i] = (
                self.households.assign_damages_to_agents(
                    agent_df,
                    building_multicurve,
                )
            )
            if export_building_damages:
                fn_for_export = self.households.model.output_folder / "building_damages"
                fn_for_export.mkdir(parents=True, exist_ok=True)
                building_multicurve.to_parquet(
                    self.households.model.output_folder
                    / "building_damages"
                    / f"building_damages_rp{return_period}_{self.households.model.current_time.year}.parquet"
                )

            if verbose:
                print(
                    f"Damages rp{return_period}: {round(damages_do_not_adapt[i].sum() / 1e6)} million"
                )
                print(
                    f"Damages adapt rp{return_period}: {round(damages_adapt[i].sum() / 1e6)} million"
                )
        # set attributes
        self._damages_do_not_adapt = damages_do_not_adapt
        self._damages_adapt = damages_adapt

        return self.damages_do_not_adapt, self.damages_adapt

    def calculate_ead_per_gdl_region(
        self,
        ead_per_household: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate and accumulate expected annual damages (EAD) per GDL region.

        The method aggregates household-level EAD values to GDL regions for the
        current model year, stores the result in a wide dataframe with years as
        rows and GDL regions as columns, and preserves the final dataframe on
        the last timestep.

        Args:
            ead_per_household: Expected annual damages per household (USD per year).

        Returns:
            A dataframe with years as rows and GDL regions as columns.
        """
        # Get gdl region for each household
        gdl_regions: pd.Series = self.households.buildings.loc[
            self.households.var.building_id_of_household, "GDLcode"
        ]

        current_year: int = self.households.model.current_time.year

        # Aggregate household EAD to regions for this timestep and keep a stable
        # set of columns across the full simulation.
        ead_per_gdl_region: pd.Series = (
            pd.DataFrame({"GDLcode": gdl_regions, "EAD": ead_per_household})
            .groupby("GDLcode", sort=True)["EAD"]
            .sum()
        )

        if not hasattr(self, "ead_per_gdl_region") or not isinstance(
            self.ead_per_gdl_region, pd.DataFrame
        ):
            self.ead_per_gdl_region = pd.DataFrame()

        # Keep one column per unique GDL region and one row per model year.
        all_regions: list[str] = sorted(
            set(self.ead_per_gdl_region.columns).union(ead_per_gdl_region.index)
        )
        self.ead_per_gdl_region = self.ead_per_gdl_region.reindex(columns=all_regions)
        self.ead_per_gdl_region.loc[current_year, ead_per_gdl_region.index] = (
            ead_per_gdl_region.astype(np.float32).values
        )
        self.ead_per_gdl_region.index.name = "year"

        if (
            self.households.model.current_timestep
            == self.households.model.n_timesteps - 1
        ):
            self.ead_per_gdl_region = self.ead_per_gdl_region.sort_index()
            self.ead_per_gdl_region.to_csv(
                self.households.model.output_folder / "ead_per_gdl_region.csv"
            )

        return self.ead_per_gdl_region

    def calculate_ead(
        self,
        damages_do_not_adapt: np.ndarray,
        damages_adapt: np.ndarray,
        adapted: np.ndarray,
        altered_flood_protection_standard: int | None = None,
        update_gdl_ead: bool = True,
    ) -> np.ndarray:
        """Calculate expected annual damages (EAD) for each household.

        Integrates damages across return periods using trapezoid rule. Handles
        adapted households differently and can apply an alternative flood protection
        standard that eliminates damages below a threshold return period.

        Args:
            damages_do_not_adapt: Damages by return period (rows) and household (columns) for non-adapted.
            damages_adapt: Damages by return period (rows) and household (columns) for adapted.
            adapted: Boolean array indicating which households have adapted.
            altered_flood_protection_standard: If provided, set damages to 0 for return periods
                below this threshold (damages protected against by higher standard).

        Returns:
            1D array of annual expected damages (USD) for each household.
        """
        # Start with baseline (non-adapted) damages
        all_damages = damages_do_not_adapt.copy()

        # Use adapted damages for households that have adapted
        adapted_mask = adapted.astype(bool)
        all_damages[:, adapted_mask] = damages_adapt[:, adapted_mask]

        # Apply higher flood protection standard if provided
        if altered_flood_protection_standard is not None:
            # Zero out damages for return periods protected by the higher standard
            protected_mask = (
                self.households.return_periods < altered_flood_protection_standard
            )
            all_damages[protected_mask, :] = 0.0

        # Integrate damages across return periods (exceedance probability integration)
        probabilities = 1.0 / self.households.return_periods
        sort_idx = np.argsort(probabilities)

        # Calculate EAD via trapezoid integration
        ead_usd_per_year = np.trapezoid(
            y=all_damages[sort_idx, :], x=probabilities[sort_idx], axis=0
        )
        if update_gdl_ead:
            self.calculate_ead_per_gdl_region(ead_usd_per_year)
        return ead_usd_per_year

    def flood(self, flood_depth: xr.DataArray) -> float:
        """This function computes the damages for the assets and land use types in the model.

        Args:
            flood_depth: The flood map containing water levels for the flood event [m].

        Returns:
            The total flood damages for the event for all assets and land use types.

        Raises:
            NotImplementedError: If the flood function is not implemented for the global damage model.
            ValueError: If both warning response and adaptation are enabled in the model configuration, as this may lead to unintended consequences.
        """
        if (
            "damage_model/flood/residential/content/maximum_damage"
            not in self.model.files["dict"]
        ):
            raise NotImplementedError(
                "The model was probably build with the damage_model set to global. This funcion is not yet implemented for the global damage model. Please rebuild the damage model with the local model (geul) instead."
            )

        flood_depth: xr.DataArray = flood_depth.compute()

        # subset building to those exposed to flooding
        buildings_centroids = gpd.GeoDataFrame(
            self.households.buildings,
            geometry=gpd.points_from_xy(
                self.households.buildings["x"], self.households.buildings["y"]
            ),
            crs="EPSG:4326",
        )

        # get the building ids of the flooded buildings
        # reproject centroids to the flood raster CRS so we can sample depths directly
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
            building_geometries["id"],
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
        if (
            (
                self.households.model.config["agent_settings"]["households"][
                    "warning_response"
                ]
            )
            & (self.households.config["adapt"])
        ):
            raise ValueError(
                "Warning: Both warning response and adaptation are enabled in the model configuration. This may lead to unintended consequences as both mechanisms currently influence the same protective measure of flood-proofing buildings. Please use either adapt or warning response, but not both."
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

            # Add lead_time information for timing-based damage reduction
            household_points["action_lead_time"] = self.households.var.action_lead_time

            # spatial join to get household attributes to buildings
            buildings: gpd.GeoDataFrame = gpd.sjoin_nearest(
                buildings, household_points, how="left", exclusive=True
            )
            buildings["object_type"] = "building_unprotected"  # reset
            # Assign object types for buildings centroid based on protective measures taken
            buildings_centroid = household_points.to_crs(flood_depth.rio.crs)
            buildings_centroid["maximum_damage"] = (
                self.households.var.max_dam_buildings_content
            )
            # Save the buildings with actions taken
            output_path = (
                self.households.model.output_folder
                / "action_maps"
                / "buildings_with_protective_measures.geoparquet"
            )
            # Ensure the action_maps directory exists before writing the file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            damage_reduction_over_leadtime = self.households.model.config[
                "agent_settings"
            ]["households"]["warning_system"]["damage_reduction_over_leadtime"]
            if damage_reduction_over_leadtime:
                elevated_mask = buildings["elevated_possessions"] == True
                # Early action: >48 hours lead time
                early_mask = elevated_mask & (buildings["action_lead_time"] > 48)
                buildings.loc[early_mask, "object_type"] = (
                    "building_elevated_possessions_early"
                )
                print(f"Early action buildings: {early_mask.sum()}")

                # Medium action: 24-48 hours lead time
                medium_mask = (
                    elevated_mask
                    & (buildings["action_lead_time"] > 24)
                    & (buildings["action_lead_time"] <= 48)
                )
                buildings.loc[medium_mask, "object_type"] = (
                    "building_elevated_possessions_medium"
                )
                print(f"Medium action buildings: {medium_mask.sum()}")

                # Late action: <24 hours lead time
                late_mask = elevated_mask & (buildings["action_lead_time"] <= 24)
                buildings.loc[late_mask, "object_type"] = (
                    "building_elevated_possessions_late"
                )
                print(f"Late action buildings: {late_mask.sum()}")

                # Summary of object types
                object_type_counts = buildings["object_type"].value_counts()
                print("Building object type counts:")
                for obj_type, count in object_type_counts.items():
                    print(f"  {obj_type}: {count}")
                buildings.to_parquet(output_path)
                # Timing-based object type assignment for buildings_centroid
                buildings_centroid["object_type"] = np.select(
                    [
                        (buildings_centroid["elevated_possessions"])
                        & (buildings_centroid["action_lead_time"] > 48),
                        (buildings_centroid["elevated_possessions"])
                        & (buildings_centroid["action_lead_time"] > 24)
                        & (buildings_centroid["action_lead_time"] <= 48),
                        (buildings_centroid["elevated_possessions"])
                        & (buildings_centroid["action_lead_time"] <= 24),
                    ],
                    [
                        "building_elevated_possessions_early",
                        "building_elevated_possessions_medium",
                        "building_elevated_possessions_late",
                    ],
                    default="building_unprotected",
                )
            else:
                # Assign object types for buildings based on protective measures taken
                buildings.loc[buildings["elevated_possessions"], "object_type"] = (
                    "building_elevated_possessions"
                )
                buildings.loc[buildings["sandbags"], "object_type"] = (
                    "building_with_sandbags"
                )
                buildings.loc[
                    buildings["elevated_possessions"] & buildings["sandbags"],
                    "object_type",
                ] = "building_all_forecast_based"
                buildings.to_parquet(output_path)

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
        elif self.households.config["adapt"]:
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

    def return_period_flood(self) -> np.ndarray:
        """Simulate a flood event based on return periods and determine which households are flooded.

        Returns:
            Array of indices of flooded households.
        """
        # draw a single random number
        if self.model.current_timestep == 0:
            return np.array([], dtype=int)
        u = np.random.random()
        return_period = 1 / u
        affected_subbasins = [
            subbasin
            for subbasin, protection in self.flood_protection_standard_subbasins.items()
            if protection < return_period
        ]

        if len(affected_subbasins) == 0:
            self.flood_in_last_year = False
            return np.array([], dtype=int)

        # get the indices of households in the affected subbasins
        mask = np.isin(self.households.comid_of_household, affected_subbasins)
        flooded_household_indices = np.nonzero(mask)[0]

        self.flood_in_last_year = len(flooded_household_indices) > 0
        print(
            f"Flood event with return period {return_period:.2f} years affected {len(flooded_household_indices)} households."
        )
        return flooded_household_indices

    def _adjust_damages_for_flood_protection(
        self,
        damages: np.ndarray,
    ) -> np.ndarray:
        """Return damages with values below the flood protection standard set to 0.

        Args:
            damages: 2D array of damages by return period (rows) and household (columns).
        Returns:
            2D array of damages with values below the flood protection standard set to 0.
        """
        comids = self.households.comid_of_household

        household_thresholds = np.fromiter(
            (self.flood_protection_standard_subbasins.get(int(c), -1) for c in comids),
            dtype=float,
            count=comids.size,
        )

        mask = self.households.return_periods[:, None] >= household_thresholds[None, :]
        return damages * mask

    @property
    def damages_do_not_adapt(self) -> np.ndarray:
        """Return damages for households that do not adapt."""
        return self._adjust_damages_for_flood_protection(self._damages_do_not_adapt)

    @property
    def damages_adapt(self) -> np.ndarray:
        """Return damages for households that adapt."""
        return self._adjust_damages_for_flood_protection(self._damages_adapt)

    def _calculate_dike_heights(
        self, dikes, floodmap_template
    ) -> dict[int, dict[int, np.ndarray]]:
        """Calculate dike heights for each river and return period.

        This is done by sampling the flood maps along the river geometries and extracting the flood depths at those points.
        These dike heights are then stored in a dictionary for later use by the government agent to determine the required dike height for each river and return period.

        Returns:
            dict[int, dict[int, np.ndarray]]: A nested dictionary where the first key is the return period, the second key is the river ID, and the value is an array of dike heights (flood depths) along the river.
        """
        dike_heights = {}
        for coastal_dike in dikes.itertuples():
            coastal_geom = coastal_dike.geometry
            # check if geom is within bounds of floodmap_template
            if not box(*floodmap_template.rio.bounds()).contains(coastal_geom):
                continue
            # initialize idx_river_points to False to avoid recalculating for each return period
            idx_river_points = False
            # sample every 100 m (TODO: build dike lines in model build with 100 m spacing. For now use interpolation to get points along the river geometry)
            distances = np.arange(
                0, coastal_geom.length, 0.0008333
            )  # 100 m in degrees (approximate, for WGS84)

            # Extract x/y directly without creating intermediate Point objects
            x = np.array([coastal_geom.interpolate(d).x for d in distances])
            y = np.array([coastal_geom.interpolate(d).y for d in distances])

            for rp in self.households.return_periods:
                flood_map: xr.DataArray = self.households.flood_maps[rp]
                flood_map_array = flood_map.values
                if rp not in dike_heights:
                    dike_heights[rp] = {}
                if not idx_river_points:
                    idx_river_points = coords_to_pixels(
                        coords=np.column_stack((x, y)),
                        gt=flood_map.rio.transform().to_gdal(),
                    )
                depths = flood_map_array[(idx_river_points[1], idx_river_points[0])]
                depths = np.nan_to_num(depths, nan=0.0)
                dike_heights[rp][coastal_dike[0]] = depths

        return dike_heights

    def dike_heights(self) -> dict[int, dict[int, np.ndarray]]:
        """Calculate dike heights for each river and return period.

        This is done by sampling the flood maps along the river geometries and extracting the flood depths at those points.
        These dike heights are then stored in a dictionary for later use by the government agent to determine the required dike height for each river and return period.

        Returns:
            dict[int, dict[int, np.ndarray]]: A nested dictionary where the first key is the return period, the second key is the river ID, and the value is an array of dike heights (flood depths) along the river.
        """
        if hasattr(self, "_coastal_dike_heights") and hasattr(
            self, "_riverine_dike_heights"
        ):
            return self._coastal_dike_heights, self._riverine_dike_heights
        elif hasattr(self, "_coastal_dike_heights"):
            return self._coastal_dike_heights, {}
        elif hasattr(self, "_riverine_dike_heights"):
            return {}, self._riverine_dike_heights
        # load river network
        river_network = gpd.read_parquet(
            Path(self.households.model.files["geom"]["routing/rivers"])
        )
        # load the coastline
        coastline = gpd.read_parquet(
            Path(self.households.model.files["geom"]["coastal/coastlines"])
        )
        # load the subbasins
        subbasins = gpd.read_parquet(
            Path(self.households.model.files["geom"]["routing/subbasins"])
        )

        coastal_dikes = gpd.GeoDataFrame([])

        for subbasin in subbasins.reset_index().itertuples():
            # clip the coastline to the subbasin geometry
            coastline_clipped = gpd.clip(coastline, subbasin.geometry.buffer(0.0008333))
            # assign the clipped coastline to the subbasin
            coastline_clipped["COMID"] = subbasin.COMID
            coastal_dikes = pd.concat([coastal_dikes, coastline_clipped])
        coastal_dikes = coastal_dikes.set_index("COMID", drop=True)
        floodmap_template = self.households.flood_maps[
            self.households.return_periods[0]
        ]

        self._coastal_dike_heights = self._calculate_dike_heights(
            coastal_dikes, floodmap_template
        )
        self._riverine_dike_heights = self._calculate_dike_heights(
            river_network, floodmap_template
        )
        return self._coastal_dike_heights, self._riverine_dike_heights
