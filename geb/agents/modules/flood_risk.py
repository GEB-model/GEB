"""This module contains the FloodRiskModule class, which is responsible for loading and managing flood risk data for the households in the model. It loads building, road, and rail geometries, as well as damage curves and maximum damage values for different asset types. It also loads flood maps for different return periods to be used in flood risk calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from geb.workflows.io import read_geom, read_params, read_table, read_zarr

from ...workflows.damage_scanner import VectorScannerMultiCurves

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel


class FloodRiskModule:
    """Module responsible for loading and managing flood risk data for the households in the model."""

    def __init__(self, model: GEBModel, households: Agents) -> None:
        """Initialize the FloodRiskModule with the model and households, and load all necessary data.

        Args:
            model (GEBModel): The main model instance containing configuration and file paths.
            households (Agents): The households agent instance where the loaded data will be stored.

        """
        self.model = model
        self.households = households
        self.load_objects()
        self.load_damage_curves()
        self.load_max_damage_values()
        self.load_flood_maps()

    def load_objects(self) -> None:
        """Load buildings, roads, and rail geometries from model files."""
        # Load buildings
        columns_to_load = [
            "id",
            "floorspace",
            "occupancy",
            "height",
            # "geometry",
            "x",
            "y",
            "NAME_1",
            "TOTAL_REPL_COST_USD_SQM",
            "COST_STRUCTURAL_USD_SQM",
            "COST_NONSTRUCTURAL_USD_SQM",
            "COST_CONTENTS_USD_SQM",
        ]
        self.households.buildings = read_table(
            self.model.files["geom"]["assets/open_building_map"],
            columns=columns_to_load,
        )

        self.households.buildings["object_type"] = (
            "building_unprotected"  # before it was "building_structure"
        )

        # Load roads
        self.households.roads = read_geom(
            self.model.files["geom"]["assets/roads"]
        ).rename(columns={"highway": "object_type"})

        # Load rail
        self.households.rail = read_geom(self.model.files["geom"]["assets/rails"])
        self.households.rail["object_type"] = "rail"

        if self.model.config["general"]["forecasts"]["use"]:
            # Load postal codes --
            # TODO: maybe move it to another function? (not really an object)
            self.households.postal_codes = read_geom(
                self.model.files["geom"]["postal_codes"]
            )

    def load_flood_maps(self) -> None:
        """Load flood maps for different return periods. This might be quite ineffecient for RAM, but faster then loading them each timestep for now."""
        self.households.return_periods = np.array(
            self.model.config["hazards"]["floods"]["return_periods"]
        )

        flood_maps = {}
        for return_period in self.households.return_periods:
            file_path = (
                self.model.output_folder / "flood_maps" / f"{return_period}.zarr"
            )
            flood_maps[return_period] = read_zarr(file_path)
        self.households.flood_maps = flood_maps

    def load_max_damage_values(self) -> None:
        """Load maximum damage values from model files and store them in the model variables."""
        # Load maximum damages
        self.households.var.max_dam_buildings_structure = float(
            read_params(
                self.households.model.files["dict"][
                    "damage_parameters/flood/buildings/structure/maximum_damage"
                ]
            )["maximum_damage"]
        )
        self.households.buildings["maximum_damage_m2"] = (
            self.households.var.max_dam_buildings_structure
        )

        max_dam_buildings_content = read_params(
            self.households.model.files["dict"][
                "damage_parameters/flood/buildings/content/maximum_damage"
            ]
        )
        self.households.var.max_dam_buildings_content = float(
            max_dam_buildings_content["maximum_damage"]
        )

        self.households.var.max_dam_rail = float(
            read_params(
                self.households.model.files["dict"][
                    "damage_parameters/flood/rail/main/maximum_damage"
                ]
            )["maximum_damage"]
        )
        self.households.rail["maximum_damage_m"] = self.households.var.max_dam_rail

        max_dam_road_m: dict[str, float] = {}
        road_types = [
            (
                "residential",
                "damage_parameters/flood/road/residential/maximum_damage",
            ),
            (
                "unclassified",
                "damage_parameters/flood/road/unclassified/maximum_damage",
            ),
            ("tertiary", "damage_parameters/flood/road/tertiary/maximum_damage"),
            ("primary", "damage_parameters/flood/road/primary/maximum_damage"),
            (
                "primary_link",
                "damage_parameters/flood/road/primary_link/maximum_damage",
            ),
            ("secondary", "damage_parameters/flood/road/secondary/maximum_damage"),
            (
                "secondary_link",
                "damage_parameters/flood/road/secondary_link/maximum_damage",
            ),
            ("motorway", "damage_parameters/flood/road/motorway/maximum_damage"),
            (
                "motorway_link",
                "damage_parameters/flood/road/motorway_link/maximum_damage",
            ),
            ("trunk", "damage_parameters/flood/road/trunk/maximum_damage"),
            ("trunk_link", "damage_parameters/flood/road/trunk_link/maximum_damage"),
        ]

        for road_type, path in road_types:
            max_dam_road_m[road_type] = read_params(
                self.households.model.files["dict"][path]
            )["maximum_damage"]

        self.households.roads["maximum_damage_m"] = self.households.roads[
            "object_type"
        ].map(max_dam_road_m)

        self.households.var.max_dam_forest_m2 = float(
            read_params(
                self.households.model.files["dict"][
                    "damage_parameters/flood/land_use/forest/maximum_damage"
                ]
            )["maximum_damage"]
        )

        self.households.var.max_dam_agriculture_m2 = float(
            read_params(
                self.households.model.files["dict"][
                    "damage_parameters/flood/land_use/agriculture/maximum_damage"
                ]
            )["maximum_damage"]
        )

    def load_damage_curves(self) -> None:
        """Load damage curves from model files and store them in the model variables."""
        # Load vulnerability curves [look into these curves, some only max out at 0.5 damage ratio]
        road_curves = []
        road_types = [
            ("residential", "damage_parameters/flood/road/residential/curve"),
            ("unclassified", "damage_parameters/flood/road/unclassified/curve"),
            ("tertiary", "damage_parameters/flood/road/tertiary/curve"),
            ("tertiary_link", "damage_parameters/flood/road/tertiary_link/curve"),
            ("primary", "damage_parameters/flood/road/primary/curve"),
            ("primary_link", "damage_parameters/flood/road/primary_link/curve"),
            ("secondary", "damage_parameters/flood/road/secondary/curve"),
            ("secondary_link", "damage_parameters/flood/road/secondary_link/curve"),
            ("motorway", "damage_parameters/flood/road/motorway/curve"),
            ("motorway_link", "damage_parameters/flood/road/motorway_link/curve"),
            ("trunk", "damage_parameters/flood/road/trunk/curve"),
            ("trunk_link", "damage_parameters/flood/road/trunk_link/curve"),
        ]

        for road_type, path in road_types:
            df = read_table(self.households.model.files["table"][path])
            df = df.rename(columns={"damage_ratio": road_type})

            road_curves.append(df[[road_type]])

        severity_column: pd.DataFrame = df[["severity"]]

        self.households.var.road_curves = pd.concat(
            [severity_column] + road_curves, axis=1
        )
        self.households.var.road_curves.set_index("severity", inplace=True)

        self.households.var.forest_curve = read_table(
            self.households.model.files["table"][
                "damage_parameters/flood/land_use/forest/curve"
            ]
        )
        self.households.var.forest_curve.set_index("severity", inplace=True)
        self.households.var.forest_curve = self.households.var.forest_curve.rename(
            columns={"damage_ratio": "forest"}
        )
        self.households.var.agriculture_curve = read_table(
            self.households.model.files["table"][
                "damage_parameters/flood/land_use/agriculture/curve"
            ]
        )
        self.households.var.agriculture_curve.set_index("severity", inplace=True)
        self.households.var.agriculture_curve = (
            self.households.var.agriculture_curve.rename(
                columns={"damage_ratio": "agriculture"}
            )
        )

        self.households.buildings_structure_curve = read_table(
            self.households.model.files["table"][
                "damage_parameters/flood/buildings/structure/curve"
            ]
        )
        self.households.buildings_structure_curve.set_index("severity", inplace=True)
        self.households.buildings_structure_curve = (
            self.households.buildings_structure_curve.rename(
                columns={"damage_ratio": "building_unprotected"}
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

        # create another column (curve) in the buildings structure curve for flood-proofed buildings
        self.households.buildings_structure_curve["building_flood_proofed"] = (
            self.households.buildings_structure_curve["building_unprotected"] * 0.85
        )
        self.households.buildings_structure_curve.loc[0:1, "building_flood_proofed"] = (
            0.0
        )

        self.households.buildings_content_curve = read_table(
            self.households.model.files["table"][
                "damage_parameters/flood/buildings/content/curve"
            ]
        )
        self.households.buildings_content_curve.set_index("severity", inplace=True)
        self.households.buildings_content_curve = (
            self.households.buildings_content_curve.rename(
                columns={"damage_ratio": "building_unprotected"}
            )
        )

        # create another column (curve) in the buildings content curve for protected buildings
        self.households.buildings_content_curve["building_protected"] = (
            self.households.buildings_content_curve["building_unprotected"] * 0.7
        )
        # create another column (curve) in the buildings content curve for flood-proofed buildings
        self.households.buildings_content_curve["building_flood_proofed"] = (
            self.households.buildings_content_curve["building_unprotected"] * 0.85
        )

        self.households.buildings_content_curve.loc[0:1, "building_flood_proofed"] = 0.0

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

        # create damage curves for adaptation
        buildings_content_curve_adapted = self.households.buildings_content_curve.copy()
        buildings_content_curve_adapted.loc[0:1] = (
            0  # assuming zero damages untill 1m water depth
        )
        buildings_content_curve_adapted.loc[1:] *= (
            0.8  # assuming 80% damages above 1m water depth
        )
        self.households.buildings_content_curve_adapted = (
            buildings_content_curve_adapted
        )

        self.households.var.rail_curve = read_table(
            self.households.model.files["table"][
                "damage_parameters/flood/rail/main/curve"
            ]
        )
        self.households.var.rail_curve.set_index("severity", inplace=True)
        self.households.var.rail_curve = self.households.var.rail_curve.rename(
            columns={"damage_ratio": "rail"}
        )

    def calculate_building_flood_damages(
        self, verbose: bool = True, export_building_damages: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """This function calculates the flood damages for the households in the model.

        It iterates over the return periods and calculates the damages for each household
        based on the flood maps and the building footprints.

        Args:
            verbose: Verbosity flag.
            export_building_damages: Whether to export the building damages to parquet files.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the damage arrays for unprotected and protected buildings.
        """
        damages_do_not_adapt = np.zeros(
            (self.households.return_periods.size, self.households.n), np.float32
        )
        damages_adapt = np.zeros(
            (self.households.return_periods.size, self.households.n), np.float32
        )

        # create a pandas data array for assigning damage to the agents:
        agent_df = pd.DataFrame(
            {"building_id_of_household": self.households.var.building_id_of_household}
        )

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
            damage_buildings: pd.DataFrame = VectorScannerMultiCurves(
                features=building_multicurve.rename(
                    columns={
                        "COST_STRUCTURAL_USD_SQM": "maximum_damage_structure",
                        "COST_CONTENTS_USD_SQM": "maximum_damage_content",
                    }
                ),
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

            if export_building_damages:
                fn_for_export = self.households.model.output_folder / "building_damages"
                fn_for_export.mkdir(parents=True, exist_ok=True)
                building_multicurve.to_parquet(
                    self.households.model.output_folder
                    / "building_damages"
                    / f"building_damages_rp{return_period}_{self.households.model.current_time.year}.parquet"
                )
            building_multicurve = building_multicurve[
                ["id", "damages", "damages_flood_proofed"]
            ]
            # merged["damage"] is aligned with agents
            damages_do_not_adapt[i], damages_adapt[i] = (
                self.households.assign_damages_to_agents(
                    agent_df,
                    building_multicurve,
                )
            )
            if verbose:
                print(
                    f"Damages rp{return_period}: {round(damages_do_not_adapt[i].sum() / 1e6)} million"
                )
                print(
                    f"Damages adapt rp{return_period}: {round(damages_adapt[i].sum() / 1e6)} million"
                )
        return damages_do_not_adapt, damages_adapt
