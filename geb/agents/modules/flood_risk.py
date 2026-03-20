"""This module contains the FloodRiskModule class, which is responsible for loading and managing flood risk data for the households in the model. It loads building, road, and rail geometries, as well as damage curves and maximum damage values for different asset types. It also loads flood maps for different return periods to be used in flood risk calculations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from geb.hydrology.landcovers import FOREST
from geb.workflows.io import read_geom, read_params, read_table, read_zarr

from ...workflows.damage_scanner import VectorScanner, VectorScannerMultiCurves
from ..workflows.helpers import from_landuse_raster_to_polygon

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
        self.load_damage_curves()
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

    def flood(self, flood_depth: xr.DataArray) -> float:
        """This function computes the damages for the assets and land use types in the model.

        Args:
            flood_depth: The flood map containing water levels for the flood event [m].

        Returns:
            The total flood damages for the event for all assets and land use types.

        """
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
            household_points = household_points.merge(
                buildings[["id", "flood_proofed"]],
                left_on="building_id",
                right_on="id",
                how="left",
            )  # now merge to get flood proofed status

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
            features=roads.rename(columns={"maximum_damage_m": "maximum_damage"}),
            hazard=flood_depth,
            vulnerability_curves=self.households.var.road_curves,
        )
        total_damages_roads = damages_roads.sum()
        print(f"damages to roads are: {total_damages_roads} ")

        rail = self.households.rail.to_crs(flood_depth.rio.crs)
        damages_rail = VectorScanner(
            features=rail.rename(columns={"maximum_damage_m": "maximum_damage"}),
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
