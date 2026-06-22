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
        self.load_damage_curves()
        self.alter_damage_curves_based_on_actions()
        self.load_max_damage_values()
        if self.model.config["hazards"]["floods"]["flood_risk"]:
            self.load_return_period_flood_maps()

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

            # Late action (<24h lead time): 80% damage (20% reduction) - same as standard
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
        if not dynamic:
            self.damages_do_not_adapt = damages_do_not_adapt
            self.damages_adapt = damages_adapt
        return damages_do_not_adapt, damages_adapt

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

        Raises:
            NotImplementedError: If the flood function is not implemented for the global damage model.
            ValueError: If both warning response and adaptation are enabled in the model configuration, as this may lead to unintended consequences.
        """
        if self.model.config["hazards"]["floods"]["damage_model"] == "global":
            raise NotImplementedError(
                "The flood function is not implemented for the global damage model yet."
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

    def return_period_flood(self, flood_protection_standard: int = 10) -> np.ndarray:
        """Simulate a flood event based on return periods and determine which households are flooded.

        Returns:
            Array of indices of flooded households.
        """
        # draw a single random number
        p_random = np.random.random()
        # Work with a locally sorted copy of return periods to ensure correct event selection
        return_periods_arr = np.asarray(self.households.return_periods, dtype=float)
        sort_idx = np.argsort(return_periods_arr)  # ascending order
        sorted_return_periods = return_periods_arr[sort_idx]
        probabilities = 1.0 / sorted_return_periods

        if p_random >= probabilities.max() or p_random >= 1 / flood_protection_standard:
            return np.array([], dtype=int)

        # find the event corresponding to the random draw
        event_idx = np.searchsorted(probabilities[::-1], p_random)
        event_idx = len(probabilities) - 1 - event_idx
        event = sorted_return_periods[event_idx]
        self.model.logger.info(
            "Return period flood event: %s years (p=%.4f, random draw=%.4f)",
            event,
            probabilities[event_idx],
            p_random,
        )

        # get the flood map for this event
        flood_map: xr.DataArray = self.households.flood_maps[event]

        # cache household coordinates in flood_map CRS (Nx2 numpy array)
        if not hasattr(self, "_household_xy"):
            import pyproj

            x, y = (
                np.array(self.households.buildings.x),
                np.array(self.households.buildings.y),
            )
            transformer = pyproj.Transformer.from_crs(
                "EPSG:4326", flood_map.rio.crs, always_xy=True
            )
            self._building_xy = np.array(transformer.transform(x, y)).T

        # sample flood map using clipped coordinates
        sampled_values = sample_from_map(
            array=flood_map.values,
            coords=self._building_xy,
            gt=flood_map.rio.transform(recalc=True).to_gdal(),
            out_of_bounds_value=np.nan,
        )
        # Use the same minimum flood depth threshold (0.05 m) as elsewhere in the model
        minimum_flood_depth_m = 0.05
        # np.where will return indices of flooded households relative to the original household array
        flooded_building_indices = np.where(sampled_values > minimum_flood_depth_m)[0]

        # get building IDs of flooded buildings
        flooded_building_ids = self.households.buildings.loc[
            flooded_building_indices, "id"
        ].values.astype(int)

        # get indices of households located in flooded buildings
        flooded_household_indices = np.where(
            np.isin(
                self.households.var.building_id_of_household.data, flooded_building_ids
            )
        )[0]

        return flooded_household_indices
