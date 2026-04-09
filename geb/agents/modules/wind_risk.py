"""This module contains the WindRiskModule, which calculates the wind risk for each agent based on the wind speed and the vulnerability of the agent."""
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

class WindRiskModule:
    """Module responsible for loading and managing wind risk data for the households in the model."""
    def __init__(self, model: GEBModel, households: Agents) -> None:
        """Initialize the WindRiskModule with the model and households, and load all necessary data.
         
          Args:
            model (GEBModel): The GEBModel instance to which this module belongs.
            households (Agents): The Agents instance representing the households in the model.
        """
        self.model = model
        self.households = households
        self.load_windstorm_damage_curve()
        self.load_max_damage_values()
        self.load_wind_maps()


    def load_wind_maps (self) -> None:
        """Load wind maps in this case for the 50 and 100 return periods.This maps are created separately and copied to the "wind_maps" folder.Creating the folder and copying the wind maps should be done manually."""
        self.households.windstorm_return_periods = np.array(
            self.model.config["hazards"]["windstorm"]["return_periods"]
        )


        windstorm_maps = {}
        windstorm_path = self.model.output_folder / "wind_maps"
        for return_period in self.households.windstorm_return_periods:
            file_path = (
                windstorm_path / f"return_level_rp{return_period}.tif"
            )  # adjust to file name
            windstorm_map = xr.open_dataarray(file_path, engine="rasterio")
            windstorm_maps[return_period] = windstorm_map
        #     if debug_maps:
        #         try:
        #             crs = windstorm_map.rio.crs
        #         except Exception:
        #             crs = None
        #         print(
        #             "Loaded windstorm map: "
        #             f"rp={int(return_period)}, path={file_path}, "
        #             f"shape={tuple(windstorm_map.shape)}, crs={crs}"
        #         )

        # wind_maps ={}
        # for return_period in self.households.return_periods:
        #     file_path = (
        #         self.model.output_folder/"wind_maps"/f"return_level_rp{return_period}.tif"
        #     )
        #     wind_maps[return_period] = read_zarr(file_path)
        self.households.windstorm_maps = windstorm_maps
        # print("Wind maps loaded for return periods:", self.windstorm_return_periods)    def load_max_damage_values (self) -> None:

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

    def load_windstorm_damage_curve(self):
        ###MODIFIED load_damage_curves() -> load_windstorm_damage_curves()
        """The function loads the damage curves for windstorms need to look for a windstorm damage function.

        We could use Koks & Haer., 2020 similar to what they did in the CLIMAAX handbook. (Ted WCR)
        For the purpose of building this code we will focus only on concrete building type when selecting the damage curves.
        """
        # Use only the residential concrete building type for now
        self.households.wind_buildings_structure_curve = pd.read_parquet(
            self.households.model.files["table"][
                "damage_parameters/windstorm/buildings/residential/curve"
            ]
        )
        # Set severity as index
        self.households.wind_buildings_structure_curve.set_index("severity", inplace=True)
        self.households.wind_buildings_structure_curve = (
            self.households.wind_buildings_structure_curve.rename(
                columns={"damage_ratio": "building_unprotected"}
            )
        )

        # ADAPTATION MEASURES
        # Window shutters: 75% reduction => multiplies = 0.25
        self.households.wind_buildings_structure_curve["building_window_shutters"] = (
            self.households.wind_buildings_structure_curve["building_unprotected"] * 0.25
        )

        # Strengthened windows: 34% reduction => multiplier = 0.66
        self.households.wind_buildings_structure_curve["building_strengthened_windows"] = (
            self.households.wind_buildings_structure_curve["building_unprotected"] * 0.66
        )

    def create_wind_damage_interpolators(self):
        # Create interpolators for windstorm damage curves.
        # For now only concrete and unprotected buildings, no adaptation measures are considered.

        self.households.windstorm_building_curve_interpolator = interpolate.interp1d(
            x=self.households.wind_buildings_curve.index,
            y=self.households.wind_building_curve["residential"],
            bounds_error=False,
            # fill_value="extrapolate",
        )

    
    # 
    def calculate_building_wind_damages(
        self, verbose: bool = True, export_building_damages: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        damages_unprotected_w = np.zeros(
            (self.households.windstorm_return_periods.size, self.households.n), np.float32
        )
        damages_adapt_w = np.zeros(
            (self.households.windstorm_return_periods.size, self.households.n), np.float32
        )
    
        debug_damage_stats = bool(
            self.model.config.get("hazards", {})
            .get("windstorm", {})
            .get("debug_damage_stats", False)
        )
    
        agent_df = pd.DataFrame(
            {"building_id_of_household": self.households.var.building_id_of_household}
        )
    
        buildings = self.households.buildings.copy()
    
        only_flooded_buildings = bool(
            self.model.config.get("hazards", {})
            .get("windstorm", {})
            .get("only_flooded_buildings", True)
        )
        if only_flooded_buildings:
            buildings = buildings[buildings["flooded"]]
    
        buildings = buildings[buildings["n_occupants"] > 0]
    
        building_ids = np.array(buildings["id"])
    
        building_geometries = read_geom(
            self.households.model.files["geom"]["assets/open_building_map"],
            filters=[("id", "in", building_ids)],
        )
    
        building_geometries = building_geometries.merge(
            buildings[["id", "object_type", "maximum_damage_m2"]],
            on="id",
            how="left",
        )
    
        for i, return_period in enumerate(self.households.windstorm_return_periods):
            wind_map: xr.DataArray = self.households.windstorm_maps[return_period]
    
            building_multicurve = building_geometries.copy()
    
            wind_crs = wind_map.rio.crs
            if building_multicurve.crs is not None and wind_crs is not None:
                if building_multicurve.crs != wind_crs:
                    building_multicurve = building_multicurve.to_crs(wind_crs)
    
            wind_threshold = 27.01
            wind_map_masked = wind_map.fillna(0.0)
            wind_map_masked = wind_map_masked.where(
                wind_map_masked >= wind_threshold, 0.0
            )
    
            multi_curves = {
                "damages_structure_unprotected": self.households.wind_buildings_structure_curve[
                    "building_unprotected"
                ],
                "damages_structure_wind_shutters": self.households.wind_buildings_structure_curve[
                    "building_window_shutters"
                ],
            }
    
            damage_buildings = VectorScannerMultiCurves(
                features=building_multicurve.rename(
                    columns={"maximum_damage_m2": "maximum_damage_structure"}
                ),
                hazard=wind_map_masked,
                multi_curves=multi_curves,
            )
    
            damage_buildings["damages_unprotected"] = damage_buildings[
                "damages_structure_unprotected"
            ]
            damage_buildings["damages_wind_shutters"] = damage_buildings[
                "damages_structure_wind_shutters"
            ]
    
            building_multicurve = pd.concat(
                [building_multicurve, damage_buildings], axis=1
            )
    
            if export_building_damages:
                fn_export = self.households.model.output_folder / "building_wind_damages"
                fn_export.mkdir(parents=True, exist_ok=True)
                building_multicurve.to_parquet(
                    fn_export
                    / f"building_wind_damages_rp{return_period}_{self.households.model.current_time.year}.parquet"
                )
    
            building_multicurve = building_multicurve[
                ["id", "damages_unprotected", "damages_wind_shutters"]
            ]
    
            damages_unprotected_w[i], damages_adapt_w[i] = (
                self.households.assign_wdamages_to_agents(
                    agent_df,
                    building_multicurve,
                )
            )
    
            if debug_damage_stats:
                unprot = damages_unprotected_w[i]
                prot = damages_adapt_w[i]
                frac_nonzero = float(np.mean(unprot > 0))
                print(
                    "Wind damage stats "
                    f"rp={int(return_period)}: "
                    f"sum={float(unprot.sum()):.3e}, mean={float(np.mean(unprot)):.3e}, "
                    f"p95={float(np.quantile(unprot, 0.95)):.3e}, max={float(np.max(unprot)):.3e}, "
                    f"nonzero_frac={frac_nonzero:.3f}, "
                    f"adapt_mean={float(np.mean(prot)):.3e}"
                )
    
            if verbose:
                print(
                    f"Wind Damages rp{return_period}: {round(damages_unprotected_w[i].sum() / 1e6)} million"
                )
                print(
                    f"Wind Damages adapt rp{return_period}: {round(damages_adapt_w[i].sum() / 1e6)} million"
                )
    
        return damages_unprotected_w, damages_adapt_w

    def get_max_wind_at_buildings(
        self, buildings_gdf: gpd.GeoDataFrame, wind_map: xr.DataArray
    ) -> np.ndarray:
        """This function extracts the maximum wind speed at the location of each building.

        Args:
            buildings_gdf: GeoDataFrame containing building geometries.
            wind_map: xarray DataArray containing the wind speed map.

        Returns:
            A numpy array containing the maximum wind speed at each building location.
        """
        # Extract the maximum wind speed at the location of each building
        # by sampling the wind map at the building centroids
        max_winds = []
        for geom in buildings_gdf.geometry:
            masked = wind_map.rio.clip([geom], buildings_gdf.crs, drop=False)
            max_winds.append(float(masked.max()))

        return np.array(max_winds)

    # def wind (self) -> None:
    


