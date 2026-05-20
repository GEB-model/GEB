"""This module contains the WindRiskModule, which calculates the wind risk for each agent based on the wind speed and the vulnerability of the agent."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

#from geb.hydrology.landcovers import FOREST
from geb.workflows.io import read_params, read_table #read_geom, read_params, read_table, read_zarr

from ...workflows.damage_scanner import VectorScannerMultiCurves #VectorScanner, VectorScannerMultiCurves
#from ..workflows.helpers import from_landuse_raster_to_polygon

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
        self.alter_damage_curves_for_windstorm_adaptation()
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
       
        self.households.windstorm_maps = windstorm_maps
   

    def load_max_damage_values(self) -> None:
        """Load maximum damage values from model files and store them in the model variables."""
        if (
            "damage_model/windstorm/residential/structure/maximum_damage"
            in self.households.model.files["dict"]
        ):
            self.households.var.max_dam_buildings_structure = float(
                read_params(
                    self.households.model.files["dict"][
                        "damage_model/windstorm/residential/structure/maximum_damage"
                    ]
                )["maximum_damage"]
            )
            self.households.buildings["maximum_damage_m2"] = self.households.var.max_dam_buildings_structure

    def load_windstorm_damage_curve(self) -> None:
        """Load windstorm damage curves from CLIMADA"""

        self.households.wind_buildings_structure_curve = read_table(
            self.households.model.files["table"][
                "damage_model/windstorm/buildings/residential/curve"
            ]
        )
        self.households.wind_buildings_structure_curve.set_index("severity", inplace=True)

    def alter_damage_curves_for_windstorm_adaptation(self) -> None:
        """Alter the damage curves to account for adaptation measures. For example, we can apply a reduction factor to the damage ratios for buildings with wind shutters or strengthened windows."""
        self.households.wind_buildings_structure_curve["building_unprotected"] = (self.households.wind_buildings_structure_curve["damage_ratio"])
        # Window shutters: 75% reduction => multiplier = 0.25
        self.households.wind_buildings_structure_curve["building_window_shutters"] = (
            self.households.wind_buildings_structure_curve["building_unprotected"] * 0.25
        )

        # Strengthened windows: 34% reduction => multiplier = 0.66
        self.households.wind_buildings_structure_curve["building_strengthened_windows"] = (
            self.households.wind_buildings_structure_curve["building_unprotected"] * 0.66
        )

    def calculate_building_wind_damages(
        self, verbose: bool = True, export_building_damages: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """This function calculates the wind damages for the households in the model.
        
        It iterates over the return periods and calculates the damages for each household
        based on the flood maps and the building footprints.
        
        Args:
            verbose: Verbosity flag.
            export_building_damages: Whether to export the building damages to parquet files.
        Returns:
            Tuple[np.ndarray, np.nadarray]: A tuple containing the damage arrays for unprotected and protected buildings."""
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
    
        # create a pandas data array for assigning damage to agents
        agent_df = pd.DataFrame(
            {"building_id_of_household": self.households.var.building_id_of_household}
        )
    
        buildings = self.households.buildings.copy()
       
        # Optional filter to focus only on flooded households
        only_flooded_buildings = bool(
            self.model.config.get("hazards", {})
            .get("windstorm", {})
            .get("only_flooded_buildings", True)
        )
        if only_flooded_buildings:
            buildings = buildings[buildings["flooded"]]
    
        buildings = buildings[buildings["n_occupants"] > 0]
    
        # threshold used only for prefiltering
        wind_threshold = float(
            self.model.config.get("hazards", {})
            .get("windstorm", {})
            .get("wind_threshold_ms", 20.0)
        )
    
        rps = list(self.households.windstorm_return_periods)
        if len(rps) == 0:
            return damages_unprotected_w, damages_adapt_w
    
        # Ensure buildings is a GeoDataFrame with a CRS (typically EPSG:4326 in this project)
        buildings_gdf = buildings
    
        if isinstance(buildings, gpd.GeoDataFrame):
            buildings_gdf = buildings
        elif "geometry" in buildings.columns:
            geom = buildings["geometry"]
            if len(geom) > 0 and isinstance(geom.iloc[0], (bytes, bytearray, memoryview, np.bytes_)):
                geom = gpd.GeoSeries.from_wkb(geom)
            buildings_gdf = gpd.GeoDataFrame(buildings, geometry=geom)
        else:
            buildings_gdf = gpd.GeoDataFrame(
                buildings,
                geometry=gpd.points_from_xy(buildings["x"], buildings["y"]),
            )

        if buildings_gdf.crs is None:
            buildings_gdf = buildings_gdf.set_crs("EPSG:4326")

        # Centroid points in WGS84 (for sampling)
        buildings_points_wgs84 = gpd.GeoDataFrame(
            buildings_gdf[["id"]].copy(),
            geometry=gpd.points_from_xy(buildings_gdf["x"], buildings_gdf["y"]),
            crs="EPSG:4326",
        ).set_index("id")
        
        # Prefilter buildings via centroid wind speed
        exposed_ids_by_rp: dict[float, np.ndarray] = {}
        for rp in rps:
            wind_map: xr.DataArray = self.households.windstorm_maps[rp]
            wind_crs = wind_map.rio.crs

            buildings_points = (
                buildings_points_wgs84.to_crs(wind_crs)
                if wind_crs is not None and buildings_points_wgs84.crs != wind_crs
                else buildings_points_wgs84
            )

            sampled_winds = wind_map.interp(
                {wind_map.rio.x_dim: ("points", buildings_points.geometry.x.values),
                 wind_map.rio.y_dim: ("points", buildings_points.geometry.y.values),},
                method="nearest",
            )
            sampled_vals = np.nan_to_num(np.asarray(sampled_winds.values).squeeze(), nan=0.0)
            exposed_ids_by_rp[rp] = np.asarray(buildings_points.index.values[sampled_vals >= wind_threshold], dtype=int)

            if debug_damage_stats: #What?
                n_total = int(buildings_points.shape[0])
                n_exposed = int(exposed_ids_by_rp[rp].size)
                frac = (n_exposed / n_total) if n_total else 0.0
                print(
                    f"Wind prefilter rp={int(rp)}: "
                    f"{n_exposed}/{n_total} buildings (frac={frac:.3f})"
                    f"above wind threshold {wind_threshold} m/s"
                )        
    
        all_exposed = [v for v in exposed_ids_by_rp.values() if v.size > 0] #What?
        union_ids = np.unique(np.concatenate(all_exposed)) if all_exposed else np.array([], dtype=int)

        if union_ids.size == 0:
            if verbose:
                for i, rp in enumerate(rps):
                    print(f"Wind Damages rp{rp}: 0 million (no exposed buildings)")
                    print(f"Wind Damages adapt rp{rp}: 0 million (no exposed buildings)")
            return damages_unprotected_w, damages_adapt_w

        buildings_union = buildings_gdf[buildings_gdf["id"].isin(union_ids)].copy().set_index("id")

        footprint_m2 = buildings_union.to_crs(buildings_union.estimate_utm_crs()).geometry.area.astype(np.float32)
    
        # index is building id so scanner output aligns nicely.
        features_pts_wgs84 = gpd.GeoDataFrame(
            buildings_union[["object_type", "COST_STRUCTURAL_USD_SQM"]].copy(),
            geometry=gpd.points_from_xy(buildings_union["x"], buildings_union["y"]),
            crs="EPSG:4326",
        )
    
        # Multi-curve dict (wind curves are severity->damage_ratio; keep as-is)
        multi_curves = {
            "damages_structure_unprotected": self.households.wind_buildings_structure_curve[
                "building_unprotected"
            ],
            "damages_structure_wind_shutters": self.households.wind_buildings_structure_curve[
                "building_window_shutters"
            ],
        }
    
        # return period loop scan ONLY exposed point-features
        for i, rp in enumerate(rps):
            ids = exposed_ids_by_rp[rp]
            if ids.size == 0:
                empty = pd.DataFrame(columns=["id", "damages_unprotected", "damages_wind_shutters"])
                damages_unprotected_w[i], damages_adapt_w[i] = (
                    self.households.assign_wdamages_to_agents(agent_df, empty)
                )
                if verbose:
                    print(f"Wind Damages rp{rp}: 0 million (no exposed buildings)")
                    print(f"Wind Damages adapt rp{rp}: 0 million (no exposed buildings)")
                continue
    
            # wind_map: xr.DataArray = self.households.windstorm_maps[rp]
            # wind_crs = wind_map.rio.crs
            wind_map: xr.DataArray = self.households.windstorm_maps[rp]
            wind_map_masked = self.reproject_to_utm(wind_map.where(wind_map >= wind_threshold))

            features_pts = features_pts_wgs84.loc[ids].copy().to_crs(wind_map_masked.rio.crs)

            # Cell area in m2 from UTM transform (dx_m * dy_m)
            transform = wind_map_masked.rio.transform(recalc=True)
            cell_area_m2 = float(abs(transform.a * transform.e)) #What?

            # Pre-scale max damage so that (coverage_m2 * max_damage_structure) ~= footprint_m2 * cost_per_m2
            fp = footprint_m2.reindex(features_pts.index).to_numpy(np.float32)
            cost_m2 = features_pts["COST_STRUCTURAL_USD_SQM"].to_numpy(np.float32)
            features_pts["maximum_damage_structure"] = (cost_m2 * fp / max(cell_area_m2, 1e-6)).astype(np.float32)
    
            damage_buildings = VectorScannerMultiCurves(
                features=features_pts,
                hazard=wind_map_masked,
                multi_curves=multi_curves,
            )
    
            # Convert scanner output to the needed columns 
            out = pd.DataFrame({"id": damage_buildings.index.astype(int)})
            out["damages_unprotected"] = damage_buildings["damages_structure_unprotected"].to_numpy()
            out["damages_wind_shutters"] = damage_buildings["damages_structure_wind_shutters"].to_numpy()
    
            if export_building_damages:
                fn_export = self.households.model.output_folder / "building_wind_damages"
                fn_export.mkdir(parents=True, exist_ok=True)
                out.to_parquet(
                    fn_export / f"building_wind_damages_rp{rp}_{self.households.model.current_time.year}.parquet"
                )
    
            damages_unprotected_w[i], damages_adapt_w[i] = (
                self.households.assign_wdamages_to_agents(agent_df, out)
            )
    
            if verbose:
                print(f"Wind Damages rp{rp}: {round(damages_unprotected_w[i].sum() / 1e6)} million")
                print(f"Wind Damages adapt rp{rp}: {round(damages_adapt_w[i].sum() / 1e6)} million")
    
        return damages_unprotected_w, damages_adapt_w
    
    @staticmethod #What? Why these numbers?
    def reproject_to_utm(hazard: xr.DataArray) -> xr.DataArray:
        """Reproject the hazard rater to a metric (UTM) CRS"""
        if not hazard.rio.crs.is_geographic:
            return hazard
        bounds = hazard.rio.bounds()
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_epsg = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
        return hazard.rio.reproject(f"EPSG:{utm_epsg}")

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

    


