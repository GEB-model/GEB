"""This module contains the WindRiskModule, which calculates the wind risk for each agent based on the wind speed and the vulnerability of the agent."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from pyproj import Geod

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
        # print("Wind maps loaded for return periods:", self.windstorm_return_periods)    

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
                "damage_model/windstorm/residential/structure/curve"
            ]
        )
        self.households.wind_buildings_structure_curve.set_index("severity", inplace=True)

    def alter_damage_curves_for_windstorm_adaptation(self) -> None:
        """Alter the damage curves to account for adaptation measures. For example, we can apply a reduction factor to the damage ratios for buildings with wind shutters or strengthened windows."""
        self.households.wind_buildings_structure_curve["building_unprotected"] = (self.households.wind_buildings_structure_curve["damage_ratio"])
        #self.households.wind_buildings_structure_curve["building_window_shutters"] = (self.households.wind_buildings_structure_curve["damage_ratio"])
        # Window shutters: 75% reduction => multiplier = 0.25
        self.households.wind_buildings_structure_curve["building_window_shutters"] = (
            self.households.wind_buildings_structure_curve["building_unprotected"] * 0.25
        )

        # Strengthened windows: 34% reduction => multiplier = 0.66
        self.households.wind_buildings_structure_curve["building_strengthened_windows"] = (
            self.households.wind_buildings_structure_curve["building_unprotected"] * 0.66
        )

    # def create_wind_damage_interpolators(self):
    #     # Create interpolators for windstorm damage curves.
    #     # For now only concrete and unprotected buildings, no adaptation measures are considered.

    #     self.households.windstorm_building_curve_interpolator = interpolate.interp1d(
    #         x=self.households.wind_buildings_structure_curve.index,
    #         y=self.households.wind_building_curve["residential"],
    #         bounds_error=False,
    #         # fill_value="extrapolate",
    #     )


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
    
        # --- tunable threshold used only for prefiltering ---
        wind_threshold = float(
            self.model.config.get("hazards", {})
            .get("windstorm", {})
            .get("wind_threshold_ms", 30.0)
        )
    
        rps = list(self.households.windstorm_return_periods)
        if len(rps) == 0:
            return damages_unprotected_w, damages_adapt_w
    
        # Ensure buildings is a GeoDataFrame with a CRS (typically EPSG:4326 in this project)
        buildings_gdf = buildings
        
        if not isinstance(buildings_gdf, gpd.GeoDataFrame):
            if "geometry" in buildings_gdf.columns:
                geom0 = buildings_gdf["geometry"].iloc[0]
                if isinstance(geom0, (bytes, bytearray, memoryview, np.bytes_)):
                    geom = gpd.GeoSeries.from_wkb(buildings_gdf["geometry"])
                else:
                    geom = buildings_gdf["geometry"]
                buildings_gdf = gpd.GeoDataFrame(buildings_gdf, geometry=geom)
            else:
                buildings_gdf = gpd.GeoDataFrame(
                    buildings_gdf,
                    geometry=gpd.points_from_xy(buildings_gdf["x"], buildings_gdf["y"]),
                )
        
        if buildings_gdf.crs is None:
            buildings_gdf = buildings_gdf.set_crs("EPSG:4326")
    
        # Centroid points in WGS84 (for sampling)
        buildings_points_wgs84 = gpd.GeoDataFrame(
            buildings_gdf[["id"]].copy(),
            geometry=gpd.points_from_xy(buildings_gdf["x"], buildings_gdf["y"]),
            crs="EPSG:4326",
        ).set_index("id")
    
        # Check CRS consistency across wind rasters for efficient reprojection
        wind_crs_first = self.households.windstorm_maps[rps[0]].rio.crs
        all_same_crs = True
        for rp in rps[1:]:
            if self.households.windstorm_maps[rp].rio.crs != wind_crs_first:
                all_same_crs = False
                break
    
       
        if wind_crs_first is not None and all_same_crs and buildings_points_wgs84.crs != wind_crs_first:
            buildings_points_common = buildings_points_wgs84.to_crs(wind_crs_first)
        else:
            buildings_points_common = buildings_points_wgs84

        # --- Prefilter: exposed building ids per RP via centroid wind speed ---
        exposed_ids_by_rp: dict[float, np.ndarray] = {}
        for rp in rps:
            wind_map: xr.DataArray = self.households.windstorm_maps[rp]
            wind_crs = wind_map.rio.crs
    
            if (not all_same_crs) and (wind_crs is not None) and (buildings_points_wgs84.crs != wind_crs):
                buildings_points = buildings_points_wgs84.to_crs(wind_crs)
            else:
                buildings_points = buildings_points_common
    
            x_dim = wind_map.rio.x_dim
            y_dim = wind_map.rio.y_dim
            x_coords = buildings_points.geometry.x.values
            y_coords = buildings_points.geometry.y.values
    
            sampled_winds = wind_map.interp(
                {x_dim: ("points", x_coords), y_dim: ("points", y_coords)},
                method="nearest",
            )
            sampled_vals = np.nan_to_num(np.asarray(sampled_winds.values).squeeze(), nan=0.0)
    
            exposed_ids = buildings_points.index.values[sampled_vals >= wind_threshold]
            exposed_ids_by_rp[rp] = np.asarray(exposed_ids, dtype=int)
    
            if debug_damage_stats:
                n_total = int(buildings_points.shape[0])
                n_exposed = int(exposed_ids_by_rp[rp].size)
                frac = (n_exposed / n_total) if n_total else 0.0
                print(
                    f"Wind prefilter rp={int(rp)}: "
                    f"{n_exposed}/{n_total} buildings (frac={frac:.3f}) "
                    f"above wind threshold {wind_threshold} m/s"
                )
    
        union_ids = (
            np.unique(np.concatenate([v for v in exposed_ids_by_rp.values() if v.size > 0]))
            if any(v.size > 0 for v in exposed_ids_by_rp.values())
            else np.array([], dtype=int)
        )
    
        if union_ids.size == 0:
            if verbose:
                for i, rp in enumerate(rps):
                    print(f"Wind Damages rp{rp}: 0 million (no exposed buildings)")
                    print(f"Wind Damages adapt rp{rp}: 0 million (no exposed buildings)")
            return damages_unprotected_w, damages_adapt_w
    
        # --- Build centroid FEATURES for the scanner (point geometries) ---
        # We’ll compute footprint_m2 from the *polygon* geometry once, but scan using points.
        buildings_union = buildings_gdf[buildings_gdf["id"].isin(union_ids)].copy()
        buildings_union = buildings_union.set_index("id")
    
        # footprint area in m²
        projected_crs = buildings_union.estimate_utm_crs()
        buildings_union_m = buildings_union.to_crs(projected_crs)
        footprint_m2 = buildings_union_m.geometry.area.astype(np.float32)
    
        # Prepare a base point feature table with required columns.
        # IMPORTANT: index is building id so scanner output aligns nicely.
        features_pts_wgs84 = gpd.GeoDataFrame(
            buildings_union[["object_type", "COST_STRUCTURAL_USD_SQM"]].copy(),
            geometry=gpd.points_from_xy(buildings_union["x"], buildings_union["y"]),
            crs="EPSG:4326",
        )
    
        # Reproject points once if possible
        if wind_crs_first is not None and all_same_crs and features_pts_wgs84.crs != wind_crs_first:
            features_pts_common = features_pts_wgs84.to_crs(wind_crs_first)
        else:
            features_pts_common = features_pts_wgs84
    
        # Multi-curve dict (wind curves are severity->damage_ratio; keep as-is)
        multi_curves = {
            "damages_structure_unprotected": self.households.wind_buildings_structure_curve[
                "building_unprotected"
            ],
            "damages_structure_wind_shutters": self.households.wind_buildings_structure_curve[
                "building_window_shutters"
            ],
        }
    
        # --- RP loop: scan ONLY exposed point-features ---
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
    
            wind_map: xr.DataArray = self.households.windstorm_maps[rp]
            wind_crs = wind_map.rio.crs
    
            # Subset point-features to exposed ids
            features_pts = features_pts_common.loc[ids].copy()
    
            # If wind rasters don’t share CRS, reproject per RP
            if (not all_same_crs) and (wind_crs is not None) and (features_pts.crs != wind_crs):
                features_pts = features_pts.to_crs(wind_crs)
    
            # Hazard mask: keep NaNs outside threshold for sparsity
            wind_map_masked = wind_map.where(wind_map >= wind_threshold)
    
            # Cell area in m² (assumes projected CRS; if wind CRS is degrees this won’t be meaningful)
            transform = wind_map_masked.rio.transform(recalc=True)
            dx_deg = float(abs(transform.a))
            dy_deg = float(abs(transform.e))
            
            # representative lon/lat (use the points you are scanning)
            if features_pts.crs and str(features_pts.crs).upper() != "EPSG:4326":
                pts_ll = features_pts.to_crs("EPSG:4326")
            else:
                pts_ll = features_pts
            
            lon0 = float(pts_ll.geometry.x.mean())
            lat0 = float(pts_ll.geometry.y.mean())
            
            geod = Geod(ellps="WGS84")
            half_dx = dx_deg / 2.0
            half_dy = dy_deg / 2.0
            lons = [lon0 - half_dx, lon0 + half_dx, lon0 + half_dx, lon0 - half_dx]
            lats = [lat0 - half_dy, lat0 - half_dy, lat0 + half_dy, lat0 + half_dy]
            area_m2, _ = geod.polygon_area_perimeter(lons, lats)
            cell_area_m2 = float(abs(area_m2))
    
            # Pre-scale max damage so that (coverage_m2 * max_damage_structure) ~= footprint_m2 * cost_per_m2
            # Assumption: point “coverage” behaves like ~1 cell (so coverage_m2 ~ cell_area_m2).
            fp = footprint_m2.reindex(features_pts.index).to_numpy(np.float32)
            cost_m2 = features_pts["COST_STRUCTURAL_USD_SQM"].to_numpy(np.float32)
            features_pts["maximum_damage_structure"] = (cost_m2 * fp / max(cell_area_m2, 1e-6)).astype(np.float32)
    
            #print(f"No of buildings to scan for wind damages rp{rp}: {features_pts.shape[0]}")
    
            damage_buildings = VectorScannerMultiCurves(
                features=features_pts,
                hazard=wind_map_masked,
                multi_curves=multi_curves,
            )
    
            # Convert scanner output to the columns assign_wdamages_to_agents expects
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

    


