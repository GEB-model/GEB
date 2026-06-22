"""This module contains the WindRiskModule, which calculates the wind risk for each agent based on the wind speed and the vulnerability of the agent."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from geb.workflows.io import read_params, read_table
from geb.workflows.raster import sample_from_map

from ...workflows.damage_scanner import VectorScannerMultiCurves #VectorScanner, VectorScannerMultiCurves
#from ..workflows.helpers import from_landuse_raster_to_polygon

if TYPE_CHECKING:
    from geb.agents.households import Households
    from geb.model import GEBModel

class WindRiskModule:
    """Module responsible for loading and managing wind risk data for the households in the model."""
    def __init__(self, model: GEBModel, households: Households) -> None:
        """Initialize the WindRiskModule with the model and households, and load all necessary data.
         
          Args:
            model (GEBModel): The GEBModel instance to which this module belongs.
            households (Agents): The Agents instance representing the households in the model.
        """
        self.model = model
        self.households = households
        self._wind_maps_utm_cache: dict = {}    # cache reprojected+masked wind maps
        self._wind_exposure_cache: dict = {}    # cache VectorExposure results per return period
        self._wind_prefilter_cache: dict | None = None  # cache exposed_ids_by_rp + derived features
        self.load_windstorm_damage_curve()
        self.alter_damage_curves_for_windstorm_adaptation()
        self.load_max_damage_values()
        self.load_wind_maps()


    def load_wind_maps(self) -> None:
        """Load wind maps for all configured return periods from the 'wind_maps' output folder."""
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
    
        windstorm_config = self.model.config.get("hazards", {}).get("windstorm", {})
        debug_damage_stats = bool(windstorm_config.get("debug_damage_stats", False))
        #only_flooded_buildings = bool(windstorm_config.get("only_flooded_buildings", True))
        # wind_threshold = float(windstorm_config.get("wind_threshold_ms", 20.0))

        # DataFrame mapping buildings to household agents
        # agent_df = pd.DataFrame(
        #     {"building_id_of_household": self.households.var.building_id_of_household}
        # )


        # if "n_occupants" in self.households.buildings.columns:
        #     mask = self.households.buildings["n_occupants"] > 0
        # else:
        #     mask = pd.Series(True, index=self.households.buildings.index)
        # # Optional filter to focus only on flooded households (only if flood data available)
        # if only_flooded_buildings and "flooded" in self.households.buildings.columns:
        #     mask &= self.households.buildings["flooded"]
        # buildings = self.households.buildings[mask]

        #LECZ mask
        #lecz_mask = self.households.var.in_lecz.data == 1
        lecz_config = self.model.config.get("agent_settings", {}).get("households",{})
        only_lecz_buildings = bool(lecz_config.get("debug_damage_stats", True))

        if only_lecz_buildings:
            lecz_mask = self.households.var.in_lecz.data == 1

            lecz_building_ids = (
                self.households.var.building_id_of_household[lecz_mask]
            )

            mask = pd.Series(True, index=self.households.buildings.index)

            mask &= self.households.buildings["id"].isin(
                lecz_building_ids
            )

        buildings = self.households.buildings[mask]

        # lecz_building_ids = np.unique(
        #     self.households.var.building_id_of_household[lecz_mask]
        # )

        agent_df = pd.DataFrame(
            {
                "building_id_of_household":
                    self.households.var.building_id_of_household[lecz_mask]
            }
        )

        buildings= self.households.buildings[(self.households.buildings["id"].isin(lecz_building_ids))].copy()
    
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
            buildings_gdf[["id"]],
            geometry=gpd.points_from_xy(buildings_gdf["x"], buildings_gdf["y"]),
            crs="EPSG:4326",
        ).set_index("id")
        
        # Prefilter buildings via centroid wind speed.
        # Cache this — building positions and wind maps are both static across years.
        prefilter_buildings_key = tuple(sorted(buildings_gdf["id"].values))
        if (
            self._wind_prefilter_cache is None
            or self._wind_prefilter_cache["key"] != prefilter_buildings_key
        ):
            # All wind maps cover the same region and share the same CRS.
            # Reproject building points once using the first map's CRS, then
            # reuse xs/ys for every return period instead of repeating per-rp.
            wind_crs_ref = self.households.windstorm_maps[rps[0]].rio.crs
            buildings_points = (
                buildings_points_wgs84.to_crs(wind_crs_ref)
                if wind_crs_ref is not None and buildings_points_wgs84.crs != wind_crs_ref
                else buildings_points_wgs84
            )
            xs = xr.DataArray(buildings_points.geometry.x.values, dims="points")
            ys = xr.DataArray(buildings_points.geometry.y.values, dims="points")

            exposed_ids_by_rp: dict[float, np.ndarray] = {}
            for rp in rps:
                wind_map: xr.DataArray = self.households.windstorm_maps[rp]
                sampled_vals = np.nan_to_num(
                    wind_map.sel(
                        {wind_map.rio.x_dim: xs, wind_map.rio.y_dim: ys},
                        method="nearest",
                    ).values.squeeze().astype(np.float32),
                    nan=0.0,
                )
                mask = sampled_vals >= wind_threshold
                exposed_ids_by_rp[rp] = buildings_points.index.values[mask].astype(np.int32)

                if debug_damage_stats:
                    n_total = int(buildings_points.shape[0])
                    n_exposed = int(exposed_ids_by_rp[rp].size)
                    frac = (n_exposed / n_total) if n_total else 0.0
                    print(
                        f"Wind prefilter rp={int(rp)}: "
                        f"{n_exposed}/{n_total} buildings (frac={frac:.3f})"
                        f"above wind threshold {wind_threshold} m/s"
                    )

            all_exposed = [v for v in exposed_ids_by_rp.values() if v.size > 0]
            union_ids = np.unique(np.concatenate(all_exposed)) if all_exposed else np.array([], dtype=int)

            if union_ids.size == 0:
                if verbose:
                    for i, rp in enumerate(rps):
                        print(f"Wind Damages rp{rp}: 0 million (no exposed buildings)")
                        print(f"Wind Damages adapt rp{rp}: 0 million (no exposed buildings)")
                return damages_unprotected_w, damages_adapt_w

            buildings_union = buildings_gdf[buildings_gdf["id"].isin(union_ids)].copy().set_index("id")
            footprint_m2 = buildings_union.to_crs(buildings_union.estimate_utm_crs()).geometry.area.astype(np.float32)
            features_pts_wgs84 = gpd.GeoDataFrame(
                buildings_union[["object_type", "COST_STRUCTURAL_USD_SQM"]],
                geometry=buildings_union.centroid,
                crs="EPSG:4326",
            )
            self._wind_prefilter_cache = {
                "key": prefilter_buildings_key,
                "exposed_ids_by_rp": exposed_ids_by_rp,
                "union_ids": union_ids,
                "buildings_union": buildings_union,
                "footprint_m2": footprint_m2,
                "features_pts_wgs84": features_pts_wgs84,
            }
            # Invalidate exposure cache when building set changes
            self._wind_exposure_cache.clear()
        else:
            exposed_ids_by_rp = self._wind_prefilter_cache["exposed_ids_by_rp"]
            union_ids = self._wind_prefilter_cache["union_ids"]
            buildings_union = self._wind_prefilter_cache["buildings_union"]
            footprint_m2 = self._wind_prefilter_cache["footprint_m2"]
            features_pts_wgs84 = self._wind_prefilter_cache["features_pts_wgs84"]
    
        multi_curves = {
            "damages_structure_unprotected": self.households.wind_buildings_structure_curve[
                "building_unprotected"
            ],
            "damages_structure_wind_shutters": self.households.wind_buildings_structure_curve[
                "building_window_shutters"
            ],
        }

        # All wind maps cover the same region → same UTM zone. Reproject point
        # features and compute cell area once before the return-period loop.
        # Cache the reprojected reference map since it never changes.
        if rps[0] not in self._wind_maps_utm_cache:
            wind_map_ref0 = self.households.windstorm_maps[rps[0]]
            self._wind_maps_utm_cache[rps[0]] = self.reproject_to_utm(
                wind_map_ref0.where(wind_map_ref0 >= wind_threshold)
            )
        wind_map_ref = self._wind_maps_utm_cache[rps[0]]
        utm_crs = wind_map_ref.rio.crs
        features_pts_utm = features_pts_wgs84.to_crs(utm_crs)
        transform_ref = wind_map_ref.rio.transform(recalc=True)
        cell_area_m2 = float(abs(transform_ref.a * transform_ref.e))

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
    
            # Cache the reprojected masked wind map for this return period.
            if rp not in self._wind_maps_utm_cache:
                wind_map_rp: xr.DataArray = self.households.windstorm_maps[rp]
                self._wind_maps_utm_cache[rp] = self.reproject_to_utm(
                    wind_map_rp.where(wind_map_rp >= wind_threshold)
                )
            wind_map_masked = self._wind_maps_utm_cache[rp]

            features_pts = features_pts_utm.loc[ids].copy()

            # Pre-scale max damage so that (coverage_m2 * max_damage_structure) ~= footprint_m2 * cost_per_m2
            fp = footprint_m2.reindex(features_pts.index).to_numpy(np.float32)
            cost_m2 = features_pts["COST_STRUCTURAL_USD_SQM"].to_numpy(np.float32)
            features_pts["maximum_damage_structure"] = (cost_m2 * fp / max(cell_area_m2, 1e-6)).astype(np.float32)

            damage_buildings = VectorScannerMultiCurves(
                features=features_pts,
                hazard=wind_map_masked,
                multi_curves=multi_curves,
                exposure_cache=self._wind_exposure_cache,
                cache_key=rp,
            )
    
            # Convert scanner output to the needed columns 
            # out = pd.DataFrame({"id": damage_buildings.index.astype(int)})
            # out["damages_unprotected"] = damage_buildings["damages_structure_unprotected"].to_numpy()
            # out["damages_wind_shutters"] = damage_buildings["damages_structure_wind_shutters"].to_numpy()
            out = pd.DataFrame({
                "id": damage_buildings.index.astype(int),
                "damages_unprotected": damage_buildings["damages_structure_unprotected"].to_numpy(),
                "damages_wind_shutters": damage_buildings["damages_structure_wind_shutters"].to_numpy(),
            })

            if export_building_damages:
                fn_export = self.households.model.output_folder / "building_wind_damages"
                fn_export.mkdir(parents=True, exist_ok=True)
                out.to_parquet(
                    fn_export / f"building_wind_damages_rp{rp}_{self.households.model.current_time.year}.parquet"
                )
            
            damages_do_not_adapt_lecz, damages_adapt_lecz = (
                self.households.assign_wdamages_to_agents(agent_df, out)
            )

            damages_unprotected_w[i, lecz_mask] = damages_do_not_adapt_lecz
            damages_adapt_w[i, lecz_mask] = damages_adapt_lecz
            # damages_unprotected_w[i], damages_adapt_w[i] = (
            #     self.households.assign_wdamages_to_agents(agent_df, out)
            # )
    
            if verbose:
                print(f"Wind Damages rp{rp}: {round(damages_unprotected_w[i].sum() / 1e6)} million")
                print(f"Wind Damages adapt rp{rp}: {round(damages_adapt_w[i].sum() / 1e6)} million")
    
        return damages_unprotected_w, damages_adapt_w
    
    def calculate_ead(
            self,
            damages_do_not_adapt: np.ndarray,
            damages_adapt: np.ndarray,
            adapted: np.ndarray,
    ) -> np.ndarray:
        """Calculate the Expected Annual Damages (EAD) based on the damages for different return periods."""
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
        w_ead_usd_per_year = np.trapezoid(y=damages_sorted, x=prob_sorted, axis=0)

        return w_ead_usd_per_year

    @staticmethod
    def reproject_to_utm(hazard: xr.DataArray) -> xr.DataArray:
        """Reproject the hazard raster to a metric (UTM) CRS."""
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
    
    def return_period_windstorm(self) -> np.ndarray:
        """Simulate a windstorm event based on return periods and determine which households are affected.

        Returns:
            Array of indices of affected households.
        """
        #draw a single random number
        p_random = np.random.random()
        # Work with a locally sorted copy of return periods to ensure correct event selection
        return_periods_arr = np.asarray(self.households.windstorm_return_periods, dtype=float)
        sort_idx = np.argsort(return_periods_arr)  # ascending order
        sorted_return_periods = return_periods_arr[sort_idx]
        probabilities = 1.0 / sorted_return_periods

        if p_random >= probabilities.max():
            return np.array([], dtype=int)
        
        # find the event corresponding to the random draw
        event_idx = np.searchsorted(probabilities[::-1], p_random)
        event_idx = len(probabilities) - 1 - event_idx
        event = sorted_return_periods[event_idx]
        self.model.logger.info(
            "Return period windstorm event: %s years (p=%.4f, random draw=%.4f)",
            event,
            probabilities[event_idx],
            p_random,
        )

        windstorm_map: xr.DataArray = self.households.windstorm_maps[event]

        # cache household coordinates in windstorm_map CRS (Nx2 numpy array)
        if not hasattr(self, "_household_xy_wind"):
            import pyproj

            x, y = np.array(self.households.buildings.x), np.array(self.households.buildings.y)
            transformer = pyproj.Transformer.from_crs(
                "EPSG:4326", windstorm_map.rio.crs, always_xy=True
            )
            self._building_xy_wind = np.array(transformer.transform(x, y)).T

        # sample windstorm map using clipped coordinates
        sampled_values = sample_from_map(
            array=windstorm_map.values,
            coords=self._building_xy_wind,
            gt=windstorm_map.rio.transform(recalc=True).to_gdal(),
            out_of_bounds_value=np.nan,
        )

        # Use a wind speed threshold to determine affected households — must match
        # the wind_threshold_ms used in wind_risk.py for damage prefiltering
        minimum_wind_speed_ms = float(
            self.model.config.get("hazards", {})
            .get("windstorm", {})
            .get("wind_threshold_ms", 20.0)
        )
        windstorm_building_indices = np.where(sampled_values > minimum_wind_speed_ms)[0]
        #get building IDs of affected buildings
        windstorm_building_ids = self.households.buildings.loc[
            windstorm_building_indices, "id"
        ].values.astype(int)

        #get indices of households located in affected buildings
        windstorm_household_indices = np.where(
            np.isin(self.households.var.building_id_of_household.data, windstorm_building_ids)
        )[0]

        return windstorm_household_indices
