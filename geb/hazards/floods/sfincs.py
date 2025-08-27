import json
from collections import deque
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import xarray as xr
import zarr
from rioxarray.merge import merge_arrays
from shapely.geometry import shape
from shapely.geometry.point import Point

from ...hydrology.HRUs import load_geom, load_grid
from ...hydrology.landcover import OPEN_WATER, SEALED
from ...workflows.io import open_zarr, to_zarr
from ...workflows.raster import reclassify
from .build_model import build_sfincs, build_sfincs_coastal
from .estimate_discharge_for_return_periods import estimate_discharge_for_return_periods
from .postprocess_model import read_maximum_flood_depth
from .run_sfincs_for_return_periods import (
    run_sfincs_for_return_periods,
    run_sfincs_for_return_periods_coastal,
)
from .sfincs_utils import run_sfincs_simulation
from .update_model_forcing import (
    update_sfincs_model_forcing,
)


class SFINCS:
    """The class that implements all methods to setup, run, and post-process the SFINCS hydrodynamic model.

    Args:
        model: The GEB model instance.
        n_timesteps: The number of timesteps to keep in memory for discharge calculations (default is 10).
    """

    def __init__(self, model, n_timesteps=10) -> None:
        self.model = model
        self.config = (
            self.model.config["hazards"]["floods"]
            if "floods" in self.model.config["hazards"]
            else {}
        )

        self.HRU = model.hydrology.HRU

        if self.model.simulate_hydrology:
            self.hydrology = model.hydrology
            self.n_timesteps = n_timesteps
            self.discharge_per_timestep = deque(maxlen=self.n_timesteps)
            self.soil_moisture_per_timestep = deque(maxlen=self.n_timesteps)
            self.max_water_storage_per_timestep = deque(maxlen=self.n_timesteps)
            self.saturated_hydraulic_conductivity_per_timestep = deque(
                maxlen=self.n_timesteps
            )
            self.soil_storage_capacity_per_timestep = deque(maxlen=self.n_timesteps)

    def sfincs_model_root(self, basin_id) -> Path:
        folder: Path = self.model.simulation_root / "SFINCS" / str(basin_id)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def sfincs_simulation_root(self, event) -> Path:
        name: str = self.get_event_name(event)
        folder: Path = (
            self.sfincs_model_root(name)
            / "simulations"
            / f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}"
        )
        if self.model.multiverse_name:
            folder: Path = folder / self.model.multiverse_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def get_event_name(self, event):
        if "name" in event:
            return event["name"]
        elif "basin_id" in event:
            return event["basin_id"]
        else:
            return "run"

    def get_utm_zone(self, region_file: Path | str) -> str:
        """Determine the UTM zone based on the centroid of the region geometry.

        Args:
            region_file: Path to the region geometry file.

        Returns:
            The EPSG code for the UTM zone of the centroid of the region.
        """
        region: gpd.GeoDataFrame = load_geom(region_file)

        # Calculate the central longitude of the dataset
        centroid: Point = region.union_all().centroid

        # Determine the UTM zone based on the longitude
        utm_zone: int = int((centroid.x + 180) // 6) + 1

        # Determine if the data is in the Northern or Southern Hemisphere
        # The EPSG code for UTM in the northern hemisphere is EPSG:326xx (xx = zone)
        # The EPSG code for UTM in the southern hemisphere is EPSG:327xx (xx = zone)
        if centroid.y > 0:
            utm_crs: str = f"EPSG:326{utm_zone}"  # Northern hemisphere
        else:
            utm_crs: str = f"EPSG:327{utm_zone}"  # Southern hemisphere
        return utm_crs

    def mirror_dataset_horizontal(self, ds, var_name="initial_soil_moisture"):
        """Mirror an xarray Dataset horizontally.

        Mirrors an xarray Dataset across a horizontal line
        based on the bottomleft coordinate.

        Parameters:
            ds (xr.Dataset): Input dataset with 'x' and 'y' dimensions.
            var_name (str): Name of the data variable to mirror.

        Returns:
            xr.Dataset: Horizontally mirrored dataset.
        """
        # Bottom y
        y0 = ds.y.min().item()

        # Mirror y coordinates only
        mirrored_y = 2 * y0 - ds.y

        # Assign mirrored y
        ds_mirrored = ds.assign_coords(y=mirrored_y)

        # Flip the data along y-axis only
        ds_mirrored[var_name] = ds_mirrored[var_name].isel(y=slice(None, None, -1))

        return ds_mirrored

    def build(self, event) -> None:
        build_parameters = {}

        event_name = self.get_event_name(event)
        if "region" in event:
            if event["region"] is None:
                model_bbox = self.hydrology.grid.bounds
                build_parameters["bbox"] = (
                    model_bbox[0] + 0.1,
                    model_bbox[1] + 0.1,
                    model_bbox[2] - 0.1,
                    model_bbox[3] - 0.1,
                )
            else:
                raise NotImplementedError

        if (
            "simulate_coastal_floods" in self.model.config["general"]
            and self.model.config["general"]["simulate_coastal_floods"]
        ):
            build_parameters["simulate_coastal_floods"] = True

        model_root: Path = self.sfincs_model_root(event_name)

        if (
            self.config["force_overwrite"]
            or not (self.sfincs_model_root(event_name) / "sfincs.inp").exists()
        ):
            build_parameters = self.get_build_parameters(model_root)
            build_sfincs(**build_parameters)

    def set_forcing(self, event, start_time, precipitation_scale_factor=1.0) -> None:
        if self.model.simulate_hydrology:
            n_timesteps: int = min(self.n_timesteps, len(self.discharge_per_timestep))
            routing_substeps: int = self.discharge_per_timestep[0].shape[0]
            discharge_grid = self.hydrology.grid.decompress(
                np.vstack(self.discharge_per_timestep)
            )
            first_timestep_of_event = len(self.discharge_per_timestep) - n_timesteps  #

            # when SFINCS starts with high values, this leads to numerical instabilities. Therefore, we first start with very low discharge and then build up slowly to timestep 0
            # TODO: Check if this is a right approach
            discharge_grid = np.vstack(
                [
                    np.full_like(
                        discharge_grid[:routing_substeps, :, :], fill_value=np.nan
                    ),
                    discharge_grid,
                ]
            )  # prepend zeros

            for i in range(routing_substeps - 1, -1, -1):
                discharge_grid[i] = discharge_grid[i + 1] * 0.3

            # convert the discharge grid to an xarray DataArray
            discharge_grid = xr.DataArray(
                data=discharge_grid,
                coords={
                    "time": pd.date_range(
                        end=self.model.current_time
                        + self.model.timestep_length
                        - self.model.timestep_length / routing_substeps,
                        periods=(n_timesteps + 1)
                        * routing_substeps,  # +1 because we prepend the discharge above
                        freq=self.model.timestep_length / routing_substeps,
                        inclusive="right",
                    ),
                    "y": self.hydrology.grid.lat,
                    "x": self.hydrology.grid.lon,
                },
                dims=["time", "y", "x"],
                name="discharge",
            )

            self.initial_soil_moisture_grid = xr.Dataset(
                {
                    "initial_soil_moisture": (
                        ["y", "x"],
                        np.flipud(
                            self.HRU.decompress(
                                self.soil_moisture_per_timestep[first_timestep_of_event]
                            )
                        ),  # take first time step of soil moisture
                    )
                },  # deque
                coords={
                    "y": self.HRU.lat,
                    "x": self.HRU.lon,
                },
            )

            # GEB HRU data is somehow mirrored, so we need to mirror it back
            self.initial_soil_moisture_grid = self.mirror_dataset_horizontal(
                self.initial_soil_moisture_grid, "initial_soil_moisture"
            )
            self.initial_soil_moisture_grid = self.initial_soil_moisture_grid.sortby(
                "x"
            )
            self.initial_soil_moisture_grid = self.initial_soil_moisture_grid.sortby(
                "y"
            )

            self.max_water_storage_grid = xr.Dataset(
                {
                    "max_water_storage": (
                        ["y", "x"],
                        self.HRU.decompress(
                            self.max_water_storage_per_timestep[
                                first_timestep_of_event
                            ],
                        ),  # take first time step of soil moisture
                    )
                },  # deque
                coords={
                    "y": self.HRU.lat,
                    "x": self.HRU.lon,
                },
            )

            self.max_water_storage_grid = self.mirror_dataset_horizontal(
                self.max_water_storage_grid, "max_water_storage"
            )
            self.max_water_storage_grid = self.max_water_storage_grid.sortby("x")
            self.max_water_storage_grid = self.max_water_storage_grid.sortby("y")

            soil_storage_capacity_grid = xr.Dataset(
                {
                    "soil_storage_capacity": (
                        ["y", "x"],
                        self.HRU.decompress(
                            self.soil_storage_capacity_per_timestep[
                                first_timestep_of_event
                            ]
                        ),  # take first time step of soil moisture
                    )
                },  # deque
                coords={
                    "y": self.HRU.lat,
                    "x": self.HRU.lon,
                },
            )

            soil_storage_capacity_grid = self.mirror_dataset_horizontal(
                soil_storage_capacity_grid, "soil_storage_capacity"
            )
            soil_storage_capacity_grid = soil_storage_capacity_grid.sortby("x")
            soil_storage_capacity_grid = soil_storage_capacity_grid.sortby("y")

            saturated_hydraulic_conductivity_grid = xr.Dataset(
                {
                    "saturated_hydraulic_conductivity": (
                        ["y", "x"],
                        self.HRU.decompress(
                            self.saturated_hydraulic_conductivity_per_timestep[
                                first_timestep_of_event
                            ]
                        ),  # take first time step of soil moisture
                    )
                },  # deque
                coords={
                    "y": self.HRU.lat,
                    "x": self.HRU.lon,
                },
            )

            saturated_hydraulic_conductivity_grid = self.mirror_dataset_horizontal(
                saturated_hydraulic_conductivity_grid,
                "saturated_hydraulic_conductivity",
            )
            saturated_hydraulic_conductivity_grid = (
                saturated_hydraulic_conductivity_grid.sortby("x")
            )
            saturated_hydraulic_conductivity_grid = (
                saturated_hydraulic_conductivity_grid.sortby("y")
            )

            self.initial_soil_moisture_grid.raster.set_crs(
                self.model.crs
            )  # for soil moisture
            saturated_hydraulic_conductivity_grid.raster.set_crs(
                self.model.crs
            )  # for saturated hydraulic conductivity
            self.max_water_storage_grid.raster.set_crs(
                self.model.crs
            )  # for max water storage
            soil_storage_capacity_grid.raster.set_crs(
                self.model.crs
            )  # for the soil water storage capacity

        else:
            routing_substeps: int = 24  # when setting 0 it doesn't matter so much how many routing_substeps. 24 is a reasonable default.
            n_timesteps: int = (event["end_time"] - event["start_time"]).days
            time = pd.date_range(
                end=self.model.current_time
                - self.model.timestep_length / routing_substeps,
                periods=(n_timesteps + 1)
                * routing_substeps,  # +1 because we prepend the discharge above
                freq=self.model.timestep_length / routing_substeps,
                inclusive="right",
            )
            discharge_grid: npt.NDArray[np.float32] = np.zeros(
                shape=(len(time), *self.model.hydrology.grid.mask.shape),
                dtype=np.float32,
            )
            discharge_grid: xr.DataArray = xr.DataArray(
                data=discharge_grid,
                coords={
                    "time": time,
                    "y": self.hydrology.grid.lat,
                    "x": self.hydrology.grid.lon,
                },
                dims=["time", "y", "x"],
                name="discharge",
            )

        discharge_grid: xr.Dataset = xr.Dataset({"discharge": discharge_grid})

        discharge_grid.raster.set_crs(self.model.crs)
        end_time = discharge_grid.time[-1] + pd.Timedelta(
            self.model.timestep_length / routing_substeps
        )
        discharge_grid: xr.Dataset = discharge_grid.sel(
            time=slice(start_time, end_time)
        )

        precipitation_grid: list[xr.DataArray] | xr.DataArray = self.model.forcing[
            "pr_hourly"
        ]

        if isinstance(precipitation_grid, list):
            precipitation_grid: list[xr.DataArray] = [
                pr * precipitation_scale_factor for pr in precipitation_grid
            ]
        else:
            precipitation_grid: xr.DataArray = (
                precipitation_grid * precipitation_scale_factor
            )

        event_name: str = self.get_event_name(event)

        update_sfincs_model_forcing(
            model_root=self.sfincs_model_root(event_name),
            simulation_root=self.sfincs_simulation_root(event),
            event=event,
            forcing_method="precipitation",
            discharge_grid=discharge_grid,
            soil_water_capacity_grid=soil_storage_capacity_grid,
            max_water_storage_grid=self.max_water_storage_grid,
            saturated_hydraulic_conductivity_grid=saturated_hydraulic_conductivity_grid,
            precipitation_grid=precipitation_grid,
            uparea_discharge_grid=None,
        )

    def run_single_event(self, event, start_time, precipitation_scale_factor=1.0):
        self.build(event)
        model_root: Path = self.sfincs_model_root(self.get_event_name(event))
        simulation_root: Path = self.sfincs_simulation_root(event)

        self.set_forcing(event, start_time, precipitation_scale_factor)
        self.model.logger.info(f"Running SFINCS for {self.model.current_time}...")

        run_sfincs_simulation(
            simulation_root=simulation_root,
            model_root=model_root,
            gpu=self.config["SFINCS"]["gpu"],
        )
        flood_map: xr.DataArray = read_maximum_flood_depth(
            model_root=model_root,
            simulation_root=simulation_root,
        )  # xc, yc is for x and y in rotated grid

        flood_map_name: str = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
        if self.model.multiverse_name:
            flood_map_name: str = self.model.multiverse_name + " - " + flood_map_name
        flood_map: xr.DataArray = to_zarr(
            flood_map,
            self.model.output_folder / "flood_maps" / flood_map_name,
            crs=flood_map.rio.crs,
        )
        print(f"Saving flood map for member: {self.model.multiverse_name}")

        damages = self.flood(flood_map=flood_map)
        return damages

    def build_mask_for_coastal_sfincs(self) -> gpd.GeoDataFrame:
        """Builds a mask to define the active cells and boundaries for the coastal SFINCS model.

        Returns:
            GeoDataFrame: A GeoDataFrame containing the coastal mask.
        """
        # Load the dataset (assumes NetCDF with CF conventions and georeferencing info)
        mask = xr.load_dataset(self.model.files["other"]["drainage/mask"])

        # Extract the mask variable
        mask_var = mask["mask"]

        # Make sure it has a CRS
        if mask_var.rio.crs is None:
            mask_var = mask_var.rio.write_crs(
                "EPSG:4326", inplace=False
            )  # or your known CRS

        # Extract binary mask values
        mask_data = mask_var.values.astype(np.uint8)

        # Get transform from raster metadata
        transform = mask_var.rio.transform()

        # Use rasterio.features.shapes() to get polygons for each contiguous region with same value
        shapes = rasterio.features.shapes(mask_data, mask=None, transform=transform)

        # Build GeoDataFrame from the shapes generator
        records = [{"geometry": shape(geom), "value": value} for geom, value in shapes]

        gdf = gpd.GeoDataFrame.from_records(records)
        gdf.set_geometry("geometry", inplace=True)
        gdf.crs = mask_var.rio.crs
        # include a 1km buffer to the mask to include the coastal areas
        # Keep only mask == 1
        gdf = gdf[gdf["value"] == 1]
        # gdf.geometry = gdf.geometry.buffer(0.00833)

        return gdf

    def build_coastal_boundary_mask(self) -> gpd.GeoDataFrame:
        """Builds a mask to define the coastal boundaries for the SFINCS model.

        Returns:
            GeoDataFrame: A GeoDataFrame containing the coastal boundary mask.
        """
        lecz = xr.load_dataset(
            self.model.files["other"]["landsurface/low_elevation_coastal_zone"]
        )

        # Make sure it has a CRS
        if lecz.rio.crs is None:
            lecz = lecz.rio.write_crs(
                "EPSG:4326", inplace=False
            )  # check CRS for later applications

        # Extract binary mask values
        lecz_data = lecz["low_elevation_coastal_zone"].values.astype(np.uint8)

        # Get transform from raster metadata
        transform = lecz.rio.transform()

        # Use rasterio.features.shapes() to get polygons for each contiguous region with same value
        shapes = rasterio.features.shapes(lecz_data, mask=None, transform=transform)

        # Build GeoDataFrame from the shapes generator
        records = [{"geometry": shape(geom), "value": value} for geom, value in shapes]

        gdf = gpd.GeoDataFrame.from_records(records)
        gdf.set_geometry("geometry", inplace=True)
        gdf = gdf.set_crs(lecz.rio.crs, inplace=True)
        gdf = gdf[gdf["value"] == 1]  # Keep only mask == 1
        return gdf

    def get_coastal_return_period_maps(self) -> dict[int, xr.DataArray]:
        """This function models coastal flooding for the return periods specified in the model config.

        Returns:
            dict[int, xr.DataArray]: A dictionary mapping return periods to their respective flood maps.
        """
        coastal_mask = self.build_mask_for_coastal_sfincs()
        boundary_mask = self.build_coastal_boundary_mask()
        model_root: Path = self.sfincs_model_root("entire_region_coastal")
        build_parameters = self.get_build_parameters(model_root)
        build_parameters["region"] = coastal_mask
        build_parameters["boundary_mask"] = boundary_mask
        build_sfincs_coastal(
            **build_parameters,
        )

        rp_maps_coastal = run_sfincs_for_return_periods_coastal(
            model=self.model,
            model_root=model_root,
            gpu=self.config["SFINCS"]["gpu"],
            export_dir=self.model.output_folder / "flood_maps",
            clean_working_dir=True,
            return_periods=self.config["return_periods"],
        )
        return rp_maps_coastal

    def get_riverine_return_period_maps(self) -> dict[int, xr.DataArray]:
        """This function models riverine flooding for the return periods specified in the model config.

        Returns:
            dict[int, xr.DataArray]: A dictionary mapping return periods to their respective flood maps.
        """
        # close the zarr store
        if hasattr(self.model, "reporter"):
            self.model.reporter.variables["discharge_daily"].close()

        model_root: Path = self.sfincs_model_root("entire_region")
        if self.config["force_overwrite"] or not (model_root / "sfincs.inp").exists():
            build_sfincs(
                **self.get_build_parameters(model_root),
            )
        estimate_discharge_for_return_periods(
            model_root,
            discharge=self.discharge_spinup_ds,
            waterbody_ids=self.model.hydrology.grid.decompress(
                self.model.hydrology.grid.var.waterBodyID
            ),
            rivers=self.rivers,
            return_periods=self.config["return_periods"],
        )

        rp_maps_riverine = run_sfincs_for_return_periods(
            model_root=model_root,
            return_periods=self.config["return_periods"],
            gpu=self.config["SFINCS"]["gpu"],
            export_dir=self.model.output_folder / "flood_maps",
            clean_working_dir=True,
        )

        if hasattr(self.model, "reporter"):
            # and re-open afterwards
            self.model.reporter.variables["discharge_daily"] = zarr.ZipStore(
                self.model.config["report_hydrology"]["discharge_daily"]["path"],
                mode="a",
            )
        return rp_maps_riverine

    def merge_return_period_maps(
        self,
        rp_maps_coastal: dict[int, xr.DataArray],
        rp_maps_riverine: dict[int, xr.DataArray],
    ) -> None:
        """Merges the return period maps for riverine and coastal floods into a single dataset.

        Args:
            rp_maps_coastal: Dictionary of coastal return period maps.
            rp_maps_riverine: Dictionary of riverine return period maps.
        """
        for return_period in self.config["return_periods"]:
            if rp_maps_coastal is None:
                riverine_da = rp_maps_riverine[return_period]
                riverine_da.to_zarr(
                    self.model.output_folder / "flood_maps" / f"{return_period}.zarr",
                    mode="w",
                )
                continue

            coastal_da = rp_maps_coastal[return_period]
            riverine_da = rp_maps_riverine[return_period]

            # --- 2. Get union bounds ---
            riv_bounds = riverine_da.rio.bounds()  # (minx, miny, maxx, maxy)
            coa_bounds = coastal_da.rio.bounds()

            minx = min(riv_bounds[0], coa_bounds[0])
            miny = min(riv_bounds[1], coa_bounds[1])
            maxx = max(riv_bounds[2], coa_bounds[2])
            maxy = max(riv_bounds[3], coa_bounds[3])

            # --- 3. Pick resolution ---
            # Use riverine resolution (y is negative if north-up, so take abs)
            res_x, res_y = riverine_da.rio.resolution()
            res_x = abs(res_x)
            res_y = abs(res_y)

            # --- 4. Build template coords ---
            width = int(np.ceil((maxx - minx) / res_x))
            height = int(np.ceil((maxy - miny) / res_y))

            x_coords = minx + (np.arange(width) + 0.5) * res_x
            y_coords = maxy - (np.arange(height) + 0.5) * res_y  # top→bottom

            template = xr.DataArray(
                np.full((height, width), np.nan, dtype=riverine_da.dtype),
                coords={"y": y_coords, "x": x_coords},
                dims=("y", "x"),
            ).rio.write_crs(riverine_da.rio.crs)

            # --- 5. Reproject both datasets to the template ---
            riverine_reproj = riverine_da.rio.reproject_match(template)
            coastal_reproj = coastal_da.rio.reproject_match(template)

            # --- 6. Merge via maximum ---
            rp_map = xr.concat([riverine_reproj, coastal_reproj], dim="stacked").max(
                dim="stacked", skipna=True
            )
            rp_map.rio.write_crs(riverine_da.rio.crs)

            rp_map.to_zarr(
                self.model.output_folder / "flood_maps" / f"{return_period}.zarr",
                mode="w",
            )

    def run(self, event) -> None:
        start_time = event["start_time"]

        if self.model.config["hazards"]["floods"]["flood_risk"]:
            scale_factors = pd.read_parquet(
                self.model.files["table"]["hydrodynamics/risk_scaling_factors"]
            )
            scale_factors["return_period"] = 1 / scale_factors["exceedance_probability"]
            damages_list = []
            return_periods_list = []
            exceedence_probabilities_list = []

            for _, row in scale_factors.iterrows():
                return_period = row["return_period"]
                exceedence_probability = row["exceedance_probability"]

                damages = self.run_single_event(
                    event, start_time, precipitation_scale_factor=row["scaling_factor"]
                )

                damages_list.append(damages)
                return_periods_list.append(return_period)
                exceedence_probabilities_list.append(exceedence_probability)

            print(damages_list)
            print(return_periods_list)
            print(exceedence_probabilities_list)

            plt.plot(return_periods_list, damages_list)
            plt.xlabel("Return period")
            plt.ylabel("Flood damages [euro]")
            plt.title("Damages per return period")
            plt.show()

            inverted_damage_list = damages_list[::-1]
            inverted_exceedence_probabilities_list = exceedence_probabilities_list[::-1]

            expected_annual_damage = np.trapz(
                y=inverted_damage_list, x=inverted_exceedence_probabilities_list
            )  # np.trapezoid or np.trapz -> depends on np version
            print(f"exptected annual damage is: {expected_annual_damage}")

        else:
            self.run_single_event(event, start_time)

    def flood(self, flood_map):
        damages = self.model.agents.households.flood(flood_map=flood_map)
        return damages

    def save_discharge(self) -> None:
        self.discharge_per_timestep.append(
            self.hydrology.grid.var.discharge_m3_s_per_substep
        )  # this is a deque, so it will automatically remove the oldest discharge

    def save_soil_moisture(self) -> None:  # is used in driver.py on every timestep
        # load and process initial soil moisture grid
        w_copy = self.HRU.var.w.copy()
        w_copy[:, self.HRU.var.land_use_type == SEALED] = 0
        w_copy[:, self.HRU.var.land_use_type == OPEN_WATER] = 0
        self.initial_soil_moisture_grid = w_copy[:2].sum(axis=0)
        self.soil_moisture_per_timestep.append(self.initial_soil_moisture_grid)

    def save_max_soil_moisture(self) -> None:
        # smax
        ws_copy = self.HRU.var.ws.copy()
        ws_copy[:, self.HRU.var.land_use_type == SEALED] = 0
        ws_copy[:, self.HRU.var.land_use_type == OPEN_WATER] = 0
        self.max_water_storage_grid = ws_copy[:2].sum(axis=0)
        self.max_water_storage_per_timestep.append(self.max_water_storage_grid)

    def save_soil_storage_capacity(self) -> None:
        self.soil_storage_capacity_grid = (
            self.max_water_storage_grid - self.initial_soil_moisture_grid
        )
        self.soil_storage_capacity_per_timestep.append(self.soil_storage_capacity_grid)

    def save_saturated_hydraulic_conductivity(self) -> None:
        saturated_hydraulic_conductivity_copy = (
            self.HRU.var.saturated_hydraulic_conductivity.copy()
        )
        saturated_hydraulic_conductivity_copy[
            :, self.HRU.var.land_use_type == SEALED
        ] = 0
        saturated_hydraulic_conductivity_copy[
            :, self.HRU.var.land_use_type == OPEN_WATER
        ] = 0
        saturated_hydraulic_conductivity_copy = saturated_hydraulic_conductivity_copy[
            :2
        ].sum(axis=0)

        saturated_hydraulic_conductivity_copy = (
            saturated_hydraulic_conductivity_copy * 1000
        ) / 24
        self.saturated_hydraulic_conductivity_per_timestep.append(
            saturated_hydraulic_conductivity_copy
        )

    @property
    def discharge_spinup_ds(self) -> xr.DataArray:
        """Open the discharge datasets from the model output folder."""
        da: xr.DataArray = open_zarr(
            self.model.output_folder
            / "report"
            / "spinup"
            / "hydrology.routing"
            / "discharge_daily.zarr"
        )

        # start_time = pd.to_datetime(ds.time[0].item()) + pd.DateOffset(years=10)
        # ds = ds.sel(time=slice(start_time, ds.time[-1]))

        # # make sure there is at least 20 years of data
        # if not len(ds.time.groupby(ds.time.dt.year).groups) >= 20:
        #     raise ValueError(
        #         """Not enough data available for reliable spinup, should be at least 20 years of data left.
        #         Please run the model for at least 30 years (10 years of data is discarded)."""
        #     )

        return da

    @property
    def rivers(self):
        """Load the river geometry from the model files.

        Returns:
            A GeoDataFrame containing the river geometry.
        """
        return load_geom(self.model.files["geom"]["routing/rivers"])

    @property
    def mannings(self) -> xr.DataArray:
        """Get the Manning's n values for the land cover types."""
        mannings = reclassify(
            self.land_cover,
            self.land_cover_mannings_rougness_classification.set_index(
                "esa_worldcover"
            )["N"].to_dict(),
            method="lookup",
        )
        return mannings

    @property
    def land_cover(self) -> xr.DataArray:
        """Get the land cover classification for the model.

        Returns:
            An xarray DataArray containing the land cover classification.
        """
        return open_zarr(self.model.files["other"]["landcover/classification"])

    @property
    def land_cover_mannings_rougness_classification(self) -> pd.DataFrame:
        """Get the land cover classification table for Manning's roughness.

        Returns:
            A DataFrame containing the land cover classification for Manning's roughness.
        """
        return pd.DataFrame(
            data=[
                [10, "Tree cover", 10, 0.12],
                [20, "Shrubland", 20, 0.05],
                [30, "Grasland", 30, 0.034],
                [40, "Cropland", 40, 0.037],
                [50, "Built-up", 50, 0.1],
                [60, "Bare / sparse vegetation", 60, 0.023],
                [70, "Snow and Ice", 70, 0.01],
                [80, "Permanent water bodies", 80, 0.02],
                [90, "Herbaceous wetland", 90, 0.035],
                [95, "Mangroves", 95, 0.07],
                [100, "Moss and lichen", 100, 0.025],
                [0, "No data", 0, 0.1],
            ],
            columns=["esa_worldcover", "description", "landuse", "N"],
        )

    @property
    def crs(self) -> str:
        """Get the coordinate reference system (CRS) for the model.

        When the CRS is set in the configuration, it will return that value.
        If the CRS is set to "auto", it will determine the UTM zone based on the routing subbasins geometry.

        Returns:
             The CRS string, either "auto" or the determined UTM zone.
        """
        crs: str = self.config["crs"]
        if crs == "auto":
            crs: str = self.get_utm_zone(self.model.files["geom"]["routing/subbasins"])
        return crs

    def get_build_parameters(self, model_root: Path) -> dict[str, Any]:
        """Get the parameters needed to build the SFINCS model.

        Args:
            model_root: The root directory for the SFINCS model.

        Returns:
            A dictionary containing the parameters needed to build the SFINCS model.
        """
        with open(self.model.files["dict"]["hydrodynamics/DEM_config"]) as f:
            DEM_config = json.load(f)
        for entry in DEM_config:
            entry["elevtn"] = open_zarr(
                self.model.files["other"][entry["path"]]
            ).to_dataset(name="elevtn")

        return {
            "model_root": model_root,
            "region": load_geom(self.model.files["geom"]["routing/subbasins"]),
            "DEMs": DEM_config,
            "rivers": self.rivers,
            "discharge": self.discharge_spinup_ds,
            "waterbody_ids": self.model.hydrology.grid.decompress(
                self.model.hydrology.grid.var.waterBodyID
            ),
            "river_width_alpha": self.model.hydrology.grid.decompress(
                self.model.var.river_width_alpha
            ),
            "river_width_beta": self.model.hydrology.grid.decompress(
                self.model.var.river_width_beta
            ),
            "mannings": self.mannings,
            "resolution": self.config["resolution"],
            "nr_subgrid_pixels": self.config["nr_subgrid_pixels"],
            "crs": self.crs,
            "depth_calculation_method": self.model.config["hydrology"]["routing"][
                "river_depth"
            ]["method"],
            "depth_calculation_parameters": self.model.config["hydrology"]["routing"][
                "river_depth"
            ]["parameters"]
            if "parameters" in self.model.config["hydrology"]["routing"]["river_depth"]
            else {},
            "mask_flood_plains": False,  # setting this to True sometimes leads to errors
        }
