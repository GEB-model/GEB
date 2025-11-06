"""Class to setup, run, and post-process the SFINCS hydrodynamic model."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import zarr
from shapely.geometry import shape
from shapely.geometry.point import Point

from geb.module import Module
from geb.typing import ArrayFloat32, TwoDArrayInt32
from geb.workflows.io import load_geom

from ...hydrology.landcovers import OPEN_WATER as OPEN_WATER, SEALED as SEALED
from ...workflows.io import open_zarr, to_zarr
from ...workflows.raster import reclassify
from .sfincs import (
    MultipleSFINCSSimulations,
    SFINCSRootModel,
    SFINCSSimulation,
    set_river_outflow_boundary_condition,
)

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology as Hydrology


class Floods(Module):
    """The class that implements all methods to setup, run, and post-process hydrodynamic flood models.

    Args:
        model: The GEB model instance.
        n_timesteps: The number of timesteps to keep in memory for discharge calculations (default is 10).
    """

    def __init__(self, model: GEBModel, longest_flood_event_in_days: int = 10) -> None:
        """Initializes the Floods class.

        Args:
            model: The GEB model instance.
            longest_flood_event_in_days: The number of timesteps to keep in memory for discharge calculations (default is 10).
        """
        super().__init__(model)

        self.model: GEBModel = model
        self.config: dict[str, Any] = (
            self.model.config["hazards"]["floods"]
            if "floods" in self.model.config["hazards"]
            else {}
        )

        self.HRU = model.hydrology.HRU

        if self.model.simulate_hydrology:
            self.hydrology: Hydrology = model.hydrology
            self.longest_flood_event_in_days: int = longest_flood_event_in_days

            self.var.discharge_per_timestep: deque[ArrayFloat32] = deque(
                maxlen=self.longest_flood_event_in_days
            )
            self.var.runoff_m_per_timestep: deque[ArrayFloat32] = deque(
                maxlen=self.longest_flood_event_in_days
            )

    @property
    def name(self) -> str:
        """The name of the module."""
        return "floods"

    def spinup(self) -> None:
        """Spinup method for the Floods module.

        Currently, this method does nothing as flood simulations do not require spinup.
        """
        pass

    def step(self) -> None:
        """Steps the Floods module.

        Currently, this method does nothing as flood simulations are handled in the HazardDriver.
        """
        pass

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

    def build(self, name: str) -> SFINCSRootModel:
        """Builds or reads a SFINCS model without any forcing.

        Before using this model, forcing must be set.

        When the model already exists and force_overwrite is False, the existing model is read.

        Args:
            name: Name of the SFINCS model (used for the model root directory).

        Returns:
            The built or read SFINCSRootModel instance.
        """
        sfincs_model = SFINCSRootModel(self.model, name)
        if self.config["force_overwrite"] or not sfincs_model.exists():
            with open(self.model.files["dict"]["hydrodynamics/DEM_config"]) as f:
                DEM_config = json.load(f)
                for entry in DEM_config:
                    entry["elevtn"] = open_zarr(
                        self.model.files["other"][entry["path"]]
                    ).to_dataset(name="elevtn")

            sfincs_model.build(
                region=load_geom(self.model.files["geom"]["routing/subbasins"]),
                DEMs=DEM_config,
                rivers=self.model.hydrology.routing.rivers,
                discharge=self.discharge_spinup_ds,
                river_width_alpha=self.model.hydrology.grid.decompress(
                    self.model.var.river_width_alpha
                ),
                river_width_beta=self.model.hydrology.grid.decompress(
                    self.model.var.river_width_beta
                ),
                mannings=self.mannings,
                resolution=self.config["resolution"],
                nr_subgrid_pixels=self.config["nr_subgrid_pixels"],
                crs=self.crs,
                depth_calculation_method=self.model.config["hydrology"]["routing"][
                    "river_depth"
                ]["method"],
                depth_calculation_parameters=self.model.config["hydrology"]["routing"][
                    "river_depth"
                ]["parameters"]
                if "parameters"
                in self.model.config["hydrology"]["routing"]["river_depth"]
                else {},
                mask_flood_plains=False,  # setting this to True sometimes leads to errors
            )
        else:
            sfincs_model.read()

        return sfincs_model

    def set_forcing(
        self,
        sfincs_model: SFINCSRootModel,
        start_time: datetime,
        end_time: datetime,
    ) -> SFINCSSimulation:
        """Sets the forcing for a SFINCS simulation.

        Depending on the forcing method in the config, either headwater point discharge
        or precipitation is used as forcing.

        Args:
            sfincs_model: The SFINCSRootModel instance to create the simulation from.
            start_time: The start time of the flood event.
            end_time: The end time of the flood event.

        Returns:
            The created SFINCSSimulation instance with the forcing set.

        Raises:
            ValueError: If the forcing method is unknown.
        """
        # Save the flood depth to a zarr file
        sfincs_simulation_name: str = f"{start_time.strftime(format='%Y%m%dT%H%M%S')} - {end_time.strftime(format='%Y%m%dT%H%M%S')}"
        if self.model.multiverse_name:
            sfincs_simulation_name: str = (
                self.model.multiverse_name + "/" + sfincs_simulation_name
            )

        simulation: SFINCSSimulation = sfincs_model.create_simulation(
            simulation_name=sfincs_simulation_name,
            start_time=start_time,
            end_time=end_time,
            write_figures=self.config["write_figures"],
        )

        routing_substeps: int = self.var.discharge_per_timestep[0].shape[0]
        if self.config["forcing_method"] == "headwater_points":
            forcing_grid = self.hydrology.grid.decompress(
                np.vstack(self.var.discharge_per_timestep)
            )
        elif self.config["forcing_method"] in ("runoff", "accumulated_runoff"):
            forcing_grid = self.hydrology.grid.decompress(
                np.vstack(self.var.runoff_m_per_timestep)
            )
        else:
            raise ValueError(
                f"Unknown forcing method {self.config['forcing_method']}. Supported are 'headwater_points' and 'runoff'."
            )

        substep_size: timedelta = self.model.timestep_length / routing_substeps

        # convert the forcing grid to an xarray DataArray
        forcing_grid: xr.DataArray = xr.DataArray(
            data=forcing_grid,
            coords={
                "time": pd.date_range(
                    end=self.model.current_time + (routing_substeps - 1) * substep_size,
                    periods=len(self.var.discharge_per_timestep) * routing_substeps,
                    freq=substep_size,
                ),
                "y": self.hydrology.grid.lat,
                "x": self.hydrology.grid.lon,
            },
            dims=["time", "y", "x"],
            name="forcing",
        )

        # ensure that we have forcing data for the entire event period
        assert (
            pd.to_datetime(forcing_grid.time.values[-1]).to_pydatetime() + substep_size
            >= end_time
        )
        assert pd.to_datetime(forcing_grid.time.values[0]).to_pydatetime() <= start_time

        forcing_grid: xr.DataArray = forcing_grid.rio.write_crs(self.model.crs)

        if self.config["forcing_method"] == "headwater_points":
            simulation.set_headwater_forcing_from_grid(
                discharge_grid=forcing_grid,
            )
        elif self.config["forcing_method"] == "runoff":
            simulation.set_runoff_forcing(
                runoff_m=forcing_grid,
            )

        elif self.config["forcing_method"] == "accumulated_runoff":
            river_ids: TwoDArrayInt32 = self.hydrology.grid.load(
                self.model.files["grid"]["routing/river_ids"], compress=False
            )
            simulation.set_accumulated_runoff_forcing(
                runoff_m=forcing_grid,
                river_network=self.model.hydrology.routing.river_network,
                mask=~self.model.hydrology.grid.mask,
                river_ids=river_ids,
                upstream_area=self.model.hydrology.grid.decompress(
                    self.model.hydrology.grid.var.upstream_area
                ),
                cell_area=self.model.hydrology.grid.decompress(
                    self.model.hydrology.grid.var.cell_area
                ),
                river_geometry=self.model.hydrology.routing.rivers,
            )
        else:
            raise ValueError(
                f"Unknown forcing method {self.config['forcing_method']}. Supported are 'headwater_points', 'runoff' and 'accumulated_runoff'."
            )

        # Set up river outflow boundary condition for all simulations
        set_river_outflow_boundary_condition(
            sf=simulation.sfincs_model,
            model_root=sfincs_model.path,
            simulation_root=simulation.path,
            write_figures=simulation.write_figures,
        )

        return simulation

    def run_single_event(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Runs a single flood event using the SFINCS model.

        Also updates the flood status of households in the model based on the flood depth results.

        Args:
            start_time: The start time of the flood event.
            end_time: The end time of the flood event.
        """
        sfincs_root_model = self.build("entire_region")  # build or read the model
        sfincs_simulation = self.set_forcing(  # set the forcing
            sfincs_root_model, start_time, end_time
        )
        self.model.logger.info(
            f"Running SFINCS for {self.model.current_time}..."
        )  # log the start of the simulation

        sfincs_simulation.run(
            gpu=self.config["SFINCS"]["gpu"],
        )  # run the simulation
        flood_depth: xr.DataArray = sfincs_simulation.read_max_flood_depth(
            self.config["minimum_flood_depth"]
        )  # read the flood depth results

        filename = (
            self.model.output_folder / "flood_maps" / (sfincs_simulation.name + ".zarr")
        )

        flood_depth: xr.DataArray = to_zarr(
            da=flood_depth,
            path=filename,
            crs=flood_depth.rio.crs,
        )  # save the flood depth to a zarr file

        self.model.agents.households.flood(flood_depth=flood_depth)

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
        transform = mask_var.rio.transform(recalc=True)

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
        transform = lecz.rio.transform(recalc=True)

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

        sfincs_root_model: SFINCSRootModel = self.build("entire_region")

        sfincs_root_model.estimate_discharge_for_return_periods(
            discharge=self.discharge_spinup_ds,
            rivers=self.model.hydrology.routing.rivers,
            return_periods=self.config["return_periods"],
        )

        rp_maps_riverine = {}
        for return_period in self.config["return_periods"]:
            print(
                f"Estimated discharge for return period {return_period} years for all rivers."
            )

            simulation: MultipleSFINCSSimulations = (
                sfincs_root_model.create_simulation_for_return_period(
                    return_period,
                )
            )
            simulation.run(
                gpu=self.config["SFINCS"]["gpu"],
            )
            flood_depth_return_period: xr.DataArray = simulation.read_max_flood_depth(
                self.config["minimum_flood_depth"]
            )
            rp_maps_riverine[return_period] = flood_depth_return_period

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
                to_zarr(
                    da=rp_maps_riverine[return_period],
                    path=self.model.output_folder
                    / "flood_maps"
                    / f"{return_period}.zarr",
                    crs=rp_maps_riverine[return_period].rio.crs,
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
            to_zarr(
                da=rp_map,
                path=self.model.output_folder / "flood_maps" / f"{return_period}.zarr",
                crs=rp_map.rio.crs,
            )

    def run(self, event: dict[str, Any]) -> None:
        """Runs the SFINCS model for a given flood event.

        Args:
            event: A dictionary containing the flood event details, including 'start_time' and 'end_time'.
        """
        start_time = event["start_time"]
        end_time = event["end_time"]

        if self.model.config["hazards"]["floods"]["flood_risk"]:
            raise NotImplementedError(
                "Flood risk calculations are not yet implemented. Need to adapt old calculations to new flood model."
            )

        else:
            self.run_single_event(start_time, end_time)

    def save_discharge(self) -> None:
        """Saves the current discharge for the current timestep.

        SFINCS is run at the end of an event rather than at the beginning. Therefore,
        we need to trace back the conditions at the beginning of the event. This function
        saves the current discharge at the beginning of each timestep,
        so it can be used later when setting up the SFINCS model.
        """
        self.var.discharge_per_timestep.append(
            self.hydrology.grid.var.discharge_m3_s_per_substep
        )  # this is a deque, so it will automatically remove the oldest discharge

    def save_runoff_m(self) -> None:
        """Saves the current runoff for the current timestep."""
        self.var.runoff_m_per_timestep.append(
            self.model.hydrology.grid.var.total_runoff_m
        )  # this is a deque, so it will automatically remove the oldest runoff

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
