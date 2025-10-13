"""Class to setup, run, and post-process the SFINCS hydrodynamic model."""

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from shapely.geometry.point import Point

from ...hydrology.HRUs import load_geom
from ...hydrology.landcover import OPEN_WATER, SEALED
from ...workflows.io import open_zarr, to_zarr
from ...workflows.raster import reclassify
from .sfincs import MultipleSFINCSSimulations, SFINCSRootModel, SFINCSSimulation


class Floods:
    """The class that implements all methods to setup, run, and post-process hydrodynamic flood models.

    Args:
        model: The GEB model instance.
        n_timesteps: The number of timesteps to keep in memory for discharge calculations (default is 10).
    """

    def __init__(self, model: "GEBModel", n_timesteps: int = 10) -> None:
        """Initializes the Floods class.

        Args:
            model: The GEB model instance.
            n_timesteps: The number of timesteps to keep in memory for discharge calculations (default is 10).
        """
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

    def build(
        self,
        name: str,
        region: gpd.GeoDataFrame = None,
        coastal: bool = False,
        include_mask: gpd.GeoDataFrame = None,
        bnd_exclude_mask: gpd.GeoDataFrame = None,
    ) -> SFINCSRootModel:
        """Builds or reads a SFINCS model without any forcing.

        Before using this model, forcing must be set.

        When the model already exists and force_overwrite is False, the existing model is read.

        Args:
            name: Name of the SFINCS model (used for the model root directory).
            region: The region geometry for the model (optional).
            coastal: Whether to setup the model for coastal flooding (default is False).
            bnd_exclude_mask: A geometry to exclude from the boundary cells (optional).
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

            if region is None:
                region = load_geom(self.model.files["geom"]["routing/subbasins"])
            sfincs_model.build(
                region=region,
                DEMs=DEM_config,
                rivers=self.rivers,
                discharge=self.discharge_spinup_ds,
                waterbody_ids=self.model.hydrology.grid.decompress(
                    self.model.hydrology.grid.var.waterBodyID
                ),
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
                coastal=coastal,
                include_mask=include_mask,
                bnd_exclude_mask=bnd_exclude_mask,
            )
        else:
            sfincs_model.read()

        return sfincs_model

    def set_forcing(
        self,
        sfincs_model: SFINCSRootModel,
        start_time: datetime,
        end_time: datetime,
        precipitation_scale_factor: float = 1.0,
    ) -> SFINCSSimulation:
        """Sets the forcing for a SFINCS simulation.

        Depending on the forcing method in the config, either headwater point discharge
        or precipitation is used as forcing.

        Args:
            sfincs_model: The SFINCSRootModel instance to create the simulation from.
            start_time: The start time of the flood event.
            end_time: The end time of the flood event.
            precipitation_scale_factor: Scale factor for precipitation (default is 1.0).
                Only used if forcing_method is 'precipitation'.

        Returns:
            The created SFINCSSimulation instance with the forcing set.

        Raises:
            ValueError: If the forcing method is unknown.
        """
        simulation: SFINCSSimulation = sfincs_model.create_simulation(
            simulation_name=f"{start_time.strftime(format='%Y%m%dT%H%M%S')} - {end_time.strftime(format='%Y%m%dT%H%M%S')}",
            start_time=start_time,
            end_time=end_time,
        )
        if self.config["forcing_method"] == "headwater_points":
            n_timesteps: int = min(self.n_timesteps, len(self.discharge_per_timestep))
            routing_substeps: int = self.discharge_per_timestep[0].shape[0]
            discharge_grid = self.hydrology.grid.decompress(
                np.vstack(self.discharge_per_timestep)
            )
            first_timestep_of_event: int = (
                len(self.discharge_per_timestep) - n_timesteps
            )  # when SFINCS starts with
            # high values, this leads to numerical instabilities. Therefore, we first start with very
            # low discharge and then build up slowly to timestep 0
            # TODO: Check if this is a right approach
            discharge_grid = np.vstack(
                tup=[
                    np.full_like(
                        discharge_grid[:routing_substeps, :, :], fill_value=np.nan
                    ),
                    discharge_grid,
                ]
            )  # prepend zeros

            for i in range(routing_substeps - 1, -1, -1):
                discharge_grid[i] = discharge_grid[i + 1] * 0.3

            # convert the discharge grid to an xarray DataArray
            discharge_grid: xr.DataArray = xr.DataArray(
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
            discharge_grid.raster.set_crs(self.model.crs)
            end_time = discharge_grid.time[-1] + pd.Timedelta(
                self.model.timestep_length / routing_substeps
            )
            discharge_grid: xr.DataArray = discharge_grid.sel(
                time=slice(start_time, end_time)
            )
            simulation.set_headwater_forcing_from_grid(
                discharge_grid=discharge_grid,
                waterbody_ids=self.model.hydrology.grid.decompress(
                    self.model.hydrology.grid.var.waterBodyID
                ),
            )

        elif self.config["forcing_method"] == "precipitation":
            first_timestep_of_event: int = (
                len(self.discharge_per_timestep) - self.n_timesteps
            )

            precipitation_grid: xr.DataArray = self.model.forcing["pr_hourly"]
            assert isinstance(precipitation_grid, xr.DataArray)
            precipitation_grid: xr.DataArray = (
                precipitation_grid * precipitation_scale_factor
            )

            current_water_storage_grid: xr.DataArray = xr.DataArray(
                data=self.HRU.decompress(
                    self.soil_moisture_per_timestep[first_timestep_of_event]
                ),
                coords={
                    "y": self.HRU.lat,
                    "x": self.HRU.lon,
                },
                dims=["y", "x"],
                name="current_water_storage",
            )

            max_water_storage_grid: xr.DataArray = xr.DataArray(
                data=self.HRU.decompress(
                    self.max_water_storage_per_timestep[first_timestep_of_event],
                ),  # take first time step of soil moisture
                coords={
                    "y": self.HRU.lat,
                    "x": self.HRU.lon,
                },
                dims=["y", "x"],
                name="max_water_storage",
            )

            saturated_hydraulic_conductivity_grid: xr.DataArray = xr.DataArray(
                data=self.HRU.decompress(
                    self.saturated_hydraulic_conductivity_per_timestep[
                        first_timestep_of_event
                    ]
                ),  # take first time step of soil moisture
                coords={
                    "y": self.HRU.lat,
                    "x": self.HRU.lon,
                },
                dims=["y", "x"],
                name="saturated_hydraulic_conductivity",
            )

            current_water_storage_grid.raster.set_crs(self.model.crs)
            saturated_hydraulic_conductivity_grid.raster.set_crs(self.model.crs)
            max_water_storage_grid.raster.set_crs(self.model.crs)

            simulation.set_precipitation_forcing_grid(
                current_water_storage_grid=current_water_storage_grid,
                max_water_storage_grid=max_water_storage_grid,
                saturated_hydraulic_conductivity_grid=saturated_hydraulic_conductivity_grid,
                precipitation_grid=precipitation_grid,
            )
        else:
            raise ValueError(
                f"Unknown forcing method {self.config['forcing_method']}. Supported are 'headwater_points' and 'precipitation'."
            )

        return simulation

    def run_single_event(
        self,
        start_time: datetime,
        end_time: datetime,
        precipitation_scale_factor: float = 1.0,
    ) -> None:
        """Runs a single flood event using the SFINCS model.

        Also updates the flood status of households in the model based on the flood depth results.

        Args:
            start_time: The start time of the flood event.
            end_time: The end time of the flood event.
            precipitation_scale_factor: Scale factor for precipitation (default is 1.0).
        """
        assert precipitation_scale_factor >= 0, (
            "Precipitation scale factor must be non-negative."
        )

        sfincs_root_model = self.build("entire_region")
        sfincs_simulation = self.set_forcing(
            sfincs_root_model, start_time, end_time, precipitation_scale_factor
        )
        self.model.logger.info(f"Running SFINCS for {self.model.current_time}...")

        sfincs_simulation.run(
            gpu=self.config["SFINCS"]["gpu"],
        )
        flood_depth: xr.DataArray = sfincs_simulation.read_max_flood_depth(
            self.config["minimum_flood_depth"]
        )
        flood_depth: xr.DataArray = to_zarr(
            da=flood_depth,
            path=self.model.output_folder / "flood_maps" / sfincs_simulation.name,
            crs=flood_depth.rio.crs,
        )
        self.model.agents.households.flood(flood_depth=flood_depth)

    def get_coastal_return_period_maps(self) -> dict[int, xr.DataArray]:
        """This function models coastal flooding for the return periods specified in the model config.

        Returns:
            dict[int, xr.DataArray]: A dictionary mapping return periods to their respective flood maps.
        """
        # close the zarr store
        if hasattr(self.model, "reporter"):
            self.model.reporter.variables["discharge_daily"].close()
        # load osm land polygons to exclude from coastal boundary cells
        bnd_exclude_mask = load_geom(
            self.model.files["geom"]["coastal/land_polygons"],
        )
        # load gtsm tide gauge locations for water level boundary conditions
        locations = (
            gpd.GeoDataFrame(
                gpd.read_parquet(self.model.files["geom"]["gtsm/stations_coast_rp"])
            )
            .rename(columns={"station_id": "stations"})
            .set_index("stations")
        )
        # convert index to int
        locations.index = locations.index.astype(int)

        # build model for the different sfincs model regions
        model_regions = load_geom(self.model.files["geom"]["coastal/model_regions"])
        lecz_regions = load_geom(self.model.files["geom"]["coastal/lecz_regions"])

        # iterate over the different regions and run the coastal model for each region
        for idx, region in lecz_regions.iterrows():
            print(f"Run coastal model for region {idx}.")
            include_mask = lecz_regions[lecz_regions["idx"] == idx]
            # create geodataframe for the region
            region = gpd.GeoDataFrame(
                [region], geometry="geometry", crs=model_regions.crs
            )
            sfincs_root_model: SFINCSRootModel = self.build(
                f"coastal_region_{idx}",
                region=region,
                coastal=True,
                bnd_exclude_mask=bnd_exclude_mask,
                include_mask=include_mask,
            )

            sfincs_root_model.sfincs_model.setup_waterlevel_forcing(locations=locations)

            rp_maps_coastal = {}
            for return_period in self.config["return_periods"]:
                print(
                    f"Run coastal model for return period {return_period} years for all rivers."
                )

                simulation: SFINCSSimulation = (
                    sfincs_root_model.create_coastal_simulation_for_return_period(
                        return_period,
                    )
                )
                try:
                    simulation.run(
                        gpu=self.config["SFINCS"]["gpu"],
                    )
                    flood_depth_return_period: xr.DataArray = (
                        simulation.read_max_flood_depth(
                            self.config["minimum_flood_depth"]
                        )
                    )
                    rp_maps_coastal[return_period] = flood_depth_return_period
                    to_zarr(
                        flood_depth_return_period,
                        self.model.output_folder
                        / "flood_maps"
                        / f"{return_period}_coastal_reg_{idx}.zarr",
                        crs=flood_depth_return_period.rio.crs,
                    )
                except Exception as e:
                    print(
                        f"Error running coastal model for return period {return_period} years for region {idx}: {e}"
                    )
                    continue
            if hasattr(self.model, "reporter"):
                # and re-open afterwards
                self.model.reporter.variables["discharge_daily"] = zarr.ZipStore(
                    self.model.config["report_hydrology"]["discharge_daily"]["path"],
                    mode="a",
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
            waterbody_ids=self.model.hydrology.grid.decompress(
                self.model.hydrology.grid.var.waterBodyID
            ),
            rivers=self.rivers,
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

            to_zarr(
                flood_depth_return_period,
                self.model.output_folder
                / "flood_maps"
                / f"{return_period}_riverine.zarr",
                crs=flood_depth_return_period.rio.crs,
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
                    start_time,
                    end_time,
                    precipitation_scale_factor=row["scaling_factor"],
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
            self.run_single_event(start_time, end_time)

    def save_discharge(self) -> None:
        """Saves the current discharge for the current timestep.

        SFINCS is run at the end of an event rather than at the beginning. Therefore,
        we need to trace back the conditions at the beginning of the event. This function
        saves the current discharge at the beginning of each timestep,
        so it can be used later when setting up the SFINCS model.
        """
        self.discharge_per_timestep.append(
            self.hydrology.grid.var.discharge_m3_s_per_substep
        )  # this is a deque, so it will automatically remove the oldest discharge

    def save_current_soil_moisture(
        self,
    ) -> None:
        """Saves the current soil moisture for the current timestep.

        SFINCS is run at the end of an event rather than at the beginning. Therefore,
        we need to trace back the conditions at the beginning of the event. This function
        saves the current soil moisture at the beginning of each timestep,
        so it can be used later when setting up the SFINCS model.
        """
        w_copy = self.HRU.var.w.copy()
        w_copy[:, self.HRU.var.land_use_type == SEALED] = 0
        w_copy[:, self.HRU.var.land_use_type == OPEN_WATER] = 0
        self.initial_soil_moisture_grid = w_copy[:2].sum(axis=0)
        self.soil_moisture_per_timestep.append(self.initial_soil_moisture_grid)

    def save_max_soil_moisture(self) -> None:
        """Saves the maximum soil moisture for the current timestep.

        SFINCS is run at the end of an event rather than at the beginning. Therefore,
        we need to trace back the conditions at the beginning of the event. This function
        saves the maximum soil moisture at the beginning of each timestep,
        so it can be used later when setting up the SFINCS model.
        """
        ws_copy = self.HRU.var.ws.copy()
        ws_copy[:, self.HRU.var.land_use_type == SEALED] = 0
        ws_copy[:, self.HRU.var.land_use_type == OPEN_WATER] = 0
        self.max_water_storage_grid = ws_copy[:2].sum(axis=0)
        self.max_water_storage_per_timestep.append(self.max_water_storage_grid)

    def save_saturated_hydraulic_conductivity(self) -> None:
        """Saves the saturated hydraulic conductivity for the current timestep.

        SFINCS is run at the end of an event rather than at the beginning. Therefore,
        we need to trace back the conditions at the beginning of the event. This function
        saves the saturated hydraulic conductivity at the beginning of each timestep,
        so it can be used later when setting up the SFINCS model.
        """
        saturated_hydraulic_conductivity = (
            self.HRU.var.saturated_hydraulic_conductivity.copy()
        )
        saturated_hydraulic_conductivity[:, self.HRU.var.land_use_type == SEALED] = 0
        saturated_hydraulic_conductivity[
            :, self.HRU.var.land_use_type == OPEN_WATER
        ] = 0
        saturated_hydraulic_conductivity = saturated_hydraulic_conductivity[:2].sum(
            axis=0
        )

        self.saturated_hydraulic_conductivity_per_timestep.append(
            saturated_hydraulic_conductivity / 24 / 3600  # convert from m/day to m/s
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
    def rivers(self) -> gpd.GeoDataFrame:
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
