import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel
from hydromt_sfincs.workflows import burn_river_rect, river_source_points
from tqdm import tqdm

from geb.hydrology.routing import get_river_width
from geb.model import GEBModel

from .workflows import do_mask_flood_plains, get_river_depth, get_river_manning
from .workflows.return_periods import (
    assign_calculation_group,
    get_topological_stream_order,
)
from .workflows.utils import (
    assign_return_periods,
    create_hourly_hydrograph,
    export_rivers,
    get_discharge_and_river_parameters_by_river,
    get_representative_river_points,
    get_start_point,
    import_rivers,
    make_relative_paths,
    read_maximum_flood_depth,
    run_sfincs_simulation,
    to_sfincs_datetime,
)


class SFINCSRootModel:
    """Builds and updates SFINCS model files for flood hazard modeling."""

    def __init__(self, model: GEBModel, event_name: str) -> None:
        self.model = model
        self.event_name = event_name

    @property
    def path(self) -> Path:
        """Returns the root directory for the SFINCS model files."""

        folder: Path = self.model.simulation_root / "SFINCS" / self.event_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def exists(self) -> bool:
        """Checks if the SFINCS model already exists in the model root directory.

        Returns:
            True if the SFINCS model exists, False otherwise.
        """
        return Path(self.path / "sfincs.inp").is_file()

    def read(self) -> None:
        """Reads an existing SFINCS model from the model root directory."""
        assert os.path.isfile(os.path.join(self.path, "sfincs.inp")), (
            f"model root does not exist {self.path}"
        )
        self.sfincs_model = SfincsModel(root=str(self.path), mode="r")

    def build(
        self,
        DEMs: list[dict[str, str]],
        region: gpd.GeoDataFrame,
        rivers: gpd.GeoDataFrame,
        discharge: xr.DataArray,
        waterbody_ids: npt.NDArray[np.int32],
        river_width_alpha: npt.NDArray[np.float32],
        river_width_beta: npt.NDArray[np.float32],
        mannings: xr.DataArray,
        resolution: float | int,
        nr_subgrid_pixels: int,
        crs: str,
        depth_calculation_method: str,
        depth_calculation_parameters: dict[str, float | int] | None = None,
        mask_flood_plains: bool = False,
    ) -> None:
        """Build a SFINCS model.

        Notes:
            mask_flood_plains is currently quite unstable and should be used with caution. Sometimes it leads
            to wrong regions being masked, which can lead to errors in the model.

        Args:
            DEMs: List of DEM datasets to use for the model. Should be a list of dictionaries with 'path' and 'name' keys.
            region: A GeoDataFrame defining the region of interest.
            rivers: A GeoDataFrame containing river segments.
            discharge: An xarray DataArray containing discharge values for the rivers in m^3/s.
            waterbody_ids: An numpy array of waterbody IDs specifying lakes and reservoirs. Should have same x and y dimensions as the discharge.
            river_width_alpha: An numpy array of river width alpha parameters. Used for calculating river width.
            river_width_beta: An numpy array of river width beta parameters. Used for calculating river width
            mannings: A xarray DataArray of Manning's n values for the rivers.
            resolution: The resolution of the requested SFINCS model grid in meters.
            nr_subgrid_pixels: The number of subgrid pixels to use for the SFINCS model. Must be an even number.
            crs: The coordinate reference system to use for the model.
            depth_calculation_method: The method to use for calculating river depth. Can be 'manning' or 'power_law'.
            depth_calculation_parameters: A dictionary of parameters for the depth calculation method. Only used if
                depth_calculation_method is 'power_law', in which case it should contain 'c' and 'd' keys.
            mask_flood_plains: Whether to autodelineate flood plains and mask them. Defaults to False.

        Raises:
            ValueError: if depth_calculation_method is not 'manning' or 'power_law',
            ValueError: if nr_subgrid_pixels is not None and not positive even number.
            ValueError: if resolution is not positive.

        """
        if nr_subgrid_pixels is not None and nr_subgrid_pixels <= 0:
            raise ValueError("nr_subgrid_pixels must be a positive number")
        if nr_subgrid_pixels is not None and nr_subgrid_pixels % 2 != 0:
            raise ValueError("nr_subgrid_pixels must be an even number")
        if resolution <= 0:
            raise ValueError("Resolution must be a positive number")

        assert depth_calculation_method in [
            "manning",
            "power_law",
        ], "Method should be 'manning' or 'power_law'"

        # build base model
        sf: SfincsModel = SfincsModel(root=str(self.path), mode="w+")

        sf.setup_grid_from_region(
            {"geom": region}, res=resolution, crs=crs, rotated=False
        )

        DEMs = [{**DEM, **{"reproj_method": "bilinear"}} for DEM in DEMs]

        # HydroMT-SFINCS only accepts datasets with an 'elevtn' variable. Therefore, the following
        # is a bit convoluted. We first open the dataarray, then convert it to a dataset,
        # and set the name as elevtn.
        sf.setup_dep(datasets_dep=DEMs)

        if mask_flood_plains:
            do_mask_flood_plains(sf)
        else:
            sf.setup_mask_active(
                region, zmin=-21, reset_mask=True
            )  # TODO: Improve mask setup

        # Setup river inflow points
        sf.setup_river_inflow(
            rivers=rivers,
            keep_rivers_geom=True,
            river_upa=0,
            river_len=0,
        )

        # find outflow points and save for later use
        outflow_points = river_source_points(
            gdf_riv=rivers.to_crs(sf.crs),
            gdf_mask=sf.region,
            src_type="outflow",
            buffer=sf.reggrid.dx,  # type: ignore
            river_upa=0,
            river_len=0,
        )
        # give error if outflow greater than 1
        if len(outflow_points) > 1:
            raise ValueError(
                "More than one outflow point found, outflow boundary condition will fail to setup"
            )
        elif len(outflow_points) == 0:
            raise ValueError(
                "No outflow point found, outflow boundary condition will fail to setup"
            )
        # print crs of outflow_points
        assert outflow_points.crs == sf.crs, (
            "CRS of outflow_points is not the same as the model crs"
        )
        # set crs before saving
        outflow_points = outflow_points.set_crs(sf.crs)
        # save to model root as a gpkg file
        outflow_points.to_file(self.path / "gis/outflow_points.gpkg", driver="GPKG")
        # Get the single outflow point coordinates
        x_coord = outflow_points.geometry.x.iloc[0]
        y_coord = outflow_points.geometry.y.iloc[0]
        assert sf.grid.dep.rio.crs == outflow_points.crs, (
            "CRS of sf.grid.dep is not the same as the outflow_points crs"
        )
        # Sample from sf.grid.dep (which is the DEM DataArray)
        elevation_value = sf.grid.dep.sel(
            x=x_coord, y=y_coord, method="nearest"
        ).values.item()

        # Optional: sanity check
        if elevation_value is None or elevation_value <= 0:
            raise ValueError(
                f"Invalid outflow elevation ({elevation_value}), must be > 0"
            )

        # Save elevation value to a file in model_root/gis
        outflow_elev_path = self.path / "gis" / "outflow_elevation.json"
        with open(outflow_elev_path, "w") as f:
            json.dump({"outflow_elevation": elevation_value}, f)

        river_representative_points = []
        for ID in rivers.index:
            river_representative_points.append(
                get_representative_river_points(ID, rivers, waterbody_ids)
            )

        discharge_by_river, river_parameters = (
            get_discharge_and_river_parameters_by_river(
                rivers.index.tolist(),
                river_representative_points,
                discharge=discharge,
                river_width_alpha=river_width_alpha,
                river_width_beta=river_width_beta,
            )
        )

        rivers = assign_return_periods(rivers, discharge_by_river, return_periods=[2])

        river_width_unknown_mask = rivers["width"].isnull()

        rivers.loc[river_width_unknown_mask, "width"] = get_river_width(
            river_parameters["river_width_alpha"][river_width_unknown_mask],
            river_parameters["river_width_beta"][river_width_unknown_mask],
            rivers.loc[river_width_unknown_mask, "Q_2"],
        )

        rivers["depth"] = get_river_depth(
            rivers,
            method=depth_calculation_method,
            parameters=depth_calculation_parameters,
            bankfull_column="Q_2",
        )

        rivers["manning"] = get_river_manning(rivers)

        export_rivers(self.path, rivers)

        # Because hydromt-sfincs does a lot of filling default values when data
        # is missing, we need to be extra sure that the required columns are
        # present and contain valid data.
        assert rivers["width"].notnull().all(), "River width cannot be null"
        assert rivers["depth"].notnull().all(), "River depth cannot be null"
        assert rivers["manning"].notnull().all(), "River Manning's n cannot be null"

        # if sfincs is run with subgrid, we set up the subgrid, with burned in rivers and mannings
        # roughness within the subgrid. If not, we burn the rivers directly into the main grid,
        # including mannings roughness.
        if nr_subgrid_pixels is not None:
            print("Setting up SFINCS subgrid...")
            sf.setup_subgrid(
                datasets_dep=DEMs,
                datasets_rgh=[
                    {
                        "manning": mannings.to_dataset(name="manning"),
                    }
                ],
                datasets_riv=[
                    {
                        "centerlines": rivers.rename(
                            columns={"width": "rivwth", "depth": "rivdph"}
                        )
                    }
                ],
                write_dep_tif=True,
                write_man_tif=True,
                nr_subgrid_pixels=nr_subgrid_pixels,
                nlevels=20,
                nrmax=500,
            )

            sf.write_subgrid()
        else:
            print("Skipping SFINCS subgrid...")
            # first set up the mannings roughness with the default method
            # (we already have the DEM set up)
            sf.setup_manning_roughness(
                datasets_rgh=[
                    {
                        "manning": mannings.to_dataset(name="manning"),
                    }
                ]
            )
            # retrieve the elevation and mannings grids
            # burn the rivers into these grids
            elevation, mannings = burn_river_rect(
                da_elv=sf.grid.dep,
                gdf_riv=rivers,
                da_man=sf.grid.manning,
                rivwth_name="width",
                rivdph_name="depth",
                manning_name="manning",
                segment_length=sf.reggrid.dx,
            )
            # set the modified grids back to the model
            sf.set_grid(elevation, name="dep")
            sf.set_grid(mannings, name="manning")

        # write all components, except forcing which must be done after the model building
        sf.write_grid()
        sf.write_geoms()
        sf.write_config()

        sf.plot_basemap(fn_out="basemap.png")

        self.sfincs_model = sf

    def estimate_discharge_for_return_periods(
        self,
        discharge: xr.DataArray,
        waterbody_ids: npt.NDArray[np.int32],
        rivers: gpd.GeoDataFrame,
        rising_limb_hours: int | float = 72,
        return_periods: list[int | float] = [2, 5, 10, 20, 50, 100, 250, 500, 1000],
    ) -> None:
        """Estimate discharge for specified return periods and create hydrographs.

        Args:
            model_root: path to the SFINC model root directory
            discharge: xr.DataArray containing the discharge data
            waterbody_ids: array of waterbody IDs, of identical x and y dimensions as discharge
            rivers: GeoDataFrame containing river segments
            rising_limb_hours: number of hours for the rising limb of the hydrograph.
            return_periods: list of return periods for which to estimate discharge.
        """
        recession_limb_hours = rising_limb_hours

        # here we only select the rivers that have an upstream forcing point
        rivers_with_forcing_point = rivers[~rivers["is_downstream_outflow_subbasin"]]

        river_representative_points = []
        for ID in rivers_with_forcing_point.index:
            river_representative_points.append(
                get_representative_river_points(
                    ID, rivers_with_forcing_point, waterbody_ids
                )
            )

        discharge_by_river, _ = get_discharge_and_river_parameters_by_river(
            rivers_with_forcing_point.index,
            river_representative_points,
            discharge=discharge,
        )
        rivers_with_forcing_point = assign_return_periods(
            rivers_with_forcing_point, discharge_by_river, return_periods=return_periods
        )

        for return_period in return_periods:
            rivers_with_forcing_point[f"hydrograph_{return_period}"] = None

        for river_idx in rivers_with_forcing_point.index:
            for return_period in return_periods:
                discharge_for_return_period = rivers_with_forcing_point.at[
                    river_idx, f"Q_{return_period}"
                ]
                hydrograph = create_hourly_hydrograph(
                    discharge_for_return_period,
                    rising_limb_hours,
                    recession_limb_hours,
                )
                hydrograph = {
                    time.isoformat(): Q.item() for time, Q in hydrograph.iterrows()
                }
                rivers_with_forcing_point.at[
                    river_idx, f"hydrograph_{return_period}"
                ] = hydrograph

        export_rivers(self.path, rivers_with_forcing_point, postfix="_return_periods")

    def create_simulation(
        self,
        simulation_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> "SFINCSSimulation":
        """Sets forcing for a SFINCS model based on the provided parameters.

        Creates a new simulation directory and creteas a new sfincs model
        in that folder. Variables that do not change between simulations
        (e.g., grid, mask, etc.) are retained in the original model folder
        and inherited by the new simulation using relative paths.

        Args:
            simulation_root: Path to the simulation root directory.
            event: Dictionary containing event details such as start and end times.
            discharge_grid: Discharge grid as xarray Dataset or path to netcdf file.
            waterbody_ids: Array of waterbody IDs, of identical x and y dimensions as discharge_grid.
            soil_water_capacity_grid: Dataset containing soil water capacity (seff).
            max_water_storage_grid: Dataset containing maximum water storage (smax).
            saturated_hydraulic_conductivity_grid: Dataset containing saturated hydraulic conductivity (ks).
            forcing_method: Method to set forcing, either "headwater_points" or "precipitation".
            precipitation_grid: Precipitation grid as xarray DataArray or list of DataArrays. Can also be None when
                forcing method is headwater_points. Defaults to None.

        Returns:
            An instance of SFINCSSimulation with the configured forcing.

        """

        return SFINCSSimulation(
            self,
            simulation_name,
            start_time=start_time,
            end_time=end_time,
        )

    def create_simulation_for_return_period(self, return_period: int | float):
        rivers: gpd.GeoDataFrame = import_rivers(self.path, postfix="_return_periods")
        assert (~rivers["is_downstream_outflow_subbasin"]).all()

        rivers["topological_stream_order"] = get_topological_stream_order(rivers)
        rivers: gpd.GeoDataFrame = assign_calculation_group(rivers)

        working_dir: Path = self.path / "working_dir"

        print(f"Running SFINCS for return period {return_period} years")
        simulations = []

        working_dir_return_period: Path = working_dir / f"rp_{return_period}"
        for group, group_rivers in tqdm(rivers.groupby("calculation_group")):
            simulation_root = working_dir_return_period / str(group)

            shutil.rmtree(simulation_root, ignore_errors=True)
            simulation_root.mkdir(parents=True, exist_ok=True)

            inflow_nodes = group_rivers.copy()
            inflow_nodes = inflow_nodes.reset_index(drop=True)
            inflow_nodes["geometry"] = inflow_nodes["geometry"].apply(get_start_point)

            Q: list[pd.DataFrame] = [
                pd.DataFrame.from_dict(
                    inflow_nodes[f"hydrograph_{return_period}"].iloc[idx],
                    orient="index",
                    columns=[idx],
                )
                for idx in inflow_nodes.index
            ]
            Q: pd.DataFrame = pd.concat(Q, axis=1)
            Q.index = pd.to_datetime(Q.index)

            Q: pd.DataFrame = (
                Q.fillna(method="ffill").fillna(  # pad with 0's before
                    method="bfill"
                )  # and after
            )

            simulation = self.create_simulation(
                simulation_name=f"rp_{return_period}_group_{group}",
                start_time=Q.index[0],
                end_time=Q.index[-1],
            )

            simulation.set_discharge_forcing_from_nodes(
                nodes=inflow_nodes.to_crs(self.sfincs_model.crs),
                timeseries=Q,
            )
            simulations.append(simulation)

        return MultipleSFINCSSimulations(simulations)


class MultipleSFINCSSimulations:
    def __init__(self, simulations) -> None:
        self.simulations = simulations

    def run(self, gpu: bool) -> None:
        for simulation in self.simulations:
            simulation.run(gpu=gpu)

    def read_flood_depth(self) -> xr.DataArray:
        """Reads the maximum flood depth map from the simulation output.

        Returns:
            An xarray DataArray containing the maximum flood depth.
        """
        flood_depths: list[xr.DataArray] = []
        for simulation in self.simulations:
            flood_depths.append(simulation.read_flood_depth())

        rp_map: xr.DataArray = xr.concat(flood_depths, dim="node")
        rp_map: xr.DataArray = rp_map.max(dim="node")
        rp_map.attrs["_FillValue"] = flood_depths[0].attrs["_FillValue"]
        assert rp_map.rio.crs is not None

        return rp_map


class SFINCSSimulation:
    def __init__(
        self,
        sfincs_root_model: SFINCSRootModel,
        simulation_name,
        start_time: datetime,
        end_time: datetime,
    ):
        self._name = simulation_name
        self.start_time = start_time
        self.end_time = end_time
        self.sfincs_root_model = sfincs_root_model

        sfincs_model = sfincs_root_model.sfincs_model
        sfincs_model.set_root(self.path, mode="w+")

        # update mode time based on event tstart and tend from event dict
        sfincs_model.setup_config(
            alpha=0.5
        )  # alpha is the parameter for the CFL-condition reduction. Decrease for additional numerical stability, minimum value is 0.1 and maximum is 0.75 (0.5 default value)
        sfincs_model.setup_config(tspinup=86400)  # spinup time in seconds
        sfincs_model.setup_config(dtout=900)  # output time step in seconds
        sfincs_model._write_gis = True
        sfincs_model.setup_config(
            tref=to_sfincs_datetime(start_time),
            tstart=to_sfincs_datetime(start_time),
            tstop=to_sfincs_datetime(end_time),
        )
        sfincs_model.setup_config(
            **make_relative_paths(sfincs_model.config, self.root_path, self.path)
        )
        sfincs_model.write_config()
        self.sfincs_model = sfincs_model

        # self.setup_outflow_boundary()

    def set_headwater_forcing_from_grid(
        self,
        discharge_grid: str | xr.DataArray,
        waterbody_ids: npt.NDArray[np.int32],
    ) -> None:
        rivers = import_rivers(self.root_path)
        rivers_with_forcing_point = rivers[~rivers["is_downstream_outflow_subbasin"]]
        headwater_rivers = rivers_with_forcing_point[
            rivers_with_forcing_point["maxup"] == 0
        ]

        inflow_nodes = headwater_rivers.copy()

        # Only select headwater points. Maxup is the number of upstream river segments.
        inflow_nodes["geometry"] = inflow_nodes["geometry"].apply(get_start_point)

        river_representative_points = []
        for ID in headwater_rivers.index:
            river_representative_points.append(
                get_representative_river_points(ID, headwater_rivers, waterbody_ids)
            )

        discharge_by_river, _ = get_discharge_and_river_parameters_by_river(
            headwater_rivers.index,
            river_representative_points,
            discharge=discharge_grid,
        )

        locations = inflow_nodes.to_crs(self.sfincs_model.crs)
        index_mapping = {
            idx: i + 1
            for i, idx in enumerate(locations.index)  # SFINCS index starts at 1
        }
        locations.index = locations.index.map(index_mapping)
        locations.index.name = "sfincs_idx"
        discharge_by_river.columns = discharge_by_river.columns.map(index_mapping)

        # Give discharge_forcing_points as forcing points
        self.sfincs_model.setup_discharge_forcing(
            locations=inflow_nodes.to_crs(self.sfincs_model.crs),
            timeseries=discharge_by_river,
        )
        self.sfincs_model.plot_basemap(fn_out="src_points_check.png")

        self.sfincs_model.write_forcing()
        self.sfincs_model.plot_forcing(fn_out="forcing.png")

    def set_discharge_forcing_from_nodes(self, nodes, timeseries) -> None:
        self.sfincs_model.setup_discharge_forcing(
            locations=nodes,
            timeseries=timeseries,
        )
        self.sfincs_model.write_forcing()

    def set_precipitation_forcing(
        self,
        soil_water_capacity_grid: xr.Dataset,
        max_water_storage_grid: xr.Dataset,
        saturated_hydraulic_conductivity_grid: xr.Dataset,
        precipitation_grid: None | xr.DataArray = None,
    ):
        assert precipitation_grid.raster.crs is not None, (
            "precipitation_grid should have a crs"
        )
        assert (
            pd.to_datetime(precipitation_grid.time[0].item()).to_pydatetime()
            <= self.start_time
        )
        assert (
            pd.to_datetime(precipitation_grid.time[-1].item()).to_pydatetime()
            >= self.end_time
        )

        precipitation_grid: xr.DataArray = precipitation_grid.sel(
            time=slice(self.start_time, self.end_time)
        )

        self.sfincs_model.set_forcing(
            (precipitation_grid * 3600).to_dataset(name="precip_2d"),
            name="precip_2d",
        )  # convert from kg/m2/s to mm/h

        self._setup_infiltration_capacity(
            max_water_storage=max_water_storage_grid,
            soil_water_capacity=soil_water_capacity_grid,
            saturated_hydraulic_conductivity=saturated_hydraulic_conductivity_grid,
        )
        self.sfincs_model.write_grid()
        self.sfincs_model.write_forcing()
        self.sfincs_model.plot_forcing(fn_out="forcing.png")

    def setup_outflow_boundary(self) -> None:
        # detect whether water level forcing should be set (use this under forcing == coastal) PLot basemap and forcing to check
        if (
            self.sfincs_model.grid["msk"] == 2
        ).any():  # if mask is 2, the model requires water level forcing
            waterlevel = self.sfincs_model.data_catalog.get_dataset(
                "waterlevel"
            ).compute()  # define water levels and stations in data_catalog.yml

            locations = gpd.GeoDataFrame(
                index=waterlevel.stations,
                geometry=gpd.points_from_xy(
                    waterlevel.station_x_coordinate, waterlevel.station_y_coordinate
                ),
                crs=4326,
            )

            timeseries = pd.DataFrame(
                index=waterlevel.time, columns=waterlevel.stations, data=waterlevel.data
            )
            assert timeseries.columns.equals(locations.index)

            locations = locations.reset_index(names="stations")
            locations.index = (
                locations.index + 1
            )  # for hydromt/SFINCS index should start at 1
            timeseries.columns = locations.index

            self.sfincs_model.setup_waterlevel_forcing(
                timeseries=timeseries, locations=locations
            )

        self.sfincs_model.write_forcing()

    def _setup_infiltration_capacity(
        self,
        max_water_storage: xr.Dataset,
        soil_water_capacity: xr.Dataset,
        saturated_hydraulic_conductivity: xr.Dataset,
    ) -> None:
        """Set up infiltration parameters in the SFINCS model.

        Uses the curve number method with recovery.

        Args:
            sfincs_model: SfincsModel object to update.
            max_water_storage: xarray Dataset containing maximum water storage (smax).
            soil_water_capacity: xarray Dataset containing soil water capacity (seff).
            saturated_hydraulic_conductivity: xarray Dataset containing saturated hydraulic conductivity (ks).
        """
        max_water_storage = max_water_storage.raster.reproject_like(
            self.sfincs_model.grid, method="average"
        )
        max_water_storage = max_water_storage.rename_vars({"max_water_storage": "smax"})
        max_water_storage.attrs.update(**self.sfincs_model._ATTRS.get("smax", {}))
        self.sfincs_model.set_grid(max_water_storage, name="smax")
        self.sfincs_model.set_config("smaxfile", "sfincs.smax")

        soil_water_capacity = soil_water_capacity.raster.reproject_like(
            self.sfincs_model.grid, method="average"
        )
        soil_water_capacity = soil_water_capacity.rename_vars(
            {"soil_storage_capacity": "seff"}
        )
        soil_water_capacity.attrs.update(**self.sfincs_model._ATTRS.get("seff", {}))
        self.sfincs_model.set_grid(soil_water_capacity, name="seff")
        self.sfincs_model.set_config("sefffile", "sfincs.seff")

        saturated_hydraulic_conductivity = (
            saturated_hydraulic_conductivity.raster.reproject_like(
                self.sfincs_model.grid, method="average"
            )
        )
        saturated_hydraulic_conductivity = saturated_hydraulic_conductivity.rename_vars(
            {"saturated_hydraulic_conductivity": "ks"}
        )
        saturated_hydraulic_conductivity.attrs.update(
            **self.sfincs_model._ATTRS.get("ks", {})
        )
        self.sfincs_model.set_grid(saturated_hydraulic_conductivity, name="ks")
        self.sfincs_model.set_config("ksfile", "sfincs.ks")

        self.sfincs_model.write_grid()
        self.sfincs_model.write_config()

    def run(self, gpu: bool) -> None:
        run_sfincs_simulation(
            simulation_root=self.path,
            model_root=self.root_path,
            gpu=gpu,
        )

    def read_flood_depth(self) -> xr.DataArray:
        """Reads the maximum flood depth map from the simulation output.

        Returns:
            An xarray DataArray containing the maximum flood depth.
        """
        flood_map: xr.DataArray = read_maximum_flood_depth(
            model_root=self.root_path,
            simulation_root=self.path,
        )
        return flood_map

    @property
    def root_path(self) -> Path:
        """Returns the root directory for the SFINCS model files."""
        return self.sfincs_root_model.path

    @property
    def path(self) -> Path:
        """Returns the root directory for the SFINCS simulation files."""
        folder: Path = self.root_path / "simulations" / self.name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def name(self) -> str:
        """Returns the name of the simulation."""
        return self._name
