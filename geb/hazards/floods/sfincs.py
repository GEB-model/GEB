"""This module contains classes and functions to build and run SFINCS models for flood hazard assessment.

The main class is `SFINCSRootModel`, which is used to create and manage SFINCS models.
It provides methods to build the model, set up simulations with different forcing methods,
and read simulation results.

"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel
from hydromt_sfincs.workflows import burn_river_rect, river_source_points
from tqdm import tqdm

from geb.hydrology.routing import get_river_width

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
    read_flood_depth,
    run_sfincs_simulation,
    to_sfincs_datetime,
)


def set_river_outflow_boundary_condition(
    sf: "SfincsModel",
    model_root: Path,
    simulation_root: Path,
    write_figures: bool = True,
) -> None:
    """Set up river outflow boundary condition with constant elevation.

    This function reads the outflow point and elevation from the model setup,
    creates a constant water level time series, and applies it as a boundary condition.

    Args:
        sf: The SFINCS model instance.
        model_root: Path to the model root directory.
        simulation_root: Path to the simulation directory.
        write_figures: Whether to generate and save forcing plots. Defaults to True.
    """
    outflow = gpd.read_file(model_root / "gis/outflow_points.gpkg")
    # only one point location is expected
    assert len(outflow) == 1, "Only one outflow point is expected"

    # before changing root read dem value from gis folder from .json file
    dem_json_path = model_root / "gis" / "outflow_elevation.json"
    with open(dem_json_path, "r") as f:
        dem_values = json.load(f)
    elevation = dem_values.get("outflow_elevation", None)

    if elevation is None or elevation == 0:
        assert False, (
            "Elevation should have positive value to set up outflow waterlevel boundary"
        )

    # Get the model's start and stop time using the get_model_time function
    tstart, tstop = sf.get_model_time()

    # Define the time range (e.g., 1 month of hourly data)
    time_range = pd.date_range(start=tstart, end=tstop, freq="H")

    # Create DataFrame with constant elevation value
    elevation_time_series_constant = pd.DataFrame(
        data={"water_level": elevation},  # Use extracted elevation value
        index=time_range,
    )

    # Extract a unique index from the outflow point. Here, we use 1 as an example.
    outflow_index = 1  # This should be the index or a suitable ID of the outflow point
    elevation_time_series_constant.columns = [
        outflow_index
    ]  # Use an integer as column name

    # Ensure outflow has the correct index as well
    outflow["index"] = outflow_index  # Set the matching index to outflow location

    # Now set the water level forcing
    sf.setup_waterlevel_forcing(
        timeseries=elevation_time_series_constant,  # Constant time series
        locations=outflow,  # Outflow point
    )
    sf.set_root(simulation_root, mode="w+")

    sf.write_forcing()

    if write_figures:
        sf.plot_forcing(fn_out="waterlevel_forcing.png")
        sf.plot_basemap(fn_out="basemap.png")


class SFINCSRootModel:
    """Builds and updates SFINCS model files for flood hazard modeling."""

    def __init__(self, model: "GEBModel", name: str) -> None:
        """Initializes the SFINCSRootModel with a GEBModel and event name.

        Sets up the constant parts of the model (grid, mask, rivers, etc.),
        and has methods to create simulations with actual forcing.

        Args:
            model: An instance of GEBModel containing hydrological and geographical data.
            name: A string representing the name of the event (e.g., "flood_event_2023").
                Also used to create the path to write the file to disk.
        """
        self.model = model
        self._name = name

    @property
    def name(self) -> str:
        """Gets the name of the SFINCS model.

        Returns:
            The name of the SFINCS model.
        """
        return self._name

    @property
    def path(self) -> Path:
        """Gets the root directory for the SFINCS model files.

        Returns:
            The path to the SFINCS model root directory.
        """
        folder: Path = self.model.simulation_root / "SFINCS" / self.name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def exists(self) -> bool:
        """Checks if the SFINCS model already exists in the model root directory.

        Returns:
            True if the SFINCS model exists, False otherwise.
        """
        return Path(self.path / "sfincs.inp").is_file()

    def read(self) -> "SFINCSRootModel":
        """Reads an existing SFINCS model from the model root directory.

        Returns:
            The SFINCSRootModel instance with the read model.

        Raises:
            FileNotFoundError: if the SFINCS model does not exist in the specified path.
        """
        if not self.exists():
            raise FileNotFoundError(f"SFINCS model not found in {self.path}")
        self.sfincs_model = SfincsModel(root=str(self.path), mode="r")
        self.sfincs_model.read()
        return self

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
        nr_subgrid_pixels: int | None,
        crs: str,
        depth_calculation_method: str,
        depth_calculation_parameters: dict[str, float | int] | None = None,
        mask_flood_plains: bool = False,
        coastal: bool = False,
        include_mask: gpd.GeoDataFrame | None = None,
        bnd_exclude_mask: gpd.GeoDataFrame | None = None,
        gtsm_stations: gpd.GeoDataFrame | None = None,
    ) -> "SFINCSRootModel":
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
            coastal: Whether to set up the model for coastal flooding. Defaults to False.
            include_mask: A GeoDataFrame defining areas to include as active cells. Required if coastal is True.
            bnd_exclude_mask: A GeoDataFrame defining areas to exclude from coastal boundary conditions. Required if coastal is True.
            gtsm_stations: A GeoDataFrame containing locations of GTSM stations for coastal boundary conditions. Required if coastal is True.

        Returns:
            The SFINCSRootModel instance with the built model.

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

        logger = logging.getLogger(__name__)

        # Configure HydroMT logging to capture internal logs
        for logger_name in ["hydromt", "hydromt_sfincs", "hydromt_sfincs.workflows"]:
            hydromt_logger = logging.getLogger(logger_name)
            hydromt_logger.setLevel(logging.INFO)
            hydromt_logger.propagate = True

        # Get the main HydroMT-SFINCS logger for level adjustments
        hydromt_logger = logging.getLogger("hydromt_sfincs")

        logger.info("Starting SFINCS model build...")

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
        elif coastal:
            sf.setup_mask_active(
                mask=include_mask,
                zmin=-21,  # minimum elevation for valid cells
                zmax=25,  # Now set quite high to include dunes. Otherwise weird bounding mask shapes can occur.
                drop_area=1,  # drops areas that are smaller than 1km2,
                reset_mask=True,
            )

            # set zsini based on the minimum elevation within the include_mask
            sf.config["zsini"] = include_mask["zsini"].values[0]

            # set the spinup period to 24 hours (also set in class init, move this to better place)
            sf.config["tspinup"] = 24 * 3600

            # setup the coastal boundary conditions
            sf.setup_mask_bounds(
                btype="waterlevel",
                zmax=2,  # maximum elevation for valid boundary cells
                exclude_mask=bnd_exclude_mask,
                all_touched=True,
            )
            sf.setup_waterlevel_forcing(locations=gtsm_stations)

        else:
            sf.setup_mask_active(
                region, zmin=-21, reset_mask=True
            )  # TODO: Improve mask setup

        # Temporarily set HydroMT logging to DEBUG to capture detailed internal logs
        hydromt_logger.setLevel(logging.DEBUG)
        # in one plot plot the region boundary as well as the rivers and save to file
        fig, ax = plt.subplots(figsize=(10, 10))
        region.boundary.plot(ax=ax, color="black")
        rivers.plot(ax=ax, color="blue")
        plt.savefig(self.path / "gis" / "rivers.png")

        sf.setup_river_inflow(
            rivers=rivers.to_crs(sf.crs),
            keep_rivers_geom=True,
            river_upa=0,
            river_len=0,
        )

        sf.setup_river_outflow(
            rivers=rivers.to_crs(sf.crs),
            keep_rivers_geom=True,
            river_upa=0,
            river_len=0,
            btype="waterlevel",
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
            print(
                "More than one outflow point found, outflow boundary condition will fail to setup"
            )
        elif len(outflow_points) == 0:
            print(
                "No outflow point found, outflow boundary condition will fail to setup. Skipping outflow elevation calculation..."
            )
        elif len(outflow_points) == 1:
            print("One outflow point found, proceeding...")
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
        ).astype(np.float64)

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
            logger.info(
                f"Setting up SFINCS subgrid with {nr_subgrid_pixels} subgrid pixels..."
            )
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
            logger.info(
                "Setting up SFINCS without subgrid - burning rivers into main grid..."
            )
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
        sf.write()

        sf.plot_basemap(fn_out="basemap.png")

        self.sfincs_model = sf
        return self

    @property
    def area(self) -> float | int:
        """Returns the area of the SFINCS model region in square kilometers.

        Returns:
            The area of the SFINCS model region in m².
        """
        return self.sfincs_model.grid["msk"].sum().item() * self.cell_area

    @property
    def cell_area(self) -> float | int:
        """Returns the area of a single cell in the SFINCS model grid in square meters.

        Returns:
            The area of a single cell in the SFINCS model grid in m².
        """
        return (
            self.sfincs_model.grid["msk"].rio.resolution()[0]
            * self.sfincs_model.grid["msk"].rio.resolution()[1]
        )

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
        recession_limb_hours: int | float = rising_limb_hours

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
        *args: Any,
        **kwargs: Any,
    ) -> "SFINCSSimulation":
        """Sets forcing for a SFINCS model based on the provided parameters.

        Creates a new simulation directory and creteas a new sfincs model
        in that folder. Variables that do not change between simulations
        (e.g., grid, mask, etc.) are retained in the original model folder
        and inherited by the new simulation using relative paths.

        Args:
            *args: Positional arguments to pass to the SFINCSSimulation constructor.
            **kwargs: Keyword arguments to pass to the SFINCSSimulation constructor.

        Returns:
            An instance of SFINCSSimulation with the configured forcing.
        """
        return SFINCSSimulation(
            self,
            *args,
            **kwargs,
        )

    def create_coastal_simulation_for_return_period(
        self, return_period: int | float
    ) -> "SFINCSSimulation":
        """Creates a SFINCS simulation for coastal regions for a specified return period.

        This method sets up a simulation that includes coastal
        boundary conditions, such as storm surge or tidal influences, for the specified
        return period.

        Args:
            return_period: The return period for which to create the coastal simulation.
        Returns:
            An instance of SFINCSSimulation configured for coastal conditions.
        """
        timeseries = pd.read_csv(
            Path(
                f"output/hydrographs/gtsm_spring_tide_hydrograph_rp{return_period:04d}.csv"
            ),
            index_col=0,
        )
        timeseries.index = pd.to_datetime(timeseries.index, format="%Y-%m-%d %H:%M:%S")
        # convert columns to int
        timeseries.columns = timeseries.columns.astype(int)

        # Align timeseries columns with locations index
        timeseries = timeseries.iloc[250:-250]  # trim the first and last 250 rows

        simulation: SFINCSSimulation = self.create_simulation(
            simulation_name=f"rp_{return_period}_coastal",
            start_time=timeseries.index[0],
            end_time=timeseries.index[-1],
        )

        # set forcing and configure model
        simulation.set_coastal_waterlevel_forcing(timeseries=timeseries)
        return simulation

    def create_simulation_for_return_period(
        self, return_period: int | float
    ) -> "MultipleSFINCSSimulations":
        """Creates multiple SFINCS simulations for a specified return period.

        The method groups rivers by their calculation group and creates a separate
        simulation for each group. Each simulation is configured with discharge
        hydrographs corresponding to the specified return period.
        Args:
            return_period: The return period for which to create simulations.

        Returns:
            An instance of MultipleSFINCSSimulations containing the created simulations.
                This class aims to emulate a single SFINCSSimulation instance as if
                it was one.
        """
        rivers: gpd.GeoDataFrame = import_rivers(self.path, postfix="_return_periods")
        assert (~rivers["is_downstream_outflow_subbasin"]).all()

        rivers["topological_stream_order"] = get_topological_stream_order(rivers)
        rivers: gpd.GeoDataFrame = assign_calculation_group(rivers)

        working_dir: Path = self.path / "working_dir"

        print(f"Running SFINCS for return period {return_period} years")
        simulations: list[SFINCSSimulation] = []

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
                    data=inflow_nodes[f"hydrograph_{return_period}"].iloc[idx],
                    orient="index",
                    columns=[idx],
                )
                for idx in inflow_nodes.index
            ]
            Q: pd.DataFrame = pd.concat(Q, axis=1)
            Q.index = pd.to_datetime(Q.index)

            assert not np.isnan(Q.values).any(), (
                "NaN values found in discharge hydrographs"
            )

            simulation: SFINCSSimulation = self.create_simulation(
                simulation_name=f"rp_{return_period}_group_{group}",
                start_time=Q.index[0],
                end_time=Q.index[-1],
            )

            simulation.set_discharge_forcing_from_nodes(
                nodes=inflow_nodes.to_crs(self.sfincs_model.crs),
                timeseries=Q,
            )

            # Set up river outflow boundary condition for this simulation
            set_river_outflow_boundary_condition(
                sf=simulation.sfincs_model,
                model_root=self.path,
                simulation_root=simulation.path,
                write_figures=simulation.write_figures,
            )

            simulations.append(simulation)

        return MultipleSFINCSSimulations(simulations=simulations)


class MultipleSFINCSSimulations:
    """Manages multiple SFINCS simulations as a single entity."""

    def __init__(self, simulations: list["SFINCSSimulation"]) -> None:
        """Simulates running multiple SFINCS simulations as one.

        Args:
            simulations: A list of SFINCSSimulation instances to manage together.
        """
        self.simulations = simulations

    def run(self, gpu: bool) -> None:
        """Runs all contained SFINCS simulations.

        Args:
            gpu: Whether to use GPU acceleration for the simulations.
        """
        for simulation in self.simulations:
            simulation.run(gpu=gpu)

    def merge_flood_depths(self, flood_depths: list[xr.DataArray]) -> xr.DataArray:
        """Merges multiple flood depth outputs into a single DataArray.

        Args:
            flood_depths: A list of xarray DataArray objects containing flood depth data.

        Returns:
            An xarray DataArray that contains the merged flood depth results.
        """
        # --- 2. Get union bounds ---

        minx = min([flood_depth.rio.bounds()[0] for flood_depth in flood_depths])
        miny = min([flood_depth.rio.bounds()[1] for flood_depth in flood_depths])
        maxx = max([flood_depth.rio.bounds()[2] for flood_depth in flood_depths])
        maxy = max([flood_depth.rio.bounds()[3] for flood_depth in flood_depths])

        # --- 3. Pick resolution ---
        # Use resolution of the first map (all have the same; y is negative if north-up, so take abs)
        res_x, res_y = flood_depths[0].rio.resolution()
        res_x = abs(res_x)
        res_y = abs(res_y)

        # --- 4. Build template coords ---
        width = int(np.ceil((maxx - minx) / res_x))
        height = int(np.ceil((maxy - miny) / res_y))

        x_coords = minx + (np.arange(width) + 0.5) * res_x
        y_coords = maxy - (np.arange(height) + 0.5) * res_y  # top→bottom

        template = xr.DataArray(
            np.full((height, width), np.nan, dtype=flood_depths[0].dtype),
            coords={"y": y_coords, "x": x_coords},
            dims=("y", "x"),
        ).rio.write_crs(flood_depths[0].rio.crs)

        # --- 5. Reproject all floodmaps to the template ---
        flood_depths_reproj: list[xr.DataArray] = []
        for flood_depth in flood_depths:
            assert flood_depth.rio.crs is not None, "Flood depth data must have a CRS"
            flood_depth_reproj = flood_depth.rio.reproject_match(template)
            flood_depths_reproj.append(flood_depth_reproj)

        # --- 6. Merge via maximum ---
        rp_map = xr.concat(flood_depths_reproj, dim="stacked").max(
            dim="stacked", skipna=True
        )
        rp_map.rio.write_crs(template.rio.crs)
        return rp_map

    def read_max_flood_depth_coastal(
        self, minimum_flood_depth: float | int
    ) -> xr.DataArray:
        """Reads the maximum flood depth map from a coastal simulation output.

        Args:
            minimum_flood_depth: Minimum flood depth to consider in the output.
        Returns:
            An xarray DataArray containing the maximum flood depth.
        """
        flood_depths: list[xr.DataArray] = []
        for simulation in self.simulations:
            flood_depths.append(simulation.read_max_flood_depth(minimum_flood_depth))

        # Merge all flood maps in regions
        rp_map = self.merge_flood_depths(flood_depths)
        assert rp_map.rio.crs is not None

        return rp_map

    def read_max_flood_depth(self, minimum_flood_depth: float | int) -> xr.DataArray:
        """Reads the maximum flood depth map from the simulation output.

        Args:
            minimum_flood_depth: Minimum flood depth to consider in the output.

        Returns:
            An xarray DataArray containing the maximum flood depth.
        """
        flood_depths: list[xr.DataArray] = []
        for simulation in self.simulations:
            flood_depths.append(simulation.read_max_flood_depth(minimum_flood_depth))

        rp_map: xr.DataArray = xr.concat(flood_depths, dim="node")
        rp_map: xr.DataArray = rp_map.max(dim="node")
        rp_map.attrs["_FillValue"] = flood_depths[0].attrs["_FillValue"]
        assert rp_map.rio.crs is not None

        return rp_map

    def cleanup(self) -> None:
        """Cleans up all simulation directories."""
        for simulation in self.simulations:
            simulation.cleanup()


class SFINCSSimulation:
    """A SFINCS simulation with specific forcing and configuration.

    Created fro m a SFINCSRootModel instance which already contains the constant parts of the model.
    """

    def __init__(
        self,
        sfincs_root_model: SFINCSRootModel,
        simulation_name: str,
        start_time: datetime,
        end_time: datetime,
        spinup_seconds: int = 86400,
        write_gis_files: bool = True,
        write_figures: bool = True,
    ) -> None:
        """Initializes a SFINCSSimulation with specific forcing and configuration.

        Args:
            sfincs_root_model: An instance of SFINCSRootModel containing the base model.
            simulation_name: A string representing the name of the simulation.
                Also used to create the path to write the file to disk.
            start_time: The start time of the simulation as a datetime object.
            end_time: The end time of the simulation as a datetime object.
            spinup_seconds: The number of seconds to use for model spin-up. Defaults to 86400 (1 day).
            write_gis_files: Whether to write GIS files for the model. Defaults to True.
            write_figures: Whether to generate and save figures for the model. Defaults to False.
        """
        self._name = simulation_name
        self.write_figures = write_figures
        self.start_time = start_time
        self.end_time = end_time
        self.sfincs_root_model = sfincs_root_model

        sfincs_model = sfincs_root_model.sfincs_model
        sfincs_model.set_root(self.path, mode="w+")

        # update mode time based on event tstart and tend from event dict
        sfincs_model.setup_config(
            alpha=0.5
        )  # alpha is the parameter for the CFL-condition reduction. Decrease for additional numerical stability, minimum value is 0.1 and maximum is 0.75 (0.5 default value)
        sfincs_model.setup_config(tspinup=spinup_seconds)  # spinup time in seconds
        sfincs_model.setup_config(dtout=900)  # output time step in seconds
        sfincs_model._write_gis = write_gis_files
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

    def set_headwater_forcing_from_grid(
        self,
        discharge_grid: str | xr.DataArray,
        waterbody_ids: npt.NDArray[np.int32],
    ) -> None:
        """Sets up discharge forcing for the SFINCS model from a gridded dataset.

        Args:
            discharge_grid: Path to a raster file or an xarray DataArray containing discharge values in m^3/s.
                Usually this is from a hydrological model.
            waterbody_ids: An numpy array of waterbody IDs specifying lakes and reservoirs.
                Should have same x and y dimensions as the discharge.
        """
        rivers: gpd.GeoDataFrame = import_rivers(self.root_path)
        rivers_with_forcing_point: gpd.GeoDataFrame = rivers[
            ~rivers["is_downstream_outflow_subbasin"]
        ]
        headwater_rivers: gpd.GeoDataFrame = rivers_with_forcing_point[
            rivers_with_forcing_point["maxup"] == 0
        ]

        inflow_nodes: gpd.GeoDataFrame = headwater_rivers.copy()

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

        self.set_discharge_forcing_from_nodes(
            nodes=locations,
            timeseries=discharge_by_river,
        )

    def set_coastal_waterlevel_forcing(self, timeseries: pd.DataFrame) -> None:
        """Sets up coastal water level forcing for the SFINCS model from a timeseries.

        Args:
            timeseries: A DataFrame containing the water level timeseries for each node.
                The columns should match the index of the locations GeoDataFrame.
        """
        # assert np.array_equal(locations.index, np.arange(1, len(locations) + 1))

        # select only locations that are in the model
        station_ids_in_model = self.sfincs_model.forcing["bzs"].index.values
        timeseries = timeseries.loc[:, timeseries.columns.isin(station_ids_in_model)]

        self.sfincs_model.setup_waterlevel_forcing(timeseries=timeseries)

        self.sfincs_model.write_forcing()
        self.sfincs_model.write_config()

        if self.write_figures:
            self.sfincs_model.plot_forcing(fn_out="forcing.png")
            self.sfincs_model.plot_basemap(fn_out="basemap.png")

    def set_discharge_forcing_from_nodes(
        self, nodes: gpd.GeoDataFrame, timeseries: pd.DataFrame
    ) -> None:
        """Sets up discharge forcing for the SFINCS model from specified nodes (locations) and timeseries.

        Args:
            nodes: A GeoDataFrame containing the locations of the discharge forcing points.
                The index should start at 1 and be consecutive (1, 2, 3, ...).
            timeseries: A DataFrame containing the discharge timeseries for each node.
                The columns should match the index of the nodes GeoDataFrame.
        """
        # assert np.array_equal(nodes.index, np.arange(1, len(nodes) + 1))
        assert set(timeseries.columns) == set(nodes.index)

        self.sfincs_model.setup_discharge_forcing(
            locations=nodes,
            timeseries=timeseries,
        )

        self.sfincs_model.write_forcing()
        self.sfincs_model.write_config()

    def set_precipitation_forcing_grid(
        self,
        current_water_storage_grid: xr.DataArray,
        max_water_storage_grid: xr.DataArray,
        saturated_hydraulic_conductivity_grid: xr.DataArray,
        precipitation_grid: xr.DataArray,
    ) -> None:
        """Sets up precipitation forcing for the SFINCS model from a gridded dataset.

        Args:
            current_water_storage_grid: xarray DataArray containing current soil water capacity [m].
            max_water_storage_grid: xarray DataArray containing maximum water storage [m].
            saturated_hydraulic_conductivity_grid: xarray DataArray containing saturated hydraulic conductivity [m/s].
            precipitation_grid: xarray DataArray containing precipitation values in kg/m²/s (equivalent to mm/s).
        """
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

        self.sfincs_model.setup_precip_forcing_from_grid(
            precip=(precipitation_grid * 3600).to_dataset(name="precip")
        )  # convert from kg/m2/s to mm/h

        # self._setup_infiltration_capacity(
        #     max_water_storage=max_water_storage_grid,
        #     current_water_storage=current_water_storage_grid,
        #     saturated_hydraulic_conductivity=saturated_hydraulic_conductivity_grid,
        # )
        # warning
        logging.warning(
            "Infiltration capacity setup is currently disabled due to an issue."
        )

        self.sfincs_model.write_forcing()
        self.sfincs_model.write_config()

    # def setup_outflow_boundary(self) -> None:
    #     # detect whether water level forcing should be set (use this under forcing == coastal) PLot basemap and forcing to check
    #     if (
    #         self.sfincs_model.grid["msk"] == 2
    #     ).any():  # if mask is 2, the model requires water level forcing
    #         waterlevel = self.sfincs_model.data_catalog.get_dataset(
    #             "waterlevel"
    #         ).compute()  # define water levels and stations in data_catalog.yml

    #         locations = gpd.GeoDataFrame(
    #             index=waterlevel.stations,
    #             geometry=gpd.points_from_xy(
    #                 waterlevel.station_x_coordinate, waterlevel.station_y_coordinate
    #             ),
    #             crs=4326,
    #         )

    #         timeseries = pd.DataFrame(
    #             index=waterlevel.time, columns=waterlevel.stations, data=waterlevel.data
    #         )
    #         assert timeseries.columns.equals(locations.index)

    #         locations = locations.reset_index(names="stations")
    #         locations.index = (
    #             locations.index + 1
    #         )  # for hydromt/SFINCS index should start at 1
    #         timeseries.columns = locations.index

    #         self.sfincs_model.setup_waterlevel_forcing(
    #             timeseries=timeseries, locations=locations
    #         )

    #     self.sfincs_model.write_forcing()

    def _setup_infiltration_capacity(
        self,
        max_water_storage: xr.DataArray,
        current_water_storage: xr.DataArray,
        saturated_hydraulic_conductivity: xr.DataArray,
        initial_substractiion: float = 0.0,
    ) -> None:
        """Set up infiltration parameters in the SFINCS model.

        Uses the curve number method with recovery.

        Args:
            sfincs_model: SfincsModel object to update.
            max_water_storage: xarray DataArray containing maximum water storage [m].
            current_water_storage: xarray DataArray containing soil water capacity [m].
            saturated_hydraulic_conductivity: xarray DataArray containing saturated hydraulic conductivity [m/s].
            initial_substractiion: Initial abstraction ratio [-].
                Is removed from rainfall before infiltration. Defaults to 0.0.
        """
        remaining_water_storage = max_water_storage - current_water_storage
        remaining_water_storage = remaining_water_storage.compute()

        # maximum water storage (smax in SFINCS)
        max_water_storage = max_water_storage.raster.reproject_like(
            self.sfincs_model.grid, method="average"
        )
        assert not np.isnan(max_water_storage.values[self.active_cells]).any(), (
            "max_water_storage contains NaN values in active cells"
        )
        max_water_storage = max_water_storage.to_dataset(name="smax")
        self.sfincs_model.set_grid(max_water_storage, name="smax")
        self.sfincs_model.set_config("smaxfile", "sfincs.smax")

        # remaining water storage (seff in SFINCS)
        remaining_water_storage = remaining_water_storage.raster.reproject_like(
            self.sfincs_model.grid, method="nearest"
        )
        assert not np.isnan(remaining_water_storage.values[self.active_cells]).any(), (
            "current_water_storage contains NaN values in active cells"
        )
        remaining_water_storage = remaining_water_storage.to_dataset(name="seff")
        self.sfincs_model.set_grid(remaining_water_storage, name="seff")
        self.sfincs_model.set_config("sefffile", "sfincs.seff")

        # saturated hydraulic conductivity (ks in SFINCS)
        saturated_hydraulic_conductivity = (
            (
                saturated_hydraulic_conductivity.raster.reproject_like(
                    self.sfincs_model.grid, method="average"
                )
            )
            * 3600
            * 1000
        )  # convert from m/s to mm/h for SFINCS
        assert not np.isnan(
            saturated_hydraulic_conductivity.values[self.active_cells]
        ).any(), "saturated_hydraulic_conductivity contains NaN values in active cells"

        saturated_hydraulic_conductivity: xr.Dataset = (
            saturated_hydraulic_conductivity.to_dataset(name="ks")
        )
        self.sfincs_model.set_grid(
            saturated_hydraulic_conductivity, name="ks"
        )  # convert from m/s to mm/h
        self.sfincs_model.set_config("ksfile", "sfincs.ks")

        # initial abstraction ratio
        assert initial_substractiion >= 0.0, "initial_substraction must be non-negative"
        assert initial_substractiion < 1.0, "initial_substraction must be < 1.0"

        self.sfincs_model.set_config("sfacinf", initial_substractiion)

        # remove constant infiltration rate if present
        self.sfincs_model.config.pop("qinf", None)

        # write grids and config
        self.sfincs_model.write_grid(data_vars=["smax", "seff", "ks"])
        self.sfincs_model.write_config()

    def run(self, gpu: bool | str) -> None:
        """Runs the SFINCS simulation.

        Args:
            gpu: Whether to use GPU acceleration for the simulation. Can be
                True, False, or 'auto' to automatically detect GPU availability.
        """
        assert gpu in [True, False, "auto"], "gpu must be True, False, or 'auto'"
        run_sfincs_simulation(
            simulation_root=self.path,
            model_root=self.root_path,
            gpu=gpu,
        )

    def read_max_flood_depth(self, minimum_flood_depth: int | float) -> xr.DataArray:
        """Reads the maximum flood depth map from the simulation output.

        Args:
            minimum_flood_depth: Minimum flood depth to consider [m]. Values below this threshold are set to zero.

        Returns:
            An xarray DataArray containing the maximum flood depth.
        """
        flood_map: xr.DataArray = read_flood_depth(
            model_root=self.root_path,
            simulation_root=self.path,
            method="max",
            minimum_flood_depth=minimum_flood_depth,
        )
        return flood_map

    def read_final_flood_depth(self, minimum_flood_depth: float | int) -> xr.DataArray:
        """Reads the final flood depth map from the simulation output.

        Args:
            minimum_flood_depth: Minimum flood depth to consider [m]. Values below this threshold are set to zero.

        Returns:
            An xarray DataArray containing the final flood depth.
        """
        flood_map: xr.DataArray = read_flood_depth(
            model_root=self.root_path,
            simulation_root=self.path,
            method="final",
            minimum_flood_depth=minimum_flood_depth,
        )
        return flood_map

    def get_flood_volume(self, flood_depth: xr.DataArray) -> float:
        """Compute the total flood volume from the flood depth map.

        Args:
            flood_depth: An xarray DataArray containing the flood depth.

        Returns:
            The total flood volume in cubic meters.
        """
        pixel_area = abs(
            flood_depth.rio.resolution()[0] * flood_depth.rio.resolution()[1]
        )
        flooded_pixels = flood_depth.where(flood_depth > 0).sum().item()
        if hasattr(flooded_pixels, "compute"):
            flooded_pixels = flooded_pixels.compute()
        return flooded_pixels * pixel_area

    def cleanup(self) -> None:
        """Cleans up the simulation directory by removing temporary files."""
        shutil.rmtree(path=self.path, ignore_errors=True)

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

    def has_outflow_boundary(self) -> bool:
        """Checks if the SFINCS model has an outflow boundary condition.

        Returns:
            True if the SFINCS model has an outflow boundary condition, False otherwise.
        """
        return (self.sfincs_model.grid["msk"] == 2).any()

    @property
    def active_cells(self) -> xr.DataArray:
        """Returns a boolean mask of the active cells in the SFINCS model.

        Returns:
            A boolean mask of the active cells in the SFINCS model.
        """
        return self.sfincs_model.grid["msk"] == 1

    def get_cumulative_precipitation(self) -> xr.DataArray:
        """Reads the cumulative precipitation from the SFINCS model results.

        Returns:
            An xarray DataArray containing the cumulative precipitation.
        """
        self.sfincs_model.read_results()
        return self.sfincs_model.results["cumprcp"].isel(timemax=-1)

    def get_cumulative_infiltration(self) -> xr.DataArray:
        """Reads the cumulative infiltration from the SFINCS model results.

        Returns:
            An xarray DataArray containing the cumulative infiltration.
        """
        self.sfincs_model.read_results()
        return self.sfincs_model.results["cuminf"].isel(timemax=-1)
