"""This module contains classes and functions to build and run SFINCS models for flood hazard assessment.

The main class is `SFINCSRootModel`, which is used to create and manage SFINCS models.
It provides methods to build the model, set up simulations with different forcing methods,
and read simulation results.

"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel
from hydromt_sfincs.workflows import burn_river_rect, river_source_points
from pyflwdir import FlwdirRaster
from scipy.ndimage import value_indices
from tqdm import tqdm

from geb.hydrology.routing import get_river_width
from geb.typing import (
    ArrayInt32,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
    TwoDArrayInt32,
)
from geb.workflows.raster import rasterize_like

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

if TYPE_CHECKING:
    from geb.model import GEBModel


def set_river_outflow_boundary_condition(
    sf: SfincsModel,
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
    outflow: gpd.GeoDataFrame = gpd.read_file(model_root / "gis/outflow_points.gpkg")
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
    time_range: pd.DatetimeIndex = pd.date_range(start=tstart, end=tstop, freq="h")

    # Create DataFrame with constant elevation value
    elevation_time_series_constant: pd.DataFrame = pd.DataFrame(
        data={"water_level": elevation},  # Use extracted elevation value
        index=time_range,
    )

    # Extract a unique index from the outflow point. Here, we use 1 as an example.
    outflow_index: int = (
        1  # This should be the index or a suitable ID of the outflow point
    )
    elevation_time_series_constant.columns: list[int] = [
        outflow_index
    ]  # Use an integer as column name

    # Ensure outflow has the correct index as well
    outflow["index"] = outflow_index  # Set the matching index to outflow location

    # Now set the water level forcing
    sf.setup_waterlevel_forcing(
        timeseries=elevation_time_series_constant,  # Constant time series
        locations=outflow,  # Outflow point
    )
    sf.set_root(str(simulation_root), mode="w+")

    sf.write_forcing()

    if write_figures:
        sf.plot_forcing(fn_out="waterlevel_forcing.png")
        sf.plot_basemap(fn_out="basemap.png")


class SFINCSRootModel:
    """Builds and updates SFINCS model files for flood hazard modeling."""

    def __init__(self, model: GEBModel, name: str) -> None:
        """Initializes the SFINCSRootModel with a GEBModel and event name.

        Sets up the constant parts of the model (grid, mask, rivers, etc.),
        and has methods to create simulations with actual forcing.

        Args:
            model: An instance of GEBModel containing hydrological and geographical data.
            name: A string representing the name of the event (e.g., "flood_event_2023").
                Also used to create the path to write the file to disk.
        """
        self.model = model
        self.logger = self.model.logger
        self._name: str = name

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

    def read(self) -> SFINCSRootModel:
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
        DEMs: list[dict[str, str | Path | xr.DataArray]],
        region: gpd.GeoDataFrame,
        rivers: gpd.GeoDataFrame,
        discharge: xr.DataArray,
        river_width_alpha: npt.NDArray[np.float32],
        river_width_beta: npt.NDArray[np.float32],
        mannings: xr.DataArray,
        resolution: float | int,
        nr_subgrid_pixels: int | None,
        crs: str,
        depth_calculation_method: str,
        depth_calculation_parameters: dict[str, float | int] | None = None,
        mask_flood_plains: bool = False,
        setup_outflow: bool = True,
    ) -> SFINCSRootModel:
        """Build a SFINCS model.

        Notes:
            mask_flood_plains is currently quite unstable and should be used with caution. Sometimes it leads
            to wrong regions being masked, which can lead to errors in the model.

        Args:
            DEMs: List of DEM datasets to use for the model. Should be a list of dictionaries with 'path' and 'name' keys.
            region: A GeoDataFrame defining the region of interest.
            rivers: A GeoDataFrame containing river segments.
            discharge: An xarray DataArray containing discharge values for the rivers in m^3/s.
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
            setup_outflow: Whether to set up an outflow boundary condition. Defaults to True. Mostly used for testing purposes.

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

        assert rivers.intersects(region.union_all()).all(), (
            "All rivers must intersect the model region"
        )

        self.logger.info("Starting SFINCS model build...")

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

        # in one plot plot the region boundary as well as the rivers and save to file
        fig, ax = plt.subplots(figsize=(10, 10))
        region.boundary.plot(ax=ax, color="black")

        # Remove rivers that are not represented in the grid and have no upstream rivers
        # TODO: Make an upstream flag in preprocessing for upstream rivers that is more
        # general than the MERIT-hydro specific 'maxup' attribute
        rivers: gpd.GeoDataFrame = rivers[
            (rivers["maxup"] > 0) | (rivers["represented_in_grid"])
        ]

        rivers.plot(ax=ax, color="blue")
        plt.savefig(self.path / "gis" / "rivers.png")

        if setup_outflow:
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

        # Save elevation value to a file in model_root/gis
        outflow_elev_path: Path = self.path / "gis" / "outflow_elevation.json"
        with open(outflow_elev_path, "w") as f:
            json.dump({"outflow_elevation": elevation_value}, f)

        river_representative_points = []
        for ID in rivers.index:
            river_representative_points.append(
                get_representative_river_points(ID, rivers)
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
            self.logger.info(
                f"Setting up SFINCS subgrid with {nr_subgrid_pixels} subgrid pixels..."
            )
            # only burn rivers that are wider than the subgrid pixel size
            rivers_to_burn: gpd.GeoDataFrame = rivers[
                rivers["width"] > resolution / nr_subgrid_pixels
            ].copy()
            sf.setup_subgrid(
                datasets_dep=DEMs,
                datasets_rgh=[
                    {
                        "manning": mannings.to_dataset(name="manning"),
                    }
                ],
                datasets_riv=[
                    {
                        "centerlines": rivers_to_burn.rename(
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
            self.logger.info(
                "Setting up SFINCS without subgrid - burning rivers into main grid..."
            )
            # only burn rivers that are wider than the grid size
            rivers_to_burn: gpd.GeoDataFrame = rivers[
                rivers["width"] > resolution
            ].copy()
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
                gdf_riv=rivers_to_burn,
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
        self.rivers = rivers
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
        rivers: gpd.GeoDataFrame,
        rising_limb_hours: int = 72,
        return_periods: list[int | float] = [2, 5, 10, 20, 50, 100, 250, 500, 1000],
    ) -> None:
        """Estimate discharge for specified return periods and create hydrographs.

        Args:
            model_root: path to the SFINC model root directory
            discharge: xr.DataArray containing the discharge data
            rivers: GeoDataFrame containing river segments
            rising_limb_hours: number of hours for the rising limb of the hydrograph.
            return_periods: list of return periods for which to estimate discharge.
        """
        recession_limb_hours: int = rising_limb_hours

        # here we only select the rivers that have an upstream forcing point
        rivers_with_forcing_point: gpd.GeoDataFrame = rivers[
            ~rivers["is_downstream_outflow_subbasin"]
        ]

        river_representative_points: list[list[tuple[int, int]]] = []
        for ID in rivers_with_forcing_point.index:
            river_representative_points.append(
                get_representative_river_points(ID, rivers_with_forcing_point)
            )

        discharge_by_river, _ = get_discharge_and_river_parameters_by_river(
            rivers_with_forcing_point.index,
            river_representative_points,
            discharge=discharge,
        )
        rivers_with_forcing_point: gpd.GeoDataFrame = assign_return_periods(
            rivers_with_forcing_point, discharge_by_river, return_periods=return_periods
        )

        for return_period in return_periods:
            rivers_with_forcing_point[f"hydrograph_{return_period}"] = None

        for river_idx in rivers_with_forcing_point.index:
            for return_period in return_periods:
                discharge_for_return_period = rivers_with_forcing_point.at[
                    river_idx, f"Q_{return_period}"
                ]
                hydrograph: pd.DataFrame = create_hourly_hydrograph(
                    discharge_for_return_period,
                    rising_limb_hours,
                    recession_limb_hours,
                )
                hydrograph: dict[str, Any] = {
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
    ) -> SFINCSSimulation:
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

    def create_simulation_for_return_period(
        self, return_period: int | float
    ) -> MultipleSFINCSSimulations:
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

    @property
    def active_cells(self) -> xr.DataArray:
        """Returns a boolean mask of the active cells in the SFINCS model.

        Returns:
            A boolean mask of the active cells in the SFINCS model.
        """
        return self.sfincs_model.grid["msk"] == 1

    @property
    def inflow_rivers(self) -> gpd.GeoDataFrame:
        """Returns a GeoDataFrame of rivers that are inflow points to the model.

        Returns:
            A GeoDataFrame of rivers that are inflow rivers.
        """
        non_headwater_rivers: gpd.GeoDataFrame = self.rivers[self.rivers["maxup"] > 0]
        non_outflow_basins: gpd.GeoDataFrame = non_headwater_rivers[
            ~non_headwater_rivers["is_downstream_outflow_subbasin"]
        ]
        upstream_branches_in_domain = np.unique(
            self.rivers["downstream_ID"], return_counts=True
        )

        rivers_with_inflow = []
        for idx, row in non_outflow_basins.iterrows():
            if idx in upstream_branches_in_domain[0] and (
                upstream_branches_in_domain[1][
                    np.where(upstream_branches_in_domain[0] == idx)
                ]
                < row["maxup"]
            ):
                rivers_with_inflow.append(idx)

        return self.rivers[self.rivers.index.isin(rivers_with_inflow)]

    @property
    def non_inflow_rivers(self) -> gpd.GeoDataFrame:
        """Returns a GeoDataFrame of rivers that are not inflow points to the model.

        Returns:
            A GeoDataFrame of rivers that are not inflow rivers.
        """
        inflow_rivers: gpd.GeoDataFrame = self.inflow_rivers
        return self.rivers[~self.rivers.index.isin(inflow_rivers.index)]

    @property
    def headwater_rivers(self) -> gpd.GeoDataFrame:
        """Returns a GeoDataFrame of headwater rivers in the model.

        Returns:
            A GeoDataFrame of headwater rivers.
        """
        return self.rivers[self.rivers["maxup"] == 0]

    @property
    def has_inflow(self) -> bool:
        """Checks if the model has any inflow rivers.

        Returns:
            True if the model has inflow rivers, False otherwise.
        """
        return not self.inflow_rivers.empty


class MultipleSFINCSSimulations:
    """Manages multiple SFINCS simulations as a single entity."""

    def __init__(self, simulations: list[SFINCSSimulation]) -> None:
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
        write_figures: bool = False,
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
        sfincs_model.set_root(str(self.path), mode="w+")

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

        # Track total volumes added via forcings (for water balance debugging)
        self.total_runoff_volume_m3: float = 0.0
        self.total_discharge_volume_m3: float = 0.0
        self.discarded_accumulated_generated_discharge_m3: float = 0.0

    def print_forcing_volume(self) -> None:
        """Print all forcing volumes for debugging the water balance."""
        msg: str = (
            f"SFINCS Forcing volumes: runoff={int(self.total_runoff_volume_m3)} m3, "
            f"discarded discharge={int(self.discarded_accumulated_generated_discharge_m3)} m3, "
            f"discharge={int(self.total_discharge_volume_m3)} m3"
        )
        print(msg)

    def set_forcing_from_grid(
        self, nodes: gpd.GeoDataFrame, discharge_grid: xr.DataArray
    ) -> None:
        """Sets up discharge forcing for the SFINCS model from a gridded dataset.

        Args:
            nodes: A GeoDataFrame containing the locations of the discharge forcing points.
            discharge_grid: Path to a raster file or an xarray DataArray containing discharge values in m^3/s.
                Usually this is from a hydrological model.
        """
        nodes: gpd.GeoDataFrame = nodes.copy()
        nodes["geometry"] = nodes["geometry"].apply(get_start_point)

        river_representative_points = []
        for ID in nodes.index:
            river_representative_points.append(
                get_representative_river_points(
                    ID,
                    nodes,
                )
            )

        discharge_by_river, _ = get_discharge_and_river_parameters_by_river(
            nodes.index,
            river_representative_points,
            discharge=discharge_grid,
        )

        locations = nodes.to_crs(self.sfincs_model.crs)

        self.set_discharge_forcing_from_nodes(
            nodes=locations,
            timeseries=discharge_by_river,
        )

    def set_headwater_forcing_from_grid(
        self,
        discharge_grid: xr.DataArray,
    ) -> None:
        """Sets up discharge forcing for the SFINCS model from a gridded dataset.

        Args:
            discharge_grid: Path to a raster file or an xarray DataArray containing discharge values in m^3/s.
                Usually this is from a hydrological model.
        """
        headwater_rivers: gpd.GeoDataFrame = self.sfincs_root_model.headwater_rivers
        self.set_forcing_from_grid(
            nodes=headwater_rivers,
            discharge_grid=discharge_grid,
        )

    def set_inflow_forcing_from_grid(
        self,
        discharge_grid: xr.DataArray,
    ) -> None:
        """Sets up discharge forcing for the SFINCS model from a gridded dataset.

        Args:
            discharge_grid: Path to a raster file or an xarray DataArray containing discharge values in m^3/s.
                Usually this is from a hydrological model.
        """
        inflow_rivers: gpd.GeoDataFrame = self.sfincs_root_model.inflow_rivers
        self.set_forcing_from_grid(
            nodes=inflow_rivers,
            discharge_grid=discharge_grid,
        )

    def set_river_inflow(
        self, nodes: gpd.GeoDataFrame, timeseries: pd.DataFrame
    ) -> None:
        """Sets up river inflow boundary conditions for the SFINCS model.

        For inflow we use a negative index.

        """
        nodes.index = [-idx for idx in nodes.index]  # SFINCS negative index for inflow
        timeseries.columns = [-col for col in timeseries.columns]
        self.set_discharge_forcing_from_nodes(
            nodes=nodes,
            timeseries=timeseries,
        )

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
        assert set(timeseries.columns) == set(nodes.index)

        if "dis" in self.sfincs_model.forcing:
            assert not np.isin(
                nodes.index, self.sfincs_model.forcing["dis"].index
            ).any(), "This forcing would overwrite existing discharge forcing points"

        assert (
            self.sfincs_model.region.union_all()
            .contains(nodes.geometry.to_crs(self.sfincs_model.crs))
            .all()
        ), "All forcing locations must be within the model region"

        self.sfincs_model.setup_discharge_forcing(
            locations=nodes, timeseries=timeseries, merge=True
        )

        self.sfincs_model.write_forcing()
        self.sfincs_model.write_config()

        assert self.end_time == timeseries.index[-1], (
            "End time of timeseries does not match simulation end time, this will lead to accounting errors"
        )

        # the last timestep will not be used in SFINCS because it is the end time, which is why we
        # discard it
        self.total_discharge_volume_m3 += (
            timeseries[:-1].sum(axis=1)
            * (timeseries.index[1:] - timeseries.index[:-1]).total_seconds()
        ).sum()

        self.print_forcing_volume()

        if self.write_figures:
            self.sfincs_model.plot_basemap(fn_out="src_points_check.png")
            self.sfincs_model.plot_forcing(fn_out="forcing.png")

    def set_runoff_forcing(
        self,
        runoff_m: xr.DataArray,
        area_m2: TwoDArrayFloat32,
    ) -> None:
        """Sets up precipitation forcing for the SFINCS model from a gridded dataset.

        Args:
            runoff_m: xarray DataArray containing runoff values in m per time step.
            area_m2: xarray DataArray containing the area of each runoff grid cell in m².
        """
        assert runoff_m.rio.crs is not None, "precipitation_grid should have a crs"
        assert (
            pd.to_datetime(runoff_m.time[0].item()).to_pydatetime() <= self.start_time
        )
        assert pd.to_datetime(runoff_m.time[-1].item()).to_pydatetime() >= self.end_time

        runoff_m: xr.DataArray = runoff_m.sel(
            time=slice(self.start_time, self.end_time)
        )

        self.sfincs_model.setup_precip_forcing_from_grid(
            precip=(runoff_m * 1000).to_dataset(name="precip")
        )  # convert from m/h to mm/h for SFINCS

        self.sfincs_model.write_forcing()
        self.sfincs_model.write_config()

        self.total_runoff_volume_m3 += (
            (runoff_m.isel(time=slice(None, -1)) * area_m2).sum().item()
        )
        self.print_forcing_volume()

    def set_accumulated_runoff_forcing(
        self,
        runoff_m: xr.DataArray,
        river_network: FlwdirRaster,
        mask: TwoDArrayBool,
        river_ids: TwoDArrayInt32,
        upstream_area: TwoDArrayFloat32,
        cell_area: TwoDArrayFloat32,
    ) -> None:
        """Sets up accumulated runoff forcing for the SFINCS model.

        This function accumulates the runoff from the provided runoff grid to the starting
        points of each river segment in the river network.

        Args:
            runoff_m: xarray DataArray containing runoff values in m per time step.
            river_network: FlwdirRaster representing the river network flow directions.
            mask: Boolean mask indicating the cells within the river basin.
            river_ids: 2D numpy array of river segment IDs for each cell in the grid.
            upstream_area: 2D numpy array of upstream area values for each cell in the grid.
            cell_area: 2D numpy array of cell area values for each cell in the grid.
        """
        # select only the time range needed
        runoff_m: xr.DataArray = runoff_m.sel(
            time=slice(self.start_time, self.end_time)
        )

        # for accounting purposes, we set all runoff outside the model region to zero
        region = self.sfincs_model.region.to_crs(runoff_m.rio.crs)
        region["value"] = 1
        region_mask = rasterize_like(
            region,
            column="value",
            raster=runoff_m.isel(time=0),
            dtype=np.int32,
            nodata=0,
            all_touched=True,
        ).astype(bool)
        original_dimensions = runoff_m.dims
        runoff_m: xr.DataArray = xr.where(region_mask, runoff_m, 0, keep_attrs=True)
        # xr.where changes the dimension order sometimes, so we ensure it is correct
        runoff_m: xr.DataArray = runoff_m.transpose(*original_dimensions)

        # we want to get all the discharge upstream from the starting point of each river segment
        # therefore, we first remove all river cells except for the starting point of each river segment
        # TODO: this can be changed so that runoff is added along the river segment, rather than
        # just the most upstream point
        xy_per_river_segment = value_indices(river_ids, ignore_value=-1)
        for COMID, (ys, xs) in xy_per_river_segment.items():
            river_upstream_area = upstream_area[ys, xs]
            up_to_downstream_ids = np.argsort(river_upstream_area)

            ys_up_to_down: npt.NDArray[np.int64] = ys[up_to_downstream_ids]
            xs_up_to_down: npt.NDArray[np.int64] = xs[up_to_downstream_ids]

            for i in range(1, len(ys_up_to_down)):
                river_ids[ys_up_to_down[i], xs_up_to_down[i]] = -1

        # confirm that each river segment is represented by exactly one cell
        assert (np.unique(river_ids, return_counts=True)[1][1:] == 1).all()

        river_cells: TwoDArrayBool = river_ids != -1
        river_ids_mapping: ArrayInt32 = river_ids[river_cells]

        # starting from each river cell, create an upstream basin map for which
        # the discharge will be accumulated
        subbasins: ArrayInt32 = river_network.basins(
            np.where(river_cells.ravel())[0],
            ids=np.arange(1, river_cells.sum() + 1, step=1, dtype=np.int32),
        )[mask]

        timestep_size: xr.DataArray = runoff_m.time.diff(dim="time").astype(
            "timedelta64[s]"
        )
        # confirm that timestep size is constant
        assert (timestep_size == timestep_size[0]).all()
        timestep_size_seconds: int = timestep_size[0].item().seconds

        # get the generated discharge in m3/s for each cell
        generated_discharge_m3_per_s: xr.DataArray = (
            runoff_m * cell_area / timestep_size_seconds
        )

        # accumulate generated discharge to the river starting points
        accumulated_generated_discharge_m3_per_s: TwoDArrayFloat64 = (
            np.apply_along_axis(
                func1d=lambda x: np.bincount(subbasins, weights=x),
                axis=1,
                arr=generated_discharge_m3_per_s.values[:, mask],
            )
        )

        # a subbasin value of 0 means that the cell does not belong to any subbasin
        # this is possible for cells that flow into the river segment closest
        # to the outlet of the river network
        # therefore, we discard the generated discharge from these cells
        if (subbasins == 0).any():
            # for testing, we return the mean discarded generated discharge
            discarded_generated_discharge_m3_per_s: np.float64 = (
                accumulated_generated_discharge_m3_per_s[:, 0].mean()
            )
            # Track discarded volume (m3) from subbasin==0 cells for debugging
            duration_seconds = (self.end_time - self.start_time).total_seconds()
            self.discarded_accumulated_generated_discharge_m3 += float(
                discarded_generated_discharge_m3_per_s * duration_seconds
            )
            accumulated_generated_discharge_m3_per_s: TwoDArrayFloat64 = (
                accumulated_generated_discharge_m3_per_s[:, 1:]
            )

        assert accumulated_generated_discharge_m3_per_s.shape[1] == river_cells.sum()

        # create the forcing timeseries for each river segment starting point
        nodes: gpd.GeoDataFrame = self.sfincs_root_model.rivers.copy()
        nodes["geometry"] = nodes["geometry"].apply(get_start_point)
        nodes: gpd.GeoDataFrame = nodes.sort_index()
        timeseries: pd.DataFrame = pd.DataFrame(
            {
                "time": generated_discharge_m3_per_s.time,
            }
        ).set_index("time")

        for i, node in nodes.iterrows():
            if node["represented_in_grid"]:
                idx = np.where(node.name == river_ids_mapping)[0]
                assert len(idx) == 1
                idx = idx[0]
                timeseries[i] = accumulated_generated_discharge_m3_per_s[:, idx]
            else:
                timeseries[i] = 0.0

        self.set_discharge_forcing_from_nodes(
            nodes=nodes,
            timeseries=timeseries,
        )

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
        if hasattr(flood_depth, "compute"):
            flood_depth = flood_depth.compute()
        flooded_pixels = flood_depth.where(flood_depth > 0).sum().item()
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
        return (self.sfincs_model.grid["msk"] == 2).any().item()

    def get_cumulative_precipitation(self) -> xr.DataArray:
        """Reads the cumulative precipitation from the SFINCS model results.

        Returns:
            An xarray DataArray containing the cumulative precipitation.
        """
        self.sfincs_model.read_results()
        cumulative_precipitation = self.sfincs_model.results["cumprcp"].isel(timemax=-1)
        assert isinstance(cumulative_precipitation, xr.DataArray)
        return cumulative_precipitation
