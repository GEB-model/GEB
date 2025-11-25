"""This module contains classes and functions to build and run SFINCS models for flood hazard assessment.

The main class is `SFINCSRootModel`, which is used to create and manage SFINCS models.
It provides methods to build the model, set up simulations with different forcing methods,
and read simulation results.

"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyflwdir
import rasterio
import xarray as xr
from hydromt_sfincs import SfincsModel
from hydromt_sfincs.workflows import burn_river_rect, river_source_points
from pyflwdir import FlwdirRaster
from pyflwdir.dem import fill_depressions
from scipy.ndimage import value_indices
from tqdm import tqdm

from geb.hydrology.routing import get_river_width
from geb.types import (
    ArrayInt32,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
    TwoDArrayInt32,
)
from geb.workflows.io import load_geom
from geb.workflows.raster import calculate_cell_area, clip_region, rasterize_like

from .workflows import get_river_depth, get_river_manning
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
        self.rivers: gpd.GeoDataFrame = load_geom(self.path / "rivers.geoparquet")
        self.region: gpd.GeoDataFrame = load_geom(self.path / "region.geoparquet")
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
        subgrid: bool,
        grid_size_multiplier: int,
        depth_calculation_method: str,
        depth_calculation_parameters: dict[str, float | int] | None = None,
        coastal: bool = False,
        low_elevation_coastal_zone_mask: gpd.GeoDataFrame | None = None,
        coastal_boundary_exclude_mask: gpd.GeoDataFrame | None = None,
        setup_outflow: bool = True,
        initial_water_level: float = 0.0,
    ) -> SFINCSRootModel:
        """Build a SFINCS model.

        Args:
            DEMs: List of DEM datasets to use for the model. Should be a list of dictionaries with 'path' and 'name' keys.
            region: A GeoDataFrame defining the region of interest.
            rivers: A GeoDataFrame containing river segments.
            discharge: An xarray DataArray containing discharge values for the rivers in m^3/s.
            river_width_alpha: An numpy array of river width alpha parameters. Used for calculating river width.
            river_width_beta: An numpy array of river width beta parameters. Used for calculating river width
            mannings: A xarray DataArray of Manning's n values for the rivers.
            grid_size_multiplier: The number of grid cells to combine when setting up the model grid.
                if 1, no combining is done.
                if 2, every 2x2 grid cells are combined into one cell, etc.
            subgrid: Whether to set up subgrid pixels for the model.
            depth_calculation_method: The method to use for calculating river depth. Can be 'manning' or 'power_law'.
            depth_calculation_parameters: A dictionary of parameters for the depth calculation method. Only used if
                depth_calculation_method is 'power_law', in which case it should contain 'c' and 'd' keys.
            coastal: Whether to set up coastal boundary conditions. Defaults to False.
            low_elevation_coastal_zone_mask: A GeoDataFrame defining the low elevation coastal zone to set as active cells.
            coastal_boundary_exclude_mask: A GeoDataFrame defining areas to exclude from the coastal boundary condition cells.
            setup_outflow: Whether to set up an outflow boundary condition. Defaults to True. Mostly used for testing purposes.
            initial_water_level: The initial water level to initiate the model. SFINCS fills all cells below this level with water.

        Returns:
            The SFINCSRootModel instance with the built model.

        Raises:
            ValueError: if depth_calculation_method is not 'manning' or 'power_law',
            ValueError: if grid_size_multiplier is not a positive integer.

        """
        if not isinstance(grid_size_multiplier, int) or grid_size_multiplier <= 0:
            raise ValueError("grid_size_multiplier must be a positive integer")
        if grid_size_multiplier == 1 and subgrid:
            raise ValueError(
                "Cannot use subgrid pixels when grid_size_multiplier is 1 (no aggregation)"
            )

        assert depth_calculation_method in [
            "manning",
            "power_law",
        ], "Method should be 'manning' or 'power_law'"

        assert rivers.intersects(region.union_all()).all(), (
            "All rivers must intersect the model region"
        )

        # Remove rivers that are not represented in the grid and have no upstream rivers
        # TODO: Make an upstream flag in preprocessing for upstream rivers that is more
        # general than the MERIT-hydro specific 'maxup' attribute
        self.rivers: gpd.GeoDataFrame = rivers[
            (rivers["maxup"] > 0) | (rivers["represented_in_grid"])
        ]
        self.rivers.to_parquet(self.path / "rivers.geoparquet")

        self.logger.info("Starting SFINCS model build...")

        # build base model
        sf: SfincsModel = SfincsModel(root=str(self.path), mode="w+", write_gis=True)
        self.sfincs_model = sf
        mask_ds = DEMs[0]["elevtn"]
        assert isinstance(mask_ds, xr.Dataset)
        mask: xr.DataArray = mask_ds["elevtn"]

        self.region: gpd.GeoDataFrame = region.to_crs(mask.rio.crs)
        self.region.to_parquet(self.path / "region.geoparquet")
        del region

        region_burned: xr.DataArray = rasterize_like(
            gdf=self.region.to_crs(mask.rio.crs),
            burn_value=1,
            raster=mask,
            dtype=np.int32,
            nodata=0,
            all_touched=True,
        ).astype(bool)
        assert isinstance(mask, xr.DataArray)

        resolution: tuple[float, float] = mask.rio.resolution()
        if abs(abs(resolution[0]) - abs(resolution[1])) > 1e-10:
            raise ValueError("DEM resolution must be square pixels")

        mask: xr.DataArray = clip_region(
            region_burned, align=abs(resolution[0]) * grid_size_multiplier
        )[0]

        # if y axis is descending (usually for geographical grids), flip it
        if mask.y[-1] < mask.y[0]:
            mask: xr.DataArray = mask.isel(y=slice(None, None, -1))

        geotransformation = mask.rio.transform(recalc=True)
        crs = mask.rio.crs
        if not crs.is_epsg_code:
            raise ValueError("CRS must be an EPSG code")

        sf.setup_grid(
            x0=geotransformation.c,
            y0=geotransformation.f,
            dx=geotransformation.a * grid_size_multiplier,
            dy=geotransformation.e * grid_size_multiplier,
            mmax=mask.rio.width // grid_size_multiplier + 1,
            nmax=mask.rio.height // grid_size_multiplier + 1,
            rotation=0.0,
            epsg=crs.to_epsg(),
        )
        if crs.is_geographic:
            sf.setup_config(crsgeo=1)

        DEMs: list[dict[str, str | Path | xr.DataArray | int]] = [
            {**DEM, **{"reproj_method": "bilinear"}} for DEM in DEMs
        ]

        # HydroMT-SFINCS only accepts datasets with an 'elevtn' variable. Therefore, the following
        # is a bit convoluted. We first open the dataarray, then convert it to a dataset,
        # and set the name as elevtn.
        sf.setup_dep(datasets_dep=DEMs)

        flood_plain: gpd.GeoDataFrame = self.get_flood_plain()
        sf.setup_mask_active(flood_plain, reset_mask=True)

        if coastal:
            # set zsini based on the minimum elevation
            assert isinstance(sf.config, dict)
            sf.config["zsini"] = initial_water_level  # ty: ignore[invalid-assignment]

            if not isinstance(coastal_boundary_exclude_mask, gpd.GeoDataFrame):
                raise ValueError(
                    "coastal_boundary_exclude_mask must be provided when coastal is True"
                )

            # activate low elevation coastal zone
            if not isinstance(low_elevation_coastal_zone_mask, gpd.GeoDataFrame):
                raise ValueError(
                    "low_elevation_coastal_zone_mask must be provided when coastal is True"
                )
            sf.setup_mask_active(
                include_mask=low_elevation_coastal_zone_mask,
                mask_buffer=5000,
                reset_mask=False,
            )

            # setup the coastal boundary conditions
            sf.setup_mask_bounds(
                btype="waterlevel",
                zmax=2,  # maximum elevation for valid boundary cells
                exclude_mask=coastal_boundary_exclude_mask,
                all_touched=True,
            )

        #
        # in one plot plot the region boundary as well as the rivers and save to file
        fig, ax = plt.subplots(figsize=(10, 10))
        self.region.boundary.plot(ax=ax, color="black")

        self.rivers.plot(ax=ax, color="blue")
        plt.savefig(self.path / "gis" / "rivers.png")

        outflow_river_upa: int = 0
        outflow_river_len: int = 0
        if setup_outflow:
            sf.setup_river_outflow(
                rivers=self.rivers.to_crs(sf.crs),
                keep_rivers_geom=True,
                river_upa=outflow_river_upa,
                river_len=outflow_river_len,
                btype="waterlevel",
            )

        # find outflow points and save for later use
        outflow_points = river_source_points(
            gdf_riv=self.rivers.to_crs(sf.crs),
            gdf_mask=sf.region,
            src_type="outflow",
            buffer=self.estimated_cell_size_m,
            river_upa=outflow_river_upa,
            river_len=outflow_river_len,
        ).to_crs(sf.crs)

        # give error if outflow greater than 1
        if len(outflow_points) > 1 and setup_outflow:
            raise ValueError(
                "More than one outflow point found, outflow boundary condition will fail to setup"
            )
        elif len(outflow_points) == 0 and setup_outflow:
            raise ValueError(
                "No outflow point found, outflow boundary condition will fail to setup"
            )
        if len(outflow_points) == 1:
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
        for ID in self.rivers.index:
            river_representative_points.append(
                get_representative_river_points(ID, self.rivers)
            )

        discharge_by_river, river_parameters = (
            get_discharge_and_river_parameters_by_river(
                self.rivers.index.tolist(),
                river_representative_points,
                discharge=discharge,
                river_width_alpha=river_width_alpha,
                river_width_beta=river_width_beta,
            )
        )

        self.rivers = assign_return_periods(
            self.rivers, discharge_by_river, return_periods=[2]
        )

        river_width_unknown_mask = self.rivers["width"].isnull()

        self.rivers.loc[river_width_unknown_mask, "width"] = get_river_width(
            river_parameters["river_width_alpha"][river_width_unknown_mask],
            river_parameters["river_width_beta"][river_width_unknown_mask],
            self.rivers.loc[river_width_unknown_mask, "Q_2"],
        ).astype(np.float64)

        self.rivers["depth"] = get_river_depth(
            self.rivers,
            method=depth_calculation_method,
            parameters=depth_calculation_parameters,
            bankfull_column="Q_2",
        )

        self.rivers["manning"] = get_river_manning(self.rivers)

        export_rivers(self.path, self.rivers)

        # Because hydromt-sfincs does a lot of filling default values when data
        # is missing, we need to be extra sure that the required columns are
        # present and contain valid data.
        assert self.rivers["width"].notnull().all(), "River width cannot be null"
        assert self.rivers["depth"].notnull().all(), "River depth cannot be null"
        assert self.rivers["manning"].notnull().all(), (
            "River Manning's n cannot be null"
        )

        # only burn rivers that are wider than the grid size
        rivers_to_burn: gpd.GeoDataFrame = self.rivers[
            self.rivers["width"] > self.estimated_cell_size_m
        ].copy()

        # if sfincs is run with subgrid, we set up the subgrid, with burned in rivers and mannings
        # roughness within the subgrid. If not, we burn the rivers directly into the main grid,
        # including mannings roughness.
        if subgrid:
            self.logger.info(
                f"Setting up SFINCS subgrid with {grid_size_multiplier} subgrid pixels..."
            )
            # only burn rivers that are wider than the subgrid pixel size
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
                nr_subgrid_pixels=grid_size_multiplier,
                nlevels=20,
                nrmax=500,
            )

            sf.write_subgrid()
        else:
            self.logger.info(
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
                gdf_riv=rivers_to_burn,
                da_man=sf.grid.manning,
                rivwth_name="width",
                rivdph_name="depth",
                manning_name="manning",
                segment_length=90.0,
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

        return self

    def get_flood_plain(self, maximum_hand: float = 30.0) -> gpd.GeoDataFrame:
        """Returns the flood plain grid of the SFINCS model.

        Args:
            maximum_hand: The maximum Height Above Nearest Drainage (HAND) value to consider as flood plain, in meters. Default is 30.0 m.

        Returns:
            The flood plain as a GeoDataFrame.
        """
        region_raster = rasterize_like(
            gdf=self.region,
            raster=self.elevation,
            dtype=np.uint8,
            nodata=0,
            burn_value=1,
            all_touched=True,
        ).astype(bool)

        # get a raster of all drainage cells (i.e., 0 HAND). We do so
        # by burning the rivers into a raster
        drainage_cells = rasterize_like(
            gdf=self.rivers.to_crs(self.elevation.rio.crs),
            raster=self.elevation,
            dtype=np.uint8,
            nodata=0,
            burn_value=1,
            all_touched=True,
        ).astype(bool)

        # obtain the elevation grid. Some cells on the outside of the region
        # drain to a location outside the region. In the HAND calculation,
        # these cells are given a very low HAND, and thus would also be part of
        # the floodplain. To avoid this, we set the cells outside the region
        # to a very low elevation, so that cells that drain outside the region
        # get an extremly high HAND value, and thus will not be part of the
        # flood plain.
        elevation = self.elevation.values.copy()
        elevation[~region_raster.values] = -10_000

        # Fill depressions to ensure proper flow direction calculation
        elevation, d8 = fill_depressions(elevation, nodata=np.nan)
        flow_raster = pyflwdir.from_array(
            d8,
            transform=self.elevation.rio.transform(),
            latlon=self.is_geographic,
        )

        height_above_nearest_drainage = xr.full_like(
            self.elevation, np.nan, dtype=np.float32
        )
        height_above_nearest_drainage.values = flow_raster.hand(
            drain=drainage_cells.values, elevtn=elevation
        )
        height_above_nearest_drainage = xr.where(
            height_above_nearest_drainage != -9999.0,
            height_above_nearest_drainage,
            np.nan,
        )
        height_above_nearest_drainage = xr.where(
            region_raster,
            height_above_nearest_drainage,
            np.nan,
        )

        flood_plain: xr.DataArray = height_above_nearest_drainage <= maximum_hand

        # convert flood plain raster to vector
        flood_plain_geom = list(
            rasterio.features.shapes(
                flood_plain.astype(np.uint8),
                mask=flood_plain,
                connectivity=8,
                transform=flood_plain.rio.transform(recalc=True),
            ),
        )
        flood_plain_geom = gpd.GeoDataFrame.from_features(
            [{"geometry": geom[0], "properties": {}} for geom in flood_plain_geom],
            crs=flood_plain.rio.crs,
        )

        return flood_plain_geom

    @property
    def mask(self) -> xr.DataArray:
        """Returns the mask grid of the SFINCS model.

        Returns:
            The mask grid as an xarray DataArray.
        """
        return self.sfincs_model.grid["msk"]

    @property
    def elevation(self) -> xr.DataArray:
        """Returns the elevation grid of the SFINCS model.

        Returns:
            The elevation grid as an xarray DataArray.
        """
        return self.sfincs_model.grid["dep"]

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the shape of the SFINCS model grid.

        Returns:
            The shape of the SFINCS model grid as a tuple (nrows, ncols).
        """
        return self.sfincs_model.grid["dep"].shape

    @property
    def is_geographic(self) -> bool:
        """Returns whether the SFINCS model uses a geographic coordinate system.

        Returns:
            True if the SFINCS model uses a geographic coordinate system, False otherwise.
        """
        return self.sfincs_model.crs.is_geographic

    @property
    def cell_area(self) -> float | int | xr.DataArray:
        """Returns the area of a single cell in the SFINCS model grid in square meters.

        Returns:
            The area of a single cell in the SFINCS model grid in m².
        """
        if self.is_geographic:
            cell_area: xr.DataArray = xr.full_like(
                self.elevation, np.nan, dtype=np.float32
            )
            cell_area.values = calculate_cell_area(
                self.elevation.rio.transform(), shape=self.shape
            )
            return cell_area
        else:
            return xr.full_like(
                self.elevation,
                (
                    self.sfincs_model.grid["dep"].rio.resolution()[0]
                    * self.sfincs_model.grid["dep"].rio.resolution()[1]
                ),
                dtype=np.float32,
            )

    @property
    def estimated_cell_size_m(self) -> float:
        """Returns the estimated cell size of the SFINCS model grid in meters.

        For geographic coordinate systems, this is an approximate value based on the average
        cell area. For projected coordinate systems, this is the actual cell size.

        Returns:
            The estimated cell size of the SFINCS model grid in meters.
        """
        if self.is_geographic:
            avg_cell_area = self.cell_area.mean().item()
            estimated_cell_size = np.sqrt(avg_cell_area)
            return estimated_cell_size
        else:
            return math.sqrt(
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
                    time.isoformat(): Q.item()  # ty: ignore[unresolved-attribute]
                    for time, Q in hydrograph.iterrows()
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

    def create_coastal_simulation(self, return_period: int) -> SFINCSSimulation:
        """
        Creates a SFINCS simulation with coastal water level forcing for a specified return period.

        It reads the coastal hydrograph timeseries and locations from pre-defined files, and sets the coastal water level forcing for the simulation.

        Args:
            return_period: The return period for which to create the coastal simulation.

        Returns:
            An instance of SFINCSSimulation configured with coastal water level forcing.
        """
        # prepare coastal timeseries and locations
        timeseries = pd.read_csv(
            Path(
                f"output/hydrographs/gtsm_spring_tide_hydrograph_rp{return_period:04d}.csv"
            ),
            index_col=0,
        )

        locations: gpd.GeoDataFrame = (  # ty: ignore[invalid-assignment]
            load_geom(self.model.files["geom"]["gtsm/stations_coast_rp"])
            .rename(columns={"station_id": "stations"})
            .set_index("stations")
        )

        # convert index to int
        locations.index = locations.index.astype(int)

        timeseries.index = pd.to_datetime(timeseries.index, format="%Y-%m-%d %H:%M:%S")
        # convert columns to int
        timeseries.columns = timeseries.columns.astype(int)

        # Align timeseries columns with locations index
        timeseries = timeseries.loc[:, locations.index]

        # now convert to incrementing integers starting from 0
        timeseries.columns = range(len(timeseries.columns))
        locations.index = range(len(locations.index))

        timeseries = timeseries.iloc[250:-250]  # trim the first and last 250 rows

        simulation: SFINCSSimulation = self.create_simulation(
            simulation_name=f"rp_{return_period}_coastal",
            start_time=timeseries.index[0],
            end_time=timeseries.index[-1],
        )

        offset = xr.open_dataarray(
            self.model.files["other"]["coastal/global_ocean_mean_dynamic_topography"]
        ).rio.write_crs("EPSG:4326")

        # set coastal forcing model
        simulation.set_coastal_waterlevel_forcing(
            timeseries=timeseries, locations=locations, offset=offset
        )
        return simulation

    def create_simulation_for_return_period(
        self, return_period: int, coastal: bool = False
    ) -> MultipleSFINCSSimulations:
        """Creates multiple SFINCS simulations for a specified return period.

        The method groups rivers by their calculation group and creates a separate
        simulation for each group. Each simulation is configured with discharge
        hydrographs corresponding to the specified return period.

        Args:
            return_period: The return period for which to create simulations.
            coastal: Whether to create a coastal simulation.

        Returns:
            An instance of MultipleSFINCSSimulations containing the created simulations.
                This class aims to emulate a single SFINCSSimulation instance as if
                it was one.

        Raises:
            ValueError: If inflow node indices cannot be converted to integers,
                if the discharge DataFrame columns cannot be converted to integers,
                or if the discharge hydrographs contain NaN values.
        """
        rivers: gpd.GeoDataFrame = import_rivers(self.path, postfix="_return_periods")
        assert (~rivers["is_downstream_outflow_subbasin"]).all()

        rivers["topological_stream_order"] = get_topological_stream_order(rivers)
        rivers: gpd.GeoDataFrame = assign_calculation_group(rivers)

        working_dir: Path = self.path / "working_dir"
        working_dir_return_period: Path = working_dir / f"rp_{return_period}"

        print(f"Running SFINCS for return period {return_period} years")
        simulations: list[SFINCSSimulation] = []

        # create coastal simulation
        if coastal:
            simulation: SFINCSSimulation = self.create_coastal_simulation(return_period)
            simulations.append(simulation)

        # create river inflow simulations
        for group, group_rivers in tqdm(rivers.groupby("calculation_group")):
            simulation_root = working_dir_return_period / str(group)

            shutil.rmtree(simulation_root, ignore_errors=True)
            simulation_root.mkdir(parents=True, exist_ok=True)

            inflow_nodes = group_rivers.copy()
            # Keep the original index (river IDs) so they don't collide with existing forcing points.
            inflow_nodes["geometry"] = inflow_nodes["geometry"].apply(get_start_point)

            # Ensure indices are integer-like (helpful for later comparisons)
            try:
                inflow_nodes.index = pd.Index(inflow_nodes.index).astype(int)
            except Exception:
                raise ValueError("Inflow node indices must be convertible to integers")

            # Build list of hydrograph DataFrames using the original node indices as column names
            Q_list: list[pd.DataFrame] = []
            for node_idx in inflow_nodes.index:
                hydro = inflow_nodes.at[node_idx, f"hydrograph_{return_period}"]
                # hydro is expected to be dict-like {iso_timestamp: Q} — convert to DataFrame with column named node_idx
                df = pd.DataFrame.from_dict(hydro, orient="index", columns=[node_idx])
                Q_list.append(df)

            # Concatenate the per-node series into a single DataFrame; index -> timestamps
            Q: pd.DataFrame = pd.concat(Q_list, axis=1)
            Q.index = pd.to_datetime(Q.index)

            # Ensure columns have consistent integer dtype (same as inflow_nodes.index)
            try:
                Q.columns = pd.Index(Q.columns).astype(int)
            except Exception:
                raise ValueError(
                    "Discharge DataFrame columns must be convertible to integers"
                )

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
            if not coastal:
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

    def cleanup(self) -> None:
        """Cleans up the SFINCS model directory."""
        shutil.rmtree(self.path, ignore_errors=True)


class MultipleSFINCSSimulations:
    """Manages multiple SFINCS simulations as a single entity."""

    def __init__(self, simulations: list[SFINCSSimulation]) -> None:
        """Simulates running multiple SFINCS simulations as one.

        Args:
            simulations: A list of SFINCSSimulation instances to manage together.
        """
        self.simulations = simulations

    def run(self, ncpus: int | str = "auto", gpu: bool | str = "auto") -> None:
        """Runs all contained SFINCS simulations.

        Args:
            ncpus: Number of CPU cores to use for the simulation. Can be
                an integer or 'auto' to automatically detect available cores.
            gpu: Whether to use GPU acceleration for the simulations. Can be
                True, False, or 'auto' to automatically detect GPU availability.
        """
        for simulation in self.simulations:
            simulation.run(ncpus=ncpus, gpu=gpu)

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
            write_figures: Whether to generate and save figures for the model. Defaults to False.
        """
        self._name = simulation_name
        self.write_figures = write_figures
        self.start_time = start_time
        self.end_time = end_time
        self.sfincs_root_model = sfincs_root_model

        sfincs_model = sfincs_root_model.sfincs_model
        sfincs_model.set_root(str(self.path), mode="w+")

        sfincs_model.setup_config(
            alpha=0.5,  # alpha is the parameter for the CFL-condition reduction. Decrease for additional numerical stability, minimum value is 0.1 and maximum is 0.75 (0.5 default value)
            h73table=1,  # use h^(7/3) table for friction calculation. This is slightly less accurate but up to 30% faster
            tspinup=spinup_seconds,  # spinup time in seconds
            dtout=900,  # output time step in seconds
            tref=to_sfincs_datetime(start_time),  # reference time for the simulation
            tstart=to_sfincs_datetime(start_time),  # simulation start time
            tstop=to_sfincs_datetime(end_time),  # simulation end time
            **make_relative_paths(sfincs_model.config, self.root_path, self.path),
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

    def set_coastal_waterlevel_forcing(
        self,
        locations: gpd.GeoDataFrame,
        timeseries: pd.DataFrame,
        buffer: int = 1e5,
        offset: xr.DataArray | None = None,
    ) -> None:
        """Sets up coastal water level forcing for the SFINCS model from a timeseries.

        Args:
            locations: A GeoDataFrame containing the locations of the water level forcing points.
            timeseries: A DataFrame containing the water level timeseries for each node.
                The columns should match the index of the locations GeoDataFrame.
            buffer: Buffer distance in meters to extend the model domain for coastal forcing points.
            offset: The offset of water levels based on the m
        """
        # select only locations that are in the model
        self.sfincs_model.read_forcing()
        self.sfincs_model.setup_waterlevel_forcing(
            locations=locations, timeseries=timeseries, buffer=buffer, offset=offset
        )
        self.sfincs_model.write_forcing()
        self.sfincs_model.write_config()

        if self.write_figures:
            self.sfincs_model.plot_forcing(fn_out="forcing.png")
            self.sfincs_model.plot_basemap(fn_out="basemap.png")

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
        """
        Sets up river inflow boundary conditions for the SFINCS model.

        Here, we use negative indices to indicate inflow boundaries, so that they are
        separate from the runoff generated in the model domain.

        Args:
            nodes: GeoDataFrame containing the locations of river inflow points.
                The index values are converted to negative values to indicate inflow boundaries.
            timeseries: DataFrame containing discharge time series (in m^3/s) for each node.
                The columns should correspond to the node indices; these are also converted to negative values.
        """
        nodes = nodes.copy()
        timeseries = timeseries.copy()
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

        Raises:
            ValueError: If forcing locations are outside the model region.
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

        reprojected_nodes = nodes.to_crs(
            self.sfincs_root_model.sfincs_model.grid["msk"].rio.crs
        )
        x_points: xr.DataArray = xr.DataArray(
            reprojected_nodes.geometry.x,
            dims="points",
        )
        y_points: xr.DataArray = xr.DataArray(
            reprojected_nodes.geometry.y,
            dims="points",
        )

        in_masked_area: xr.DataArray = self.sfincs_root_model.sfincs_model.grid[
            "msk"
        ].sel(
            x=x_points,
            y=y_points,
            method="nearest",
        )
        if not (in_masked_area > 0).all():
            raise ValueError("Some forcing locations are outside the active model area")

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
        region_mask = rasterize_like(
            self.sfincs_root_model.region.to_crs(runoff_m.rio.crs),
            burn_value=1,
            raster=runoff_m.isel(time=0),
            dtype=np.int32,
            nodata=0,
            all_touched=True,
        ).astype(bool)

        original_dimensions = runoff_m.dims
        runoff_m: xr.DataArray = xr.where(region_mask, runoff_m, 0, keep_attrs=True)
        # xr.where changes the dimension order, so we need to transpose it back
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

    def run(self, ncpus: int | str = "auto", gpu: bool | str = "auto") -> None:
        """Runs the SFINCS simulation.

        Args:
            ncpus: Number of CPU cores to use for the simulation. Can be
                an integer or 'auto' to automatically detect available cores.
            gpu: Whether to use GPU acceleration for the simulation. Can be
                True, False, or 'auto' to automatically detect GPU availability.
        """
        assert gpu in [True, False, "auto"], "gpu must be True, False, or 'auto'"
        assert ncpus == "auto" or (isinstance(ncpus, int) and ncpus > 0), (
            "ncpus must be 'auto' or a positive integer"
        )
        run_sfincs_simulation(
            simulation_root=self.path,
            model_root=self.root_path,
            ncpus=ncpus,
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
        if hasattr(flood_depth, "compute"):
            flood_depth = flood_depth.compute()
        return (flood_depth * self.sfincs_root_model.cell_area).sum().item()

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
