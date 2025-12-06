"""This module contains classes and functions to build and run SFINCS models for flood hazard assessment.

The main class is `SFINCSRootModel`, which is used to create and manage SFINCS models.
It provides methods to build the model, set up simulations with different forcing methods,
and read simulation results.

"""

from __future__ import annotations

import logging
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyflwdir
import rasterio
import xarray as xr
from hydromt_sfincs import SfincsModel
from hydromt_sfincs.workflows import burn_river_rect
from pyflwdir import FlwdirRaster
from pyflwdir.dem import fill_depressions
from scipy.ndimage import value_indices
from shapely import line_locate_point
from shapely.geometry import Point
from tqdm import tqdm

from geb.hydrology.routing import get_river_width
from geb.types import (
    ArrayInt64,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
    TwoDArrayInt32,
)
from geb.workflows.io import read_geom, write_geom, write_zarr
from geb.workflows.raster import (
    calculate_cell_area,
    clip_region,
    coord_to_pixel,
    pad_xy,
    rasterize_like,
)

from .workflows import get_river_depth, get_river_manning
from .workflows.outflow import create_outflow_in_mask
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

SFINCS_WATER_LEVEL_BOUNDARY = 2


class SFINCSRootModel:
    """Builds and updates SFINCS model files for flood hazard modeling."""

    def __init__(self, root: Path, name: str) -> None:
        """Initializes the SFINCSRootModel with a GEBModel and event name.

        Sets up the constant parts of the model (grid, mask, rivers, etc.),
        and has methods to create simulations with actual forcing.

        Args:
            files: A dictionary containing file paths for model components.
            root: The root directory for all SFINCS models.
            name: A string representing the name of the event (e.g., "flood_event_2023").
                Also used to create the path to write the file to disk.
        """
        self._name: str = name
        self._root = root

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
        folder: Path = self._root / self.name
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
        self.rivers: gpd.GeoDataFrame = read_geom(self.path / "rivers.geoparquet")
        self.region: gpd.GeoDataFrame = read_geom(self.path / "region.geoparquet")
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
        setup_river_outflow_boundary: bool = True,
        initial_water_level: float | None = 0.0,
        custom_rivers_to_burn: gpd.GeoDataFrame | None = None,
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
            setup_river_outflow_boundary: Whether to set up an outflow boundary condition. Defaults to True. Mostly used for testing purposes.
            initial_water_level: The initial water level to initiate the model. SFINCS fills all cells below this level with water.
            custom_rivers_to_burn: A GeoDataFrame of custom rivers to burn into the model grid. If None, uses the provided rivers GeoDataFrame.
                dataframe must contain 'width' and 'depth' columns.

        Returns:
            The SFINCSRootModel instance with the built model.

        Raises:
            ValueError: if depth_calculation_method is not 'manning' or 'power_law',
            ValueError: if grid_size_multiplier is not a positive integer.
            ValueError: if resolution of DEM is not square pixels.
        """
        self.cleanup()

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

        print("Starting SFINCS model build...")

        # build base model
        sf: SfincsModel = SfincsModel(root=str(self.path), mode="w+", write_gis=True)
        self.sfincs_model = sf

        # Configure the model's logger and all handlers to DEBUG level
        # This must be done AFTER creating SfincsModel when handlers are initialized
        sf.logger.setLevel(logging.DEBUG)
        for handler in sf.logger.handlers:
            handler.setLevel(logging.DEBUG)

        mask_ds = DEMs[0]["elevtn"]
        assert isinstance(mask_ds, xr.Dataset)
        mask: xr.DataArray = mask_ds["elevtn"]

        self.region: gpd.GeoDataFrame = region.to_crs(mask.rio.crs)
        del region

        # in case the first DEM does not fully cover the region, we pad it. Because
        # of the alignment, the padding must increase with the grid_size_multiplier
        minx, miny, maxx, maxy = self.region.total_bounds
        mask: xr.DataArray = pad_xy(
            mask,
            minx=minx - abs(mask.rio.resolution()[0]) * grid_size_multiplier,
            miny=miny - abs(mask.rio.resolution()[1]) * grid_size_multiplier,
            maxx=maxx + abs(mask.rio.resolution()[0]) * grid_size_multiplier,
            maxy=maxy + abs(mask.rio.resolution()[1]) * grid_size_multiplier,
            return_slice=False,
        )

        region_burned: xr.DataArray = rasterize_like(
            gdf=self.region,
            burn_value=1,
            raster=mask,
            dtype=np.int32,
            nodata=0,
            all_touched=True,
        ).astype(bool)
        assert isinstance(mask, xr.DataArray)

        resolution: tuple[float, float] = mask.rio.resolution()
        if abs(abs(resolution[0]) - abs(resolution[1])) > 1e-8:
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

        # Remove rivers that are not represented in the grid and have no upstream rivers
        # TODO: Make an upstream flag in preprocessing for upstream rivers that is more
        # general than the MERIT-hydro specific 'maxup' attribute
        self.rivers: gpd.GeoDataFrame = rivers[
            (rivers["maxup"] > 0) | (rivers["represented_in_grid"])
        ].to_crs(self.crs)
        del rivers

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

        self.plot_rivers()

        if setup_river_outflow_boundary:
            # must be performed BEFORE burning rivers.
            self.setup_river_outflow_boundary()
        else:
            self.rivers["outflow_elevation"] = np.nan
            self.rivers["outflow_point_xy"] = None

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

        if custom_rivers_to_burn is not None:
            rivers_to_burn = custom_rivers_to_burn.to_crs(sf.crs)
            if "width" not in rivers_to_burn.columns:
                raise ValueError(
                    "Custom rivers to burn must have a 'width' column when using custom rivers"
                )
            if "depth" not in rivers_to_burn.columns:
                raise ValueError(
                    "Custom rivers to burn must have a 'depth' column when using custom rivers"
                )
        else:
            rivers_to_burn = assign_return_periods(
                self.rivers, discharge_by_river, return_periods=[2]
            )

            river_width_unknown_mask = rivers_to_burn["width"].isnull()

            rivers_to_burn.loc[river_width_unknown_mask, "width"] = get_river_width(
                river_parameters["river_width_alpha"][river_width_unknown_mask],
                river_parameters["river_width_beta"][river_width_unknown_mask],
                rivers_to_burn.loc[river_width_unknown_mask, "Q_2"],
            ).astype(np.float64)

            rivers_to_burn["depth"] = get_river_depth(
                rivers_to_burn,
                method=depth_calculation_method,
                parameters=depth_calculation_parameters,
                bankfull_column="Q_2",
            )

        rivers_to_burn["manning"] = get_river_manning(rivers_to_burn)

        # Because hydromt-sfincs does a lot of filling default values when data
        # is missing, we need to be extra sure that the required columns are
        # present and contain valid data.
        assert rivers_to_burn["width"].notnull().all(), "River width cannot be null"
        assert rivers_to_burn["depth"].notnull().all(), "River depth cannot be null"
        assert rivers_to_burn["manning"].notnull().all(), (
            "River Manning's n cannot be null"
        )

        # only burn rivers that are wider than the grid size
        rivers_to_burn: gpd.GeoDataFrame = rivers_to_burn[
            rivers_to_burn["width"] > self.estimated_cell_size_m
        ]

        # if sfincs is run with subgrid, we set up the subgrid, with burned in rivers and mannings
        # roughness within the subgrid. If not, we burn the rivers directly into the main grid,
        # including mannings roughness.
        if subgrid:
            print(
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
            print(
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

        self.region.to_parquet(self.path / "region.geoparquet")
        self.rivers.to_parquet(self.path / "rivers.geoparquet")

        sf.plot_basemap(fn_out="basemap.png")

        return self

    def plot_rivers(self) -> None:
        """Plots the rivers and region boundary and saves to file."""
        fig, ax = plt.subplots(figsize=(10, 10))
        self.region.boundary.plot(ax=ax, color="black")

        self.rivers.plot(ax=ax, color="blue")
        plt.savefig(self.path / "gis" / "rivers.png")

    def setup_river_outflow_boundary(
        self,
    ) -> None:
        """Sets up river outflow boundary condition for the SFINCS model.

        Raises:
            ValueError: if the calculated outflow point is not a single point.
            ValueError: if the calculated outflow point is outside of the model grid.
        """

        def export_diagnostics() -> None:
            write_zarr(
                self.mask,
                self.path / "debug_outflow_mask.zarr",
                crs=self.mask.rio.crs,
            )
            self.rivers.to_file(self.path / "debug_rivers.geojson", driver="GeoJSON")
            self.region.to_file(self.path / "debug_region.geojson", driver="GeoJSON")
            write_geom(self.rivers, self.path / "debug_rivers.geoparquet")
            write_geom(self.region, self.path / "debug_region.geoparquet")

        downstream_most_rivers: gpd.GeoDataFrame = self.rivers.loc[
            self.rivers["is_downstream_outflow_subbasin"]
            | (self.rivers["downstream_ID"] == 0)
        ]

        self.rivers["outflow_elevation"] = np.nan
        self.rivers["outflow_point_xy"] = None

        if not downstream_most_rivers.empty:
            for river_idx, river in downstream_most_rivers.iterrows():
                # outflow point is the intersection of the river geometry with the region boundary
                # this will be used as the central point of the outflow boundary condition
                outflow_point = river.geometry.intersection(
                    self.region.union_all().boundary
                )
                if not isinstance(outflow_point, Point):
                    export_diagnostics()
                    raise ValueError(
                        "Calculated outflow point is not a single point. Please check the river geometries and region boundary."
                    )

                outflow_col, outflow_row = coord_to_pixel(
                    (outflow_point.x, outflow_point.y),
                    self.mask.rio.transform().to_gdal(),
                )
                # due to floating point precision, the intersection point
                # may be just outside the model grid. We therefore check if the
                # point is outside the grid, and if so, move it 1 m upstream along the river
                if not self.mask.values[outflow_row, outflow_col]:
                    # move outflow point 1 m upstream. 0.000008983 degrees is approximately 1 m
                    outflow_point = river.geometry.interpolate(
                        line_locate_point(river.geometry, outflow_point) - 0.000008983
                        if self.is_geographic
                        else 1.0
                    )
                    outflow_col, outflow_row = coord_to_pixel(
                        (outflow_point.x, outflow_point.y),
                        self.mask.rio.transform().to_gdal(),
                    )
                    # if still outside the grid, raise error
                    if not self.mask.values[outflow_row, outflow_col]:
                        export_diagnostics()
                        raise ValueError(
                            "Calculated outflow point is outside of the model grid. Please check the river geometries and region boundary."
                        )
                assert outflow_col >= 0 and outflow_row >= 0, (
                    "Calculated outflow point is outside of the model grid"
                )
                outflow_boundary_width_m = 500
                try:
                    outflow_mask: TwoDArrayBool = create_outflow_in_mask(
                        self.mask.values,
                        row=outflow_row,
                        col=outflow_col,
                        width_cells=(
                            math.ceil(
                                (
                                    (
                                        outflow_boundary_width_m
                                        / self.estimated_cell_size_m
                                    )
                                    - 1
                                )
                                / 2
                            )
                            * 2
                            + 1
                        ),
                    )
                except ValueError:
                    export_diagnostics()
                    raise

                outflow_elevation: float = self.elevation[
                    outflow_row, outflow_col
                ].item()
                self.rivers.at[river_idx, "outflow_elevation"] = outflow_elevation
                self.rivers.at[river_idx, "outflow_point_xy"] = (
                    outflow_point.x,
                    outflow_point.y,
                )

                assert self.sfincs_model.grid_type == "regular"
                self.mask.values[outflow_mask] = SFINCS_WATER_LEVEL_BOUNDARY

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

        # also add all cells with large upstream from the DEM to the drainage cells
        upstream_area: TwoDArrayFloat64 = flow_raster.upstream_area(unit="m2")
        drainage_cells: xr.DataArray = drainage_cells | (
            upstream_area > 25_000_000
        )  # 25 km²

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
    def crs(self) -> rasterio.crs.CRS:
        """Returns the coordinate reference system (CRS) of the SFINCS model.

        Returns:
            The CRS of the SFINCS model.
        """
        return self.elevation.rio.crs

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

    @property
    def root(self) -> Path:
        """Gets the root directory for all SFINCS models.

        Returns:
            The path to the SFINCS model root directory.
        """
        return self._root

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
            sfincs_root_model_path=self.root,
            sfincs_root_model_name=self.name,
            **kwargs,
        )

    def create_coastal_simulation(
        self, return_period: int, locations: gpd.GeoDataFrame, offset: xr.DataArray
    ) -> SFINCSSimulation:
        """
        Creates a SFINCS simulation with coastal water level forcing for a specified return period.

        It reads the coastal hydrograph timeseries and locations from pre-defined files, and sets the coastal water level forcing for the simulation.

        Args:
            return_period: The return period for which to create the coastal simulation.
            locations: A GeoDataFrame containing the locations of GTSM forcing stations.
            offset: The offset to apply to the coastal water level forcing based on mean sea level topography.
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

        # convert index to int
        # make a copy to avoid overwriting the original locations
        locations_copy = locations.copy()
        locations_copy.index = locations_copy.index.astype(int)

        timeseries.index = pd.to_datetime(timeseries.index, format="%Y-%m-%d %H:%M:%S")
        # convert columns to int
        timeseries.columns = timeseries.columns.astype(int)

        # Align timeseries columns with locations index
        timeseries = timeseries.loc[:, locations_copy.index]

        # now convert to incrementing integers starting from 0
        timeseries.columns = range(len(timeseries.columns))
        locations_copy.index = range(len(locations_copy.index))

        timeseries = timeseries.iloc[250:-250]  # trim the first and last 250 rows

        simulation: SFINCSSimulation = self.create_simulation(
            simulation_name=f"rp_{return_period}_coastal",
            start_time=timeseries.index[0],
            end_time=timeseries.index[-1],
        )

        # set coastal forcing model
        simulation.set_coastal_waterlevel_forcing(
            timeseries=timeseries, locations=locations_copy, offset=offset
        )
        return simulation

    def create_simulation_for_return_period(
        self,
        return_period: int,
        locations: gpd.GeoDataFrame,
        offset: xr.DataArray,
        coastal: bool = False,
        coastal_only: bool = False,
    ) -> MultipleSFINCSSimulations:
        """Creates multiple SFINCS simulations for a specified return period.

        The method groups rivers by their calculation group and creates a separate
        simulation for each group. Each simulation is configured with discharge
        hydrographs corresponding to the specified return period.

        Args:
            return_period: The return period for which to create simulations.
            locations: A GeoDataFrame containing the locations of GTSM forcing stations.
            offset: The offset to apply to the coastal water level forcing based on mean sea level topography.
            coastal: Whether to create a coastal simulation.
            coastal_only: Whether to only include coastal subbasins in the model.

        Returns:
            An instance of MultipleSFINCSSimulations containing the created simulations.
                This class aims to emulate a single SFINCSSimulation instance as if
                it was one.

        Raises:
            ValueError: If inflow node indices cannot be converted to integers,
                if the discharge DataFrame columns cannot be converted to integers,
                or if the discharge hydrographs contain NaN values.
        """
        working_dir: Path = self.path / "working_dir"
        working_dir_return_period: Path = working_dir / f"rp_{return_period}"

        print(f"Running SFINCS for return period {return_period} years")
        simulations: list[SFINCSSimulation] = []

        # create coastal simulation
        if coastal:
            simulation: SFINCSSimulation = self.create_coastal_simulation(
                return_period, locations, offset
            )
            simulations.append(simulation)
        if coastal_only:
            return MultipleSFINCSSimulations(simulations=simulations)

        rivers: gpd.GeoDataFrame = import_rivers(self.path, postfix="_return_periods")
        assert (~rivers["is_downstream_outflow_subbasin"]).all()

        rivers["topological_stream_order"] = get_topological_stream_order(rivers)
        rivers: gpd.GeoDataFrame = assign_calculation_group(rivers)

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
                df = pd.DataFrame.from_dict(
                    hydro, orient="index", columns=np.array([node_idx])
                )
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
        sfincs_root_model_path: Path,
        sfincs_root_model_name: str,
        simulation_name: str,
        start_time: datetime,
        end_time: datetime,
        spinup_seconds: int = 86400,
        write_figures: bool = True,
        flood_map_output_interval_seconds: int | None = None,
    ) -> None:
        """Initializes a SFINCSSimulation with specific forcing and configuration.

        Args:
            sfincs_root_model_path: Path to the root SFINCS model directory.
            sfincs_root_model_name: Name of the root SFINCS model.
            simulation_name: A string representing the name of the simulation.
                Also used to create the path to write the file to disk.
            start_time: The start time of the simulation as a datetime object.
            end_time: The end time of the simulation as a datetime object.
            spinup_seconds: The number of seconds to use for model spin-up. Defaults to 86400 (1 day).
            write_figures: Whether to generate and save figures for the model. Defaults to False.
            flood_map_output_interval_seconds: The interval in seconds at which to output flood maps. Defaults to None (15 minutes).
                None means no output.
        """
        self._name = simulation_name
        self.write_figures = write_figures
        self.start_time = start_time
        self.end_time = end_time

        # Read a new independent instance of the SFINCS model to avoid shared state
        self.root_model: SFINCSRootModel = SFINCSRootModel(
            root=sfincs_root_model_path, name=sfincs_root_model_name
        ).read()

        self.cleanup()

        sfincs_model = self.root_model.sfincs_model
        sfincs_model.set_root(str(self.path), mode="w+")

        # Configure the model's logger and all handlers to DEBUG level
        # This must be done AFTER reading SfincsModel when handlers are initialized
        sfincs_model.logger.setLevel(logging.DEBUG)
        for handler in sfincs_model.logger.handlers:
            handler.setLevel(logging.DEBUG)

        sfincs_model.setup_config(
            alpha=0.5,  # alpha is the parameter for the CFL-condition reduction. Decrease for additional numerical stability, minimum value is 0.1 and maximum is 0.75 (0.5 default value)
            h73table=1,  # use h^(7/3) table for friction calculation. This is slightly less accurate but up to 30% faster
            nc_deflate_level=9,  # compression level for netcdf output files (0-9)
            tspinup=spinup_seconds,  # spinup time in seconds
            dtout=flood_map_output_interval_seconds
            or 999999999,  # output flood maps every n seconds, 0 means no output
            dtmaxout=999999999,  # only report max flood depth at the end of the simulation
            tref=to_sfincs_datetime(start_time),  # reference time for the simulation
            tstart=to_sfincs_datetime(start_time),  # simulation start time
            tstop=to_sfincs_datetime(end_time),  # simulation end time
            **make_relative_paths(sfincs_model.config, self.root_path, self.path),
        )
        sfincs_model.write_config()

        self.sfincs_model: SfincsModel = sfincs_model

        # Track total volumes added via forcings (for water balance debugging)
        self.total_runoff_volume_m3: float = 0.0
        self.total_discharge_volume_m3: float = 0.0

        self.set_river_outflow_boundary_condition()

    def print_forcing_volume(self) -> None:
        """Print all forcing volumes for debugging the water balance."""
        msg: str = (
            f"SFINCS Forcing volumes: runoff={int(self.total_runoff_volume_m3)} m3, "
            f"discharge={int(self.total_discharge_volume_m3)} m3"
        )
        print(msg)

    def set_coastal_waterlevel_forcing(
        self,
        locations: gpd.GeoDataFrame,
        timeseries: pd.DataFrame,
        buffer: int = 100_000,
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
            locations=locations,
            timeseries=timeseries,
            buffer=buffer,
            offset=offset,  # ty: ignore[invalid-argument-type]
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
        # Load rivers from file and filter for headwater rivers
        headwater_rivers: gpd.GeoDataFrame = self.root_model.rivers[
            self.root_model.rivers["maxup"] == 0
        ]
        self.set_forcing_from_grid(
            nodes=headwater_rivers,
            discharge_grid=discharge_grid,
        )

    def set_river_outflow_boundary_condition(self) -> None:
        """Sets up river outflow boundary condition for the SFINCS model.

        We use the pre-calculated outflow elevations from the root model
        to set constant water level boundary conditions at the river outflow points.
        """
        outflow_points: gpd.GeoDataFrame = self.root_model.rivers[
            ~self.root_model.rivers["outflow_point_xy"].isna()
        ].copy()
        if not outflow_points.empty:
            outflow_points["geometry"] = outflow_points["outflow_point_xy"].apply(
                lambda xy: Point(xy[0], xy[1])
            )
            outflow_points.index = range(1, len(outflow_points) + 1)

            # Create DataFrame with constant water level for each outflow point
            elevation_time_series_constant: pd.DataFrame = pd.DataFrame(
                data=outflow_points["outflow_elevation"].to_dict(),
                index=pd.date_range(start=self.start_time, end=self.end_time, freq="h"),
            )

            self.sfincs_model.set_forcing_1d(
                gdf_locs=outflow_points,
                df_ts=elevation_time_series_constant,
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
        # Replicate the inflow_rivers property logic
        non_headwater_rivers: gpd.GeoDataFrame = self.root_model.rivers[
            self.root_model.rivers["maxup"] > 0
        ]
        non_outflow_basins: gpd.GeoDataFrame = non_headwater_rivers[
            ~non_headwater_rivers["is_downstream_outflow_subbasin"]
        ]
        upstream_branches_in_domain = np.unique(
            self.root_model.rivers["downstream_ID"], return_counts=True
        )

        rivers_with_inflow = []
        for idx, row in non_outflow_basins.iterrows():
            downstream_ID = row["downstream_ID"]
            if downstream_ID not in self.root_model.rivers.index:
                continue
            upstream_branches_count = upstream_branches_in_domain[1][
                upstream_branches_in_domain[0] == downstream_ID
            ][0]
            if upstream_branches_count > 1:
                rivers_with_inflow.append(idx)

        inflow_rivers: gpd.GeoDataFrame = self.root_model.rivers[
            self.root_model.rivers.index.isin(rivers_with_inflow)
        ]
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

        reprojected_nodes = nodes.to_crs(self.sfincs_model.grid["msk"].rio.crs)
        x_points: xr.DataArray = xr.DataArray(
            reprojected_nodes.geometry.x,
            dims="points",
        )
        y_points: xr.DataArray = xr.DataArray(
            reprojected_nodes.geometry.y,
            dims="points",
        )

        in_masked_area: xr.DataArray = self.sfincs_model.grid["msk"].sel(
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
        river_ids: TwoDArrayInt32,
        basin_ids: TwoDArrayInt32,
        upstream_area: TwoDArrayFloat32,
        cell_area: TwoDArrayFloat32,
    ) -> None:
        """Sets up accumulated runoff forcing for the SFINCS model.

        This function accumulates the runoff from the provided runoff grid to the river network
        and sets it as discharge forcing for the SFINCS model. For each river cell in the low-resolution
        runoff grid, the upstream area is determined and the runoff from all cells in that upstream
        area is accumulated to the river cell. The accumulated discharge is then set as forcing
        for the SFINCS model.

        In some cases, the upstream area of the most upstream low-res river point is larger than the
        upstream area of the most upstream high-res river point. This means that the runoff
        would be added further downstream. To avoid this, we check if this is the case, and if so,
        we scale the upstream area of the low-res river points to match the upstream area of the
        most upstream high-res river point.

        Args:
            runoff_m: xarray DataArray containing runoff values in m per time step.
            river_network: FlwdirRaster representing the river network flow directions.
            river_ids: 2D numpy array of river segment IDs for each cell in the grid.
            basin_ids: 2D numpy array of basin IDs for each cell in the grid.
            upstream_area: 2D numpy array of upstream area values for each cell in the grid.
            cell_area: 2D numpy array of cell area values for each cell in the grid.
        """
        INFLOW_MULTIPLICATION_FACTOR = 100000

        # select only the time range needed
        runoff_m: xr.DataArray = runoff_m.sel(
            time=slice(self.start_time, self.end_time)
        )

        # mask out all basins that are not in the model
        mask: TwoDArrayBool = np.isin(basin_ids, self.root_model.rivers.index)
        runoff_m = xr.where(mask, runoff_m, 0.0)  # set runoff to 0 outside model basins
        river_ids = np.where(
            mask, river_ids, -1
        )  # set river IDs to -1 outside model basins

        # first, we want to create unique IDs for each river cell. To allow us to map
        # it back to the original river segment, we create IDs as follows:
        # inflow_ID = river_segment_ID * INFLOW_MULTIPLICATION_FACTOR + offset
        # where offset is a number from 0 to N-1, with N being the number of cells in the river segment
        river_inflow_IDs = np.full_like(river_ids, -1, dtype=np.int64)
        xy_per_river_segment = value_indices(river_ids, ignore_value=-1)
        for ID, (ys, xs) in xy_per_river_segment.items():
            assert len(ys) < INFLOW_MULTIPLICATION_FACTOR - 2, (
                f"River segment has more than {INFLOW_MULTIPLICATION_FACTOR - 2} cells, which is not supported. "
                "Increase the multiplication factor in the inflow ID calculation."
            )

            river_upstream_area = upstream_area[ys, xs]
            up_to_downstream_ids = np.argsort(river_upstream_area)

            ys_up_to_down: npt.NDArray[np.int64] = ys[up_to_downstream_ids]
            xs_up_to_down: npt.NDArray[np.int64] = xs[up_to_downstream_ids]

            for i in range(len(ys_up_to_down)):
                inflow_ID: np.int64 = np.int64(ID) * np.int64(
                    INFLOW_MULTIPLICATION_FACTOR
                ) + np.int64(i)
                assert inflow_ID < 9_223_372_036_854_775_807, (
                    "Inflow ID exceeds maximum int64 value."
                )
                river_inflow_IDs[ys_up_to_down[i], xs_up_to_down[i]] = inflow_ID

        river_cells: TwoDArrayBool = river_inflow_IDs != -1
        river_ids_mapping: ArrayInt64 = river_inflow_IDs[river_cells]

        # starting from each river cell, create an upstream basin map for which
        # the discharge will be accumulated
        # .basins() must have river IDs starting from 1, so we use IDs from 1 to N + 1
        # then, we subtract 1 to make it zero based again
        subbasins: ArrayInt64 = river_network.basins(
            np.where(river_cells.ravel())[0],
            ids=np.arange(1, river_ids_mapping.size + 1, step=1, dtype=np.int64),
        )[mask]
        assert not (subbasins == 0).any()
        subbasins -= 1  # make zero based

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

        # accumulate generated discharge for each river cell
        accumulated_generated_discharge_m3_per_s: TwoDArrayFloat64 = (
            np.apply_along_axis(
                func1d=lambda x: np.bincount(subbasins, weights=x),
                axis=1,
                arr=generated_discharge_m3_per_s.values[:, mask],
            )
        )
        assert (
            accumulated_generated_discharge_m3_per_s.shape[1] == river_ids_mapping.size
        )
        assert accumulated_generated_discharge_m3_per_s.shape[1] == river_cells.sum()

        # create empty timeseries and nodes
        timeseries: pd.DataFrame = pd.DataFrame(
            {
                "time": generated_discharge_m3_per_s.time,
            }
        ).set_index("time")
        inflow_IDs_gdf: list[int] = []
        nodes: list[Point] = []

        for mapped_idx in range(accumulated_generated_discharge_m3_per_s.shape[1]):
            inflow_idx: int = river_ids_mapping[mapped_idx]

            # find the original river segment and offset
            river_ID = inflow_idx // INFLOW_MULTIPLICATION_FACTOR
            inflow_offset = inflow_idx % INFLOW_MULTIPLICATION_FACTOR

            river: gpd.GeoSeries = self.root_model.rivers.loc[river_ID]

            # confirm we have the correct inflow point
            xy_low_res = river["hydrography_xy"][inflow_offset]
            assert river_inflow_IDs[xy_low_res[1], xy_low_res[0]] == inflow_idx

            # find the high-res location corresponding to the low-res inflow point
            hydrography_upstream_area_m2 = river["hydrography_upstream_area_m2"]
            upstream_area_low_res = hydrography_upstream_area_m2[inflow_offset]
            hydrography_high_res_lons_lats = river["hydrography_high_res_lons_lats"]
            hydrography_high_res_upstream_area_m2 = river[
                "hydrography_high_res_upstream_area_m2"
            ]
            closest_upstream_area_index = np.argmin(
                np.abs(hydrography_high_res_upstream_area_m2 - upstream_area_low_res)
            )

            discharge_m3_per_s = accumulated_generated_discharge_m3_per_s[:, mapped_idx]

            # we want the runoff to start at the headwater of the river. However, sometimes due to
            # the low resolution hydrology, it is inherent that sometimes the inflow point is not at the headwater
            # but somewhere downstream. In that case, we need to add an additional inflow point at the headwater
            # and scale the discharge accordingly, based on the upstream area at the headwater and the upstream area.
            # We then place the normal inflow point at the closest high-res point, but with the discharge reduced accordingly.
            if (
                inflow_offset == 0  # check if this is the most upstream low-res point
                and closest_upstream_area_index
                != 0  # check if the closest high-res point is not the high-res headwater point
                and river["maxup"] == 0  # check if this river is a headwater river
            ):
                # create an additional inflow point at the headwater
                lon_headwater, lat_headwater = hydrography_high_res_lons_lats[0]
                nodes.append(Point(lon_headwater, lat_headwater))

                # for the index, we create a unique index based on the river ID. The headwater point
                # will always have offset INFLOW_MULTIPLICATION_FACTOR - 1 which will not
                # collide with any other inflow point
                headwater_idx = (
                    river_ID * INFLOW_MULTIPLICATION_FACTOR
                    + INFLOW_MULTIPLICATION_FACTOR
                    - 1
                )
                inflow_IDs_gdf.append(headwater_idx)

                # find the upstream areas of both the low-res downstream point
                # and the high-res headwater point
                upstream_area_downstream_point_m2 = (
                    hydrography_high_res_upstream_area_m2[closest_upstream_area_index]
                )
                upstream_area_headwater_point_m2 = (
                    hydrography_high_res_upstream_area_m2[0]
                )

                # scale the discharge based on the upstream areas
                headwater_discharge_m3_per_s = (
                    discharge_m3_per_s
                    * upstream_area_headwater_point_m2
                    / upstream_area_downstream_point_m2
                )
                timeseries[headwater_idx] = headwater_discharge_m3_per_s

                # reduce the discharge at the downstream point accordingly
                discharge_m3_per_s = discharge_m3_per_s - headwater_discharge_m3_per_s

            # get the lon lat of the closest high-res point
            lon, lat = hydrography_high_res_lons_lats[closest_upstream_area_index]

            # add the normal inflow point
            nodes.append(Point(lon, lat))
            inflow_IDs_gdf.append(inflow_idx)

            # add the timeseries
            timeseries[inflow_idx] = discharge_m3_per_s

        # create and sort the final nodes and timeseries
        nodes: gpd.GeoDataFrame = gpd.GeoDataFrame(
            index=inflow_IDs_gdf, geometry=nodes, crs=4326
        )

        nodes: gpd.GeoDataFrame = nodes.sort_index()  # ty: ignore[invalid-assignment]
        timeseries: pd.DataFrame = timeseries.sort_index(axis=1)

        self.set_discharge_forcing_from_nodes(
            nodes=nodes,
            timeseries=timeseries,
        )

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
            end_time=self.end_time,
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
            end_time=self.end_time,
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

        # Calculate cell area based on the model's CRS
        if self.sfincs_model.crs.is_geographic:
            cell_area: xr.DataArray = xr.full_like(
                self.sfincs_model.grid["dep"], np.nan, dtype=np.float32
            )
            cell_area.values = calculate_cell_area(
                self.sfincs_model.grid["dep"].rio.transform(),
                shape=self.sfincs_model.grid["dep"].shape,
            )
        else:
            cell_area = xr.full_like(
                self.sfincs_model.grid["dep"],
                (
                    self.sfincs_model.grid["dep"].rio.resolution()[0]
                    * self.sfincs_model.grid["dep"].rio.resolution()[1]
                ),
                dtype=np.float32,
            )
        return (flood_depth * cell_area).sum().item()

    def cleanup(self) -> None:
        """Cleans up the simulation directory by removing temporary files."""
        shutil.rmtree(path=self.path, ignore_errors=True)

    @property
    def root_path(self) -> Path:
        """Returns the root directory for the SFINCS model files."""
        return self.root_model.path

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
