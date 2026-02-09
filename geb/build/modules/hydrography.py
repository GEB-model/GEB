"""Build methods for the hydrography for GEB."""

import os
from datetime import timedelta
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyflwdir
import rasterio.features
import xarray as xr
from pyflwdir import FlwdirRaster
from scipy.ndimage import value_indices
from shapely.geometry import LineString, Point, shape

from geb.build.data_catalog import NewDataCatalog
from geb.build.methods import build_method
from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayInt32,
    TwoDArrayFloat64,
    TwoDArrayInt32,
    TwoDArrayInt64,
)
from geb.hydrology.waterbodies import LAKE, LAKE_CONTROL, RESERVOIR
from geb.workflows.raster import (
    calculate_height_m,
    calculate_width_m,
    rasterize_like,
    snap_to_grid,
)

from .base import BuildModelBase


def calculate_stream_length(
    original_d8_ldd: xr.DataArray,
    upstream_area_m2: xr.DataArray,
    threshold: float = 1_000_000,
) -> xr.DataArray:
    """Calculate the stream length for the high resolution grid.

    Args:
        original_d8_ldd: The high resolution flow direction raster (D8).
        upstream_area_m2: The high resolution upstream area raster (m2).
        threshold: The threshold for defining a stream based on upstream area (m2).

    Returns:
        A DataArray containing the stream length for the high resolution grid (meters).
    """
    is_stream: xr.DataArray = (
        upstream_area_m2 > threshold
    )  # threshold for streams is 1 km^2 upstream area

    cell_width_m = calculate_width_m(
        is_stream.rio.transform(),
        height=is_stream.shape[0],
        width=is_stream.shape[1],
    )
    cell_height_m = calculate_height_m(
        is_stream.rio.transform(),
        height=is_stream.shape[0],
        width=is_stream.shape[1],
    )

    stream_length = xr.DataArray(
        np.full_like(is_stream, np.nan, dtype=np.float32),
        coords=is_stream.coords,
        dims=is_stream.dims,
    )

    # pit -> no channel, so set stream length to 0
    stream_length = xr.where(original_d8_ldd != 0, stream_length, 0)
    # vertical -> set stream length to cell height
    stream_length = xr.where(
        ~((original_d8_ldd == 64) | (original_d8_ldd == 4)),
        stream_length,
        cell_height_m,
    )
    # horizontal -> set stream length to cell width
    stream_length = xr.where(
        ~((original_d8_ldd == 16) | (original_d8_ldd == 1)),
        stream_length,
        cell_width_m,
    )
    # diagonal -> set stream length to cell diagonal
    stream_length = xr.where(
        ~(
            (original_d8_ldd == 128)
            | (original_d8_ldd == 2)
            | (original_d8_ldd == 8)
            | (original_d8_ldd == 32)
        ),
        stream_length,
        np.sqrt(cell_width_m**2 + cell_height_m**2),
    )
    stream_length = xr.where(is_stream | np.isnan(stream_length), stream_length, 0)

    return stream_length


def get_all_upstream_subbasin_ids(
    river_graph: nx.DiGraph, subbasin_ids: list[int]
) -> set[int]:
    """Get all upstream subbasin IDs for the given subbasin IDs.

    Args:
        river_graph: The river graph to use for determining upstream subbasins.
        subbasin_ids: The subbasin IDs to get the upstream subbasins for.

    Returns:
        A set of all upstream subbasin IDs.
    """
    ancenstors = set()
    for subbasin_id in subbasin_ids:
        ancenstors |= nx.ancestors(river_graph, subbasin_id)
    return ancenstors


def get_all_downstream_subbasin_ids(
    river_graph: nx.DiGraph, subbasin_ids: list[int]
) -> set[int]:
    """Get all downstream subbasin IDs for the given subbasin IDs.

    Args:
        river_graph: The river graph to use for determining downstream subbasins.
        subbasin_ids: The subbasin IDs to get the downstream subbasins for.

    Returns:
        A set of all downstream subbasin IDs.
    """
    descendants = set()
    for subbasin_id in subbasin_ids:
        descendants |= nx.descendants(river_graph, subbasin_id)
    return descendants


def get_immediate_downstream_subbasins(
    river_graph: nx.DiGraph, sink_subbasin_ids: list[int]
) -> dict[int, list[int]]:
    """Get the immediately downstream subbasins for the given sink subbasin IDs.

    The downstream subbasins are returned as a dictionary mapping each downstream subbasin ID
    to a list of its upstream sink subbasin IDs. Some downstream subbasins may have multiple
    upstream sink subbasins.

    Args:
        river_graph: The river graph to use for determining downstream subbasins.
        sink_subbasin_ids: The subbasin IDs of the sink subbasins (i.e., the outflow subbasins).

    Returns:
        A dictionary mapping each downstream subbasin ID to a list of its upstream sink subbasin IDs.
    """
    downstream_subbasins: dict[int, list[int]] = {}
    for subbasin_id in sink_subbasin_ids:
        downstream_subbasin = list(river_graph.neighbors(subbasin_id))
        if len(downstream_subbasin) == 0:
            pass
        else:
            assert len(downstream_subbasin) == 1, (
                "A subbasin has more than one downstream subbasin"
            )
            downstream_subbasin = downstream_subbasin[0]
            if downstream_subbasin not in downstream_subbasins:
                downstream_subbasins[downstream_subbasin] = []
            downstream_subbasins[downstream_subbasin].append(subbasin_id)

    return downstream_subbasins


def get_rivers_geometry(
    data_catalog: NewDataCatalog, subbasin_ids: list[int]
) -> gpd.GeoDataFrame:
    """Get the subbasins geometry and other attributes for the given subbasin IDs.

    Args:
        data_catalog: The data catalog to use for accessing the MERIT dataset.
        subbasin_ids: The subbasin IDs to get the geometry for.

    Returns:
        A GeoDataFrame containing the subbasins geometry for the given subbasin IDs.
            The index of the GeoDataFrame is the subbasin ID (COMID).
    """
    rivers = data_catalog.fetch("merit_basins_rivers").read(
        filters=[
            ("COMID", "in", subbasin_ids),
        ],
        columns=[
            "COMID",
            "lengthkm",
            "slope",
            "uparea",
            "maxup",
            "NextDownID",
            "geometry",
        ],
    )
    assert isinstance(rivers, gpd.GeoDataFrame)
    assert len(rivers) == len(subbasin_ids), "Some rivers were not found"

    rivers: gpd.GeoDataFrame = rivers.rename(
        columns={
            "NextDownID": "downstream_ID",
        }
    ).set_index("COMID")
    rivers["uparea_m2"] = rivers["uparea"] * 1e6  # convert from km^2 to m^2
    rivers["is_headwater_catchment"] = rivers["maxup"] == 0
    rivers: gpd.GeoDataFrame = rivers.drop(columns=["uparea"])
    rivers.loc[rivers["downstream_ID"] == 0, "downstream_ID"] = -1

    # reverse the river lines to have the downstream direction
    rivers["geometry"] = rivers["geometry"].apply(
        lambda x: LineString(list(x.coords)[::-1])
    )

    return rivers


def extend_rivers_into_ocean(
    rivers: gpd.GeoDataFrame,
    flow_raster: FlwdirRaster,
) -> gpd.GeoDataFrame:
    """Extend rivers that end just before the ocean into the ocean.

    For some rivers that end up in oceans, the end point is not in the ocean
    but rather stops just before the ocean. Here we extend those rivers to the downstream point
    in the ocean, based on the flow raster downstream index.

    Args:
        rivers: A GeoDataFrame containing the river lines.
        flow_raster: The flow raster to use for determining the downstream point.

    Returns:
        A GeoDataFrame containing the extended river lines.

    Raises:
        ValueError: If the distance between the river end point and the downstream point is too large.
    """
    resolution = abs(flow_raster.transform.a)
    for river_id, river in rivers.iterrows():
        # only select rivers that have no downstream river (i.e., end in ocean)
        if river["downstream_ID"] == -1 and not river["is_further_downstream_outflow"]:
            river_end_point = river.geometry.coords[-1]
            lon, lat = river_end_point

            # retrieve the linear index of the river end point
            cell_index = flow_raster.index(lon, lat)

            # find the linear index of the downstream point
            downstream_index = flow_raster.idxs_ds[cell_index]

            # get the lonlat of the downstream point
            downstream_lon, downstream_lat = flow_raster.xy(downstream_index)

            # check that distance between river end point and downstream point is within one cell
            distance = np.sqrt(
                (downstream_lon - lon) ** 2 + (downstream_lat - lat) ** 2
            )
            if distance > resolution * 1.5:
                raise ValueError(
                    f"River {river_id} end point is too far from downstream point: {distance} m"
                )

            # extend the river line to the downstream point
            rivers.at[river_id, "geometry"] = LineString(
                list(river.geometry.coords) + [(downstream_lon, downstream_lat)]
            )

    return rivers


def create_river_raster_from_river_lines(
    rivers: gpd.GeoDataFrame,
    target: xr.DataArray,
    column: str | None = None,
    index: bool | None = None,
) -> TwoDArrayInt32:
    """Create a river raster from river lines.

    Args:
        rivers: A GeoDataFrame containing the river lines.
        target: The target raster to match the spatial properties.
        column: The column in the GeoDataFrame to use for rasterization values. If None, the index will be used.
        index: If True, the index will be used for rasterization values. If None or False, the column will be used.
               If both column and index are None, the index will be used.

    Returns:
        A 2D numpy array containing the rasterized river lines.

    Raises:
        ValueError: If both column and index are provided.

    """
    if column is None and (index is None or index is True):
        values = rivers.index
    elif column is not None:
        values = rivers[column]
    else:
        raise ValueError(
            "Either column or index must be provided, or both must be None"
        )
    river_raster = rasterio.features.rasterize(
        zip(rivers.geometry, values),
        out_shape=target.shape,
        fill=-1,
        dtype=np.int32,
        transform=target.rio.transform(recalc=True),
        all_touched=False,  # because this is a line, Bresenham's line algorithm is used, which is perfect here :-)
    )
    return river_raster


def get_SWORD_translation_IDs_and_lengths(
    data_catalog: NewDataCatalog, rivers: gpd.GeoDataFrame
) -> tuple[TwoDArrayInt64, TwoDArrayFloat64]:
    """Get the SWORD reach IDs and lengths for each river based on the MERIT basin ID.

    Each river can have multiple SWORD reach IDs, so the output is a 2D array of shape (N, M)
    where N is the number of SWORD reaches per river and M is the number of rivers.

    Args:
        data_catalog: The data catalog to use for accessing the MERIT and SWORD datasets.
        rivers: A GeoDataFrame containing the rivers with their MERIT basin IDs as index.

    Returns:
        A tuple of two 2D numpy arrays:
        - SWORD_reach_IDs: A 2D numpy array of shape (N, M) where N is the number of SWORD reaches per river
          and M is the number of rivers. Each element is the SWORD reach ID for that river.
        - SWORD_reach_lengths: A 2D numpy array of shape (N, M) where N is the number of SWORD reaches per river
            and M is the number of rivers. Each element is the length of the SWORD reach for that river.
    """
    MERIT_Basins_to_SWORD = (
        data_catalog.fetch("merit_sword").read().sel(mb=rivers.index.tolist())
    )
    assert isinstance(MERIT_Basins_to_SWORD, xr.Dataset)

    SWORD_reach_IDs: TwoDArrayInt64 = np.full(
        (40, len(rivers)), dtype=np.int64, fill_value=-1
    )
    SWORD_reach_lengths: TwoDArrayFloat64 = np.full(
        (40, len(rivers)), dtype=np.float64, fill_value=np.nan
    )
    for i in range(1, 41):
        SWORD_reach_IDs[i - 1] = np.where(
            MERIT_Basins_to_SWORD[f"sword_{i}"] != 0,
            MERIT_Basins_to_SWORD[f"sword_{i}"],
            -1,
        )
        SWORD_reach_lengths[i - 1] = MERIT_Basins_to_SWORD[f"part_len_{i}"]

    return SWORD_reach_IDs, SWORD_reach_lengths


def get_SWORD_river_widths(
    data_catalog: NewDataCatalog, SWORD_reach_IDs: TwoDArrayInt64
) -> TwoDArrayFloat64:
    """Get the river widths from the SWORD dataset based on the SWORD reach IDs.

    Each river can have multiple SWORD reach IDs, so the input is a 2D array of shape (N, M).

    Args:
        data_catalog: The data catalog to use for accessing the SWORD dataset.
        SWORD_reach_IDs: A 2D numpy array of shape (N, M) where N is the number of SWORD reaches per river
                         and M is the number of rivers. Each element is the SWORD reach ID for that river.

    Returns:
        A 2D numpy array of shape (N, M) where N is the number of SWORD reaches per river
        and M is the number of rivers. Each element is the river width for that SWORD reach ID.
        If a SWORD reach ID is -1, the corresponding river width will be NaN.
    """
    unique_SWORD_reach_ids = np.unique(SWORD_reach_IDs[SWORD_reach_IDs != -1])

    SWORD = data_catalog.fetch("sword").read(
        sql=f"""SELECT * FROM sword WHERE reach_id IN ({",".join([str(ID) for ID in unique_SWORD_reach_ids])})"""
    )
    assert isinstance(SWORD, pd.DataFrame)
    SWORD = SWORD.set_index("reach_id")

    assert len(SWORD) == len(unique_SWORD_reach_ids), (
        "Some SWORD reaches were not found, possibly the SWORD and MERIT data version are not correct"
    )

    def lookup_river_width(reach_id: int) -> float:
        """Lookup the river width in SWORD dataset for a given SWORD reach ID.

        Args:
            reach_id: The SWORD reach ID.

        Returns:
            The river width for the given SWORD reach ID. If the reach ID is -1, returns NaN.
        """
        if reach_id == -1:  # when no SWORD reach is found
            return np.nan  # return NaN
        return SWORD.loc[reach_id, "width"]

    SWORD_river_width = np.vectorize(lookup_river_width)(SWORD_reach_IDs)
    return SWORD_river_width


class Hydrography(BuildModelBase):
    """Contains all build methods for the hydrography for GEB."""

    def __init__(self) -> None:
        """Initializes the Hydrography class."""
        pass

    @build_method(depends_on=["setup_hydrography", "setup_cell_area"], required=True)
    def setup_mannings(self) -> None:
        """Sets up the Manning's coefficient for the model.

        Notes:
            This method sets up the Manning's coefficient for the model by calculating the coefficient based on the cell area
            and topography of the grid. It first calculates the upstream area of each cell in the grid using the
            `routing/upstream_area` attribute of the grid. It then calculates the coefficient using the formula:

                C = 0.025 + 0.015 * (2 * A / U) + 0.030 * (Z / 2000)

            where C is the Manning's coefficient, A is the cell area, U is the upstream area, and Z is the elevation of the cell.

            The resulting Manning's coefficient is then set as the `routing/mannings` attribute of the grid using the
            `set_grid()` method.
        """
        a: xr.DataArray = (2 * self.grid["cell_area"]) / self.grid[
            "routing/upstream_area_m2"
        ]
        a: xr.DataArray = xr.where(a < 1, a, 1, keep_attrs=True)
        b: xr.DataArray = self.grid["landsurface/elevation_min_m"] / 2000
        b: xr.DataArray = xr.where(b < 1, b, 1, keep_attrs=True)

        mannings: xr.DataArray = 0.025 + 0.015 * a + 0.030 * b
        mannings.attrs["_FillValue"] = np.nan

        self.set_grid(mannings, "routing/mannings")

    def get_rivers(
        self, river_graph: nx.DiGraph, sink_subbasin_ids: list[int]
    ) -> gpd.GeoDataFrame:
        """Sets the routing subbasins for the model.

        For each sink subbasin, all upstream subbasins and the immediately downstream subbasins are included.
        The downstream subbasins have a flag set to indicate that they are downstream outflow subbasins.

        If a sink subbasin has no downstream subbasin, it is assumed to be a coastal basin. This is
        also saved in the subbasins attribute.

        Args:
            river_graph: The river graph to use for determining upstream and downstream subbasins.
            sink_subbasin_ids: The subbasin IDs of the sink subbasins (i.e., the outflow subbasins).

        Returns:
            A GeoDataFrame containing the subbasins geometry for the model.
        """
        # always make a list of the subbasin ids, such that the function always gets the same type of input
        if not isinstance(sink_subbasin_ids, (list, set)):
            sink_subbasin_ids = [sink_subbasin_ids]

        subbasin_ids: set[int] = get_all_upstream_subbasin_ids(
            river_graph=river_graph, subbasin_ids=sink_subbasin_ids
        )
        subbasin_ids.update(sink_subbasin_ids)

        downstream_subbasins = get_immediate_downstream_subbasins(
            river_graph, sink_subbasin_ids
        )
        subbasin_ids.update(downstream_subbasins)

        is_further_downstream_outflow: set[int] = set()

        is_further_downstream_outflow.update(
            get_all_downstream_subbasin_ids(
                river_graph, list(downstream_subbasins.keys())
            )
        )

        subbasin_ids.update(is_further_downstream_outflow)

        # later we want to include the downstream outflow basins. However, we don't want to include
        # other branches that are upstream of those downstream basins, but are not part
        # of the area that we are interested in. Therefore, we also include
        # the immediately upstream basins of the downstream basins, so that we can stop the subbasin
        # construction there.
        upstream_basins_of_downstream_basins = set()
        for downstream_subbasin in downstream_subbasins.keys():
            for upstream_basin in river_graph.predecessors(downstream_subbasin):
                if upstream_basin not in subbasin_ids:
                    upstream_basins_of_downstream_basins.add(upstream_basin)

        subbasin_ids.update(upstream_basins_of_downstream_basins)

        rivers: gpd.GeoDataFrame = get_rivers_geometry(
            self.data_catalog, list(subbasin_ids)
        )

        rivers["is_downstream_outflow"] = pd.Series(
            True, index=downstream_subbasins
        ).reindex(rivers.index, fill_value=False)

        rivers["is_upstream_of_downstream_basin"] = pd.Series(
            True, index=upstream_basins_of_downstream_basins
        ).reindex(rivers.index, fill_value=False)

        rivers["is_further_downstream_outflow"] = pd.Series(
            True, index=is_further_downstream_outflow
        ).reindex(rivers.index, fill_value=False)

        return rivers

    @build_method(required=True)
    def setup_hydrography(
        self,
        custom_rivers: str | None = None,
        custom_rivers_width_m_column: str | None = None,
        custom_rivers_depth_m_column: str | None = None,
    ) -> None:
        """Sets up the hydrography for the model.

        Steps:
            1. Resample the original D8 elevation and flow direction to the model resolution.
            2. Calculate the flow direction (LDD) and upstream area based on the resampled elevation and flow direction.
            3. Calculate the slope based on the resampled elevation.
            4. Rasterize the river network, and setting up variables to link
                the low-resolution and high resolution grids.
            5. Calculate river attributes such as river length, river slope, and river width.

        Args:
            custom_rivers: Optional path to a custom river shapefile. The MERIT rivers will still be
                used to create the river raster, but the burning of rivers in the DEM will be done
                based on the custom river shapefile. Must be readable by geopandas.read_file() or geopandas.read_parquet().
            custom_rivers_width_m_column: The column name in the custom rivers file that contains the river width in meters.
                must be provided if custom_rivers is provided.
            custom_rivers_depth_m_column: The column name in the custom rivers file that contains the river depth in meters.
                Must be provided if custom_rivers is provided.

        Raises:
            FileNotFoundError: If the custom rivers file is not found.
            ValueError: If custom_rivers_width_m_column or custom_rivers_depth_m_column is not provided when using custom_rivers.
            KeyError: If custom_rivers_width_m_column or custom_rivers_depth_m_column is not found in the custom rivers file.
            AssertionError: If some large rivers are not represented in the grid.
        """
        if custom_rivers is not None:
            custom_rivers: Path = Path(custom_rivers)
            self.logger.info(f"Using custom rivers from {custom_rivers}")
            if not custom_rivers.exists():
                raise FileNotFoundError(f"Custom rivers file {custom_rivers} not found")
            if custom_rivers.suffix in [".parquet", ".pq", ".geoparquet"]:
                custom_rivers_gdf = gpd.read_parquet(custom_rivers)
            else:
                custom_rivers_gdf = gpd.read_file(custom_rivers)
            custom_rivers_gdf = custom_rivers_gdf.to_crs("EPSG:4326")
            if (
                custom_rivers_width_m_column is None
                or custom_rivers_depth_m_column is None
            ):
                raise ValueError(
                    "custom_rivers_width_m_column and custom_rivers_depth_m_column must be provided when using custom_rivers"
                )
            if custom_rivers_width_m_column not in custom_rivers_gdf.columns:
                raise KeyError(
                    f"custom_rivers_width_m_column '{custom_rivers_width_m_column}' not found in custom rivers file"
                )
            if custom_rivers_depth_m_column not in custom_rivers_gdf.columns:
                raise KeyError(
                    f"custom_rivers_depth_m_column '{custom_rivers_depth_m_column}' not found in custom rivers file"
                )
            custom_rivers_gdf = custom_rivers_gdf.rename(
                columns={
                    custom_rivers_width_m_column: "width",
                    custom_rivers_depth_m_column: "depth",
                }
            )
            self.set_geom(custom_rivers_gdf, name="routing/custom_rivers")

        original_d8_elevation = self.other["drainage/original_d8_elevation"]
        original_d8_ldd = self.other["drainage/original_d8_flow_directions"].compute()
        original_d8_ldd_data = original_d8_ldd.values

        flow_raster_original = pyflwdir.from_array(
            original_d8_ldd_data,
            ftype="d8",
            transform=original_d8_ldd.rio.transform(recalc=True),
            latlon=True,  # hydrography is specified in latlon
            mask=original_d8_ldd_data
            != original_d8_ldd.attrs[
                "_FillValue"
            ],  # this mask is True within study area
        )

        upstream_area_high_res = self.full_like(
            original_d8_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        upstream_area_high_res_data = flow_raster_original.upstream_area(
            unit="m2"
        ).astype(np.float32)
        upstream_area_high_res_data[upstream_area_high_res_data == -9999.0] = np.nan
        upstream_area_high_res.data = upstream_area_high_res_data
        self.set_other(
            upstream_area_high_res, name="drainage/original_d8_upstream_area_m2"
        )

        streams_length_high_res = calculate_stream_length(
            original_d8_ldd, upstream_area_high_res, threshold=1_000_000
        )

        streams_length_low_res = streams_length_high_res.coarsen(
            x=self.ldd_scale_factor,  # ty:ignore[invalid-argument-type]
            y=self.ldd_scale_factor,  # ty:ignore[invalid-argument-type]
            boundary="exact",
            coord_func="mean",
        ).sum()  # ty:ignore[unresolved-attribute]
        streams_length_low_res = snap_to_grid(streams_length_low_res, self.grid["mask"])

        drainage_density = streams_length_low_res / self.grid["cell_area"]

        hillslope_length = 1 / (2 * drainage_density)
        hillslope_length = xr.where(
            hillslope_length < 1000, hillslope_length, 1000
        )  # cap hill slope length at 1000 m
        self.set_grid(hillslope_length, name="drainage/hillslope_length_m")

        elevation_coarsened = original_d8_elevation.coarsen(
            x=self.ldd_scale_factor,
            y=self.ldd_scale_factor,
            boundary="exact",
            coord_func="mean",
        )

        elevation = elevation_coarsened.mean()
        elevation = snap_to_grid(elevation, self.grid["mask"])
        self.set_grid(elevation, name="landsurface/elevation_m")

        elevation_min = elevation_coarsened.min()
        elevation_min = snap_to_grid(elevation_min, self.grid["mask"])
        self.set_grid(elevation_min, name="landsurface/elevation_min_m")

        slope = self.full_like(
            elevation_min, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        slope_data = pyflwdir.dem.slope(
            elevation.values,
            nodata=np.nan,
            latlon=True,
            transform=elevation.rio.transform(recalc=True),
        )
        # set slope to zero on the mask boundary
        slope_data[np.isnan(slope_data) & (~self.grid["mask"].data)] = 0
        slope.data = slope_data
        self.set_grid(slope, name="landsurface/slope_m_per_m")

        flow_raster_idxs_ds = self.grid["flow_raster_idxs_ds"].compute()
        flow_raster = FlwdirRaster(
            flow_raster_idxs_ds.values.ravel(),
            shape=flow_raster_idxs_ds.shape,
            transform=flow_raster_idxs_ds.rio.transform(recalc=True),
            ftype="d8",
            latlon=True,
        )

        # flow direction
        ldd: xr.DataArray = self.full_like(
            elevation_min, fill_value=255, nodata=255, dtype=np.uint8
        )
        ldd.data = flow_raster.to_array(ftype="ldd")
        self.set_grid(ldd, name="routing/ldd")

        # upstream area
        upstream_area: xr.DataArray = self.full_like(
            elevation_min, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        upstream_area_data: npt.NDArray[np.float32] = flow_raster.upstream_area(
            unit="m2"
        ).astype(np.float32)
        upstream_area_data[upstream_area_data == -9999.0] = np.nan
        upstream_area.data = upstream_area_data
        self.set_grid(upstream_area, name="routing/upstream_area_m2")

        upstream_area_n_cells: xr.DataArray = self.full_like(
            elevation_min, fill_value=-1, nodata=-1, dtype=np.int32
        )
        upstream_area_n_cells_data: npt.NDArray[np.int32] = flow_raster.upstream_area(
            unit="cell"
        ).astype(np.int32)
        upstream_area_n_cells.data = upstream_area_n_cells_data
        self.set_grid(upstream_area_n_cells, name="routing/upstream_area_n_cells")

        # river length
        river_length: xr.DataArray = self.full_like(
            elevation_min, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_length_data: npt.NDArray[np.float32] = (
            flow_raster_original.subgrid_rivlen(
                self.grid["idxs_outflow"].values, unit="m", direction="down"
            )
        )
        river_length_data[river_length_data == -9999.0] = np.nan
        river_length.data = river_length_data
        self.set_grid(river_length, name="routing/river_length_m")

        # river slope
        river_slope: xr.DataArray = self.full_like(
            elevation_min, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_slope_data: npt.NDArray[np.float32] = flow_raster_original.subgrid_rivslp(
            self.grid["idxs_outflow"].values, original_d8_elevation
        )
        river_slope_data[river_slope_data == -9999.0] = np.nan
        river_slope.data = river_slope_data
        self.set_grid(
            river_slope,
            name="routing/river_slope_m_per_m",
        )

        self.logger.info("Retrieving river data")
        rivers: gpd.GeoDataFrame = self.geom["routing/rivers"]

        self.logger.info("Processing river data")

        river_raster_HD: npt.NDArray[np.int32] = create_river_raster_from_river_lines(
            rivers, original_d8_elevation
        )
        river_raster_LR: npt.NDArray[np.int32] = river_raster_HD.ravel()[
            self.grid["idxs_outflow"].values.ravel()
        ].reshape(self.grid["idxs_outflow"].shape)

        missing_rivers: set[int] = set(rivers.index) - set(
            np.unique(river_raster_LR[river_raster_LR != -1]).tolist()
        )

        rivers["represented_in_grid"] = True
        rivers.iloc[
            rivers.index.isin(missing_rivers),
            rivers.columns.get_loc("represented_in_grid"),
        ] = False

        assert (
            rivers[
                (~rivers["represented_in_grid"])
                & (~rivers["is_downstream_outflow"])
                & (~rivers["is_further_downstream_outflow"])
            ]["lengthkm"]
            < 5
        ).all(), (
            "Some large rivers are not represented in the grid, please check the "
            "rasterization of the river lines"
        )

        # Derive the xy coordinates of the river network. Here the coordinates
        # are the PIXEL coordinates for the coarse drainage network.
        rivers["hydrography_xy"] = [[] for _ in range(len(rivers))]
        rivers["hydrography_upstream_area_m2"] = [[] for _ in range(len(rivers))]
        xy_per_river_segment = value_indices(river_raster_LR, ignore_value=-1)
        for COMID, (ys, xs) in xy_per_river_segment.items():
            upstream_area = upstream_area_data[ys, xs]
            up_to_downstream_ids = np.argsort(upstream_area)
            upstream_area_sorted = upstream_area[up_to_downstream_ids]

            assert (river_raster_LR[ys, xs] == COMID).all(), (
                f"River segment {COMID} has inconsistent raster values"
            )

            ys: npt.NDArray[np.int64] = ys[up_to_downstream_ids]
            xs: npt.NDArray[np.int64] = xs[up_to_downstream_ids]
            assert ys.size > 0, "No xy coordinates found for river segment"
            rivers.at[COMID, "hydrography_xy"] = list(zip(xs, ys, strict=True))
            rivers.at[COMID, "hydrography_upstream_area_m2"] = (
                upstream_area_sorted.tolist()
            )

        # Derive the xy coordinates of the river network. Here the coordinates
        # are the PIXEL coordinates for the high-resolution drainage network.
        rivers["hydrography_high_res_lons_lats"] = [[] for _ in range(len(rivers))]
        rivers["hydrography_high_res_upstream_area_m2"] = [
            [] for _ in range(len(rivers))
        ]
        xy_per_river_segment = value_indices(river_raster_HD, ignore_value=-1)

        for river_ID, river in rivers.iterrows():
            if river_ID not in xy_per_river_segment:
                if river["is_downstream_outflow"] or not river["represented_in_grid"]:
                    continue
                else:
                    raise AssertionError("River xy not found, but should be found.")

            (ys, xs) = xy_per_river_segment[river_ID]
            upstream_area: ArrayFloat32 = upstream_area_high_res_data[ys, xs]
            nan_mask: ArrayBool = np.isnan(upstream_area)

            if nan_mask.all():
                if river["is_downstream_outflow"] or not river["represented_in_grid"]:
                    continue
                else:
                    raise AssertionError(
                        "River xy all NaN upstream area, but should have valid values."
                    )

            non_nan_mask: ArrayBool = ~nan_mask
            upstream_area = upstream_area[non_nan_mask]
            ys: ArrayInt32 = ys[non_nan_mask]
            xs: ArrayInt32 = xs[non_nan_mask]

            assert not np.isnan(upstream_area).any()

            up_to_downstream_ids = np.argsort(upstream_area)
            upstream_area_sorted = upstream_area[up_to_downstream_ids]

            assert (river_raster_HD[ys, xs] == river_ID).all(), (
                f"River segment {river_ID} has inconsistent raster values"
            )

            ys: npt.NDArray[np.int64] = ys[up_to_downstream_ids]
            xs: npt.NDArray[np.int64] = xs[up_to_downstream_ids]

            lats: ArrayFloat32 = upstream_area_high_res.y.values[ys]
            lons: ArrayFloat32 = upstream_area_high_res.x.values[xs]

            assert ys.size > 0, "No xy coordinates found for river segment"
            rivers.at[river_ID, "hydrography_high_res_lons_lats"] = list(
                zip(lons.tolist(), lats.tolist(), strict=True)
            )
            rivers.at[river_ID, "hydrography_high_res_upstream_area_m2"] = (
                upstream_area_sorted.tolist()
            )

        for river_ID, river in rivers.iterrows():
            if river["represented_in_grid"]:
                assert len(river["hydrography_xy"]) > 0, (
                    f"River {river_ID} has no xy coordinates, please check the "
                    "rasterization of the river lines"
                )

        COMID_IDs_raster: xr.DataArray = self.full_like(
            elevation_min, fill_value=-1, nodata=-1, dtype=np.int32
        )
        COMID_IDs_raster.data = river_raster_LR
        self.set_grid(COMID_IDs_raster, name="routing/river_ids")

        basin_ids = self.full_like(
            elevation_min, fill_value=-1, nodata=-1, dtype=np.int32
        )

        river_linear_indices = np.where(COMID_IDs_raster.values.ravel() != -1)[0]
        ids = COMID_IDs_raster.values.ravel()[river_linear_indices]
        assert 0 not in ids, (
            "COMID ID of 0 found in river raster, which would lead to errors in pyflwdir, which uses 0 as nodata value."
        )
        basin_ids.values = flow_raster.basins(
            idxs=river_linear_indices,
            ids=COMID_IDs_raster.values.ravel()[river_linear_indices],
        )
        basin_ids = xr.where(basin_ids != 0, basin_ids, -1)
        assert (
            np.unique(basin_ids.values[basin_ids.values != -1])
            == np.unique(COMID_IDs_raster.values[COMID_IDs_raster.values != -1])
        ).all()
        self.set_grid(basin_ids, name="routing/basin_ids")

        SWORD_reach_IDs, SWORD_reach_lengths = get_SWORD_translation_IDs_and_lengths(
            self.data_catalog, rivers
        )

        SWORD_river_widths: npt.NDArray[np.float64] = get_SWORD_river_widths(
            self.data_catalog, SWORD_reach_IDs
        )

        rivers["width"] = np.nansum(
            SWORD_river_widths * SWORD_reach_lengths, axis=0
        ) / np.nansum(SWORD_reach_lengths, axis=0)
        # ensure that all rivers with a SWORD ID have a width
        assert (~np.isnan(rivers["width"][(SWORD_reach_IDs != -1).any(axis=0)])).all()

        self.set_geom(rivers, name="routing/rivers")

        river_with_mapper: dict[int, float] = rivers["width"].to_dict()
        river_width_data: npt.NDArray[np.float32] = np.vectorize(
            lambda ID: river_with_mapper.get(ID, np.nan)
        )(COMID_IDs_raster.values).astype(np.float32)

        river_width: xr.DataArray = self.full_like(
            elevation_min, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_width.data = river_width_data
        self.set_grid(river_width, name="routing/river_width_m")

    @build_method(required=True)
    def setup_global_ocean_mean_dynamic_topography(self) -> None:
        """Sets up the global ocean mean dynamic topography for the model."""
        if not self.geom["routing/subbasins"]["is_coastal"].any():
            self.logger.info(
                "No coastal basins found, skipping setup_global_ocean_mean_dynamic_topography"
            )
            return

        global_ocean_mdt_fn = self.old_data_catalog.get_source(
            "global_ocean_mean_dynamic_topography"
        ).path

        global_ocean_mdt = xr.open_dataset(global_ocean_mdt_fn)

        # get the model bounds and buffer by ~10km
        model_bounds = self.bounds
        model_bounds = (
            model_bounds[0] - 0.083,  # min_lon
            model_bounds[1] - 0.083,  # min_lat
            model_bounds[2] + 0.083,  # max_lon
            model_bounds[3] + 0.083,  # max_lat
        )
        min_lon, min_lat, max_lon, max_lat = model_bounds

        # reproject global_ocean_mdt to 0.008333 grid (~1km)
        global_ocean_mdt = global_ocean_mdt["mdt"]
        global_ocean_mdt = global_ocean_mdt.rio.write_crs("EPSG:4326")

        # clip to model bounds
        global_ocean_mdt = global_ocean_mdt.rio.clip_box(
            minx=min_lon,
            miny=min_lat,
            maxx=max_lon,
            maxy=max_lat,
        )

        # write crs
        global_ocean_mdt = global_ocean_mdt.rio.write_crs("EPSG:4326")
        # drop unused columns
        global_ocean_mdt = global_ocean_mdt.squeeze(drop=True)
        # set datatype to float32 and set fillvalue to np.nan
        global_ocean_mdt = global_ocean_mdt.astype(np.float32)
        global_ocean_mdt.encoding["_FillValue"] = np.nan
        global_ocean_mdt.attrs["_FillValue"] = np.nan
        # write to model
        self.set_other(
            global_ocean_mdt,
            name="coastal/global_ocean_mean_dynamic_topography",
        )

    def create_low_elevation_coastal_zone_mask(self) -> gpd.GeoDataFrame | None:
        """creates the low elevation coastal zone (LECZ) mask for sfincs models.

        Returns:
            A GeoDataFrame containing the low elevation coastal zone mask.
        """
        if not self.geom["routing/subbasins"]["is_coastal"].any():
            self.logger.info(
                "No coastal basins found, skipping setup_low_elevation_coastal_zone_mask"
            )
            return

        # load low elevation coastal zone mask
        low_elevation_coastal_zone = self.other[
            "landsurface/low_elevation_coastal_zone"
        ]
        mask_data = low_elevation_coastal_zone.values.astype(np.uint8)

        # Get transform from raster metadata
        transform = low_elevation_coastal_zone.rio.transform()

        # Use rasterio.features.shapes() to get polygons for each contiguous region with same value
        shapes = rasterio.features.shapes(mask_data, mask=None, transform=transform)

        # Build GeoDataFrame from the shapes generator
        records = [{"geometry": shape(geom), "value": value} for geom, value in shapes]

        gdf = gpd.GeoDataFrame.from_records(records)
        gdf.set_geometry("geometry", inplace=True)
        gdf.crs = low_elevation_coastal_zone.rio.crs
        # Keep only mask == 1
        gdf = gdf[gdf["value"] == 1]

        # load mask to select coastal areas in model region
        # intersect the mask with the low elevation coastal zone mask
        low_elevation_coastal_zone_mask = gpd.overlay(
            gdf, self.geom["mask"], how="intersection"
        )
        # merge all polygons into a single polygon
        low_elevation_coastal_zone_mask = gpd.GeoDataFrame(
            geometry=[low_elevation_coastal_zone_mask.union_all()],
            crs=gdf.crs,
        )
        return low_elevation_coastal_zone_mask

    @build_method(required=True)
    def setup_coastlines(self) -> None:
        """Sets up the coastlines for the model."""
        if not self.geom["routing/subbasins"]["is_coastal"].any():
            self.logger.info("No coastal basins found, skipping setup_coastlines")
            return

        # load the coastline from the data catalog
        coastlines = self.data_catalog.fetch("open_street_map_coastlines").read()

        # clip the coastline to overlapping with mask
        coastlines = gpd.overlay(coastlines, self.geom["mask"], how="intersection")
        # merge all coastlines into a single linestring
        coastlines = gpd.GeoDataFrame(
            geometry=[coastlines.union_all()], crs=coastlines.crs
        )

        # write to model files
        self.set_geom(coastlines, name="coastal/coastlines")

        # create rectangular box around coastlines
        if not coastlines.empty:
            bbox = coastlines.minimum_rotated_rectangle().iloc[0]  # get the Polygon
            bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=coastlines.crs)
            bbox_gdf.geometry = bbox_gdf.geometry.buffer(
                0.04, join_style=2
            )  # buffer by 0.04 degree
            self.set_geom(bbox_gdf, name="coastal/coastline_bbox")

    @build_method(required=True)
    def setup_osm_land_polygons(
        self,
    ) -> None:
        """Sets up the OSM land polygons for the model."""
        if not self.geom["routing/subbasins"]["is_coastal"].any():
            self.logger.info(
                "No coastal basins found, skipping setup_osm_land_polygons"
            )
            return

        # load the land polygon from the data catalog
        land_polygons = self.data_catalog.fetch("open_street_map_land_polygons").read()
        # select only the land polygons that intersect with the region
        land_polygons = land_polygons[land_polygons.intersects(self.region.union_all())]
        # merge all land polygons into a single polygon
        land_polygons = gpd.GeoDataFrame(
            geometry=[land_polygons.union_all()], crs=land_polygons.crs
        )

        # clip and write to model files
        self.set_geom(land_polygons.clip(self.bounds), name="coastal/land_polygons")

    @build_method(depends_on=["setup_coastlines"], required=True)
    def setup_coastal_sfincs_model_regions(self) -> None:
        """Sets up the coastal sfincs model regions."""
        if not self.geom["routing/subbasins"]["is_coastal"].any():
            self.logger.info(
                "No coastal basins found, skipping setup_coastal_sfincs_model_regions"
            )
            return

        # load elevation data
        elevation = self.other["DEM/fabdem"]
        # load the lecz mask
        low_elevation_coastal_zone_mask = self.create_low_elevation_coastal_zone_mask()

        # add small buffer to ensure connection of 'islands' with coastlines
        low_elevation_coastal_zone_mask.geometry = (
            low_elevation_coastal_zone_mask.geometry.buffer(0.001)
        )

        # sample the minimum elevation present in the lecz mask
        mask = elevation.rio.clip(
            low_elevation_coastal_zone_mask.geometry,
            low_elevation_coastal_zone_mask.crs,
            all_touched=True,
            drop=False,
        )

        initial_water_levels = float(np.nanmin(mask.values))

        low_elevation_coastal_zone_mask["initial_water_level"] = initial_water_levels
        self.set_geom(
            low_elevation_coastal_zone_mask,
            name="coastal/low_elevation_coastal_zone_mask",
        )

    @build_method(required=True)
    def setup_waterbodies(
        self,
        command_areas: None | str = None,
        custom_reservoir_capacity: None | str = None,
    ) -> None:
        """Sets up the waterbodies for GEB.

        Args:
            command_areas: The path to the command areas data in the data catalog. If None, command areas are not set up.
            custom_reservoir_capacity: The path to the custom reservoir capacity data in the data catalog.
                If None, the default reservoir capacity is used. The data should be a DataFrame with
                'waterbody_id' as the index and 'volume_total' as the column for the reservoir capacity.

        Notes:
            This method sets up the waterbodies for GEB. It first retrieves the waterbody data from the
            specified data catalog and sets it as a geometry in the model. It then rasterizes the waterbody data onto the model
            grid and the subgrid using the `rasterize` method of the `raster` object. The resulting grids are set as attributes
            of the model with names of the form 'waterbodies/{grid_name}'.

            The method also retrieves the reservoir command area data from the data catalog and calculates the area of each
            command area that falls within the model region. The `waterbody_id` key is used to do the matching between these
            databases. The relative area of each command area within the model region is calculated and set as a column in
            the waterbody data. The method sets all lakes with a command area to be reservoirs and updates the waterbody data
            with any custom reservoir capacity data from the data catalog.

        Raises:
            ValueError: If the custom_reservoir_capacity file is not a .csv or .xlsx file.
        """
        waterbodies: gpd.GeoDataFrame = self.data_catalog.fetch("hydrolakes").read(
            bbox=self.bounds,
            columns=[
                "waterbody_id",
                "waterbody_type",
                "volume_total",
                "average_discharge",
                "average_area",
                "geometry",
            ],
        )
        # only select waterbodies that intersect with the region
        waterbodies: gpd.GeoDataFrame = waterbodies[
            waterbodies.intersects(self.region.union_all())
        ]

        hydrolakes_to_geb: dict[int, np.int32] = {
            1: np.int32(LAKE),
            2: np.int32(RESERVOIR),
            3: np.int32(LAKE_CONTROL),
        }
        assert set(waterbodies["waterbody_type"]).issubset(hydrolakes_to_geb.keys())
        waterbodies["waterbody_type"] = waterbodies["waterbody_type"].map(
            hydrolakes_to_geb
        )
        assert waterbodies["waterbody_type"].dtype == np.int32

        waterbody_id: xr.DataArray = rasterize_like(
            gdf=waterbodies,
            column="waterbody_id",
            raster=self.grid["mask"],
            nodata=-1,
            dtype=np.int32,
            all_touched=True,
        )

        sub_waterbody_id: xr.DataArray = rasterize_like(
            gdf=waterbodies,
            column="waterbody_id",
            raster=self.subgrid["mask"],
            nodata=-1,
            dtype=np.int32,
            all_touched=True,
        )

        self.set_grid(waterbody_id, name="waterbodies/waterbody_id")
        self.set_subgrid(sub_waterbody_id, name="waterbodies/sub_waterbody_id")

        waterbodies["volume_flood"] = waterbodies["volume_total"]

        if command_areas:
            command_areas: gpd.GeoDataFrame = gpd.read_file(
                command_areas, mask=self.region
            )
            assert isinstance(command_areas, gpd.GeoDataFrame)
            command_areas = command_areas[
                ~command_areas["waterbody_id"].isnull()
            ].reset_index(drop=True)
            command_areas["waterbody_id"] = command_areas["waterbody_id"].astype(
                np.int32
            )

            # Dissolve command areas with same reservoir
            command_areas = command_areas.dissolve(by="waterbody_id", as_index=False)

            # Set lakes with command area to reservoirs and reservoirs without command area to lakes
            ids_with_command: set = set(command_areas["waterbody_id"])
            waterbodies.loc[
                waterbodies["waterbody_id"].isin(ids_with_command),
                "waterbody_type",
            ] = RESERVOIR

            # Lastly remove command areas that have no associated water body
            reservoir_ids: set = set(
                waterbodies.loc[
                    waterbodies["waterbody_type"] == RESERVOIR, "waterbody_id"
                ]
            )
            command_areas_dissolved = command_areas[
                command_areas["waterbody_id"].isin(reservoir_ids)
            ].reset_index(drop=True)

            self.set_geom(command_areas_dissolved, name="waterbodies/command_areas")

            assert command_areas_dissolved["waterbody_id"].isin(reservoir_ids).all()

            command_area_raster = rasterize_like(
                gdf=command_areas,
                column="waterbody_id",
                raster=self.grid["mask"],
                nodata=-1,
                dtype=np.int32,
                all_touched=True,
            )
            self.set_grid(command_area_raster, name="waterbodies/command_area")

            subcommand_area_raster = rasterize_like(
                gdf=command_areas,
                column="waterbody_id",
                raster=self.subgrid["mask"],
                nodata=-1,
                dtype=np.int32,
                all_touched=True,
            )
            self.set_subgrid(
                subcommand_area_raster, name="waterbodies/subcommand_areas"
            )

        else:
            command_areas: xr.DataArray = self.full_like(
                self.grid["mask"],
                fill_value=-1,
                nodata=-1,
                dtype=np.int32,
            )
            subcommand_areas = self.full_like(
                self.subgrid["mask"],
                fill_value=-1,
                nodata=-1,
                dtype=np.int32,
            )

            self.set_grid(command_areas, name="waterbodies/command_area")
            self.set_subgrid(subcommand_areas, name="waterbodies/subcommand_areas")

        if custom_reservoir_capacity:
            if custom_reservoir_capacity.endswith(".xlsx"):
                custom_reservoir_capacity: pd.DataFrame = pd.read_excel(
                    custom_reservoir_capacity
                )
            elif custom_reservoir_capacity.endswith(".csv"):
                custom_reservoir_capacity: pd.DataFrame = pd.read_csv(
                    custom_reservoir_capacity
                )
            else:
                raise ValueError(
                    "custom_reservoir_capacity must be a .csv or .xlsx file"
                )

            custom_reservoir_capacity = custom_reservoir_capacity[
                custom_reservoir_capacity.index != -1
            ]

            waterbodies.set_index("waterbody_id", inplace=True)
            waterbodies.update(custom_reservoir_capacity)
            waterbodies.reset_index(inplace=True)

        assert "waterbody_id" in waterbodies.columns, "waterbody_id is required"
        assert "waterbody_type" in waterbodies.columns, "waterbody_type is required"
        assert "volume_total" in waterbodies.columns, "volume_total is required"
        assert "average_discharge" in waterbodies.columns, (
            "average_discharge is required"
        )
        assert "average_area" in waterbodies.columns, "average_area is required"
        self.set_geom(waterbodies, name="waterbodies/waterbody_data")

    def setup_gtsm_water_levels(self, temporal_range: npt.NDArray[np.int32]) -> None:
        """Sets up the GTSM hydrographs for station within the model bounds.

        Args:
            temporal_range: The range of years to process.
        """
        # get the model bounds and buffer by ~2km
        model_bounds = self.bounds
        model_bounds = (
            model_bounds[0] - 0.0166,  # min_lon
            model_bounds[1] - 0.0166,  # min_lat
            model_bounds[2] + 0.0166,  # max_lon
            model_bounds[3] + 0.0166,  # max_lat
        )
        min_lon, min_lat, max_lon, max_lat = model_bounds

        # First: get station indices from ONE representative file
        ref_file = self.old_data_catalog.get_source("GTSM").path.format(1979, "01")  # ty:ignore[possibly-missing-attribute]
        ref = xr.open_dataset(ref_file)

        x_coords = ref.station_x_coordinate.load()
        y_coords = ref.station_y_coordinate.load()

        mask = (
            (x_coords >= min_lon)
            & (x_coords <= max_lon)
            & (y_coords >= min_lat)
            & (y_coords <= max_lat)
        )
        station_idx = np.nonzero(mask.values)[0]

        station_df = pd.DataFrame(
            {
                "station_id": ref.stations.values[mask].astype(str),
                "longitude": x_coords[mask].values,
                "latitude": y_coords[mask].values,
            }
        )
        ref.close()

        # Then: loop through files in smaller batches
        gtsm_data_region = []
        for year in temporal_range:
            for month in range(1, 13):
                f = self.old_data_catalog.get_source("GTSM").path.format(  # ty:ignore[possibly-missing-attribute]
                    year, f"{month:02d}"
                )
                ds = xr.open_dataset(f, chunks={"time": -1})
                subset = ds.isel(stations=station_idx).drop_vars(
                    ["station_x_coordinate", "station_y_coordinate"]
                )
                gtsm_data_region.append(subset.waterlevel.to_pandas())
                print(f"Processed GTSM data for {year}-{month:02d}")
                ds.close()
        gtsm_data_region_pd = pd.concat(gtsm_data_region, axis=0)
        # set _FillValue to NaN
        self.set_table(gtsm_data_region_pd, name="gtsm/waterlevels")
        stations = gpd.GeoDataFrame(
            station_df,
            geometry=[
                Point(xy) for xy in zip(station_df.longitude, station_df.latitude)
            ],
            crs="EPSG:4326",
        )
        self.set_geom(stations, name="gtsm/stations")
        self.logger.info("GTSM station waterlevels and geometries set")

    def setup_gtsm_surge_levels(self, temporal_range: npt.NDArray[np.int32]) -> None:
        """Sets up the GTSM surge hydrographs for station within the model bounds.

        Args:
            temporal_range: The range of years to process.
        """
        self.logger.info("Setting up GTSM surge hydrographs")
        # get the model bounds and buffer by ~2km
        model_bounds = self.bounds
        model_bounds = (
            model_bounds[0] - 0.0166,  # min_lon
            model_bounds[1] - 0.0166,  # min_lat
            model_bounds[2] + 0.0166,  # max_lon
            model_bounds[3] + 0.0166,  # max_lat
        )
        min_lon, min_lat, max_lon, max_lat = model_bounds

        # First: get station indices from ONE representative file
        ref_file = self.old_data_catalog.get_source("GTSM_surge").path.format(  # ty:ignore[possibly-missing-attribute]
            1979, "01"
        )
        ref = xr.open_dataset(ref_file)

        x_coords = ref.station_x_coordinate.load()
        y_coords = ref.station_y_coordinate.load()

        mask = (
            (x_coords >= min_lon)
            & (x_coords <= max_lon)
            & (y_coords >= min_lat)
            & (y_coords <= max_lat)
        )
        station_idx = np.nonzero(mask.values)[0]

        # Then: loop through files in smaller batches
        gtsm_data_region = []
        for year in temporal_range:
            for month in range(1, 13):
                f = self.old_data_catalog.get_source("GTSM_surge").path.format(  # ty:ignore[possibly-missing-attribute]
                    year, f"{month:02d}"
                )
                ds = xr.open_dataset(f, chunks={"time": -1})
                subset = ds.isel(stations=station_idx).drop_vars(
                    ["station_x_coordinate", "station_y_coordinate"]
                )
                gtsm_data_region.append(subset.surge.to_pandas())
                print(f"Processed GTSM data for {year}-{month:02d}")
                ds.close()
        gtsm_data_region_pd = pd.concat(gtsm_data_region, axis=0)
        # set _FillValue to NaN
        self.set_table(gtsm_data_region_pd, name="gtsm/surge")
        self.logger.info("GTSM station waterlevels and geometries set")

    def setup_gtsm_sea_level_rise(self) -> None:
        """Sets up the GTSM sea level rise data for the model.

        Raises:
            ValueError: If the extrapolated sea level rise data is not monotonically increasing
                or exceeds 2 meters by 2100 for any station.
        """
        self.logger.info("Setting up GTSM sea level rise data")
        # get the model bounds and buffer by ~2km
        model_bounds = self.bounds
        model_bounds = (
            model_bounds[0] - 0.0166,  # min_lon
            model_bounds[1] - 0.0166,  # min_lat
            model_bounds[2] + 0.0166,  # max_lon
            model_bounds[3] + 0.0166,  # max_lat
        )

        gtsm_sea_level_rise = self.data_catalog.fetch("gtsm").read(bounds=model_bounds)

        # extract data arrays
        mean_sea_level = gtsm_sea_level_rise.mean_sea_level.data
        time = gtsm_sea_level_rise.time.data
        stations = gtsm_sea_level_rise.stations.data

        # create dataframe
        mean_sea_level_df = pd.DataFrame(
            data=mean_sea_level.T,
            index=pd.to_datetime(time),
            columns=stations,
        )
        # sort by datetime index
        mean_sea_level_df = mean_sea_level_df.sort_index()

        # set table for model
        self.set_table(mean_sea_level_df, name="gtsm/mean_sea_level")

        # calculate the increment in mean sea level in the time series  based on a reference year
        reference_year = 2020
        sea_level_rise_df = mean_sea_level_df.subtract(
            mean_sea_level_df.loc[f"{reference_year}-01-01"], axis=1
        )

        # extrapolate to 2100 using nonlinear trend  between 2015-2050 per station
        last_year = sea_level_rise_df.index.year.max()
        future_years = np.arange(last_year + 1, 2101)
        future_dates = pd.to_datetime([f"{year}-01-01" for year in future_years])
        future_data = {}
        for station in sea_level_rise_df.columns:
            series = sea_level_rise_df[station]
            # fit nonlinear trend all years
            recent_series = series
            coeffs = np.polyfit(
                recent_series.index.year,
                recent_series.values,
                deg=2,
            )
            trend = np.poly1d(coeffs)
            future_values = trend(future_years)
            future_data[station] = future_values
        future_df = pd.DataFrame(
            data=future_data,
            index=future_dates,
        )
        # append future data to sea_level_rise_df
        sea_level_rise_df = pd.concat([sea_level_rise_df, future_df], axis=0)

        # do some check on the extrapolated data
        for station in sea_level_rise_df.columns:
            series = sea_level_rise_df[station]
            # check that the values are monotonically increasing
            if not series.is_monotonic_increasing:
                raise ValueError(
                    f"Sea level rise data for station {station} is not monotonically increasing after extrapolation."
                )
            # check that the values are reasonable (less than 2 meters by 2100)
            if series.iloc[-1] >= 2:
                raise ValueError(
                    f"Sea level rise data for station {station} exceeds 2 meters by 2100."
                )

        # set table for model
        self.set_table(sea_level_rise_df, name="gtsm/sea_level_rise_rcp8p5")

    def setup_coast_rp(self) -> None:
        """Sets up the coastal return period data for the model."""
        self.logger.info("Setting up coastal return period data")
        stations = gpd.read_parquet(
            os.path.join("input", self.files["geom"]["gtsm/stations"])
        )

        fp_coast_rp = self.old_data_catalog.get_source("COAST_RP").path
        coast_rp = pd.read_pickle(fp_coast_rp)

        # remove stations that are not in coast_rp index
        stations = stations[
            stations["station_id"].astype(int).isin(coast_rp.index)
        ].reset_index(drop=True)

        coast_rp = coast_rp.loc[stations["station_id"].astype(int)]
        self.set_table(coast_rp, name="coast_rp")

        # also set stations (only those that are in coast_rp)
        self.set_geom(stations, "gtsm/stations_coast_rp")

    @build_method(required=True)
    def setup_gtsm_station_data(self) -> None:
        """This function sets up COAST-RP and the GTSM station data (surge and waterlevel) for the model."""
        if not self.geom["routing/subbasins"]["is_coastal"].any():
            self.logger.info("No coastal basins found, skipping GTSM hydrographs setup")
            return

        # Continue with GTSM setup
        temporal_range = np.arange(1979, 2018, 1, dtype=np.int32)
        self.setup_gtsm_water_levels(temporal_range)
        self.setup_gtsm_surge_levels(temporal_range)
        self.setup_gtsm_sea_level_rise()
        self.setup_coast_rp()

    @build_method(required=False)
    def setup_inflow(
        self,
        locations: str,
        inflow_m3_per_s: str,
        interpolate: bool = False,
        extrapolate: bool = False,
    ) -> None:
        """Sets up a inflow hydrograph for the model.

        Args:
            locations: A vector file that can be read by geopandas containing the inflow location,
                with column ID (str) and a point geometry.
            inflow_m3_per_s: A CSV file containing the inflow hydrograph in m3/s with a datetime index (date),
                and a column for each inflow location ID matching the IDs (str) in the locations file.
            interpolate: Whether to interpolate missing values in the inflow hydrograph.
            extrapolate: Whether to extrapolate missing values in the inflow hydrograph.

        Raises:
            ValueError: If the inflow location is outside of the model grid or study area.
        """
        mask = self.grid["mask"].compute()
        transform = mask.rio.transform()

        inflow_locations = gpd.read_file(locations)

        if inflow_locations.crs is None:
            raise ValueError("Locations file must have a defined CRS.")
        inflow_locations = inflow_locations.to_crs(mask.rio.crs)

        if "ID" not in inflow_locations.columns:
            raise ValueError("Locations file must have an 'ID' column.")

        for ID in inflow_locations["ID"]:
            if not isinstance(ID, str):
                raise ValueError("Locations 'ID' column must be of type string.")

        y, x = rasterio.transform.rowcol(
            transform, inflow_locations.geometry.x, inflow_locations.geometry.y
        )

        inflow_locations["y"] = y
        inflow_locations["x"] = x

        self.set_geom(inflow_locations, name="routing/inflow_locations")

        if (
            (inflow_locations["y"] < 0)
            | (inflow_locations["y"] >= mask.shape[0])
            | (inflow_locations["x"] < 0)
            | (inflow_locations["x"] >= mask.shape[1])
        ).any():
            raise ValueError("Inflow location is outside of the model grid.")

        # check if any of the locations is outside of the study area (mask is True outside the study area)
        if (mask.values[inflow_locations["y"], inflow_locations["x"]]).any():
            raise ValueError("Inflow location is outside of the study area.")

        inflow_df_m3_per_s = pd.read_csv(inflow_m3_per_s, index_col=0, parse_dates=True)
        # ensure index is datetime
        if not np.issubdtype(inflow_df_m3_per_s.index.dtype, np.datetime64):  # ty:ignore[invalid-argument-type]
            raise ValueError("Inflow hydrograph index must be datetime.")

        # ensure columns are strings
        if not all(isinstance(col, str) for col in inflow_df_m3_per_s.columns):
            raise ValueError("Inflow hydrograph columns must be of type string.")

        # check if all IDs in locations are in the inflow file
        missing_ids: set[str] = set(inflow_locations["ID"]) - set(
            inflow_df_m3_per_s.columns
        )
        if missing_ids:
            raise ValueError(
                f"Missing inflow data for location IDs: {missing_ids}. "
                "Ensure column names in CSV match location IDs."
            )

        # subset to only include the locations that are in the locations file
        inflow_df_m3_per_s = inflow_df_m3_per_s[inflow_locations["ID"]]

        # align to model time step
        model_time_step = pd.date_range(
            self.start_date,
            end=self.end_date + timedelta(hours=23),
            freq="H",
            inclusive="both",
        )

        inflow_df_m3_per_s: pd.DataFrame = inflow_df_m3_per_s.reindex(model_time_step)

        if interpolate:
            # interpolate missing values by first interpolating in time
            inflow_df_m3_per_s: pd.DataFrame = inflow_df_m3_per_s.interpolate(
                method="time"
            )
        if extrapolate:
            # extrapolate missing values by forward and backward filling
            inflow_df_m3_per_s: pd.DataFrame = inflow_df_m3_per_s.ffill().bfill()

        if inflow_df_m3_per_s.isnull().any().any():
            raise ValueError(
                "Inflow hydrograph contains missing values. "
                "Set interpolate=True to interpolate missing values. "
                "or fill missing values in the inflow hydrograph. "
                "model start and end date: "
                f"{self.start_date} to {self.end_date} "
                "with hourly frequency."
            )

        self.set_table(inflow_df_m3_per_s, name="routing/inflow_m3_per_s")
