"""Build methods for the hydrography for GEB."""

import os
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyflwdir
import rasterio
import xarray as xr
from pyflwdir import FlwdirRaster
from rasterio.features import rasterize
from scipy.ndimage import value_indices
from shapely.geometry import LineString, Point, shape

from geb.build.data_catalog import NewDataCatalog
from geb.build.methods import build_method
from geb.hydrology.lakes_reservoirs import LAKE, LAKE_CONTROL, RESERVOIR
from geb.types import ArrayBool, ArrayFloat32, ArrayInt32
from geb.workflows.raster import rasterize_like, snap_to_grid


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


def get_downstream_subbasins(
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


def get_subbasins_geometry(
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
    subbasins: gpd.GeoDataFrame = data_catalog.fetch("merit_basins_catchments").read(
        filters=[
            ("COMID", "in", subbasin_ids),
        ],
    )
    assert len(subbasins) == len(subbasin_ids), "Some subbasins were not found"
    return subbasins.set_index("COMID")


def get_rivers(
    data_catalog: NewDataCatalog, subbasin_ids: list[int]
) -> gpd.GeoDataFrame:
    """Get the rivers and its attributes for the given subbasin IDs.

    Assumes that the subbasin IDs match the COMID in the MERIT river dataset.

    Args:
        data_catalog: The data catalog to use for accessing the MERIT dataset.
        subbasin_ids: The subbasin IDs to get the rivers for.

    Returns:
        A GeoDataFrame containing the rivers for the given subbasin IDs.
    """
    rivers: gpd.GeoDataFrame = (
        data_catalog.fetch("merit_basins_rivers")
        .read(
            columns=[
                "COMID",
                "lengthkm",
                "slope",
                "uparea",
                "maxup",
                "NextDownID",
                "geometry",
            ],
            filters=[
                ("COMID", "in", subbasin_ids),
            ],
        )
        .rename(
            columns={
                "NextDownID": "downstream_ID",
            }
        )
    )
    rivers["uparea_m2"] = rivers["uparea"] * 1e6  # convert from km^2 to m^2
    rivers["is_headwater_catchment"] = rivers["maxup"] == 0
    rivers: gpd.GeoDataFrame = rivers.drop(columns=["uparea"])
    rivers.loc[rivers["downstream_ID"] == 0, "downstream_ID"] = -1
    assert len(rivers) == len(subbasin_ids), "Some rivers were not found"
    # reverse the river lines to have the downstream direction
    rivers["geometry"] = rivers["geometry"].apply(
        lambda x: LineString(list(x.coords)[::-1])
    )
    return rivers.set_index("COMID")


def create_river_raster_from_river_lines(
    rivers: gpd.GeoDataFrame,
    target: xr.DataArray,
    column: str | None = None,
    index: str | None = None,
) -> npt.NDArray[np.int32]:
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
    river_raster = rasterize(
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
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
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
    MERIT_Basins_to_SWORD: xr.Dataset = (
        data_catalog.fetch("merit_sword").read().sel(mb=rivers.index.tolist())
    )

    SWORD_reach_IDs = np.full((40, len(rivers)), dtype=np.int64, fill_value=-1)
    SWORD_reach_lengths = np.full(
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
    data_catalog: NewDataCatalog, SWORD_reach_IDs: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
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

    SWORD = (
        data_catalog.fetch("sword")
        .read(
            sql=f"""SELECT * FROM sword WHERE reach_id IN ({",".join([str(ID) for ID in unique_SWORD_reach_ids])})"""
        )
        .set_index("reach_id")
    )

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


class Hydrography:
    """Contains all build methods for the hydrography for GEB."""

    def __init__(self) -> None:
        """Initializes the Hydrography class."""
        pass

    @build_method(depends_on=["setup_hydrography", "setup_cell_area"])
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
        a = (2 * self.grid["cell_area"]) / self.grid["routing/upstream_area"]
        a = xr.where(a < 1, a, 1, keep_attrs=True)
        b = self.grid["routing/outflow_elevation"] / 2000
        b = xr.where(b < 1, b, 1, keep_attrs=True)

        mannings = 0.025 + 0.015 * a + 0.030 * b
        mannings.attrs["_FillValue"] = np.nan

        self.set_grid(mannings, "routing/mannings")

    def set_routing_subbasins(
        self, river_graph: nx.DiGraph, sink_subbasin_ids: list[int]
    ) -> None:
        """Sets the routing subbasins for the model.

        For each sink subbasin, all upstream subbasins and the immediately downstream subbasins are included.
        The downstream subbasins have a flag set to indicate that they are downstream outflow subbasins.

        If a sink subbasin has no downstream subbasin, it is assumed to be a coastal basin. This is
        also saved in the subbasins attribute.

        Args:
            river_graph: The river graph to use for determining upstream and downstream subbasins.
            sink_subbasin_ids: The subbasin IDs of the sink subbasins (i.e., the outflow subbasins).
        """
        # always make a list of the subbasin ids, such that the function always gets the same type of input
        if not isinstance(sink_subbasin_ids, (list, set)):
            sink_subbasin_ids = [sink_subbasin_ids]

        subbasin_ids: set[int] = get_all_upstream_subbasin_ids(
            river_graph=river_graph, subbasin_ids=sink_subbasin_ids
        )
        subbasin_ids.update(sink_subbasin_ids)

        downstream_subbasins = get_downstream_subbasins(river_graph, sink_subbasin_ids)
        subbasin_ids.update(downstream_subbasins)

        subbasins = get_subbasins_geometry(self.new_data_catalog, subbasin_ids)
        subbasins["is_downstream_outflow_subbasin"] = pd.Series(
            True, index=downstream_subbasins
        ).reindex(subbasins.index, fill_value=False)

        subbasins["associated_upstream_basins"] = pd.Series(
            downstream_subbasins.values(), index=downstream_subbasins
        ).reindex(subbasins.index, fill_value=[])

        # for each basin check if the basin has any downstream neighbors
        # if not, it is a coastal basin
        subbasins["is_coastal_basin"] = subbasins.apply(
            lambda row: len(list(river_graph.neighbors(row.name))) == 0, axis=1
        )

        self.set_geom(subbasins, name="routing/subbasins")

    @build_method
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
        original_d8_ldd = self.other["drainage/original_d8_flow_directions"]
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
            upstream_area_high_res, name="drainage/original_d8_upstream_area"
        )

        elevation_coarsened = original_d8_elevation.coarsen(
            x=self.ldd_scale_factor,
            y=self.ldd_scale_factor,
            boundary="exact",
            coord_func="mean",
        )

        # elevation (we only set this later, because it has to be done after setting the mask)
        elevation = elevation_coarsened.mean()
        elevation = snap_to_grid(elevation, self.grid["mask"])

        self.set_grid(elevation, name="landsurface/elevation")

        elevation_std = elevation_coarsened.std()
        elevation_std = snap_to_grid(elevation_std, self.grid["mask"])
        self.set_grid(
            elevation_std,
            name="landsurface/elevation_standard_deviation",
        )

        # outflow elevation
        outflow_elevation = elevation_coarsened.min()
        outflow_elevation = snap_to_grid(outflow_elevation, self.grid["mask"])
        self.set_grid(outflow_elevation, name="routing/outflow_elevation")

        slope = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
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
        self.set_grid(slope, name="landsurface/slope")

        flow_raster_idxs_ds = self.grid["flow_raster_idxs_ds"].compute()
        flow_raster = FlwdirRaster(
            flow_raster_idxs_ds.values.ravel(),
            shape=flow_raster_idxs_ds.shape,
            transform=flow_raster_idxs_ds.rio.transform(recalc=True),
            ftype="d8",
            latlon=True,
        )

        # flow direction
        ldd: npt.NDArray[np.uint8] = self.full_like(
            outflow_elevation, fill_value=255, nodata=255, dtype=np.uint8
        )
        ldd.data = flow_raster.to_array(ftype="ldd")
        self.set_grid(ldd, name="routing/ldd")

        # upstream area
        upstream_area: xr.DataArray = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        upstream_area_data: npt.NDArray[np.float32] = flow_raster.upstream_area(
            unit="m2"
        ).astype(np.float32)
        upstream_area_data[upstream_area_data == -9999.0] = np.nan
        upstream_area.data = upstream_area_data
        self.set_grid(upstream_area, name="routing/upstream_area")

        upstream_area_n_cells: xr.DataArray = self.full_like(
            outflow_elevation, fill_value=-1, nodata=-1, dtype=np.int32
        )
        upstream_area_n_cells_data: npt.NDArray[np.int32] = flow_raster.upstream_area(
            unit="cell"
        ).astype(np.int32)
        upstream_area_n_cells.data = upstream_area_n_cells_data
        self.set_grid(upstream_area_n_cells, name="routing/upstream_area_n_cells")

        # river length
        river_length: xr.DataArray = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_length_data: npt.NDArray[np.float32] = (
            flow_raster_original.subgrid_rivlen(
                self.grid["idxs_outflow"].values, unit="m", direction="down"
            )
        )
        river_length_data[river_length_data == -9999.0] = np.nan
        river_length.data = river_length_data
        self.set_grid(river_length, name="routing/river_length")

        # river slope
        river_slope: xr.DataArray = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_slope_data: npt.NDArray[np.float32] = flow_raster_original.subgrid_rivslp(
            self.grid["idxs_outflow"].values, original_d8_elevation
        )
        river_slope_data[river_slope_data == -9999.0] = np.nan
        river_slope.data = river_slope_data
        self.set_grid(
            river_slope,
            name="routing/river_slope",
        )

        # river width
        subbasin_ids: list[int] = self.geom["routing/subbasins"].index.tolist()

        self.logger.info("Retrieving river data")
        rivers: gpd.GeoDataFrame = get_rivers(self.new_data_catalog, subbasin_ids)

        self.logger.info("Processing river data")
        # remove all rivers that are both shorter than 1 km and have no upstream river
        rivers: gpd.GeoDataFrame = rivers[
            ~((rivers["lengthkm"] < 1) & (rivers["maxup"] == 0))
        ]

        rivers: gpd.GeoDataFrame = rivers.join(
            self.geom["routing/subbasins"][
                ["is_downstream_outflow_subbasin", "associated_upstream_basins"]
            ],
            how="left",
        )

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
                & (~rivers["is_downstream_outflow_subbasin"])
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

        represented_rivers = rivers[~rivers["is_downstream_outflow_subbasin"]]

        for river_ID, river in rivers.iterrows():
            if river_ID not in xy_per_river_segment:
                if (
                    river["is_downstream_outflow_subbasin"]
                    or not river["represented_in_grid"]
                ):
                    continue
                else:
                    raise AssertionError("River xy not found, but should be found.")

            (ys, xs) = xy_per_river_segment[river_ID]
            upstream_area: ArrayFloat32 = upstream_area_high_res_data[ys, xs]
            nan_mask: ArrayBool = np.isnan(upstream_area)

            if nan_mask.all():
                if (
                    river["is_downstream_outflow_subbasin"]
                    or not river["represented_in_grid"]
                ):
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
            outflow_elevation, fill_value=-1, nodata=-1, dtype=np.int32
        )
        COMID_IDs_raster.data = river_raster_LR
        self.set_grid(COMID_IDs_raster, name="routing/river_ids")

        basin_ids = self.full_like(
            outflow_elevation, fill_value=-1, nodata=-1, dtype=np.int32
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
            self.new_data_catalog, rivers
        )

        SWORD_river_widths: npt.NDArray[np.float64] = get_SWORD_river_widths(
            self.new_data_catalog, SWORD_reach_IDs
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
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_width.data = river_width_data
        self.set_grid(river_width, name="routing/river_width")

    @build_method
    def setup_global_ocean_mean_dynamic_topography(self) -> None:
        """Sets up the global ocean mean dynamic topography for the model."""
        if not self.geom["routing/subbasins"]["is_coastal_basin"].any():
            self.logger.info(
                "No coastal basins found, skipping setup_global_ocean_mean_dynamic_topography"
            )
            return

        global_ocean_mdt_fn = self.data_catalog.get_source(
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

    @build_method
    def setup_low_elevation_coastal_zone_mask(self) -> None:
        """Sets up the low elevation coastal zone (LECZ) mask for sfincs models."""
        if not self.geom["routing/subbasins"]["is_coastal_basin"].any():
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
        self.set_geom(
            low_elevation_coastal_zone_mask,
            name="coastal/low_elevation_coastal_zone_mask",
        )

    @build_method
    def setup_coastlines(self) -> None:
        """Sets up the coastlines for the model."""
        if not self.geom["routing/subbasins"]["is_coastal_basin"].any():
            self.logger.info("No coastal basins found, skipping setup_coastlines")
            return

        # load the coastline from the data catalog
        fp_coastlines = self.data_catalog.get_source("osm_coastlines").path
        coastlines = gpd.read_file(fp_coastlines)

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

    @build_method
    def setup_osm_land_polygons(
        self,
    ) -> None:
        """Sets up the OSM land polygons for the model."""
        if not self.geom["routing/subbasins"]["is_coastal_basin"].any():
            self.logger.info(
                "No coastal basins found, skipping setup_osm_land_polygons"
            )
            return

        # load the land polygon from the data catalog
        fp_land_polygons = self.data_catalog.get_source("osm_land_polygons").path
        land_polygons = gpd.read_file(fp_land_polygons)
        # select only the land polygons that intersect with the region
        land_polygons = land_polygons[land_polygons.intersects(self.region.union_all())]
        # merge all land polygons into a single polygon
        land_polygons = gpd.GeoDataFrame(
            geometry=[land_polygons.union_all()], crs=land_polygons.crs
        )

        # clip and write to model files
        self.set_geom(land_polygons.clip(self.bounds), name="coastal/land_polygons")

    @build_method(
        depends_on=["setup_low_elevation_coastal_zone_mask", "setup_coastlines"]
    )
    def setup_coastal_sfincs_model_regions(
        self, minimum_coastal_area_deg2: float = 0.0006449015308288645
    ) -> None:
        """Sets up the coastal sfincs model regions."""
        if not self.geom["routing/subbasins"]["is_coastal_basin"].any():
            self.logger.info(
                "No coastal basins found, skipping setup_coastal_sfincs_model_regions"
            )
            return

        # load elevation data
        elevation = self.other["DEM/fabdem"]
        # load the lecz mask
        low_elevation_coastal_zone_mask = self.geom[
            "coastal/low_elevation_coastal_zone_mask"
        ]
        # add small buffer to ensure connection of 'islands' with coastlines
        low_elevation_coastal_zone_mask.geometry = (
            low_elevation_coastal_zone_mask.geometry.buffer(0.001)
        )
        # split the low elevation coastal zone mask into individual polygons of contiguous areas
        low_elevation_coastal_zone_polygons = low_elevation_coastal_zone_mask.explode(
            index_parts=False
        ).reset_index(drop=True)

        # add area column
        low_elevation_coastal_zone_polygons["area"] = (
            low_elevation_coastal_zone_polygons.geometry.area
        )

        # load the coastlines
        coastlines = self.geom["coastal/coastlines"]
        sfincs_regions = []
        low_elevation_coastal_zone_regions = []
        initial_water_levels = []
        for (
            _,
            low_elevation_coastal_zone_polygon,
        ) in low_elevation_coastal_zone_polygons.iterrows():
            # check if the low elevation coastal zone polygon intersects with the coastline
            if (
                coastlines.intersects(low_elevation_coastal_zone_polygon.geometry).any()
                and low_elevation_coastal_zone_polygon["area"]
                > minimum_coastal_area_deg2  # approx 1 km2 at equator
            ):
                # if it does, create a sfincs region
                # get the minimum elevation within the low elevation coastal zone polygon
                mask = elevation.rio.clip_box(
                    *low_elevation_coastal_zone_polygon.geometry.bounds
                ).where(
                    elevation.rio.clip(
                        [low_elevation_coastal_zone_polygon.geometry], drop=False
                    ).notnull(),
                    drop=False,
                )
                initial_water_levels.append(float(mask.min().values))

                # create a bounding box around the low elevation coastal zone polygon
                low_elevation_coastal_zone_polygon_gpd = gpd.GeoDataFrame(
                    geometry=[low_elevation_coastal_zone_polygon.geometry],
                    crs=low_elevation_coastal_zone_mask.crs,
                )
                bbox = low_elevation_coastal_zone_polygon_gpd.minimum_rotated_rectangle().iloc[
                    0
                ]
                # add a small buffer to ensure connection with coastlines
                bbox = bbox.buffer(0.04, join_style=2)

                sfincs_regions.append(bbox)
                low_elevation_coastal_zone_regions.append(
                    low_elevation_coastal_zone_polygon[0]
                )
        bbox_gdf = gpd.GeoDataFrame(
            geometry=sfincs_regions, crs=low_elevation_coastal_zone_mask.crs
        )
        low_elevation_coastal_zone_gdf = gpd.GeoDataFrame(
            geometry=low_elevation_coastal_zone_regions,
            crs=low_elevation_coastal_zone_mask.crs,
        )

        # add idx
        bbox_gdf["idx"] = bbox_gdf.index
        low_elevation_coastal_zone_gdf["idx"] = low_elevation_coastal_zone_gdf.index
        low_elevation_coastal_zone_gdf["area"] = (
            low_elevation_coastal_zone_gdf.geometry.area
        )
        low_elevation_coastal_zone_gdf["initial_water_level"] = initial_water_levels
        self.set_geom(bbox_gdf, name="coastal/model_regions")
        self.set_geom(
            low_elevation_coastal_zone_gdf,
            name="coastal/low_elevation_coastal_zone_mask",
        )

    @build_method
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
        """
        waterbodies: gpd.GeoDataFrame = self.new_data_catalog.fetch("hydrolakes").read(
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

        water_body_id: xr.DataArray = rasterize_like(
            gdf=waterbodies,
            column="waterbody_id",
            raster=self.grid["mask"],
            nodata=-1,
            dtype=np.int32,
            all_touched=True,
        )

        sub_water_body_id: xr.DataArray = rasterize_like(
            gdf=waterbodies,
            column="waterbody_id",
            raster=self.subgrid["mask"],
            nodata=-1,
            dtype=np.int32,
            all_touched=True,
        )

        self.set_grid(water_body_id, name="waterbodies/water_body_id")
        self.set_subgrid(sub_water_body_id, name="waterbodies/sub_water_body_id")

        waterbodies["volume_flood"] = waterbodies["volume_total"]

        if command_areas:
            command_areas = self.data_catalog.get_geodataframe(
                command_areas, geom=self.region, predicate="intersects"
            )
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
            command_areas = self.full_like(
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
            custom_reservoir_capacity = self.data_catalog.get_dataframe(
                custom_reservoir_capacity
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
        ref_file = self.data_catalog.get_source("GTSM").path.format(1979, "01")
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
                f = self.data_catalog.get_source("GTSM").path.format(
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
        ref_file = self.data_catalog.get_source("GTSM_surge").path.format(1979, "01")
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
                f = self.data_catalog.get_source("GTSM_surge").path.format(
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

    def setup_coast_rp(self) -> None:
        """Sets up the coastal return period data for the model."""
        self.logger.info("Setting up coastal return period data")
        stations = gpd.read_parquet(
            os.path.join("input", self.files["geom"]["gtsm/stations"])
        )

        fp_coast_rp = self.data_catalog.get_source("COAST_RP").path
        coast_rp = pd.read_pickle(fp_coast_rp)

        # remove stations that are not in coast_rp index
        stations = stations[
            stations["station_id"].astype(int).isin(coast_rp.index)
        ].reset_index(drop=True)

        coast_rp = coast_rp.loc[stations["station_id"].astype(int)]
        self.set_table(coast_rp, name="coast_rp")

        # also set stations (only those that are in coast_rp)
        self.set_geom(stations, "gtsm/stations_coast_rp")

    @build_method
    def setup_gtsm_station_data(self) -> None:
        """This function sets up COAST-RP and the GTSM station data (surge and waterlevel) for the model."""
        if not self.geom["routing/subbasins"]["is_coastal_basin"].any():
            self.logger.info("No coastal basins found, skipping GTSM hydrographs setup")
            return

        # Continue with GTSM hydrographs setup
        temporal_range = np.arange(1979, 2018, 1, dtype=np.int32)
        self.setup_gtsm_water_levels(temporal_range)
        self.setup_gtsm_surge_levels(temporal_range)
        self.setup_coast_rp()
