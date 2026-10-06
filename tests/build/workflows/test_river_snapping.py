"""Verify vectorized grid snapping and reuse of river coordinates."""

import logging

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point

from geb.build.workflows.river_snapping import (
    SnappingResults,
    snap_point_to_river_network,
)


@pytest.mark.parametrize("sample_subgrid", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
def test_nearest_cell_tie_and_cache(use_cache: bool, sample_subgrid: bool) -> None:
    """Keep the first equidistant cell and reuse coordinates across barriers.

    Args:
        use_cache: Whether coordinate caching is enabled.
        sample_subgrid: Whether to sample subgrid diagnostics.

    Returns:
        None.

    Raises:
        AssertionError: If snapping or cache reuse changes the selected cell.
    """  # noqa: DOC202, DOC502
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "represented_in_grid": [True],
            "uparea_m2": [10.0],
            "hydrography_xy": [[(0, 0), (1, 0)]],
            "shreve_stream_order": [1],
        },
        geometry=[LineString([(0, 0), (0.02, 0)])],
        index=pd.Index([1], dtype="int64"),
        crs=4326,
    )
    area: xr.DataArray = xr.DataArray(
        [[10.0, 20.0]], dims=("y", "x"), coords={"y": [0.0], "x": [0.0, 0.02]}
    )
    cache: dict[int, npt.NDArray[np.float64]] | None = {} if use_cache else None
    result: SnappingResults | None = snap_point_to_river_network(
        Point(0.01, 0),
        rivers,
        area,
        area if sample_subgrid else None,
        river_coordinate_cache=cache,
    )
    assert result is not None
    assert result.snapped_grid_pixel_xy == (0, 0)
    saved_coordinates: npt.NDArray[np.float64] | None = (
        cache[1] if cache is not None else None
    )
    result = snap_point_to_river_network(
        Point(0.019, 0),
        rivers,
        area,
        area if sample_subgrid else None,
        river_coordinate_cache=cache,
    )
    assert result is not None
    assert result.snapped_grid_pixel_xy == (1, 0)
    assert result.geb_uparea_grid == 20.0
    if sample_subgrid:
        assert result.geb_uparea_subgrid == 20.0
        assert result.subgrid_pixel_coords == (0.02, 0.0)
    else:
        assert np.isnan(result.geb_uparea_subgrid)
        assert np.isnan(result.subgrid_pixel_coords).all()
    if cache is not None:
        assert cache[1] is saved_coordinates
    rivers.at[1, "hydrography_xy"] = []
    with pytest.raises(ValueError, match="no model grid cells"):
        snap_point_to_river_network(Point(0, 0), rivers, area, area)


@pytest.mark.parametrize("has_river", [False, True])
@pytest.mark.parametrize("use_build_logger", [False, True])
def test_distant_barrier_diagnostics(
    has_river: bool, use_build_logger: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Retain distant barriers in diagnostics, including an empty river network.

    Args:
        has_river: Whether there is a river far from the barrier.
        use_build_logger: Whether summaries use an explicitly supplied build logger.
        caplog: Captured messages for checking exclusion summaries.

    Returns:
        None.

    Raises:
        AssertionError: If the distant barrier is included or lost.
    """  # noqa: DOC202, DOC502
    from geb.build.workflows.weirs import create_weir_grids

    logger: logging.Logger | None = (
        logging.getLogger("barrier_build") if use_build_logger else None
    )
    expected_logger: str = (
        "barrier_build" if use_build_logger else "geb.build.workflows.weirs"
    )
    caplog.set_level("INFO", logger=expected_logger)
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"represented_in_grid": [True]},
        geometry=[LineString([(0, 0), (0, 0.01)])],
        crs=4326,
    ).iloc[: int(has_river)]
    points: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"amber_id": ["far"], "dam_type": ["Dam"], "dam_hgt_m": [3.0]},
        geometry=[Point(10, 52)],
        crs=4326,
    )
    area: xr.DataArray = xr.DataArray(
        np.ones((2, 2)), dims=("y", "x"), coords={"y": [0.01, 0], "x": [0, 0.01]}
    ).rio.write_crs(4326)
    heights: xr.DataArray
    records: gpd.GeoDataFrame
    heights, records = create_weir_grids(
        gpd.GeoDataFrame(),
        None,
        xr.full_like(area, -1),
        rivers,
        area,
        area,
        xr.ones_like(area, dtype=bool),
        amber_points=points,
        logger=logger,
    )
    assert len(records) == 1
    assert records.iloc[0].exclusion_reason == "no river within 250 m"
    assert np.isnan(records.iloc[0].distance_to_river_m)
    assert not records.iloc[0].included
    assert "AMBER barriers: 0 included, 1 excluded (1 total)." in caplog.text
    assert "AMBER excluded: 1 — no river within 250 m." in caplog.text
    assert "Skipping" not in caplog.text
    assert all(record.name == expected_logger for record in caplog.records)
    assert not heights.values.any()


def _unexpected_subgrid_read() -> npt.NDArray[np.float32]:
    """Fail if a diagnostic raster is read during weir placement.

    Returns:
        Never returns an array.

    Raises:
        RuntimeError: Whenever the raster is evaluated.
    """  # noqa: DOC202
    raise RuntimeError("Weir placement must not read subgrid diagnostics.")


def test_weirs_skip_subgrid_read() -> None:
    """Place a nearby weir without evaluating the diagnostic drainage raster.

    Returns:
        None.

    Raises:
        RuntimeError: If the unused drainage raster is evaluated.
    """  # noqa: DOC202, DOC502
    import dask.array as da
    from dask import delayed

    from geb.build.workflows.weirs import create_weir_grids

    area: xr.DataArray = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0.0, -0.01], "x": [0.0, 0.01]},
    ).rio.write_crs(4326)
    subgrid: xr.DataArray = area.copy(
        data=da.from_delayed(
            delayed(_unexpected_subgrid_read)(),
            shape=(2, 2),
            dtype=np.float32,
        )
    )
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "represented_in_grid": [True],
            "uparea_m2": [10.0],
            "hydrography_xy": [[(0, 0), (0, 1)]],
            "shreve_stream_order": [1],
        },
        geometry=[LineString([(0, 0), (0, -0.01)])],
        index=pd.Index([1], dtype="int64"),
        crs=4326,
    )
    points: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"amber_id": ["near"], "dam_type": ["Weir"], "dam_hgt_m": [3.0]},
        geometry=[Point(0.001, 0)],
        crs=4326,
    )
    heights: xr.DataArray
    records: gpd.GeoDataFrame
    heights, records = create_weir_grids(
        gpd.GeoDataFrame(),
        None,
        xr.full_like(area, -1),
        rivers,
        area,
        subgrid,
        xr.ones_like(area, dtype=bool),
        amber_points=points,
    )
    assert heights.values[0, 0] == 3.0
    assert records.iloc[0].included
    assert records.iloc[0].distance_to_river_m > 0
