"""Test GDW weirs outside reservoir outlines."""

import logging
from pathlib import Path
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point

from geb.build import DelayedReader, GEBModel
from geb.build.workflows.weirs import create_weir_grids
from geb.workflows import (
    USE_BANKFULL_HEIGHT,
    USE_HALF_BANKFULL_HEIGHT,
)
from geb.workflows.io import read_geom, read_zarr


@pytest.mark.parametrize("with_waterbody_outline", [False, True])
@pytest.mark.parametrize("amber_type", ["Weir", "Dam"])
@pytest.mark.parametrize("amber_longitude", [0.001, 0.003])
@pytest.mark.parametrize("has_amber_points", [True, False])
@pytest.mark.parametrize("has_gdw_points", [True, False])
@pytest.mark.parametrize("lazy_grids", [True, False])
def test_setup_weirs_from_files(
    tmp_path: Path,
    has_gdw_points: bool,
    has_amber_points: bool,
    amber_longitude: float,
    amber_type: str,
    with_waterbody_outline: bool,
    lazy_grids: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read saved GDW points during an update; allow builds without GDW points.

    Args:
        tmp_path: Folder for test files.
        has_gdw_points: Whether a GDW points file is available.
        has_amber_points: Whether the atlas contains a barrier in this region.
        amber_longitude: Barrier longitude (degrees), within or beyond 250 m.
        amber_type: Fixed weir or gated AMBER dam.
        with_waterbody_outline: Whether the AMBER point intersects a waterbody polygon.
        lazy_grids: Whether to use Dask arrays, as input updates do.
        monkeypatch: Fixture to replace the atlas download.
    """
    geometry_files: DelayedReader = DelayedReader(read_geom)
    if has_gdw_points:
        points: gpd.GeoDataFrame = gpd.GeoDataFrame(
            {
                "gdw_id": [34354],
                "inside_gdw_polygon": [False],
                "waterbody_id": [np.nan],
                "dam_hgt_m": [np.nan],
                "dam_type": ["Sluice"],
            },
            geometry=[Point(0.01, 0.01)],
            crs=4326,
        )
        points_file: Path = tmp_path / "gdw_checks.parquet"
        points.to_parquet(points_file)
        geometry_files["waterbodies/gdw_checks"] = points_file

    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "represented_in_grid": [True],
            "uparea_m2": [1000.0],
            "hydrography_xy": [[(0, 0), (0, 1)]],
            "shreve_stream_order": [1],
        },
        index=pd.Index([1], dtype="int64"),
        geometry=[LineString([(0, 0), (0, -0.1)])],
        crs=4326,
    )
    rivers_file: Path = tmp_path / "rivers.parquet"
    rivers.to_parquet(rivers_file)
    geometry_files["routing/rivers"] = rivers_file
    upstream_area: xr.DataArray = xr.DataArray(
        np.full((2, 2), 1000.0, dtype=np.float32),
        coords={"y": [0.0, -0.1], "x": [0.0, 0.1]},
        dims=("y", "x"),
        attrs={"_FillValue": np.nan},
    )
    upstream_area = upstream_area.rio.write_crs(4326)
    downstream_cells: xr.DataArray = upstream_area.copy(
        data=np.array([[2, 3], [2, 3]], dtype=np.int64)
    )
    builder: GEBModel = GEBModel(logger=logging.getLogger(__name__), root=tmp_path)
    catalog: Mock = Mock()
    catalog.fetch.return_value.read.return_value = gpd.GeoDataFrame(
        {"amber_id": ["amber-1"], "dam_type": [amber_type], "dam_hgt_m": [3.5]},
        geometry=[Point(amber_longitude, 0.0)],
        crs=4326,
    ).iloc[: 1 if has_amber_points else 0]
    monkeypatch.setattr(builder, "data_catalog", catalog)
    region_file: Path = tmp_path / "region.parquet"
    gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(1)], crs=4326).to_parquet(region_file)
    geometry_files["mask"] = region_file
    if with_waterbody_outline:
        waterbody_file: Path = tmp_path / "waterbody_data.parquet"
        gpd.GeoDataFrame(
            {"waterbody_id": [7]},
            geometry=[Point(amber_longitude, 0.0).buffer(0.0001)],
            crs=4326,
        ).to_parquet(waterbody_file)
        geometry_files["waterbodies/waterbody_data"] = waterbody_file
    dam_linked_to_waterbody: bool = with_waterbody_outline and amber_type == "Dam"
    builder.files = builder.read_or_create_file_library()
    builder.geom = geometry_files
    grids: dict[str, xr.DataArray] = {
        "mask": xr.zeros_like(upstream_area, dtype=bool),
        "routing/river_ids": xr.ones_like(upstream_area, dtype=np.int32),
        "waterbodies/waterbody_id": xr.full_like(upstream_area, -1, dtype=np.int32),
        "flow_raster_idxs_ds": downstream_cells,
        "routing/upstream_area_m2": upstream_area,
    }
    grids["mask"].values[1, 1] = True
    grid_name: str
    grid: xr.DataArray
    for grid_name, grid in grids.items():
        grid.attrs["_FillValue"] = (
            None if grid.dtype == bool else np.nan if grid.dtype.kind == "f" else -1
        )
        if lazy_grids:
            grid = grid.chunk({"x": 1, "y": 1})
        builder.set_grid(grid, name=grid_name)
    builder.set_other(upstream_area, name="drainage/original_d8_upstream_area_m2")
    builder.files["geom"] = dict(geometry_files)
    builder.write_file_library()

    # Use the same read, build and save path as an automatic input update.
    builder.update({"setup_weirs": {}})
    builder.update({"setup_weirs": {}})
    output_file: Path = tmp_path / "grid/routing/weir_height_m.zarr"
    saved_heights: xr.DataArray = read_zarr(output_file).compute()
    barrier_file: Path = tmp_path / "geom/routing/barriers.geoparquet"
    saved_barriers: gpd.GeoDataFrame = gpd.read_parquet(barrier_file)
    assert saved_barriers.crs.to_epsg() == 4326
    assert pd.api.types.is_string_dtype(saved_barriers["source"].dtype)
    assert pd.api.types.is_string_dtype(saved_barriers["barrier_id"].dtype)
    assert pd.api.types.is_string_dtype(saved_barriers["barrier_type"].dtype)
    assert saved_barriers.columns.tolist() == [
        "source",
        "barrier_id",
        "barrier_type",
        "height_m",
        "geometry",
        "distance_to_river_m",
        "included",
        "exclusion_reason",
    ]
    assert len(saved_barriers) == int(has_gdw_points) + int(has_amber_points)
    assert saved_barriers["included"].sum() == int(
        has_gdw_points
        or (
            has_amber_points
            and amber_longitude == 0.001
            and not dam_linked_to_waterbody
        )
    )
    if has_amber_points:
        amber_record: pd.Series = saved_barriers.loc[
            saved_barriers["source"] == "AMBER"
        ].iloc[0]
        assert amber_record["barrier_id"] == "amber-1"
        assert amber_record["height_m"] == 3.5
        assert amber_record.geometry == Point(amber_longitude, 0.0)
        if not dam_linked_to_waterbody:
            assert amber_record["distance_to_river_m"] > 0
        assert amber_record["exclusion_reason"] == (
            "part of lake or reservoir"
            if dam_linked_to_waterbody
            else "farther than 250 m"
            if amber_longitude == 0.003
            else "unsuitable river cell"
            if has_gdw_points
            else ""
        )
    saved_instream_dams: xr.DataArray = read_zarr(
        tmp_path / "grid/routing/instream_dam.zarr"
    ).compute()
    assert bool(saved_instream_dams.any()) == (
        has_amber_points
        and amber_type == "Dam"
        and amber_longitude == 0.001
        and not has_gdw_points
        and not dam_linked_to_waterbody
    )
    assert saved_heights.dtype == np.float32
    assert np.isnan(saved_heights.attrs["_FillValue"])
    assert np.isnan(saved_heights.values[1, 1])
    assert saved_heights.sum().item() == (
        -2.0
        if has_gdw_points
        else 3.5
        if has_amber_points and amber_longitude == 0.001 and not dam_linked_to_waterbody
        else 0.0
    )
    assert (
        "routing/weir_height_source"
        not in builder.read_or_create_file_library()["grid"]
    )
    assert saved_heights.values[1, 0] == 0.0
    assert builder.read_or_create_file_library()["grid"]["routing/weir_height_m"] == (
        "grid/routing/weir_height_m.zarr"
    )


@pytest.mark.parametrize("height_m", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_default_height(height_m: float) -> None:
    """Reject a default height that cannot be used.

    Args:
        height_m: Invalid default height (m).
    """
    grid: xr.DataArray = xr.DataArray([[-1]])
    points: gpd.GeoDataFrame = gpd.GeoDataFrame()
    with pytest.raises(ValueError, match="crest_height_m"):
        create_weir_grids(points, height_m, grid, points, grid, grid, grid)


@pytest.mark.parametrize(
    "case",
    [
        "weir",
        "dam",
        "amber_dam",
        "amber_dam_in_waterbody",
        "amber_dam_linked",
        "lock",
        "sluice",
        "fixed",
        "low_dam",
        "linked",
        "offstream",
        "boundary",
        "occupied",
        "distant",
        "empty",
        "lake_control",
        "inside",
        "height",
        "zero_height",
        "negative_height",
        "infinite_height",
    ],
)
@pytest.mark.parametrize("crest_height_m", [None, 1.0])
def test_build_weir(case: str, crest_height_m: float | None) -> None:
    """Use all points outside polygons, whatever their name or type.

    Args:
        case: Selection, overlap, or height scenario.
        crest_height_m: Missing-height override (m), or None for type-dependent heights.
    """
    barriers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": [42],
            "dam_type": ["Dam"],
            "dam_name": ["Example Weir"],
            "waterbody_id": [np.nan],
            "instream": ["Instream"],
            "lake_control": [None],
            "inside_gdw_polygon": [False],
            "dam_hgt_m": [np.nan],
        },
        geometry=[Point(1 if case == "distant" else 0.01, 0.01)],
        crs=4326,
    )
    if case == "inside":
        barriers["inside_gdw_polygon"] = True
    if case == "height":
        barriers["dam_hgt_m"] = 2.5
    if case == "zero_height":
        barriers["dam_hgt_m"] = 0.0
    if case == "negative_height":
        barriers["dam_hgt_m"] = -99.0
    if case == "infinite_height":
        barriers["dam_hgt_m"] = np.inf
    if case in {"linked", "amber_dam_linked"}:
        barriers["waterbody_id"] = 1
    if case == "dam":
        barriers["dam_name"] = None
    if case == "lock":
        barriers["dam_type"] = "Lock"
    if case == "sluice":
        barriers["dam_type"] = "Sluice"
    if case == "fixed":
        barriers["dam_type"] = "Weir"
    if case == "low_dam":
        barriers["dam_type"] = "Low Permeable Dam"
    if case == "offstream":
        barriers["instream"] = "Offstream"
    if case == "lake_control":
        barriers["dam_type"] = "Lake Control Dam"
        barriers["lake_control"] = "Yes"
    if case == "distant":
        barriers.geometry = [Point(1, 0.01)]
    if case == "empty":
        barriers = gpd.GeoDataFrame()
    upstream_area: xr.DataArray = xr.DataArray(
        np.full((2, 2), 1000.0, dtype=np.float32),
        coords={"y": [0.0, -0.1], "x": [0.0, 0.1]},
        dims=("y", "x"),
    )
    waterbody_ids: xr.DataArray = xr.full_like(upstream_area, -1, dtype=np.int32)
    river_cells: xr.DataArray = xr.full_like(upstream_area, True, dtype=bool)
    if case == "boundary":
        river_cells.values[0, 0] = False
    if case in {"occupied", "amber_dam_in_waterbody"}:
        waterbody_ids.values[0, 0] = 1
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "represented_in_grid": [True],
            "uparea_m2": [1000.0],
            "hydrography_xy": [[(0, 0), (0, 1)]],
            "shreve_stream_order": [1],
        },
        index=pd.Index([1], dtype="int64"),
        geometry=[LineString([(0, 0), (0, -0.1)])],
        crs=4326,
    )
    amber_points: gpd.GeoDataFrame | None = None
    if case.startswith("amber_dam"):
        barriers.geometry = [Point(0.001, 0.0)]
        amber_points = gpd.GeoDataFrame(
            barriers.rename(columns={"gdw_id": "amber_id"}), crs=barriers.crs
        )
        barriers = gpd.GeoDataFrame()
    result: xr.DataArray
    instream_dams: xr.DataArray
    records: gpd.GeoDataFrame
    result, instream_dams, records = create_weir_grids(
        barriers,
        crest_height_m,
        waterbody_ids,
        rivers,
        upstream_area,
        upstream_area,
        river_cells,
        amber_points=amber_points,
    )
    if case.startswith("amber_dam"):
        assert instream_dams.values.any() == (case == "amber_dam")
        assert result.values.sum() == (
            (crest_height_m if crest_height_m is not None else -2.0)
            if case == "amber_dam"
            else 0.0
        )
        if case in {"amber_dam_in_waterbody", "amber_dam_linked"}:
            assert records.iloc[0]["exclusion_reason"] == "part of lake or reservoir"
        return
    assert not instream_dams.values.any()
    assert len(records) == (0 if case == "empty" else 1)
    assert int(records["included"].sum()) == int(bool((result.values != 0).any()))
    if case == "inside":
        assert records.iloc[0]["exclusion_reason"] == "inside reservoir outline"
    elif case in {"linked", "occupied"}:
        assert records.iloc[0]["exclusion_reason"] == "part of lake or reservoir"
    elif case == "boundary":
        assert records.iloc[0]["exclusion_reason"] == "unsuitable river cell"
    elif case == "distant":
        assert records.iloc[0]["exclusion_reason"] == "no river found"
    if not records.empty:
        assert records.iloc[0]["source"] == "GDW"
        assert records.iloc[0]["barrier_id"] == "42"
    assert result.dtype == np.float32
    if case == "height":
        assert result.values.sum() == 2.5
    elif case in (
        "weir",
        "dam",
        "lock",
        "sluice",
        "fixed",
        "low_dam",
        "offstream",
        "lake_control",
        "zero_height",
        "negative_height",
        "infinite_height",
    ):
        if crest_height_m is not None:
            assert result.values.sum() == crest_height_m
        else:
            assert result.values.sum() == -2.0
    else:
        assert result.values.sum() == 0.0


@pytest.mark.parametrize(
    "dam_type",
    [
        "Dam",
        "Lock",
        "Weir",
        "Low Permeable Dam",
        "Sluice",
        "Lake Control Dam",
        "Culvert",
        "Ford",
        "Ramp",
        "Other",
        "",
    ],
)
@pytest.mark.parametrize("source", ["GDW", "AMBER"])
@pytest.mark.parametrize(
    "recorded_height_m", [np.nan, 3.5, 0.0, -99.0, np.inf, -np.inf]
)
@pytest.mark.parametrize("crest_height_m", [None, 1.25])
def test_structure_classification(
    dam_type: str,
    source: str,
    recorded_height_m: float,
    crest_height_m: float | None,
) -> None:
    """Apply source-specific type and height rules to fixed barriers.

    Args:
        dam_type: Barrier type shared by two test points.
        source: Atlas providing the two structures.
        recorded_height_m: Source height (m), or NaN when missing.
        crest_height_m: Explicit height override (m), or None for bankfull defaults.

    Returns:
        None.

    Raises:
        AssertionError: If inclusion or heights are incorrect.
    """  # noqa: DOC202, DOC502
    barriers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": [42, 43],
            "dam_type": [dam_type, dam_type],
            "waterbody_id": [np.nan, np.nan],
            "inside_gdw_polygon": [False, False],
            "dam_hgt_m": [recorded_height_m, recorded_height_m],
        },
        geometry=[Point(0.001, 0.0), Point(0.001, -0.1)],
        crs=4326,
    )
    upstream_area: xr.DataArray = xr.DataArray(
        np.full((2, 2), 1000.0, dtype=np.float32),
        coords={"y": [0.0, -0.1], "x": [0.0, 0.1]},
        dims=("y", "x"),
    )
    waterbody_ids: xr.DataArray = xr.full_like(upstream_area, -1, dtype=np.int32)
    river_cells: xr.DataArray = xr.full_like(upstream_area, True, dtype=bool)
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "represented_in_grid": [True],
            "uparea_m2": [1000.0],
            "hydrography_xy": [[(0, 0), (0, 1)]],
            "shreve_stream_order": [1],
        },
        index=pd.Index([1], dtype="int64"),
        geometry=[LineString([(0, 0), (0, -0.1)])],
        crs=4326,
    )
    heights: xr.DataArray
    records: gpd.GeoDataFrame
    amber_points: gpd.GeoDataFrame | None = None
    if source == "AMBER":
        amber_points = gpd.GeoDataFrame(
            barriers.rename(columns={"gdw_id": "amber_id"}),
            geometry="geometry",
            crs=barriers.crs,
        )
        barriers = gpd.GeoDataFrame()
    heights, instream_dams, records = create_weir_grids(
        barriers,
        crest_height_m,
        waterbody_ids,
        rivers,
        upstream_area,
        upstream_area,
        river_cells,
        amber_points=amber_points,
    )
    is_supported: bool = source == "GDW" or dam_type not in {
        "Lake Control Dam",
        "Culvert",
    }
    assert records["included"].all() == is_supported
    assert instream_dams.values.any() == (source == "AMBER" and dam_type == "Dam")
    if not is_supported:
        expected_reason: str = (
            "AMBER culvert excluded"
            if dam_type == "Culvert"
            else "AMBER dam classification only"
        )
        assert (records["exclusion_reason"] == expected_reason).all()
    expected_height: float
    if not is_supported:
        expected_height = 0.0
    elif np.isfinite(recorded_height_m) and recorded_height_m > 0:
        expected_height = recorded_height_m
    elif crest_height_m is not None:
        expected_height = crest_height_m
    elif source == "AMBER" and dam_type not in {"Dam", "Weir", "Sluice", "Lock"}:
        expected_height = USE_HALF_BANKFULL_HEIGHT
    else:
        expected_height = USE_BANKFULL_HEIGHT
    np.testing.assert_array_equal(
        heights.values[:, 0], expected_height, err_msg=records.to_string()
    )


@pytest.mark.parametrize("cell_count", [1, 2, 4])
@pytest.mark.parametrize("alternative_status", ["free", "invalid", "waterbody"])
def test_occupied_weir_cell_fallback(cell_count: int, alternative_status: str) -> None:
    """Try one alternative cell without overwriting barriers or searching further.

    Args:
        cell_count: Number of distinct routing cells on the river segment.
        alternative_status: Whether the second cell is free, invalid or a waterbody.

    Returns:
        None.

    Raises:
        AssertionError: If placement overwrites a weir or checks a third cell.
    """  # noqa: DOC202, DOC502
    barriers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": [1, 2, 3, 4],
            "dam_type": ["Weir", "Sluice", "Dam", "Weir"],
            "waterbody_id": [np.nan] * 4,
            "inside_gdw_polygon": [False] * 4,
            "dam_hgt_m": [1.0, 2.0, 3.0, 4.0],
        },
        geometry=[Point(0.0, 0.0)] * 4,
        crs=4326,
    )
    upstream_area: xr.DataArray = xr.DataArray(
        np.full((4, 2), 1000.0, dtype=np.float32),
        coords={"y": [0.0, -0.001, -0.002, -0.003], "x": [0.0, 0.001]},
        dims=("y", "x"),
    )
    waterbody_ids: xr.DataArray = xr.full_like(upstream_area, -1, dtype=np.int32)
    valid_cells: xr.DataArray = xr.full_like(upstream_area, True, dtype=bool)
    if alternative_status == "invalid":
        valid_cells.values[1, 0] = False
    elif alternative_status == "waterbody":
        waterbody_ids.values[1, 0] = 7
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "represented_in_grid": [True],
            "uparea_m2": [1000.0],
            "hydrography_xy": [[(0, row) for row in range(cell_count)]],
            "shreve_stream_order": [1],
        },
        index=pd.Index([1], dtype="int64"),
        geometry=[LineString([(0.0, 0.0), (0.0, -0.002)])],
        crs=4326,
    )
    heights: xr.DataArray
    records: gpd.GeoDataFrame
    heights, instream_dams, records = create_weir_grids(
        barriers, None, waterbody_ids, rivers, upstream_area, upstream_area, valid_cells
    )
    alternative_available: bool = cell_count > 1 and alternative_status == "free"
    np.testing.assert_array_equal(
        heights.values[:, 0],
        [
            1.0,
            2.0 if alternative_available else 0.0,
            0.0,
            0.0,
        ],
    )
    assert records["included"].tolist() == [
        True,
        alternative_available,
        False,
        False,
    ]
    expected_reason: str = (
        "cell already used"
        if cell_count == 1 or alternative_status == "free"
        else "unsuitable river cell"
        if alternative_status == "invalid"
        else "part of lake or reservoir"
    )
    assert records.iloc[3]["exclusion_reason"] == expected_reason
