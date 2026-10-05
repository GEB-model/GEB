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
from geb.workflows.io import read_geom, read_zarr


@pytest.mark.parametrize("amber_longitude", [0.001, 0.003])
@pytest.mark.parametrize("has_amber_points", [True, False])
@pytest.mark.parametrize("has_gdw_points", [True, False])
@pytest.mark.parametrize("lazy_grids", [True, False])
def test_setup_weirs_from_files(
    tmp_path: Path,
    has_gdw_points: bool,
    has_amber_points: bool,
    amber_longitude: float,
    lazy_grids: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read saved GDW points during an update; allow builds without GDW points.

    Args:
        tmp_path: Folder for test files.
        has_gdw_points: Whether a GDW points file is available.
        has_amber_points: Whether the atlas contains a barrier in this region.
        amber_longitude: Barrier longitude (degrees), within or beyond 250 m.
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
        {"amber_id": ["amber-1"], "dam_type": ["Weir"], "dam_hgt_m": [3.5]},
        geometry=[Point(amber_longitude, 0.0)],
        crs=4326,
    ).iloc[: 1 if has_amber_points else 0]
    monkeypatch.setattr(builder, "data_catalog", catalog)
    region_file: Path = tmp_path / "region.parquet"
    gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(1)], crs=4326).to_parquet(region_file)
    geometry_files["mask"] = region_file
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
    saved_gates: xr.DataArray = read_zarr(
        tmp_path / "grid/routing/weir_gate.zarr"
    ).compute()
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
        has_gdw_points or (has_amber_points and amber_longitude == 0.001)
    )
    if has_amber_points:
        amber_record: pd.Series = saved_barriers.loc[
            saved_barriers["source"] == "AMBER"
        ].iloc[0]
        assert amber_record["barrier_id"] == "amber-1"
        assert amber_record["height_m"] == 3.5
        assert amber_record.geometry == Point(amber_longitude, 0.0)
        assert amber_record["distance_to_river_m"] > 0
        assert amber_record["exclusion_reason"] == (
            "farther than 250 m"
            if amber_longitude == 0.003
            else "cell already used"
            if has_gdw_points
            else ""
        )
    assert saved_gates.dtype == bool
    assert saved_gates.values[0, 0] == has_gdw_points
    assert saved_gates.values.sum() == int(has_gdw_points)
    assert saved_heights.dtype == np.float32
    assert np.isnan(saved_heights.attrs["_FillValue"])
    assert np.isnan(saved_heights.values[1, 1])
    assert saved_heights.sum().item() == (
        -2.0
        if has_gdw_points
        else 3.5
        if has_amber_points and amber_longitude == 0.001
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
    if case == "linked":
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
    if case == "occupied":
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
    result: xr.DataArray
    gates: xr.DataArray
    records: gpd.GeoDataFrame
    result, gates, records = create_weir_grids(
        barriers,
        crest_height_m,
        waterbody_ids,
        rivers,
        upstream_area,
        upstream_area,
        river_cells,
    )
    assert gates.values.sum() == int(
        case
        in (
            "weir",
            "dam",
            "sluice",
            "offstream",
            "lake_control",
            "height",
            "zero_height",
            "negative_height",
            "infinite_height",
        )
    )
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
            expected_height: float
            if case in {"lock", "sluice", "fixed", "low_dam"}:
                expected_height = -2.0
            else:
                expected_height = -1.0
            assert result.values.sum() == expected_height
    else:
        assert result.values.sum() == 0.0


@pytest.mark.parametrize(
    "dam_type",
    ["Dam", "Lock", "Weir", "Low Permeable Dam", "Sluice", "Lake Control Dam", ""],
)
def test_gate_classification(dam_type: str) -> None:
    """Classify flow-control types independently of their default heights.

    Args:
        dam_type: GDW dam type shared by two test points.

    Returns:
        None.

    Raises:
        AssertionError: If gate flags or heights are incorrect.
    """  # noqa: DOC202, DOC502
    barriers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": [42, 43],
            "dam_type": [dam_type, dam_type],
            "waterbody_id": [np.nan, np.nan],
            "inside_gdw_polygon": [False, False],
            "dam_hgt_m": [np.nan, np.nan],
        },
        geometry=[Point(0.01, 0.01), Point(0.01, -0.09)],
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
    gates: xr.DataArray
    records: gpd.GeoDataFrame
    heights, gates, records = create_weir_grids(
        barriers,
        None,
        waterbody_ids,
        rivers,
        upstream_area,
        upstream_area,
        river_cells,
    )
    is_gated: bool = dam_type in {"Dam", "Sluice", "Lake Control Dam"}
    expected_height: float
    if dam_type in {"Dam", "Lake Control Dam"}:
        expected_height = -1.0
    else:
        expected_height = -2.0
    assert gates.values.sum() == 2 * int(is_gated)
    np.testing.assert_array_equal(heights.values[:, 0], expected_height)


@pytest.mark.parametrize("option_name", ["open_depth_m", "close_depth_m"])
def test_fixed_gate_depth_options_removed(option_name: str) -> None:
    """Reject removed fixed-depth build options.

    Args:
        option_name: Removed gate depth setting.

    Returns:
        None.

    Raises:
        AssertionError: If a removed option is accepted.
    """  # noqa: DOC202, DOC502
    grid: xr.DataArray = xr.DataArray([[-1]])
    points: gpd.GeoDataFrame = gpd.GeoDataFrame()
    with pytest.raises(TypeError, match=option_name):
        create_weir_grids(
            points,
            None,
            grid,
            points,
            grid,
            grid,
            grid,
            **{option_name: 1.0},  # ty:ignore[invalid-argument-type]
        )
