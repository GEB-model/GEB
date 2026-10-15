"""Check metric placement of AMBER weirs and the setup input grids."""

import logging
from pathlib import Path
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Geod
from shapely.geometry import LineString, Point, box

from geb.build import GEBModel, version_updates
from geb.build.workflows.weirs import snap_amber_weirs
from geb.workflows.io import read_zarr


def make_inputs(
    distance_m: float = 100.0, latitude: float = 52.0
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, xr.DataArray]:
    """Create a weir and a two-cell river with a known geodesic distance.

    Args:
        distance_m: Distance east from the source weir to the river (meters).
        latitude: Source latitude (degrees).

    Returns:
        AMBER points, represented river lines and the valid connection grid.

    Raises:
        ValueError: If fixture coordinates cannot be projected.
    """  # noqa: DOC502
    longitude: float
    river_latitude: float
    longitude, river_latitude, _ = Geod(ellps="WGS84").fwd(
        0.0, latitude, 90.0, distance_m
    )
    points: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"amber_id": ["a"], "dam_type": ["Weir"], "dam_hgt_m": [2.0]},
        geometry=[Point(0.0, latitude)],
        crs=4326,
    )
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"represented_in_grid": [True], "hydrography_xy": [[(0, 0), (1, 0)]]},
        index=pd.Index([7], dtype="int64"),
        geometry=[
            LineString(
                [(longitude, river_latitude), (longitude + 0.01, river_latitude)]
            )
        ],
        crs=4326,
    )
    valid: xr.DataArray = xr.DataArray(
        [[True, False], [False, False]],
        dims=("y", "x"),
        coords={
            "x": [longitude, longitude + 0.01],
            "y": [river_latitude, river_latitude - 0.01],
        },
    ).rio.write_crs(4326)
    return points, rivers, valid


@pytest.mark.parametrize("latitude", [0.0, 52.0, 70.0])
@pytest.mark.parametrize("distance_m", [0.0, 249.999, 250.0, 250.001, 1000.0])
def test_metric_distance_limit(latitude: float, distance_m: float) -> None:
    """Apply the same 250-meter line-distance limit at different latitudes.

    Args:
        latitude: Source latitude (degrees).
        distance_m: Known source-to-line distance (meters).

    Returns:
        None.

    Raises:
        AssertionError: If distance or grid assignment is incorrect.
    """  # noqa: DOC202, DOC502
    points: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    valid: xr.DataArray
    points, rivers, valid = make_inputs(distance_m, latitude)
    fractions: xr.DataArray
    records: gpd.GeoDataFrame
    fractions, records = snap_amber_weirs(points, rivers, valid)
    assert records.included.tolist() == [distance_m <= 250.0]
    assert records.distance_to_river_m.iloc[0] == pytest.approx(distance_m, abs=1e-5)
    assert fractions.values[0, 0] == (0.5 if distance_m <= 250.0 else 0.0)
    assert records.geometry.iloc[0] == points.geometry.iloc[0]
    if distance_m <= 250.0:
        assert records.grid_column.iloc[0] == records.grid_row.iloc[0] == 0
        assert records.snapped_longitude.iloc[0] == valid.x.values[0]
    else:
        assert records.grid_column.iloc[0] == records.grid_row.iloc[0] == -1


@pytest.mark.parametrize(
    "case",
    [
        "other_type",
        "no_rivers",
        "unsuitable",
        "duplicate",
        "no_cells",
    ],
)
def test_excluded_weirs(case: str) -> None:
    """Reject unsupported and ambiguous records without relocating structures.

    Args:
        case: Exclusion scenario applied to otherwise valid inputs.

    Returns:
        None.

    Raises:
        AssertionError: If an excluded weir changes the height grid.
    """  # noqa: DOC202, DOC502
    points: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    valid: xr.DataArray
    points, rivers, valid = make_inputs()
    if case == "other_type":
        points["dam_type"] = "Dam"
    elif case == "no_rivers":
        rivers = rivers.iloc[:0]
    elif case == "unsuitable":
        valid.values[:] = False
    elif case == "duplicate":
        points = gpd.GeoDataFrame(
            pd.concat([points, points.assign(amber_id="b")], ignore_index=True),
            crs=4326,
        )
    elif case == "no_cells":
        rivers.at[7, "hydrography_xy"] = []
    fractions: xr.DataArray
    records: gpd.GeoDataFrame
    fractions, records = snap_amber_weirs(points, rivers, valid)
    assert not fractions.values.any()
    assert not records.included.any()
    assert records.exclusion_reason.ne("").all()
    assert records.grid_row.eq(-1).all()
    if case == "duplicate":
        assert records.exclusion_reason.eq("multiple weirs on one connection").all()


@pytest.mark.parametrize("case", ["missing_crs", "invalid_grid", "invalid_cells"])
def test_invalid_inputs(case: str) -> None:
    """Fail clearly when coordinates or grid indices cannot be interpreted.

    Args:
        case: Invalid input to supply.

    Returns:
        None.

    Raises:
        AssertionError: If invalid inputs do not raise a ValueError.
    """  # noqa: DOC202, DOC502
    points: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    valid: xr.DataArray
    points, rivers, valid = make_inputs()
    if case == "missing_crs":
        rivers.crs = None
    elif case == "invalid_grid":
        valid = valid.astype(np.int32)
    else:
        rivers.at[7, "hydrography_xy"] = [(5, 0)]
    with pytest.raises(ValueError):
        snap_amber_weirs(points, rivers, valid)


@pytest.mark.parametrize("case", ["valid", "waterbody", "masked", "outlet", "empty"])
def test_setup_weirs_saved_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Build and reload height and diagnostic inputs, including lazy grids.

    Args:
        tmp_path: Directory for model inputs.
        monkeypatch: Fixture replacing the atlas download.
        case: Whether the downstream connection supports a weir.

    Returns:
        None.

    Raises:
        AssertionError: If persisted grids or connection exclusions are incorrect.
    """  # noqa: DOC202, DOC502
    points: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    valid: xr.DataArray
    points, rivers, valid = make_inputs()
    builder: GEBModel = GEBModel(logger=logging.getLogger(__name__), root=tmp_path)
    builder.files = builder.read_or_create_file_library()
    builder.set_geom(
        gpd.GeoDataFrame(geometry=[box(-1, 50, 1, 54)], crs=4326), name="mask"
    )
    builder.set_geom(rivers, name="routing/rivers")
    catalog: Mock = Mock()
    catalog.fetch.return_value.read.return_value = (
        points.iloc[:0] if case == "empty" else points
    )
    monkeypatch.setattr(builder, "data_catalog", catalog)
    mask: xr.DataArray = xr.zeros_like(valid, dtype=bool)
    waterbodies: xr.DataArray = xr.full_like(valid, -1, dtype=np.int32)
    downstream: xr.DataArray = xr.DataArray(
        [[1, 1], [2, 3]],
        dims=valid.dims,
        coords=valid.coords,
    ).rio.write_crs(4326)
    if case == "waterbody":
        waterbodies.values[0, 1] = 42
    elif case == "masked":
        mask.values[0, 1] = True
    elif case == "outlet":
        downstream.values[0, 0] = 0
    grids: dict[str, xr.DataArray] = {
        "mask": mask,
        "waterbodies/waterbody_id": waterbodies,
        "routing/river_ids": xr.full_like(valid, 7, dtype=np.int32),
        "flow_raster_idxs_ds": downstream,
    }
    name: str
    grid: xr.DataArray
    for name, grid in grids.items():
        grid.attrs["_FillValue"] = None if grid.dtype == bool else -1
        builder.set_grid(grid.chunk({"x": 1, "y": 1}), name=name)
    builder.write_file_library()
    builder.update({"setup_weirs": {}})
    builder.update({"setup_weirs": {}})
    fractions: xr.DataArray = read_zarr(
        tmp_path / "grid/routing/weir_bankfull_depth_fraction.zarr"
    )
    records: gpd.GeoDataFrame = gpd.read_parquet(
        tmp_path / "geom/routing/weirs.geoparquet"
    )
    assert bool((fractions.values > 0).any()) == (case == "valid")
    assert records.included.tolist() == ([] if case == "empty" else [case == "valid"])
    assert records.crs.to_epsg() == 4326
    if case == "valid":
        assert fractions.values[0, 0] == 0.5
        assert records.bankfull_depth_fraction.iloc[0] == 0.5
        assert records.dam_hgt_m.iloc[0] == 2.0


def test_weir_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add snapped inputs through the shared development migration.

    Args:
        monkeypatch: Fixture setting the target package version.

    Returns:
        None.

    Raises:
        AssertionError: If migration omits setup_weirs.
    """  # noqa: DOC202, DOC502
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b37")
    builder: Mock = Mock()
    version_updates.get_and_maybe_do_version_updates(
        "1.0.0b36", logging.getLogger(__name__), builder, {"setup_weirs": {}}
    )
    builder.update.assert_called_once_with({"setup_weirs": {}})
    builder.set_version.assert_called_once_with("1.0.0b37")


def test_distance_limit_applies_to_line() -> None:
    """Allow a nearby river even when its coarse model-cell center is farther away.

    Returns:
        None.

    Raises:
        AssertionError: If the line-distance rule is incorrectly applied to grid centers.
    """  # noqa: DOC202, DOC502
    points: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    valid: xr.DataArray
    points, rivers, valid = make_inputs()
    points.geometry = [Point(0.005, 52.0)]
    rivers.geometry = [LineString([(-0.01, 52.0), (0.02, 52.0)])]
    valid = valid.assign_coords(x=[-0.01, 0.02], y=[52.0, 51.99])
    valid.values[0, :] = True
    fractions: xr.DataArray
    records: gpd.GeoDataFrame
    fractions, records = snap_amber_weirs(points, rivers, valid)
    assert records.included.tolist() == [True]
    assert records.distance_to_river_m.iloc[0] < 1.0
    assert fractions.values.sum() == 0.5
    assert (
        Geod(ellps="WGS84").inv(
            points.geometry.iloc[0].x,
            points.geometry.iloc[0].y,
            records.snapped_longitude.iloc[0],
            records.snapped_latitude.iloc[0],
        )[2]
        > 250.0
    )


@pytest.mark.parametrize("recorded_height", [None, np.nan, -2.0, 0.0, 2.0, 100.0])
def test_recorded_heights_are_ignored(recorded_height: float | None) -> None:
    """Include weirs regardless of the recorded height using the assigned height.

    Args:
        recorded_height: Source height (meters), including absent or invalid values.

    Returns:
        None.

    Raises:
        AssertionError: If source height affects placement or the assigned height.
    """  # noqa: DOC202, DOC502
    points: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    valid: xr.DataArray
    points, rivers, valid = make_inputs()
    points["dam_hgt_m"] = recorded_height
    fractions: xr.DataArray
    records: gpd.GeoDataFrame
    fractions, records = snap_amber_weirs(points, rivers, valid)
    assert records.included.tolist() == [True]
    assert fractions.values[0, 0] == records.bankfull_depth_fraction.iloc[0] == 0.5
    # Source height is metadata, so its entire column may be absent as well.
    fractions, records = snap_amber_weirs(
        gpd.GeoDataFrame(points.drop(columns="dam_hgt_m")), rivers, valid
    )
    assert records.included.tolist() == [True]
    assert fractions.values[0, 0] == records.bankfull_depth_fraction.iloc[0] == 0.5
