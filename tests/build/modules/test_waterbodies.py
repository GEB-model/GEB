"""Check waterbody preprocessing through the build method."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, box

from geb.build.modules.hydrography import Hydrography
from geb.build.workflows.waterbodies_preprocessing import GDW_ID_OFFSET
from geb.workflows.raster import full_like


@pytest.mark.parametrize(
    ("mode", "expected_ids"),
    [
        ("on", [10, GDW_ID_OFFSET + 2]),
        ("reservoirs_only", [GDW_ID_OFFSET + 2]),
        ("lakes_only", [10]),
        ("off", []),
    ],
)
def test_setup_waterbodies(
    tmp_path: Path,
    mode: str,
    expected_ids: list[int],
) -> None:
    """Check GDW loading, waterbody modes, and custom capacities.

    Args:
        tmp_path: Test output directory.
        mode: Build mode to check.
        expected_ids: IDs that should remain after filtering.
    """
    lakes: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "waterbody_id": [10],
            "waterbody_type": [1],
            "volume_total": [100.0],
            "average_area": [10.0],
            "average_discharge": [1.0],
        },
        geometry=[box(0, 0, 1, 1)],
        crs=4326,
    )
    dams: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": [1, 2],
            "hydrolakes_id": pd.array([10, None], dtype="Int64"),
            "dam_type": ["Dam", "Dam"],
            "lake_control": [None, None],
            "capacity_m3": [900.0, 800.0],
            "area_m2": [50.0, 60.0],
            "average_discharge_m3_per_s": [5.0, 6.0],
            "construction_year": [1980, 1990],
        },
        # Keep the first ID match even though its dam is outside the model region.
        geometry=[Point(9, 9), Point(5, 5)],
        crs=4326,
    )
    reservoir_shapes: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"gdw_id": [2]}, geometry=[box(4, 4, 6, 6)], crs=4326
    )
    adapters: dict[str, Mock] = {
        "hydrolakes": Mock(read=Mock(return_value=lakes)),
        "gdw_barriers": Mock(read=Mock(return_value=dams)),
        "gdw_reservoirs": Mock(read=Mock(return_value=reservoir_shapes)),
    }
    grid: xr.DataArray = xr.DataArray(
        np.zeros((8, 8), dtype=bool),
        dims=("y", "x"),
        coords={"y": np.arange(7.5, -0.5, -1), "x": np.arange(0.5, 8.5)},
    ).rio.write_crs(4326)
    model: SimpleNamespace = SimpleNamespace(
        data_catalog=Mock(fetch=Mock(side_effect=adapters.__getitem__)),
        region=gpd.GeoDataFrame(geometry=[box(0, 0, 8, 8)], crs=4326),
        root=tmp_path,
        logger=Mock(),
        grid=xr.Dataset({"mask": grid}),
        subgrid=xr.Dataset({"mask": grid}),
        full_like=full_like,
        set_geom=Mock(),
        set_grid=Mock(),
        set_subgrid=Mock(),
    )
    capacity_file: Path = tmp_path / "capacity.csv"
    pd.DataFrame(
        {"waterbody_id": [GDW_ID_OFFSET + 2], "volume_total": [1234.0]}
    ).to_csv(capacity_file, index=False)
    Hydrography.setup_waterbodies.__wrapped__(
        model,
        mode=mode,
        custom_reservoir_capacity=str(capacity_file)
        if mode in ("on", "reservoirs_only")
        else None,
    )
    saved_waterbodies: gpd.GeoDataFrame = model.set_geom.call_args.args[0]
    assert model.set_geom.call_args.kwargs["name"] == "waterbodies/waterbody_data"
    assert saved_waterbodies["waterbody_id"].tolist() == expected_ids
    assert saved_waterbodies["waterbody_type"].dtype == np.int32
    if mode == "off":
        assert model.data_catalog.fetch.call_count == 1
        return
    dam_checks: pd.DataFrame = pd.read_csv(
        tmp_path / "reports/waterbodies/gdw_checks.csv"
    )
    assert dam_checks["construction_year"].tolist() == [1980, 1990]
    assert dam_checks["point_outside_lake"].tolist() == [True, False]
    assert dam_checks["type_check"].tolist() == [
        "type_1_with_barrier",
        "no_hydrolakes_record",
    ]
    if mode in ("on", "reservoirs_only"):
        added_reservoir: pd.Series = saved_waterbodies.set_index("waterbody_id").loc[
            GDW_ID_OFFSET + 2
        ]
        assert (
            added_reservoir["volume_total"] == added_reservoir["volume_flood"] == 1234
        )
        assert added_reservoir["gdw_capacity_m3"] == 800
    assert set(np.unique(model.set_grid.call_args_list[0].args[0])) == {
        -1,
        *expected_ids,
    }
