"""Tests for dashboard station exclusion checks."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geb.evaluate.workflows.discharge_station_checks import (
    find_dashboard_excluded_stations,
)


@pytest.mark.parametrize(
    ("area_values", "area_dtype", "missing_area"),
    [
        ([], "object", []),
        ([1e6, 2e6], "float64", [False, False]),
        ([1e6, None], "object", [False, True]),
        (
            ["1000000", "invalid", "0", "-1", "inf"],
            "object",
            [False, True, True, True, True],
        ),
        ([1e6, pd.NA, np.inf], "Float64", [False, True, True]),
    ],
    ids=["empty-inventory", "numeric", "object", "strings", "nullable"],
)
def test_dashboard_station_upstream_areas(
    tmp_path: Path,
    area_values: list[float | str | None],
    area_dtype: str,
    missing_area: list[bool],
) -> None:
    """Handle empty inventories and exclude stations with unusable areas.

    Args:
        tmp_path: Temporary directory for the station inventory.
        area_values: Station upstream areas (m²), including invalid entries.
        area_dtype: Pandas dtype used for the stored upstream areas.
        missing_area: Whether each station should have a missing-area reason.
    """
    inventory: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "discharge_observations_station_name": ["Station"] * len(area_values),
            "discharge_observations_upstream_area_m2": pd.Series(
                area_values, dtype=area_dtype
            ),
            "evaluation_exclusion_reason": [""] * len(area_values),
        },
        geometry=[Point(8, 50)] * len(area_values),
        crs="EPSG:4326",
    )
    inventory.to_parquet(tmp_path / "station_locations.geoparquet")
    empty_stations: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"GEB_upstream_area_from_grid": pd.Series(dtype=float)},
        geometry=gpd.GeoSeries(crs=inventory.crs),
        crs=inventory.crs,
    )

    result: gpd.GeoDataFrame = find_dashboard_excluded_stations(
        evaluated=empty_stations,
        excluded=empty_stations,
        snapped=empty_stations,
        snapped_path=tmp_path / "snapped.geoparquet",
        minimum_upstream_area_km2=0.0,
    )

    assert result.index.tolist() == inventory.index.tolist()
    assert result.crs == inventory.crs
    assert result["exclusion_reason"].tolist() == [
        "Missing station upstream area."
        if is_missing
        else "No valid river match within 1.5 km and 10% upstream area."
        for is_missing in missing_area
    ]
