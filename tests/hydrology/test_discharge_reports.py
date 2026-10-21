"""Regression tests for discharge reports across changing river representations."""

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geb.hydrology.routing import get_discharge_per_river


@pytest.mark.parametrize("source", ["memory", "consolidated", "legacy"])
@pytest.mark.parametrize("recorded_branches", [1, 2])
def test_recorded_layout(source: str, recorded_branches: int, tmp_path: Path) -> None:
    """Read historical branches independently of current river representation.

    Args:
        source: Report storage format.
        recorded_branches: Number of branches at reporting initialization.
        tmp_path: Temporary report directory.
    """
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(index=[24027753])
    times: pd.DatetimeIndex = pd.date_range("1990-01-01", periods=3, freq="h")
    reports: dict[str, Any] = {}
    columns: dict[str, np.ndarray] = {}
    prefix: str = "river_outflow_hourly_m3_per_s_"
    branch: int
    for branch in range(recorded_branches):
        key: str = "24027753" + (f"_{branch}" if recorded_branches > 1 else "")
        values: np.ndarray = np.array([1.0, 2.0, 999.0]) * (branch + 1)
        reports[f"{prefix}{key}"] = {
            "_data_array": values,
            "_time_array": times.to_numpy().astype("datetime64[s]").astype(np.int64),
            "_var_index": 2,
        }
        columns[key] = values[:2]
        if source == "legacy":
            pd.DataFrame({f"{prefix}{key}": values[:2]}, index=times[:2]).to_parquet(
                tmp_path / f"{prefix}{key}.parquet"
            )
    if source == "consolidated":
        pd.DataFrame(columns, index=times[:2]).to_parquet(
            tmp_path / "river_outflow_hourly_m3_per_s.parquet"
        )
    result: pd.DataFrame = get_discharge_per_river(
        rivers,
        source="memory" if source == "memory" else "file",
        folder=tmp_path,
        variables_to_report=reports,
    )
    expected: np.ndarray = np.array([1.0, 2.0]) * sum(range(1, recorded_branches + 1))
    np.testing.assert_array_equal(result[24027753], expected)
    np.testing.assert_array_equal(result.index.to_numpy(), times[:2].to_numpy())
