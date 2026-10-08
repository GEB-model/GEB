"""Verify exact barrier/report joins and the exported AMBER page."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from geb.evaluate.workflows.waterworks_dashboard import (
    build_waterworks_payload,
    write_waterworks_dashboard,
)


def waterworks_fixture(folder: Path) -> gpd.GeoDataFrame:
    """Write runtime metadata and two hourly snapshots for three AMBER objects.

    Args:
        folder: Temporary simulation folder.

    Returns:
        Catalog with a gated dam, a fixed sluice and an excluded structure.

    Raises:
        OSError: If fixture files cannot be written.
    """  # noqa: DOC502
    metadata: pd.DataFrame = pd.DataFrame(
        {
            "grid_row": [2, 3],
            "grid_column": [4, 5],
            "grid_cell_index": [17, 19],
            "instream_dam": [True, False],
            "effective_height_m": [2.0, 1.0],
            "latitude_deg": [51.0, 51.1],
            "longitude_deg": [5.0, 5.1],
        }
    )
    metadata.to_csv(folder / "weir_heights.csv", index=False)
    report_folder: Path = folder / "report" / "hydrology.routing"
    report_folder.mkdir(parents=True, exist_ok=True)
    times: pd.DatetimeIndex = pd.date_range("2020-01-01 00:30", periods=2, freq="h")
    quantity: str
    values: list[float]
    for quantity, values in {
        "open_fraction": [0.1, 1.0],
        "outflow_m3_s": [2.0, 8.0],
        "inflow_m3_s": [3.0, 9.0],
    }.items():
        pd.DataFrame(
            {
                "17": values,
                "19": [-1.0, -1.0] if quantity == "open_fraction" else [1.0, 4.0],
            },
            index=times,
        ).to_parquet(report_folder / f"waterworks_{quantity}.parquet")
    return gpd.GeoDataFrame(
        {
            "source": ["AMBER"] * 3,
            "barrier_id": ["<dam>", "sluice", "excluded"],
            "barrier_type": ["Dam", "Sluice", "Weir"],
            "included": [True, True, False],
            "grid_row": [2, 3, -1],
            "grid_column": [4, 5, -1],
            "exclusion_reason": ["", "", "part of lake or reservoir"],
        },
        geometry=[Point(4.99, 50.99), Point(5.09, 51.09), Point(5.2, 51.2)],
        crs=4326,
    )


def test_waterworks_exact_join_and_missing_data(tmp_path: Path) -> None:
    """Join by actual pixel, distinguish fixed structures and retain missing samples.

    Args:
        tmp_path: Temporary simulation folder.

    Returns:
        None.

    Raises:
        AssertionError: If exact joins or missing-data behavior differ.
    """  # noqa: DOC202, DOC502
    barriers: gpd.GeoDataFrame = waterworks_fixture(tmp_path)
    payload: dict = build_waterworks_payload(barriers, tmp_path)
    dam, sluice, excluded = payload["structures"]
    assert dam["operation"] == "gate"
    assert dam["lon"] == 5.0
    assert dam["series"]["open_fraction"] == [0.1, 1.0]
    assert dam["series"]["outflow_m3_s"] == [2.0, 8.0]
    assert sluice["operation"] == "fixed"
    assert excluded["operation"] == "excluded"
    assert len(payload["timeline"]) == 2
    assert payload["timeline"][0] == pd.Timestamp("2020-01-01 01:00").value // 1_000_000
    report_path: Path = (
        tmp_path / "report/hydrology.routing/waterworks_inflow_m3_s.parquet"
    )
    report: pd.DataFrame = pd.read_parquet(report_path)
    report.loc[report.index[1], "17"] = np.nan
    report.to_parquet(report_path)
    assert build_waterworks_payload(barriers, tmp_path)["structures"][0]["series"][
        "inflow_m3_s"
    ] == [3.0, None]
    assert (
        build_waterworks_payload(barriers, None)["structures"][0]["operation"]
        == "unknown"
    )
    assert (
        build_waterworks_payload(
            barriers.drop(columns=["grid_row", "grid_column"]), tmp_path
        )["structures"][0]["series"]
        == {}
    )


def test_waterworks_validation_and_export(tmp_path: Path) -> None:
    """Reject duplicate runtime mappings and export safe navigation and controls.

    Args:
        tmp_path: Temporary simulation folder.

    Returns:
        None.

    Raises:
        AssertionError: If validation or page controls are missing.
    """  # noqa: DOC202, DOC502
    barriers: gpd.GeoDataFrame = waterworks_fixture(tmp_path)
    region: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[box(4.8, 50.8, 5.3, 51.3)], crs=4326
    )
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[LineString([(5, 51), (5.1, 51.1)])], crs=4326
    )
    path: Path = tmp_path / "dashboard_amber.html"
    write_waterworks_dashboard(
        path, tmp_path / "dashboard.html", region, rivers, barriers, tmp_path
    )
    page: str = path.read_text()
    assert "Waterworks in motion" in page
    assert "Simulation time" in page
    assert "Opening history" in page
    assert "dashboard.html" in page
    assert "<dam>" not in page
    metadata: pd.DataFrame = pd.read_csv(tmp_path / "weir_heights.csv")
    pd.concat([metadata, metadata.iloc[[0]]]).to_csv(
        tmp_path / "weir_heights.csv", index=False
    )
    with pytest.raises(ValueError, match="unique"):
        build_waterworks_payload(barriers, tmp_path)
    with pytest.raises(ValueError, match="CRS"):
        build_waterworks_payload(barriers.drop(columns="included"), None)
