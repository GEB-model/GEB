"""Tests for the observations build module and helper functions."""

import math
from pathlib import Path

import numpy as np
import pytest

from geb.build.modules.observations import parse_custom_station_filename


def test_parse_custom_station_filename_with_upstream_area() -> None:
    """Test parsing filename containing coordinates, upstream area in m2, and station name."""
    station_path: Path = Path("-0.02754_51.839051_136790000+Rib_at_Wadesmill.parquet")
    lon: float
    lat: float
    upstream_area_m2: float
    station_name: str
    lon, lat, upstream_area_m2, station_name = parse_custom_station_filename(
        station_path
    )

    assert math.isclose(lon, -0.02754)
    assert math.isclose(lat, 51.839051)
    assert math.isclose(upstream_area_m2, 136790000.0)
    assert station_name == "Rib_at_Wadesmill"


def test_parse_custom_station_filename_without_upstream_area() -> None:
    """Test parsing legacy filename containing only coordinates and station name."""
    station_path: Path = Path("0.12345_52.6789+Sample_Station.csv")
    lon: float
    lat: float
    upstream_area_m2: float
    station_name: str
    lon, lat, upstream_area_m2, station_name = parse_custom_station_filename(
        station_path
    )

    assert math.isclose(lon, 0.12345)
    assert math.isclose(lat, 52.6789)
    assert np.isnan(upstream_area_m2)
    assert station_name == "Sample_Station"


def test_parse_custom_station_filename_missing_plus_separator() -> None:
    """Test that missing '+' separator raises ValueError."""
    station_path: Path = Path("-0.02754_51.839051_136790000_Rib_at_Wadesmill.parquet")
    with pytest.raises(ValueError, match=r"does not contain '\+' separator"):
        parse_custom_station_filename(station_path)


def test_parse_custom_station_filename_invalid_part_count() -> None:
    """Test that invalid number of metadata parts before '+' raises ValueError."""
    station_path_single: Path = Path("51.839051+Rib_at_Wadesmill.parquet")
    with pytest.raises(ValueError, match="metadata parts"):
        parse_custom_station_filename(station_path_single)

    station_path_four: Path = Path("1.0_2.0_3.0_4.0+Rib_at_Wadesmill.parquet")
    with pytest.raises(ValueError, match="metadata parts"):
        parse_custom_station_filename(station_path_four)


def test_parse_custom_station_filename_invalid_numbers() -> None:
    """Test that non-numeric coordinates or upstream area raise ValueError."""
    station_path_coords: Path = Path("invalid_51.839051+Rib_at_Wadesmill.parquet")
    with pytest.raises(ValueError, match="valid numeric coordinates"):
        parse_custom_station_filename(station_path_coords)

    station_path_area: Path = Path(
        "-0.02754_51.839051_invalid+Rib_at_Wadesmill.parquet"
    )
    with pytest.raises(ValueError, match="valid numeric coordinates or upstream area"):
        parse_custom_station_filename(station_path_area)
