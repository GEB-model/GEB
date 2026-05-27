"""Tests for MIRCA-OS crop calendar parsing workflows."""

import logging

import numpy as np
import pandas as pd
import pytest

from geb.build.data_catalog import DataCatalog
from geb.build.workflows.crop_calendars import (
    MIRCA_OS_CROP_CLASS_MAP,
    parse_MIRCA_crop_calendar,
)

from ...testconfig import IN_GITHUB_ACTIONS


def test_parse_mirca_os_others_annual_numbered_variant() -> None:
    """Map MIRCA-OS numbered Others annual variants to class 25."""
    calendar_data = pd.DataFrame(
        {
            "unit_code": [12345],
            "Crop": ["Others annual3"],
            "Growing_area": [100.0],
            "Planting_Month": [4],
            "Maturity_Month": [10],
        }
    )

    parsed = parse_MIRCA_crop_calendar(
        parsed_calendar={},
        crop_calendar_data=calendar_data,
        MIRCA_units=[12345],
        is_irrigated=False,
    )

    assert 12345 in parsed
    assert len(parsed[12345]) == 1

    _, rotation_matrix = parsed[12345][0]
    assert int(rotation_matrix[0, 0]) == 25
    assert int(rotation_matrix[0, 1]) == 0


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Requires MIRCA-OS data files.")
def test_parse_mirca_os_crop_calendar_from_data_catalog() -> None:
    """Parse MIRCA-OS rainfed and irrigated calendars from the data catalog."""
    logger = logging.getLogger("test_parse_mirca_os_crop_calendar_from_data_catalog")
    data_catalog = DataCatalog(logger=logger)

    rainfed_source = data_catalog.fetch("mirca_os_crop_calendar_2000_rf").read()
    irrigated_source = data_catalog.fetch("mirca_os_crop_calendar_2000_ir").read()

    assert isinstance(rainfed_source, pd.DataFrame)
    assert isinstance(irrigated_source, pd.DataFrame)

    required_columns = {
        "unit_code",
        "Crop",
        "Growing_area",
        "Planting_Month",
        "Maturity_Month",
    }
    assert required_columns.issubset(rainfed_source.columns)
    assert required_columns.issubset(irrigated_source.columns)

    rainfed_units = set(
        rainfed_source.loc[rainfed_source["Growing_area"] > 0, "unit_code"]
        .astype(int)
        .tolist()
    )
    irrigated_units = set(
        irrigated_source.loc[irrigated_source["Growing_area"] > 0, "unit_code"]
        .astype(int)
        .tolist()
    )
    test_units = sorted(rainfed_units & irrigated_units)

    assert test_units, "Expected at least one MIRCA unit with both rf and ir crops."

    parsed_calendar: dict[int, list[tuple[float, np.ndarray]]] = {}
    parsed_calendar = parse_MIRCA_crop_calendar(
        parsed_calendar=parsed_calendar,
        crop_calendar_data=rainfed_source,
        MIRCA_units=test_units,
        is_irrigated=False,
    )
    parsed_calendar = parse_MIRCA_crop_calendar(
        parsed_calendar=parsed_calendar,
        crop_calendar_data=irrigated_source,
        MIRCA_units=test_units,
        is_irrigated=True,
    )

    assert parsed_calendar

    seen_crop_classes: set[int] = set()
    seen_irrigation_flags: set[int] = set()

    for unit_code, crop_rotations in parsed_calendar.items():
        assert unit_code in test_units[:3]
        assert crop_rotations

        for area, rotation_matrix in crop_rotations:
            assert area > 0
            assert isinstance(rotation_matrix, np.ndarray)
            assert rotation_matrix.shape == (3, 5)
            assert rotation_matrix.dtype == np.int32

            valid_rows = rotation_matrix[rotation_matrix[:, 0] != -1]
            assert valid_rows.size > 0

            seen_crop_classes.update(valid_rows[:, 0].astype(int).tolist())
            seen_irrigation_flags.update(valid_rows[:, 1].astype(int).tolist())

            # Planting day index and growth length must be in valid annual ranges.
            assert np.all((valid_rows[:, 2] >= 0) & (valid_rows[:, 2] <= 364))
            assert np.all((valid_rows[:, 3] >= 1) & (valid_rows[:, 3] <= 365))

    assert seen_crop_classes.issubset(set(MIRCA_OS_CROP_CLASS_MAP.values()))
    assert seen_irrigation_flags == {0, 1}
