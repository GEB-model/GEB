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


def test_parse_mirca_os_duplicate_crop_name_aggregated() -> None:
    """Duplicate rows for the same named crop variant have their areas summed.

    Two rows both named "Wheat1" with the same growing season should be
    collapsed into a single rotation entry whose area is the sum of both rows,
    rather than being treated as two separate rotation candidates.
    """
    calendar_data = pd.DataFrame(
        {
            "unit_code": [1, 1],
            "Crop": ["Wheat1", "Wheat1"],
            "Growing_area": [100.0, 50.0],
            "Planting_Month": [11, 11],
            "Maturity_Month": [5, 5],
        }
    )

    parsed = parse_MIRCA_crop_calendar(
        parsed_calendar={},
        crop_calendar_data=calendar_data,
        MIRCA_units=[1],
        is_irrigated=False,
    )

    assert 1 in parsed
    assert len(parsed[1]) == 1

    area, rotation_matrix = parsed[1][0]
    assert area == 150.0
    assert int(rotation_matrix[0, 0]) == MIRCA_OS_CROP_CLASS_MAP["Wheat"]


def test_parse_mirca_os_two_variants_non_overlapping_split() -> None:
    """Two non-overlapping wheat variants produce one single and one double rotation.

    With variant 1 at 100 ha (Jan–Mar) and variant 2 at 200 ha (Jun–Sep), the
    expected output is:
    - 100 ha with only variant 2's rotation (the excess area that cannot also
      grow variant 1 within the same year).
    - 100 ha with both variants in sequence (the area that can fit both rotations
      in one year).
    """
    calendar_data = pd.DataFrame(
        {
            "unit_code": [1, 1],
            "Crop": ["Wheat1", "Wheat2"],
            "Growing_area": [100.0, 200.0],
            "Planting_Month": [1, 6],
            "Maturity_Month": [3, 9],
        }
    )

    parsed = parse_MIRCA_crop_calendar(
        parsed_calendar={},
        crop_calendar_data=calendar_data,
        MIRCA_units=[1],
        is_irrigated=False,
    )

    assert 1 in parsed
    assert len(parsed[1]) == 2

    areas = sorted(e[0] for e in parsed[1])
    assert areas == [100.0, 100.0]

    # One entry has a single active rotation row, the other has two.
    active_rotation_counts = sorted(
        int((arr[:, 0] != -1).sum()) for _, arr in parsed[1]
    )
    assert active_rotation_counts == [1, 2]


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
        assert unit_code in test_units
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
