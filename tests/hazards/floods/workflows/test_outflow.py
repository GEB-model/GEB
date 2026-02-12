"""Tests to for outflow workflow."""

import numpy as np
import pytest

from geb.geb_types import TwoDArrayBool
from geb.hazards.floods.workflows.outflow import create_outflow_in_mask


@pytest.fixture
def mask() -> TwoDArrayBool:
    """Provides a simple flood mask for testing.

    Returns:
        A 2D boolean array representing a flood mask.
    """
    return np.array(
        [
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ]
    )


def test_create_outflow_in_mask_width_1(mask: TwoDArrayBool) -> None:
    """Test the create_outflow_in_mask function with a simple mask."""
    row, col = 2, 2
    width_cells = 1

    outflow_mask: TwoDArrayBool = create_outflow_in_mask(mask, row, col, width_cells)

    # Expected outflow points (this is just an example; adjust as needed)
    expected_outflow = np.array(
        [
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]
    )

    assert np.array_equal(outflow_mask, expected_outflow)


def test_create_outflow_in_mask_width_3(mask: TwoDArrayBool) -> None:
    """Test the create_outflow_in_mask function with a simple mask and width 3."""
    row, col = 2, 2
    width_cells = 3

    outflow_mask: TwoDArrayBool = create_outflow_in_mask(mask, row, col, width_cells)

    expected_outflow = np.array(
        [
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    assert np.array_equal(outflow_mask, expected_outflow)


def test_create_outflow_in_mask_width_5_difficult_mask() -> None:
    """Test the create_outflow_in_mask function with a simple mask and width 3."""
    row, col = 2, 2
    width_cells = 5

    mask: TwoDArrayBool = np.array(
        [
            [False, False, False, False, False],
            [False, True, True, False, True],
            [False, True, True, True, False],
            [False, False, False, True, False],
        ]
    )

    outflow_mask: TwoDArrayBool = create_outflow_in_mask(mask, row, col, width_cells)

    expected_outflow = np.array(
        [
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, True, True, True, False],
            [False, False, False, True, False],
        ]
    )
    assert np.array_equal(outflow_mask, expected_outflow)


def test_create_outflow_in_mask_invalid_width(mask: TwoDArrayBool) -> None:
    """Test that create_outflow_in_mask raises ValueError for invalid width_cells."""
    row, col = 2, 2
    width_cells = 4  # Even number, should raise ValueError

    with pytest.raises(ValueError):
        create_outflow_in_mask(mask, row, col, width_cells)


def test_create_outflow_in_mask_too_high_width(mask: TwoDArrayBool) -> None:
    """Test that create_outflow_in_mask handles too high width_cells gracefully."""
    row, col = 2, 2
    width_cells = 101  # Wider than the mask

    outflow_mask: TwoDArrayBool = create_outflow_in_mask(mask, row, col, width_cells)

    # In this case, the entire mask should be marked as outflow
    expected_outflow = np.array(
        [
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    assert np.array_equal(outflow_mask, expected_outflow)


def test_create_outflow_in_mask_outside_mask(mask: TwoDArrayBool) -> None:
    """Test that create_outflow_in_mask handles starting point outside the mask."""
    row, col = 0, 0  # Outside the True area
    width_cells = 3

    with pytest.raises(ValueError):
        create_outflow_in_mask(mask, row, col, width_cells)


def test_create_outflow_in_mask_not_on_border() -> None:
    """Test that create_outflow_in_mask handles starting point not on the border."""
    mask: TwoDArrayBool = np.array(
        [
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    row, col = 1, 2  # Inside the True area but not on the border
    width_cells = 3

    with pytest.raises(ValueError):
        create_outflow_in_mask(mask, row, col, width_cells)
