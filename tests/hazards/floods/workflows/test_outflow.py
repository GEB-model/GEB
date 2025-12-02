"""Tests to for outflow workflow."""

import numpy as np
import pytest

from geb.hazards.floods.workflows.outflow import detect_outflow
from geb.types import TwoDArrayBool


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


def test_detect_outflow_width_1(mask: TwoDArrayBool) -> None:
    """Test the detect_outflow function with a simple mask."""
    row, col = 2, 2
    width_cells = 1

    outflow_mask: TwoDArrayBool = detect_outflow(mask, row, col, width_cells)

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


def test_detect_outflow_width_3(mask: TwoDArrayBool) -> None:
    """Test the detect_outflow function with a simple mask and width 3."""
    row, col = 2, 2
    width_cells = 3

    outflow_mask: TwoDArrayBool = detect_outflow(mask, row, col, width_cells)

    expected_outflow = np.array(
        [
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    assert np.array_equal(outflow_mask, expected_outflow)


def test_detect_outflow_width_5_difficult_mask() -> None:
    """Test the detect_outflow function with a simple mask and width 3."""
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

    outflow_mask: TwoDArrayBool = detect_outflow(mask, row, col, width_cells)

    expected_outflow = np.array(
        [
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, True, True, True, False],
            [False, False, False, True, False],
        ]
    )
    assert np.array_equal(outflow_mask, expected_outflow)


def test_detect_outflow_invalid_width(mask: TwoDArrayBool) -> None:
    """Test that detect_outflow raises ValueError for invalid width_cells."""
    row, col = 2, 2
    width_cells = 4  # Even number, should raise ValueError

    with pytest.raises(ValueError):
        detect_outflow(mask, row, col, width_cells)


def test_detect_outflow_too_high_width(mask: TwoDArrayBool) -> None:
    """Test that detect_outflow handles too high width_cells gracefully."""
    row, col = 2, 2
    width_cells = 101  # Wider than the mask

    outflow_mask: TwoDArrayBool = detect_outflow(mask, row, col, width_cells)

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


def test_detect_outflow_outside_mask(mask: TwoDArrayBool) -> None:
    """Test that detect_outflow handles starting point outside the mask."""
    row, col = 0, 0  # Outside the True area
    width_cells = 3

    with pytest.raises(ValueError):
        detect_outflow(mask, row, col, width_cells)


def test_detect_outflow_not_on_border() -> None:
    """Test that detect_outflow handles starting point not on the border."""
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
        detect_outflow(mask, row, col, width_cells)
