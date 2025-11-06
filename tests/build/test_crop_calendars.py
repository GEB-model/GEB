"""Tests for the crop calendar functions."""

from datetime import date

from geb.build.workflows.crop_calendars import (
    get_day_index,
    get_growing_season_length,
)


def test_growing_season_length() -> None:
    """Test growing season length calculation for various start and end days.

    Especially important are cases where the season wraps around the end of the year.
    """
    assert get_growing_season_length(0, 364) == 364
    assert get_growing_season_length(0, 0) == 365
    assert get_growing_season_length(0, 1) == 1
    assert get_growing_season_length(300, 1) == 66
    assert get_growing_season_length(2, 1) == 364


def test_day_of_year() -> None:
    """Test day index calculation for various dates."""
    assert get_day_index(date(2000, 1, 1)) == 0
    assert get_day_index(date(2000, 1, 2)) == 1
    assert get_day_index(date(2000, 2, 1)) == 31
    assert get_day_index(date(2000, 12, 31)) == 365  # Leap year
    assert get_day_index(date(2001, 12, 31)) == 364  # Non-leap year
