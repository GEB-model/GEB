"""Utility functions for datetime operations in GEB workflows."""

from datetime import datetime, timedelta


def round_up_to_start_of_next_day_unless_midnight(dt_object: datetime) -> datetime:
    """Round up a datetime object to the start of the next day unless it is already midnight.

    Args:
        dt_object: the datetime object to round up.

    Returns:
        A datetime object rounded up to the start of the next day if it is not already midnight.
        If it is midnight, the original datetime object is returned.
    """
    # Check if the time is already exactly midnight
    if (
        dt_object.hour == 0
        and dt_object.minute == 0
        and dt_object.second == 0
        and dt_object.microsecond == 0
    ):
        return dt_object  # If it's midnight, return as is
    else:
        # Otherwise, round up to the start of the next day
        next_day_start = (dt_object + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return next_day_start


def map_date_to_dekad(dt: datetime) -> int:
    """Map a date to a dekadal index (0-35).

    Args:
        dt: Date to map.

    Returns:
        Dekadal index (0-35).
    """
    month: int = dt.month
    day: int = dt.day

    # find dekadal index for start of month
    dekadal_index = (month - 1) * 3

    # adjust for day of month
    if day <= 10:
        dekadal_index += 0
    elif day <= 20:
        dekadal_index += 1
    else:
        dekadal_index += 2

    return dekadal_index
