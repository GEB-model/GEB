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
