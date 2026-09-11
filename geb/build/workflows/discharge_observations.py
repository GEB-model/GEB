"""Small workflows for preparing discharge observations."""

from typing import Any

import numpy as np
import pandas as pd


def find_duplicate_discharge_stations(
    observations: pd.DataFrame,
    minimum_years: float = 5.0,
) -> dict[Any, list[Any]]:
    """Find stations with identical overlapping daily observations.

    Args:
        observations: Daily discharge observations (m³/s), one station per column.
        minimum_years: Required number of identical overlapping years.

    Returns:
        Duplicate station IDs mapped to all matching station IDs.

    Raises:
        ValueError: If the index is not a DatetimeIndex or minimum_years is not positive.
    """
    if not isinstance(observations.index, pd.DatetimeIndex):
        raise ValueError("Discharge observations must use a DatetimeIndex.")
    if minimum_years <= 0:
        raise ValueError("minimum_years must be positive.")
    if observations.empty:
        return {}

    daily: pd.DataFrame = observations.resample("D").mean()
    minimum_days: int = int(minimum_years * 365.25)
    station_ids: list[Any] = list(daily.columns)
    # Store each station contiguously because every comparison reads its full record.
    values: np.ndarray = np.ascontiguousarray(daily.to_numpy().T)
    finite: np.ndarray = np.isfinite(values)
    duplicate_sets: dict[Any, set[Any]] = {}
    for left_position, left_id in enumerate(station_ids):
        left_values: np.ndarray = values[left_position]
        for right_position in range(left_position + 1, len(station_ids)):
            right_id: Any = station_ids[right_position]
            right_values: np.ndarray = values[right_position]
            shared: np.ndarray = finite[left_position] & finite[right_position]
            if np.count_nonzero(shared) >= minimum_days and np.array_equal(
                left_values[shared], right_values[shared]
            ):
                duplicate_sets.setdefault(left_id, set()).add(right_id)
                duplicate_sets.setdefault(right_id, set()).add(left_id)
    return {
        station_id: sorted(matching_ids, key=str)
        for station_id, matching_ids in duplicate_sets.items()
    }
