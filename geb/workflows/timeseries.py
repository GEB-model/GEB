"""This module contains functions for processing and regularizing time series data."""

import numpy as np
import pandas as pd


def regularize_discharge_timeseries(discharge: pd.DataFrame) -> pd.DataFrame:
    """Regularize the discharge timeseries to ensure consistent time steps.

    This function checks if the time steps in the discharge timeseries are regular (i.e. all time steps are multiples of the minimum time step). If they are not regular, it raises a ValueError. If they are regular, it reindexes the timeseries to have a consistent frequency based on the minimum time step, filling any missing values with NaN.

    Args:
        discharge: DataFrame with a DateTimeIndex representing the discharge timeseries.

    Returns:
        Regularized DataFrame with a consistent DateTimeIndex frequency.

    Raises:
        ValueError: If the time steps in the discharge timeseries are not regular.
    """
    steps = np.diff(discharge.index)
    minimum_step = steps.min()
    # check if all time steps are multiples of the minimum step (i.e. regular time steps)
    if not (steps % minimum_step == pd.Timedelta(0)).all():
        raise ValueError(
            "discharge_observations time steps are not regular. Please ensure the index is a regular time series."
        )
    discharge = discharge.asfreq(
        pd.to_timedelta(minimum_step)
    )  # reindex to regular time steps, filling missing values with NaN
    return discharge
