"""Workflow helpers used in the GEB."""

from time import time
from typing import Iterable

import numpy as np


class TimingModule:
    """A timing module to measure the time taken for different parts of a workflow."""

    def __init__(self, name: str) -> None:
        """Initializes the TimingModule with a name and starts the timer.

        Args:
            name: The name of the timing module. Will be used when printing the timing results.
        """
        self.name = name
        self.times = [time()]
        self.split_names = []

    def finish_split(self, name: str) -> None:
        """Finish split with with name given.

        Appends the current time and the name of the split to their respective lists, which will be
        used to calculate the time taken for each split and the total time when converting to string.

        Args:
            name: The name of the split. This is the name of the previous split.
        """
        self.times.append(time())
        self.split_names.append(name)

    def __str__(self) -> str:
        """Converts the timing information into a readable string format for logging or display.

        Returns:
            A formatted string summarizing the time taken for each split and the total time.
        """
        messages = []
        for i in range(1, len(self.times)):
            time_difference = self.times[i] - self.times[i - 1]
            messages.append(
                "{}: {:.4f}s".format(self.split_names[i - 1], time_difference)
            )

        # Calculate total time
        total_time = self.times[-1] - self.times[0]
        messages.append("Total: {:.4f}s".format(total_time))

        to_print = ", ".join(messages)
        to_print = "{} - {}".format(self.name, to_print)

        return to_print


def balance_check(
    name: str,
    how: str = "cellwise",
    influxes: Iterable = [],
    outfluxes: Iterable = [],
    prestorages: Iterable = [],
    poststorages: Iterable = [],
    tolerance: float = 1e-10,
    error_identifiers: dict = {},
    raise_on_error: bool = False,
) -> bool:
    """Check the balance of a system, usually for water.

    Essentially checks that influxes + prestorages = outfluxes + poststorages,
    within a given tolerance.

    Args:
        name: Name of the balance check, used for printing.
        how: Method to use for balance check, either 'cellwise' or 'sum'.
        influxes: List of influx arrays.
        outfluxes: List of outflux arrays.
        prestorages: List of pre-storage arrays.
        poststorages: List of post-storage arrays.
        tolerance: tolerance for the balance check.
        error_identifiers: Dictionary of identifiers to help locate errors, e.g. {'x': x_array, 'y': y_array}.
            Can only be used with how='cellwise'.
            When an error is found, the values of these identifiers at the location of the maximum error will be printed.
        raise_on_error: Whether to raise an error if the balance check fails.

    Returns:
        True if the balance check passes, False otherwise.

    Raises:
        ValueError: If NaN values are found in the balance calculation.
        AssertionError: If the balance check fails and raise_on_error is True.
    """
    income = 0
    out = 0
    store = 0

    if not isinstance(influxes, (list, tuple)):
        influxes = [influxes]
    if not isinstance(outfluxes, (list, tuple)):
        outfluxes = [outfluxes]
    if not isinstance(prestorages, (list, tuple)):
        prestorages = [prestorages]
    if not isinstance(poststorages, (list, tuple)):
        poststorages = [poststorages]

    if how == "cellwise":
        for fluxIn in influxes:
            income += fluxIn
        for fluxOut in outfluxes:
            out += fluxOut
        for preStorage in prestorages:
            store += preStorage
        for endStorage in poststorages:
            store -= endStorage
        balance = income + store - out

        if np.isnan(balance).any():
            raise ValueError("Balance check failed, NaN values found.")

        if balance.size == 0:
            return True
        elif np.abs(balance).max() > tolerance:
            index = np.abs(balance).argmax()
            text = f"{balance[np.abs(balance).argmax()]} > tolerance {tolerance}, max imbalance at index {index}."

            if error_identifiers:
                text += " Error identifiers: " + ", ".join(
                    f"{key}={value[index]}" for key, value in error_identifiers.items()
                )
            if name:
                print(name, text)
            else:
                print(text)
            if raise_on_error:
                raise AssertionError(text)
            return False
        else:
            return True

    elif how == "sum":
        assert not error_identifiers, (
            "Error identifiers not supported for 'sum' method."
        )
        for fluxIn in influxes:
            income += fluxIn.sum()
        for fluxOut in outfluxes:
            out += fluxOut.sum()
        for preStorage in prestorages:
            store += preStorage.sum()
        for endStorage in poststorages:
            store -= endStorage.sum()

        balance = abs(income + store - out)
        if np.isnan(balance):
            raise ValueError("Balance check failed, NaN values found.")
        if balance > tolerance:
            text = f"{balance} is larger than tolerance {tolerance}"
            if name:
                print(name, text)
            else:
                print(text)
            if raise_on_error:
                raise AssertionError(text)
            return False
        else:
            return True
    else:
        raise ValueError(f"Method {how} not recognized.")
