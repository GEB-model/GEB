from time import time

import numpy as np


class TimingModule:
    """A timing module to measure the time taken for different parts of a workflow."""

    def __init__(self, name):
        self.name = name
        self.times = [time()]
        self.split_names = []

    def new_split(self, name):
        self.times.append(time())
        self.split_names.append(name)

    def __str__(self):
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
    name,
    how="cellwise",
    influxes=[],
    outfluxes=[],
    prestorages=[],
    poststorages=[],
    tollerance=1e-10,
    raise_on_error=False,
):
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
        elif np.abs(balance).max() > tollerance:
            text = f"{balance[np.abs(balance).argmax()]} > tollerance {tollerance}, max imbalance at index {np.abs(balance).argmax()}"
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
        if balance > tollerance:
            text = f"{balance} is larger than tollerance {tollerance}"
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
