import asyncio
from time import time
from concurrent.futures import ThreadPoolExecutor

import cftime
import zarr
import numpy as np
import pandas as pd

all_async_readers = []


class TimingModule:
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


class AsyncForcingReader:
    shared_loop = (
        asyncio.new_event_loop()
    )  # Shared event loop, note running in a separate thread is slower
    asyncio.set_event_loop(shared_loop)

    def __init__(self, filepath, variable_name):
        self.filepath = filepath

        self.ds = zarr.open_consolidated(self.filepath, mode="r")
        self.var = self.ds[variable_name]

        self.datetime_index = cftime.num2date(
            self.ds.time[:],
            units=self.ds.time.attrs.get("units"),
            calendar=self.ds.time.attrs.get("calendar"),
        )
        self.datetime_index = pd.DatetimeIndex(
            pd.to_datetime([obj.isoformat() for obj in self.datetime_index])
        ).to_numpy()
        self.time_size = self.datetime_index.size

        all_async_readers.append(self)
        self.preloaded_data_future = None
        self.current_index = -1  # Initialize to -1 to indicate no data loaded yet
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.loop = AsyncForcingReader.shared_loop

    def load(self, index):
        return self.var[index]

    async def load_await(self, index):
        return await self.loop.run_in_executor(self.executor, lambda: self.load(index))

    async def preload_next(self, index):
        # Preload the next timestep asynchronously
        if index + 1 < self.time_size:
            return await self.load_await(index + 1)
        return None

    async def read_timestep_async(self, index):
        assert index < self.time_size, "Index out of bounds."
        assert index >= 0, "Index out of bounds."
        # Check if the requested data is already preloaded, if so, just return that data
        if index == self.current_index:
            return self.current_data
        # Check if the data for the next timestep is preloaded, if so, await for it to complete
        if self.preloaded_data_future is not None and self.current_index + 1 == index:
            data = await self.preloaded_data_future
        # Load the requested data if not preloaded
        else:
            data = await self.load_await(index)

        # Initiate preloading the next timestep, do not await here, this returns a future
        self.preloaded_data_future = asyncio.create_task(self.preload_next(index))
        self.current_index = index
        self.current_data = data
        return data

    def get_index(self, date):
        # convert datetime object to dtype of time coordinate. There is a very high probability
        # that the dataset is the same as the previous one or the next one in line,
        # so we can just check the current index and the next one. Only if those do not match
        # we have to search for the correct index.
        numpy_date = np.datetime64(date, "ns")
        if self.datetime_index[self.current_index] == numpy_date:
            return self.current_index
        elif self.datetime_index[self.current_index + 1] == numpy_date:
            return self.current_index + 1
        else:
            indices = self.datetime_index == numpy_date
            assert np.count_nonzero(indices) == 1, "Date not found in the dataset."
            return indices.argmax()

    def read_timestep(self, date):
        index = self.get_index(date)
        fn = self.read_timestep_async(index)
        data = self.loop.run_until_complete(fn)
        return data

    def read_timestep_not_async(self, date):
        index = self.get_index(date)
        return self.load(index)

    def close(self):
        # cancel the preloading of the next timestep
        if self.preloaded_data_future is not None:
            self.preloaded_data_future.cancel()
        # close the executor
        self.executor.shutdown(wait=False)

        AsyncForcingReader.shared_loop.call_soon_threadsafe(
            AsyncForcingReader.shared_loop.stop
        )


def balance_check(
    name,
    how="cellwise",
    influxes=[],
    outfluxes=[],
    prestorages=[],
    poststorages=[],
    tollerance=1e-10,
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

        if balance.size == 0:
            return True
        elif np.abs(balance).max() > tollerance:
            text = f"{balance[np.abs(balance).argmax()]} is larger than tollerance {tollerance}"
            if name:
                print(name, text)
            else:
                print(text)
            # raise AssertionError(text)
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

        balance = income + store - out
        if balance > tollerance:
            text = f"{np.abs(balance).max()} is larger than tollerance {tollerance}"
            if name:
                print(name, text)
            else:
                print(text)
            # raise AssertionError(text)
            return False
        else:
            return True
    else:
        raise ValueError(f"Method {how} not recognized.")
