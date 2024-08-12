from time import time
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import netCDF4

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


class AsyncXarrayReader:
    shared_loop = (
        asyncio.new_event_loop()
    )  # Shared event loop, note running in a separate thread is slower
    asyncio.set_event_loop(shared_loop)

    def __init__(self, filepath, variable_name):
        self.filepath = filepath

        # netCDF4 is faster than xarray for reading data
        self.ds = netCDF4.Dataset(filepath)
        self.var = self.ds.variables[variable_name]
        self.time_index = self.ds.variables["time"][:]

        self.time_index = self.convert_times_to_numpy_datetime(
            self.ds.variables["time"][:], self.ds.variables["time"].units
        )

        self.time_size = self.time_index.size
        self.var.set_auto_maskandscale(False)

        all_async_readers.append(self)
        self.preloaded_data_future = None
        self.current_index = -1  # Initialize to -1 to indicate no data loaded yet
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.loop = AsyncXarrayReader.shared_loop

        self.lock = threading.Lock()

    def convert_times_to_numpy_datetime(self, times, units):
        # Convert NetCDF times to datetime objects
        datetimes = netCDF4.num2date(times, units)
        # Convert datetime objects to numpy.datetime64
        numpy_datetimes = np.array(datetimes, dtype="datetime64[ns]")
        return numpy_datetimes

    def load_with_lock(self, index):
        with self.lock:
            return self.var[index]

    async def load(self, index):
        # return await asyncio.sleep(1)
        return await self.loop.run_in_executor(
            self.executor, lambda: self.load_with_lock(index)
        )

    async def preload_next(self, index):
        # Preload the next timestep asynchronously
        if index + 1 < self.time_size:
            return await self.load(index + 1)
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
            data = await self.load(index)

        # Initiate preloading the next timestep, do not await here, this returns a future
        self.preloaded_data_future = asyncio.create_task(self.preload_next(index))
        self.current_index = index
        self.current_data = data
        return data

    def get_index(self, date):
        # convert datetime object to dtype of time coordinate
        numpy_date = np.datetime64(date, "ns")
        if self.time_index[self.current_index] == numpy_date:
            return self.current_index
        elif self.time_index[self.current_index + 1] == numpy_date:
            return self.current_index + 1
        else:
            indices = self.time_index == numpy_date
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
        # close the dataset and the executor
        self.ds.close()
        self.executor.shutdown(wait=False)

        AsyncXarrayReader.shared_loop.call_soon_threadsafe(
            AsyncXarrayReader.shared_loop.stop
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
