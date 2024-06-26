from time import time
import xarray as xr
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
                "{}: {:.3f}s".format(self.split_names[i - 1], time_difference)
            )

        # Calculate total time
        total_time = self.times[-1] - self.times[0]
        messages.append("Total: {:.3f}s".format(total_time))

        to_print = ", ".join(messages)
        to_print = "{} - {}".format(self.name, to_print)

        return to_print


class AsyncXarrayReader:
    def __init__(self, filepath, variable_name, loop):
        self.filepath = filepath
        self.ds = xr.open_dataset(filepath)[
            variable_name
        ]  # Adjust chunk size based on your data
        self.executor = ThreadPoolExecutor(max_workers=1)
        all_async_readers.append(self)
        self.preloaded_data_future = None
        self.current_index = -1  # Initialize to -1 to indicate no data loaded yet
        self.time_index = self.ds.time.values
        self.loop = loop
        return None

    def load(self, index):
        return self.ds.isel(time=index).values

    def preload_next(self, index):
        # Preload the next timestep asynchronously
        if index + 1 < self.ds.time.size:
            future = self.loop.run_in_executor(
                self.executor, lambda: self.load(index + 1)
            )
            return future
        return None

    async def read_timestep_async(self, index):
        assert index < self.ds.time.size, "Index out of bounds."
        assert index >= 0, "Index out of bounds."
        # Check if the requested data is already preloaded, if so, just return that data
        if index == self.current_index:
            return self.current_data
        # Check if the data for the next timestep is preloaded, if so, await for it to complete
        if self.preloaded_data_future is not None and self.current_index + 1 == index:
            data = await self.preloaded_data_future
        # Load the requested data if not preloaded
        else:
            data = await self.loop.run_in_executor(
                self.executor, lambda: self.load(index)
            )

        # Initiate preloading the next timestep, do not await here, this returns a future
        self.preloaded_data_future = self.preload_next(index)
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
        future = asyncio.run_coroutine_threadsafe(
            self.read_timestep_async(index), self.loop
        )
        data = future.result()
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
