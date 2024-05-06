from time import time
import xarray as xr
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor


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
                "{}: {:.2f}s".format(self.split_names[i - 1], time_difference)
            )

        # Calculate total time
        total_time = self.times[-1] - self.times[0]
        messages.append("Total: {:.2f}s".format(total_time))

        to_print = ", ".join(messages)
        to_print = "{} - {}".format(self.name, to_print)

        return to_print


class AsyncXarrayReader:
    def __init__(self, filepath, variable_name):
        self.filepath = filepath
        self.ds = xr.open_dataset(filepath, chunks={"time": 1})[
            variable_name
        ]  # Adjust chunk size based on your data
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.preloaded_data_future = None
        self.current_index = -1  # Initialize to -1 to indicate no data loaded yet
        self.time_index = self.ds.time.values
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()

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
        date = np.datetime64(date, "s")
        indices = self.time_index == np.datetime64(date, "ns")
        assert np.count_nonzero(indices) == 1, "Date not found in the dataset."
        return indices.argmax()

    def read_timestep(self, date):
        index = self.get_index(date)
        return self.loop.run_until_complete(self.read_timestep_async(index))

    def read_timestep_not_async(self, date):
        index = self.get_index(date)
        return self.load(index)

    def close(self):
        self.ds.close()
        self.executor.shutdown()
