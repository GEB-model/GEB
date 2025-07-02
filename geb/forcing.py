from datetime import datetime
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from .module import Module
from .workflows.io import AsyncGriddedForcingReader, open_zarr


class Forcing(Module):
    """Module to handle climate forcing data.

    This module is responsible for loading and validating climate forcing data such as temperature, humidity, pressure, and radiation.
    It provides methods to load specific datasets and ensures that the data meets certain validation criteria.

    Args:
        model: The GEB model instance.
    """

    def __init__(self, model):
        self.model = model
        self._forcings = {}
        self.validators = {
            "tas": lambda x: (x > 170).all() and (x < 370).all(),
            "tasmin": lambda x: (x > 170).all() and (x < 370).all(),
            "tasmax": lambda x: (x > 170).all() and (x < 370).all(),
            "rsds": lambda x: (x >= 0).all(),
            "rlds": lambda x: (x >= 0).all(),
            "sfcwind": lambda x: (x >= 0).all() and (x < 150).all(),
            "ps": lambda x: (x > 30_000).all() and (x < 120_000).all(),
            "pr": lambda x: (x >= 0).all()
            and (x < 500 / 3600).all(),  # 500 mm/h converted to kg/m²/s
            "pr_hourly": lambda x: (x >= 0).all(),
            "hurs": lambda x: (x >= 0).all() and (x <= 100).all(),
            "SPEI": lambda x: not np.isnan(x).any(),
            "CO2": lambda x: x > 270 and x < 2000,
        }

    @property
    def name(self):
        return "forcing"

    def load_forcing_ds(self, name: str) -> AsyncGriddedForcingReader | xr.DataArray:
        """Load the reader for forcing dataset for a given name.

        Args:
            name: name of forcing dataset, e.g. "tas", "tasmin", "tasmax", "hurs", "ps", "rlds", "rsds", "sfcWind".

        Returns:
            An AsyncGriddedForcingReader or xarray DataArray for the specified forcing dataset.

        """
        if name == "CO2":
            reader: xr.DataArray = open_zarr(
                self.model.files["other"]["climate/CO2_ppm"],
            ).compute()
        elif name == "pr_hourly":
            reader: xr.DataArray = open_zarr(
                self.model.files["other"]["climate/pr_hourly"],
            )
        else:
            reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
                self.model.files["other"][f"climate/{name}"],
                name,
            )
            assert reader.ds["y"][0] > reader.ds["y"][-1]
        return reader

    def __setitem__(self, name: str, reader: AsyncGriddedForcingReader | xr.DataArray):
        """Set the forcing data for a given name.

        Args:
            name: name of forcing dataset, e.g. "tas", "tasmin", "tasmax", "hurs", "ps", "rlds", "rsds", "sfcWind".
            reader: An AsyncGriddedForcingReader or xarray DataArray for the specified forcing dataset.
        """
        self._forcings[name] = reader

    @overload
    def __getitem__(self, name: Literal["pr_hourly", "CO2"]) -> xr.DataArray:
        pass

    @overload
    def __getitem__(
        self,
        name: Literal[
            "tas", "tasmin", "tasmax", "hurs", "ps", "rlds", "rsds", "sfcwind", "SPEI"
        ],
    ) -> AsyncGriddedForcingReader:
        pass

    def __getitem__(self, name: str) -> AsyncGriddedForcingReader | xr.DataArray:
        """Get the forcing data for a given name.

        Args:
            name: name of forcing dataset, e.g. "tas", "tasmin", "tasmax", "hurs", "ps", "rlds", "rsds", "sfcWind".

        Returns:
            An AsyncGriddedForcingReader or xarray DataArray for the specified forcing dataset.
        """
        if name not in self._forcings.keys():
            self[name] = self.load_forcing_ds(name)
        return self._forcings[name]

    def load(self, name: str, time: datetime | None = None) -> npt.NDArray[Any]:
        """Load forcing data for a given name and time.

        Args:
            name: name of forcing dataset, e.g. "tas", "tasmin", "tasmax", "hurs", "ps", "rlds", "rsds", "sfcWind".
            time: time of forcing data to be returned. Defaults to None, in  which case the current time of the model is used.

        Returns:
            Forcing data as a numpy array.
        """
        if time is None:
            time: datetime = self.model.current_time

        if name == "CO2":
            # tollerance is given in nanoseconds. Calulate it as 1e9 * 366 days * 24 hours * 3600 seconds
            data = (
                self[name]
                .sel(
                    time=time,
                    method="pad",
                    tolerance=1e9 * 366 * 24 * 3600,
                )
                .item()
            )
        elif name == "pr_hourly":
            # For hourly precipitation, we need to read the data for the current time
            # and return it as a numpy array.
            data = self[name].sel(time=time)
        else:
            data = self[name].read_timestep(time)
        if __debug__ and not self.validators[name](data):
            data = data.compute()
            raise ValueError(
                f"Invalid data for {name} at time {time}. "
                f"\tMin data value: {data.min()}"
                f"\tMax data value: {data.max()}"
            )
        return data

    def spinup(self):
        pass

    def step(self):
        pass
