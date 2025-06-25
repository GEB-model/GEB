from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from .module import Module
from .workflows.io import AsyncGriddedForcingReader, open_zarr


class Forcing(Module):
    def __init__(self, model):
        self.model = model
        self.forcings = {}
        self.validators = {
            "tas": lambda x: (x > 170).all() and (x < 370).all(),
            "tasmin": lambda x: (x > 170).all() and (x < 370).all(),
            "tasmax": lambda x: (x > 170).all() and (x < 370).all(),
            "rsds": lambda x: (x >= 0).all(),
            "rlds": lambda x: (x >= 0).all(),
            "sfcwind": lambda x: (x >= 0).all() and (x < 150).all(),
            "ps": lambda x: (x > 30_000).all() and (x < 120_000).all(),
            "pr": lambda x: (x >= 0).all(),
            "hurs": lambda x: (x >= 0).all() and (x <= 100).all(),
            "SPEI": lambda x: not np.isnan(x).any(),
            "CO2": lambda x: x > 270 and x < 2000,
        }

        if self.model.in_spinup:
            self.spinup()

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
            reader = open_zarr(
                self.model.files["other"]["climate/CO2_ppm"],
            ).compute()
        else:
            reader = AsyncGriddedForcingReader(
                self.model.files["other"][f"climate/{name}"],
                name,
            )
            assert reader.ds["y"][0] > reader.ds["y"][-1]
        return reader

    def load(self, name: str, time: datetime | None = None) -> npt.NDArray[Any]:
        """Load forcing data for a given name and time.

        Args:
            name: name of forcing dataset, e.g. "tas", "tasmin", "tasmax", "hurs", "ps", "rlds", "rsds", "sfcWind".
            time: time of forcing data to be returned. Defaults to None, in  which case the current time of the model is used.

        Returns:
            Forcing data as a numpy array.
        """
        if name not in self.forcings:
            reader = self.load_forcing_ds(name)
            self.forcings[name] = reader

        else:
            reader = self.forcings[name]

        if time is None:
            time = self.model.current_time

        if name == "CO2":
            # tollerance is given in nanoseconds. Calulate it as 1e9 * 366 days * 24 hours * 3600 seconds
            data = reader.sel(
                time=self.model.current_time,
                method="pad",
                tolerance=1e9 * 366 * 24 * 3600,
            ).item()
        else:
            data = reader.read_timestep(time)
        assert self.validators[name](data)
        return data

    def spinup(self):
        pass

    def step(self):
        pass
