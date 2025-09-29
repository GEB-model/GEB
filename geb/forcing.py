"""Module to handle climate forcing data."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from .module import Module
from .workflows.io import AsyncGriddedForcingReader


def generate_bilinear_interpolation_weights(
    src_x: npt.NDArray[np.float32],
    src_y: npt.NDArray[np.float32],
    tgt_x: npt.NDArray[np.float32],
    tgt_y: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Generates indices and weights for bilinear interpolation using pure NumPy.

    Assumes the source grid is rectilinear and monotonic.

    Args:
        src_x: Source grid x-coordinates (must be monotonic).
        src_y: Source grid y-coordinates (must be monotonic).
        tgt_x: Target points x-coordinates.
        tgt_y: Target points y-coordinates.

    Returns:
        tuple: A tuple containing:
            - indices (ndarray): Shape (n_targets, 4) of flat source grid indices.
            - weights (ndarray): Shape (n_targets, 4) of interpolation weights.

    Raises:
        ValueError: If target points are outside the source grid bounds.
    """
    # Create the full grid of target points and flatten them
    # The 'ij' indexing creates a target grid of shape (len(tgt_y_1d), len(tgt_x_1d))
    # Note: len(tgt_y_1d) is the row count (Ny_tgt), len(tgt_x_1d) is the col count (Nx_tgt)
    tgt_x_2d, tgt_y_2d = np.meshgrid(tgt_x, tgt_y, indexing="ij")

    # Flatten the target coordinates into (N_targets,) arrays
    tgt_x = tgt_x_2d.flatten()
    tgt_y = tgt_y_2d.flatten()

    # Source Grid Dimensions
    nx: int = len(src_x)
    ny: int = len(src_y)

    # ix is the index of the source x-coordinate that is just less than or equal to tgt_x
    ix = np.searchsorted(src_x, tgt_x) - 1
    # Check bounds for ix
    if not np.all((ix >= 0) & (ix < nx - 1)):
        raise ValueError("Target x-coordinates are outside the source grid bounds.")

    is_y_descending = src_y[0] > src_y[-1]

    if is_y_descending:
        # Search on a reversed copy of the y-axis
        iy_search = np.searchsorted(src_y[::-1], tgt_y) - 1
        # Map the index from the reversed array back to the original descending array
        iy = (ny - 1) - (iy_search + 1)
    else:
        # Standard case for ascending y-axis
        iy = np.searchsorted(src_y, tgt_y) - 1

    # Check bounds for iy
    if not np.all((iy >= 0) & (iy < ny - 1)):
        raise ValueError("Target y-coordinates are outside the source grid bounds.")

    # Get corner coordinates for each grid cell
    x0 = src_x[ix]
    x1 = src_x[ix + 1]
    y0 = src_y[iy]
    y1 = src_y[iy + 1]

    # Calculate normalized distances; denominators are guaranteed non-zero for monotonic grids
    dx = (tgt_x - x0) / (x1 - x0)
    dy = (tgt_y - y0) / (y1 - y0)

    # Calculate Weights
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy

    # Calculate Flat Indices (Assumes C-style row-major: index = iy * nx + ix)
    idx00 = iy * nx + ix  # (y0, x0)
    idx01 = iy * nx + (ix + 1)  # (y0, x1)
    idx10 = (iy + 1) * nx + ix  # (y1, x0)
    idx11 = (iy + 1) * nx + (ix + 1)  # (y1, x1)

    # Stack the indices and weights for the final output
    indices = np.stack([idx00, idx01, idx10, idx11], axis=1).astype(np.int32)
    weights = np.stack([w00, w10, w01, w11], axis=1).astype(np.float32)

    return indices, weights


class ForcingLoader(ABC):
    """Abstract base class for loading and validating forcing data."""

    def __init__(self, model: "GEBModel", variable: str, n: int) -> None:
        """Initialize the ForcingLoader.

        Args:
            model: The GEB model instance.
            variable: The variable name to load (e.g., "pr" for precipitation).
            n: Number of time steps to load at once (default is 1).
        """
        self.n = n
        self.reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
            model.files["other"][f"climate/{variable}"],
            variable,
        )

        self.indices, self.weights = generate_bilinear_interpolation_weights(
            self.reader.x,
            self.reader.y,
            model.hydrology.grid.lon,
            model.hydrology.grid.lat,
        )
        self.xsize = model.hydrology.grid.lon.size
        self.ysize = model.hydrology.grid.lat.size

    def load(self, time: datetime) -> npt.NDArray[Any]:
        """Load and validate forcing data for a given time.

        Args:
            time: The datetime for which to load the data.

        Returns:
            The interpolated and validated data as a numpy array.

        Raises:
            ValueError: If the data is invalid according to the validation criteria.
        """
        data: npt.NDArray[np.float32] = self.reader.read_timestep(time, n=self.n)
        # data: npt.NDArray[np.float32] = data.mean(axis=0)
        interpolated: npt.NDArray[np.float32] = self.interpolate(data)
        valid: bool = self.validate(interpolated)
        if not valid:
            raise ValueError(f"Invalid data for time {time}.")
        assert interpolated.dtype == np.dtype(np.float32)
        return interpolated

    def interpolate(self, data: npt.NDArray[np.float32]) -> npt.NDArray[Any]:
        """Interpolate data to the model grid using bilinear interpolation.

        Args:
            data: The input data array to interpolate.

        Returns:
            The interpolated data array.
        """
        data_flattened_xy_dims = data.reshape(data.shape[0], -1)
        corner_values = data_flattened_xy_dims[:, self.indices]
        interpolated_flattened_xy_dims = np.sum(
            corner_values * self.weights[np.newaxis, :, :], axis=2
        )
        return interpolated_flattened_xy_dims.reshape(
            interpolated_flattened_xy_dims.shape[0], self.ysize, self.xsize
        )

    @abstractmethod
    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate the data array."""
        pass


class Precipitation(ForcingLoader):
    """Loader for precipitation data with specific validation."""

    def __init__(self, model: "GEBModel") -> None:
        """Initialize the Precipitation loader."""
        super().__init__(model, "pr_kg_per_m2_per_s", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate precipitation data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        return (v >= 0).all() and (v < 500 / 3600).all()


class Temperature(ForcingLoader):
    """Loader for temperature data with specific validation."""

    def __init__(self, model: "GEBModel") -> None:
        """Initialize the Temperature loader."""
        super().__init__(model, "tas_2m_K", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate temperature data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        return (v > 170).all() and (v < 370).all()


class DewPointTemperature(ForcingLoader):
    """Loader for dew point temperature data with specific validation."""

    def __init__(self, model: "GEBModel") -> None:
        """Initialize the DewPointTemperature loader."""
        super().__init__(model, "dewpoint_tas_2m_K", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate dew point temperature data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        return (v > 170).all() and (v < 370).all()


class Wind(ForcingLoader):
    """Loader for wind data with specific validation.

    Note that wind can be both positive and negative.
    """

    def __init__(self, model: "GEBModel", direction: str) -> None:
        """Initialize the Wind loader.

        Args:
            model: The GEB model instance.
            direction: The wind direction, either "u" or "v".

        Raises:
            ValueError: If the direction is not "u" or "v".
        """
        if direction not in ["u", "v"]:
            raise ValueError("Direction must be 'u' or 'v'")
        super().__init__(model, f"wind_{direction}10m_m_per_s", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate wind data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        return (v >= -150).all() and (v < 150).all()


class Pressure(ForcingLoader):
    """Loader for surface pressure data with specific validation."""

    def __init__(self, model: "GEBModel") -> None:
        """Initialize the Pressure loader."""
        super().__init__(model, "ps_pascal", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate surface pressure data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        return (v > 30_000).all() and (v < 120_000).all()


class RSDS(ForcingLoader):
    """Loader for surface downwelling shortwave radiation data with specific validation."""

    def __init__(self, model: "GEBModel") -> None:
        """Initialize the RSDS loader."""
        super().__init__(model, "rsds_W_per_m2", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate surface downwelling shortwave radiation data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        return (v >= 0).all()


class RLDS(ForcingLoader):
    """Loader for surface downwelling longwave radiation data with specific validation."""

    def __init__(self, model: "GEBModel") -> None:
        """Initialize the RLDS loader."""
        super().__init__(model, "rlds_W_per_m2", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate surface downwelling longwave radiation data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        return (v >= 0).all()


class SPEI(ForcingLoader):
    """Loader for Standardized Precipitation-Evapotranspiration Index (SPEI) data with specific validation."""

    def __init__(self, model: "GEBModel") -> None:
        """Initialize the SPEI loader."""
        super().__init__(model, "SPEI", 1)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate SPEI data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        return not np.isnan(v).any()


class CO2:
    """Loader for CO2 concentration data with specific validation."""

    def __init__(self, model: "GEBModel") -> None:
        """Initialize the CO2 loader."""
        self.df = pd.read_parquet(model.files["table"]["climate/CO2_ppm"])

    def load(self, time: datetime) -> float:
        """Load CO2 concentration data for a given time.

        Args:
            time: The datetime for which to load the data.

        Returns:
            The CO2 concentration value.

        Raises:
            ValueError: If the data is invalid according to the validation criteria.
        """
        data: float = self.df.loc[time.year].item()
        valid: bool = self.validate(data)
        if not valid:
            raise ValueError(f"Invalid CO2 data for time {time}.")
        return data

    def validate(self, v: float) -> bool:
        """Validate CO2 concentration data.

        Args:
            v: the data value to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        return v > 270 and v < 2000


class Forcing(Module):
    """Module to handle climate forcing data.

    This module is responsible for loading and validating climate forcing data such as temperature, humidity, pressure, and radiation.
    It provides methods to load specific datasets and ensures that the data meets certain validation criteria.

    Args:
        model: The GEB model instance.
    """

    def __init__(self, model: "GEBModel") -> None:
        """Initialize the Forcing module.

        The datasets themselves are loaded on demand when accessed via the `__getitem__` method.
        This is to avoid loading datasets in memory that are not actually used in the simulation,
        due to specific model configurations.

        Also, sets up the forcing validation functions.

        Notes:

        Args:
            model: The GEB model instance.
        """
        self.model = model
        self._forcings = {}

    @property
    def name(self) -> str:
        """Return the name of the module.

        Returns:
            The name of the module.
        """
        return "forcing"

    def load_forcing_ds(self, name: str) -> ForcingLoader | CO2:
        """Load the reader for forcing dataset for a given name.

        Args:
            name: name of forcing dataset, e.g. "tas", "tasmin", "tasmax", "hurs", "ps", "rlds", "rsds", "sfcWind".

        Returns:
            An AsyncGriddedForcingReader or xarray DataArray for the specified forcing dataset.

        """
        if name == "pr_kg_per_m2_per_s":
            reader: Precipitation = Precipitation(self.model)
        elif name == "tas_2m_K":
            reader: Temperature = Temperature(self.model)
        elif name == "dewpoint_tas_2m_K":
            reader: DewPointTemperature = DewPointTemperature(self.model)
        elif name == "wind_u10m_m_per_s":
            reader: Wind = Wind(self.model, direction="u")
        elif name == "wind_v10m_m_per_s":
            reader: Wind = Wind(self.model, direction="v")
        elif name == "ps_pascal":
            reader: Pressure = Pressure(self.model)
        elif name == "rsds_W_per_m2":
            reader: RSDS = RSDS(self.model)
        elif name == "rlds_W_per_m2":
            reader: RLDS = RLDS(self.model)
        elif name == "SPEI":
            reader: SPEI = SPEI(self.model)
        elif name == "CO2_ppm":
            reader: CO2 = CO2(self.model)
        else:
            raise NotImplementedError(f"Forcing dataset '{name}' not implemented.")
        return reader

    def __setitem__(
        self, name: str, reader: AsyncGriddedForcingReader | xr.DataArray
    ) -> None:
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

        return self[name].load(time)

    def spinup(self) -> None:
        """Prepare the forcing module for the simulation.

        Does not do anything as no spinup is needed.
        """
        pass

    def step(self) -> None:
        """Advance the forcing module by one time step.

        Does not do anything as the forcing data is read on demand.
        """
        pass
