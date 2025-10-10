"""Module to handle climate forcing data."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from geb.workflows.io import load_grid

from .module import Module
from .workflows.io import AsyncGriddedForcingReader


def generate_bilinear_interpolation_weights(
    src_x: npt.NDArray[np.float32],
    src_y: npt.NDArray[np.float32],
    tgt_x: npt.NDArray[np.float32],
    tgt_y: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Generates indices and weights for bilinear interpolation.

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
    # The 'xy' indexing creates a target grid of shape (len(tgt_y), len(tgt_x))
    # Note: len(tgt_y) is the row count (Ny_tgt), len(tgt_x) is the col count (Nx_tgt)
    tgt_x_2d, tgt_y_2d = np.meshgrid(tgt_x, tgt_y, indexing="xy")

    # Flatten the target coordinates into (N_targets,) arrays
    tgt_x = tgt_x_2d.flatten()
    tgt_y = tgt_y_2d.flatten()

    # Source Grid Dimensions
    nx: int = len(src_x)
    ny: int = len(src_y)

    # Validate that x-coordinates are ascending
    if not np.all(src_x[1:] > src_x[:-1]):
        raise ValueError("Source x-coordinates must be strictly ascending.")

    # ix is the index of the source x-coordinate that is just less than or equal to tgt_x
    ix = np.searchsorted(src_x, tgt_x) - 1
    # Clip indices to be within valid bounds for x
    ix = np.clip(ix, 0, nx - 2)
    # Check bounds for ix after clipping - if any targets are still out of bounds, it's an error
    if not np.all((tgt_x >= src_x[0]) & (tgt_x <= src_x[-1])):
        raise ValueError("Target x-coordinates are outside the source grid bounds.")

    is_y_descending = src_y[0] > src_y[-1]

    if not is_y_descending:
        # For ascending y-axis (normal case)
        iy = np.searchsorted(src_y, tgt_y, side="right") - 1
        iy = np.clip(iy, 0, ny - 2)
        y0 = src_y[iy]
        y1 = src_y[iy + 1]
        dy = (tgt_y - y0) / (y1 - y0)
    else:
        # For descending y-axis, use vectorized NumPy operations
        # Create a matrix comparison to find intervals efficiently
        # src_y[j] >= tgt_y >= src_y[j+1] for descending coordinates

        # Reshape for broadcasting: tgt_y as column, src_y as row
        tgt_y_col = tgt_y[:, np.newaxis]  # Shape: (n_targets, 1)
        src_y_intervals = src_y[:-1]  # Shape: (ny-1,) - upper bounds of intervals
        src_y_intervals_next = src_y[1:]  # Shape: (ny-1,) - lower bounds of intervals

        # Find where target points fall within each interval
        # For descending: src_y[j] >= target_y >= src_y[j+1]
        in_interval = (src_y_intervals >= tgt_y_col) & (
            tgt_y_col >= src_y_intervals_next
        )

        # Find the first (leftmost) interval index for each target point
        iy = np.argmax(in_interval, axis=1)

        # Handle out-of-bounds cases
        no_interval_found = ~np.any(in_interval, axis=1)
        above_highest = tgt_y > src_y[0]
        below_lowest = tgt_y < src_y[-1]

        # Assign boundary indices for out-of-bounds points
        iy = np.where(no_interval_found & above_highest, 0, iy)
        iy = np.where(no_interval_found & below_lowest, ny - 2, iy)

        # Calculate dy for descending coordinates
        y0 = src_y[iy]
        y1 = src_y[iy + 1]
        # For descending coordinates, the weight should be calculated as:
        # dy = (y0 - target_y) / (y0 - y1) since y0 > y1
        dy = (y0 - tgt_y) / (y0 - y1)  # Get the corner coordinates
    x0 = src_x[ix]
    x1 = src_x[ix + 1]

    # Calculate normalized distances
    dx = (tgt_x - x0) / (x1 - x0)

    # Weights for the four corners
    w00 = (1 - dx) * (1 - dy)  # Corresponds to (y0, x0)
    w01 = dx * (1 - dy)  # Corresponds to (y0, x1)
    w10 = (1 - dx) * dy  # Corresponds to (y1, x0)
    w11 = dx * dy  # Corresponds to (y1, x1)

    # Flat indices for the four corners
    idx00 = iy * nx + ix
    idx01 = iy * nx + (ix + 1)
    idx10 = (iy + 1) * nx + ix
    idx11 = (iy + 1) * nx + (ix + 1)

    # Stack indices and weights in a consistent order
    indices = np.stack([idx00, idx01, idx10, idx11], axis=1).astype(np.int32)
    weights = np.stack([w00, w01, w10, w11], axis=1).astype(np.float32)

    assert (weights >= 0).all() and (weights <= 1).all(), "Weights must be in [0, 1]"
    assert np.allclose(weights.sum(axis=1), 1.0), "Weights must sum to 1"

    return indices, weights


def get_pressure_correction_factor(
    DEM: npt.NDArray[np.float32],
    g: float,
    Mo: float,
    lapse_rate: float,
) -> npt.NDArray[np.float32]:
    """Calculate pressure correction factor based on elevation.

    Args:
        DEM: Digital elevation model data (meters).
        g: Gravitational constant (m/s²).
        Mo: Molecular weight of dry air (kg/mol).
        lapse_rate: Temperature lapse rate (K/m).

    Returns:
        Pressure correction factor (dimensionless).
    """
    return (288.15 / (288.15 + lapse_rate * DEM)) ** (g * Mo / (8.3144621 * lapse_rate))


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
        # the corner values must be gathered in the same order as the weights
        corner_values = data_flattened_xy_dims[:, self.indices]
        interpolated_flattened_xy_dims = np.sum(
            corner_values * self.weights[np.newaxis, :, :], axis=2
        )
        output = interpolated_flattened_xy_dims.reshape(
            interpolated_flattened_xy_dims.shape[0], self.ysize, self.xsize
        )
        return output

    @abstractmethod
    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate the data array."""
        pass


class Precipitation(ForcingLoader):
    """Loader for precipitation data with specific validation."""

    def __init__(self, model: "GEBModel", grid_mask: npt.NDArray[np.bool_]) -> None:
        """Initialize the Precipitation loader.

        Args:
            model: The GEB model instance.
            grid_mask: The mask of the model grid (True for valid points, False for invalid).
        """
        self.grid_mask = grid_mask
        super().__init__(model, "pr_kg_per_m2_per_s", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate precipitation data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return (v_non_masked >= 0).all() and (v_non_masked < 500 / 3600).all()


class Temperature(ForcingLoader):
    """Loader for temperature data with specific validation."""

    def __init__(
        self,
        model: "GEBModel",
        forcing_DEM: npt.NDArray[np.float32],
        grid_DEM: npt.NDArray[np.float32],
        grid_mask: npt.NDArray[np.bool_],
        dewpoint: bool = False,
        lapse_rate: float = -0.0065,
    ) -> None:
        """Initialize the Temperature loader.

        This class performs a lapse rate correction based on the difference between the
        forcing DEM and the model grid DEM.

        Args:
            model: The GEB model instance.
            forcing_DEM: The DEM used in the forcing data.
            grid_DEM: The DEM used in the model grid.
            grid_mask: The mask of the model grid (True for valid points, False for invalid).
            dewpoint: If True, load dewpoint temperature; otherwise, load air temperature.
            lapse_rate: The lapse rate in K/m (default is -0.0065 K/m).
        """
        self.forcing_DEM = forcing_DEM
        self.grid_DEM = grid_DEM
        self.grid_mask = grid_mask
        self.lapse_rate = lapse_rate

        if dewpoint:
            super().__init__(model, "dewpoint_tas_2m_K", 24)
        else:
            super().__init__(model, "tas_2m_K", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate temperature data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return (v_non_masked > 170).all() and (v_non_masked < 370).all()

    def interpolate(self, data: npt.NDArray[np.float32]) -> npt.NDArray[Any]:
        """Interpolate data to the model grid using bilinear interpolation.

        Overrides the base class method to handle temperature with a lapse rate correction.
        First, the temperature is adjusted to sea level using the forcing DEM and lapse rate.
        Then, after interpolation, it is adjusted back to the model grid elevation.

        This ensures a smooth temperature gradient with elevation.

        Args:
            data: The input data array to interpolate.

        Returns:
            The interpolated data array.
        """
        temperature_sea_level = (
            data - self.forcing_DEM[np.newaxis, :, :] * self.lapse_rate
        )
        interpolated_temperature_sea_level = super().interpolate(data)
        interpolated_temperature = (
            interpolated_temperature_sea_level
            + self.grid_DEM[np.newaxis, :, :] * self.lapse_rate
        )

        return interpolated_temperature


class Wind(ForcingLoader):
    """Loader for wind data with specific validation.

    Note that wind can be both positive and negative.
    """

    def __init__(
        self, model: "GEBModel", direction: str, grid_mask: npt.NDArray[np.bool_]
    ) -> None:
        """Initialize the Wind loader.

        Args:
            model: The GEB model instance.
            direction: The wind direction, either "u" or "v".
            grid_mask: The mask of the model grid (True for valid points, False for invalid).

        Raises:
            ValueError: If the direction is not "u" or "v".
        """
        if direction not in ["u", "v"]:
            raise ValueError("Direction must be 'u' or 'v'")
        self.grid_mask = grid_mask
        super().__init__(model, f"wind_{direction}10m_m_per_s", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate wind data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return (v_non_masked >= -150).all() and (v_non_masked < 150).all()


class Pressure(ForcingLoader):
    """Loader for surface pressure data with elevation-based correction and validation."""

    def __init__(
        self,
        model: "GEBModel",
        forcing_DEM: npt.NDArray[np.float32],
        grid_DEM: npt.NDArray[np.float32],
        grid_mask: npt.NDArray[np.bool_],
        g: float = 9.80665,
        Mo: float = 0.0289644,
        lapse_rate: float = -0.0065,
    ) -> None:
        """Initialize the Pressure loader.

        This class performs an elevation-based pressure correction based on the difference
        between the forcing DEM and the model grid DEM.

        Args:
            model: The GEB model instance.
            forcing_DEM: The DEM used in the forcing data (meters).
            grid_DEM: The DEM used in the model grid (meters).
            grid_mask: The mask of the model grid (True for valid points, False for invalid).
            g: Gravitational constant (m/s², default is 9.80665).
            Mo: Molecular weight of dry air (kg/mol, default is 0.0289644).
            lapse_rate: Temperature lapse rate (K/m, default is -0.0065).
        """
        self.forcing_DEM = forcing_DEM
        self.grid_DEM = grid_DEM
        self.grid_mask = grid_mask
        self.g = g
        self.Mo = Mo
        self.lapse_rate = lapse_rate

        super().__init__(model, "ps_pascal", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate surface pressure data.

        Args:
            v: The data array to validate (Pa).

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return (v_non_masked > 30_000).all() and (v_non_masked < 120_000).all()

    def interpolate(self, data: npt.NDArray[np.float32]) -> npt.NDArray[Any]:
        """Interpolate data to the model grid using bilinear interpolation.

        Overrides the base class method to handle pressure with elevation correction.
        First, the pressure is adjusted to sea level using the forcing DEM and atmospheric model.
        Then, after interpolation, it is adjusted back to the model grid elevation.

        This ensures proper pressure gradients with elevation.

        Args:
            data: The input pressure data array to interpolate (Pa).

        Returns:
            The interpolated pressure data array (Pa).
        """
        # Convert pressure to sea level by dividing by correction factor
        # for the forcing grid
        pressure_sea_level = (
            data
            / get_pressure_correction_factor(
                self.forcing_DEM, self.g, self.Mo, self.lapse_rate
            )[np.newaxis, :, :]
        )

        # Interpolate the sea level pressure
        interpolated_pressure_sea_level = super().interpolate(pressure_sea_level)

        # Convert back to grid elevation by multiplying by correction factor, now
        # for the model grid
        interpolated_pressure = (
            interpolated_pressure_sea_level
            * get_pressure_correction_factor(
                self.grid_DEM, self.g, self.Mo, self.lapse_rate
            )[np.newaxis, :, :]
        )

        return interpolated_pressure


class RSDS(ForcingLoader):
    """Loader for surface downwelling shortwave radiation data with specific validation."""

    def __init__(self, model: "GEBModel", grid_mask: npt.NDArray[np.bool_]) -> None:
        """Initialize the RSDS loader.

        Args:
            model: The GEB model instance.
            grid_mask: The mask of the model grid (True for valid points, False for invalid).
        """
        self.grid_mask = grid_mask
        super().__init__(model, "rsds_W_per_m2", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate surface downwelling shortwave radiation data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return (v_non_masked >= 0).all()


class RLDS(ForcingLoader):
    """Loader for surface downwelling longwave radiation data with specific validation."""

    def __init__(self, model: "GEBModel", grid_mask: npt.NDArray[np.bool_]) -> None:
        """Initialize the RLDS loader.

        Args:
            model: The GEB model instance.
            grid_mask: The mask of the model grid (True for valid points, False for invalid).
        """
        self.grid_mask = grid_mask
        super().__init__(model, "rlds_W_per_m2", 24)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate surface downwelling longwave radiation data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return (v_non_masked >= 0).all()


class SPEI(ForcingLoader):
    """Loader for Standardized Precipitation-Evapotranspiration Index (SPEI) data with specific validation."""

    def __init__(self, model: "GEBModel", grid_mask: npt.NDArray[np.bool_]) -> None:
        """Initialize the SPEI loader.

        Args:
            model: The GEB model instance.
            grid_mask: The mask of the model grid (True for valid points, False for invalid).
        """
        self.grid_mask = grid_mask
        super().__init__(model, "SPEI", 1)

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate SPEI data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return not np.isnan(v_non_masked).any()


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
        self.forcing_DEM = load_grid(model.files["other"]["climate/elevation_forcing"])
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
            reader: Precipitation = Precipitation(
                self.model, grid_mask=~self.model.hydrology.grid.mask
            )
        elif name == "tas_2m_K":
            reader: Temperature = Temperature(
                self.model,
                forcing_DEM=self.forcing_DEM,
                grid_DEM=self.model.hydrology.grid.decompress(
                    self.model.hydrology.grid.var.elevation
                ),
                grid_mask=~self.model.hydrology.grid.mask,
                dewpoint=False,
            )
        elif name == "dewpoint_tas_2m_K":
            reader: Temperature = Temperature(
                self.model,
                forcing_DEM=self.forcing_DEM,
                grid_DEM=self.model.hydrology.grid.decompress(
                    self.model.hydrology.grid.var.elevation
                ),
                grid_mask=~self.model.hydrology.grid.mask,
                dewpoint=True,
            )
        elif name == "wind_u10m_m_per_s":
            reader: Wind = Wind(
                self.model, direction="u", grid_mask=~self.model.hydrology.grid.mask
            )
        elif name == "wind_v10m_m_per_s":
            reader: Wind = Wind(
                self.model, direction="v", grid_mask=~self.model.hydrology.grid.mask
            )
        elif name == "ps_pascal":
            reader: Pressure = Pressure(
                self.model,
                forcing_DEM=self.forcing_DEM,
                grid_DEM=self.model.hydrology.grid.decompress(
                    self.model.hydrology.grid.var.elevation
                ),
                grid_mask=~self.model.hydrology.grid.mask,
            )
        elif name == "rsds_W_per_m2":
            reader: RSDS = RSDS(self.model, grid_mask=~self.model.hydrology.grid.mask)
        elif name == "rlds_W_per_m2":
            reader: RLDS = RLDS(self.model, grid_mask=~self.model.hydrology.grid.mask)
        elif name == "SPEI":
            reader: SPEI = SPEI(self.model, grid_mask=~self.model.hydrology.grid.mask)
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
