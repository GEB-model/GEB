"""Module to handle climate forcing data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from geb.types import ThreeDArrayFloat32
from geb.workflows.io import read_grid

from .module import Module
from .workflows.io import AsyncGriddedForcingReader

if TYPE_CHECKING:
    from geb.model import GEBModel


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

    def __init__(
        self, model: GEBModel, variable: str, n: int, supports_forecast: bool = True
    ) -> None:
        """Initialize the ForcingLoader.

        Args:
            model: The GEB model instance.
            variable: The variable name to load (e.g., "pr" for precipitation).
            n: Number of time steps to load at once (default is 1).
            supports_forecast: Whether the loader supports forecast mode.
        """
        self.model: GEBModel = model
        self.n: int = n
        self.variable: str = variable
        self.reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
            model.files["other"][f"climate/{variable}"], variable, asynchronous=True
        )

        self.indices, self.weights = generate_bilinear_interpolation_weights(
            self.reader.x,
            self.reader.y,
            model.hydrology.grid.lon,
            model.hydrology.grid.lat,
        )
        self.xsize: int = model.hydrology.grid.lon.size
        self.ysize: int = model.hydrology.grid.lat.size

        self._supports_forecast = supports_forecast

        self._in_forecast_mode: bool = False
        self.ds_forecast: xr.DataArray | None = None
        self.forecast_issue_datetime: datetime | None = None

    @property
    def in_forecast_mode(self) -> bool:
        """Indicates whether the loader is in forecast mode.

        Returns:
            True if in forecast mode, False otherwise.
        """
        return self._in_forecast_mode

    @in_forecast_mode.setter
    def in_forecast_mode(self, value: bool) -> None:
        self._in_forecast_mode = value

    @property
    def supports_forecast(self) -> bool:
        """Indicates whether the loader supports forecast mode.

        Returns:
            True if forecast mode is supported, False otherwise.
        """
        return self._supports_forecast

    def load(self, dt: datetime) -> ThreeDArrayFloat32:
        """Load and validate forcing data for a given time.

        If in forecast mode and the time is after the forecast issue date,
        the data is loaded from the forecast dataset.

        Otherwise, data is loaded from the standard reader.

        Args:
            dt: The datetime for which to load the data.

        Returns:
            The interpolated and validated data as a numpy array.

        Raises:
            ValueError: If the data is invalid according to the validation criteria.
        """
        # check if we are in forecasting mode, and if the end of the timestep is after the
        # start of the forecast
        if (
            self.forecast_issue_datetime is not None
            and dt + self.model.timestep_length >= self.forecast_issue_datetime
        ):
            # find how many substeps to load from the normal data source
            substeps_to_forecast: int = int(
                (self.forecast_issue_datetime - dt).total_seconds()
                / (self.model.timestep_length / self.n).total_seconds()
            )
            # if some substeps are before the forecast, load them from the normal reader
            if substeps_to_forecast > 0:
                # TODO: The reader breaks when loading less than n timesteps, so we load n and slice
                # the data
                non_forecast_data: npt.NDArray[np.float32] = self.reader.read_timestep(
                    dt, n=self.n
                )[:substeps_to_forecast, :, :]

                forecast_data: npt.NDArray[np.float32] = self.ds_forecast.isel(
                    time=slice(None, self.n - substeps_to_forecast)
                ).values
                data: npt.NDArray[np.float32] = np.concatenate(
                    [non_forecast_data, forecast_data], axis=0
                )
            else:
                # if the current day is already fully in forecast, load all from forecast
                data: npt.NDArray[np.float32] = self.ds_forecast.sel(
                    time=slice(
                        dt,
                        dt
                        + self.model.timestep_length
                        - self.model.timestep_length / self.n,
                    )
                ).values
        else:
            data: npt.NDArray[np.float32] = self.reader.read_timestep(dt, n=self.n)

        interpolated: npt.NDArray[np.float32] = self.interpolate(data)
        valid: bool = self.validate(interpolated)
        if not valid:
            raise ValueError(
                f"Invalid data for time {dt} for variable {self.variable}."
            )
        assert interpolated.dtype == np.dtype(np.float32)
        return interpolated

    def set_forecast(self, forecast_issue_datetime: datetime, da: xr.DataArray) -> None:
        """Sets the loader to forecast mode.

        This means that data from before the forecast issue date will be loaded
        normally, while data from the forecast issue date onwards will be loaded
        from the forecast data file. Data beyond the available forecast data will
        raise an error.

        Args:
            forecast_issue_datetime: The datetime when the forecast starts.
            da: The xarray DataArray containing the forecast data.
        """
        self.forecast_issue_datetime: datetime = forecast_issue_datetime
        self.ds_forecast: xr.DataArray = da

    def unset_forecast(self) -> None:
        """Unset forecast mode."""
        self.ds_forecast: None = None
        self.forecast_issue_datetime: None = None

    @property
    def in_forecast_mode(self) -> bool:
        """Indicates whether the loader is in forecast mode.

        Returns:
            True if in forecast mode, False otherwise.
        """
        return self.forecast_issue_datetime is not None

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

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate the data array.

        Args:
            v: The data array to validate.

        Returns:
            True, indicating that by default all data is valid.

        Raises:
            ValueError: If the data shape does not match the expected grid shape.
        """
        if v.shape[0] != self.n:
            raise ValueError(f"Data time dimension does not match expected n {self.n}.")
        if v.shape[1] != self.ysize or v.shape[2] != self.xsize:
            raise ValueError(
                f"Data shape {v.shape[1:]} does not match expected grid shape "
                f"({self.ysize}, {self.xsize})."
            )
        if not self.validate_values(v):
            raise ValueError("Data validation failed.")

        return True

    @abstractmethod
    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate the data array.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        pass


class Precipitation(ForcingLoader):
    """Loader for precipitation data with specific validation."""

    def __init__(self, model: GEBModel, grid_mask: npt.NDArray[np.bool_]) -> None:
        """Initialize the Precipitation loader.

        Args:
            model: The GEB model instance.
            grid_mask: The mask of the model grid (True for valid points, False for invalid).
        """
        self.grid_mask = grid_mask
        super().__init__(model, "pr_kg_per_m2_per_s", 24)

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate precipitation data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return ((v_non_masked >= 0).all() and (v_non_masked < 500 / 3600).all()).item()


class Temperature(ForcingLoader):
    """Loader for temperature data with specific validation."""

    def __init__(
        self,
        model: GEBModel,
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

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate temperature data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return ((v_non_masked > 170).all() and (v_non_masked < 370).all()).item()

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
        self, model: GEBModel, direction: str, grid_mask: npt.NDArray[np.bool_]
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

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate wind data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return ((v_non_masked >= -150).all() and (v_non_masked < 150).all()).item()


class Pressure(ForcingLoader):
    """Loader for surface pressure data with elevation-based correction and validation."""

    def __init__(
        self,
        model: GEBModel,
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

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate surface pressure data.

        Args:
            v: The data array to validate (Pa).

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return ((v_non_masked > 30_000).all() and (v_non_masked < 120_000).all()).item()

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

    def __init__(self, model: GEBModel, grid_mask: npt.NDArray[np.bool_]) -> None:
        """Initialize the RSDS loader.

        Args:
            model: The GEB model instance.
            grid_mask: The mask of the model grid (True for valid points, False for invalid).
        """
        self.grid_mask = grid_mask
        super().__init__(model, "rsds_W_per_m2", 24)

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate surface downwelling shortwave radiation data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return (v_non_masked >= 0).all().item()


class RLDS(ForcingLoader):
    """Loader for surface downwelling longwave radiation data with specific validation."""

    def __init__(self, model: GEBModel, grid_mask: npt.NDArray[np.bool_]) -> None:
        """Initialize the RLDS loader.

        Args:
            model: The GEB model instance.
            grid_mask: The mask of the model grid (True for valid points, False for invalid).
        """
        self.grid_mask = grid_mask
        super().__init__(model, "rlds_W_per_m2", 24)

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate surface downwelling longwave radiation data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        v_non_masked = v[:, self.grid_mask]
        return (v_non_masked >= 0).all().item()


class SPEI(ForcingLoader):
    """Loader for Standardized Precipitation-Evapotranspiration Index (SPEI) data with specific validation."""

    def __init__(self, model: GEBModel, grid_mask: npt.NDArray[np.bool_]) -> None:
        """Initialize the SPEI loader.

        Args:
            model: The GEB model instance.
            grid_mask: The mask of the model grid (True for valid points, False for invalid).
        """
        self.grid_mask = grid_mask
        super().__init__(model, "SPEI", 1, supports_forecast=False)

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
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

    def __init__(self, model: GEBModel) -> None:
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
        valid: bool = self.validate_values(data)
        if not valid:
            raise ValueError(f"Invalid CO2 data for time {time}.")
        return data

    def validate_values(self, v: float) -> bool:
        """Validate CO2 concentration data.

        Args:
            v: the data value to validate.

        Returns:
            True if valid, otherwise raises ValueError.
        """
        return v > 270 and v < 2000

    @property
    def supports_forecast(self) -> bool:
        """Indicates whether the loader supports forecast mode.

        Returns:
            False as CO2 loader does not support forecast mode.
        """
        return False

    def set_forecast(self, forecast_issue_datetime: datetime, da: xr.DataArray) -> None:
        """Set forecast mode.

        CO2 loader does not support forecast mode.

        Raises:
            NotImplementedError: As CO2 loader does not support forecast mode.
        """
        raise NotImplementedError("CO2 loader does not support forecast mode.")

    def unset_forecast(self) -> None:
        """Unset forecast mode.

        CO2 loader does not support forecast mode.

        Raises:
            NotImplementedError: As CO2 loader does not support forecast mode.
        """
        raise NotImplementedError("CO2 loader does not support forecast mode.")


class Forcing(Module):
    """Module to handle climate forcing data.

    This module is responsible for loading and validating climate forcing data such as temperature, humidity, pressure, and radiation.
    It provides methods to load specific datasets and ensures that the data meets certain validation criteria.

    Args:
        model: The GEB model instance.
    """

    def __init__(self, model: GEBModel) -> None:
        """Initialize the Forcing module.

        All forcing loaders are initialized upfront to allow for efficient batch loading
        and inter-variable dependencies during interpolation.

        Args:
            model: The GEB model instance.
        """
        self.model = model
        self.forcing_DEM = read_grid(model.files["other"]["climate/elevation_forcing"])

        # Initialize all forcing loaders upfront
        self._initialize_loaders()

    def _initialize_loaders(self) -> None:
        """Initialize all forcing loaders.

        This creates instances of all forcing loaders so they're ready to use.
        """
        grid_mask = ~self.model.hydrology.grid.mask
        grid_DEM = self.model.hydrology.grid.decompress(
            self.model.hydrology.grid.var.elevation
        )

        # Initialize all loaders
        self._loaders: dict[str, ForcingLoader | CO2] = {
            "pr_kg_per_m2_per_s": Precipitation(self.model, grid_mask=grid_mask),
            "tas_2m_K": Temperature(
                self.model,
                forcing_DEM=self.forcing_DEM,
                grid_DEM=grid_DEM,
                grid_mask=grid_mask,
                dewpoint=False,
            ),
            "dewpoint_tas_2m_K": Temperature(
                self.model,
                forcing_DEM=self.forcing_DEM,
                grid_DEM=grid_DEM,
                grid_mask=grid_mask,
                dewpoint=True,
            ),
            "wind_u10m_m_per_s": Wind(self.model, direction="u", grid_mask=grid_mask),
            "wind_v10m_m_per_s": Wind(self.model, direction="v", grid_mask=grid_mask),
            "ps_pascal": Pressure(
                self.model,
                forcing_DEM=self.forcing_DEM,
                grid_DEM=grid_DEM,
                grid_mask=grid_mask,
            ),
            "rsds_W_per_m2": RSDS(self.model, grid_mask=grid_mask),
            "rlds_W_per_m2": RLDS(self.model, grid_mask=grid_mask),
            "SPEI": SPEI(self.model, grid_mask=grid_mask),
            "CO2_ppm": CO2(self.model),
        }

    @property
    def name(self) -> str:
        """Return the name of the module.

        Returns:
            The name of the module.
        """
        return "forcing"

    def __getitem__(self, name: str) -> ForcingLoader | CO2:
        """Get the forcing loader for a given name.

        Args:
            name: name of forcing dataset, e.g. "pr_kg_per_m2_per_s", "tas_2m_K", etc.

        Returns:
            The forcing loader for the specified dataset.

        Raises:
            KeyError: If the forcing variable name is not recognized.
        """
        if name not in self._loaders:
            raise KeyError(
                f"Forcing variable '{name}' not found. Available variables: {list(self._loaders.keys())}"
            )
        return self._loaders[name]

    @overload
    def load(self, name: Literal["CO2_ppm"], dt: datetime | None = None) -> float: ...

    @overload
    def load(self, name: str, dt: datetime | None = None) -> ThreeDArrayFloat32: ...

    def load(self, name: str, dt: datetime | None = None) -> ThreeDArrayFloat32 | float:
        """Load forcing data for a given name and time.

        Args:
            name: name of forcing dataset, e.g. "pr_kg_per_m2_per_s", "tas_2m_K", etc.
            dt: time of forcing data to be returned. Defaults to None, in which case
                the current time of the model is used.

        Returns:
            Forcing data as a numpy array or float.
        """
        if dt is None:
            dt: datetime = self.model.current_time

        return self[name].load(dt)

    @property
    def loaders(self) -> dict[str, ForcingLoader | CO2]:
        """Get all forcing loaders.

        Returns:
            A dictionary of all forcing loaders.
        """
        return self._loaders

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
