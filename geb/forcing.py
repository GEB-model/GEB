"""Module to handle climate forcing data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from geb.geb_types import (
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    TwoDArrayBool,
    TwoDArrayFloat32,
)
from geb.workflows.io import read_grid, read_table

from .module import Module
from .workflows.io import AsyncGriddedForcingReader

if TYPE_CHECKING:
    from geb.model import GEBModel


def generate_bilinear_interpolation_weights(
    src_x: ArrayFloat64,
    src_y: ArrayFloat64,
    tgt_x: ArrayFloat64,
    tgt_y: ArrayFloat64,
    mask: TwoDArrayBool | None = None,
    src_mask: TwoDArrayBool | None = None,
) -> tuple[ArrayInt32, ArrayFloat32]:
    """
    Generates indices and weights for bilinear interpolation.

    Assumes the source grid is rectilinear and monotonic.

    Args:
        src_x: Source grid x-coordinates (must be monotonic).
        src_y: Source grid y-coordinates (must be monotonic).
        tgt_x: Target points x-coordinates.
        tgt_y: Target points y-coordinates.
        mask: Optional mask of the target grid (True for masked/invalid points).
            If provided, only unmasked points will have indices and weights generated.
        src_mask: Optional mask of the source grid (True for kept/available
            source cells). If provided, the returned indices are mapped onto the
            compressed source-cell numbering defined by this mask.

    Returns:
        tuple[ArrayInt32, ArrayFloat32]: A tuple containing indices (int32) and
            weights (float32).

    Raises:
        ValueError: If target points are outside the source grid bounds.
        ValueError: If source x or y coordinates are not monotonic.
        ValueError: If source mask excludes cells needed for interpolation.
    """
    # Create the full grid of target points and flatten them
    # The 'xy' indexing creates a target grid of shape (len(tgt_y), len(tgt_x))
    # Note: len(tgt_y) is the row count (Ny_tgt), len(tgt_x) is the col count (Nx_tgt)
    tgt_x_2d, tgt_y_2d = np.meshgrid(tgt_x, tgt_y, indexing="xy")

    if mask is not None:
        # Filter target points by mask (keep unmasked points)
        tgt_x_flat = tgt_x_2d[~mask]
        tgt_y_flat = tgt_y_2d[~mask]
    else:
        # Flatten the target coordinates into (N_targets,) arrays
        tgt_x_flat = tgt_x_2d.flatten()
        tgt_y_flat = tgt_y_2d.flatten()

    # Source Grid Dimensions
    nx: int = len(src_x)
    ny: int = len(src_y)

    # Validate that x-coordinates are ascending
    if not np.all(src_x[1:] > src_x[:-1]):
        raise ValueError("Source x-coordinates must be strictly ascending.")

    # ix is the index of the source x-coordinate that is just less than or equal to tgt_x
    ix = np.searchsorted(src_x, tgt_x_flat) - 1
    # Clip indices to be within valid bounds for x
    ix = np.clip(ix, 0, nx - 2)
    # Check bounds for ix after clipping - if any targets are still out of bounds, it's an error
    if not np.all((tgt_x_flat >= src_x[0]) & (tgt_x_flat <= src_x[-1])):
        raise ValueError("Target x-coordinates are outside the source grid bounds.")

    is_y_descending = src_y[0] > src_y[-1]

    if not is_y_descending:
        # For ascending y-axis (normal case)
        iy = np.searchsorted(src_y, tgt_y_flat, side="right") - 1
        iy = np.clip(iy, 0, ny - 2)
        y0 = src_y[iy]
        y1 = src_y[iy + 1]
        dy = (tgt_y_flat - y0) / (y1 - y0)
    else:
        # For descending y-axis, use vectorized NumPy operations
        # Create a matrix comparison to find intervals efficiently
        # src_y[j] >= tgt_y >= src_y[j+1] for descending coordinates

        # Reshape for broadcasting: tgt_y as column, src_y as row
        tgt_y_col = tgt_y_flat[:, np.newaxis]  # Shape: (n_targets, 1)
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
        above_highest = tgt_y_flat > src_y[0]
        below_lowest = tgt_y_flat < src_y[-1]

        # Assign boundary indices for out-of-bounds points
        iy = np.where(no_interval_found & above_highest, 0, iy)
        iy = np.where(no_interval_found & below_lowest, ny - 2, iy)

        # Calculate dy for descending coordinates
        y0 = src_y[iy]
        y1 = src_y[iy + 1]
        # For descending coordinates, the weight should be calculated as:
        # dy = (y0 - target_y) / (y0 - y1) since y0 > y1
        dy = (y0 - tgt_y_flat) / (y0 - y1)  # Get the corner coordinates

    x0 = src_x[ix]
    x1 = src_x[ix + 1]

    # Calculate normalized distances
    dx = (tgt_x_flat - x0) / (x1 - x0)

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

    if src_mask is not None:
        if src_mask.shape != (ny, nx):
            raise ValueError(
                f"Source mask shape {src_mask.shape} does not match source grid shape {(ny, nx)}."
            )

        compressed_index_lookup = np.full(nx * ny, -1, dtype=np.int32)
        compressed_index_lookup[src_mask.reshape(-1)] = np.arange(
            src_mask.sum(), dtype=np.int32
        )

        idx00 = compressed_index_lookup[idx00]
        idx01 = compressed_index_lookup[idx01]
        idx10 = compressed_index_lookup[idx10]
        idx11 = compressed_index_lookup[idx11]

        if (
            np.any(idx00 < 0)
            or np.any(idx01 < 0)
            or np.any(idx10 < 0)
            or np.any(idx11 < 0)
        ):
            raise ValueError(
                "Source mask excludes cells needed by the bilinear interpolation stencils."
            )

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
        self,
        model: GEBModel,
        variable: str,
        n: int,
        grid_mask: TwoDArrayBool,
        supports_forecast: bool = True,
    ) -> None:
        """Initialize the ForcingLoader.

        Args:
            model: The GEB model instance.
            variable: The variable name to load (e.g., "pr" for precipitation).
            n: Number of time steps to load at once (default is 1).
            grid_mask: The mask of the model grid (True for masked cells, False
                for active cells).
            supports_forecast: Whether the loader supports forecast mode.
        """
        self.model: GEBModel = model
        self.n: int = n
        self.variable: str = variable
        self.grid_mask = grid_mask
        self.reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
            model.files["other"][f"climate/{variable}"], variable, asynchronous=True
        )
        self.forcing_mask = self._load_forcing_mask()

        self.indices, self.weights = generate_bilinear_interpolation_weights(
            self.forcing_mask.x.values,
            self.forcing_mask.y.values,
            model.hydrology.grid.lon,
            model.hydrology.grid.lat,
            mask=self.grid_mask,
            src_mask=self.forcing_mask.values,
        )
        self.output_size: int = self.indices.shape[0]
        self.xsize: int = model.hydrology.grid.lon.size
        self.ysize: int = model.hydrology.grid.lat.size

        self._supports_forecast = supports_forecast
        self.forecast_issue_datetime: datetime | None = None
        self.ds_forecast: xr.DataArray | None = None

    def _load_forcing_mask(self) -> xr.DataArray:
        """Load the forcing-grid keep mask if it is available.

        Returns:
            The forcing-grid keep mask where True marks source cells kept during
            forcing preprocessing.
        """
        mask_key = f"climate/{self.variable}_mask"
        forcing_mask: xr.DataArray = xr.open_dataarray(
            self.model.files["other"][mask_key], consolidated=False
        )
        return forcing_mask

    def _compress_source_values(
        self,
        values: npt.NDArray[np.float32],
        name: str,
    ) -> ArrayFloat32:
        """Compress a source-grid field with the forcing keep mask.

        Args:
            values: Values on the forcing grid or already compressed to the
                kept source cells.
            name: Name used in validation errors.

        Returns:
            Compressed values aligned with the source-cell dimension read from
            forcing storage.

        Raises:
            ValueError: If the input shape does not match the forcing mask.
        """
        if values.ndim == 1:
            expected_size = int(self.forcing_mask.values.sum())
            if values.size != expected_size:
                raise ValueError(
                    f"{name} size {values.size} does not match the expected "
                    f"compressed source size {expected_size}."
                )
            return values.astype(np.float32, copy=False)

        if values.shape != self.forcing_mask.shape:
            raise ValueError(
                f"{name} shape {values.shape} does not match forcing mask shape "
                f"{self.forcing_mask.shape}."
            )

        return values[self.forcing_mask.values]

    @property
    def in_forecast_mode(self) -> bool:
        """Indicates whether the loader is in forecast mode.

        Returns:
            True if in forecast mode, False otherwise.
        """
        return self.forecast_issue_datetime is not None

    @property
    def supports_forecast(self) -> bool:
        """Indicates whether the loader supports forecast mode.

        Returns:
            True if forecast mode is supported, False otherwise.
        """
        return self._supports_forecast

    def load(self, dt: datetime) -> TwoDArrayFloat32:
        """Load and validate forcing data for a given time.

        If in forecast mode and the time is after the forecast issue date,
        the data is loaded from the forecast dataset.

        Otherwise, data is loaded from the standard reader.

        Args:
            dt: The datetime for which to load the data.

        Returns:
            The interpolated and validated compressed data with shape
            (n_active_cells, time).

        Raises:
            ValueError: If the data is invalid according to the validation criteria.
        """
        # check if we are in forecasting mode, and if the end of the timestep is after the
        # start of the forecast
        if (
            self.forecast_issue_datetime is not None
            and dt + self.model.timestep_length >= self.forecast_issue_datetime
        ):
            if self.ds_forecast is None:
                raise ValueError(
                    "Forecast data array is not set, but loader is in forecast mode."
                )
            # find how many substeps to load from the normal data source
            substeps_to_forecast: int = int(
                (self.forecast_issue_datetime - dt).total_seconds()
                / (self.model.timestep_length / self.n).total_seconds()
            )
            # if some substeps are before the forecast, load them from the normal reader
            if substeps_to_forecast > 0:
                # TODO: The reader breaks when loading less than n timesteps, so we load n and slice
                # the data
                non_forecast_data_all, normal_start_date = self.reader.read_timestep(
                    dt, n=self.n
                )

                if normal_start_date != np.datetime64(dt, "ns"):
                    raise ValueError(
                        f"Standard reader returned data starting at {normal_start_date}, but expected {dt}."
                    )

                non_forecast_data: npt.NDArray[np.float32] = non_forecast_data_all[
                    :substeps_to_forecast
                ]

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
            data_all, normal_start_date = self.reader.read_timestep(dt, n=self.n)
            if normal_start_date != np.datetime64(dt, "ns"):
                raise ValueError(
                    f"Standard reader returned data starting at {normal_start_date}, but expected {dt}."
                )
            data: npt.NDArray[np.float32] = data_all

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
        self.forecast_issue_datetime = forecast_issue_datetime
        self.ds_forecast = da

    def unset_forecast(self) -> None:
        """Unset forecast mode."""
        self.ds_forecast = None
        self.forecast_issue_datetime = None

    def interpolate(self, data: npt.NDArray[np.float32]) -> npt.NDArray[Any]:
        """Interpolate data to the model grid using bilinear interpolation.

        If a mask was provided during initialization, the output will be
        compressed (only unmasked points returned).

        Args:
            data: The input data array to interpolate.

        Returns:
            The interpolated data array.
        """
        # data has shape (n_active_cells, n_timesteps)
        # the corner values must be gathered in the same order as the weights
        corner_values = data[self.indices, :]  # (N_target, 4, n_timesteps)
        interpolated = np.sum(
            corner_values * self.weights[:, :, np.newaxis], axis=1
        )  # Result is (N_target, n_timesteps)

        return interpolated

    def validate(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate the data array.

        Args:
            v: The data array to validate (possibly compressed).

        Returns:
            True if valid.

        Raises:
            ValueError: If the data shape does not match the expected dimensions.
        """
        if v.shape[1] != self.n:
            raise ValueError(f"Data time dimension does not match expected n {self.n}.")

        if v.ndim != 2:
            raise ValueError(
                f"Compressed data must be 2D, received array with shape {v.shape}."
            )
        if v.shape[0] != self.output_size:
            raise ValueError(
                f"Compressed data shape {v.shape[0]} does not match "
                f"expected number of active points {self.output_size}."
            )

        if not self.validate_values(v):
            raise ValueError("Data validation failed.")

        return True

    @abstractmethod
    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate the data array values.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise False.
        """
        pass


class Precipitation(ForcingLoader):
    """Loader for precipitation data with specific validation."""

    def __init__(self, model: GEBModel, grid_mask: TwoDArrayBool) -> None:
        """Initialize the Precipitation loader.

        Args:
            model: The GEB model instance.
            grid_mask: The mask of the model grid.
        """
        super().__init__(model, "pr_kg_per_m2_per_s", 24, grid_mask=grid_mask)

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate precipitation data.

        Args:
            v: The data array to validate (compressed).

        Returns:
            True if valid, otherwise False.
        """
        return ((v >= 0).all() and (v < 500 / 3600).all()).item()


class Temperature(ForcingLoader):
    """Loader for temperature data with specific validation."""

    def __init__(
        self,
        model: GEBModel,
        forcing_DEM: npt.NDArray[np.float32],
        grid_DEM: npt.NDArray[np.float32],
        grid_mask: TwoDArrayBool,
        dewpoint: bool = False,
        lapse_rate: float = -0.0065,
    ) -> None:
        """Initialize the Temperature loader.

        Args:
            model: The GEB model instance.
            forcing_DEM: The DEM used in the forcing data.
            grid_DEM: The DEM used in the model grid.
            grid_mask: The mask of the model grid.
            dewpoint: If True, load dewpoint temperature.
            lapse_rate: The lapse rate in K/m.
        """
        self.lapse_rate = lapse_rate

        super().__init__(
            model,
            "tas_2m_K" if not dewpoint else "dewpoint_tas_2m_K",
            24,
            grid_mask=grid_mask,
        )
        self.forcing_DEM_compressed = self._compress_source_values(
            forcing_DEM,
            "forcing_DEM",
        )
        self.grid_DEM_compressed: ArrayFloat32 = grid_DEM[~grid_mask]

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate temperature data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise False.
        """
        return ((v > 170).all() and (v < 370).all()).item()

    def interpolate(self, data: npt.NDArray[np.float32]) -> npt.NDArray[Any]:
        """Interpolate temperature with lapse-rate correction.

        Args:
            data: The input data array to interpolate.

        Returns:
            The interpolated temperature data (compressed).
        """
        # data has shape (n_source_cells, n_timesteps)
        temperature_sea_level = (
            data - self.forcing_DEM_compressed[:, np.newaxis] * self.lapse_rate
        )
        interpolated_temperature = (
            super().interpolate(temperature_sea_level)
            + self.grid_DEM_compressed[:, np.newaxis] * self.lapse_rate
        )

        return interpolated_temperature


class Wind(ForcingLoader):
    """Loader for wind data with specific validation."""

    def __init__(
        self, model: GEBModel, direction: str, grid_mask: TwoDArrayBool
    ) -> None:
        """Initialize the Wind loader.

        Args:
            model: The GEB model instance.
            direction: either "u" or "v".
            grid_mask: The mask of the model grid.

        Raises:
            ValueError: If direction is not "u" or "v".
        """
        if direction not in ["u", "v"]:
            raise ValueError("Direction must be 'u' or 'v'")
        super().__init__(model, f"wind_{direction}10m_m_per_s", 24, grid_mask=grid_mask)

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate wind data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise False.
        """
        return ((v >= -150).all() and (v < 150).all()).item()


class Pressure(ForcingLoader):
    """Loader for surface pressure data with elevation-based correction."""

    def __init__(
        self,
        model: GEBModel,
        forcing_DEM: npt.NDArray[np.float32],
        grid_DEM: npt.NDArray[np.float32],
        grid_mask: TwoDArrayBool,
        g: float = 9.80665,
        Mo: float = 0.0289644,
        lapse_rate: float = -0.0065,
    ) -> None:
        """Initialize the Pressure loader.

        Args:
            model: The GEB model instance.
            forcing_DEM: The DEM for forcing.
            grid_DEM: The DEM for the grid.
            grid_mask: The mask for the grid.
            g: Gravity constant.
            Mo: Molecular weight of air.
            lapse_rate: Lapse rate.
        """
        self.g = g
        self.Mo = Mo
        self.lapse_rate = lapse_rate

        super().__init__(model, "ps_pascal", 24, grid_mask=grid_mask)
        self.forcing_pressure_correction_factor = get_pressure_correction_factor(
            self._compress_source_values(forcing_DEM, "forcing_DEM"),
            g,
            Mo,
            lapse_rate,
        )
        self.grid_pressure_correction_factor: ArrayFloat32 = (
            get_pressure_correction_factor(
                grid_DEM[~grid_mask],
                g,
                Mo,
                lapse_rate,
            )
        )

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate surface pressure data.

        Args:
            v: The data array to validate.

        Returns:
            True if valid, otherwise False.
        """
        return ((v > 30_000).all() and (v < 120_000).all()).item()

    def interpolate(self, data: npt.NDArray[np.float32]) -> npt.NDArray[Any]:
        """Interpolate pressure with elevation correction.

        Args:
            data: The input data array to interpolate.

        Returns:
            The interpolated pressure data (compressed).
        """
        # data has shape (n_source_cells, n_timesteps)
        pressure_sea_level = (
            data / self.forcing_pressure_correction_factor[:, np.newaxis]
        )

        # Interpolate the sea level pressure
        interpolated_pressure_sea_level = super().interpolate(pressure_sea_level)

        interpolated_pressure = (
            interpolated_pressure_sea_level
            * self.grid_pressure_correction_factor[:, np.newaxis]
        )

        return interpolated_pressure


class RSDS(ForcingLoader):
    """Loader for surface downwelling shortwave radiation data."""

    def __init__(self, model: GEBModel, grid_mask: TwoDArrayBool) -> None:
        """Initialize RSDS.

        Args:
            model: model instance.
            grid_mask: grid mask.
        """
        super().__init__(model, "rsds_W_per_m2", 24, grid_mask=grid_mask)

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate values.

        Args:
            v: data array.

        Returns:
            True if valid.
        """
        return (v >= 0).all().item()


class RLDS(ForcingLoader):
    """Loader for surface downwelling longwave radiation data."""

    def __init__(self, model: GEBModel, grid_mask: TwoDArrayBool) -> None:
        """Initialize RLDS.

        Args:
            model: model instance.
            grid_mask: grid mask.
        """
        super().__init__(model, "rlds_W_per_m2", 24, grid_mask=grid_mask)

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate values.

        Args:
            v: data array.

        Returns:
            True if valid.
        """
        return (v >= 0).all().item()


class SPEI(ForcingLoader):
    """Loader for SPEI data."""

    def __init__(self, model: GEBModel, grid_mask: TwoDArrayBool) -> None:
        """Initialize SPEI.

        Args:
            model: model instance.
            grid_mask: grid mask.
        """
        super().__init__(model, "SPEI", 1, grid_mask=grid_mask, supports_forecast=False)

    def validate_values(self, v: npt.NDArray[np.float32]) -> bool:
        """Validate values.

        Args:
            v: data array.

        Returns:
            True if valid.
        """
        return not np.isnan(v).any()


class CO2:
    """Loader for CO2 concentration data."""

    def __init__(self, model: GEBModel) -> None:
        """Initialize CO2 loader.

        Args:
            model: model instance.
        """
        self.df = read_table(model.files["table"]["climate/CO2_ppm"])

    def load(self, time: datetime) -> float:
        """Load CO2.

        Args:
            time: datetime.

        Returns:
            CO2 concentration.

        Raises:
            ValueError: If the annual CO2 concentration is outside the expected
                range.
        """
        data: float = self.df.loc[time.year].item()
        if not self.validate_values(data):
            raise ValueError(f"Invalid CO2 data for time {time}.")
        return data

    def validate_values(self, v: float) -> bool:
        """Validate values.

        Args:
            v: value.

        Returns:
            True if valid.
        """
        return v > 270 and v < 2000

    @property
    def supports_forecast(self) -> bool:
        """Return whether forecast mode is supported.

        Returns:
            False.
        """
        return False

    def set_forecast(self, forecast_issue_datetime: datetime, da: xr.DataArray) -> None:
        """Set forecast.

        Args:
            forecast_issue_datetime: dt.
            da: dataarray.
        """
        raise NotImplementedError("CO2 loader does not support forecast mode.")

    def unset_forecast(self) -> None:
        """Unset forecast."""
        raise NotImplementedError("CO2 loader does not support forecast mode.")


class Forcing(Module):
    """Module to handle climate forcing data."""

    def __init__(self, model: GEBModel) -> None:
        """Initialize forcing module.

        Args:
            model: model instance.
        """
        self.model = model
        self.forcing_DEM = read_grid(model.files["other"]["climate/elevation_forcing"])
        self._initialize_loaders()

    def _initialize_loaders(self) -> None:
        """Initialize loaders."""
        grid_mask = self.model.hydrology.grid.mask
        grid_DEM = self.model.hydrology.grid.decompress(
            self.model.hydrology.grid.var.elevation
        )

        self._loaders: dict[str, CO2 | ForcingLoader] = {
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
        """Module name.

        Returns:
            Name.
        """
        return "forcing"

    @overload
    def __getitem__(self, name: Literal["CO2_ppm"]) -> CO2: ...

    @overload
    def __getitem__(self, name: str) -> ForcingLoader: ...

    def __getitem__(self, name: str) -> ForcingLoader | CO2:
        """Get loader.

        Args:
            name: variable name.

        Returns:
            Loader.

        Raises:
            KeyError: If the forcing variable name is not available.
        """
        if name not in self._loaders:
            raise KeyError(
                f"Forcing variable '{name}' not found. Available variables: {list(self._loaders.keys())}"
            )
        return self._loaders[name]

    @overload
    def load(self, name: Literal["CO2_ppm"], dt: datetime | None = None) -> float: ...

    @overload
    def load(
        self,
        name: Literal[
            "pr_kg_per_m2_per_s",
            "tas_2m_K",
            "dewpoint_tas_2m_K",
            "wind_u10m_m_per_s",
            "wind_v10m_m_per_s",
            "ps_pascal",
            "rsds_W_per_m2",
            "rlds_W_per_m2",
        ],
        dt: datetime | None = None,
    ) -> TwoDArrayFloat32: ...

    @overload
    def load(
        self, name: Literal["SPEI"], dt: datetime | None = None
    ) -> TwoDArrayFloat32: ...

    @overload
    def load(self, name: str, dt: datetime | None = None) -> TwoDArrayFloat32: ...

    def load(self, name: str, dt: datetime | None = None) -> TwoDArrayFloat32 | float:
        """Load data.

        Args:
            name: variable name.
            dt: datetime.

        Returns:
            Data.
        """
        if dt is None:
            dt = self.model.current_time + timedelta(hours=1)
        return self[name].load(dt)

    @property
    def loaders(self) -> dict[str, CO2 | ForcingLoader]:
        """All loaders.

        Returns:
            loaders.
        """
        return self._loaders

    @property
    def forcing_loaders(self) -> dict[str, ForcingLoader]:
        """Forcing loaders.

        Returns:
            forcing loaders.
        """
        return {k: v for k, v in self._loaders.items() if isinstance(v, ForcingLoader)}

    def spinup(self) -> None:
        """Spinup."""
        pass

    def step(self) -> None:
        """Step."""
        pass
