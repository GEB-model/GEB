"""This module contains the Reporter class, which is used to report data to disk."""

import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import zarr.storage
from dateutil.relativedelta import relativedelta
from xarray.backends.zarr import FillValueCoder
from zarr.abc.codec import ArrayArrayCodec, BytesBytesCodec
from zarr.codecs import ZstdCodec
from zarr.codecs.numcodecs import (
    BitRound,
    PackBits,
    Shuffle,
)

from geb.geb_types import ArrayFloat32, ArrayFloat64, ArrayInt64, TwoDArrayInt32
from geb.hydrology.routing import get_upstream_represented_xys
from geb.module import Module
from geb.store import DynamicArray
from geb.workflows.io import fast_rmtree, read_geom, write_table
from geb.workflows.methods import multi_level_merge
from geb.workflows.raster import coord_to_pixel

if TYPE_CHECKING:
    from geb.model import GEBModel

WATER_CIRCLE_REPORT_CONFIG: dict[str, str | dict[str, str | dict[str, str]]] = {
    "hydrology": {
        "_current_storage": {
            "varname": ".current_storage",
            "type": "scalar",
        },
        "_routing_loss_m3": {
            "varname": ".routing_loss_m3",
            "type": "scalar",
        },
    },
    "hydrology.landsurface": {
        "_rain_m": {
            "varname": ".rain_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_snow_m": {
            "varname": ".snow_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_transpiration_m": {
            "varname": ".transpiration_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_bare_soil_evaporation_m": {
            "varname": ".bare_soil_evaporation_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_open_water_evaporation_m": {
            "varname": ".open_water_evaporation_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_interception_evaporation_m": {
            "varname": ".interception_evaporation_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_sublimation_or_deposition_m": {
            "varname": ".sublimation_or_deposition_m",
            "type": "HRU",
            "function": "weightedsum",
        },
    },
    "hydrology.routing": {
        "_total_evaporation_in_rivers_m3": {
            "varname": ".total_evaporation_in_rivers_m3",
            "type": "scalar",
        },
        "_total_waterbody_evaporation_m3": {
            "varname": ".total_waterbody_evaporation_m3",
            "type": "scalar",
        },
        "_total_outflow_at_pits_m3": {
            "varname": ".total_outflow_at_pits_m3",
            "type": "scalar",
        },
    },
    "hydrology.water_demand": {
        "_domestic_water_loss_m3": {
            "varname": ".domestic_water_loss_m3",
            "type": "scalar",
        },
        "_industry_water_loss_m3": {
            "varname": ".industry_water_loss_m3",
            "type": "scalar",
        },
        "_livestock_water_loss_m3": {
            "varname": ".livestock_water_loss_m3",
            "type": "scalar",
        },
    },
}

WATER_BALANCE_REPORT_CONFIG: dict[str, dict[str, dict[str, str]]] = {
    "hydrology": {
        "_current_storage": {
            "varname": ".current_storage",
            "type": "scalar",
        },
        "_routing_loss_m3": {
            "varname": ".routing_loss_m3",
            "type": "scalar",
        },
    },
    "hydrology.landsurface": {
        "_rain_m": {
            "varname": ".rain_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_snow_m": {
            "varname": ".snow_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_transpiration_m": {
            "varname": ".transpiration_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_bare_soil_evaporation_m": {
            "varname": ".bare_soil_evaporation_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_open_water_evaporation_m": {
            "varname": ".open_water_evaporation_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_interception_evaporation_m": {
            "varname": ".interception_evaporation_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_potential_evapotranspiration_m": {
            "varname": ".potential_evapotranspiration_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_sublimation_or_deposition_m": {
            "varname": ".sublimation_or_deposition_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_interflow_m": {
            "varname": ".interflow_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        # _rain_m and _snow_m above already cover top-soil precipitation and snow inputs
        "_top_soil_water_content_m": {
            "varname": "HRU.var.water_content_m[0]",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_runoff_m_daily": {
            "varname": ".runoff_m_daily",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_top_soil_evaporation_m": {
            "varname": ".top_soil_evaporation_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_top_soil_infiltration_m": {
            "varname": ".top_soil_infiltration_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_top_soil_rise_from_layer_2_m": {
            "varname": ".top_soil_rise_from_layer_2_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_top_soil_percolation_to_layer_2_m": {
            "varname": ".top_soil_percolation_to_layer_2_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_top_soil_transpiration_m": {
            "varname": ".top_soil_transpiration_m",
            "type": "HRU",
            "function": "weightedsum",
        },
    },
    "hydrology.routing": {
        "_total_evaporation_in_rivers_m3": {
            "varname": ".total_evaporation_in_rivers_m3",
            "type": "scalar",
        },
        "_total_waterbody_evaporation_m3": {
            "varname": ".total_waterbody_evaporation_m3",
            "type": "scalar",
        },
        "_total_outflow_at_pits_m3": {
            "varname": ".total_outflow_at_pits_m3",
            "type": "scalar",
        },
    },
    "hydrology.water_demand": {
        "_domestic_water_loss_m3": {
            "varname": ".domestic_water_loss_m3",
            "type": "scalar",
        },
        "_industry_water_loss_m3": {
            "varname": ".industry_water_loss_m3",
            "type": "scalar",
        },
        "_livestock_water_loss_m3": {
            "varname": ".livestock_water_loss_m3",
            "type": "scalar",
        },
    },
}

WATER_STORAGE_REPORT_CONFIG: dict[str, dict[str, dict[str, str]]] = {
    "hydrology.landsurface": {
        "_soil_water_content_layer_0_m": {
            "varname": "HRU.var.water_content_m[0]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_soil_water_content_layer_1_m": {
            "varname": "HRU.var.water_content_m[1]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_soil_water_content_layer_2_m": {
            "varname": "HRU.var.water_content_m[2]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_soil_water_content_layer_3_m": {
            "varname": "HRU.var.water_content_m[3]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_soil_water_content_layer_4_m": {
            "varname": "HRU.var.water_content_m[4]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_soil_water_content_layer_5_m": {
            "varname": "HRU.var.water_content_m[5]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_snow_water_equivalent_m_top": {
            "varname": "HRU.var.snow_water_equivalent_m[:,0]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_snow_water_equivalent_m_bottom": {
            "varname": "HRU.var.snow_water_equivalent_m[:,1]",
            "type": "HRU",
            "function": "weightedmean",
        },
    },
}

OUTFLOW_PLOT_CONTEXT_REPORT_CONFIG: dict[str, dict[str, dict[str, str]]] = {
    "hydrology.landsurface": {
        "_top_soil_frozen_fraction": {
            "varname": ".top_soil_frozen_fraction",
            "type": "HRU",
            "function": "weightedmean",
        },
    },
}

ENERGY_BALANCE_REPORT_CONFIG: dict[str, dict[str, dict[str, str]]] = {
    "hydrology.landsurface": {
        "_soil_temperature_layer_0_C": {
            "varname": ".soil_temperature_C[0]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_soil_temperature_layer_1_C": {
            "varname": ".soil_temperature_C[1]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_soil_temperature_layer_2_C": {
            "varname": ".soil_temperature_C[2]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_soil_temperature_layer_3_C": {
            "varname": ".soil_temperature_C[3]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_soil_temperature_layer_4_C": {
            "varname": ".soil_temperature_C[4]",
            "type": "HRU",
            "function": "weightedmean",
        },
        "_soil_temperature_layer_5_C": {
            "varname": ".soil_temperature_C[5]",
            "type": "HRU",
            "function": "weightedmean",
        },
    },
}


def get_fill_value(
    data: np.ndarray[Any] | DynamicArray,
) -> tuple[int | float | None, Any]:
    """Get the fill value for a zarr array based on the data type.

    Args:
        data: The data array.

    Returns:
        A tuple containing the fill value and the encoded fill value for the zarr array.

    Raises:
        ValueError: If the data type is not recognized.
    """
    if np.issubdtype(data.dtype, np.floating):
        kind = np.dtype(data.dtype)
        fill_value: int | float = np.nan
    elif np.issubdtype(data.dtype, np.integer):
        kind = np.dtype(data.dtype)
        fill_value = -1
    elif data.dtype == bool:
        fill_value = None
    else:
        raise ValueError(f"Value dtype {data.dtype} not recognized.")

    if fill_value is not None:
        encoded_fill_value = FillValueCoder.encode(fill_value, kind)
    else:
        encoded_fill_value = None

    return fill_value, encoded_fill_value


def get_time_chunk_size(
    dtype: np.dtype, *other_dim_sizes: int, target_size_bytes: int = 100_000_000
) -> int:
    """Get the time chunk size for a zarr array.

    Args:
        dtype: The data type of the array.
        other_dim_sizes: The sizes of the other dimensions of the array.
        target_size_bytes: The target size of the chunk in bytes.

    Returns:
        time_chunk_size: The size of the time chunk.
    """
    type_size: int = np.dtype(dtype).itemsize
    other_dims_size: np.int64 = np.prod(other_dim_sizes)
    time_chunk_size: int = (target_size_bytes // (type_size * other_dims_size)).item()
    return max(1, time_chunk_size)


def create_time_array(
    start: datetime.datetime,
    end: datetime.datetime,
    timestep: datetime.timedelta | relativedelta,
    conf: dict,
    substeps: None | int = None,
) -> ArrayInt64:
    """Create a time array based on the start and end time, the timestep, and the frequency.

    Args:
        start: The start time.
        end: The end time.
        timestep: The timestep length.
        conf: The configuration for the frequency.
        substeps: The number of substeps per timestep.

    Returns:
        time: The time array.

    Raises:
        ValueError: If the frequency is not recognized.
        ValueError: If substeps are provided for a frequency that does not support them.
    """
    if "frequency" not in conf:
        frequency = {"every": "day"}
    else:
        frequency = conf["frequency"]
    if "every" in frequency:
        every = frequency["every"]
        time = []
        current_time = start
        while current_time <= end:
            if every == "year":
                if substeps is not None:
                    raise ValueError(
                        "Substeps not supported for yearly frequency in create_time_array."
                    )
                if (
                    frequency["month"] == current_time.month
                    and frequency["day"] == current_time.day
                ):
                    time.append(current_time)
            elif every == "month":
                if substeps is not None:
                    raise ValueError(
                        "Substeps not supported for monthly frequency in create_time_array."
                    )
                if frequency["day"] == current_time.day:
                    time.append(current_time)
            elif every == "day":
                if substeps is None:
                    time.append(current_time)
                else:
                    for substep in range(substeps):
                        time.append(current_time + substep * (timestep / substeps))
            current_time += timestep
    elif frequency == "initial":
        if substeps is not None:
            raise ValueError(
                "Substeps not supported for initial frequency in create_time_array."
            )
        time = [start]
    elif frequency == "final":
        if substeps is not None:
            raise ValueError(
                "Substeps not supported for final frequency in create_time_array."
            )
        time = [end]
    else:
        raise ValueError(f"Frequency {frequency} not recognized.")

    time_array = (
        np.array(time, dtype="datetime64[ns]").astype("datetime64[s]").astype(np.int64)
    )
    return time_array


def get_filters_and_compressors(
    dtype: np.dtype,
    compression_level: int,
) -> tuple[list[ArrayArrayCodec], list[BytesBytesCodec]]:
    """Select zarr filters and compressors appropriate for a given dtype.

    The chosen pipeline balances precision loss, byte-shuffle efficiency,
    and final compression ratio:
    - Floating-point types: BitRound (lossy precision reduction) followed by
      byte Shuffle (improves compressibility of mantissa bits) then ZstdCodec.
      keepbits values are chosen to retain ~0.01% relative error,
      which is negligible for most applications.
    - Boolean: PackBits to pack 8 booleans per byte, then ZstdCodec.
    - All other types (integers, etc.): only ZstdCodec; Shuffle is omitted
      because unstructured integer data rarely benefits from byte-shuffling.

    Args:
        dtype: NumPy dtype of the array to be stored.
        compression_level: Zstd compression level (1 = fast, 22 = best).

    Returns:
        filters: Array-to-array codecs applied before byte conversion.
        compressors: Bytes-to-bytes codecs applied after filters, always
            ending with ZstdCodec at the given compression level.
    """
    filters: list[ArrayArrayCodec]
    compressors: list[BytesBytesCodec]
    if dtype == bool:
        filters = [PackBits()]
        compressors = []
    elif np.issubdtype(dtype, np.float64):
        # keepbits=14 retains ~0.006% relative error; Shuffle exploits
        # the repetitive byte patterns in rounded mantissa bits
        filters = [BitRound(keepbits=14)]
        compressors = [Shuffle(elementsize=8)]
    elif np.issubdtype(dtype, np.float32):
        # keepbits=13 retains ~0.012% relative error
        filters = [BitRound(keepbits=13)]
        compressors = [Shuffle(elementsize=4)]
    elif np.issubdtype(dtype, np.float16):
        # keepbits=10 retains ~0.098% relative error
        filters = [BitRound(keepbits=10)]
        compressors = [Shuffle(elementsize=2)]
    else:
        filters = []
        compressors = []
    compressors.append(ZstdCodec(level=compression_level))
    return filters, compressors


def prepare_gridded_group(
    name: str,
    config: dict,
    lon: ArrayFloat32 | ArrayFloat64,
    lat: ArrayFloat32 | ArrayFloat64,
    crs: str,
    example_value: np.ndarray[Any],
    time: ArrayInt64,
    time_chunk_size: int,
    compression_level: int,
) -> None:
    """Create a zarr group for gridded data.

    Args:
        name: The name of the variable.
        config: The configuration for the variable.
        lon: The longitude array.
        lat: The latitude array.
        crs: The coordinate reference system in WKT format.
        example_value: An example value to determine the data type.
        time: The time array.
        time_chunk_size: The size of the time chunk.
        compression_level: The compression level for the zarr array.
    """
    root_group = config["_root_group"]

    # Create the y coordinate array
    y_group = root_group.create_array(
        "y",
        shape=lat.shape,
        dtype=lat.dtype,
        dimension_names=["y"],
        attributes={
            "standard_name": "latitude",
            "units": "degrees_north",
        },
    )
    y_group[:] = lat

    # Create the x coordinate array
    x_group = root_group.create_array(
        "x",
        shape=lon.shape,
        dtype=lon.dtype,
        dimension_names=["x"],
        attributes={
            "standard_name": "longitude",
            "units": "degrees_east",
        },
    )
    x_group[:] = lon

    # Create the time coordinate array
    time_group = root_group.create_array(
        "time",
        shape=time.shape,
        dtype=time.dtype,
        dimension_names=["time"],
        attributes={
            "standard_name": "time",
            "units": "seconds since 1970-01-01T00:00:00",
            "calendar": "gregorian",
        },
    )
    time_group[:] = time

    fill_value, encoded_fill_value = get_fill_value(example_value)

    attributes = {
        "grid_mapping": "crs",
        "coordinates": "time y x",
        "units": "unknown",
        "long_name": name,
        "_CRS": {"wkt": crs},
    }
    if encoded_fill_value is not None:
        attributes["_FillValue"] = encoded_fill_value

    filters, compressors = get_filters_and_compressors(
        example_value.dtype, compression_level
    )

    # Create the variable array
    variable_data: zarr.Array[Any] = root_group.create_array(
        name,
        shape=(
            lat.size,
            lon.size,
            time_group.size,
        ),
        chunks=(
            lat.size,
            lon.size,
            time_chunk_size,
        ),
        dtype=example_value.dtype,
        compressors=compressors,
        filters=filters,
        fill_value=fill_value,
        dimension_names=[
            "y",
            "x",
            "time",
        ],
        attributes=attributes,
    )

    # Pre-allocate the writing buffer for the chunks
    config["_chunk_data"] = np.full(
        (lat.size, lon.size, time_chunk_size),
        fill_value,
        dtype=example_value.dtype,
    )


def prepare_agent_group(
    name: str,
    config: dict,
    time: ArrayInt64,
    example_value: np.ndarray[Any] | int | float | bool,
    chunk_target_size_bytes: int,
    compression_level: int,
) -> None:
    """Create a zarr group for agent data.

    Args:
        name: The name of the variable.
        config: The configuration for the variable.
        time: The time array.
        example_value: An example value to determine the data type.
        chunk_target_size_bytes: The target size of the chunk in bytes.
        compression_level: The compression level for the zarr array.

    Raises:
        ValueError: If the example value type is not recognized.
    """
    root_group = config["_root_group"]

    # Create the time coordinate array
    time_group = root_group.create_array(
        "time",
        shape=time.shape,
        dtype=time.dtype,
        dimension_names=["time"],
        attributes={
            "standard_name": "time",
            "units": "seconds since 1970-01-01T00:00:00",
            "calendar": "gregorian",
        },
    )
    time_group[:] = time

    # Determine chunk size and shape based on example value
    if isinstance(example_value, (float, int, bool)):
        shape = (time_group.size,)
        time_chunk_size = get_time_chunk_size(
            np.dtype(type(example_value)),
            1,
            target_size_bytes=chunk_target_size_bytes,
        )
        if isinstance(example_value, float):
            dtype_ = np.float32
            example_value = np.array(example_value, dtype=np.float32)
        elif isinstance(example_value, bool):
            dtype_ = bool
            example_value = np.array(example_value, dtype=bool)
        elif isinstance(example_value, int):
            dtype_ = np.int32
            example_value = np.array(example_value, dtype=np.int32)
        else:
            raise ValueError(
                f"Example value of type {type(example_value)} not recognized."
            )
        time_chunk_size = min(time_chunk_size, time_group.size)
        chunks = (time_chunk_size,)
        array_dimensions = ["time"]
    else:
        shape = (time_group.size, example_value.size)
        dtype_ = example_value.dtype
        time_chunk_size = get_time_chunk_size(
            np.dtype(example_value.dtype),
            example_value.size,
            target_size_bytes=chunk_target_size_bytes,
        )
        time_chunk_size = min(time_chunk_size, time_group.size)
        chunks = (time_chunk_size, example_value.size)
        array_dimensions = ["time", "agents"]

    filters, compressors = get_filters_and_compressors(
        np.dtype(dtype_), compression_level
    )

    fill_value, encoded_fill_value = get_fill_value(example_value)
    attributes = {
        "coordinates": "time",
        "units": "unknown",
        "long_name": name,
    }
    if encoded_fill_value is not None:
        attributes["_FillValue"] = encoded_fill_value

    root_group.create_array(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype_,
        filters=filters,
        compressors=compressors,
        fill_value=fill_value,
        dimension_names=array_dimensions,
        attributes=attributes,
    )
    # Pre-allocate the writing buffer for the chunks
    if isinstance(example_value, (float, int)):
        config["_chunk_data"] = np.full((time_chunk_size,), fill_value)
    else:
        config["_chunk_data"] = np.full(
            (time_chunk_size, example_value.size), fill_value
        )


RE_SQUARE_BRACKETS = re.compile(r"\[.*?\]")


class Reporter:
    """This class is used to report data to disk."""

    def __init__(self, model: GEBModel, report_folder: Path, clean: bool) -> None:
        """The constructor for the Reporter class.

        Loops over the reporter configuration and creates the necessary files and data structures,
        that are then used while the model is run to add data to.

        For full documentation of the report configuration, see the documentation.

        There are also several pre-defined report configurations that can be activated by adding
        special keys to the report configuration. These are:
        - _discharge_stations: if set to True, discharge at all discharge stations is reported.
        - _meteorological_stations: if set to True, meteorological variables at all meteorological stations are reported.
        - _outflow_points: if set to True, outflow at all outflow points is reported.
        - _water_circle: if set to True, a standard set of variables to monitor the water circle is reported.
        - _water_balance: if set to True, a standard set of variables to monitor the water balance is reported.
        - _energy_balance: if set to True, a standard set of variables to monitor the energy balance is reported.

        Args:
            model: The GEB model instance.
            report_folder: The folder where the reports will be saved.
            clean: If True, the report folder is cleaned at the start of the model run.

        Raises:
            ValueError: If the variable type is not recognized.
        """
        self.model = model
        if "_config" not in self.model.config["report"]:
            self.config: dict[str, int] = {}
        else:
            self.config: dict[str, int] = self.model.config["report"]["_config"].copy()
            del self.model.config["report"]["_config"]

        if self.model.simulate_hydrology:
            self.hydrology = model.hydrology
        self.report_folder = report_folder
        # optionally clean report model at start of run
        if clean and self.report_folder.exists():
            fast_rmtree(self.report_folder)
        self.report_folder.mkdir(parents=True, exist_ok=True)

        self.variables = {}
        self.timesteps = []

        if (
            self.model.mode == "w"
            and "report" in self.model.config
            and self.model.config["report"]
        ):
            self.is_activated = True

            report_config: dict[str, Any] = self.model.config["report"]

            to_delete: list[str] = []
            for module_name, module_values in list(report_config.items()):
                if module_name.startswith("_"):
                    if module_name == "_discharge_stations":
                        if module_values is True:
                            stations = read_geom(
                                self.model.files["geom"][
                                    "discharge/discharge_snapped_locations"
                                ]
                            )

                            station_reporters = {}
                            for station_ID, station_info in stations.iterrows():
                                xy_grid = station_info["snapped_grid_pixel_xy"]
                                station_reporters[
                                    f"discharge_hourly_m3_per_s_{station_ID}"
                                ] = {
                                    "varname": f"grid.var.discharge_m3_s_per_substep",
                                    "type": "grid",
                                    "function": f"sample_xy,{xy_grid[0]},{xy_grid[1]}",
                                    "substeps": 24,
                                }
                            report_config = multi_level_merge(
                                report_config,
                                {"hydrology.routing": station_reporters},
                            )
                    elif module_name == "_retention_basins":
                        if module_values is True:
                            retention_basins = self.model.hydrology.grid.load2d(
                                self.model.files["grid"]["routing/retention_basin_ids"],
                                compress=False,
                            )

                            retention_basin_yx = np.where(retention_basins != -1)
                            retentinion_basin_IDs = retention_basins[retention_basin_yx]

                            retention_basin_reporters: dict[
                                str, dict[str, str | int]
                            ] = {}
                            for basin_ID, yx in zip(
                                retentinion_basin_IDs, zip(*retention_basin_yx)
                            ):
                                retention_basin_reporters[
                                    f"retention_basin_discharge_m3_per_s_{basin_ID}"
                                ] = {
                                    "varname": "grid.var.discharge_m3_s_per_substep",
                                    "type": "grid",
                                    "function": f"sample_xy,{yx[1]},{yx[0]}",
                                    "substeps": 24,
                                }

                            report_config = multi_level_merge(
                                report_config,
                                {"hydrology.routing": retention_basin_reporters},
                            )

                    elif module_name == "_meteorological_stations":
                        if module_values is True:
                            meteorological_station_locations: gpd.GeoDataFrame = (
                                read_geom(
                                    self.model.files["geom"][
                                        "observations/meteorological_station_locations"
                                    ]
                                )
                            )
                            for (
                                station_ID,
                                station_info,
                            ) in meteorological_station_locations.iterrows():
                                station_reporters: dict[str, dict[str, str | int]] = {
                                    f"evapotranspiration_m_per_hour{station_ID}": {
                                        "varname": f".evapotranspiration_m",
                                        "type": "HRU",
                                        "function": f"sample_lonlat,{station_info['geometry'].x},{station_info['geometry'].y}",
                                        "substeps": 24,
                                    },
                                }
                                report_config = multi_level_merge(
                                    report_config,
                                    {"hydrology.landsurface": station_reporters},
                                )
                    elif module_name == "_outflow_points":
                        if module_values is True:
                            routing = self.model.hydrology.routing
                            outflow_rivers = (
                                routing.get_active_and_downstream_outflow_rivers()
                            )
                            all_rivers = routing.rivers

                            outflow_reporters = {}

                            for river_ID, river in outflow_rivers.iterrows():
                                assert isinstance(river_ID, int)
                                xys: list[tuple[int, int]] = (
                                    get_upstream_represented_xys(river_ID, all_rivers)
                                )
                                for i, xy in enumerate(xys):
                                    # if there are multiple branches, we append a suffix to the name
                                    suffix = f"_{i}" if len(xys) > 1 else ""
                                    outflow_reporters[
                                        f"river_outflow_hourly_m3_per_s_{river_ID}{suffix}"
                                    ] = {
                                        "varname": "grid.var.discharge_m3_s_per_substep",
                                        "type": "grid",
                                        "function": f"sample_xy,{xy[0]},{xy[1]}",
                                        "substeps": 24,
                                    }
                            report_config = multi_level_merge(
                                report_config,
                                {"hydrology.routing": outflow_reporters},
                            )
                            report_config = multi_level_merge(
                                report_config,
                                OUTFLOW_PLOT_CONTEXT_REPORT_CONFIG,
                            )
                    elif module_name == "_water_circle":
                        if module_values is True:
                            report_config = multi_level_merge(
                                report_config,
                                WATER_CIRCLE_REPORT_CONFIG,
                            )
                    elif module_name == "_water_balance":
                        if module_values is True:
                            report_config = multi_level_merge(
                                report_config,
                                WATER_BALANCE_REPORT_CONFIG,
                            )
                    elif module_name == "_water_storage":
                        if module_values is True:
                            report_config = multi_level_merge(
                                report_config,
                                WATER_STORAGE_REPORT_CONFIG,
                            )
                    elif module_name == "_energy_balance":
                        if module_values is True:
                            report_config = multi_level_merge(
                                report_config,
                                ENERGY_BALANCE_REPORT_CONFIG,
                            )
                    else:
                        raise ValueError(
                            f"Module {module_name} is not a valid module for reporting."
                        )

                    to_delete.append(module_name)

            for module_name in to_delete:
                del self.model.config["report"][module_name]

            for module_name, configs in self.model.config["report"].items():
                self.variables[module_name] = {}
                for name, config in configs.items():
                    assert isinstance(config, dict), (
                        f"Configuration for {module_name}.{name} must be a dictionary, but is {type(config)}."
                    )
                    self.variables[module_name][name] = self.create_variable(
                        config, module_name, name
                    )
        else:
            self.is_activated = False

    def create_variable(self, config: dict, module_name: str, name: str) -> None:
        """This function creates a variable for the reporter.

        For configurations without an aggregation function, a zarr file is created.
        For configurations with an aggregation function, both the time array and
        the data array are lazily created on the first write so that runtime-only
        information (e.g. substep count) is available when sizing the arrays.

        Args:
            config: The configuration for the variable (mutated in-place to add
                pre-allocated time array and tracking structures).
            module_name: The name of the module to which the variable belongs.
            name: The name of the variable.

        Returns:
            None in all cases; data is tracked via the config dict.

        Raises:
            ValueError: If the variable type is not recognized.
        """
        if config["type"] == "scalar":
            assert "function" not in config or config["function"] is None, (
                "Scalar variables cannot have a function. "
            )
            initialize_tracking_arrays = True
        elif config["type"] in ("grid", "HRU", "agents"):
            initialize_tracking_arrays = config["function"] is not None
        else:
            raise ValueError(
                f"Type {config['type']} not recognized. Must be 'scalar', 'grid', 'agents' or 'HRU'."
            )

        if initialize_tracking_arrays:
            # Time array and data array are created lazily on first write.
            # For grid/HRU this allows sizing based on runtime substep count.
            config["_time_array"] = None
            config["_data_array"] = None
            config["_var_index"] = 0

        return None

    def maybe_report_value(
        self,
        module_name: str,
        name: str,
        module: Any,
        local_variables: dict,
        config: dict,
    ) -> None:
        """This method is used to save and/or export model values.

        Args:
            module_name: Name of the module to which the value belongs.
            name: Name of the value to be exported.
            module: The module to report data from.
            local_variables: A dictionary of local variables from the function
                that calls this one.
            config: Configuration for saving the file. Contains options such a file format, and whether to export the data or save the data in the model.

        Raises:
            ValueError: If the frequency is not recognized.
            KeyError: If the variable is not found in the local variables or module attributes.
            AttributeError: If the attribute is not found in the module.
        """
        if "frequency" in config:
            current_time = self.model.current_time
            current_timestep = self.model.current_timestep
            # here we return None if the value is not to be reported on this timestep
            if config["frequency"] == "initial":
                if current_timestep != 0:
                    return None
            elif config["frequency"] == "final":
                if current_timestep != self.model.n_timesteps - 1:
                    return None
            elif "every" in config["frequency"]:
                every = config["frequency"]["every"]
                if every == "year":
                    month = config["frequency"]["month"]
                    day = config["frequency"]["day"]
                    if current_time.month != month or current_time.day != day:
                        return None
                elif every == "month":
                    day = config["frequency"]["day"]
                    if current_time.day != day:
                        return None
                elif every == "day":
                    pass
                else:
                    raise ValueError(
                        f"Frequency every {config['every']} not recognized (must be 'yearly', or 'monthly')."
                    )
            else:
                raise ValueError(f"Frequency {config['frequency']} not recognized.")

        varname = config["varname"]
        fancy_index: re.Match[str] | None = RE_SQUARE_BRACKETS.search(varname)
        if fancy_index:
            fancy_index: str = fancy_index.group(0)
            varname: str = varname.replace(fancy_index, "")

        # get the variable
        if varname.startswith("."):
            varname = varname[1:]
            try:
                value = local_variables[varname]
            except KeyError:
                raise KeyError(
                    f"Variable {varname} not found in local variables of {module_name}. "
                )
        else:
            try:
                value = attrgetter(varname)(module)
            except AttributeError:
                raise AttributeError(
                    f"Attribute {varname} not found in module {module_name}. "
                )

        if fancy_index:
            value = eval(f"value{fancy_index}")

        if np.isscalar(value):
            assert isinstance(value, np.generic)
            assert not np.isnan(value) and not np.isinf(value)

        self.process_value(module_name, name, value, config)

    def _write_grid_hru_to_zarr(
        self,
        module_name: str,
        name: str,
        value: np.ndarray,
        config: dict,
        type_: str,
    ) -> None:
        """Write a grid or HRU array directly to a zarr store without aggregation.

        Decompresses the compressed array and appends it to the zarr buffer,
        flushing a full chunk to disk when the buffer is full.

        Args:
            module_name: Name of the module to which the value belongs.
            name: Name of the variable being written.
            value: The compressed 1-D (or 2-D with substeps) spatial array.
            config: Reporter configuration dict for this variable (mutated in-place
                to track zarr store, index, and buffer).
            type_: Either ``"grid"`` or ``"HRU"``.
        """
        if type_ == "HRU":
            value = self.hydrology.HRU.decompress(value)
        else:
            value = self.hydrology.grid.decompress(value)

        # in the first timestep, we create the array that will hold the actual data
        if value.ndim == 3:
            substeps: int = value.shape[0]
            # move time axis to the end, so that we can write the data to zarr in chunks along the time axis
            value = np.moveaxis(value, 0, -1)
        else:
            substeps: int = 1
            value = np.expand_dims(value, axis=-1)

        if config["_index"] == 0:  # first time writing data
            if config["type"] == "HRU":
                raster = self.hydrology.HRU
            else:
                raster = self.hydrology.grid

            time = create_time_array(
                start=self.model.simulation_start,
                end=self.model.simulation_end,
                timestep=self.model.timestep_length,
                conf=config,
                substeps=substeps,
            )

            time_chunk_size: int = get_time_chunk_size(
                value.dtype,
                raster.lat.size,
                raster.lon.size,
                target_size_bytes=self.config["chunk_target_size_bytes"],
            )
            time_chunk_size = min(time_chunk_size, time.size)

            # ensure chunk size is multiple of substeps
            time_chunk_size = max(time_chunk_size // substeps, 1) * substeps

            assert isinstance(raster.crs, str)
            prepare_gridded_group(
                name,
                config,
                raster.lon,
                raster.lat,
                raster.crs,
                value,
                time,
                time_chunk_size=time_chunk_size,
                compression_level=self.config["compression_level"],
            )

        root_group = config["_root_group"]

        buffer = config["_chunk_data"]
        chunk_size: int = buffer.shape[-1]

        # Calculate the index in the buffer
        start_index = config["_index"] % chunk_size
        end_index = start_index + substeps

        # Write the values to the buffer
        buffer[..., start_index:end_index] = value

        # If the buffer is full, flush it to disk
        if end_index == chunk_size:
            chunk_index = config["_index"] // chunk_size
            self._flush_chunk_data(root_group, name, buffer, chunk_index, axis=2)

        config["_index"] += substeps

    def _apply_grid_hru_function(
        self,
        module_name: str,
        name: str,
        value: np.ndarray,
        config: dict,
        type_: str,
    ) -> np.ndarray:
        """Apply a spatial aggregation function to a grid or HRU array.

        Args:
            module_name: Name of the module to which the value belongs.
            name: Name of the variable.
            value: The compressed spatial array.
            config: Reporter configuration dict for this variable (mutated in-place
                to record substep count when present).
            type_: Either ``"grid"`` or ``"HRU"``.

        Returns:
            The aggregated value (scalar or 1-D array of substep values).

        Raises:
            ValueError: If the function name is not recognised.
            IndexError: If a sampled coordinate lies outside the model domain.
        """
        # in the first timestep, we create the array that will hold the actual data
        if value.ndim == 2:
            substeps: int = value.shape[0]
            config["substeps"] = substeps

        function, *args = config["function"].split(",")
        if function == "mean":
            return np.mean(value, axis=-1)
        elif function == "nanmean":
            return np.nanmean(value, axis=-1)
        elif function == "sum":
            return np.sum(value, axis=-1)
        elif function == "nansum":
            return np.nansum(value, axis=-1)
        elif function in ("sample_xy", "sample_lonlat"):
            # for sample_xy, args are pixel coordinates, which we
            # first need to convert to their pixel index in x and y
            if function == "sample_lonlat":
                if type_ == "grid":
                    gt: tuple[float, float, float, float, float, float] = (
                        self.model.hydrology.grid.gt
                    )
                elif type_ == "HRU":
                    gt = self.hydrology.HRU.gt
                else:
                    raise ValueError(f"Unknown varname type {config['varname']}")
                px, py = coord_to_pixel((float(args[0]), float(args[1])), gt)
            else:
                # for sample_xy, args are pixel indices
                px: int = int(args[0])
                py: int = int(args[1])

            # both for the grid and HRU, the data is stored efficiently, so that
            # all data is in a 1D array, and we need to map the pixel coordinates
            # to the index in that 1D array
            # therefore, we first get the linear mapping and use that to get the index
            # then we extract the value at that index
            if type_ == "grid":
                linear_mapping: TwoDArrayInt32 = self.hydrology.grid.linear_mapping
            elif type_ == "HRU":
                linear_mapping = self.hydrology.HRU.linear_mapping
            else:
                raise ValueError(f"Unknown varname type {config['varname']}")

            try:
                idx: int = linear_mapping[py, px]
            except IndexError:
                raise IndexError(
                    f"Coordinate ({px}, {py}) is outside the model domain, which has shape {linear_mapping.shape}."
                )
            if idx == -1:
                raise IndexError(f"Coordinate ({px}, {py}) is not a valid cell.")

            # extract the value at that index
            return value[..., idx]
        elif function in (
            "weightedmean",
            "weightednanmean",
            "weightedsum",
            "weightednansum",
        ):
            if type_ == "HRU":
                cell_area = self.hydrology.HRU.var.cell_area
            else:
                cell_area = self.hydrology.grid.var.cell_area
            if function == "weightedmean":
                return np.average(value, weights=cell_area, axis=-1)
            elif function == "weightednanmean":
                return np.nansum(value * cell_area, axis=-1) / np.sum(cell_area)
            elif function == "weightedsum":
                return np.sum(value * cell_area, axis=-1)
            elif function == "weightednansum":
                return np.nansum(value * cell_area, axis=-1)

        raise ValueError(f"Function {function} not recognized")

    def _write_agents_to_zarr(
        self,
        module_name: str,
        name: str,
        value: np.ndarray | DynamicArray,
        config: dict,
    ) -> None:
        """Write an agent array directly to a zarr store without aggregation.

        Initialises the zarr group on the first call, then appends each
        timestep's agent values to the buffer, flushing a full chunk to disk
        when the buffer is full.

        Args:
            module_name: Name of the module to which the value belongs.
            name: Name of the variable being written.
            value: Array of per-agent values for the current timestep.
            config: Reporter configuration dict for this variable (mutated in-place
                to track zarr store, index, and buffer).
        """
        if config["_index"] == 0:
            time = create_time_array(
                start=self.model.simulation_start,
                end=self.model.simulation_end,
                timestep=self.model.timestep_length,
                conf=config,
            )

            prepare_agent_group(
                name,
                config,
                time,
                value,  # ty:ignore[invalid-argument-type]
                self.config["chunk_target_size_bytes"],
                self.config["compression_level"],
            )

        root_group = config["_root_group"]
        value_store_array = root_group[name]
        assert isinstance(value_store_array, zarr.Array)

        assert isinstance(value, (np.ndarray, DynamicArray))
        if value.size < value_store_array.shape[1]:
            self.model.logger.warning(
                "Padding array with NaNs or -1 - temporary solution"
            )
            value = np.pad(
                value.data if isinstance(value, DynamicArray) else value,
                (0, value_store_array.shape[1] - value.size),
                mode="constant",
                constant_values=get_fill_value(value)[0] or False,
            )

        buffer = config["_chunk_data"]
        chunk_size: int = buffer.shape[0]

        # Calculate the index in the buffer
        index = config["_index"] % chunk_size
        buffer[index, ...] = value

        # If the buffer is full, flush it to disk
        if index + 1 == chunk_size:
            chunk_index = config["_index"] // chunk_size
            self._flush_chunk_data(root_group, name, buffer, chunk_index, axis=0)

        config["_index"] += 1

    def _apply_agent_function(
        self,
        module_name: str,
        name: str,
        value: np.ndarray | DynamicArray,
        config: dict,
    ) -> np.ndarray | float:
        """Apply a scalar aggregation function over all agents.

        Args:
            module_name: Name of the module to which the value belongs.
            name: Name of the variable.
            value: Array of per-agent values for the current timestep.
            config: Reporter configuration dict for this variable.

        Returns:
            A single aggregated scalar value.

        Raises:
            ValueError: If the function name is not recognised.
        """
        function, *args = config["function"].split(",")
        if function == "mean":
            return np.mean(value)
        elif function == "nanmean":
            return np.nanmean(value)
        elif function == "sum":
            return np.sum(value)
        elif function == "nansum":
            return np.nansum(value)
        else:
            raise ValueError(f"Function {function} not recognized")

    def process_value(
        self,
        module_name: str,
        name: str,
        value: np.ndarray | np.generic,
        config: dict,
    ) -> None:
        """Exports an array of values to the export folder.

        Args:
            module_name: Name of the module to which the value belongs.
            name: Name of the value to be exported.
            value: The array itself.
            config: Configuration for saving the file. Contains options such a file format, and whether to export the array in this timestep at all.

        Raises:
            ValueError: If the function is not recognized,
                or if the variable type is not recognized.
        """
        type_: str | None = config.get("type", None)
        if type_ is None:
            raise ValueError(f"Type not specified for {config}.")

        if "function" in config and config["function"] is None and "path" not in config:
            zarr_path: Path = self.report_folder / module_name / (name + ".zarr")
            zarr_path.parent.mkdir(parents=True, exist_ok=True)
            config["path"] = str(zarr_path)

            store = zarr.storage.LocalStore(zarr_path, read_only=False)
            config["_store"] = store

            root_group = zarr.open_group(store, mode="w")
            config["_root_group"] = root_group

            config["_index"] = 0

        if type_ in ("grid", "HRU"):
            if not isinstance(value, np.ndarray):
                raise ValueError(
                    f"Value for {module_name}.{name} must be a numpy ndarray, but is {type(value)}."
                )

            grid_size = self.hydrology.grid.compressed_size
            HRU_size = self.hydrology.HRU.compressed_size
            if type_ == "grid" and value.shape[-1] != grid_size:
                error = f"Value for {module_name}.{name} has size {value.shape[-1]}, but grid size is {grid_size}."
                if value.shape[-1] == HRU_size:
                    error += f" HRU is size {HRU_size}. Did you mean to set type to 'HRU' instead of 'grid'?"
                raise ValueError(error)

            if type_ == "HRU" and value.shape[-1] != HRU_size:
                error = f"Value for {module_name}.{name} has size {value.shape[-1]}, but HRU size is {HRU_size}."
                if value.shape[-1] == grid_size:
                    error += f" grid id size {grid_size}. Did you mean to set type to 'grid' instead of 'HRU'?"
                raise ValueError(error)

            if config["function"] is None:
                self._write_grid_hru_to_zarr(module_name, name, value, config, type_)
                return
            else:
                value = self._apply_grid_hru_function(
                    module_name, name, value, config, type_
                )

        elif type_ == "agents":
            assert isinstance(value, (np.ndarray, DynamicArray))
            if config["function"] is None:
                self._write_agents_to_zarr(module_name, name, value, config)
                return
            else:
                value = self._apply_agent_function(module_name, name, value, config)

        elif type_ == "scalar":
            pass  # no processing needed for scalar values

        else:
            raise ValueError(
                f"Type {type_} not recognized. Check your configuration for {module_name}.{name}."
            )

        # Initialize time and data arrays on the first write.
        if config["_time_array"] is None:
            substeps: int | None = config.get("substeps")
            config["_time_array"] = create_time_array(
                start=self.model.simulation_start,
                end=self.model.simulation_end,
                timestep=self.model.timestep_length,
                conf=config,
                substeps=substeps,
            )

        if config["_data_array"] is None:
            n: int = len(config["_time_array"])
            # Use numpy value dtype
            if isinstance(value, np.ndarray):
                dtype = value.dtype
            elif isinstance(value, np.generic):
                dtype = np.dtype(value)
            else:
                raise ValueError(
                    f"Value for {module_name}.{name} has unsupported type {type(value)}. Must be a numpy array or a scalar of type int, float or bool."
                )
            config["_data_array"] = np.empty(n, dtype=dtype)

        if "substeps" in config:
            assert isinstance(value, np.ndarray)
            assert len(value) == config["substeps"], (
                f"Value for {module_name}.{name} has length {len(value)}, but {config['substeps']} substeps are expected."
            )
            n_substeps: int = config["substeps"]
            idx: int = config["_var_index"]
            config["_data_array"][idx : idx + n_substeps] = value
            config["_var_index"] = idx + n_substeps
        else:
            idx: int = config["_var_index"]
            config["_data_array"][idx] = value
            config["_var_index"] = idx + 1

    def _flush_chunk_data(
        self,
        group: zarr.Group,
        name: str,
        buffer: np.ndarray,
        chunk_index: int,
        axis: int,
    ) -> None:
        """Flush a specific chunk for grid/HRU variables.

        We write directly to the zarr blocks to avoid the overhead of zarr's
        indexing and slicing. This is possible because we are writing full chunks.

        Args:
            group: The zarr group where the variable is stored.
            name: The name of the variable.
            buffer: The buffer containing the data to flush.
            chunk_index: The index of the chunk to flush.
            axis: The time axis along which to flush the data.
        """
        zarr_array = group[name]
        assert isinstance(zarr_array, zarr.Array)
        # Create a slice tuple with Ellipsis for all dimensions except the target axis
        selection: list[slice | int] = [slice(None)] * len(zarr_array.shape)
        selection[axis] = chunk_index
        zarr_array.blocks[tuple(selection)] = buffer

    def finalize(self) -> None:
        """At the end of the model run, all previously collected data is reported to disk.

        Raises:
            ValueError: If the variable type is not recognized.
        """
        # If no data has been collected, we return
        if self.model.config["report"] is None:
            self.model.logger.info("No report configuration found. No data to report.")
            return

        # Limit thread pool to avoid excessive threading that could prevent disk flushes
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []

            # Flush any remaining buffers
            for module_name, configs in self.model.config["report"].items():
                for name, config in configs.items():
                    if "function" in config and config["function"] is None:
                        if config["type"] == "agents":
                            chunk_time_size: int = config["_chunk_data"].shape[0]
                        elif config["type"] in ("grid", "HRU"):
                            chunk_time_size: int = config["_chunk_data"].shape[-1]
                        else:
                            raise ValueError(
                                f"Unknown type {config['type']} for variable {module_name}.{name}"
                            )

                        buffer_end: int = config["_index"] % chunk_time_size
                        if buffer_end == 0:
                            continue  # nothing to flush
                        if config["type"] == "agents":
                            axis = 0
                            buffer = config["_chunk_data"][:buffer_end, ...]
                        elif config["type"] in ("grid", "HRU"):
                            axis = 2
                            buffer = config["_chunk_data"][..., :buffer_end]
                        else:
                            raise ValueError(
                                f"Unknown type {config['type']} for variable {module_name}.{name}"
                            )

                        chunk_index: int = config["_index"] // chunk_time_size

                        futures.append(
                            executor.submit(
                                self._flush_chunk_data,
                                group=config["_root_group"],
                                name=name,
                                buffer=buffer,
                                chunk_index=chunk_index,
                                axis=axis,
                            )
                        )

            # Export all scalar and aggregated variables to parquet files
            for module_name, module_configs in self.model.config["report"].items():
                for name, config in module_configs.items():
                    if "_time_array" not in config or config["_data_array"] is None:
                        continue
                    # Convert Unix timestamps (seconds) to datetime
                    time_values: pd.DatetimeIndex = pd.to_datetime(
                        config["_time_array"], unit="s"
                    )
                    df = pd.DataFrame(
                        {name: config["_data_array"]},
                        index=pd.Index(time_values, name="time"),
                    )

                    folder = self.report_folder / module_name
                    folder.mkdir(parents=True, exist_ok=True)

                    futures.append(
                        executor.submit(write_table, df, folder / (name + ".parquet"))
                    )

            # Wait for all futures to complete before exiting context manager
            for future in as_completed(futures):
                # re-raise any exception from the worker thread
                future.result()

    def report(
        self,
        module: Module,
        local_variables: dict[str, Any],
        module_name: str,
        variables_to_report: dict[str, Any],
    ) -> None:
        """This method is in every step function to report data to disk.

        Args:
            module: The module to report data from.
            local_variables: A dictionary of local variables from the function
                that calls this one.
            module_name: The name of the module.
            variables_to_report: A dictionary of variable names to report for this module, with their configurations.
        """
        if not self.is_activated:
            return None
        if variables_to_report is not None:
            for name, config in variables_to_report.items():
                self.maybe_report_value(
                    module_name=module_name,
                    name=name,
                    module=module,
                    local_variables=local_variables,
                    config=config,
                )

    @property
    def is_activated(self) -> bool:
        """Returns whether the reporter is activated.

        Returns:
            True if the reporter is activated, False otherwise.
        """
        return self._is_activated

    @is_activated.setter
    def is_activated(self, value: bool) -> None:
        self._is_activated = value
