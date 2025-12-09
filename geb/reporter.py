"""This module contains the Reporter class, which is used to report data to disk."""

from __future__ import annotations

import datetime
import re
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import zarr.storage
from dateutil.relativedelta import relativedelta
from zarr.codecs import ZstdCodec

from geb.module import Module
from geb.store import DynamicArray
from geb.types import ArrayFloat32, ArrayFloat64, ArrayInt64, TwoDArrayInt32
from geb.workflows.io import fast_rmtree
from geb.workflows.methods import multi_level_merge
from geb.workflows.raster import coord_to_pixel

if TYPE_CHECKING:
    from geb.model import GEBModel

WATER_CIRCLE_REPORT_CONFIG: dict[str, str | dict[str, str | dict[str, str]]] = {
    "hydrology": {
        "_water_circle_storage": {
            "varname": ".current_storage",
            "type": "scalar",
        },
        "_water_circle_routing_loss": {
            "varname": ".routing_loss_m3",
            "type": "scalar",
        },
    },
    "hydrology.landsurface": {
        "_water_circle_rain": {
            "varname": ".rain_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_water_circle_snow": {
            "varname": ".snow_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_water_circle_transpiration": {
            "varname": ".transpiration_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_water_circle_bare_soil_evaporation": {
            "varname": ".bare_soil_evaporation_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_water_circle_open_water_evaporation": {
            "varname": ".open_water_evaporation_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_water_circle_interception_evaporation": {
            "varname": ".interception_evaporation_m",
            "type": "HRU",
            "function": "weightedsum",
        },
        "_water_circle_sublimation_or_deposition": {
            "varname": ".sublimation_or_deposition_m",
            "type": "HRU",
            "function": "weightedsum",
        },
    },
    "hydrology.routing": {
        "_water_circle_river_evaporation": {
            "varname": ".total_evaporation_in_rivers_m3",
            "type": "scalar",
        },
        "_water_circle_waterbody_evaporation": {
            "varname": ".total_waterbody_evaporation_m3",
            "type": "scalar",
        },
        "_water_circle_river_outflow": {
            "varname": ".total_outflow_at_pits_m3",
            "type": "scalar",
        },
    },
    "hydrology.water_demand": {
        "_water_circle_domestic_water_loss": {
            "varname": ".domestic_water_loss_m3",
            "type": "scalar",
        },
        "_water_circle_industry_water_loss": {
            "varname": ".industry_water_loss_m3",
            "type": "scalar",
        },
        "_water_circle_livestock_water_loss": {
            "varname": ".livestock_water_loss_m3",
            "type": "scalar",
        },
    },
}


def get_fill_value(
    data: np.ndarray[Any] | int | float | bool | np.floating | np.integer,
) -> int | float | None:
    """Get the fill value for a zarr array based on the data type.

    Args:
        data: The data array.

    Returns:
        fill_value: The fill value for the zarr array.

    Raises:
        ValueError: If the data type is not recognized.
    """
    if isinstance(data, float):
        fill_value = np.nan
    elif isinstance(data, int):
        fill_value = -1
    elif isinstance(data, bool):
        fill_value = None
    elif np.issubdtype(data.dtype, np.floating):
        fill_value: int | float = np.nan
    elif np.issubdtype(data.dtype, np.integer):
        fill_value = -1
    elif data.dtype == bool:
        fill_value = None
    else:
        raise ValueError(f"Value dtype {data.dtype} not recognized.")
    return fill_value


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


def prepare_gridded_group(
    name: str,
    config: dict,
    lon: ArrayFloat32 | ArrayFloat64,
    lat: ArrayFloat32 | ArrayFloat64,
    crs: str,
    example_value: np.ndarray[Any],
    time: ArrayInt64,
    chunk_size: int,
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
        chunk_size: The size of the time chunk.
        compression_level: The compression level for the zarr array.
    """
    root_group = config["_root_group"]

    # Create the y coordinate array
    y_group = root_group.create_array(
        "y",
        shape=lat.shape,
        dtype=lat.dtype,
        dimension_names=["y"],
    )
    y_group[:] = lat
    y_group.attrs.update(
        {
            "standard_name": "latitude",
            "units": "degrees_north",
        }
    )

    # Create the x coordinate array
    x_group = root_group.create_array(
        "x",
        shape=lon.shape,
        dtype=lon.dtype,
        dimension_names=["x"],
    )
    x_group[:] = lon
    x_group.attrs.update(
        {
            "standard_name": "longitude",
            "units": "degrees_east",
        }
    )

    # Create the time coordinate array
    time_group = root_group.create_array(
        "time",
        shape=time.shape,
        dtype=time.dtype,
        dimension_names=["time"],
    )
    time_group[:] = time
    time_group.attrs.update(
        {
            "standard_name": "time",
            "units": "seconds since 1970-01-01T00:00:00",
            "calendar": "gregorian",
        }
    )

    # Create the variable array
    variable_data: zarr.Array[Any] = root_group.create_array(
        name,
        shape=(
            time_group.size,
            lat.size,
            lon.size,
        ),
        chunks=(
            chunk_size,
            lat.size,
            lon.size,
        ),
        dtype=example_value.dtype,
        compressors=(
            ZstdCodec(
                level=compression_level,
            ),
        ),
        fill_value=get_fill_value(example_value),
        dimension_names=["time", "y", "x"],
    )

    variable_data.attrs.update(
        {
            "grid_mapping": "crs",
            "coordinates": "time y x",
            "units": "unknown",
            "long_name": name,
            "_CRS": {"wkt": crs},
        }
    )
    # Pre-allocate the writing buffer for the chunks
    config["_chunk_data"] = np.full(
        (chunk_size, lat.size, lon.size),
        np.nan,
        dtype=np.float32,
    )


def prepare_agent_group(
    name: str,
    config: dict,
    time: ArrayInt64,
    example_value: np.ndarray[Any] | int | float | bool | np.floating | np.integer,
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
    """
    root_group = config["_root_group"]

    # Create the time coordinate array
    time_group = root_group.create_array(
        "time",
        shape=time.shape,
        dtype=time.dtype,
        dimension_names=["time"],
    )
    time_group[:] = time

    time_group.attrs.update(
        {
            "standard_name": "time",
            "units": "seconds since 1970-01-01T00:00:00",
            "calendar": "gregorian",
        }
    )

    # Determine chunk size and shape based on example value
    if isinstance(example_value, (float, int)):
        shape = (time_group.size,)
        chunk_size = get_time_chunk_size(
            np.dtype(type(example_value)),
            1,
            target_size_bytes=chunk_target_size_bytes,
        )
        if isinstance(example_value, float):
            dtype_ = np.float32
        else:
            dtype_ = np.int32
        chunk_size = min(chunk_size, time_group.size)
        chunks = (chunk_size,)
        compressors = None
        array_dimensions = ["time"]
    else:
        shape = (time_group.size, example_value.size)
        dtype_ = example_value.dtype
        chunk_size = get_time_chunk_size(
            np.dtype(example_value.dtype),
            example_value.size,
            target_size_bytes=chunk_target_size_bytes,
        )
        chunk_size = min(chunk_size, time_group.size)
        chunks = (chunk_size, example_value.size)
        compressors = (
            ZstdCodec(
                level=compression_level,
            ),
        )
        array_dimensions = ["time", "agents"]

    fill_value = get_fill_value(example_value)
    root_group.create_array(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype_,
        compressors=compressors,
        fill_value=fill_value,
        dimension_names=array_dimensions,
    )
    # Pre-allocate the writing buffer for the chunks
    if isinstance(example_value, (float, int)):
        config["_chunk_data"] = np.full((chunk_size,), fill_value)
    else:
        config["_chunk_data"] = np.full((chunk_size, example_value.size), fill_value)


class Reporter:
    """This class is used to report data to disk."""

    def __init__(self, model: GEBModel, clean: bool) -> None:
        """The constructor for the Reporter class.

        Loops over the reporter configuration and creates the necessary files and data structures,
        that are then used while the model is run to add data to.

        For full documentation of the report configuration, see the documentation.

        There are also several pre-defined report configurations that can be activated by adding
        special keys to the report configuration. These are:
        - _discharge_stations: if set to True, discharge at all discharge stations is reported.
        - _outflow_points: if set to True, outflow at all outflow points is reported.
        - _water_circle: if set to True, a standard set of variables to monitor the water circle is reported.

        Args:
            model: The GEB model instance.
            clean: If True, the report folder is cleaned at the start of the model run.

        Raises:
            ValueError: If the variable type is not recognized.
        """
        self.model = model
        self.config: dict[str, int] = self.model.config["report"]["_config"].copy()
        del self.model.config["report"]["_config"]

        if self.model.simulate_hydrology:
            self.hydrology = model.hydrology
        self.report_folder = self.model.output_folder / "report" / self.model.run_name
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
                    if module_name == "_discharge_stations" and module_values is True:
                        stations = gpd.read_parquet(
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
                    elif module_name == "_outflow_points" and module_values is True:
                        outflow_rivers = self.model.hydrology.routing.outflow_rivers

                        outflow_reporters = {}
                        for river_ID, river in outflow_rivers.iterrows():
                            xy = river["hydrography_xy"][-1]  # last point is outflow
                            outflow_reporters[
                                f"river_outflow_hourly_m3_per_s_{river_ID}"
                            ] = {
                                "varname": f"grid.var.discharge_m3_s_per_substep",
                                "type": "grid",
                                "function": f"sample_xy,{xy[0]},{xy[1]}",
                                "substeps": 24,
                            }
                        report_config = multi_level_merge(
                            report_config,
                            {"hydrology.routing": outflow_reporters},
                        )
                    elif module_name == "_water_circle":
                        if module_values is True:
                            report_config = multi_level_merge(
                                report_config,
                                WATER_CIRCLE_REPORT_CONFIG,
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

    def create_variable(self, config: dict, module_name: str, name: str) -> list | None:
        """This function creates a variable for the reporter.

        For
        configurations without a aggregation function, a zarr file is created. For
        configurations with an aggregation function, the data
        is stored in memory and exported on the final timestep.

        Args:
            config: The configuration for the variable.
            module_name: The name of the module to which the variable belongs.
            name: The name of the variable.

        Returns:
            A list of values if the variable is scalar, None otherwise.

        Raises:
            ValueError: If the variable type is not recognized.
        """
        if config["type"] == "scalar":
            assert "function" not in config or config["function"] is None, (
                "Scalar variables cannot have a function. "
            )
            return []
        elif config["type"] in ("grid", "HRU"):
            if config["function"] is not None:
                return []
            else:
                return

        elif config["type"] == "agents":
            if config["function"] is not None:
                return []
            else:
                return

        else:
            raise ValueError(
                f"Type {config['type']} not recognized. Must be 'scalar', 'grid', 'agents' or 'HRU'."
            )

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
        # here we return None if the value is not to be reported on this timestep
        if "frequency" in config:
            if config["frequency"] == "initial":
                if self.model.current_timestep != 0:
                    return None
            elif config["frequency"] == "final":
                if self.model.current_timestep != self.model.n_timesteps - 1:
                    return None
            elif "every" in config["frequency"]:
                every = config["frequency"]["every"]
                if every == "year":
                    month = config["frequency"]["month"]
                    day = config["frequency"]["day"]
                    if (
                        self.model.current_time.month != month
                        or self.model.current_time.day != day
                    ):
                        return None
                elif every == "month":
                    day = config["frequency"]["day"]
                    if self.model.current_time.day != day:
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
        fancy_index = re.search(r"\[.*?\]", varname)
        if fancy_index:
            fancy_index = fancy_index.group(0)
            varname = varname.replace(fancy_index, "")

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

        # if the value is not None, we check whether the value is valid
        if isinstance(value, list):
            value = np.array([v.item() for v in value])
            for v in value:
                assert not np.isnan(value) and not np.isinf(v)
        elif np.isscalar(value):
            value = value.item()
            assert not np.isnan(value) and not np.isinf(value)

        self.process_value(module_name, name, value, config)

    def process_value(
        self,
        module_name: str,
        name: str,
        value: np.ndarray | int | float | np.floating | np.integer,
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
            IndexError: If the coordinate is in sample_lon_lat is outside the model domain.
        """
        type_: str | None = config.get("type", None)
        if type_ is None:
            raise ValueError(f"Type not specified for {config}.")

        if config["function"] is None and "path" not in config:
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

            # in case of no aggregation function, we write the data directly to zarr
            if config["function"] is None:
                if type_ == "HRU":
                    value: np.ndarray = self.hydrology.HRU.decompress(value)
                else:
                    value: np.ndarray = self.hydrology.grid.decompress(value)

                # in the first timestep, we create the array that will hold the actual data
                if value.ndim == 3:
                    substeps: int = value.shape[0]
                else:
                    substeps: int = 1

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

                    chunk_size: int = get_time_chunk_size(
                        value.dtype,
                        raster.lat.size,
                        raster.lon.size,
                        target_size_bytes=self.config["chunk_target_size_bytes"],
                    )
                    chunk_size: int = min(chunk_size, time.size)

                    # ensure chunk size is multiple of substeps
                    chunk_size: int = max(chunk_size // substeps, 1) * substeps

                    assert isinstance(raster.crs, str)
                    prepare_gridded_group(
                        name,
                        config,
                        raster.lon,
                        raster.lat,
                        raster.crs,
                        value,
                        time,
                        chunk_size,
                        self.config["compression_level"],
                    )

                root_group = config["_root_group"]

                buffer = config["_chunk_data"]
                chunk_size: int = buffer.shape[0]

                # Calculate the index in the buffer
                start_index = config["_index"] % chunk_size
                end_index = start_index + substeps
                buffer[start_index:end_index, ...] = value

                # If the buffer is full, flush it to disk
                if end_index == chunk_size:
                    chunk_index = config["_index"] // chunk_size
                    self._flush_chunk_data(root_group, name, buffer, chunk_index)

                config["_index"] += substeps
                return
            else:
                function, *args = config["function"].split(",")
                if function == "mean":
                    value = np.mean(value)
                elif function == "nanmean":
                    value = np.nanmean(value)
                elif function == "sum":
                    value = np.sum(value)
                elif function == "nansum":
                    value = np.nansum(value)
                elif function in ("sample_xy", "sample_lonlat"):
                    # for sample_xy, args are pixel coordinates, which we
                    # first need to convert to their pixel index in x and y
                    if function == "sample_lonlat":
                        if type_ == "grid":
                            gt: tuple[float, float, float, float, float, float] = (
                                self.model.hydrology.grid.gt
                            )
                        elif type_ == "HRU":
                            gt: tuple[float, float, float, float, float, float] = (
                                self.hydrology.HRU.gt
                            )
                        else:
                            raise ValueError(
                                f"Unknown varname type {config['varname']}"
                            )
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
                        linear_mapping: TwoDArrayInt32 = (
                            self.hydrology.grid.linear_mapping
                        )
                    elif type_ == "HRU":
                        linear_mapping: TwoDArrayInt32 = (
                            self.hydrology.HRU.linear_mapping
                        )
                    else:
                        raise ValueError(f"Unknown varname type {config['varname']}")

                    try:
                        idx = linear_mapping[py, px]
                    except IndexError:
                        raise IndexError(
                            f"Coordinate ({px}, {py}) is outside the model domain, which has shape {linear_mapping.shape}."
                        )
                    if idx == -1:
                        raise IndexError(
                            f"Coordinate ({px}, {py}) is not a valid cell."
                        )

                    # extract the value at that index
                    value = value[..., idx]
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
                        value = np.average(value, weights=cell_area)
                    elif function == "weightednanmean":
                        value = np.nansum(value * cell_area) / np.sum(cell_area)
                    elif function == "weightedsum":
                        value = np.sum(value * cell_area)
                    elif function == "weightednansum":
                        value = np.nansum(value * cell_area)

                else:
                    raise ValueError(f"Function {function} not recognized")

        elif type_ == "agents":
            if config["function"] is None:
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
                        value,
                        self.config["chunk_target_size_bytes"],
                        self.config["compression_level"],
                    )

                root_group = config["_root_group"]

                assert isinstance(value, (np.ndarray, DynamicArray))
                if value.size < root_group[name].shape[0]:
                    print("Padding array with NaNs or -1 - temporary solution")
                    value = np.pad(
                        value.data if isinstance(value, DynamicArray) else value,
                        (0, root_group[name].shape[0] - value.size),
                        mode="constant",
                        constant_values=get_fill_value(value) or False,
                    )

                buffer = config["_chunk_data"]
                chunk_size: int = buffer.shape[0]

                # Calculate the index in the buffer
                index = config["_index"] % chunk_size
                buffer[index, ...] = value

                # If the buffer is full, flush it to disk
                if index + 1 == chunk_size:
                    chunk_index = config["_index"] // chunk_size
                    self._flush_chunk_data(root_group, name, buffer, chunk_index)

                config["_index"] += 1

                return
            else:
                function, *args = config["function"].split(",")
                if function == "mean":
                    value = np.mean(value)
                elif function == "nanmean":
                    value = np.nanmean(value)
                elif function == "sum":
                    value = np.sum(value)
                elif function == "nansum":
                    value = np.nansum(value)
                else:
                    raise ValueError(f"Function {function} not recognized")

        if module_name not in self.variables:
            self.variables[module_name] = {}

        if name not in self.variables[module_name]:
            self.variables[module_name][name] = []

        if "substeps" in config:
            assert isinstance(value, np.ndarray)
            assert len(value) == config["substeps"], (
                f"Value for {module_name}.{name} has length {len(value)}, but {config['substeps']} substeps are expected."
            )
            self.variables[module_name][name].extend(
                [
                    (
                        self.model.current_time
                        + i * self.model.timestep_length / config["substeps"],
                        v,
                    )
                    for i, v in enumerate(value)
                ]
            )
        else:
            self.variables[module_name][name].append((self.model.current_time, value))

    def _flush_chunk_data(
        self, group: zarr.Group, name: str, buffer: np.ndarray, chunk_index: int
    ) -> None:
        """Flush a specific chunk for grid/HRU variables.

        We write directly to the zarr blocks to avoid the overhead of zarr's
        indexing and slicing. This is possible because we are writing full chunks.

        Args:
            group: The zarr group where the variable is stored.
            name: The name of the variable.
            buffer: The buffer containing the data to flush.
            chunk_index: The index of the chunk to flush.
        """
        zarr_array = group[name]
        assert isinstance(zarr_array, zarr.Array)
        zarr_array.blocks[chunk_index, ...] = buffer

    def finalize(self) -> None:
        """At the end of the model run, all previously collected data is reported to disk."""
        # If no data has been collected, we return
        if self.model.config["report"] is None:
            self.model.logger.info("No report configuration found. No data to report.")
            return

        # Flush any remaining buffers
        for module_name, configs in self.model.config["report"].items():
            for name, config in configs.items():
                if config["function"] is None:
                    chunk_time_size: int = config["_chunk_data"].shape[0]
                    buffer_end: int = config["_index"] % chunk_time_size
                    if buffer_end == 0:
                        continue  # nothing to flush
                    chunk_index: int = config["_index"] // chunk_time_size
                    self._flush_chunk_data(
                        group=config["_root_group"],
                        name=name,
                        buffer=config["_chunk_data"][:buffer_end, ...],
                        chunk_index=chunk_index,
                    )

        # Export all scalar and aggregated variables to CSV
        for module_name, variables in self.variables.items():
            for name, values in variables.items():
                if self.model.config["report"][module_name][name][
                    "type"
                ] == "scalar" or (
                    self.model.config["report"][module_name][name]["function"]
                    is not None
                ):
                    # if the variable is a scalar or has an aggregation function, we report
                    df = pd.DataFrame.from_records(
                        values, columns=["time", name], index="time"
                    )

                    folder = self.report_folder / module_name
                    folder.mkdir(parents=True, exist_ok=True)

                    df.to_csv(folder / (name + ".csv"))

    def report(
        self, module: Module, local_variables: dict[str, Any], module_name: str
    ) -> None:
        """This method is in every step function to report data to disk.

        Args:
            module: The module to report data from.
            local_variables: A dictionary of local variables from the function
                that calls this one.
            module_name: The name of the module.
        """
        if not self.is_activated:
            return None
        report = self.model.config["report"].get(module_name, None)
        if report is not None:
            for name, config in report.items():
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
