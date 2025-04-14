# -*- coding: utf-8 -*-
import datetime
from operator import attrgetter
from typing import Any, Union

import numpy as np
import pandas as pd
import zarr
from honeybees.library.raster import coord_to_pixel


def create_time_array(
    start: datetime.datetime,
    end: datetime.datetime,
    timestep: datetime.timedelta,
    conf: dict,
) -> list:
    """Create a time array based on the start and end time, the timestep, and the frequency.

    Args:
        start: The start time.
        end: The end time.
        timestep: The timestep length.
        conf: The configuration for the frequency.

    Returns:
        time: The time array.
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
                if (
                    frequency["month"] == current_time.month
                    and frequency["day"] == current_time.day
                ):
                    time.append(current_time)
            elif every == "month":
                if frequency["day"] == current_time.day:
                    time.append(current_time)
            elif every == "day":
                time.append(current_time)
            current_time += timestep
        return time
    elif frequency == "initial":
        return [start]
    elif frequency == "final":
        return [end]
    else:
        raise ValueError(f"Frequency {frequency} not recognized.")


class Reporter:
    """This class is used to report data to disk.

    Args:
        model: The GEB model.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.hydrology = model.hydrology
        self.report_folder = self.model.output_folder / "reporter"
        self.report_folder.mkdir(parents=True, exist_ok=True)

        self.variables = {}
        self.timesteps = []

        if self.model.mode == "w":
            if "report" in self.model.config:
                for module_name, configs in self.model.config["report"].items():
                    self.variables[module_name] = {}
                    for name, config in configs.items():
                        self.variables[module_name][name] = self.create_variable(
                            config, module_name, name
                        )

    def create_variable(self, config: dict, module_name: str, name: str) -> None:
        """This function creates a variable for the reporter. For
        configurations without a aggregation function, a zarr file is created. For
        configurations with an aggregation function, the data
        is stored in memory and exported on the final timestep.

        Args:
            config: The configuration for the variable.
            module_name: The name of the module to which the variable belongs.
            name: The name of the variable.
        """
        if config["type"] in ("grid", "HRU"):
            if config["function"] is not None:
                return []
            else:
                if config["type"] == "HRU":
                    raster = self.hydrology.HRU
                else:
                    raster = self.hydrology.grid
                zarr_path = self.report_folder / module_name / (name + ".zarr")
                zarr_path.parent.mkdir(parents=True, exist_ok=True)
                config["path"] = str(zarr_path)

                zarr_store = zarr.storage.LocalStore(zarr_path, read_only=False)
                zarr_group = zarr.open_group(zarr_store, mode="w")

                time = create_time_array(
                    start=self.model.current_time,
                    end=self.model.end_time,
                    timestep=self.model.timestep_length,
                    conf=config,
                )

                time = (
                    np.array(time, dtype="datetime64[ns]")
                    .astype("datetime64[s]")
                    .astype(np.int64)
                )
                time_group = zarr_group.create_array(
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

                y_group = zarr_group.create_array(
                    "y",
                    shape=raster.lat.shape,
                    dtype=raster.lat.dtype,
                    dimension_names=["y"],
                )
                y_group[:] = raster.lat
                y_group.attrs.update(
                    {
                        "standard_name": "latitude",
                        "units": "degrees_north",
                    }
                )

                x_group = zarr_group.create_array(
                    "x",
                    shape=raster.lon.shape,
                    dtype=raster.lon.dtype,
                    dimension_names=["x"],
                )
                x_group[:] = raster.lon
                x_group.attrs.update(
                    {
                        "standard_name": "longitude",
                        "units": "degrees_east",
                    }
                )

                zarr_data = zarr_group.create_array(
                    name,
                    shape=(
                        time.size,
                        raster.lat.size,
                        raster.lon.size,
                    ),
                    chunks=(
                        1,
                        raster.lat.size,
                        raster.lon.size,
                    ),
                    dtype=np.float32,
                    compressor=zarr.codecs.BloscCodec(
                        cname="zlib",
                        clevel=9,
                        shuffle=zarr.codecs.BloscShuffle.shuffle,
                    ),
                    fill_value=np.nan,
                    dimension_names=["time", "y", "x"],
                )

                zarr_data.attrs.update(
                    {
                        "grid_mapping": "crs",
                        "coordinates": "time y x",
                        "units": "unknown",
                        "long_name": name,
                        "_CRS": raster.crs,
                    }
                )
                return zarr_store

        elif config["type"] == "agents":
            if config["function"] is not None:
                return []
            else:
                filepath = zarr_path = (
                    self.report_folder / module_name / (name + ".zarr")
                )
                filepath.parent.mkdir(parents=True, exist_ok=True)
                store = zarr.storage.LocalStore(filepath, read_only=False)
                ds = zarr.open_group(store, mode="w")

                time = create_time_array(
                    start=self.model.current_time,
                    end=self.model.end_time,
                    timestep=self.model.timestep_length,
                    conf=config,
                )
                time = np.array(
                    [np.datetime64(t, "s").astype(np.int64) for t in time],
                    dtype=np.int64,
                )

                time_group = ds.create_array(
                    "time",
                    shape=time.shape,
                    dtype=time.dtype,
                )
                time_group[:] = time

                time_group.attrs.update(
                    {
                        "standard_name": "time",
                        "units": "seconds since 1970-01-01T00:00:00",
                        "calendar": "gregorian",
                        "_ARRAY_DIMENSIONS": ["time"],
                    }
                )

                config["_file"] = ds
                config["_time_index"] = time
        else:
            raise ValueError(
                f"Type {config['type']} not recognized. Must be 'grid', 'agents' or 'HRU'."
            )

    def maybe_report_value(
        self,
        module_name: str,
        name: Union[str, tuple[str, Any]],
        value: Any,
        conf: dict,
    ) -> None:
        """This method is used to save and/or export model values.

        Args:
            name: Name of the value to be exported.
            value: The array itself.
            conf: Configuration for saving the file. Contains options such a file format, and whether to export the data or save the data in the model.
        """

        # here we return None if the value is not to be reported on this timestep
        if "frequency" in conf:
            if conf["frequency"] == "initial":
                if self.model.current_timestep != 0:
                    return None
            elif conf["frequency"] == "final":
                if self.model.current_timestep != self.model.n_timesteps - 1:
                    return None
            elif "every" in conf["frequency"]:
                every = conf["frequency"]["every"]
                if every == "year":
                    month = conf["frequency"]["month"]
                    day = conf["frequency"]["day"]
                    if (
                        self.model.current_time.month != month
                        or self.model.current_time.day != day
                    ):
                        return None
                elif every == "month":
                    day = conf["frequency"]["day"]
                    if self.model.current_time.day != day:
                        return None
                elif every == "day":
                    pass
                else:
                    raise ValueError(
                        f"Frequency every {conf['every']} not recognized (must be 'yearly', or 'monthly')."
                    )
            else:
                raise ValueError(f"Frequency {conf['frequency']} not recognized.")

        # if the value is not None, we check whether the value is valid
        if isinstance(value, list):
            value = [v.item() for v in value]
            for v in value:
                self.check_value(v)
        elif np.isscalar(value):
            value = value.item()
            self.check_value(value)

        self.process_value(module_name, name, value, conf)

    def process_value(
        self, module_name: str, name: str, value: np.ndarray, conf: dict
    ) -> None:
        """Exports an array of values to the export folder.

        Args:
            module_name: Name of the module to which the value belongs.
            name: Name of the value to be exported.
            value: The array itself.
            conf: Configuration for saving the file. Contains options such a file format, and whether to export the array in this timestep at all.
        """
        if conf["type"] in ("grid", "HRU"):
            if conf["function"] is None:
                if conf["type"] == "HRU":
                    value = self.hydrology.HRU.decompress(value)
                else:
                    value = self.hydrology.grid.decompress(value)

                zarr_group = zarr.open_group(self.variables[module_name][name])
                if (
                    np.isin(self.model.current_time_unix_s, zarr_group["time"][:])
                    and value is not None
                ):
                    time_index = np.where(
                        zarr_group["time"][:] == self.model.current_time_unix_s
                    )[0].item()
                    if "substeps" in conf:
                        time_index_start = np.where(time_index)[0][0]
                        time_index_end = time_index_start + conf["substeps"]
                        zarr_group[name][time_index_start:time_index_end, ...] = value
                    else:
                        zarr_group[name][time_index, ...] = value
                return None
            else:
                function, *args = conf["function"].split(",")
                if function == "mean":
                    value = np.mean(value)
                elif function == "nanmean":
                    value = np.nanmean(value)
                elif function == "sum":
                    value = np.sum(value)
                elif function == "nansum":
                    value = np.nansum(value)
                elif function == "sample":
                    decompressed_array = self.decompress(conf["varname"], value)
                    value = decompressed_array[int(args[0]), int(args[1])]
                elif function == "sample_coord":
                    if conf["varname"].startswith("hydrology.grid"):
                        gt = self.model.hydrology.grid.gt
                    elif conf["varname"].startswith("hydrology.HRU"):
                        gt = self.hydrology.HRU.gt
                    else:
                        raise ValueError
                    px, py = coord_to_pixel((float(args[0]), float(args[1])), gt)
                    decompressed_array = self.decompress(conf["varname"], value)
                    try:
                        value = decompressed_array[py, px]
                    except IndexError:
                        raise IndexError(
                            f"The coordinate ({args[0]},{args[1]}) is outside the model domain."
                        )
                elif function in ("weightedmean", "weightednanmean"):
                    if conf["type"] == "HRU":
                        cell_area = self.hydrology.HRU.var.cell_area
                    else:
                        cell_area = self.hydrology.grid.var.cell_area
                    if function == "weightedmean":
                        value = np.average(value, weights=cell_area)
                    elif function == "weightednanmean":
                        value = np.nansum(value * cell_area) / np.sum(cell_area)

                else:
                    raise ValueError(f"Function {function} not recognized")

        elif conf["type"] == "agents":
            if conf["function"] is None:
                ds = conf["_file"]
                if name not in ds:
                    # zarr file has not been created yet
                    if isinstance(value, (float, int)):
                        shape = (ds["time"].size,)
                        chunks = (1,)
                        compressor = None
                        dtype = type(value)
                        array_dimensions = ["time"]
                    else:
                        shape = (ds["time"].size, value.size)
                        chunks = (1, value.size)
                        compressor = zarr.codecs.BloscCodec(
                            cname="zlib",
                            clevel=9,
                            shuffle=zarr.codecs.BloscShuffle.shuffle,
                        )
                        dtype = value.dtype
                        array_dimensions = ["time", "agents"]
                    if dtype in (float, np.float32, np.float64):
                        fill_value = np.nan
                    elif dtype in (int, np.int32, np.int64):
                        fill_value = -1
                    else:
                        raise ValueError(
                            f"Value {dtype} of type {type(dtype)} not recognized."
                        )
                    ds.create_array(
                        name,
                        shape=shape,
                        chunks=chunks,
                        dtype=dtype,
                        compressor=compressor,
                        fill_value=fill_value,
                    )
                    ds[name].attrs["_ARRAY_DIMENSIONS"] = array_dimensions
                index = np.argwhere(
                    conf["_time_index"]
                    == np.datetime64(self.model.current_time, "s").astype(
                        conf["_time_index"].dtype
                    )
                ).item()
                if value.size < ds[name][index].size:
                    print("Padding array with NaNs or -1 - temporary solution")
                    value = np.pad(
                        value,
                        (0, ds[name][index].size - value.size),
                        mode="constant",
                        constant_values=np.nan
                        if value.dtype in (float, np.float32, np.float64)
                        else -1,
                    )
                ds[name][index] = value
                return None
            else:
                function, *args = conf["function"].split(",")
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

        if np.isnan(value):
            value = None

        if module_name not in self.variables:
            self.variables[module_name] = {}

        if name not in self.variables[module_name]:
            self.variables[module_name][name] = []

        self.variables[module_name][name].append((self.model.current_time, value))
        return None

    def finalize(self) -> None:
        """At the end of the model run, all previously collected data is reported to disk."""
        for module_name, variables in self.variables.items():
            for name, values in variables.items():
                if (
                    self.model.config["report"][module_name][name]["function"]
                    is not None
                ):
                    df = pd.DataFrame.from_records(
                        values, columns=["date", name], index="date"
                    )

                    folder = self.report_folder / module_name
                    folder.mkdir(parents=True, exist_ok=True)

                    df.to_csv(folder / (name + ".csv"))

    def report(self, module, local_variables, module_name) -> None:
        """This method is in every step function to report data to disk.

        Args:
            module: The module to report data from.
            variables: A dictionary of local variables from the function
                that calls this one.
            module_name: The name of the module.
        """
        report = self.model.config["report"].get(module_name, None)
        if report is not None:
            for name, config in report.items():
                varname = config["varname"]
                if varname.startswith("."):
                    varname = varname[1:]
                    value = local_variables[varname]
                else:
                    value = attrgetter(varname)(module)

                self.maybe_report_value(
                    module_name=module_name,
                    name=name,
                    value=value,
                    conf=config,
                )
