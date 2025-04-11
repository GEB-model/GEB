# -*- coding: utf-8 -*-
"""This module is used to report data to the disk. After initialization, the :meth:`reporter.Report.step` method is called every timestep, which in turn calls the equivalent methods in honeybees's reporter (to report data from the agents) and the CWatM reporter, to report data from CWatM. The variables to report can be configured in `model.yml` (see :doc:`configuration`). All data is saved in a subfolder (see :doc:`configuration`)."""

import os
import re
import warnings
from collections.abc import Iterable
from operator import attrgetter
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import zarr


def create_time_array(start, end, timestep, conf):
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
    """This class is used to report CWatM data to disk. On initialization the export folder is created if it does not yet exist. Then all variables to report are on read from the configuration folder, and the datastructures to save the data are created.

    Args:
        model: The GEB model.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.hydrology = model.hydrology
        self.folder = self.model.report_folder / "reporter"
        self.folder.mkdir(parents=True, exist_ok=True)

        self.variables = {}
        self.timesteps = []

        if self.model.mode == "w":
            if "report" in self.model.config:
                for module_name, configs in self.model.config["report"].items():
                    self.variables[module_name] = {}
                    for name, config in configs.items():
                        self.variables[module_name][name] = self.create_variable(
                            config, name
                        )

    def create_variable(self, config, name) -> None:
        if config["type"] in ("grid", "HRU"):
            if config["format"] != "zarr":
                return []
            else:
                if config["type"] == "HRU":
                    raster = self.hydrology.HRU
                else:
                    raster = self.hydrology.grid
                assert "single_file" in config and config["single_file"] is True, (
                    "Only single_file=True is supported for zarr format."
                )
                zarr_path = Path(self.folder, name + ".zarr")
                config["path"] = str(zarr_path)
                if zarr_path.exists():
                    zarr_path.unlink()
                if "time_ranges" not in config:
                    if "substeps" in config:
                        time = pd.date_range(
                            start=self.model.current_time,
                            periods=(self.model.n_timesteps + 1) * config["substeps"],
                            freq=self.model.timestep_length / config["substeps"],
                            inclusive="left",
                        )
                    else:
                        time = pd.date_range(
                            start=self.model.current_time,
                            periods=self.model.n_timesteps,
                            freq=self.model.timestep_length,
                        )
                else:
                    time = []
                    for time_range in config["time_ranges"]:
                        start = time_range["start"]
                        end = time_range["end"]
                        if "substeps" in config:
                            time.extend(
                                pd.date_range(
                                    start=start,
                                    end=end + self.model.timestep_length,
                                    freq=self.model.timestep_length
                                    / config["substeps"],
                                    inclusive="left",
                                )
                            )
                        else:
                            time.extend(
                                pd.date_range(
                                    start=start,
                                    end=end,
                                    freq=self.model.timestep_length,
                                )
                            )
                    # exlude time ranges that are not in the simulation period
                    time = [
                        t
                        for t in time
                        if t >= self.model.current_time
                        and t
                        <= self.model.current_time
                        + (self.model.n_timesteps + 1) * self.model.timestep_length
                    ]
                    # remove duplicates and sort
                    time = list(dict.fromkeys(time))
                    time.sort()
                    if not time:
                        print(
                            f"WARNING: None of the time ranges for {name} are in the simulation period."
                        )

                zarr_store = zarr.storage.LocalStore(zarr_path, read_only=False)
                zarr_group = zarr.open_group(zarr_store, mode="w")

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
            if config["format"] == "zarr":
                filepath = Path(self.folder) / (name + ".zarr.zip")
                store = zarr.storage.ZipStore(filepath, mode="w")
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

    def decompress(self, attr: str, array: np.ndarray) -> np.ndarray:
        """This function decompresses an array for given attribute.

        Args:
            attr: Attribute which was used to get array.
            array: The array itself.

        Returns:
            decompressed_array: The decompressed array.
        """
        return attrgetter(".".join(attr.split(".")[:-1]).replace(".var", ""))(
            self.model
        ).decompress(array)

    def get_array(self, attr: str, decompress: bool = False) -> np.ndarray:
        """This function retrieves a NumPy array from the model based the name of the variable. Optionally decompresses the array.

        Args:
            attr: Name of the variable to retrieve. Name can contain "." to specify variables are a "deeper" level.
            decompress: Boolean value whether to decompress the array. If True, the class to which the top variable name belongs to must have an equivalent function called `decompress`.

        Returns:
            array: The requested array.

        Example:
            Read discharge from `data.grid`. Because :code:`decompress=True`, `data.grid` must have a `decompress` method.
            ::

                >>> get_array(data.grid.discharge, decompress=True)
        """
        slicer = re.search(r"\[([0-9]+)\]$", attr)
        if slicer:
            try:
                array = attrgetter(attr[: slicer.span(0)[0]])(self.model)
            except AttributeError:
                return None
            else:
                array = array[int(slicer.group(1))]
        else:
            try:
                array = attrgetter(attr)(self.model)
            except AttributeError:
                return None
        if decompress:
            decompressed_array = self.decompress(attr, array)
            return array, decompressed_array

        assert isinstance(array, np.ndarray)

        return array

    def report_value(
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
        if isinstance(value, list):
            value = [v.item() for v in value]
            for v in value:
                self.check_value(v)
        elif np.isscalar(value):
            value = value.item()
            self.check_value(value)

        if "save" in conf:
            if conf["save"] not in ("save", "export"):
                raise ValueError(
                    f"Save type for {name} in config file must be 'save', 'save+export' or 'export')."
                )

            warnings.warn(
                "The `save` option is deprecated and will be removed in future versions. If you use 'save: export' the option can simply be removed (new default). If you use 'save: save', please replace with 'single_file: true'",
                DeprecationWarning,
            )
            if conf["save"] == "save":
                conf["single_file"] = True
            del conf["save"]

        if (
            "single_file" in conf
            and conf["single_file"] is True
            and conf["format"] != "zarr"  # for zarr, we always save per timestep
        ):
            try:
                if isinstance(name, tuple):
                    name, ID = name
                    if name not in self.variables:
                        self.variables[name] = {}
                    if ID not in self.variables[name]:
                        self.variables[name][ID] = []
                    self.variables[name][ID].append(value)
                else:
                    if name not in self.variables:
                        self.variables[name] = []
                    self.variables[name].append(value)
            except KeyError:
                raise KeyError(
                    f"Variable {name} not initialized. This likely means that an agent is reporting for a group that was not is not the reporter"
                )

        else:
            if "frequency" in conf and conf["frequency"] is not None:
                if conf["frequency"] == "initial":
                    if self.model.current_timestep == 0:
                        self.export_value(module_name, name, value, conf)
                elif conf["frequency"] == "final":
                    if (
                        self.model.current_timestep == self.model.n_timesteps - 1
                    ):  # timestep is 0-indexed, so n_timesteps - 1 is the last timestep
                        self.export_value(module_name, name, value, conf)
                elif "every" in conf["frequency"]:
                    every = conf["frequency"]["every"]
                    if every == "year":
                        month = conf["frequency"]["month"]
                        day = conf["frequency"]["day"]
                        if (
                            self.model.current_time.month == month
                            and self.model.current_time.day == day
                        ):
                            self.export_value(module_name, name, value, conf)
                    elif every == "month":
                        day = conf["frequency"]["day"]
                        if self.model.current_time.day == day:
                            self.export_value(module_name, name, value, conf)
                    elif every == "day":
                        self.export_value(module_name, name, value, conf)
                    else:
                        raise ValueError(
                            f"Frequency every {conf['every']} not recognized (must be 'yearly', or 'monthly')."
                        )
                else:
                    raise ValueError(f"Frequency {conf['frequency']} not recognized.")
            else:
                self.export_value(module_name, name, value, conf)

    def export_value(
        self, module_name: str, name: str, value: np.ndarray, conf: dict
    ) -> None:
        """Exports an array of values to the export folder.

        Args:
            name: Name of the value to be exported.
            value: The array itself.
            conf: Configuration for saving the file. Contains options such a file format, and whether to export the array in this timestep at all.
        """
        if conf["type"] in ("grid", "HRU"):
            if conf["type"] == "HRU":
                value = self.hydrology.HRU.decompress(value)
            else:
                value = self.hydrology.grid.decompress(value)
            if "format" not in conf:
                raise ValueError(
                    f"Export format must be specified for {name} in config file (npy/npz/csv/xlsx/zarr)."
                )
            if conf["format"] == "zarr":
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
            else:
                folder = os.path.join(self.folder, name)
                os.makedirs(folder, exist_ok=True)
                fn = f"{self.timesteps[-1].isoformat().replace('-', '').replace(':', '')}"
                if conf["format"] == "npy":
                    fn += ".npy"
                    fp = os.path.join(folder, fn)
                    np.save(fp, value)
                elif conf["format"] == "npz":
                    fn += ".npz"
                    fp = os.path.join(folder, fn)
                    np.savez_compressed(fp, data=value)
                elif conf["format"] == "csv":
                    fn += ".csv"
                    fp = os.path.join(folder, fn)
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    if isinstance(value, (float, int)):
                        value = [value]
                    if len(value) > 100_000:
                        self.model.logger.info(
                            f"Exporting {len(value)} items to csv. This might take a long time and take a lot of space. Consider using NumPy (compressed) binary format (npy/npz)."
                        )
                    with open(fp, "w") as f:
                        f.write("\n".join([str(v) for v in value]))
                else:
                    raise ValueError(f"{conf['format']} not recognized")
        elif conf["type"] == "agents":
            if conf["format"] == "zarr":
                ds = conf["_file"]
                if name not in ds:
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
                    ds.create_dataset(
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
            else:
                folder = os.path.join(self.folder, name)
                os.makedirs(folder, exist_ok=True)
                fn = f"{self.timesteps[-1].isoformat().replace('-', '').replace(':', '')}"
                if conf["format"] == "npy":
                    fn += ".npy"
                    fp = os.path.join(folder, fn)
                    np.save(fp, value)
                elif conf["format"] == "npz":
                    fn += ".npz"
                    fp = os.path.join(folder, fn)
                    np.savez_compressed(fp, data=value)
                elif conf["format"] == "csv":
                    fn += ".csv"
                    fp = os.path.join(folder, fn)
                    if len(value) > 100_000:
                        self.model.logger.info(
                            f"Exporting {len(value)} items to csv. This might take a long time and take a lot of space. Consider using NumPy (compressed) binary format (npy/npz)."
                        )
                    with open(fp, "w") as f:
                        f.write("\n".join([str(v) for v in value]))
                else:
                    raise ValueError(f"{conf['format']} not recognized")
        else:
            raise ValueError(
                f"Type {conf['type']} not recognized. Must be 'grid', 'agents' or 'HRU'."
            )

    # def step(self) -> None:
    #     """This method is called after every timestep, to collect data for reporting from the model."""
    #     for name, conf in self.model.config["report_hydrology"].items():
    #         array = self.get_array(conf["varname"])
    #         if array is None:
    #             print(
    #                 f"variable {name} not found at timestep {self.model.current_time}"
    #             )
    #             self.report_value(name, None, conf)
    #         else:
    #             if conf["varname"].endswith("crop"):
    #                 crop_map = self.get_array("HRU.crop_map")
    #                 array = array[crop_map == conf["crop"]]
    #             if array.size == 0:
    #                 value = None
    #             else:
    #                 if conf["function"] is None:
    #                     value = self.decompress(conf["varname"], array)
    #                 else:
    #                     function, *args = conf["function"].split(",")
    #                     if function == "mean":
    #                         value = np.mean(array)
    #                         if np.isnan(value):
    #                             value = None
    #                     elif function == "nanmean":
    #                         value = np.nanmean(array)
    #                         if np.isnan(value):
    #                             value = None
    #                     elif function == "sum":
    #                         value = np.sum(array)
    #                         if np.isnan(value):
    #                             value = None
    #                     elif function == "nansum":
    #                         value = np.nansum(array)
    #                         if np.isnan(value):
    #                             value = None
    #                     elif function == "sample":
    #                         decompressed_array = self.decompress(conf["varname"], array)
    #                         value = decompressed_array[int(args[0]), int(args[1])]
    #                         assert not np.isnan(value)
    #                     elif function == "sample_coord":
    #                         if conf["varname"].startswith("hydrology.grid"):
    #                             gt = self.model.hydrology.grid.gt
    #                         elif conf["varname"].startswith("hydrology.HRU"):
    #                             gt = self.hydrology.HRU.gt
    #                         else:
    #                             raise ValueError
    #                         px, py = coord_to_pixel(
    #                             (float(args[0]), float(args[1])), gt
    #                         )
    #                         decompressed_array = self.decompress(conf["varname"], array)
    #                         try:
    #                             value = decompressed_array[py, px]
    #                         except IndexError as e:
    #                             index_error = f"{e}. Most likely the coordinate ({args[0]},{args[1]}) is outside the model domain."
    #                             raise IndexError(index_error)
    #                     else:
    #                         raise ValueError(f"Function {function} not recognized")
    #             self.report_value(name, value, conf)

    def finalize(self) -> None:
        """At the end of the model run, all previously collected data is reported to disk."""
        for module_name, variables in self.variables.items():
            for name, values in variables.items():
                if self.model.config["report"][module_name][name]["format"] == "zarr":
                    continue
                else:
                    if isinstance(values[0], Iterable):
                        df = pd.DataFrame.from_dict(
                            {k: v for k, v in zip(self.timesteps, values)}
                        )
                    else:
                        df = pd.DataFrame(values, index=self.timesteps, columns=[name])
                    df.index.name = "time"
                    export_format = self.model.config["report_hydrology"][name][
                        "format"
                    ]
                    if export_format == "csv":
                        df.to_csv(os.path.join(self.folder, name + "." + export_format))
                    elif export_format == "xlsx":
                        df.to_excel(
                            os.path.join(self.folder, name + "." + export_format)
                        )
                    else:
                        raise ValueError(f"save_to format {export_format} unknown")

    def report(self, module, variables, module_name):
        report = self.model.config["report"].get(module_name, None)

        if report is not None:
            for name, config in report.items():
                varname = config["varname"]
                if varname.startswith("."):
                    varname = varname[1:]
                    value = variables[varname]
                else:
                    value = attrgetter(varname)(module)
                if "decompressor" in config:
                    decompressor = attrgetter("hydrology.grid.decompress")(self)
                    value = decompressor(value)

                self.report_value(
                    module_name=module.name,
                    name=name,
                    value=value,
                    conf=config,
                )
