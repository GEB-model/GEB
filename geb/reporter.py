# -*- coding: utf-8 -*-
"""This module is used to report data to the disk. After initialization, the :meth:`reporter.Report.step` method is called every timestep, which in turn calls the equivalent methods in honeybees's reporter (to report data from the agents) and the CWatM reporter, to report data from CWatM. The variables to report can be configured in `model.yml` (see :doc:`configuration`). All data is saved in a subfolder (see :doc:`configuration`)."""

import os
import re
from collections.abc import Iterable
from operator import attrgetter
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from honeybees.library.raster import coord_to_pixel
from honeybees.reporter import Reporter as ABMReporter


class hydrology_reporter(ABMReporter):
    """This class is used to report CWatM data to disk. On initialization the export folder is created if it does not yet exist. Then all variables to report are on read from the configuration folder, and the datastructures to save the data are created.

    Args:
        model: The GEB model.
    """

    def __init__(self, model, folder: str) -> None:
        self.model = model
        self.hydrology = model.hydrology

        self.export_folder = folder

        self.variables = {}
        self.timesteps = []

        if self.model.mode == "w":
            if (
                "report_hydrology" in self.model.config
                and self.model.config["report_hydrology"]
            ):
                for name, config in self.model.config["report_hydrology"].items():
                    if config["format"] == "zarr":
                        assert (
                            "single_file" in config and config["single_file"] is True
                        ), "Only single_file=True is supported for zarr format."
                        zarr_path = Path(self.export_folder, name + ".zarr")
                        config["path"] = str(zarr_path)
                        if zarr_path.exists():
                            zarr_path.unlink()
                        if "time_ranges" not in config:
                            if "substeps" in config:
                                time = pd.date_range(
                                    start=self.model.current_time,
                                    periods=(self.model.n_timesteps + 1)
                                    * config["substeps"],
                                    freq=self.model.timestep_length
                                    / config["substeps"],
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
                                + (self.model.n_timesteps + 1)
                                * self.model.timestep_length
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
                            shape=self.hydrology.grid.lat.shape,
                            dtype=self.hydrology.grid.lat.dtype,
                            dimension_names=["y"],
                        )
                        y_group[:] = self.hydrology.grid.lat
                        y_group.attrs.update(
                            {
                                "standard_name": "latitude",
                                "units": "degrees_north",
                            }
                        )

                        x_group = zarr_group.create_array(
                            "x",
                            shape=self.hydrology.grid.lon.shape,
                            dtype=self.hydrology.grid.lon.dtype,
                            dimension_names=["x"],
                        )
                        x_group[:] = self.hydrology.grid.lon
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
                                self.hydrology.grid.lat.size,
                                self.hydrology.grid.lon.size,
                            ),
                            chunks=(
                                1,
                                self.hydrology.grid.lat.size,
                                self.hydrology.grid.lon.size,
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
                                "_CRS": self.hydrology.grid.crs,
                            }
                        )
                        self.variables[name] = zarr_store

                    else:
                        self.variables[name] = []

            self.step()  # report on inital state

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

    def export_value(self, name: str, value: np.ndarray, conf: dict) -> None:
        """Exports an array of values to the export folder.

        Args:
            name: Name of the value to be exported.
            value: The array itself.
            conf: Configuration for saving the file. Contains options such a file format, and whether to export the array in this timestep at all.
        """
        if "format" not in conf:
            raise ValueError(
                f"Export format must be specified for {name} in config file (npy/npz/csv/xlsx/zarr)."
            )
        if conf["format"] == "zarr":
            zarr_group = zarr.open_group(self.variables[name])
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
            folder = os.path.join(self.export_folder, name)
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

    def step(self) -> None:
        """This method is called after every timestep, to collect data for reporting from the model."""
        self.timesteps.append(self.model.current_time)
        if (
            "report_hydrology" in self.model.config
            and self.model.config["report_hydrology"]
        ):
            for name, conf in self.model.config["report_hydrology"].items():
                array = self.get_array(conf["varname"])
                if array is None:
                    print(
                        f"variable {name} not found at timestep {self.model.current_time}"
                    )
                    self.report_value(name, None, conf)
                else:
                    if conf["varname"].endswith("crop"):
                        crop_map = self.get_array("HRU.crop_map")
                        array = array[crop_map == conf["crop"]]
                    if array.size == 0:
                        value = None
                    else:
                        if conf["function"] is None:
                            value = self.decompress(conf["varname"], array)
                        else:
                            function, *args = conf["function"].split(",")
                            if function == "mean":
                                value = np.mean(array)
                                if np.isnan(value):
                                    value = None
                            elif function == "nanmean":
                                value = np.nanmean(array)
                                if np.isnan(value):
                                    value = None
                            elif function == "sum":
                                value = np.sum(array)
                                if np.isnan(value):
                                    value = None
                            elif function == "nansum":
                                value = np.nansum(array)
                                if np.isnan(value):
                                    value = None
                            elif function == "sample":
                                decompressed_array = self.decompress(
                                    conf["varname"], array
                                )
                                value = decompressed_array[int(args[0]), int(args[1])]
                                assert not np.isnan(value)
                            elif function == "sample_coord":
                                if conf["varname"].startswith("data.grid"):
                                    gt = self.model.hydrology.grid.gt
                                elif conf["varname"].startswith("data.HRU"):
                                    gt = self.hydrology.HRU.gt
                                else:
                                    raise ValueError
                                px, py = coord_to_pixel(
                                    (float(args[0]), float(args[1])), gt
                                )
                                decompressed_array = self.decompress(
                                    conf["varname"], array
                                )
                                try:
                                    value = decompressed_array[py, px]
                                except IndexError as e:
                                    index_error = f"{e}. Most likely the coordinate ({args[0]},{args[1]}) is outside the model domain."
                                    raise IndexError(index_error)
                            else:
                                raise ValueError(f"Function {function} not recognized")
                    self.report_value(name, value, conf)

    def finalize(self) -> None:
        """At the end of the model run, all previously collected data is reported to disk."""
        for name, values in self.variables.items():
            if self.model.config["report_hydrology"][name]["format"] == "zarr":
                continue
            else:
                if isinstance(values[0], Iterable):
                    df = pd.DataFrame.from_dict(
                        {k: v for k, v in zip(self.timesteps, values)}
                    )
                else:
                    df = pd.DataFrame(values, index=self.timesteps, columns=[name])
                df.index.name = "time"
                export_format = self.model.config["report_hydrology"][name]["format"]
                if export_format == "csv":
                    df.to_csv(
                        os.path.join(self.export_folder, name + "." + export_format)
                    )
                elif export_format == "xlsx":
                    df.to_excel(
                        os.path.join(self.export_folder, name + "." + export_format)
                    )
                else:
                    raise ValueError(f"save_to format {export_format} unknown")


class Reporter:
    """This is the main reporter class for the GEB model. On initialization the ABMReporter and hydrology_reporter classes are initalized.

    Args:
        model: The GEB model.
    """

    def __init__(self, model):
        self.model = model
        self.abm_reporter = ABMReporter(model, folder=self.model.report_folder)

        if self.model.simulate_hydrology:
            self.hydrology_reporter = hydrology_reporter(
                model, folder=self.model.report_folder
            )

    @property
    def variables(self):
        return {**self.abm_reporter.variables, **self.hydrology_reporter.variables}

    @property
    def timesteps(self):
        return self.abm_reporter.timesteps

    def step(self) -> None:
        """This function is called at the end of every timestep. This function only forwards the step function to the reporter for the ABM model and CWatM."""
        self.abm_reporter.step()
        if self.model.simulate_hydrology:
            self.hydrology_reporter.step()

    def finalize(self):
        """At the end of the model run, all previously collected data is reported to disk. This function only forwards the report function to the reporter for the ABM model and CWatM."""
        self.abm_reporter.finalize()
        if self.model.simulate_hydrology:
            self.hydrology_reporter.finalize()
        print("Reported data")
