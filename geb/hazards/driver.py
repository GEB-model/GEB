"""Module for managing short-lived hazard simulations such as floods."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from geb.hazards.floods import Floods
from geb.module import Module
from geb.types import ThreeDArrayFloat32, TwoDArrayFloat32

if TYPE_CHECKING:
    from geb.model import GEBModel


class HazardDriver(Module):
    """Class that manages the simulation of short-lived hazards such as floods.

    Currently it only supports floods but can be extended to include other hazards such as landslides in the future.
    """

    def __init__(self, model: GEBModel) -> None:
        """Initializes the HazardDriver class.

        If flood simulation is enabled in the configuration, it initializes the flood simulation by determining
        the longest flood event duration and setting up the SFINCS model accordingly.
        """
        super().__init__(model)

        self.model: GEBModel = model
        # extract the longest flood event in days
        flood_events: list[dict[str, Any]] = self.model.config["hazards"]["floods"][
            "events"
        ]
        if flood_events == [] or flood_events is None:
            longest_flood_event_in_days: int = 0
        else:
            flood_event_lengths: list[timedelta] = [
                event["end_time"] - event["start_time"] for event in flood_events
            ]
            longest_flood_event_in_days = max(flood_event_lengths).days

        self.initialize(
            longest_flood_event_in_days=longest_flood_event_in_days + 10
        )  # Here we add 10 days as a buffer to ensure we capture the full event

        self.next_detection_time: datetime | None = (
            None  # Variable to keep track of when a flood has happened
        )
        self.discharge_log: list = []

    def spinup(self) -> None:
        """Spinup method for the hazard driver.

        Currently does nothing as hazards do not require spinup.
        """
        pass

    @property
    def name(self) -> str:
        """Returns the name of the module."""
        return "hazard_driver"

    def initialize(self, longest_flood_event_in_days: int) -> None:
        """Initializes the hazard driver.

        Used to set up the SFINCS model for flood simulation and in the future perhaps other hazards.

        Args:
            longest_flood_event_in_days: The longest flood event in days. This is needed because
                the SFINCS model is initiated at the end of the flood event, but requires
                the conditions at the start of the flood event. Therefore, the conditions during the
                last n_timesteps is saved in memory to be used at the start of the flood event.

        """
        self.floods: Floods = Floods(
            self.model, longest_flood_event_in_days=longest_flood_event_in_days
        )

    def step(self) -> None:
        """Steps the hazard driver.

        If flood simulation is enabled in the configuration, it runs the SFINCS model for each flood event
        that ends during the current timestep.
        """
        if self.model.config["hazards"]["floods"]["simulate"]:
            if self.model.config["hazards"]["floods"]["events"] is None:
                return

            if self.model.config["hazards"]["floods"]["detect_floods_from_discharge"]:
                print("Detecting floods from discharge...")
                if self.model.in_spinup:
                    return
                else:
                    if (
                        self.next_detection_time
                        and self.model.current_time < self.next_detection_time
                    ):
                        print(
                            "Flood has recently happened, no detection"
                        )  # Within 10 days of the first flood, no second flood can happen
                    else:
                        discharge_grid: ThreeDArrayFloat32 = (
                            self.model.hydrology.grid.decompress(
                                np.vstack(list(self.floods.var.discharge_per_timestep))
                            )
                        )

                        discharge_grid_current_timestep: TwoDArrayFloat32 = (
                            discharge_grid[-1]
                        )  # Extract discharge from last timestep

                        # Convert discharge grid to an xarray DataArray
                        discharge_grid: xr.DataArray = xr.DataArray(
                            data=discharge_grid_current_timestep,
                            coords={
                                "y": self.model.hydrology.grid.lat,
                                "x": self.model.hydrology.grid.lon,
                            },
                            dims=["y", "x"],
                            name="forcing",
                        )

                        discharge_grid: xr.DataArray = discharge_grid.rio.write_crs(
                            self.model.crs
                        )

                        # Get location of the threshold from config file
                        threshold_location: tuple[float, float] = self.model.config[
                            "hazards"
                        ]["floods"]["threshold_location"]
                        x, y = threshold_location

                        # Extract discharge from the previously extracted location
                        discharge_location: xr.DataArray = discharge_grid.sel(
                            x=x, y=y, method="nearest"
                        ).compute()

                        self.discharge_log.append(
                            {
                                "time": self.model.current_time,
                                "discharge": float(discharge_location.values),
                            }
                        )

                        # Load in discharge_threshold after which there is a flood from the config file
                        threshold: int = self.model.config["hazards"]["floods"][
                            "discharge_threshold"
                        ]

                        # Check if discharge > threshold
                        if discharge_location > threshold:
                            print(
                                f"Flood detected at {self.model.current_time}, discharge = {discharge_location:.2f} m3/s"
                            )

                            start_time = self.model.current_time - timedelta(
                                days=5
                            )  # Here we assume a flood duration of 10 days
                            end_time = self.model.current_time + timedelta(days=5)

                            new_event_mem = {
                                "start_time": start_time,
                                "end_time": end_time,
                            }

                            new_event_yaml = {
                                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                            }

                            hazards_cfg = self.model.config.setdefault("hazards", {})
                            floods_cfg = hazards_cfg.setdefault("floods", {})
                            events_mem = floods_cfg.setdefault("events", [])

                            event_exists_mem = any(
                                e["start_time"] == new_event_mem["start_time"]
                                and e["end_time"] == new_event_mem["end_time"]
                                for e in events_mem
                            )

                            if not event_exists_mem:
                                events_mem.append(new_event_mem)
                                print("Flood event added to in-memory config.")
                            else:
                                print("Flood event already in in-memory config.")

                            config_path = Path.cwd() / "model.yml"

                            with open(config_path, "r") as f:
                                config_yaml = yaml.safe_load(f) or {}

                            hazards_yaml = config_yaml.setdefault("hazards", {})
                            floods_yaml = hazards_yaml.setdefault("floods", {})
                            events_yaml = floods_yaml.setdefault("events", [])

                            event_exists_yaml = any(
                                e.get("start_time") == new_event_yaml["start_time"]
                                and e.get("end_time") == new_event_yaml["end_time"]
                                for e in events_yaml
                            )

                            if not event_exists_yaml:
                                events_yaml.append(new_event_yaml)

                                tmp_path = config_path.with_suffix(".tmp")
                                with open(tmp_path, "w") as f:
                                    yaml.safe_dump(config_yaml, f, sort_keys=False)

                                tmp_path.replace(config_path)
                            else:
                                print("Flood event already in model.yml.")

                            self.next_detection_time = (
                                self.model.current_time + timedelta(days=10)
                            )

                    end_time: datetime = datetime.combine(
                        self.model.config["general"]["end_time"], datetime.min.time()
                    )

                    if self.model.current_time == end_time:
                        df_all: pd.DataFrame = pd.DataFrame(self.discharge_log)
                        df_all.to_csv(
                            Path(self.model.output_folder) / "discharge_timeseries.csv",
                            index=False,
                        )

            for event in self.model.config["hazards"]["floods"]["events"]:
                assert isinstance(event["start_time"], datetime), (
                    f"Start time {event['start_time']} must be a datetime object."
                )
                assert isinstance(event["end_time"], datetime), (
                    f"End time {event['end_time']} must be a datetime object."
                )
                assert event["end_time"] >= event["start_time"], (
                    f"End time {event['end_time']} must be greater than or equal to start time {event['start_time']}."
                )

                routing_substeps: int = self.floods.var.discharge_per_timestep[0].shape[
                    0
                ]

                # since we are at the end of the timestep, we need to check if the current time plus the timestep length is greater than or equal to the start time of the event
                timestep_end_time: datetime = (
                    self.model.current_time
                    + self.model.timestep_length
                    - self.model.timestep_length / routing_substeps
                )
                if (
                    timestep_end_time >= event["end_time"]
                    and event["end_time"] + self.model.timestep_length
                    > timestep_end_time
                ) or (
                    event["end_time"] > self.model.simulation_end
                    and event["start_time"] < timestep_end_time
                    and self.model.current_timestep == self.model.n_timesteps - 1
                ):
                    event: dict[str, datetime] = copy.deepcopy(event)

                    # the actual end time is the end of the day of the simulation. Therefore,
                    # its the simulation end time plus one timestep length
                    if (
                        event["end_time"]
                        > self.model.simulation_end + self.model.timestep_length
                    ):
                        print(
                            f"Warning: Flood event {event} ends after the model end time {self.model.simulation_end}. Simulating only part of flood event."
                        )
                        event["end_time"] = (
                            self.model.simulation_end + self.model.timestep_length
                        )
                        assert event["end_time"] > event["start_time"]

                    print("Running floods for event:", event)
                    self.floods.run(event)
