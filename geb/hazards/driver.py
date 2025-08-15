import copy
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from honeybees.library.raster import coord_to_pixel


class HazardDriver:
    """Class that manages the simulation of short-lived hazards such as floods.

    Currently it only supports floods but can be extended to include other hazards such as landslides in the future.
    """

    def __init__(self):
        if self.config["hazards"]["floods"]["simulate"]:
            # extract the longest flood event in days
            flood_events = self.config["hazards"]["floods"]["events"]
            flood_event_lengths = [
                event["end_time"] - event["start_time"] for event in flood_events
            ]
            longest_flood_event = max(flood_event_lengths).days
            self.initialize(longest_flood_event)

            self.next_detection_time = (
                None  # Variable to keep track of when a flood has happened
            )

    def initialize(self, longest_flood_event):
        from geb.hazards.floods.sfincs import SFINCS

        self.sfincs = SFINCS(self, n_timesteps=longest_flood_event)

    def step(self):
        if self.config["hazards"]["floods"]["simulate"]:
            if self.simulate_hydrology:
                self.sfincs.save_discharge_timestep()
                self.sfincs.save_soil_moisture()
                self.sfincs.save_max_soil_moisture()
                self.sfincs.save_soil_storage_capacity()
                self.sfincs.save_ksat()
                if not self.spinup: # Dont detect floods within spinup 
                    if self.config["hazards"]["floods"]["detect_from_discharge"]:
                        if (
                            self.next_detection_time
                            and self.current_time < self.next_detection_time
                        ):
                            print(
                                "Flood has recently happened, no detection"
                            )  # Within 5 days of the first flood, no second flood can happen
                        else:
                            discharge = self.sfincs.save_discharge_step()
                            if discharge is not None:
                                discharge_current_step = discharge[
                                    -1
                                ]  # last value from discharge deque corresponds to current timestep

                                # discharge is on hydrology.grid so we can use the spatial information to extract the discharge from a certain location
                                gt = self.model.hydrology.grid.gt
                                decompressed_array = self.hydrology.grid.decompress(
                                    discharge_current_step
                                )
                                x, y = self.config["hazards"]["floods"][
                                    "threshold_location"
                                ]
                                px, py = coord_to_pixel((float(x), float(y)), gt)
                                try:
                                    discharge_location = decompressed_array[py, px]
                                except IndexError:
                                    raise IndexError(
                                        f"The coordinate ({x},{y}) is outside the model domain."
                                    )

                                # Load in user-define discharge_threshold after which there is a flood
                                threshold = self.config["hazards"]["floods"][
                                    "discharge_threshold"
                                ]
                                # check if discharge > discharge
                                if discharge_location > threshold:
                                    print(
                                        f"Flood detected at {self.current_time}, discharge = {discharge_location:.2f} "
                                    )
                                    start_time = self.current_time - timedelta(days=5)
                                    end_time = self.current_time + timedelta(days=5)

                                    # Block of code to save the start and end time of the detected event
                                    def represent_datetime(dumper, data):
                                        return dumper.represent_scalar(
                                            "tag:yaml.org,2002:timestamp",
                                            data.strftime("%Y-%m-%d %H:%M:%S"),
                                        )

                                    yaml.add_representer(datetime, represent_datetime)
                                    new_event = {
                                        "start_time": start_time,
                                        "end_time": end_time,
                                    }
                                    # Check if event already exists (exact match on start and end)
                                    existing_events = self.config["hazards"]["floods"].get(
                                        "events", []
                                    )
                                    event_exists = any(
                                        e["start_time"] == new_event["start_time"]
                                        and e["end_time"] == new_event["end_time"]
                                        for e in existing_events
                                    )

                                    if not event_exists:
                                        self.config["hazards"]["floods"]["events"].append(
                                            new_event
                                        )
                                        config_path = Path.cwd() / "model.yml"
                                        with open(config_path, "w") as f:
                                            yaml.safe_dump(self.config, f, sort_keys=False)
                                        print("Flood event saved to config.")
                                    else:
                                        print(
                                            "Flood event already in config, skipping save."
                                        )

                                    self.next_detection_time = (
                                        self.current_time + timedelta(days=5)
                                    )

            for event in self.config["hazards"]["floods"]["events"]:
                assert isinstance(event["start_time"], datetime), (
                    f"Start time {event['start_time']} must be a datetime object."
                )
                assert isinstance(event["end_time"], datetime), (
                    f"End time {event['end_time']} must be a datetime object."
                )
                assert event["end_time"] >= event["start_time"], (
                    f"End time {event['end_time']} must be greater than or equal to start time {event['start_time']}."
                )

                routing_substeps: int = self.sfincs.discharge_per_timestep[0].shape[0]

                # since we are at the end of the timestep, we need to check if the current time plus the timestep length is greater than or equal to the start time of the event

                timestep_end_time = (
                    self.current_time
                    + self.timestep_length
                    - self.timestep_length / routing_substeps
                )
                if (
                    timestep_end_time >= event["end_time"]
                    and event["end_time"] + self.timestep_length > timestep_end_time
                ) or (
                    event["end_time"] > self.end_time
                    and event["start_time"] < timestep_end_time
                    and self.current_timestep == self.n_timesteps - 1
                ):
                    event = copy.deepcopy(event)
                    if isinstance(self.model.forcing["pr_hourly"], list):
                        final_forcing_dataset = self.model.forcing["pr_hourly"][-1]
                    else:
                        final_forcing_dataset = self.model.forcing["pr_hourly"]
                    end_of_forcing_date: datetime = pd.to_datetime(
                        final_forcing_dataset.time[-1].item()
                    ).to_pydatetime()
                    if event["end_time"] > end_of_forcing_date:
                        print(
                            f"Warning: Flood event {event} ends after the model end time {self.end_time}. Simulating only part of flood event."
                        )
                        event["end_time"] = end_of_forcing_date

                    print("Running SFINCS for event:", event)
                    self.sfincs.run(event)
