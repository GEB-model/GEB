import copy
from datetime import datetime

import pandas as pd


class HazardDriver:
    """Class that manages the simulation of short-lived hazards such as floods.

    Currently it only supports floods but can be extended to include other hazards such as landslides in the future.
    """

    def __init__(self):
        if self.config["hazards"]["floods"]["simulate"]:
            # exract the longest flood event in days
            flood_events = self.config["hazards"]["floods"]["events"]
            flood_event_lengths = [
                event["end_time"] - event["start_time"] for event in flood_events
            ]
            longest_flood_event = max(flood_event_lengths).days
            self.initialize(longest_flood_event)

    def initialize(self, longest_flood_event):
        from geb.hazards.floods.sfincs import SFINCS

        self.sfincs = SFINCS(self, n_timesteps=longest_flood_event)

    def step(self):
        if self.config["hazards"]["floods"]["simulate"]:
            if self.simulate_hydrology:
                self.sfincs.save_discharge()

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
