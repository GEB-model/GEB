"""Module for managing short-lived hazard simulations such as floods."""

import copy
from datetime import datetime, timedelta
from typing import Any

from geb.hazards.floods import Floods


class HazardDriver:
    """Class that manages the simulation of short-lived hazards such as floods.

    Currently it only supports floods but can be extended to include other hazards such as landslides in the future.
    """

    def __init__(self) -> None:
        """Initializes the HazardDriver class.

        If flood simulation is enabled in the configuration, it initializes the flood simulation by determining
        the longest flood event duration and setting up the SFINCS model accordingly.
        """
        # extract the longest flood event in days
        flood_events: list[dict[str, Any]] = self.config["hazards"]["floods"]["events"]
        if flood_events == [] or flood_events is None:
            longest_flood_event_in_days: int = 0
        else:
            flood_event_lengths: list[timedelta] = [
                event["end_time"] - event["start_time"] for event in flood_events
            ]
            longest_flood_event_in_days = max(flood_event_lengths).days
        self.initialize(longest_flood_event_in_days=longest_flood_event_in_days + 1)

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
            self, longest_flood_event_in_days=longest_flood_event_in_days
        )

    def step(self) -> None:
        """Steps the hazard driver.

        If flood simulation is enabled in the configuration, it runs the SFINCS model for each flood event
        that ends during the current timestep.
        """
        if self.config["hazards"]["floods"]["simulate"]:
            if self.simulate_hydrology:
                self.floods.save_discharge()
                self.floods.save_runoff_m()

            if self.config["hazards"]["floods"]["events"] is None:
                return

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

                routing_substeps: int = self.floods.var.discharge_per_timestep[0].shape[
                    0
                ]

                # since we are at the end of the timestep, we need to check if the current time plus the timestep length is greater than or equal to the start time of the event
                timestep_end_time: datetime = (
                    self.current_time
                    + self.timestep_length
                    - self.timestep_length / routing_substeps
                )
                if (
                    timestep_end_time >= event["end_time"]
                    and event["end_time"] + self.timestep_length > timestep_end_time
                ) or (
                    event["end_time"] > self.simulation_end
                    and event["start_time"] < timestep_end_time
                    and self.current_timestep == self.n_timesteps - 1
                ):
                    event: dict[str, datetime] = copy.deepcopy(event)

                    # the actual end time is the end of the day of the simulation. Therefore,
                    # its the simulation end time plus one timestep length
                    if (
                        event["end_time"]
                        > self.model.simulation_end + self.model.timestep_length
                    ):
                        print(
                            f"Warning: Flood event {event} ends after the model end time {self.simulation_end}. Simulating only part of flood event."
                        )
                        event["end_time"] = (
                            self.model.simulation_end + self.model.timestep_length
                        )
                        assert event["end_time"] > event["start_time"]

                    print("Running floods for event:", event)
                    self.floods.run(event)
