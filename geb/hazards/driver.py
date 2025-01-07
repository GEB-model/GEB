class HazardDriver:
    def __init__(self):
        if self.config["hazards"]["floods"]["simulate"]:
            from geb.hazards.floods.sfincs import SFINCS

            # exract the longest flood event in days
            flood_events = self.config["hazards"]["floods"]["events"]
            flood_event_lengths = [
                event["end_time"] - event["start_time"] for event in flood_events
            ]
            longest_flood_event = max(flood_event_lengths).days
            self.sfincs = SFINCS(
                self, config=self.config, max_number_of_timesteps=longest_flood_event
            )

    def step(self, step_size):
        if self.config["hazards"]["floods"]["simulate"]:
            if self.config["general"]["simulate_hydrology"]:
                self.sfincs.save_discharge()
                self.sfincs.save_soil_moisture()
                self.sfincs.save_max_soil_moisture()
                self.sfincs.save_ksat()

            for event in self.config["hazards"]["floods"]["events"]:
                assert type(self.current_time.date()) is type(event["end_time"])
                if self.current_time.date() == event["end_time"]:
                    self.sfincs.run(event)

            if (
                "calculate_return_period_maps_at_and_of_spinup"
                in self.config["hazards"]["floods"]
                and self.config["hazards"]["floods"][
                    "calculate_return_period_maps_at_and_of_spinup"
                ]
                is True
                and self.current_timestep == self.n_timesteps - 1
                and self.spinup
            ):
                self.sfincs.get_return_period_maps()
