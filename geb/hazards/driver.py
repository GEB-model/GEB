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

    def step(self, step_size):
        if self.config["hazards"]["floods"]["simulate"]:
            if self.simulate_hydrology:
                self.sfincs.save_discharge()

            for event in self.config["hazards"]["floods"]["events"]:
                assert type(self.current_time.date()) is type(event["end_time"])
                if self.current_time.date() == event["end_time"]:
                    self.sfincs.run(event)
