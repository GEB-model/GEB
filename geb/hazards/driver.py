"""Module for managing short-lived hazard simulations such as floods."""

import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from geb.geb_types import TwoDArrayFloat32
from geb.hazards.event import Event
from geb.hazards.floods import Floods
from geb.module import Module
from geb.workflows.io import read_geom

if TYPE_CHECKING:
    from geb.model import GEBModel


def first_discharge_exceedance(
    discharge_m3_s: np.ndarray,
    threshold_m3_s: float,
    start_time: datetime,
    timestep_length: timedelta,
) -> datetime | None:
    """Find the first sampled discharge strictly above a threshold.

    Args:
        discharge_m3_s: One-dimensional discharge samples (m³/s).
        threshold_m3_s: Discharge threshold (m³/s).
        start_time: Start of the model timestep, in model time.
        timestep_length: Duration covered by the equally spaced samples.

    Returns:
        First exceedance time, or None if no sample exceeds the threshold.
        Samples follow the interval-start convention used by SFINCS forcing.

    Raises:
        ValueError: If samples, threshold, or timestep duration are invalid.
    """
    # Validate inputs
    if discharge_m3_s.ndim != 1 or discharge_m3_s.size == 0:
        raise ValueError("Discharge must be a nonempty one-dimensional array.")
    if not np.isfinite(discharge_m3_s).all():
        raise ValueError("Discharge samples must all be finite.")
    if not np.isfinite(threshold_m3_s) or threshold_m3_s < 0:
        raise ValueError("Discharge threshold must be finite and nonnegative.")
    if timestep_length <= timedelta(0):
        raise ValueError("Timestep duration must be positive.")

    # Get the indices of the samples that exceed the threshold
    indices: np.ndarray = np.flatnonzero(discharge_m3_s > threshold_m3_s)
    if indices.size == 0:
        return None
    substep_length = timestep_length / discharge_m3_s.size
    return start_time + int(indices[0]) * substep_length


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
        self.config = self.model.config["hazards"]
        if "events" not in self.model.config["hazards"]["floods"]:
            self.config["floods"]["events"] = []

        self.flood_events: list[Event] = [
            Event(
                kind="flood",
                name=f"{event['start_time'].strftime(format='%Y%m%dT%H%M%S')} - {event['end_time'].strftime(format='%Y%m%dT%H%M%S')}",
                start_time=event["start_time"],
                end_time=event["end_time"],
                create_max_intensity_map=True,
            )
            for event in (self.config["floods"]["events"] or [])
        ]

        if (
            not self.model.in_spinup
            and self.config["floods"]["run_for_validation_events"]
        ):
            validation_events = read_geom(
                self.model.files["geom"]["observations/floods"]
            )
            for _, event in validation_events.iterrows():
                date: pd.Timestamp = event["observation_date"]
                start_time = date.to_pydatetime().replace(
                    minute=0, second=0, microsecond=0
                ) - timedelta(days=5)

                event = Event(
                    kind="flood",
                    name=event["name"],
                    start_time=start_time,
                    end_time=date,
                    create_final_intensity_map=True,
                )

                self.flood_events.append(event)

        # extract the longest flood event in days
        if self.flood_events == [] or self.flood_events is None:
            longest_flood_event_in_days: int = 0
        else:
            flood_event_lengths: list[timedelta] = [
                event.end_time - event.start_time for event in self.flood_events
            ]
            longest_flood_event_in_days = max(flood_event_lengths).days

        self.initialize(
            longest_flood_event_in_days=longest_flood_event_in_days + 10
        )  # Here we add 10 days as a buffer to ensure we capture the full event

        self.next_detection_time: datetime | None = (
            None  # Variable to keep track of when a flood has happened
        )

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

        Raises:
            ValueError: If a flood simulation is requested during spinup or the
                forecast window does not end after its issue date.
        """
        if self.model.config["hazards"]["floods"]["simulate"]:
            # Get the configuration for flood detection and simulation
            # This is used to help distinguish between forecast and regular flood detection
            forecasts_enabled: bool = self.model.config["general"]["forecasts"]["use"]
            forecast_detection: bool = (
                forecasts_enabled
                and self.model.multiverse_name is not None
                and self.model.multiverse_mode == "discharge_detection"
            )
            regular_detection: bool = (
                not forecasts_enabled and self.model.multiverse_name is None
            )
            detection_enabled: bool = self.config["floods"][
                "detect_floods_from_discharge"
            ]
            detection_allowed: bool = (
                self.next_detection_time is None
                or self.model.current_time >= self.next_detection_time
            )

            if (
                detection_enabled
                and (forecast_detection or regular_detection)
                and not self.model.in_spinup
                and detection_allowed
            ):
                # Sample all substeps of the current timestep to later check for exceedances
                discharge_current_timestep: TwoDArrayFloat32 = (
                    self.floods.var.discharge_per_timestep[-1]
                )
                discharge_grid: xr.DataArray = xr.DataArray(
                    data=self.model.hydrology.grid.decompress(
                        discharge_current_timestep
                    ),
                    coords={
                        "substep": np.arange(discharge_current_timestep.shape[0]),
                        "y": self.model.hydrology.grid.lat,
                        "x": self.model.hydrology.grid.lon,
                    },
                    dims=["substep", "y", "x"],
                    name="discharge",
                ).rio.write_crs(self.model.crs)

                # Get the discharge for all substeps at the threshold location and check for exceedances
                threshold_location: tuple[float, float] = self.config["floods"][
                    "threshold_location"
                ]
                station_discharge: xr.DataArray = discharge_grid.sel(
                    x=threshold_location[0],
                    y=threshold_location[1],
                    method="nearest",
                ).compute()

                threshold_m3_s: float = self.config["floods"]["discharge_threshold"]
                maximum_discharge_m3_s: float = float(
                    station_discharge.max(dim="substep", skipna=False).values
                )

                # Check at which substep the discharge first exceeds the threshold, if at all
                # This will later be used to compute the lead time of the event
                first_exceedance_time: datetime | None = first_discharge_exceedance(
                    station_discharge.values,
                    threshold_m3_s,
                    self.model.current_time,
                    self.model.current_time
                    + self.model.timestep_length
                    - self.model.current_time,
                )

                if forecast_detection:
                    self.model.discharge_log.append(
                        {
                            "forecast_issue_date": self.model.forecast_issue_datetime,
                            "time": self.model.current_time,
                            "discharge": maximum_discharge_m3_s,
                            "member": self.model.forecast_member,
                            "first_exceedance_time": first_exceedance_time,
                        }
                    )
                    return

                self.model.discharge_log.append(
                    {
                        "time": self.model.current_time,
                        "discharge": maximum_discharge_m3_s,
                    }
                )

                # If the discharge exceeds the threshold, schedule a flood event for simulation
                if first_exceedance_time is not None:
                    start_time: datetime = self.model.current_time
                    end_time: datetime = start_time + timedelta(days=10)
                    event_already_scheduled: bool = any(
                        event.start_time == start_time and event.end_time == end_time
                        for event in self.flood_events
                    )
                    if not event_already_scheduled:
                        self.flood_events.append(
                            Event(
                                kind="flood",
                                name=(
                                    f"{start_time.strftime('%Y%m%dT%H%M%S')} - "
                                    f"{end_time.strftime('%Y%m%dT%H%M%S')}"
                                ),
                                start_time=start_time,
                                end_time=end_time,
                                create_max_intensity_map=True,
                            )
                        )

                    new_event: dict[str, datetime] = {
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                    if self.config["floods"].get("events") is None:
                        self.config["floods"]["events"] = []
                    events_in_memory: list[dict[str, datetime]] = self.config["floods"][
                        "events"
                    ]
                    if new_event not in events_in_memory:
                        events_in_memory.append(new_event.copy())

                    config_path: Path = Path.cwd() / "model.yml"
                    with config_path.open("r") as config_file:
                        config_yaml: dict = yaml.safe_load(config_file) or {}
                    floods_yaml: dict = config_yaml.setdefault(
                        "hazards", {}
                    ).setdefault("floods", {})
                    if floods_yaml.get("events") is None:
                        floods_yaml["events"] = []
                    events_in_yaml: list[dict[str, datetime]] = floods_yaml["events"]
                    if new_event not in events_in_yaml:
                        events_in_yaml.append(new_event.copy())
                        # Replace atomically to avoid leaving a partially written config.
                        temporary_config_path: Path = config_path.with_suffix(".tmp")
                        with temporary_config_path.open("w") as config_file:
                            yaml.safe_dump(config_yaml, config_file, sort_keys=False)
                        temporary_config_path.replace(config_path)

            if regular_detection and detection_enabled and not self.model.in_spinup:
                reporting_end_time: datetime = datetime.combine(
                    self.model.config["general"]["end_time"], datetime.min.time()
                )
                if self.model.current_time == reporting_end_time:
                    discharge_log: pd.DataFrame = pd.DataFrame(self.model.discharge_log)
                    discharge_log.to_csv(
                        Path(self.model.output_folder) / "discharge_timeseries.csv",
                        index=False,
                    )

            if (
                self.model.multiverse_name is None
                or not self.model.config["general"]["forecasts"]["long_term_analysis"]
            ):
                for event in self.flood_events:
                    event: Event = copy.deepcopy(event)
                    if self.model.multiverse_name is not None:
                        event.name = f"{self.model.multiverse_name}/{event.name}"
                    assert isinstance(event.start_time, datetime), (
                        f"Start time {event.start_time} must be a datetime object."
                    )
                    assert isinstance(event.end_time, datetime), (
                        f"End time {event.end_time} must be a datetime object."
                    )
                    assert event.end_time >= event.start_time, (
                        f"End time {event.end_time} must be greater than or equal to start time {event.start_time}."
                    )

                    routing_substeps: int = self.floods.var.discharge_per_timestep[
                        0
                    ].shape[0]

                    # since we are at the end of the timestep, we need to check if the current time plus the timestep length is greater than or equal to the start time of the event
                    timestep_end_time: datetime = (
                        self.model.current_time
                        + self.model.timestep_length
                        - self.model.timestep_length / routing_substeps
                    )
                    if (
                        timestep_end_time >= event.end_time
                        and event.end_time + self.model.timestep_length
                        > timestep_end_time
                    ) or (
                        event.end_time > self.model.simulation_end
                        and event.start_time < timestep_end_time
                        and self.model.current_timestep == self.model.n_timesteps - 1
                    ):
                        # the actual end time is the end of the day of the simulation. Therefore,
                        # its the simulation end time plus one timestep length
                        if (
                            event.end_time
                            > self.model.simulation_end + self.model.timestep_length
                        ):
                            print(
                                f"Warning: Flood event {event} ends after the model end time {self.model.simulation_end}. Simulating only part of flood event."
                            )
                            event.end_time = (
                                self.model.simulation_end + self.model.timestep_length
                            )
                            assert event.end_time > event.start_time

                        if self.model.in_spinup:
                            msg = f"Flood event {event} cannot be simulated during spinup. Please adjust the flood event times or the spinup time in the configuration."
                            self.model.logger.error(msg)
                            raise ValueError(msg)

                        print(
                            f"Running floods for event from {event.start_time} to {event.end_time}"
                        )
                        self.floods.run_single_event(event)

            if self.model.multiverse_mode == "flood_simulation":
                if self.model.in_spinup:
                    raise ValueError(
                        "Forecast flood simulations cannot run during spinup."
                    )

                # Stored forcing is already complete, run SFINCS once per member.
                if self.model.current_time != self.model.forecast_issue_datetime:
                    return

                forecast_start_time: datetime = self.model.forecast_issue_datetime
                # Include the entire final hydrological timestep saved in the reports
                forecast_end_time: datetime = (
                    self.model.simulation_end + self.model.timestep_length
                )
                if forecast_end_time <= forecast_start_time:
                    raise ValueError(
                        "The forecast simulation window must end after its issue date."
                    )

                forecasted_event: Event = Event(
                    kind="flood",
                    name=(
                        f"{self.model.multiverse_name}/"
                        f"{forecast_start_time.strftime('%Y%m%dT%H%M%S')} - "
                        f"{forecast_end_time.strftime('%Y%m%dT%H%M%S')}"
                    ),
                    start_time=forecast_start_time,
                    end_time=forecast_end_time,
                    create_max_intensity_map=True,
                )
                self.model.logger.info(
                    "Running forecast flood simulation from "
                    f"{forecasted_event.start_time} to {forecasted_event.end_time} "
                    f"for member {self.model.forecast_member}."
                )
                self.floods.run_single_event(forecasted_event)
