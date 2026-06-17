"""Event class object."""

from datetime import datetime
from typing import Literal


class Event:
    """Class representing a hazard event, such as a flood."""

    def __init__(
        self,
        kind: Literal["flood"],
        name: str,
        start_time: datetime,
        end_time: datetime,
        create_max_intensity_map: bool = False,
        create_final_intensity_map: bool = False,
        create_interval_maps: int | None = None,
    ) -> None:
        """Initializes the Event class.

        Args:
            kind: The kind of the hazard event (e.g., "flood", "drought").
            name: The name of the hazard event.
            start_time: The start time of the event.
            end_time: The end time of the event.
            create_max_intensity_map: Whether to create maximum intensity map for the event.
            create_final_intensity_map: Whether to create final intensity map for the event.
            create_interval_maps: The interval at which to create maps for the event.

        Raises:
            ValueError: If none of the map creation options are set to True or a positive integer
        """
        self.kind = kind
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.export_max_intensity = create_max_intensity_map
        self.export_final_intensity = create_final_intensity_map
        self.export_interval = create_interval_maps

        if (
            not self.export_max_intensity
            and not self.export_final_intensity
            and self.export_interval is None
        ):
            raise ValueError(
                "At least one of create_max_intensity_map, create_final_intensity_map, or create_interval_maps must be set to True or a positive integer."
            )

    def __repr__(self) -> str:
        """Returns a string representation of the Event object."""
        return (
            f"Event(kind={self.kind}, name={self.name}, "
            f"start_time={self.start_time}, end_time={self.end_time}, "
            f"export_max_intensity={self.export_max_intensity}, "
            f"export_final_intensity={self.export_final_intensity}, "
            f"export_interval={self.export_interval})"
        )
