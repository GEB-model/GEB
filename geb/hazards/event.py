"""Event class object."""

from datetime import datetime


class Event:
    """Class representing a hazard event, such as a flood."""

    def __init__(
        self,
        kind: str,
        name: str,
        start_time: datetime,
        end_time: datetime,
        export_max_intensity: bool = False,
        export_final_intensity: bool = False,
        export_interval: int | None = None,
    ) -> None:
        """Initializes the Event class.

        Args:
            kind: The kind of the hazard event (e.g., "flood", "drought").
            name: The name of the hazard event.
            start_time: The start time of the event.
            end_time: The end time of the event.
            export_max_intensity: Whether to export the maximum intensity of the event.
            export_final_intensity: Whether to export the final intensity of the event.
            export_interval: The interval at which to export the event data.
        """
        self.kind = kind
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.export_max_intensity = export_max_intensity
        self.export_final_intensity = export_final_intensity
        self.export_interval = export_interval

    def __repr__(self) -> str:
        """Returns a string representation of the Event object."""
        return (
            f"Event(kind={self.kind}, name={self.name}, "
            f"start_time={self.start_time}, end_time={self.end_time}, "
            f"export_max_intensity={self.export_max_intensity}, "
            f"export_final_intensity={self.export_final_intensity}, "
            f"export_interval={self.export_interval})"
        )
