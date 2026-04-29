"""FieldBoundaries adapter."""

from typing import Any

import geopandas as gpd

from geb.workflows.io import fetch_and_save, read_geom

from .base import Adapter


class FieldBoundaries(Adapter):
    """Adapter for the FieldBoundaries dataset.

    This datasets provides global information on aquifer properties.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the FieldBoundaries adapter.

        Args:
            *args: Arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.

        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str, *args: Any, **kwargs: Any) -> FieldBoundaries:
        """Fetch the FieldBoundaries dataset.

        Args:
            url: URL to fetch the dataset from.
            *args: Additional arguments to pass to fetch_and_save.
            **kwargs: Additional keyword arguments to pass to fetch_and_save.

        Returns:
            The FieldBoundaries adapter.
        """
        if not self.path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            fetch_and_save(
                url=url,
                file_path=self.path,
                **kwargs,
            )
        return self

    def read(
        self,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> gpd.GeoDataFrame:
        """Read the FieldBoundaries dataset from parquet.

        Args:
            bounds: Optional bounding box as
                ``(min_x, min_y, max_x, max_y)``.
                If provided, only features intersecting the bounding box
                are read.

        Returns:
            The field boundaries as a GeoDataFrame.

        Raises:
            FileNotFoundError: If the parquet file does not exist yet.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"Field boundaries dataset was not found at {self.path}. "
                "Fetch the dataset before reading it."
            )

        if bounds is None:
            return read_geom(self.path)

        return read_geom(self.path, bbox=bounds)
