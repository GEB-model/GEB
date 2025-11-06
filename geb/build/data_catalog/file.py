"""File adapter for datasets stored as files.

Useful for simple files that only need to be downloaded and stored without additional processing.
"""

from __future__ import annotations

from typing import Any

from geb.workflows.io import fetch_and_save

from .base import Adapter


class File(Adapter):
    """Adapter for File datasets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the File adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> File:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The GlobGM adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path)
        return self
