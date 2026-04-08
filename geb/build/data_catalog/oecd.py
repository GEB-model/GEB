"""Adapter for OECD datasets."""

from __future__ import annotations

from geb.workflows.io import fetch_and_save

from .base import Adapter


class OECD(Adapter):
    """Adapter for OECD datasets."""

    def fetch(self, url: str) -> OECD:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The OECD adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path)
        return self
