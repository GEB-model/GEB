"""Adapter for COAST-RP datasets."""

from __future__ import annotations

from geb.workflows.io import fetch_and_save

from .base import Adapter


class CoastRP(Adapter):
    """Adapter for COAST-RP datasets."""

    def fetch(
        self,
        url: str = "https://raw.githubusercontent.com/jobdullaart/HGRAPHER/refs/tags/v0.1/COAST-RP.pkl",
    ) -> CoastRP:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The CoastRP adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path)
        return self
