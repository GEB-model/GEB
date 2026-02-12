"""Adapter for MIRCA2000 datasets."""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path
from typing import Any

from geb.workflows.io import fetch_and_save

from .base import Adapter


class MIRCA2000(Adapter):
    """Adapter for MIRCA2000 ZIP archives containing a single target file."""

    def __init__(
        self,
        *args: Any,
        target_member_suffix: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the MIRCA2000 ZIP adapter.

        Args:
            *args: Positional arguments passed to the base Adapter.
            target_member_suffix: File name suffix to extract from the ZIP archive.
            **kwargs: Keyword arguments passed to the base Adapter.
        """
        super().__init__(*args, **kwargs)
        self.target_member_suffix: str = target_member_suffix

    def _find_member(self, zip_ref: zipfile.ZipFile) -> str:
        """Find a ZIP member matching the configured suffix.

        Args:
            zip_ref: Open ZIP file handle.

        Returns:
            The ZIP member name that matches the configured suffix.

        Raises:
            FileNotFoundError: If no matching member or multiple matching members are found.
        """
        members: list[str] = zip_ref.namelist()
        matches: list[str] = [
            member for member in members if member.endswith(self.target_member_suffix)
        ]
        if not len(matches) == 1:
            raise FileNotFoundError(
                f"Expected exactly one member with suffix "
                f"'{self.target_member_suffix}', found {len(matches)}."
            )

        return matches[0]

    def fetch(self, url: str) -> MIRCA2000:
        """Download and extract a target file from a MIRCA2000 ZIP archive.

        Args:
            url: URL of the MIRCA2000 ZIP archive.

        Returns:
            The MIRCA2000 ZIP adapter instance.
        """
        if self.is_ready:
            return self

        zip_filename: str = f"{self.path.stem}.zip"
        download_path: Path = self.root / zip_filename
        fetch_and_save(url=url, file_path=download_path)

        with zipfile.ZipFile(file=download_path, mode="r") as zip_ref:
            member_name: str = self._find_member(zip_ref)
            zip_ref.extract(member_name, self.root)

        extracted_path: Path = self.root / member_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if extracted_path.suffix == ".gz":
            with (
                gzip.open(extracted_path, "rb") as source,
                open(self.path, "wb") as target,
            ):
                target.write(source.read())
            extracted_path.unlink()
        elif extracted_path != self.path:
            extracted_path.replace(self.path)

        download_path.unlink()

        return self

    def read(self, **kwargs: Any) -> Any:
        """Read the extracted MIRCA2000 file.

        Args:
            **kwargs: Additional keyword arguments forwarded to the base reader.

        Returns:
            For text files, a list of lines. For other files, the base adapter
            read result.
        """
        if self.path.suffix == ".txt":
            return self.path.read_text().splitlines()
        return Adapter.read(self, **kwargs)
