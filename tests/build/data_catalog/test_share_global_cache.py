"""Test that data written to the global cache is shared with other users."""

import logging
import stat
from pathlib import Path
from typing import Any

import pytest

from geb.build.data_catalog import DataCatalog
from geb.build.data_catalog.base import Adapter


class PrivateFileAdapter(Adapter):
    """Adapter that writes a file with private permissions, like tempfile does."""

    def fetch(self, *args: Any, **kwargs: Any) -> Adapter:
        """Write a private file to the adapter path.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            The Adapter instance.
        """
        self.path.write_bytes(b"data")
        self.path.chmod(0o600)
        self.root.chmod(0o700)
        return self


@pytest.fixture
def data_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DataCatalog, dict[str, PrivateFileAdapter]]:
    """Create a data catalog with one global and one local adapter.

    Args:
        tmp_path: Temporary directory used as global cache and working directory.
        monkeypatch: Fixture for setting the environment and working directory.

    Returns:
        The data catalog and the adapters it contains.
    """
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path / "global"))
    monkeypatch.chdir(tmp_path)
    adapters: dict[str, PrivateFileAdapter] = {
        "global_entry": PrivateFileAdapter(
            folder="global_entry", filename="data.bin", local_version=1, cache="global"
        ),
        "local_entry": PrivateFileAdapter(
            folder="local_entry", filename="data.bin", local_version=1, cache="local"
        ),
    }
    catalog = DataCatalog(logger=logging.getLogger(__name__))
    catalog.catalog = {
        name: {"adapter": adapter, "url": None} for name, adapter in adapters.items()
    }
    return catalog, adapters


def test_global_cache_is_shared(
    data_catalog: tuple[DataCatalog, dict[str, PrivateFileAdapter]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that only global cache data is shared when the flag is set."""
    monkeypatch.setenv("GEB_SHARE_GLOBAL_CACHE", "true")
    catalog, adapters = data_catalog

    catalog.fetch("global_entry")
    catalog.fetch("local_entry")

    global_adapter = adapters["global_entry"]
    assert stat.S_IMODE(global_adapter.path.stat().st_mode) == 0o664
    assert stat.S_IMODE(global_adapter.root.stat().st_mode) == 0o2775

    local_adapter = adapters["local_entry"]
    assert stat.S_IMODE(local_adapter.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(local_adapter.root.stat().st_mode) == 0o700


def test_global_cache_not_shared_without_flag(
    data_catalog: tuple[DataCatalog, dict[str, PrivateFileAdapter]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that permissions are left untouched when the flag is false."""
    monkeypatch.setenv("GEB_SHARE_GLOBAL_CACHE", "false")
    catalog, adapters = data_catalog

    catalog.fetch("global_entry")

    global_adapter = adapters["global_entry"]
    assert stat.S_IMODE(global_adapter.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(global_adapter.root.stat().st_mode) == 0o700
