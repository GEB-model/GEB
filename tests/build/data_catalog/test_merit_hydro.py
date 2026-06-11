"""Tests for the MERIT Hydro data catalog adapter."""

from pathlib import Path

import pytest

from geb.build.data_catalog.merit_hydro import MeritHydroDir


def test_merit_hydro_fetch_uses_cached_variable_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify MERIT Hydro uses tiles from the variable cache folder."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    adapter = MeritHydroDir(
        folder="merit_hydro_dir",
        local_version=1,
        filename="tiles",
        cache="global",
    )
    tile_dir: Path = adapter.root / "dir"
    tile_dir.mkdir(parents=True)
    (tile_dir / "n30w120_dir.tif").write_bytes(b"cached")

    result = adapter.fetch(
        xmin=-120.0,
        xmax=-119.0,
        ymin=30.0,
        ymax=31.0,
        url="https://global-hydrodynamics.github.io/MERIT_Hydro/",
    )

    assert result is adapter


def test_merit_hydro_fetch_uses_cached_root_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify MERIT Hydro still accepts tiles in the dataset root folder."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    adapter = MeritHydroDir(
        folder="merit_hydro_dir",
        local_version=1,
        filename="tiles",
        cache="global",
    )
    (adapter.root / "n30w120_dir.tif").write_bytes(b"cached")

    result = adapter.fetch(
        xmin=-120.0,
        xmax=-119.0,
        ymin=30.0,
        ymax=31.0,
        url="https://global-hydrodynamics.github.io/MERIT_Hydro/",
    )

    assert result is adapter


def test_merit_hydro_fetch_raises_clear_error_for_missing_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify missing MERIT Hydro land tiles raise download instructions."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    adapter = MeritHydroDir(
        folder="merit_hydro_dir",
        local_version=1,
        filename="tiles",
        cache="global",
    )
    expected_folder: Path = adapter.root / "dir"

    with pytest.raises(FileNotFoundError) as error:
        adapter.fetch(
            xmin=-120.0,
            xmax=-119.0,
            ymin=30.0,
            ymax=31.0,
            url="https://global-hydrodynamics.github.io/MERIT_Hydro/",
        )

    message: str = str(error.value)
    assert "n30w120_dir.tif" in message
    assert "https://global-hydrodynamics.github.io/MERIT_Hydro/" in message
    assert str(expected_folder) in message
    assert "$GEB_DATA_ROOT/merit_hydro_dir/v1/dir/" in message
    assert "$GEB_DATA_ROOT/merit_hydro_elv/v1/elv/" in message
