"""Tests for the MERIT Hydro data catalog adapter."""

from pathlib import Path

import numpy as np
import pytest
import rioxarray  # noqa: F401  # registers the .rio accessor
import xarray as xr
from affine import Affine

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


def test_merged_tiles_get_identical_coordinates_across_variables(
    tmp_path: Path,
) -> None:
    """Tiles whose coordinates differ by float noise are clipped to the same pixels.

    Regression test: a bbox edge on a pixel coordinate selected a different number of rows for
    flow direction and elevation when their coordinates differed by floating-point noise.
    """
    pixel_size: float = 1 / 1200
    values: np.ndarray = np.arange(20 * 10, dtype=np.float32).reshape(20, 10)
    merged: list[xr.DataArray] = []
    for name, y_top in (("exact", 5.0), ("noisy", 5.0 + 1e-12)):
        tile: xr.DataArray = xr.DataArray(
            values,
            dims=("y", "x"),
            coords={
                "y": y_top - (np.arange(20) + 0.5) * pixel_size,
                "x": 43.0 + (np.arange(10) + 0.5) * pixel_size,
            },
        ).rio.write_crs(4326)
        tile.rio.write_transform(
            Affine(pixel_size, 0.0, 43.0, 0.0, -pixel_size, y_top), inplace=True
        )
        tile.rio.to_raster(tmp_path / f"{name}.tif")
        adapter = MeritHydroDir(
            folder="merit_hydro_dir", local_version=1, filename="tiles", cache="global"
        )
        adapter._source_nodata = -9999.0  # normally set by fetch()
        merged.append(adapter._merge_merit_tiles([tmp_path / f"{name}.tif"]))

    exact, noisy = merged
    assert np.array_equal(exact.y.values, noisy.y.values)
    assert np.array_equal(exact.x.values, noisy.x.values)
    # clipping at an edge that lies exactly on a pixel coordinate selects the same rows
    y_edge: float = float(exact.y.values[3])
    assert (
        exact.sel(y=slice(y_edge, 4.99)).shape == noisy.sel(y=slice(y_edge, 4.99)).shape
    )
    # values and dtype are unchanged
    np.testing.assert_array_equal(exact.values, values)
    assert exact.dtype == np.float32
