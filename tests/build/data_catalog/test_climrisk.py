"""Tests for the CLIMRISK data catalog adapter."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import geopandas as gpd
import h5py
import numpy as np
import pytest
from shapely.geometry import box

from geb.build.data_catalog import climrisk as climrisk_module
from geb.build.data_catalog.climrisk import CLIMRISK


def _adapter(path: Path) -> CLIMRISK:
    """Create an adapter whose data path points to a temporary directory.

    Args:
        path: Temporary directory used as the adapter data path.

    Returns:
        An uninitialized adapter configured with the supplied path.
    """

    class TemporaryCLIMRISK(CLIMRISK):
        @property
        def path(self) -> Path:
            """Return the temporary path used by this test adapter."""
            return path

    return TemporaryCLIMRISK.__new__(TemporaryCLIMRISK)


def _region() -> gpd.GeoDataFrame:
    """Create a small region covering the test grid.

    Returns:
        A WGS84 GeoDataFrame containing one polygon.
    """
    return gpd.GeoDataFrame(geometry=[box(-1, -1, 1, 1)], crs="EPSG:4326")


def _use_small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the adapter's global coordinate arrays with a tiny test grid.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    """
    original_numpy = np

    def small_arange(
        start: float,
        stop: float | None = None,
        step: float = 1,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return short coordinate vectors for CLIMRISK's fixed grid definitions.

        Args:
            start: Start of the requested range.
            stop: End of the requested range, if supplied.
            step: Step between values.
            args: Additional positional arguments passed to ``numpy.arange``.
            kwargs: Additional keyword arguments passed to ``numpy.arange``.

        Returns:
            A short coordinate vector for recognized grid ranges, or the
            original NumPy range for other inputs.
        """
        if (start, stop, step) == (2010, 2101, 1):
            return original_numpy.array([2010, 2011, 2012])
        if (start, stop, step) in (
            (-179.75, 180.25, 0.5),
            (-89.75, 90.25, 0.5),
        ):
            return original_numpy.array([-0.25, 0.25])
        return original_numpy.arange(start, stop, step, *args, **kwargs)

    numpy_proxy = SimpleNamespace(
        **{
            name: getattr(original_numpy, name)
            for name in dir(original_numpy)
            if not name.startswith("_")
        }
    )
    numpy_proxy.arange = small_arange
    monkeypatch.setattr(climrisk_module, "np", numpy_proxy)


def test_read_rejects_invalid_ssp(tmp_path: Path) -> None:
    """Reject an unsupported SSP before attempting to load source maps."""
    with pytest.raises(ValueError, match="Possible climrisk SSPs"):
        _adapter(tmp_path).read(
            region=_region(),
            ssp="9",
            rcp=None,
            damage_function=None,
            warming_percentile=None,
        )


def test_read_rejects_invalid_reference_year(tmp_path: Path) -> None:
    """Reject a reference year outside the supported range."""
    with pytest.raises(ValueError, match="reference_year"):
        _adapter(tmp_path).read(
            region=_region(),
            ssp="2",
            rcp=None,
            damage_function=None,
            warming_percentile=None,
            reference_year=2009,
        )


def test_read_requires_region_crs(tmp_path: Path) -> None:
    """Reject a region without a CRS."""
    region_without_crs = gpd.GeoDataFrame(geometry=[box(-1, -1, 1, 1)])

    with pytest.raises(ValueError, match="must have a CRS"):
        _adapter(tmp_path).read(
            region=region_without_crs,
            ssp="2",
            rcp=None,
            damage_function=None,
            warming_percentile=None,
        )


def test_read_requires_country_map_file(tmp_path: Path) -> None:
    """Raise a clear error when the country map file is absent."""
    with pytest.raises(FileNotFoundError, match="Country map file"):
        _adapter(tmp_path).read(
            region=_region(),
            ssp="2",
            rcp=None,
            damage_function=None,
            warming_percentile=None,
        )


def test_read_requires_country_map_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a country map file without its required MATLAB variable."""
    (tmp_path / "KompasCountry.mat").touch()
    monkeypatch.setattr(climrisk_module, "loadmat", lambda _: {})

    with pytest.raises(ValueError, match="CountryMap"):
        _adapter(tmp_path).read(
            region=_region(),
            ssp="2",
            rcp=None,
            damage_function=None,
            warming_percentile=None,
        )


def test_read_population_returns_cropped_float32_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read and crop population data from mocked small source arrays."""
    (tmp_path / "KompasCountry.mat").touch()
    (tmp_path / "OECD_POP_SSP2_SPATIAL13R.mat").touch()
    _use_small_grid(monkeypatch)

    population_source = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    monkeypatch.setattr(
        climrisk_module,
        "loadmat",
        lambda path: (
            {"CountryMap": np.ones((2, 2), dtype=np.int16)}
            if Path(path).name == "KompasCountry.mat"
            else {"POP_MAP": population_source}
        ),
    )

    result = _adapter(tmp_path).read(
        region=_region(),
        ssp="2",
        rcp=None,
        damage_function=None,
        warming_percentile=None,
        obtain_population=True,
        reference_year=2010,
    )

    assert result.name == "population"
    assert result.dtype == np.float32
    assert dict(result.sizes) == {"year": 3, "lon": 2, "lat": 2}
    assert result.attrs["lon_resolution"] == 0.5
    assert result.attrs["lat_resolution"] == 0.5


def test_read_rejects_region_without_country_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise an error when the region overlaps no country-grid cells."""
    (tmp_path / "KompasCountry.mat").touch()
    (tmp_path / "OECD_POP_SSP2_SPATIAL13R.mat").touch()
    _use_small_grid(monkeypatch)

    population_source = np.ones((2, 2, 3), dtype=np.float32)
    monkeypatch.setattr(
        climrisk_module,
        "loadmat",
        lambda path: (
            {"CountryMap": np.zeros((2, 2), dtype=np.int16)}
            if Path(path).name == "KompasCountry.mat"
            else {"POP_MAP": population_source}
        ),
    )

    with pytest.raises(ValueError, match="No country-grid cells"):
        _adapter(tmp_path).read(
            region=_region(),
            ssp="2",
            rcp=None,
            damage_function=None,
            warming_percentile=None,
            obtain_population=True,
        )


def test_read_gdp_damage_uses_gdp_formula(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that the GDP option uses the GDP damage variables and formula."""
    (tmp_path / "KompasCountry.mat").touch()
    _use_small_grid(monkeypatch)
    monkeypatch.setattr(
        climrisk_module,
        "loadmat",
        lambda _: {"CountryMap": np.ones((2, 2), dtype=np.int16)},
    )

    expected_file = tmp_path / "SSP226_Kompas_results_p10" / "SSP226_K_KU_results10.mat"

    class FakeHDF5File:
        """Provide the two GDP variables expected by the adapter."""

        def __init__(self) -> None:
            self.data = {
                "CLIMRISK_DOLLARS_K_GRID_WORLD_S": np.full(
                    (3, 2, 2), 20.0, dtype=np.float32
                ),
                "CLIMRISK_PROP_K_GRID_WORLD_S": np.full(
                    (3, 2, 2), 50.0, dtype=np.float32
                ),
            }

        def __enter__(self) -> FakeHDF5File:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def __contains__(self, key: str) -> bool:
            return key in self.data

        def __getitem__(self, key: str) -> np.ndarray:
            return self.data[key]

    opened_paths: list[Path] = []

    def fake_hdf5_file(path: Path, mode: str) -> FakeHDF5File:
        opened_paths.append(Path(path))
        return FakeHDF5File()

    monkeypatch.setattr(h5py, "File", fake_hdf5_file)

    result = _adapter(tmp_path).read(
        region=_region(),
        ssp="2",
        rcp=None,
        damage_function="GDP",
        warming_percentile=None,
    )

    assert opened_paths == [expected_file]
    assert result.name == "GDP"
    np.testing.assert_allclose(result.to_numpy(), 40.0)
