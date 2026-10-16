"""Tests for hydrography module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine

from geb.build.modules.hydrography import (
    Hydrography,
    calculate_dem_floodplain_width,
    calculate_stream_length,
)


def test_calculate_stream_length_high_res_diagonal() -> None:
    """Test calculate_stream_length_high_res with diagonal flow."""
    # Create simple grid: 1x1 degree cells
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    # 2x2 grid
    # Cell 0,0: Diagonal flow (2 is SE)
    # Cell 0,1: Below threshold
    # The dimensions in calculate_width_m rely on lat/lon
    # Using lat=0 for simplicity (equator)

    ups = xr.DataArray(
        np.array([[2_000_000, 500_000], [500_000, 500_000]], dtype=float),
        coords={"y": [0.5, -0.5], "x": [0.5, 1.5]},
        dims=("y", "x"),
    )
    ups.rio.write_transform(transform, inplace=True)

    # LDD: 2 is SE (diagonal), others irrelevant
    ldd = xr.DataArray(
        np.array([[2, 0], [0, 0]], dtype=np.uint8), coords=ups.coords, dims=ups.dims
    )

    length = calculate_stream_length(ldd, ups, threshold_m2=1_000_000)

    # Check shape
    assert length.shape == ups.shape

    # Calculate expected length for cell at equator
    # 1 degree approx 111320 m
    # Diagonal length = sqrt(111320**2 + 111320**2) approx 157424
    val = length.isel(y=0, x=0).item()
    assert 150000 < val < 160000

    # Check non-stream cell is 0
    assert length.isel(y=0, x=1).item() == 0


def test_calculate_stream_length_high_res_cardinal() -> None:
    """Test calculate_stream_length_high_res with cardinal flow."""
    # Create simple grid: 1x1 degree cells
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    ups = xr.DataArray(
        np.array([[2_000_000, 2_000_000], [0, 0]], dtype=float),
        coords={"y": [0.5, -0.5], "x": [0.5, 1.5]},
        dims=("y", "x"),
    )
    ups.rio.write_transform(transform, inplace=True)

    # LDD: 1 is East (horizontal), 4 is South (vertical)
    ldd = xr.DataArray(
        np.array([[1, 4], [0, 0]], dtype=np.uint8), coords=ups.coords, dims=ups.dims
    )

    length = calculate_stream_length(ldd, ups, threshold_m2=1_000_000)

    # Horizontal
    val_h = length.isel(y=0, x=0).item()
    # Vertical
    val_v = length.isel(y=0, x=1).item()

    # Approx 111km
    assert 110000 < val_h < 112000
    assert 110000 < val_v < 112000


def test_calculate_dem_floodplain_width() -> None:
    """Test calculate_dem_floodplain_width extracts floodplain width per river reach using subbasins."""
    import pyflwdir

    # 4x4 high-res grid with 2x2 coarsening factor (ldd_scale_factor=2) -> 2x2 low-res grid
    # Low valley in columns 1-2, higher hills on borders
    elev_vals = np.array(
        [
            [12.0, 5.0, 5.0, 12.0],
            [11.0, 4.0, 4.0, 11.0],
            [10.0, 3.0, 3.0, 10.0],
            [9.0, 2.0, 2.0, 9.0],
        ],
        dtype=np.float32,
    )
    ldd_vals = np.array(
        [
            [2, 2, 2, 4],
            [2, 2, 2, 4],
            [2, 2, 2, 4],
            [5, 5, 5, 5],
        ],
        dtype=np.uint8,
    )

    transform = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 0.0)
    elev_high_res = xr.DataArray(
        elev_vals,
        coords={"y": [0.0, -0.01, -0.02, -0.03], "x": [0.0, 0.01, 0.02, 0.03]},
        dims=("y", "x"),
    )
    elev_high_res.rio.write_transform(transform, inplace=True)

    flw_high_res = pyflwdir.from_array(ldd_vals, ftype="ldd", transform=transform)
    upstream_area_vals = flw_high_res.upstream_area(unit="m2").astype(np.float32)
    upstream_area_high_res = xr.DataArray(
        upstream_area_vals, coords=elev_high_res.coords, dims=elev_high_res.dims
    )
    upstream_area_high_res.rio.write_transform(transform, inplace=True)

    # Low-res coordinates
    cell_area_low_res = xr.DataArray(
        np.full((2, 2), 4_000_000.0, dtype=np.float32),
        coords={"y": [-0.005, -0.025], "x": [0.005, 0.025]},
        dims=("y", "x"),
    )
    river_length_low_res = xr.DataArray(
        np.full((2, 2), 2000.0, dtype=np.float32),
        coords=cell_area_low_res.coords,
        dims=cell_area_low_res.dims,
    )
    idxs_outflow_low_res = xr.DataArray(
        np.array([[5, 7], [13, 15]], dtype=np.int64),
        coords=cell_area_low_res.coords,
        dims=cell_area_low_res.dims,
    )
    # Right column (col 1) has river segments; left column (col 0) is non-river
    river_raster_low_res = np.array([[-1, 101], [-1, 102]], dtype=np.int32)

    fp_width = calculate_dem_floodplain_width(
        flow_raster_high_res=flw_high_res,
        elevation_high_res=elev_high_res,
        upstream_area_high_res_m2=upstream_area_high_res,
        idxs_outflow_low_res=idxs_outflow_low_res,
        river_raster_low_res=river_raster_low_res,
        cell_area_low_res=cell_area_low_res,
        river_length_low_res=river_length_low_res,
        ldd_scale_factor=2,
    )

    assert fp_width.shape == (2, 2)
    # Non-river cells must be NaN
    assert np.isnan(fp_width.values[0, 0])
    assert np.isnan(fp_width.values[1, 0])
    # River cells must have valid floodplain widths
    assert not np.isnan(fp_width.values[0, 1])
    assert not np.isnan(fp_width.values[1, 1])
    assert (fp_width.values[:, 1] >= 0.0).all()


AR6_YEARS: np.ndarray = np.arange(2020, 2151, 10)
STATIONS_FILE: str = "geom/gtsm/stations.parquet"


def make_ar6_dataset(
    lon_deg: list[float],
    lat_deg: list[float],
    slope_mm_per_year: list[float],
    offset_mm: list[float],
    years: np.ndarray = AR6_YEARS,
    transpose: bool = False,
) -> xr.Dataset:
    """Create a synthetic AR6 dataset with linear sea level change per location.

    Args:
        lon_deg: Longitude of each AR6 location (degrees).
        lat_deg: Latitude of each AR6 location (degrees).
        slope_mm_per_year: Sea level change rate of each location (mm/year).
        offset_mm: Sea level change of each location in 2020 (mm).
        years: Years in the dataset.
        transpose: Whether to store the variable as (years, locations).

    Returns:
        Dataset mimicking the structure of the AR6 data read by the data catalog.
    """
    slope: np.ndarray = np.asarray(slope_mm_per_year, dtype=float)[:, None]
    offset: np.ndarray = np.asarray(offset_mm, dtype=float)[:, None]
    values_mm: np.ndarray = offset + slope * (years[None, :] - 2020)
    dims: tuple[str, str] = ("locations", "years")
    if transpose:
        values_mm = values_mm.T
        dims = ("years", "locations")
    return xr.Dataset(
        data_vars={"sea_level_change": (dims, values_mm)},
        coords={
            "years": years,
            "lon": ("locations", np.asarray(lon_deg, dtype=float)),
            "lat": ("locations", np.asarray(lat_deg, dtype=float)),
        },
    )


def run_setup_sea_level_projections_ar6(
    ar6: xr.Dataset,
    station_lon_deg: list[float],
    station_lat_deg: list[float],
    station_ids: list[int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: int = 245,
    percentile: float = 0.5,
) -> tuple[MagicMock, pd.DataFrame]:
    """Run setup_sea_level_projections_ar6 on a mocked Hydrography object.

    Args:
        ar6: AR6 dataset returned by the mocked data catalog.
        station_lon_deg: Station longitudes (degrees).
        station_lat_deg: Station latitudes (degrees).
        station_ids: Station index values.
        tmp_path: Temporary directory used as working directory.
        monkeypatch: Pytest monkeypatch fixture.
        scenario: SSP scenario passed to the method.
        percentile: Quantile passed to the method.

    Returns:
        The mocked Hydrography object and the table passed to ``set_table``,
        re-indexed by integer year.
    """
    stations_path: Path = tmp_path / "input" / STATIONS_FILE
    stations_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(station_lon_deg, station_lat_deg),
        index=pd.Index(station_ids),
        crs="EPSG:4326",
    ).to_parquet(stations_path)
    monkeypatch.chdir(tmp_path)

    mock_hydrography: MagicMock = MagicMock()
    mock_hydrography.logger = logging.getLogger("test")
    mock_hydrography.files = {"geom": {"gtsm/stations": STATIONS_FILE}}
    mock_hydrography.data_catalog.fetch.return_value.read.return_value = ar6

    Hydrography.setup_sea_level_projections_ar6(
        mock_hydrography, scenario=scenario, percentile=percentile
    )

    mock_hydrography.set_table.assert_called_once()
    table: pd.DataFrame = mock_hydrography.set_table.call_args.args[0]
    assert mock_hydrography.set_table.call_args.kwargs["name"] == "ar6/sea_level_rise"

    # Annual values on 1 January of 2000-2100; re-index by year to simplify assertions.
    pd.testing.assert_index_equal(
        table.index,
        pd.DatetimeIndex(pd.date_range("2000-01-01", "2100-01-01", freq="YS")),
        exact=False,
        check_names=False,
    )
    table.index = pd.Index(table.index.year)
    return mock_hydrography, table


def test_setup_sea_level_projections_ar6_nearest_location_relative_to_2020(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each station gets the nearest AR6 location, in m relative to 2020, for 2000-2100."""
    ar6: xr.Dataset = make_ar6_dataset(
        lon_deg=[0.0, 90.0, -90.0],
        lat_deg=[0.0, 0.0, 0.0],
        slope_mm_per_year=[10.0, 20.0, 30.0],
        offset_mm=[500.0, 700.0, 900.0],
    )
    mock_hydrography, table = run_setup_sea_level_projections_ar6(
        ar6,
        station_lon_deg=[1.0, 88.0, -91.0],
        station_lat_deg=[1.0, -2.0, 0.5],
        station_ids=[7, 3, 11],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        scenario=370,
        percentile=0.83,
    )

    mock_hydrography.data_catalog.fetch.assert_called_once_with("ar6")
    mock_hydrography.data_catalog.fetch.return_value.read.assert_called_once_with(
        370, 0.83
    )

    assert list(table.index) == list(range(2000, 2101))
    assert list(table.columns) == [7, 3, 11]
    assert not table.isna().any().any()

    # Data is linear in time, so linear interpolation and extrapolation (both before
    # 2020 and, here, within the range) must be exact. Offsets cancel out.
    years: np.ndarray = table.index.to_numpy()
    for station_id, slope_mm_per_year in zip([7, 3, 11], [10.0, 20.0, 30.0]):
        expected_m: np.ndarray = slope_mm_per_year * (years - 2020) / 1000.0
        np.testing.assert_allclose(table[station_id].to_numpy(), expected_m, atol=1e-12)

    assert (table.loc[2020] == 0.0).all()


def test_setup_sea_level_projections_ar6_interpolates_nonlinear_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Values between AR6 time steps are linearly interpolated and 2020 is the reference."""
    years: np.ndarray = np.array([2020, 2050, 2100])
    ar6: xr.Dataset = xr.Dataset(
        {"sea_level_change": (("locations", "years"), [[100.0, 400.0, 1400.0]])},
        coords={
            "years": years,
            "lon": ("locations", [10.0]),
            "lat": ("locations", [10.0]),
        },
    )
    _, table = run_setup_sea_level_projections_ar6(
        ar6, [10.0], [10.0], [0], tmp_path, monkeypatch
    )

    assert table.loc[2020, 0] == pytest.approx(0.0)
    assert table.loc[2035, 0] == pytest.approx(0.15)
    assert table.loc[2050, 0] == pytest.approx(0.3)
    assert table.loc[2075, 0] == pytest.approx(0.8)
    assert table.loc[2100, 0] == pytest.approx(1.3)
    # Linear extrapolation backwards using the first segment (300 mm / 30 years)
    assert table.loc[2000, 0] == pytest.approx(-0.2)


def test_setup_sea_level_projections_ar6_extrapolates_beyond_last_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Years beyond the last AR6 year are linearly extrapolated."""
    years: np.ndarray = np.array([2020, 2040, 2060])
    ar6: xr.Dataset = xr.Dataset(
        {"sea_level_change": (("locations", "years"), [[0.0, 100.0, 300.0]])},
        coords={
            "years": years,
            "lon": ("locations", [0.0]),
            "lat": ("locations", [0.0]),
        },
    )
    _, table = run_setup_sea_level_projections_ar6(
        ar6, [0.0], [0.0], [0], tmp_path, monkeypatch
    )

    # Last segment slope is 200 mm / 20 years = 10 mm/year
    assert table.loc[2060, 0] == pytest.approx(0.3)
    assert table.loc[2100, 0] == pytest.approx(0.7)


def test_setup_sea_level_projections_ar6_uses_great_circle_distance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nearest location is found across the antimeridian and near the poles."""
    ar6: xr.Dataset = make_ar6_dataset(
        # Station at 179.9 E is 0.2 degrees from -179.9 but 9.9 degrees from 170.
        lon_deg=[170.0, -179.9, 0.0, 120.0],
        lat_deg=[0.0, 0.0, 89.0, 88.0],
        slope_mm_per_year=[1.0, 2.0, 3.0, 4.0],
        offset_mm=[0.0, 0.0, 0.0, 0.0],
    )
    _, table = run_setup_sea_level_projections_ar6(
        ar6,
        # Second station: near the pole, 89N/0E is on the sphere much closer than
        # 88N/120E, although 0 and 180 degrees longitude are far apart.
        station_lon_deg=[179.9, 180.0],
        station_lat_deg=[0.0, 89.5],
        station_ids=[0, 1],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert table.loc[2030, 0] == pytest.approx(2.0 * 10 / 1000.0)
    assert table.loc[2030, 1] == pytest.approx(3.0 * 10 / 1000.0)


def test_setup_sea_level_projections_ar6_stations_can_share_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple stations mapping to one AR6 location get identical series in station order."""
    ar6: xr.Dataset = make_ar6_dataset(
        lon_deg=[0.0, 50.0],
        lat_deg=[0.0, 0.0],
        slope_mm_per_year=[5.0, 15.0],
        offset_mm=[0.0, 0.0],
    )
    _, table = run_setup_sea_level_projections_ar6(
        ar6,
        station_lon_deg=[49.0, 1.0, 51.0, 2.0],
        station_lat_deg=[0.0, 0.0, 0.0, 0.0],
        station_ids=[40, 10, 30, 20],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert list(table.columns) == [40, 10, 30, 20]
    np.testing.assert_allclose(table[40], table[30])
    np.testing.assert_allclose(table[10], table[20])
    assert table.loc[2100, 40] == pytest.approx(15.0 * 80 / 1000.0)
    assert table.loc[2100, 10] == pytest.approx(5.0 * 80 / 1000.0)


def test_setup_sea_level_projections_ar6_ignores_invalid_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AR6 locations with NaN coordinates are never selected."""
    ar6: xr.Dataset = make_ar6_dataset(
        # The NaN-coordinate locations would otherwise be the closest or poison the tree.
        lon_deg=[np.nan, 0.0, 30.0, 0.0],
        lat_deg=[0.0, np.nan, 0.0, np.nan],
        slope_mm_per_year=[100.0, 200.0, 7.0, 300.0],
        offset_mm=[0.0, 0.0, 0.0, 0.0],
    )
    _, table = run_setup_sea_level_projections_ar6(
        ar6, [0.0], [0.0], [0], tmp_path, monkeypatch
    )

    assert not table.isna().any().any()
    assert table.loc[2100, 0] == pytest.approx(7.0 * 80 / 1000.0)


def test_setup_sea_level_projections_ar6_handles_transposed_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Variable stored as (years, locations) gives the same result as (locations, years)."""
    kwargs: dict[str, list[float]] = {
        "lon_deg": [0.0, 40.0],
        "lat_deg": [0.0, 0.0],
        "slope_mm_per_year": [2.0, 9.0],
        "offset_mm": [10.0, 20.0],
    }
    station_args: tuple[list[float], list[float], list[int]] = (
        [41.0, 1.0],
        [0.0, 0.0],
        [1, 2],
    )
    _, table_regular = run_setup_sea_level_projections_ar6(
        make_ar6_dataset(**kwargs), *station_args, tmp_path / "a", monkeypatch
    )
    _, table_transposed = run_setup_sea_level_projections_ar6(
        make_ar6_dataset(**kwargs, transpose=True),
        *station_args,
        tmp_path / "b",
        monkeypatch,
    )

    pd.testing.assert_frame_equal(table_regular, table_transposed)


def test_setup_sea_level_projections_ar6_raises_without_valid_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ValueError is raised if no AR6 location has finite coordinates."""
    ar6: xr.Dataset = make_ar6_dataset(
        lon_deg=[np.nan, 10.0],
        lat_deg=[0.0, np.nan],
        slope_mm_per_year=[1.0, 1.0],
        offset_mm=[0.0, 0.0],
    )
    with pytest.raises(ValueError, match="valid coordinates"):
        run_setup_sea_level_projections_ar6(
            ar6, [0.0], [0.0], [0], tmp_path, monkeypatch
        )


@pytest.mark.parametrize(
    "years",
    [
        np.array([2030, 2040, 2050]),  # 2020 missing
        np.array([2020, 2020, 2030]),  # 2020 duplicated
    ],
)
def test_setup_sea_level_projections_ar6_raises_for_invalid_reference_year(
    years: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ValueError is raised unless 2020 occurs exactly once in the AR6 years."""
    ar6: xr.Dataset = make_ar6_dataset(
        lon_deg=[0.0],
        lat_deg=[0.0],
        slope_mm_per_year=[1.0],
        offset_mm=[0.0],
        years=years,
    )
    with pytest.raises(ValueError, match="2020"):
        run_setup_sea_level_projections_ar6(
            ar6, [0.0], [0.0], [0], tmp_path, monkeypatch
        )
