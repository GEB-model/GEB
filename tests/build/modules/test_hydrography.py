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
    _count_flooded_subpixels_per_reach,
    _extract_1sec_subgrid,
    calculate_dem_floodplain_geometry,
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


def test_calculate_dem_floodplain_geometry() -> None:
    """Test calculate_dem_floodplain_geometry extracts floodplain width and side slope per reach."""
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

    # 1 arc-second DEM: 3x3 refinement of 3-arcsecond elevations
    dx1, dy1 = 0.01 / 3.0, -0.01 / 3.0
    x1 = 0.0 + (np.arange(12) - 1) * dx1
    y1 = 0.0 + (np.arange(12) - 1) * dy1
    elev_1sec = xr.DataArray(
        np.repeat(np.repeat(elev_vals, 3, axis=0), 3, axis=1),
        coords={"y": y1, "x": x1},
        dims=("y", "x"),
    )

    fp_slope_m_per_m, fp_profile_w_m = calculate_dem_floodplain_geometry(
        flow_raster_high_res=flw_high_res,
        elevation_high_res=elev_high_res,
        elevation_1sec=elev_1sec,
        upstream_area_high_res_m2=upstream_area_high_res,
        idxs_outflow_low_res=idxs_outflow_low_res,
        river_raster_low_res=river_raster_low_res,
        cell_area_low_res=cell_area_low_res,
        river_length_low_res=river_length_low_res,
        ldd_scale_factor=2,
    )

    assert fp_slope_m_per_m.shape == (2, 2)
    assert np.isnan(fp_slope_m_per_m.values[0, 0])
    assert np.isnan(fp_slope_m_per_m.values[1, 0])
    assert not np.isnan(fp_slope_m_per_m.values[0, 1])
    assert not np.isnan(fp_slope_m_per_m.values[1, 1])
    assert (fp_slope_m_per_m.values[:, 1] >= 0.0).all()
    # When top width is constant across the final interval, side slope is 0.0 (vertical continuation)
    assert np.allclose(fp_slope_m_per_m.values[:, 1], 0.0)

    # 3D 10-point profile assertions
    assert fp_profile_w_m.shape == (10, 2, 2)
    assert np.isnan(fp_profile_w_m.values[:, 0, 0]).all()
    assert np.isnan(fp_profile_w_m.values[:, 1, 0]).all()
    assert not np.isnan(fp_profile_w_m.values[:, 0, 1]).any()
    assert not np.isnan(fp_profile_w_m.values[:, 1, 1]).any()
    assert (fp_profile_w_m.values[:, :, 1] >= 0.0).all()
    for k in range(1, 10):
        assert (
            fp_profile_w_m.values[k, :, 1] >= fp_profile_w_m.values[k - 1, :, 1]
        ).all()


def test_extract_1sec_subgrid() -> None:
    """Test _extract_1sec_subgrid subgrid alignment and error handling."""
    dx3, dy3 = 3.0 / 3600.0, -3.0 / 3600.0
    dx1, dy1 = 1.0 / 3600.0, -1.0 / 3600.0

    x3 = 10.0 + (np.arange(4) + 0.5) * dx3
    y3 = 50.0 + (np.arange(4) + 0.5) * dy3
    da3 = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32), coords={"y": y3, "x": x3}, dims=("y", "x")
    )

    # Larger 1-sec DEM covering a wider region
    x1 = 9.998 + (np.arange(20) + 0.5) * dx1
    y1 = 50.002 + (np.arange(20) + 0.5) * dy1
    da1 = xr.DataArray(
        np.arange(400, dtype=np.float32).reshape(20, 20),
        coords={"y": y1, "x": x1},
        dims=("y", "x"),
    )

    extracted = _extract_1sec_subgrid(da1, da3)
    assert extracted.shape == (12, 12)
    assert not np.isnan(extracted).any()

    # Missing coordinates raises ValueError
    da3_nocoords = xr.DataArray(np.zeros((4, 4), dtype=np.float32))
    da1_nocoords = xr.DataArray(np.ones((12, 12), dtype=np.float32))
    with pytest.raises(ValueError, match="missing 'x' or 'y' coordinates"):
        _extract_1sec_subgrid(da1_nocoords, da3_nocoords)

    # Non-overlapping coordinates raises ValueError
    da1_nonoverlapping = xr.DataArray(
        np.ones((20, 20), dtype=np.float32),
        coords={"y": y1 + 100.0, "x": x1 + 100.0},
        dims=("y", "x"),
    )
    with pytest.raises(ValueError, match="does not overlap"):
        _extract_1sec_subgrid(da1_nonoverlapping, da3)

    # Incompatible single-point coordinates raises ValueError
    da1_single = xr.DataArray(
        np.zeros((1, 1), dtype=np.float32),
        coords={"y": [50.0], "x": [10.0]},
        dims=("y", "x"),
    )
    with pytest.raises(ValueError, match="must have length > 1"):
        _extract_1sec_subgrid(da1_single, da3)


def test_setup_floodplains_method() -> None:
    """Test setup_floodplains build method on Hydrography class."""
    elev_vals = np.array(
        [
            [12.0, 5.0, 5.0, 12.0],
            [11.0, 4.0, 4.0, 11.0],
            [10.0, 3.0, 3.0, 10.0],
            [9.0, 2.0, 2.0, 9.0],
        ],
        dtype=np.float32,
    )
    d8_vals = np.array(
        [
            [4, 4, 4, 16],
            [4, 4, 4, 16],
            [4, 4, 4, 16],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    transform = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 0.0)
    coords = {"y": [0.0, -0.01, -0.02, -0.03], "x": [0.0, 0.01, 0.02, 0.03]}

    elev_high_res = xr.DataArray(elev_vals, coords=coords, dims=("y", "x"))
    elev_high_res.rio.write_transform(transform, inplace=True)

    d8_da = xr.DataArray(d8_vals, coords=coords, dims=("y", "x"))
    d8_da.attrs["_FillValue"] = 255
    d8_da.rio.write_transform(transform, inplace=True)

    import pyflwdir

    flw = pyflwdir.from_array(d8_vals, ftype="d8", transform=transform)
    ups_vals = flw.upstream_area(unit="m2").astype(np.float32)
    ups_da = xr.DataArray(ups_vals, coords=coords, dims=("y", "x"))

    dx1, dy1 = 0.01 / 3.0, -0.01 / 3.0
    x1 = 0.0 + (np.arange(12) - 1) * dx1
    y1 = 0.0 + (np.arange(12) - 1) * dy1
    elev_1sec = xr.DataArray(
        np.repeat(np.repeat(elev_vals, 3, axis=0), 3, axis=1),
        coords={"y": y1, "x": x1},
        dims=("y", "x"),
    )

    cell_area = xr.DataArray(
        np.full((2, 2), 4_000_000.0, dtype=np.float32),
        coords={"y": [-0.005, -0.025], "x": [0.005, 0.025]},
        dims=("y", "x"),
    )
    river_len = xr.DataArray(
        np.full((2, 2), 2000.0, dtype=np.float32),
        coords=cell_area.coords,
        dims=cell_area.dims,
    )
    idxs_outflow = xr.DataArray(
        np.array([[5, 7], [13, 15]], dtype=np.int64),
        coords=cell_area.coords,
        dims=cell_area.dims,
    )
    river_ids = xr.DataArray(
        np.array([[-1, 101], [-1, 102]], dtype=np.int32),
        coords=cell_area.coords,
        dims=cell_area.dims,
    )

    mock_model = MagicMock()
    mock_model.logger = logging.getLogger("test")
    mock_model.ldd_scale_factor = 2
    mock_model.other = {
        "DEM/fathomdem": elev_1sec,
        "drainage/original_d8_flow_directions": d8_da,
        "drainage/original_d8_elevation": elev_high_res,
        "drainage/original_d8_upstream_area_m2": ups_da,
    }
    mock_model.grid = {
        "idxs_outflow": idxs_outflow,
        "routing/river_ids": river_ids,
        "cell_area": cell_area,
        "routing/river_length_m": river_len,
    }

    saved_grids: dict[str, xr.DataArray] = {}
    mock_model.set_grid.side_effect = lambda da, name: saved_grids.setdefault(name, da)

    Hydrography.setup_floodplains(mock_model)

    assert "routing/floodplain_side_slope_m_per_m" in saved_grids
    assert "routing/floodplain_profile_widths_m" in saved_grids
    slope_res = saved_grids["routing/floodplain_side_slope_m_per_m"]
    profile_res = saved_grids["routing/floodplain_profile_widths_m"]
    assert slope_res.shape == (2, 2)
    assert profile_res.shape == (10, 2, 2)

    # Missing DEM raises KeyError
    del mock_model.other["DEM/fathomdem"]
    with pytest.raises(KeyError, match="DEM 'fathomdem' not found"):
        Hydrography.setup_floodplains(mock_model)


AR6_YEARS: np.ndarray = np.arange(2020, 2151, 10)
STATIONS_FILE: str = "geom/gtsm/stations.parquet"


def make_ar6_dataset(
    lon_deg: list[float],
    lat_deg: list[float],
    slope_m_per_year: list[float],
    years: np.ndarray = AR6_YEARS,
    transpose: bool = False,
) -> xr.Dataset:
    """Create a synthetic AR6 dataset with linear sea level change per location.

    The adapter already expresses sea level change in m relative to 2020, so the
    synthetic values are 0 in 2020.

    Args:
        lon_deg: Longitude of each AR6 location (degrees).
        lat_deg: Latitude of each AR6 location (degrees).
        slope_m_per_year: Sea level change rate of each location (m/year).
        years: Years in the dataset.
        transpose: Whether to store the variable as (years, locations).

    Returns:
        Dataset mimicking the structure of the AR6 data read by the data catalog.
    """
    slope: np.ndarray = np.asarray(slope_m_per_year, dtype=float)[:, None]
    values_m: np.ndarray = slope * (years[None, :] - 2020)
    dims: tuple[str, str] = ("locations", "years")
    if transpose:
        values_m = values_m.T
        dims = ("years", "locations")
    return xr.Dataset(
        data_vars={"sea_level_change": (dims, values_m)},
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
    projection_quantile: float = 0.5,
) -> tuple[MagicMock, pd.DataFrame]:
    """Run setup_sea_level_projections on a mocked Hydrography object.

    Args:
        ar6: AR6 dataset returned by the mocked data catalog.
        station_lon_deg: Station longitudes (degrees).
        station_lat_deg: Station latitudes (degrees).
        station_ids: Station index values.
        tmp_path: Temporary directory used as working directory.
        monkeypatch: Pytest monkeypatch fixture.
        projection_quantile: Quantile passed to the method.

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
    mock_hydrography.files = {"geom": {"water_levels/stations": STATIONS_FILE}}
    mock_hydrography.ssp_rcp = 370
    mock_hydrography.data_catalog.fetch.return_value.read.return_value = ar6

    Hydrography.setup_sea_level_projections(
        mock_hydrography, projection_quantile=projection_quantile
    )

    mock_hydrography.set_table.assert_called_once()
    table: pd.DataFrame = mock_hydrography.set_table.call_args.args[0]
    assert (
        mock_hydrography.set_table.call_args.kwargs["name"]
        == "water_levels/sea_level_rise"
    )

    # Annual values on 1 January of 2000-2100; re-index by year to simplify assertions.
    pd.testing.assert_index_equal(
        table.index,
        pd.DatetimeIndex(pd.date_range("2000-01-01", "2100-01-01", freq="YS")),
        exact=False,
        check_names=False,
        check_freq=False,
    )
    assert isinstance(table.index, pd.DatetimeIndex)
    table.index = pd.Index(table.index.year)
    return mock_hydrography, table


def test_setup_sea_level_projections_ar6_nearest_location_relative_to_2020(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each station gets the nearest AR6 location, in m relative to 2020, for 2000-2100."""
    ar6: xr.Dataset = make_ar6_dataset(
        lon_deg=[0.0, 90.0, -90.0],
        lat_deg=[0.0, 0.0, 0.0],
        slope_m_per_year=[0.01, 0.02, 0.03],
    )
    mock_hydrography, table = run_setup_sea_level_projections_ar6(
        ar6,
        station_lon_deg=[1.0, 88.0, -91.0],
        station_lat_deg=[1.0, -2.0, 0.5],
        station_ids=[7, 3, 11],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        projection_quantile=0.83,
    )

    mock_hydrography.data_catalog.fetch.assert_called_once_with("ar6")
    mock_hydrography.data_catalog.fetch.return_value.read.assert_called_once_with(
        370, 0.83
    )

    assert list(table.index) == list(range(2000, 2101))
    assert list(table.columns) == [7, 3, 11]
    assert not table.isna().any().any()

    # Data is linear in time, so linear interpolation and extrapolation (both before
    # 2020 and, here, within the range) must be exact.
    years: np.ndarray = table.index.to_numpy()
    for station_id, slope_m_per_year in zip([7, 3, 11], [0.01, 0.02, 0.03]):
        expected_m: np.ndarray = slope_m_per_year * (years - 2020)
        np.testing.assert_allclose(table[station_id].to_numpy(), expected_m, atol=1e-12)

    assert (table.loc[2020] == 0.0).all()


def test_setup_sea_level_projections_ar6_interpolates_nonlinear_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Values between AR6 time steps are linearly interpolated and 2020 is the reference."""
    years: np.ndarray = np.array([2020, 2050, 2100])
    ar6: xr.Dataset = xr.Dataset(
        {"sea_level_change": (("locations", "years"), [[0.0, 0.3, 1.3]])},
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
        {"sea_level_change": (("locations", "years"), [[0.0, 0.1, 0.3]])},
        coords={
            "years": years,
            "lon": ("locations", [0.0]),
            "lat": ("locations", [0.0]),
        },
    )
    _, table = run_setup_sea_level_projections_ar6(
        ar6, [0.0], [0.0], [0], tmp_path, monkeypatch
    )

    # Last segment slope is 0.2 m / 20 years = 0.01 m/year
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
        slope_m_per_year=[0.001, 0.002, 0.003, 0.004],
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

    assert table.loc[2030, 0] == pytest.approx(0.002 * 10)
    assert table.loc[2030, 1] == pytest.approx(0.003 * 10)


def test_setup_sea_level_projections_ar6_stations_can_share_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple stations mapping to one AR6 location get identical series in station order."""
    ar6: xr.Dataset = make_ar6_dataset(
        lon_deg=[0.0, 50.0],
        lat_deg=[0.0, 0.0],
        slope_m_per_year=[0.005, 0.015],
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
    assert table.loc[2100, 40] == pytest.approx(0.015 * 80)
    assert table.loc[2100, 10] == pytest.approx(0.005 * 80)


def test_setup_sea_level_projections_ar6_handles_transposed_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Variable stored as (years, locations) gives the same result as (locations, years)."""
    lon_deg: list[float] = [0.0, 40.0]
    lat_deg: list[float] = [0.0, 0.0]
    slope_m_per_year: list[float] = [0.002, 0.009]
    station_args: tuple[list[float], list[float], list[int]] = (
        [41.0, 1.0],
        [0.0, 0.0],
        [1, 2],
    )
    _, table_regular = run_setup_sea_level_projections_ar6(
        make_ar6_dataset(
            lon_deg=lon_deg, lat_deg=lat_deg, slope_m_per_year=slope_m_per_year
        ),
        *station_args,
        tmp_path / "a",
        monkeypatch,
    )
    _, table_transposed = run_setup_sea_level_projections_ar6(
        make_ar6_dataset(
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            slope_m_per_year=slope_m_per_year,
            transpose=True,
        ),
        *station_args,
        tmp_path / "b",
        monkeypatch,
    )

    pd.testing.assert_frame_equal(table_regular, table_transposed)


@pytest.mark.parametrize("rows", [2, 257])
def test_floodplain_row_blocks_match_full_arrays(rows: int) -> None:
    """Match the original full-array calculation across block edges and missing data.

    Args:
        rows: Drainage-grid height, including a case crossing the 256-row block edge.
    """
    from geb.build.modules.hydrography import FLOODPLAIN_PROFILE_DEPTHS

    drainage: np.ndarray = np.full((rows, 4), 10.0, dtype=np.float32)
    reach_ids: np.ndarray = (np.arange(rows * 4).reshape(rows, 4) % 4).astype(np.uint32)
    elevation: np.ndarray = (
        np.random.default_rng(42)
        .uniform(9.0, 16.0, size=(3 * rows, 12))
        .astype(np.float32)
    )
    elevation[0, 0] = np.nan
    elevation[0, 3] = np.nan
    original: np.ndarray = elevation.copy()
    height: np.ndarray = np.maximum(
        elevation - np.repeat(np.repeat(drainage, 3, axis=0), 3, axis=1), 0.0
    )
    expanded_ids: np.ndarray = np.repeat(np.repeat(reach_ids, 3, axis=0), 3, axis=1)
    expected: np.ndarray = np.array(
        [
            np.bincount(
                expanded_ids[(expanded_ids > 0) & (height <= depth)], minlength=4
            )[1:]
            for depth in FLOODPLAIN_PROFILE_DEPTHS
        ]
    )
    actual: np.ndarray = _count_flooded_subpixels_per_reach(
        elevation, drainage, reach_ids, 3
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(elevation, original)


def test_floodplain_row_blocks_reject_misaligned_grids() -> None:
    """Reject elevation arrays without the required 3:1 refinement."""
    with pytest.raises(ValueError, match="3:1 refinement"):
        _count_flooded_subpixels_per_reach(
            np.zeros((5, 6), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.uint32),
            1,
        )


def test_extract_dem_subgrid_loads_only_selected_chunks() -> None:
    """Leave DEM chunks outside the selected region unread."""
    import dask.array as da
    from dask import delayed

    def unavailable_block() -> np.ndarray:
        """Represent an out-of-region DEM chunk that must not be read.

        Raises:
            RuntimeError: If the calculation reads outside the selected window.
        """
        raise RuntimeError("Out-of-region DEM chunk was read")

    available: da.Array = da.ones((10, 10), chunks=(10, 10), dtype=np.float32)
    unavailable: da.Array = da.from_delayed(
        delayed(unavailable_block)(), shape=(10, 10), dtype=np.float32
    )
    source: xr.DataArray = xr.DataArray(
        da.block([[available, unavailable], [unavailable, unavailable]]),
        dims=("y", "x"),
        coords={"y": -np.arange(20), "x": np.arange(20)},
    )
    target: xr.DataArray = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [-1, -4], "x": [1, 4]},
    )
    result: np.ndarray = _extract_1sec_subgrid(source, target)
    np.testing.assert_array_equal(result, np.ones((6, 6), dtype=np.float32))
