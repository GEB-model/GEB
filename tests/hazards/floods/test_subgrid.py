"""Tests for the parallel replacements of the hydromt-sfincs subgrid functions."""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import pytest
import xarray as xr
from affine import Affine
from hydromt.gis.raster_utils import full_from_transform
from hydromt_sfincs import SfincsModel
from hydromt_sfincs.components.grid.subgrid import process_tile_regular
from hydromt_sfincs.utils import find_uv_indices
from hydromt_sfincs.workflows.merge import merge_multi_dataarrays
from shapely.geometry import LineString, Point, box

from geb.hazards.floods.workflows.subgrid import (
    HUTHRESH,
    RiverBurner,
    _fill_nodata,
    _merge_rasters,
    _prepare_merge_rasters,
    _process_block_tables,
    build_subgrid,
    read_subgrid_bed_elevation,
    sfincs_uv_indices,
    subgrid_water_volume,
)

LOGGER: logging.Logger = logging.getLogger(__name__)

CRS: int = 32631
X0: float = 500000.0
Y0: float = 5000000.0
CELL_SIZE: float = 20.0
N_ROWS: int = 30
N_COLS: int = 40
NR_SUBGRID_PIXELS: int = 4


def create_model(root: Path, n_rows: int = N_ROWS, n_cols: int = N_COLS) -> SfincsModel:
    """Creates a small SFINCS model with an irregular active area.

    Args:
        root: Model root directory.
        n_rows: Number of grid rows.
        n_cols: Number of grid columns.

    Returns:
        The model, with grid and mask.
    """
    sf = SfincsModel(root=str(root), mode="w+")
    sf.grid.create(
        x0=X0,
        y0=Y0 - n_rows * CELL_SIZE,  # southern edge
        dx=CELL_SIZE,
        dy=CELL_SIZE,
        mmax=n_cols,
        nmax=n_rows,
        rotation=0.0,
        epsg=CRS,
    )
    # two overlapping boxes, scaled with the grid size
    sx: float = n_cols / N_COLS
    sy: float = n_rows / N_ROWS
    active_area = gpd.GeoDataFrame(
        geometry=[
            box(X0 + 30 * sx, Y0 - 500 * sy, X0 + 700 * sx, Y0 - 50 * sy).union(
                box(X0 + 300 * sx, Y0 - 590 * sy, X0 + 790 * sx, Y0 - 300 * sy)
            )
        ],
        crs=CRS,
    )
    sf.mask.create_active(include_polygon=active_area, reset_mask=True, fill_area=0.0)
    return sf


def create_dem(
    n_rows: int = N_ROWS, n_cols: int = N_COLS
) -> tuple[xr.DataArray, xr.DataArray]:
    """Creates an elevation and roughness raster aligned with the subgrid pixels.

    The rasters extend two cells beyond the grid (to the east and north), so that the subgrid blocks (with
    one cell of overlap) are fully covered.

    Args:
        n_rows: Number of grid rows.
        n_cols: Number of grid columns.

    Returns:
        Elevation (m) and Manning's n (s/m^(1/3)).
    """
    pixel_size: float = CELL_SIZE / NR_SUBGRID_PIXELS
    margin: int = 2 * NR_SUBGRID_PIXELS
    x: np.ndarray = X0 + (np.arange(n_cols * NR_SUBGRID_PIXELS + margin) + 0.5) * (
        pixel_size
    )
    # south-up, as the hydromt-sfincs regular grid
    y: np.ndarray = (Y0 - n_rows * CELL_SIZE) + (
        np.arange(n_rows * NR_SUBGRID_PIXELS + margin) + 0.5
    ) * pixel_size
    xx, yy = np.meshgrid(x - X0, Y0 - y)
    rng = np.random.default_rng(42)
    elevation: np.ndarray = (
        12.0 - 0.01 * xx + 2.0 * np.sin(yy / 60.0) + rng.normal(0, 0.3, size=xx.shape)
    ).astype(np.float32)
    elevation_da = xr.DataArray(
        elevation, coords={"y": y, "x": x}, dims=("y", "x"), name="elevation"
    ).rio.write_crs(CRS)
    manning_da = (0.03 + 0.02 * (elevation_da > 10.0)).astype(np.float32)
    manning_da.name = "manning"
    return elevation_da, manning_da.rio.write_crs(CRS)


def create_rivers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Creates a river crossing the grid and river bed points along it.

    Returns:
        River center line (width, Manning's n) and bed points (bed_elevation_m,
            depth_m).
    """
    river = gpd.GeoDataFrame(
        {"width": [15.0], "manning": [0.025], "shape_exponent": [0.5]},
        geometry=[LineString([(X0 + 10, Y0 - 280), (X0 + 790, Y0 - 300)])],
        crs=CRS,
    )
    distances: np.ndarray = np.linspace(0, river.geometry.iloc[0].length, 12)
    points = gpd.GeoDataFrame(
        {"bed_elevation_m": 6.0 - 0.004 * distances, "depth_m": 1.5},
        geometry=[river.geometry.iloc[0].interpolate(d) for d in distances],
        crs=CRS,
    )
    return river, points


def elevation_list(elevation: xr.DataArray) -> list[dict]:
    """Elevation datasets in the form used by GEB.

    Args:
        elevation: Elevation raster.

    Returns:
        List with one elevation dataset.
    """
    return [
        {
            "elevation": elevation.to_dataset(name="elevation"),
            "da": elevation,
            "reproj_method": "bilinear",
        }
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sfincs_uv_indices_match_hydromt(seed: int) -> None:
    """The vectorized indices equal those of hydromt-sfincs.

    Args:
        seed: Random seed for the mask.
    """
    mask_values: np.ndarray = (
        np.random.default_rng(seed).random((17, 23)) > 0.4
    ).astype(np.uint8)
    mask_values[0, :] = 1  # an active edge row
    mask = xr.DataArray(mask_values, dims=("y", "x"))
    for expected, result in zip(
        find_uv_indices(mask), sfincs_uv_indices(mask_values), strict=True
    ):
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("is_geographic", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_process_block_tables_matches_hydromt(seed: int, is_geographic: bool) -> None:
    """The subgrid tables of one block equal those of hydromt-sfincs' kernel.

    Compared with `process_tile_regular` with GEB's options (q_table_option 2,
    weight_option "min", no maximum gradient), on random elevation and roughness
    with a flat area and pixels below the lowest table level (-20 m).

    Args:
        seed: Random seed.
        is_geographic: Whether the grid is geographic (pixel size in degrees).
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols, refi, nr_levels = 12, 9, 4, 10
    mask: np.ndarray = (rng.random((n_rows, n_cols)) > 0.3).astype(np.uint8)
    # one extra cell in both directions, for the u/v points of the last cells
    shape: tuple[int, int] = ((n_rows + 1) * refi, (n_cols + 1) * refi)
    elevation: np.ndarray = (5.0 + rng.normal(0.0, 2.0, shape)).astype(np.float32)
    elevation[: 3 * refi, : 3 * refi] = 2.0  # flat
    elevation[-2 * refi :, :] -= 30.0  # below the lowest table level
    manning: np.ndarray = rng.uniform(0.02, 0.1, shape)
    pixel_size: float = 0.0001 if is_geographic else 5.0
    y: np.ndarray = (50.0 if is_geographic else 5e6) + np.arange(shape[0]) * pixel_size

    tables = _process_block_tables(
        mask,
        elevation,
        manning,
        pixel_size,
        pixel_size,
        refi,
        nr_levels,
        y,
        is_geographic,
    )
    expected = process_tile_regular(
        mask,
        elevation,
        manning,
        pixel_size,
        pixel_size,
        refi,
        nr_levels,
        np.repeat(y[:, None], shape[1], axis=1),
        99999.0,  # max_gradient: none
        HUTHRESH,
        2,  # q_table_option
        "min",  # weight_option
        is_geographic,
    )
    assert len(tables) == len(expected)
    for table, expected_table in zip(tables, expected, strict=True):
        np.testing.assert_array_equal(table, expected_table)


def compare_with_hydromt(
    tmp_path: Path,
    elevations: list[dict],
    manning: xr.DataArray,
    n_rows: int,
    n_cols: int,
    hydromt_block_cells: int,
    block_size_pixels: int,
) -> None:
    """Asserts that `build_subgrid` gives the tables of hydromt-sfincs.

    Args:
        tmp_path: Temporary directory.
        elevations: Elevation datasets, as for `SfincsSubgridTable.create`.
        manning: Manning's n raster.
        n_rows: Number of grid rows.
        n_cols: Number of grid columns.
        hydromt_block_cells: Maximum number of cells along each side of a
            hydromt-sfincs block (`nrmax`).
        block_size_pixels: Block size of `build_subgrid` in subgrid pixels.
    """
    hydromt_model: SfincsModel = create_model(tmp_path / "hydromt", n_rows, n_cols)
    with np.errstate(invalid="ignore"):
        hydromt_model.subgrid.create(
            elevation_list=elevations,
            roughness_list=[{"manning": manning.to_dataset(name="manning")}],
            write_dep_tif=True,
            nr_subgrid_pixels=NR_SUBGRID_PIXELS,
            nr_levels=10,
            nrmax=hydromt_block_cells,
        )
    hydromt_model.subgrid.write_netcdf(tmp_path / "hydromt" / "sfincs_subgrid.nc")

    geb_model: SfincsModel = create_model(tmp_path / "geb", n_rows, n_cols)
    result = build_subgrid(
        sfincs_model=geb_model,
        elevation_list=elevations,
        roughness_list=[{"manning": manning.to_dataset(name="manning")}],
        river_burner=None,
        nr_subgrid_pixels=NR_SUBGRID_PIXELS,
        nr_levels=10,
        logger=LOGGER,
        block_size_pixels=block_size_pixels,
        n_workers=4,
    )

    with (
        xr.open_dataset(tmp_path / "hydromt" / "sfincs_subgrid.nc") as expected,
        xr.open_dataset(tmp_path / "geb" / "sfincs_subgrid.nc") as tables,
    ):
        assert dict(tables.sizes) == dict(expected.sizes)
        assert list(tables.data_vars) == list(expected.data_vars)
        for name in expected.data_vars:
            assert tables[name].dims == expected[name].dims
            np.testing.assert_allclose(
                tables[name].values, expected[name].values, rtol=1e-6, atol=1e-5
            )

    with (
        xr.open_dataarray(tmp_path / "hydromt" / "subgrid" / "dep_subgrid.tif") as a,
        xr.open_dataarray(tmp_path / "geb" / "subgrid" / "dep_subgrid.tif") as b,
    ):
        np.testing.assert_allclose(b.values, a.values, atol=1e-5)

    # the bed elevation returned and read back equals hydromt's z_zmin
    np.testing.assert_allclose(
        result.bed_elevation.values,
        hydromt_model.subgrid.data["z_zmin"].values,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        read_subgrid_bed_elevation(
            tmp_path / "geb" / "sfincs_subgrid.nc", geb_model.grid.mask
        ).values,
        result.bed_elevation.values,
    )
    assert geb_model.config.get("sbgfile") == "sfincs_subgrid.nc"
    assert result.river_initial_water_level.isnull().all()


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "hydromt_block_cells", "block_size_pixels"),
    [
        # one block in both
        (N_ROWS, N_COLS, 2000, 4000),
        # several blocks in both, with different block boundaries (GEB: 64 x 64
        # cells, hydromt-sfincs: 37 x 37 cells)
        (150, 140, 37, 256),
    ],
    ids=["one_block", "multiple_blocks"],
)
def test_build_subgrid_matches_hydromt(
    tmp_path: Path,
    n_rows: int,
    n_cols: int,
    hydromt_block_cells: int,
    block_size_pixels: int,
) -> None:
    """Without rivers, the tables equal those of hydromt-sfincs.

    Args:
        tmp_path: Temporary directory.
        n_rows: Number of grid rows.
        n_cols: Number of grid columns.
        hydromt_block_cells: Maximum block size of hydromt-sfincs in cells.
        block_size_pixels: Block size of `build_subgrid` in subgrid pixels.
    """
    elevation, manning = create_dem(n_rows, n_cols)
    compare_with_hydromt(
        tmp_path,
        elevation_list(elevation),
        manning,
        n_rows,
        n_cols,
        hydromt_block_cells,
        block_size_pixels,
    )


def test_build_subgrid_matches_hydromt_two_elevation_datasets(
    tmp_path: Path,
) -> None:
    """With a partial fine DEM and a coarse DEM behind it, the tables equal hydromt's.

    The fine DEM (at subgrid resolution) covers the western part of the grid and has
    a gap; the coarse DEM (at grid resolution, bilinearly resampled) fills the rest.
    The grid has several blocks, so the merge is also tested across block edges.

    Args:
        tmp_path: Temporary directory.
    """
    n_rows, n_cols = 150, 140
    elevation, manning = create_dem(n_rows, n_cols)

    fine: xr.DataArray = elevation.isel(x=slice(0, 300)).copy()
    fine[200:260, 100:180] = np.nan  # a gap, filled from the coarse DEM
    fine = fine.rio.write_nodata(np.nan)

    # the coarse DEM is smoother and 1 m higher, so that its pixels are recognizable
    coarse: xr.DataArray = (
        elevation.coarsen(x=NR_SUBGRID_PIXELS, y=NR_SUBGRID_PIXELS, boundary="trim")
        .mean()
        .astype(np.float32)
        + 1.0
    )
    coarse = coarse.rio.write_crs(CRS).rio.write_nodata(np.nan)
    coarse.name = "elevation"

    elevations: list[dict] = [
        {
            "elevation": dem.to_dataset(name="elevation"),
            "da": dem,
            "reproj_method": "bilinear",
        }
        for dem in (fine, coarse)
    ]
    compare_with_hydromt(
        tmp_path,
        elevations,
        manning,
        n_rows,
        n_cols,
        hydromt_block_cells=37,
        block_size_pixels=256,
    )


def test_river_burner_does_not_depend_on_blocks() -> None:
    """Burning a raster at once or in two parts gives the same result."""
    elevation, manning = create_dem()
    river, points = create_rivers()
    burner = RiverBurner(
        river,
        points,
        grid_crs=CRS,
        utm_crs=CRS,
        min_width_m=5.0,
    )
    transform: Affine = elevation.rio.transform()

    whole_elevation: np.ndarray = elevation.values.copy()
    whole_manning: np.ndarray = manning.values.astype(np.float64)
    rows, cols, unburned = burner.burn(whole_elevation, whole_manning, transform)
    assert rows.size > 0
    np.testing.assert_array_equal(unburned, elevation.values[rows, cols])
    assert (whole_elevation[rows, cols] < unburned).all()
    # the parabolic channel lies between the center-line bed (interpolated along the
    # river) and the bank (bed + depth)
    burned: np.ndarray = whole_elevation[rows, cols]
    assert burned.min() >= points["bed_elevation_m"].min() - 0.05
    assert burned.min() < points["bed_elevation_m"].min() + points["depth_m"].max()
    assert np.isclose(whole_manning[rows, cols], 0.025).all()

    parts_elevation: np.ndarray = elevation.values.copy()
    parts_manning: np.ndarray = manning.values.astype(np.float64)
    split: int = 61  # not a multiple of anything in particular
    for column_slice in (slice(0, split), slice(split, None)):
        part_elevation: np.ndarray = parts_elevation[:, column_slice]
        part_manning: np.ndarray = parts_manning[:, column_slice]
        part_transform: Affine = transform * Affine.translation(column_slice.start, 0)
        burner.burn(part_elevation, part_manning, part_transform)
    np.testing.assert_array_equal(parts_elevation, whole_elevation)
    np.testing.assert_array_equal(parts_manning, whole_manning)


def test_river_burner_outside_rivers() -> None:
    """A block without rivers is left unchanged."""
    elevation, manning = create_dem()
    river, points = create_rivers()
    burner = RiverBurner(
        river,
        points,
        grid_crs=CRS,
        utm_crs=CRS,
        min_width_m=5.0,
    )
    block: np.ndarray = elevation.values[:20, :20].copy()
    rows, _, _ = burner.burn(
        block, manning.values[:20, :20].astype(np.float64), elevation.rio.transform()
    )
    assert rows.size == 0
    np.testing.assert_array_equal(block, elevation.values[:20, :20])


def test_build_subgrid_river_initial_water_level(tmp_path: Path) -> None:
    """Active cells with burned pixels get the lowest unburned elevation of those.

    Args:
        tmp_path: Temporary directory.
    """
    elevation, manning = create_dem()
    river, points = create_rivers()
    sf: SfincsModel = create_model(tmp_path)
    burner = RiverBurner(
        river,
        points,
        grid_crs=CRS,
        utm_crs=CRS,
        min_width_m=5.0,
    )
    result = build_subgrid(
        sfincs_model=sf,
        elevation_list=elevation_list(elevation),
        roughness_list=[{"manning": manning.to_dataset(name="manning")}],
        river_burner=burner,
        nr_subgrid_pixels=NR_SUBGRID_PIXELS,
        nr_levels=10,
        logger=LOGGER,
        n_workers=2,
    )

    # expected: burn the whole raster and take the per-cell minimum
    burned: np.ndarray = elevation.values.copy()
    rows, cols, unburned = burner.burn(
        burned, manning.values.astype(np.float64), elevation.rio.transform()
    )
    inside: np.ndarray = (rows < N_ROWS * NR_SUBGRID_PIXELS) & (
        cols < N_COLS * NR_SUBGRID_PIXELS
    )
    expected: np.ndarray = np.full((N_ROWS, N_COLS), np.inf)
    np.minimum.at(
        expected,
        (rows[inside] // NR_SUBGRID_PIXELS, cols[inside] // NR_SUBGRID_PIXELS),
        unburned[inside],
    )
    expected[~np.isfinite(expected) | (sf.grid.mask.values != 1)] = np.nan
    assert np.isfinite(expected).any()
    np.testing.assert_allclose(
        result.river_initial_water_level.values, expected, rtol=1e-6
    )

    # the burned river is the lowest part of the subgrid elevation
    with xr.open_dataarray(tmp_path / "subgrid" / "dep_subgrid.tif") as dep:
        river_pixel = Point(X0 + 400, Y0 - 290)
        value = dep.sel(x=river_pixel.x, y=river_pixel.y, method="nearest").item()
        assert value < 6.0


def test_subgrid_water_volume(tmp_path: Path) -> None:
    """The water volume follows the subgrid volume tables.

    Args:
        tmp_path: Temporary directory.
    """
    elevation, manning = create_dem()
    sf: SfincsModel = create_model(tmp_path)
    build_subgrid(
        sfincs_model=sf,
        elevation_list=elevation_list(elevation),
        roughness_list=[{"manning": manning.to_dataset(name="manning")}],
        river_burner=None,
        nr_subgrid_pixels=NR_SUBGRID_PIXELS,
        nr_levels=10,
        logger=LOGGER,
        n_workers=2,
    )
    mask: xr.DataArray = sf.grid.mask
    tables_path: Path = tmp_path / "sfincs_subgrid.nc"
    cell_area = xr.full_like(mask, CELL_SIZE**2, dtype=np.float64)
    index_nm, _, _ = sfincs_uv_indices(mask.values)
    with xr.open_dataset(tables_path) as ds:
        z_level: np.ndarray = ds["z_level"].transpose("np", "levels").values
        volume_max: np.ndarray = ds["z_volmax"].values
    row, col = 10, 20
    assert mask.values[row, col] > 0
    cell: int = int(index_nm[col * N_ROWS + row])

    def volume_at(level: float) -> float:
        """Volume with a water level in one cell only.

        Args:
            level: Water level in the cell (m).

        Returns:
            Volume (m3).
        """
        water_level = xr.full_like(mask, np.nan, dtype=np.float64)
        water_level.values[row, col] = level
        return subgrid_water_volume(tables_path, mask, water_level, cell_area)

    # exact at the table levels, each holding an equal share of the maximum volume
    for k in (0, 3, 9):
        np.testing.assert_allclose(
            volume_at(z_level[cell, k]), k / 9 * volume_max[cell], rtol=1e-5
        )
    assert volume_at(z_level[cell, 0] - 1.0) == 0.0
    np.testing.assert_allclose(
        volume_at(z_level[cell, -1] + 0.5),
        volume_max[cell] + 0.5 * CELL_SIZE**2,
        rtol=1e-5,
    )

    # close to the volume of the subgrid pixels
    level: float = float(np.mean(z_level[cell, [4, 5]]))
    with xr.open_dataarray(tmp_path / "subgrid" / "dep_subgrid.tif") as dep:
        pixels: np.ndarray = dep.values.squeeze()[
            row * NR_SUBGRID_PIXELS : (row + 1) * NR_SUBGRID_PIXELS,
            col * NR_SUBGRID_PIXELS : (col + 1) * NR_SUBGRID_PIXELS,
        ]
    pixel_volume: float = float(
        np.maximum(level - pixels, 0.0).sum() * (CELL_SIZE / NR_SUBGRID_PIXELS) ** 2
    )
    np.testing.assert_allclose(volume_at(level), pixel_volume, rtol=0.05)


@pytest.mark.parametrize(
    ("cross_section_exponent", "expected_area_m2"),
    [(0.25, 96.0), (0.5, 80.0), (1.0, 60.0)],
)
def test_river_burner_cross_section(
    cross_section_exponent: float, expected_area_m2: float
) -> None:
    """The burned cross-section has the area of the channel shape: W * depth / (r + 1).

    Args:
        cross_section_exponent: Power-law exponent of the channel shape.
        expected_area_m2: Expected cross-section area (m2) for a 40 m wide, 3 m deep
            channel.
    """
    width, depth, bank = 40.0, 3.0, 12.0
    river = gpd.GeoDataFrame(
        dict(width=[width], manning=[0.03], shape_exponent=[cross_section_exponent]),
        geometry=[LineString([(X0, Y0 - 100), (X0 + 400, Y0 - 100)])],
        crs=CRS,
    )
    distances: np.ndarray = np.linspace(0, 400, 9)
    points = gpd.GeoDataFrame(
        dict(
            bed_elevation_m=np.full(distances.size, bank - depth),
            depth_m=np.full(distances.size, depth),
        ),
        geometry=[river.geometry.iloc[0].interpolate(d) for d in distances],
        crs=CRS,
    )
    burner = RiverBurner(
        river,
        points,
        grid_crs=CRS,
        utm_crs=CRS,
        min_width_m=1.0,
    )
    # 0.5 m pixels, north-up, covering 200 m across the river
    pixel = 0.5
    elevation: np.ndarray = np.full((400, 600), bank, dtype=np.float32)
    transform: Affine = Affine(pixel, 0, X0 + 50, 0, -pixel, Y0)
    burner.burn(elevation, np.full(elevation.shape, 0.04), transform)
    area_per_column: np.ndarray = (bank - elevation).sum(axis=0) * pixel
    np.testing.assert_allclose(area_per_column.mean(), expected_area_m2, rtol=0.03)


def _raster(
    values: np.ndarray, transform: Affine, crs: int, nodata: float
) -> xr.DataArray:
    """Creates a DataArray with CRS and no-data value (rioxarray attributes).

    Args:
        values: Values (y, x).
        transform: Transform of the values.
        crs: EPSG code.
        nodata: No-data value.

    Returns:
        The raster.
    """
    rows, cols = values.shape
    da = xr.DataArray(
        values,
        coords=dict(
            y=transform.f + (np.arange(rows) + 0.5) * transform.e,
            x=transform.c + (np.arange(cols) + 0.5) * transform.a,
        ),
        dims=("y", "x"),
    )
    return da.rio.write_crs(crs).rio.write_nodata(nodata, encoded=False)


@pytest.mark.parametrize("merge_method", ["first", "last", "mean", "max", "min"])
def test_merge_rasters_matches_hydromt(merge_method: str) -> None:
    """The numpy merge of a block equals hydromt-sfincs' merge_multi_dataarrays.

    A coarse DEM with a gap and a finer DEM in another CRS with an offset and a
    valid minimum, merged on a south-up grid as in the subgrid build.

    Args:
        merge_method: Merge method of the second DEM.
    """
    rng = np.random.default_rng(5)
    coarse = rng.uniform(5.0, 15.0, (60, 60)).astype(np.float32)
    coarse[20:30, 10:25] = -9999.0
    coarse_da = _raster(
        coarse, Affine(30.0, 0, 500000.0, 0, -30.0, 5001800.0), 32631, -9999.0
    )
    fine = rng.uniform(-2.0, 12.0, (300, 300)).astype(np.float32)
    fine[100:140, :] = np.nan  # a gap band across the block
    fine_da = _raster(fine, Affine(5.0, 0, 28500.0, 0, -5.0, 5019000.0), 32632, np.nan)
    datasets = [
        dict(da=coarse_da, reproj_method="bilinear"),
        dict(da=fine_da, merge_method=merge_method, offset=-0.2, zmin=0.0),
    ]
    # south-up block grid of 2 m pixels inside both DEMs
    transform = Affine(2.0, 0, 500300.0, 0, 2.0, 5000600.0)
    shape = (300, 400)
    like = full_from_transform(
        transform, shape, nodata=np.nan, dtype=np.float32, crs=32631
    )
    expected = merge_multi_dataarrays(da_list=datasets, da_like=like).values
    merged = _merge_rasters(
        _prepare_merge_rasters(datasets), transform, shape, pyproj.CRS(32631)
    )
    assert np.isfinite(expected).sum() > 1000
    # the second DEM is used: the result differs from the first DEM alone
    first_only = merge_multi_dataarrays(da_list=datasets[:1], da_like=like).values
    assert not np.allclose(expected, first_only, equal_nan=True)
    assert np.isnan(expected).any()
    np.testing.assert_allclose(merged, expected, atol=1e-5)
    np.testing.assert_array_equal(np.isnan(merged), np.isnan(expected))

    expected_filled = (
        _raster(expected, transform, 32631, np.nan)
        .raster.interpolate_na(method="rio_idw", extrapolate=True)
        .values
    )
    np.testing.assert_allclose(
        _fill_nodata(merged, transform), expected_filled, atol=1e-5
    )
