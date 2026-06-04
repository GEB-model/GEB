"""Tests for init-multiple clustering helpers."""

from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import box

from geb.build import (
    create_cluster_outline_geodataframe,
    save_clusters_as_merged_geometries,
)
from geb.runner import _resolve_init_multiple_directory


class _FakeAdapter:
    """Minimal adapter exposing a parquet path for tests."""

    def __init__(self, path: Path) -> None:
        """Initialize the fake adapter.

        Args:
            path: Path to the synthetic GeoParquet file.
        """
        self.path = path


class _FakeDataCatalog:
    """Minimal data catalog for MERIT Basins catchment tests."""

    def __init__(self, catchments_path: Path) -> None:
        """Initialize the fake data catalog.

        Args:
            catchments_path: Path to the synthetic MERIT Basins catchments.
        """
        self.catchments_path = catchments_path

    def fetch(self, name: str) -> _FakeAdapter:
        """Return the fake MERIT Basins catchments adapter.

        Args:
            name: Data catalog entry name.

        Returns:
            Fake adapter with a parquet path.

        Raises:
            ValueError: If a dataset other than MERIT Basins catchments is requested.
        """
        if name != "merit_basins_catchments":
            raise ValueError(f"Unsupported fake dataset: {name}")
        return _FakeAdapter(self.catchments_path)


def test_resolve_init_multiple_directory_uses_parent_models(
    tmp_path: Path,
) -> None:
    """Test that plain init-multiple names can resolve to a parent models folder.

    Args:
        tmp_path: Temporary directory for a synthetic repository layout.
    """
    working_directory: Path = tmp_path / "GEB"
    working_directory.mkdir()
    models_directory: Path = tmp_path / "models"
    models_directory.mkdir()

    resolved_path: Path = _resolve_init_multiple_directory(
        working_directory=working_directory,
        init_multiple_dir="large_scale",
    )

    assert resolved_path == models_directory / "large_scale"


def test_resolve_init_multiple_directory_accepts_explicit_path(
    tmp_path: Path,
) -> None:
    """Test that explicit relative paths do not require a models folder.

    Args:
        tmp_path: Temporary directory for the path resolver.
    """
    working_directory: Path = tmp_path / "checkout"
    working_directory.mkdir()

    resolved_path: Path = _resolve_init_multiple_directory(
        working_directory=working_directory,
        init_multiple_dir="snellius/output/large_scale",
    )

    assert (
        resolved_path == (working_directory / "snellius/output/large_scale").resolve()
    )


def test_cluster_outlines_are_one_exact_geometry_per_cluster(
    tmp_path: Path,
) -> None:
    """Test that upstream catchments are dissolved to one outline row per cluster.

    Args:
        tmp_path: Temporary directory for synthetic GeoParquet files.
    """
    catchments_path: Path = tmp_path / "catchments.geoparquet"
    catchments: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "COMID": [1, 2, 3],
            "geometry": [
                box(0.0, 0.0, 1.0, 1.0),
                box(1.0, 0.0, 2.0, 1.0),
                box(3.0, 0.0, 4.0, 1.0),
            ],
        },
        crs="EPSG:4326",
    )
    catchments.to_parquet(catchments_path)

    river_graph: nx.DiGraph = nx.DiGraph()
    river_graph.add_edges_from([(1, 3), (2, 3)])
    fake_catalog: _FakeDataCatalog = _FakeDataCatalog(catchments_path)

    cluster_outlines: gpd.GeoDataFrame = create_cluster_outline_geodataframe(
        clusters=[[3]],
        data_catalog=fake_catalog,  # ty:ignore[invalid-argument-type]
        river_graph=river_graph,
        cluster_prefix="Europe",
    )

    assert len(cluster_outlines) == 1
    assert cluster_outlines.loc[0, "cluster_id"] == "Europe_000"
    assert cluster_outlines.loc[0, "num_total_subbasins"] == 3
    assert cluster_outlines.geometry.iloc[0].geom_type == "MultiPolygon"

    output_path: Path = tmp_path / "cluster_outlines.geoparquet"
    save_clusters_as_merged_geometries(cluster_outlines, output_path)
    saved_outlines: gpd.GeoDataFrame = gpd.read_parquet(output_path)

    assert len(saved_outlines) == 1
    assert saved_outlines.geometry.iloc[0].equals(cluster_outlines.geometry.iloc[0])
