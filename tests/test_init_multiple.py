"""Tests for init-multiple path handling."""

from pathlib import Path

import geopandas as gpd
import networkx
import pytest
from shapely.geometry import box

from geb.runner import init_multiple_fn


def test_init_multiple_rejects_directory_path(tmp_path: Path) -> None:
    """Test that init-multiple requires a model set name, not a path."""
    (tmp_path / "models").mkdir()

    with pytest.raises(ValueError, match="must be a name"):
        init_multiple_fn(
            working_directory=tmp_path,
            from_example="geul",
            geometry_bounds="0,0,1,1",
            target_area_km2=1.0,
            cluster_prefix="cluster",
            init_multiple_dir="large_scale/europe",
        )


def test_init_multiple_requires_models_directory(tmp_path: Path) -> None:
    """Test that init-multiple requires a models directory."""
    with pytest.raises(FileNotFoundError, match="Models directory not found"):
        init_multiple_fn(
            working_directory=tmp_path,
            from_example="geul",
            geometry_bounds="0,0,1,1",
            target_area_km2=1.0,
            cluster_prefix="cluster",
            init_multiple_dir="europe",
        )


def test_init_multiple_creates_model_set_under_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that init-multiple writes outputs under the working-directory models folder."""
    (tmp_path / "models").mkdir()
    created_directories: list[Path] = []

    monkeypatch.setattr("geb.runner.DataCatalog", lambda logger: object())
    monkeypatch.setattr(
        "geb.runner.get_river_graph", lambda data_catalog: networkx.DiGraph()
    )
    monkeypatch.setattr(
        "geb.runner.get_init_multiple_region_geometry",
        lambda geometry_bounds, region_shapefile, working_directory: gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
        ),
    )
    monkeypatch.setattr(
        "geb.runner.get_all_downstream_subbasins_in_geom",
        lambda data_catalog, region_geometry, ocean_outlets_only, logger: [1],
    )
    monkeypatch.setattr(
        "geb.runner.cluster_subbasins_following_coastline",
        lambda *args, **kwargs: [[1]],
    )
    monkeypatch.setattr(
        "geb.runner.remove_init_multiple_excluded_outlets",
        lambda data_catalog, clusters: (clusters, []),
    )
    monkeypatch.setattr(
        "geb.runner.create_cluster_outline_geodataframe",
        lambda **kwargs: gpd.GeoDataFrame(
            {"cluster_number": [0], "total_basin_area_km2": [1.0]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        ),
    )
    monkeypatch.setattr(
        "geb.runner.create_multi_basin_configs",
        lambda working_directory, **kwargs: created_directories.append(
            working_directory
        ),
    )
    monkeypatch.setattr("geb.runner.save_clusters_to_geoparquet", lambda **kwargs: None)
    monkeypatch.setattr(
        "geb.runner.save_clusters_as_merged_geometries", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "geb.runner.create_cluster_visualization_map", lambda **kwargs: None
    )

    init_multiple_fn(
        working_directory=tmp_path,
        from_example="geul",
        geometry_bounds="0,0,1,1",
        target_area_km2=1.0,
        cluster_prefix="cluster",
        init_multiple_dir="europe",
    )

    assert created_directories == [tmp_path / "models" / "europe"]
    assert (tmp_path / "models" / "europe").is_dir()
