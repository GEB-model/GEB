"""Tests for init-multiple configuration generation."""

from pathlib import Path

from geb.build import create_multi_basin_configs
from geb.workflows.io import read_params


def test_create_multi_basin_configs_uses_standard_config_file_names(
    tmp_path: Path,
) -> None:
    """Test that init-multiple creates standard config file names."""
    create_multi_basin_configs(
        clusters=[[101, 102]],
        working_directory=tmp_path,
        cluster_basin_areas_km2={0: 1234.5},
    )

    cluster_base_directory: Path = tmp_path / "cluster_000" / "base"

    assert (tmp_path / "model.yml").exists()
    assert (tmp_path / "build.yml").exists()
    assert (tmp_path / "update.yml").exists()

    model_config: dict = read_params(cluster_base_directory / "model.yml")
    build_config: dict = read_params(cluster_base_directory / "build.yml")
    update_config: dict = read_params(cluster_base_directory / "update.yml")

    assert model_config["inherits"] == "../../model.yml"
    assert build_config["inherits"] == "../../build.yml"
    assert update_config["inherits"] == "../../update.yml"
    assert model_config["general"]["region"]["subbasin"] == [101, 102]
