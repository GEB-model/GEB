"""Tests for init-multiple configuration generation."""

from pathlib import Path

from geb.build import create_multi_basin_configs
from geb.workflows.io import read_params


def test_create_multi_basin_configs_uses_config_file_names(tmp_path: Path) -> None:
    """Test that init-multiple config file names are propagated to clusters."""
    create_multi_basin_configs(
        clusters=[[101, 102]],
        working_directory=tmp_path,
        config=Path("model.custom.yml"),
        build_config=Path("build.custom.yml"),
        update_config=Path("update.custom.yml"),
        cluster_basin_areas_km2={0: 1234.5},
    )

    cluster_base_directory: Path = tmp_path / "cluster_000" / "base"

    assert (tmp_path / "model.custom.yml").exists()
    assert (tmp_path / "build.custom.yml").exists()
    assert (tmp_path / "update.custom.yml").exists()

    model_config: dict = read_params(cluster_base_directory / "model.yml")
    build_config: dict = read_params(cluster_base_directory / "build.yml")
    update_config: dict = read_params(cluster_base_directory / "update.yml")

    assert model_config["inherits"] == "../../model.custom.yml"
    assert build_config["inherits"] == "../../build.custom.yml"
    assert update_config["inherits"] == "../../update.custom.yml"
    assert model_config["general"]["region"]["subbasin"] == [101, 102]
