"""Tests for hydrology evaluation metric calculations."""

import numpy as np
import pandas as pd
import pytest

from geb.evaluate.hydrology import (
    _calculate_discharge_validation_metrics,
    _recreate_output_folder,
    _resolve_report_folder,
)


def test_discharge_metrics_use_coefficient_of_determination_for_r2() -> None:
    """R2 is coefficient of determination, not squared Pearson correlation."""
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [1.0, 2.0, 3.0],
            "discharge_simulations": [2.0, 3.0, 4.0],
        }
    )

    metrics = _calculate_discharge_validation_metrics(validation_df)

    assert metrics.KGE_correlation == pytest.approx(1.0)
    assert metrics.R2 == pytest.approx(-0.5)


def test_discharge_metrics_include_kge_components() -> None:
    """KGE decomposition stores original KGE rho, beta, alpha, and modified KGE."""
    observed_discharge_m3_per_s: np.ndarray = np.array([1.0, 2.0, 3.0])
    simulated_discharge_m3_per_s: np.ndarray = np.array([2.0, 3.0, 4.0])
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": observed_discharge_m3_per_s,
            "discharge_simulations": simulated_discharge_m3_per_s,
        }
    )

    metrics = _calculate_discharge_validation_metrics(validation_df)

    expected_bias_ratio: float = float(
        simulated_discharge_m3_per_s.mean() / observed_discharge_m3_per_s.mean()
    )
    expected_variability_ratio: float = float(
        simulated_discharge_m3_per_s.std() / observed_discharge_m3_per_s.std()
    )
    expected_observed_variation: float = float(
        observed_discharge_m3_per_s.std() / observed_discharge_m3_per_s.mean()
    )
    expected_simulated_variation: float = float(
        simulated_discharge_m3_per_s.std() / simulated_discharge_m3_per_s.mean()
    )
    expected_modified_variability_ratio: float = float(
        expected_simulated_variation / expected_observed_variation
    )
    expected_modified_kge: float = float(
        1.0
        - np.sqrt(
            (1.0 - 1.0) ** 2
            + (expected_bias_ratio - 1.0) ** 2
            + (expected_modified_variability_ratio - 1.0) ** 2
        )
    )
    assert metrics.KGE == pytest.approx(0.5)
    assert metrics.KGE_modified == pytest.approx(expected_modified_kge)
    assert metrics.KGE_correlation == pytest.approx(1.0)
    assert metrics.KGE_bias_ratio == pytest.approx(expected_bias_ratio)
    assert metrics.KGE_variability_ratio == pytest.approx(expected_variability_ratio)


def test_rrmse_is_normalized_by_observed_standard_deviation() -> None:
    """RRMSE is RMSE divided by observed discharge standard deviation."""
    observed_discharge_m3_per_s: np.ndarray = np.array([1.0, 2.0, 3.0])
    simulated_discharge_m3_per_s: np.ndarray = np.array([2.0, 3.0, 4.0])
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": observed_discharge_m3_per_s,
            "discharge_simulations": simulated_discharge_m3_per_s,
        }
    )

    metrics = _calculate_discharge_validation_metrics(validation_df)

    expected_rrmse: float = float(
        metrics.RMSE / observed_discharge_m3_per_s.std()
    )
    assert metrics.RRMSE == pytest.approx(expected_rrmse)


def test_recreate_output_folder_replaces_symlink_without_deleting_target(
    tmp_path: Path,
) -> None:
    """Output cleanup unlinks symlinks instead of deleting their target folder."""
    target_folder: Path = tmp_path / "target"
    target_folder.mkdir()
    target_file: Path = target_folder / "keep.txt"
    target_file.write_text("keep me")
    output_folder: Path = tmp_path / "output"
    output_folder.symlink_to(target_folder, target_is_directory=True)

    _recreate_output_folder(output_folder)

    assert output_folder.is_dir()
    assert not output_folder.is_symlink()
    assert target_file.read_text() == "keep me"


def test_resolve_report_folder_supports_merged_output_layout(tmp_path: Path) -> None:
    """Report lookup falls back to output/report/<run_name> for merged models."""
    run_output_folder: Path = tmp_path / "output" / "default"
    merged_report_folder: Path = tmp_path / "output" / "report" / "default"
    merged_report_folder.mkdir(parents=True)

    report_folder: Path = _resolve_report_folder(run_output_folder, "default")

    assert report_folder == merged_report_folder
