"""Tests for hydrology evaluation metric calculations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geb.evaluate.hydrology import (
    DEFAULT_EXTERNAL_EVALUATION_FOLDER,
    _calculate_discharge_validation_metrics,
    _get_effective_external_evaluation_folder,
    create_validation_df,
)


def test_discharge_metrics_use_squared_pearson_correlation_for_r2() -> None:
    """R2 is squared Pearson correlation; COD is represented by NSE."""
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [1.0, 2.0, 3.0],
            "discharge_simulations": [2.0, 3.0, 4.0],
        }
    )

    metrics = _calculate_discharge_validation_metrics(validation_df)

    assert metrics.KGE_correlation == pytest.approx(1.0)
    assert metrics.R2 == pytest.approx(1.0)
    assert metrics.NSE == pytest.approx(-0.5)


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

    expected_rrmse: float = float(metrics.RMSE / observed_discharge_m3_per_s.std())
    assert metrics.RRMSE == pytest.approx(expected_rrmse)


def test_external_folder_default_is_only_used_for_external_plots() -> None:
    """Default external data folder is derived only when comparisons are enabled."""
    disabled_folder: str | Path | None = _get_effective_external_evaluation_folder(
        external_evaluation_folder=None,
        configured_external_evaluation_folder=None,
        include_external=False,
    )
    enabled_folder: str | Path | None = _get_effective_external_evaluation_folder(
        external_evaluation_folder=None,
        configured_external_evaluation_folder=None,
        include_external=True,
    )
    custom_folder: str | Path | None = _get_effective_external_evaluation_folder(
        external_evaluation_folder=Path("custom_external"),
        configured_external_evaluation_folder=None,
        include_external=True,
    )

    assert disabled_folder is None
    assert enabled_folder == DEFAULT_EXTERNAL_EVALUATION_FOLDER
    assert custom_folder == Path("custom_external")


def _write_hourly_discharge_report(
    output_folder: Path,
    station_id: int,
    simulated_discharge: pd.Series,
) -> None:
    """Write a station discharge report in the evaluation input format.

    Args:
        output_folder: Temporary model output folder.
        station_id: Discharge station identifier.
        simulated_discharge: Hourly simulated discharge (m³/s).
    """
    routing_folder: Path = output_folder / "report" / "hydrology.routing"
    routing_folder.mkdir(parents=True)
    column_name: str = f"discharge_hourly_m3_per_s_{station_id}"
    simulated_discharge.rename(column_name).to_frame().to_parquet(
        routing_folder / f"{column_name}.parquet"
    )


@pytest.mark.parametrize(
    ("timezone_utc_offset", "expected_start", "expected_end"),
    [
        (3.0, "1999-12-31 21:00", "2000-01-01 20:00"),
        (0.0, "2000-01-01 00:00", "2000-01-01 23:00"),
        (-4.0, "2000-01-01 04:00", "2000-01-02 03:00"),
    ],
)
def test_daily_grdc_alignment_uses_fixed_local_calendar_days(
    tmp_path: Path,
    timezone_utc_offset: float,
    expected_start: str,
    expected_end: str,
) -> None:
    """Daily GEB means use the fixed UTC offset supplied by GRDC."""
    station_id: int = 1
    simulation_index: pd.DatetimeIndex = pd.date_range(
        "1999-12-31 00:00",
        "2000-01-03 23:00",
        freq="h",
    )
    simulated_discharge: pd.Series = pd.Series(
        np.arange(len(simulation_index), dtype=float),
        index=simulation_index,
    )
    _write_hourly_discharge_report(tmp_path, station_id, simulated_discharge)
    observed_discharge: pd.Series = pd.Series(
        [1.0],
        index=pd.date_range("2000-01-01", periods=1, freq="D"),
    )

    validation_df: pd.DataFrame = create_validation_df(
        output_folder=tmp_path,
        run_name="default",
        station_id=station_id,
        observed_discharge=observed_discharge,
        correct_discharge_observations=False,
        discharge_observations_to_GEB_upstream_area_ratio=1.0,
        timezone_utc_offset=timezone_utc_offset,
    )

    expected_mean: float = float(
        simulated_discharge.loc[expected_start:expected_end].mean()
    )
    assert validation_df.loc[
        pd.Timestamp("2000-01-01"), "discharge_simulations"
    ] == pytest.approx(expected_mean)


def test_hourly_observations_are_not_shifted_by_grdc_daily_offset(
    tmp_path: Path,
) -> None:
    """Sub-daily observations retain their timestamps despite GRDC metadata."""
    station_id: int = 1
    simulation_index: pd.DatetimeIndex = pd.date_range(
        "2000-01-01 00:00",
        periods=4,
        freq="h",
    )
    simulated_discharge: pd.Series = pd.Series(
        [10.0, 20.0, 30.0, 40.0],
        index=simulation_index,
    )
    _write_hourly_discharge_report(tmp_path, station_id, simulated_discharge)
    observed_discharge: pd.Series = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=simulation_index,
    )

    validation_df: pd.DataFrame = create_validation_df(
        output_folder=tmp_path,
        run_name="default",
        station_id=station_id,
        observed_discharge=observed_discharge,
        correct_discharge_observations=False,
        discharge_observations_to_GEB_upstream_area_ratio=1.0,
        timezone_utc_offset=3.0,
    )

    pd.testing.assert_series_equal(
        validation_df["discharge_simulations"],
        simulated_discharge.rename("discharge_simulations"),
        check_freq=True,
    )


@pytest.mark.parametrize("timezone_utc_offset", [float("nan"), -13.0, 15.0])
def test_discharge_alignment_rejects_invalid_utc_offsets(
    tmp_path: Path,
    timezone_utc_offset: float,
) -> None:
    """Invalid fixed UTC offsets fail with an explicit error."""
    station_id: int = 1
    simulation_index: pd.DatetimeIndex = pd.date_range(
        "2000-01-01",
        periods=48,
        freq="h",
    )
    _write_hourly_discharge_report(
        tmp_path,
        station_id,
        pd.Series(np.ones(48), index=simulation_index),
    )
    observed_discharge: pd.Series = pd.Series(
        [1.0, 2.0],
        index=pd.date_range("2000-01-01", periods=2, freq="D"),
    )

    with pytest.raises(ValueError, match="UTC offset"):
        create_validation_df(
            output_folder=tmp_path,
            run_name="default",
            station_id=station_id,
            observed_discharge=observed_discharge,
            correct_discharge_observations=False,
            discharge_observations_to_GEB_upstream_area_ratio=1.0,
            timezone_utc_offset=timezone_utc_offset,
        )
