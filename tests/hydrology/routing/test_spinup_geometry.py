"""Check that runs retain channel dimensions from weir-free spinup."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geb.hydrology.routing import Routing, RoutingVariables


@pytest.mark.parametrize("invalid_depth", [None, 0.0, np.nan])
def test_restore_spinup_geometry(invalid_depth: float | None) -> None:
    """Restore saved dimensions exactly and reject invalid checkpoint depths.

    Args:
        invalid_depth: Invalid depth (meters), or None to test valid restoration.

    Returns:
        None. Checks saved dimensions are used without fitting run discharge.

    Raises:
        AssertionError: If the run recomputes geometry or accepts invalid depths.
    """  # noqa: DOC202, DOC502
    routing: Routing = Routing.__new__(Routing)
    routing.model = SimpleNamespace(in_spinup=False)  # ty:ignore[invalid-assignment]
    routing.var = RoutingVariables()
    routing.var.river_ids = np.array([7, 7], dtype=np.int32)
    routing.var.river_width = np.array([4, 5], dtype=np.float32)
    routing.var.bankfull_depth_m = np.array([1, 2], dtype=np.float32)
    routing.calculate_bankfull_width = Mock(side_effect=AssertionError("Width refit"))
    routing.calculate_bankfull_depth = Mock(side_effect=AssertionError("Depth refit"))
    if invalid_depth is not None:
        routing.var.bankfull_depth_m[0] = invalid_depth
        with pytest.raises(ValueError, match="Saved spinup"):
            routing.get_bankfull_geometry()
    else:
        width_m: np.ndarray
        depth_m: np.ndarray
        width_m, depth_m = routing.get_bankfull_geometry()
        np.testing.assert_array_equal(width_m, [4, 5])
        np.testing.assert_array_equal(depth_m, [1, 2])
        width_m[0] = 100
        depth_m[0] = 100
        assert routing.var.river_width[0] == 4
        assert routing.var.bankfull_depth_m[0] == 1


@pytest.mark.parametrize("in_spinup", [False, True])
def test_geometry_estimation_for_spinup_and_old_checkpoints(in_spinup: bool) -> None:
    """Derive geometry during spinup and for legacy checkpoints without grid depths.

    Args:
        in_spinup: Whether geometry is being estimated during spinup.

    Returns:
        None. Checks that saved spinup return periods determine dimensions.

    Raises:
        AssertionError: If saved dimensions prevent spinup geometry updates.
    """  # noqa: DOC202, DOC502
    routing: Routing = Routing.__new__(Routing)
    routing.model = SimpleNamespace(in_spinup=in_spinup)  # ty:ignore[invalid-assignment]
    routing.var = RoutingVariables()
    routing.var.rivers = gpd.GeoDataFrame(
        {"represented_in_grid": [True], "return_period_2_years_daily_m3_per_s": [8.0]}
    )
    if in_spinup:
        routing.var.bankfull_depth_m = np.array([100], dtype=np.float32)
    routing.calculate_bankfull_width = Mock(
        return_value=np.array([4], dtype=np.float32)
    )
    routing.calculate_bankfull_depth = Mock(
        return_value=np.array([2], dtype=np.float32)
    )
    width_m: np.ndarray
    depth_m: np.ndarray
    width_m, depth_m = routing.get_bankfull_geometry()
    np.testing.assert_array_equal(width_m, [4])
    np.testing.assert_array_equal(depth_m, [2])
    routing.calculate_bankfull_width.assert_called_once_with(True)


def test_run_return_periods_leave_geometry_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep explicit flood-statistics estimation independent of run discharge.

    Args:
        monkeypatch: Fixture replacing report reads and statistical fitting.

    Returns:
        None. Checks that only spinup reports are read and dimensions stay fixed.

    Raises:
        AssertionError: If run discharge is fitted or channel geometry changes.
    """  # noqa: DOC202, DOC502
    routing: Routing = Routing.__new__(Routing)
    routing.model = SimpleNamespace(  # ty:ignore[invalid-assignment]
        in_spinup=False,
        current_timestep=365,
        report_folder=Path("/tmp/geometry-test/output/default/report"),
        config={"general": {"spinup_name": "spinup"}},
    )
    routing.config = {
        "retention_basins": {"activation_threshold_return_period_years": 2}
    }
    routing.var = RoutingVariables()
    routing.var.rivers = gpd.GeoDataFrame(index=[7])
    routing.var.river_width = np.array([4], dtype=np.float32)
    routing.var.bankfull_depth_m = np.array([2], dtype=np.float32)
    routing.get_active_rivers = Mock(return_value=routing.var.rivers)
    routing.router = Mock()
    routing.retention_basin_data = pd.DataFrame()
    routing.default_minimum_discharge_m3_per_s = 0.001
    discharge: pd.DataFrame = pd.DataFrame(
        {7: np.ones(72)}, index=pd.date_range("1999-01-01", periods=72, freq="h")
    )
    read: Mock = Mock(return_value=discharge)
    monkeypatch.setattr("geb.hydrology.routing.get_discharge_per_river", read)
    fit: Mock = Mock()
    fit.return_value.rl_table = pd.DataFrame({"T_years": [2], "GPD_POT_RL": [1000.0]})
    monkeypatch.setattr("geb.hydrology.routing.ReturnPeriodModel", fit)
    routing.update_return_periods()
    read.assert_called_once()
    assert read.call_args.kwargs["source"] == "file"
    assert read.call_args.kwargs["folder"] == Path(
        "/tmp/geometry-test/output/spinup/report/hydrology.routing"
    )
    np.testing.assert_array_equal(routing.var.river_width, [4])
    np.testing.assert_array_equal(routing.var.bankfull_depth_m, [2])
    routing.router.update_channel_geometry.assert_not_called()


def test_channel_dimensions_survive_checkpoint(tmp_path: Path) -> None:
    """Restore exact grid dimensions from a saved spinup bucket.

    Args:
        tmp_path: Directory receiving a temporary routing checkpoint.

    Returns:
        None. Checks width and depth persistence through a real save and load.

    Raises:
        AssertionError: If the checkpoint loses or changes channel dimensions.
    """  # noqa: DOC202, DOC502
    from concurrent.futures import ThreadPoolExecutor

    saved: RoutingVariables = RoutingVariables()
    saved.river_ids = np.array([7, 7], dtype=np.int32)
    saved.river_width = np.array([4, 5], dtype=np.float32)
    saved.bankfull_depth_m = np.array([1, 2], dtype=np.float32)
    with ThreadPoolExecutor() as executor:
        for future in saved.save(tmp_path, executor):
            future.result()
    routing: Routing = Routing.__new__(Routing)
    routing.model = SimpleNamespace(in_spinup=False)  # ty:ignore[invalid-assignment]
    routing.var = RoutingVariables()
    routing.var.load(tmp_path)
    width_m: np.ndarray
    depth_m: np.ndarray
    width_m, depth_m = routing.get_bankfull_geometry()
    np.testing.assert_array_equal(width_m, saved.river_width)
    np.testing.assert_array_equal(depth_m, saved.bankfull_depth_m)


@pytest.mark.parametrize("active", [False, True])
def test_retention_thresholds_use_saved_spinup_statistics(active: bool) -> None:
    """Restore retention activation without fitting weir-affected run discharge.

    Args:
        active: Whether the retention basin is enabled.

    Returns:
        None. Checks active and disabled thresholds and missing-cell validation.

    Raises:
        AssertionError: If activation differs from the saved spinup return period.
    """  # noqa: DOC202, DOC502
    routing: Routing = Routing.__new__(Routing)
    routing.var = RoutingVariables()
    routing.var.river_ids = np.array([7], dtype=np.int32)
    routing.var.rivers = gpd.GeoDataFrame(
        {"return_period_2_years_daily_m3_per_s": [10.0]}, index=[7]
    )
    routing.config = {
        "retention_basins": {"activation_threshold_return_period_years": 2}
    }
    routing.retention_basin_data = pd.DataFrame(index=[0])
    routing.retention_basin_ids = np.array([0], dtype=np.int32)
    routing.retention_basin_is_active = pd.Series([active])
    routing.retention_activation_threshold_m3_s = np.array([np.inf], dtype=np.float32)
    routing.default_minimum_discharge_m3_per_s = 0.001
    routing.set_retention_activation_thresholds()
    assert routing.retention_activation_threshold_m3_s[0] == (10 if active else np.inf)
    routing.retention_basin_ids[:] = -1
    with pytest.raises(ValueError, match="no associated grid cells"):
        routing.set_retention_activation_thresholds()
