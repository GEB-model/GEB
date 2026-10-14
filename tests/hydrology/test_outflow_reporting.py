"""Regression tests for discharge reports across changing river representations."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geb.hydrology.routing import get_discharge_per_river

if TYPE_CHECKING:
    from geb.model import GEBModel


@pytest.mark.parametrize("outflow_setting", [False, True, None])
def test_required_outflow_history(
    outflow_setting: bool | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collect required routing history with disabled, enabled, or omitted reports.

    Args:
        outflow_setting: Configured outflow flag, or None to omit the flag.
        tmp_path: Temporary directory for exported discharge history.
        monkeypatch: Fixture for isolating the extreme-value fit.

    Returns:
        None.

    Raises:
        AssertionError: If recording, annual updating, or export fails.
    """  # noqa: DOC202, DOC502
    import datetime
    from typing import cast
    from unittest.mock import MagicMock

    from geb.hydrology.routing import Routing
    from geb.reporter import Reporter

    model: MagicMock = MagicMock()
    report_settings: dict[str, Any] = {
        "_config": {"compression_level": 1, "chunk_target_size_bytes": 1000000}
    }
    if outflow_setting is not None:
        report_settings["_outflow_points"] = outflow_setting
    model.config = {"report": report_settings}
    model.mode = "w"
    model.simulate_hydrology = True
    model.in_spinup = True
    model.current_time = datetime.datetime(1990, 12, 29)
    model.simulation_end = datetime.datetime(1990, 12, 31)
    model.timestep_length = datetime.timedelta(days=1)
    routing: MagicMock = model.hydrology.routing
    # Use the failed river's actual two-cell layout; no topology change is needed.
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"represented_in_grid": [True], "hydrography_xy": [[(327, 54), (328, 53)]]},
        index=[24027753],
    )
    routing.var.rivers = rivers
    routing.get_active_rivers.return_value = rivers
    model.hydrology.grid.compressed_size = 2
    model.hydrology.HRU.compressed_size = 2
    cell_mapping: np.ndarray = np.full((55, 329), -1, dtype=np.int32)
    cell_mapping[54, 327] = 0
    cell_mapping[53, 328] = 1
    model.hydrology.grid.linear_mapping = cell_mapping
    reporter: Reporter = Reporter(
        cast("GEBModel", model), tmp_path / "report", clean=True
    )
    model.reporter = reporter
    routing_reports: dict[str, Any] = reporter.variables_to_report["hydrology.routing"]
    report_name: str = "river_outflow_hourly_m3_per_s_24027753"
    assert report_name in routing_reports
    assert report_settings.get("_outflow_points") is outflow_setting
    assert "_outflow_points" not in reporter.variables_to_report
    model.logger.warning.assert_not_called()

    day: int
    for day in range(3):
        model.current_time = datetime.datetime(1990, 12, 29) + datetime.timedelta(
            days=day
        )
        routing.var.discharge_m3_s_per_substep = np.tile(
            np.array([1.0, 10.0 + day], dtype=np.float32), (24, 1)
        )
        reporter.maybe_report_value(
            "hydrology.routing", report_name, routing, {}, routing_reports[report_name]
        )
    assert routing_reports[report_name]["_var_index"] == 72

    # Exercise the January 1 call that failed, including the geometry update.
    model.current_time = datetime.datetime(1991, 1, 1)
    routing.variables_to_report = routing_reports
    routing.config = {
        "retention_basins": {"activation_threshold_return_period_years": 5}
    }
    routing.default_minimum_discharge_m3_per_s = 0.1
    routing.retention_basin_data = pd.DataFrame()
    routing.calculate_bankfull_width.return_value = np.array(
        [7.0, 7.0], dtype=np.float32
    )
    routing.calculate_bankfull_depth.return_value = np.array(
        [1.0, 1.0], dtype=np.float32
    )
    return_period_fit: MagicMock = MagicMock()
    return_period_fit.return_value.rl_table = pd.DataFrame(
        {"T_years": [2, 5], "GPD_POT_RL": [20.0, 30.0]}
    )
    monkeypatch.setattr("geb.hydrology.routing.ReturnPeriodModel", return_period_fit)
    Routing.update_return_periods(cast(Routing, routing))
    np.testing.assert_array_equal(
        return_period_fit.call_args.kwargs["series"].to_numpy(), [10.0, 11.0, 12.0]
    )
    assert rivers.loc[24027753, "return_period_2_years_daily_m3_per_s"] == 20.0
    routing.router.update_channel_geometry.assert_called_once()

    reporter.finalize()
    exported_history: pd.DataFrame = get_discharge_per_river(
        rivers, rivers, source="file", folder=tmp_path / "report" / "hydrology.routing"
    )
    np.testing.assert_array_equal(
        exported_history[24027753].to_numpy(), np.repeat([10.0, 11.0, 12.0], 24)
    )


def test_no_required_outflow_history_without_hydrology(tmp_path: Path) -> None:
    """Skip required outflow reporting when hydrology is not simulated.

    Args:
        tmp_path: Temporary report directory.

    Returns:
        None.

    Raises:
        AssertionError: If hydrological reporting is enabled unnecessarily.
    """  # noqa: DOC202, DOC502
    from typing import cast
    from unittest.mock import MagicMock

    from geb.reporter import Reporter

    model: MagicMock = MagicMock()
    model.config = {
        "report": {
            "_config": {"compression_level": 1, "chunk_target_size_bytes": 1000000},
            "_outflow_points": False,
        }
    }
    model.mode = "w"
    model.simulate_hydrology = False
    reporter: Reporter = Reporter(
        cast("GEBModel", model), tmp_path / "report", clean=True
    )
    assert "hydrology.routing" not in reporter.variables_to_report
    model.logger.warning.assert_not_called()
