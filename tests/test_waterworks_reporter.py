"""Verify hourly waterworks diagnostics use the existing grouped reporter."""

import datetime
import logging
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from geb.build import version_updates
from geb.config_schema import ReportConfig
from geb.reporter import Reporter


def test_waterworks_hourly_export(tmp_path: Path) -> None:
    """Export 24 hourly samples keyed by original grid cells, including fixed crests.

    Args:
        tmp_path: Temporary simulation folder.

    Returns:
        None.

    Raises:
        AssertionError: If sampling columns, times or values are incorrect.
    """  # noqa: DOC202, DOC502
    model: Mock = Mock()
    model.mode = "w"
    model.simulate_hydrology = True
    model.current_time = datetime.datetime(2020, 1, 1)
    model.simulation_end = datetime.datetime(2020, 1, 1)
    model.timestep_length = datetime.timedelta(days=1)
    model.config = {
        "report": {
            "_config": {"compression_level": 1, "chunk_target_size_bytes": 1000000},
            "_waterworks": True,
        }
    }
    routing: Mock = model.hydrology.routing
    routing.router._weir_height_inertial = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    routing.router._inertial_cells = np.array([17, 18, 19], dtype=np.int32)
    reporter: Reporter = Reporter(model, tmp_path / "report", clean=True)
    variables: dict = reporter.variables_to_report["hydrology.routing"]
    samples: dict[str, np.ndarray] = {
        "waterworks_open_fraction": np.column_stack(
            [np.linspace(0.1, 1.0, 24), np.full(24, -1)]
        ).astype(np.float32),
        "waterworks_outflow_m3_s": np.column_stack(
            [np.arange(24), np.arange(24) + 10]
        ).astype(np.float32),
        "waterworks_inflow_m3_s": np.column_stack(
            [np.arange(24) + 1, np.arange(24) + 11]
        ).astype(np.float32),
    }
    reporter.report(routing, samples, "hydrology.routing", variables)
    reporter.finalize()
    quantity: str
    for quantity, values in samples.items():
        report: pd.DataFrame = pd.read_parquet(
            tmp_path / "report/hydrology.routing" / f"{quantity}.parquet"
        )
        assert report.columns.tolist() == ["17", "19"]
        assert len(report) == 24
        assert report.index[0] == pd.Timestamp("2020-01-01 00:30")
        np.testing.assert_allclose(report.to_numpy(), values)
    assert ReportConfig.model_validate({}).waterworks
    assert not ReportConfig.model_validate({"_waterworks": False}).waterworks


def test_waterworks_input_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require exact barrier placement metadata when upgrading existing inputs.

    Args:
        monkeypatch: Fixture controlling the target version.

    Returns:
        None.

    Raises:
        AssertionError: If rebuilding barriers or rerunning simulations is omitted.
    """  # noqa: DOC202, DOC502
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b34")
    updates: list[str] = version_updates.get_and_maybe_do_version_updates(
        "1.0.0b33",
        logging.getLogger(__name__),
    )
    assert any("Re-run `setup_weirs`" in update for update in updates)
    assert any("Rerun spinup and simulation" in update for update in updates)
