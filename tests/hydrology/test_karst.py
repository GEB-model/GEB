"""Tests for conservative karst recharge and its hydrology integration."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from geb.config_schema import KarstConfig
from geb.hydrology import Hydrology
from geb.hydrology.HRUs import GridVariables
from geb.store import Store


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_capture_fraction(fraction: float) -> None:
    """Reject capture settings outside the allowed range.

    Args:
        fraction: Invalid dimensionless capture fraction.
    """
    with pytest.raises(ValueError):
        KarstConfig(capture_fraction=fraction)


@pytest.mark.parametrize("coverage", [None, np.nan, -0.1, 1.1])
def test_enabled_karst_requires_valid_map(
    monkeypatch: pytest.MonkeyPatch, coverage: float | None
) -> None:
    """Reject missing or invalid karst maps when the feature is enabled.

    Args:
        monkeypatch: Fixture bypassing unrelated data and module initialization.
        coverage: Invalid mapped coverage, or None for a missing map.
    """
    monkeypatch.setattr("geb.hydrology.Data.__init__", MagicMock(return_value=None))
    monkeypatch.setattr("geb.hydrology.Module.__init__", MagicMock(return_value=None))
    hydrology: Hydrology = Hydrology.__new__(Hydrology)
    hydrology.model = MagicMock()
    hydrology.model.simulate_hydrology = True
    hydrology.model.config = {"hydrology": {"karst": {"enabled": True}}}
    hydrology.model.files = {
        "grid": {} if coverage is None else {"groundwater/karst_fraction": "karst.zarr"}
    }
    hydrology.grid = MagicMock()
    hydrology.grid.load2d.return_value = np.array([coverage], dtype=np.float32)
    with pytest.raises(ValueError, match="Karst"):
        hydrology.__init__(hydrology.model)


@pytest.mark.parametrize(
    "enabled,coverage,capture_fraction",
    [(False, 0.6, 0.5), (True, 0, 0.5), (True, 0.6, 0), (True, 0.6, 0.5), (True, 1, 1)],
)
@pytest.mark.parametrize("release_time_days", [0.01, 2.0, 10.0, 1000.0])
def test_hydrology_karst_integration(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    coverage: float,
    capture_fraction: float,
    tmp_path: Path,
    release_time_days: float,
) -> None:
    """Send captured runoff to MODFLOW and conserve water at zero and full capture.

    Args:
        monkeypatch: Fixture replacing unrelated whole-model balance checks.
        enabled: Whether the karst diversion is enabled.
        coverage: Karst area fraction of the grid cell (0–1).
        capture_fraction: Fraction of runoff captured on karst land (0–1).
        tmp_path: Temporary checkpoint directory.
        release_time_days: Time controlling release of stored water (days).
    """
    monkeypatch.setattr("geb.hydrology.balance_check", MagicMock())
    hydrology: Hydrology = Hydrology.__new__(Hydrology)
    hydrology.model = MagicMock()
    hydrology.model.config = {
        "hydrology": {
            "karst": {
                "enabled": enabled,
                "capture_fraction": capture_fraction,
                "release_time_days": release_time_days,
            }
        },
        "hazards": {"floods": {"simulate": False}},
    }
    hydrology.model.timing = False
    hydrology.model.reporter.variables_to_report = {
        "hydrology": {
            "karst_recharge": {"varname": ".karst_recharge_m"},
            "karst_capture": {"varname": ".karst_capture_m"},
        }
    }
    hydrology.model.agents.reservoir_operators.command_area_release_m3 = np.zeros(1)
    hydrology.HRU = MagicMock()
    hydrology.HRU.var.land_use_type = np.arange(6)
    hydrology.HRU.var.cell_area = np.ones(6)
    hydrology.grid = MagicMock()
    hydrology.model.hydrology = hydrology
    store: Store = Store(hydrology.model)
    hydrology.grid.var = cast(GridVariables, store.create_bucket("hydrology.grid.var"))
    hydrology.grid.var.cell_area = np.array([6.0], dtype=np.float32)
    hydrology.grid.var.capillar = np.zeros(1, dtype=np.float32)
    hydrology.grid.var.karst_storage_m = np.zeros(1, dtype=np.float64)
    hydrology.grid.compressed_size = 1
    hydrology.karst_fraction = (
        np.array([coverage], dtype=np.float32) if enabled else None
    )
    hydrology.to_HRU = MagicMock(return_value=np.full(6, coverage, dtype=np.float32))
    hydrology.to_grid = MagicMock(
        side_effect=lambda *, HRU_data: HRU_data.mean(axis=-1, keepdims=True)
    )
    hydrology.get_landsurface_storage_m3 = MagicMock(return_value=np.float64(0))
    hydrology.get_overland_flow_buffer_storage_m3 = MagicMock(
        return_value=np.float64(0)
    )
    hydrology.get_routing_storage_m3 = MagicMock(return_value=np.float64(0))
    hydrology.get_waterbodies_storage_m3 = MagicMock(return_value=np.float64(0))
    hydrology.get_groundwater_storage_m3 = MagicMock(return_value=np.float64(0))
    hydrology.waterbodies = MagicMock()
    hydrology.landsurface = MagicMock()
    hydrology.landsurface.step.return_value = (
        np.zeros((24, 1), dtype=np.float32),
        np.full((24, 6), 0.0005, dtype=np.float32),
        np.full((24, 6), 0.001, dtype=np.float32),
        np.full(6, 0.002, dtype=np.float32),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        0.0,
        np.zeros(6),
        np.zeros(6),
        np.float64(0),
    )
    hydrology.groundwater = MagicMock()
    hydrology.groundwater.step.return_value = np.zeros(1, dtype=np.float32)
    hydrology.groundwater.boundary_inflow_m3 = np.float64(0)
    hydrology.groundwater.boundary_outflow_m3 = np.float64(0)
    hydrology.runoff_concentrator = MagicMock()
    hydrology.runoff_concentrator.step.return_value = np.zeros(
        (24, 1), dtype=np.float32
    )
    hydrology.routing = MagicMock()
    hydrology.routing.step.return_value = (0.0, 0.0, 0.0)
    hydrology.hillslope_erosion = MagicMock()
    hydrology.report = MagicMock()

    hydrology.step()

    recharge: np.ndarray = hydrology.groundwater.step.call_args.args[0]
    runoff: np.ndarray = hydrology.runoff_concentrator.step.call_args.kwargs["runoff_m"]
    interflow: np.ndarray = hydrology.runoff_concentrator.step.call_args.kwargs[
        "interflow_m"
    ]
    np.testing.assert_allclose(
        recharge
        + runoff.sum(axis=0)
        + interflow.sum(axis=0)
        + hydrology.grid.var.karst_storage_m,
        [0.038],
        rtol=1e-6,
    )
    if enabled:
        expected_capture_m: float = 0.036 * coverage * capture_fraction
        expected_release_m: float = (
            expected_capture_m
            * 4
            / 6
            * (1.0 - release_time_days * -np.expm1(-1.0 / release_time_days))
        )
        np.testing.assert_allclose(recharge, [0.002 + expected_release_m], rtol=1e-6)
        captured: np.ndarray = hydrology.report.call_args.args[0]["karst_capture_m"]
        np.testing.assert_allclose(captured[:4], expected_capture_m, rtol=1e-6)
        np.testing.assert_array_equal(captured[4:], 0)
        np.testing.assert_allclose(
            hydrology.get_current_storage(),
            (expected_capture_m * 4 / 6 - expected_release_m) * 6,
            rtol=1e-6,
        )

        # Dry days must keep releasing stored water, even with capture set to zero.
        hydrology.model.config["hydrology"]["karst"]["capture_fraction"] = 0.0
        for day in range(1, 31):
            hydrology.step()
            np.testing.assert_allclose(
                hydrology.grid.var.karst_storage_m,
                np.array([expected_capture_m * 4 / 6 - expected_release_m])
                * np.exp(-day / release_time_days),
                rtol=1e-6,
                atol=1e-15,
            )
            if day == 15:
                saved_storage_m: np.ndarray = hydrology.grid.var.karst_storage_m.copy()
                store.save(tmp_path / "checkpoint", metadata={})
                store.load(tmp_path / "checkpoint")
                np.testing.assert_array_equal(
                    hydrology.grid.var.karst_storage_m, saved_storage_m
                )
    else:
        np.testing.assert_allclose(recharge, [0.002], rtol=1e-6)
        np.testing.assert_array_equal(runoff, np.full((24, 1), 0.001, dtype=np.float32))
        hydrology.to_HRU.assert_not_called()
        np.testing.assert_array_equal(
            hydrology.report.call_args.args[0]["karst_recharge_m"], 0
        )
        np.testing.assert_array_equal(
            hydrology.report.call_args.args[0]["karst_capture_m"], 0
        )


@pytest.mark.parametrize("days", [0.0, -1.0, np.nan, np.inf])
def test_invalid_release_time(days: float) -> None:
    """Reject invalid release times.

    Args:
        days: Invalid release time (days).

    """
    with pytest.raises(ValueError):
        KarstConfig(release_time_days=days)


def test_missing_karst_storage_requires_spinup() -> None:
    """Reject an old checkpoint rather than silently starting with an empty store."""
    hydrology: Hydrology = Hydrology.__new__(Hydrology)
    hydrology.karst_fraction = np.ones(1, dtype=np.float32)
    hydrology.grid = MagicMock()
    hydrology.grid.var = SimpleNamespace()
    with pytest.raises(ValueError, match="Rerun spinup"):
        hydrology.step()
