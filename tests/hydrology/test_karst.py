"""Tests for conservative karst recharge and its hydrology integration."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from geb.config_schema import KarstConfig
from geb.hydrology import Hydrology
from geb.hydrology.HRUs import GridVariables


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
    [
        (False, 0.6, 0.5),
        (True, 0, 0.5),
        (True, 0.6, 0),
        (True, 0.6, 0.5),
        (True, 1, 1),
    ],
)
def test_hydrology_karst_integration(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    coverage: float,
    capture_fraction: float,
) -> None:
    """Recharge captured runoff immediately and conserve water.

    Args:
        monkeypatch: Fixture replacing unrelated whole-model balance checks.
        enabled: Whether the karst diversion is enabled.
        coverage: Karst area fraction of the grid cell (0–1).
        capture_fraction: Fraction of runoff captured on karst land (0–1).
    """
    monkeypatch.setattr("geb.hydrology.balance_check", MagicMock())
    hydrology: Hydrology = Hydrology.__new__(Hydrology)
    hydrology.model = MagicMock()
    hydrology.model.config = {
        "hydrology": {
            "karst": {
                "enabled": enabled,
                "capture_fraction": capture_fraction,
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
    hydrology.grid.var = cast(
        GridVariables,
        SimpleNamespace(
            cell_area=np.array([6.0], dtype=np.float32),
            capillar=np.zeros(1, dtype=np.float32),
        ),
    )
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
        recharge + runoff.sum(axis=0) + interflow.sum(axis=0),
        [0.038],
        rtol=1e-6,
    )
    if enabled:
        expected_capture_m: float = 0.036 * coverage * capture_fraction
        expected_recharge_m: float = 0.002 + expected_capture_m * 4 / 6
        np.testing.assert_allclose(recharge, [expected_recharge_m], rtol=1e-6)
        captured: np.ndarray = hydrology.report.call_args.args[0]["karst_capture_m"]
        np.testing.assert_allclose(captured[:4], expected_capture_m, rtol=1e-6)
        np.testing.assert_array_equal(captured[4:], 0)
        np.testing.assert_allclose(
            hydrology.report.call_args.args[0]["karst_recharge_m"],
            [expected_capture_m * 4 / 6],
            rtol=1e-6,
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


def test_legacy_karst_checkpoint_requires_spinup() -> None:
    """Reject old stored-karst state rather than silently discarding its water."""
    hydrology: Hydrology = Hydrology.__new__(Hydrology)
    hydrology.karst_fraction = np.ones(1, dtype=np.float32)
    hydrology.grid = MagicMock()
    hydrology.grid.var = SimpleNamespace(karst_storage_m=np.ones(1, dtype=np.float64))
    with pytest.raises(ValueError, match="Rerun spinup"):
        hydrology.step()
