"""Test karst routing, water conservation and saved storage."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from geb.hydrology import Hydrology
from geb.hydrology.HRUs import GridVariables
from geb.store import Store


@pytest.mark.parametrize("coverage", [0.0, 0.6, 1.0])
def test_hydrology_karst_integration(
    monkeypatch: pytest.MonkeyPatch,
    coverage: float,
    tmp_path: Path,
) -> None:
    """Delay karst recharge and conserve water at zero and full coverage.

    Args:
        monkeypatch: Fixture replacing unrelated whole-model balance checks.
        coverage: Karst area fraction of the grid cell (0–1).
        tmp_path: Temporary checkpoint directory.

    Returns:
        None.

    Raises:
        AssertionError: If routing, storage or checkpoint conservation is incorrect.
    """  # noqa: DOC202, DOC502
    capture_fraction: float = 0.5
    release_time_days: float = 10.0
    soil_recharge_m: float = 0.002
    monkeypatch.setattr("geb.hydrology.balance_check", MagicMock())
    hydrology: Hydrology = Hydrology.__new__(Hydrology)
    hydrology.model = MagicMock()
    hydrology.model.config = {
        "hydrology": {
            "karst": {
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
            "karst_soil_recharge": {"varname": ".karst_soil_recharge_m"},
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
    hydrology.karst_fraction = np.array([coverage], dtype=np.float32)
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
        np.full(6, soil_recharge_m, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
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

    assert all(
        call.kwargs["HRU_data"].dtype == np.float32
        for call in hydrology.to_grid.call_args_list
    )
    recharge: np.ndarray = hydrology.groundwater.step.call_args.kwargs[
        "groundwater_recharge_m"
    ]
    runoff: np.ndarray = hydrology.runoff_concentrator.step.call_args.kwargs["runoff_m"]
    interflow: np.ndarray = hydrology.runoff_concentrator.step.call_args.kwargs[
        "interflow_m"
    ]
    np.testing.assert_allclose(
        recharge
        + runoff.sum(axis=0)
        + interflow.sum(axis=0)
        + hydrology.grid.var.karst_storage_m,
        [0.036 + soil_recharge_m],
        rtol=1e-6,
    )
    expected_capture_m: float = 0.036 * coverage * capture_fraction
    expected_soil_capture_m: float = soil_recharge_m * coverage
    expected_inflow_m: float = (expected_capture_m + expected_soil_capture_m) * 4 / 6
    expected_release_m: float = expected_inflow_m * (
        1.0 - release_time_days * -np.expm1(-1.0 / release_time_days)
    )
    np.testing.assert_allclose(
        recharge,
        [soil_recharge_m - expected_soil_capture_m * 4 / 6 + expected_release_m],
        rtol=1e-6,
    )
    captured: np.ndarray = hydrology.report.call_args.args[0]["karst_capture_m"]
    np.testing.assert_allclose(captured[:4], expected_capture_m, rtol=1e-6)
    np.testing.assert_array_equal(captured[4:], 0)
    soil_captured: np.ndarray = hydrology.report.call_args.args[0][
        "karst_soil_recharge_m"
    ]
    np.testing.assert_allclose(soil_captured[:4], expected_soil_capture_m)
    np.testing.assert_array_equal(soil_captured[4:], 0)
    np.testing.assert_allclose(
        hydrology.get_current_storage(),
        (expected_inflow_m - expected_release_m) * 6,
        rtol=1e-6,
    )

    # Dry days must keep releasing stored water, even with capture set to zero.
    hydrology.model.config["hydrology"]["karst"]["capture_fraction"] = 0.0
    dry_inputs: list[np.ndarray | float] = list(hydrology.landsurface.step.return_value)
    dry_inputs[3] = np.zeros(6, dtype=np.float32)
    hydrology.landsurface.step.return_value = tuple(dry_inputs)
    for day in range(1, 31):
        hydrology.step()
        np.testing.assert_allclose(
            hydrology.grid.var.karst_storage_m,
            np.array([expected_inflow_m - expected_release_m])
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
