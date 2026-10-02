"""Weirs hold back water without adding or losing water."""

from pathlib import Path

import numpy as np
import pyflwdir
import pytest

from geb.geb_types import ArrayFloat32, ArrayFloat64
from geb.hydrology.routing.inertial_substeps import (
    GEOM_IN_GATE_CLOSING_COUNT,
    GEOM_IN_GATE_OPEN_SECONDS,
    GEOM_IN_INTERFACE_BED_ELEVATION_MAX,
)
from geb.hydrology.routing.local_inertial import LocalInertial

from .test_routing import _make_local_inertial


def make_router(
    heights: ArrayFloat32 | None,
    gated: bool = False,
    gate_open: np.ndarray | None = None,
    dt_seconds: int = 60,
) -> LocalInertial:
    """Create three river cells with a weir at the end of the first cell.

    Args:
        heights: Weir heights above the river bed (m), or None for no weirs.
        gated: Whether the first weir has a gate.
        gate_open: Saved gate states, or None to start closed.
        dt_seconds: Routing interval (seconds).

    Returns:
        Router with 1 km reaches, 20 m width, and 4 m bankfull depth.
    """
    network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        np.array([[6, 6, 5]], dtype=np.uint8), ftype="ldd"
    )
    return _make_local_inertial(
        dt=dt_seconds,
        river_network=network,
        river_length=np.full(3, 1000, dtype=np.float32),
        river_width=np.full(3, 20, dtype=np.float32),
        bankfull_river_elevation_m=np.array([10, 9, 8], dtype=np.float32),
        bankfull_depth_m=4.0,
        weir_height_m=heights,
        weir_gate=np.array([gated, False, False]),
        gate_open=gate_open,
    )


@pytest.mark.parametrize("depth_m", [1.0, 3.0])
def test_sill_flow_and_mass_balance(depth_m: float) -> None:
    """Stop flow below the weir, allow flow above it, and keep the water balance.

    Args:
        depth_m: Initial upstream water depth (m).
    """
    router: LocalInertial = make_router(np.array([2, 0, 0], dtype=np.float32))
    # Parabolic cross-section volume: L * W / (1.5 * sqrt(bankfull depth)) * h^1.5.
    initial_storage: ArrayFloat64 = (
        1000 * 20 / 3 * np.array([depth_m, 0.2, 0.2], dtype=np.float64) ** 1.5
    )
    result: tuple = router.step(
        Q_prev_m3_s=np.full(3, 5, dtype=np.float32),
        river_storage_m3=initial_storage.copy(),
        sideflow_m3=np.zeros(3, dtype=np.float32),
        evaporation_m3=np.zeros(3, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )
    if depth_m < 2:
        assert result[0][0] == 0
        assert result[1][0] == pytest.approx(initial_storage[0])
    else:
        assert result[0][0] > 0
        assert result[1][0] < initial_storage[0]
    assert np.all(result[1] >= 0)
    assert result[1].sum() + result[6] == pytest.approx(initial_storage.sum(), rel=1e-5)


@pytest.mark.parametrize(
    "heights",
    [[-3, 0, 0], [float("nan"), 0, 0], [0, 0, 1], [1, 0]],
)
def test_invalid_weirs(heights: list[float]) -> None:
    """Reject invalid heights and a weir placed at the domain outlet.

    Args:
        heights: Invalid crest heights (m).
    """
    with pytest.raises(ValueError, match="[Ww]eir"):
        make_router(np.array(heights, dtype=np.float32))


def test_weir_backwater() -> None:
    """Higher water downstream reduces flow over the weir."""
    discharges: list[float] = []
    downstream_depth: float
    for downstream_depth in (0.2, 3.9):
        router: LocalInertial = make_router(np.array([2, 0, 0], dtype=np.float32))
        initial_storage: ArrayFloat64 = (
            1000
            * 20
            / 3
            * np.array([3.0, downstream_depth, 0.2], dtype=np.float64) ** 1.5
        )
        result: tuple = router.step(
            Q_prev_m3_s=np.zeros(3, dtype=np.float32),
            river_storage_m3=initial_storage.copy(),
            sideflow_m3=np.zeros(3, dtype=np.float32),
            evaporation_m3=np.zeros(3, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharges.append(float(result[0][0]))
        assert result[1].sum() + result[6] == pytest.approx(
            initial_storage.sum(), rel=1e-5
        )
    assert 0 < discharges[1] < discharges[0]


def test_optional_weirs_and_width_update() -> None:
    """Zero heights leave the river unchanged; changing width keeps the weir."""
    original: LocalInertial = make_router(None)
    disabled: LocalInertial = make_router(np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(original._geom_inbank, disabled._geom_inbank)
    weir: LocalInertial = make_router(np.array([2, 0, 0], dtype=np.float32))
    # The weir does not change how much water fits in the channel.
    np.testing.assert_array_equal(original._bankfull_volume, weir._bankfull_volume)
    weir.update_channel_width(np.full(3, 25, dtype=np.float32))
    index: int = int(np.flatnonzero(weir._weir_height_inertial > 0)[0])
    assert weir._geom_inbank[index, GEOM_IN_INTERFACE_BED_ELEVATION_MAX] == 12.0


@pytest.mark.parametrize("bankfull_depth", [2.0, 8.0])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    "height_marker",
    [-1.0, -2.0],
)
def test_missing_height_scales_with_depth(
    bankfull_depth: float, gated: bool, height_marker: float
) -> None:
    """Missing heights scale with the channel; known heights stay fixed.

    Args:
        bankfull_depth: River depth at bankfull flow (m).
        gated: Whether the missing-height structure is controlled.
        height_marker: Missing-height marker (-1 or -2).

    Returns:
        None.

    Raises:
        AssertionError: If heights depend on the gate flag.
    """  # noqa: DOC202, DOC502
    network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        np.array([[6, 6, 5]], dtype=np.uint8), ftype="ldd"
    )
    input_heights: ArrayFloat32 = np.array([height_marker, 3, 0], dtype=np.float32)
    router: LocalInertial = _make_local_inertial(
        dt=60,
        river_network=network,
        river_length=np.full(3, 1000, dtype=np.float32),
        river_width=np.full(3, 20, dtype=np.float32),
        bankfull_depth_m=bankfull_depth,
        weir_height_m=input_heights,
        weir_gate=np.array([gated, gated, False]),
    )
    np.testing.assert_array_equal(input_heights, [height_marker, 3, 0])
    expected_height_m: float = (
        bankfull_depth + 1 if height_marker == -1.0 else bankfull_depth / 2
    )
    np.testing.assert_allclose(
        router._weir_height_inertial, np.array([expected_height_m, 3, 0])
    )


def test_gate_levels_and_backwater() -> None:
    """Check threshold boundaries and overflow through a closed barrier."""
    from geb.hydrology.routing.inertial_substeps import (
        closed_gate_overflow,
        update_gate_state,
    )

    assert not update_gate_state(199.99, 200.0, 5.0, False)
    assert update_gate_state(200.0, 200.0, 5.0, False)
    assert update_gate_state(100.0, 200.0, 5.0, True)
    assert not update_gate_state(100.0, 200.0, 5.0, False)
    assert update_gate_state(5.01, 200.0, 5.0, True)
    assert not update_gate_state(5.0, 200.0, 5.0, True)
    assert not update_gate_state(0.0, 200.0, 5.0, False)
    assert closed_gate_overflow(4, 0, 0, 0, 5, 20) == 0
    assert closed_gate_overflow(5, 0, 0, 0, 5, 20) == 0
    flow: float = closed_gate_overflow(6, 0, 0, 0, 5, 20)
    assert flow == pytest.approx(34.0)
    assert 0 < closed_gate_overflow(6, 5.5, 0, 0, 5, 20) < flow
    assert closed_gate_overflow(6, 6, 0, 0, 5, 20) == 0
    assert closed_gate_overflow(6, 7, 0, 0, 5, 20) == 0
    assert closed_gate_overflow(6, 0, 0, 7, 5, 20) == 0


@pytest.mark.parametrize("downstream_depth_m", [0.2, 1.0, 1.5])
def test_open_gate_matches_unobstructed_river(downstream_depth_m: float) -> None:
    """A fully open gate matches river routing, including backwater.

    Args:
        downstream_depth_m: Downstream channel depth (m).

    """
    storage: ArrayFloat64 = (
        1000 * 20 / 3 * np.array([1.6, downstream_depth_m, 0.2]) ** 1.5
    )
    results: list[tuple] = []
    router: LocalInertial
    for router in (
        make_router(None),
        make_router(
            np.array([2, 0, 0], dtype=np.float32),
            gated=True,
            gate_open=np.array([True, False, False]),
        ),
    ):
        results.append(
            router.step(
                Q_prev_m3_s=np.full(3, 5, dtype=np.float32),
                river_storage_m3=storage.copy(),
                sideflow_m3=np.zeros(3, dtype=np.float32),
                evaporation_m3=np.zeros(3, dtype=np.float32),
                waterbody_storage_m3=np.zeros(0, dtype=np.float64),
                outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
                retention_storage_m3=np.zeros(0, dtype=np.float32),
                retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
            )
        )
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert router.gate_open[0]


def test_gate_routing_and_restart() -> None:
    """Opening and closing conserve water, and saved state survives a restart."""
    heights: ArrayFloat32 = np.array([2, 0, 0], dtype=np.float32)
    saved_state: np.ndarray = np.zeros(3, dtype=bool)
    router: LocalInertial = make_router(heights, gated=True, gate_open=saved_state)
    router.update_channel_width(np.full(3, 200, dtype=np.float32))
    depth: float
    expected_open: bool
    for depth, expected_open in (
        (1.4, False),
        (1.9, True),
        (1.6, True),
        (1.5, True),
        (0.1, False),
    ):
        previously_open: bool = bool(saved_state[0])
        storage: ArrayFloat64 = (
            1000 * 200 / 3 * np.array([depth, 0.2, 0.2], dtype=np.float64) ** 1.5
        )
        result: tuple = router.step(
            Q_prev_m3_s=np.full(3, 5, dtype=np.float32),
            river_storage_m3=storage.copy(),
            sideflow_m3=np.zeros(3, dtype=np.float32),
            evaporation_m3=np.zeros(3, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        assert bool(saved_state[0]) == expected_open
        gate_index: int = int(np.flatnonzero(router._gate_height > 0)[0])
        assert router._geom_inbank[
            gate_index, GEOM_IN_GATE_OPEN_SECONDS
        ] == pytest.approx(60.0 if expected_open else 0.0)
        assert router._geom_inbank[gate_index, GEOM_IN_GATE_CLOSING_COUNT] == int(
            previously_open and not expected_open
        )
        assert np.all(
            router._geom_inbank[router._gate_height == 0, GEOM_IN_GATE_OPEN_SECONDS]
            == 0
        )
        assert (result[0][0] > 0) == expected_open
        assert np.all(result[1] >= 0)
        assert result[1].sum() + result[6] == pytest.approx(storage.sum(), rel=1e-5)
        # Rebuild as on a checkpoint restart, including a channel-width refresh.
        router = make_router(
            heights,
            gated=True,
            gate_open=saved_state.copy(),
        )
        saved_state = router.gate_open
        router.update_channel_width(np.full(3, 200, dtype=np.float32))


@pytest.mark.parametrize("gates", [[False, True, False], [True, False]])
def test_invalid_gates(gates: list[bool]) -> None:
    """Reject gates without a weir or with the wrong array size.

    Args:
        gates: Invalid gate flags.
    """
    with pytest.raises(ValueError, match="Gate"):
        _make_local_inertial(
            dt=60,
            river_network=pyflwdir.from_array(
                np.array([[6, 6, 5]], dtype=np.uint8), ftype="ldd"
            ),
            river_length=np.full(3, 1000, dtype=np.float32),
            river_width=np.full(3, 20, dtype=np.float32),
            weir_height_m=np.array([2, 0, 0], dtype=np.float32),
            weir_gate=np.array(gates),
        )


def test_gate_diagnostics_capture_internal_closure() -> None:
    """Capture an opening and closure within one routing interval.

    Returns:
        None.

    Raises:
        AssertionError: If diagnostics miss an internal event or are not reset.
    """  # noqa: DOC202, DOC502
    router: LocalInertial = make_router(
        np.array([2, 0, 0], dtype=np.float32),
        gated=True,
        dt_seconds=86400,
    )
    storage: ArrayFloat64 = 1000 * 20 / 3 * np.array([1.9, 0.2, 0.2]) ** 1.5
    discharge: ArrayFloat32 = np.zeros(3, dtype=np.float32)
    gate_index: int = int(np.flatnonzero(router._gate_height > 0)[0])
    interval: int
    for interval in range(2):
        result: tuple = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=np.zeros(3, dtype=np.float32),
            evaporation_m3=np.zeros(3, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = result[:2]
        assert not router.gate_open[0]
        assert router._geom_inbank[gate_index, GEOM_IN_GATE_CLOSING_COUNT] == (
            1 if interval == 0 else 0
        )
        open_seconds: float = float(
            router._geom_inbank[gate_index, GEOM_IN_GATE_OPEN_SECONDS]
        )
        if interval == 0:
            assert 0 < open_seconds < 86400
        else:
            assert open_seconds == 0


def test_gate_stays_open_with_steady_inflow() -> None:
    """Conserve water when inflow sustains depth above the closing threshold.

    Notes:
        A gate need not cycle: steady inflow can maintain an equilibrium depth
        above the closing depth threshold after the first opening.
    """
    router: LocalInertial = make_stage_gate_router(0.5, 0.05)
    router.update_channel_width(np.full(3, 20, dtype=np.float32))
    storage: ArrayFloat64 = 1000 * 20 / 3 * np.array([0.9, 0.2, 0.2]) ** 1.5
    initial_total: float = float(storage.sum())
    discharge: ArrayFloat32 = np.zeros(3, dtype=np.float32)
    total_outflow: float = 0.0
    states: list[bool] = []
    result: tuple
    for _ in range(500):
        result = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=np.array([60, 0, 0], dtype=np.float32),
            evaporation_m3=np.zeros(3, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = result[:2]
        total_outflow += float(result[6])
        states.append(bool(router.gate_open[0]))
        assert np.all(storage >= 0)
    assert np.count_nonzero(np.diff(states)) == 1
    assert states[-1]
    assert storage.sum() + total_outflow == pytest.approx(
        initial_total + 500 * 60, rel=1e-5
    )


@pytest.mark.parametrize("has_weirs", [True, False])
@pytest.mark.parametrize("gated", [True, False])
def test_saved_weir_heights(tmp_path: Path, has_weirs: bool, gated: bool) -> None:
    """Write resolved heights, including an empty file when no weirs exist.

    Args:
        tmp_path: Output folder for the run.
        has_weirs: Whether to include a depth-based and a known-height weir.
        gated: Whether the depth-based structure uses the controlled fallback.
    """
    from unittest.mock import Mock

    import pandas as pd

    from geb.hydrology.routing import Routing

    heights: ArrayFloat32 = np.array(
        [-1, 3, 0] if has_weirs else [0, 0, 0], dtype=np.float32
    )
    routing: Mock = Mock()
    routing.router = make_router(heights, gated=has_weirs and gated)
    routing.weir_height_m = heights
    routing.weir_gate = np.array([has_weirs and gated, False, False])
    routing.var.river_ids = np.array([11, 12, 13], dtype=np.int32)
    routing.grid.lonlat = np.array([[-6, 54], [-6, 53], [-6, 52]], dtype=np.float32)
    routing.model.output_folder = tmp_path
    Routing.save_weirs(routing)
    saved: pd.DataFrame = pd.read_csv(tmp_path / "weir_heights.csv")
    assert len(saved) == (2 if has_weirs else 0)
    assert "effective_height_m" in saved.columns
    if has_weirs:
        assert saved.effective_height_m.tolist() == [5, 3]
        assert saved.bankfull_depth_m.tolist() == [4, 4]
        assert saved.gate_controlled.tolist() == [gated, False]
        assert saved.gate_report_index.tolist() == [0 if gated else -1, -1]
        assert saved.grid_cell_index.tolist() == [0, 1]
        assert saved.latitude_deg.tolist() == [54, 53]


def make_stage_gate_router(
    opening_fraction: float, closing_fraction: float, gated: bool = True
) -> LocalInertial:
    """Create three river cells with a stage-controlled gate on the first cell.

    Args:
        opening_fraction: Opening depth divided by crest height (dimensionless).
        closing_fraction: Closing depth divided by crest height (dimensionless).
        gated: Whether the first cell has a gate flag.

    Returns:
        Router with thresholds derived from crest-height fractions.

    Raises:
        ValueError: If the fractions do not define valid gate thresholds.
    """  # noqa: DOC502
    return _make_local_inertial(
        dt=60,
        river_network=pyflwdir.from_array(
            np.array([[6, 6, 5]], dtype=np.uint8), ftype="ldd"
        ),
        river_length=np.full(3, 1000, dtype=np.float32),
        river_width=np.full(3, 200, dtype=np.float32),
        bankfull_river_elevation_m=np.array([10, 9, 8], dtype=np.float32),
        bankfull_depth_m=4.0,
        weir_height_m=np.array([2, 0, 0], dtype=np.float32),
        weir_gate=np.array([gated, False, False]),
        gate_opening_level_fraction=opening_fraction,
        gate_closing_level_fraction=closing_fraction,
    )


def test_stage_gate_opens_and_closes_on_upstream_depth() -> None:
    """Open above the opening depth, stay open until below the closing depth."""
    router: LocalInertial = make_stage_gate_router(0.75, 0.25)
    depth: float
    expected_open: bool
    for depth, expected_open in (
        (1.4, False),
        (1.9, True),
        (1.0, True),
        (0.3, False),
    ):
        storage: ArrayFloat64 = (
            1000 * 200 / 3 * np.array([depth, 0.2, 0.2], dtype=np.float64) ** 1.5
        )
        router.step(
            Q_prev_m3_s=np.full(3, 5, dtype=np.float32),
            river_storage_m3=storage,
            sideflow_m3=np.zeros(3, dtype=np.float32),
            evaporation_m3=np.zeros(3, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        assert bool(router.gate_open[0]) == expected_open


@pytest.mark.parametrize("opening_fraction,closing_fraction", [(0.7, 0.3), (0.9, 0.0)])
@pytest.mark.parametrize("gated", [True, False])
def test_fraction_gate_thresholds(
    opening_fraction: float, closing_fraction: float, gated: bool
) -> None:
    """Scale thresholds by crest height and leave ungated cells inactive.

    Args:
        opening_fraction: Opening fraction of crest height (dimensionless).
        closing_fraction: Closing fraction of crest height (dimensionless).
        gated: Whether the first structure has a gate.

    Returns:
        None.

    Raises:
        AssertionError: If derived thresholds differ from the configured fractions.
    """  # noqa: DOC202, DOC502
    router: LocalInertial = make_stage_gate_router(
        opening_fraction, closing_fraction, gated
    )
    np.testing.assert_allclose(
        router._gate_open_depth_m, [2 * opening_fraction * gated, 0, 0]
    )
    np.testing.assert_allclose(
        router._gate_close_depth_m, [2 * closing_fraction * gated, 0, 0]
    )


def test_default_stage_gate_cycles_and_conserves_water() -> None:
    """Resolve default levels and produce releases from stored water.

    Returns:
        None.

    Raises:
        AssertionError: If the gate does not cycle or water is lost.
    """  # noqa: DOC202, DOC502
    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=pyflwdir.from_array(
            np.array([[6, 6, 5]], dtype=np.uint8), ftype="ldd"
        ),
        river_length=np.full(3, 1000, dtype=np.float32),
        river_width=np.full(3, 20, dtype=np.float32),
        bankfull_river_elevation_m=np.array([10, 9, 8], dtype=np.float32),
        bankfull_depth_m=4.0,
        weir_height_m=np.array([2, 0, 0], dtype=np.float32),
        weir_gate=np.array([True, False, False]),
    )
    np.testing.assert_allclose(router._gate_open_depth_m, [1.8, 0, 0])
    np.testing.assert_allclose(router._gate_close_depth_m, [1.4, 0, 0])
    storage: ArrayFloat64 = 1000 * 20 / 3 * np.array([1.4, 0.2, 0.2]) ** 1.5
    initial_storage_m3: float = float(storage.sum())
    discharge: ArrayFloat32 = np.zeros(3, dtype=np.float32)
    outflow_m3: float = 0.0
    closing_count: float = 0.0
    gate_flows: list[float] = []
    for _ in range(72):
        result: tuple = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=np.array([3600, 0, 0], dtype=np.float32),
            evaporation_m3=np.zeros(3, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = result[:2]
        gate_flows.append(float(discharge[0]))
        outflow_m3 += float(result[6])
        closing_count += float(router._geom_inbank[0, GEOM_IN_GATE_CLOSING_COUNT])
        assert (storage >= 0).all()
    assert closing_count > 2
    assert max(gate_flows) > 2.0  # Releases exceed the steady 1 m3/s inflow.
    assert min(gate_flows) == 0.0
    assert storage.sum() + outflow_m3 == pytest.approx(
        initial_storage_m3 + 72 * 3600, rel=1e-5
    )


@pytest.mark.parametrize(
    "opening,closing",
    [
        (1, 0.7),
        (0, 0),
        (0.5, 0.5),
        (0.9, -0.1),
        (float("nan"), 0.7),
        (0.9, float("inf")),
    ],
)
def test_invalid_stage_level_fractions(opening: float, closing: float) -> None:
    """Reject invalid automatic stage thresholds.

    Args:
        opening: Invalid opening fraction of crest height (dimensionless).
        closing: Invalid closing fraction of crest height (dimensionless).

    Returns:
        None.

    Raises:
        AssertionError: If invalid fractions are accepted.
    """  # noqa: DOC202, DOC502
    from geb.config_schema import RoutingConfig

    with pytest.raises(ValueError):
        RoutingConfig(
            gate_opening_level_fraction=opening, gate_closing_level_fraction=closing
        )
    with pytest.raises(ValueError, match="Gate level fractions"):
        _make_local_inertial(
            dt=60,
            river_network=pyflwdir.from_array(
                np.array([[6, 6, 5]], dtype=np.uint8), ftype="ldd"
            ),
            river_length=np.full(3, 1000, dtype=np.float32),
            gate_opening_level_fraction=opening,
            gate_closing_level_fraction=closing,
        )


@pytest.mark.parametrize(
    "heights,bankfull_depths",
    [
        ([-3, 0, 0], [4, 4, 4]),
        ([float("nan"), 0, 0], [4, 4, 4]),
        ([-2, 0, 0], [0, 4, 4]),
        ([-1, 0, 0], [float("nan"), 4, 4]),
        ([0, 0], [4, 4, 4]),
    ],
)
def test_invalid_height_markers(
    heights: list[float], bankfull_depths: list[float]
) -> None:
    """Reject unknown height markers and invalid required depths.

    Args:
        heights: Heights (m) or missing-height markers.
        bankfull_depths: River depths at bankfull flow (m).

    Returns:
        None.

    Raises:
        AssertionError: If invalid height data are accepted.
    """  # noqa: DOC202, DOC502
    with pytest.raises(ValueError, match="[Ww]eir|bankfull_depth"):
        _make_local_inertial(
            dt=60,
            river_network=pyflwdir.from_array(
                np.array([[6, 6, 5]], dtype=np.uint8), ftype="ldd"
            ),
            river_length=np.full(3, 1000, dtype=np.float32),
            bankfull_depth_m=np.array(bankfull_depths, dtype=np.float32),
            weir_height_m=np.array(heights, dtype=np.float32),
        )
