"""Weirs hold back water without adding or losing water."""

from pathlib import Path

import numpy as np
import pyflwdir
import pytest

from geb.geb_types import ArrayBool, ArrayFloat32, ArrayFloat64
from geb.hydrology.routing.inertial_substeps import (
    GEOM_IN_INTERFACE_BED_ELEVATION_MAX,
)
from geb.hydrology.routing.local_inertial import LocalInertial
from geb.workflows import (
    USE_BANKFULL_HEIGHT,
    USE_HALF_BANKFULL_HEIGHT,
)

from .test_routing import _make_local_inertial


def make_router(
    heights: ArrayFloat32 | None,
    dt_seconds: int = 60,
    instream_dam: ArrayBool | None = None,
) -> LocalInertial:
    """Create three river cells with a weir at the end of the first cell.

    Args:
        heights: Weir heights above the river bed (m), or None for no weirs.
        dt_seconds: Routing interval (seconds).
        instream_dam: Boolean mask of dams with gradual opening, or None for fixed weirs.

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
        instream_dam=instream_dam,
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
    [[-4, 0, 0], [float("nan"), 0, 0], [0, 0, 1], [1, 0]],
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
    weir.update_channel_geometry(np.full(3, 25, dtype=np.float32), weir.bankfull_depth)
    index: int = int(np.flatnonzero(weir._weir_height_inertial > 0)[0])
    assert weir._geom_inbank[index, GEOM_IN_INTERFACE_BED_ELEVATION_MAX] == 12.0


@pytest.mark.parametrize("bankfull_depth", [2.0, 8.0])
@pytest.mark.parametrize(
    "height_marker",
    [USE_BANKFULL_HEIGHT, USE_HALF_BANKFULL_HEIGHT],
)
def test_missing_height_scales_with_depth(
    bankfull_depth: float, height_marker: float
) -> None:
    """Missing heights scale with the channel; known heights stay fixed.

    Args:
        bankfull_depth: River depth at bankfull flow (m).
        height_marker: Missing-height marker (-2 or -3).

    Returns:
        None.

    Raises:
        AssertionError: If known heights or depth-based proxies resolve incorrectly.
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
    )
    np.testing.assert_array_equal(input_heights, [height_marker, 3, 0])
    expected_height_m: float = (
        bankfull_depth * 0.5
        if height_marker == USE_HALF_BANKFULL_HEIGHT
        else bankfull_depth
    )
    np.testing.assert_allclose(
        router._weir_height_inertial, np.array([expected_height_m, 3, 0])
    )


@pytest.mark.parametrize("has_weirs", [True, False])
def test_save_weirs(tmp_path: Path, has_weirs: bool) -> None:
    """Write resolved heights, including an empty file when no weirs exist.

    Args:
        tmp_path: Output folder for the run.
        has_weirs: Whether to include a depth-based and a known-height weir.
    """
    from unittest.mock import Mock

    import pandas as pd

    from geb.hydrology.routing import Routing

    heights: ArrayFloat32 = np.array(
        [-2, 3, 0] if has_weirs else [0, 0, 0], dtype=np.float32
    )
    routing: Mock = Mock()
    routing.router = make_router(
        heights, instream_dam=np.array([has_weirs, False, False])
    )
    routing.weir_height_m = heights
    routing.var.river_ids = np.array([11, 12, 13], dtype=np.int32)
    routing.grid.lonlat = np.array([[-6, 54], [-6, 53], [-6, 52]], dtype=np.float32)
    routing.model.output_folder = tmp_path
    Routing.save_weirs(routing)
    saved: pd.DataFrame = pd.read_csv(tmp_path / "weir_heights.csv")
    assert len(saved) == (2 if has_weirs else 0)
    assert "effective_height_m" in saved.columns
    if has_weirs:
        assert saved.effective_height_m.tolist() == [4, 3]
        assert saved.instream_dam.tolist() == [True, False]
        assert saved.bankfull_depth_m.tolist() == [4, 4]
        assert saved.grid_cell_index.tolist() == [0, 1]
        assert saved.latitude_deg.tolist() == [54, 53]


@pytest.mark.parametrize(
    "heights,bankfull_depths",
    [
        ([-4, 0, 0], [4, 4, 4]),
        ([-1, 0, 0], [4, 4, 4]),
        ([float("nan"), 0, 0], [4, 4, 4]),
        ([-2, 0, 0], [0, 4, 4]),
        ([-3, 0, 0], [0, 4, 4]),
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


@pytest.mark.parametrize(
    "upstream_depth_m,downstream_depth_m,expected_discharge_m3_s",
    [
        (1.0, 0.2, 0.0),
        (2.0, 0.2, 0.0),
        (3.0, 0.2, 0.24843513333333334),
        (3.0, 3.5, 0.032688833333333334),
        (3.0, 4.0, 0.0),
        (3.0, 4.5, 0.0),
    ],
)
@pytest.mark.parametrize("river_width_m", [20.0, 25.0])
def test_fixed_weir_uses_raised_sill(
    upstream_depth_m: float,
    downstream_depth_m: float,
    expected_discharge_m3_s: float,
    river_width_m: float,
) -> None:
    """Use river momentum over a raised sill for fixed weirs.

    Args:
        upstream_depth_m: Initial upstream water depth above the bed (m).
        downstream_depth_m: Initial downstream water depth above the bed (m).
        expected_discharge_m3_s: Expected weir discharge (m3/s).
        river_width_m: Updated river width (m).

    Returns:
        None.

    Raises:
        AssertionError: If flow differs from raised-sill hydraulics or loses water.
    """  # noqa: DOC202, DOC502
    router: LocalInertial = make_router(
        np.array([2, 0, 0], dtype=np.float32), dt_seconds=1
    )
    router.update_channel_geometry(
        np.full(3, river_width_m, dtype=np.float32), router.bankfull_depth
    )
    # One short substep isolates acceleration over the sill. The positive
    # reference flows use Q = g * dt * flow_area * water_surface_slope.
    initial_storage_m3: ArrayFloat64 = (
        1000
        * river_width_m
        / 3
        * np.array([upstream_depth_m, downstream_depth_m, 0.2], dtype=np.float64) ** 1.5
    )
    result: tuple = router.step(
        Q_prev_m3_s=np.zeros(3, dtype=np.float32),
        river_storage_m3=initial_storage_m3.copy(),
        sideflow_m3=np.zeros(3, dtype=np.float32),
        evaporation_m3=np.zeros(3, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )
    assert result[0][0] == pytest.approx(
        expected_discharge_m3_s * river_width_m / 20.0, abs=1e-5
    )
    assert np.all(result[1] >= 0)
    assert result[1].sum() + result[6] == pytest.approx(
        initial_storage_m3.sum(), rel=1e-5
    )


@pytest.mark.parametrize(
    ("depth_fraction", "expected_opening"),
    [(0.0, 0.1), (0.64, 0.1), (0.65, 0.1), (0.775, 0.55), (0.9, 1.0), (1.5, 1.0)],
)
def test_dam_opening_thresholds(depth_fraction: float, expected_opening: float) -> None:
    """Open smoothly between the requested bankfull thresholds.

    Args:
        depth_fraction: Upstream depth divided by bankfull depth (dimensionless).
        expected_opening: Expected fraction of the open river passage.
    """
    from geb.hydrology.routing.inertial_substeps import (
        compute_instream_dam_open_fraction,
    )

    opening: np.float32 = compute_instream_dam_open_fraction(
        np.float32(depth_fraction * 4), np.float32(4)
    )
    assert opening == pytest.approx(expected_opening, abs=1e-6)


@pytest.mark.parametrize(
    ("depth_m", "bankfull_m"),
    [(float("nan"), 4), (1, 0), (1, -1), (1, float("inf"))],
)
def test_invalid_dam_depths(depth_m: float, bankfull_m: float) -> None:
    """Reject depths that cannot define the dam operation rule.

    Args:
        depth_m: Upstream water depth (m).
        bankfull_m: Bankfull channel depth (m).
    """
    from geb.hydrology.routing.inertial_substeps import (
        compute_instream_dam_open_fraction,
    )

    with pytest.raises(ValueError, match="depth"):
        compute_instream_dam_open_fraction(np.float32(depth_m), np.float32(bankfull_m))


@pytest.mark.parametrize("crest_height_m", [1.0, 4.0, 8.0])
@pytest.mark.parametrize("depth_m", [1.0, 3.1, 3.8, 4.0])
def test_instream_dam_flow_and_mass_balance(
    depth_m: float, crest_height_m: float
) -> None:
    """Pass water below the crest and recover ordinary river flow above 90%.

    Args:
        depth_m: Initial upstream depth above the original river bed (m).
        crest_height_m: Recorded dam height, independent of bankfull depth (m).
    """
    ordinary: LocalInertial = make_router(None, dt_seconds=1)
    fixed: LocalInertial = make_router(
        np.array([crest_height_m, 0, 0], dtype=np.float32), dt_seconds=1
    )
    gated: LocalInertial = make_router(
        np.array([crest_height_m, 0, 0], dtype=np.float32),
        dt_seconds=1,
        instream_dam=np.array([True, False, False]),
    )
    initial_storage: ArrayFloat64 = np.array(
        [1000 * 20 / 3 * depth_m**1.5, 500, 500], dtype=np.float64
    )
    results: list[tuple] = []
    router: LocalInertial
    for router in (ordinary, fixed, gated):
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
        results.append(result)
        assert np.all(result[1] >= 0)
        assert result[1].sum() + result[6] == pytest.approx(
            initial_storage.sum(), rel=1e-6
        )
    assert results[2][0][0] > 0
    if depth_m >= 3.6:
        np.testing.assert_allclose(results[2][0], results[0][0], rtol=1e-6)
        np.testing.assert_allclose(results[2][1], results[0][1], rtol=1e-6)
    else:
        if depth_m <= crest_height_m:
            assert results[1][0][0] == 0
        assert results[1][0][0] < results[2][0][0] < results[0][0][0]


@pytest.mark.parametrize("flags", [[True, False], [1, 0, 0], [False, False, True]])
def test_invalid_instream_dam_flags(flags: list[bool] | list[int]) -> None:
    """Reject mismatched flags, non-boolean flags, and dams without a crest.

    Args:
        flags: Invalid instream-dam flags in original river cell order.
    """
    with pytest.raises(ValueError, match="[Ii]nstream dam"):
        make_router(
            np.array([-2, 0, 0], dtype=np.float32), instream_dam=np.array(flags)
        )


@pytest.mark.parametrize("substep_seconds", [0.5, 5.0])
def test_dam_steady_flow_is_independent_of_substep(substep_seconds: float) -> None:
    """A 10% passage carries 10% of steady river flow for either substep size.

    Args:
        substep_seconds: Momentum integration interval (seconds).
    """
    from geb.hydrology.routing.inertial_substeps import _solve_inertial_momentum

    router: LocalInertial = make_router(None)
    discharge: np.float32 = np.float32(0.0)
    elapsed_step: int
    for elapsed_step in range(int(2000 / substep_seconds)):
        discharge = _solve_inertial_momentum(
            reach_idx=0,
            effective_depth=np.float32(1.0),
            water_slope=np.float32(-0.001),
            curr_discharge=discharge,
            min_wet_depth_m=np.float32(0.001),
            geom_inbank=router._geom_inbank,
            geom_overbank=router._geom_overbank,
            g_dt_substep=np.float32(9.81 * substep_seconds),
            sqrt_gravity=np.float32(np.sqrt(9.81)),
            can_reverse=False,
            river_storage_m3_inertial=np.full(3, 1e9, dtype=np.float64),
            inv_dt_substep=np.float32(1 / substep_seconds),
            instream_dam_open_fraction=np.float32(0.1),
            overflow_depth_m=np.float32(0.0),
        )
    passage_area_m2: float = 0.1 * 10 / 1.5
    hydraulic_radius_m: float = (10 / 1.5) / (10 + 8 / 30)
    expected_discharge: float = (
        passage_area_m2 * hydraulic_radius_m ** (2 / 3) * np.sqrt(0.001) / 0.03
    )
    assert discharge == pytest.approx(expected_discharge, rel=1e-4)
