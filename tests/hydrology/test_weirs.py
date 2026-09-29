"""Weirs hold back water without adding or losing water."""

import numpy as np
import pyflwdir
import pytest

from geb.geb_types import ArrayFloat32, ArrayFloat64
from geb.hydrology.routing.inertial_substeps import GEOM_IN_INTERFACE_BED_ELEVATION_MAX
from geb.hydrology.routing.local_inertial import LocalInertial

from .test_routing import _make_local_inertial


def make_router(heights: ArrayFloat32 | None) -> LocalInertial:
    """Create three river cells with a weir at the end of the first cell.

    Args:
        heights: Weir heights above the river bed (m), or None for no weirs.

    Returns:
        Router with 1 km reaches, 20 m width, and 4 m bankfull depth.
    """
    network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        np.array([[6, 6, 5]], dtype=np.uint8), ftype="ldd"
    )
    return _make_local_inertial(
        dt=60,
        river_network=network,
        river_length=np.full(3, 1000, dtype=np.float32),
        river_width=np.full(3, 20, dtype=np.float32),
        bankfull_river_elevation_m=np.array([10, 9, 8], dtype=np.float32),
        bankfull_depth_m=4.0,
        weir_height_m=heights,
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
    "heights", [[-1, 0, 0], [float("nan"), 0, 0], [0, 0, 1], [1, 0]]
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
