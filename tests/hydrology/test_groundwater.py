"""Tests for the groundwater module of GEB.

The groundwater module uses MODFLOW to simulate groundwater flow and interactions
with surface water and the unsaturated zone.
"""

import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from affine import Affine

from geb.geb_types import (
    ArrayFloat32,
    ArrayFloat64,
    ThreeDArrayFloat64,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
)
from geb.hydrology.groundwater.model import (
    ModFlowSimulation,
    distribute_well_abstraction_m3_per_layer,
    get_groundwater_storage_m,
    get_water_table_depth,
)
from geb.workflows.raster import calculate_cell_area_m2, compress

from ..testconfig import GEB_PACKAGE_DIR, output_folder, tmp_folder

logger: logging.Logger = logging.getLogger(__name__)


class ModFlowParams(TypedDict):
    """Type definition for ModFlowSimulation parameters."""

    working_directory: Path
    modflow_bin_folder: Path
    topography: ArrayFloat32
    gt: tuple[float, float, float, float, float, float]
    specific_storage: TwoDArrayFloat32
    specific_yield: TwoDArrayFloat32
    layer_boundary_elevation: TwoDArrayFloat32
    basin_mask: TwoDArrayBool
    hydraulic_conductivity: TwoDArrayFloat32
    heads: TwoDArrayFloat64
    heads_update_callback: Callable[[ArrayFloat64], None]
    verbose: bool
    never_load_from_disk: bool
    logger: logging.Logger
    boundary_heads: ThreeDArrayFloat64
    boundary_mask: TwoDArrayBool
    boundary_layer_boundary_elevation: ThreeDArrayFloat64
    boundary_hydraulic_conductivity: ThreeDArrayFloat64


def decompress(
    array: npt.NDArray[Any], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[Any]:
    """Decompress an array using the basin mask.

    Args:
        array: The array to decompress.
        mask: The basin mask.

    Returns:
        The decompressed array.

    Raises:
        ValueError: If the array is not 1D or 2D.
    """
    if array.ndim == 1:
        out = np.full(mask.shape, np.nan)
    elif array.ndim == 2:
        out = np.full((array.shape[0], *mask.shape), np.nan)
    else:
        raise ValueError(f"Array must be 1D or 2D, got {array.ndim}D")
    out[..., ~mask] = array
    return out


XSIZE: Literal[12] = 12
YSIZE: Literal[10] = 10
NLAY: Literal[2] = 2

# Create a topography 2D map
x = np.linspace(-5, 5, XSIZE)
y = np.linspace(-5, 5, YSIZE)
x, y = np.meshgrid(x, y)

topography = np.exp2(-(x**2) - y**2 + 5).astype(np.float32)
basin_mask = np.zeros((YSIZE, XSIZE), dtype=bool)
basin_mask[0] = True
basin_mask[-3:-1, 0:3] = True

gt: tuple[float, float, float, float, float, float] = (
    4.864242872511027,
    0.0001,
    0,
    52.33412139354429,
    0,
    -0.0001,
)

cell_area = calculate_cell_area_m2(Affine.from_gdal(*gt), YSIZE, XSIZE)


layer_boundary_elevation = np.full((NLAY + 1, YSIZE, XSIZE), np.nan, dtype=np.float32)
layer_boundary_elevation[0] = topography
for layer in range(1, NLAY + 1):
    layer_boundary_elevation[layer] = (
        layer_boundary_elevation[layer - 1] - 5
    )  # each layer is 5 m thick

heads = np.full((NLAY, YSIZE, XSIZE), 0, dtype=np.float32)
for layer in range(NLAY):
    heads[layer] = topography - 2

# Boundary conditions on 1-cell extended grid (NLAY, YSIZE + 2, XSIZE + 2)
# Outside boundary cell (row 10, col 13) directly adjacent to active domain cell (row 9, col 11)
default_boundary_mask: TwoDArrayBool = np.zeros((YSIZE + 2, XSIZE + 2), dtype=bool)
default_boundary_mask[9 + 1, 11 + 2] = True

default_boundary_heads: ThreeDArrayFloat64 = np.zeros(
    (NLAY, YSIZE + 2, XSIZE + 2), dtype=np.float64
)
for layer in range(NLAY):
    default_boundary_heads[layer, 1:-1, 1:-1] = heads[layer]
    default_boundary_heads[layer, 0, 1:-1] = heads[layer, 0, :]
    default_boundary_heads[layer, -1, 1:-1] = heads[layer, -1, :]
    default_boundary_heads[layer, 1:-1, 0] = heads[layer, :, 0]
    default_boundary_heads[layer, 1:-1, -1] = heads[layer, :, -1]

default_boundary_layer_boundary_elevation: ThreeDArrayFloat64 = np.zeros(
    (NLAY + 1, YSIZE + 2, XSIZE + 2), dtype=np.float64
)
for b_idx in range(NLAY + 1):
    default_boundary_layer_boundary_elevation[b_idx, 1:-1, 1:-1] = (
        layer_boundary_elevation[b_idx]
    )
    default_boundary_layer_boundary_elevation[b_idx, 0, 1:-1] = (
        layer_boundary_elevation[b_idx, 0, :]
    )
    default_boundary_layer_boundary_elevation[b_idx, -1, 1:-1] = (
        layer_boundary_elevation[b_idx, -1, :]
    )
    default_boundary_layer_boundary_elevation[b_idx, 1:-1, 0] = (
        layer_boundary_elevation[b_idx, :, 0]
    )
    default_boundary_layer_boundary_elevation[b_idx, 1:-1, -1] = (
        layer_boundary_elevation[b_idx, :, -1]
    )

default_boundary_hydraulic_conductivity: ThreeDArrayFloat64 = np.ones(
    (NLAY, YSIZE + 2, XSIZE + 2), dtype=np.float64
)

default_params: ModFlowParams = {
    "working_directory": tmp_folder / "modflow",
    "modflow_bin_folder": GEB_PACKAGE_DIR / "modflow" / "bin",
    "gt": gt,
    "specific_storage": compress(np.full((NLAY, YSIZE, XSIZE), 0), basin_mask),
    "specific_yield": compress(np.full((NLAY, YSIZE, XSIZE), 0.8), basin_mask),
    "topography": compress(topography, basin_mask),
    "layer_boundary_elevation": compress(layer_boundary_elevation, basin_mask),
    "basin_mask": basin_mask,
    "heads": compress(heads, basin_mask),
    "hydraulic_conductivity": compress(np.full((NLAY, YSIZE, XSIZE), 1), basin_mask),
    "boundary_heads": default_boundary_heads,
    "boundary_mask": default_boundary_mask,
    "boundary_layer_boundary_elevation": default_boundary_layer_boundary_elevation,
    "boundary_hydraulic_conductivity": default_boundary_hydraulic_conductivity,
    "verbose": True,
    "never_load_from_disk": True,
    "heads_update_callback": lambda heads: None,
    "logger": logger,
}


def test_modflow_simulation_initialization() -> None:
    """Test initialization of MODFLOW simulation.

    Verifies that the ModFlowSimulation object is correctly
    initialized with the expected number of active cells and area.
    """
    sim: ModFlowSimulation = ModFlowSimulation(**default_params)
    assert sim.n_active_cells == (~basin_mask).sum()
    # In the Netherlands, the average area of a cell with this gt is ~75.8 m2
    assert np.allclose(sim.area, 75.8, atol=0.1)


def test_modflow_bmi_pointer_stability() -> None:
    """Test that repeated BMI pointer retrieval returns the same underlying memory.

    Verifies that values are updated in place and that a pointer retrieved before
    a time step still points to the same memory location afterwards.
    """
    sim: ModFlowSimulation = ModFlowSimulation(**deepcopy(default_params))
    try:
        head_tag = sim.mf6.get_var_address("X", sim.name)
        heads_ptr_first = sim.mf6.get_value_ptr(head_tag)
        heads_ptr_second = sim.mf6.get_value_ptr(head_tag)

        assert np.shares_memory(heads_ptr_first, heads_ptr_second)
        assert (
            heads_ptr_first.__array_interface__["data"][0]
            == heads_ptr_second.__array_interface__["data"][0]
        )

        heads_first_2d = heads_ptr_first.reshape(sim.nlay, sim.n_mf_cells)
        heads_second_2d = heads_ptr_second.reshape(sim.nlay, sim.n_mf_cells)

        initial_column = heads_second_2d[:, 0].copy()
        heads_first_2d[:, 0] = initial_column - 0.25
        np.testing.assert_allclose(heads_second_2d[:, 0], initial_column - 0.25)

        pointer_address_before_step = heads_ptr_first.__array_interface__["data"][0]
        sim.step()
        heads_ptr_after_step = sim.mf6.get_value_ptr(head_tag)
        assert (
            heads_ptr_after_step.__array_interface__["data"][0]
            == pointer_address_before_step
        )
    finally:
        sim.finalize()


def test_step() -> None:
    """Test single time step execution of MODFLOW simulation.

    Verifies that the simulation step maintains water balance
    between groundwater content, recharge, and drainage.
    """
    parameters = deepcopy(default_params)

    sim: ModFlowSimulation = ModFlowSimulation(**parameters)

    groundwater_content_prev = sim.groundwater_content_m3.sum()
    sim.step()

    drainage_m3 = sim.drainage_m3.sum()
    groundwater_content = sim.groundwater_content_m3.sum()
    recharge_m3 = sim.recharge_m3.sum()

    balance_pre = (
        groundwater_content_prev
        + recharge_m3
        - drainage_m3
        + sim.boundary_flow_m3.sum()
    )
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, rel_tol=1e-5)

    sim.finalize()


def test_recharge() -> None:
    """Test groundwater recharge functionality.

    Verifies that recharge water is correctly added to groundwater
    storage and maintains proper water balance.
    """
    parameters = deepcopy(default_params)
    parameters["heads"] = parameters["heads"] - 2
    parameters["boundary_heads"] = parameters["boundary_heads"] - 2

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = sim.groundwater_content_m3.sum()
    recharge_m = np.full((YSIZE, XSIZE), 0.1)
    recharge_m[9, 11] = 0.0  # Do not recharge constant-head boundary cell
    recharge_m3 = recharge_m * cell_area
    sim.set_recharge_m3(compress(recharge_m3, sim.basin_mask))
    sim.step()

    drainage_m3 = np.nansum(sim.drainage_m3)
    assert np.nansum(drainage_m3) == 0
    groundwater_content = np.nansum(sim.groundwater_content_m3)

    assert np.nansum(sim.recharge_m) == np.nansum(sim.recharge_m3 / sim.area)

    recharge_m3 = np.nansum(sim.recharge_m3)
    balance_pre = (
        groundwater_content_prev
        + recharge_m3
        - drainage_m3
        + sim.boundary_flow_m3.sum()
    )
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, abs_tol=1, rel_tol=1e-5)

    sim.finalize()


def test_drainage() -> None:
    """Test groundwater drainage functionality.

    Verifies that drainage occurs when water table is at surface
    and that drainage is zero when heads are below drainage level.
    """
    parameters = deepcopy(default_params)
    layer_boundary_elevation = parameters["layer_boundary_elevation"]
    topography = np.full((YSIZE, XSIZE), 0)

    parameters["topography"] = np.zeros_like(parameters["topography"])

    layer_boundary_elevation[0] = compress(topography - 2, basin_mask)
    layer_boundary_elevation[1] = compress(topography - 10, basin_mask)
    layer_boundary_elevation[2] = compress(topography - 20, basin_mask)

    parameters["heads"][0] = layer_boundary_elevation[0]
    parameters["heads"][1] = layer_boundary_elevation[0]
    parameters["boundary_heads"] = parameters["boundary_heads"] - 2

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = np.nansum(sim.groundwater_content_m3)

    recharge_m = np.full((YSIZE, XSIZE), 0.1)
    recharge_m[9, 11] = 0.0  # Do not recharge constant-head boundary cell
    recharge_m3 = recharge_m * cell_area

    sim.set_recharge_m3(compress(recharge_m3, sim.basin_mask))
    sim.step()

    drainage = np.nansum(sim.drainage_m3)
    assert drainage.sum() > 0

    assert math.isclose(np.nansum(sim._drainage_m * sim.area), np.nansum(drainage))

    groundwater_content = np.nansum(sim.groundwater_content_m3)

    balance_pre = (
        groundwater_content_prev
        - drainage
        + sim.recharge_m3.sum()
        + sim.boundary_flow_m3.sum()
    )
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, rel_tol=1e-5)

    sim.finalize()

    parameters["heads"][0] = layer_boundary_elevation[0] - 1
    parameters["heads"][1] = layer_boundary_elevation[0] - 1
    parameters["boundary_heads"] = np.full_like(parameters["boundary_heads"], -3.0)

    sim = ModFlowSimulation(**parameters)

    sim.step()

    drainage = np.nansum(sim.drainage_m3)
    assert drainage == 0

    sim.finalize()


def test_wells() -> None:
    """Test groundwater well abstraction functionality.

    Verifies that well abstraction correctly removes water from
    groundwater storage and maintains water balance.
    """
    parameters = deepcopy(default_params)
    parameters["heads"][:,] = compress(
        topography - 2, basin_mask
    )  # set head lower than drainage

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = sim.groundwater_content_m3.sum()

    groundwater_abstracton = np.full((YSIZE, XSIZE), 0.10)
    groundwater_abstracton[0, 0] = 0.10
    groundwater_abstracton[1, 1] = 0.05
    groundwater_abstracton[4, 5] = 0.20
    groundwater_abstracton[9, 11] = 0.0  # Do not abstract from constant-head cell
    groundwater_abstracton[parameters["basin_mask"]] = np.nan
    groundwater_abstracton = compress(groundwater_abstracton, sim.basin_mask) * sim.area

    # setting an abstraction that is too high should raise an error
    try:
        sim.set_groundwater_abstraction_m3(groundwater_abstracton * 1e9)
        assert False
    except AssertionError:
        pass

    sim.set_groundwater_abstraction_m3(groundwater_abstracton)
    sim.step()

    drainage = np.nansum(sim.drainage_m3)
    assert drainage.sum() == 0

    groundwater_content = sim.groundwater_content_m3.sum()

    total_abstraction = groundwater_abstracton.sum()
    balance_pre = (
        groundwater_content_prev - total_abstraction + sim.boundary_flow_m3.sum()
    )
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, rel_tol=1e-6)

    for _ in range(100):
        sim.set_groundwater_abstraction_m3(
            np.minimum(groundwater_abstracton, sim.available_groundwater_m3)
        )
        sim.step()

    sim.finalize()


def visualize_modflow_results(
    sim: ModFlowSimulation,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes, plt.Axes],
) -> None:
    """This function is used to visualize the current state of a ModFlowSimulation.

    Plots the topography, groundwater head, groundwater depth, and drainage, on
    axes 1 to 5 respectively.

    Args:
        sim: The ModFlowSimulation object. This contains the current state of the simulation.
        axes: An array of matplotlib axes to plot on. Should be of shape (5,).
    """
    (ax1, ax2, ax3, ax4, ax5) = axes

    # Plot topography
    im1 = ax1.imshow(
        decompress(
            sim.topography,
            sim.basin_mask,
        ),
        cmap="terrain",
    )
    ax1.set_title("Topography")
    plt.colorbar(im1, ax=ax1, label="Elevation (m)")

    # Plot groundwater head
    im2 = ax2.imshow(decompress(sim.heads[0], sim.basin_mask), cmap="viridis")
    ax2.set_title("Groundwater Head Top layer")
    plt.colorbar(im2, ax=ax2, label="Head (m)")

    im3 = ax3.imshow(decompress(sim.heads[1], sim.basin_mask), cmap="viridis")
    ax3.set_title("Groundwater Head Bottom layer")
    plt.colorbar(im3, ax=ax3, label="Head (m)")

    # Plot groundwater depth
    im4 = ax4.imshow(decompress(sim.groundwater_depth, sim.basin_mask), cmap="RdYlBu")
    ax4.set_title("Groundwater Depth")
    plt.colorbar(im4, ax=ax4, label="Depth (m)")

    # Plot drainage
    drainage_m3 = decompress(sim.drainage_m3, sim.basin_mask)
    drainage = drainage_m3 / cell_area
    im5 = ax5.imshow(drainage, cmap="Blues")
    ax5.set_title("Drainage")
    plt.colorbar(im5, ax=ax5, label="Drainage (m/day)")


def test_modflow_simulation_with_visualization() -> None:
    """Test MODFLOW simulation with visualization output.

    Runs a simulation with random recharge and abstraction,
    generating visualization plots of the results.
    """
    parameters = deepcopy(default_params)
    parameters["heads"][:,] = compress(topography, basin_mask)
    sim = ModFlowSimulation(**parameters)

    fig, axes = plt.subplots(5, 5, figsize=(15, 10))
    plt.tight_layout()

    # Run the simulation for a few steps
    for i in range(5):
        recharge_m = np.random.uniform(0, 0.001, size=(YSIZE, XSIZE))
        recharge_m3 = recharge_m * cell_area
        sim.set_recharge_m3(compress(recharge_m3, sim.basin_mask))

        groundwater_abstracton = np.full((YSIZE, XSIZE), 0.0)
        groundwater_abstracton[2, 2] = 1.25
        groundwater_abstracton[3, 5] = 0.20

        sim.set_groundwater_abstraction_m3(
            compress(groundwater_abstracton, sim.basin_mask) * sim.area
        )
        sim.step()

        # Visualize the results
        visualize_modflow_results(sim, axes[i])

    sim.finalize()
    plt.savefig(output_folder / "modflow_simulation.png")


def test_modflow_simulation_with_restore() -> None:
    """Test MODFLOW simulation state restoration.

    Verifies that the simulation can be restored to a previous
    state and continue simulation correctly.
    """
    parameters = deepcopy(default_params)
    parameters["heads"][:,] = compress(topography, basin_mask)
    sim = ModFlowSimulation(**parameters)

    recharge_m = np.random.uniform(0, 0.001, size=(YSIZE, XSIZE))
    recharge_m3 = recharge_m * cell_area
    sim.set_recharge_m3(compress(recharge_m3, sim.basin_mask))

    groundwater_abstracton = np.full((YSIZE, XSIZE), 0.0)
    groundwater_abstracton[2, 2] = 1.25
    groundwater_abstracton[3, 5] = 0.20

    sim.set_groundwater_abstraction_m3(
        compress(groundwater_abstracton, sim.basin_mask) * sim.area
    )

    # Run the simulation for a few steps
    for i in range(2):
        sim.step()

    heads_mid = sim.heads.copy()

    for i in range(3):
        print("before restore", sim.heads.mean())
        sim.step()

    heads_end = sim.heads.copy()

    sim.restore(heads_mid)

    for i in range(3):
        print("after restore", sim.heads.mean())
        sim.step()

    np.testing.assert_allclose(sim.heads, heads_end)

    sim.finalize()


def test_get_water_table_depth() -> None:
    """Test calculation of water table depth.

    Verifies that water table depth is correctly calculated
    from layer boundaries, heads, and surface elevation.
    """
    layer_boundary_elevation = np.array(
        [
            [100, 100, 100, 100, 100, 100, 100],
            [50, 50, 50, 50, 50, 50, 50],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    head = np.array(
        [
            [110, 90, 45, 45, -1, 50.05, 100.01],
            [115, 60, 60, 40, -1, 50.01, 100.01],
        ]
    )
    elevation = np.array([103, 103, 103, 103, 103, 103, 103])
    water_table_depth = get_water_table_depth(
        layer_boundary_elevation, head, elevation, min_remaining_layer_storage_m=0
    )
    np.testing.assert_allclose(
        water_table_depth, np.array([3, 13, 53, 63, 103, 52.95, 3])
    )

    water_table_depth = get_water_table_depth(
        layer_boundary_elevation, head, elevation, min_remaining_layer_storage_m=0.1
    )
    np.testing.assert_allclose(water_table_depth, np.array([3, 13, 53, 63, 103, 53, 3]))


def test_get_groundwater_storage_m() -> None:
    """Test calculation of groundwater storage in meters.

    Verifies that groundwater storage is correctly calculated
    from layer boundaries, heads, and specific yield.
    """
    layer_boundary_elevation = np.array(
        [
            [100, 100, 100, 100, 100],
            [50, 50, 50, 50, 50],
            [0, 0, 0, 0, 0],
        ]
    )
    head = np.array(
        [
            [110, 90, np.nan, np.nan, np.nan],
            [115, 60, 60, 40, -1],
        ]
    )
    specific_yield = np.array(
        [
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25, 0.25, 0.25],
        ]
    )
    specific_storage = np.zeros_like(specific_yield)
    storage = get_groundwater_storage_m(
        layer_boundary_elevation, head, specific_yield, specific_storage
    )
    np.testing.assert_allclose(storage, np.array([37.5, 32.5, 12.5, 10, 0]))
    storage = get_groundwater_storage_m(
        layer_boundary_elevation,
        head,
        specific_yield,
        specific_storage,
        min_remaining_layer_storage_m=1,
    )
    np.testing.assert_allclose(storage, np.array([36.75, 31.75, 12.25, 9.75, 0.0]))


def test_distribute_well_abstraction_m3_per_layer() -> None:
    """Test distribution of well abstraction across layers.

    Verifies that well abstraction rates are correctly distributed
    across groundwater layers based on available storage.
    """
    layer_boundary_elevation = np.array(
        [
            [100, 100, 100, 100, 100, 100, 100],
            [50, 50, 50, 50, 50, 50, 50],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    heads = np.array(
        [
            [110, 90, 0, 0, 0, 110, 110],
            [115, 60, 60, 40, -11, 115, 115],
        ],
        dtype=np.float64,
    )
    specific_yield = np.array(
        [
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float64,
    )
    area = np.array([100, 100, 100, 100, 100, 100, 100], dtype=np.float64)

    well_rate = np.array([-10, -10, -10, -10, -0, -3000, -0], dtype=np.float64)

    well_rate_per_layer = distribute_well_abstraction_m3_per_layer(
        well_rate, layer_boundary_elevation, heads, specific_yield, area
    )

    np.testing.assert_allclose(
        well_rate_per_layer,
        np.array(
            [
                [-10, -10, 0, 0, 0, -2500, 0],
                [0, 0, -10, -10, 0, -500, 0],
            ],
            dtype=np.float64,
        ),
    )

    well_rate_per_layer = distribute_well_abstraction_m3_per_layer(
        well_rate,
        layer_boundary_elevation,
        heads,
        specific_yield,
        area,
        min_remaining_layer_storage_m=1,
    )

    np.testing.assert_allclose(
        well_rate_per_layer,
        np.array(
            [
                [-10, -10, 0, 0, 0, -2450, 0],
                [0, 0, -10, -10, 0, -550, 0],
            ],
            dtype=np.float64,
        ),
    )


def test_modflow_simulation_with_groundwater_level_boundary() -> None:
    """Test MODFLOW simulation with specified groundwater level boundary condition (CHD).

    Verifies that:
    1. Boundary heads are correctly applied.
    2. Boundary heads have a physical hydrodynamic effect on interior cells of the modelled domain.
    3. Boundary heads can be dynamically updated via BMI.
    4. Custom boundary mask correctly configures specified constant-head boundary cells.
    """
    # 0. Set up outside boundary mask surrounding the active domain
    ext_active: np.ndarray = np.zeros((YSIZE + 2, XSIZE + 2), dtype=bool)
    ext_active[1:-1, 1:-1] = ~basin_mask
    touches_active: np.ndarray = np.zeros_like(ext_active, dtype=bool)
    touches_active[:-1, :] |= ext_active[1:, :]
    touches_active[1:, :] |= ext_active[:-1, :]
    touches_active[:, :-1] |= ext_active[:, 1:]
    touches_active[:, 1:] |= ext_active[:, :-1]
    bnd_mask_full: TwoDArrayBool = ~ext_active & touches_active

    # Baseline simulation with low boundary heads (10.0 m)
    p_base: ModFlowParams = deepcopy(default_params)
    p_base["working_directory"] = tmp_folder / "modflow_boundary_base"
    p_base["boundary_heads"] = np.full(
        (NLAY, YSIZE + 2, XSIZE + 2), 10.0, dtype=np.float64
    )
    p_base["boundary_mask"] = bnd_mask_full
    sim_base: ModFlowSimulation = ModFlowSimulation(**p_base)
    for _ in range(3):
        sim_base.step()
    decomp_base = decompress(sim_base.heads, sim_base.basin_mask)
    head_base_interior: float = float(decomp_base[0, 1, 5])
    sim_base.finalize()

    # 1. Test with high boundary heads (50.0 m)
    ext_boundary_heads: ThreeDArrayFloat64 = np.full(
        (NLAY, YSIZE + 2, XSIZE + 2), 50.0, dtype=np.float64
    )
    parameters: ModFlowParams = deepcopy(default_params)
    parameters["working_directory"] = tmp_folder / "modflow_boundary_test"
    parameters["boundary_heads"] = ext_boundary_heads
    parameters["boundary_mask"] = bnd_mask_full
    sim: ModFlowSimulation = ModFlowSimulation(**parameters)

    assert sim.boundary_heads is not None
    assert len(sim.boundary_heads) > 0

    for _ in range(3):
        sim.step()

    # Verify that the boundary heads have a measurable physical effect on interior active cells
    decomp_heads = decompress(sim.heads, sim.basin_mask)
    head_bnd_interior: float = float(decomp_heads[0, 1, 5])
    assert head_bnd_interior > head_base_interior + 0.1, (
        f"Expected boundary heads to raise interior cell head from {head_base_interior} "
        f"to higher than {head_base_interior + 0.1}, but got {head_bnd_interior}"
    )

    # 2. Test dynamically updating boundary heads via BMI setter
    new_boundary_heads = np.full_like(sim.boundary_heads, 60.0)
    sim.set_boundary_heads(new_boundary_heads)
    np.testing.assert_allclose(sim.boundary_heads, 60.0)
    sim.step()

    sim.finalize()

    # 3. Test with custom boundary mask for outside boundary cell adjacent to (row 9, col 11)
    custom_bnd_mask = np.zeros((YSIZE + 2, XSIZE + 2), dtype=bool)
    custom_bnd_mask[9 + 1, 11 + 2] = True
    custom_bnd_heads = np.full((NLAY, YSIZE + 2, XSIZE + 2), 45.0, dtype=np.float64)
    parameters_custom: ModFlowParams = deepcopy(default_params)
    parameters_custom["working_directory"] = tmp_folder / "modflow_boundary_custom_test"
    parameters_custom["boundary_heads"] = custom_bnd_heads
    parameters_custom["boundary_mask"] = custom_bnd_mask
    sim_custom: ModFlowSimulation = ModFlowSimulation(**parameters_custom)

    assert len(sim_custom.boundary_heads) == NLAY
    np.testing.assert_allclose(sim_custom.boundary_heads, 45.0)

    sim_custom.step()
    sim_custom.finalize()


def test_modflow_groundwater_level_boundary_edge_cases() -> None:
    """Test boundary condition error handling and edge cases.

    Verifies that invalid boundary head/mask shapes raise ValueError, and attempting
    to set boundary heads with NaN or wrong shape raises AssertionError.
    """
    # 1. Invalid boundary_heads shape raises ValueError
    params: dict[str, Any] = dict(deepcopy(default_params))
    params["working_directory"] = tmp_folder / "modflow_boundary_err_heads"
    params["boundary_heads"] = np.ones((5, 5, 5, 5))
    with pytest.raises(ValueError, match="boundary_heads must have shape"):
        ModFlowSimulation(**params)

    # 2. Invalid boundary_mask shape raises ValueError
    params_mask: dict[str, Any] = dict(deepcopy(default_params))
    params_mask["working_directory"] = tmp_folder / "modflow_boundary_err_mask"
    params_mask["boundary_mask"] = np.ones((5, 5, 5), dtype=bool)
    with pytest.raises(ValueError, match="boundary_mask must have shape"):
        ModFlowSimulation(**params_mask)

    # 3. Setting boundary heads with NaN raises ValueError
    sim = ModFlowSimulation(**default_params)
    with pytest.raises(ValueError, match="NaN"):
        sim.set_boundary_heads(np.full_like(sim.boundary_heads, np.nan))

    # 4. Setting boundary heads with incorrect shape raises ValueError
    with pytest.raises(ValueError, match="shape"):
        sim.set_boundary_heads(np.array([10.0]))

    sim.finalize()

    # 5. Boundary cell inside active domain raises ValueError
    params_inside: dict[str, Any] = dict(deepcopy(default_params))
    params_inside["working_directory"] = tmp_folder / "modflow_boundary_err_inside"
    bad_mask = np.zeros((YSIZE + 2, XSIZE + 2), dtype=bool)
    bad_mask[9 + 1, 11 + 1] = True  # inside active cell (9, 11)
    params_inside["boundary_mask"] = bad_mask
    with pytest.raises(ValueError, match="strictly outside"):
        ModFlowSimulation(**params_inside)


def test_modflow_boundary_flows_inflow_and_water_balance() -> None:
    """Test constant-head boundary inflow and water balance closure in absolute terms.

    Verifies that when boundary heads exceed interior groundwater levels:
    1. Inflow rates (m3/step) are strictly positive and outflows are zero.
    2. Simulated net boundary flow equals inflow minus outflow.
    3. Per-active-cell fluxes map exclusively to the designated boundary cells.
    4. The domain water balance is closed in absolute volumetric terms:
       storage change (m3) equals the net boundary influx within < 0.001 m3.
    """
    parameters: ModFlowParams = deepcopy(default_params)
    parameters["working_directory"] = tmp_folder / "modflow_wb_chd_inflow"
    parameters["heads"] = parameters["heads"] - 2
    parameters["boundary_heads"] = parameters["boundary_heads"] - 2

    # Prescribe boundary head 0.5 m higher than domain heads at outside boundary cell
    parameters["boundary_heads"][:, 9 + 1, 11 + 2] += 0.5

    sim: ModFlowSimulation = ModFlowSimulation(**parameters)
    initial_storage_m3: float = float(sim.groundwater_content_m3.sum())
    sim.step()
    final_storage_m3: float = float(sim.groundwater_content_m3.sum())

    boundary_flows_m3: ArrayFloat64 = sim.boundary_flow_m3
    boundary_inflows_m3: ArrayFloat64 = sim.boundary_inflow_m3
    boundary_outflows_m3: ArrayFloat64 = sim.boundary_outflow_m3

    # Verify vector dimensions match the number of active boundary conditions (NLAY)
    assert len(boundary_flows_m3) == NLAY
    assert len(boundary_inflows_m3) == NLAY
    assert len(boundary_outflows_m3) == NLAY

    # Check individual flux properties in absolute terms
    np.testing.assert_allclose(
        boundary_flows_m3, boundary_inflows_m3 - boundary_outflows_m3
    )
    assert (boundary_inflows_m3 > 0.0).all()
    assert (boundary_outflows_m3 == 0.0).all()
    np.testing.assert_allclose(boundary_flows_m3, boundary_inflows_m3)

    # Water balance closure in absolute volumetric terms (m3)
    storage_change_m3: float = final_storage_m3 - initial_storage_m3
    recharge_volume_m3: float = float(sim.recharge_m3.sum())
    drainage_volume_m3: float = float(sim.drainage_m3.sum())
    total_boundary_flux_m3: float = float(boundary_flows_m3.sum())
    net_flux_m3: float = (
        recharge_volume_m3 - drainage_volume_m3 + total_boundary_flux_m3
    )
    water_balance_discrepancy_m3: float = abs(storage_change_m3 - net_flux_m3)

    assert storage_change_m3 > 0.0
    assert math.isclose(storage_change_m3, net_flux_m3, rel_tol=1e-5, abs_tol=1e-3)
    assert water_balance_discrepancy_m3 < 1e-3

    sim.finalize()


def test_modflow_boundary_flows_outflow_and_water_balance() -> None:
    """Test constant-head boundary outflow and water balance closure in absolute terms.

    Verifies that when boundary heads are below interior groundwater levels:
    1. Inflow rates are zero and outflow rates (m3/step) are strictly positive.
    2. Simulated net boundary flow is negative and equals -outflow.
    3. Per-active-cell fluxes map exclusively to the designated boundary cells.
    4. Storage change is negative and matches net boundary outflow within < 0.001 m3.
    """
    parameters: ModFlowParams = deepcopy(default_params)
    parameters["working_directory"] = tmp_folder / "modflow_wb_chd_outflow"
    parameters["heads"] = parameters["heads"] - 2
    parameters["boundary_heads"] = parameters["boundary_heads"] - 2

    # Prescribe boundary head 0.5 m lower than domain heads at outside boundary cell
    parameters["boundary_heads"][:, 9 + 1, 11 + 2] -= 0.5

    sim: ModFlowSimulation = ModFlowSimulation(**parameters)
    initial_storage_m3: float = float(sim.groundwater_content_m3.sum())
    sim.step()
    final_storage_m3: float = float(sim.groundwater_content_m3.sum())

    boundary_flows_m3: ArrayFloat64 = sim.boundary_flow_m3
    boundary_inflows_m3: ArrayFloat64 = sim.boundary_inflow_m3
    boundary_outflows_m3: ArrayFloat64 = sim.boundary_outflow_m3

    assert len(boundary_flows_m3) == NLAY
    assert len(boundary_inflows_m3) == NLAY
    assert len(boundary_outflows_m3) == NLAY

    np.testing.assert_allclose(
        boundary_flows_m3, boundary_inflows_m3 - boundary_outflows_m3
    )
    assert (boundary_inflows_m3 == 0.0).all()
    assert (boundary_outflows_m3 > 0.0).all()
    assert (boundary_flows_m3 < 0.0).all()
    np.testing.assert_allclose(boundary_flows_m3, -boundary_outflows_m3)

    # Water balance closure in absolute volumetric terms (m3)
    storage_change_m3: float = final_storage_m3 - initial_storage_m3
    recharge_volume_m3: float = float(sim.recharge_m3.sum())
    drainage_volume_m3: float = float(sim.drainage_m3.sum())
    total_boundary_flux_m3: float = float(boundary_flows_m3.sum())
    net_flux_m3: float = (
        recharge_volume_m3 - drainage_volume_m3 + total_boundary_flux_m3
    )
    water_balance_discrepancy_m3: float = abs(storage_change_m3 - net_flux_m3)

    assert storage_change_m3 < 0.0
    assert math.isclose(storage_change_m3, net_flux_m3, rel_tol=1e-5, abs_tol=1e-3)
    assert water_balance_discrepancy_m3 < 1e-3

    sim.finalize()


def test_modflow_boundary_flows_simultaneous_inflow_outflow_water_balance() -> None:
    """Test boundary fluxes with concurrent inflow and outflow cells and closed water balance.

    Configures two constant-head cells simultaneously:
    - One cell with prescribed head higher than interior domain (inflow).
    - Another cell with prescribed head lower than interior domain (outflow).

    Verifies that:
    1. Both inflow and outflow rates (m3/step) are non-zero.
    2. Simulated net boundary flow vector equals inflow minus outflow.
    3. Active cell mapping correctly distinguishes inflow and outflow locations.
    4. Total storage change balances the combined net fluxes within < 0.001 m3.
    """
    parameters: ModFlowParams = deepcopy(default_params)
    parameters["working_directory"] = tmp_folder / "modflow_wb_chd_simultaneous"
    parameters["heads"] = parameters["heads"] - 2

    # Two boundary cells strictly outside the active domain:
    # (row 10, col 13) adjacent to active cell (9, 11) for inflow
    # (row 1, col 6) adjacent to active cell (1, 5) for outflow (row 0 in domain is inactive)
    bnd_mask: TwoDArrayBool = np.zeros((YSIZE + 2, XSIZE + 2), dtype=bool)
    bnd_mask[9 + 1, 11 + 2] = True
    bnd_mask[1, 5 + 1] = True

    decomp_heads: ThreeDArrayFloat64 = decompress(
        parameters["heads"], parameters["basin_mask"]
    )

    bnd_heads: ThreeDArrayFloat64 = np.zeros(
        (NLAY, YSIZE + 2, XSIZE + 2), dtype=np.float64
    )
    bnd_heads[:, 9 + 1, 11 + 2] = decomp_heads[:, 9, 11] + 0.5
    bnd_heads[:, 1, 5 + 1] = decomp_heads[:, 1, 5] - 0.5

    parameters["boundary_mask"] = bnd_mask
    parameters["boundary_heads"] = bnd_heads

    sim: ModFlowSimulation = ModFlowSimulation(**parameters)
    initial_storage_m3: float = float(sim.groundwater_content_m3.sum())
    sim.step()
    final_storage_m3: float = float(sim.groundwater_content_m3.sum())

    boundary_flows_m3: ArrayFloat64 = sim.boundary_flow_m3
    boundary_inflows_m3: ArrayFloat64 = sim.boundary_inflow_m3
    boundary_outflows_m3: ArrayFloat64 = sim.boundary_outflow_m3

    # 2 boundary cells * NLAY layers = 4 boundary conditions
    assert len(boundary_flows_m3) == 2 * NLAY
    assert len(boundary_inflows_m3) == 2 * NLAY
    assert len(boundary_outflows_m3) == 2 * NLAY

    np.testing.assert_allclose(
        boundary_flows_m3, boundary_inflows_m3 - boundary_outflows_m3
    )
    assert float(boundary_inflows_m3.sum()) > 0.0
    assert float(boundary_outflows_m3.sum()) > 0.0

    storage_change_m3: float = final_storage_m3 - initial_storage_m3
    recharge_volume_m3: float = float(sim.recharge_m3.sum())
    drainage_volume_m3: float = float(sim.drainage_m3.sum())
    total_boundary_flux_m3: float = float(boundary_flows_m3.sum())
    net_flux_m3: float = (
        recharge_volume_m3 - drainage_volume_m3 + total_boundary_flux_m3
    )
    water_balance_discrepancy_m3: float = abs(storage_change_m3 - net_flux_m3)

    assert math.isclose(storage_change_m3, net_flux_m3, rel_tol=1e-5, abs_tol=1e-3)
    assert water_balance_discrepancy_m3 < 1e-3

    sim.finalize()


def test_modflow_boundary_flows_combined_fluxes_and_dynamic_update() -> None:
    """Test boundary fluxes with concurrent recharge and dynamic boundary head update.

    Verifies that:
    1. Multi-step simulation maintains water balance when boundary flows and surface recharge coincide.
    2. Dynamic update of boundary heads via set_boundary_heads() updates boundary fluxes.
    3. Subsequent timesteps maintain water balance closure in absolute terms (< 0.001 m3).
    """
    parameters: ModFlowParams = deepcopy(default_params)
    parameters["working_directory"] = tmp_folder / "modflow_wb_chd_dynamic"
    parameters["heads"] = parameters["heads"] - 2
    parameters["boundary_heads"] = parameters["boundary_heads"] - 2

    sim: ModFlowSimulation = ModFlowSimulation(**parameters)

    # Initial time step in steady equilibrium
    sim.step()

    # Dynamically raise boundary head by 0.25 m
    updated_boundary_heads: ArrayFloat64 = sim.boundary_heads.copy() + 0.25
    sim.set_boundary_heads(updated_boundary_heads)

    for step_index in range(3):
        step_storage_pre_m3: float = float(sim.groundwater_content_m3.sum())

        # Apply recharge to interior cells
        recharge_m: TwoDArrayFloat64 = np.zeros((YSIZE, XSIZE), dtype=np.float64)
        recharge_m[3:6, 3:6] = 0.005
        recharge_m3: TwoDArrayFloat64 = recharge_m * cell_area
        sim.set_recharge_m3(compress(recharge_m3, sim.basin_mask))

        sim.step()
        step_storage_post_m3: float = float(sim.groundwater_content_m3.sum())

        boundary_flows_m3: ArrayFloat64 = sim.boundary_flow_m3
        boundary_inflows_m3: ArrayFloat64 = sim.boundary_inflow_m3
        boundary_outflows_m3: ArrayFloat64 = sim.boundary_outflow_m3

        np.testing.assert_allclose(
            boundary_flows_m3, boundary_inflows_m3 - boundary_outflows_m3
        )
        assert float(boundary_inflows_m3.sum()) > 0.0

        # Water balance must be closed at each step, including the dynamic update step
        step_balance_pre_m3: float = (
            step_storage_pre_m3
            + float(sim.recharge_m3.sum())
            - float(sim.drainage_m3.sum())
            + float(boundary_flows_m3.sum())
        )
        discrepancy_m3: float = abs(step_balance_pre_m3 - step_storage_post_m3)
        assert math.isclose(
            step_balance_pre_m3, step_storage_post_m3, rel_tol=1e-5, abs_tol=1e-3
        )
        assert discrepancy_m3 < 1e-3

    sim.finalize()


def test_groundwater_step_epsilon_safeguard() -> None:
    """Test that minor epsilon excesses are clamped to available storage while large excesses raise AssertionError."""
    from unittest.mock import MagicMock

    from geb.hydrology.groundwater import GroundWater

    gw: GroundWater = GroundWater.__new__(GroundWater)
    gw.grid = MagicMock()
    gw.grid.var.cell_area = np.array([1000.0, 1000.0], dtype=np.float32)
    gw.modflow = MagicMock()
    gw.modflow.available_groundwater_m3 = np.array([100.0, 200.0], dtype=np.float64)
    gw.modflow.groundwater_content_m3 = np.array([500.0, 500.0], dtype=np.float64)
    gw.modflow.drainage_m3 = np.array([0.0, 0.0], dtype=np.float64)
    gw.modflow.boundary_inflow_m3 = np.array([0.0, 0.0], dtype=np.float64)
    gw.modflow.boundary_outflow_m3 = np.array([0.0, 0.0], dtype=np.float64)
    gw.report = MagicMock()

    # 1. Minor excess within epsilon (e.g. 1e-6 m3 over available_groundwater_m3)
    recharge: ArrayFloat32 = np.array([0.0, 0.0], dtype=np.float32)
    abstraction: ArrayFloat32 = np.array([100.000001, 200.0], dtype=np.float32)
    capillary: ArrayFloat32 = np.array([0.0, 0.0], dtype=np.float32)

    gw.step(
        groundwater_recharge_m=recharge,
        groundwater_abstraction_m3=abstraction,
        capillary_rise_m=capillary,
    )
    # The set_groundwater_abstraction_m3 should have received clamped value 100.0
    passed_abstraction: ArrayFloat64 = (
        gw.modflow.set_groundwater_abstraction_m3.call_args[0][0]
    )
    assert passed_abstraction[0] == 100.0
    assert passed_abstraction[1] == 200.0

    # 2. Large excess exceeding epsilon tolerance (e.g. 50 m3 over available_groundwater_m3)
    large_abstraction: ArrayFloat32 = np.array([150.0, 200.0], dtype=np.float32)
    with pytest.raises(AssertionError, match="exceeds available groundwater storage"):
        gw.step(
            groundwater_recharge_m=recharge,
            groundwater_abstraction_m3=large_abstraction,
            capillary_rise_m=capillary,
        )
