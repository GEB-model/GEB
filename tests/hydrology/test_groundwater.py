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

output_folder_groundwater: Path = output_folder / "groundwater"
output_folder_groundwater.mkdir(exist_ok=True, parents=True)

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
    between groundwater content, recharge and boundary flow.
    """
    parameters = deepcopy(default_params)

    sim: ModFlowSimulation = ModFlowSimulation(**parameters)

    groundwater_content_prev = sim.groundwater_content_m3.sum()
    sim.step()

    groundwater_content = sim.groundwater_content_m3.sum()
    recharge_m3 = sim.recharge_m3.sum()

    balance_pre = groundwater_content_prev + recharge_m3 + sim.boundary_flow_m3.sum()
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

    groundwater_content = np.nansum(sim.groundwater_content_m3)

    assert np.nansum(sim.recharge_m) == np.nansum(sim.recharge_m3 / sim.area)

    recharge_m3 = np.nansum(sim.recharge_m3)
    balance_pre = groundwater_content_prev + recharge_m3 + sim.boundary_flow_m3.sum()
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, abs_tol=1, rel_tol=1e-5)

    sim.finalize()


def test_wells() -> None:
    """Test groundwater well abstraction functionality.

    Verifies that well abstraction correctly removes water from
    groundwater storage and maintains water balance.
    """
    parameters = deepcopy(default_params)
    parameters["heads"][:,] = compress(
        topography - 2, basin_mask
    )  # set head below the land surface

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

    Plots the topography, groundwater head of both layers, groundwater depth and
    recharge, on axes 1 to 5 respectively.

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

    # Plot recharge
    im5 = ax5.imshow(decompress(sim.recharge_m, sim.basin_mask), cmap="Blues")
    ax5.set_title("Recharge")
    plt.colorbar(im5, ax=ax5, label="Recharge (m/day)")


def test_modflow_simulation_with_visualization() -> None:
    """Test MODFLOW simulation with visualization output.

    Runs a simulation with random recharge and abstraction,
    generating visualization plots of the results.
    """
    parameters = deepcopy(default_params)
    # below the land surface: there is no drain in MODFLOW to remove water above it
    parameters["heads"][:,] = compress(topography - 2, basin_mask)
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
    plt.savefig(output_folder_groundwater / "modflow_simulation.png")


def test_modflow_simulation_with_restore() -> None:
    """Test MODFLOW simulation state restoration.

    Verifies that the simulation can be restored to a previous
    state and continue simulation correctly.
    """
    parameters = deepcopy(default_params)
    # below the land surface: there is no drain in MODFLOW to remove water above it
    parameters["heads"][:,] = compress(topography - 2, basin_mask)
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
    total_boundary_flux_m3: float = float(boundary_flows_m3.sum())
    net_flux_m3: float = recharge_volume_m3 + total_boundary_flux_m3
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
    total_boundary_flux_m3: float = float(boundary_flows_m3.sum())
    net_flux_m3: float = recharge_volume_m3 + total_boundary_flux_m3
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
    total_boundary_flux_m3: float = float(boundary_flows_m3.sum())
    net_flux_m3: float = recharge_volume_m3 + total_boundary_flux_m3
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
    gw.update_drain_conductance = MagicMock()
    gw.modflow.available_groundwater_m3 = np.array([100.0, 200.0], dtype=np.float64)
    gw.modflow.groundwater_content_m3 = np.array([500.0, 500.0], dtype=np.float64)
    gw.modflow.boundary_inflow_m3 = np.array([0.0, 0.0], dtype=np.float64)
    gw.modflow.boundary_outflow_m3 = np.array([0.0, 0.0], dtype=np.float64)
    gw.report = MagicMock()
    gw.model = MagicMock(timing=False)

    # 1. Minor excess within epsilon (e.g. 1e-6 m3 over available_groundwater_m3)
    recharge: ArrayFloat32 = np.array([0.0, 0.0], dtype=np.float32)
    abstraction: ArrayFloat32 = np.array([100.000001, 200.0], dtype=np.float32)
    capillary: ArrayFloat32 = np.array([0.0, 0.0], dtype=np.float32)

    gw.step(
        groundwater_recharge_m=recharge,
        groundwater_abstraction_m3=abstraction,
        capillary_rise_m=capillary,
        river_seepage_m3=np.zeros(2, dtype=np.float64),
        river_drainage_m3=np.zeros(2, dtype=np.float64),
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
            river_seepage_m3=np.zeros(2, dtype=np.float64),
            river_drainage_m3=np.zeros(2, dtype=np.float64),
        )


def test_get_groundwater_storage_m_thin_full_layer() -> None:
    """A full layer thinner than the minimum remaining storage gives no negative storage.

    Two-layer models can have an upper layer of 0.1 m (in float32 just below 0.1),
    the same as the minimum remaining storage used for available groundwater.
    """
    layer_boundary_elevation = np.array([[100.0], [np.float32(99.9)], [50.0]])
    heads = np.array([[100.5], [100.5]])  # both layers full
    specific_yield = np.array([[0.1], [0.1]])
    specific_storage = np.zeros((2, 1))
    available = get_groundwater_storage_m(
        layer_boundary_elevation,
        heads,
        specific_yield,
        specific_storage,
        min_remaining_layer_storage_m=0.1,
    )
    lower_layer_available = (99.9 - 50.0 - 0.1) * 0.1
    assert available[0] >= 0.0
    assert available[0] == pytest.approx(lower_layer_available, abs=1e-6)


def test_hydraulic_conductivity_multiplier_applied_at_run_time() -> None:
    """The stored conductivity has no multiplier; every user gets the configured one."""
    from types import SimpleNamespace

    from geb.hydrology.groundwater import GroundWater

    groundwater = GroundWater.__new__(GroundWater)
    groundwater.model = SimpleNamespace(  # ty:ignore[invalid-assignment]
        config={
            "hydrology": {"groundwater": {"hydraulic_conductivity_multiplier": 3.0}}
        }
    )
    groundwater.grid = SimpleNamespace(  # ty:ignore[invalid-assignment]
        var=SimpleNamespace(
            groundwater_hydraulic_conductivity_m_per_day=np.array(
                [[1.0, 2.0]], dtype=np.float32
            )
        )
    )
    conductivity = groundwater.hydraulic_conductivity_m_per_day
    assert conductivity.dtype == np.float32
    np.testing.assert_allclose(conductivity, [[3.0, 6.0]])
    # the stored value is unchanged
    np.testing.assert_allclose(
        groundwater.grid.var.groundwater_hydraulic_conductivity_m_per_day, [[1.0, 2.0]]
    )


def test_drain_groundwater_to_rivers_exact_exponential() -> None:
    """The head above the drain level decays exponentially and never drops below it."""
    from geb.hydrology.groundwater import drain_groundwater_to_rivers

    head = np.array([12.0, 9.0, 15.0], dtype=np.float64)
    drain = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    # cell 0: conductance = storage coefficient -> exp(-1); cell 2: huge conductance
    conductance = np.array([1000.0, 1000.0, 1e9], dtype=np.float64)
    storage = np.array([1000.0, 1000.0, 1000.0], dtype=np.float64)
    drainage = np.zeros(3, dtype=np.float64)
    remaining = np.full(3, 1e9, dtype=np.float64)
    drain_groundwater_to_rivers(head, drain, conductance, storage, drainage, remaining)
    np.testing.assert_allclose(head[0], 10.0 + 2.0 * np.exp(-1.0), rtol=1e-6)
    np.testing.assert_allclose(
        drainage[0], 2.0 * (1 - np.exp(-1.0)) * 1000.0, rtol=1e-5
    )
    # head below the drain level: no drainage (the river may seep into the aquifer)
    assert head[1] == 9.0 and drainage[1] == 0.0
    # stiff drain: head reaches the drain level, the drained volume is the excess storage
    np.testing.assert_allclose(head[2], 10.0, atol=1e-5)
    np.testing.assert_allclose(drainage[2], 5.0 * 1000.0, rtol=1e-6)
    np.testing.assert_allclose(remaining[2], 1e9 - 5.0 * 1000.0, rtol=1e-9)


def test_response_time_conductance_drains_in_the_response_time() -> None:
    """With the response-time conductance the head above the drain falls to 1/e in that time."""
    from geb.hydrology.groundwater import (
        drain_groundwater_to_rivers,
        get_response_time_conductance,
    )

    specific_yield = np.array([0.03, 0.075, 0.28], dtype=np.float32)
    cell_area_m2 = np.array([2.0e5, 3.0e5, 4.0e5], dtype=np.float32)
    response_time_days = 10.0
    conductance_m2_per_day = get_response_time_conductance(
        specific_yield, cell_area_m2, response_time_days
    )
    assert conductance_m2_per_day.dtype == np.float32
    head = np.array([12.0, 11.0, 14.0], dtype=np.float64)
    drain = np.full(3, 10.0, dtype=np.float64)
    excess_start = head - drain
    storage = (specific_yield * cell_area_m2).astype(np.float64)
    drainage = np.zeros(3, dtype=np.float64)
    remaining = np.full(3, 1e12, dtype=np.float64)
    conductance_m2_per_hour = conductance_m2_per_day.astype(np.float64) / 24.0
    for _ in range(int(response_time_days * 24)):
        drain_groundwater_to_rivers(
            head, drain, conductance_m2_per_hour, storage, drainage, remaining
        )
    # the same share is left in every cell, whatever its specific yield and area
    np.testing.assert_allclose((head - drain) / excess_start, np.exp(-1.0), rtol=1e-5)


def test_aquifer_drain_conductance_drains_in_the_aquifer_drainage_time() -> None:
    """The aquifer drain empties a cell in 4 Sy L^2 / (pi^2 T) (Kraijenhoff van de Leur)."""
    from geb.hydrology.groundwater import (
        drain_groundwater_to_rivers,
        get_aquifer_drain_conductance,
    )

    specific_yield = np.float32(0.1)
    # 1 m of aquifer below the stream, thinner than its wetted perimeter: no radial
    # resistance, only horizontal flow
    conductivity = np.array([50.0, 5.0], dtype=np.float32)  # m/day
    thickness = np.ones(2, dtype=np.float32)
    transmissivity = conductivity * thickness
    hillslope_length = np.array([300.0, 300.0], dtype=np.float32)  # m
    cell_area = np.array([1e6, 1e6], dtype=np.float32)
    conductance = get_aquifer_drain_conductance(
        hydraulic_conductivity_m_per_day=conductivity,
        thickness_below_stream_m=thickness,
        height_above_stream_m=np.zeros(2, dtype=np.float32),
        hillslope_length_m=hillslope_length,
        cell_area_m2=cell_area,
    )
    drainage_time_days = (
        4.0 * specific_yield * hillslope_length**2 / (np.pi**2 * transmissivity)
    )
    # ten times less transmissive rock drains ten times more slowly
    np.testing.assert_allclose(drainage_time_days[1] / drainage_time_days[0], 10.0)

    head = np.array([12.0, 12.0], dtype=np.float64)
    drain = np.full(2, 10.0, dtype=np.float64)
    storage = (specific_yield * cell_area).astype(np.float64)
    drainage = np.zeros(2, dtype=np.float64)
    remaining = np.full(2, 1e12, dtype=np.float64)
    conductance_m2_per_hour = conductance.astype(np.float64) / 24.0
    for _ in range(int(round(drainage_time_days[0] * 24))):
        drain_groundwater_to_rivers(
            head, drain, conductance_m2_per_hour, storage, drainage, remaining
        )
    # after one drainage time of the first cell its head above the drain is 1 / e
    np.testing.assert_allclose((head[0] - drain[0]) / 2.0, np.exp(-1.0), rtol=0.01)


def test_karst_drain_conductance_follows_the_karst_fraction() -> None:
    """The karst part of a cell drains with the karst conductance, the rest with the matrix."""
    from geb.hydrology.groundwater import get_karst_drain_conductance

    conductance = get_karst_drain_conductance(
        matrix_conductance_m2_per_day=np.full(4, 10.0, dtype=np.float32),
        karst_conductance_m2_per_day=np.full(4, 100.0, dtype=np.float32),
        karst_fraction=np.array([0.0, 0.5, 1.0, 1.2], dtype=np.float32),
    )
    np.testing.assert_allclose(conductance, [10.0, 55.0, 100.0, 100.0], rtol=1e-6)
    assert conductance.dtype == np.float32


def test_drain_groundwater_to_rivers_limited_by_storage() -> None:
    """A cell does not drain more than the groundwater it still holds."""
    from geb.hydrology.groundwater import drain_groundwater_to_rivers

    head = np.array([15.0], dtype=np.float64)
    drain = np.array([10.0], dtype=np.float64)
    conductance = np.array([1e9], dtype=np.float64)
    storage = np.array([1000.0], dtype=np.float64)
    drainage = np.zeros(1, dtype=np.float64)
    # 5 m above the drain level would give 5000 m3, but only 2000 m3 can be taken
    remaining = np.array([2000.0], dtype=np.float64)
    drain_groundwater_to_rivers(head, drain, conductance, storage, drainage, remaining)
    np.testing.assert_allclose(drainage[0], 2000.0, rtol=1e-9)
    np.testing.assert_allclose(head[0], 13.0, rtol=1e-9)
    assert remaining[0] == 0.0
    # nothing left: no drainage in the next hour
    drain_groundwater_to_rivers(head, drain, conductance, storage, drainage, remaining)
    assert drainage[0] == 0.0 and head[0] == 13.0


def test_get_conductivity_depth_decay_factor() -> None:
    """The conductivity of a layer decreases with its depth and with the slope."""
    from geb.hydrology.groundwater import get_conductivity_depth_decay_factor

    slope = np.array([0.0, 0.1, 0.3])
    # one layer of 100 m
    boundaries = np.array([[100.0, 100.0, 100.0], [0.0, 0.0, 0.0]])
    factor = get_conductivity_depth_decay_factor(boundaries, slope)
    assert factor.shape == (1, 3)
    # flat: e-folding depth 120 m; steep: 7.5 m; very steep: 5 m
    expected = [
        (1.5 + 120.0 * (1.0 - math.exp(-98.5 / 120.0))) / 100.0,
        (1.5 + 7.5 * (1.0 - math.exp(-98.5 / 7.5))) / 100.0,
        (1.5 + 5.0 * (1.0 - math.exp(-98.5 / 5.0))) / 100.0,
    ]
    np.testing.assert_allclose(factor[0], expected, rtol=1e-6)
    assert (np.diff(factor[0]) < 0).all()

    # a layer within the upper 1.5 m keeps its conductivity
    shallow = get_conductivity_depth_decay_factor(
        np.array([[10.0], [9.0]]), np.array([0.1])
    )
    np.testing.assert_allclose(shallow, 1.0)

    # two layers have the same transmissivity as one layer over the same depth
    two = get_conductivity_depth_decay_factor(
        np.array([[100.0], [60.0], [0.0]]), np.array([0.1])
    )
    one = get_conductivity_depth_decay_factor(
        np.array([[100.0], [0.0]]), np.array([0.1])
    )
    np.testing.assert_allclose(two[0] * 40.0 + two[1] * 60.0, one[0] * 100.0)
    assert two[1, 0] < two[0, 0]


def test_radial_resistance_matches_van_der_molen_wesseling() -> None:
    """Horizontal plus radial resistance give the exact steady state of parallel drains.

    Van der Molen and Wesseling (1991), Table 2 (exact solution, drain radius 0.05 m):
    thickness of Hooghoudt's equivalent layer d for aquifer thickness D and drain
    spacing S. In steady state the head midway is s S^2 / (8 K d), the horizontal part
    s S^2 / (8 K D) plus the radial part s c_radial.
    """
    from geb.hydrology.groundwater import get_aquifer_drain_conductance

    # (D, S, d) in m
    table = [
        (0.5, 10.0, 0.44),
        (1.0, 20.0, 0.81),
        (2.0, 20.0, 1.21),
        (2.0, 100.0, 1.77),
        (5.0, 40.0, 2.38),
        (5.0, 200.0, 4.10),
        (10.0, 25.0, 1.93),
        (10.0, 200.0, 6.54),
    ]
    for thickness, spacing, equivalent_depth in table:
        length = spacing / 2.0
        conductance = get_aquifer_drain_conductance(
            hydraulic_conductivity_m_per_day=np.array([1.0], dtype=np.float32),
            thickness_below_stream_m=np.array([thickness], dtype=np.float32),
            height_above_stream_m=np.array([0.0], dtype=np.float32),
            hillslope_length_m=np.array([length], dtype=np.float32),
            cell_area_m2=np.array([1.0], dtype=np.float32),
            stream_wetted_perimeter_m=np.pi * 0.05,
        )[0]
        radial_days = 1.0 / conductance - 4.0 * length**2 / (np.pi**2 * thickness)
        computed = spacing**2 / 8.0 / (spacing**2 / (8.0 * thickness) + radial_days)
        np.testing.assert_allclose(computed, equivalent_depth, rtol=0.02)


def test_radial_resistance_slows_thick_aquifers() -> None:
    """No radial resistance when the stream reaches the base; it grows with the thickness."""
    from geb.hydrology.groundwater import get_aquifer_drain_conductance

    thickness = np.array([0.1, 2.0, 50.0, 200.0, 2000.0], dtype=np.float32)
    length = np.full(5, 300.0, dtype=np.float32)
    conductance = get_aquifer_drain_conductance(
        hydraulic_conductivity_m_per_day=np.ones(5, dtype=np.float32),
        thickness_below_stream_m=thickness,
        height_above_stream_m=np.zeros(5, dtype=np.float32),
        hillslope_length_m=length,
        cell_area_m2=np.ones(5, dtype=np.float32),
    )
    horizontal_only = np.pi**2 * thickness / (4.0 * length**2)
    np.testing.assert_allclose(conductance[:2], horizontal_only[:2], rtol=1e-5)
    factor = conductance / horizontal_only
    assert np.all(np.diff(factor[1:]) < 0) and np.all(factor > 0)


def test_drainage_conductance_includes_flow_above_stream_level() -> None:
    """T = K (D + 2 h / pi) with h the mean head; with the stream at the layer base only h counts."""
    from types import SimpleNamespace

    from geb.hydrology.groundwater import GroundWater, get_aquifer_drain_conductance

    gw = GroundWater.__new__(GroundWater)
    gw.grid = SimpleNamespace(
        var=SimpleNamespace(
            cell_area=np.full(2, 1e6, dtype=np.float32),
            # top layer from 0 to 50 m in both cells
            layer_boundary_elevation=np.array(
                [[50.0, 50.0], [0.0, 0.0], [-50.0, -50.0]], dtype=np.float32
            ),
            groundwater_hydraulic_conductivity_m_per_day=np.full(
                (2, 2), 1.0, dtype=np.float32
            ),
        )
    )
    gw.hydrology = SimpleNamespace(
        karst=SimpleNamespace(karst_fraction=np.zeros(2, dtype=np.float32))
    )
    gw.model = SimpleNamespace(
        config={
            "hydrology": {"groundwater": {"hydraulic_conductivity_multiplier": 1.0}}
        }
    )
    # cell 0: stream at the layer base (D ~ 0); cell 1: 30 m of aquifer below the stream
    gw._drain_stream_level_m = np.array([0.1, 30.0], dtype=np.float32)
    gw._drain_hillslope_length_m = np.full(2, 300.0, dtype=np.float32)
    gw._drain_thickness_below_stream_m = np.array([0.1, 30.0], dtype=np.float32)
    gw._drain_karst_conductance_m2_per_day = np.zeros(2, dtype=np.float32)

    head = np.array([10.1, 40.0])  # 10 m above the stream in both cells
    conductance = gw.get_drainage_conductance(head)
    expected = get_aquifer_drain_conductance(
        hydraulic_conductivity_m_per_day=np.ones(2, dtype=np.float32),
        thickness_below_stream_m=gw._drain_thickness_below_stream_m,
        height_above_stream_m=np.full(2, 10.0, dtype=np.float32),
        hillslope_length_m=gw._drain_hillslope_length_m,
        cell_area_m2=gw.grid.var.cell_area,
    )
    np.testing.assert_allclose(conductance, expected, rtol=1e-6)
    # below the stream level only the aquifer below the stream conducts
    low = gw.get_drainage_conductance(np.array([0.0, 20.0]))
    assert low[0] < conductance[0] / 10 and low[1] < conductance[1]


def test_aquifer_drain_conductance_rejects_invalid_input() -> None:
    """Non-positive conductivity, thickness or hillslope length, or negative height, raise."""
    from geb.hydrology.groundwater import get_aquifer_drain_conductance

    valid = {
        "hydraulic_conductivity_m_per_day": np.ones(1, dtype=np.float32),
        "thickness_below_stream_m": np.ones(1, dtype=np.float32),
        "height_above_stream_m": np.zeros(1, dtype=np.float32),
        "hillslope_length_m": np.ones(1, dtype=np.float32),
        "cell_area_m2": np.ones(1, dtype=np.float32),
    }
    get_aquifer_drain_conductance(**valid)
    for name, value in [
        ("hydraulic_conductivity_m_per_day", 0.0),
        ("thickness_below_stream_m", 0.0),
        ("height_above_stream_m", -1.0),
        ("hillslope_length_m", np.nan),
    ]:
        with pytest.raises(ValueError):
            get_aquifer_drain_conductance(
                **{**valid, name: np.array([value], dtype=np.float32)}
            )
