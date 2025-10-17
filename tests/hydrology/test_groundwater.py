"""Tests for the groundwater module of GEB.

The groundwater module uses MODFLOW to simulate groundwater flow and interactions
with surface water and the unsaturated zone.
"""

import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from affine import Affine

from geb.build.workflows.general import calculate_cell_area
from geb.hydrology.groundwater.model import (
    ModFlowSimulation,
    distribute_well_abstraction_m3_per_layer,
    get_groundwater_storage_m,
    get_water_table_depth,
)
from geb.workflows.raster import compress

from ..testconfig import output_folder, tmp_folder


def decompress(
    array: npt.NDArray[Any], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[Any]:
    """Decompress an array using the basin mask.

    Args:
        array: The array to decompress.
        mask: The basin mask.

    Returns:
        The decompressed array.
    """
    if array.ndim == 1:
        out = np.full(mask.shape, np.nan)
    elif array.ndim == 2:
        out = np.full((array.shape[0], *mask.shape), np.nan)
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

cell_area = calculate_cell_area(Affine.from_gdal(*gt), (YSIZE, XSIZE))


layer_boundary_elevation = np.full((NLAY + 1, YSIZE, XSIZE), np.nan, dtype=np.float32)
layer_boundary_elevation[0] = topography
for layer in range(1, NLAY + 1):
    layer_boundary_elevation[layer] = (
        layer_boundary_elevation[layer - 1] - 5
    )  # each layer is 5 m thick

heads = np.full((NLAY, YSIZE, XSIZE), 0, dtype=np.float32)
for layer in range(NLAY):
    heads[layer] = topography - 2


class DummyGrid:
    """A dummy grid class to simulate the grid structure."""

    def __init__(self) -> None:
        """Initializes a DummyGrid instance of the GEB grid with required attributes for the MODFLOW simulation to work."""
        pass

    def decompress(self, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Decompress an array from 1D to 2D using the basin mask.

        Args:
            array: The compressed array.

        Returns:
            The decompressed array.
        """
        return decompress(array, basin_mask)


class DummyHydrology:
    """A dummy hydrology class to simulate the hydrology structure."""

    def __init__(self) -> None:
        """Initializes a DummyHydrology instance of the GEB hydrology with required attributes for the MODFLOW simulation to work."""
        self.grid = DummyGrid()


class DummyModel:
    """A dummy model class to simulate the MODFLOW model structure."""

    def __init__(self) -> None:
        """Initializes a DummyModel instance of the GEB model with required attributes for the MODFLOW simulation to work.

        For testing purposes only.
        """
        self.simulation_root_spinup = tmp_folder / "modflow"
        self.hydrology = DummyHydrology()

    @property
    def bin_folder(self) -> Path:
        """Gets the folder where MODFLOW binaries are stored.

        Returns:
            Path to the folder with MODFLOW binaries.
        """
        return Path(os.environ.get("GEB_PACKAGE_DIR")) / "bin"


default_params = {
    "model": DummyModel(),
    "gt": gt,
    "specific_storage": compress(np.full((NLAY, YSIZE, XSIZE), 0), basin_mask),
    "specific_yield": compress(np.full((NLAY, YSIZE, XSIZE), 0.8), basin_mask),
    "topography": compress(topography, basin_mask),
    "layer_boundary_elevation": compress(layer_boundary_elevation, basin_mask),
    "basin_mask": basin_mask,
    "heads": compress(heads, basin_mask),
    "hydraulic_conductivity": compress(np.full((NLAY, YSIZE, XSIZE), 1), basin_mask),
    "verbose": True,
    "never_load_from_disk": True,
    "heads_update_callback": lambda heads: None,
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

    balance_pre = groundwater_content_prev + recharge_m3 - drainage_m3
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

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = sim.groundwater_content_m3.sum()

    recharge_m = np.full((YSIZE, XSIZE), 0.1)
    recharge_m3 = recharge_m * cell_area
    sim.set_recharge_m3(compress(recharge_m3, sim.basin_mask))
    sim.step()

    drainage_m3 = np.nansum(sim.drainage_m3)
    assert np.nansum(drainage_m3) == 0
    groundwater_content = np.nansum(sim.groundwater_content_m3)

    assert np.nansum(sim.recharge_m) == np.nansum(sim.recharge_m3 / sim.area)

    recharge_m3 = np.nansum(sim.recharge_m3)
    balance_pre = groundwater_content_prev + recharge_m3 - drainage_m3
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

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = np.nansum(sim.groundwater_content_m3)

    recharge_m = np.full((YSIZE, XSIZE), 0.1)
    recharge_m3 = recharge_m * cell_area

    sim.set_recharge_m3(compress(recharge_m3, sim.basin_mask))
    sim.step()

    drainage = np.nansum(sim.drainage_m3)
    assert drainage.sum() > 0

    assert math.isclose(np.nansum(sim._drainage_m * sim.area), np.nansum(drainage))

    groundwater_content = np.nansum(sim.groundwater_content_m3)

    balance_pre = groundwater_content_prev - drainage + sim.recharge_m3.sum()
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, rel_tol=1e-5)

    sim.finalize()

    parameters["heads"][0] = layer_boundary_elevation[0] - 1
    parameters["heads"][1] = layer_boundary_elevation[0] - 1

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
    balance_pre = groundwater_content_prev - total_abstraction
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, rel_tol=1e-6)

    for _ in range(100):
        sim.set_groundwater_abstraction_m3(
            np.minimum(groundwater_abstracton, sim.available_groundwater_m3)
        )
        sim.step()

    sim.finalize()


def visualize_modflow_results(
    sim: ModFlowSimulation, axes: npt.NDArray[plt.Axes]
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
    storage = get_groundwater_storage_m(layer_boundary_elevation, head, specific_yield)
    np.testing.assert_allclose(storage, np.array([37.5, 32.5, 12.5, 10, 0]))
    storage = get_groundwater_storage_m(
        layer_boundary_elevation, head, specific_yield, min_remaining_layer_storage_m=1
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
