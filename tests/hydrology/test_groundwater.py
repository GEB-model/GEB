import math
import numpy as np
import matplotlib.pyplot as plt
from geb.hydrology.groundwater.model import (
    ModFlowSimulation,
    get_water_table_depth,
    get_groundwater_storage_m,
    distribute_well_rate_per_layer,
)
from copy import deepcopy

from ..setup import output_folder, tmp_folder


def decompress(array, mask):
    if array.ndim == 1:
        out = np.full(mask.shape, np.nan)
    elif array.ndim == 2:
        out = np.full((array.shape[0], *mask.shape), np.nan)
    out[..., ~mask] = array
    return out


XSIZE = 12
YSIZE = 10
NLAY = 2

# Create a topography 2D map
x = np.linspace(-5, 5, XSIZE)
y = np.linspace(-5, 5, YSIZE)
x, y = np.meshgrid(x, y)

topography = np.exp2(-(x**2) - y**2 + 5) / 2
basin_mask = np.zeros((YSIZE, XSIZE), dtype=bool)
basin_mask[0] = True
basin_mask[-3:-1, 0:3] = True

gt = (4.864242872511027, 0.0001, 0, 52.33412139354429, 0, -0.0001)


def compress(array, mask):
    return array[..., ~mask]


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
    def __init__(self):
        pass

    def decompress(self, array):
        return decompress(array, basin_mask)


class DummyData:
    def __init__(self):
        self.grid = DummyGrid()


class DummyModel:
    def __init__(self):
        self.simulation_root = tmp_folder / "modflow"
        self.data = DummyData()


default_params = {
    "model": DummyModel(),
    "gt": gt,
    "ndays": 20,
    "specific_storage": compress(np.full((NLAY, YSIZE, XSIZE), 0), basin_mask),
    "specific_yield": compress(np.full((NLAY, YSIZE, XSIZE), 0.8), basin_mask),
    "topography": compress(topography, basin_mask),
    "layer_boundary_elevation": compress(layer_boundary_elevation, basin_mask),
    "basin_mask": basin_mask,
    "heads": compress(heads, basin_mask),
    "hydraulic_conductivity": compress(np.full((NLAY, YSIZE, XSIZE), 1), basin_mask),
    "verbose": True,
}


def test_modflow_simulation_initialization():
    sim = ModFlowSimulation(**default_params)
    assert sim.n_active_cells == (~basin_mask).sum()
    # In the Netherlands, the average area of a cell with this gt is ~75.8 m2
    assert np.allclose(sim.area, 75.8, atol=0.1)


def test_step():
    parameters = deepcopy(default_params)

    parameters["heads"][:,] = compress(layer_boundary_elevation[-1], basin_mask) + 5.0

    sim = ModFlowSimulation(**parameters)

    # sim.step()
    groundwater_content_prev = np.nansum(sim.groundwater_content_m3)
    print("groundwater_content_prev", groundwater_content_prev)

    for i in range(5):
        sim.step()
        drainage_m3 = np.nansum(sim.drainage_m3)
        groundwater_content = np.nansum(sim.groundwater_content_m3)
        recharge_m3 = np.nansum(sim.recharge_m3)
        print("drainge", drainage_m3)
        print("groundwater_content", groundwater_content)
        print("recharge_m3", recharge_m3)

    print("drainge", drainage_m3)
    print("recharge", recharge_m3)
    print("groundwater_content", groundwater_content)

    balance_pre = groundwater_content_prev + recharge_m3 - drainage_m3
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, rel_tol=1e-5)

    sim.finalize()


def test_recharge():
    parameters = deepcopy(default_params)
    parameters["heads"] = parameters["heads"] - 3  # set head lower than drainage

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = np.nansum(sim.groundwater_content_m3)

    recharge = np.full((YSIZE, XSIZE), 0.01)
    sim.set_recharge_m(compress(recharge, sim.basin_mask))
    sim.step()

    drainage_m3 = np.nansum(sim.drainage_m3)
    assert np.nansum(drainage_m3) == 0
    groundwater_content = np.nansum(sim.groundwater_content_m3)

    assert np.nansum(sim.recharge_m) == np.nansum(sim.recharge_m3 / sim.area)

    recharge_m3 = np.nansum(sim.recharge_m3)
    balance_pre = groundwater_content_prev + recharge_m3 - drainage_m3
    balance_post = groundwater_content

    print("drainge", drainage_m3)
    print("recharge", recharge_m3)
    print("groundwater_content", groundwater_content)
    print("groundwater_content_prev", groundwater_content_prev)

    assert math.isclose(balance_pre, balance_post, abs_tol=1, rel_tol=1e-5)

    sim.finalize()


def test_drainage():
    parameters = deepcopy(default_params)
    layer_boundary_elevation = parameters["layer_boundary_elevation"]
    topography = np.full((YSIZE, XSIZE), 0)

    parameters["topography"] = compress(topography, basin_mask)
    parameters["heads"][0] = compress(topography, basin_mask)
    parameters["heads"][1] = compress(topography, basin_mask)

    layer_boundary_elevation[0] = compress(topography - 2, basin_mask)
    layer_boundary_elevation[1] = compress(topography - 10, basin_mask)
    layer_boundary_elevation[2] = compress(topography - 20, basin_mask)

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = np.nansum(sim.groundwater_content_m3)

    sim.step()

    drainage = np.nansum(sim.drainage_m3)
    assert drainage.sum() > 0

    assert np.nansum(sim.drainage_m * sim.area) == np.nansum(drainage)

    groundwater_content = np.nansum(sim.groundwater_content_m3)

    balance_pre = groundwater_content_prev - drainage
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


def test_wells():
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

    total_abstraction = np.nansum(groundwater_abstracton[~basin_mask] * sim.area)

    sim.set_groundwater_abstraction_m(compress(groundwater_abstracton, sim.basin_mask))
    sim.step()

    drainage = np.nansum(sim.drainage_m3)
    assert drainage.sum() == 0

    groundwater_content = sim.groundwater_content_m3.sum()

    balance_pre = groundwater_content_prev - total_abstraction
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, rel_tol=1e-6)

    sim.finalize()


def visualize_modflow_results(sim, axes):
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
    im5 = ax5.imshow(decompress(sim.drainage_m, sim.basin_mask), cmap="Blues")
    ax5.set_title("Drainage")
    plt.colorbar(im5, ax=ax5, label="Drainage (m/day)")


def test_modflow_simulation_with_visualization():
    parameters = deepcopy(default_params)
    parameters["heads"][:,] = compress(topography, basin_mask) - 1
    sim = ModFlowSimulation(**parameters)

    fig, axes = plt.subplots(5, 5, figsize=(15, 10))
    plt.tight_layout()

    # Run the simulation for a few steps
    for i in range(5):
        recharge = np.random.uniform(0, 0.001, size=(YSIZE, XSIZE))
        sim.set_recharge_m(compress(recharge, sim.basin_mask))

        groundwater_abstracton = np.full((YSIZE, XSIZE), 0.0)
        groundwater_abstracton[2, 2] = 1.25
        groundwater_abstracton[3, 5] = 0.20

        sim.set_groundwater_abstraction_m(
            compress(groundwater_abstracton, sim.basin_mask)
        )
        sim.step()

        # Visualize the results
        visualize_modflow_results(sim, axes[i])

    sim.finalize()
    plt.savefig(output_folder / "modflow_simulation.png")


def test_get_water_table_depth():
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
    elevation = np.array([103, 103, 103, 103, 103])
    water_table_depth = get_water_table_depth(layer_boundary_elevation, head, elevation)
    np.testing.assert_allclose(water_table_depth, np.array([3, 13, 53, 63, 103]))


def test_get_groundwater_storage_m():
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


def test_distribute_well_rate_per_layer():
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

    well_rate_per_layer = distribute_well_rate_per_layer(
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
