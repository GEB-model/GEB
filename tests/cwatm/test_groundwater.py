import math
import numpy as np
import matplotlib.pyplot as plt
from cwatm.modules.groundwater_modflow.modflow_model import ModFlowSimulation

from ..setup import output_folder, tmp_folder


class DummyModel:
    def __init__(self):
        self.simulation_root = tmp_folder / "modflow"


XSIZE = 12
YSIZE = 10
NLAY = 1

# Create a topography 2D map
x = np.linspace(-5, 5, XSIZE)
y = np.linspace(-5, 5, YSIZE)
x, y = np.meshgrid(x, y)

topography = np.exp2(-(x**2) - y**2 + 5)
basin_mask = np.zeros((YSIZE, XSIZE), dtype=bool)
basin_mask[0] = True
basin_mask[-3:-1, 0:3] = True

# Create a topography 2D map
x_vertices, y_vertices = np.meshgrid(
    np.linspace(0, XSIZE * 10, XSIZE + 1), np.linspace(0, YSIZE * 10, YSIZE + 1)
)


def compress(array, mask):
    return array[..., ~mask]


def decompress(array, mask):
    out = np.full(mask.shape, np.nan)
    out[~mask] = array
    return out


default_params = {
    "model": DummyModel(),
    "name": "test_model",
    "ndays": 20,
    "specific_storage": compress(np.full((NLAY, YSIZE, XSIZE), 0), basin_mask),
    "specific_yield": compress(np.full((NLAY, YSIZE, XSIZE), 0.4), basin_mask),
    "nrow": YSIZE,
    "ncol": XSIZE,
    "x_coordinates_vertices": x_vertices,
    "y_coordinates_vertices": y_vertices,
    "topography": compress(topography, basin_mask),
    "bottom_soil": compress(topography - 2, basin_mask),
    "bottom": compress(topography - np.full((NLAY, YSIZE, XSIZE), 10), basin_mask),
    "basin_mask": basin_mask,
    "head": compress(topography - 2, basin_mask),
    "hydraulic_conductivity": compress(np.full((NLAY, YSIZE, XSIZE), 1), basin_mask),
    "verbose": True,
}


def test_modflow_simulation_initialization():
    sim = ModFlowSimulation(**default_params)
    assert sim.name == "TEST_MODEL"
    assert sim.nrow == YSIZE
    assert sim.ncol == XSIZE
    assert sim.n_active_cells == (~basin_mask).sum()
    assert (sim.area == 100.0).all()


def test_recharge():
    parameters = default_params.copy()
    parameters["head"] = compress(
        topography - 5, basin_mask
    )  # set head lower than drainage

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

    assert math.isclose(balance_pre, balance_post, rel_tol=1e-5)

    sim.finalize()


def test_drainage():
    parameters = default_params.copy()
    topography = np.full((YSIZE, XSIZE), 0)
    parameters["topography"] = compress(topography, basin_mask)
    parameters["bottom_soil"] = compress(topography - 2, basin_mask)
    parameters["bottom"] = compress(
        topography - np.full((NLAY, YSIZE, XSIZE), 10), basin_mask
    )
    parameters["head"] = compress(topography, basin_mask)

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

    parameters["head"] = parameters["bottom_soil"] - 1

    sim = ModFlowSimulation(**parameters)

    sim.step()

    drainage = np.nansum(sim.drainage_m3)
    assert drainage == 0

    sim.finalize()


def test_wells():
    parameters = default_params.copy()
    parameters["head"] = compress(
        topography - 5, basin_mask
    )  # set head lower than drainage

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = np.nansum(sim.groundwater_content_m3)

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

    groundwater_content = np.nansum(sim.groundwater_content_m3)

    balance_pre = groundwater_content_prev - total_abstraction
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, rel_tol=1e-6)

    sim.finalize()


def visualize_modflow_results(sim, axes):
    (ax1, ax2, ax3, ax4) = axes

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
    im2 = ax2.imshow(decompress(sim.head, sim.basin_mask), cmap="viridis")
    ax2.set_title("Groundwater Head")
    plt.colorbar(im2, ax=ax2, label="Head (m)")

    # Plot groundwater depth
    im3 = ax3.imshow(decompress(sim.groundwater_depth, sim.basin_mask), cmap="RdYlBu")
    ax3.set_title("Groundwater Depth")
    plt.colorbar(im3, ax=ax3, label="Depth (m)")

    # Plot drainage
    im4 = ax4.imshow(decompress(sim.drainage_m, sim.basin_mask), cmap="Blues")
    ax4.set_title("Drainage")
    plt.colorbar(im4, ax=ax4, label="Drainage (m/day)")


def test_modflow_simulation_with_visualization():
    sim = ModFlowSimulation(**default_params)

    fig, axes = plt.subplots(5, 4, figsize=(15, 10))
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
