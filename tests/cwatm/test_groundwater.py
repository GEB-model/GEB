import math
import numpy as np
import matplotlib.pyplot as plt
from cwatm.modules.groundwater_modflow.modflow_model import ModFlowSimulation

from ..setup import output_folder, tmp_folder


class DummyModel:
    def __init__(self):
        self.simulation_root = tmp_folder / "modflow"


# Create a topography 2D map
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
x, y = np.meshgrid(x, y)

topography = np.exp2(-(x**2) - y**2 + 5)
basin_mask = np.zeros((10, 10), dtype=bool)
basin_mask[-1] = True
basin_mask[-3:-1, 0:3] = True

XSIZE = 10
YSIZE = 10
NLAY = 1

default_params = {
    "model": DummyModel(),
    "name": "test_model",
    "ndays": 20,
    "specific_storage": np.full((NLAY, YSIZE, XSIZE), 0),
    "specific_yield": np.full((NLAY, YSIZE, XSIZE), 0.4),
    "nlay": NLAY,
    "nrow": 10,
    "ncol": 10,
    "row_resolution": 10,
    "col_resolution": 10,
    "topography": topography,
    "bottom_soil": topography - 2,
    "bottom": topography - np.full((NLAY, YSIZE, XSIZE), 10),
    "basin_mask": basin_mask,
    "head": topography - 2,
    "hydraulic_conductivity": np.full((NLAY, YSIZE, XSIZE), 1),
    "verbose": True,
}


def test_modflow_simulation_initialization():
    sim = ModFlowSimulation(**default_params)
    assert sim.name == "TEST_MODEL"
    assert sim.nrow == 10
    assert sim.ncol == 10
    assert sim.n_active_cells == 84


def test_recharge():
    parameters = default_params.copy()
    parameters["head"] = topography - 5  # set head lower than drainage

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = np.nansum(sim.groundwater_content_m3)

    sim.set_recharge_m(np.full((10, 10), 0.01))
    sim.step()

    drainage = np.nansum(sim.drainage_m3)
    assert np.nansum(drainage) == 0
    groundwater_content = np.nansum(sim.groundwater_content_m3)
    recharge = np.nansum(sim.recharge_m3)

    assert np.nansum(
        sim.recharge_m * sim.row_resolution * sim.col_resolution
    ) == np.nansum(recharge)

    balance_pre = groundwater_content_prev + recharge - drainage
    balance_post = groundwater_content

    assert math.isclose(balance_pre, balance_post, rel_tol=1e-5)

    sim.finalize()


def test_drainage():
    parameters = default_params.copy()
    parameters["topography"] = np.full((10, 10), 0)
    parameters["bottom_soil"] = parameters["topography"] - 2
    parameters["bottom"] = parameters["topography"] - np.full((NLAY, YSIZE, XSIZE), 10)
    parameters["head"] = parameters["topography"]

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = np.nansum(sim.groundwater_content_m3)

    sim.step()

    drainage = np.nansum(sim.drainage_m3)
    assert drainage.sum() > 0

    assert np.nansum(
        sim.drainage_m * sim.row_resolution * sim.col_resolution
    ) == np.nansum(drainage)

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
    parameters["head"] = topography - 5  # set head lower than drainage

    sim = ModFlowSimulation(**parameters)

    groundwater_content_prev = np.nansum(sim.groundwater_content_m3)

    groundwater_abstracton = np.full((10, 10), 0.10)
    groundwater_abstracton[0, 0] = 0.10
    groundwater_abstracton[1, 1] = 0.05
    groundwater_abstracton[4, 5] = 0.20
    groundwater_abstracton[parameters["basin_mask"]] = np.nan

    total_abstraction = (
        np.nansum(groundwater_abstracton)
        * default_params["row_resolution"]
        * default_params["col_resolution"]
    )

    sim.set_groundwater_abstraction_m(groundwater_abstracton)
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
    im1 = ax1.imshow(sim.topography, cmap="terrain")
    ax1.set_title("Topography")
    plt.colorbar(im1, ax=ax1, label="Elevation (m)")

    # Plot groundwater head
    im2 = ax2.imshow(sim.decompress(sim.head), cmap="viridis")
    ax2.set_title("Groundwater Head")
    plt.colorbar(im2, ax=ax2, label="Head (m)")

    # Plot groundwater depth
    im3 = ax3.imshow(sim.groundwater_depth, cmap="RdYlBu")
    ax3.set_title("Groundwater Depth")
    plt.colorbar(im3, ax=ax3, label="Depth (m)")

    # Plot drainage
    im4 = ax4.imshow(sim.drainage_m, cmap="Blues")
    ax4.set_title("Drainage")
    plt.colorbar(im4, ax=ax4, label="Drainage (m/day)")


def test_modflow_simulation_with_visualization():
    sim = ModFlowSimulation(**default_params)

    fig, axes = plt.subplots(5, 4, figsize=(15, 10))
    plt.tight_layout()

    # Run the simulation for a few steps
    for i in range(5):
        sim.set_recharge_m(np.random.uniform(0, 0.001, size=(10, 10)))

        groundwater_abstracton = np.full((10, 10), 0.0)
        groundwater_abstracton[1, 1] = 0.05
        groundwater_abstracton[3, 5] = 0.20

        sim.set_groundwater_abstraction_m(groundwater_abstracton)
        sim.step()

        # Visualize the results
        visualize_modflow_results(sim, axes[i])

    sim.finalize()
    plt.savefig(output_folder / "modflow_simulation.png")
