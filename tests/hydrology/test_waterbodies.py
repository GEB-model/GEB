"""Tests for lake and reservoir functions in GEB."""

import math

import matplotlib.pyplot as plt
import numpy as np

from geb.hydrology.waterbodies import (
    estimate_lake_outflow,
    estimate_outflow_height,
    get_lake_factor,
    get_lake_height_above_outflow,
    get_lake_height_from_bottom,
    get_lake_outflow,
    get_lake_storage_from_height_above_bottom,
    get_river_width,
)

from ..testconfig import output_folder


def test_get_lake_height_from_bottom() -> None:
    """Test calculation of lake height from storage and area.

    Verifies that lake height is correctly calculated as storage divided by area,
    and that the inverse function produces consistent results.
    """
    lake_area = np.array([100]).astype(np.float32)
    lake_storage = np.linspace(0, 1000, 100).astype(np.float32)

    lake_height = get_lake_height_from_bottom(
        lake_storage=lake_storage, lake_area=lake_area
    )

    np.testing.assert_allclose(
        lake_height,
        np.linspace(0, 10, 100),
    )

    np.testing.assert_allclose(
        lake_storage,
        get_lake_storage_from_height_above_bottom(
            lake_height=lake_height, lake_area=lake_area
        ),
    )


def test_get_lake_storage_from_height_above_bottom() -> None:
    """Test calculation of lake storage from height and area.

    Verifies that lake storage is correctly calculated as height times area,
    and that the inverse function produces consistent results.
    """
    lake_area = np.array([100])
    lake_height = np.linspace(0, 10, 100).astype(np.float32)

    lake_storage = get_lake_storage_from_height_above_bottom(
        lake_height=lake_height, lake_area=lake_area
    )

    np.testing.assert_allclose(
        lake_storage,
        np.linspace(0, 1000, 100),
    )

    np.testing.assert_allclose(
        lake_height,
        get_lake_height_from_bottom(lake_storage=lake_storage, lake_area=lake_area),
    )


def test_estimate_initial_lake_storage_and_outflow_height() -> None:
    """Test estimation of lake outflow height and storage dynamics.

    Tests the complete lake outflow estimation workflow including:
    - Calculation of river width from average discharge
    - Lake factor computation using overflow coefficients
    - Outflow height estimation from capacity and average outflow
    - Verification of outflow-to-height inverse relationships
    - Storage dynamics simulation with inflow, outflow, and evaporation
    - Generation of diagnostic plots for lake behavior analysis
    """
    lake_area = np.array([3_480_000.0])
    lake_capacity = np.array([7_630_000.0])
    avg_outflow = np.array([2.494])
    river_width = get_river_width(avg_outflow)
    lake_factor = get_lake_factor(
        river_width=river_width,
        overflow_coefficient_mu=np.float32(0.577),
        lake_a_factor=np.float32(1),
    )

    outflow_height = estimate_outflow_height(
        lake_capacity=lake_capacity,
        lake_factor=lake_factor,
        lake_area=lake_area,
        avg_outflow=avg_outflow,
    )

    lake_storage = lake_capacity.copy()
    height_above_outflow = get_lake_height_above_outflow(
        lake_storage=lake_storage, lake_area=lake_area, outflow_height=outflow_height
    )

    assert math.isclose(
        ((height_above_outflow + outflow_height) * lake_area)[0], lake_storage[0]
    )
    outflow = estimate_lake_outflow(
        lake_factor=lake_factor, height_above_outflow=height_above_outflow
    )

    # test if outflow_to_height_above_outflow is indeed the inverse of estimate_lake_outflow
    assert math.isclose(outflow[0], avg_outflow[0])

    lake_storage = np.linspace(0, lake_storage[0] * 1.2, 100)
    height_above_outflow = get_lake_height_above_outflow(
        lake_storage=lake_storage, lake_area=lake_area, outflow_height=outflow_height
    )
    outflow = estimate_lake_outflow(
        lake_factor=lake_factor, height_above_outflow=height_above_outflow
    )

    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()

    ax_left.plot(
        lake_storage, height_above_outflow, color="red", label="height_above_outflow"
    )
    ax_right.plot(lake_storage, outflow, color="blue", label="outflow")

    # plot vertical line of initial storage
    ax_left.axvline(x=lake_storage[0], color="green", linestyle="--")

    # plot horizontal line of average outflow
    ax_right.axhline(y=avg_outflow[0], color="black", linestyle="--")

    ax_left.set_ylim(0, None)
    ax_right.set_ylim(0, None)

    ax_left.set_xlim(lake_storage[0].item(), lake_storage[-1].item())

    ax_left.set_xlabel("Storage")
    ax_left.set_ylabel("Height above outflow (red)")
    ax_right.set_ylabel("Outflow (blue)")

    plt.savefig(output_folder / "estimate_outflow_height.png")
    plt.close()

    storage = lake_storage.copy()

    dt = 3600

    inflow_m3_s = np.full(1_000, 0, dtype=np.float32)
    inflow_m3_s[:200] = np.linspace(0, 10, 200)
    inflow_m3_s[200:400] = avg_outflow[0]
    inflow_m3_s[400:] = np.linspace(avg_outflow[0], 0, 600)
    inflow = inflow_m3_s * dt

    evaporation = avg_outflow[0] * 3600 * 0.5

    new_storages = np.full(inflow.size, np.nan)
    new_outflows = np.full(inflow.size, np.nan)
    new_height_above_outflows = np.full(inflow.size, np.nan)
    for i in range(inflow.size):
        storage += inflow[i]

        outflow, lake_height_above_outflow = get_lake_outflow(
            dt,
            storage,
            lake_factor,
            lake_area,
            outflow_height,
        )

        # evaporation
        storage -= outflow
        storage -= evaporation

        new_storages[i] = storage[0]
        new_outflows[i] = outflow[0]
        new_height_above_outflows[i] = lake_height_above_outflow[0]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

    ax0.plot(new_storages, color="red", label="storage (red)")
    ax0.axhline(y=lake_storage[0], color="red", linestyle="--")
    ax0.axhline(y=outflow_height[0] * lake_area[0], color="black", linestyle="--")

    ax1.plot(new_outflows, color="blue", label="outflow (blue)")
    ax1.plot(inflow, color="green", label="inflow (green)")
    ax2.plot(new_height_above_outflows, color="black", label="height_above_outflow")

    ax0.set_ylabel("Storage")
    ax1.set_ylabel("Outflow (blue) / Inflow (green)")
    ax2.set_ylabel("Height above outflow")

    ax0.set_ylim(0, None)
    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)

    plt.savefig(output_folder / "get_lake_outflow_and_storage.png")

    plt.close()
