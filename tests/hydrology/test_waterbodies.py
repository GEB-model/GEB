import math

import matplotlib.pyplot as plt
import numpy as np

from geb.hydrology.lakes_reservoirs import (
    estimate_lake_outflow,
    estimate_outflow_height,
    get_lake_factor,
    get_lake_height_above_outflow,
    get_lake_outflow_and_storage,
    get_river_width,
)

from ..testconfig import output_folder


def test_estimate_initial_lake_storage_and_outflow_height():
    lake_area = np.array([3_480_000.0])
    lake_storage = np.array([7_630_000.0])
    avg_outflow = np.array([2.494])
    river_width = get_river_width(avg_outflow)
    lake_factor = get_lake_factor(
        river_width=river_width, overflow_coefficient_mu=0.577, lake_a_factor=1
    )

    outflow_height = estimate_outflow_height(
        lake_storage=lake_storage,
        lake_factor=lake_factor,
        lake_area=lake_area,
        avg_outflow=avg_outflow,
    )

    storage = lake_storage.copy()
    height_above_outflow = get_lake_height_above_outflow(
        storage=storage, lake_area=lake_area, outflow_height=outflow_height
    )

    assert math.isclose(
        ((height_above_outflow + outflow_height) * lake_area)[0], lake_storage[0]
    )
    outflow = estimate_lake_outflow(
        lake_factor=lake_factor, height_above_outflow=height_above_outflow
    )

    # test if outflow_to_height_above_outflow is indeed the inverse of estimate_lake_outflow
    assert math.isclose(outflow[0], avg_outflow[0])

    storage = np.linspace(0, lake_storage * 1.2, 100)
    height_above_outflow = get_lake_height_above_outflow(
        storage=storage, lake_area=lake_area, outflow_height=outflow_height
    )
    outflow = estimate_lake_outflow(
        lake_factor=lake_factor, height_above_outflow=height_above_outflow
    )

    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()

    ax_left.plot(
        storage, height_above_outflow, color="red", label="height_above_outflow"
    )
    ax_right.plot(storage, outflow, color="blue", label="outflow")

    # plot vertical line of initial storage
    ax_left.axvline(x=lake_storage[0], color="green", linestyle="--")

    # plot horizontal line of average outflow
    ax_right.axhline(y=avg_outflow[0], color="black", linestyle="--")

    ax_left.set_ylim(0, None)
    ax_right.set_ylim(0, None)

    ax_left.set_xlim(storage[0].item(), storage[-1].item())

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
        outflow, storage, lake_height_above_outflow = get_lake_outflow_and_storage(
            dt,
            storage,
            inflow[i],
            lake_factor,
            lake_area,
            outflow_height,
        )

        # evaporation
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
