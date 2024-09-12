import numpy as np
import matplotlib.pyplot as plt

from geb.agents.reservoir_operators import regulate_reservoir_outflow
from tests.testconfig import output_folder


def test_regulate_reservoir_outflow():
    current_storage = np.linspace(0, 105, 1000)
    volume = np.full_like(current_storage, 100)
    inflow = np.full_like(current_storage, 5)
    minQ = np.full_like(current_storage, 1)
    normQ = np.full_like(current_storage, 10)
    nondmgQ = np.full_like(current_storage, 20)
    conservative_limit_ratio = np.full_like(current_storage, 0.1)
    normal_limit_ratio = np.full_like(current_storage, 0.9)
    flood_limit_ratio = np.full_like(current_storage, 1.0)

    outflow = regulate_reservoir_outflow(
        current_storage,
        volume,
        inflow,
        minQ,
        normQ,
        nondmgQ,
        conservative_limit_ratio,
        normal_limit_ratio,
        flood_limit_ratio,
    )

    fig, (ax0, ax1) = plt.subplots(1, 2)

    ax0.plot(current_storage, outflow, label="outflow")
    ax0.plot(current_storage, inflow, label="inflow")

    inflow = np.full_like(current_storage, 100)
    outflow = regulate_reservoir_outflow(
        current_storage,
        volume,
        inflow,
        minQ,
        normQ,
        nondmgQ,
        conservative_limit_ratio,
        normal_limit_ratio,
        flood_limit_ratio,
    )

    ax1.plot(current_storage, outflow, label="outflow")
    ax1.plot(current_storage, inflow, label="inflow")

    for ax in [ax0, ax1]:
        ax.legend()
        ax.set_xlabel("storage (%) of 100 m3 reservoir")
        ax.set_ylabel("in-/outflow (m3/s)")

    plt.savefig(output_folder / "test_regulate_reservoir_outflow.png")
