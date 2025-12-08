"""Module for concentrating runoff from different sources."""

import numpy as np

from geb.types import ArrayFloat32


def concentrate_runoff(
    interflow: ArrayFloat32,
    baseflow: ArrayFloat32,
    runoff: ArrayFloat32,
) -> ArrayFloat32:
    """Combines all sources of runoff.

    Args:
        interflow: The interflow [m] for the cell.
        baseflow: The baseflow [m] for the cell.
        runoff: The surface runoff [m] for the cell.

    Returns:
        The total runoff [m] for the cell.
    """
    assert (runoff >= 0).all()
    assert (interflow >= 0).all()
    assert (baseflow >= 0).all()

    assert interflow.shape[0] == 24
    assert runoff.shape[0] == 24

    assert interflow.ndim == 2
    assert baseflow.ndim == 1
    assert runoff.ndim == 2

    baseflow_per_timestep = baseflow / np.float32(24)

    return interflow + baseflow_per_timestep + runoff
