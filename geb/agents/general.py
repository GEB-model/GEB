"""Module containing general agent functions and the base class for all agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numba import njit

from geb.module import Module
from geb.store import DynamicArray

if TYPE_CHECKING:
    from geb.model import GEBModel


@njit(cache=True)
def downscale_volume(
    data_gt: tuple[float, float, float, float, float, float],
    model_gt: tuple[float, float, float, float, float, float],
    data: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    mapping_grid_to_HRU_uncompressed: npt.NDArray[np.int32],
    downscale_mask: npt.NDArray[np.bool_],
    HRU_land_size: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Downscale a gridded volume to HRU level using area-weighted averaging.

    Args:
        data_gt: Geotransform of the data to be downscaled.
        model_gt: Geotransform of the model (target) decompressed HRU grid.
        data: 2D array of data to be downscaled.
        mask: 2D boolean array where True indicates no data (e.g., water bodies).
        mapping_grid_to_HRU_uncompressed: 1D array mapping each grid cell to an HRU index.
        downscale_mask: 1D boolean array where True indicates HRUs to be excluded from downscaling.
        HRU_land_size: 1D array of land area for each HRU.

    Returns:
        1D array of downscaled data at HRU level.
    """
    xoffset = (model_gt[0] - data_gt[0]) / model_gt[1]
    assert 0.0001 > xoffset - round(xoffset) > -0.0001
    xoffset = round(xoffset)
    assert xoffset >= 0

    yoffset = (model_gt[3] - data_gt[3]) / model_gt[5]
    assert 0.0001 > yoffset - round(yoffset) > -0.0001
    yoffset = round(yoffset)
    assert yoffset >= 0

    xratio = data_gt[1] / model_gt[1]
    assert 0.0001 > xratio - round(xratio) > -0.0001
    assert xratio > 0
    xratio = round(xratio)

    yratio = data_gt[5] / model_gt[5]
    assert 0.0001 > yratio - round(yratio) > -0.0001
    assert yratio > 0
    yratio = round(yratio)

    downscale_invmask = ~downscale_mask
    assert xratio > 0
    assert yratio > 0
    assert xoffset >= 0
    assert yoffset >= 0
    ysize, xsize = data.shape
    yvarsize, xvarsize = mask.shape
    downscaled_array = np.zeros(HRU_land_size.size, dtype=np.float32)
    i = 0
    for y in range(ysize):
        y_left = y * yratio - yoffset
        y_right = min(y_left + yratio, yvarsize)
        y_left = max(y_left, 0)
        for x in range(xsize):
            x_left = x * xratio - xoffset
            x_right = min(x_left + xratio, xvarsize)
            x_left = max(x_left, 0)

            land_area_cell = np.float32(0.0)
            for yvar in range(y_left, y_right):
                for xvar in range(x_left, x_right):
                    if not mask[yvar, xvar]:
                        k = yvar * xvarsize + xvar
                        HRU_right = mapping_grid_to_HRU_uncompressed[k]
                        # assert HRU_right != -1
                        if k > 0:
                            HRU_left = mapping_grid_to_HRU_uncompressed[k - 1]
                            # assert HRU_left != -1
                        else:
                            HRU_left = 0
                        land_area_cell += (
                            downscale_invmask[HRU_left:HRU_right]
                            * HRU_land_size[HRU_left:HRU_right]
                        ).sum()
                        i += 1

            if land_area_cell:
                for yvar in range(y_left, y_right):
                    for xvar in range(x_left, x_right):
                        if not mask[yvar, xvar]:
                            k = yvar * xvarsize + xvar
                            HRU_right = mapping_grid_to_HRU_uncompressed[k]
                            # assert HRU_right != -1
                            if k > 0:
                                HRU_left = mapping_grid_to_HRU_uncompressed[k - 1]
                                # assert HRU_left != -1
                            else:
                                HRU_left = 0
                            downscaled_array[HRU_left:HRU_right] = (
                                downscale_invmask[HRU_left:HRU_right]
                                * HRU_land_size[HRU_left:HRU_right]
                                / land_area_cell
                                * data[y, x]
                            )

    assert i == mask.size - mask.sum()
    return downscaled_array


class AgentBaseClass(Module):
    """Base class for all agent classes."""

    def __init__(self, model: GEBModel) -> None:
        """Initialize the agent base class.

        Args:
            model: The GEB model instance.
        """
        Module.__init__(self, model)

    @property
    def agent_arrays(self) -> dict[str, DynamicArray]:
        """Return a dictionary of all DynamicArray attributes of the agent.

        Raises:
            AssertionError: If there are duplicate DynamicArray attributes.
        """
        agent_arrays = {
            name: value
            for name, value in vars(self.var).items()
            if isinstance(value, DynamicArray)
        }
        ids = [id(v) for v in agent_arrays.values()]
        if len(set(ids)) != len(ids):
            duplicate_arrays = [
                name for name, value in agent_arrays.items() if ids.count(id(value)) > 1
            ]
            raise AssertionError(
                f"Duplicate agent array names: {', '.join(duplicate_arrays)}."
            )
        return agent_arrays
