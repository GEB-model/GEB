"""Filling of values along a river network, shared by the build and the river routing."""

from typing import Any

import numpy as np
import numpy.typing as npt


def propagate_downstream(
    data: npt.NDArray,
    idxs_ds: npt.NDArray[np.integer],
    upstream_area: npt.NDArray[np.float32],
    missing_val: Any = -1,
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray:
    """Propagates values downstream along flow paths ordered from upstream to downstream.

    Args:
        data: Input 1D or 2D array containing values to propagate.
        idxs_ds: 1D or 2D array of downstream linear indices.
        upstream_area: 1D or 2D array of upstream areas used to order traversal.
        missing_val: Sentinel value representing unassigned/missing data (e.g., -1 or np.nan).
        mask: Optional boolean mask restricting propagation cells.

    Returns:
        npt.NDArray with values propagated downstream into missing cells.
    """
    flat_data = data.ravel().copy()
    downstream_index_flat = idxs_ds.ravel()
    flat_ua = upstream_area.ravel()

    if mask is not None:
        indices = np.where(mask.ravel())[0]
    else:
        indices = np.arange(flat_data.size)

    if indices.size > 0:
        sorted_indices = indices[np.argsort(flat_ua[indices])]
        missing_is_nan = isinstance(missing_val, float) and np.isnan(missing_val)

        for idx in sorted_indices:
            downstream_index = downstream_index_flat[idx]
            if downstream_index != -1 and downstream_index < flat_data.size:
                if mask is not None and not mask.ravel()[downstream_index]:
                    continue

                value = flat_data[idx]
                downstream_value = flat_data[downstream_index]

                value_is_set = (
                    not np.isnan(value) if missing_is_nan else value != missing_val
                )
                downstream_is_missing = (
                    np.isnan(downstream_value)
                    if missing_is_nan
                    else downstream_value == missing_val
                )

                if value_is_set and downstream_is_missing:
                    flat_data[downstream_index] = value

    return flat_data.reshape(data.shape)


def propagate_upstream(
    data: npt.NDArray,
    idxs_ds: npt.NDArray[np.integer],
    upstream_area: npt.NDArray[np.float32],
    missing_val: Any = -1,
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray:
    """Propagates values upstream along flow paths ordered from downstream to upstream.

    The counterpart of propagate_downstream: a cell with a missing value takes the value
    of the cell it drains to, so tributaries and headwaters take the value of the river
    they flow into.

    Args:
        data: Input 1D or 2D array containing values to propagate.
        idxs_ds: 1D or 2D array of downstream linear indices.
        upstream_area: 1D or 2D array of upstream areas used to order traversal.
        missing_val: Sentinel value representing unassigned/missing data (e.g., -1 or np.nan).
        mask: Optional boolean mask restricting propagation cells.

    Returns:
        npt.NDArray with values propagated upstream into missing cells.
    """
    flat_data = data.ravel().copy()
    downstream_index_flat = idxs_ds.ravel()
    flat_ua = upstream_area.ravel()

    if mask is not None:
        indices = np.where(mask.ravel())[0]
    else:
        indices = np.arange(flat_data.size)

    if indices.size > 0:
        sorted_indices = indices[np.argsort(-flat_ua[indices])]
        missing_is_nan = isinstance(missing_val, float) and np.isnan(missing_val)

        for idx in sorted_indices:
            downstream_index = downstream_index_flat[idx]
            if (
                downstream_index != -1
                and downstream_index < flat_data.size
                and downstream_index != idx
            ):
                if mask is not None and not mask.ravel()[downstream_index]:
                    continue

                value = flat_data[idx]
                downstream_value = flat_data[downstream_index]

                value_is_missing = (
                    np.isnan(value) if missing_is_nan else value == missing_val
                )
                downstream_is_set = (
                    not np.isnan(downstream_value)
                    if missing_is_nan
                    else downstream_value != missing_val
                )

                if value_is_missing and downstream_is_set:
                    flat_data[idx] = downstream_value

    return flat_data.reshape(data.shape)
