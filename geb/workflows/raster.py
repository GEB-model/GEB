import numpy as np
import xarray as xr


def reclassify(
    data_array: xr.DataArray | np.ndarray, remap_dict: dict, method: str = "dict"
) -> xr.DataArray | np.ndarray:
    """Reclassify values in an xarray DataArray using a dictionary.

    Parameters
    ----------
    data_array : xarray.DataArray
        The input data array to be reclassified
    remap_dict : dict
        Dictionary mapping old values to new values
    method : str
        The method to use for reclassification. Dict or lookup.
        If lookup, it will use a lookup array for faster performance, but
        keys must be positive integers only and be limited in size.

    Returns:
    -------
    xarray.DataArray or numpy.ndarray
        Reclassified array with the same dimensions and coordinates
    """
    # create numpy array from dictionary values to get the dtype of the output
    values = np.array(list(remap_dict.values()))
    # assert that dtype is not object
    assert values.dtype != np.dtype("O")

    if method == "dict":
        # Create a vectorized function using the dictionary approach
        remap_func_dict = np.vectorize(remap_dict.get)
    elif method == "lookup":
        # Get all keys from remap_dict to create a lookup array
        keys = np.array(list(remap_dict.keys()))

        if keys.min() < 0:
            raise ValueError(
                "Keys must be positive integers for lookup method, recommend using dict method"
            )

        lookup_array = np.full(keys.max() + 1, -1, dtype=values.dtype)
        lookup_array[keys] = values
    else:
        raise ValueError("Method must be either 'dict' or 'lookup'")

    if isinstance(data_array, xr.DataArray):
        # Apply the function and keep attributes
        if method == "dict":
            # Use the dictionary method
            result = xr.apply_ufunc(
                remap_func_dict,
                data_array,
                keep_attrs=True,
                dask="parallelized",
                output_dtypes=[values.dtype],
            )
        elif method == "lookup":
            # Use the lookup method
            result = xr.apply_ufunc(
                lambda x: lookup_array[x],
                data_array,
                keep_attrs=True,
                dask="parallelized",
                output_dtypes=[values.dtype],
            )
        else:
            raise ValueError("Method must be either 'dict' or 'lookup'")

    elif isinstance(data_array, np.ndarray):
        # Apply the function and keep attributes
        if method == "dict":
            # Use the dictionary method
            # result = remap_func_dict(data_array)
            result = remap_func_dict(data_array)
        elif method == "lookup":
            result = lookup_array[data_array]
        else:
            raise ValueError("Method must be either 'dict' or 'lookup'")
    else:
        raise TypeError("Input must be either xarray.DataArray or numpy.ndarray")

    return result
