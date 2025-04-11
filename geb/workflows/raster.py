import numpy as np
import xarray as xr


def reclassify(data_array, remap_dict):
    """Reclassify values in an xarray DataArray using a dictionary.

    Parameters
    ----------
    data_array : xarray.DataArray
        The input data array to be reclassified
    remap_dict : dict
        Dictionary mapping old values to new values

    Returns
    -------
    xarray.DataArray
        Reclassified array with the same dimensions and coordinates
    """

    # create numpy array from dictionary values to get the dtype of the output
    remap_array = np.array(list(remap_dict.values()))
    # assert that dtype is not object
    assert remap_array.dtype != np.dtype("O")

    # Create a vectorized function
    remap_func = np.vectorize(lambda x: remap_dict[x])

    data_array = data_array.compute()
    # Apply the function and keep attributes
    result = xr.apply_ufunc(
        remap_func,
        data_array,
        keep_attrs=True,
        dask="parallelized",
        output_dtypes=[remap_array.dtype],
    )

    return result
