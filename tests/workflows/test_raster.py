import numpy as np
import pytest
import xarray as xr

from geb.workflows.raster import reclassify


def test_reclassify() -> None:
    """Test the reclassify function with various methods and inputs."""
    classification_dict = {
        1: 10,
        2: 20,
        3: 30,
        4: 40,
        5: 50,
    }

    array = np.array([[1, 2, 3], [4, 5, 1]])
    expected_output = array * 10

    output_dict = reclassify(array, classification_dict, method="dict")
    assert np.array_equal(output_dict, expected_output)

    output_lookup = reclassify(array, classification_dict, method="lookup")
    assert np.array_equal(output_lookup, expected_output)

    xr_array = xr.DataArray(array, dims=["x", "y"])
    output_xr_dict = reclassify(xr_array, classification_dict, method="dict")
    assert np.array_equal(output_xr_dict.values, expected_output)

    output_xr_lookup = reclassify(xr_array, classification_dict, method="lookup")
    assert np.array_equal(output_xr_lookup.values, expected_output)

    classification_dict_with_negative = {
        -1: 10,
        -2: 20,
        -3: 30,
        -4: 40,
        -5: 50,
    }
    array_with_negative = np.array([[-1, -2, -3], [-4, -5, -1]])
    expected_output_with_negative = array_with_negative * 10 * -1

    output_dict_with_negative = reclassify(
        array_with_negative, classification_dict_with_negative, method="dict"
    )
    assert np.array_equal(output_dict_with_negative, expected_output_with_negative)

    # Test if ValueError is raised when using lookup method with negative keys
    with pytest.raises(
        ValueError,
        match="Keys must be positive integers for lookup method, recommend using dict method",
    ):
        reclassify(
            array_with_negative, classification_dict_with_negative, method="lookup"
        )
