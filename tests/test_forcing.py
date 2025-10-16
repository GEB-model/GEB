import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from geb.forcing import generate_bilinear_interpolation_weights


def test_bilinear_interpolation_ascending_coordinates() -> None:
    """Test bilinear interpolation with ascending source coordinates."""
    # Create a simple test grid with ascending coordinates
    src_x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    src_y = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    # Create test data with a known function: z = x^2 + y^2
    X, Y = np.meshgrid(src_x, src_y, indexing="xy")
    test_data = (X**2 + Y**2).astype(np.float32)

    # Target points for interpolation
    tgt_x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    tgt_y = np.array([0.5, 1.5], dtype=np.float32)

    # Our custom interpolation
    indices, weights = generate_bilinear_interpolation_weights(
        src_x, src_y, tgt_x, tgt_y
    )

    # Flatten test data for our interpolation
    test_data_flat = test_data.flatten()
    corner_values = test_data_flat[indices]
    our_result = np.sum(corner_values * weights, axis=1)
    our_result = our_result.reshape(len(tgt_y), len(tgt_x))

    # SciPy interpolation
    scipy_interp = RegularGridInterpolator(
        (src_x, src_y), test_data.T, method="linear", bounds_error=True
    )
    tgt_x_2d, tgt_y_2d = np.meshgrid(tgt_x, tgt_y, indexing="xy")
    target_points = np.column_stack([tgt_x_2d.ravel(), tgt_y_2d.ravel()])
    scipy_result = scipy_interp(target_points).reshape(len(tgt_y), len(tgt_x))

    # Compare results (should be very close)
    np.testing.assert_allclose(our_result, scipy_result, rtol=1e-6, atol=1e-8)


def test_bilinear_interpolation_descending_y_coordinates() -> None:
    """Test bilinear interpolation with descending y-coordinates (common in climate data)."""
    # Create test grid with ascending x but descending y coordinates
    src_x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    src_y = np.array([2.0, 1.0, 0.0], dtype=np.float32)  # Descending

    # Create test data with the same function: z = x^2 + y^2
    X, Y = np.meshgrid(src_x, src_y, indexing="xy")
    test_data = (X**2 + Y**2).astype(np.float32)

    # Target points for interpolation
    tgt_x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    tgt_y = np.array([0.5, 1.5], dtype=np.float32)

    # Our custom interpolation
    indices, weights = generate_bilinear_interpolation_weights(
        src_x, src_y, tgt_x, tgt_y
    )

    # Flatten test data for our interpolation
    test_data_flat = test_data.flatten()
    corner_values = test_data_flat[indices]
    our_result = np.sum(corner_values * weights, axis=1)
    our_result = our_result.reshape(len(tgt_y), len(tgt_x))

    # SciPy interpolation (note: still pass in original coordinate order)
    scipy_interp = RegularGridInterpolator(
        (src_x, src_y), test_data.T, method="linear", bounds_error=True
    )
    tgt_x_2d, tgt_y_2d = np.meshgrid(tgt_x, tgt_y, indexing="xy")
    target_points = np.column_stack([tgt_x_2d.ravel(), tgt_y_2d.ravel()])
    scipy_result = scipy_interp(target_points).reshape(len(tgt_y), len(tgt_x))

    # Compare results (should be very close)
    np.testing.assert_allclose(our_result, scipy_result, rtol=1e-6, atol=1e-8)


def test_bilinear_interpolation_descending_x_coordinates_error() -> None:
    """Test that descending x-coordinates raise an appropriate error."""
    # Create test grid with descending x coordinates (should raise error)
    src_x = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)  # Descending
    src_y = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    # Target points
    tgt_x = np.array([2.5, 1.5, 0.5], dtype=np.float32)
    tgt_y = np.array([0.5, 1.5], dtype=np.float32)

    # Should raise ValueError for descending x-coordinates
    with pytest.raises(
        ValueError, match="Source x-coordinates must be strictly ascending"
    ):
        generate_bilinear_interpolation_weights(src_x, src_y, tgt_x, tgt_y)


def test_bilinear_interpolation_non_monotonic_x_coordinates_error() -> None:
    """Test that non-monotonic x-coordinates raise an appropriate error."""
    # Create test grid with non-monotonic x coordinates
    src_x = np.array([0.0, 2.0, 1.0, 3.0], dtype=np.float32)  # Non-monotonic
    src_y = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    # Target points
    tgt_x = np.array([0.5, 1.5], dtype=np.float32)
    tgt_y = np.array([0.5, 1.5], dtype=np.float32)

    # Should raise ValueError for non-monotonic x-coordinates
    with pytest.raises(
        ValueError, match="Source x-coordinates must be strictly ascending"
    ):
        generate_bilinear_interpolation_weights(src_x, src_y, tgt_x, tgt_y)


def test_bilinear_interpolation_climate_like_data() -> None:
    """Test with realistic climate data coordinates (latitude descending, longitude ascending)."""
    # Typical climate data: longitude ascending, latitude descending
    src_x = np.array(
        [-180.0, -90.0, 0.0, 90.0, 180.0], dtype=np.float32
    )  # Longitude (ascending)
    src_y = np.array(
        [90.0, 45.0, 0.0, -45.0, -90.0], dtype=np.float32
    )  # Latitude (descending)

    # Create synthetic temperature-like data
    X, Y = np.meshgrid(src_x, src_y, indexing="xy")
    # Temperature decreasing towards poles and with longitude variation
    test_data = (20.0 - 0.3 * np.abs(Y) + 0.1 * np.cos(np.pi * X / 180.0)).astype(
        np.float32
    )

    # Target points (some cities)
    tgt_x = np.array(
        [-74.0, 2.3, 139.7], dtype=np.float32
    )  # NYC, Paris, Tokyo longitude
    tgt_y = np.array([40.7, 48.9, 35.7], dtype=np.float32)  # NYC, Paris, Tokyo latitude

    # Our custom interpolation
    indices, weights = generate_bilinear_interpolation_weights(
        src_x, src_y, tgt_x, tgt_y
    )

    # Flatten test data for our interpolation
    test_data_flat = test_data.flatten()
    corner_values = test_data_flat[indices]
    our_result = np.sum(corner_values * weights, axis=1)
    our_result = our_result.reshape(len(tgt_y), len(tgt_x))

    # SciPy interpolation
    scipy_interp = RegularGridInterpolator(
        (src_x, src_y), test_data.T, method="linear", bounds_error=True
    )
    tgt_x_2d, tgt_y_2d = np.meshgrid(tgt_x, tgt_y, indexing="xy")
    target_points = np.column_stack([tgt_x_2d.ravel(), tgt_y_2d.ravel()])
    scipy_result = scipy_interp(target_points).reshape(len(tgt_y), len(tgt_x))

    # Compare results (should be very close)
    np.testing.assert_allclose(our_result, scipy_result, rtol=1e-6, atol=1e-8)


def test_bilinear_interpolation_edge_cases() -> None:
    """Test edge cases like single grid cell and exact coordinate matches."""
    # Test case 1: Target points exactly on grid points
    src_x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    src_y = np.array([0.0, 1.0], dtype=np.float32)

    X, Y = np.meshgrid(src_x, src_y, indexing="xy")
    test_data = (X + Y).astype(np.float32)

    # Target points exactly on grid
    tgt_x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    tgt_y = np.array([0.0, 1.0], dtype=np.float32)

    indices, weights = generate_bilinear_interpolation_weights(
        src_x, src_y, tgt_x, tgt_y
    )

    test_data_flat = test_data.flatten()
    corner_values = test_data_flat[indices]
    our_result = np.sum(corner_values * weights, axis=1)
    our_result = our_result.reshape(len(tgt_y), len(tgt_x))

    # SciPy interpolation
    scipy_interp = RegularGridInterpolator(
        (src_x, src_y), test_data.T, method="linear", bounds_error=True
    )
    tgt_x_2d, tgt_y_2d = np.meshgrid(tgt_x, tgt_y, indexing="xy")
    target_points = np.column_stack([tgt_x_2d.ravel(), tgt_y_2d.ravel()])
    scipy_result = scipy_interp(target_points).reshape(len(tgt_y), len(tgt_x))

    np.testing.assert_allclose(our_result, scipy_result, rtol=1e-6, atol=1e-8)


def test_bilinear_interpolation_weights_properties() -> None:
    """Test that interpolation weights have the correct mathematical properties."""
    src_x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    src_y = np.array([2.0, 1.0, 0.0], dtype=np.float32)  # Descending

    tgt_x = np.array([0.3, 1.7, 2.8], dtype=np.float32)
    tgt_y = np.array([1.8, 0.2], dtype=np.float32)

    indices, weights = generate_bilinear_interpolation_weights(
        src_x, src_y, tgt_x, tgt_y
    )

    # Check that weights are non-negative
    assert np.all(weights >= 0), "All weights should be non-negative"

    # Check that weights are at most 1
    assert np.all(weights <= 1), "All weights should be at most 1"

    # Check that weights sum to 1 for each target point
    weight_sums = np.sum(weights, axis=1)
    np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-6, atol=1e-8)

    # Check indices are within valid range
    n_src_points = len(src_x) * len(src_y)
    assert np.all(indices >= 0), "All indices should be non-negative"
    assert np.all(indices < n_src_points), "All indices should be within grid bounds"


def test_bilinear_interpolation_bounds_error() -> None:
    """Test that interpolation raises appropriate errors for out-of-bounds targets."""
    src_x = np.array([0.0, 1.0, 2.0], dtype=np.float32)  # Ascending
    src_y = np.array([0.0, 1.0], dtype=np.float32)

    # Target points outside x bounds
    tgt_x_bad = np.array(
        [-0.5, 1.0, 2.5], dtype=np.float32
    )  # -0.5 and 2.5 out of bounds
    tgt_y_good = np.array([0.5], dtype=np.float32)

    with pytest.raises(ValueError, match="Target x-coordinates are outside"):
        generate_bilinear_interpolation_weights(src_x, src_y, tgt_x_bad, tgt_y_good)
