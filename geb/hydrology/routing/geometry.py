"""Cross-sectional geometry calculations for 1D river reaches."""

import numpy as np
from numba import njit

from geb.geb_types import ArrayFloat32

__all__: list[str] = [
    "compute_cfl_area_and_top_width",
    "compute_cross_section_from_depth",
    "compute_overbank_depth",
    "compute_static_geometry",
]


def compute_static_geometry(
    river_length: ArrayFloat32,
    river_width: ArrayFloat32,
    shape_exponent: ArrayFloat32,
    bankfull_depth: ArrayFloat32,
    floodplain_width: ArrayFloat32,
) -> tuple[
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
]:
    """Precomputes reach-static cross-sectional geometric constants.

    Args:
        river_length: Channel length of each reach (meters). Must be positive.
        river_width: Bankfull top width of the main channel (meters). Must be positive.
        shape_exponent: Power-law cross-sectional shape exponent r (dimensionless). Must be positive.
        bankfull_depth: Bankfull channel depth before floodplain spilling (meters). Must be positive.
        floodplain_width: Additional floodplain width beyond bankfull channel (meters). Must be non-negative.

    Returns:
        A tuple containing 1D float32 arrays:
            - inverse_reach_length: Inverse channel length 1 / L (1/meters).
            - inverse_shape_exponent_plus_one: Inverse shape exponent term 1 / (r + 1) (dimensionless).
            - bankfull_area: Bankfull cross-sectional area (m²).
            - bankfull_volume: Bankfull channel storage volume (m³).
            - bankfull_perimeter: Bankfull wetted perimeter (meters).
            - floodplain_side_slope: Effective floodplain side-slope parameter z_fp (dimensionless).
            - floodplain_depth_threshold: Floodplain depth threshold for vertical expansion (meters).
            - floodplain_area_threshold: Floodplain area threshold for vertical expansion (m²).
            - sqrt_one_plus_floodplain_slope_squared: Precalculated factor sqrt(1 + z_fp²) (dimensionless).
            - width_over_sqrt_bankfull_depth: Precalculated factor W_bf / sqrt(h_bf) for r=0.5 (m^(1/2)).

    Raises:
        ValueError: If input arrays have mismatched shapes or contain non-finite (NaN, inf)
            or non-positive values (negative for floodplain width).
    """
    if not (
        river_length.shape
        == river_width.shape
        == shape_exponent.shape
        == bankfull_depth.shape
        == floodplain_width.shape
    ):
        raise ValueError("All input arrays must have the same shape.")

    if not np.all(np.isfinite(river_length)) or np.any(river_length <= 0):
        raise ValueError("river_length must contain positive, finite numbers.")
    if not np.all(np.isfinite(river_width)) or np.any(river_width <= 0):
        raise ValueError("river_width must contain positive, finite numbers.")
    if not np.all(np.isfinite(shape_exponent)) or np.any(shape_exponent <= 0):
        raise ValueError("shape_exponent must contain positive, finite numbers.")
    if not np.all(np.isfinite(bankfull_depth)) or np.any(bankfull_depth <= 0):
        raise ValueError("bankfull_depth must contain positive, finite numbers.")
    if not np.all(np.isfinite(floodplain_width)) or np.any(floodplain_width < 0):
        raise ValueError("floodplain_width must contain non-negative, finite numbers.")

    inverse_reach_length: ArrayFloat32 = np.float32(1.0) / river_length
    inverse_shape_exponent_plus_one: ArrayFloat32 = np.float32(1.0) / (
        shape_exponent + np.float32(1.0)
    )
    bankfull_area: ArrayFloat32 = (
        river_width * bankfull_depth * inverse_shape_exponent_plus_one
    )
    bankfull_volume: ArrayFloat32 = river_length * bankfull_area
    bankfull_perimeter: ArrayFloat32 = (
        river_width
        + np.float32(8.0 / 3.0) * (bankfull_depth * bankfull_depth) / river_width
    )

    reference_depth: ArrayFloat32 = np.maximum(bankfull_depth, np.float32(0.5))
    floodplain_side_slope: ArrayFloat32 = np.maximum(
        floodplain_width / (np.float32(2.0) * reference_depth), np.float32(2.0)
    )
    floodplain_depth_threshold: ArrayFloat32 = np.where(
        floodplain_width > np.float32(0.0),
        floodplain_width / (np.float32(2.0) * floodplain_side_slope),
        np.float32(0.0),
    ).astype(np.float32)
    floodplain_area_threshold: ArrayFloat32 = floodplain_side_slope * (
        floodplain_depth_threshold * floodplain_depth_threshold
    )
    sqrt_one_plus_floodplain_slope_squared: ArrayFloat32 = np.sqrt(
        np.float32(1.0) + floodplain_side_slope * floodplain_side_slope
    ).astype(np.float32)

    width_over_sqrt_bankfull_depth: ArrayFloat32 = river_width / np.sqrt(bankfull_depth)

    return (
        inverse_reach_length,
        inverse_shape_exponent_plus_one,
        bankfull_area,
        bankfull_volume,
        bankfull_perimeter,
        floodplain_side_slope,
        floodplain_depth_threshold,
        floodplain_area_threshold,
        sqrt_one_plus_floodplain_slope_squared,
        width_over_sqrt_bankfull_depth,
    )


@njit(inline="always")
def compute_overbank_depth(
    overbank_volume_m3: np.float32,
    inverse_reach_length: np.float32,
    floodplain_width_m: np.float32,
    floodplain_side_slope: np.float32,
    floodplain_depth_threshold_m: np.float32,
    floodplain_area_threshold_m2: np.float32,
) -> np.float32:
    """Calculates floodplain inundation depth for water volume exceeding bankfull storage.

    Args:
        overbank_volume_m3: Water volume above bankfull capacity (m³).
        inverse_reach_length: Precomputed inverse reach length 1 / L (1/meters).
        floodplain_width_m: Additional floodplain width beyond bankfull channel (meters).
        floodplain_side_slope: Precomputed effective floodplain side-slope parameter z_fp (dimensionless).
        floodplain_depth_threshold_m: Precomputed floodplain depth threshold for vertical expansion (meters).
        floodplain_area_threshold_m2: Precomputed floodplain area threshold for vertical expansion (m²).

    Returns:
        Floodplain inundation depth (meters).
    """
    overbank_area: np.float32 = overbank_volume_m3 * inverse_reach_length
    if (
        floodplain_width_m > np.float32(0.0)
        and overbank_area > floodplain_area_threshold_m2
    ):
        return floodplain_depth_threshold_m + (
            overbank_area - floodplain_area_threshold_m2
        ) / max(floodplain_width_m, np.float32(1e-3))
    elif floodplain_side_slope > np.float32(0.0):
        return np.sqrt(overbank_area / floodplain_side_slope)
    else:
        return overbank_area / max(floodplain_width_m, np.float32(1e-3))


@njit(inline="always")
def compute_cross_section_from_depth(
    effective_depth_m: np.float32,
    bankfull_width_m: np.float32,
    shape_exponent: np.float32,
    bankfull_depth_m: np.float32,
    floodplain_width_m: np.float32,
    inverse_shape_exponent_plus_one: np.float32,
    bankfull_area_m2: np.float32,
    bankfull_perimeter_m: np.float32,
    floodplain_side_slope: np.float32,
    floodplain_depth_threshold_m: np.float32,
    floodplain_area_threshold_m2: np.float32,
    sqrt_one_plus_floodplain_slope_squared: np.float32,
    width_over_sqrt_bankfull_depth: np.float32,
) -> tuple[np.float32, np.float32, np.float32]:
    """Calculates cross-sectional area, top width, and wetted perimeter from effective interface depth.

    Uses a continuous power-law channel cross-section (W(y) = W_bf * (y / h_bf)^r) within the main channel
    and a continuous compound trapezoidal floodplain that expands gradually above bankfull depth (Dingman 2009, Neal et al. 2012).

    Args:
        effective_depth_m: Effective flow depth across reach interface (meters).
        bankfull_width_m: Bankfull top width of the channel (meters).
        shape_exponent: Power-law cross-sectional shape exponent r (dimensionless).
        bankfull_depth_m: Bankfull channel depth (meters).
        floodplain_width_m: Floodplain width beyond bankfull channel (meters).
        inverse_shape_exponent_plus_one: Precomputed 1 / (r + 1) (dimensionless).
        bankfull_area_m2: Precomputed bankfull cross-sectional area (m²).
        bankfull_perimeter_m: Precomputed bankfull wetted perimeter (meters).
        floodplain_side_slope: Precomputed effective floodplain side-slope parameter z_fp (dimensionless).
        floodplain_depth_threshold_m: Precomputed floodplain depth threshold for vertical expansion (meters).
        floodplain_area_threshold_m2: Precomputed floodplain area threshold for vertical expansion (m²).
        sqrt_one_plus_floodplain_slope_squared: Precomputed factor sqrt(1 + z_fp²) (dimensionless).
        width_over_sqrt_bankfull_depth: Precalculated factor W_bf / sqrt(h_bf) for r=0.5 (m^(1/2)).

    Returns:
        A tuple containing:
            - area: Cross-sectional flow area (m²).
            - top_width: Free surface top width (meters).
            - wetted_perimeter: Wetted perimeter (meters).
    """
    if effective_depth_m <= bankfull_depth_m or bankfull_depth_m <= np.float32(0.0):
        water_depth: np.float32 = max(effective_depth_m, np.float32(0.0))
        if shape_exponent == np.float32(0.5):
            top_width: np.float32 = max(
                np.sqrt(water_depth) * width_over_sqrt_bankfull_depth,
                np.float32(1e-3),
            )
        else:
            depth_ratio: np.float32 = max(water_depth, np.float32(0.0)) / max(
                bankfull_depth_m, np.float32(1e-4)
            )
            top_width = max(
                bankfull_width_m * (depth_ratio**shape_exponent),
                np.float32(1e-3),
            )
        area: np.float32 = top_width * water_depth * inverse_shape_exponent_plus_one
        wetted_perimeter: np.float32 = (
            top_width + np.float32(8.0 / 3.0) * (water_depth * water_depth) / top_width
        )
    else:
        floodplain_depth: np.float32 = max(
            effective_depth_m - bankfull_depth_m, np.float32(0.0)
        )

        if (
            floodplain_width_m > np.float32(0.0)
            and floodplain_depth > floodplain_depth_threshold_m
        ):
            area = (
                bankfull_area_m2
                + floodplain_area_threshold_m2
                + floodplain_width_m * (floodplain_depth - floodplain_depth_threshold_m)
            )
            top_width = bankfull_width_m + floodplain_width_m
            wetted_perimeter = (
                bankfull_perimeter_m
                + floodplain_width_m
                + np.float32(2.0) * (floodplain_depth - floodplain_depth_threshold_m)
            )
        else:
            area = bankfull_area_m2 + floodplain_side_slope * (
                floodplain_depth * floodplain_depth
            )
            actual_floodplain_width: np.float32 = (
                np.float32(2.0) * floodplain_side_slope * floodplain_depth
            )
            top_width = bankfull_width_m + actual_floodplain_width
            wetted_perimeter = (
                bankfull_perimeter_m
                + np.float32(2.0)
                * sqrt_one_plus_floodplain_slope_squared
                * floodplain_depth
            )

    return area, top_width, wetted_perimeter


@njit(inline="always")
def compute_cfl_area_and_top_width(
    volume_m3: np.float32,
    bankfull_width_m: np.float32,
    shape_exponent: np.float32,
    bankfull_depth_m: np.float32,
    floodplain_width_m: np.float32,
    inverse_reach_length: np.float32,
    inverse_shape_exponent_plus_one: np.float32,
    bankfull_area_m2: np.float32,
    bankfull_volume_m3: np.float32,
    floodplain_side_slope: np.float32,
    floodplain_depth_threshold_m: np.float32,
    floodplain_area_threshold_m2: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculates flow area and top width directly from storage volume for CFL evaluation.

    Args:
        volume_m3: Water volume stored in reach (m³).
        bankfull_width_m: Bankfull top width of the channel (meters).
        shape_exponent: Power-law cross-sectional shape exponent r (dimensionless).
        bankfull_depth_m: Bankfull channel depth before floodplain spilling occurs (meters).
        floodplain_width_m: Additional floodplain width beyond bankfull channel (meters).
        inverse_reach_length: Precomputed inverse reach length 1 / L (1/meters).
        inverse_shape_exponent_plus_one: Precomputed 1 / (r + 1) (dimensionless).
        bankfull_area_m2: Precomputed bankfull cross-sectional area (m²).
        bankfull_volume_m3: Precomputed bankfull channel storage volume (m³).
        floodplain_side_slope: Precomputed effective floodplain side-slope parameter z_fp (dimensionless).
        floodplain_depth_threshold_m: Precomputed floodplain depth threshold for vertical expansion (meters).
        floodplain_area_threshold_m2: Precomputed floodplain area threshold for vertical expansion (m²).

    Returns:
        A tuple containing:
            - area: Cross-sectional flow area (m²).
            - top_width: Free surface top width (meters).
    """
    if volume_m3 <= bankfull_volume_m3 or bankfull_depth_m <= np.float32(0.0):
        if volume_m3 <= np.float32(0.0):
            top_width: np.float32 = max(
                bankfull_width_m * np.float32(0.01), np.float32(1e-3)
            )
            return np.float32(0.0), top_width

        volume_ratio: np.float32 = volume_m3 / max(bankfull_volume_m3, np.float32(1e-6))
        depth_ratio: np.float32 = volume_ratio**inverse_shape_exponent_plus_one
        area: np.float32 = volume_m3 * inverse_reach_length

        if shape_exponent == np.float32(0.5):
            top_width = max(
                bankfull_width_m * np.sqrt(depth_ratio),
                np.float32(1e-3),
            )
        else:
            top_width = max(
                bankfull_width_m * (depth_ratio**shape_exponent),
                np.float32(1e-3),
            )
        return area, top_width

    overbank_volume: np.float32 = volume_m3 - bankfull_volume_m3
    overbank_area: np.float32 = overbank_volume * inverse_reach_length

    if (
        floodplain_width_m > np.float32(0.0)
        and overbank_area > floodplain_area_threshold_m2
    ):
        top_width = bankfull_width_m + floodplain_width_m
    elif floodplain_side_slope > np.float32(0.0):
        floodplain_depth: np.float32 = np.sqrt(overbank_area / floodplain_side_slope)
        top_width = (
            bankfull_width_m
            + np.float32(2.0) * floodplain_side_slope * floodplain_depth
        )
    else:
        floodplain_depth = overbank_area / max(floodplain_width_m, np.float32(1e-3))
        top_width = bankfull_width_m

    area = bankfull_area_m2 + overbank_area
    return area, top_width
