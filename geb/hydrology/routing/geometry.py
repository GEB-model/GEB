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

    The river channel has curved sides (a bowl or parabola shape). When water
    rises above the river banks, it spills onto the flat floodplain next to it.

    When water floods over the banks, deep water in the river still moves fast,
    while shallow water on the floodplain moves slowly. To prevent the shallow
    floodplain water from making the whole river appear to suddenly slow down,
    we calculate the river and the floodplain separately and combine how easily
    water can move through both.

    Args:
        effective_depth_m: Water depth at the boundary between two river reaches (meters).
        bankfull_width_m: River channel width when full to the top of its banks (meters).
        shape_exponent: Channel shape factor: how curved the river bed is, where 0.5 is a parabola (dimensionless).
        bankfull_depth_m: Channel depth when full to the top of its banks (meters).
        floodplain_width_m: Extra width available on the floodplain next to the channel (meters).
        inverse_shape_exponent_plus_one: Precomputed value of 1 / (shape_exponent + 1) (dimensionless).
        bankfull_area_m2: Water area when the channel is full to the banks (m²).
        bankfull_perimeter_m: Channel bed length touching water when full to the banks (meters).
        floodplain_side_slope: Side slope of the floodplain banks (horizontal to vertical ratio) (dimensionless).
        floodplain_depth_threshold_m: Water depth where the sloping floodplain reaches its full width (meters).
        floodplain_area_threshold_m2: Extra water area when the floodplain reaches its full width (m²).
        sqrt_one_plus_floodplain_slope_squared: Precomputed factor sqrt(1 + slope²) (dimensionless).
        width_over_sqrt_bankfull_depth: Precomputed shortcut for parabola channels (r=0.5) (m^(1/2)).

    Returns:
        A tuple containing:
            - area: Cross-sectional flow area (m²).
            - top_width: Free surface top width (meters).
            - wetted_perimeter: Wetted perimeter (meters).
    """
    if effective_depth_m <= bankfull_depth_m or bankfull_depth_m <= np.float32(0.0):
        # Water stays inside the river banks.
        water_depth: np.float32 = max(effective_depth_m, np.float32(0.0))
        if shape_exponent == np.float32(0.5):
            # Fast shortcut for a standard bowl / parabola shape.
            top_width: np.float32 = max(
                np.sqrt(water_depth) * width_over_sqrt_bankfull_depth,
                np.float32(1e-3),
            )
        else:
            # General curved shape: the channel gets wider as the water gets deeper.
            depth_ratio: np.float32 = max(water_depth, np.float32(0.0)) / max(
                bankfull_depth_m, np.float32(1e-4)
            )
            top_width = max(
                bankfull_width_m * (depth_ratio**shape_exponent),
                np.float32(1e-3),
            )
        # Flow area under the curved channel bed.
        area: np.float32 = top_width * water_depth * inverse_shape_exponent_plus_one
        # Length of the river bed touching water (wetted perimeter).
        wetted_perimeter: np.float32 = (
            top_width + np.float32(8.0 / 3.0) * (water_depth * water_depth) / top_width
        )
    else:
        # Water has risen above the river banks and spilled onto the floodplain.
        floodplain_depth: np.float32 = max(
            effective_depth_m - bankfull_depth_m, np.float32(0.0)
        )

        if (
            floodplain_width_m > np.float32(0.0)
            and floodplain_depth > floodplain_depth_threshold_m
        ):
            # Water is deep enough that the floodplain has expanded to its full width.
            # Above this level, water rises straight up between the outer valley walls.
            area = (
                bankfull_area_m2
                + floodplain_area_threshold_m2
                + floodplain_width_m * (floodplain_depth - floodplain_depth_threshold_m)
            )
            top_width = bankfull_width_m + floodplain_width_m
            actual_floodplain_width: np.float32 = floodplain_width_m
        else:
            # Water is still spreading out sideways along the sloping floodplain edges.
            area = bankfull_area_m2 + floodplain_side_slope * (
                floodplain_depth * floodplain_depth
            )
            actual_floodplain_width = (
                np.float32(2.0) * floodplain_side_slope * floodplain_depth
            )
            top_width = bankfull_width_m + actual_floodplain_width

        # Why we separate the river and the floodplain:
        # Deep river water moves fast, while shallow floodplain water moves slowly.
        # If we mixed them together into one shape, the huge width of the floodplain
        # would make it look like the fast river suddenly slowed down.
        # To avoid this, we calculate how easily water flows through each part separately,
        # then add their flow capacities together.

        # 1. Main channel flow: river channel area plus the column of water directly above it.
        channel_area: np.float32 = min(
            bankfull_area_m2 + bankfull_width_m * floodplain_depth, area
        )
        # 2. Floodplain flow: shallow water spreading out to the sides.
        floodplain_area: np.float32 = max(area - channel_area, np.float32(0.0))

        # Flow efficiency (hydraulic radius = area / bed length touching water) for each part.
        r_channel: np.float32 = channel_area / max(
            bankfull_perimeter_m, np.float32(1e-3)
        )
        p_floodplain: np.float32 = max(actual_floodplain_width, np.float32(1e-3))
        r_floodplain: np.float32 = floodplain_area / p_floodplain

        # How easily water flows through each part (Manning's flow factor: area * radius^(2/3)).
        conveyance_channel: np.float32 = channel_area * (
            r_channel ** np.float32(2.0 / 3.0)
        )
        conveyance_floodplain: np.float32 = floodplain_area * (
            r_floodplain ** np.float32(2.0 / 3.0)
        )
        total_conveyance: np.float32 = conveyance_channel + conveyance_floodplain

        # Convert the total flow capacity back into an equivalent wetted perimeter
        # so the standard flow equation can use it.
        effective_r_two_thirds: np.float32 = total_conveyance / max(
            area, np.float32(1e-6)
        )
        effective_hydraulic_radius: np.float32 = effective_r_two_thirds ** np.float32(
            1.5
        )
        wetted_perimeter = area / max(effective_hydraulic_radius, np.float32(1e-6))

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
