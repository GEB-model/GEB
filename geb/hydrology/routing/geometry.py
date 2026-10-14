"""Cross-sectional geometry calculations for 1D river reaches."""

from typing import NamedTuple

import numpy as np
from numba import njit

from geb.geb_types import (
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    TwoDArrayFloat32,
)

__all__: list[str] = [
    "GEOM_CFL_CONSTANT",
    "GEOM_CFL_MANNING_COEFFICIENT",
    "GEOM_CFL_NUM_COLS",
    "GEOM_IN_BANKFULL_DEPTH",
    "GEOM_IN_BANKFULL_VOLUME",
    "GEOM_IN_BED_ELEVATION",
    "GEOM_IN_INTERFACE_BED_ELEVATION_MAX",
    "GEOM_IN_INVERSE_INTERFACE_LENGTH",
    "GEOM_IN_INVERSE_LENGTH",
    "GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE",
    "GEOM_IN_MANNING_N_SQUARED",
    "GEOM_IN_NUM_COLS",
    "GEOM_IN_STAGE_VOLUME_COEFFICIENT",
    "GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH",
    "GEOM_OV_BANKFULL_PERIMETER",
    "GEOM_OV_FLOODPLAIN_AREA_THRESHOLD",
    "GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD",
    "GEOM_OV_FLOODPLAIN_SIDE_SLOPE",
    "GEOM_OV_FLOODPLAIN_WIDTH",
    "GEOM_OV_NUM_COLS",
    "GEOM_OV_RIVER_WIDTH",
    "InertialGeometry",
    "build_inertial_geometry",
    "compute_cfl_area_and_top_width",
    "compute_cross_section_from_depth",
    "compute_overbank_depth",
    "compute_static_geometry",
    "estimate_lake_outflow",
    "estimate_outflow_height",
    "get_lake_factor",
]

# Column indices for in-bank channel geometry (geom_inbank)
GEOM_IN_INVERSE_LENGTH: int = 0
GEOM_IN_BED_ELEVATION: int = 1
GEOM_IN_BANKFULL_DEPTH: int = 2
GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE: int = 3
GEOM_IN_BANKFULL_VOLUME: int = 4
GEOM_IN_STAGE_VOLUME_COEFFICIENT: int = 5
GEOM_IN_INTERFACE_BED_ELEVATION_MAX: int = 6
GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH: int = 7
GEOM_IN_MANNING_N_SQUARED: int = 8
GEOM_IN_INVERSE_INTERFACE_LENGTH: int = 9
GEOM_IN_NUM_COLS: int = 10

# Column indices for overbank floodplain geometry (geom_overbank)
GEOM_OV_FLOODPLAIN_WIDTH: int = 0
GEOM_OV_RIVER_WIDTH: int = 1
GEOM_OV_BANKFULL_PERIMETER: int = 2
GEOM_OV_FLOODPLAIN_SIDE_SLOPE: int = 3
GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD: int = 4
GEOM_OV_FLOODPLAIN_AREA_THRESHOLD: int = 5
GEOM_OV_NUM_COLS: int = 6

# Column indices for CFL stability parameters (geom_cfl)
GEOM_CFL_MANNING_COEFFICIENT: int = 0
GEOM_CFL_CONSTANT: int = 1
GEOM_CFL_NUM_COLS: int = 2


class InertialGeometry(NamedTuple):
    """Encapsulates static geometric and roughness lookup tables for inertial reaches.

    Attributes:
        geom_inbank: Compact 2D array of in-bank channel geometry attributes (meters, dimensionless).
        geom_overbank: Secondary 2D array of overbank floodplain geometry attributes (meters, dimensionless).
        geom_cfl: 2D array of CFL stability condition parameters per reach.
        pit_slope_inertial: Baseline bed slope assigned to terminal pit/boundary reaches (m/m).
    """

    geom_inbank: TwoDArrayFloat32
    geom_overbank: TwoDArrayFloat32
    geom_cfl: TwoDArrayFloat32
    pit_slope_inertial: ArrayFloat32


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


def build_inertial_geometry(
    river_length: ArrayFloat32,
    river_width: ArrayFloat32,
    shape_exponent: ArrayFloat32,
    bankfull_depth: ArrayFloat32,
    floodplain_width: ArrayFloat32,
    bed_elevation: ArrayFloat32,
    ds_bed_elevation: ArrayFloat32,
    interface_bed_elevation_max: ArrayFloat32,
    manning_n_sq: ArrayFloat32,
    inv_dx_interface: ArrayFloat32,
    up_count: ArrayInt32,
    pit_slope: ArrayFloat32,
) -> InertialGeometry:
    """Precomputes reach-static channel cross-sections, thresholds, and CFL parameters.

    Constructs dense 2D matrices containing in-bank geometry, overbank floodplain
    properties, and CFL stability scaling terms to optimize CPU cache line utilization
    during local inertial substepping.

    Args:
        river_length: Channel length of each reach (meters).
        river_width: Bankfull top width of the main channel (meters).
        shape_exponent: Power-law cross-sectional shape exponent r (dimensionless).
        bankfull_depth: Bankfull channel depth before floodplain spilling (meters).
        floodplain_width: Additional floodplain width beyond bankfull channel (meters).
        bed_elevation: Bed elevation of each reach (meters).
        ds_bed_elevation: Downstream bed elevation for slope calculation (meters).
        interface_bed_elevation_max: Max upstream interface bed elevation (meters).
        manning_n_sq: Square of Manning roughness coefficient n^2 (s^2 / m^(2/3)).
        inv_dx_interface: Inverse spatial step at reach boundaries (1/meters).
        up_count: Count of upstream tributaries entering each reach (dimensionless).
        pit_slope: Default channel slope for boundary pit reaches (dimensionless).

    Returns:
        Structured InertialGeometry containing the packed 2D geometric matrices.
    """
    (
        inv_reach_length,
        inv_shape_exponent_plus_one,
        _bankfull_area,
        bankfull_volume,
        bankfull_perimeter,
        fp_side_slope,
        fp_depth_threshold,
        fp_area_threshold,
        _sqrt_one_plus_fp_slope_sq,
        width_over_sqrt_bankfull_depth,
    ) = compute_static_geometry(
        river_length=river_length,
        river_width=river_width,
        shape_exponent=shape_exponent,
        bankfull_depth=bankfull_depth,
        floodplain_width=floodplain_width,
    )

    stage_vol_coeff: ArrayFloat32 = (
        bankfull_depth / (bankfull_volume**inv_shape_exponent_plus_one)
    ).astype(np.float32)

    n_inertial: int = river_length.shape[0]

    # Precompute in-bank channel geometry
    geom_inbank: TwoDArrayFloat32 = np.empty(
        (n_inertial, GEOM_IN_NUM_COLS), dtype=np.float32
    )
    geom_inbank[:, GEOM_IN_INVERSE_LENGTH] = inv_reach_length
    geom_inbank[:, GEOM_IN_BED_ELEVATION] = bed_elevation
    geom_inbank[:, GEOM_IN_BANKFULL_DEPTH] = bankfull_depth
    geom_inbank[:, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE] = (
        inv_shape_exponent_plus_one
    )
    geom_inbank[:, GEOM_IN_BANKFULL_VOLUME] = bankfull_volume
    geom_inbank[:, GEOM_IN_STAGE_VOLUME_COEFFICIENT] = stage_vol_coeff
    geom_inbank[:, GEOM_IN_INTERFACE_BED_ELEVATION_MAX] = interface_bed_elevation_max
    geom_inbank[:, GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH] = (
        width_over_sqrt_bankfull_depth
    )
    geom_inbank[:, GEOM_IN_MANNING_N_SQUARED] = manning_n_sq
    geom_inbank[:, GEOM_IN_INVERSE_INTERFACE_LENGTH] = inv_dx_interface

    # Precompute overbank floodplain geometry
    geom_overbank: TwoDArrayFloat32 = np.empty(
        (n_inertial, GEOM_OV_NUM_COLS), dtype=np.float32
    )
    geom_overbank[:, GEOM_OV_FLOODPLAIN_WIDTH] = floodplain_width
    geom_overbank[:, GEOM_OV_RIVER_WIDTH] = river_width
    geom_overbank[:, GEOM_OV_BANKFULL_PERIMETER] = bankfull_perimeter
    geom_overbank[:, GEOM_OV_FLOODPLAIN_SIDE_SLOPE] = fp_side_slope
    geom_overbank[:, GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD] = fp_depth_threshold
    geom_overbank[:, GEOM_OV_FLOODPLAIN_AREA_THRESHOLD] = fp_area_threshold

    # Precompute CFL stability parameters
    geom_cfl: TwoDArrayFloat32 = np.empty(
        (n_inertial, GEOM_CFL_NUM_COLS), dtype=np.float32
    )
    bed_drop: ArrayFloat32 = np.maximum(
        bed_elevation - ds_bed_elevation,
        np.float32(0.0),
    )
    eff_slope: ArrayFloat32 = np.maximum(bed_drop * inv_reach_length, np.float32(1e-4))
    manning_n: ArrayFloat32 = np.sqrt(manning_n_sq)
    geom_cfl[:, GEOM_CFL_MANNING_COEFFICIENT] = (
        (manning_n / (river_width * np.sqrt(eff_slope))) ** np.float32(0.6)
    ).astype(np.float32)
    cfl_safety_factor: np.float32 = np.float32(0.85)
    sqrt_g: np.float32 = np.sqrt(np.float32(9.80665))
    cfl_dx: ArrayFloat32 = river_length
    geom_cfl[:, GEOM_CFL_CONSTANT] = (cfl_safety_factor * cfl_dx / sqrt_g).astype(
        np.float32
    )

    return InertialGeometry(
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        geom_cfl=geom_cfl,
        pit_slope_inertial=pit_slope,
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
    inverse_shape_exponent_plus_one: np.float32,
    bankfull_area_m2: np.float32,
    bankfull_perimeter_m: np.float32,
    width_over_sqrt_bankfull_depth: np.float32,
) -> tuple[np.float32, np.float32, np.float32]:
    """Calculates cross-sectional area, top width, and wetted perimeter from effective interface depth.

    The river channel has curved sides (a bowl or parabola shape). When water
    rises above the river banks, it spills onto the floodplain next to it.

    In CaMa-Flood style routing (Yamazaki et al. 2011, 2012), water exceeding bankfull
    depth spills onto the floodplain where it is stored (acting as a capacitive retention /
    dead zone that attenuates the flood wave and raises the water level), while downstream
    longitudinal conveyance between reaches is carried by the river channel section (channel
    area plus the column of water directly above the river channel).

    Args:
        effective_depth_m: Water depth at the boundary between two river reaches (meters).
        bankfull_width_m: River channel width when full to the top of its banks (meters).
        shape_exponent: Channel shape factor: how curved the river bed is, where 0.5 is a parabola (dimensionless).
        bankfull_depth_m: Channel depth when full to the top of its banks (meters).
        inverse_shape_exponent_plus_one: Precomputed value of 1 / (shape_exponent + 1) (dimensionless).
        bankfull_area_m2: Water area when the channel is full to the banks (m²).
        bankfull_perimeter_m: Channel bed length touching water when full to the banks (meters).
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

        # In CaMa-Flood style routing, the floodplain acts as a storage reservoir (dead zone):
        # water stored on the floodplain attenuates the flood wave and raises the water surface level,
        # but downstream conveyance between reaches is carried by the river channel section (channel
        # area plus the column of water directly above the river channel).
        area = bankfull_area_m2 + bankfull_width_m * floodplain_depth
        top_width = bankfull_width_m
        wetted_perimeter = bankfull_perimeter_m

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


def get_lake_factor(
    river_width: ArrayFloat32,
    overflow_coefficient_mu: np.float32 = np.float32(0.612),
    lake_a_factor: np.float32 = np.float32(1.0),
    reference_depth_m: np.float32 = np.float32(1.0),
) -> ArrayFloat32:
    """Calculates the parabolic weir rating discharge multiplier for lake outlets.

    For a parabolic weir notch with equation z = c * x^2, the notch curvature is
    c = 4 * h_ref / b_w^2, where b_w is the outlet river width at reference depth h_ref.
    The discharge formula is Q = mu * sqrt(g / c) * h^2 = lake_factor * h^2.

    References:
        http://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf

    Args:
        river_width: Width of the outlet river (meters).
        overflow_coefficient_mu: Weir discharge coefficient (dimensionless, 0.612 for parabola).
        lake_a_factor: Calibration multiplier factor for lake outflow (dimensionless).
        reference_depth_m: Reference flow depth for parabolic notch curvature (meters).

    Returns:
        A lake-specific constant factor used in the parabolic lake outflow equations (m/s).
    """
    gravity: np.float32 = np.float32(9.80665)
    return (
        lake_a_factor
        * overflow_coefficient_mu
        * (river_width * np.float32(0.5))
        * np.sqrt(gravity / reference_depth_m)
    )


def estimate_outflow_height(
    lake_capacity: ArrayFloat64,
    lake_factor: ArrayFloat32,
    lake_area: ArrayFloat32,
    avg_outflow: ArrayFloat32,
) -> ArrayFloat32:
    """Estimates the weir sill height of a lake above bottom from capacity, area, and average outflow.

    Args:
        lake_capacity: Capacity of the lake (m3).
        lake_factor: Parabolic weir discharge rating factor (m/s).
        lake_area: Surface area of the lake (m2).
        avg_outflow: Long-term average outflow discharge (m3/s).

    Returns:
        Weir sill outflow height above the lake bottom (meters).
    """
    height_above_outflow: ArrayFloat32 = np.sqrt(avg_outflow / lake_factor).astype(
        np.float32
    )
    lake_height_when_full: ArrayFloat32 = (lake_capacity / lake_area).astype(np.float32)
    outflow_height: ArrayFloat32 = np.maximum(
        lake_height_when_full - height_above_outflow, np.float32(0.0)
    )
    return outflow_height


def estimate_lake_outflow(
    lake_factor: ArrayFloat32,
    height_above_outflow: ArrayFloat32,
) -> ArrayFloat32:
    """Estimates the outflow from a lake given its head above the weir sill for a parabolic weir.

    References:
        http://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf

    Args:
        lake_factor: Parabolic weir discharge rating factor (m/s).
        height_above_outflow: Head of lake surface above weir sill (meters).

    Returns:
        Outflow discharge (m3/s).
    """
    return lake_factor * (height_above_outflow * height_above_outflow)
