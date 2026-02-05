"""Module for computing damages using the VectorScanner from the damagescanner package.

Since the DamageScanner has a slightly complicated interface, we wrap it in a function that checks for common issues
and provides a simpler interface.

"""

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from damagescanner.vector import (
    VectorExposure as VectorExposureDS,
    VectorScanner as VectorScannerDS,
)
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def compute_all_numba(
    values: np.ndarray,
    coverage: np.ndarray,
    max_damage_arr: np.ndarray,
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    slopes: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated function to compute damages for multiple objects and multiple curves.

    Args:
        values (np.ndarray): Array of shape (n_obj,) containing inundation
            depth in meters for each building part.
        coverage (np.ndarray): Array of shape (n_obj,) containing the covered
            area (in m²) for each building part.
        max_damage_arr (np.ndarray): Array of shape (n_obj,) containing the
            maximum damage (in EUR) for the building that each part belongs to.
        curve_x (np.ndarray): Array of shape (n_points,) containing the
            inundation depths for the damage curves.
        curve_y (np.ndarray): Array of shape (n_curves, n_points) containing
            the damage factors (0–1) for each damage curve.
        slopes (np.ndarray): Array of shape (n_curves, n_points - 1)
            containing pre-computed slopes for fast interpolation.

    Returns:
        np.ndarray: Output array of shape (n_obj, n_curves) containing computed
        damages for each building part and each curve.
    """
    n_obj = values.size
    n_curves = curve_y.shape[0]
    out = np.empty((n_obj, n_curves))

    for i in prange(n_obj):  # ty: ignore[not-iterable]
        v = values[i]
        cov = coverage[i]
        m = max_damage_arr[i]

        j = max(np.searchsorted(curve_x, v) - 1, 0)
        dx = v - curve_x[j]

        for c in range(n_curves):
            s = curve_y[c, j] + slopes[c, j] * dx
            out[i, c] = s * cov * m

    return out


def VectorScannerMultiCurves(
    features: gpd.GeoDataFrame,
    hazard: xr.DataArray,
    multi_curves: dict[str, pd.Series],
) -> pd.DataFrame:
    """Computes flood damages for a set of building features using multiple depth-damage curves, with heavy optimizations for large datasets.

    This function wraps the DamageScanner VectorExposure and applies
    efficient numerical routines to compute damage across multiple curves
    in a single pass.

    Args:
        features (gpd.GeoDataFrame):
            GeoDataFrame of building footprints or asset features for which
            flood damages should be computed.
        hazard (xr.DataArray):
            Flood raster (inundation depth in meters) aligned to a regular grid.
        multi_curves (dict):
            Dictionary mapping curve names → pandas Series of
            damage fraction (01) indexed by inundation depth (m).
            Example:
                {
                    "unmitigated": pd.Series(...),
                    "floodproofed": pd.Series(...)
                }
    Returns:
        pd.DataFrame:
            A dataframe indexed by building ID (same index as input features),
            with one column per damage curve, containing damages in EUR.
    Raises:
        ValueError: If the curves do not share the same x-values or if required columns are missing from the features GeoDataFrame.
    """
    curve_names: list[str] = list(multi_curves.keys())

    # Shared x-values
    curve_x = multi_curves[curve_names[0]].index.values.astype(np.float64)

    # assert all curves have the same x-values
    for n in curve_names[1:]:
        if not np.array_equal(curve_x, multi_curves[n].index.values.astype(np.float64)):
            raise ValueError(
                f"All curves must have the same x-values. Curve '{n}' does not match."
            )

    curve_x = np.append(curve_x, 1e10)  # sentinel value for safe searchsorted

    # Stack curve y-values
    curve_y = np.vstack([multi_curves[n].values for n in curve_names]).astype(
        np.float64
    )
    curve_y = np.hstack(
        (curve_y, curve_y[:, -1][:, None])
    )  # extend to same length as curve_x

    # Precompute slopes to avoid calling numpy interpolators inside Numba
    curve_slopes = np.diff(curve_y) / np.diff(curve_x)

    # Ensure C-contiguous for Numba
    curve_x = np.ascontiguousarray(curve_x)
    curve_y = np.ascontiguousarray(curve_y)
    curve_slopes = np.ascontiguousarray(curve_slopes)

    # Extract exposure geometry
    features, _, _, cell_area_m2 = VectorExposureDS(
        hazard_file=hazard,
        feature_file=features,
        object_col="object_type",
        disable_progress=True,
        gridded=False,
    )

    # Keep only inundated buildings
    filtered = features[features["values"].str.len() > 0].copy()
    filtered = filtered[filtered["values"].apply(sum) > 0]
    filtered["len_values"] = filtered["values"].apply(len).astype(np.int32)

    # Efficient list flattening
    vals = (v for sub in filtered["values"].array for v in sub)
    covs = (c for sub in filtered["coverage"].array for c in sub)

    inundation_parts = np.fromiter(vals, dtype=np.float64)
    coverage_parts = np.fromiter(covs, dtype=np.float64) * cell_area_m2

    # Clip hazard values for stable searchsorted
    inundation_parts = np.clip(inundation_parts, curve_x[0], curve_x[-2])

    # check if maximum damage columns are present
    if "maximum_damage_structure" not in filtered.columns:
        raise ValueError(
            "The features GeoDataFrame must contain a 'maximum_damage_structure' column."
        )
    if "maximum_damage_content" not in filtered.columns:
        raise ValueError(
            "The features GeoDataFrame must contain a 'maximum_damage_content' column."
        )

    # Maximum damage per building-part (broadcasted)
    max_damage_arr_structure = np.fromiter(
        (
            dmg
            for len_v, dmg in zip(
                filtered["len_values"], filtered["maximum_damage_structure"]
            )
            for _ in range(len_v)
        ),
        dtype=np.float64,
    )

    max_damage_arr_content = np.fromiter(
        (
            dmg
            for len_v, dmg in zip(
                filtered["len_values"], filtered["maximum_damage_content"]
            )
            for _ in range(len_v)
        ),
        dtype=np.float64,
    )

    # Initiate aggregate per building (vectorized)
    lengths = filtered["len_values"].to_numpy()
    starts = np.r_[0, lengths.cumsum()[:-1]]

    # Compute damages for every part
    # only select curves relevant for structure
    # find index of curves relevant for structure based on index searching for "structure" in curve names
    i_curves_structure = [
        i for i, n in enumerate(curve_names) if "structure" in n.lower()
    ]
    curve_structure = curve_y[i_curves_structure, :]
    slopes_structure = curve_slopes[i_curves_structure, :]
    damage_matrix_structure = compute_all_numba(
        inundation_parts,
        coverage_parts,
        max_damage_arr_structure,
        curve_x,
        curve_structure,
        slopes_structure,
    )

    damage_matrix_structure_final = np.add.reduceat(
        damage_matrix_structure, starts, axis=0
    )
    # Return as DataFrame
    df_damage_structure = pd.DataFrame(
        damage_matrix_structure_final,
        columns=np.array(curve_names)[i_curves_structure],
        index=filtered.index,
    )
    # fill missing buildings with zero damage
    df_damage_structure = df_damage_structure.reindex(features.index, fill_value=0.0)

    # only select curves relevant for content
    i_curves_content = [i for i, n in enumerate(curve_names) if "content" in n.lower()]
    curve_content = curve_y[i_curves_content, :]
    slopes_content = curve_slopes[i_curves_content, :]
    damage_matrix_content = compute_all_numba(
        inundation_parts,
        coverage_parts,
        max_damage_arr_content,
        curve_x,
        curve_content,
        slopes_content,
    )

    damage_matrix_content_final = np.add.reduceat(damage_matrix_content, starts, axis=0)

    # Return as DataFrame
    df_damage_content = pd.DataFrame(
        damage_matrix_content_final,
        columns=np.array(curve_names)[i_curves_content],
        index=filtered.index,
    )
    # fill missing buildings with zero damage
    df_damage_content = df_damage_content.reindex(features.index, fill_value=0.0)

    # concat both dataframes
    df_damage_combined = pd.concat([df_damage_structure, df_damage_content], axis=1)
    return df_damage_combined


def VectorScanner(
    features: gpd.GeoDataFrame,
    hazard: xr.DataArray,
    vulnerability_curves: pd.DataFrame,
    disable_progress: bool = False,
) -> pd.Series:
    """VectorScanner function to compute damages based on features, hazard, and vulnerability curves.

    Notes:
        Gridded may be turned on some time in the future when bugs are fixed in the damagescanner package.

    Args:
        features: GeoDataFrame containing features such as buildings, roads, etc. Must have a column 'maximum_damage' and 'object_type'.
        hazard: DataArray representing the hazard data.
        vulnerability_curves: DataFrame containing vulnerability curves for different object types. For each 'object_type', there should be a corresponding curve.
            The index is used to map the severity (hazard level) to the damage ratio.
        disable_progress: If True, disables the progress bar during processing.

    Returns:
       Series containing the computed damages for each feature.
    """
    assert "maximum_damage" in features.columns, (
        "The features GeoDataFrame must contain a 'maximum_damage' column."
    )
    assert "object_type" in features.columns, (
        "The features GeoDataFrame must contain an 'object_type' column."
    )
    assert features["object_type"].isin(vulnerability_curves.columns).all(), (
        "All unique object_types in the features GeoDataFrame must be present as columns in the vulnerability_curves DataFrame."
    )
    return VectorScannerDS(
        feature_file=features,
        hazard_file=hazard,
        curve_path=vulnerability_curves,
        gridded=False,
        disable_progress=disable_progress,
    )["damage"]
