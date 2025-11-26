"""Module for computing damages using the VectorScanner from the damagescanner package.

Since the DamageScanner has a slightly complicated interface, we wrap it in a function that checks for common issues
and provides a simpler interface.

"""

import geopandas as gpd
import pandas as pd
import xarray as xr
from damagescanner.vector import VectorScanner as VectorScannerDS
from damagescanner.vector import VectorExposure as VectorExposureDS
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def compute_all_numba(values, coverage, max_arr, curve_x, curve_y, slopes):
    n_obj = values.size
    n_curves = curve_y.shape[0]
    out = np.empty((n_obj, n_curves))

    for i in prange(n_obj):
        v = values[i]
        cov = coverage[i]
        m = max_arr[i]

        # --- searchsorted (scalar) ---
        j = np.searchsorted(curve_x, v) - 1
        if j < 0:
            j = 0
        elif j >= curve_x.size - 1:
            j = curve_x.size - 2

        # interpolate x coordinate offset
        dx = v - curve_x[j]

        for c in range(n_curves):
            # slope-based linear interpolation
            s = curve_y[c, j] + slopes[c, j] * dx

            out[i, c] = s * cov * m

    return out


def VectorScannerMultiCurves(
    features: gpd.GeoDataFrame,
    hazard: xr.DataArray,
    multi_curves: dict,
    disable_progress: bool = False,
):
    # get vector exposure
    features, object_col, hazard_crs, cell_area_m2 = VectorExposureDS(
        hazard_file=hazard,
        feature_file=features,
        asset_type=None,
        object_col="object_type",
        disable_progress=disable_progress,
        gridded=False,
    )
    # Filter
    filtered = features[features["values"].str.len() > 1].copy()

    vals = filtered["values"].tolist()
    covs = filtered["coverage"].tolist()

    # Vectorized list comprehensions (very fast)
    filtered["coverage_summed"] = [np.sum(c) for c in covs]
    filtered["average_inundation"] = [np.mean(v) for v in vals]

    # calculate damages
    curve_names = list(multi_curves.keys())
    curve_x = multi_curves[curve_names[0]].index.values
    curve_y = np.vstack([multi_curves[name].values for name in curve_names])

    average_inundation_arr = np.stack(
        filtered["average_inundation"].to_numpy()
    )  # shape (n_obj, n_cells)
    coverage_arr = np.stack(filtered["coverage_summed"].to_numpy()) * cell_area_m2
    max_arr = filtered["maximum_damage"].to_numpy()

    curve_x = multi_curves[curve_names[0]].index.values.astype(np.float64)
    curve_y = np.vstack([multi_curves[k].values for k in curve_names]).astype(
        np.float64
    )
    curve_slopes = np.diff(curve_y) / np.diff(curve_x)

    damage_matrix = compute_all_numba(
        average_inundation_arr.astype(np.float64),
        coverage_arr.astype(np.float64),
        max_arr.astype(np.float64),
        curve_x,
        curve_y,
        curve_slopes,
    )

    damage_df = pd.DataFrame(damage_matrix, columns=curve_names, index=filtered.index)

    return damage_df


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
