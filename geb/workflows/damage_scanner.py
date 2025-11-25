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
from tqdm import tqdm


# TODO replace this with numba njit
def _get_damage_per_object(_object, multi_curves, cell_area_m2):
    coverage = np.array(_object["coverage"]) * cell_area_m2
    for curve in multi_curves:
        _object[curve] = (
            np.sum(
                np.interp(
                    _object["values"],
                    multi_curves[curve].index,
                    multi_curves[curve].values,
                )
                * coverage
            )
            * _object["maximum_damage"]
        )

    return _object[multi_curves.keys()]


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
    # calculate damages
    tqdm.pandas(desc="Calculating damage", disable=disable_progress)

    result = features.progress_apply(
        lambda _object: _get_damage_per_object(_object, multi_curves, cell_area_m2),
        axis=1,
    )
    return result


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
