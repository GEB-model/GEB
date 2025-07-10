import logging

import numpy as np
import numpy.typing as npt
import pyflwdir
import xarray as xr
from hydromt_sfincs import SfincsModel

from geb.hydrology.routing import get_river_width

from .io import export_rivers
from .sfincs_utils import (
    assign_return_periods,
    get_discharge_by_river,
    get_logger,
    get_representative_river_points,
)

logger = logging.getLogger(__name__)


def get_river_depth(river_segments, method, parameters, bankfull_column):
    if method == "manning":
        # Set a minimum value for 'rivslp'
        min_rivslp = 1e-5
        # Replace NaN values in 'rivslp' with the minimum value
        slope = river_segments["slope"].fillna(min_rivslp)
        # Replace 'rivslp' values with the minimum value where they are less than the minimum
        slope = np.where(
            slope < min_rivslp,
            min_rivslp,
            slope,
        )
        # Calculate 'river depth' using the Manning equation
        depth = (
            (0.030 * river_segments[bankfull_column])
            / (np.sqrt(slope) * river_segments["width"])
        ) ** (3 / 5)

    elif method == "power_law":
        # Calculate 'river depth' using the power law equation
        # Powerlaw equation from Andreadis et al (2013)
        c = parameters["c"]
        d = parameters["d"]
        depth = c * (river_segments[bankfull_column].astype(float) ** d)

    else:
        raise ValueError(f"Unknown depth calculation method: {method}")

    # Set a minimum value for 'river depth'
    # Note: Making this value higher or lower can affect results
    min_rivdph = 0
    # Replace 'rivslp' values with the minimum value where they are less than the minimum
    depth = np.where(depth < min_rivdph, min_rivdph, depth)
    # Convert 'rivdph' to float
    return depth.astype(np.float32)


def get_river_manning(river_segments):
    return np.full(len(river_segments), 0.02)


def do_mask_flood_plains(sf):
    elevation, d8 = pyflwdir.dem.fill_depressions(sf.grid.dep.values)

    flw = pyflwdir.from_array(
        d8,
        transform=sf.grid.raster.transform,
        latlon=False,
    )
    floodplains = flw.floodplains(elevation, upa_min=10)

    mask = xr.full_like(sf.grid.dep, 0, dtype=np.uint8).rename("mask")
    mask.raster.set_nodata(0)
    mask.values = floodplains.astype(mask.dtype)
    sf.set_grid(mask, name="msk")
    sf.config.update({"mskfile": "sfincs.msk"})
    sf.config.update({"indexfile": "sfincs.ind"})


def build_sfincs(
    model_root,
    DEMs,
    region,
    rivers,
    discharge,
    river_width_alpha: npt.NDArray[np.float32],
    river_width_beta: npt.NDArray[np.float32],
    mannings,
    resolution,
    nr_subgrid_pixels,
    crs,
    depth_calculation_method,
    depth_calculation_parameters,
    derive_river_method=None,
    mask_flood_plains=False,
):
    """Build a SFINCS model for a given basin_id and configuration file.

    Parameters
    ----------
    basin_id : int or None
        The Pfafstetter ID of the basin to build the model for.
    config_fn : str
        The path to the configuration file.
    model_root : str
        The path to the model root directory.
    depth_calculation : str
        The method to use for calculating river depth. Can be 'manning' or 'power_law'
    unsnapped_method : str
        The method to use for setting up the unsnapped points. Can be 'outflow' or 'nearest'
    derive_river_method : str
        The method to use for setting up the river segments. Can be 'default' or 'detailed'
    upstream_area_threshold : float
        If using the detailed method, set the minimum upstream area size for something to be considered a river. Standard 1 km2
    """
    assert "width" in rivers.columns, "Width must be provided in rivers"

    assert not (derive_river_method and rivers), (
        "Specify either derive_river_method or rivers, not both"
    )
    assert depth_calculation_method in [
        "manning",
        "power_law",
    ], "Method should be 'manning' or 'power_law'"

    # build base model
    sf = SfincsModel(root=model_root, mode="w+", logger=get_logger())

    sf.setup_grid_from_region({"geom": region}, res=resolution, crs=crs, rotated=False)

    DEMs = [{**DEM, **{"reproj_method": "bilinear"}} for DEM in DEMs]

    # HydroMT-SFINCS only accepts datasets with an 'elevtn' variable. Therefore, the following
    # is a bit convoluted. We first open the dataarray, then convert it to a dataset,
    # and set the name as elevtn.
    sf.setup_dep(datasets_dep=DEMs)

    if mask_flood_plains:
        do_mask_flood_plains(sf)
    else:
        sf.setup_mask_active(
            region, zmin=-21, reset_mask=True
        )  # TODO: Improve mask setup

    # Setup river inflow points
    sf.setup_river_inflow(
        rivers=rivers,
        keep_rivers_geom=True,
        river_upa=0,
        river_len=0,
    )

    # Setup river outflow points
    sf.setup_river_outflow(
        rivers=rivers,
        keep_rivers_geom=True,
        river_upa=0,
        river_len=0,
    )

    river_representative_points = []
    for ID in rivers.index:
        river_representative_points.append(get_representative_river_points(ID, rivers))

    discharge_by_river, river_width_alpha_per_river, river_width_beta_per_river = (
        get_discharge_by_river(
            rivers.index.tolist(),
            river_representative_points,
            discharge=discharge,
            river_width_alpha=river_width_alpha,
            river_width_beta=river_width_beta,
        )
    )
    rivers = assign_return_periods(rivers, discharge_by_river, return_periods=[2])

    river_width_unknown_mask = rivers["width"].isnull()

    rivers.loc[river_width_unknown_mask, "width"] = get_river_width(
        river_width_alpha_per_river[river_width_unknown_mask],
        river_width_beta_per_river[river_width_unknown_mask],
        rivers.loc[river_width_unknown_mask, "Q_2"],
    )

    rivers["depth"] = get_river_depth(
        rivers,
        method=depth_calculation_method,
        parameters=depth_calculation_parameters,
        bankfull_column="Q_2",
    )
    rivers["manning"] = get_river_manning(rivers)

    export_rivers(model_root, rivers)

    # Because hydromt-sfincs does a lot of filling default values when data
    # is missing, we need to be extra sure that the required columns are
    # present and contain valid data.
    assert rivers["depth"].notnull().all(), "River depth cannot be null"
    assert rivers["manning"].notnull().all(), "River Manning's n cannot be null"
    assert rivers["width"].notnull().all(), "River width cannot be null"

    sf.setup_subgrid(
        datasets_dep=DEMs,
        datasets_rgh=[
            {
                "manning": mannings.to_dataset(name="manning"),
            }
        ],
        datasets_riv=[
            {
                "centerlines": rivers.rename(
                    columns={"width": "rivwth", "depth": "rivdph"}
                )
            }
        ],
        write_dep_tif=True,
        write_man_tif=True,
        nr_subgrid_pixels=nr_subgrid_pixels,
        nlevels=20,
        nrmax=500,
    )

    # write all components, except forcing which must be done after the model building
    sf.write_grid()
    sf.write_geoms()
    sf.write_config()
    sf.write_subgrid()

    sf.plot_basemap(fn_out="basemap.png")
