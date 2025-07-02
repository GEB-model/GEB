from pathlib import Path

import xarray as xr
from hydromt_sfincs import SfincsModel, utils


def read_maximum_flood_depth(model_root: Path, simulation_root: Path) -> xr.DataArray:
    """Read the maximum flood depth from the SFINCS model results downscaled to subgrid resolution.

    Args:
        model_root: The root path of the SFINCS model directory. This should contain the subgrid directory with the depth file.
        simulation_root: The root path of the SFINCS simulation directory. This should contain the simulation results files.

    Returns:
        The maximum flood depth downscaled to subgrid resolution.
    """
    mod: SfincsModel = SfincsModel(
        simulation_root,
        mode="r",
    )
    mod.read_config()
    # we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
    mod.read_results()
    zsmax: xr.DataArray = mod.results["zsmax"].max(dim="timemax")

    # read subgrid elevation
    depfile: Path = model_root / "subgrid" / "dep_subgrid.tif"
    dep: xr.DataArray = mod.data_catalog.get_rasterdataset(depfile)

    # we use a threshold to mask minimum flood depth (use 0.05 m for fluvial model and 0.15 for pluvial/coastal model)
    # check if the model is fluvial or pluvial/coastal
    precip_file: Path = simulation_root / "precip_2d.nc"
    if precip_file.exists():
        print("Pluvial/Coastal model")
        hmin: float = 0.15
    else:
        print("Fluvial model")
        hmin: float = 0.05

    # there are some fundamental issues with the current generation of subgrid maps,
    # especially in steep terrain, see conclusion here: https://doi.org/10.5194/gmd-18-843-2025
    # Here, we use bilinear interpolation as a goodish solution.
    # However, we keep our eyes out for better solutions.
    hmax: xr.DataArray = utils.downscale_floodmap(
        zsmax=zsmax,
        dep=dep,
        hmin=hmin,
        reproj_method="bilinear",
    )

    return hmax
