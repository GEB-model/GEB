import os
from os.path import join

from hydromt_sfincs import SfincsModel, utils


def read_flood_map(model_root, simulation_root, floodmap_name):
    mod = SfincsModel(
        simulation_root,
        mode="r",
    )
    mod.read_config()
    # we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
    mod.read_results()
    zsmax = mod.results["zsmax"].max(dim="timemax")

    # read subgrid elevation
    depfile = join(model_root, "subgrid", "dep_subgrid.tif")
    dep = mod.data_catalog.get_rasterdataset(depfile)

    # we use a threshold to mask minimum flood depth (use 0.05 m for fluvial model and 0.15 for pluvial/coastal model)
    # check if the model is fluvial or pluvial/coastal
    precip_file = join(simulation_root, "precip_2d.nc")
    if os.path.exists(precip_file):
        print("Pluvial/Coastal model")
        hmin = 0.15
    else:
        print("Fluvial model")
        hmin = 0.05

    # there are some fundamental issues with the current generation of subgrid maps,
    # especially in steep terrain, see conclusion here: https://doi.org/10.5194/gmd-18-843-2025
    # Here, we use bilinear interpolation as a goodish solution.
    # However, we keep our eyes out for better solutions.
    hmax = utils.downscale_floodmap(
        zsmax=zsmax,
        dep=dep,
        hmin=hmin,
        reproj_method="bilinear",
    )

    hmax.rio.to_raster(join(simulation_root, f"{floodmap_name}.tif"))

    return hmax
