from pathlib import Path

import xarray as xr
from hydromt_sfincs import SfincsModel, utils


def read_maximum_flood_depth(model_root: Path, simulation_root: Path) -> xr.DataArray:
    """Read the maximum flood depth from the SFINCS model results.

    If SFINCS was run with subgrid, the flood depth is downscaled to subgrid resolution.

    Notes:
        If SFINCS was run as a fluvial model, a minimum flood depth of 0.15 m is applied.
        If SFINCS was run as a pluvial/coastal model, a minimum flood depth of 0.05 m is applied.
        There are some fundamental issues with the current generation of subgrid maps,
            especially in steep terrain, see conclusion here: https://doi.org/10.5194/gmd-18-843-2025
            Here, we use bilinear interpolation as a goodish solution. However, we keep our
            eyes out for better solutions.

    Args:
        model_root: The root path of the SFINCS model directory. This should contain the subgrid directory with the depth file.
        simulation_root: The root path of the SFINCS simulation directory. This should contain the simulation results files.

    Returns:
        The maximum flood depth downscaled to subgrid resolution.
    """

    # Read SFINCS model, config and results
    model: SfincsModel = SfincsModel(
        str(simulation_root),
        mode="r",
    )
    model.read_config()
    model.read_results()

    # get maximum water surface elevation (with respect to sea level)
    water_surface_elevation: xr.DataArray = model.results["zsmax"].max(dim="timemax")

    # to detect whether SFINCS was run with subgrid, we check if the 'sbgfile' key exists in the config
    # to be extra safe, we also check if the value is not None or has has length > 0
    if (
        "sbgfile" in model.config
        and model.config["sbgfile"] is not None
        and len(model.config["sbgfile"]) > 0
    ):
        # read subgrid elevation
        surface_elevation: xr.DataArray = xr.open_dataarray(
            model_root / "subgrid" / "dep_subgrid.tif"
        )["dep_subgrid"]
    else:
        # the the grid elevation from the model
        surface_elevation: xr.DataArray = model.grid.get("dep")

    # Detect whether SFINCS was run with or without precipitation
    # we do this by checking if the 'netamprfile' key exists in the config
    # to be extra safe, we also check if the value is not None or has has length > 0
    if (
        "netamprfile" in model.config
        and model.config["netamprfile"] is not None
        and len(model.config["netamprfile"]) > 0
    ):
        print("Precipitation input detected, applying minimum flood depth of 0.15 m")
        minimum_flood_depth: float = 0.15
    else:
        print("No precipitation input detected, applying minimum flood depth of 0.05 m")
        minimum_flood_depth: float = 0.05

    flood_depth_m: xr.DataArray = utils.downscale_floodmap(
        zsmax=water_surface_elevation,
        dep=surface_elevation,
        hmin=minimum_flood_depth,
        reproj_method="bilinear",  # maybe use "nearest" for coastal
    )

    print(
        f"Maximum flood depth: {float(flood_depth_m.max().values):.2f} m, "
        f"Mean flood depth: {float(flood_depth_m.mean().values):.2f} m"
    )

    return flood_depth_m
