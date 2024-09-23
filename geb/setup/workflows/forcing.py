def reproject_and_apply_lapse_rate_temperature(T, DEM, grid_mask, lapse_rate=-0.0065):
    DEM.raster.mask_nodata().fillna(
        0
    )  # assuming 0 for missing DEM values above the ocean
    DEM_grid = DEM.raster.reproject_like(grid_mask, method="average").drop_vars(
        ["dparams", "inplace"]
    )
    DEM_forcing = DEM.raster.reproject_like(T, method="average")

    t_at_sea_level = T - DEM_forcing * lapse_rate
    t_at_sea_level_reprojected = t_at_sea_level.raster.reproject_like(
        DEM_grid, method="average"
    )
    T_grid = t_at_sea_level_reprojected + lapse_rate * DEM_grid
    T_grid.name = T.name
    T_grid.attrs = T.attrs
    return T_grid


def get_pressure_correction_factor(DEM, g, Mo, lapse_rate):
    return (288.15 / (288.15 + lapse_rate * DEM)) ** (g * Mo / (8.3144621 * lapse_rate))


def reproject_and_apply_lapse_rate_pressure(
    pressure,
    DEM,
    grid_mask,
    g=9.80665,
    Mo=0.0289644,
    lapse_rate=-0.0065,
):
    """Pressure correction based on elevation lapse_rate.

    Parameters
    ----------
    dem_model : xarray.DataArray
        DataArray with high res lat/lon axis and elevation data
    g : float, default 9.80665
        gravitational constant [m s-2]

    Mo : float, default 0.0289644
        molecular weight of gas [kg / mol]
    lapse_rate : float, deafult -0.0065
        lapse rate of temperature [C m-1]

    Returns
    -------
    press_fact : xarray.DataArray
        pressure correction factor
    """
    DEM.raster.mask_nodata().fillna(
        0
    )  # assuming 0 for missing DEM values above the ocean
    DEM_grid = DEM.raster.reproject_like(grid_mask, method="average").drop_vars(
        ["dparams", "inplace"]
    )
    DEM_forcing = DEM.raster.reproject_like(pressure, method="average")

    pressure_at_sea_level = pressure / get_pressure_correction_factor(
        DEM_forcing, g, Mo, lapse_rate
    )  # divide by pressure factor to get pressure at sea level
    pressure_at_sea_level_reprojected = pressure_at_sea_level.raster.reproject_like(
        DEM_grid, method="average"
    )
    pressure_grid = (
        pressure_at_sea_level_reprojected
        * get_pressure_correction_factor(DEM_grid, g, Mo, lapse_rate)
    )  # multiply by pressure factor to get pressure at DEM grid, corrected for elevation
    pressure_grid.name = pressure.name
    pressure_grid.attrs = pressure.attrs

    return pressure_grid
