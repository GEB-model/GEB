import numpy as np


def reproject_and_apply_lapse_rate_temperature(
    T, DEM_forcing, DEM_grid, lapse_rate=-0.0065
):
    t_at_sea_level = T - DEM_forcing * lapse_rate
    t_at_sea_level_reprojected = t_at_sea_level.raster.reproject_like(
        DEM_grid, method="average"
    )
    T_grid = t_at_sea_level_reprojected + lapse_rate * DEM_grid
    return T_grid


def get_pressure_correction_factor(DEM, g, R_air, Mo, lapse_rate):
    return np.power(288.15 / (288.15 + lapse_rate * DEM), g * Mo / (R_air * lapse_rate))


def reproject_and_apply_lapse_rate_pressure(
    pressure,
    DEM_forcing,
    DEM_grid,
    g=9.80665,
    R_air=8.3144621,
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
    R_air : float, default 8.3144621
        specific gas constant for dry air [J mol-1 K-1]
    Mo : float, default 0.0289644
        molecular weight of gas [kg / mol]
    lapse_rate : float, deafult -0.0065
        lapse rate of temperature [C m-1]

    Returns
    -------
    press_fact : xarray.DataArray
        pressure correction factor
    """
    pressure_at_sea_level = pressure / get_pressure_correction_factor(
        DEM_forcing, g, R_air, Mo, lapse_rate
    )  # divide by pressure factor to get pressure at sea level
    pressure_at_sea_level_reprojected = pressure_at_sea_level.raster.reproject_like(
        DEM_grid, method="average"
    )
    pressure_grid = (
        pressure_at_sea_level_reprojected
        * get_pressure_correction_factor(DEM_grid, g, R_air, Mo, lapse_rate)
    )  # multiply by pressure factor to get pressure at DEM grid, corrected for elevation

    return pressure_grid
