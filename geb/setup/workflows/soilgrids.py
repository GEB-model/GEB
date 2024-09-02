# --------------------------------------------------------------------------------
# This file contains code that has been adapted from an original source available
# in a public repository under the GNU General Public License. The original code
# has been modified to fit the specific needs of this project.
#
# Original source repository: https://github.com/Deltares/hydromt_wflow/
# Files:
# - https://github.com/Deltares/hydromt_wflow/blob/main/hydromt_wflow/workflows/ptf.py
# - https://github.com/Deltares/hydromt_wflow/blob/main/hydromt_wflow/workflows/soilgrids.py
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------

import xarray as xr
import numpy as np


def get_pore_size_index_brakensiek(sand, thetas, clay):
    """
    Determine Brooks-Corey pore size distribution index [-].

    Thetas is equal to porosity (Φ) in this case.

    Based on:
      Rawls,W. J., and Brakensiek, D. L.: Estimation of SoilWater Retention and
      Hydraulic Properties, In H. J. Morel-Seytoux (Ed.),
      Unsaturated flow in hydrologic modelling - Theory and practice, NATO ASI Series 9,
      275-300, Dordrecht, The Netherlands: Kluwer Academic Publishing, 1989.

    Parameters
    ----------
    sand: float
        sand percentage [%].
    thetas : float
        saturated water content [m3/m3].
    clay: float
        clay percentage [%].

    Returns
    -------
    poresizeindex : float
        pore size distribution index [-].

    """
    poresizeindex = np.exp(
        -0.7842831
        + 0.0177544 * sand
        - 1.062498 * thetas
        - (5.304 * 10**-5) * (sand**2)
        - 0.00273493 * (clay**2)
        + 1.11134946 * (thetas**2)
        - 0.03088295 * sand * thetas
        + (2.6587 * 10**-4) * (sand**2) * (thetas**2)
        - 0.00610522 * (clay**2) * (thetas**2)
        - (2.35 * 10**-6) * (sand**2) * clay
        + 0.00798746 * (clay**2) * thetas
        - 0.00674491 * (thetas**2) * clay
    )

    return poresizeindex


def get_pore_size_index(ds):
    """
    Determine pore size distribution index per soil layer depth based on PTF.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil properties at each soil depth [sl1 - sl7].
    thetas: xarray.Dataset
        Dataset containing thetaS at each soil layer depth.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing pore size distribution index [-] for each soil layer
        depth.

    """
    ds_out = xr.apply_ufunc(
        get_pore_size_index_brakensiek,
        ds["sand"],
        ds["thetas"],
        ds["clay"],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )
    ds_out.name = "pore_size"
    return ds_out


def kv_brakensiek(thetas, clay, sand):
    """
    Determine saturated hydraulic conductivity kv [m/day].

    Based on:
      Brakensiek, D.L., Rawls, W.J.,and Stephenson, G.R.: Modifying scs hydrologic
      soil groups and curve numbers for range land soils, ASAE Paper no. PNR-84-203,
      St. Joseph, Michigan, USA, 1984.

    Parameters
    ----------
    thetas: float
        saturated water content [m3/m3].
    clay : float
        clay percentage [%].
    sand: float
        sand percentage [%].

    Returns
    -------
    kv : float
        saturated hydraulic conductivity [m/day].

    """
    kv = (
        np.exp(
            19.52348 * thetas
            - 8.96847
            - 0.028212 * clay
            + (1.8107 * 10**-4) * sand**2
            - (9.4125 * 10**-3) * clay**2
            - 8.395215 * thetas**2
            + 0.077718 * sand * thetas
            - 0.00298 * sand**2 * thetas**2
            - 0.019492 * clay**2 * thetas**2
            + (1.73 * 10**-5) * sand**2 * clay
            + 0.02733 * clay**2 * thetas
            + 0.001434 * sand**2 * thetas
            - (3.5 * 10**-6) * clay**2 * sand
        )
        * (2.78 * 10**-6)
        * 1000
        * 3600
        * 24
    )

    return kv / 1000


def kv_cosby(sand, clay):
    """
    Determine saturated hydraulic conductivity kv [m/day].

    based on:
      Cosby, B.J., Hornberger, G.M., Clapp, R.B., Ginn, T.R., 1984.
      A statistical exploration of the relationship of soil moisture characteristics to
      the physical properties of soils. Water Resour. Res. 20(6) 682-690.

    Parameters
    ----------
    sand: float
        sand percentage [%].
    clay : float
        clay percentage [%].

    Returns
    -------
    kv : float
        saturated hydraulic conductivity [m/day].

    """
    kv = 60.96 * 10.0 ** (-0.6 + 0.0126 * sand - 0.0064 * clay) * 10.0

    return kv / 1000


def get_hydraulic_conductivity(ds, ptf_name="brakensiek"):
    """
    Determine vertical saturated hydraulic conductivity (KsatVer) per soil layer depth.

    Based on PTF.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil properties at each soil depth [sl1 - sl7].
    thetas: xarray.Dataset
        Dataset containing thetaS at each soil layer depth.
    ptf_name : str
        PTF to use for calculation KsatVer.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing KsatVer [mm/day] for each soil layer depth.
    """
    if ptf_name == "brakensiek":
        ds_out = xr.apply_ufunc(
            kv_brakensiek,
            ds["thetas"],
            ds["clay"],
            ds["sand"],
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )
    elif ptf_name == "cosby":
        ds_out = xr.apply_ufunc(
            kv_cosby,
            ds["clay"],
            ds["sand"],
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )

    ds_out.name = "kv"
    ds_out.raster.set_nodata(np.nan)

    return ds_out


def thetar_brakensiek(sand, clay, thetas):
    """
    Determine residual water content [m3/m3].

    Thetas is equal to porosity (Φ) in this case.

    Equation found in https://archive.org/details/watershedmanagem0000unse_d4j9/page/294/mode/1up (p. 294)

    Based on:
        Brakensiek, D.L., Rawls, W.J.,and Stephenson, G.R.: Modifying scs hydrologic
        soil groups and curve numbers for range land soils, ASAE Paper no. PNR-84-203,
        St. Joseph, Michigan, USA, 1984.

    Parameters
    ----------
    sand: float
        sand percentage [%].
    clay: float
        clay percentage [%].
    thetas : float
        saturated water content [m3/m3].

    Returns
    -------
    thetar : float
        residual water content [m3/m3].

    """
    return (
        -0.0182482
        + 0.00087269 * sand
        + 0.00513488 * clay
        + 0.02939286 * thetas
        - 0.00015395 * clay**2
        - 0.0010827 * sand * thetas
        - 0.00018233 * clay**2 * thetas**2
        + 0.00030703 * clay**2 * thetas
        - 0.0023584 * thetas**2 * clay
    )


def thetas_toth(soc, bdod, clay, silt, is_top_soil):
    """
    Determine saturated water content [m3/m3].

    Based on:
      Tóth, B., Weynants, M., Nemes, A., Makó, A., Bilas, G., and Tóth, G.:
      New generation of hydraulic pedotransfer functions for Europe, Eur. J.
      Soil Sci., 66, 226-238. doi: 10.1111/ejss.121921211, 2015.

    Parameters
    ----------
    bdod : float
        bulk density [g /cm3].
    sand: float
        sand percentage [%].
    silt: float
        silt percentage [%].
    is_top_soil: bool
        top soil flag.

    Returns
    -------
    thetas : float
        saturated water content [cm3/cm3].

    """

    # this is an alternative version of the thetas_toth function
    # which also requires the ph value as input
    # thetas = (
    #     0.5653
    #     - 0.07918 * bdod**2
    #     + 0.001671 * ph**2
    #     + 0.0005438 * clay
    #     + 0.001065 * silt
    #     + 0.06836 * is_top_soil
    #     - 0.00001382 * clay * ph**2
    #     - 0.00001270 * silt * clay
    #     - 0.0004784 * bdod**2 * ph**2
    #     - 0.0002836 * silt * bdod**2
    #     + 0.0004158 * clay * bdod**2
    #     - 0.01686 * is_top_soil * bdod**2
    #     - 0.0003541 * silt
    #     - 0.0003152 * is_top_soil * ph**2
    # )

    thetas = (
        0.6819
        - 0.06480 * (1 / (soc + 1))
        - 0.11900 * bdod**2
        - 0.02668 * is_top_soil
        + 0.001489 * clay
        + 0.0008031 * silt
        + 0.02321 * (1 / (soc + 1)) * bdod**2
        + 0.01908 * bdod**2 * is_top_soil
        - 0.0011090 * clay * is_top_soil
        - 0.00002315 * silt * clay
        - 0.0001197 * silt * bdod**2
        - 0.0001068 * clay * bdod**2
    )

    return thetas


def thetawp_toth(soc, clay, silt):
    """
    Determine water content at wilting point [m3/m3].

    Based on:
      Tóth, B., Weynants, M., Nemes, A., Makó, A., Bilas, G., and Tóth, G.:
      New generation of hydraulic pedotransfer functions for Europe, Eur. J. Soil Sci.,
      66, 226-238. doi: 10.1111/ejss.121921211, 2015.

    Parameters
    ----------
    soc : float
        organic carbon [%].
    clay: float
        sand percentage [%].
    silt: float
        silt percentage [%].

    Returns
    -------
    thetawp : float
        residual water content [m3/m3].

    """
    thetawp = (
        0.09878
        + 0.002127 * clay
        - 0.0008366 * silt
        - 0.07670 / (soc + 1)
        + 0.00003853 * silt * clay
        + (0.002330 * clay) / (soc + 1)
        + (0.0009498 * silt) / (soc + 1)
    )

    return thetawp


def thetafc_toth(soc, clay, silt):
    """
    Determine field capacity water content [m3/m3].

    Based on:
      Tóth, B., Weynants, M., Nemes, A., Makó, A., Bilas, G., and Tóth, G.:
      New generation of hydraulic pedotransfer functions for Europe, Eur. J. Soil Sci.,
      66, 226-238. doi: 10.1111/ejss.121921211, 2015.

    Parameters
    ----------
    soc : float
        organic carbon [%].
    clay: float
        sand percentage [%].
    silt: float
        silt percentage [%].

    Returns
    -------
    thetafc : float
        field capacity water content [m3/m3].

    """
    thetafc = (
        0.2449
        - 0.1887 * (1 / (soc + 1))
        + 0.004527 * clay
        + 0.001535 * silt
        + 0.001442 * silt * (1 / (soc + 1))
        - 0.00005110 * silt * clay
        + 0.0008676 * clay * (1 / (soc + 1))
    )

    return thetafc


def get_bubble_pressure(clay, sand, thetas):
    bubbling_pressure = np.exp(
        5.3396738
        + 0.1845038 * clay
        - 2.48394546 * thetas
        - 0.00213853 * clay**2
        - 0.0435649 * sand * thetas
        - 0.61745089 * clay * thetas
        - 0.00001282 * sand**2 * clay
        + 0.00895359 * clay**2 * thetas
        - 0.00072472 * sand**2 * thetas
        + 0.0000054 * sand * clay**2
        + 0.00143598 * sand**2 * thetas**2
        - 0.00855375 * clay**2 * thetas**2
        + 0.50028060 * thetas**2 * clay
    )
    return bubbling_pressure


def interpolate_soil_layers(ds):
    assert ds.ndim == 3
    for layer in range(ds.shape[0]):
        ds[layer] = ds[layer].raster.interpolate_na("nearest")
    return ds


def load_soilgrids(data_catalog, grid, region):
    variables = ["bdod", "clay", "silt", "sand", "soc"]
    layers = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

    ds = []
    for variable in variables:
        variable_layers = []
        for i, layer in enumerate(layers, start=1):
            da = data_catalog.get_rasterdataset(
                f"soilgrids_2020_{variable}_{layer}", geom=region
            ).compute()
            da.name = f"{variable}_{i}"
            variable_layers.append(da)
        ds_variable = xr.concat(variable_layers, dim="soil_layers", compat="equals")
        ds_variable.name = variable
        ds.append(ds_variable)
    ds = xr.merge(ds, join="exact")
    ds = ds.raster.mask_nodata()  # set all nodata values to nan

    total = ds["sand"] + ds["clay"] + ds["silt"]
    assert total.min() >= 99.8
    assert total.max() <= 100.2

    # the top 30 cm is considered as top soil (https://www.fao.org/uploads/media/Harm-World-Soil-DBv7cv_1.pdf)
    is_top_soil = np.zeros_like(ds["sand"], dtype=bool)
    is_top_soil[0:3] = True
    is_top_soil = xr.DataArray(
        is_top_soil, dims=ds["sand"].dims, coords=ds["sand"].coords
    )
    ds["is_top_soil"] = is_top_soil

    depth_to_bedrock = data_catalog.get_rasterdataset(
        "soilgrids_2017_BDTICM", geom=region
    )
    depth_to_bedrock = depth_to_bedrock.raster.mask_nodata()
    depth_to_bedrock = depth_to_bedrock.raster.reproject_like(
        grid, method="bilinear"
    ).raster.interpolate_na("nearest")

    ds["thetas"] = xr.apply_ufunc(
        thetas_toth,
        ds["soc"],
        ds["bdod"],
        ds["clay"],
        ds["silt"],
        ds["is_top_soil"],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )
    thetas = ds["thetas"].raster.reproject_like(grid, method="average")
    thetas = interpolate_soil_layers(thetas)

    assert thetas.notnull().all()
    assert (thetas >= 0).all()
    assert (thetas <= 1).all()

    # thetafc = xr.apply_ufunc(
    #     thetafc_toth,
    #     ds["soc"],
    #     ds["clay"],
    #     ds["silt"],
    #     dask="parallelized",
    #     output_dtypes=[float],
    #     keep_attrs=True,
    # )
    # thetafc = thetafc.raster.reproject_like(grid, method="average")
    # thetafc = interpolate_soil_layers(thetafc)

    # assert thetafc.notnull().all()
    # assert (thetafc >= 0).all()
    # assert (thetafc < thetas).all()

    # ds["thetawp"] = xr.apply_ufunc(
    #     thetawp_toth,
    #     ds["soc"],
    #     ds["clay"],
    #     ds["silt"],
    #     dask="parallelized",
    #     output_dtypes=[float],
    #     keep_attrs=True,
    # )
    # thetawp = ds["thetawp"].raster.reproject_like(grid, method="average")
    # thetawp = interpolate_soil_layers(thetawp)

    # assert thetawp.notnull().all()
    # assert (thetawp >= 0).all()
    # assert (thetawp < thetafc).all()

    ds["thetar"] = xr.apply_ufunc(
        thetar_brakensiek,
        ds["sand"],
        ds["clay"],
        ds["thetas"],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )
    thetar = ds["thetar"].raster.reproject_like(grid, method="average")
    thetar = interpolate_soil_layers(thetar)

    assert thetar.notnull().all()
    assert (thetar >= 0).all()
    assert (thetar < thetas).all()

    ds["bubbling_pressure_cm"] = get_bubble_pressure(
        ds["clay"], ds["sand"], ds["thetas"]
    )
    bubbling_pressure_cm = ds["bubbling_pressure_cm"].raster.reproject_like(
        grid, method="average"
    )
    bubbling_pressure_cm = interpolate_soil_layers(bubbling_pressure_cm)

    assert (bubbling_pressure_cm > 0).all()
    assert (bubbling_pressure_cm < 1000).all()

    hydraulic_conductivity = get_hydraulic_conductivity(ds, "brakensiek")
    # Hydraulic conductivity (K) values can vary over several orders of magnitude within
    # a relatively small spatial area. This wide range of values can lead to issues
    # during interpolation (e.g., reprojecting), as standard interpolation methods might be
    # overly influenced by high values. Moreover, hydraulic conductivity values are
    # typically log-normally distributed. To address these issues, we apply a log
    # before reprojecting and then exponentiate the result.
    hydraulic_conductivity_log = np.log(hydraulic_conductivity)
    hydraulic_conductivity_log = hydraulic_conductivity_log.raster.reproject_like(
        grid, method="average"
    )
    hydraulic_conductivity = np.exp(hydraulic_conductivity_log)
    hydraulic_conductivity = interpolate_soil_layers(hydraulic_conductivity)

    assert hydraulic_conductivity.min() >= 1e-7
    assert hydraulic_conductivity.max() <= 10

    # same for pore_size_index lambda
    lambda_ = get_pore_size_index(ds)
    lambda_log = np.log(lambda_)
    lambda_log = lambda_log.raster.reproject_like(grid, method="average")
    lambda_ = np.exp(lambda_log)
    lambda_ = interpolate_soil_layers(lambda_)

    assert lambda_.min() >= 0.01
    assert lambda_.max() <= 1

    soil_layer_height = xr.DataArray(
        np.zeros(hydraulic_conductivity.shape, dtype=np.float32),
        dims=hydraulic_conductivity.dims,
        coords=hydraulic_conductivity.coords,
    )
    for layer, height in enumerate((0.05, 0.10, 0.15, 0.30, 0.40, 1.00)):
        soil_layer_height[layer] = height

    assert (soil_layer_height.sum(axis=0) == 2.0).all()

    return (
        hydraulic_conductivity,
        bubbling_pressure_cm,
        lambda_,
        thetas,
        thetar,
        soil_layer_height,
    )
