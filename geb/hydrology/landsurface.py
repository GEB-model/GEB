"""Land surface module for GEB."""

import numpy as np
import numpy.typing as npt
import zarr
from numba import njit, prange  # noqa: F401

from geb.module import Module
from geb.workflows import balance_check

from .evaporation import (
    get_CO2_induced_crop_factor_adustment,
    get_crop_factors_and_root_depths,
    get_potential_bare_soil_evaporation,
    get_potential_evapotranspiration,
    get_potential_transpiration,
)
from .interception import get_interception_capacity, interception
from .potential_evapotranspiration import get_reference_evapotranspiration
from .snow_glaciers import snow_model
from .soil_scalar import (
    add_water_to_topwater_and_evaporate_open_water,
    get_flux,
    get_mean_unsaturated_hydraulic_conductivity,
    get_soil_water_flow_parameters,
    infiltration,
    rise_from_groundwater,
)


# @njit(parallel=True, cache=True)
def land_surface_model(
    land_use_type: npt.NDArray[np.int32],
    w: npt.NDArray[np.float32],  # TODO: Check if fortran order speeds up
    wres: npt.NDArray[np.float32],  # TODO: Check if fortran order speeds up
    ws: npt.NDArray[np.float32],  # TODO: Check if fortran order speeds up
    delta_z: npt.NDArray[np.float32],  # TODO: Check if fortran order speeds up
    topwater_m: npt.NDArray[np.float32],
    snow_water_equivalent_m: npt.NDArray[np.float32],
    liquid_water_in_snow_m: npt.NDArray[np.float32],
    snow_temperature_C: npt.NDArray[np.float32],
    interception_storage_m: npt.NDArray[np.float32],
    interception_capacity_m: npt.NDArray[np.float32],
    pr_kg_per_m2_per_s: npt.NDArray[np.float32],
    tas_2m_K: npt.NDArray[np.float32],
    dewpoint_tas_2m_K: npt.NDArray[np.float32],
    ps_pascal: npt.NDArray[np.float32],
    rlds_W_per_m2: npt.NDArray[np.float32],
    rsds_W_per_m2: npt.NDArray[np.float32],
    wind_u10m_m_per_s: npt.NDArray[np.float32],
    wind_v10m_m_per_s: npt.NDArray[np.float32],
    CO2_ppm: np.float32,
    crop_factor: npt.NDArray[np.float32],
    actual_irrigation_consumption_m: npt.NDArray[np.float32],
    capillar_rise_m: npt.NDArray[np.float32],
    saturated_hydraulic_conductivity_m_per_s: npt.NDArray[np.float32],
    lambda_pore_size_distribution: npt.NDArray[np.float32],
    bubbing_pressure_cm: npt.NDArray[np.float32],
    frost_index: npt.NDArray[np.float32],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """The main land surface model of GEB.

    Args:
        land_use_type: Land use type of the hydrological response unit.
        w: Current soil moisture content [m3/m3].
        wres: Soil moisture content at residual [m3/m3].
        ws: Soil moisture content at saturation [m3/m3].
        delta_z: Thickness of soil layers [m].
        topwater_m: Topwater in meters, which is >=0 for paddy and 0 for non-paddy. Within
            this function topwater is used to add water from natural infiltration and
            irrigation and to calculate open water evaporation.
        snow_water_equivalent_m: Snow water equivalent in meters.
        liquid_water_in_snow_m: Liquid water in snow in meters.
        snow_temperature_C: Snow temperature in Celsius.
        interception_storage_m: Interception storage in meters.
        interception_capacity_m: Interception capacity in meters.
        pr_kg_per_m2_per_s: Precipitation rate in kg/m^2/s.
        tas_2m_K: 2m air temperature in Kelvin.
        dewpoint_tas_2m_K: 2m dewpoint temperature in Kelvin.
        ps_pascal: Surface pressure in Pascals.
        rlds_W_per_m2: Downward longwave radiation in W/m^2.
        rsds_W_per_m2: Downward shortwave radiation in W/m^2.
        wind_u10m_m_per_s: U component of 10m wind speed in m/s.
        wind_v10m_m_per_s: V component of 10m wind speed in m/s.
        CO2_ppm: Atmospheric CO2 concentration in ppm.
        crop_factor: Crop factor for each HRU. Dimensionless.
        actual_irrigation_consumption_m: Actual irrigation consumption in meters.
        capillar_rise_m: Capillary rise in meters.
        saturated_hydraulic_conductivity_m_per_s: Saturated hydraulic conductivity in m/s.
        lambda_pore_size_distribution: Van Genuchten pore size distribution parameter.
        bubbing_pressure_cm: Bubbling pressure in cm.
        frost_index: Frost index. TODO: Add unit and description.

    Returns:
        Tuple of:
        - reference_evapotranspiration_grass_m: Reference evapotranspiration for
          grass in meters.
        - reference_evapotranspiration_water_m: Reference evapotranspiration for
          water in meters.
        - snow_water_equivalent_m: Updated snow water equivalent in meters.
        - liquid_water_in_snow_m: Updated liquid water in snow in meters.
        - sublimation_m: Sublimation in meters.
        - snow_temperature_C: Updated snow temperature in Celsius.
        - interception_storage_m: Updated interception storage in meters.
        - interception_evaporation_m: Evaporation from interception storage in meters.
    """
    N_SOIL_LAYERS = 6

    CO2_induced_crop_factor_adustment = get_CO2_induced_crop_factor_adustment(CO2_ppm)

    # convert values to substep (i.e., per hour)
    actual_irrigation_consumption_m = actual_irrigation_consumption_m / 24.0
    capillar_rise_m = capillar_rise_m / 24.0
    saturated_hydraulic_conductivity_m_per_hour = (
        saturated_hydraulic_conductivity_m_per_s
    ) * 3600.0

    reference_evapotranspiration_grass_m = np.zeros_like(pr_kg_per_m2_per_s)
    reference_evapotranspiration_water_m = np.zeros_like(pr_kg_per_m2_per_s)

    # total per day variables for water balance
    sublimation_m = np.zeros_like(snow_water_equivalent_m)
    interception_evaporation_m = np.zeros_like(snow_water_equivalent_m)
    open_water_evaporation_m = np.zeros_like(snow_water_equivalent_m)
    runoff = np.zeros_like(snow_water_equivalent_m)
    groundwater_recharge_m = np.zeros_like(snow_water_equivalent_m)
    interflow_m = np.zeros_like(snow_water_equivalent_m)

    for i in prange(snow_water_equivalent_m.size):
        pr_kg_per_m2_per_s_cell = pr_kg_per_m2_per_s[:, i]
        tas_2m_K_cell = tas_2m_K[:, i]
        dewpoint_tas_2m_K_cell = dewpoint_tas_2m_K[:, i]
        ps_pascal_cell = ps_pascal[:, i]
        rlds_W_per_m2_cell = rlds_W_per_m2[:, i]
        rsds_W_per_m2_cell = rsds_W_per_m2[:, i]
        wind_u10m_m_per_s_cell = wind_u10m_m_per_s[:, i]
        wind_v10m_m_per_s_cell = wind_v10m_m_per_s[:, i]

        snow_water_equivalent_m_cell = snow_water_equivalent_m[i]
        liquid_water_in_snow_m_cell = liquid_water_in_snow_m[i]
        snow_temperature_C_cell = snow_temperature_C[i]

        for hour in range(24):
            tas_C = tas_2m_K_cell[hour] - np.float32(273.15)
            dewpoint_tas_C = tas_2m_K_cell[hour] - np.float32(273.15)

            wind_10m_m_per_s: np.float32 = np.sqrt(
                wind_u10m_m_per_s_cell[hour] ** 2 + wind_v10m_m_per_s_cell[hour] ** 2
            )  # Wind speed at 10m height

            (
                reference_evapotranspiration_grass_m[hour, i],
                reference_evapotranspiration_water_m[hour, i],
                net_absorbed_radiation_vegetation_MJ_m2_dt,
                actual_vapour_pressure_Pa,
            ) = get_reference_evapotranspiration(
                tas_C=tas_C,
                dewpoint_tas_C=dewpoint_tas_C,
                ps_pa=ps_pascal_cell[hour],
                rlds_W_per_m2=rlds_W_per_m2_cell[hour],
                rsds_W_per_m2=rsds_W_per_m2_cell[hour],
                wind_10m_m_per_s=wind_10m_m_per_s,
            )

            (
                snow_water_equivalent_m_cell,
                liquid_water_in_snow_m_cell,
                snow_temperature_C_cell,
                _,  # melt (before refreezing)
                runoff_from_melt_m,  # after refreezing
                rainfall_m,
                sublimation_m_cell_hour,
                _,  # refreezing
                _,  # snow surface temperature
                _,  # net shortwave radiation
                _,  # net longwave radiation
                _,  # sensible heat flux
                _,  # latent heat flux
            ) = snow_model(
                pr_kg_per_m2_per_s=pr_kg_per_m2_per_s_cell[hour],
                air_temperature_C=tas_C,
                snow_water_equivalent_m=snow_water_equivalent_m_cell,
                liquid_water_in_snow_m=liquid_water_in_snow_m_cell,
                snow_temperature_C=snow_temperature_C_cell,
                shortwave_radiation_W_per_m2=rsds_W_per_m2_cell[hour],
                downward_longwave_radiation_W_per_m2=rlds_W_per_m2_cell[hour],
                vapor_pressure_air_Pa=actual_vapour_pressure_Pa,
                air_pressure_Pa=ps_pascal_cell[hour],
                wind_10m_m_per_s=wind_10m_m_per_s,
            )

            sublimation_m[i] += sublimation_m_cell_hour

            potential_bare_soil_evaporation_m: np.float32 = (
                get_potential_bare_soil_evaporation(
                    reference_evapotranspiration_grass_m[hour, i],
                    sublimation_m_cell_hour,
                )
            )

            potential_evapotranspiration_m: np.float32 = get_potential_evapotranspiration(
                reference_evapotranspiration_grass_m=reference_evapotranspiration_grass_m[
                    hour, i
                ],
                crop_factor=crop_factor[i],
                CO2_induced_crop_factor_adustment=CO2_induced_crop_factor_adustment,
            )

            potential_transpiration_m: np.float32 = get_potential_transpiration(
                potential_evapotranspiration_m=potential_evapotranspiration_m,
                potential_bare_soil_evaporation_m=potential_bare_soil_evaporation_m,
            )
            (
                interception_storage_m[i],
                throughfall_m,
                interception_evaporation_m_cell_hour,
            ) = interception(
                rainfall_m=rainfall_m,
                storage_m=interception_storage_m[i],
                capacity_m=interception_capacity_m[i],
                potential_transpiration_m=potential_transpiration_m,
            )

            interception_evaporation_m[i] += interception_evaporation_m_cell_hour

            natural_available_water_infiltration_m: np.float32 = (
                throughfall_m + runoff_from_melt_m
            )

            # TODO: Test if removing if-statements in the function speeds up the code
            topwater_m[i], open_water_evaporation_m_cell_hour = (
                add_water_to_topwater_and_evaporate_open_water(
                    natural_available_water_infiltration_m=natural_available_water_infiltration_m,
                    actual_irrigation_consumption_m=actual_irrigation_consumption_m[i],
                    land_use_type=land_use_type[i],
                    reference_evapotranspiration_water_m_per_day=reference_evapotranspiration_water_m[
                        hour, i
                    ],
                    topwater_m=topwater_m[i],
                )
            )
            open_water_evaporation_m[i] += open_water_evaporation_m_cell_hour

            runoff[i] += rise_from_groundwater(
                w=w[:, i],
                ws=ws[:, i],
                capillary_rise_from_groundwater=capillar_rise_m[i],
            )

            (
                topwater_m[i],
                direct_runoff_m,
                groundwater_recharge_from_infiltraton_m,
                infiltration_amount,
            ) = infiltration(
                ws=ws[:, i],
                saturated_hydraulic_conductivity=saturated_hydraulic_conductivity_m_per_hour[
                    :, i
                ],
                land_use_type=land_use_type[i],
                frost_index=frost_index[i],
                w=w[:, i],
                topwater_m=topwater_m[i],
            )
            runoff[i] += direct_runoff_m
            groundwater_recharge_m[i] += groundwater_recharge_from_infiltraton_m

            bottom_layer = N_SOIL_LAYERS - 1

            psi, unsaturated_hydraulic_conductivity_m_per_hour = (
                get_soil_water_flow_parameters(
                    w=w[bottom_layer, i],
                    wres=wres[bottom_layer, i],
                    ws=ws[bottom_layer, i],
                    lambda_pore_size_distribution=lambda_pore_size_distribution[
                        bottom_layer, i
                    ],
                    saturated_hydraulic_conductivity=saturated_hydraulic_conductivity_m_per_hour[
                        bottom_layer, i
                    ],
                    bubbling_pressure_cm=bubbing_pressure_cm[bottom_layer, i],
                )
            )

            # We assume that the bottom layer is draining under gravity
            # i.e., assuming homogeneous soil water potential below
            # bottom layer all the way to groundwater
            # Assume draining under gravity. If there is capillary rise from groundwater, there will be no
            # percolation to the groundwater. A potential capillary rise from
            # the groundwater is already accounted for in rise_from_groundwater
            flux = unsaturated_hydraulic_conductivity_m_per_hour * (
                capillar_rise_m[i] <= np.float32(0)
            )
            available_water_source = w[bottom_layer, i] - wres[bottom_layer, i]
            flux = min(flux, available_water_source)
            w[bottom_layer, i] -= flux
            w[bottom_layer, i] = max(w[bottom_layer, i], wres[bottom_layer, i])
            groundwater_recharge_m[i] += flux

            psi_layer_below = psi
            unsaturated_hydraulic_conductivity_layer_below = (
                unsaturated_hydraulic_conductivity_m_per_hour
            )

            # iterate from bottom to top layer (ignoring the bottom layer which is treated above)
            for layer in range(N_SOIL_LAYERS - 2, -1, -1):
                psi, unsaturated_hydraulic_conductivity_m_per_hour = (
                    get_soil_water_flow_parameters(
                        w=w[layer, i],
                        wres=wres[layer, i],
                        ws=ws[layer, i],
                        lambda_pore_size_distribution=lambda_pore_size_distribution[
                            layer, i
                        ],
                        saturated_hydraulic_conductivity=saturated_hydraulic_conductivity_m_per_hour[
                            layer, i
                        ],
                        bubbling_pressure_cm=bubbing_pressure_cm[layer, i],
                    )
                )

                # Compute the mean of the conductivities
                mean_unsaturated_hydraulic_conductivity: np.float32 = (
                    get_mean_unsaturated_hydraulic_conductivity(
                        unsaturated_hydraulic_conductivity_m_per_hour,
                        unsaturated_hydraulic_conductivity_layer_below,
                    )
                )

                # Compute flux using Darcy's law. The -1 accounts for gravity.
                # Positive flux is downwards; see minus sign in the equation, which negates
                # the -1 of gravity and other terms.
                flux: np.float32 = get_flux(
                    mean_unsaturated_hydraulic_conductivity,
                    psi_layer_below,
                    psi,
                    delta_z[layer, i],
                )

                # Determine the positive flux and source/sink layers without if statements
                positive_flux = abs(flux)
                flux_direction = flux >= 0  # 1 if flux >= 0, 0 if flux < 0
                source = layer + (
                    1 - flux_direction
                )  # layer if flux >= 0, layer + 1 if flux < 0
                sink = (
                    layer + flux_direction
                )  # layer + 1 if flux >= 0, layer if flux < 0

                # Limit flux by available water in source and storage capacity of sink
                remaining_storage_capacity_sink = ws[sink, i] - w[sink, i]
                available_water_source = w[source, i] - wres[source, i]

                positive_flux = min(
                    positive_flux,
                    remaining_storage_capacity_sink,
                    available_water_source,
                )

                # Update water content in source and sink layers
                w[source, i] -= positive_flux
                w[sink, i] += positive_flux

                # Ensure water content stays within physical bounds
                w[sink, i] = min(w[sink, i], ws[sink, i])
                w[source, i] = max(w[source, i], wres[source, i])

                psi_layer_below = psi
                unsaturated_hydraulic_conductivity_layer_below = (
                    unsaturated_hydraulic_conductivity_m_per_hour
                )

        snow_water_equivalent_m[i] = snow_water_equivalent_m_cell
        liquid_water_in_snow_m[i] = liquid_water_in_snow_m_cell
        snow_temperature_C[i] = snow_temperature_C_cell

    # TODO: Also solve vertical soil water balance for non-bio land use types
    # some of the above calculations for non-bio land use types will lead to NaNs
    # but these can be safely converted to zeros
    groundwater_recharge_m = np.nan_to_num(groundwater_recharge_m)

    return (
        topwater_m,
        reference_evapotranspiration_grass_m,
        reference_evapotranspiration_water_m,
        snow_water_equivalent_m,
        liquid_water_in_snow_m,
        sublimation_m,
        snow_temperature_C,
        interception_storage_m,
        interception_evaporation_m,
        open_water_evaporation_m,
        runoff,
        groundwater_recharge_m,
        interflow_m,
    )


class LandSurface(Module):
    """Land surface module for GEB."""

    def __init__(self, model: "GEBModel", hydrology: "Hydrology") -> None:
        """Initialize the potential evaporation module.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology module.
        """
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Name of the module."""
        return "hydrology.landsurface"

    def spinup(self) -> None:
        """Spinup function for the land surface module."""
        self.HRU.var.topwater = self.HRU.full_compressed(0.0, dtype=np.float32)

        self.HRU.var.snow_water_equivalent_m = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )
        self.HRU.var.liquid_water_in_snow_m = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )
        self.HRU.var.snow_temperature_C = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )
        self.HRU.var.interception_storage_m = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )

        store = zarr.storage.LocalStore(
            self.model.files["grid"]["landcover/forest/interception_capacity"],
            read_only=True,
        )

        self.grid.var.interception_capacity_forest = self.grid.compress(
            zarr.open_group(store, mode="r")["interception_capacity"][:]
        )

        store = zarr.storage.LocalStore(
            self.model.files["grid"]["landcover/grassland/interception_capacity"],
            read_only=True,
        )

        self.grid.var.interception_capacity_grassland = self.grid.compress(
            zarr.open_group(store, mode="r")["interception_capacity"][:]
        )

        store = zarr.storage.LocalStore(
            self.model.files["grid"]["landcover/forest/crop_coefficient"],
            read_only=True,
        )
        self.grid.var.forest_crop_factor_per_10_days = zarr.open_group(store, mode="r")[
            "crop_coefficient"
        ][:]

    def step(
        self,
    ) -> tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
    ]:
        """Step function for the land surface module.

        Currently, this function calculates the reference evapotranspiration
        for grass and water surfaces using meteorological data.

        Returns:
            A tuple containing:
            - snow_melt_m: Snow melt in meters.
            - rain_m: Rainfall in meters.
            - sublimation_m: Sublimation in meters.

        """
        if __debug__:
            snow_water_equivalent_prev = self.HRU.var.snow_water_equivalent_m.copy()
            liquid_water_in_snow_prev = self.HRU.var.liquid_water_in_snow_m.copy()
            interception_storage_prev = self.HRU.var.interception_storage_m.copy()
            topwater_prev = self.HRU.var.topwater.copy()
            w_prev = np.nansum(self.HRU.var.w, axis=0).copy()

        forest_crop_factor = self.hydrology.to_HRU(
            data=self.grid.compress(
                self.grid.var.forest_crop_factor_per_10_days[
                    (self.model.current_day_of_year - 1) // 10
                ]
            ),
            fn=None,
        )

        crop_stage_lenghts = np.column_stack(
            [
                self.model.agents.crop_farmers.var.crop_data["l_ini"],
                self.model.agents.crop_farmers.var.crop_data["l_dev"],
                self.model.agents.crop_farmers.var.crop_data["l_mid"],
                self.model.agents.crop_farmers.var.crop_data["l_late"],
            ]
        )

        get_crop_sub_stage = self.model.agents.crop_farmers.var.crop_data_type == "GAEZ"
        if get_crop_sub_stage:
            crop_sub_stage_lengths = np.column_stack(
                [
                    self.model.agents.crop_farmers.var.crop_data["d1"],
                    self.model.agents.crop_farmers.var.crop_data["d2a"],
                    self.model.agents.crop_farmers.var.crop_data["d2b"],
                    self.model.agents.crop_farmers.var.crop_data["d3a"],
                    self.model.agents.crop_farmers.var.crop_data["d3b"],
                    self.model.agents.crop_farmers.var.crop_data["d4"],
                ]
            )
        else:
            crop_sub_stage_lengths = np.full(
                (self.model.agents.crop_farmers.var.crop_data.shape[0], 6),
                np.nan,
                dtype=np.float32,
            )

        crop_factor_per_crop_stage = np.column_stack(
            [
                self.model.agents.crop_farmers.var.crop_data["kc_initial"],
                self.model.agents.crop_farmers.var.crop_data["kc_mid"],
                self.model.agents.crop_farmers.var.crop_data["kc_end"],
            ]
        )

        crop_root_depths = np.column_stack(
            [
                self.model.agents.crop_farmers.var.crop_data["rd_rain"],
                self.model.agents.crop_farmers.var.crop_data["rd_irr"],
            ]
        )

        crop_factor, root_depth_m, crop_sub_stage = get_crop_factors_and_root_depths(
            land_use_map=self.HRU.var.land_use_type,
            crop_factor_forest_map=forest_crop_factor,
            crop_map=self.HRU.var.crop_map,
            crop_age_days_map=self.HRU.var.crop_age_days_map,
            crop_harvest_age_days=self.HRU.var.crop_harvest_age_days,
            crop_stage_lengths=crop_stage_lenghts,
            crop_sub_stage_lengths=crop_sub_stage_lengths,
            crop_factor_per_crop_stage=crop_factor_per_crop_stage,
            crop_root_depths=crop_root_depths,
            get_crop_sub_stage=get_crop_sub_stage,
        )

        interception_capacity_m_forest_HRU = self.hydrology.to_HRU(
            data=self.grid.var.interception_capacity_forest[
                (self.model.current_day_of_year - 1) // 10
            ],
            fn=None,
        )
        interception_capacity_m_grassland_HRU = self.hydrology.to_HRU(
            data=self.grid.var.interception_capacity_grassland[
                (self.model.current_day_of_year - 1) // 10
            ],
            fn=None,
        )

        interception_capacity_m: npt.NDArray[np.float32] = get_interception_capacity(
            land_use_type=self.HRU.var.land_use_type,
            interception_capacity_m_forest_HRU=interception_capacity_m_forest_HRU,
            interception_capacity_m_grassland_HRU=interception_capacity_m_grassland_HRU,
        )

        pr_kg_per_m2_per_s = self.HRU.pr_kg_per_m2_per_s

        (
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow,  # from all sources
            irrigation_loss_to_evaporation_m,
            total_water_demand_loss_m3,
            actual_irrigation_consumption_m,
        ) = self.hydrology.water_demand.step(root_depth_m)

        # Obtain capillary rise for the HRUs
        capillar_rise_m = self.hydrology.to_HRU(data=self.grid.var.capillar, fn=None)
        if capillar_rise_m.sum() > 0.0:
            raise NotImplementedError(
                "Capillary rise is not implemented in the land surface model yet."
            )

        self.HRU.var.frost_index = np.full_like(self.HRU.var.topwater, np.float32(0.0))
        print("warning: setting frost index to zero")

        # TODO: pre-compute this
        delta_z = (
            self.HRU.var.soil_layer_height[:-1, :]
            + self.HRU.var.soil_layer_height[1:, :]
        ) / 2

        (
            self.HRU.var.topwater,
            reference_evapotranspiration_grass_m_dt,
            reference_evapotranspiration_water_m_dt,
            self.HRU.var.snow_water_equivalent_m,
            self.HRU.var.liquid_water_in_snow_m,
            sublimation_m,
            self.HRU.var.snow_temperature_C,
            self.HRU.var.interception_storage_m,
            interception_evaporation_m,
            open_water_evaporation_m,
            runoff_m,
            groundwater_recharge_m,
            interflow_m,
        ) = land_surface_model(
            w=self.HRU.var.w,
            wres=self.HRU.var.wres,
            ws=self.HRU.var.ws,
            delta_z=delta_z,
            land_use_type=self.HRU.var.land_use_type,
            topwater_m=self.HRU.var.topwater,
            snow_water_equivalent_m=self.HRU.var.snow_water_equivalent_m,
            liquid_water_in_snow_m=self.HRU.var.liquid_water_in_snow_m,
            snow_temperature_C=self.HRU.var.snow_temperature_C,
            interception_storage_m=self.HRU.var.interception_storage_m,
            interception_capacity_m=interception_capacity_m,
            pr_kg_per_m2_per_s=np.asfortranarray(
                pr_kg_per_m2_per_s
            ),  # Due to the access pattern in numba (iterate over hours), the fortran order is much faster in this case
            tas_2m_K=np.asfortranarray(
                self.HRU.tas_2m_K
            ),  # Due to the access pattern in numba (iterate over hours), the fortran order is much faster in this case
            dewpoint_tas_2m_K=np.asfortranarray(
                self.HRU.dewpoint_tas_2m_K
            ),  # Due to the access pattern in numba (iterate over hours), the fortran order is much faster in this case
            ps_pascal=np.asfortranarray(
                self.HRU.ps_pascal
            ),  # Due to the access pattern in numba (iterate over hours), the fortran order is much faster in this case
            rlds_W_per_m2=np.asfortranarray(
                self.HRU.rlds_W_per_m2
            ),  # Due to the access pattern in numba (iterate over hours), the fortran order is much faster in this case
            rsds_W_per_m2=np.asfortranarray(
                self.HRU.rsds_W_per_m2
            ),  # Due to the access pattern in numba (iterate over hours), the fortran order is much faster in this case
            wind_u10m_m_per_s=np.asfortranarray(
                self.HRU.wind_u10m_m_per_s
            ),  # Due to the access pattern in numba (iterate over hours), the fortran order is much faster in this case
            wind_v10m_m_per_s=np.asfortranarray(
                self.HRU.wind_v10m_m_per_s
            ),  # Due to the access pattern in numba (iterate over hours), the fortran order is much faster in this case
            CO2_ppm=self.model.forcing.load("CO2_ppm"),
            crop_factor=crop_factor,
            actual_irrigation_consumption_m=actual_irrigation_consumption_m,
            capillar_rise_m=capillar_rise_m,
            saturated_hydraulic_conductivity_m_per_s=self.HRU.var.saturated_hydraulic_conductivity
            / (24 * 3600),
            lambda_pore_size_distribution=self.HRU.var.lambda_pore_size_distribution,
            bubbing_pressure_cm=self.HRU.var.bubbling_pressure_cm,
            frost_index=self.HRU.var.frost_index,
        )

        assert interflow_m.sum() == 0.0, "Interflow is not implemented yet."

        assert balance_check(
            name="land surface 1",
            how="cellwise",
            influxes=[
                pr_kg_per_m2_per_s.sum(axis=0)
                * 3.6,  # from kg/m2/s to m/hr and sum over hours to make it day (but then re-arranged for efficiency)
                actual_irrigation_consumption_m,
                capillar_rise_m,
            ],
            outfluxes=[
                -sublimation_m,
                interception_evaporation_m,
                open_water_evaporation_m,
                runoff_m,
                groundwater_recharge_m,
            ],
            prestorages=[
                snow_water_equivalent_prev,
                liquid_water_in_snow_prev,
                interception_storage_prev,
                topwater_prev,
                w_prev,
            ],
            poststorages=[
                self.HRU.var.snow_water_equivalent_m,
                self.HRU.var.liquid_water_in_snow_m,
                self.HRU.var.interception_storage_m,
                self.HRU.var.topwater,
                np.nansum(self.HRU.var.w, axis=0),
            ],
            tolerance=1e-6,
        )

        self.HRU.var.reference_evapotranspiration_water_m_per_day = np.sum(
            reference_evapotranspiration_water_m_dt, axis=0
        )
        self.HRU.var.reference_evapotranspiration_grass_m_per_day = np.sum(
            reference_evapotranspiration_grass_m_dt, axis=0
        )

        self.grid.var.reference_evapotranspiration_water_m_per_day = (
            self.hydrology.to_grid(
                HRU_data=self.HRU.var.reference_evapotranspiration_water_m_per_day,
                fn="weightedmean",
            )
        )

        self.model.agents.crop_farmers.save_water_deficit(
            self.HRU.var.reference_evapotranspiration_grass_m_per_day
        )
        self.report(locals())

        return snow_melt_m, throughfall_m, sublimation_m
