"""Tests for the land surface model in GEB."""

import numpy as np
import pytest

from geb.hydrology import landsurface
from geb.hydrology.landsurface import land_surface_model
from geb.workflows import balance_check


@pytest.mark.parametrize(
    "asfloat64,tolerance",
    [
        (False, 1e-6),  # float32 inputs, looser tolerance
        (True, 1e-8),  # float64 inputs, tighter tolerance
    ],
    ids=["float32", "float64"],
)
def test_land_surface_model_with_error_case(asfloat64: bool, tolerance: float) -> None:
    """Test the land surface model with a previously failing mass balance case.

    Parameterized to run with float32 (default) and float64 inputs. When asfloat64=True,
    all floating inputs are cast to float64 before calling the model; a tighter tolerance
    is used for the balance check.

    To extract the input data, run the extract_landsurface_data.py script with the path
    to the landsurface_model_error.npz file and the desired cell index.
    """
    # Set the global N_SOIL_LAYERS variable required by the numba function
    landsurface.N_SOIL_LAYERS: int = 6

    land_use_type_data = np.int32(1)
    root_depth_m_data = np.float32(0.10000000149011612)
    topwater_m_data = np.float32(0.0)
    snow_water_equivalent_m_data = np.float32(2.357403516769409)
    liquid_water_in_snow_m_data = np.float32(0.0)
    snow_temperature_C_data = np.float32(-14.251362800598145)
    interception_storage_m_data = np.float32(0.0)
    interception_capacity_m_data = np.float32(0.00016463636711705476)
    crop_factor_data = np.float32(1.0)
    crop_map_data = np.int32(-1)
    actual_irrigation_consumption_m_data = np.float32(0.0)
    capillar_rise_m_data = np.float32(-0.0)
    natural_crop_groups_data = np.float32(3.0)
    w_data = np.array(
        [
            0.0051544439047575,
            0.010467353276908398,
            0.018293600529432297,
            0.08036193996667862,
            0.12977348268032074,
            0.33741647005081177,
        ],
        dtype=np.float32,
    )
    wres_data = np.array(
        [
            0.004129087086766958,
            0.00837691966444254,
            0.014061669819056988,
            0.03336339071393013,
            0.044351156800985336,
            0.11162598431110382,
        ],
        dtype=np.float32,
    )
    wwp_data = np.array(
        [
            0.0069597745314240456,
            0.013941819779574871,
            0.02337738312780857,
            0.05890510603785515,
            0.0802571177482605,
            0.20636671781539917,
        ],
        dtype=np.float32,
    )
    wfc_data = np.array(
        [
            0.017656346783041954,
            0.03488186001777649,
            0.05448378250002861,
            0.12172050029039383,
            0.15979675948619843,
            0.40186113119125366,
        ],
        dtype=np.float32,
    )
    ws_data = np.array(
        [
            0.028130417689681053,
            0.05417105555534363,
            0.07983528077602386,
            0.16036146879196167,
            0.20222823321819305,
            0.4983390271663666,
        ],
        dtype=np.float32,
    )
    delta_z_data = np.array(
        [
            0.07500000298023224,
            0.125,
            0.22500000894069672,
            0.3500000238418579,
            0.699999988079071,
        ],
        dtype=np.float32,
    )
    saturated_hydraulic_conductivity_m_per_s_data = np.array(
        [
            1.4807392290094867e-06,
            3.2266912057821173e-06,
            3.838434167846572e-06,
            1.802484007384919e-06,
            1.4587878922611708e-06,
            1.2659297681238968e-06,
        ],
        dtype=np.float32,
    )
    lambda_pore_size_distribution_data = np.array(
        [
            0.3132339417934418,
            0.31312474608421326,
            0.29538458585739136,
            0.25215277075767517,
            0.23889631032943726,
            0.22994033992290497,
        ],
        dtype=np.float32,
    )
    bubbling_pressure_cm_data = np.array(
        [
            17.230770111083984,
            18.913177490234375,
            21.203218460083008,
            27.398061752319336,
            32.20917892456055,
            34.9641227722168,
        ],
        dtype=np.float32,
    )
    pr_kg_per_m2_per_s_data = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    tas_2m_K_data = np.array(
        [
            255.0272216796875,
            254.10165405273438,
            253.6627197265625,
            252.8765869140625,
            252.2982177734375,
            251.7760009765625,
            252.34597778320312,
            255.46380615234375,
            259.33831787109375,
            261.64715576171875,
            263.0116271972656,
            263.94097900390625,
            264.61480712890625,
            265.01629638671875,
            265.4912414550781,
            265.0434265136719,
            264.3658752441406,
            263.15765380859375,
            261.99530029296875,
            261.9620361328125,
            261.5413818359375,
            260.8979797363281,
            260.3808898925781,
            259.7225036621094,
        ],
        dtype=np.float32,
    )
    dewpoint_tas_2m_K_data = np.array(
        [
            253.4105224609375,
            252.69598388671875,
            252.13119506835938,
            251.44378662109375,
            250.8382568359375,
            250.33804321289062,
            250.53549194335938,
            253.34689331054688,
            252.48822021484375,
            252.14804077148438,
            251.87521362304688,
            253.72872924804688,
            254.45169067382812,
            255.03048706054688,
            254.92291259765625,
            255.59716796875,
            255.84841918945312,
            256.2546081542969,
            256.5147705078125,
            255.89111328125,
            254.64678955078125,
            254.37774658203125,
            254.10726928710938,
            253.89950561523438,
        ],
        dtype=np.float32,
    )
    ps_pascal_data = np.array(
        [
            80013.3828125,
            79973.2890625,
            79910.2109375,
            79902.96875,
            79849.75,
            79849.75,
            79849.75,
            79907.5703125,
            79952.25,
            79952.25,
            80016.828125,
            80021.4296875,
            80023.4140625,
            80016.828125,
            79995.7890625,
            79995.7890625,
            79995.7890625,
            80016.828125,
            80023.4140625,
            80072.5234375,
            80138.25,
            80140.8984375,
            80138.25,
            80138.25,
        ],
        dtype=np.float32,
    )
    rlds_W_per_m2_data = np.array(
        [
            240.66983032226562,
            227.31858825683594,
            226.01441955566406,
            226.7765655517578,
            232.40240478515625,
            239.0749053955078,
            230.7742462158203,
            239.92845153808594,
            246.57806396484375,
            240.4760284423828,
            235.6538848876953,
            237.00503540039062,
            241.2831268310547,
            246.939453125,
            252.58811950683594,
            257.177001953125,
            270.8632507324219,
            280.9817810058594,
            295.815673828125,
            286.4650573730469,
            285.0284729003906,
            265.1391296386719,
            258.7989196777344,
            246.726318359375,
        ],
        dtype=np.float32,
    )
    rsds_W_per_m2_data = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            27.679231643676758,
            175.49609375,
            368.4490966796875,
            584.81787109375,
            756.2760620117188,
            876.4261474609375,
            930.758056640625,
            915.9420166015625,
            837.817138671875,
            698.3944091796875,
            506.23956298828125,
            272.4990234375,
            102.66202545166016,
            12.4603271484375,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    wind_u10m_m_per_s_data = np.array(
        [
            0.1171874925494194,
            0.2202148288488388,
            0.2202148288488388,
            0.14209187030792236,
            0.14209187030792236,
            0.2060546725988388,
            0.2060546725988388,
            0.1030273362994194,
            -0.12255655229091644,
            -0.33740234375,
            -0.2202148288488388,
            -0.2202148288488388,
            -0.2060546725988388,
            -0.1030273362994194,
            0.0,
            -0.05322469025850296,
            -0.11181843280792236,
            0.0,
            -0.04443359375,
            0.1030273362994194,
            0.28417766094207764,
            0.3232421875,
            0.3232421875,
            0.27880859375,
        ],
        dtype=np.float32,
    )
    wind_v10m_m_per_s_data = np.array(
        [
            0.8330097794532776,
            0.9414061903953552,
            0.9414061903953552,
            1.0532246828079224,
            1.04443359375,
            1.04443359375,
            0.9360371232032776,
            0.6181640625,
            -0.0283203125,
            -0.52734375,
            -0.7333984375,
            -0.83642578125,
            -0.8842813968658447,
            -0.9536132216453552,
            -1.12939453125,
            -1.2856465578079224,
            -1.6513690948486328,
            -1.4755879640579224,
            -1.09912109375,
            -0.7758788466453552,
            -0.3798828125,
            0.0605468675494194,
            0.1635742038488388,
            0.21679890155792236,
        ],
        dtype=np.float32,
    )
    CO2_ppm_data = np.float32(372.52)
    minimum_effective_root_depth_m_data = np.float32(0.25)
    soil_layer_height_data = np.array(
        [
            0.05000000074505806,
            0.10000000149011612,
            0.15000000596046448,
            0.30000001192092896,
            0.4000000059604645,
            1.0,
        ],
        dtype=np.float32,
    )
    crop_group_number_per_group_data = np.array(
        [
            4.5,
            4.5,
            3.5,
            4.5,
            3.0,
            4.5,
            5.0,
            5.0,
            3.5,
            3.0,
            4.5,
            5.0,
            2.0,
            5.0,
            4.5,
            4.0,
            4.5,
            4.0,
            5.0,
            3.0,
            4.5,
            2.5,
            2.5,
            4.0,
            3.0,
            2.5,
        ],
        dtype=np.float32,
    )

    # Reshape for single cell test
    land_use_type = np.array([land_use_type_data], dtype=np.int32)
    crop_map = np.array([crop_map_data], dtype=np.int32)

    # Choose dtypes based on parameter
    flt = np.float64 if asfloat64 else np.float32

    root_depth_m = np.array([root_depth_m_data], dtype=flt)
    slope_m_per_m = np.array([0.01], dtype=flt)
    hillslope_length_m = np.array([100.0], dtype=flt)
    groundwater_toplayer_conductivity_m_per_day = np.array([0.0001], dtype=flt)
    topwater_m = np.array([topwater_m_data], dtype=flt)
    snow_water_equivalent_m = np.array([snow_water_equivalent_m_data], dtype=flt)
    liquid_water_in_snow_m = np.array([liquid_water_in_snow_m_data], dtype=flt)
    snow_temperature_C = np.array([snow_temperature_C_data], dtype=flt)
    interception_storage_m = np.array([interception_storage_m_data], dtype=flt)
    interception_capacity_m = np.array([interception_capacity_m_data], dtype=flt)
    crop_factor = np.array([crop_factor_data], dtype=flt)
    actual_irrigation_consumption_m = np.array(
        [actual_irrigation_consumption_m_data], dtype=flt
    )
    capillar_rise_m = np.array([capillar_rise_m_data], dtype=flt)
    natural_crop_groups = np.array([natural_crop_groups_data], dtype=flt)
    variable_runoff_shape_beta = np.array([0.0], dtype=flt)  # not used in this test
    wetting_front_depth_m = np.array([0.0], dtype=flt)
    wetting_front_suction_head_m = np.array([0.0], dtype=flt)
    wetting_front_moisture_deficit = np.array([0.0], dtype=flt)
    green_ampt_active_layer_idx = np.array([0], dtype=np.int32)

    # 2D arrays: add cell dimension and set dtype
    w = w_data.reshape(-1, 1).astype(flt)
    wres = wres_data.reshape(-1, 1).astype(flt)
    wwp = wwp_data.reshape(-1, 1).astype(flt)
    wfc = wfc_data.reshape(-1, 1).astype(flt)
    ws = ws_data.reshape(-1, 1).astype(flt)
    delta_z = delta_z_data.reshape(-1, 1).astype(flt)
    saturated_hydraulic_conductivity_m_per_s = (
        saturated_hydraulic_conductivity_m_per_s_data.reshape(-1, 1).astype(flt)
    )
    lambda_pore_size_distribution = lambda_pore_size_distribution_data.reshape(
        -1, 1
    ).astype(flt)
    bubbling_pressure_cm = bubbling_pressure_cm_data.reshape(-1, 1).astype(flt)
    soil_layer_height = soil_layer_height_data.reshape(-1, 1).astype(flt)

    # Time series: add cell dimension and set dtype
    pr_kg_per_m2_per_s = pr_kg_per_m2_per_s_data.reshape(24, 1).astype(flt)
    tas_2m_K = tas_2m_K_data.reshape(24, 1).astype(flt)
    dewpoint_tas_2m_K = dewpoint_tas_2m_K_data.reshape(24, 1).astype(flt)
    ps_pascal = ps_pascal_data.reshape(24, 1).astype(flt)
    rlds_W_per_m2 = rlds_W_per_m2_data.reshape(24, 1).astype(flt)
    rsds_W_per_m2 = rsds_W_per_m2_data.reshape(24, 1).astype(flt)
    wind_u10m_m_per_s = wind_u10m_m_per_s_data.reshape(24, 1).astype(flt)
    wind_v10m_m_per_s = wind_v10m_m_per_s_data.reshape(24, 1).astype(flt)

    # Scalars by dtype
    CO2_ppm = flt(CO2_ppm_data)
    minimum_effective_root_depth_m = flt(minimum_effective_root_depth_m_data)
    interflow_multiplier = flt(1.0)
    crop_group_number_per_group = crop_group_number_per_group_data.astype(flt)

    soil_temperature_C = np.full_like(w, -5.0)  # Assume frozen/cold
    solid_heat_capacity_J_per_m2_K = np.full_like(w, 2e5)

    # Capture previous values before calling the model (arrays get modified in-place)
    snow_water_equivalent_prev = snow_water_equivalent_m.copy()
    liquid_water_in_snow_prev = liquid_water_in_snow_m.copy()
    interception_storage_prev = interception_storage_m.copy()
    topwater_m_prev = topwater_m.copy()
    w_prev = w.copy()

    # Call the land surface model
    results = land_surface_model(
        land_use_type=land_use_type,
        slope_m_per_m=slope_m_per_m,
        hillslope_length_m=hillslope_length_m,
        w=w,
        wres=wres,
        wwp=wwp,
        wfc=wfc,
        ws=ws,
        soil_temperature_C=soil_temperature_C,
        solid_heat_capacity_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
        delta_z=delta_z,
        soil_layer_height=soil_layer_height,
        root_depth_m=root_depth_m,
        topwater_m=topwater_m,
        variable_runoff_shape_beta=variable_runoff_shape_beta,
        snow_water_equivalent_m=snow_water_equivalent_m,
        liquid_water_in_snow_m=liquid_water_in_snow_m,
        snow_temperature_C=snow_temperature_C,
        interception_storage_m=interception_storage_m,
        interception_capacity_m=interception_capacity_m,
        pr_kg_per_m2_per_s=pr_kg_per_m2_per_s,
        tas_2m_K=tas_2m_K,
        dewpoint_tas_2m_K=dewpoint_tas_2m_K,
        ps_pascal=ps_pascal,
        rlds_W_per_m2=rlds_W_per_m2,
        rsds_W_per_m2=rsds_W_per_m2,
        wind_u10m_m_per_s=wind_u10m_m_per_s,
        wind_v10m_m_per_s=wind_v10m_m_per_s,
        CO2_ppm=CO2_ppm,
        crop_factor=crop_factor,
        crop_map=crop_map,
        actual_irrigation_consumption_m=actual_irrigation_consumption_m,
        capillar_rise_m=capillar_rise_m,
        groundwater_toplayer_conductivity_m_per_day=groundwater_toplayer_conductivity_m_per_day,
        saturated_hydraulic_conductivity_m_per_s=saturated_hydraulic_conductivity_m_per_s,
        wetting_front_depth_m=wetting_front_depth_m,
        wetting_front_suction_head_m=wetting_front_suction_head_m,
        wetting_front_moisture_deficit=wetting_front_moisture_deficit,
        lambda_pore_size_distribution=lambda_pore_size_distribution,
        bubbling_pressure_cm=bubbling_pressure_cm,
        natural_crop_groups=natural_crop_groups,
        crop_group_number_per_group=crop_group_number_per_group,
        minimum_effective_root_depth_m=minimum_effective_root_depth_m,
        green_ampt_active_layer_idx=green_ampt_active_layer_idx,
        interflow_multiplier=interflow_multiplier,
    )

    # Unpack the results
    (
        rain_m,
        snow_m,
        topwater_m_out,
        reference_evapotranspiration_grass_m,
        reference_evapotranspiration_water_m,
        snow_water_equivalent_m_out,
        liquid_water_in_snow_m_out,
        sublimation_or_deposition_m,
        snow_temperature_C_out,
        interception_storage_m_out,
        interception_evaporation_m,
        open_water_evaporation_m,
        runoff_m,
        groundwater_recharge_m,
        interflow_m,
        bare_soil_evaporation_m,
        transpiration_m,
        potential_transpiration_m,
    ) = results
    # Construct the balance check parameters
    influxes = [
        (pr_kg_per_m2_per_s.sum(axis=0) * 3.6).astype(np.float64),  # kg/m2/s -> m/hr
        actual_irrigation_consumption_m.astype(np.float64),
        capillar_rise_m.astype(np.float64),
    ]

    outfluxes = [
        (-sublimation_or_deposition_m).astype(np.float64),
        interception_evaporation_m.astype(np.float64),
        open_water_evaporation_m.astype(np.float64),
        runoff_m.sum(axis=0).astype(np.float64),  # sum over hours
        interflow_m.sum(axis=0).astype(np.float64),  # sum over hours
        groundwater_recharge_m.astype(np.float64),
        bare_soil_evaporation_m.astype(np.float64),
        transpiration_m.astype(np.float64),
    ]

    prestorages = [
        snow_water_equivalent_prev.astype(np.float64),
        liquid_water_in_snow_prev.astype(np.float64),
        interception_storage_prev.astype(np.float64),
        topwater_m_prev.astype(np.float64),
        np.nansum(w_prev, axis=0).astype(np.float64),
    ]

    poststorages = [
        snow_water_equivalent_m_out.astype(np.float64),
        liquid_water_in_snow_m_out.astype(np.float64),
        interception_storage_m_out.astype(np.float64),
        topwater_m_out.astype(np.float64),
        np.nansum(w, axis=0).astype(np.float64),
    ]

    # Check that the balance closes within tolerance
    assert balance_check(
        name=f"land surface test {'float64' if asfloat64 else 'float32'}",
        how="cellwise",
        influxes=influxes,
        outfluxes=outfluxes,
        prestorages=prestorages,
        poststorages=poststorages,
        tolerance=tolerance,
    )
