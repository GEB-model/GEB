import math

from .modules.plantFATE import Model


def test_soil_water_potential() -> None:
    model = Model("tests/p_daily.ini")
    wilting_point = -1500  # kPa
    field_capacity = -33  # kPa

    # assert soil water potential is wilting point when soil moisture is wilting point
    assert (
        model.calculate_soil_water_potential_MPa(
            soil_moisture=0.126,  # saturation
            soil_moisture_wilting_point=0.126,
            soil_moisture_field_capacity=0.268,
            soil_tickness=1,
            wilting_point=wilting_point,  # kPa
            field_capacity=field_capacity,  # kPa
        )
        * 1000
        == wilting_point
    )  # Loam

    assert (
        model.calculate_soil_water_potential_MPa(
            soil_moisture=0.268,  # saturation
            soil_moisture_wilting_point=0.126,
            soil_moisture_field_capacity=0.268,
            soil_tickness=1,
            wilting_point=wilting_point,  # kPa
            field_capacity=field_capacity,  # kPa
        )
        * 1000
        == field_capacity
    )  # Loam

    # import numpy as np
    # import matplotlib.pyplot as plt
    # s = np.linspace(0.126, 0.268, 100)
    # swp = []
    # for soil_moisture in s:
    #     swp.append(model.calculate_soil_water_potential_MPa(
    #         soil_moisture=soil_moisture,  # saturation
    #         soil_moisture_wilting_point=0.126,
    #         soil_moisture_field_capacity=0.268,
    #         soil_tickness=1,
    #         wilting_point=wilting_point,  # kPa
    #         field_capacity=field_capacity  # kPa
    #     ) * 1000) # Loam

    # plt.plot(s, swp)
    # # invert y axis
    # plt.gca().invert_yaxis()
    # # set xlim to 0
    # plt.xlim(0, .5)
    # plt.savefig('swp.png')


def test_vapour_pressure_deficit() -> None:
    model = Model("tests/p_daily.ini")
    assert (
        model.calculate_vapour_pressure_deficit_kPa(
            temperature=20, relative_humidity=60
        )
        == 0.9345917710195297
    )


def test_calculate_photosynthetic_photon_flux_density() -> None:
    model = Model("tests/p_daily.ini")
    assert math.isclose(
        model.calculate_photosynthetic_photon_flux_density(
            shortwave_radiation=200, xi=0.5
        ),
        460.0,
    )


if __name__ == "__main__":
    test_soil_water_potential()
    test_vapour_pressure_deficit()
    test_calculate_photosynthetic_photon_flux_density()
