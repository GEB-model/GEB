import numpy as np

from geb.hydrology.evaporation import get_CO2_induced_crop_factor_adustment


def test_get_CO2_induced_crop_factor_adustment() -> None:
    assert get_CO2_induced_crop_factor_adustment(369.41) == 1.0
    assert get_CO2_induced_crop_factor_adustment(550.0) == 0.95
    assert get_CO2_induced_crop_factor_adustment(550.0 + (550.0 - 369.41) == 0.9)
