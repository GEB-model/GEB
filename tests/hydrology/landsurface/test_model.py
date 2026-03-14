"""Tests for the land surface model in GEB."""

from pathlib import Path

import numpy as np
import pytest

from geb.hydrology.landsurface import model as landsurface
from geb.hydrology.landsurface.model import land_surface_model
from geb.workflows import balance_check


def get_error_cases() -> list[Path]:
    """Returns a list of paths to error cases in the test folder."""
    error_cases_dir = Path(__file__).parent / "landsurface_error_cases"
    if not error_cases_dir.exists():
        return []
    return list(error_cases_dir.glob("*.npz"))


@pytest.mark.parametrize(
    "error_case_path, asfloat64",
    [(case, False) for case in get_error_cases()]
    + [(case, True) for case in get_error_cases()],
    ids=lambda x: f"{x.name}-float64" if isinstance(x, Path) else str(x),
)
def test_land_surface_model_error_cases(error_case_path: Path, asfloat64: bool) -> None:
    """Test the land surface model with previous error cases."""
    # Set the global N_SOIL_LAYERS variable required by the numba function
    landsurface.N_SOIL_LAYERS = 6

    # Load the error case data
    with np.load(error_case_path) as data:
        inputs = {key: data[key] for key in data.files}

    # Cast inputs if requested
    if asfloat64:
        for key, value in inputs.items():
            if isinstance(value, np.ndarray) and np.issubdtype(
                value.dtype, np.floating
            ):
                inputs[key] = value.astype(np.float64)
            elif isinstance(value, (float, np.floating)):
                inputs[key] = np.float64(value)

    inputs["snow_water_equivalent_m"] = inputs["snow_water_equivalent_m"].astype(
        np.float64
    )
    inputs["liquid_water_in_snow_m"] = inputs["liquid_water_in_snow_m"].astype(
        np.float64
    )

    # Extract initial storages for balance check
    pre_water_content_m = inputs["water_content_m"].copy()
    pre_topwater_m = inputs["topwater_m"].copy()
    pre_snow_water_equivalent_m = inputs["snow_water_equivalent_m"].copy()
    pre_liquid_water_in_snow_m = inputs["liquid_water_in_snow_m"].copy()
    pre_interception_storage_m = inputs["interception_storage_m"].copy()
    pre_soil_enthalpy = inputs["soil_enthalpy_J_per_m2"].copy()

    results = land_surface_model(**inputs)

    (
        out_rain_m,
        out_snow_m,
        post_topwater_m,
        out_ref_et_grass_m,
        out_ref_et_water_m,
        post_snow_water_equivalent_m,
        post_liquid_water_in_snow_m,
        out_sublimation_m,
        post_snow_temperature_C,
        post_interception_storage_m,
        out_interception_evaporation_m,
        out_open_water_evaporation_m,
        out_runoff_m,  # This is the 2D runoff_m [substep, indices]
        out_groundwater_recharge_m,
        out_interflow_m,  # This is the 2D interflow_m [substep, indices]
        out_bare_soil_evaporation,
        out_transpiration_m,
        out_potential_transpiration_m,
        out_potential_evapotranspiration_m,
        out_soil_boundary_h_flux,
        out_rain_adv_h_flux,
        out_evap_cool_h_loss,
        out_interflow_h_loss,
        out_gw_recharge_h_loss,
        out_transpiration_h_loss,
    ) = results

    # Perform water balance check
    post_water_content_m = inputs["water_content_m"]

    is_water_balanced = balance_check(
        name=f"Water balance: {error_case_path.name}",
        how="cellwise",
        influxes=[
            inputs["pr_kg_per_m2_per_s"].sum(axis=0) * 3.6,
            inputs["actual_irrigation_consumption_m"],
            inputs["capillar_rise_m"],
        ],
        outfluxes=[
            -out_sublimation_m,
            out_interception_evaporation_m,
            out_open_water_evaporation_m,
            out_runoff_m.sum(axis=0),
            out_interflow_m.sum(axis=0),
            out_groundwater_recharge_m,
            out_bare_soil_evaporation,
            out_transpiration_m,
        ],
        prestorages=[
            pre_snow_water_equivalent_m,
            pre_liquid_water_in_snow_m,
            pre_interception_storage_m,
            pre_topwater_m,
            pre_water_content_m.sum(axis=0),
        ],
        poststorages=[
            post_snow_water_equivalent_m,
            post_liquid_water_in_snow_m,
            post_interception_storage_m,
            post_topwater_m,
            post_water_content_m.sum(axis=0),
        ],
        tolerance=1e-5,
        raise_on_error=False,
    )

    # Perform enthalpy balance check
    post_soil_enthalpy = inputs["soil_enthalpy_J_per_m2"]

    is_enthalpy_balanced = balance_check(
        name=f"Enthalpy balance: {error_case_path.name}",
        how="cellwise",
        influxes=[
            out_soil_boundary_h_flux,
            out_rain_adv_h_flux,
        ],
        outfluxes=[
            out_evap_cool_h_loss,
            out_interflow_h_loss,
            out_gw_recharge_h_loss,
            out_transpiration_h_loss,
        ],
        prestorages=[pre_soil_enthalpy.sum(axis=0)],
        poststorages=[post_soil_enthalpy.sum(axis=0)],
        tolerance=1e3,
        raise_on_error=False,
    )

    # The test passes if the balances are correct.
    # If they fail, we can see which one failed in the output.
    assert is_water_balanced, f"Water balance failed for {error_case_path.name}"
    assert is_enthalpy_balanced, f"Enthalpy balance failed for {error_case_path.name}"
