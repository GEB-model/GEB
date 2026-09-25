"""Tests for the land surface model in GEB."""

from pathlib import Path

import numpy as np
import pytest

from geb.hydrology.landsurface.constants import (
    RHO_WATER_KG_PER_M3,
    SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K,
    THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN,
    THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN,
)
from geb.hydrology.landsurface.landsurface_model import (
    LandSurfaceInputs,
    _pad_hru_arrays,
    land_surface_model,
)
from geb.workflows import balance_check


def get_error_cases() -> list[Path]:
    """Returns a list of paths to error cases in the test folder."""
    error_cases_dir: Path = Path(__file__).parent / "landsurface_error_cases"
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
    """Test the land surface model with previous error cases.

    Args:
        error_case_path: Path to the error case file.
        asfloat64: Whether to cast float inputs to float64 before running.
    """
    # Load the error case data
    with np.load(error_case_path) as data:
        inputs: dict = {key: data[key] for key in data.files}

    # Old error-case files were saved in layer-major layout (N_LAYERS, num_cells) or
    # (N_HOURS, num_cells). The refactored land_surface_model expects cell-major layout
    # (num_cells, N_LAYERS) / (num_cells, N_HOURS). Detect the old format by checking
    # for shape (N, 1) with N > 1 and transpose to (1, N) accordingly.
    for key, value in inputs.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            and value.shape[1] == 1
            and value.shape[0] > 1
        ):
            inputs[key] = np.ascontiguousarray(value.T)

    if "thermal_conductivity_saturated_unfrozen_W_per_m_K" not in inputs:
        porosity: np.ndarray = (
            inputs["water_content_saturated_m"] / inputs["soil_layer_height"]
        )
        solid_factor: np.ndarray = inputs["solid_thermal_conductivity_W_per_m_K"] ** (
            np.float32(1.0) - porosity
        )
        inputs["thermal_conductivity_saturated_unfrozen_W_per_m_K"] = solid_factor * (
            THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN**porosity
        )
        inputs["thermal_conductivity_saturated_frozen_W_per_m_K"] = solid_factor * (
            THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN**porosity
        )
    inputs.pop("solid_thermal_conductivity_W_per_m_K", None)
    if "daily_reference_evapotranspiration_grass_m" not in inputs:
        inputs["daily_reference_evapotranspiration_grass_m"] = np.full(
            inputs["root_depth_m"].shape, 0.003, dtype=np.float32
        )

    # Cast inputs if requested
    if asfloat64:
        for key, value in inputs.items():
            if isinstance(value, np.ndarray) and np.issubdtype(
                value.dtype, np.floating
            ):
                inputs[key] = value.astype(np.float64)
            elif isinstance(value, (float, np.floating)):
                inputs[key] = np.float64(value)
    else:
        # Match kernel dtype expectations
        inputs["snow_water_equivalent_m"] = inputs["snow_water_equivalent_m"].astype(
            np.float64
        )
        inputs["liquid_water_in_snow_m"] = inputs["liquid_water_in_snow_m"].astype(
            np.float64
        )

    # Convert 0-D arrays to Python/NumPy scalars so Numba typing succeeds
    for key, value in inputs.items():
        if isinstance(value, np.ndarray) and value.ndim == 0:
            inputs[key] = value.item()

    if "snow_enthalpy_J_per_m2" not in inputs and "snow_temperature_C" in inputs:
        snow_temperature_C: np.ndarray = inputs.pop("snow_temperature_C").astype(
            np.float32
        )
        inputs["snow_enthalpy_J_per_m2"] = (
            inputs["snow_water_equivalent_m"].astype(np.float32)
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K
            * np.minimum(snow_temperature_C, np.float32(0.0))
        ).astype(np.float32)

    # Extract initial storages for balance check
    pre_water_content_m: np.ndarray = inputs["water_content_m"].copy()
    pre_topwater_m: np.ndarray = inputs["topwater_m"].copy()
    pre_snow_water_equivalent_m: np.ndarray = inputs["snow_water_equivalent_m"].copy()
    pre_liquid_water_in_snow_m: np.ndarray = inputs["liquid_water_in_snow_m"].copy()
    pre_interception_storage_m: np.ndarray = inputs["interception_storage_m"].copy()
    pre_soil_enthalpy: np.ndarray = inputs["soil_enthalpy_J_per_m2"].copy()

    # The land surface kernel processes in SIMD blocks of 16 cells; pad inputs
    # to the next multiple of 16 cells, and slice outputs back to original cell count.
    num_original_cells: int = inputs["slope_m_per_m"].shape[0]
    padded_inputs: LandSurfaceInputs = _pad_hru_arrays(LandSurfaceInputs(**inputs))
    raw_results = land_surface_model(**padded_inputs._asdict())

    trimmed_results = tuple(
        r[:num_original_cells]
        if isinstance(r, np.ndarray) and r.ndim == 1
        else r[:num_original_cells, :]
        if isinstance(r, np.ndarray) and r.ndim == 2
        else r
        for r in raw_results
    )

    (
        out_rain_m,
        out_snow_m,
        post_topwater_m,
        out_ref_et_grass_m,
        out_ref_et_water_m,
        post_snow_water_equivalent_m,
        post_liquid_water_in_snow_m,
        out_sublimation_m,
        post_snow_enthalpy_J_per_m2,
        post_snow_density_kg_per_m3,
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
        _,
        out_top_soil_rise_from_layer_2_m,
        _,
        out_top_soil_transpiration_m,
        out_evapotranspiration_m,
    ) = trimmed_results

    assert np.all(out_top_soil_rise_from_layer_2_m >= 0.0)
    assert np.all(out_top_soil_transpiration_m >= 0.0)
    assert np.all(out_top_soil_transpiration_m <= out_transpiration_m + 1e-6)

    # Perform water balance check using in-place updated water content
    post_water_content_m: np.ndarray = padded_inputs.water_content_m[
        :num_original_cells
    ]

    is_water_balanced: bool = balance_check(
        name=f"Water balance: {error_case_path.name}",
        how="cellwise",
        influxes=[
            inputs["pr_kg_per_m2_per_s"].sum(axis=1) * 3.6,
            inputs["actual_irrigation_consumption_m"],
            inputs["capillar_rise_m"],
        ],
        outfluxes=[
            -out_sublimation_m,
            out_interception_evaporation_m,
            out_open_water_evaporation_m,
            out_runoff_m.sum(axis=1),
            out_interflow_m.sum(axis=1),
            out_groundwater_recharge_m,
            out_bare_soil_evaporation,
            out_transpiration_m,
        ],
        prestorages=[
            pre_snow_water_equivalent_m.sum(axis=1)
            if pre_snow_water_equivalent_m.ndim == 2
            else pre_snow_water_equivalent_m,
            pre_liquid_water_in_snow_m.sum(axis=1)
            if pre_liquid_water_in_snow_m.ndim == 2
            else pre_liquid_water_in_snow_m,
            pre_interception_storage_m,
            pre_topwater_m,
            pre_water_content_m.sum(axis=1),
        ],
        poststorages=[
            post_snow_water_equivalent_m.sum(axis=1)
            if post_snow_water_equivalent_m.ndim == 2
            else post_snow_water_equivalent_m,
            post_liquid_water_in_snow_m.sum(axis=1)
            if post_liquid_water_in_snow_m.ndim == 2
            else post_liquid_water_in_snow_m,
            post_interception_storage_m,
            post_topwater_m,
            post_water_content_m.sum(axis=1),
        ],
        tolerance=1e-5,
        raise_on_error=False,
    )

    # Perform enthalpy balance check using in-place updated soil enthalpy
    post_soil_enthalpy: np.ndarray = padded_inputs.soil_enthalpy_J_per_m2[
        :num_original_cells
    ]

    is_enthalpy_balanced: bool = balance_check(
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
        prestorages=[pre_soil_enthalpy.sum(axis=1)],
        poststorages=[post_soil_enthalpy.sum(axis=1)],
        tolerance=1e3,
        raise_on_error=False,
    )

    # The test passes if the balances are correct.
    # If they fail, we can see which one failed in the output.
    assert is_water_balanced, f"Water balance failed for {error_case_path.name}"
    assert is_enthalpy_balanced, f"Enthalpy balance failed for {error_case_path.name}"
