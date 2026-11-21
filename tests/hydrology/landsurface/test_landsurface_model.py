"""Tests for the land surface model in GEB."""

from pathlib import Path

import numpy as np
import pytest

from geb.hydrology.landsurface.constants import (
    N_SOIL_LAYERS,
    N_STATIC_SOIL_PROPS,
    RHO_WATER_KG_PER_M3,
    SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K,
    STATIC_SOIL_PROPERTY_NAMES,
    THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN,
    THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN,
)
from geb.hydrology.landsurface.energy import get_soil_heat_flux_net_of_evaporation
from geb.hydrology.landsurface.evapotranspiration import (
    get_dew,
    limit_evaporation_under_snow,
)
from geb.hydrology.landsurface.landsurface_model import (
    LandSurfaceInputs,
    land_surface_model,
)
from geb.hydrology.landsurface.potential_evapotranspiration import (
    get_forest_leaf_area_index_with_evergreen,
)
from geb.hydrology.landsurface.water import (
    get_coarse_fragment_factors,
    get_regolith_layer_thickness_m,
    get_regolith_vertical_conductivity_factor,
    get_slope_soil_depth_factor,
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
    if "groundwater_depth_m" not in inputs:
        inputs["groundwater_depth_m"] = np.full(
            inputs["root_depth_m"].shape, 10.0, dtype=np.float32
        )
    if "groundwater_depth_previous_day_m" not in inputs:
        inputs["groundwater_depth_previous_day_m"] = inputs[
            "groundwater_depth_m"
        ].copy()
    if "max_capillary_rise_m" not in inputs:
        inputs["max_capillary_rise_m"] = np.full(
            inputs["root_depth_m"].shape, np.inf, dtype=np.float32
        )
    if "groundwater_toplayer_conductivity_m_per_s" not in inputs:
        if "groundwater_toplayer_conductivity_m_per_day" in inputs:
            inputs["groundwater_toplayer_conductivity_m_per_s"] = (
                inputs.pop("groundwater_toplayer_conductivity_m_per_day") / 86400.0
            ).astype(np.float32)
        else:
            inputs["groundwater_toplayer_conductivity_m_per_s"] = np.zeros(
                inputs["root_depth_m"].shape, dtype=np.float32
            )
    inputs.pop("capillar_rise_m", None)
    if "snow_albedo" not in inputs:
        inputs["snow_albedo"] = np.full(
            inputs["root_depth_m"].shape, 0.85, dtype=np.float32
        )
    if "HRU_to_grid" not in inputs:
        num_cells_err: int = inputs["slope_m_per_m"].shape[0]
        inputs["HRU_to_grid"] = np.arange(num_cells_err, dtype=np.int32)

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

    if "packed_soil_properties" not in inputs:
        num_cells_input: int = inputs["slope_m_per_m"].shape[0]
        n_layers_input: int = inputs["water_content_residual_m"].shape[1]
        dtype = np.float64 if asfloat64 else np.float32
        packed_soil: np.ndarray = np.empty(
            (num_cells_input, N_STATIC_SOIL_PROPS, n_layers_input), dtype=dtype
        )
        for idx, prop in enumerate(STATIC_SOIL_PROPERTY_NAMES):
            packed_soil[:, idx, :] = inputs.pop(prop)
        inputs["packed_soil_properties"] = packed_soil

    raw_results = land_surface_model(**inputs)
    trimmed_results = raw_results

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
        out_capillar_rise_m,
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
        _,  # dew_m, included in out_interception_evaporation_m
    ) = trimmed_results

    assert np.all(out_top_soil_rise_from_layer_2_m >= 0.0)
    assert np.all(out_top_soil_transpiration_m >= 0.0)
    assert np.all(out_top_soil_transpiration_m <= out_transpiration_m + 1e-6)
    assert np.all(post_topwater_m >= 0.0), (
        f"Negative topwater: {np.min(post_topwater_m)}"
    )
    assert np.all(out_runoff_m >= 0.0), f"Negative runoff: {np.min(out_runoff_m)}"

    # Perform water balance check using in-place updated water content
    post_water_content_m: np.ndarray = inputs["water_content_m"]

    is_water_balanced: bool = balance_check(
        name=f"Water balance: {error_case_path.name}",
        how="cellwise",
        influxes=[
            inputs["pr_kg_per_m2_per_s"].sum(axis=1) * 3.6,
            inputs["actual_irrigation_consumption_m"],
            out_capillar_rise_m,
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
    post_soil_enthalpy: np.ndarray = inputs["soil_enthalpy_J_per_m2"]

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


from unittest.mock import MagicMock, patch

from geb.geb_types import TwoDArrayFloat32
from geb.hydrology.landsurface.landsurface_model import LandSurface


def _create_mock_land_surface(
    tmp_path: Path,
) -> tuple[LandSurface, LandSurfaceInputs]:
    """Create a mock LandSurface instance and minimal LandSurfaceInputs for testing.

    Args:
        tmp_path: Path to temporary directory for diagnostics output.

    Returns:
        A tuple of (mock LandSurface instance, mock LandSurfaceInputs namedtuple).
    """
    land_surface: LandSurface = object.__new__(LandSurface)
    mock_model: MagicMock = MagicMock()
    mock_model.diagnostics_folder = tmp_path
    mock_model.logger = MagicMock()
    land_surface.model = mock_model

    num_cells: int = 2
    mock_hru: MagicMock = MagicMock()
    mock_hru_var: MagicMock = MagicMock()
    mock_hru_var.land_use_type = np.array([1, 2], dtype=np.int32)
    mock_hru_var.topwater_m = np.zeros(num_cells, dtype=np.float32)
    mock_hru_var.water_content_m = np.full(
        (num_cells, N_SOIL_LAYERS), 0.1, dtype=np.float32
    )
    mock_hru_var.snow_water_equivalent_m = np.zeros((num_cells, 2), dtype=np.float64)
    mock_hru_var.liquid_water_in_snow_m = np.zeros((num_cells, 2), dtype=np.float64)
    mock_hru_var.interception_storage_m = np.zeros(num_cells, dtype=np.float32)
    mock_hru.var = mock_hru_var
    land_surface.HRU = mock_hru

    inputs_dict: dict = {}
    for field in LandSurfaceInputs._fields:
        if field == "packed_soil_properties":
            inputs_dict[field] = np.zeros(
                (num_cells, N_STATIC_SOIL_PROPS, N_SOIL_LAYERS), dtype=np.float32
            )
        elif "water_content" in field or "soil_enthalpy" in field:
            inputs_dict[field] = np.zeros((num_cells, N_SOIL_LAYERS), dtype=np.float32)
        elif field in (
            "pr_kg_per_m2_per_s",
            "tas_2m_K",
            "direct_radiation_W_per_m2",
            "diffuse_radiation_W_per_m2",
        ):
            inputs_dict[field] = np.zeros((num_cells, 24), dtype=np.float32)
        elif field in ("snow_water_equivalent_m", "liquid_water_in_snow_m"):
            inputs_dict[field] = np.zeros((num_cells, 2), dtype=np.float64)
        elif field in ("snow_enthalpy_J_per_m2", "snow_density_kg_per_m3"):
            inputs_dict[field] = np.zeros((num_cells, 2), dtype=np.float32)
        elif field in (
            "land_use_type",
            "crop_map",
            "green_ampt_active_layer_idx",
            "HRU_to_grid",
        ):
            inputs_dict[field] = np.zeros(num_cells, dtype=np.int32)
        elif field == "unix_time_seconds":
            inputs_dict[field] = np.int64(0)
        else:
            inputs_dict[field] = np.zeros(num_cells, dtype=np.float32)

    mock_inputs: LandSurfaceInputs = LandSurfaceInputs(**inputs_dict)
    return land_surface, mock_inputs


def test_check_land_surface_water_valid(tmp_path: Path) -> None:
    """Test that valid, balanced water states pass without raising an error."""
    land_surface, mock_inputs = _create_mock_land_surface(tmp_path)
    num_cells: int = 2

    # All zeros and balanced
    land_surface._check_land_surface_water(
        land_surface_inputs=mock_inputs,
        water_content_m_prev=np.full((num_cells, N_SOIL_LAYERS), 0.1, dtype=np.float32),
        topwater_m_prev=np.zeros(num_cells, dtype=np.float32),
        snow_water_equivalent_prev=np.zeros((num_cells, 2), dtype=np.float64),
        liquid_water_in_snow_prev=np.zeros((num_cells, 2), dtype=np.float64),
        snow_enthalpy_J_per_m2_prev=np.zeros((num_cells, 2), dtype=np.float32),
        snow_density_kg_per_m3_prev=np.zeros((num_cells, 2), dtype=np.float32),
        interception_storage_prev=np.zeros(num_cells, dtype=np.float32),
        soil_enthalpy_J_per_m2_prev=np.zeros(
            (num_cells, N_SOIL_LAYERS), dtype=np.float32
        ),
        deep_soil_temperature_C_prev=np.zeros(num_cells, dtype=np.float32),
        wetting_front_depth_prev=np.zeros(num_cells, dtype=np.float32),
        wetting_front_suction_head_prev=np.zeros(num_cells, dtype=np.float32),
        wetting_front_moisture_deficit_prev=np.zeros(num_cells, dtype=np.float32),
        green_ampt_active_layer_idx_prev=np.zeros(num_cells, dtype=np.int32),
        pr_kg_per_m2_per_s=np.zeros((num_cells, 24), dtype=np.float32),
        actual_irrigation_consumption_m=np.zeros(num_cells, dtype=np.float32),
        capillar_rise_m=np.zeros(num_cells, dtype=np.float32),
        sublimation_or_deposition_m=np.zeros(num_cells, dtype=np.float32),
        interception_evaporation_m=np.zeros(num_cells, dtype=np.float32),
        dew_m=np.zeros(num_cells, dtype=np.float32),
        open_water_evaporation_m=np.zeros(num_cells, dtype=np.float32),
        runoff_m=np.zeros((num_cells, 24), dtype=np.float32),
        interflow_m=np.zeros((num_cells, 24), dtype=np.float32),
        groundwater_recharge_m=np.zeros(num_cells, dtype=np.float32),
        bare_soil_evaporation_m=np.zeros(num_cells, dtype=np.float32),
        transpiration_m=np.zeros(num_cells, dtype=np.float32),
    )


def test_check_land_surface_water_negative_topwater(tmp_path: Path) -> None:
    """Test that negative topwater triggers export of diagnostic snapshot and raises."""
    land_surface, mock_inputs = _create_mock_land_surface(tmp_path)
    num_cells: int = 2
    # Set negative topwater on cell 1
    land_surface.HRU.var.topwater_m = np.array([0.0, -0.005], dtype=np.float32)

    with patch(
        "geb.hydrology.landsurface.landsurface_model.land_surface_model"
    ) as mock_kernel:
        with pytest.raises(AssertionError, match="Negative topwater detected"):
            land_surface._check_land_surface_water(
                land_surface_inputs=mock_inputs,
                water_content_m_prev=np.full(
                    (num_cells, N_SOIL_LAYERS), 0.1, dtype=np.float32
                ),
                topwater_m_prev=np.zeros(num_cells, dtype=np.float32),
                snow_water_equivalent_prev=np.zeros((num_cells, 2), dtype=np.float64),
                liquid_water_in_snow_prev=np.zeros((num_cells, 2), dtype=np.float64),
                snow_enthalpy_J_per_m2_prev=np.zeros((num_cells, 2), dtype=np.float32),
                snow_density_kg_per_m3_prev=np.zeros((num_cells, 2), dtype=np.float32),
                interception_storage_prev=np.zeros(num_cells, dtype=np.float32),
                soil_enthalpy_J_per_m2_prev=np.zeros(
                    (num_cells, N_SOIL_LAYERS), dtype=np.float32
                ),
                deep_soil_temperature_C_prev=np.zeros(num_cells, dtype=np.float32),
                wetting_front_depth_prev=np.zeros(num_cells, dtype=np.float32),
                wetting_front_suction_head_prev=np.zeros(num_cells, dtype=np.float32),
                wetting_front_moisture_deficit_prev=np.zeros(
                    num_cells, dtype=np.float32
                ),
                green_ampt_active_layer_idx_prev=np.zeros(num_cells, dtype=np.int32),
                pr_kg_per_m2_per_s=np.zeros((num_cells, 24), dtype=np.float32),
                actual_irrigation_consumption_m=np.zeros(num_cells, dtype=np.float32),
                capillar_rise_m=np.zeros(num_cells, dtype=np.float32),
                sublimation_or_deposition_m=np.zeros(num_cells, dtype=np.float32),
                interception_evaporation_m=np.zeros(num_cells, dtype=np.float32),
                dew_m=np.zeros(num_cells, dtype=np.float32),
                open_water_evaporation_m=np.zeros(num_cells, dtype=np.float32),
                runoff_m=np.zeros((num_cells, 24), dtype=np.float32),
                interflow_m=np.zeros((num_cells, 24), dtype=np.float32),
                groundwater_recharge_m=np.zeros(num_cells, dtype=np.float32),
                bare_soil_evaporation_m=np.zeros(num_cells, dtype=np.float32),
                transpiration_m=np.zeros(num_cells, dtype=np.float32),
            )

        # Verify diagnostic snapshot was exported for cell 1
        diag_file: Path = tmp_path / "diagnostic_landsurface_water_error_cell_1.npz"
        assert diag_file.exists()
        mock_kernel.assert_called_once()


def test_check_land_surface_water_negative_runoff(tmp_path: Path) -> None:
    """Test that negative runoff triggers export of diagnostic snapshot and raises."""
    land_surface, mock_inputs = _create_mock_land_surface(tmp_path)
    num_cells: int = 2
    runoff_m: TwoDArrayFloat32 = np.zeros((num_cells, 24), dtype=np.float32)
    runoff_m[0, 12] = np.float32(-0.002)

    with patch(
        "geb.hydrology.landsurface.landsurface_model.land_surface_model"
    ) as mock_kernel:
        with pytest.raises(AssertionError, match="Negative runoff detected"):
            land_surface._check_land_surface_water(
                land_surface_inputs=mock_inputs,
                water_content_m_prev=np.full(
                    (num_cells, N_SOIL_LAYERS), 0.1, dtype=np.float32
                ),
                topwater_m_prev=np.zeros(num_cells, dtype=np.float32),
                snow_water_equivalent_prev=np.zeros((num_cells, 2), dtype=np.float64),
                liquid_water_in_snow_prev=np.zeros((num_cells, 2), dtype=np.float64),
                snow_enthalpy_J_per_m2_prev=np.zeros((num_cells, 2), dtype=np.float32),
                snow_density_kg_per_m3_prev=np.zeros((num_cells, 2), dtype=np.float32),
                interception_storage_prev=np.zeros(num_cells, dtype=np.float32),
                soil_enthalpy_J_per_m2_prev=np.zeros(
                    (num_cells, N_SOIL_LAYERS), dtype=np.float32
                ),
                deep_soil_temperature_C_prev=np.zeros(num_cells, dtype=np.float32),
                wetting_front_depth_prev=np.zeros(num_cells, dtype=np.float32),
                wetting_front_suction_head_prev=np.zeros(num_cells, dtype=np.float32),
                wetting_front_moisture_deficit_prev=np.zeros(
                    num_cells, dtype=np.float32
                ),
                green_ampt_active_layer_idx_prev=np.zeros(num_cells, dtype=np.int32),
                pr_kg_per_m2_per_s=np.zeros((num_cells, 24), dtype=np.float32),
                actual_irrigation_consumption_m=np.zeros(num_cells, dtype=np.float32),
                capillar_rise_m=np.zeros(num_cells, dtype=np.float32),
                sublimation_or_deposition_m=np.zeros(num_cells, dtype=np.float32),
                interception_evaporation_m=np.zeros(num_cells, dtype=np.float32),
                dew_m=np.zeros(num_cells, dtype=np.float32),
                open_water_evaporation_m=np.zeros(num_cells, dtype=np.float32),
                runoff_m=runoff_m,
                interflow_m=np.zeros((num_cells, 24), dtype=np.float32),
                groundwater_recharge_m=np.zeros(num_cells, dtype=np.float32),
                bare_soil_evaporation_m=np.zeros(num_cells, dtype=np.float32),
                transpiration_m=np.zeros(num_cells, dtype=np.float32),
            )

        diag_file: Path = tmp_path / "diagnostic_landsurface_water_error_cell_0.npz"
        assert diag_file.exists()
        mock_kernel.assert_called_once()


def test_check_land_surface_water_imbalance(tmp_path: Path) -> None:
    """Test that water balance imbalance triggers diagnostic export and raises."""
    land_surface, mock_inputs = _create_mock_land_surface(tmp_path)
    num_cells: int = 2

    with patch(
        "geb.hydrology.landsurface.landsurface_model.land_surface_model"
    ) as mock_kernel:
        with pytest.raises(
            AssertionError, match="Land surface water balance check failed"
        ):
            land_surface._check_land_surface_water(
                land_surface_inputs=mock_inputs,
                water_content_m_prev=np.full(
                    (num_cells, N_SOIL_LAYERS), 0.1, dtype=np.float32
                ),
                topwater_m_prev=np.zeros(num_cells, dtype=np.float32),
                snow_water_equivalent_prev=np.zeros((num_cells, 2), dtype=np.float64),
                liquid_water_in_snow_prev=np.zeros((num_cells, 2), dtype=np.float64),
                snow_enthalpy_J_per_m2_prev=np.zeros((num_cells, 2), dtype=np.float32),
                snow_density_kg_per_m3_prev=np.zeros((num_cells, 2), dtype=np.float32),
                interception_storage_prev=np.zeros(num_cells, dtype=np.float32),
                soil_enthalpy_J_per_m2_prev=np.zeros(
                    (num_cells, N_SOIL_LAYERS), dtype=np.float32
                ),
                deep_soil_temperature_C_prev=np.zeros(num_cells, dtype=np.float32),
                wetting_front_depth_prev=np.zeros(num_cells, dtype=np.float32),
                wetting_front_suction_head_prev=np.zeros(num_cells, dtype=np.float32),
                wetting_front_moisture_deficit_prev=np.zeros(
                    num_cells, dtype=np.float32
                ),
                green_ampt_active_layer_idx_prev=np.zeros(num_cells, dtype=np.int32),
                pr_kg_per_m2_per_s=np.zeros((num_cells, 24), dtype=np.float32),
                actual_irrigation_consumption_m=np.array([5.0, 0.0], dtype=np.float32),
                capillar_rise_m=np.zeros(num_cells, dtype=np.float32),
                sublimation_or_deposition_m=np.zeros(num_cells, dtype=np.float32),
                interception_evaporation_m=np.zeros(num_cells, dtype=np.float32),
                dew_m=np.zeros(num_cells, dtype=np.float32),
                open_water_evaporation_m=np.zeros(num_cells, dtype=np.float32),
                runoff_m=np.zeros((num_cells, 24), dtype=np.float32),
                interflow_m=np.zeros((num_cells, 24), dtype=np.float32),
                groundwater_recharge_m=np.zeros(num_cells, dtype=np.float32),
                bare_soil_evaporation_m=np.zeros(num_cells, dtype=np.float32),
                transpiration_m=np.zeros(num_cells, dtype=np.float32),
            )

        diag_file: Path = tmp_path / "diagnostic_landsurface_water_error_cell_0.npz"
        assert diag_file.exists()
        mock_kernel.assert_called_once()


def test_get_slope_soil_depth_factor() -> None:
    """Soil thins with slope down to 0.1 of its depth."""
    slope = np.array([[0.0, 0.2], [5.0, 0.0]], dtype=np.float32)
    factor = get_slope_soil_depth_factor(slope, 10.0)
    assert factor.dtype == np.float32
    np.testing.assert_allclose(factor, [[1.0, 1.0 / 3.0], [0.1, 1.0]], rtol=1e-6)


def test_get_coarse_fragment_factors() -> None:
    """Stones hold no water and lower conductivity following Peck & Watson (1979)."""
    coarse_fragments_percentage = np.array([[0.0, 30.0, 95.0]], dtype=np.float32)
    fine_earth_fraction, conductivity_factor = get_coarse_fragment_factors(
        coarse_fragments_percentage
    )
    assert fine_earth_fraction.dtype == np.float32
    assert conductivity_factor.dtype == np.float32
    # 95% is capped at 90% so a layer always keeps some fine earth.
    np.testing.assert_allclose(fine_earth_fraction, [[1.0, 0.7, 0.1]], rtol=1e-6)
    np.testing.assert_allclose(
        conductivity_factor,
        [[1.0, 2 * 0.7 / 2.3, 2 * 0.1 / 2.9]],
        rtol=1e-6,
    )
    # Conductivity drops faster than the fine-earth fraction.
    assert (conductivity_factor <= fine_earth_fraction).all()


def test_get_regolith_layer_thickness_m() -> None:
    """Regolith spans from 2 m to the depth to bedrock, at most 2 m, at least 5 cm."""
    depth_to_bedrock_m = np.array([1.0, 2.5, 3.0, 15.0], dtype=np.float32)
    thickness = get_regolith_layer_thickness_m(depth_to_bedrock_m)
    assert thickness.dtype == np.float32
    np.testing.assert_allclose(thickness, [0.05, 0.5, 1.0, 2.0], rtol=1e-6)


def test_get_regolith_vertical_conductivity_factor() -> None:
    """Area-weighted mean of slow weathered rock and fast epikarst."""
    karst_fraction = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    factor = get_regolith_vertical_conductivity_factor(10.0, 1.0, karst_fraction)
    assert factor.dtype == np.float32
    np.testing.assert_allclose(factor, [0.1, 0.55, 1.0], rtol=1e-6)
    np.testing.assert_allclose(
        get_regolith_vertical_conductivity_factor(10.0, 10.0, karst_fraction),
        [0.1, 5.05, 10.0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        get_regolith_vertical_conductivity_factor(1.0, 1.0, karst_fraction), 1.0
    )
    with pytest.raises(AssertionError):
        get_regolith_vertical_conductivity_factor(0.5, 1.0, karst_fraction)


def test_get_forest_leaf_area_index_with_evergreen() -> None:
    """Evergreen forest keeps its annual maximum LAI; deciduous forest is seasonal."""
    lai = np.array(
        [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]], dtype=np.float32
    )  # (time, cells)
    evergreen_fraction = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    adjusted = get_forest_leaf_area_index_with_evergreen(lai, evergreen_fraction)
    assert adjusted.dtype == np.float32
    np.testing.assert_allclose(adjusted, [[1.0, 3.0, 5.0], [5.0, 5.0, 5.0]], rtol=1e-6)


def test_get_surface_crust_factor() -> None:
    """No crust under forest, crust in proportion to exposed soil elsewhere."""
    from geb.hydrology.landcovers import FOREST, GRASSLAND_LIKE, SEALED
    from geb.hydrology.landsurface.water import get_surface_crust_factor

    lai = np.float32(2.0)
    developed = np.float32(10.0)  # m of rain on exposed soil: fully developed crust
    assert get_surface_crust_factor(
        np.int32(FOREST),
        lai,
        developed,
        bare_soil_crust_factor=np.float32(0.2),
        crust_formation_rain_m=np.float32(0.05),
    ) == np.float32(1.0)
    # sealed land: the grass of the unpaved 65 % protects its soil
    assert get_surface_crust_factor(
        np.int32(SEALED),
        np.float32(0.65) * lai,
        developed,
        bare_soil_crust_factor=np.float32(0.2),
        crust_formation_rain_m=np.float32(0.05),
    ) == pytest.approx(0.2 + 0.8 * (1 - np.exp(-1.4)), rel=1e-5)
    assert get_surface_crust_factor(
        np.int32(GRASSLAND_LIKE),
        np.float32(0.0),
        developed,
        bare_soil_crust_factor=np.float32(0.2),
        crust_formation_rain_m=np.float32(0.05),
    ) == pytest.approx(0.2)
    assert get_surface_crust_factor(
        np.int32(GRASSLAND_LIKE),
        lai,
        developed,
        bare_soil_crust_factor=np.float32(0.2),
        crust_formation_rain_m=np.float32(0.05),
    ) == pytest.approx(0.2 + 0.8 * (1 - np.exp(-1.4)), rel=1e-5)


def test_surface_crust_develops_with_rain_on_exposed_soil() -> None:
    """Loose bare soil has no crust; the crust develops with rain on the exposed soil."""
    from geb.hydrology.landcovers import GRASSLAND_LIKE
    from geb.hydrology.landsurface.water import get_surface_crust_factor

    bare = np.float32(0.0)
    # freshly loosened by frost or tillage: no crust
    assert get_surface_crust_factor(
        np.int32(GRASSLAND_LIKE),
        bare,
        np.float32(0.0),
        bare_soil_crust_factor=np.float32(0.2),
        crust_formation_rain_m=np.float32(0.05),
    ) == pytest.approx(1.0)
    # after the formation rain (50 mm) the crust is 63 % developed
    assert get_surface_crust_factor(
        np.int32(GRASSLAND_LIKE),
        bare,
        np.float32(0.05),
        bare_soil_crust_factor=np.float32(0.2),
        crust_formation_rain_m=np.float32(0.05),
    ) == pytest.approx(0.2 + 0.8 * np.exp(-1.0), rel=1e-5)
    # more rain, lower conductivity
    factors = [
        get_surface_crust_factor(
            np.int32(GRASSLAND_LIKE),
            bare,
            np.float32(rain),
            bare_soil_crust_factor=np.float32(0.2),
            crust_formation_rain_m=np.float32(0.05),
        )
        for rain in (0.0, 0.02, 0.05, 0.2)
    ]
    assert all(a > b for a, b in zip(factors, factors[1:], strict=False))


def test_limit_evaporation_under_snow() -> None:
    """Snow stops soil evaporation everywhere; only forest crowns keep transpiring."""
    from geb.hydrology.landcovers import FOREST, GRASSLAND_LIKE, NON_PADDY_IRRIGATED

    evaporation, transpiration = np.float32(0.2), np.float32(0.3)
    # no snow: unchanged
    assert limit_evaporation_under_snow(
        np.int32(GRASSLAND_LIKE), np.float32(0.0), evaporation, transpiration
    ) == (evaporation, transpiration)
    # snow on grass and crops: nothing
    for land_use in (GRASSLAND_LIKE, NON_PADDY_IRRIGATED):
        assert limit_evaporation_under_snow(
            np.int32(land_use), np.float32(0.05), evaporation, transpiration
        ) == (0.0, 0.0)
    # snow in forest: no soil evaporation, transpiration continues
    assert limit_evaporation_under_snow(
        np.int32(FOREST), np.float32(0.05), evaporation, transpiration
    ) == (0.0, transpiration)


def test_accumulate_crop_evapotranspiration_skips_unknown_stage() -> None:
    """Per-stage accumulators are only updated for known crop sub-stages."""
    from geb.hydrology.landsurface.evapotranspiration import (
        accumulate_crop_evapotranspiration,
    )

    crop_map = np.array([0, 0, -1], dtype=np.int32)
    crop_sub_stage = np.array([2, -1, -1], dtype=np.int8)
    actual = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    potential = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    actual_life = np.zeros(3, dtype=np.float32)
    potential_life = np.zeros(3, dtype=np.float32)
    actual_stage = np.zeros((6, 3), dtype=np.float32)
    potential_stage = np.zeros((6, 3), dtype=np.float32)

    accumulate_crop_evapotranspiration(
        crop_map,
        crop_sub_stage,
        actual,
        potential,
        actual_life,
        potential_life,
        actual_stage,
        potential_stage,
    )

    np.testing.assert_array_equal(actual_life, [1.0, 2.0, 0.0])
    np.testing.assert_array_equal(potential_life, [2.0, 3.0, 0.0])
    expected_actual_stage = np.zeros((6, 3), dtype=np.float32)
    expected_actual_stage[2, 0] = 1.0
    np.testing.assert_array_equal(actual_stage, expected_actual_stage)
    # stage -1 must not write into the last stage row
    assert potential_stage[5].sum() == 0
    assert potential_stage[2, 0] == 2.0


def test_resolve_crop_group_number_rainfed_crop() -> None:
    """A crop sets the crop group number, also on grassland-like (rainfed) land."""
    from geb.hydrology.landcovers import (
        FOREST,
        GRASSLAND_LIKE,
        NON_PADDY_IRRIGATED,
    )
    from geb.hydrology.landsurface.evapotranspiration import (
        resolve_crop_group_number,
    )

    crop_group_numbers = resolve_crop_group_number(
        crop_map=np.array([1, 1, -1, -1], dtype=np.int32),
        land_use_type=np.array(
            [GRASSLAND_LIKE, NON_PADDY_IRRIGATED, GRASSLAND_LIKE, FOREST],
            dtype=np.int32,
        ),
        crop_group_number_forest=np.full(4, 5.0, dtype=np.float32),
        crop_group_number_grassland_like=np.full(4, 4.0, dtype=np.float32),
        crop_group_number_per_group=np.array([1.0, 2.0], dtype=np.float32),
        forest_land_use=FOREST,
        grassland_like_land_use=GRASSLAND_LIKE,
    )
    np.testing.assert_array_equal(crop_group_numbers, [2.0, 2.0, 4.0, 5.0])


def test_get_dew_from_negative_wet_surface_evaporation() -> None:
    """Condensation (negative evaporation) is dew; none on snow or with evaporation."""
    assert get_dew(np.float32(-5e-5), np.float64(0.0)) == pytest.approx(5e-5)
    assert get_dew(np.float32(2e-4), np.float64(0.0)) == 0.0
    assert get_dew(np.float32(-5e-5), np.float64(0.01)) == 0.0


def test_clear_humid_night_gives_dew() -> None:
    """A clear, calm, humid night gives condensation at a realistic dew rate."""
    from geb.hydrology.landsurface.potential_evapotranspiration import (
        get_hourly_weather,
        get_reference_evapotranspiration,
    )

    _, wet_vegetation_m, _ = get_reference_evapotranspiration(
        get_hourly_weather(
            np.float32(10.0),
            np.float32(9.5),
            np.float32(100000.0),
            np.float32(0.0),
            np.float32(290.0),
            np.float32(1.0),
        ),
        np.float32(0.0),
    )
    dew_mm_per_hour = get_dew(wet_vegetation_m, np.float64(0.0)) * 1000
    # observed dew rates are at most about 0.05-0.07 mm/hour
    assert 0.0 < dew_mm_per_hour < 0.07


def test_get_soil_heat_flux_net_of_evaporation() -> None:
    """1 mm/hour of soil evaporation removes about 695 W/m2 of latent heat."""
    flux = get_soil_heat_flux_net_of_evaporation(np.float32(50.0), np.float32(0.0))
    assert flux == pytest.approx(50.0)
    flux = get_soil_heat_flux_net_of_evaporation(np.float32(50.0), np.float32(1e-4))
    assert flux == pytest.approx(50.0 - 69.5, abs=0.5)
