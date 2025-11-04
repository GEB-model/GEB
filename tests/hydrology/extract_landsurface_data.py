#!/usr/bin/env python3
"""Utility script to extract hardcoded data from landsurface_model_error.npz for testing.

This script loads an NPZ file containing error case data and generates Python code
that can be pasted into test functions for reproducing water balance errors.

Usage:
    python extract_landsurface_data.py <npz_path> <cell_idx>

Example:
    python extract_landsurface_data.py /path/to/landsurface_model_error.npz 1675495

The output can be pasted directly into a test function in test_landsurface.py.
"""

import sys
from pathlib import Path

import numpy as np


def extract_landsurface_error_data(npz_path: str, cell_idx: int) -> str:
    """Extract hardcoded data from landsurface_model_error.npz for a specific cell.

    This utility function loads the NPZ file, extracts data for the specified cell,
    and generates Python code that can be pasted into a test function.

    Args:
        npz_path: Path to the NPZ file containing the error case data
        cell_idx: Index of the cell to extract data for

    Returns:
        String containing the Python code with hardcoded array definitions
    """
    # Load the data
    data = np.load(npz_path)

    # Extract data for the specific cell
    land_use_type_data = data["land_use_type"][cell_idx]
    root_depth_m_data = data["root_depth_m"][cell_idx]
    topwater_m_data = data["topwater_m"][cell_idx]
    snow_water_equivalent_m_data = data["snow_water_equivalent_m"][cell_idx]
    liquid_water_in_snow_m_data = data["liquid_water_in_snow_m"][cell_idx]
    snow_temperature_C_data = data["snow_temperature_C"][cell_idx]
    interception_storage_m_data = data["interception_storage_m"][cell_idx]
    interception_capacity_m_data = data["interception_capacity_m"][cell_idx]
    crop_factor_data = data["crop_factor"][cell_idx]
    crop_map_data = data["crop_map"][cell_idx]
    actual_irrigation_consumption_m_data = data["actual_irrigation_consumption_m"][
        cell_idx
    ]
    capillar_rise_m_data = data["capillar_rise_m"][cell_idx]
    frost_index_data = data["frost_index"][cell_idx]
    natural_crop_groups_data = data["natural_crop_groups"][cell_idx]

    # Extract arrays
    w_data = data["w"][:, cell_idx]
    wres_data = data["wres"][:, cell_idx]
    wwp_data = data["wwp"][:, cell_idx]
    wfc_data = data["wfc"][:, cell_idx]
    ws_data = data["ws"][:, cell_idx]
    delta_z_data = data["delta_z"][:, cell_idx]
    saturated_hydraulic_conductivity_m_per_s_data = data[
        "saturated_hydraulic_conductivity_m_per_s"
    ][:, cell_idx]
    lambda_pore_size_distribution_data = data["lambda_pore_size_distribution"][
        :, cell_idx
    ]
    bubbing_pressure_cm_data = data["bubbing_pressure_cm"][:, cell_idx]

    # Time series data
    pr_kg_per_m2_per_s_data = data["pr_kg_per_m2_per_s"][:, cell_idx]
    tas_2m_K_data = data["tas_2m_K"][:, cell_idx]
    dewpoint_tas_2m_K_data = data["dewpoint_tas_2m_K"][:, cell_idx]
    ps_pascal_data = data["ps_pascal"][:, cell_idx]
    rlds_W_per_m2_data = data["rlds_W_per_m2"][:, cell_idx]
    rsds_W_per_m2_data = data["rsds_W_per_m2"][:, cell_idx]
    wind_u10m_m_per_s_data = data["wind_u10m_m_per_s"][:, cell_idx]
    wind_v10m_m_per_s_data = data["wind_v10m_m_per_s"][:, cell_idx]

    # Scalars
    CO2_ppm_data = data["CO2_ppm"]
    minimum_effective_root_depth_m_data = data["minimum_effective_root_depth_m"]
    soil_layer_height_data = data["soil_layer_height"][:, cell_idx]
    crop_group_number_per_group_data = data["crop_group_number_per_group"]

    # Generate the Python code
    code_lines = []
    code_lines.append(f"# Hardcoded test data for cell {cell_idx}")
    code_lines.append(f"# Generated from {npz_path}")
    code_lines.append(
        f"# To regenerate: run extraction script with cell_idx = {cell_idx}"
    )
    code_lines.append("")

    # Scalars
    code_lines.append(f"land_use_type_data = np.int32({land_use_type_data})")
    code_lines.append(f"root_depth_m_data = np.float32({root_depth_m_data})")
    code_lines.append(f"topwater_m_data = np.float32({topwater_m_data})")
    code_lines.append(
        f"snow_water_equivalent_m_data = np.float32({snow_water_equivalent_m_data})"
    )
    code_lines.append(
        f"liquid_water_in_snow_m_data = np.float32({liquid_water_in_snow_m_data})"
    )
    code_lines.append(
        f"snow_temperature_C_data = np.float32({snow_temperature_C_data})"
    )
    code_lines.append(
        f"interception_storage_m_data = np.float32({interception_storage_m_data})"
    )
    code_lines.append(
        f"interception_capacity_m_data = np.float32({interception_capacity_m_data})"
    )
    code_lines.append(f"crop_factor_data = np.float32({crop_factor_data})")
    code_lines.append(f"crop_map_data = np.int32({crop_map_data})")
    code_lines.append(
        f"actual_irrigation_consumption_m_data = np.float32({actual_irrigation_consumption_m_data})"
    )
    code_lines.append(f"capillar_rise_m_data = np.float32({capillar_rise_m_data})")
    code_lines.append(f"frost_index_data = np.float32({frost_index_data})")
    code_lines.append(
        f"natural_crop_groups_data = np.float32({natural_crop_groups_data})"
    )

    # Arrays
    code_lines.append(f"w_data = np.array({w_data.tolist()}, dtype=np.float32)")
    code_lines.append(f"wres_data = np.array({wres_data.tolist()}, dtype=np.float32)")
    code_lines.append(f"wwp_data = np.array({wwp_data.tolist()}, dtype=np.float32)")
    code_lines.append(f"wfc_data = np.array({wfc_data.tolist()}, dtype=np.float32)")
    code_lines.append(f"ws_data = np.array({ws_data.tolist()}, dtype=np.float32)")
    code_lines.append(
        f"delta_z_data = np.array({delta_z_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"saturated_hydraulic_conductivity_m_per_s_data = np.array({saturated_hydraulic_conductivity_m_per_s_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"lambda_pore_size_distribution_data = np.array({lambda_pore_size_distribution_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"bubbing_pressure_cm_data = np.array({bubbing_pressure_cm_data.tolist()}, dtype=np.float32)"
    )

    # Time series
    code_lines.append(
        f"pr_kg_per_m2_per_s_data = np.array({pr_kg_per_m2_per_s_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"tas_2m_K_data = np.array({tas_2m_K_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"dewpoint_tas_2m_K_data = np.array({dewpoint_tas_2m_K_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"ps_pascal_data = np.array({ps_pascal_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"rlds_W_per_m2_data = np.array({rlds_W_per_m2_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"rsds_W_per_m2_data = np.array({rsds_W_per_m2_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"wind_u10m_m_per_s_data = np.array({wind_u10m_m_per_s_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"wind_v10m_m_per_s_data = np.array({wind_v10m_m_per_s_data.tolist()}, dtype=np.float32)"
    )

    # Scalars
    code_lines.append(f"CO2_ppm_data = np.float32({CO2_ppm_data})")
    code_lines.append(
        f"minimum_effective_root_depth_m_data = np.float32({minimum_effective_root_depth_m_data})"
    )
    code_lines.append(
        f"soil_layer_height_data = np.array({soil_layer_height_data.tolist()}, dtype=np.float32)"
    )
    code_lines.append(
        f"crop_group_number_per_group_data = np.array({crop_group_number_per_group_data.tolist()}, dtype=np.float32)"
    )

    return "\n".join(code_lines)


def main() -> None:
    """Main function to run the extraction script from command line."""
    if len(sys.argv) != 3:
        print("Usage: python extract_landsurface_data.py <npz_path> <cell_idx>")
        print(
            "Example: python extract_landsurface_data.py /path/to/landsurface_model_error.npz 1675495"
        )
        sys.exit(1)

    npz_path = sys.argv[1]
    cell_idx = int(sys.argv[2])

    if not Path(npz_path).exists():
        print(f"Error: NPZ file '{npz_path}' does not exist")
        sys.exit(1)

    try:
        code = extract_landsurface_error_data(npz_path, cell_idx)
        print(code)
    except Exception as e:
        print(f"Error extracting data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
