"""This module contains the version updates for the GEB model. It is used to keep track of the changes that need to be made when updating the model to a new version. The VERSION_UPDATES dictionary contains the version as the key and a list of updates as the value. Each update is a string that describes the change that needs to be made. This module is imported in the build module and used to display the updates when running `geb update-version`."""

VERSION_UPDATES: dict[str, list[str]] = {
    "1.0.0b18": [
        "[manual] Re-run `setup_forcing`: `geb update -b build.yml::setup_forcing`.",
        "[manual] Re-run `setup_SPEI`: `geb update -b build.yml::setup_SPEI`.",
        "[manual] Re-run `setup_pr_GEV`: `geb update -b build.yml::setup_pr_GEV`.",
        "[manual] Re-run `setup_buildings`: `geb update -b build.yml::setup_buildings`.",
    ],
    "1.0.0b12": [
        "[manual] Re-run `setup_hydrography`: `geb update -b build.yml::setup_hydrography`.",
        "[manual] Re-name `setup_mannings` to `setup_geomorphology` and run `setup_geomorphology`: `geb update -b build.yml::setup_geomorphology`.",
        "[manual] Re-run `setup_discharge_observations`: `geb update -b build.yml::setup_discharge_observations`.",
        "[manual] Only in case of build errors (or later in spinup/run): re-run `setup_household_characteristics`, `setup_crops`, `setup_income_distribution_parameters`, and `setup_create_farms` using `geb update -b build.yml::<method>`.",
        "[manual] Optional but recommended: Re-run `setup_forcing` and `setup_SPEI` for a significant speedup and better SPEI estimation: `geb update -b build.yml::setup_forcing` and `geb update -b build.yml::setup_SPEI`.",
    ],
    "1.0.0b11": [
        "[manual] Rename `setup_soil_parameters` to `setup_soil` in `build.yml`.",
        "[manual] Re-run `setup_soil` and `setup_household_characteristics`.",
        "[manual] Re-run `setup_coastal_sfincs_model_regions`.",
        "[manual] Remove `setup_low_elevation_coastal_zone_mask` from your `build.yml`.",
        "[manual] Add `setup_buildings` to your `build.yml`.",
        "[manual] Models for inland regions need to be rebuild if floods need to be run.",
        "[manual] Re-run `setup_gtsm_station_data` and `setup_gtsm_water_levels`.",
        "[manual] Re-run `setup_buildings`.",
        "[manual] Setup cdsapi for gtsm download (see Copernicus instructions).",
        "[manual] Rename `setup_crops_from_source` to `setup_crops` and use `source_type` rather than `type`.",
        "[manual] Add and run `setup_vegetation` to `build.yml` (e.g., after `setup_soil`).",
        "[manual] Run `uv sync` to update `damagescanner`.",
    ],
}
