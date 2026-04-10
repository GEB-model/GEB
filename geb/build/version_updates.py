"""This module contains the version updates for the GEB model. It is used to keep track of the changes that need to be made when updating the model to a new version. The VERSION_UPDATES dictionary contains the version as the key and a list of updates as the value. Each update is a string that describes the change that needs to be made. This module is imported in the build module and used to display the updates when running `geb update-version`."""

import re
import sys
from typing import TYPE_CHECKING, Any

from packaging.version import Version

from geb import __version__

if TYPE_CHECKING:
    from geb.build import GEBModel as GEBModelBuild

VERSION_UPDATES: dict[str, list[str]] = {
    "1.0.0b20": ["[update-python;3.14.4]"],
    "1.0.0b19": [
        "[create-file;build_complete.txt]",
        "[update-method;setup_hydrography]",
        "[manual] Add `setup_global_flood_damage_model` and `setup_geul_flood_damage_model` to your `build.yml`.",
        "[manual] Run `setup_global_flood_damage_model`: `geb update -b build.yml::setup_global_flood_damage_model`.",
        "[manual] Run `setup_geul_flood_damage_model`: `geb update -b build.yml::setup_geul_flood_damage_model`.",
    ],
    "1.0.0b18": [
        "[update-method;setup_forcing]",
        "[update-method;setup_SPEI]",
        "[update-method;setup_pr_GEV]",
        "[update-method;setup_buildings]",
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


def get_and_maybe_do_version_updates(
    version_info: str,
    perform_auto_update: bool = False,
    build_model: GEBModelBuild | None = None,
    methods: dict[str, Any] | None = None,
) -> list[str]:
    """Get the version updates that need to be made to update from the stored version to the current version.

    Args:
        version_info: The version string stored in the version file, e.g. "1.2.3".
        perform_auto_update: Whether to perform auto updates.
        build_model: The GEB model instance for building. Must be provided if perform_auto_update is True.
        methods: A dictionary of loaded methods from the build configuration. Must be provided if perform_auto_update is True.

    Returns:
        A list of strings describing the updates that need to be made to update to the current version.

    Raises:
        ValueError: If the version update text is not in the expected format.
    """
    if perform_auto_update and build_model is None:
        raise ValueError("build_model must be provided if perform_auto_update is True")

    if perform_auto_update and methods is None:
        raise ValueError("methods must be provided if perform_auto_update is True")

    current_v = Version(__version__)
    stored_v = Version(version_info)

    versions: list[str] = sorted(
        VERSION_UPDATES.keys(), key=Version
    )  # iterate from oldest to newest version
    updates_to_print: list[str] = []
    for v_str in versions:
        v = Version(v_str)
        if v > stored_v and v <= current_v:
            version_updates: list[str] = VERSION_UPDATES[v_str]
            for version_update in version_updates:
                match: re.Match[str] | None = re.search(r"^\[(.*?)\]", version_update)
                if match is None:
                    raise ValueError(
                        f"Version update text should start with the update type in square brackets, e.g. [update-python], but got: {version_update}"
                    )
                update_type, *update_type_arguments = match.group(1).split(";")
                if update_type == "update-python":
                    if len(update_type_arguments) != 1:
                        raise ValueError(
                            f"update-python update type should have exactly one argument, the python version to update to, but got: {update_type_arguments}"
                        )
                    python_version: str = update_type_arguments[0]
                    # if current version is lower than the required python version, we need to update python
                    if Version(python_version) > Version(
                        ".".join(map(str, sys.version_info[:3]))
                    ):
                        updates_to_print.append(
                            f"Update to Python {python_version}. If you use uv, ensure your uv is updated first: `uv self update`. Then use `uv sync`."
                        )

                elif update_type == "update-method":
                    if len(update_type_arguments) != 1:
                        raise ValueError(
                            f"update-method update type should have exactly one argument, the method to update, but got: {update_type_arguments}"
                        )
                    method_name: str = update_type_arguments[0]
                    if perform_auto_update:
                        assert methods is not None
                        assert build_model is not None
                        update_method = getattr(build_model, method_name, None)
                        if update_method is None:
                            raise ValueError(
                                f"Method {method_name} not found in geb.cli.update module"
                            )

                        build_model.logger.info(
                            f"Performing auto-update for method {method_name}..."
                        )

                        update_method(**(methods[method_name] or {}))
                    else:
                        updates_to_print.append(
                            f"Re-run `{method_name}`: `geb update -b build.yml::{method_name}`."
                        )

                elif update_type == "create-file":
                    if len(update_type_arguments) != 1:
                        raise ValueError(
                            f"create-file update type should have exactly one argument, the file to create, but got: {update_type_arguments}"
                        )
                    file_path: str = update_type_arguments[0]
                    if perform_auto_update:
                        assert build_model is not None
                        build_model.logger.info(
                            f"Creating file {file_path} as part of auto-update..."
                        )
                        full_file_path = build_model.root / file_path
                        full_file_path.parent.mkdir(parents=True, exist_ok=True)
                        full_file_path.touch(exist_ok=True)
                    else:
                        updates_to_print.append(
                            f"Add a new file called '{file_path}' in your input folder. In future versions this file will be made automatically."
                        )

                elif update_type == "manual":
                    if len(update_type_arguments) != 0:
                        raise ValueError(
                            f"manual update type should have no arguments, but got: {update_type_arguments}"
                        )
                    updates_to_print.append(
                        version_update.replace(f"[{match.group(1)}]", "").strip()
                    )

                else:
                    raise ValueError(f"Unknown update type: {update_type}")

    return updates_to_print
