"""Runner module for GEB functions."""

import cProfile
import importlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from operator import attrgetter
from pathlib import Path
from pstats import Stats
from typing import Any

import geopandas as gpd
import yaml
from shapely.geometry import box

from geb import GEB_PACKAGE_DIR
from geb.build import GEBModel as GEBModelBuild
from geb.build.methods import build_method
from geb.model import GEBModel
from geb.workflows.io import WorkingDirectory, read_params, write_params
from geb.workflows.methods import multi_level_merge

PROFILING_DEFAULT: bool = False
OPTIMIZE_DEFAULT: bool = False
TIMING_DEFAULT: bool = False
WORKING_DIRECTORY_DEFAULT: Path = Path(".")
CONFIG_DEFAULT: Path = Path("model.yml")
UPDATE_DEFAULT: Path = Path("update.yml")
BUILD_DEFAULT: Path = Path("build.yml")

DATA_CATALOG_DEFAULT: Path = GEB_PACKAGE_DIR / "data_catalog.yml"
DATA_PROVIDER_DEFAULT: str = os.environ.get("GEB_DATA_PROVIDER", "default")
DATA_ROOT_DEFAULT: Path = Path(
    os.environ.get(
        "GEB_DATA_ROOT",
        GEB_PACKAGE_DIR.parent.parent / "data_catalog",
    )
)
ALTER_FROM_MODEL_DEFAULT: Path = Path("../base")


class DetectDuplicateKeysYamlLoader(yaml.SafeLoader):
    """Custom YAML loader that detects duplicate keys in mappings.

    Raises:
        ValueError: If a duplicate key is found in the YAML mapping.
    """

    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict:
        """Construct a mapping from a YAML node, checking for duplicate keys.

        Args:
            node: The YAML node to construct the mapping from.
            deep: Whether to perform a deep construction of the mapping. Defaults to False.

        Raises:
            ValueError: If a duplicate key is found in the YAML mapping.

        Returns:
            dict: The constructed mapping.
        """
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate key found: {key}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def parse_config(
    config_path: dict | Path | str, current_directory: Path | None = None
) -> dict[str, Any]:
    """Parse config.

    This method recursively parses the config file and resolves any 'inherits' keys.

    Args:
        config_path: Path to the config file or a dict with the config.
        current_directory: Current directory to resolve relative paths.
            If None, the current working directory is used.

    Returns:
        Full model configuation of the model without any remaining 'inherits' keys.
    """
    if current_directory is None:
        current_directory = Path.cwd()

    if isinstance(config_path, dict):
        config = config_path
    else:
        config: dict | None = yaml.load(
            open(current_directory / config_path, "r"),
            Loader=DetectDuplicateKeysYamlLoader,
        )
        if config is None:
            config = {}
        current_directory = current_directory / Path(config_path).parent

    if "inherits" in config:
        inherit_config_path = config["inherits"]
        inherit_config_path = inherit_config_path.format(**os.environ)
        # replace {VAR} with environment variable VAR if it exists
        inherit_config_path = os.path.expandvars(inherit_config_path)
        # if inherits is not an absolute path, we assume it is relative to the config file
        if not Path(inherit_config_path).is_absolute():
            inherit_config_path = current_directory / config["inherits"]
        inherited_config = yaml.load(
            open(inherit_config_path, "r"),
            Loader=yaml.FullLoader,
        )
        current_directory = current_directory / Path(inherit_config_path).parent
        del config[
            "inherits"
        ]  # remove inherits key from config to avoid infinite recursion
        config = multi_level_merge(inherited_config, config)
        config = parse_config(config, current_directory=current_directory)

    # Validate config
    from pydantic import ValidationError

    from geb.config_schema import Config

    try:
        Config(**config)
    except ValidationError as e:
        # We warn instead of raising an error to allow for extra fields or partial configs
        # during development, but ideally this should be strict.
        logging.warning(f"Configuration validation failed: {e}")

    return config


def create_logger(fp: Path) -> logging.Logger:
    """Create logger with console and file handler.

    Args:
        fp: Path to the log file.
    Returns:
        Logger instance.
    """
    logger = logging.getLogger("GEB")
    # remove any previous handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # set log level to debug
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    # add file handler
    Path(fp).parent.mkdir(exist_ok=True, parents=True)
    fh = logging.FileHandler(fp)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def run_model_with_method(
    method: str | None,
    config: dict | str | Path = CONFIG_DEFAULT,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    timing: bool = TIMING_DEFAULT,
    profiling: bool = PROFILING_DEFAULT,
    optimize: bool = OPTIMIZE_DEFAULT,
    method_args: dict = {},
    close_after_run: bool = True,
) -> GEBModel:
    """Run model with a specific method.

    Args:
        method: Method to run on the model. If None, the model is created but no method is run.
        config: Path to the model configuration file or a dict with the config.
        working_directory: Working directory for the model.
        timing: If True, run the model with timing, printing the time taken for specific methods.
        profiling: If True, run the model with profiling.
        optimize: If True, run the model in optimized mode, skipping asserts and water balance checks.
        method_args: Optional arguments to pass to the method.
        close_after_run: If True, close the model after running the method. Defaults to True.

    Returns:
        Instance of GEBModel

    Raises:
        SystemExit: If the model is restarted in optimized mode.
    """
    # check if we need to run the model in optimized mode
    # if the model is already running in optimized mode, we don't need to restart it
    # or else we start an infinite loop
    if optimize and sys.flags.optimize == 0:
        # If the script is not a .py file, we need to add the .exe extension
        if platform.system() == "Windows" and not sys.argv[0].endswith(".py"):
            sys.argv[0] = sys.argv[0] + ".exe"
        command: list[str] = [sys.executable, "-O"] + sys.argv
        raise SystemExit(subprocess.run(command).returncode)

    with WorkingDirectory(working_directory):
        config: dict[str, Any] = parse_config(config)

        # TODO: This can be removed in 2026
        if not Path("input/files.yml").exists() and Path("input/files.json").exists():
            # convert input/files.json to input/files.yml
            json_files: dict[str, Any] = read_params(
                Path("input/files.json"),
            )
            write_params(json_files, Path("input/files.yml"))

        files: dict[str, Any] = parse_config(
            read_params(Path("input/files.yml"))
            if "files" not in config["general"]
            else config["general"]["files"]
        )

        if profiling:
            profile = cProfile.Profile()
            profile.enable()

        geb = GEBModel(config=config, files=files, timing=timing)
        if method is not None:
            getattr(geb, method)(**method_args)
        if close_after_run:
            geb.close()

        if profiling:
            profile.disable()
            with open("profiling_stats.cprof", "w") as stream:
                stats = Stats(profile, stream=stream)
                stats.strip_dirs()
                stats.sort_stats("cumtime")
                stats.dump_stats(".prof_stats")
                stats.print_stats()
            profile.dump_stats("profile.prof")

        return geb


def get_model_builder_class(custom_model: None | str) -> type:
    """Get model builder class.

    This is usually the GEBModelBuild class, but can be a custom model builder class
    from the geb.build.custom_models module. This class would usually
    specify some custom build methods, but largely re-use the existing GEBModelBuild methods.

    Args:
        custom_model: Name of the custom model to use. If None, the default GEBModelBuild is used.
            custom_models are available in the geb.build.custom_models module.

    Returns:
        Model builder class.

    Raises:
        ValueError: If the custom model is not found in the geb.build.custom_models module.
    """
    if custom_model is None:
        return GEBModelBuild
    else:
        from geb import build as geb_build

        importlib.import_module(
            "." + custom_model.split(".")[0], package="geb.build.custom_models"
        )
        if not hasattr(geb_build, "custom_models"):
            raise ValueError("Custom models module not found")
        return attrgetter(custom_model)(geb_build.custom_models)


def customize_data_catalog(data_catalog: Path, data_root: None | Path = None) -> Path:
    """This functions adds the GEB_DATA_ROOT to the data catalog if it is set as an environment variable.

    This enables reading the data catalog from a different location than the location of the yml-file
    without the need to specify root in the meta of the data catalog.

    Args:
        data_catalog: List of paths to data catalog yml files.
        data_root: Root folder where the data is located. If None, the data catalog is not modified.

    Returns:
        List of paths to data catalog yml files, possibly modified to include the data_root.
    """
    if data_root:
        with open(data_catalog, "r") as stream:
            data_catalog_yml = yaml.load(stream, Loader=yaml.FullLoader)

            if "meta" not in data_catalog_yml:
                data_catalog_yml["meta"] = {}
            data_catalog_yml["meta"]["root"] = str(data_root)

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
            yaml.dump(data_catalog_yml, tmp, default_flow_style=False)
        return Path(tmp.name)
    else:
        return data_catalog


def get_builder(
    config: Path | dict[str, Any],
    data_catalog: Path,
    custom_model: str | None,
    data_provider: str | None,
    data_root: Path | None,
) -> GEBModelBuild:
    """Get model builder.

    Args:
        config: Path to the model configuration file.
        data_catalog: Path to the data catalog file.
        custom_model: Name of the custom model to use. If None, the default GEBModelBuild is used.
            custom_models are available in the geb.build.custom_models module.
        data_provider: Data variant to use from data catalog (see hydroMT documentation).
        data_root: Root folder where the data is located. If None, the data catalog is not modified.

    Returns:
        Instance of the model builder.
    """
    config = parse_config(config)
    input_folder = Path(config["general"]["input_folder"])

    data_catalog = customize_data_catalog(data_catalog, data_root=data_root)

    arguments = {
        "root": input_folder,
        "data_catalog": data_catalog,
        "logger": create_logger(Path("build.log")),
        "data_provider": data_provider,
    }

    builder_class = get_model_builder_class(custom_model)(**arguments)
    build_method.validate_tree()
    return builder_class


def init_fn(
    config: str | Path,
    build_config: str | Path,
    update_config: str | Path,
    working_directory: str | Path,
    from_example: str,
    basin_id: str | None = None,
    ISO3: str | None = None,
    overwrite: bool = False,
) -> None:
    """Create a new model.

    Args:
        config: Path to the model configuration file to create.
        build_config: Path to the model build configuration file to create.
        update_config: Path to the model update configuration file to create.
        working_directory: Working directory for the model.
        from_example: Name of the example to use as a base for the model.
        basin_id: Basin ID(s) to use for the model. Can be a comma-separated list of integers.
            If not set, the basin ID is taken from the config file.
        ISO3: ISO3 country code to use for the model. Cannot be used together with --basin-id.
        overwrite: If True, overwrite existing config and build config files. Defaults to False.

    Raises:
        FileExistsError: If the config or build config file already exists and overwrite is False.
        FileNotFoundError: If the example folder does not exist.
        ValueError: If both basin_id and ISO3 are set.

    """
    if basin_id is not None and ISO3 is not None:
        raise ValueError("Cannot use --basin-id and --ISO3 together.")

    config: Path = Path(config)
    build_config: Path = Path(build_config)
    update_config: Path = Path(update_config)
    working_directory: Path = Path(working_directory)

    if not working_directory.exists():
        working_directory.mkdir(parents=True, exist_ok=True)

    with WorkingDirectory(working_directory):
        if config.exists() and not overwrite:
            raise FileExistsError(
                f"Config file {config} already exists. Please remove it or use a different name, or use --overwrite."
            )

        if build_config.exists() and not overwrite:
            raise FileExistsError(
                f"Build config file {build_config} already exists. Please remove it or use a different name, or use --overwrite."
            )

        if update_config.exists() and not overwrite:
            raise FileExistsError(
                f"Update config file {update_config} already exists. Please remove it or use a different name, or use --overwrite."
            )

        example_folder: Path = GEB_PACKAGE_DIR / "examples" / from_example
        if not example_folder.exists():
            raise FileNotFoundError(
                f"Example folder {example_folder} does not exist. Did you use the right --from-example option?"
            )

        config_dict: dict = yaml.load(
            open(example_folder / CONFIG_DEFAULT, "r"),
            Loader=DetectDuplicateKeysYamlLoader,
        )

        if basin_id is not None:
            # Allow passing a comma-separated list of integers
            if "," in basin_id:
                basin_ids: list[int] = [
                    int(x) for x in basin_id.split(",") if x.strip()
                ]
            else:
                basin_ids: int = int(basin_id)

            config_dict["general"]["region"]["subbasin"] = basin_ids
        elif ISO3 is not None:
            del config_dict["general"]["region"]["subbasin"]
            config_dict["general"]["region"] = {
                "geom": {
                    "source": "GADM_level1",
                    "key": ISO3,
                    "column": "GID_0",
                }
            }

        with open(config, "w") as f:
            # do not sort keys, to keep the order of the config file
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        shutil.copy(example_folder / BUILD_DEFAULT, build_config)
        shutil.copy(example_folder / UPDATE_DEFAULT, update_config)


def set_fn(
    config: Path,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    **kwargs: Any,
) -> None:
    """Set model configuration values by updating a YAML configuration file.

    This function loads the existing configuration from the specified file,
    updates it with the provided keyword arguments (supporting nested keys
    using dot notation, e.g., 'section.subsection.key'), and saves the
    modified configuration back to the file.

        config: Path to the model configuration file (as a string or Path object).
        working_directory: Working directory for the model.
        **kwargs: Keyword arguments representing keys and values to set in the config.
                  Keys can be nested using dots (e.g., 'model.lr' sets 'lr' under 'model').

    Note:
        If a nested key does not exist, intermediate dictionaries are created automatically.
        The file is overwritten with the updated configuration in YAML format.

    Args:
        config: Path to the model configuration file.
        working_directory: Working directory for the model.
        **kwargs: Keyword arguments to set in the config file.

    Raises:
        KeyError: If a specified key does not exist in the config and cannot be created.
    """
    with WorkingDirectory(working_directory):
        config_dict: dict[str, Any] = parse_config(config)
        for key, value in kwargs.items():
            if key.endswith("+"):
                key: str = key[:-1]
                create: bool = True
            else:
                create: bool = False

            keys: list[str] = key.split(".")
            d = config_dict

            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    if create:
                        d[k] = {}
                    else:
                        raise KeyError(
                            f"Key '{k}' not found in config. If you want to create it, use the '+' suffix for the KEY."
                        )
                d: dict[str, Any] = d[k]

            if value == "null":
                value = None
            elif value == "true":
                value = True
            elif value == "false":
                value = False

            if not create and keys[-1] not in d:
                raise KeyError(
                    f"Key '{keys[-1]}' not found in config. If you want to create it, use the '+' suffix for the KEY."
                )
            d[keys[-1]] = value

        with open(config, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def build_fn(
    data_catalog: Path = DATA_CATALOG_DEFAULT,
    config: Path | dict[str, Any] = CONFIG_DEFAULT,
    build_config: Path | dict[str, Any] = BUILD_DEFAULT,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    data_provider: str = DATA_PROVIDER_DEFAULT,
    data_root: Path = DATA_ROOT_DEFAULT,
    continue_: bool = False,
) -> None:
    """Build model.

    Args:
        data_catalog: Path to the data catalog file.
        config: Path to the model configuration file.
        build_config: Path to the model build configuration file.
        working_directory: Working directory for the model.
        data_provider: Data variant to use from data catalog (see hydroMT documentation).
        data_root: Root folder where the data is located. If None, the data catalog is not modified.
        continue_: Continue previous build if it was interrupted or failed.
    """
    with WorkingDirectory(working_directory):
        build_config = parse_config(build_config)
        model = get_builder(
            config,
            data_catalog,
            build_config["_custom_model"] if "_custom_model" in build_config else None,
            data_provider,
            data_root,
        )
        methods = {
            method: args
            for method, args in build_config.items()
            if not method.startswith("_")
        }
        model.build(
            methods=methods,
            region=parse_config(config)["general"]["region"],
            continue_=continue_,
        )


def alter_fn(
    data_catalog: Path = DATA_CATALOG_DEFAULT,
    config: Path = CONFIG_DEFAULT,
    build_config: Path | dict[str, Any] = BUILD_DEFAULT,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    from_model: Path = ALTER_FROM_MODEL_DEFAULT,
    data_provider: str = DATA_PROVIDER_DEFAULT,
    data_root: Path = DATA_ROOT_DEFAULT,
) -> None:
    """Create alternative version from base model with only changed files.

    This function is useful to create a new model based on an existing one, but with
    only a few changes. It will copy the base model and overwrite the files that are
    specified in the config and build config files. The rest of the files will be
    linked to the original model to reduce disk space.

    Args:
        data_catalog: Path to the data catalog file.
        config: Path to the model configuration file.
        build_config: Path to the model build configuration file.
        working_directory: Working directory for the model.
        from_model: Folder for the existing model.
        data_provider: Data variant to use from data catalog (see hydroMT documentation).
        data_root: Root folder where the data is located. If None, the data catalog is not modified.
    """
    from_model: Path = Path(from_model)

    with WorkingDirectory(working_directory):
        original_config: Path = from_model / config

        # if config does not exist, create a new config that inherits from the original model
        if not config.exists():
            original_config: Path = from_model / config

            config_dict: dict[str, str] = {"inherits": str(original_config)}
            with open(config, "w") as f:
                # do not sort keys, to keep the order of the config file
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # if config exists, we just make sure it inherits from the original model
        # if this is set already, we just leave it as is and assume the user knows what they are doing
        else:
            # Read existing config
            with open(config, "r") as f:
                raw_config = yaml.load(f, Loader=DetectDuplicateKeysYamlLoader)
            if "inherits" not in raw_config:
                raw_config["inherits"] = str(from_model / config)
            with open(config, "w") as f:
                yaml.dump(raw_config, f, default_flow_style=False, sort_keys=False)

        config_from_original_model = parse_config(from_model / config)
        input_folder: Path = Path(config_from_original_model["general"]["input_folder"])

        original_input_path: Path = from_model / input_folder

        # TODO: This can be removed in 2026
        if (
            not (original_input_path / "files.yml").exists()
            and (original_input_path / "files.json").exists()
        ):
            # convert input/files.json to input/files.yml
            json_files: dict[str, Any] = read_params(
                (original_input_path / "files.json"),
            )
            write_params(json_files, original_input_path / "files.yml")
            # remove the original json file
            (original_input_path / "files.json").unlink()

        original_files = read_params(original_input_path / "files.yml")

        for file_class, files in original_files.items():
            for file_name, file_path in files.items():
                if not file_path.startswith("/"):
                    original_files[file_class][file_name] = str(
                        Path("..") / original_input_path / file_path
                    )

        input_folder.mkdir(parents=True, exist_ok=True)
        with open(input_folder / "files.yml", "w") as f:
            yaml.dump(original_files, f, default_flow_style=False)

        build_config = parse_config(build_config)
        model = get_builder(
            config,
            data_catalog,
            build_config["_custom_model"] if "_custom_model" in build_config else None,
            data_provider,
            data_root,
        )
        methods = {
            method: args
            for method, args in build_config.items()
            if not method.startswith("_")
        }

        model.update(
            methods=methods,
        )


def update_fn(
    data_catalog: Path = DATA_CATALOG_DEFAULT,
    config: Path | dict[str, Any] = CONFIG_DEFAULT,
    build_config: Path = BUILD_DEFAULT,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    data_provider: str = DATA_PROVIDER_DEFAULT,
    data_root: Path = DATA_ROOT_DEFAULT,
) -> None:
    """Update model.

    Args:
        data_catalog: Path to the data catalog file.
        config: Path to the model configuration file.
        build_config: Path to the model build configuration file or a specific method within the build file using :: syntax, e.g., 'build.yml::setup_economic_data' to only run the setup_economic_data method. If the method ends with a '+', all subsequent methods are run as well.
        working_directory: Working directory for the model.
        data_provider: Data variant to use from data catalog (see hydroMT documentation).
        data_root: Root folder where the data is located. If None, the data catalog is not modified.

    Raises:
        FileNotFoundError: if the build config file is not found.
        KeyError: if the specified method is not found in the build config file.
        ValueError: if build_config is not a str or dict.
    """
    with WorkingDirectory(working_directory):
        if isinstance(build_config, Path):
            build_config_list: list[str] = str(build_config).split("::")
            build_config_file: Path = Path(build_config_list[0])

            try:
                build_config: dict[str, Any] = parse_config(build_config_file)
            except FileNotFoundError:
                if ":" in str(build_config_file) and "::" not in str(build_config_file):
                    raise FileNotFoundError(
                        f"Build config file '{str(build_config_file)}' not found. Did you mean '{str(build_config_file).replace(':', '::')}'?"
                    )
                raise

            methods = {
                method: args
                for method, args in build_config.items()
                if not method.startswith("_")
            }

            if len(build_config_list) > 1:
                assert len(build_config_list) == 2
                build_config_function: str = build_config_list[1]

                # Check if the method is specified with a trailing '+' or '#'.
                # If + we set a flag to keep all subsequent methods
                # If # we set a flag to keep all dependent methods
                if build_config_function.endswith("+"):
                    build_config_function: str = build_config_function[:-1]
                    keys_to_remove: list[str] = []

                    for key in methods.keys():
                        if key == build_config_function:
                            break
                        else:
                            keys_to_remove.append(key)
                    else:
                        raise KeyError(
                            f"Method '{build_config_function}' not found in build config file '{build_config_file}'. "
                            "Available methods: "
                            f"{', '.join(methods.keys())}"
                        )

                    # remove all functions from the methods dict except the one we want to run
                    for key in keys_to_remove:
                        del methods[key]

                elif build_config_function.endswith("#"):
                    build_config_function: str = build_config_function[:-1]
                    dependents: list[str] = build_method.get_dependents(
                        build_config_function
                    )
                    dependents.append(build_config_function)
                    methods = {
                        dependent: methods[dependent]
                        for dependent in dependents
                        if dependent in methods
                    }

                else:
                    methods = {build_config_function: methods[build_config_function]}

        elif isinstance(build_config, dict):
            methods = {
                method: args
                for method, args in build_config.items()
                if not method.startswith("_")
            }

        else:
            raise ValueError("build_config must be a str or dict.")

        model = get_builder(
            config,
            data_catalog,
            build_config["_custom_model"] if "_custom_model" in build_config else None,
            data_provider,
            data_root,
        )

        model.update(methods=methods)


def share_fn(
    working_directory: Path,
    name: str,
    include_cache: bool,
    include_output: bool,
) -> None:
    """Share model."""
    with WorkingDirectory(working_directory):
        # create a zip file called model.zip with the folders input, and model files
        # in the working directory
        folders: list = ["input"]
        if include_cache:
            folders.append("cache")
        if include_output:
            folders.append("output")
        files: list = [CONFIG_DEFAULT, BUILD_DEFAULT]
        optional_files: list = [UPDATE_DEFAULT, DATA_CATALOG_DEFAULT]
        optional_folders: list = ["data"]
        zip_filename: str = f"{name}.zip"
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            total_files: int = (
                sum(
                    [
                        sum(len(files) for _, _, files in os.walk(folder))
                        for folder in folders
                    ]
                )
                + sum(
                    [
                        sum(len(files) for _, _, files in os.walk(folder))
                        for folder in optional_folders
                        if os.path.exists(folder)
                    ]
                )
                + len(files)
                + len(optional_files)
            )  # Count total number of files
            progress: int = 0  # Initialize progress counter
            for folder in folders:
                for root, _, filenames in os.walk(folder):
                    for filename in filenames:
                        zipf.write(os.path.join(root, filename))
                        progress += 1  # Increment progress counter
                        print(
                            f"Exporting file {progress}/{total_files} to {zip_filename}",
                            end="\r",
                        )  # Print progress
            for folder in optional_folders:
                if os.path.exists(folder):
                    for root, _, filenames in os.walk(folder):
                        for filename in filenames:
                            zipf.write(os.path.join(root, filename))
                            progress += 1
                            print(
                                f"Exporting file {progress}/{total_files} to {zip_filename}",
                                end="\r",
                            )
            for file in files:
                zipf.write(file)
                progress += 1  # Increment progress counter
                print(
                    f"Exporting file {progress}/{total_files} to {zip_filename}",
                    end="\r",
                )  # Print progress
            for file in optional_files:
                if os.path.exists(file):
                    zipf.write(file)
                progress += 1  # Increment progress counter
                print(
                    f"Exporting file {progress}/{total_files} to {zip_filename}",
                    end="\r",
                )  # Print progress
            print(f"Exporting file {progress}/{total_files} to {zip_filename}")
            print("Done!")


def init_multiple_fn(
    config: str | Path,
    build_config: str | Path,
    update_config: str | Path,
    working_directory: str | Path,
    from_example: str,
    geometry_bounds: str,
    region_shapefile: str | None,
    target_area_km2: float,
    area_tolerance: float,
    cluster_prefix: str,
    overwrite: bool,
    save_geoparquet: Path | None,
    save_map: str | Path | None,
) -> None:
    """Create multiple models from a geometry by clustering downstream subbasins.

    Args:
        config: Path to the base model configuration file.
        build_config: Path to the base model build configuration file.
        update_config: Path to the base model update configuration file.
        working_directory: Working directory for the models.
        from_example: Name of the example to use as a base for the models.
        geometry_bounds: Bounding box as "xmin,ymin,xmax,ymax" to select subbasins.
        region_shapefile: Shapefile to use for the region geometry. If the file is not specified, a bounding box geometry is created from geometry_bounds.
        target_area_km2: Target cumulative upstream area per cluster (default: Danube basin ~817,000 km2).
        area_tolerance: Tolerance for target area (0.3 = 30% tolerance).
        cluster_prefix: Prefix for cluster directory names.
        overwrite: If True, overwrite existing directories and files.
        save_geoparquet: Path to save clusters as geoparquet file. If None, no file is saved.
        save_map: Path to save visualization map as PNG file. If None, no map is created.

    Raises:
        FileExistsError: If directories already exist and overwrite is False.
        FileNotFoundError: If the example folder does not exist.
        ValueError: If geometry_bounds format is invalid.
    """
    from geb.build import (
        cluster_subbasins_by_area_and_proximity,
        create_cluster_visualization_map,
        create_multi_basin_configs,
        get_all_downstream_subbasins_in_geom,
        get_river_graph,
        save_clusters_as_merged_geometries,
        save_clusters_to_geoparquet,
    )
    from geb.build.data_catalog import NewDataCatalog

    config: Path = Path(config)
    build_config: Path = Path(build_config)
    update_config: Path = Path(update_config)
    working_directory: Path = Path(working_directory)
    if region_shapefile:
        region_shapefile: Path = Path(region_shapefile)
    # Create the models/large_scale directory structure
    large_scale_dir = working_directory / "large_scale"
    if not large_scale_dir.exists():
        large_scale_dir.mkdir(parents=True, exist_ok=True)

    # create logger
    logger = create_logger(working_directory / "init_multiple.log")

    # Always create geoparquet and map files in large_scale directory if not specified
    if save_geoparquet is None:
        save_geoparquet = large_scale_dir / f"{cluster_prefix}_clusters.geoparquet"
    if save_map is None:
        save_map = large_scale_dir / f"{cluster_prefix}_clusters_map.png"

    # Parse geometry bounds
    try:
        bounds = [float(x.strip()) for x in geometry_bounds.split(",")]
        if len(bounds) != 4:
            raise ValueError
        xmin, ymin, xmax, ymax = bounds
    except ValueError:
        raise ValueError(
            "geometry_bounds must be in format 'xmin,ymin,xmax,ymax' with numeric values"
        )
    logger.info("Starting multiple model initialization")
    logger.info(f"Target area: {target_area_km2:,.0f} km²")
    logger.info(f"Area tolerance: {area_tolerance:.1%}")
    # Create bounding box geometry or read region shapefile
    if not region_shapefile:
        logger.info(f"Using geometry bounds: {geometry_bounds}")
        bbox_geom = gpd.GeoDataFrame(
            geometry=[box(xmin, ymin, xmax, ymax)], crs="EPSG:4326"
        )
    else:
        logger.info(f"Using region shapefile: {region_shapefile}")
        region_shapefile_path: Path = working_directory / region_shapefile
        if not region_shapefile_path.exists():
            raise FileNotFoundError(
                f"Region shapefile not found at: {region_shapefile_path}"
            )
        bbox_geom = gpd.read_file(region_shapefile_path)

    # check crs bounding box geometry
    if bbox_geom.crs != "EPSG:4326":
        bbox_geom = bbox_geom.to_crs("EPSG:4326")
    # Initialize data catalog and logger
    data_catalog_instance = NewDataCatalog()

    logger.info("Loading river network...")
    river_graph = get_river_graph(data_catalog_instance)

    logger.info("Finding downstream subbasins in geometry...")
    downstream_subbasins = get_all_downstream_subbasins_in_geom(
        data_catalog_instance, bbox_geom, logger
    )

    if not downstream_subbasins:
        raise ValueError("No downstream subbasins found in the specified geometry")

    logger.info(f"Found {len(downstream_subbasins)} downstream subbasins")

    logger.info("Clustering subbasins by area and proximity...")
    clusters = cluster_subbasins_by_area_and_proximity(
        data_catalog_instance,
        downstream_subbasins,
        target_area_km2=target_area_km2,
        area_tolerance=area_tolerance,
        logger=logger,
    )

    logger.info(f"Created {len(clusters)} clusters")

    # Check for existing directories if not overwriting
    if not overwrite:
        for i in range(len(clusters)):
            cluster_dir = large_scale_dir / f"{cluster_prefix}_{i:03d}"
            if cluster_dir.exists():
                raise FileExistsError(
                    f"Cluster directory {cluster_dir} already exists. Remove --no-overwrite flag to overwrite."
                )

    # Verify example folder exists
    example_folder: Path = GEB_PACKAGE_DIR / "examples" / from_example
    if not example_folder.exists():
        raise FileNotFoundError(
            f"Example folder {example_folder} does not exist. Did you use the right --from-example option?"
        )

    logger.info(f"Creating cluster configurations using example: {from_example}")
    # Create cluster configurations
    cluster_directories = create_multi_basin_configs(
        clusters=clusters,
        working_directory=large_scale_dir,
        cluster_prefix=cluster_prefix,
    )

    logger.info(f"Saving clusters to geoparquet: {save_geoparquet}")
    # Save clusters to geoparquet (always create)
    save_clusters_to_geoparquet(
        clusters=clusters,
        data_catalog=data_catalog_instance,
        output_path=save_geoparquet,
        cluster_prefix=cluster_prefix,
    )

    # Save clusters as merged geometries (complete basins as single polygons)
    merged_basins_path = (
        save_geoparquet.parent / f"complete_basins_{save_geoparquet.stem}.geoparquet"
    )
    logger.info(f"Saving complete basins as merged geometries: {merged_basins_path}")
    save_clusters_as_merged_geometries(
        clusters=clusters,
        data_catalog=data_catalog_instance,
        river_graph=river_graph,
        output_path=merged_basins_path,
        cluster_prefix=cluster_prefix,
        include_upstream=True,  # Include all upstream subbasins in merged geometry
    )

    logger.info(f"Creating visualization map: {save_map}")
    # Create visualization map (always create)
    create_cluster_visualization_map(
        clusters=clusters,
        data_catalog=data_catalog_instance,
        output_path=save_map,
        cluster_prefix=cluster_prefix,
    )

    logger.info(
        f"Successfully created {len(cluster_directories)} model configurations:"
    )
    for cluster_dir in cluster_directories:
        logger.info(f"  {cluster_dir.relative_to(large_scale_dir)}")

    logger.info("To build all models, run:")
    logger.info(f"  cd {large_scale_dir}")
    logger.info(f"  for dir in {cluster_prefix}_*/; do")
    logger.info(f"    echo 'Building model in $dir'")
    logger.info(f"    cd $dir && geb build && cd ..")
    logger.info(f"  done")

    logger.info("Multiple model initialization completed successfully")
