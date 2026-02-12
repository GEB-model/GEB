"""MCP Server for GEB."""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Define defaults locally to avoid top-level import of geb.runner
CONFIG_DEFAULT = "model.yml"
WORKING_DIRECTORY_DEFAULT = "."
UPDATE_DEFAULT = "update.yml"
BUILD_DEFAULT = "build.yml"

mcp = FastMCP("GEB")


@mcp.tool()
def run_model(
    config: str = CONFIG_DEFAULT,
    working_directory: str = WORKING_DIRECTORY_DEFAULT,
    optimize: bool = False,
    profiling: bool = False,
) -> str:
    """Run the GEB model.

    Args:
        config: Path to model configuration file.
        working_directory: Working directory.
        optimize: Run in optimized mode.
        profiling: Run with profiling.

    Returns:
        Status message.
    """
    try:
        from geb.runner import run_model_with_method

        run_model_with_method(
            method="run",
            config=Path(config),
            working_directory=Path(working_directory),
            optimize=optimize,
            profiling=profiling,
        )
        return "Model run completed successfully."
    except Exception as e:
        return f"Error running model: {e}"


@mcp.tool()
def spinup_model(
    config: str = CONFIG_DEFAULT,
    working_directory: str = WORKING_DIRECTORY_DEFAULT,
    optimize: bool = False,
) -> str:
    """Run model spinup.

    Args:
        config: Path to model configuration file.
        working_directory: Working directory.
        optimize: Run in optimized mode.

    Returns:
        Status message.
    """
    try:
        from geb.runner import run_model_with_method

        run_model_with_method(
            method="spinup",
            config=Path(config),
            working_directory=Path(working_directory),
            optimize=optimize,
        )
        return "Model spinup completed successfully."
    except Exception as e:
        return f"Error spinning up model: {e}"


@mcp.tool()
def build_model(
    config: str = CONFIG_DEFAULT,
    build_config: str = BUILD_DEFAULT,
    working_directory: str = WORKING_DIRECTORY_DEFAULT,
    data_provider: str = "default",
    continue_substep: bool = False,
) -> str:
    """Build the model.

    Returns:
        Status message.

    Args:
        config: Path to model configuration file.
        build_config: Path to build configuration file.
        working_directory: Working directory.
        data_provider: Data provider to use.
        continue_substep: Continue previous build.
    """
    try:
        from geb.runner import build_fn

        build_fn(
            config=Path(config),
            build_config=Path(build_config),
            working_directory=Path(working_directory),
            data_provider=data_provider,
            continue_=continue_substep,
        )
        return "Model build completed."
    except Exception as e:
        return f"Error building model: {e}"


@mcp.tool()
def update_model(
    config: str = CONFIG_DEFAULT,
    build_config: str = UPDATE_DEFAULT,
    working_directory: str = WORKING_DIRECTORY_DEFAULT,
    data_provider: str = "default",
) -> str:
    """Update the model.

    Args:
        config: Path to model configuration file.
        build_config: Path to update configuration file.
        working_directory: Working directory.
        data_provider: Data provider to use.

    Returns:
        Status message.
    """
    try:
        from geb.runner import update_fn

        update_fn(
            config=Path(config),
            build_config=Path(build_config),
            working_directory=Path(working_directory),
            data_provider=data_provider,
        )
        return "Model updated."
    except Exception as e:
        return f"Error updating model: {e}"


@mcp.tool()
def init_model(
    config: str = CONFIG_DEFAULT,
    build_config: str = BUILD_DEFAULT,
    update_config: str = UPDATE_DEFAULT,
    working_directory: str = WORKING_DIRECTORY_DEFAULT,
    from_example: str = "geul",
    basin_id: str | None = None,
    ISO3: str | None = None,
    overwrite: bool = False,
) -> str:
    """Initialize a new model.

    Args:
        config: Path to config file.
        build_config: Path to build config file.
        update_config: Path to update config file.
        working_directory: Working directory.
        from_example: Example model to copy.
        basin_id: Basin ID(s).
        ISO3: ISO3 code.
        overwrite: Overwrite existing files.

    Returns:
        Status message.
    """
    try:
        from geb.runner import init_fn

        init_fn(
            config=Path(config),
            build_config=Path(build_config),
            update_config=Path(update_config),
            working_directory=Path(working_directory),
            from_example=from_example,
            basin_id=basin_id,
            ISO3=ISO3,
            overwrite=overwrite,
        )
        return "Model initialized."
    except Exception as e:
        return f"Error initializing model: {e}"


if __name__ == "__main__":
    mcp.run()
