"""Utilities for reading and plotting household attribute results."""

import os
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

PLOT_Y_LIMS_DEFAULT: dict[str, tuple[float, float]] = {
    "expected_annual_damage": (0.0, 1.0),
    "n_adaptation_uptake": (0.0, 1.0),
    "n_households_exposed_to_flooding": (0.0, 1.0),
}


PLOT_Y_LIMS_MULTIRUN_BASE_FUTURE: dict[str, tuple[float, float]] = {
    "expected_annual_damage": (0.0, 3e9),
    "n_adaptation_uptake": (0.0, 1.0),
    "n_households_exposed_to_flooding": (0.0, 1.0),
}
PLOT_Y_LABELS: dict[str, str] = {
    "expected_annual_damage": "Expected Annual Damage (USD)",
    "n_adaptation_uptake": "Adaptation Uptake (fraction of households)",
    "n_households_exposed_to_flooding": "Number of Households Exposed to Flooding",
}
SCENARIOS_BASE_FUTURE: tuple[str, str] = ("base", "_future")


def _list_household_attribute_files(results_path: str) -> list[str]:
    """Return sorted household attribute parquet filenames.

    Args:
        results_path: Path to agents.households report folder.

    Returns:
        Sorted list of parquet file names.
    """
    if not os.path.exists(results_path):
        return []
    return sorted(fn for fn in os.listdir(results_path) if fn.endswith(".parquet"))


def _ensure_axes_array(axes: object, ncols: int) -> list[plt.Axes]:
    """Normalize matplotlib subplot output to a flat list of axes.

    Notes:
        Matplotlib returns a scalar Axes when ncols == 1. Normalizing here avoids
        branch logic in plotting functions.

    Args:
        axes: Return value from plt.subplots.
        ncols: Number of requested subplot columns.

    Returns:
        List of axes with length ncols.
    """
    if ncols == 1:
        return [axes]  # type: ignore[list-item]
    return list(axes)  # type: ignore[arg-type]


def _resolve_ylim(
    df: pd.DataFrame,
    household_attribute_name: str,
    predefined_ylims: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float] | None:
    """Resolve a sensible y-axis range for one household attribute.

    Notes:
        If predefined limits exist and are too small for observed values, the upper
        bound is expanded to 110% of the data maximum to prevent clipping.

    Args:
        df: Timeseries values for one household attribute.
        household_attribute_name: Variable name (e.g., expected_annual_damage).
        predefined_ylims: Optional mapping of variable to preferred y-axis limits.

    Returns:
        A y-limit tuple or None when limits should be left automatic.
    """
    ylims_variable: tuple[float, float] | None = None
    if predefined_ylims is not None:
        ylims_variable = predefined_ylims.get(household_attribute_name)

    if ylims_variable is None:
        return None

    max_value: float = float(df.max().max())
    if max_value > ylims_variable[1]:
        return (0.0, max_value * 1.1)
    return ylims_variable


def _validate_x_axis_mode(x_axis: str) -> Literal["year", "timestep"]:
    """Validate and normalize x-axis mode.

    Args:
        x_axis: X-axis mode, either year or timestep.

    Returns:
        Validated x-axis mode.

    Raises:
        ValueError: If x_axis is not one of the supported values.
    """
    if x_axis not in {"year", "timestep"}:
        raise ValueError("x_axis must be either 'year' or 'timestep'.")
    return x_axis


def _resolve_x_axis_values(
    df: pd.DataFrame,
    x_axis: Literal["year", "timestep"],
) -> pd.Index:
    """Resolve x-axis values for plotting household time series.

    Args:
        df: Time-indexed dataframe to plot.
        x_axis: X-axis mode.

    Returns:
        Index used for plotting.
    """
    if x_axis == "year":
        return df.index
    return pd.RangeIndex(start=0, stop=len(df.index), step=1)


def _resolve_x_axis_label(x_axis: Literal["year", "timestep"]) -> str:
    """Resolve x-axis label text from x-axis mode.

    Args:
        x_axis: X-axis mode.

    Returns:
        Label for the x-axis.
    """
    if x_axis == "year":
        return "Year"
    return "Timestep"


def _read_multirun_results(
    model_path: str,
    scenario_to_prefixes: dict[str, list[str]],
    process_adaptation_uptake: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Read and merge household attribute results for multiple runs.

    Notes:
        Columns are concatenated per attribute so each column represents one run.

    Args:
        model_path: Root model path containing scenario folders.
        scenario_to_prefixes: Mapping of scenario name to run prefixes to include.
        process_adaptation_uptake: Whether to process adaptation uptake data.
    Returns:
        Nested dictionary {scenario: {attribute_name: dataframe}}.
    """
    results: dict[str, dict[str, pd.DataFrame]] = {
        scenario: {} for scenario in scenario_to_prefixes
    }

    for scenario, run_prefixes in scenario_to_prefixes.items():
        output_root: str = os.path.join(model_path, scenario, "output")
        if not os.path.exists(output_root):
            continue

        model_runs: list[str] = sorted(
            model_dir
            for model_dir in os.listdir(output_root)
            if any(model_dir.startswith(prefix) for prefix in run_prefixes)
        )

        for model_run in model_runs:
            results_path: str = os.path.join(
                model_path,
                scenario,
                "output",
                model_run,
                "report",
                "agents.households",
            )
            for household_attribute_fn in _list_household_attribute_files(results_path):
                household_attribute_name: str = household_attribute_fn.removesuffix(
                    ".parquet"
                )
                attribute_df: pd.DataFrame = pd.read_parquet(
                    os.path.join(results_path, household_attribute_fn)
                )
                if (
                    household_attribute_name == "n_adaptation_uptake"
                    and "n_households_exposed_to_flooding.parquet"
                    in _list_household_attribute_files(results_path)
                    and process_adaptation_uptake
                ):
                    # Calculate the fraction of households that have adopted adaptation measures
                    exposed_df: pd.DataFrame = pd.read_parquet(
                        os.path.join(
                            results_path, "n_households_exposed_to_flooding.parquet"
                        )
                    )
                    attribute_df["n_adaptation_uptake"] = (
                        attribute_df["n_adaptation_uptake"]
                        / exposed_df["n_households_exposed_to_flooding"]
                    ).fillna(0)

                # Keep run identity in column names so downstream aggregation can
                # align runs by name instead of by positional column index.
                if attribute_df.shape[1] == 1:
                    attribute_df.columns = [model_run]
                else:
                    attribute_df.columns = [
                        f"{model_run}_{column_name}"
                        for column_name in attribute_df.columns
                    ]

                if household_attribute_name not in results[scenario]:
                    results[scenario][household_attribute_name] = attribute_df
                else:
                    results[scenario][household_attribute_name] = pd.concat(
                        [results[scenario][household_attribute_name], attribute_df],
                        axis=1,
                    )

    return results


def process_household_attributes(
    model_path: str,
    scenario: str,
    model_name: str = "default",
    x_axis: Literal["year", "timestep"] = "year",
) -> None:
    """Plot household attribute timeseries for one model run.

    Args:
        model_path: Root model path.
        scenario: Scenario name (e.g., base or future).
        model_name: Name of the model output folder.
        x_axis: X-axis mode, either year (data index) or timestep (0..n-1).
    """
    x_axis = _validate_x_axis_mode(x_axis)

    results_path: str = os.path.join(
        model_path, scenario, "output", model_name, "report", "agents.households"
    )
    household_attributes_fns: list[str] = _list_household_attribute_files(results_path)
    if not household_attributes_fns:
        return

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(household_attributes_fns),
        figsize=(7 * len(household_attributes_fns), 5),
    )
    axes_list: list[plt.Axes] = _ensure_axes_array(axes, len(household_attributes_fns))
    x_axis_label: str = _resolve_x_axis_label(x_axis)

    for ax, fn in zip(axes_list, household_attributes_fns):
        household_attribute_name: str = fn.removesuffix(".parquet")
        df: pd.DataFrame = pd.read_parquet(os.path.join(results_path, fn))

        if (
            household_attribute_name == "n_adaptation_uptake"
            and "n_households_exposed_to_flooding.parquet" in household_attributes_fns
        ):
            # Calculate the fraction of households that have adopted adaptation measures
            exposed_df: pd.DataFrame = pd.read_parquet(
                os.path.join(results_path, "n_households_exposed_to_flooding.parquet")
            )
            df["n_adaptation_uptake"] = (
                df["n_adaptation_uptake"]
                / exposed_df["n_households_exposed_to_flooding"]
            )
        x_values: pd.Index = _resolve_x_axis_values(df, x_axis)

        ylims_variable: tuple[float, float] | None = _resolve_ylim(
            df,
            household_attribute_name,
            PLOT_Y_LIMS_DEFAULT,
        )
        ax.plot(x_values, df.to_numpy())
        if ylims_variable is not None:
            ax.set_ylim(ylims_variable)
        if household_attribute_name in PLOT_Y_LABELS:
            ax.set_ylabel(PLOT_Y_LABELS[household_attribute_name])
        ax.set_xlabel(x_axis_label)

    fig.suptitle("Household Attributes Over Time")
    fig.tight_layout()
    plt.savefig(os.path.join(results_path, "household_attributes_histograms.png"))
    plt.close(fig)


def read_multirun_results_for_base_and_future(
    model_path: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compatibility wrapper for reading base and future scenarios.

    Args:
        model_path: Root model path containing scenario folders.

    Returns:
        Nested dictionary {scenario: {attribute_name: dataframe}}.
    """
    return read_multirun_results_for_scenarios(
        model_path=model_path,
        scenarios=list(SCENARIOS_BASE_FUTURE),
        run_prefix="run_",
    )


def read_multirun_results_for_scenarios(
    model_path: str,
    scenarios: list[str],
    run_prefix: str = "run_",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Read run-prefixed outputs for any set of scenarios.

    Args:
        model_path: Root model path containing scenario folders.
        scenarios: Scenario names to read (e.g., base, future, policy_a).
        run_prefix: Run folder prefix to include.

    Returns:
        Nested dictionary {scenario: {attribute_name: dataframe}}.
    """
    scenario_to_prefixes: dict[str, list[str]] = {
        scenario: [run_prefix] for scenario in scenarios
    }
    return _read_multirun_results(model_path, scenario_to_prefixes)


def read_multirun_results_within_scenario(
    model_path: str,
    scenario: str,
    run_prefixes: list[str],
    process_adaptation_uptake: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Read grouped runs for one scenario and return one dataframe per prefix.

    Notes:
        This function groups runs by prefix (e.g., nogov_ vs cba_) so they can be
        compared within a single scenario.

    Args:
        model_path: Root model path containing scenario folders.
        scenario: Scenario name to read.
        run_prefixes: Prefixes used to group runs.
        process_adaptation_uptake: Whether to normalize adaptation uptake by
            exposed households.

    Returns:
        Nested dictionary {run_prefix: {attribute_name: dataframe}}.
    """
    results_by_prefix: dict[str, dict[str, pd.DataFrame]] = {
        run_prefix: {} for run_prefix in run_prefixes
    }

    output_root: str = os.path.join(model_path, scenario, "output")
    if not os.path.exists(output_root):
        return results_by_prefix

    model_runs: list[str] = sorted(os.listdir(output_root))
    for run_prefix in run_prefixes:
        prefix_runs: list[str] = [
            model_run for model_run in model_runs if model_run.startswith(run_prefix)
        ]

        for model_run in prefix_runs:
            results_path: str = os.path.join(
                model_path,
                scenario,
                "output",
                model_run,
                "report",
                "agents.households",
            )

            for household_attribute_fn in _list_household_attribute_files(results_path):
                household_attribute_name: str = household_attribute_fn.removesuffix(
                    ".parquet"
                )
                attribute_df: pd.DataFrame = pd.read_parquet(
                    os.path.join(results_path, household_attribute_fn)
                )

                if (
                    household_attribute_name == "n_adaptation_uptake"
                    and "n_households_exposed_to_flooding.parquet"
                    in _list_household_attribute_files(results_path)
                    and process_adaptation_uptake
                ):
                    # Calculate the fraction of households that have adopted adaptation measures
                    exposed_df: pd.DataFrame = pd.read_parquet(
                        os.path.join(
                            results_path, "n_households_exposed_to_flooding.parquet"
                        )
                    )
                    attribute_df["n_adaptation_uptake"] = (
                        attribute_df["n_adaptation_uptake"]
                        / exposed_df["n_households_exposed_to_flooding"]
                    ).fillna(0)

                # Keep run identity in column names so cluster-level sums keep
                # no_reloc_0..N as separate columns.
                if attribute_df.shape[1] == 1:
                    attribute_df.columns = [model_run]
                else:
                    attribute_df.columns = [
                        f"{model_run}_{column_name}"
                        for column_name in attribute_df.columns
                    ]

                if household_attribute_name not in results_by_prefix[run_prefix]:
                    results_by_prefix[run_prefix][household_attribute_name] = (
                        attribute_df
                    )
                else:
                    results_by_prefix[run_prefix][household_attribute_name] = pd.concat(
                        [
                            results_by_prefix[run_prefix][household_attribute_name],
                            attribute_df,
                        ],
                        axis=1,
                    )

    return results_by_prefix


def read_multirun_results_within_base_or_future(
    model_path: str,
    scenario: str,
    run_prefixes: list[str],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compatibility wrapper for the old function name.

    Notes:
        Kept to avoid breaking existing scripts that still import the old name.

    Args:
        model_path: Root model path containing scenario folders.
        scenario: Scenario name to read.
        run_prefixes: Prefixes used to group runs.

    Returns:
        Nested dictionary {run_prefix: {attribute_name: dataframe}}.
    """
    return read_multirun_results_within_scenario(model_path, scenario, run_prefixes)


def _comparison_output_path(
    model_path: str,
    mode: str,
    scenario: str | None = None,
) -> str:
    """Build a non-overlapping output path for comparison plots.

    Args:
        model_path: Root model path.
        mode: Comparison mode (e.g., base_future, within_scenario).
        scenario: Optional scenario name to include in file name.

    Returns:
        Absolute path where the comparison plot should be written.
    """
    if scenario is None:
        return os.path.join(model_path, f"comparisons_{mode}.png")
    return os.path.join(model_path, f"comparisons_{mode}_{scenario}.png")


def _build_group_colors(group_names: list[str]) -> dict[str, str]:
    """Create deterministic colors for scenario or prefix groups.

    Notes:
        The tab10 colormap is repeated when more than ten groups are provided.

    Args:
        group_names: Group names shown in the legend.

    Returns:
        Mapping from group name to matplotlib color string.
    """
    cmap = plt.get_cmap("tab10")
    return {
        group_name: cmap(index % 10)
        for index, group_name in enumerate(sorted(group_names))
    }


def _plot_multirun_results(
    results: dict[str, dict[str, pd.DataFrame]],
    colors: dict[str, str],
    output_path: str,
    predefined_ylims: dict[str, tuple[float, float]] | None = None,
    x_axis: Literal["year", "timestep"] = "year",
) -> None:
    """Plot multirun household attributes for grouped scenarios or prefixes.

    Args:
        results: Nested dictionary {group_name: {attribute_name: dataframe}}.
        colors: Line colors per group name.
        output_path: Path to save the plot image.
        predefined_ylims: Optional y-axis bounds per attribute.
        x_axis: X-axis mode, either year (data index) or timestep (0..n-1).
    """
    x_axis = _validate_x_axis_mode(x_axis)

    if not results:
        return

    attribute_names: list[str] = sorted(
        {
            attribute_name
            for group_results in results.values()
            for attribute_name in group_results
        }
    )
    if not attribute_names:
        return

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(attribute_names),
        figsize=(5 * len(attribute_names), 5),
    )
    axes_list: list[plt.Axes] = _ensure_axes_array(axes, len(attribute_names))
    x_axis_label: str = _resolve_x_axis_label(x_axis)

    for ax, household_attribute_name in zip(axes_list, attribute_names):
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        has_individual_runs: bool = False

        for group_name, group_results in results.items():
            if household_attribute_name not in group_results:
                continue
            df: pd.DataFrame = group_results[household_attribute_name]
            if df.empty:
                continue
            x_values: pd.Index = _resolve_x_axis_values(df, x_axis)

            ax.plot(x_values, df.to_numpy(), alpha=0.3, color="grey")
            has_individual_runs = True
            ax.plot(
                x_values,
                df.to_numpy().mean(axis=1),
                label=f"Mean {group_name}",
                linewidth=2,
                color=colors.get(group_name, "black"),
            )

            ylims_variable: tuple[float, float] | None = _resolve_ylim(
                df,
                household_attribute_name,
                predefined_ylims,
            )
            if ylims_variable is not None:
                ax.set_ylim(ylims_variable)

        if has_individual_runs:
            ax.plot([], [], color="grey", alpha=0.3, label="Individual Runs")
        ax.set_title(household_attribute_name)
        if household_attribute_name in PLOT_Y_LABELS:
            ax.set_ylabel(PLOT_Y_LABELS[household_attribute_name])
        ax.set_xlabel(x_axis_label)
        ax.legend()

    fig.suptitle("Household Attributes Over Time for Multiple Runs")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_multirun_results_for_scenarios(
    model_path: str,
    scenarios: list[str],
    output_path: str | None = None,
    x_axis: Literal["year", "timestep"] = "year",
) -> None:
    """Plot run_* household attributes for a list of scenarios.

    Args:
        model_path: Root model path containing scenario folders.
        scenarios: Scenario names to compare in one plot.
        output_path: Optional output image path.
        x_axis: X-axis mode, either year (data index) or timestep (0..n-1).
    """
    plot_multirun_results_across_scenarios(
        model_path=model_path,
        scenarios=scenarios,
        x_axis=x_axis,
        output_path=(
            output_path
            if output_path is not None
            else _comparison_output_path(model_path, "scenarios")
        ),
    )


def plot_multirun_results_base_and_future(
    model_path: str,
    output_path: str | None = None,
    x_axis: Literal["year", "timestep"] = "year",
) -> None:
    """Compatibility wrapper for plotting base and future scenarios.

    Args:
        model_path: Root model path containing scenario folders.
        output_path: Optional output image path.
        x_axis: X-axis mode, either year (data index) or timestep (0..n-1).
    """
    plot_multirun_results_for_scenarios(
        model_path=model_path,
        scenarios=list(SCENARIOS_BASE_FUTURE),
        x_axis=x_axis,
        output_path=(
            output_path
            if output_path is not None
            else _comparison_output_path(model_path, "base_future")
        ),
    )


def plot_multirun_results_across_scenarios(
    model_path: str,
    scenarios: list[str],
    output_path: str | None = None,
    run_prefix: str = "nogov_",
    colors: dict[str, str] | None = None,
    x_axis: Literal["year", "timestep"] = "year",
) -> None:
    """Plot run-prefixed household attributes for any list of scenarios.

    Args:
        model_path: Root model path containing scenario folders.
        scenarios: Scenario names to compare in one plot.
        output_path: Optional output image path.
        run_prefix: Run folder prefix to include from each scenario.
        colors: Optional custom line colors keyed by scenario.
        x_axis: X-axis mode, either year (data index) or timestep (0..n-1).
    """
    if not scenarios:
        return

    resolved_colors: dict[str, str] = (
        colors if colors is not None else _build_group_colors(scenarios)
    )
    results: dict[str, dict[str, pd.DataFrame]] = read_multirun_results_for_scenarios(
        model_path=model_path,
        scenarios=scenarios,
        run_prefix=run_prefix,
    )
    _plot_multirun_results(
        results=results,
        colors=resolved_colors,
        x_axis=x_axis,
        output_path=(
            output_path
            if output_path is not None
            else _comparison_output_path(model_path, "scenarios")
        ),
        predefined_ylims=PLOT_Y_LIMS_MULTIRUN_BASE_FUTURE,
    )


def plot_multirun_results_within_scenario(
    model_path: str,
    scenario: str = "base",
    prefixes: list[str] | None = None,
    output_path: str | None = None,
    x_axis: Literal["year", "timestep"] = "year",
) -> None:
    """Plot multirun household attributes for run groups in one scenario.

    Args:
        model_path: Root model path containing scenario folders.
        scenario: Scenario name to compare prefixes within.
        prefixes: Optional run prefixes to compare.
        output_path: Optional output image path.
        x_axis: X-axis mode, either year (data index) or timestep (0..n-1).
    """
    if prefixes is None:
        prefixes = ["no_gov_", "no_adapt", "full"]

    colors: dict[str, str] = {
        "no_reloc": "black",
        "no_gov_": "red",
        "no_adapt": "green",
        "full": "blue",
    }
    results: dict[str, dict[str, pd.DataFrame]] = read_multirun_results_within_scenario(
        model_path,
        scenario,
        prefixes,
    )
    _plot_multirun_results(
        results=results,
        colors=colors,
        x_axis=x_axis,
        output_path=(
            output_path
            if output_path is not None
            else _comparison_output_path(model_path, "within_scenario", scenario)
        ),
        predefined_ylims=PLOT_Y_LIMS_DEFAULT,
    )


def plot_multirun_results_within_base_or_future(
    model_path: str,
    scenario: str = "base",
    prefixes: list[str] | None = None,
    output_path: str | None = None,
    x_axis: Literal["year", "timestep"] = "year",
) -> None:
    """Compatibility wrapper for the old function name.

    Args:
        model_path: Root model path containing scenario folders.
        scenario: Scenario name to compare prefixes within.
        prefixes: Optional run prefixes to compare.
        output_path: Optional output image path.
        x_axis: X-axis mode, either year (data index) or timestep (0..n-1).
    """
    plot_multirun_results_within_scenario(
        model_path=model_path,
        scenario=scenario,
        prefixes=prefixes,
        output_path=output_path,
        x_axis=x_axis,
    )


def combine_cluster_results(
    model_path: str,
    scenario: str,
    run_prefixes: list[str] | None = None,
) -> None:
    """Combine household attribute results from multiple cluster runs.

    Args:
        model_path: Root model path.
        scenario: Scenario name to read from each cluster folder.
        run_prefixes: Optional list of prefixes to include in the combined results.

    Raises:
        ValueError: If run_prefixes is empty.
    """
    if run_prefixes is None or len(run_prefixes) == 0:
        raise ValueError("run_prefixes must contain at least one prefix.")

    output_root: str = model_path
    if not os.path.exists(output_root):
        return

    cluster_folders: list[str] = sorted(
        model_dir
        for model_dir in os.listdir(output_root)
        if model_dir.startswith("cluster_")
        and os.path.isdir(os.path.join(output_root, model_dir))
    )
    if not cluster_folders:
        return

    combined_results: dict[str, dict[str, pd.DataFrame]] = {
        run_prefix: {} for run_prefix in run_prefixes
    }

    for cluster_folder in cluster_folders:
        cluster_results: dict[str, dict[str, pd.DataFrame]] = (
            read_multirun_results_within_scenario(
                model_path=os.path.join(model_path, cluster_folder),
                scenario=scenario,
                run_prefixes=run_prefixes,
                process_adaptation_uptake=False,
            )
        )
        for run_prefix, prefix_results in cluster_results.items():
            for household_attribute_name, attribute_df in prefix_results.items():
                if household_attribute_name not in combined_results[run_prefix]:
                    combined_results[run_prefix][household_attribute_name] = (
                        attribute_df.copy()
                    )
                else:
                    # Sum aligned runs and timesteps across clusters.
                    combined_results[run_prefix][household_attribute_name] = (
                        combined_results[run_prefix][household_attribute_name].add(
                            attribute_df,
                            fill_value=0.0,
                        )
                    )

    combined_results_path: str = os.path.join(
        model_path,
        "combined_results",
        scenario,
        "report",
        "agents.households",
    )
    for run_prefix, prefix_results in combined_results.items():
        prefix_results_path: str = os.path.join(combined_results_path, run_prefix)
        os.makedirs(prefix_results_path, exist_ok=True)
        for household_attribute_name, attribute_df in prefix_results.items():
            attribute_output_path: str = os.path.join(
                prefix_results_path,
                f"{household_attribute_name}.parquet",
            )
            attribute_df.to_parquet(attribute_output_path)


def read_combined_cluster_results(
    model_path: str,
    scenario: str,
    run_prefixes: list[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Read combined household attribute results stored by run prefix.

    Args:
        model_path: Root model path.
        scenario: Scenario name under combined_results.
        run_prefixes: Optional run prefixes to read. If omitted, all available
            prefixes in the combined results folder are read.

    Returns:
        Nested dictionary {run_prefix: {attribute_name: dataframe}}.
    """
    combined_results_root: str = os.path.join(
        model_path,
        "combined_results",
        scenario,
        "report",
        "agents.households",
    )
    if not os.path.exists(combined_results_root):
        return {}

    available_prefixes: list[str] = sorted(
        prefix_name
        for prefix_name in os.listdir(combined_results_root)
        if os.path.isdir(os.path.join(combined_results_root, prefix_name))
    )
    prefixes_to_read: list[str] = (
        run_prefixes if run_prefixes is not None else available_prefixes
    )

    results: dict[str, dict[str, pd.DataFrame]] = {}
    for run_prefix in prefixes_to_read:
        prefix_results_path: str = os.path.join(combined_results_root, run_prefix)
        if not os.path.exists(prefix_results_path):
            continue

        prefix_results: dict[str, pd.DataFrame] = {}
        for household_attribute_fn in _list_household_attribute_files(
            prefix_results_path
        ):
            household_attribute_name: str = household_attribute_fn.removesuffix(
                ".parquet"
            )
            prefix_results[household_attribute_name] = pd.read_parquet(
                os.path.join(prefix_results_path, household_attribute_fn)
            )

        if prefix_results:
            results[run_prefix] = prefix_results

    return results


def plot_combined_cluster_results_within_scenario(
    model_path: str,
    scenario: str = "base",
    run_prefixes: list[str] | None = None,
    output_path: str | None = None,
    x_axis: Literal["year", "timestep"] = "year",
) -> None:
    """Plot merged cluster household attributes for run prefixes in one scenario.

    Notes:
        This expects files created by combine_cluster_results under
        combined_results/<scenario>/report/agents.households/<run_prefix>/.

    Args:
        model_path: Root model path.
        scenario: Scenario name to plot.
        run_prefixes: Optional run prefixes to include.
        output_path: Optional output image path.
        x_axis: X-axis mode, either year (data index) or timestep (0..n-1).
    """
    results: dict[str, dict[str, pd.DataFrame]] = read_combined_cluster_results(
        model_path=model_path,
        scenario=scenario,
        run_prefixes=run_prefixes,
    )
    if not results:
        return

    colors: dict[str, str] = _build_group_colors(list(results.keys()))
    _plot_multirun_results(
        results=results,
        colors=colors,
        x_axis=x_axis,
        output_path=(
            output_path
            if output_path is not None
            else _comparison_output_path(model_path, "combined_clusters", scenario)
        ),
        predefined_ylims=PLOT_Y_LIMS_DEFAULT,
    )


if __name__ == "__main__":
    model_path = os.path.join("..", "..", "models", "models", "mex")

    model_name = "default"
    scenario = "base"
    scenarios_to_compare = list(SCENARIOS_BASE_FUTURE)
    prefixes = ["no_reloc", "no_gov", "no_adapt", "full"]

    # 1) Build merged (cluster-summed) results while keeping per-run columns.
    combine_cluster_results(
        model_path=model_path, scenario=scenario, run_prefixes=prefixes
    )

    # 2) Plot merged cluster results for the selected scenario.
    plot_combined_cluster_results_within_scenario(
        model_path=model_path,
        scenario=scenario,
        run_prefixes=prefixes,
        x_axis="year",
    )

    # 3) Plot non-merged reference views from regular multirun outputs.
    model_path = os.path.join("..", "..", "models", "models", "etaple_new")
    plot_multirun_results_within_scenario(model_path, scenario, prefixes)
    plot_multirun_results_for_scenarios(model_path, scenarios_to_compare, x_axis="year")

    # 4) Single-run timeseries diagnostics.
    process_household_attributes(model_path, scenario, model_name)
