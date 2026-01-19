"""Sensitivity analysis module for warning system evaluation.

This module provides tools for conducting sensitivity analyses on warning system
parameters to understand their impact on key outcomes like warning effectiveness,
evacuation rates, and damage reduction.
"""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

if TYPE_CHECKING:
    import geopandas as gpd

    from geb.agents.households import Households


@dataclass
class SensitivityParameter:
    """Configuration for a single sensitivity parameter.

    Args:
        name: Parameter name (must match config key).
        enabled: Whether to vary this parameter in sensitivity analysis.
        base_value: Baseline value used when parameter is not varied.
        test_values: List of values to test for this parameter.
        is_categorical: Whether parameter is categorical (e.g., warning_type) vs numeric.
    """

    name: str
    enabled: bool
    base_value: float | str
    test_values: list[float | str]
    is_categorical: bool = False


@dataclass
class SensitivityRun:
    """Results from a single sensitivity analysis run.

    Args:
        run_id: Unique identifier for this parameter combination.
        parameters: Dictionary mapping parameter names to values used.
        forecast_date: Forecast initialization datetime.
        metrics: Dictionary of output metrics (e.g., n_warned, damage_avoided).
        computational_time_s: Time taken to run this combination (seconds).
        warned_postal_codes: List of postal codes that received warnings.
        postal_code_stats: Dictionary with detailed stats per postal code.
    """

    run_id: int
    parameters: dict[str, float | str]
    forecast_date: datetime.datetime
    metrics: dict[str, float]
    computational_time_s: float
    warned_postal_codes: list[str] = field(default_factory=list)
    postal_code_stats: dict[str, dict[str, float]] = field(default_factory=dict)


class SensitivityAnalyzer:
    """Manages sensitivity analysis runs for warning system parameters.

    This class handles configuration, execution, and result storage for
    sensitivity analyses. It supports different sampling methods (grid search,
    Latin Hypercube, Monte Carlo) and provides structured output.

    Args:
        households: The Households agent instance.
        config: Sensitivity configuration dictionary from model.yml.
    """

    def __init__(self, households: Households, config: dict[str, Any]) -> None:
        """Initialize the sensitivity analyzer.

        Args:
            households: The Households agent instance to run sensitivity analysis on.
            config: Sensitivity configuration from model.yml under warning_system.sensitivity.
        """
        self.households = households
        self.config = config
        self.method = config.get("method", "grid_search")

        # Parse parameter configurations
        self.parameters = self._parse_parameters(config["parameters"])

        # Setup output paths
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = (
            households.model.output_folder
            / config.get("output", {}).get("folder", "sensitivity_analysis")
            / f"run_{timestamp}"
        )
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Save configuration copy
        self._save_config()

        # Storage for results
        self.results: list[SensitivityRun] = []

    def _parse_parameters(
        self, params_config: dict[str, dict[str, Any]]
    ) -> list[SensitivityParameter]:
        """Parse parameter configurations from YAML.

        Args:
            params_config: Dictionary of parameter configurations.

        Returns:
            List of SensitivityParameter objects for enabled parameters.

        Raises:
            ValueError: If parameter configuration is missing required fields.
        """
        parameters = []
        for param_name, param_dict in params_config.items():
            if not param_dict.get("enabled", True):
                continue

            # Check if parameter is categorical (has string values)
            is_categorical = param_dict.get("type") == "categorical"

            test_values = param_dict.get("test_values")
            if test_values is None:
                # Generate from min/max/step if provided (numeric only)
                if is_categorical:
                    raise ValueError(
                        f"Categorical parameter {param_name} must have 'test_values'"
                    )

                min_val = param_dict.get("min")
                max_val = param_dict.get("max")
                step = param_dict.get("step")

                if all(v is not None for v in [min_val, max_val, step]):
                    test_values = list(np.arange(min_val, max_val + step / 2, step))
                else:
                    raise ValueError(
                        f"Parameter {param_name} must have either 'test_values' "
                        f"or 'min', 'max', and 'step'"
                    )

            parameters.append(
                SensitivityParameter(
                    name=param_name,
                    enabled=param_dict.get("enabled", True),
                    base_value=param_dict["base_value"],
                    test_values=test_values,
                    is_categorical=is_categorical,
                )
            )

        return parameters

    def _save_config(self) -> None:
        """Save a copy of the sensitivity configuration used."""
        config_path = self.output_folder / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def generate_parameter_combinations(self) -> list[dict[str, float]]:
        """Generate parameter combinations based on sampling method.

        Returns:
            List of dictionaries, each containing one parameter combination.

        Raises:
            ValueError: If an unsupported sampling method is specified.
        """
        if self.method == "grid_search":
            return self._grid_search()
        elif self.method == "latin_hypercube":
            return self._latin_hypercube_sampling()
        elif self.method == "monte_carlo":
            return self._monte_carlo_sampling()
        else:
            raise ValueError(f"Unsupported sampling method: {self.method}")

    def _grid_search(self) -> list[dict[str, float | str]]:
        """Generate full grid of parameter combinations.

        Returns:
            List of all possible parameter combinations (numeric and categorical).
        """
        param_names = [p.name for p in self.parameters]
        param_values = [p.test_values for p in self.parameters]

        combinations = []
        for combo in product(*param_values):
            combinations.append(dict(zip(param_names, combo)))

        return combinations

    def _latin_hypercube_sampling(self) -> list[dict[str, float | str]]:
        """Generate Latin Hypercube sample of parameter space.

        Returns:
            List of parameter combinations using LHS.

        Notes:
            Categorical parameters are sampled uniformly from their test_values,
            while numeric parameters use Latin Hypercube sampling.
        """
        n_samples = self.config.get("n_samples", 100)

        # Separate numeric and categorical parameters
        numeric_params = [p for p in self.parameters if not p.is_categorical]
        categorical_params = [p for p in self.parameters if p.is_categorical]

        # Generate LHS samples for numeric parameters
        if numeric_params:
            n_numeric = len(numeric_params)
            sampler = qmc.LatinHypercube(d=n_numeric, seed=42)
            numeric_samples = sampler.random(n=n_samples)
        else:
            numeric_samples = np.zeros((n_samples, 0))

        # Generate uniform random samples for categorical parameters
        rng = np.random.default_rng(seed=42)
        categorical_samples = []
        for _ in range(n_samples):
            cat_combo = {}
            for param in categorical_params:
                cat_combo[param.name] = rng.choice(param.test_values)
            categorical_samples.append(cat_combo)

        # Combine numeric and categorical samples
        combinations = []
        for i, sample in enumerate(numeric_samples):
            combo = categorical_samples[i].copy()
            for j, param in enumerate(numeric_params):
                # Map [0,1] to parameter range
                min_val = min(param.test_values)
                max_val = max(param.test_values)
                combo[param.name] = min_val + sample[j] * (max_val - min_val)
            combinations.append(combo)

        return combinations

    def _monte_carlo_sampling(self) -> list[dict[str, float | str]]:
        """Generate Monte Carlo sample of parameter space.

        Returns:
            List of random parameter combinations (numeric and categorical).
        """
        n_samples = self.config.get("n_samples", 100)
        rng = np.random.default_rng(seed=42)

        combinations = []
        for _ in range(n_samples):
            combo = {}
            for param in self.parameters:
                # Random sample from test values
                combo[param.name] = rng.choice(param.test_values)
            combinations.append(combo)

        return combinations

    def _extract_warned_postal_codes(
        self,
    ) -> tuple[list[str], dict[str, dict[str, float]]]:
        """Extract list of postal codes that received warnings and their statistics.

        Returns:
            Tuple of (warned_postal_codes, postal_code_stats) where:
            - warned_postal_codes: List of postal code strings
            - postal_code_stats: Dict mapping postal code to stats dict
        """
        # Get households with warnings
        warned_households_mask = self.households.var.warning_reached.data == 1

        if not hasattr(self.households.var, "household_points"):
            return [], {}

        # Get postal codes of warned households
        households_gdf = self.households.var.household_points
        if "postcode" not in households_gdf.columns:
            return [], {}

        # Filter to warned households using index
        warned_indices = np.where(warned_households_mask)[0]
        warned_postcodes_series = households_gdf.loc[warned_indices, "postcode"]

        # Get unique postal codes that were warned
        unique_warned_postcodes = warned_postcodes_series.dropna().unique().tolist()

        # Calculate statistics per postal code
        postal_code_stats = {}
        for postcode in unique_warned_postcodes:
            postcode_mask = households_gdf["postcode"] == postcode
            postcode_indices = households_gdf[postcode_mask].index

            # Count households in this postal code
            n_total_in_postcode = len(postcode_indices)
            n_warned_in_postcode = np.sum(warned_households_mask[postcode_indices])
            n_evacuated_in_postcode = np.sum(
                self.households.var.evacuated.data[postcode_indices]
            )

            postal_code_stats[str(postcode)] = {
                "n_households_total": float(n_total_in_postcode),
                "n_households_warned": float(n_warned_in_postcode),
                "n_households_evacuated": float(n_evacuated_in_postcode),
                "warning_coverage_pct": float(
                    n_warned_in_postcode / n_total_in_postcode * 100
                    if n_total_in_postcode > 0
                    else 0
                ),
            }

        return unique_warned_postcodes, postal_code_stats

    def run_single_combination(
        self,
        run_id: int,
        parameters: dict[str, float | str],
        forecast_date: datetime.datetime,
    ) -> SensitivityRun:
        """Execute warning system with specific parameter combination.

        Args:
            run_id: Unique identifier for this run.
            parameters: Dictionary of parameter values to use.
            forecast_date: Forecast initialization date to evaluate.

        Returns:
            SensitivityRun object containing results.
        """
        start_time = time.time()

        # Temporarily override config parameters
        original_config = {}
        for param_name, param_value in parameters.items():
            # Handle nested warning_type parameter (under strategies)
            if param_name == "warning_type":
                original_config[param_name] = self.households.model.config[
                    "agent_settings"
                ]["households"]["warning_system"]["strategies"][param_name]
                self.households.model.config["agent_settings"]["households"][
                    "warning_system"
                ]["strategies"][param_name] = param_value
            else:
                original_config[param_name] = self.households.model.config[
                    "agent_settings"
                ]["households"]["warning_system"][param_name]
                self.households.model.config["agent_settings"]["households"][
                    "warning_system"
                ][param_name] = param_value

        # Reset household warning states to ensure clean slate for each run
        self.households.var.warning_reached.data[:] = 0
        self.households.var.warning_level.data[:] = 0
        self.households.var.evacuated.data[:] = 0
        self.households.var.actions_taken.data[:] = 0
        self.households.var.recommended_measures.data[:] = 0
        self.households.var.warning_trigger.data[:] = 0

        # Run warning strategy (strategy is determined by config warning_type)
        self.households.water_level_warning_strategy(
            date_time=forecast_date, exceedance=True, strategy_id=1
        )

        # Run household decision-making to determine actual actions taken
        # This translates recommended_measures into actions_taken and sets evacuated status
        self.households.household_decision_making(date_time=forecast_date)

        # Collect metrics
        metrics = {
            "n_households_warned": np.sum(self.households.var.warning_reached.data),
            "n_households_evacuated": np.sum(self.households.var.evacuated.data),
            "n_possessions_elevated": np.sum(
                self.households.var.actions_taken.data[:, 0]
            ),
            "warning_coverage_pct": (
                np.sum(self.households.var.warning_reached.data)
                / len(self.households.var.warning_reached.data)
                * 100
            ),
        }

        # Extract warned postal codes and their statistics
        warned_postal_codes, postal_code_stats = self._extract_warned_postal_codes()
        metrics["n_postal_codes_warned"] = float(len(warned_postal_codes))

        # Restore original config
        for param_name, original_value in original_config.items():
            if param_name == "warning_type":
                self.households.model.config["agent_settings"]["households"][
                    "warning_system"
                ]["strategies"][param_name] = original_value
            else:
                self.households.model.config["agent_settings"]["households"][
                    "warning_system"
                ][param_name] = original_value

        computational_time = time.time() - start_time

        return SensitivityRun(
            run_id=run_id,
            parameters=parameters,
            forecast_date=forecast_date,
            metrics=metrics,
            computational_time_s=computational_time,
            warned_postal_codes=warned_postal_codes,
            postal_code_stats=postal_code_stats,
        )

    def run_sensitivity_analysis(
        self, forecast_dates: list[datetime.datetime]
    ) -> pd.DataFrame:
        """Execute full sensitivity analysis across parameter combinations.

        Args:
            forecast_dates: List of forecast dates to evaluate.

        Returns:
            DataFrame with all results.
        """
        combinations = self.generate_parameter_combinations()

        print(f"\n{'=' * 80}")
        print(f"Starting sensitivity analysis: {self.method}")
        print(f"Parameters: {[p.name for p in self.parameters]}")
        print(f"Combinations: {len(combinations)}")
        print(f"Forecast dates: {len(forecast_dates)}")
        print(f"Total runs: {len(combinations) * len(forecast_dates)}")
        print(f"{'=' * 80}\n")

        run_id = 1
        for forecast_date in forecast_dates:
            for combo in combinations:
                print(
                    f"Running combination {run_id}/{len(combinations) * len(forecast_dates)}"
                )
                print(f"  Forecast: {forecast_date}")
                print(f"  Parameters: {combo}")

                result = self.run_single_combination(run_id, combo, forecast_date)
                self.results.append(result)

                print(f"  Metrics: {result.metrics}")
                print(f"  Time: {result.computational_time_s:.2f}s\n")

                # Save intermediate results if configured
                if self.config.get("output", {}).get("save_intermediate", False):
                    self._save_intermediate_result(result)

                run_id += 1

        # Create summary DataFrame
        summary_df = self._create_summary_dataframe()

        # Save final results
        self._save_summary_results(summary_df)

        # Create spatial warning maps
        print("\nCreating spatial warning maps...")
        self.create_spatial_warning_map()  # Overview comparison

        return summary_df

    def _save_intermediate_result(self, result: SensitivityRun) -> None:
        """Save individual run result to parquet file.

        Args:
            result: The SensitivityRun to save.
        """
        detailed_folder = self.output_folder / "detailed_results"
        detailed_folder.mkdir(exist_ok=True)

        # Convert to DataFrame
        row = {**result.parameters, **result.metrics}
        row["run_id"] = result.run_id
        row["forecast_date"] = result.forecast_date
        row["computational_time_s"] = result.computational_time_s

        df = pd.DataFrame([row])
        output_path = detailed_folder / f"combination_{result.run_id:03d}.parquet"
        df.to_parquet(output_path, index=False)

        # Save postal code details separately
        if result.warned_postal_codes and result.postal_code_stats:
            self._save_postal_code_details(result)

    def _save_postal_code_details(self, result: SensitivityRun) -> None:
        """Save detailed postal code information for a run.

        Args:
            result: The SensitivityRun with postal code data.
        """
        postal_folder = self.output_folder / "postal_code_details"
        postal_folder.mkdir(exist_ok=True)

        # Create DataFrame with postal code statistics
        postal_rows = []
        for postcode, stats in result.postal_code_stats.items():
            row = {
                "run_id": result.run_id,
                "forecast_date": result.forecast_date,
                "postcode": postcode,
                **result.parameters,
                **stats,
            }
            postal_rows.append(row)

        if postal_rows:
            postal_df = pd.DataFrame(postal_rows)
            output_path = postal_folder / f"postal_codes_run_{result.run_id:03d}.csv"
            postal_df.to_csv(output_path, index=False)

    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Aggregate all results into summary DataFrame.

        Returns:
            DataFrame with all run results.
        """
        rows = []
        for result in self.results:
            row = {
                "run_id": result.run_id,
                "forecast_date": result.forecast_date,
                **result.parameters,
                **result.metrics,
                "computational_time_s": result.computational_time_s,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _save_summary_results(self, summary_df: pd.DataFrame) -> None:
        """Save summary results in multiple formats.

        Args:
            summary_df: Summary DataFrame to save.
        """
        output_format = self.config.get("output", {}).get("format", "parquet")

        if output_format in ["parquet", "both"]:
            output_path = self.output_folder / "summary_results.parquet"
            summary_df.to_parquet(output_path, index=False)
            print(f"Saved summary results to {output_path}")

        if output_format in ["csv", "both"]:
            output_path = self.output_folder / "summary_results.csv"
            summary_df.to_csv(output_path, index=False)
            print(f"Saved summary results to {output_path}")

    def create_spatial_warning_map(self, run_id: int | None = None) -> None:
        """Create spatial map showing which postal codes were warned.

        Args:
            run_id: Specific run ID to visualize. If None, creates overview of all runs.
        """
        try:
            import geopandas as gpd  # noqa: F401
            import matplotlib.pyplot as plt
        except ImportError:
            print("GeoPandas or matplotlib not available for spatial visualization")
            return

        # Load postal codes geometry
        if not hasattr(self.households, "postal_codes"):
            print("Postal codes geometry not available")
            return

        postal_codes = self.households.postal_codes.copy()

        viz_folder = self.output_folder / "visualizations" / "spatial_maps"
        viz_folder.mkdir(parents=True, exist_ok=True)

        if run_id is not None:
            # Visualize specific run
            result = next((r for r in self.results if r.run_id == run_id), None)
            if result is None or not result.warned_postal_codes:
                print(f"No warned postal codes found for run {run_id}")
                return

            # Mark warned postal codes
            postal_codes["warned"] = postal_codes["postcode"].isin(
                result.warned_postal_codes
            )

            # Create map
            fig, ax = plt.subplots(figsize=(12, 10))
            postal_codes.plot(
                column="warned",
                ax=ax,
                legend=True,
                cmap="RdYlGn_r",
                edgecolor="black",
                linewidth=0.5,
            )

            param_str = ", ".join([f"{k}={v}" for k, v in result.parameters.items()])
            ax.set_title(
                f"Run {run_id}: Warned Postal Codes\n{param_str}\n"
                f"Date: {result.forecast_date.strftime('%Y-%m-%d')}"
            )
            ax.set_axis_off()

            output_path = viz_folder / f"warned_postcodes_run_{run_id:03d}.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved spatial map to {output_path}")

        else:
            # Create comparison maps for different parameter values
            if "warning_type" in self.results[0].parameters:
                self._create_warning_type_comparison_maps(postal_codes, viz_folder)

    def _create_warning_type_comparison_maps(
        self, postal_codes: gpd.GeoDataFrame, viz_folder: Path
    ) -> None:
        """Create side-by-side comparison of warned postcodes for different warning types.

        Args:
            postal_codes: GeoDataFrame with postal code geometries.
            viz_folder: Folder to save visualizations.
        """
        import matplotlib.pyplot as plt

        # Group results by warning_type
        impact_based_runs = [
            r
            for r in self.results
            if r.parameters.get("warning_type") == "impact_based"
        ]
        flood_general_runs = [
            r
            for r in self.results
            if r.parameters.get("warning_type") == "flood_general"
        ]

        if not impact_based_runs or not flood_general_runs:
            return

        # Aggregate warned postal codes across all runs for each type
        impact_warned = set()
        for run in impact_based_runs:
            if run.warned_postal_codes:
                impact_warned.update(run.warned_postal_codes)

        flood_warned = set()
        for run in flood_general_runs:
            if run.warned_postal_codes:
                flood_warned.update(run.warned_postal_codes)

        # Create comparison
        postal_codes["impact_based"] = postal_codes["postcode"].isin(impact_warned)
        postal_codes["flood_general"] = postal_codes["postcode"].isin(flood_warned)
        postal_codes["both"] = (
            postal_codes["impact_based"] & postal_codes["flood_general"]
        )
        postal_codes["only_impact"] = (
            postal_codes["impact_based"] & ~postal_codes["flood_general"]
        )
        postal_codes["only_flood"] = (
            ~postal_codes["impact_based"] & postal_codes["flood_general"]
        )

        # Create side-by-side plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        postal_codes.plot(
            column="impact_based",
            ax=ax1,
            legend=True,
            cmap="RdYlGn_r",
            edgecolor="black",
            linewidth=0.3,
        )
        ax1.set_title("Impact-Based Warnings", fontsize=14, fontweight="bold")
        ax1.set_axis_off()

        postal_codes.plot(
            column="flood_general",
            ax=ax2,
            legend=True,
            cmap="RdYlGn_r",
            edgecolor="black",
            linewidth=0.3,
        )
        ax2.set_title("Flood-General Warnings", fontsize=14, fontweight="bold")
        ax2.set_axis_off()

        plt.suptitle(
            f"Comparison: Warned Postal Codes\n"
            f"Impact-based: {len(impact_warned)} postcodes | "
            f"Flood-general: {len(flood_warned)} postcodes | "
            f"Both: {len(impact_warned & flood_warned)} | "
            f"Difference: {len(impact_warned ^ flood_warned)}",
            fontsize=16,
            fontweight="bold",
        )

        output_path = viz_folder / "comparison_warned_postcodes.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison map to {output_path}")

    def create_visualizations(self, summary_df: pd.DataFrame) -> None:
        """Generate visualization plots for sensitivity analysis results.

        Args:
            summary_df: Summary DataFrame with all results.
        """
        import matplotlib.pyplot as plt

        viz_folder = self.output_folder / "visualizations"
        viz_folder.mkdir(exist_ok=True)

        # 1. Heatmaps for pairs of parameters (if grid search)
        if self.method == "grid_search" and len(self.parameters) >= 2:
            try:
                import seaborn as sns
            except ImportError:
                print("Seaborn not available for heatmap visualization")
                sns = None

            if sns is not None:
                param_names = [p.name for p in self.parameters]

            for i, param1 in enumerate(param_names[:-1]):
                for param2 in param_names[i + 1 :]:
                    for metric in self.config.get("output", {}).get(
                        "metrics", list(summary_df.columns)
                    ):
                        if metric not in summary_df.columns:
                            continue

                        # Create pivot table
                        pivot = summary_df.pivot_table(
                            values=metric,
                            index=param1,
                            columns=param2,
                            aggfunc="mean",
                        )

                        # Plot heatmap
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", ax=ax)
                        ax.set_title(f"{metric} vs {param1} and {param2}")

                        output_path = (
                            viz_folder / f"heatmap_{metric}_{param1}_vs_{param2}.png"
                        )
                        fig.savefig(output_path, dpi=300, bbox_inches="tight")
                        plt.close(fig)

        # 2. Parallel coordinates plot
        self._create_parallel_coordinates(summary_df, viz_folder)

        print(f"Visualizations saved to {viz_folder}")

    def _create_parallel_coordinates(
        self, summary_df: pd.DataFrame, output_folder: Path
    ) -> None:
        """Create parallel coordinates plot showing parameter relationships.

        Args:
            summary_df: Summary DataFrame with results.
            output_folder: Folder to save visualization.
        """
        import matplotlib.pyplot as plt
        from pandas.plotting import parallel_coordinates

        # Select columns to plot
        param_cols = [p.name for p in self.parameters]
        metric_col = self.config.get("output", {}).get("metrics", [])[0]

        if metric_col not in summary_df.columns:
            return

        # Normalize parameters and metric to [0, 1]
        plot_df = summary_df[param_cols + [metric_col]].copy()
        for col in plot_df.columns:
            min_val = plot_df[col].min()
            max_val = plot_df[col].max()
            if max_val > min_val:
                plot_df[col] = (plot_df[col] - min_val) / (max_val - min_val)

        # Create bins for metric to use as color classes
        plot_df["metric_bin"] = pd.qcut(
            plot_df[metric_col],
            q=5,
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
            duplicates="drop",
        )

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        parallel_coordinates(
            plot_df, "metric_bin", cols=param_cols, ax=ax, colormap="viridis"
        )
        ax.set_title(f"Parallel Coordinates: Impact on {metric_col}")
        ax.legend(title=metric_col, bbox_to_anchor=(1.05, 1), loc="upper left")

        output_path = output_folder / "parallel_coordinates.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
