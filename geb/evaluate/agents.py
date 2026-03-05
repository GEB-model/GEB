"""Module implementing agent evaluation functions for the GEB model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from geb.workflows.io import read_zarr

if TYPE_CHECKING:
    from geb.evaluate import Evaluate
    from geb.model import GEBModel


class Agents:
    """Implements several functions to evaluate the agent-based module of GEB."""

    def __init__(self, model: GEBModel, evaluator: Evaluate) -> None:
        """Initialize the Agents evaluation module."""
        self.model = model
        self.evaluator = evaluator

    def evaluate_household_adaptation(
        self,
        spinup_name: str = "spinup",
        run_name: str = "default",
        include_spinup: bool = False,
        include_yearly_plots: bool = True,
        correct_discharge_observations: bool = False,
    ) -> dict[str, Any]:
        """Evaluate the household adaptation module of GEB.

        Compares simulated household adaptation (dry- and wetproofing measures)
        with observed adoption rates. Reads saved simulation data
        and compares it against observed values from the configuration.

        Returns:
            Dictionary containing:
                - simulated_data: DataFrame with simulated dryproofing and wetproofing counts
                - observed_data: DataFrame with observed adaptation rates
                - ratios: DataFrame with simulated/observed ratios for each measure
                - summary: Dict with aggregate statistics (mean ratios, counts)

        Note:
            Observed data must be configured in the model config under
            agent_settings.households with 'observed_adaptation' containing:
            - timestamps: list of datetime strings or indices
            - dryproofing_percentage: percentage of households with dryproofing
            - wetproofing_percentage: percentage of households with wetproofing
        """
        # Unused in this evaluator but accepted for compatibility with Evaluate.run
        _ = (
            spinup_name,
            include_spinup,
            include_yearly_plots,
            correct_discharge_observations,
        )
        print("Evaluating household adaptation...")

        # Load simulated adaptation data
        simulated_df = self._load_simulated_adaptation_data(run_name)
        if simulated_df is None or simulated_df.empty:
            return {"error": "No simulated adaptation data found"}

        print(simulated_df)

        # Load observed adaptation data from config
        observed_df, config_total_households = self._load_observed_adaptation_data()
        if observed_df is None or observed_df.empty:
            return {"error": "No observed adaptation data configured"}

        # Align the two datasets temporally
        aligned_sim, aligned_obs = self._align_datasets(simulated_df, observed_df)

        print(aligned_sim)
        print(aligned_obs)

        # Determine total households (argument > observed_adaptation config > fallback)
        if config_total_households is not None:
            total_households = int(config_total_households)
        else:
            total_households = self._get_total_households(simulated_df)

        # Calculate ratios (simulated / observed) 1 indicates a perfect match, >1 indicates overestimation, <1 indicates underestimation
        ratios = self._calculate_adaptation_ratios(
            aligned_sim, aligned_obs, total_households
        )

        print(ratios)

        # Save ratios to CSV in evaluate folder
        self._save_ratios_to_csv(ratios, run_name)

        # Calculate summary statistics and get the balanced score
        balanced_ratio_score = self._calculate_summary_statistics(
            ratios, aligned_sim, aligned_obs
        )
        print(balanced_ratio_score)
        return {
            "balanced_ratio_score": balanced_ratio_score,
        }

    def _get_total_households(self, simulated_df: pd.DataFrame) -> int:
        """Get total number of households from config or simulated data.

        Tries to load from config first, falls back to calculating from simulated data.

        Args:
            simulated_df: Simulated adaptation data

        Returns:
            Total number of households

        Raises:
            ValueError: If total households cannot be determined from config or simulated data.
        """
        # Try to get from config first
        try:
            households_config = self.model.config.get("agent_settings", {}).get(
                "households", {}
            )
            if "total_households" in households_config:
                total = households_config.get("total_households")
                if total is not None and total > 0:
                    return int(total)
        except (KeyError, TypeError, ValueError):
            pass

        # Fall back to calculating from simulated data
        # Sum of all adaptation categories in the first timestep
        if (
            "dryproofing" in simulated_df.columns
            and "wetproofing" in simulated_df.columns
            and "not_adapting" in simulated_df.columns
        ):
            total = (
                simulated_df[["dryproofing", "wetproofing", "not_adapting"]]
                .iloc[0]
                .sum()
            )
            return int(total)

        # If columns don't match, try summing all numeric columns except time
        numeric_cols = simulated_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return int(simulated_df[numeric_cols].iloc[0].sum())

        raise ValueError(
            "Could not determine total households from config or simulated data"
        )

    def _load_simulated_adaptation_data(self, run_name: str) -> pd.DataFrame | None:
        """Load saved simulation data on household adaptations from reporter output.

        Tries to load from reporter's zarr files.
        The reporter should be configured to save household adaptation data.

        Returns:
            DataFrame with columns: time, dryproofing, wetproofing, not_adapting
            or None if data not found.
        """
        adaptation = read_zarr(
            self.model.output_folder
            / "report"
            / run_name
            / "agents.households"
            / "adaptation_type.zarr"
        )

        # Convert from dask array to DataFrame by aggregating per timestep
        times = adaptation.coords["time"].values
        values = adaptation.values  # This loads dask array

        # Count adaptation types per timestep
        # adaptation_type: 0=not adapted, 1=dry-proofed, 2=wet-proofed
        dryproofing = []
        wetproofing = []
        not_adapting = []

        print(f"Aggregating {len(times)} timesteps...")
        for t in range(len(times)):
            # Get timestep data (will compute this chunk of dask array)
            timestep_data = values[t, :]

            # Count each adaptation type
            dryproofing.append(int(np.sum(timestep_data == 1)))
            wetproofing.append(int(np.sum(timestep_data == 2)))
            not_adapting.append(int(np.sum(timestep_data == 0)))

        df = pd.DataFrame(
            {
                "time": pd.to_datetime(times),
                "dryproofing": dryproofing,
                "wetproofing": wetproofing,
                "not_adapting": not_adapting,
            }
        )

        return df

    def _load_observed_adaptation_data(self) -> tuple[pd.DataFrame | None, int | None]:
        """Load observed adaptation data from model configuration.

        The configuration should contain observed adaptation rates at specific
        timestamps as percentages. Also loads total_households if provided.

        Returns:
            Tuple of (DataFrame with observed data, total_households or None)
            DataFrame has columns: time, dryproofing_pct, wetproofing_pct
        """
        households_config = self.model.config.get("agent_settings", {}).get(
            "households", {}
        )
        observed_config = households_config.get("observed_adaptation", {})

        if not observed_config:
            print("No observed adaptation data configured.")

        timestamps = observed_config.get("timestamps", [])
        dryproof_pct = observed_config.get("dryproofing_percentage", [])
        wetproof_pct = observed_config.get("wetproofing_percentage", [])
        total_hh = observed_config.get("total_households", None)

        # Convert timestamps to datetime
        times = pd.to_datetime(timestamps)

        df = pd.DataFrame(
            {
                "time": times,
                "dryproofing_pct": dryproof_pct,
                "wetproofing_pct": wetproof_pct,
            }
        )

        print(df)
        return df, total_hh

    def _align_datasets(
        self, simulated_df: pd.DataFrame, observed_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align simulated and observed datasets temporally using exact date matching.

        Only keeps dates that exist in both datasets (inner join).

        Args:
            simulated_df: DataFrame with simulated data
            observed_df: DataFrame with observed data

        Returns:
            Tuple of (aligned_simulated, aligned_observed) DataFrames,
            containing only exact date matches between both datasets
        """
        # Merge on exact time match (inner join)
        merged = pd.merge(
            simulated_df,
            observed_df,
            on="time",
            how="inner",
        )

        if merged.empty:
            print(
                "Warning: No exact date matches found between simulated and observed data."
            )
            return pd.DataFrame(), pd.DataFrame()

        # Split back into simulated and observed columns
        sim_cols = ["time", "dryproofing", "wetproofing", "not_adapting"]
        obs_cols = ["time", "dryproofing_pct", "wetproofing_pct"]

        sim_aligned = merged[sim_cols].reset_index(drop=True)
        obs_aligned = merged[obs_cols].reset_index(drop=True)

        print(
            f"Exact date match: found {len(sim_aligned)} dates with data in both simulated and observed"
        )

        return sim_aligned, obs_aligned

    def _calculate_adaptation_ratios(
        self,
        simulated_df: pd.DataFrame,
        observed_df: pd.DataFrame,
        total_households: int,
    ) -> pd.DataFrame:
        """Calculate ratio of simulated to observed adaptation.

        Args:
            simulated_df: Simulated data with absolute household counts
            observed_df: Observed data with percentages
            total_households: Total number of households

        Returns:
            DataFrame with calculated ratios for dryproofing and wetproofing
        """
        # Convert observed percentages to household counts
        obs_dry_count = (observed_df["dryproofing_pct"] / 100) * total_households
        obs_wet_count = (observed_df["wetproofing_pct"] / 100) * total_households

        # Calculate ratios (simulated / observed)
        # Avoid division by zero
        dry_ratio = np.divide(
            simulated_df["dryproofing"].values,
            obs_dry_count.values,
            where=obs_dry_count.values != 0,
            out=np.full_like(simulated_df["dryproofing"].values, np.nan, dtype=float),
        )

        wet_ratio = np.divide(
            simulated_df["wetproofing"].values,
            obs_wet_count.values,
            where=obs_wet_count.values != 0,
            out=np.full_like(simulated_df["wetproofing"].values, np.nan, dtype=float),
        )

        ratios_df = pd.DataFrame(
            {
                "time": simulated_df["time"],
                "dryproofing_ratio": dry_ratio,
                "wetproofing_ratio": wet_ratio,
                "simulated_dryproofing": simulated_df["dryproofing"],
                "observed_dryproofing_count": obs_dry_count,
                "simulated_wetproofing": simulated_df["wetproofing"],
                "observed_wetproofing_count": obs_wet_count,
            }
        )

        # Round numeric columns to 2 decimals
        numeric_cols = ratios_df.select_dtypes(include=[np.number]).columns
        ratios_df[numeric_cols] = ratios_df[numeric_cols].round(2)

        return ratios_df

    def _save_ratios_to_csv(self, ratios: pd.DataFrame, run_name: str) -> None:
        """Save adaptation ratios to CSV file in evaluate folder.

        Args:
            ratios: DataFrame with adaptation ratios over time
            run_name: Name of the model run
        """
        # Create output folder
        output_folder = self.evaluator.output_folder_evaluate / "agents"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Save ratios to CSV
        output_path = output_folder / f"household_adaptation_ratios_{run_name}.csv"
        ratios.to_csv(output_path, index=False)
        print(f"Saved adaptation ratios to {output_path}")

    def _calculate_balanced_ratio_metric(
        self, ratios_df: pd.DataFrame, ratio_columns: list[str] | None = None
    ) -> dict[str, float | int]:
        """Calculate one balanced metric across any number of ratio columns and timesteps.

        This metric is robust to different table sizes (e.g., 2 ratios x 2 timesteps,
        or many more). It treats over- and underestimation symmetrically using
        absolute log-deviation from 1.

        Args:
            ratios_df: DataFrame containing ratio columns.
            ratio_columns: Optional explicit ratio columns. If None, all columns
                ending with "_ratio" are used.

        Returns:
            Dictionary with aggregated balanced metrics.
        """
        if ratio_columns is None:
            ratio_columns = [c for c in ratios_df.columns if c.endswith("_ratio")]

        if len(ratio_columns) == 0:
            return {
                "balanced_ratio_error": np.nan,
                "balanced_ratio_score": np.nan,
                "balanced_ratio_geomean": np.nan,
                "n_ratio_values": 0,
            }

        # Flatten all ratio values across selected columns and all timesteps
        ratio_values = ratios_df[ratio_columns].to_numpy(dtype=float).ravel()

        # Keep only finite positive values (needed for log transform)
        valid = ratio_values[np.isfinite(ratio_values) & (ratio_values > 0)]
        if valid.size == 0:
            return {
                "balanced_ratio_error": np.nan,
                "balanced_ratio_score": np.nan,
                "balanced_ratio_geomean": np.nan,
                "n_ratio_values": 0,
            }

        # = 1 when perfect agreement, > 1 otherwise
        balanced_error = float(np.exp(np.mean(np.abs(np.log(valid)))))
        # Bounded score in (0, 1], where 1 is perfect agreement
        balanced_score = float(1.0 / balanced_error)
        # Central tendency of ratios (can indicate over/under bias)
        balanced_geomean = float(np.exp(np.mean(np.log(valid))))

        return {
            "balanced_ratio_error": round(balanced_error, 2),
            "balanced_ratio_score": round(balanced_score, 2),
            "balanced_ratio_geomean": round(balanced_geomean, 2),
            "n_ratio_values": int(valid.size),
        }

    def _calculate_summary_statistics(
        self,
        ratios_df: pd.DataFrame,
        simulated_df: pd.DataFrame,
        observed_df: pd.DataFrame,
    ) -> float:
        """Calculate summary statistics for adaptation evaluation.

        Computes per-measure statistics (mean, std, median ratios, totals) and
        a balanced aggregate metric across all ratio columns and timesteps.

        Returns:
            The balanced_ratio_score (float in (0, 1], where 1 = perfect agreement).
        """
        # Calculate per-measure statistics (for internal tracking/debugging)
        dry_ratio = ratios_df["dryproofing_ratio"].replace([np.inf, -np.inf], np.nan)
        wet_ratio = ratios_df["wetproofing_ratio"].replace([np.inf, -np.inf], np.nan)

        dry_ratio_valid = dry_ratio.dropna()
        wet_ratio_valid = wet_ratio.dropna()

        dry_stats = {
            "mean_ratio": round(float(dry_ratio_valid.mean()), 2)
            if not dry_ratio_valid.empty
            else np.nan,
            "std_ratio": round(float(dry_ratio_valid.std()), 2)
            if not dry_ratio_valid.empty
            else np.nan,
            "median_ratio": round(float(dry_ratio_valid.median()), 2)
            if not dry_ratio_valid.empty
            else np.nan,
            "simulated_total": int(simulated_df["dryproofing"].sum()),
            "observed_total": round(float(observed_df["dryproofing_pct"].sum()), 2),
        }

        wet_stats = {
            "mean_ratio": round(float(wet_ratio_valid.mean()), 2)
            if not wet_ratio_valid.empty
            else np.nan,
            "std_ratio": round(float(wet_ratio_valid.std()), 2)
            if not wet_ratio_valid.empty
            else np.nan,
            "median_ratio": round(float(wet_ratio_valid.median()), 2)
            if not wet_ratio_valid.empty
            else np.nan,
            "simulated_total": int(simulated_df["wetproofing"].sum()),
            "observed_total": round(float(observed_df["wetproofing_pct"].sum()), 2),
        }

        # Calculate balanced metric across all ratio columns
        balanced = self._calculate_balanced_ratio_metric(
            ratios_df, ratio_columns=["dryproofing_ratio", "wetproofing_ratio"]
        )

        # Print summary for logging
        print(f"Dryproofing stats: {dry_stats}")
        print(f"Wetproofing stats: {wet_stats}")
        print(f"Balanced metric: {balanced}")

        # Return only the balanced score
        return balanced["balanced_ratio_score"]
