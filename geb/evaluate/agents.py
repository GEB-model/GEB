"""Module implementing agent evaluation functions for the GEB model.

This function allows to evaluate floodproofing and windstorm-shutter uptake across GEB clusters.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import zarr

if TYPE_CHECKING:
    from geb.evaluate import Evaluate
    from geb.model import GEBModel

LECZ_REL_PATH = Path("input/geom/coastal/low_elevation_coastal_zone_mask.geoparquet")
HOUSEHOLD_OUTPUT_REL_PATH = Path("output")
HOUSEHOLD_TABLE_REL_PATH = Path("buildings_each_step")
HOUSEHOLD_LOCATION_REL_PATH = Path("input/array/agents/households/location.zarr")


class Agents:
    """Implements several functions to evaluate the agent-based module of GEB."""

    def __init__(self, model: GEBModel, evaluator: Evaluate) -> None:
        """Initialize the Agents evaluation module."""
        self.model = model
        self.evaluator = evaluator

    def evaluate_household_adaptation(
        self,
        spinup_name: str = "spinup",
        run_name: str | None = None,
        include_spinup: bool = False,
        include_yearly_plots: bool = True,
        correct_discharge_observations: bool = False,  # NEED-CHECK
        flood_snapshot: str | None = None,
        wind_snapshot: str | None = None,
        baseline_snapshot: str | None = None,
        map_realization: str | None = None,
    ) -> dict[str, Any]:
        """Compares simulated household adaptation (dry- and wetproofing measures)
        with observed adoption rates. Reads saved simulation data
        and compares it against observed values from the configuration.

        Returns:
            Dictionary containing:
            - balanced_ratio_score: A summary metric of how well the simulated adaptation matches observed data, where 1 is a perfect match and values <1 indicate underestimation

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

        settings = self._settings()  # this lines are different, why?
        run_name = run_name or settings.get("run_name", "default")

        simulated_df = self._load_simulated_adaptation_data(
            run_name,
            flood_snapshot=flood_snapshot,
            wind_snapshot=wind_snapshot,
        )
        observed_df, _ = self._load_simulated_adaptation_data(simulated_df)
        aligned_sim, aligned_obs = self._align_datasets(simulated_df, observed_df)
        ratios = self._calculated_adaptation_ratios(aligned_sim, aligned_obs)

        # This part is different NEED-CHECK
        map_realization = map_realization or settings.get("map_realization")
        ratios["map_realization"] = map_realization
        self._save_ratios_to_csv(ratios, run_name)
        result = self._calculate_summary_statistics(ratios, simulated_df, observed_df)
        if map_realization is not None:
            result["map_realization"] = map_realization

        baseline_snapshot = baseline_snapshot or settings.get("baseline_snapshot")
        if baseline_snapshot is not None:
            baseline = self._load_simulated_adaptation_data(
                run_name,
                flood_snapshot=baseline_snapshot,
                wind_snapshot=baseline_snapshot,
            )
            baseline_flood = float(
                baseline.loc[baseline["hazard"] == "flood", "simulated_fraction"].iloc[
                    0
                ]
            )
            baseline_wind = float(
                baseline.loc[baseline["hazard"] == "wind", "simulated_fraction"].iloc[0]
            )
            result.update(
                {
                    "baseline_snapshot": baseline.attrs["snapshots"]["flood"],
                    "baseline_flood_fraction": baseline_flood,
                    "baseline_wind_fraction": baseline_wind,
                    "flood_fractions_change_since_baseline": (
                        result["flood_simulated_fraction"] - baseline_flood
                    ),
                    "wind_fraction_change_since_baseline": (
                        result["wind_simulated_fraction"] - baseline_wind
                    ),
                }
            )
        return result

        # # Load simulated adaptation data
        # simulated_df = self._load_simulated_adaptation_data(run_name)
        # if simulated_df is None or simulated_df.empty:
        #     return {"error": "No simulated adaptation data found"}

        # # Load observed adaptation data from config
        # observed_df, config_total_households = self._load_observed_adaptation_data()
        # if observed_df is None or observed_df.empty:
        #     return {"error": "No observed adaptation data configured"}

        # # Align the two datasets temporally
        # aligned_sim, aligned_obs = self._align_datasets(simulated_df, observed_df)

        # # Determine total households (observed_adaptation config > fallback)
        # if config_total_households is not None:
        #     total_households = int(config_total_households)
        # else:
        #     total_households = self._get_total_households(simulated_df)

        # # Calculate ratios (simulated / observed) 1 indicates a perfect match, >1 indicates overestimation, <1 indicates underestimation
        # ratios = self._calculate_adaptation_ratios(
        #     aligned_sim, aligned_obs, total_households
        # )

        # print(ratios)

        # # Save ratios to CSV in evaluate folder
        # self._save_ratios_to_csv(ratios, run_name)

        # # Calculate summary statistics and get the balanced score
        # balanced_ratio_score = self._calculate_summary_statistics(
        #     ratios, aligned_sim, aligned_obs
        # )
        # print(balanced_ratio_score)
        # return {
        #     "balanced_ratio_score": balanced_ratio_score,
        # }

    def _get_total_households(
        self,
        simulated_df: pd.DataFrame,
        hazard: str | None = None,
    ) -> int:
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
        # try:
        #     households_config = self.model.config.get("agent_settings", {}).get(
        #         "households", {}
        #     )
        #     if "total_households" in households_config:
        #         total = households_config.get("total_households")
        #         if total is not None and total > 0:
        #             return int(total)
        # except (KeyError, TypeError, ValueError):
        #     passi

        # Fall back to calculating from simulated data
        # Sum of all adaptation categories in the first timestep
        # if (
        #     "dryproofing" in simulated_df.columns
        #     and "wetproofing" in simulated_df.columns
        #     and "not_adapting" in simulated_df.columns
        # ):
        #     total = (
        #         simulated_df[["dryproofing", "wetproofing", "not_adapting"]]
        #         .iloc[0]
        #         .sum()
        #     )
        #     return int(total)

        # # If columns don't match, try summing all numeric columns except time
        # numeric_cols = simulated_df.select_dtypes(include=[np.number]).columns
        # if len(numeric_cols) > 0:
        #     return int(simulated_df[numeric_cols].iloc[0].sum())

        # raise ValueError(
        #     "Could not determine total households from config or simulated data"
        # )

        selected = simulated_df
        if hazard is not None:
            selected = selected.loc[selected["hazard"] == hazard]
        if selected.empty:
            raise ValueError("No simulated household totals are available.")
        return int(selected["total_households"].iloc[0])

    def _load_simulated_adaptation_data(
        self,
        run_name: str,
        flood_snapshot: str | None = None,
        wind_snapshot: str | None = None,
    ) -> pd.DataFrame:
        """Load saved simulation data on household adaptations from reporter output.

        Tries to load from reporter's zarr files.
        The reporter should be configured to save household adaptation data.

        Returns:
            DataFrame with columns: time, dryproofing, wetproofing, not_adapting
            or None if data not found.
        """
        # adaptation = read_zarr(
        #     self.model.output_folder
        #     / "report"
        #     / run_name
        #     / "agents.households"
        #     / "adaptation_type.zarr"
        # )

        # print(adaptation)

        # Convert from dask array to DataFrame by aggregating per timestep
        #
        settings = self._settings()
        cluster_dirs = [Path(p) for p in settings.get("cluster_base_dirs", [])]
        if not cluster_dirs:
            raise ValueError(
                "Configure adaptation_evaluation.cluster_base_dirs with every cluster."
            )

        snapshots_by_cluster = self._snapshot_files(cluster_dirs, run_name)
        common = sorted(set.intersection(*snapshots_by_cluster.values()))
        if not common:
            raise FileNotFoundError("No common household snapshots across clusters.")
        flood_snapshot = self._resolve_snapshot(
            "flood", flood_snapshot or settings.get("flood_snapshot"), settings, common
        )
        wind_snapshot = self._resolve_snapshot(
            "wind", wind_snapshot or settings.get("wind_snapshot"), settings, common
        )
        lecz = self._load_lecz(cluster_dirs)

        records: list[dict[str, Any]] = []
        cluster_diagnostics: list[tuple[str, str, list[dict[str, Any]]]] = []
        for hazard, snapshot in (("flood", flood_snapshot), ("wind", wind_snapshot)):
            cluster_counts = self._aggregate_snapshot(
                cluster_dirs, run_name, snapshot, lecz
            )
            if hazard == "flood":
                adapted_count = sum(row["flood_adapted"] for row in cluster_counts)
            else:
                adapted_count = sum(row["wind_adapted"] for row in cluster_counts)
            total_households = sum(row["lecz_households"] for row in cluster_counts)
            records.append(
                {
                    "hazard": hazard,
                    "time": pd.to_datetime(snapshot, format="%Y%m%d"),
                    "snapshot": snapshot,
                    "simulated_count": int(adapted_count),
                    "total_households": int(total_households),
                    "simulated_fraction": adapted_count / total_households,
                }
            )
            cluster_diagnostics.append((snapshot, hazard, cluster_counts))

            simulated = pd.DataFrame(records)
            simulated.attrs["snapshots"] = {
                "flood": flood_snapshot,
                "wind": wind_snapshot,
            }
            simulated.attrs["cluster_diagnostics"] = cluster_diagnostics
            return simulated

    def _load_observed_adaptation_data(self) -> tuple[pd.DataFrame | None, int | None]:
        """Load observed adaptation data from model configuration.

        The configuration should contain observed adaptation rates at specific
        timestamps as percentages. Also loads total_households if provided.

        Returns:
            Tuple of (DataFrame with observed data, total_households or None)
            DataFrame has columns: time, dryproofing_pct, wetproofing_pct
        """
        settings = self._settings()
        targets = {
            "flood": self._fraction_setting(settings, "target_flood_fraction"),
            "wind": self._fraction_setting(settings, "target_wind_fraction"),
        }
        snapshots = simulated_df.attrs["snapshots"]
        observed = pd.DataFrame(
            [
                {
                    "hazard": hazard,
                    "time": pd.to_datetime(snapshot, format="%Y%m%d"),
                    "observed_fraction": target,
                }
            ]
            for hazard, target in targets.items()
            for snapshot in [snapshots[hazard]]
        )

        return observed, None

        # households_config = self.model.config.get("agent_settings", {}).get(
        #     "households", {}
        # )
        # observed_config = households_config.get("observed_adaptation", {})

        # if not observed_config:
        #     print("No observed adaptation data configured.")

        # timestamps = observed_config.get("timestamps", [])
        # dryproof_pct = observed_config.get("dryproofing_percentage", [])
        # wetproof_pct = observed_config.get("wetproofing_percentage", [])
        # total_hh = observed_config.get("total_households", None)

        # # Convert timestamps to datetime
        # times = pd.to_datetime(timestamps)

        # df = pd.DataFrame(
        #     {
        #         "time": times,
        #         "dryproofing_pct": dryproof_pct,
        #         "wetproofing_pct": wetproof_pct,
        #     }
        # )

        # return df, total_hh

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
        # merged = pd.merge(
        #     simulated_df,
        #     observed_df,
        #     on="time",
        #     how="inner",
        # )

        # if merged.empty:
        #     print(
        #         "Warning: No exact date matches found between simulated and observed data."
        #     )
        #     return pd.DataFrame(), pd.DataFrame()
        merged = simulated_df.merge(
            observed_df, on=["hazard", "time"], how="inner", validate="one_to_one"
        )
        if merged.empty or len(merged) != 2:
            raise ValueError(
                "Could not align both hazard targets with simulated snapshots"
            )
        return merged, merged[
            ["hazard", "time", "observed_fraction"]
        ].copy()  # NEED-CHECK why double []

        # # Split back into simulated and observed columns
        # sim_cols = ["time", "dryproofing", "wetproofing", "not_adapting"]
        # obs_cols = ["time", "dryproofing_pct", "wetproofing_pct"]

        # sim_aligned = merged[sim_cols].reset_index(drop=True)
        # obs_aligned = merged[obs_cols].reset_index(drop=True)

        # return sim_aligned, obs_aligned

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
        # obs_dry_count = (observed_df["dryproofing_pct"] / 100) * total_households
        # obs_wet_count = (observed_df["wetproofing_pct"] / 100) * total_households

        # # Calculate ratios (simulated / observed)
        # # Avoid division by zero
        # dry_ratio = np.divide(
        #     simulated_df["dryproofing"].values,
        #     obs_dry_count.values,
        #     where=obs_dry_count.values != 0,
        #     out=np.full_like(simulated_df["dryproofing"].values, np.nan, dtype=float),
        # )

        # wet_ratio = np.divide(
        #     simulated_df["wetproofing"].values,
        #     obs_wet_count.values,
        #     where=obs_wet_count.values != 0,
        #     out=np.full_like(simulated_df["wetproofing"].values, np.nan, dtype=float),
        # )

        # ratios_df = pd.DataFrame(
        #     {
        #         "time": simulated_df["time"],
        #         "dryproofing_ratio": dry_ratio,
        #         "wetproofing_ratio": wet_ratio,
        #         "simulated_dryproofing": simulated_df["dryproofing"],
        #         "observed_dryproofing_count": obs_dry_count,
        #         "simulated_wetproofing": simulated_df["wetproofing"],
        #         "observed_wetproofing_count": obs_wet_count,
        #     }
        # )

        # # Round numeric columns to 2 decimals
        # numeric_cols = ratios_df.select_dtypes(include=[np.number]).columns
        # ratios_df[numeric_cols] = ratios_df[numeric_cols].round(2)

        _ = total_householdsratios = simulated_df.copy()
        ratios = simulated_df.copy()
        ratios["observed_fraction"] = observed_df["observed_fraction"].to_numpy()
        ratios["target_count"] = (
            ratios["observed_fraction"] * ratios["total_households"]
        )
        ratios["fraction_error"] = (
            ratios["simulated_count"] - ratios["target_count"]
        ).abs()
        ratios["count_error"] = (
            ratios["simulated_count"] - ratios["target_count"]
        ).abs()
        ratios["score"] = 1.0 - ratios["fraction_error"]

        return ratios

    def _save_ratios_to_csv(self, ratios: pd.DataFrame, run_name: str) -> None:
        """Save adaptation ratios to CSV file in evaluate folder.

        Args:
            ratios: DataFrame with adaptation ratios over time
            run_name: Name of the model run
        """
        # Create output folder
        output_folder = Path(self.evaluator.output_folder_evaluate) / "agents"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Save ratios to CSV
        ratios.to_csv(
            output_folder / f"household_adaptation_objectives_{run_name}.csv",
            index=False,
        )

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
        # if ratio_columns is None:
        #     ratio_columns = [c for c in ratios_df.columns if c.endswith("_ratio")]

        # if len(ratio_columns) == 0:
        #     return {
        #         "balanced_ratio_error": np.nan,
        #         "balanced_ratio_score": np.nan,
        #         "balanced_ratio_geomean": np.nan,
        #         "n_ratio_values": 0,
        #     }

        # # Flatten all ratio values across selected columns and all timesteps
        # ratio_values = ratios_df[ratio_columns].to_numpy(dtype=float).ravel()

        # # Keep only finite positive values (needed for log transform)
        # valid = ratio_values[np.isfinite(ratio_values) & (ratio_values > 0)]
        # if valid.size == 0:
        #     return {
        #         "balanced_ratio_error": np.nan,
        #         "balanced_ratio_score": np.nan,
        #         "balanced_ratio_geomean": np.nan,
        #         "n_ratio_values": 0,
        #     }

        # # = 1 when perfect agreement, > 1 otherwise
        # balanced_error = float(np.exp(np.mean(np.abs(np.log(valid)))))
        # # Bounded score in (0, 1], where 1 is perfect agreement
        # balanced_score = float(1.0 / balanced_error)
        # # Central tendency of ratios (can indicate over/under bias)
        # balanced_geomean = float(np.exp(np.mean(np.log(valid))))

        # return {
        #     "balanced_ratio_error": round(balanced_error, 2),
        #     "balanced_ratio_score": round(balanced_score, 2),
        #     "balanced_ratio_geomean": round(balanced_geomean, 2),
        #     "n_ratio_values": int(valid.size),
        # }
        _ = ratio_columns
        metrics: dict[str, float | int] = {}
        for hazard in ("flood", "wind"):
            rows = ratios_df.loc[ratios_df["hazard"] == hazard]
            if rows.empty:
                raise ValueError(f"No aligned {hazard} adaptation objective")
            error = float(rows["fraction_error"].mean())
            metrics[f"{hazard}_adaptation_objective"] = error
            metrics[f"{hazard}_adaptation_score"] = 1.0 - error

        return metrics

    def _calculate_summary_statistics(
        self,
        ratios_df: pd.DataFrame,
        simulated_df: pd.DataFrame,
        observed_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate summary statistics for adaptation evaluation.

        Computes per-measure statistics (mean, std, median ratios, totals) and
        a balanced aggregate metric across all ratio columns and timesteps.

        Returns:
            The balanced_ratio_score (float in (0, 1], where 1 = perfect agreement).
        """
        # Calculate per-measure statistics (for internal tracking/debugging)
        # dry_ratio = ratios_df["dryproofing_ratio"].replace([np.inf, -np.inf], np.nan)
        # wet_ratio = ratios_df["wetproofing_ratio"].replace([np.inf, -np.inf], np.nan)

        # dry_ratio_valid = dry_ratio.dropna()
        # wet_ratio_valid = wet_ratio.dropna()

        # dry_stats = {
        #     "mean_ratio": round(float(dry_ratio_valid.mean()), 2)
        #     if not dry_ratio_valid.empty
        #     else np.nan,
        #     "std_ratio": round(float(dry_ratio_valid.std()), 2)
        #     if not dry_ratio_valid.empty
        #     else np.nan,
        #     "median_ratio": round(float(dry_ratio_valid.median()), 2)
        #     if not dry_ratio_valid.empty
        #     else np.nan,
        #     "simulated_total": int(simulated_df["dryproofing"].sum()),
        #     "observed_total": round(float(observed_df["dryproofing_pct"].sum()), 2),
        # }

        # wet_stats = {
        #     "mean_ratio": round(float(wet_ratio_valid.mean()), 2)
        #     if not wet_ratio_valid.empty
        #     else np.nan,
        #     "std_ratio": round(float(wet_ratio_valid.std()), 2)
        #     if not wet_ratio_valid.empty
        #     else np.nan,
        #     "median_ratio": round(float(wet_ratio_valid.median()), 2)
        #     if not wet_ratio_valid.empty
        #     else np.nan,
        #     "simulated_total": int(simulated_df["wetproofing"].sum()),
        #     "observed_total": round(float(observed_df["wetproofing_pct"].sum()), 2),
        # }

        # # Calculate balanced metric across all ratio columns
        # balanced = self._calculate_balanced_ratio_metric(
        #     ratios_df, ratio_columns=["dryproofing_ratio", "wetproofing_ratio"]
        # )

        # # Return only the balanced score
        # return balanced["balanced_ratio_score"]
        _ = observed_df
        result: dict[str, Any] = self._calculate_balanced_ratio_metric(ratios_df)
        for hazard in ("flood", "wind"):
            row = ratios_df.loc[ratios_df["hazard"] == hazard].iloc[0]
            result.update(
                {
                    f"{hazard}_simulated_fraction": float(row["simulated_fraction"]),
                    f"{hazard}_target_fraction": float(row["observed_fraction"]),
                    f"{hazard}_simulated_count": int(row["simulated_count"]),
                    f"{hazard}_target_count": float(row["target_count"]),
                    f"{hazard}_lecz_households": int(row["total_households"]),
                    f"{hazard}_count_error": float(row["count_error"]),
                    f"{hazard}_snapshot": row["snapshot"],
                }
            )

        result["map_realizarion"] = self._settings().get("map_realizarion")
        return result

    def _settings(self) -> dict[str, Any]:
        return (
            self.model.config.get("agent_settings", {})
            .get("households", {})
            .get("adaptation_evaluation", {})
        )

    @staticmethod
    def _fraction_setting(settings: dict[str, Any], key: str) -> float:
        if key not in settings:
            raise ValueError(
                f"Missing {key!r} inagent_settings.households.adaptation_evaluation"
            )
        value = float(settings[key])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be a fraction between 0 and 1")
        return value

    @staticmethod  # NEED-CHECK what does this truly does
    def _normalize_snapshot(value: str | date | datetime) -> str:
        if isinstance(value, datetime):
            return value.strftime("%Y%m%d")
        if isinstance(value, date):
            return value.strftime("%Y%m%d")
        text = str(value).replace("-", "")
        if len(text) != 8 or not text.isdigit():
            raise ValueError(f"Snapshot {value!r} must be a date such as '20220101")
        return text

    def _resolve_snapshot(
        self,
        hazard: str,
        explicit_snapshot: str | None,
        settings: dict[str, Any],
        common_snapshots: list[str],
    ) -> str:

        if explicit_snapshot is not None:
            selected = self._normalize_snapshot(explicit_snapshot)
            if selected not in common_snapshots:
                raise FileNotFoundError(
                    f"Requested {hazard} snapshot {selected} is not available "
                    "in every configured cluster."
                )
            return selected

        event_end = settings.get(f"{hazard}_event_end")
        if event_end is None:
            raise ValueError(
                f"Set {hazard}_snapshot or {hazard}_event_end in "
                "agent_settings.households.adaptation_evaluation."
            )

        event_date = pd.to_datetime(event_end).normalize()
        delay = settings.get(f"{hazard}_response_delay_days")
        if delay is None:
            delay = 0
            if hazard == "flood":
                household_settings = self.model.config.get("agent_settings", {}).get(
                    "households", {}
                )
                if household_settings.get("adapt_to_actual_floods", False):
                    delay = 14
        eligible_date = event_date + pd.Timedelta(days=int(delay))
        eligible = [
            snap
            for snap in common_snapshots
            if pd.to_datetime(snap, format="%Y%m%d") >= eligible_date
        ]
        if not eligible:
            raise FileNotFoundError(
                f"No common {hazard} snapshot is on/after {eligible_date.isoformat()}."
            )
        return eligible[0]

    @staticmethod
    def _snapshot_files(
        cluster_dirs: Sequence[Path], run_name: str
    ) -> dict[Path, set[str]]:
        found: dict[Path, set[str]] = {}
        for cluster_dir in cluster_dirs:
            parquet_dir = (
                cluster_dir
                / HOUSEHOLD_OUTPUT_REL_PATH
                / run_name
                / HOUSEHOLD_TABLE_REL_PATH
            )
            if not parquet_dir.is_dir():
                raise FileNotFoundError(
                    f"Household output directory does not exist: {parquet_dir}"
                )
            files = {
                path.stem.removeprefix("household_data_")
                for path in parquet_dir.glob("household_data_*.parquet")
            }
        if not files:
            raise FileNotFoundError(
                f"No household snapshot parquet files found in {parquet_dir}"
            )
        found[cluster_dir] = files
        return found

    @staticmethod
    def _load_lecz(cluster_dirs: Sequence[Path]) -> gpd.GeoDataFrame:
        masks: list[gpd.GeoDataFrame] = []
        for cluster_dir in cluster_dirs:
            path = cluster_dir / LECZ_REL_PATH
            if not path.is_file():
                raise FileNotFoundError(f"LECZ mask not found: {path}")
            masks.append(gpd.read_parquet(path))
        combined = pd.concat(masks, ignore_index=True)
        mask = gpd.GeoDataFrame(combined, geometry="geometry", crs=masks[0].crs)
        if mask.crs is None:
            raise ValueError("LECZ mask has no CRS.")
        dissolved = mask.dissolve().reset_index(drop=True)
        return gpd.GeoDataFrame(
            dissolved,
            geometry="geometry",
            crs=mask.crs,
        )

    ##NEED-CHEK
    @staticmethod
    def _load_cluster_snapshot(
        cluster_dir: Path,
        run_name: str,
        snapshot: str,
        lecz: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        parquet_path = (
            cluster_dir
            / HOUSEHOLD_OUTPUT_REL_PATH
            / run_name
            / HOUSEHOLD_TABLE_REL_PATH
            / f"household_data_{snapshot}.parquet"
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Snapshot not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        required = {"flood_adaptation", "wind_adaptation"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{parquet_path} is missing columns: {sorted(missing)}")
        if "household_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "household_id"})
        if df["household_id"].duplicated().any():
            raise ValueError(f"Duplicate household_id values in {parquet_path}")

        household_ids = pd.to_numeric(df["household_id"], errors="raise").to_numpy(
            dtype=np.int64
        )
        location_path = cluster_dir / HOUSEHOLD_LOCATION_REL_PATH
        if not location_path.exists():
            raise FileNotFoundError(f"Household locations not found: {location_path}")
        location = np.asarray(zarr.open_array(str(location_path), mode="r")[:])
        valid = (household_ids >= 0) & (household_ids < len(location))
        if not valid.all():
            raise IndexError(
                f"Invalid household_id values in {parquet_path}; cannot map "
                "all rows to location.zarr."
            )

        points = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(
                location[household_ids, 0], location[household_ids, 1]
            ),
            crs="EPSG:4326",
        )
        points = points.to_crs(lecz.crs)
        selected = gpd.sjoin(
            points,
            lecz[["geometry"]],
            how="inner",
            predicate="within",
        ).drop(columns=["index_right"], errors="ignore")

        for column in ("flood_adaptation", "wind_adaptation"):
            values = pd.to_numeric(selected[column], errors="coerce")
            if values.isna().any() or not values.isin([0, 1]).all():
                raise ValueError(
                    f"{column} must contain only non-missing binary 0/1 values "
                    f"in {parquet_path}."
                )
            selected[column] = values.astype(np.int64)
        return selected

    def _aggregate_snapshot(
        self,
        cluster_dirs: Sequence[Path],
        run_name: str,
        snapshot: str,
        lecz: gpd.GeoDataFrame,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for cluster_dir in cluster_dirs:
            households = self._load_cluster_snapshot(
                cluster_dir, run_name, snapshot, lecz
            )
            records.append(
                {
                    "cluster": cluster_dir.name,
                    "snapshot": snapshot,
                    "lecz_households": int(len(households)),
                    "flood_adapted": int(households["flood_adaptation"].sum()),
                    "wind_adapted": int(households["wind_adaptation"].sum()),
                }
            )
        if sum(row["lecz_households"] for row in records) == 0:
            raise ValueError(f"No LECZ households found across clusters at {snapshot}.")
        return records
