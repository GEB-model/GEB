"""Load discharge observations and simulations and locate evaluation outputs."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd

from geb.hydrology import routing as hydrology_routing
from geb.workflows.io import read_geom, read_table

if TYPE_CHECKING:
    from geb.evaluate.hydrology import Hydrology


DISCHARGE_OBSERVATION_FREQUENCIES: dict[str, str] = {
    "hourly": "h",
    "daily": "D",
}


# Evaluation output paths


class DischargeEvaluationPaths(NamedTuple):
    """Output paths for one discharge evaluation period."""

    suffix: str
    label: str
    plot_folder: Path
    metrics_excel: Path
    metrics_geoparquet: Path


def _get_discharge_evaluation_paths(
    output_folder: Path,
    start_year: int | None,
    end_year: int | None,
) -> DischargeEvaluationPaths:
    """Get output paths for one discharge evaluation period.

    Args:
        output_folder: Root discharge evaluation output folder.
        start_year: First calendar year included in the evaluation, or `None`
            for the first available year.
        end_year: Last calendar year included in the evaluation, or `None` for
            the last available year.

    Returns:
        Period-specific output suffix, label, plot folder, and metric file paths.

    Raises:
        ValueError: If both years are provided and `start_year` is after
            `end_year`.
    """
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("start_year must be smaller than or equal to end_year.")
    suffix: str = ""
    if start_year is not None or end_year is not None:
        start_label: str = str(start_year) if start_year is not None else "start"
        end_label: str = str(end_year) if end_year is not None else "end"
        suffix = f"_{start_label}_{end_label}"
    return DischargeEvaluationPaths(
        suffix=suffix,
        label="the full overlapping period" if not suffix else suffix[1:],
        plot_folder=output_folder if not suffix else output_folder / f"period{suffix}",
        metrics_excel=output_folder / f"evaluation_metrics{suffix}.xlsx",
        metrics_geoparquet=output_folder / f"evaluation_metrics{suffix}.geoparquet",
    )


# River discharge


def get_discharge_per_river(
    self: Hydrology, run_name: str
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Get the discharge per river from the report directory.

    Args:
        self: Hydrology evaluator providing model settings and output paths.
        run_name: Name of the simulation run to evaluate. Must correspond to an existing
            run directory in the model output folder.

    Raises:
        FileNotFoundError: If the discharge file for the specified run does not exist
            in the report directory.

    Returns:
        A GeoDataFrame containing the river geometries and a DataFrame containing the discharge data for each river.
    """
    discharge_folder: Path = (
        self.evaluator.output_folder_evaluate.parent / "report" / "hydrology.routing"
    )
    if not discharge_folder.exists():
        raise FileNotFoundError(
            f"Discharge files for run '{run_name}' does not exist in the report directory. Did you run the model?"
        )

    all_rivers: gpd.GeoDataFrame = read_geom(self.model.files["geom"]["routing/rivers"])
    rivers_of_interest: gpd.GeoDataFrame = all_rivers[
        ~(
            all_rivers["is_downstream_outflow"]
            | all_rivers["is_upstream_of_downstream_basin"]
            | all_rivers["is_further_downstream_outflow"]
        )
    ].copy()

    # Merged runs can omit files when outflow reporting was disabled for a cluster.
    has_discharge_output: list[bool] = [
        (
            discharge_folder / f"river_outflow_hourly_m3_per_s_{river_id}.parquet"
        ).exists()
        for river_id in rivers_of_interest.index
    ]
    rivers_of_interest = rivers_of_interest[has_discharge_output].copy()

    discharge: pd.DataFrame = hydrology_routing.get_discharge_per_river(
        folder=discharge_folder,
        rivers=rivers_of_interest,
        all_rivers=all_rivers,
    )
    return rivers_of_interest, discharge


# Station observations and simulation alignment


def load_discharge_observations(self: Hydrology) -> dict[str, pd.DataFrame]:
    """Read station discharge observations on regular hourly and daily indices.

    Args:
        self: Hydrology evaluator providing observation file paths.

    Returns:
        Observation tables (m³/s) keyed by frequency, with station ID columns.
    """
    observations_by_frequency: dict[str, pd.DataFrame] = {}
    for frequency, timestep in DISCHARGE_OBSERVATION_FREQUENCIES.items():
        observations: pd.DataFrame = read_table(
            self.model.files["table"][f"discharge/discharge_observations_{frequency}"]
        )
        observations_by_frequency[frequency] = (
            observations.asfreq(timestep) if not observations.empty else observations
        )
    return observations_by_frequency


def load_station_discharge_comparison(
    output_folder: Path,
    station_id: str | int,
    observed_discharge: pd.Series,
    apply_upstream_area_correction: bool,
    upstream_area_ratio: float,
    timezone_utc_offset: float = 0.0,
) -> pd.DataFrame:
    """Align observed and simulated discharge for one gauging station.

    Args:
        output_folder: Path to the model output folder.
        station_id: Station identifier to create the validation dataframe for.
        observed_discharge: Observed station discharge (m3/s).
        apply_upstream_area_correction: Whether to scale simulated discharge to
            the observed station's upstream area.
        upstream_area_ratio: Observed upstream area divided by the modeled
            upstream area (dimensionless).
        timezone_utc_offset: Fixed UTC offset for the GRDC station metadata
            (hours). For daily or coarser observations, GEB's hourly UTC
            timestamps are converted to this fixed local offset before
            aggregation. Sub-daily observations are not shifted because their
            timestamp convention is not defined by the GRDC daily product.
            Defaults to 0 (UTC).
    Returns:
        Aligned observed and simulated discharge (m3/s).

    Raises:
        FileNotFoundError: If the hydrology routing directory does not exist.
        ValueError: If discharge values, timestamp frequencies, the upstream-area
            correction, or the fixed UTC offset are invalid.
    """
    report_folder: Path = output_folder / "report"
    routing_dir: Path = report_folder / "hydrology.routing"
    if not routing_dir.exists():
        raise FileNotFoundError(
            f"Hydrology routing directory does not exist: {routing_dir}"
        )

    station_file_path: Path = (
        routing_dir / f"discharge_hourly_m3_per_s_{station_id}.parquet"
    )
    simulated_discharge: pd.Series = pd.read_parquet(station_file_path)[
        f"discharge_hourly_m3_per_s_{station_id}"
    ]

    if not np.isfinite(simulated_discharge.to_numpy()).all():
        raise ValueError(
            f"Non-finite values found in GEB discharge data for station {station_id}. Please check the station file {station_file_path}."
        )

    simulated_index: pd.Index = simulated_discharge.index
    if not isinstance(simulated_index, pd.DatetimeIndex):
        raise ValueError("Simulated discharge must have a DateTimeIndex.")
    if len(simulated_index) < 3 or pd.infer_freq(simulated_index) is None:
        raise ValueError("Simulated discharge must have a regular frequency.")

    if apply_upstream_area_correction:
        if not np.isfinite(upstream_area_ratio) or upstream_area_ratio <= 0:
            raise ValueError("Upstream area ratio must be finite and positive.")
        simulated_discharge = simulated_discharge * upstream_area_ratio

    observed_index: pd.Index = observed_discharge.index
    if not isinstance(observed_index, pd.DatetimeIndex):
        raise ValueError("Observed discharge must have a DateTimeIndex.")
    if not observed_index.is_monotonic_increasing:
        raise ValueError(
            "Observed discharge index must be a regular time series with a monotonic increasing DateTimeIndex."
        )

    if observed_index.freq is None:
        raise ValueError("Observed discharge index must have a defined frequency.")
    if len(observed_index) < 2:
        raise ValueError("Observed discharge must contain at least two timestamps.")
    if not np.isfinite(timezone_utc_offset) or not (
        -12.0 <= timezone_utc_offset <= 14.0
    ):
        raise ValueError(
            "Station UTC offset must be finite and between UTC-12 and UTC+14 hours."
        )
    observed_frequency: Any = observed_index.freq
    simulated_timestep: pd.Timedelta = simulated_index[1] - simulated_index[0]
    observed_timestep: pd.Timedelta = observed_index[1] - observed_index[0]
    if (
        observed_timestep < simulated_timestep
        or observed_timestep % simulated_timestep != pd.Timedelta(0)
    ):
        raise ValueError(
            "Observed discharge timestep must be a multiple of the simulated timestep."
        )

    if observed_timestep >= pd.Timedelta(days=1) and timezone_utc_offset != 0.0:
        # GRDC daily observations represent local calendar days.
        simulated_discharge.index = simulated_discharge.index + pd.Timedelta(
            hours=timezone_utc_offset
        )

    simulated_resampler: Any = simulated_discharge.resample(
        observed_frequency, closed="left", label="left"
    )
    # Local-time shifts can leave partial days at either end of a report.
    # Compare observations only with complete simulation intervals.
    expected_steps: float = observed_timestep / simulated_timestep
    simulated_discharge = simulated_resampler.mean().where(
        simulated_resampler.count() == expected_steps
    )

    # cut both observed and simulated discharge to the same time range
    start_time = max(observed_discharge.index.min(), simulated_discharge.index.min())
    end_time = min(observed_discharge.index.max(), simulated_discharge.index.max())
    observed_discharge = observed_discharge.loc[start_time:end_time]
    simulated_discharge = simulated_discharge.loc[start_time:end_time]

    # Create a combined dataframe with the union of all timestamps.
    # Values will be NaN where data is missing in either series.
    discharge_comparison = pd.DataFrame(
        {
            "discharge_observations": observed_discharge,
            "discharge_simulations": simulated_discharge,
        }
    )

    return discharge_comparison
