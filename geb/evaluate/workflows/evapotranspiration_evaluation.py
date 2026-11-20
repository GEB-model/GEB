"""Evaluate simulated evapotranspiration against eddy covariance (FLUXNET) towers.

The model reports daily evaporation components at the land unit (HRU) under each
tower when ``report._meteorological_stations`` is enabled. Observed evapotranspiration
is the tower latent heat flux divided by the latent heat of vaporization.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd

from geb.workflows.io import read_geom, read_table

if TYPE_CHECKING:
    from geb.evaluate.hydrology import Hydrology

# Latent heat of vaporization of water at about 20 °C (J/kg); 1 kg/m2 = 1 mm.
LATENT_HEAT_OF_VAPORIZATION_J_PER_KG: float = 2.45e6
SECONDS_PER_DAY: float = 86400.0
DAYS_PER_YEAR: float = 365.25

STATION_REPORT_MODULE: str = "hydrology.landsurface"
# scores per tower; NaN where a tower has too few good days
SCORE_COLUMNS: tuple[str, ...] = (
    "n_days",
    "observed_mm_per_year",
    "simulated_mm_per_year",
    "potential_mm_per_year",
    "bias_ratio",
    "bias_ratio_corrected",
    "correlation_daily",
    "KGE_daily",
)
SIMULATED_COMPONENTS: tuple[str, ...] = (
    "transpiration_m",
    "bare_soil_evaporation_m",
    "interception_evaporation_m",
    "sublimation_or_deposition_m",
    "potential_evapotranspiration_m",
    "reference_evapotranspiration_grass_m",
)

# IGBP land cover classes used by FLUXNET.
IGBP_NAMES: dict[str, str] = {
    "ENF": "evergreen needleleaf forest",
    "EBF": "evergreen broadleaf forest",
    "DNF": "deciduous needleleaf forest",
    "DBF": "deciduous broadleaf forest",
    "MF": "mixed forest",
    "CSH": "closed shrubland",
    "OSH": "open shrubland",
    "WSA": "woody savanna",
    "SAV": "savanna",
    "GRA": "grassland",
    "WET": "wetland",
    "CRO": "cropland",
    "URB": "urban",
    "SNO": "snow and ice",
    "BSV": "barren",
    "WAT": "water",
}


class TowerComparison(NamedTuple):
    """Daily observed and simulated evapotranspiration per tower and summary scores."""

    stations: gpd.GeoDataFrame
    daily: dict[str, pd.DataFrame]


def latent_heat_flux_to_mm_per_day(
    latent_heat_flux_W_per_m2: pd.Series,
) -> pd.Series:
    """Convert a daily mean latent heat flux to evapotranspiration.

    Args:
        latent_heat_flux_W_per_m2: Daily mean latent heat flux (W/m2).

    Returns:
        Evapotranspiration (mm/day).
    """
    return (
        latent_heat_flux_W_per_m2
        * SECONDS_PER_DAY
        / LATENT_HEAT_OF_VAPORIZATION_J_PER_KG
    )


def read_simulated_station_components(
    report_folder: Path,
) -> dict[str, pd.DataFrame] | None:
    """Read the daily evaporation components reported at the stations.

    Args:
        report_folder: The ``report`` folder of a model run.

    Returns:
        Component name to a DataFrame (daily index, column per station) in mm/day,
        or None if the run did not report meteorological stations.
    """
    components: dict[str, pd.DataFrame] = {}
    for component in SIMULATED_COMPONENTS:
        path: Path = (
            report_folder
            / STATION_REPORT_MODULE
            / f"meteorological_station_{component}.parquet"
        )
        if not path.exists():
            return None
        df: pd.DataFrame = read_table(path) * 1000.0  # m -> mm
        df.index = pd.DatetimeIndex(df.index).normalize()
        components[component] = df
    return components


def kling_gupta_efficiency(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Kling-Gupta efficiency (Gupta et al., 2009).

    Args:
        observed: Observed values.
        simulated: Simulated values, same length as observed.

    Returns:
        KGE (-), or NaN with fewer than three values or zero observed variance.
    """
    if observed.size < 3 or np.std(observed) == 0 or np.mean(observed) == 0:
        return math.nan
    correlation: float = float(np.corrcoef(observed, simulated)[0, 1])
    variability: float = float(np.std(simulated) / np.std(observed))
    bias: float = float(np.mean(simulated) / np.mean(observed))
    return 1.0 - math.sqrt(
        (correlation - 1.0) ** 2 + (variability - 1.0) ** 2 + (bias - 1.0) ** 2
    )


def compare_towers(
    station_locations: gpd.GeoDataFrame,
    observations: pd.DataFrame,
    simulated: dict[str, pd.DataFrame],
    minimum_quality_fraction: float = 0.8,
) -> TowerComparison:
    """Pair observed and simulated daily evapotranspiration and score each tower.

    Only days on which at least ``minimum_quality_fraction`` of the half-hourly
    latent heat fluxes are measured or good-quality gap-filled are scored.

    Args:
        station_locations: Tower locations indexed by station id, optionally with
            ``site_name`` and ``igbp`` columns.
        observations: Long table with ``station_id``, ``date``,
            ``latent_heat_flux_W_per_m2``, ``latent_heat_flux_corrected_W_per_m2``
            and ``latent_heat_flux_quality_fraction``.
        simulated: Daily simulated components in mm/day (see
            read_simulated_station_components).
        minimum_quality_fraction: Minimum daily quality fraction of the tower data (0-1).

    Returns:
        Stations with scores and, per station, the paired daily data (mm/day).
    """
    observations = observations.copy()
    observations["date"] = pd.to_datetime(observations["date"]).dt.normalize()
    rows: list[dict[str, Any]] = []
    daily: dict[str, pd.DataFrame] = {}
    for station_id, station in station_locations.iterrows():
        key: str = str(station_id)
        row: dict[str, Any] = {
            "station_id": key,
            "site_name": station["site_name"],
            "igbp": station["igbp"],
            "geometry": station.geometry,
        }
        if key not in simulated["transpiration_m"].columns:
            rows.append(row)
            continue
        sim: pd.DataFrame = pd.DataFrame(
            {name: df[key] for name, df in simulated.items()}
        )
        sim["evapotranspiration"] = (
            sim["transpiration_m"]
            + sim["bare_soil_evaporation_m"]
            + sim["interception_evaporation_m"]
            - sim["sublimation_or_deposition_m"]
        )
        obs: pd.DataFrame = (
            observations[observations["station_id"].astype(str) == key]
            .set_index("date")
            .sort_index()
        )
        paired: pd.DataFrame = sim.join(
            pd.DataFrame(
                {
                    "observed": latent_heat_flux_to_mm_per_day(
                        obs["latent_heat_flux_W_per_m2"]
                    ),
                    "observed_corrected": latent_heat_flux_to_mm_per_day(
                        obs["latent_heat_flux_corrected_W_per_m2"]
                    ),
                    "quality_fraction": obs["latent_heat_flux_quality_fraction"],
                }
            ),
            how="left",
        )
        good: pd.Series = (
            paired["observed"].notna()
            & (paired["quality_fraction"] >= minimum_quality_fraction)
            & paired["evapotranspiration"].notna()
        )
        paired["scored"] = good
        daily[key] = paired
        scored: pd.DataFrame = paired[good]
        row["n_days"] = int(good.sum())
        if row["n_days"] >= 3:
            observed: np.ndarray = scored["observed"].to_numpy()
            simulated_et: np.ndarray = scored["evapotranspiration"].to_numpy()
            row["observed_mm_per_year"] = float(observed.mean() * DAYS_PER_YEAR)
            row["simulated_mm_per_year"] = float(simulated_et.mean() * DAYS_PER_YEAR)
            row["potential_mm_per_year"] = float(
                scored["potential_evapotranspiration_m"].mean() * DAYS_PER_YEAR
            )
            row["bias_ratio"] = float(simulated_et.mean() / observed.mean())
            corrected: pd.Series = scored["observed_corrected"].dropna()
            row["bias_ratio_corrected"] = (
                float(
                    scored.loc[corrected.index, "evapotranspiration"].mean()
                    / corrected.mean()
                )
                if len(corrected) >= 3
                else math.nan
            )
            row["correlation_daily"] = float(np.corrcoef(observed, simulated_et)[0, 1])
            row["KGE_daily"] = kling_gupta_efficiency(observed, simulated_et)
        rows.append(row)
    stations: gpd.GeoDataFrame = gpd.GeoDataFrame(
        pd.DataFrame(
            rows,
            columns=["station_id", "site_name", "igbp", "geometry", *SCORE_COLUMNS],
        ).set_index("station_id"),
        geometry="geometry",
        crs=station_locations.crs,
    )
    return TowerComparison(stations=stations, daily=daily)


def load_tower_comparison(
    run_output_folder: Path,
    geometry_files: dict[str, Path],
    table_files: dict[str, Path],
    minimum_quality_fraction: float = 0.8,
) -> TowerComparison | None:
    """Load tower observations and the simulated components of a run and compare them.

    Args:
        run_output_folder: Output folder of the run.
        geometry_files: Model geometry paths keyed by dataset name.
        table_files: Model table paths keyed by dataset name.
        minimum_quality_fraction: Minimum daily quality fraction of the tower data (0-1).

    Returns:
        The comparison, or None if the model has no towers or the run did not
        report them.
    """
    locations_key: str = "observations/meteorological_station_locations"
    observations_key: str = "observations/meteorological_stations_timeseries"
    if locations_key not in geometry_files or observations_key not in table_files:
        return None
    station_locations: gpd.GeoDataFrame = read_geom(geometry_files[locations_key])
    if station_locations.empty:
        return None
    observations: pd.DataFrame = read_table(table_files[observations_key])
    simulated: dict[str, pd.DataFrame] | None = read_simulated_station_components(
        run_output_folder / "report"
    )
    if simulated is None:
        return None
    return compare_towers(
        station_locations, observations, simulated, minimum_quality_fraction
    )


def build_tower_dashboard_data(comparison: TowerComparison) -> list[dict[str, Any]]:
    """Monthly series and scores per tower for the dashboard.

    Monthly means use only scored days, for observed and simulated alike, so both
    lines describe the same days.

    Args:
        comparison: Tower comparison from compare_towers.

    Returns:
        One dictionary per tower with location, metadata, scores and monthly series
        (mm/day).
    """

    def clean(values: pd.Series) -> list[float | None]:
        return [None if pd.isna(v) else round(float(v), 3) for v in values]

    towers: list[dict[str, Any]] = []
    for station_id, station in comparison.stations.iterrows():
        tower: dict[str, Any] = {
            "id": str(station_id),
            "name": str(station["site_name"]),
            "igbp": str(station["igbp"]),
            "igbp_name": IGBP_NAMES[str(station["igbp"])],
            "lat": float(station.geometry.y),
            "lon": float(station.geometry.x),
            "scores": {
                name: (
                    None if pd.isna(station[name]) else round(float(station[name]), 3)
                )
                for name in SCORE_COLUMNS
            },
        }
        paired: pd.DataFrame | None = comparison.daily.get(str(station_id))
        if paired is not None and paired["scored"].any():
            scored: pd.DataFrame = paired[paired["scored"]]
            monthly: pd.DataFrame = scored.resample("MS").mean(numeric_only=True)
            monthly = monthly[scored["observed"].resample("MS").count() >= 10]
            tower["monthly"] = {
                "time": [t.strftime("%Y-%m-%d") for t in monthly.index],
                "observed": clean(monthly["observed"]),
                "observed_corrected": clean(monthly["observed_corrected"]),
                "simulated": clean(monthly["evapotranspiration"]),
                "transpiration": clean(monthly["transpiration_m"]),
                "interception": clean(monthly["interception_evaporation_m"]),
                "soil": clean(monthly["bare_soil_evaporation_m"]),
                "potential": clean(monthly["potential_evapotranspiration_m"]),
                "reference": clean(monthly["reference_evapotranspiration_grass_m"]),
            }
        towers.append(tower)
    return towers


def evaluate_evapotranspiration(
    self: Hydrology,
    run_name: str = "default",
    minimum_quality_fraction: float = 0.8,
) -> dict[str, float | None]:
    """Evaluate simulated evapotranspiration against eddy covariance towers.

    Requires a run with ``report._meteorological_stations: true`` and towers from
    the build method ``setup_meteorological_stations_observations``. Writes
    ``tower_metrics.geoparquet`` and ``tower_daily.parquet`` to
    ``evaluate/hydrology/evaluate_evapotranspiration``.

    Args:
        self: Hydrology evaluator providing model settings and output paths.
        run_name: Name of the simulation run to evaluate.
        minimum_quality_fraction: Minimum daily quality fraction of the tower data (0-1).

    Returns:
        Median bias ratio, daily correlation and daily KGE over the towers (None
        if no tower could be scored).

    Raises:
        FileNotFoundError: If the run did not report the meteorological stations or
            the model has no tower observations.
    """
    run_output_folder: Path = Path(self.model.config["output_folder"]) / run_name
    comparison: TowerComparison | None = load_tower_comparison(
        run_output_folder=run_output_folder,
        geometry_files=self.model.files["geom"],
        table_files=self.model.files["table"],
        minimum_quality_fraction=minimum_quality_fraction,
    )
    if comparison is None:
        raise FileNotFoundError(
            "No tower comparison possible: run the model with "
            "report._meteorological_stations: true, and (re)run the build method "
            "setup_meteorological_stations_observations."
        )
    output_folder: Path = (
        self.evaluator.output_folder_evaluate
        / "hydrology"
        / "evaluate_evapotranspiration"
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    comparison.stations.to_parquet(output_folder / "tower_metrics.geoparquet")
    if comparison.daily:
        pd.concat(comparison.daily, names=["station_id", "time"]).to_parquet(
            output_folder / "tower_daily.parquet"
        )
    for station_id, station in comparison.stations.iterrows():
        if pd.notna(station["bias_ratio"]):
            self.model.logger.info(
                "%s (%s): observed %.0f mm/yr, simulated %.0f mm/yr, bias %.2f, r %.2f, KGE %.2f (%d days)",
                station_id,
                station["igbp"],
                station["observed_mm_per_year"],
                station["simulated_mm_per_year"],
                station["bias_ratio"],
                station["correlation_daily"],
                station["KGE_daily"],
                station["n_days"],
            )

    def median(column: str) -> float | None:
        if column not in comparison.stations:
            return None
        values: pd.Series = comparison.stations[column].dropna()
        return round(float(values.median()), 4) if len(values) else None

    return {
        "ET_bias_ratio_daily": median("bias_ratio"),
        "ET_correlation_daily": median("correlation_daily"),
        "ET_KGE_daily": median("KGE_daily"),
    }
