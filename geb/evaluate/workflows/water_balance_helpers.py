"""Load water-balance and storage series and calculate area and yearly summaries."""

from pathlib import Path
from typing import Any

import pandas as pd

from geb.workflows.io import read_table


def _read_evaluation_series_with_date_index(
    folder: Path,
    module: str,
    name: str,
    skip_first_day: bool = False,
) -> pd.Series:
    """Read an evaluation time series from a parquet file.

    Args:
        folder: Path to the report folder for one model run.
        module: Name of the module subfolder containing the parquet file.
        name: Name of the parquet file without the `.parquet` suffix.
        skip_first_day: Drop the initial record when it represents initialization.

    Returns:
        Time-indexed series in the reported units, optionally without its first record.
    """
    series: pd.Series = read_table((folder / module / name).with_suffix(".parquet"))[
        name
    ]
    return series.iloc[1:] if skip_first_day else series


def _load_named_evaluation_series(
    folder: Path,
    series_specs: dict[str, tuple[str, str]],
) -> dict[str, pd.Series]:
    """Load a named collection of evaluation time series from parquet files.

    Args:
        folder: Path to the report folder for one model run.
        series_specs: Mapping from output name used by the caller to a tuple of
            `(module, reported_name)` describing where the parquet series lives.

    Returns:
        Mapping of caller-defined series names to time-indexed pandas series.
    """
    return {
        series_name: _read_evaluation_series_with_date_index(
            folder,
            module_name,
            reported_name,
        )
        for series_name, (module_name, reported_name) in series_specs.items()
    }


def _flatten_water_balance_hierarchy(
    prefix: str,
    hierarchy: dict[str, Any],
    flattened_series: dict[str, pd.Series],
) -> None:
    """Flatten a nested water balance hierarchy into a flat column mapping.

    Args:
        prefix: Current prefix for nested names.
        hierarchy: Nested mapping with dict nodes and `pd.Series` leaves.
        flattened_series: Output mapping populated in place.
    """
    for key, value in hierarchy.items():
        column_name: str = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_water_balance_hierarchy(column_name, value, flattened_series)
        elif isinstance(value, pd.Series):
            flattened_series[column_name] = value


def _load_water_balance_dataframe(folder: Path) -> pd.DataFrame:
    """Load water balance component time series for one run.

    Notes:
        Output components remain positive in the returned dataframe. Callers that
        want a signed plotting convention should negate the `out_` columns.

    Args:
        folder: Path to the report folder for one model run.

    Returns:
        Dataframe with one column per water balance component (m3 per timestep).
    """
    balance_series: dict[str, pd.Series] = _load_named_evaluation_series(
        folder,
        {
            "storage": ("hydrology", "_current_storage"),
            "rain": ("hydrology.landsurface", "_rain_m"),
            "snow": ("hydrology.landsurface", "_snow_m"),
            "domestic_water_loss": (
                "hydrology.water_demand",
                "_domestic_water_loss_m3",
            ),
            "industry_water_loss": (
                "hydrology.water_demand",
                "_industry_water_loss_m3",
            ),
            "livestock_water_loss": (
                "hydrology.water_demand",
                "_livestock_water_loss_m3",
            ),
            "river_outflow": ("hydrology.routing", "_total_outflow_at_pits_m3"),
            "transpiration": (
                "hydrology.landsurface",
                "_transpiration_m",
            ),
            "bare_soil_evaporation": (
                "hydrology.landsurface",
                "_bare_soil_evaporation_m",
            ),
            "open_water_evaporation": (
                "hydrology.landsurface",
                "_open_water_evaporation_m",
            ),
            "interception_evaporation": (
                "hydrology.landsurface",
                "_interception_evaporation_m",
            ),
            "sublimation_or_deposition": (
                "hydrology.landsurface",
                "_sublimation_or_deposition_m",
            ),
            "river_evaporation": (
                "hydrology.routing",
                "_total_evaporation_in_rivers_m3",
            ),
            "waterbody_evaporation": (
                "hydrology.routing",
                "_total_waterbody_evaporation_m3",
            ),
        },
    )

    storage_m3: pd.Series = balance_series["storage"]
    rain_m3: pd.Series = balance_series["rain"]
    snow_m3: pd.Series = balance_series["snow"]
    domestic_water_loss_m3: pd.Series = balance_series["domestic_water_loss"]
    industry_water_loss_m3: pd.Series = balance_series["industry_water_loss"]
    livestock_water_loss_m3: pd.Series = balance_series["livestock_water_loss"]
    river_outflow_m3: pd.Series = balance_series["river_outflow"]
    transpiration_m3: pd.Series = balance_series["transpiration"]
    bare_soil_evaporation_m3: pd.Series = balance_series["bare_soil_evaporation"]
    open_water_evaporation_m3: pd.Series = balance_series["open_water_evaporation"]
    interception_evaporation_m3: pd.Series = balance_series["interception_evaporation"]
    sublimation_or_deposition_m3: pd.Series = balance_series[
        "sublimation_or_deposition"
    ]
    river_evaporation_m3: pd.Series = balance_series["river_evaporation"]
    waterbody_evaporation_m3: pd.Series = balance_series["waterbody_evaporation"]

    storage_change_m3: pd.Series = storage_m3.diff().fillna(0)
    hierarchy: dict[str, Any] = {
        "in": {
            "rain": rain_m3,
            "snow": snow_m3,
        },
        "out": {
            "evapotranspiration": {
                "transpiration": transpiration_m3,
                "bare_soil_evaporation": bare_soil_evaporation_m3,
                "open_water_evaporation": open_water_evaporation_m3,
                "interception_evaporation": interception_evaporation_m3,
                "river_evaporation": river_evaporation_m3,
                "waterbody_evaporation": waterbody_evaporation_m3,
            },
            "water_demand": {
                "domestic_water_loss": domestic_water_loss_m3,
                "industry_water_loss": industry_water_loss_m3,
                "livestock_water_loss": livestock_water_loss_m3,
            },
            "river_outflow": river_outflow_m3,
        },
        "storage_change": storage_change_m3,
    }

    if sublimation_or_deposition_m3.sum() > 0:
        hierarchy["in"]["deposition"] = sublimation_or_deposition_m3
    else:
        hierarchy["out"]["evapotranspiration"]["sublimation"] = abs(
            sublimation_or_deposition_m3
        )

    flattened_series: dict[str, pd.Series] = {}
    _flatten_water_balance_hierarchy("", hierarchy, flattened_series)
    return pd.DataFrame(flattened_series).sort_index()


def _load_contextual_water_balance_series(folder: Path) -> dict[str, pd.Series]:
    """Load optional context series that support water-balance interpretation.

    Notes:
        These series are not part of the actual water balance and therefore must
        not be included in the balance dataframe, signed output conversion, or
        annual balance summaries.

    Args:
        folder: Path to the report folder for one model run.

    Returns:
        Mapping of context series names to their time series (m3 per timestep),
        or an empty mapping when potential evapotranspiration was not reported.
    """
    potential_evapotranspiration_path: Path = (
        folder / "hydrology.landsurface" / "_potential_evapotranspiration_m.parquet"
    )
    if not potential_evapotranspiration_path.is_file():
        return {}

    return _load_named_evaluation_series(
        folder,
        {
            "potential_evapotranspiration": (
                "hydrology.landsurface",
                "_potential_evapotranspiration_m",
            )
        },
    )


def _load_top_soil_water_balance_dataframe(folder: Path) -> pd.DataFrame:
    """Load top-soil-layer water balance diagnostics for one run.

    Notes:
        This dataframe is limited to terms that contribute directly to the
        reported top-soil storage balance. Additional land-surface terms that
        help interpret the plot, such as precipitation and runoff before
        infiltration enters the control volume, are loaded separately as
        context series.

    Args:
        folder: Path to the report folder for one model run.

    Returns:
        Dataframe with one column per top-soil water balance component (m3 per timestep).
    """
    top_soil_series: dict[str, pd.Series] = _load_named_evaluation_series(
        folder,
        {
            "storage": ("hydrology.landsurface", "_top_soil_water_content_m"),
            "infiltration": (
                "hydrology.landsurface",
                "_top_soil_infiltration_m",
            ),
            "rise_from_layer_2": (
                "hydrology.landsurface",
                "_top_soil_rise_from_layer_2_m",
            ),
            "evaporation": (
                "hydrology.landsurface",
                "_top_soil_evaporation_m",
            ),
            "transpiration": (
                "hydrology.landsurface",
                "_top_soil_transpiration_m",
            ),
            "percolation_to_layer_2": (
                "hydrology.landsurface",
                "_top_soil_percolation_to_layer_2_m",
            ),
        },
    )

    top_soil_storage_m3: pd.Series = top_soil_series["storage"]
    top_soil_infiltration_m3: pd.Series = top_soil_series["infiltration"]
    top_soil_rise_from_layer_2_m3: pd.Series = top_soil_series["rise_from_layer_2"]
    top_soil_evaporation_m3: pd.Series = top_soil_series["evaporation"]
    top_soil_transpiration_m3: pd.Series = top_soil_series["transpiration"]
    top_soil_percolation_to_layer_2_m3: pd.Series = top_soil_series[
        "percolation_to_layer_2"
    ]

    top_soil_storage_change_m3: pd.Series = top_soil_storage_m3.diff().fillna(0)
    hierarchy: dict[str, Any] = {
        "in": {
            "infiltration": top_soil_infiltration_m3,
            "rise_from_layer_2": top_soil_rise_from_layer_2_m3,
        },
        "out": {
            "evaporation": top_soil_evaporation_m3,
            "transpiration": top_soil_transpiration_m3,
            "percolation_to_layer_2": top_soil_percolation_to_layer_2_m3,
        },
        "storage_change": top_soil_storage_change_m3,
    }

    flattened_series: dict[str, pd.Series] = {}
    _flatten_water_balance_hierarchy("", hierarchy, flattened_series)
    return pd.DataFrame(flattened_series).sort_index()


def _load_contextual_top_soil_water_balance_series(
    folder: Path,
) -> dict[str, pd.Series]:
    """Load land-surface context series for the top-soil water balance plots.

    Notes:
        These series help explain how precipitation is partitioned before water
        enters or leaves the top-soil control volume, and how atmospheric
        demand linked to that store varies over time. They stay outside the
        strict top-soil storage balance and the balance totals.

    Args:
        folder: Path to the report folder for one model run.

    Returns:
        Mapping of context series names to their time series.
    """
    return _load_named_evaluation_series(
        folder,
        {
            # _rain_m is identical to the former top_soil_precipitation (same varname)
            "precipitation": (
                "hydrology.landsurface",
                "_rain_m",
            ),
            "runoff": (
                "hydrology.landsurface",
                "_runoff_m_daily",
            ),
            # _snow_m is identical to the former top_soil_snow (same varname)
            "snow": (
                "hydrology.landsurface",
                "_snow_m",
            ),
            "potential_evapotranspiration": (
                "hydrology.landsurface",
                "_potential_evapotranspiration_m",
            ),
        },
    )


def _get_datetime_index_step_label(time_index: pd.DatetimeIndex) -> str:
    """Infer a compact timestep label from a datetime index.

    Args:
        time_index: Datetime index for the plotted series.

    Returns:
        Compact timestep label such as `H`, `D`, or `MS`.

    Raises:
        ValueError: If the datetime frequency cannot be determined from the index.
    """
    frequency_label_map: dict[str, str] = {
        "D": "day",
    }

    if time_index.freq is not None and time_index.freq.freqstr is not None:
        frequency_label: str = str(time_index.freq.freqstr).upper()
        return frequency_label_map.get(frequency_label, frequency_label)

    inferred_frequency: str | None = pd.infer_freq(time_index)
    if inferred_frequency is not None:
        normalized_frequency: str = inferred_frequency.upper()
        return frequency_label_map.get(normalized_frequency, normalized_frequency)

    raise ValueError(
        "Could not determine the timestep frequency from the datetime index."
    )


def _create_yearly_totals_summary_mm(
    water_balance_df_m3_per_timestep: pd.DataFrame,
    total_area_m2: float,
) -> pd.DataFrame:
    """Summarize annual water balance totals per component as depths.

    Notes:
        Output components are converted to negative depths and storage change
        retains its sign. This mirrors the signed plotting convention used in
        the time-series figures.

    Args:
        water_balance_df_m3_per_timestep: Water balance components (m3 per timestep).
        total_area_m2: Total model area represented by the reported fluxes (m2).

    Returns:
        Dataframe indexed by calendar year with one column per component in mm/year.
    """
    annual_totals_m3: pd.DataFrame = water_balance_df_m3_per_timestep.resample(
        "YE"
    ).sum()
    conversion_factor_mm_per_m3: float = 1000.0 / total_area_m2

    summary_mm: pd.DataFrame = annual_totals_m3 * conversion_factor_mm_per_m3
    output_columns: list[str] = [
        column_name
        for column_name in summary_mm.columns
        if column_name.startswith("out_")
    ]
    summary_mm.loc[:, output_columns] = -summary_mm.loc[:, output_columns]
    summary_mm.index = summary_mm.index.year  # ty:ignore[unresolved-attribute]
    return summary_mm
