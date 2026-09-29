"""Plot water circles, water balances, and water storage.

Public plotting commands come first; shared axis and caption formatting follows.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as mcolormaps
from matplotlib.lines import Line2D

from geb.evaluate.workflows.water_balance_helpers import (
    _create_yearly_totals_summary_mm,
    _get_datetime_index_step_label,
    _load_contextual_top_soil_water_balance_series,
    _load_contextual_water_balance_series,
    _load_named_evaluation_series,
    _load_top_soil_water_balance_dataframe,
    _load_water_balance_dataframe,
    _read_evaluation_series_with_date_index,
)
from geb.reporter import WATER_STORAGE_REPORT_CONFIG
from geb.workflows.visualise import plot_sunburst

if TYPE_CHECKING:
    from geb.evaluate.hydrology import Hydrology


# Configure global style for all plots in this module
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["axes.edgecolor"] = "0.15"
mpl.rcParams["axes.labelcolor"] = "black"
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["text.color"] = "black"
mpl.rcParams["figure.edgecolor"] = "black"
mpl.rcParams["grid.color"] = "0.8"
mpl.rcParams["legend.labelcolor"] = "black"
mpl.rcParams["savefig.facecolor"] = "white"
mpl.rcParams["savefig.edgecolor"] = "white"


def plot_water_circle(
    self: Hydrology,
    run_name: str,
    *args: Any,
    export: bool = True,
    **kwargs: Any,
) -> plt.Figure:
    """Create a water circle plot for the GEB model.

    Adapted from: https://github.com/mikhailsmilovic/flowplot
    Also see the paper: https://doi.org/10.1088/1748-9326/ad18de

    Args:
        self: Hydrology evaluator providing model settings and output paths.
        run_name: Name of the run to evaluate.
        export: Whether to export the water circle plot to a file.
        *args: ignored.
        **kwargs: ignored.

    Returns:
        A matplotlib Figure object representing the water circle.
    """
    folder: Path = (
        Path(self.model.config["general"]["output_folder"]) / run_name / "report"
    )

    # because storage is the storage at the end of the timestep, we need to calculate the change
    # across the entire simulation period. For all other variables we do skip the first day.
    storage = _read_evaluation_series_with_date_index(
        folder, "hydrology", "_current_storage", skip_first_day=False
    )
    storage_change = storage.iloc[-1] - storage.iloc[0]

    rain = _read_evaluation_series_with_date_index(
        folder, "hydrology.landsurface", "_rain_m", skip_first_day=True
    ).sum()
    snow = _read_evaluation_series_with_date_index(
        folder, "hydrology.landsurface", "_snow_m", skip_first_day=True
    ).sum()

    domestic_water_loss = _read_evaluation_series_with_date_index(
        folder, "hydrology.water_demand", "_domestic_water_loss_m3", skip_first_day=True
    ).sum()
    industry_water_loss = _read_evaluation_series_with_date_index(
        folder, "hydrology.water_demand", "_industry_water_loss_m3", skip_first_day=True
    ).sum()
    livestock_water_loss = _read_evaluation_series_with_date_index(
        folder,
        "hydrology.water_demand",
        "_livestock_water_loss_m3",
        skip_first_day=True,
    ).sum()

    river_outflow = _read_evaluation_series_with_date_index(
        folder, "hydrology.routing", "_total_outflow_at_pits_m3", skip_first_day=True
    ).sum()

    transpiration = _read_evaluation_series_with_date_index(
        folder, "hydrology.landsurface", "_transpiration_m", skip_first_day=True
    ).sum()
    bare_soil_evaporation = _read_evaluation_series_with_date_index(
        folder, "hydrology.landsurface", "_bare_soil_evaporation_m", skip_first_day=True
    ).sum()
    open_water_evaporation = _read_evaluation_series_with_date_index(
        folder,
        "hydrology.landsurface",
        "_open_water_evaporation_m",
        skip_first_day=True,
    ).sum()
    interception_evaporation = _read_evaluation_series_with_date_index(
        folder,
        "hydrology.landsurface",
        "_interception_evaporation_m",
        skip_first_day=True,
    ).sum()
    sublimation_or_deposition = _read_evaluation_series_with_date_index(
        folder,
        "hydrology.landsurface",
        "_sublimation_or_deposition_m",
        skip_first_day=True,
    ).sum()
    river_evaporation = _read_evaluation_series_with_date_index(
        folder,
        "hydrology.routing",
        "_total_evaporation_in_rivers_m3",
        skip_first_day=True,
    ).sum()
    waterbody_evaporation = _read_evaluation_series_with_date_index(
        folder,
        "hydrology.routing",
        "_total_waterbody_evaporation_m3",
        skip_first_day=True,
    ).sum()

    hierarchy: dict[str, Any] = {
        "in": {
            "rain": rain,
            "snow": snow,
        },
        "out": {
            "evapotranspiration": {
                "transpiration": transpiration,
                "bare soil evaporation": bare_soil_evaporation,
                "open water evaporation": open_water_evaporation,
                "interception evaporation": interception_evaporation,
                "river evaporation": river_evaporation,
                "waterbody evaporation": waterbody_evaporation,
            },
            "river outflow": river_outflow,
            "water demand": {
                "domestic water loss": domestic_water_loss,
                "industry water loss": industry_water_loss,
                "livestock water loss": livestock_water_loss,
            },
        },
        "storage change": abs(storage_change),
    }

    if sublimation_or_deposition > 0:
        hierarchy["in"]["deposition"] = sublimation_or_deposition
    else:
        hierarchy["out"]["evapotranspiration"]["sublimation"] = abs(
            sublimation_or_deposition
        )

    if storage_change > 0:
        order: list[str] = ["in", "out", "storage change"]
    else:
        order: list[str] = ["storage change", "in", "out"]

    hierarchy = {key: hierarchy[key] for key in order}

    water_circle = plot_sunburst(hierarchy, title="water circle")

    if export:
        water_circle.savefig(
            self.water_circle_output_folder / "water_circle.svg",
        )

    return water_circle


def plot_water_balance(
    self: Hydrology,
    run_name: str,
    export: bool = True,
) -> None:
    """Create a csv file and plot showing the water balance components.

    Args:
        self: Hydrology evaluator providing model settings and output paths.
        run_name: Name of the run to evaluate.
        export: Whether to export the water balance plot to a file.

    Notes:
        Potential evapotranspiration is shown as an optional context series when
        the corresponding report output is available. It is not included in
        the actual water balance totals.

    Raises:
        ValueError: If the water balance dataframe does not contain any rows.
    """
    folder: Path = (
        Path(self.model.config["general"]["output_folder"]) / run_name / "report"
    )
    water_balance_df_m3_per_timestep: pd.DataFrame = _load_water_balance_dataframe(
        folder
    )
    context_series: dict[str, pd.Series] = _load_contextual_water_balance_series(folder)

    if water_balance_df_m3_per_timestep.empty:
        raise ValueError("No water balance data available for plotting.")

    yearly_totals_df_m3_per_year: pd.DataFrame = (
        water_balance_df_m3_per_timestep.resample("YE").sum()
    )
    yearly_totals_df_m3_per_year.to_csv(folder / "water_balance_yearly.csv")
    self.model.logger.info("Water balance yearly values saved.")

    years: pd.Index = yearly_totals_df_m3_per_year.index.year  # ty:ignore[unresolved-attribute]
    n_years: int = len(years)

    bar_chart_fig, bar_chart_axes = plt.subplots(
        n_years, 1, figsize=(16, 4 * n_years), sharex=True
    )
    if n_years == 1:
        bar_chart_axes = [bar_chart_axes]

    inputs_cols = [
        c for c in yearly_totals_df_m3_per_year.columns if c.startswith("in_")
    ]
    outputs_cols = [
        c for c in yearly_totals_df_m3_per_year.columns if c.startswith("out_")
    ]
    storage_cols = [
        c for c in yearly_totals_df_m3_per_year.columns if "storage" in c.lower()
    ]
    yearly_context_series: dict[str, pd.Series] = {
        series_name: series.resample("YE").sum()
        for series_name, series in context_series.items()
    }

    # legend building
    legend_handles = []
    legend_labels = []

    # Colormaps
    input_cmap = mcolormaps["Blues"]
    output_cmap = mcolormaps["Set3"]
    storage_cmap = mcolormaps["Greens"]

    # Assign distinct colors per column
    input_colors = {
        col: input_cmap(0.4 + 0.5 * i / max(1, len(inputs_cols) - 1))
        for i, col in enumerate(inputs_cols)
    }

    output_colors = {
        col: output_cmap(i % output_cmap.N) for i, col in enumerate(outputs_cols)
    }

    storage_colors = {
        col: storage_cmap(0.5 + 0.4 * i / max(1, len(storage_cols) - 1))
        for i, col in enumerate(storage_cols)
    }

    def add_legend_entry(handle: Any, label: str) -> None:
        if label not in legend_labels:
            legend_handles.append(handle)
            legend_labels.append(label)

    for ax, year in zip(bar_chart_axes, years):
        row = yearly_totals_df_m3_per_year.loc[
            [d.year == year for d in yearly_totals_df_m3_per_year.index]
        ].iloc[0]

        bottom = 0
        for col in inputs_cols:
            label = col.replace("in_", "").replace("_", " ")
            bar_container = ax.bar(
                "inputs",
                row[col],
                bottom=bottom,
                color=input_colors[col],
            )
            add_legend_entry(bar_container[0], f"input • {label}")
            bottom += row[col]

        bottom = 0
        for col in outputs_cols:
            label = col.replace("out_", "").replace("_", " ")
            bar_container = ax.bar(
                "outputs",
                row[col],
                bottom=bottom,
                color=output_colors[col],
            )
            add_legend_entry(bar_container[0], f"output • {label}")
            bottom += row[col]

        for col in storage_cols:
            label = col.replace("_", " ")
            bar_container = ax.bar(
                "storage",
                row[col],
                color=storage_colors[col],
            )
            add_legend_entry(bar_container[0], label)

        for series_name, yearly_series in yearly_context_series.items():
            label = _format_water_balance_context_label(series_name)
            yearly_context_positions: list[int] = [
                position
                for position, timestamp in enumerate(yearly_series.index)
                if pd.Timestamp(timestamp).year == year
            ]
            context_value_m3_per_year: float = float(
                yearly_series.iloc[yearly_context_positions[0]]
            )
            bar_container = ax.bar(
                "context",
                context_value_m3_per_year,
                color="none",
                edgecolor="black",
                linewidth=1.5,
                hatch="//",
            )
            add_legend_entry(bar_container[0], label)

        ax.set_title(f"Water Balance – {year}")
        ax.set_ylabel("m3/year")

    bar_chart_fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
    )

    if export:
        bar_chart_fig_path = (
            self.water_balance_output_folder / "water_balance_yearly_subplots.svg"
        )
        plt.savefig(bar_chart_fig_path)
        self.model.logger.info(
            f"Water balance yearly plot saved as: {bar_chart_fig_path}"
        )

    plt.show()
    plt.close(bar_chart_fig)

    signed_water_balance_df_m3_per_timestep: pd.DataFrame = (
        water_balance_df_m3_per_timestep.copy()
    )
    output_columns: list[str] = [
        column_name
        for column_name in signed_water_balance_df_m3_per_timestep.columns
        if column_name.startswith("out_")
    ]
    # Plot outputs below zero so the full balance can be read on a single axis.
    signed_water_balance_df_m3_per_timestep.loc[
        :, output_columns
    ] = -signed_water_balance_df_m3_per_timestep.loc[:, output_columns]

    component_columns: list[str] = list(signed_water_balance_df_m3_per_timestep.columns)
    component_colors: dict[str, Any] = {
        column_name: mcolormaps["tab20"](
            color_index / max(1, len(component_columns) - 1)
        )
        for color_index, column_name in enumerate(component_columns)
    }
    component_labels: dict[str, str] = {
        column_name: _format_water_balance_component_label(column_name)
        for column_name in component_columns
    }
    total_area_m2: float = self.model.total_area_m2
    conversion_factor_mm_per_m3: float = 1000.0 / total_area_m2
    yearly_context_totals_mm: pd.DataFrame = pd.DataFrame(
        {
            series_name: series.resample("YE").sum() * 1000.0 / total_area_m2
            for series_name, series in context_series.items()
        }
    )
    yearly_context_totals_mm.index = pd.Index(
        [pd.Timestamp(timestamp).year for timestamp in yearly_context_totals_mm.index]
    )
    context_colors: dict[str, str] = {
        "potential_evapotranspiration": "#555555",
    }
    context_linestyles: dict[str, Literal["-", "--", "-.", ":"]] = {
        "potential_evapotranspiration": ":",
    }
    context_linewidths: dict[str, float] = {
        "potential_evapotranspiration": 1.1,
    }
    context_labels: dict[str, str] = {
        series_name: _format_water_balance_context_label(series_name)
        for series_name in context_series
    }
    yearly_totals_mm: pd.DataFrame | None = None

    yearly_totals_mm = _create_yearly_totals_summary_mm(
        water_balance_df_m3_per_timestep,
        total_area_m2,
    )

    time_index: pd.DatetimeIndex = pd.DatetimeIndex(
        signed_water_balance_df_m3_per_timestep.index
    )
    timestep_label: str = _get_datetime_index_step_label(time_index)
    signed_water_balance_df_mm_per_timestep: pd.DataFrame = (
        signed_water_balance_df_m3_per_timestep * conversion_factor_mm_per_m3
    )
    context_series_mm_per_timestep: dict[str, pd.Series] = {
        series_name: series * conversion_factor_mm_per_m3
        for series_name, series in context_series.items()
    }
    full_figure, full_axis = plt.subplots(figsize=(15, 14))
    for column_name in component_columns:
        full_axis.plot(
            signed_water_balance_df_mm_per_timestep.index,
            signed_water_balance_df_mm_per_timestep[column_name],
            label=component_labels[column_name],
            color=component_colors[column_name],
            linewidth=0.7,
        )
    for series_name, series in context_series_mm_per_timestep.items():
        full_axis.plot(
            series.index,
            series,
            label=context_labels[series_name],
            color=context_colors.get(series_name, "black"),
            linewidth=context_linewidths.get(series_name, 1.0),
            linestyle=context_linestyles.get(series_name, ":"),
            alpha=0.9,
        )

    _format_timeseries_axis(
        full_axis,
        title=f"Water Balance Over Time - {run_name}",
        y_label=f"mm/{timestep_label}",
        time_index=time_index,
        draw_zero_line=True,
    )
    full_axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(3, len(component_columns)),
        fontsize=9,
    )
    full_figure.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.26)

    year_values: np.ndarray = pd.Series(time_index).dt.year.to_numpy(dtype=int)
    years: list[int] = sorted(np.unique(year_values).tolist())
    yearly_figure, yearly_axes = plt.subplots(
        len(years),
        1,
        figsize=(15, max(7.2 * len(years), 11.0)),
        sharey=True,
    )
    if len(years) == 1:
        yearly_axes = [yearly_axes]

    for axis, year in zip(yearly_axes, years, strict=True):
        year_mask: np.ndarray = year_values == year
        yearly_df_mm_per_timestep: pd.DataFrame = (
            signed_water_balance_df_mm_per_timestep.loc[year_mask]
        )
        for column_name in component_columns:
            axis.plot(
                yearly_df_mm_per_timestep.index,
                yearly_df_mm_per_timestep[column_name],
                color=component_colors[column_name],
                linewidth=0.55,
            )
        for series_name, series in context_series_mm_per_timestep.items():
            yearly_context_series: pd.Series = series.loc[year_mask]
            axis.plot(
                yearly_context_series.index,
                yearly_context_series,
                color=context_colors.get(series_name, "black"),
                linewidth=context_linewidths.get(series_name, 1.0),
                linestyle=context_linestyles.get(series_name, ":"),
                alpha=0.9,
            )

        _format_timeseries_axis(
            axis,
            title=f"Water Balance Over Time - {year}",
            y_label=f"mm/{timestep_label}",
            year=year,
            draw_zero_line=True,
        )
        if yearly_totals_mm is not None:
            _add_yearly_totals_caption(
                axis,
                year,
                yearly_totals_mm,
                component_labels,
                yearly_context_totals_mm,
                context_labels,
            )

    yearly_axes[-1].set_xlabel("Time")
    yearly_handles: list[Line2D] = [
        Line2D([0], [0], color=component_colors[column_name], linewidth=0.9)
        for column_name in component_columns
    ]
    yearly_labels: list[str] = [
        component_labels[column_name] for column_name in component_columns
    ]
    yearly_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=context_colors.get(series_name, "black"),
                linewidth=context_linewidths.get(series_name, 1.0),
                linestyle=context_linestyles.get(series_name, ":"),
            )
            for series_name in context_series
        ]
    )
    yearly_labels.extend(
        [context_labels[series_name] for series_name in context_series]
    )
    yearly_figure.legend(
        yearly_handles,
        yearly_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(3, len(yearly_labels)),
        frameon=False,
    )
    yearly_figure.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.99,
        bottom=0.12 + 0.03 * max(0, (len(yearly_labels) - 1) // 3),
        hspace=0.7,
    )

    if export:
        full_path: Path = (
            self.water_balance_output_folder / "water_balance_timeseries.svg"
        )
        yearly_path: Path = (
            self.water_balance_output_folder / "water_balance_timeseries_yearly.svg"
        )
        full_figure.savefig(full_path)
        yearly_figure.savefig(yearly_path)
        self.model.logger.info(f"Water balance time-series plot saved as: {full_path}")
        self.model.logger.info(
            f"Water balance yearly time-series plot saved as: {yearly_path}"
        )

    plt.show()
    plt.close(full_figure)
    plt.close(yearly_figure)

    top_soil_water_balance_df_m3_per_timestep: pd.DataFrame = (
        _load_top_soil_water_balance_dataframe(folder)
    )
    top_soil_context_series: dict[str, pd.Series] = (
        _load_contextual_top_soil_water_balance_series(folder)
    )

    signed_top_soil_water_balance_df_m3_per_timestep: pd.DataFrame = (
        top_soil_water_balance_df_m3_per_timestep.copy()
    )
    top_soil_output_columns: list[str] = [
        column_name
        for column_name in signed_top_soil_water_balance_df_m3_per_timestep.columns
        if column_name.startswith("out_")
    ]
    signed_top_soil_water_balance_df_m3_per_timestep.loc[
        :, top_soil_output_columns
    ] = -signed_top_soil_water_balance_df_m3_per_timestep.loc[
        :, top_soil_output_columns
    ]

    top_soil_component_columns: list[str] = list(
        signed_top_soil_water_balance_df_m3_per_timestep.columns
    )
    top_soil_component_colors: dict[str, Any] = {
        column_name: mcolormaps["Dark2"](
            color_index / max(1, len(top_soil_component_columns) - 1)
        )
        for color_index, column_name in enumerate(top_soil_component_columns)
    }
    if "storage_change" in top_soil_component_colors:
        top_soil_component_colors["storage_change"] = "black"
    top_soil_component_labels: dict[str, str] = {
        column_name: (
            "storage change (from top-soil storage)"
            if column_name == "storage_change"
            else _format_water_balance_component_label(column_name)
        )
        for column_name in top_soil_component_columns
    }
    top_soil_yearly_context_totals_mm: pd.DataFrame = pd.DataFrame(
        {
            series_name: series.resample("YE").sum() * 1000.0 / total_area_m2
            for series_name, series in top_soil_context_series.items()
        }
    )
    top_soil_yearly_context_totals_mm.index = pd.Index(
        [
            pd.Timestamp(timestamp).year
            for timestamp in top_soil_yearly_context_totals_mm.index
        ]
    )
    top_soil_context_colors: dict[str, str] = {
        "precipitation": "#72b7b2",
        "runoff": "#f58518",
        "snow": "#4c78a8",
        "potential_evapotranspiration": "#555555",
        "transpiration": "#54a24b",
    }
    top_soil_context_linestyles: dict[str, Literal["-", "--", "-.", ":"]] = {
        "precipitation": ":",
        "runoff": "--",
        "snow": "-.",
        "potential_evapotranspiration": ":",
        "transpiration": "-",
    }
    top_soil_context_linewidths: dict[str, float] = {
        "precipitation": 1.0,
        "runoff": 1.0,
        "snow": 1.0,
        "potential_evapotranspiration": 1.1,
        "transpiration": 1.0,
    }
    top_soil_context_labels: dict[str, str] = {
        series_name: _format_water_balance_context_label(series_name)
        for series_name in top_soil_context_series
    }
    top_soil_yearly_totals_mm: pd.DataFrame | None = None
    if total_area_m2 is not None:
        top_soil_yearly_totals_mm = _create_yearly_totals_summary_mm(
            top_soil_water_balance_df_m3_per_timestep,
            total_area_m2,
        )
    top_soil_component_linewidths: dict[str, float] = {
        column_name: 1.0 if column_name == "storage_change" else 0.7
        for column_name in top_soil_component_columns
    }
    top_soil_component_linestyles: dict[str, str] = {
        column_name: "--" if column_name == "storage_change" else "-"
        for column_name in top_soil_component_columns
    }
    top_soil_component_zorders: dict[str, int] = {
        column_name: 4 if column_name == "storage_change" else 2
        for column_name in top_soil_component_columns
    }
    if "storage_change" in top_soil_component_colors:
        top_soil_component_colors["storage_change"] = "black"

    top_soil_time_index: pd.DatetimeIndex = pd.DatetimeIndex(
        signed_top_soil_water_balance_df_m3_per_timestep.index
    )
    top_soil_timestep_label: str = _get_datetime_index_step_label(top_soil_time_index)
    signed_top_soil_water_balance_df_mm_per_timestep: pd.DataFrame = (
        signed_top_soil_water_balance_df_m3_per_timestep * conversion_factor_mm_per_m3
    )
    top_soil_context_series_mm_per_timestep: dict[str, pd.Series] = {
        series_name: series * conversion_factor_mm_per_m3
        for series_name, series in top_soil_context_series.items()
    }
    top_soil_full_figure, top_soil_full_axis = plt.subplots(figsize=(15, 13.0))
    for column_name in top_soil_component_columns:
        top_soil_full_axis.plot(
            signed_top_soil_water_balance_df_mm_per_timestep.index,
            signed_top_soil_water_balance_df_mm_per_timestep[column_name],
            label=top_soil_component_labels[column_name],
            color=top_soil_component_colors[column_name],
            linewidth=top_soil_component_linewidths[column_name],
            linestyle=top_soil_component_linestyles[column_name],
            zorder=top_soil_component_zorders[column_name],
        )
    for series_name, series in top_soil_context_series_mm_per_timestep.items():
        top_soil_full_axis.plot(
            series.index,
            series,
            label=top_soil_context_labels[series_name],
            color=top_soil_context_colors.get(series_name, "black"),
            linewidth=top_soil_context_linewidths.get(series_name, 1.0),
            linestyle=top_soil_context_linestyles.get(series_name, ":"),
            alpha=0.9,
            zorder=3,
        )

    _format_timeseries_axis(
        top_soil_full_axis,
        title=f"Top-Soil Water Balance Over Time - {run_name}",
        y_label=f"mm/{top_soil_timestep_label}",
        time_index=top_soil_time_index,
        draw_zero_line=True,
    )
    top_soil_full_axis.legend(
        frameon=False,
        loc="upper right",
        ncol=min(
            3,
            len(top_soil_component_columns) + len(top_soil_context_series),
        ),
        fontsize=9,
    )
    top_soil_full_figure.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.14)

    top_soil_year_values: np.ndarray = pd.Series(top_soil_time_index).dt.year.to_numpy(
        dtype=int
    )
    top_soil_years: list[int] = sorted(np.unique(top_soil_year_values).tolist())
    top_soil_yearly_figure, top_soil_yearly_axes = plt.subplots(
        len(top_soil_years),
        1,
        figsize=(15, max(6.4 * len(top_soil_years), 10.0)),
        sharey=True,
    )
    if len(top_soil_years) == 1:
        top_soil_yearly_axes = [top_soil_yearly_axes]

    for axis_index, (axis, year) in enumerate(
        zip(top_soil_yearly_axes, top_soil_years, strict=True)
    ):
        year_mask: np.ndarray = top_soil_year_values == year
        yearly_df_mm: pd.DataFrame = (
            signed_top_soil_water_balance_df_mm_per_timestep.loc[year_mask]
        )
        for column_name in top_soil_component_columns:
            axis.plot(
                yearly_df_mm.index,
                yearly_df_mm[column_name],
                color=top_soil_component_colors[column_name],
                linewidth=(0.8 if column_name == "storage_change" else 0.55),
                linestyle=top_soil_component_linestyles[column_name],
                zorder=top_soil_component_zorders[column_name],
                label=(
                    top_soil_component_labels[column_name] if axis_index == 0 else None
                ),
            )
        for series_name, series in top_soil_context_series_mm_per_timestep.items():
            yearly_context_series: pd.Series = series.loc[year_mask]
            axis.plot(
                yearly_context_series.index,
                yearly_context_series,
                color=top_soil_context_colors.get(series_name, "black"),
                linewidth=top_soil_context_linewidths.get(series_name, 1.0),
                linestyle=top_soil_context_linestyles.get(series_name, ":"),
                alpha=0.9,
                zorder=3,
                label=(
                    top_soil_context_labels[series_name] if axis_index == 0 else None
                ),
            )

        _format_timeseries_axis(
            axis,
            title=f"Top-Soil Water Balance Over Time - {year}",
            y_label=f"mm/{top_soil_timestep_label}",
            year=year,
            draw_zero_line=True,
        )
        if top_soil_yearly_totals_mm is not None:
            _add_yearly_totals_caption(
                axis,
                year,
                top_soil_yearly_totals_mm,
                top_soil_component_labels,
                top_soil_yearly_context_totals_mm,
                top_soil_context_labels,
            )

        if axis_index == 0:
            axis.legend(
                frameon=False,
                loc="upper right",
                ncol=min(
                    3,
                    len(top_soil_component_columns) + len(top_soil_context_series),
                ),
                fontsize=8.5,
            )

    top_soil_yearly_axes[-1].set_xlabel("Time")
    top_soil_yearly_figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.96,
        bottom=0.08,
        hspace=0.68,
    )

    if export:
        top_soil_full_path: Path = (
            self.water_balance_output_folder / "water_balance_top_soil_timeseries.svg"
        )
        top_soil_yearly_path: Path = (
            self.water_balance_output_folder
            / "water_balance_top_soil_timeseries_yearly.svg"
        )
        top_soil_full_figure.savefig(top_soil_full_path)
        top_soil_yearly_figure.savefig(top_soil_yearly_path)
        self.model.logger.info(
            f"Top-soil water balance time-series plot saved as: {top_soil_full_path}"
        )
        self.model.logger.info(
            f"Top-soil water balance yearly time-series plot saved as: {top_soil_yearly_path}"
        )

    plt.show()
    plt.close(top_soil_full_figure)
    plt.close(top_soil_yearly_figure)


def plot_water_storage(
    self: Hydrology,
    run_name: str,
    export: bool = True,
) -> None:
    """Plot reported water storage component time series for the full run and per year.

    Notes:
        The currently available storage components come directly from
        `WATER_STORAGE_REPORT_CONFIG` in `geb.reporter`. At present these are the
        reported soil water content layers.

    Args:
        self: Hydrology evaluator providing model settings and output paths.
        run_name: Name of the run to evaluate.
        export: Whether to export the water storage plots to files.

    Raises:
        ValueError: If the water storage dataframe does not contain any rows.
    """
    folder: Path = (
        Path(self.model.config["general"]["output_folder"]) / run_name / "report"
    )
    storage_module: str = "hydrology.landsurface"
    storage_specs: dict[str, tuple[str, str]] = {
        reported_name.removeprefix("_").removesuffix("_m"): (
            storage_module,
            reported_name,
        )
        for reported_name in WATER_STORAGE_REPORT_CONFIG[storage_module]
    }
    try:
        water_storage_df_m: pd.DataFrame = pd.DataFrame(
            _load_named_evaluation_series(folder, storage_specs)
        ).sort_index()
    except FileNotFoundError as error:
        raise ValueError(
            "Water storage outputs are missing. Enable report._water_storage "
            "during the run before plotting water storage."
        ) from error

    if water_storage_df_m.empty:
        raise ValueError("No water storage data available for plotting.")

    component_columns: list[str] = list(water_storage_df_m.columns)
    component_colors: dict[str, Any] = {
        column_name: mcolormaps["viridis"](
            0.15 + 0.75 * color_index / max(1, len(component_columns) - 1)
        )
        for color_index, column_name in enumerate(component_columns)
    }
    component_labels: dict[str, str] = {
        column_name: column_name.replace("_", " ") for column_name in component_columns
    }

    time_index: pd.DatetimeIndex = pd.DatetimeIndex(water_storage_df_m.index)
    full_figure, full_axis = plt.subplots(figsize=(14, 6.5))
    for column_name in component_columns:
        full_axis.plot(
            water_storage_df_m.index,
            water_storage_df_m[column_name],
            label=component_labels[column_name],
            color=component_colors[column_name],
            linewidth=1.6,
        )

    _format_timeseries_axis(
        full_axis,
        title=f"Water Storage Over Time - {run_name}",
        y_label="m",
        time_index=time_index,
    )
    full_axis.legend(
        frameon=False,
        loc="upper right",
        ncol=min(2, len(component_columns)),
        fontsize=9,
    )
    full_figure.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.12)

    year_values: np.ndarray = pd.Series(time_index).dt.year.to_numpy(dtype=int)
    years: list[int] = sorted(np.unique(year_values).tolist())
    yearly_figure, yearly_axes = plt.subplots(
        len(years),
        1,
        figsize=(14, max(3.2 * len(years), 5.0)),
        sharey=True,
    )
    if len(years) == 1:
        yearly_axes = [yearly_axes]

    for axis_index, (axis, year) in enumerate(zip(yearly_axes, years, strict=True)):
        year_mask: np.ndarray = year_values == year
        yearly_df_m: pd.DataFrame = water_storage_df_m.loc[year_mask]
        for column_name in component_columns:
            axis.plot(
                yearly_df_m.index,
                yearly_df_m[column_name],
                color=component_colors[column_name],
                linewidth=1.3,
                label=component_labels[column_name] if axis_index == 0 else None,
            )

        _format_timeseries_axis(
            axis, title=f"Water Storage Over Time - {year}", y_label="m", year=year
        )

        if axis_index == 0:
            axis.legend(
                frameon=False,
                loc="upper right",
                ncol=min(2, len(component_columns)),
                fontsize=8.5,
            )

    yearly_axes[-1].set_xlabel("Time")
    yearly_figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.96,
        bottom=0.08,
        hspace=0.26,
    )

    if export:
        full_path: Path = (
            self.water_storage_output_folder / "water_storage_timeseries.svg"
        )
        yearly_path: Path = (
            self.water_storage_output_folder / "water_storage_timeseries_yearly.svg"
        )
        full_figure.savefig(full_path)
        yearly_figure.savefig(yearly_path)
        self.model.logger.info(f"Water storage time-series plot saved as: {full_path}")
        self.model.logger.info(
            f"Water storage yearly time-series plot saved as: {yearly_path}"
        )

    plt.show()
    plt.close(full_figure)
    plt.close(yearly_figure)


def _format_timeseries_axis(
    axis: plt.Axes,
    title: str,
    y_label: str,
    time_index: pd.DatetimeIndex | None = None,
    year: int | None = None,
    draw_zero_line: bool = False,
) -> None:
    """Format a full-period or calendar-year water-balance axis.

    Args:
        axis: Axis receiving date ticks, labels, and a grid.
        title: Axis title.
        y_label: Y-axis label including units.
        time_index: Full-period timestamps when year is omitted.
        year: Calendar year for monthly ticks and full-year limits.
        draw_zero_line: Whether to draw a dashed zero reference.

    Returns:
        None. Updates the axis in place.

    Raises:
        ValueError: If neither a year nor a time index is provided.
    """  # noqa: DOC202
    if year is None and time_index is None:
        raise ValueError("Provide a year or a time index for the axis.")
    if draw_zero_line:
        axis.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    axis.set(title=title, ylabel=y_label)
    axis.margins(x=0)
    axis.grid(True, alpha=0.5, color="0.8")
    if year is not None:
        axis.set_xlim(
            mdates.date2num(pd.Timestamp(year=year, month=1, day=1)),
            mdates.date2num(pd.Timestamp(year=year, month=12, day=31, hour=23)),
        )
        axis.xaxis.set_major_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        assert time_index is not None
        axis.set(xlabel="Time", xlim=(time_index.min(), time_index.max()))
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
        axis.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(axis.xaxis.get_major_locator())
        )


def _format_water_balance_component_label(column_name: str) -> str:
    """Format a water balance column name for plot legends.

    Args:
        column_name: Raw dataframe column name.

    Returns:
        Human-readable legend label.
    """
    simplified_column_name: str = column_name
    if simplified_column_name.startswith("in_"):
        simplified_column_name = simplified_column_name.removeprefix("in_")
    elif simplified_column_name.startswith("out_"):
        simplified_column_name = simplified_column_name.removeprefix("out_")

    simplified_column_name = simplified_column_name.removeprefix("evapotranspiration_")
    simplified_column_name = simplified_column_name.removeprefix("water_demand_")
    return simplified_column_name.replace("_", " ")


def _format_water_balance_context_label(series_name: str) -> str:
    """Format an optional water-balance context series label for plot legends.

    Args:
        series_name: Raw context series name.

    Returns:
        Human-readable legend label.
    """
    if series_name == "potential_evapotranspiration":
        return "potential ET"
    return series_name.replace("_", " ")


def _add_yearly_totals_caption(
    axis: plt.Axes,
    year: int,
    yearly_totals_mm: pd.DataFrame,
    component_labels: dict[str, str],
    yearly_context_totals_mm: pd.DataFrame | None = None,
    context_labels: dict[str, str] | None = None,
) -> None:
    """Add a compact annual totals caption to a yearly water-balance subplot.

    Args:
        axis: Parent axis that receives the caption.
        year: Calendar year represented by the subplot.
        yearly_totals_mm: Annual totals indexed by year and expressed in mm/year.
        component_labels: Human-readable labels for each component column.
        yearly_context_totals_mm: Optional annual context totals indexed by year and
            expressed in mm/year.
        context_labels: Human-readable labels for each context series column.
    """
    if yearly_totals_mm.empty or year not in yearly_totals_mm.index:
        return

    yearly_values_mm: pd.Series = yearly_totals_mm.loc[year]
    ordered_columns: list[str] = list(yearly_totals_mm.columns)
    input_columns: list[str] = [
        column_name for column_name in ordered_columns if column_name.startswith("in_")
    ]
    output_columns: list[str] = [
        column_name for column_name in ordered_columns if column_name.startswith("out_")
    ]
    storage_columns: list[str] = [
        column_name
        for column_name in ordered_columns
        if column_name not in input_columns and column_name not in output_columns
    ]
    caption_lines: list[str] = []
    components_per_line: int = 4
    input_total_mm: float = float(yearly_values_mm[input_columns].sum())
    output_total_mm: float = float(-yearly_values_mm[output_columns].sum())
    caption_lines.extend(
        _format_yearly_totals_caption_lines(
            prefix="inputs",
            column_names=input_columns,
            values_mm=yearly_values_mm,
            labels=component_labels,
            items_per_line=components_per_line,
        )
    )
    caption_lines.extend(
        _format_yearly_totals_caption_lines(
            prefix="outputs",
            column_names=output_columns,
            values_mm=yearly_values_mm,
            labels=component_labels,
            items_per_line=components_per_line,
        )
    )
    if storage_columns:
        caption_lines.extend(
            _format_yearly_totals_caption_lines(
                prefix="storage",
                column_names=storage_columns,
                values_mm=yearly_values_mm,
                labels=component_labels,
                items_per_line=components_per_line,
            )
        )

    caption_text_lines: list[str] = ["mm/year\n" + "\n".join(caption_lines)]
    caption_text_lines.append(
        f"sum input: {input_total_mm:.1f} | sum output: {output_total_mm:.1f}"
    )

    if (
        yearly_context_totals_mm is not None
        and context_labels is not None
        and not yearly_context_totals_mm.empty
        and year in yearly_context_totals_mm.index
    ):
        context_values_mm: pd.Series = yearly_context_totals_mm.loc[year]
        ordered_context_columns: list[str] = list(yearly_context_totals_mm.columns)
        context_caption_parts: list[str] = [
            f"{context_labels[column_name]}: {context_values_mm[column_name]:.1f}"
            for column_name in ordered_context_columns
        ]
        caption_text_lines.append("context: " + " | ".join(context_caption_parts))

    caption_text: str = "\n".join(caption_text_lines)
    axis.text(
        0.01,
        -0.24,
        caption_text,
        transform=axis.transAxes,
        fontsize=6,
        va="top",
        ha="left",
        linespacing=1.15,
        clip_on=False,
    )


def _format_yearly_totals_caption_lines(
    prefix: str,
    column_names: list[str],
    values_mm: pd.Series,
    labels: dict[str, str],
    items_per_line: int,
) -> list[str]:
    """Format grouped yearly totals caption lines for one component direction.

    Args:
        prefix: Direction label such as `inputs` or `outputs`.
        column_names: Ordered component columns to render.
        values_mm: Annual component totals for one year (mm/year).
        labels: Human-readable labels for each component column.
        items_per_line: Maximum number of caption items per rendered line.

    Returns:
        Caption lines with the direction prefix shown only once per line.
    """
    if not column_names:
        return []

    lines: list[str] = []
    for start_index in range(0, len(column_names), items_per_line):
        chunk: list[str] = column_names[start_index : start_index + items_per_line]
        chunk_text: str = " | ".join(
            f"{labels[column_name]}: {values_mm[column_name]:.1f}"
            for column_name in chunk
        )
        lines.append(f"{prefix}: {chunk_text}")
    return lines
