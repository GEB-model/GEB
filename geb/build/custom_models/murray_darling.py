"""Utilities to set up Australian water prices and drip irrigation prices."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geb.build.methods import build_method

from .. import GEBModel


class Agents(GEBModel):
    """Build methods for agents in GEB, including Murray–Darling-specific logic."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize an Agents model.

        Forwards all arguments to the base ``GEBModel`` initializer.

        Args:
            *args: Positional arguments forwarded to ``GEBModel``.
            **kwargs: Keyword arguments forwarded to ``GEBModel``.
        """
        super().__init__(*args, **kwargs)

    @build_method(depends_on=["set_time_range", "setup_economic_data"])
    def setup_crop_cultivation_costs_by_reference_year(
        self,
        crop_costs: dict[str, dict[str, Any]],
        n_crops: int = 26,
    ) -> None:
        """Build crop cultivation cost time series using crop-specific reference years.

        For each region and crop, construct an annual time series of cultivation
        costs (e.g. USD per square metre) from a single reference-year value. The
        given reference-year value is inflated forward and deflated backward using
        region-specific inflation rates, and the resulting time series are stored
        in the internal dictionary.

        Notes:
            The input ``crop_costs`` is expected to contain, for each crop key,
            at least the fields ``reference_year`` and ``value``. Reference years
            must fall within the model time range and be present in the inflation
            time axis.

        Args:
            crop_costs: Mapping from crop identifier to a specification that
                includes a reference year and a per-area cost value. Keys are crop
                IDs that can be cast to integers.
            n_crops: Total number of potential crops. This is used to ensure that
                the output dictionary has entries for all crop IDs in the range
                from 0 to ``n_crops - 1``, even if some are missing from
                ``crop_costs``.


        Raises:
            ValueError: If a crop specification is missing the required fields
                ``reference_year`` or ``value``, or if the reference year lies
                outside the model time range or is missing from the inflation
                time axis.
        """
        inflation_rates = self.dict["socioeconomics/inflation_rates"]
        regions = list(inflation_rates["data"].keys())

        start_year = int(self.start_date.year)
        end_year = int(self.end_date.year)
        years = list(range(start_year, end_year + 1))

        # Inflation time axis lookup (often stored as strings)
        infl_time_years = [int(y) for y in inflation_rates["time"]]
        year_to_idx = {y: i for i, y in enumerate(infl_time_years)}

        # Normalize YAML keys to canonical string crop ids
        crop_costs_norm: dict[str, dict[str, Any]] = {
            str(int(k)): v for k, v in crop_costs.items()
        }

        def inflate_series_from_reference(
            initial_value: float | None,
            reference_year: int,
            region_id: str,
        ) -> list[float]:
            """Create an annual series from a reference-year cultivation cost.

            Given a single reference-year cost, build a time series over the model
            time range by applying regional inflation rates backward and forward
            in time.

            Notes:
                If ``initial_value`` is missing or not a valid number, the function
                returns a series of missing values for all years. The reference
                year must lie within the model time range and be present in the
                inflation rate time axis.

            Args:
                initial_value: Cultivation cost in the reference year. If None or
                    not a valid number, the result is a list of missing values.
                reference_year: Year for which ``initial_value`` applies. Must lie
                    between ``start_year`` and ``end_year``.
                region_id: Region identifier used to select the appropriate
                    inflation series.

            Returns:
                A list of annual cultivation costs for each year between
                ``start_year`` and ``end_year`` (inclusive).

            Raises:
                ValueError: If the reference year is outside the model time range
                    or not present in the inflation time axis.
            """
            if initial_value is None or (
                isinstance(initial_value, float) and np.isnan(initial_value)
            ):
                return [np.nan] * len(years)

            reference_year = int(reference_year)

            if reference_year < start_year or reference_year > end_year:
                raise ValueError(
                    f"reference_year={reference_year} must be within [{start_year}, {end_year}] "
                    f"(region_id={region_id})."
                )
            if reference_year not in year_to_idx:
                raise ValueError(
                    "reference_year="
                    f"{reference_year} not present in socioeconomics/"
                    "inflation_rates['time']."
                )

            s = pd.Series(index=years, dtype="float64")
            s.loc[reference_year] = float(initial_value)

            # Forward: y-1 -> y multiply by inflation rate for year y
            for y in range(reference_year + 1, end_year + 1):
                infl = inflation_rates["data"][region_id][year_to_idx[y]]
                s.loc[y] = s.loc[y - 1] * infl

            # Backward: y+1 -> y divide by inflation rate for year y+1
            for y in range(reference_year - 1, start_year - 1, -1):
                infl_next = inflation_rates["data"][region_id][year_to_idx[y + 1]]
                s.loc[y] = s.loc[y + 1] / infl_next

            return s.tolist()

        cultivation_costs: dict[str, Any] = {
            "type": "time_series",
            "time": years,
            "data": {},
        }

        for region_id in regions:
            region_dict: dict[str, list[float]] = {}

            for crop_id in range(int(n_crops)):
                crop_key = str(crop_id)
                spec = crop_costs_norm.get(crop_key)

                if spec is None:
                    region_dict[crop_key] = [np.nan] * len(years)
                    continue

                if "reference_year" not in spec or "value" not in spec:
                    raise ValueError(
                        f"crop_costs['{crop_key}'] must contain 'reference_year' and "
                        f"'value'. Got keys: {list(spec.keys())}"
                    )

                region_dict[crop_key] = inflate_series_from_reference(
                    initial_value=spec["value"],
                    reference_year=spec["reference_year"],
                    region_id=str(region_id),
                )

            cultivation_costs["data"][str(region_id)] = region_dict

        self.set_dict(cultivation_costs, name="crops/cultivation_costs")

    @build_method(depends_on=["setup_economic_data"])
    def setup_irrigation_efficiency_australia(
        self,
        start_year: int,
        end_year: int,
    ) -> None:
        """Set up irrigation efficiency trajectories for Australian regions.

        Read historical irrigation type data, estimate missing values, fit several
        time-series models (linear, linear plus sinusoidal, and logistic), and
        create region-specific trajectories of irrigation type fractions for the
        model period.

        Notes:
            The method creates diagnostic plots on disk and then stores the final
            trajectories as dictionaries that can be used as model inputs. The
            stored irrigation fractions are constrained to be between zero and
            one and to sum to one for each year and region.

        Args:
            start_year: First simulation year (inclusive) for which trajectories
                are generated.
            end_year: Last simulation year (inclusive) for which trajectories
                are generated.

        """
        fp = Path("calibration_data/water_use/murray_water_use.csv")

        data_df = pd.read_csv(fp)
        data_df["Value"] = pd.to_numeric(data_df["Value"], errors="coerce")

        # Filter and rename columns
        irrigation_types = [
            "surface_irrigation",
            "drip_or_trickle_irrigation",
            "sprinkler_irrigation",
        ]
        irrigation_df = data_df[data_df["Description"].isin(irrigation_types)]
        irrigation_df = irrigation_df.rename(columns={"Year": "time"})

        # Parse e.g. "2002/3" -> datetime(2003-06-30)
        def parse_water_year(wy_str: str) -> pd.Timestamp:
            """Parse a water year string into a June-30 timestamp.

            Notes:
                The expected format is a string with a slash, such as "2002/3",
                where the calendar year is taken as the first part plus one, and
                June 30 of that year is returned.

            Args:
                wy_str: Water year string to parse.

            Returns:
                A pandas timestamp corresponding to the June 30 date of the
                water year.
            """
            return pd.to_datetime(f"{int(wy_str.split('/')[0]) + 1}-06-30")

        irrigation_df["time"] = irrigation_df["time"].apply(parse_water_year)

        # Pivot and compute total + fraction
        irrigation_pivot = irrigation_df.pivot_table(
            index=["time", "Region"],
            columns="Description",
            values="Value",
            aggfunc="first",
        )
        irrigation_pivot["total_irrigation"] = irrigation_pivot[irrigation_types].sum(
            axis=1
        )
        for irr_type in irrigation_types:
            irrigation_pivot["fraction_" + irr_type] = (
                irrigation_pivot[irr_type] / irrigation_pivot["total_irrigation"]
            )

        # Melt fractions into long format, combine with original
        fraction_cols = ["fraction_" + x for x in irrigation_types]
        fraction_df = (
            irrigation_pivot[fraction_cols]
            .reset_index()
            .melt(
                id_vars=["time", "Region"],
                value_vars=fraction_cols,
                var_name="Description",
                value_name="Value",
            )
        )
        combined_data_df = pd.concat(
            [data_df, fraction_df[["time", "Region", "Description", "Value"]]],
            ignore_index=True,
        )

        # Pivot for ratio-based estimation
        pivot_df = combined_data_df.pivot_table(
            index=["time", "Description"],
            columns="Region",
            values="Value",
            aggfunc="first",
        )

        # Estimate missing data via ratio
        ref_regions = {
            "murray": "NSW",
            "goulburn_broken": "VICT",
            "north_central": "VICT",
            "north_east": "VICT",
        }
        for target, ref in ref_regions.items():
            if target in pivot_df.columns and ref in pivot_df.columns:
                pivot_df["Ratio"] = pivot_df[target] / pivot_df[ref]
                desc_ratio = pivot_df.groupby(level="Description")["Ratio"].mean()
                pivot_df["Avg_Ratio"] = pivot_df.index.get_level_values(
                    "Description"
                ).map(desc_ratio)
                pivot_df["Estimated"] = pivot_df[ref] * pivot_df["Avg_Ratio"]
                pivot_df[target] = pivot_df[target].fillna(pivot_df["Estimated"])
                pivot_df = pivot_df.drop(columns=["Ratio", "Avg_Ratio", "Estimated"])

        # Normalize fraction rows to sum to 1
        frac_desc = ["fraction_" + x for x in irrigation_types]
        frac_df = pivot_df.loc[
            pivot_df.index.get_level_values("Description").isin(frac_desc)
        ].copy()
        frac_df = frac_df.fillna(0)
        frac_sum = frac_df.groupby(level=["time"]).sum()
        frac_norm = frac_df.div(frac_sum)
        pivot_df.update(frac_norm)

        # Flatten for interpolation
        df_flat = pivot_df.reset_index()
        df_flat = df_flat.sort_values("time")
        min_date = df_flat["time"].min()
        max_date = df_flat["time"].max()

        # For each Description, reindex on full date range, interpolate by time
        all_desc = df_flat["Description"].unique()
        groups: list[pd.DataFrame] = []
        for d in all_desc:
            g = df_flat[df_flat["Description"] == d].copy()

            g = g.set_index("time")
            g.index = pd.to_datetime(g.index, errors="coerce")
            g = g.sort_index()

            dates = pd.date_range(min_date, max_date, freq="YE-JUN")
            g = g.reindex(dates, fill_value=np.nan)

            non_numeric_cols = g.select_dtypes(exclude=["number"]).columns
            temp_non_numeric = g[non_numeric_cols]
            g = g.drop(columns=non_numeric_cols)

            g = g.infer_objects(copy=False)
            g = g.interpolate(method="time", limit_direction="both")

            g[non_numeric_cols] = temp_non_numeric
            g["Description"] = d
            groups.append(g)

        df_flat_int = pd.concat(groups).reset_index().rename(columns={"index": "time"})
        pivot_df = df_flat_int.pivot_table(
            index=["time", "Description"],
            aggfunc="first",
        ).sort_index(level="time")

        import matplotlib.pyplot as plt

        output_dir = Path("calibration_data/water_use/plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        pivot = pivot_df.copy().reset_index()
        pivot["time"] = pd.to_datetime(pivot["time"])
        pivot["year"] = pivot["time"].dt.year.astype(float)

        regions_to_fit = ["NSW", "VICT"]

        # =========================
        # 1. SIMPLE PROGRESSION PLOTS (ALL REGIONS)
        # =========================

        long_all = pivot.melt(
            id_vars=["time", "year", "Description"],
            var_name="Region",
            value_name="fraction",
        )

        for region, g in long_all.groupby("Region"):
            fig, ax = plt.subplots()
            for desc, gg in g.groupby("Description"):
                ax.plot(gg["time"], gg["fraction"], marker="o", label=desc)

            ax.set_title(f"Irrigation fractions over time – {region}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Fraction")
            ax.legend()
            ax.grid(True)

            fig.savefig(
                output_dir / f"irrigation_fractions_{region}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        print("Saved simple progression plots for all regions.")

        # =========================
        # 2. PREPARE DATA FOR NSW & VICT (WIDE, COMPOSITIONAL)
        # =========================

        desc_map = {
            "fraction_drip_or_trickle_irrigation": "drip",
            "fraction_sprinkler_irrigation": "sprinkler",
            "fraction_surface_irrigation": "surface",
        }

        def prepare_region_wide(
            long_all_df: pd.DataFrame,
            region: str,
        ) -> pd.DataFrame:
            """Prepare a wide compositional table for a single region.

            The returned table contains one row per year with columns for the
            irrigation type fractions. Fractions are renormalised per year so that
            they sum exactly to one.

            Args:
                long_all_df: Long-format irrigation fraction table for all regions.
                region: Name of the region to extract.

            Returns:
                A wide-format table with columns ``"year"``, ``"drip"``,
                ``"sprinkler"``, and ``"surface"`` containing yearly fractions.
            """
            sub = long_all_df[long_all_df["Region"] == region].copy()
            sub["Var"] = sub["Description"].map(desc_map)
            wide = sub.pivot(
                index="year",
                columns="Var",
                values="fraction",
            ).sort_index()

            sum_frac = wide[["drip", "sprinkler", "surface"]].sum(axis=1)
            wide[["drip", "sprinkler", "surface"]] = wide[
                ["drip", "sprinkler", "surface"]
            ].div(sum_frac, axis=0)

            wide = wide.reset_index()
            return wide

        # =========================
        # 3. LINEAR MODEL
        # =========================

        def fit_linear_model(
            years: Sequence[float],
            y: Sequence[float],
        ) -> dict[str, Any]:
            """Fit a linear trend model for irrigation fractions.

            Notes:
                The model is of the form f(year) = a * year + b and is fitted via
                least squares. Model quality is summarised by the coefficient of
                determination.

            Args:
                years: Sequence of years at which observations are available.
                y: Observed fractions corresponding to ``years``.

            Returns:
                A dictionary containing the model parameters, the fitted type
                label, and the coefficient of determination.
            """
            years_arr = np.asarray(years, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            coeffs = np.polyfit(years_arr, y_arr, 1)
            yhat = np.polyval(coeffs, years_arr)
            ss_res = np.sum((y_arr - yhat) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
            return {"type": "linear", "coeffs": coeffs, "r2": r2}

        def predict_linear_model(
            years: Sequence[float],
            model: dict[str, Any],
        ) -> np.ndarray:
            """Predict irrigation fractions using a linear trend model.

            Args:
                years: Years at which to compute model predictions.
                model: Fitted linear model dictionary.

            Returns:
                Array of predicted fractions for each year in ``years``.
            """
            years_arr = np.asarray(years, dtype=float)
            return np.polyval(model["coeffs"], years_arr)

        # =========================
        # 4. LINEAR + SINUSOID MODEL
        # =========================

        def fit_linear_sine_model(
            years: Sequence[float],
            y: Sequence[float],
            period: float,
        ) -> dict[str, Any]:
            """Fit a linear-plus-sinusoidal model for irrigation fractions.

            The model combines a linear trend and a sinusoidal component with a
            prescribed period, fitted via least squares.

            Args:
                years: Years at which observations are available.
                y: Observed fractions corresponding to ``years``.
                period: Period of the sinusoidal component in years.

            Returns:
                A dictionary with fitted parameters, the reference time shift, the
                angular frequency, and the coefficient of determination.
            """
            years_arr = np.asarray(years, dtype=float)
            y_arr = np.asarray(y, dtype=float)

            t0 = years_arr.mean()
            t = years_arr - t0
            omega = 2.0 * np.pi / period

            X = np.column_stack(
                [
                    np.ones_like(t),
                    t,
                    np.sin(omega * t),
                ]
            )

            params, *_ = np.linalg.lstsq(X, y_arr, rcond=None)

            yhat = X @ params
            ss_res = np.sum((y_arr - yhat) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

            return {
                "type": "linear_sine",
                "params": params,
                "t0": t0,
                "omega": omega,
                "r2": r2,
            }

        def predict_linear_sine_model(
            years: Sequence[float],
            model: dict[str, Any],
        ) -> np.ndarray:
            """Predict irrigation fractions using a linear-plus-sinusoidal model.

            Args:
                years: Years at which to compute model predictions.
                model: Fitted linear-plus-sinusoidal model dictionary.

            Returns:
                Array of predicted fractions for each year in ``years``.
            """
            years_arr = np.asarray(years, dtype=float)
            t0 = model["t0"]
            omega = model["omega"]
            a, b, c = model["params"]

            t = years_arr - t0
            return a + b * t + c * np.sin(omega * t)

        # =========================
        # 4b. LOGISTIC (S-SHAPED) MODEL
        # =========================

        def fit_logistic_model(
            years: Sequence[float],
            y: Sequence[float],
        ) -> dict[str, Any]:
            """Fit a logistic curve model to irrigation fractions.

            The model uses a logit transform of the fractions and fits a linear
            function in the transformed space. Predictions are mapped back via the
            logistic function.

            Notes:
                Fractions are clipped slightly away from 0 and 1 before applying
                the logit transform to avoid numerical issues.

            Args:
                years: Years at which observations are available.
                y: Observed fractions corresponding to ``years``.

            Returns:
                A dictionary containing the fitted parameters, the model type, and
                the coefficient of determination.
            """
            years_arr = np.asarray(years, dtype=float)
            y_arr = np.asarray(y, dtype=float)

            eps = 1e-6
            y_clip = np.clip(y_arr, eps, 1 - eps)
            logit = np.log(y_clip / (1 - y_clip))

            coeffs = np.polyfit(years_arr, logit, 1)
            logit_hat = np.polyval(coeffs, years_arr)
            yhat = 1.0 / (1.0 + np.exp(-logit_hat))

            ss_res = np.sum((y_arr - yhat) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

            return {"type": "logistic", "coeffs": coeffs, "r2": r2}

        def predict_logistic_model(
            years: Sequence[float],
            model: dict[str, Any],
        ) -> np.ndarray:
            """Predict irrigation fractions using a logistic curve model.

            Args:
                years: Years at which to compute model predictions.
                model: Fitted logistic model dictionary.

            Returns:
                Array of predicted fractions for each year in ``years``.
            """
            years_arr = np.asarray(years, dtype=float)
            logit_hat = np.polyval(model["coeffs"], years_arr)
            return 1.0 / (1.0 + np.exp(-logit_hat))

        # =========================
        # 5. FIT MODELS FOR NSW & VICT
        # =========================
        period_years_NSW = 7
        period_years_VICT = 8

        fit_linear: dict[str, dict[str, Any]] = {}
        fit_sine: dict[str, dict[str, Any]] = {}
        fit_logistic: dict[str, dict[str, Any]] = {}

        for region in regions_to_fit:
            wide = prepare_region_wide(long_all, region)

            fit_linear[region] = {"data": wide, "models": {}}
            fit_sine[region] = {"data": wide, "models": {}}
            fit_logistic[region] = {"data": wide, "models": {}}

            for var in ["drip", "sprinkler", "surface"]:
                # linear
                m_lin = fit_linear_model(wide["year"].values, wide[var].values)
                fit_linear[region]["models"][var] = m_lin

                # linear + sine
                if region == "VICT":
                    m_sin = fit_linear_sine_model(
                        wide["year"].values,
                        wide[var].values,
                        period_years_VICT,
                    )
                else:
                    m_sin = fit_linear_sine_model(
                        wide["year"].values,
                        wide[var].values,
                        period_years_NSW,
                    )
                fit_sine[region]["models"][var] = m_sin

                # logistic
                m_log = fit_logistic_model(wide["year"].values, wide[var].values)
                fit_logistic[region]["models"][var] = m_log

        # Optional: print R² to see how they behave
        for region in regions_to_fit:
            print(f"=== {region} – LINEAR ===")
            for var, m in fit_linear[region]["models"].items():
                print(f"  {var}: R² = {m['r2']:.4f}")
            print(f"=== {region} – LINEAR+SINE ===")
            for var, m in fit_sine[region]["models"].items():
                print(f"  {var}: R² = {m['r2']:.4f}")
            print(f"=== {region} – LOGISTIC ===")
            for var, m in fit_logistic[region]["models"].items():
                print(f"  {var}: R² = {m['r2']:.4f}")

        # =========================
        # 6. BUILD EXPANDED PREDICTIONS WITH COMPOSITION CONSTRAINTS
        # =========================

        years_expanded = np.arange(start_year, end_year + 1)

        def predict_region_composition(
            years: Sequence[float],
            region: str,
            mode: str = "linear",
        ) -> pd.DataFrame:
            """Predict regional irrigation type composition over time.

            Select and evaluate one of the fitted models for a given region, then
            enforce compositional constraints on the resulting fractions.

            Notes:
                The predicted fractions are clipped between zero and one and then
                rescaled so that drip, sprinkler, and surface irrigation fractions
                sum to one for each year.

            Args:
                years: Years for which to generate predictions.
                region: Name of the region to use.
                mode: Model type to use. Supported values are "linear",
                    "linear_sine", and "logistic".

            Returns:
                A table with columns for the region name, year, predicted
                fractions for each irrigation type, and the chosen model mode.

            Raises:
                ValueError: If ``mode`` is not one of the supported values.
            """
            years_arr = np.asarray(years, dtype=float)

            if mode == "linear":
                reg_fit = fit_linear[region]
                f_drip = predict_linear_model(years_arr, reg_fit["models"]["drip"])
                f_spr = predict_linear_model(years_arr, reg_fit["models"]["sprinkler"])
                f_surf = predict_linear_model(years_arr, reg_fit["models"]["surface"])

            elif mode == "linear_sine":
                reg_fit = fit_sine[region]
                f_drip = predict_linear_sine_model(
                    years_arr,
                    reg_fit["models"]["drip"],
                )
                f_spr = predict_linear_sine_model(
                    years_arr,
                    reg_fit["models"]["sprinkler"],
                )
                f_surf = predict_linear_sine_model(
                    years_arr,
                    reg_fit["models"]["surface"],
                )

            elif mode == "logistic":
                reg_fit = fit_logistic[region]
                f_drip = predict_logistic_model(years_arr, reg_fit["models"]["drip"])
                f_spr = predict_logistic_model(
                    years_arr,
                    reg_fit["models"]["sprinkler"],
                )
                f_surf = predict_logistic_model(years_arr, reg_fit["models"]["surface"])

            else:
                raise ValueError("mode must be 'linear', 'linear_sine', or 'logistic'")

            comp = np.vstack([f_drip, f_spr, f_surf]).T

            comp = np.clip(comp, 0.0, 1.0)
            row_sums = comp.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            comp = comp / row_sums

            drip, spr, surf = comp.T

            return pd.DataFrame(
                {
                    "Region": region,
                    "year": years_arr,
                    "drip": drip,
                    "sprinkler": spr,
                    "surface": surf,
                    "mode": mode,
                }
            )

        # Build expanded predictions for all three modes
        expanded_linear_list: list[pd.DataFrame] = []
        expanded_sine_list: list[pd.DataFrame] = []
        expanded_logistic_list: list[pd.DataFrame] = []

        for region in regions_to_fit:
            expanded_linear_list.append(
                predict_region_composition(years_expanded, region, mode="linear")
            )
            expanded_sine_list.append(
                predict_region_composition(years_expanded, region, mode="linear_sine")
            )
            expanded_logistic_list.append(
                predict_region_composition(years_expanded, region, mode="logistic")
            )

        expanded_linear_df = pd.concat(expanded_linear_list, ignore_index=True)
        expanded_sine_df = pd.concat(expanded_sine_list, ignore_index=True)
        expanded_logistic_df = pd.concat(expanded_logistic_list, ignore_index=True)

        # =========================
        # 7. PLOTS: ORIGINAL + EXPANDED, FOR ALL MODES
        # =========================

        for region in regions_to_fit:
            wide = fit_linear[region]["data"]

            # ---- Linear ----
            preds_lin = expanded_linear_df[expanded_linear_df["Region"] == region]

            fig, ax = plt.subplots()
            ax.plot(wide["year"], wide["drip"], "o", label="drip (obs)")
            ax.plot(wide["year"], wide["sprinkler"], "o", label="sprinkler (obs)")
            ax.plot(wide["year"], wide["surface"], "o", label="surface (obs)")

            ax.plot(preds_lin["year"], preds_lin["drip"], "-", label="drip (linear)")
            ax.plot(
                preds_lin["year"],
                preds_lin["sprinkler"],
                "-",
                label="sprinkler (linear)",
            )
            ax.plot(
                preds_lin["year"],
                preds_lin["surface"],
                "-",
                label="surface (linear)",
            )

            ax.set_title(f"Irrigation fractions – linear fit & expanded – {region}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Fraction")
            ax.legend()
            ax.grid(True)

            fig.savefig(
                output_dir / f"irrigation_fit_expanded_linear_{region}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

            # ---- Linear + Sine ----
            preds_sin = expanded_sine_df[expanded_sine_df["Region"] == region]

            fig, ax = plt.subplots()
            ax.plot(wide["year"], wide["drip"], "o", label="drip (obs)")
            ax.plot(wide["year"], wide["sprinkler"], "o", label="sprinkler (obs)")
            ax.plot(wide["year"], wide["surface"], "o", label="surface (obs)")

            ax.plot(
                preds_sin["year"],
                preds_sin["drip"],
                "-",
                label="drip (lin+sine)",
            )
            ax.plot(
                preds_sin["year"],
                preds_sin["sprinkler"],
                "-",
                label="sprinkler (lin+sine)",
            )
            ax.plot(
                preds_sin["year"],
                preds_sin["surface"],
                "-",
                label="surface (lin+sine)",
            )

            ax.set_title(
                f"Irrigation fractions – linear+sinusoidal fit & expanded – {region}"
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("Fraction")
            ax.legend()
            ax.grid(True)

            fig.savefig(
                output_dir / f"irrigation_fit_expanded_linear_sine_{region}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

            # ---- Logistic (S-shaped) ----
            preds_log = expanded_logistic_df[expanded_logistic_df["Region"] == region]

            fig, ax = plt.subplots()
            ax.plot(wide["year"], wide["drip"], "o", label="drip (obs)")
            ax.plot(wide["year"], wide["sprinkler"], "o", label="sprinkler (obs)")
            ax.plot(wide["year"], wide["surface"], "o", label="surface (obs)")

            ax.plot(
                preds_log["year"],
                preds_log["drip"],
                "-",
                label="drip (logistic)",
            )
            ax.plot(
                preds_log["year"],
                preds_log["sprinkler"],
                "-",
                label="sprinkler (logistic)",
            )
            ax.plot(
                preds_log["year"],
                preds_log["surface"],
                "-",
                label="surface (logistic)",
            )

            ax.set_title(
                f"Irrigation fractions – logistic (S-curve) fit & expanded – {region}"
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("Fraction")
            ax.legend()
            ax.grid(True)

            fig.savefig(
                output_dir / f"irrigation_fit_expanded_logistic_{region}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        print(
            "Saved combined original + expanded plots for NSW and VICT, for linear, "
            "linear+sinusoidal and logistic models."
        )

        # =========================
        # 8. STORE RESULTS AS DICTIONARIES (VICT=0, NSW=1)
        # =========================

        region_ids = {"VICT": 0, "NSW": 1}
        years = list(range(start_year, end_year + 1))

        irrigation_frac_map = {
            "fraction_drip_or_trickle_irrigation": "drip",
            "fraction_sprinkler_irrigation": "sprinkler",
            "fraction_surface_irrigation": "surface",
        }

        def build_and_store_dicts(expanded_df: pd.DataFrame) -> None:
            """Build and store irrigation fraction dictionaries for the model.

            Notes:
                For each irrigation type and region, the function reindexes onto
                the model year range, fills gaps by interpolation and edge filling,
                and stores the resulting series under the appropriate key in the
                model dictionary.

            Args:
                expanded_df: Expanded prediction table containing regional
                    irrigation fractions by year.

            """
            expanded_df = expanded_df[
                (expanded_df["year"] >= start_year) & (expanded_df["year"] <= end_year)
            ].copy()

            for dict_name, col in irrigation_frac_map.items():
                irrig_dict: dict[str, Any] = {"time": years, "data": {}}

                for region_name, region_id in region_ids.items():
                    sub = (
                        expanded_df[expanded_df["Region"] == region_name]
                        .set_index("year")
                        .sort_index()
                    )
                    s = sub[col].reindex(years)
                    s = s.interpolate().bfill().ffill()
                    irrig_dict["data"][region_id] = s.tolist()

                key_name = f"irrigation/{col}"
                self.set_dict(irrig_dict, name=key_name)

        # Choose which trajectories to use for the model dictionaries:
        # build_and_store_dicts(expanded_linear_df)      # linear
        # build_and_store_dicts(expanded_sine_df)        # linear+sine
        build_and_store_dicts(expanded_logistic_df)  # logistic (S-shaped)

    @build_method(depends_on=["setup_economic_data"])
    def setup_water_prices_australia(
        self,
        start_year: int,
        end_year: int,
    ) -> None:
        """Set up Australian water price and diversion time series.

        Construct monthly water price and annual diversion series for Australian
        regions based on observed datasets and inflation adjustments. The resulting
        time series are stored in the internal socio-economic dictionaries.

        Notes:
            Water prices are converted from local currency units to USD using
            pre-loaded conversion rates and then deflated using region-specific
            inflation rates. Diversions are kept at annual resolution.

        Args:
            start_year: First simulation year (inclusive) for which data is
                generated.
            end_year: Last simulation year (inclusive) for which data is
                generated.

        """

        def get_observed_diversion_price() -> tuple[pd.DataFrame, pd.DataFrame]:
            """Load and preprocess observed diversion and water price data.

            Read MDB catchment and market outlook datasets, filter regions of
            interest, convert units and frequencies, and return aligned monthly
            water prices and annual diversions for the relevant states.

            Notes:
                Water prices are interpolated to monthly resolution, while
                diversions remain at annual resolution with a water year ending in
                June. All prices are converted to USD per cubic metre.

            Args:
                None.

            Returns:
                A tuple containing the monthly observed water prices and annual
                observed diversions, both indexed by time and with columns for
                aggregated regions.
            """
            full_southern_mdb = "southern" in os.getcwd()
            # Regions of interest
            if full_southern_mdb:
                regions_of_interest = [
                    "VIC Goulburn-Broken",
                    "VIC Murray Above",
                    "VIC Murray Below",
                    "VIC Loddon-Campaspe",
                    "NSW Murray Below",
                    "NSW Murray Above",
                    "NSW Murrumbidgee",
                    "NSW Lachlan",
                ]
            else:
                regions_of_interest = [
                    "VIC Goulburn-Broken",
                    "VIC Murray Above",
                    "VIC Murray Below",
                    "VIC Loddon-Campaspe",
                    "NSW Murray Below",
                    "NSW Murray Above",
                    "NSW Murrumbidgee",
                ]

            ################ Water price #######################
            # Observed data path
            water_price_observed_fp = Path(
                "calibration_data/water_use/"
                "WaterMarketOutlook_2023-04_data_tables_v1.0.0.xlsx"
            )
            water_price_observed_df = pd.read_excel(
                water_price_observed_fp,
                sheet_name=3,
            )

            # Filter relevant regions, rename columns
            filtered_prices_df = water_price_observed_df[
                water_price_observed_df["Region"].isin(regions_of_interest)
            ].copy()
            filtered_prices_df = filtered_prices_df[
                ["Date", "Region", "Monthly average price ($/ML)"]
            ]
            filtered_prices_df.columns = ["time", "Region", "water_price_observed"]
            filtered_prices_df["time"] = pd.to_datetime(
                filtered_prices_df["time"]
            ).dt.normalize()

            pivot_df = filtered_prices_df.pivot(
                index="time",
                columns="Region",
                values="water_price_observed",
            ).sort_index()
            pivot_df = pivot_df.apply(pd.to_numeric, errors="coerce").interpolate(
                method="time"
            )
            pivot_df["Victoria"] = pivot_df[
                [
                    "VIC Goulburn-Broken",
                    "VIC Murray Above",
                    "VIC Murray Below",
                    "VIC Loddon-Campaspe",
                ]
            ].mean(axis=1)

            pivot_df["New South Wales"] = pivot_df[
                [
                    "NSW Murray Below",
                    "NSW Murray Above",
                    "NSW Murrumbidgee",
                ]
            ].mean(axis=1)

            if full_southern_mdb:
                pivot_df["Australian Capital Territory"] = pivot_df[
                    [
                        "NSW Murray Below",
                        "NSW Murray Above",
                        "NSW Murrumbidgee",
                    ]
                ].mean(axis=1)

            # Convert from AUD to USD if needed
            conversion_rates = self.dict["socioeconomics/LCU_per_USD"]
            yearly_rates = dict(
                zip(
                    map(int, conversion_rates["time"]),
                    map(float, conversion_rates["data"]["0"]),
                )
            )
            pivot_df["year"] = pivot_df.index.year
            pivot_df["conversion_rate"] = pivot_df["year"].map(yearly_rates)

            pivot_df = pivot_df.div(pivot_df["conversion_rate"], axis=0)

            df_observed_price = pivot_df.drop(columns=["year", "conversion_rate"])
            df_observed_price.index.name = "time"

            # Load the new dataset to supplement prices earlier than 2004
            diversions_price_observed_fp = Path(
                "calibration_data/water_use/"
                "MDBWaterMarketCatchmentDataset_Supply_v1.0.0.xlsx"
            )
            diversions_price_observed_df = pd.read_excel(
                diversions_price_observed_fp,
                sheet_name=1,
            )

            # Filter to only the regions of interest
            filtered_diversions_price_df = diversions_price_observed_df[
                diversions_price_observed_df["Region"].isin(regions_of_interest)
            ].copy()

            filtered_water_price_df = filtered_diversions_price_df[
                ["Year", "Region", "P"]
            ].rename(
                columns={
                    "Year": "time",
                    "P": "water_price_observed",
                }
            )
            # Convert 'time' to datetime
            filtered_water_price_df["time"] = pd.to_datetime(
                filtered_water_price_df["time"].astype(str) + "-06-30"
            )

            pivot_water_price_df = filtered_water_price_df.pivot(
                index="time",
                columns="Region",
                values="water_price_observed",
            ).sort_index()

            pivot_water_price_df = pivot_water_price_df.apply(
                pd.to_numeric,
                errors="coerce",
            ).interpolate("time")

            pivot_water_price_df["Victoria"] = pivot_water_price_df[
                [
                    "VIC Goulburn-Broken",
                    "VIC Murray Above",
                    "VIC Murray Below",
                    "VIC Loddon-Campaspe",
                ]
            ].mean(axis=1)

            pivot_water_price_df["New South Wales"] = pivot_water_price_df[
                [
                    "NSW Murray Below",
                    "NSW Murray Above",
                    "NSW Murrumbidgee",
                ]
            ].mean(axis=1)

            if full_southern_mdb:
                pivot_water_price_df["Australian Capital Territory"] = (
                    pivot_water_price_df[
                        [
                            "NSW Murray Below",
                            "NSW Murray Above",
                            "NSW Murrumbidgee",
                        ]
                    ].mean(axis=1)
                )

            start_date = "2000-07-01"
            end_date = "2004-06-30"
            monthly_index = pd.date_range(start=start_date, end=end_date, freq="MS")

            def find_year_ended_june_for_month(
                ts: pd.Timestamp,
            ) -> pd.Timestamp:
                """Map a calendar month to the corresponding June-30 water year date.

                Args:
                    ts: Calendar month timestamp.

                Returns:
                    Timestamp representing the June 30 date of the water year.
                """
                if ts.month < 7:
                    return pd.Timestamp(year=ts.year, month=6, day=30)
                return pd.Timestamp(year=ts.year + 1, month=6, day=30)

            df_list: list[pd.DataFrame] = []
            for m in monthly_index:
                match_date = find_year_ended_june_for_month(m)
                if match_date in pivot_water_price_df.index:
                    row_vals = pivot_water_price_df.loc[match_date]
                    df_list.append(pd.DataFrame(row_vals).T.assign(time=m))
                else:
                    df_list.append(
                        pd.DataFrame(
                            np.nan,
                            index=[0],
                            columns=pivot_water_price_df.columns,
                        ).assign(time=m)
                    )
            df_annual_to_monthly = pd.concat(df_list, ignore_index=True).set_index(
                "time"
            )

            df_observed_post_aug_2004 = df_observed_price[
                df_observed_price.index >= "2004-08-01"
            ]
            df_combined = pd.concat([df_annual_to_monthly, df_observed_post_aug_2004])
            full_monthly_index = pd.date_range(
                start=df_combined.index.min(),
                end=df_observed_price.index.max(),
                freq="MS",
            )
            df_combined = df_combined.reindex(full_monthly_index)
            df_observed_price = df_combined.interpolate(method="time")
            df_observed_price_usd_m3 = df_observed_price / 1000  # from ML to m3

            ################ Diversions #######################

            # Keep only relevant columns
            filtered_diversions_df = filtered_diversions_price_df[
                ["Year", "Region", "U"]
            ].rename(columns={"Year": "time", "U": "diversion_observed"})

            # Convert 'time' to datetime
            filtered_diversions_df["time"] = pd.to_datetime(
                filtered_diversions_df["time"].astype(str) + "-06-30"
            )

            pivot_diversions_df = filtered_diversions_df.pivot(
                index="time",
                columns="Region",
                values="diversion_observed",
            ).sort_index()

            pivot_diversions_df = (
                pivot_diversions_df.apply(pd.to_numeric, errors="coerce")
                .replace(0, np.nan)
                .interpolate("time")
            )

            pivot_diversions_df["Victoria"] = pivot_diversions_df[
                [
                    "VIC Goulburn-Broken",
                    "VIC Murray Above",
                    "VIC Murray Below",
                    "VIC Loddon-Campaspe",
                ]
            ].sum(axis=1)
            if full_southern_mdb:
                pivot_diversions_df["New South Wales"] = pivot_diversions_df[
                    [
                        "NSW Murray Below",
                        "NSW Murray Above",
                        "NSW Murrumbidgee",
                        "NSW Lachlan",
                    ]
                ].sum(axis=1)
                pivot_diversions_df["Australian Capital Territory"] = (
                    pivot_diversions_df[["NSW Lachlan"]]
                )
            else:
                pivot_diversions_df["New South Wales"] = pivot_diversions_df[
                    [
                        "NSW Murray Below",
                        "NSW Murray Above",
                    ]
                ].sum(axis=1)

                pivot_diversions_df["New South Wales"] + (
                    pivot_diversions_df["NSW Murrumbidgee"] * 0.5
                )

            pivot_diversions_df_m3 = pivot_diversions_df * 1000  # from ML to m3

            return df_observed_price_usd_m3, pivot_diversions_df_m3

        df_observed_price, df_observed_diversion = get_observed_diversion_price()

        inflation_rates = self.dict["socioeconomics/inflation_rates"]

        dictionary_types: dict[str, pd.DataFrame] = {
            "diversions": df_observed_diversion,
            "water_price": df_observed_price,
        }

        for dictionary_type, data in dictionary_types.items():
            # If we're dealing with water_price, create monthly time steps
            if dictionary_type == "water_price":
                monthly_dates = pd.date_range(
                    start=pd.Timestamp(start_year, 1, 1),
                    end=pd.Timestamp(end_year, 12, 31),
                    freq="MS",
                )
                dictionary: dict[str, list[str] | dict[str, list[float]]] = {
                    "time": [d.strftime("%Y-%m-%d") for d in monthly_dates],
                    "data": {},
                }
            else:
                # For diversions, keep the original annual time steps
                dictionary = {
                    "time": list(range(start_year, end_year + 1)),
                    "data": {},
                }

            for _, region_row in self.geom["regions"].iterrows():
                region_id = str(region_row["region_id"])
                region_state = region_row["NAME_1"]

                data_region = data[region_state]

                if dictionary_type == "diversions":
                    # Keep the existing (annual) logic for diversions exactly as before

                    min_obs_year = data_region.index.year.min()
                    max_obs_year = data_region.index.year.max()
                    if max_obs_year > np.int32(inflation_rates["time"][-1]):
                        max_obs_year = np.int32(inflation_rates["time"][-1])

                    real_values: list[float] = []
                    for year in range(min_obs_year, max_obs_year + 1):
                        nominal = data_region.loc[data_region.index.year == year]
                        if len(nominal) == 0:
                            continue
                        factor = 1.0
                        for y in range(min_obs_year + 1, year + 1):
                            inf = inflation_rates["data"][region_id][
                                inflation_rates["time"].index(str(y))
                            ]
                            factor *= inf
                        real_values.append(float(nominal.iloc[0] / factor))

                    baseline = real_values[0]
                    prices = pd.Series(
                        index=range(start_year, end_year + 1),
                        dtype=float,
                    )
                    prices.loc[min_obs_year] = baseline

                    for y in range(start_year, end_year + 1):
                        if y in data_region.index.year:
                            prices.loc[y] = data_region.loc[
                                data_region.index.year == y
                            ].iloc[0]
                        elif y < min_obs_year or y > max_obs_year:
                            prices.loc[y] = baseline
                        else:
                            if pd.isna(prices.loc[y]):
                                prices.loc[y] = baseline

                    dictionary["data"][region_id] = prices.tolist()

                else:
                    # WATER PRICE: handle monthly data
                    # Ensure we have a sorted DatetimeIndex for monthly data
                    data_region = data_region.sort_index()
                    min_obs_date = data_region.index.min()
                    max_obs_date = data_region.index.max()

                    # Compute a baseline by deflating observed monthly values back to min_obs_date.year
                    real_values: list[float] = []
                    if pd.notna(min_obs_date) and pd.notna(max_obs_date):
                        min_yr = min_obs_date.year
                        for dt_obs in data_region.index:
                            val = data_region.loc[dt_obs]
                            factor = 1.0
                            for y in range(min_yr + 1, dt_obs.year + 1):
                                y_str = str(y)
                                if y_str in inflation_rates["time"]:
                                    idx = inflation_rates["time"].index(y_str)
                                    factor *= inflation_rates["data"][region_id][idx]
                            real_values.append(float(val / factor))

                    if len(real_values) > 0:
                        baseline = real_values[0]
                    else:
                        baseline = 0.0

                    prices = pd.Series(index=monthly_dates, dtype=float)

                    # Fill any known monthly data directly
                    for dt_obs in data_region.index:
                        if dt_obs in prices.index:
                            prices.loc[dt_obs] = data_region.loc[dt_obs]

                    # Helper to apply annual inflation only if we crossed a year boundary
                    def inflation_factor(
                        prev_date: pd.Timestamp,
                        curr_date: pd.Timestamp,
                        region: str,
                    ) -> float:
                        """Compute the annual inflation factor between two dates.

                        Args:
                            prev_date: Previous time step in the monthly series.
                            curr_date: Current time step in the monthly series.
                            region: Region identifier used to look up inflation
                                rates.

                        Returns:
                            Multiplicative inflation factor to apply when moving
                            from ``prev_date`` to ``curr_date``.
                        """
                        if curr_date.year != prev_date.year:
                            y_str = str(curr_date.year)
                            if y_str in inflation_rates["time"]:
                                idx = inflation_rates["time"].index(y_str)
                                return float(inflation_rates["data"][region][idx])
                        return 1.0

                    all_dates = prices.index

                    # If we have at least one observed monthly date, forward/backward fill
                    if pd.notna(min_obs_date):
                        # Set the earliest known date to baseline if not set
                        prices.loc[min_obs_date] = baseline

                        # Go forward
                        start_idx = all_dates.get_loc(min_obs_date)
                        for i in range(start_idx + 1, len(all_dates)):
                            prev_date = all_dates[i - 1]
                            curr_date = all_dates[i]
                            if pd.isna(prices.loc[curr_date]):
                                f = inflation_factor(prev_date, curr_date, region_id)
                                prices.loc[curr_date] = prices.loc[prev_date] * f

                        # Go backward
                        for i in range(start_idx - 1, -1, -1):
                            next_date = all_dates[i + 1]
                            curr_date = all_dates[i]
                            if pd.isna(prices.loc[curr_date]):
                                f = inflation_factor(curr_date, next_date, region_id)
                                if f == 0:
                                    f = 1.0
                                prices.loc[curr_date] = prices.loc[next_date] / f
                    else:
                        # No observed data at all
                        prices[:] = baseline

                    dictionary["data"][region_id] = prices.tolist()

            self.set_dict(dictionary, name=f"socioeconomics/{dictionary_type}")

    @build_method(depends_on=["setup_economic_data"])
    def setup_drip_irrigation_prices_by_reference_year(
        self,
        drip_irrigation_price: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ) -> None:
        """Create region-specific drip-irrigation price time series via inflation.

        Construct an annual drip-irrigation price series for each region based on
        a single reference-year price and region-specific inflation rates, and
        store the result in the model dictionary.

        Notes:
            Prices in years after the reference year are obtained by inflating
            forward using the annual inflation rates; prices in earlier years are
            obtained by deflating backward. If inflation data is missing for a
            given year, the last available price is carried forward or backward.

        Args:
            drip_irrigation_price: Baseline drip-irrigation price in the
                reference year, in the relevant currency per unit area.
            reference_year: Calendar year to which ``drip_irrigation_price``
                applies.
            start_year: First year of the model period (inclusive).
            end_year: Last year of the model period (inclusive).

        """
        self.logger.info("Setting up drip irrigation prices by reference year")

        inflation = self.new_data_catalog.fetch("wb_inflation_rate").read()
        regions = list(inflation["data"].keys())
        infl_years: list[str] = [str(y) for y in inflation["time"]]

        price_name = "drip_irrigation_price"
        out: dict[str, Any] = {
            "time": list(range(start_year, end_year + 1)),
            "data": {},
        }

        for region in regions:
            series = pd.Series(index=out["time"], dtype=float)
            series.loc[reference_year] = float(drip_irrigation_price)

            # forward
            for y in range(reference_year + 1, end_year + 1):
                y_str = str(y)
                if y_str in infl_years:
                    idx = infl_years.index(y_str)
                    series.loc[y] = series.loc[y - 1] * float(
                        inflation["data"][region][idx]
                    )
                else:
                    series.loc[y] = series.loc[y - 1]

            # backward
            for y in range(reference_year - 1, start_year - 1, -1):
                y_plus_1 = str(y + 1)
                if y_plus_1 in infl_years:
                    idx = infl_years.index(y_plus_1)
                    factor = float(inflation["data"][region][idx])
                    series.loc[y] = series.loc[y + 1] / (factor if factor != 0 else 1.0)
                else:
                    series.loc[y] = series.loc[y + 1]

            out["data"][region] = series.tolist()

        self.set_dict(out, name=f"socioeconomics/{price_name}")
