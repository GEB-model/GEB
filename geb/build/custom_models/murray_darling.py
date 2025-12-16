"""Utilities to set up Australian water prices and drip irrigation prices."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geb.build.methods import build_method

from .. import GEBModel


class Agents(GEBModel):
    """Contains copied build methods for the agents for GEB and additional functions for the murray-darling basin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Forward arguments to the base ``GEBModel`` initializer."""
        super().__init__(*args, **kwargs)

    @build_method(depends_on=["setup_economic_data"])
    def setup_irrigation_efficiency_australia(
        self,
        start_year: int,
        end_year: int,
    ) -> None:
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
        def parse_water_year(wy_str):
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
        groups = []
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

        # Range for expansion (arguments)
        start_year = start_year
        end_year = end_year

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

        def prepare_region_wide(long_all, region):
            """
            Return wide table for a region:
            columns: year, drip, sprinkler, surface
            Fractions are renormalised per year to sum exactly to 1.
            """
            sub = long_all[long_all["Region"] == region].copy()
            sub["Var"] = sub["Description"].map(desc_map)
            wide = sub.pivot(
                index="year", columns="Var", values="fraction"
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

        def fit_linear_model(years, y):
            years = np.asarray(years, dtype=float)
            y = np.asarray(y, dtype=float)
            coeffs = np.polyfit(years, y, 1)  # [b, a]
            yhat = np.polyval(coeffs, years)
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
            return {"type": "linear", "coeffs": coeffs, "r2": r2}

        def predict_linear_model(years, model):
            years = np.asarray(years, dtype=float)
            return np.polyval(model["coeffs"], years)

        # =========================
        # 4. LINEAR + SINUSOID MODEL
        # =========================

        def fit_linear_sine_model(years, y, period):
            years = np.asarray(years, dtype=float)
            y = np.asarray(y, dtype=float)

            t0 = years.mean()
            t = years - t0
            omega = 2.0 * np.pi / period

            X = np.column_stack(
                [
                    np.ones_like(t),
                    t,
                    np.sin(omega * t),
                ]
            )

            params, *_ = np.linalg.lstsq(X, y, rcond=None)  # [a, b, c]

            yhat = X @ params
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

            return {
                "type": "linear_sine",
                "params": params,
                "t0": t0,
                "omega": omega,
                "r2": r2,
            }

        def predict_linear_sine_model(years, model):
            years = np.asarray(years, dtype=float)
            t0 = model["t0"]
            omega = model["omega"]
            a, b, c = model["params"]

            t = years - t0
            return a + b * t + c * np.sin(omega * t)

        # =========================
        # 4b. LOGISTIC (S-SHAPED) MODEL
        # =========================

        def fit_logistic_model(years, y):
            """
            Logistic S-curve via logit transform:
                logit(y) = ln(y/(1-y)) = a + b * year
            with y clipped to (eps, 1-eps).
            """
            years = np.asarray(years, dtype=float)
            y = np.asarray(y, dtype=float)

            eps = 1e-6
            y_clip = np.clip(y, eps, 1 - eps)
            logit = np.log(y_clip / (1 - y_clip))

            coeffs = np.polyfit(years, logit, 1)  # [b, a]
            logit_hat = np.polyval(coeffs, years)
            yhat = 1.0 / (1.0 + np.exp(-logit_hat))

            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

            return {"type": "logistic", "coeffs": coeffs, "r2": r2}

        def predict_logistic_model(years, model):
            years = np.asarray(years, dtype=float)
            logit_hat = np.polyval(model["coeffs"], years)
            return 1.0 / (1.0 + np.exp(-logit_hat))

        # =========================
        # 5. FIT MODELS FOR NSW & VICT
        # =========================
        period_years_NSW = 7
        period_years_VICT = 8

        fit_linear = {}
        fit_sine = {}
        fit_logistic = {}

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
                        wide["year"].values, wide[var].values, period_years_VICT
                    )
                else:
                    m_sin = fit_linear_sine_model(
                        wide["year"].values, wide[var].values, period_years_NSW
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

        def predict_region_composition(years, region, mode="linear"):
            """
            mode: "linear", "linear_sine", or "logistic"
            Uses the corresponding fitted models, then enforces:
            - 0 <= fraction <= 1
            - drip + sprinkler + surface = 1
            """
            years = np.asarray(years, dtype=float)

            if mode == "linear":
                reg_fit = fit_linear[region]
                f_drip = predict_linear_model(years, reg_fit["models"]["drip"])
                f_spr = predict_linear_model(years, reg_fit["models"]["sprinkler"])
                f_surf = predict_linear_model(years, reg_fit["models"]["surface"])

            elif mode == "linear_sine":
                reg_fit = fit_sine[region]
                f_drip = predict_linear_sine_model(years, reg_fit["models"]["drip"])
                f_spr = predict_linear_sine_model(years, reg_fit["models"]["sprinkler"])
                f_surf = predict_linear_sine_model(years, reg_fit["models"]["surface"])

            elif mode == "logistic":
                reg_fit = fit_logistic[region]
                f_drip = predict_logistic_model(years, reg_fit["models"]["drip"])
                f_spr = predict_logistic_model(years, reg_fit["models"]["sprinkler"])
                f_surf = predict_logistic_model(years, reg_fit["models"]["surface"])

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
                    "year": years,
                    "drip": drip,
                    "sprinkler": spr,
                    "surface": surf,
                    "mode": mode,
                }
            )

        # Build expanded predictions for all three modes
        expanded_linear_list = []
        expanded_sine_list = []
        expanded_logistic_list = []

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
                preds_lin["year"], preds_lin["surface"], "-", label="surface (linear)"
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

            ax.plot(preds_sin["year"], preds_sin["drip"], "-", label="drip (lin+sine)")
            ax.plot(
                preds_sin["year"],
                preds_sin["sprinkler"],
                "-",
                label="sprinkler (lin+sine)",
            )
            ax.plot(
                preds_sin["year"], preds_sin["surface"], "-", label="surface (lin+sine)"
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

            ax.plot(preds_log["year"], preds_log["drip"], "-", label="drip (logistic)")
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
            "Saved combined original + expanded plots for NSW and VICT, for linear, linear+sinusoidal and logistic models."
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

        def build_and_store_dicts(expanded_df):
            expanded_df = expanded_df[
                (expanded_df["year"] >= start_year) & (expanded_df["year"] <= end_year)
            ].copy()

            for dict_name, col in irrigation_frac_map.items():
                irrig_dict = {"time": years, "data": {}}

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

        pass

    @build_method(depends_on=["setup_economic_data"])
    def setup_water_prices_australia(
        self,
        start_year: int,
        end_year: int,
    ) -> None:
        """Set up Australian water price and diversion time series.

        This method constructs monthly water price and annual diversion series for
        Australian regions based on observed datasets and inflation adjustments.
        The resulting time series are stored in the internal economics dictionaries.

        Notes:
            Water prices are converted from AUD to USD using pre-loaded conversion
            rates and then deflated using region-specific inflation rates.
            Diversions are kept at annual resolution.

        Args:
            start_year: First simulation year (inclusive) for which data is generated.
            end_year: Last simulation year (inclusive) for which data is generated.
        """

        def get_observed_diversion_price() -> tuple[pd.DataFrame, pd.DataFrame]:
            """Load and preprocess observed diversion and water price data.

            The function reads MDB and market outlook datasets, filters regions of
            interest, converts units and frequencies, and returns aligned monthly
            water prices and annual diversions for Victoria and New South Wales.

            Notes:
                Water prices are interpolated to monthly resolution, while diversions
                remain at annual resolution (water year ending in June).

            Returns:
                A tuple containing:
                    A DataFrame with monthly observed water prices by state.
                    A DataFrame with annual observed diversions by region or state.
            """
            full_southern_MDB = "southern" in os.getcwd()
            # Regions of interest
            if full_southern_MDB:
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
                "calibration_data/water_use/WaterMarketOutlook_2023-04_data_tables_v1.0.0.xlsx"
            )
            water_price_observed_df = pd.read_excel(
                water_price_observed_fp, sheet_name=3
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
                index="time", columns="Region", values="water_price_observed"
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

            if full_southern_MDB:
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

            # Load the new dataset so supplement prices earlier than 2004
            diversions_price__observed_fp = Path(
                "calibration_data/water_use/MDBWaterMarketCatchmentDataset_Supply_v1.0.0.xlsx"
            )
            diversions_price_observed_df = pd.read_excel(
                diversions_price__observed_fp, sheet_name=1
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
                index="time", columns="Region", values="water_price_observed"
            ).sort_index()

            # pivot_price_df = pivot_price_df.shift(1)
            pivot_water_price_df = pivot_water_price_df.apply(
                pd.to_numeric, errors="coerce"
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

            if full_southern_MDB:
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

            def find_year_ended_june_for_month(m: pd.Timestamp) -> pd.Timestamp:
                """Map a calendar month to the corresponding June-30 water year date.

                Args:
                    m: Calendar month timestamp.

                Returns:
                    Timestamp representing the June 30 date of the water year.
                """
                if m.month < 7:
                    return pd.Timestamp(year=m.year, month=6, day=30)
                return pd.Timestamp(year=m.year + 1, month=6, day=30)

            df_list: list[pd.DataFrame] = []
            for m in monthly_index:
                match_date = find_year_ended_june_for_month(m)
                if match_date in pivot_water_price_df.index:
                    row_vals = pivot_water_price_df.loc[match_date]
                    df_list.append(pd.DataFrame(row_vals).T.assign(time=m))
                else:
                    df_list.append(
                        pd.DataFrame(
                            np.nan, index=[0], columns=pivot_water_price_df.columns
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
            df_observed_price_USD_m3 = df_observed_price / 1000  # from ML to m3

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
                index="time", columns="Region", values="diversion_observed"
            ).sort_index()

            # pivot_price_df = pivot_price_df.shift(1)
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
            if full_southern_MDB:
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

            return df_observed_price_USD_m3, pivot_diversions_df_m3

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
                        index=range(start_year, end_year + 1), dtype=float
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
                            region: Region identifier used to look up inflation rates.

                        Returns:
                            Multiplicative inflation factor to apply when moving from
                            ``prev_date`` to ``curr_date``.
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
        self: Any,
        drip_irrigation_price: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ) -> None:
        """Create region-specific drip-irrigation price time series via annual inflation.

        Stores under ``economics/drip_irrigation_price`` with:
        - ``time``: list[int] of years
        - ``data``: dict[region_id] → list[float] (one per year)

        The reference year's price is given; subsequent years are inflated forward,
        previous years are deflated backward.
        """
        self.logger.info("Setting up drip irrigation prices by reference year")

        inflation = self.new_data_catalog.fetch("wb_inflation_rate").read()
        regions = list(inflation["data"].keys())
        infl_years: list[str] = [str(y) for y in inflation["time"]]

        price_name = "drip_irrigation_price"
        out = {"time": list(range(start_year, end_year + 1)), "data": {}}

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
