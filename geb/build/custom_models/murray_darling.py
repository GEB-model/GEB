"""Utilities to set up Australian water prices and drip irrigation prices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geb.build.modules.agents import Agents as BaseAgents


class Agents(BaseAgents):
    """Contains copied build methods for the agents for GEB and additional functions for the murray-darling basin."""

    def setup_water_prices_australia(self: Any, start_year: int, end_year: int) -> None:
        """Build and store observed monthly water prices (USD) and annual diversions.

        Creates two dictionaries under ``self.dict`` via ``self.set_dict``:

        - ``economics/water_price``: monthly USD prices per region (YYYY-MM-01 steps)
        - ``economics/diversions``: annual diversions per region (integers years)


        -----
        * Prices come from:
        - ABARES Water Market Outlook monthly table (sheet 4, zero-based 3)
        - MDB Catchment dataset (annual), mapped to monthly before 2004-08
        * AUD→USD conversion is applied using ``economics/lcu_per_usd_conversion_rates``.
        * Diversions are kept as annual values. NSW total includes Murrumbidgee
        at a 0.5 weight (as in your original intent).
        """
        inflation_rates = self.new_data_catalog.fetch("wb_inflation_rate").read()
        inflation_rates_country_index = inflation_rates.set_index("Country Code")
        price_ratio = self.new_data_catalog.fetch("world_bank_price_ratio").read()
        LCU_per_USD = self.new_data_catalog.fetch("world_bank_LCU_per_USD").read()

        def _get_observed_price_and_diversion() -> tuple[pd.DataFrame, pd.DataFrame]:
            regions_of_interest = [
                "VIC Goulburn-Broken",
                "VIC Murray Above",
                "VIC Murray Below",
                "VIC Loddon-Campaspe",
                "NSW Murray Below",
                "NSW Murray Above",
                "NSW Murrumbidgee",
            ]

            # ---------- Monthly water prices (ABARES) ----------
            price_xlsx = Path(
                "calibration_data/water_use/WaterMarketOutlook_2023-04_data_tables_v1.0.0.xlsx"
            )
            price_df = pd.read_excel(price_xlsx, sheet_name=3)

            price_df = price_df.loc[
                price_df["Region"].isin(regions_of_interest),
                ["Date", "Region", "Monthly average price ($/ML)"],
            ].rename(
                columns={"Date": "time", "Monthly average price ($/ML)": "price_aud"}
            )
            price_df["time"] = pd.to_datetime(price_df["time"]).dt.normalize()

            monthly_aud = (
                price_df.pivot(index="time", columns="Region", values="price_aud")
                .sort_index()
                .apply(pd.to_numeric, errors="coerce")
                .interpolate(method="time")
            )

            # State averages
            monthly_aud["Victoria"] = monthly_aud[
                [
                    "VIC Goulburn-Broken",
                    "VIC Murray Above",
                    "VIC Murray Below",
                    "VIC Loddon-Campaspe",
                ]
            ].mean(axis=1)
            monthly_aud["New South Wales"] = monthly_aud[
                ["NSW Murray Below", "NSW Murray Above", "NSW Murrumbidgee"]
            ].mean(axis=1)

            # Expect conv["time"] as a list of years (str/int) and conv["data"]["0"] as rates
            years_conv = [int(y) for y in LCU_per_USD["time"]]
            rates_conv = [float(r) for r in LCU_per_USD["data"]["0"]]
            year_to_rate: dict[int, float] = dict(zip(years_conv, rates_conv))

            monthly_aud["year"] = monthly_aud.index.year
            monthly_aud["conv"] = monthly_aud["year"].map(year_to_rate)
            monthly_usd = monthly_aud.div(monthly_aud["conv"], axis=0).drop(
                columns=["year", "conv"]
            )
            monthly_usd.index.name = "time"

            # ---------- Annual price & diversions (MDB catchment dataset) ----------
            mdb_xlsx = Path(
                "calibration_data/water_use/MDBWaterMarketCatchmentDataset_Supply_v1.0.0.xlsx"
            )
            mdb_df = pd.read_excel(mdb_xlsx, sheet_name=1)

            mdb_df = mdb_df.loc[mdb_df["Region"].isin(regions_of_interest)].copy()

            # Annual price (P)
            annual_price = mdb_df.loc[:, ["Year", "Region", "P"]].rename(
                columns={"Year": "time", "P": "price_aud"}
            )
            annual_price["time"] = pd.to_datetime(
                annual_price["time"].astype(str) + "-06-30"
            )

            annual_price_pivot = (
                annual_price.pivot(index="time", columns="Region", values="price_aud")
                .sort_index()
                .apply(pd.to_numeric, errors="coerce")
                .interpolate(method="time")
            )
            annual_price_pivot["Victoria"] = annual_price_pivot[
                [
                    "VIC Goulburn-Broken",
                    "VIC Murray Above",
                    "VIC Murray Below",
                    "VIC Loddon-Campaspe",
                ]
            ].mean(axis=1)
            annual_price_pivot["New South Wales"] = annual_price_pivot[
                ["NSW Murray Below", "NSW Murray Above", "NSW Murrumbidgee"]
            ].mean(axis=1)

            # Map annual (year ended June) → monthly (2000-07 to 2004-06)
            def _fy_end_for_month(m: pd.Timestamp) -> pd.Timestamp:
                return pd.Timestamp(
                    year=m.year if m.month <= 6 else m.year + 1, month=6, day=30
                )

            annual_monthly_index = pd.date_range("2000-07-01", "2004-06-30", freq="MS")
            rows = []
            for m in annual_monthly_index:
                fy = _fy_end_for_month(m)
                if fy in annual_price_pivot.index:
                    vals = annual_price_pivot.loc[fy]
                    rows.append(pd.DataFrame(vals).T.assign(time=m))
                else:
                    rows.append(
                        pd.DataFrame(
                            np.nan, index=[0], columns=annual_price_pivot.columns
                        ).assign(time=m)
                    )
            annual_as_monthly = pd.concat(rows, ignore_index=True).set_index("time")

            # Stitch: annual-as-monthly (pre 2004-08) + observed monthly thereafter
            observed_post = monthly_usd.loc[monthly_usd.index >= "2004-08-01"]
            stitched = pd.concat([annual_as_monthly, observed_post])

            full_monthly_index = pd.date_range(
                stitched.index.min(), monthly_usd.index.max(), freq="MS"
            )
            monthly_usd_final = stitched.reindex(full_monthly_index).interpolate(
                method="time"
            )
            monthly_usd_final.index.name = "time"

            # ---------- Diversions (U) annual ----------
            annual_div = mdb_df.loc[:, ["Year", "Region", "U"]].rename(
                columns={"Year": "time", "U": "diversion"}
            )
            annual_div["time"] = pd.to_datetime(
                annual_div["time"].astype(str) + "-06-30"
            )
            annual_div_pivot = (
                annual_div.pivot(index="time", columns="Region", values="diversion")
                .sortindex()
                .apply(pd.to_numeric, errors="coerce")
                .interpolate("time")
            )

            annual_div_pivot["Victoria"] = annual_div_pivot[
                [
                    "VIC Goulburn-Broken",
                    "VIC Murray Above",
                    "VIC Murray Below",
                    "VIC Loddon-Campaspe",
                ]
            ].sum(axis=1)

            # NSW total (explicitly include 0.5 * Murrumbidgee)
            annual_div_pivot["New South Wales"] = (
                annual_div_pivot[["NSW Murray Below", "NSW Murray Above"]].sum(axis=1)
                + 0.5 * annual_div_pivot["NSW Murrumbidgee"]
            )

            return monthly_usd_final, annual_div_pivot

        # Get observed series
        df_price_monthly_usd, df_diversions_annual = _get_observed_price_and_diversion()

        inflation_years: list[str] = [str(y) for y in inflation_rates["time"]]

        sources = {
            "diversions": df_diversions_annual,  # annual
            "water_price": df_price_monthly_usd,  # monthly
        }

        for key, df in sources.items():
            if key == "water_price":
                monthly_dates = pd.date_range(
                    start=pd.Timestamp(start_year, 1, 1),
                    end=pd.Timestamp(end_year, 12, 31),
                    freq="MS",
                )
                out = {
                    "time": [d.strftime("%Y-%m-%d") for d in monthly_dates],
                    "data": {},
                }
            else:
                out = {"time": list(range(start_year, end_year + 1)), "data": {}}

            # Expect self.geoms["areamaps/regions"] with columns: region_id, NAME_1 (state)
            for _, row in self.geoms["areamaps/regions"].iterrows():
                region_id = str(row["region_id"])
                state_name = row["NAME_1"]

                if state_name not in df.columns:
                    # If a region/state has no column, fill with zeros to avoid KeyErrors
                    if key == "water_price":
                        out["data"][region_id] = [0.0] * len(out["time"])
                    else:
                        out["data"][region_id] = [0.0] * len(out["time"])
                    continue

                series = df[state_name].sort_index()

                if key == "diversions":
                    # keep annual values mapped into the requested window, fallback to baseline
                    years_available = series.index.year
                    if len(years_available) == 0:
                        out["data"][region_id] = [0.0] * len(out["time"])
                        continue

                    min_obs, max_obs = years_available.min(), years_available.max()
                    max_obs = (
                        min(max_obs, int(inflation_years[-1]))
                        if inflation_years
                        else max_obs
                    )

                    # baseline (deflated to first year using inflation)
                    real_vals = []
                    for yr in range(min_obs, max_obs + 1):
                        nom = series.loc[series.index.year == yr]
                        if nom.empty:
                            continue
                        factor = 1.0
                        for y in range(min_obs + 1, yr + 1):
                            y_str = str(y)
                            if y_str in inflation_years:
                                idx = inflation_years.index(y_str)
                                factor *= inflation_rates["data"][region_id][idx]
                        real_vals.append(nom.iloc[0] / factor)

                    baseline = float(real_vals[0]) if real_vals else 0.0

                    out_series = pd.Series(index=out["time"], dtype=float)
                    for y in out_series.index:
                        if y in years_available:
                            out_series.loc[y] = float(
                                series.loc[series.index.year == y].iloc[0]
                            )
                        elif y < min_obs or y > max_obs:
                            out_series.loc[y] = baseline
                        else:
                            # inside gap → baseline (simple)
                            out_series.loc[y] = baseline

                    out["data"][region_id] = out_series.tolist()

                else:  # water_price (monthly)
                    # prepare inflation helper (apply once when crossing a year)
                    def infl_factor(
                        prev_date: pd.Timestamp, curr_date: pd.Timestamp, reg: str
                    ) -> float:
                        if curr_date.year != prev_date.year:
                            y_str = str(curr_date.year)
                            if y_str in inflation_years:
                                idx = inflation_years.index(y_str)
                                return float(inflation_rates["data"][reg][idx])
                        return 1.0

                    prices = pd.Series(index=pd.to_datetime(out["time"]), dtype=float)

                    # seed known monthly values
                    common = series.index.intersection(prices.index)
                    prices.loc[common] = series.loc[common].astype(float)

                    min_obs_date = series.index.min() if not series.empty else None
                    baseline = 0.0
                    if min_obs_date is not None:
                        # deflate all observed prices back to min_obs_date.year → take first as baseline
                        base_year = min_obs_date.year
                        reals = []
                        for ts in series.index:
                            val = float(series.loc[ts])
                            factor = 1.0
                            for y in range(base_year + 1, ts.year + 1):
                                y_str = str(y)
                                if y_str in inflation_years:
                                    idx = inflation_years.index(y_str)
                                    factor *= float(
                                        inflation_rates["data"][region_id][idx]
                                    )
                            reals.append(val / factor)
                        baseline = float(reals[0]) if reals else 0.0

                        prices.loc[min_obs_date] = baseline

                        # forward fill with annual inflation at year boundaries
                        idx_all = prices.index
                        start_i = idx_all.get_loc(min_obs_date)
                        for i in range(start_i + 1, len(idx_all)):
                            prev_d, cur_d = idx_all[i - 1], idx_all[i]
                            if pd.isna(prices.iloc[i]):
                                prices.iloc[i] = prices.loc[prev_d] * infl_factor(
                                    prev_d, cur_d, region_id
                                )
                        # backward (undo year boundary once when stepping back)
                        for i in range(start_i - 1, -1, -1):
                            cur_d, next_d = idx_all[i], idx_all[i + 1]
                            if pd.isna(prices.iloc[i]):
                                f = infl_factor(cur_d, next_d, region_id)
                                prices.iloc[i] = prices.loc[next_d] / (
                                    f if f != 0 else 1.0
                                )
                    else:
                        prices[:] = baseline

                    out["data"][region_id] = prices.tolist()

            self.set_dict(out, name=f"economics/{key}")

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

        self.set_dict(out, name=f"economics/{price_name}")
