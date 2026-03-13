from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from geb.build.methods import build_method

from .. import GEBModel


class Agents(GEBModel):
    """Build methods for agents in GEB, including hetao-specific logic."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize an Agents model.

        Forwards all arguments to the base ``GEBModel`` initializer.

        Args:
            *args: Positional arguments forwarded to ``GEBModel``.
            **kwargs: Keyword arguments forwarded to ``GEBModel``.
        """
        super().__init__(*args, **kwargs)

    @build_method(required=True, depends_on=["setup_economic_data"])
    def setup_water_prices_hetao(
        self,
        start_year: int,
        end_year: int,
        water_price_xlsx: str,
        sheet_water_price: str = "Sheet1",
        col_year: str = "year",
        col_price_cny_m3: str = "price_cny_m3",
        out_name: str = "socioeconomics/water_price",
    ) -> None:
        """
        Build yearly water price dict (USD/m3) for Hetao, same for all regions.

        Rules:
        - Convert CNY/m3 -> USD/m3 using socioeconomics/LCU_per_USD.yml (CNY per USD).
        - If observed period shorter than simulation:
            earlier years -> 0
            later years   -> last observed value (constant)
        """
        years = list(range(int(start_year), int(end_year) + 1))

        # --- 1) FX: LCU_per_USD (CNY per USD). USD = CNY / (CNY per USD)
        fx_dict: dict[str, Any] = self.params["socioeconomics/LCU_per_USD"]
        fx_years = [int(y) for y in fx_dict["time"]]

        fx_data = fx_dict["data"]
        fx_key0 = (
            "0" if "0" in fx_data else (0 if 0 in fx_data else list(fx_data.keys())[0])
        )
        fx_rates = [float(v) for v in fx_data[fx_key0]]
        year_to_fx = dict(zip(fx_years, fx_rates))

        def cny_to_usd(price_cny: float, year: int) -> float:
            if year in year_to_fx:
                fx = year_to_fx[year]
            else:
                # 超出 FX 覆盖年份：用最后一年 FX（你这边一般是 2024）
                fx = year_to_fx[max(year_to_fx.keys())]
            return float(price_cny) / float(fx) if fx else 0.0

        # --- 2) Load water price (CNY/m3), one series
        wp_df = pd.read_excel(
            Path(water_price_xlsx), sheet_name=sheet_water_price
        ).copy()
        wp_df[col_year] = wp_df[col_year].astype(int)
        wp_df = wp_df.sort_values(col_year)

        obs_min = int(wp_df[col_year].min())
        obs_max = int(wp_df[col_year].max())
        year_to_cny = dict(
            zip(
                wp_df[col_year].tolist(), wp_df[col_price_cny_m3].astype(float).tolist()
            )
        )
        last_obs_cny = float(year_to_cny[obs_max])

        # --- 3) Build padded yearly USD series
        series_usd: list[float] = []
        for y in years:
            if y < obs_min:
                series_usd.append(0.0)
            elif y > obs_max:
                series_usd.append(cny_to_usd(last_obs_cny, y))
            else:
                series_usd.append(cny_to_usd(float(year_to_cny[y]), y))

        # --- 4) Fill ALL regions with the same series
        try:
            region_ids = [
                str(rid) for rid in self.geom["regions"]["region_id"].tolist()
            ]
        except Exception:
            # 如果 regions 还没 build 出来，就先写一个默认 region_id=0
            region_ids = ["0"]

        out: dict[str, Any] = {
            "time": years,
            "data": {rid: series_usd for rid in region_ids},
        }

        # --- 5) Write to model/input/dict and register in files.yml via file library
        self.set_params(out, name=out_name)

    def _get_region_ids_or_default(self) -> list[str]:
        try:
            return [str(rid) for rid in self.geom["regions"]["region_id"].tolist()]
        except Exception:
            return ["0"]

    def _make_fx_map_from_lcu_per_usd(self) -> dict[int, float]:
        """
        Build year -> (LCU per USD) map from existing socioeconomics/LCU_per_USD dict.
        Convention in GEB: data["0"] is the national series.
        """
        fx_dict: dict[str, Any] = self.params["socioeconomics/LCU_per_USD"]
        fx_years = [int(y) for y in fx_dict["time"]]

        fx_data = fx_dict["data"]
        fx_key0 = (
            "0" if "0" in fx_data else (0 if 0 in fx_data else list(fx_data.keys())[0])
        )
        fx_rates = [float(v) for v in fx_data[fx_key0]]

        return dict(zip(fx_years, fx_rates))

    def _cny_to_usd(
        self, value_cny: float, year: int, fx_map: dict[int, float]
    ) -> float:
        """
        Convert CNY -> USD using LCU_per_USD (CNY per USD).
        If year exceeds fx coverage (e.g., 2025), use the last available FX year.
        """
        if year in fx_map:
            fx = fx_map[year]
        else:
            fx = fx_map[max(fx_map.keys())]
        return float(value_cny) / float(fx) if fx else 0.0

    @build_method(required=True, depends_on=["setup_economic_data"])
    def setup_operation_costs_hetao(
        self,
        start_year: int,
        end_year: int,
        xlsx: str,
        sheet: str = "Sheet1",
        col_year: str = "year",
        col_surface: str = "operation_cost_surface_CNY/m2/year",
        col_sprinkler: str = "operation_cost_sprinkler_CNY/m2/year",
        col_drip: str = "operation_cost_drip_CNY/m2/year",
        input_currency: str = "CNY",  # "CNY" or "USD"
    ) -> None:
        """
        Build irrigation operation costs for Hetao (USD/m2/year), same for all regions.
        If input_currency == "CNY", convert to USD using socioeconomics/LCU_per_USD.
        """
        df = pd.read_excel(Path(xlsx), sheet_name=sheet).copy()

        for c in [col_year, col_surface, col_sprinkler, col_drip]:
            if c not in df.columns:
                raise ValueError(
                    f"Column '{c}' not found in {xlsx}:{sheet}. Columns={list(df.columns)}"
                )

        df[col_year] = df[col_year].astype(int)
        df = df.sort_values(col_year)

        years = list(range(int(start_year), int(end_year) + 1))
        df_indexed = df.set_index(col_year)

        fx_map = (
            self._make_fx_map_from_lcu_per_usd()
            if input_currency.upper() == "CNY"
            else {}
        )

        def maybe_convert(v: float, y: int) -> float:
            return (
                self._cny_to_usd(v, y, fx_map)
                if input_currency.upper() == "CNY"
                else float(v)
            )

        surface_filled = (
            df_indexed[col_surface].astype(float).reindex(years).ffill().bfill()
        )
        sprinkler_filled = (
            df_indexed[col_sprinkler].astype(float).reindex(years).ffill().bfill()
        )
        drip_filled = df_indexed[col_drip].astype(float).reindex(years).ffill().bfill()

        series_surface = [
            maybe_convert(v, y) for y, v in zip(years, surface_filled.tolist())
        ]
        series_sprinkler = [
            maybe_convert(v, y) for y, v in zip(years, sprinkler_filled.tolist())
        ]
        series_drip = [maybe_convert(v, y) for y, v in zip(years, drip_filled.tolist())]

        region_ids = self._get_region_ids_or_default()

        out_surface: dict[str, Any] = {
            "time": years,
            "data": {rid: series_surface for rid in region_ids},
        }
        out_sprinkler: dict[str, Any] = {
            "time": years,
            "data": {rid: series_sprinkler for rid in region_ids},
        }
        out_drip: dict[str, Any] = {
            "time": years,
            "data": {rid: series_drip for rid in region_ids},
        }

        self.set_params(out_surface, name="socioeconomics/operation_cost_surface")
        self.set_params(out_sprinkler, name="socioeconomics/operation_cost_sprinkler")
        self.set_params(out_drip, name="socioeconomics/operation_cost_drip")

    @build_method(required=True, depends_on=["setup_economic_data"])
    def setup_capital_costs_hetao(
        self,
        start_year: int,
        end_year: int,
        xlsx: str,
        sheet: str = "Sheet1",
        col_year: str = "year",
        col_surface: str = "capital_cost_surface_CNY/m2",
        col_sprinkler: str = "capital_cost_sprinkler_CNY/m2",
        col_drip: str = "capital_cost_drip_CNY/m2",
        input_currency: str = "CNY",  # "CNY" or "USD"
    ) -> None:
        """
        Build irrigation capital costs for Hetao (USD/m2, one-time), same for all regions.
        If input_currency == "CNY", convert to USD using socioeconomics/LCU_per_USD.
        """
        df = pd.read_excel(Path(xlsx), sheet_name=sheet).copy()

        for c in [col_year, col_surface, col_sprinkler, col_drip]:
            if c not in df.columns:
                raise ValueError(
                    f"Column '{c}' not found in {xlsx}:{sheet}. Columns={list(df.columns)}"
                )

        df[col_year] = df[col_year].astype(int)
        df = df.sort_values(col_year)

        years = list(range(int(start_year), int(end_year) + 1))
        df_indexed = df.set_index(col_year)

        fx_map = (
            self._make_fx_map_from_lcu_per_usd()
            if input_currency.upper() == "CNY"
            else {}
        )

        def maybe_convert(v: float, y: int) -> float:
            return (
                self._cny_to_usd(v, y, fx_map)
                if input_currency.upper() == "CNY"
                else float(v)
            )

        surface_filled = (
            df_indexed[col_surface].astype(float).reindex(years).ffill().bfill()
        )
        sprinkler_filled = (
            df_indexed[col_sprinkler].astype(float).reindex(years).ffill().bfill()
        )
        drip_filled = df_indexed[col_drip].astype(float).reindex(years).ffill().bfill()

        series_surface = [
            maybe_convert(v, y) for y, v in zip(years, surface_filled.tolist())
        ]
        series_sprinkler = [
            maybe_convert(v, y) for y, v in zip(years, sprinkler_filled.tolist())
        ]
        series_drip = [maybe_convert(v, y) for y, v in zip(years, drip_filled.tolist())]

        region_ids = self._get_region_ids_or_default()

        out_surface: dict[str, Any] = {
            "time": years,
            "data": {rid: series_surface for rid in region_ids},
        }
        out_sprinkler: dict[str, Any] = {
            "time": years,
            "data": {rid: series_sprinkler for rid in region_ids},
        }
        out_drip: dict[str, Any] = {
            "time": years,
            "data": {rid: series_drip for rid in region_ids},
        }

        self.set_params(out_surface, name="socioeconomics/capital_cost_surface")
        self.set_params(out_sprinkler, name="socioeconomics/capital_cost_sprinkler")
        self.set_params(out_drip, name="socioeconomics/capital_cost_drip")
