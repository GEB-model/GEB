# -*- coding: utf-8 -*-
import json

import numpy as np
import statsmodels.api as sm

from ..data import load_regional_crop_data_from_dict
from ..store import DynamicArray
from .general import AgentBaseClass


class Market(AgentBaseClass):
    """This class is used to simulate the Market.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).

    Note:
        Currently assume single market for all crops.
    """

    def __init__(self, model, agents):
        super().__init__(model)
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["market"]
            if "market" in self.model.config["agent_settings"]
            else {}
        )

        if self.model.in_spinup:
            self.spinup()

        self._crop_prices = load_regional_crop_data_from_dict(
            self.model, "crops/crop_prices"
        )

    @property
    def name(self):
        return "agents.market"

    def spinup(self) -> None:
        with open(self.model.files["dict"]["socioeconomics/inflation_rates"], "r") as f:
            inflation = json.load(f)
            inflation["time"] = [int(time) for time in inflation["time"]]
            start_idx = inflation["time"].index(
                self.model.config["general"]["spinup_time"].year
            )
            end_idx = inflation["time"].index(
                self.model.config["general"]["end_time"].year
            )
            for region in inflation["data"]:
                region_inflation = [1] + inflation["data"][region][
                    start_idx + 1 : end_idx + 1
                ]
                self.var.cumulative_inflation_per_region = np.cumprod(region_inflation)

        n_crops = len(self.agents.crop_farmers.var.crop_ids.keys())
        n_years = (
            self.model.config["general"]["end_time"].year
            - self.model.config["general"]["spinup_time"].year
        ) + 1
        self.var.production = DynamicArray(
            n=n_crops,
            max_n=n_crops,
            dtype=np.float32,
            fill_value=np.nan,
            extra_dims=(n_years,),
            extra_dims_names=("years",),
        )
        self.var.total_farmer_income = DynamicArray(
            n=n_crops,
            max_n=n_crops,
            dtype=np.float32,
            fill_value=np.nan,
            extra_dims=(n_years,),
            extra_dims_names=("years",),
        )

    def estimate_price_model(self) -> None:
        self.var.parameters = np.full((self.var.production.shape[0], 2), np.nan)

        estimation_start_year = 1  # skip first year
        estimation_end_year = (
            self.model.config["general"]["start_time"].year
            - self.model.config["general"]["spinup_time"].year
        )

        production = self.var.production[
            :,
            estimation_start_year:estimation_end_year,
        ]
        total_farmer_income = self.var.total_farmer_income[
            :,
            estimation_start_year:estimation_end_year,
        ]

        print("Look into increasing yield and increasing price")
        for crop in range(self.var.production.shape[0]):
            if production[crop].sum() == 0:
                continue
            # Defining the independent variables (add a constant term for the intercept)
            X = sm.add_constant(np.log(production[crop]))

            # Defining the dependent variable
            price = total_farmer_income[crop] / production[crop]

            y = np.log(price)

            # Fitting the model
            model = sm.OLS(y, X).fit()
            model_parameters = model.params
            # assert model_parameters[-1] < 0, "Price increase with decreasing yield"
            self.var.parameters[crop] = model_parameters

        print(self.var.parameters)

    def get_modelled_crop_prices(self) -> np.ndarray:
        number_of_regions = self._crop_prices[1].shape[1]

        price_pred_per_region = np.full(
            (number_of_regions, self.var.production.shape[0]),
            np.nan,
            dtype=np.float32,
        )
        for region_idx in range(number_of_regions):
            production = self.var.production[
                :, self.year_index - 1
            ]  # for now taking the previous year, should be updated
            price_pred = np.exp(
                1 * self.var.parameters[:, 0]
                + np.log(production) * self.var.parameters[:, 1]
            )
            price_pred_per_region[region_idx, :] = price_pred

        assert np.all(price_pred_per_region > 0), "Negative prices predicted"

        # TODO: This assumes that the inflation is the same for all regions (region_idx=0)
        return (
            price_pred_per_region
            * self.var.cumulative_inflation_per_region[self.year_index]
        )

    def track_production_and_price(self) -> None:
        if self.model.current_day_of_year == 1:
            self.var.production[:, self.year_index] = 0
            self.var.total_farmer_income[:, self.year_index] = 0
        mask = self.agents.crop_farmers.var.harvested_crop != -1
        # TODO: This does not yet diffentiate per region
        yield_per_crop = np.bincount(
            self.agents.crop_farmers.var.harvested_crop[mask],
            weights=self.agents.crop_farmers.var.actual_yield_per_farmer[mask],
            minlength=self.var.production.shape[0],
        )
        income_per_crop = np.bincount(
            self.agents.crop_farmers.var.harvested_crop[mask],
            weights=self.agents.crop_farmers.income_farmer[mask],
            minlength=self.var.production.shape[0],
        )
        self.var.production[:, self.year_index] += yield_per_crop
        # TODO: This assumes that the inflation is the same for all crops
        self.var.total_farmer_income[:, self.year_index] += (
            income_per_crop / self.var.cumulative_inflation_per_region[self.year_index]
        )

    def step(self) -> None:
        """This function is run each timestep."""
        if not self.model.simulate_hydrology:
            return
        self.track_production_and_price()
        self.report(self, locals())

    @property
    def crop_prices(self) -> np.ndarray:
        if (
            not self.model.in_spinup
            and "dynamic_market" in self.config
            and self.config["dynamic_market"] is True
        ):
            return self.get_modelled_crop_prices()
        else:
            if self._crop_prices[0] is None:
                print("WARNING: Using static crop prices")
                return np.full(
                    (
                        len(self.model.regions),
                        len(self.agents.crop_farmers.var.crop_ids),
                    ),
                    self._crop_prices[1],
                )
            else:
                index = self._crop_prices[0].get(self.model.current_time)
                return self._crop_prices[1][index]

    @property
    def year_index(self) -> int:
        return (
            self.model.current_time.year
            - self.model.config["general"]["spinup_time"].year
        )
