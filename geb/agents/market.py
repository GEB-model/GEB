"""This module contains the Market agent class for simulating market dynamics in the GEB model."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import statsmodels.api as sm
from numpy.linalg import LinAlgError

from geb.types import TwoDArrayFloat32
from geb.workflows.io import read_dict

from ..data import load_regional_crop_data_from_dict
from ..store import DynamicArray
from .general import AgentBaseClass

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel


class Market(AgentBaseClass):
    """This class is used to simulate the Market.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).

    Note:
        Currently assume single market for all crops.
    """

    def __init__(self, model: GEBModel, agents: Agents) -> None:
        """Initialize the Market agent module.

        Args:
            model: The GEB model.
            agents: The class that includes all agent types (allowing easier communication between agents).
        """
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

        if (
            "calibration" in self.model.config
            and "KGE_crops" in self.model.config["calibration"]["calibration_targets"]
        ):
            self.production_influence_calibration_factor = np.array(
                [
                    self.model.config["agent_settings"]["calibration_crops"][
                        f"price_{i}"
                    ]
                    for i in range(self._crop_prices[1].shape[2])
                ],
                dtype=np.float32,
            )
        else:
            self.production_influence_calibration_factor = np.ones(
                self._crop_prices[1].shape[2], dtype=np.float32
            )

    @property
    def name(self) -> str:
        """Get the name of the module.

        This is used for saving data to disk

        Returns:
            The name of the module.
        """
        return "agents.market"

    def spinup(self) -> None:
        """Initialize market arrays for production, income, and model parameters.

        This method sets up the data structures needed to track market dynamics over
        the simulation period. It initializes arrays for crop production, farmer income,
        and the parameters of the price model. It also loads and processes
        historical inflation data to adjust prices over time.
        """
        n_crops = len(self.agents.crop_farmers.var.crop_ids.keys())
        n_years = (
            self.model.config["general"]["end_time"].year
            - self.model.config["general"]["spinup_time"].year
        ) + 20
        self.var.production = DynamicArray(
            n=n_crops,
            max_n=n_crops,
            dtype=np.float32,
            fill_value=np.nan,
            extra_dims=(n_years,),
            extra_dims_names=["years"],
        )
        self.var.total_farmer_income = DynamicArray(
            n=n_crops,
            max_n=n_crops,
            dtype=np.float32,
            fill_value=np.nan,
            extra_dims=(n_years,),
            extra_dims_names=["years"],
        )

        self.var.parameters = DynamicArray(
            n=n_crops,
            max_n=n_crops,
            dtype=np.float32,
            fill_value=np.nan,
            extra_dims=(2,),
            extra_dims_names=["params"],
        )

        inflation = read_dict(
            self.model.files["dict"]["socioeconomics/inflation_rates"]
        )
        inflation["time"] = [int(time) for time in inflation["time"]]
        start_idx = inflation["time"].index(
            self.model.config["general"]["spinup_time"].year
        )
        end_idx = inflation["time"].index(self.model.config["general"]["end_time"].year)
        for region in inflation["data"]:
            region_inflation = [1] + inflation["data"][region][
                start_idx + 1 : end_idx + 1
            ]
            self.var.cumulative_inflation_per_region = np.cumprod(region_inflation)

    def estimate_price_model(self) -> None:
        """Estimate the parameters of the crop price model using OLS regression.

        This method fits a log-log ordinary least squares (OLS) model to estimate
        the relationship between crop production and price. The model is specified as:
        log(price) = β₀ + β₁ * log(production)

        The estimated parameters (β₀ and β₁) are stored for each crop. The model
        is fitted using data from the beginning of the simulation up to the
        current year. It handles cases where the regression fails to converge.
        """
        estimation_start_year = 1  # skip first year
        estimation_end_year = (
            self.model.current_time.year
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
            prod = production[crop]
            if prod.sum() == 0:
                continue
            # Defining the independent variables (add a constant term for the intercept)
            X = sm.add_constant(np.log(prod))

            # Defining the dependent variable
            price = total_farmer_income[crop] / prod

            y = np.log(price)

            # Fitting the model
            try:
                model = sm.OLS(y, X).fit()
            except LinAlgError:  # SVD did not converge
                warnings.warn(f"Crop {crop}: SVD did not converge – skipped")
                continue
            except ValueError as e:  # any other statsmodels problem
                warnings.warn(f"Crop {crop}: {e} – skipped")
                continue

            model_parameters = model.params
            # assert model_parameters[-1] < 0, "Price increase with decreasing yield"
            self.var.parameters[crop] = model_parameters

        print(self.var.parameters)

    def get_modelled_crop_prices(self) -> TwoDArrayFloat32:
        """Calculate and return crop prices based on the estimated price model.

        This method uses the previously estimated OLS model parameters to predict
        crop prices for the current year. The prediction is based on the previous
        year's production levels. The formula for the prediction is:
        price = exp(β₀ + β₁ * log(production))

        The predicted prices are then adjusted for inflation using a cumulative
        inflation factor.

        Returns:
            A 2D numpy array of predicted crop prices for each region and crop,
            adjusted for inflation (USD/ton). Infilation is included based on the
            current year index.
        """
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
                + self.production_influence_calibration_factor
                * np.log(production)
                * self.var.parameters[:, 1]
            )
            price_pred_per_region[region_idx, :] = price_pred

        assert np.all(
            price_pred_per_region[:, self.var.production[:, self.year_index - 1] > 0]
            > 0
        ), "Negative prices predicted"

        # TODO: This assumes that the inflation is the same for all regions (region_idx=0)
        return (
            price_pred_per_region
            * self.var.cumulative_inflation_per_region[self.year_index]
        )

    def track_production_and_price(self) -> None:
        """Aggregate and record total crop production and farmer income for the current timestep.

        This method is called at each timestep to update the yearly production and
        income totals. It aggregates the harvested yield and income from all
        crop farmers and stores it in the respective yearly arrays. Income is
        adjusted for inflation before being stored.
        """
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
        """Execute the market agent's actions for the current timestep.

        This function tracks production and income daily. It also triggers the
        re-estimation of the price model at the end of the spinup period and
        at regular 5-year intervals thereafter.
        """
        if not self.model.simulate_hydrology:
            return
        self.track_production_and_price()
        if (
            # run price model at the end of the spinup
            (self.model.current_time == self.model.spinup_end and self.model.in_spinup)
            or
            # and on 5-year anniversaries
            (
                not self.model.in_spinup
                and (self.model.current_time.year - self.model.run_start.year) % 5 == 0
                and (
                    self.model.current_time.month == 1
                    and self.model.current_time.day == 1
                )
                and (self.model.current_time.year - self.model.run_start.year) >= 5
            )
        ):
            self.estimate_price_model()

        self.report(locals())

    @property
    def crop_prices(self) -> TwoDArrayFloat32:
        """Get the crop prices for the current timestep per region.

        If dynamic market is enabled, it returns the modelled crop prices. Otherwise, it returns the static crop prices.

        Returns:
            The crop prices for the current timestep per region. The first dimension corresponds to regions, and the second to crop IDs.
        """
        if (
            not self.model.in_spinup
            and "dynamic_market" in self.config
            and self.config["dynamic_market"] is True
        ):
            simulated_price = self.get_modelled_crop_prices()
            return simulated_price
        else:
            index = self._crop_prices[0].get(self.model.current_time)
            return self._crop_prices[1][index]

    @property
    def year_index(self) -> int:
        """Get the current year index since the start of the simulation.

        Returns:
            The current year index.
        """
        return (
            self.model.current_time.year
            - self.model.config["general"]["spinup_time"].year
        )
