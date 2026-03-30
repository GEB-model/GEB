"""Module containing build methods for the agents for GEB."""

import unicodedata
import warnings
from datetime import datetime
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from geb.build.methods import build_method
from geb.build.workflows.crop_calendars import donate_and_receive_crop_prices
from geb.geb_types import TwoDArrayBool, TwoDArrayInt32
from geb.workflows.io import get_window
from geb.workflows.raster import (
    clip_with_grid,
    pixels_to_coords,
    sample_from_map,
)

from ..workflows.conversions import (
    COUNTRY_NAME_TO_ISO3,
    TRADE_REGIONS,
    setup_donor_countries,
)
from ..workflows.farmers import create_farm_distributions, create_farms
from .base import BuildModelBase


class Agents(BuildModelBase):
    """Contains all build methods for the agents for GEB."""

    def __init__(self) -> None:
        """Initialize the Agents build methods."""
        pass

    @build_method(
        depends_on=[
            "set_ssp",
            "set_time_range",
            "setup_regions_and_land_use",
            "setup_household_characteristics",
        ],
        required=True,
    )
    def setup_water_demand(self) -> None:
        """Sets up the water demand data for GEB.

        Notes:
            This method sets up the water demand data for GEB. It retrieves the domestic, industry, and
            livestock water demand data from the specified data catalog and sets it as forcing data in the model. The domestic
            water demand and consumption data are retrieved from the 'cwatm_domestic_water_demand' dataset, while the industry
            water demand and consumption data are retrieved from the 'cwatm_industry_water_demand' dataset. The livestock water
            consumption data is retrieved from the 'cwatm_livestock_water_demand' dataset.

            The domestic water demand and consumption data are provided at a monthly time step, while the industry water demand
            and consumption data are provided at an annual time step. The livestock water consumption data is provided at a
            monthly time step, but is assumed to be constant over the year.

            The resulting water demand data is set as forcing data in the model with names of the form 'water_demand/{demand_type}'.
        """
        start_model_time = self.start_date.year
        end_model_time = self.end_date.year

        municipal_water_withdrawal_m3_per_capita_per_year = self.data_catalog.fetch(
            "aquastat"
        ).read(
            indicator="Municipal water withdrawal per capita (total population) [m3/inhab/year]"
        )

        # Filter the data for the model time
        municipal_water_withdrawal_m3_per_capita_per_year = (
            municipal_water_withdrawal_m3_per_capita_per_year[
                (
                    municipal_water_withdrawal_m3_per_capita_per_year["Year"]
                    >= start_model_time
                )
                & (
                    municipal_water_withdrawal_m3_per_capita_per_year["Year"]
                    <= end_model_time
                )
            ]
        )

        municipal_water_withdrawal_per_capita = np.full_like(
            self.array["agents/households/region_id"],
            np.nan,
            dtype=np.float32,
        )

        municipal_water_withdrawal_m3_per_capita_per_day_multiplier = pd.DataFrame()
        for _, region in self.geom["regions"].iterrows():
            ISO3 = region["ISO3"]
            region_id = region["region_id"]

            def load_water_demand_data(
                ISO3: str,
            ) -> pd.DataFrame:
                """Load municipal water demand data for a given ISO3 code.

                Args:
                    ISO3: The ISO3 code of the region.

                Returns:
                    The municipal water withdrawal data for the given ISO3 code.
                """
                # Load the municipal water demand data for the given ISO3 code
                if ISO3 not in municipal_water_withdrawal_m3_per_capita_per_year.index:
                    countries_with_data = municipal_water_withdrawal_m3_per_capita_per_year.index.unique().tolist()
                    donor_countries = setup_donor_countries(
                        self.data_catalog,
                        self.geom["global_countries"],
                        countries_with_data,
                        alternative_countries=self.geom["regions"]["ISO3"]
                        .unique()
                        .tolist(),
                    )
                    ISO3 = donor_countries[ISO3]

                    self.logger.warning(
                        f"Country {region['ISO3']} not present in municipal water demand data, using donor country {ISO3}"
                    )

                # now that we have data for a single country, we can set the index to year and return the value column
                municipal_water_withdrawal_m3_per_capita_per_year_country = (
                    municipal_water_withdrawal_m3_per_capita_per_year.loc[ISO3]
                ).set_index("Year")["Value"]

                municipal_water_withdrawal_m3_per_capita_per_day_country = (
                    municipal_water_withdrawal_m3_per_capita_per_year_country / 365.2425
                )
                return municipal_water_withdrawal_m3_per_capita_per_day_country

            municipal_water_withdrawal_m3_per_capita_per_day_country = (
                load_water_demand_data(ISO3)
            )

            if len(municipal_water_withdrawal_m3_per_capita_per_day_country) == 0:
                countries_with_water_withdrawal_data = (
                    municipal_water_withdrawal_m3_per_capita_per_year.dropna(
                        axis=0, how="any"
                    )
                    .index.unique()
                    .tolist()
                )

                donor_countries = setup_donor_countries(
                    self.data_catalog,
                    self.geom["global_countries"],
                    countries_with_water_withdrawal_data,
                    alternative_countries=self.geom["regions"]["ISO3"]
                    .unique()
                    .tolist(),
                )
                donor_country = donor_countries[ISO3]
                self.logger.info(
                    f"Missing municipal water withdrawal data for {ISO3}, filling with donor country {donor_country}"
                )
                municipal_water_withdrawal_m3_per_capita_per_day_country = (
                    load_water_demand_data(donor_country)
                )

            assert len(municipal_water_withdrawal_m3_per_capita_per_day_country) > 0, (
                f"Missing municipal water withdrawal data for {ISO3}"
            )

            if municipal_water_withdrawal_m3_per_capita_per_day_country.isna().any():
                missing_years = (
                    municipal_water_withdrawal_m3_per_capita_per_day_country[
                        municipal_water_withdrawal_m3_per_capita_per_day_country.isna()
                    ].index.tolist()
                )
                # Find countries that have data for all the missing years
                countries_with_data: set[str] = set()
                for (
                    country,
                    group,
                ) in municipal_water_withdrawal_m3_per_capita_per_year.groupby(level=0):
                    if np.isin(missing_years, group["Year"]).all():
                        countries_with_data.add(country)

                # fill the municipal water withdrawal data for missing years from donor countries
                donor_countries = setup_donor_countries(
                    self.data_catalog,
                    self.geom["global_countries"],
                    countries_with_data,
                    alternative_countries=self.geom["regions"]["ISO3"]
                    .unique()
                    .tolist(),
                )
                donor_country = donor_countries[ISO3]
                self.logger.info(
                    f"Missing municipal water demand data for {ISO3}, using donor country {donor_country}"
                )
                municipal_water_withdrawal_m3_per_capita_per_day_donor = (
                    load_water_demand_data(donor_country)
                )

                # use the donor country data to fill the missing values
                for year in missing_years:
                    municipal_water_withdrawal_m3_per_capita_per_day_country.loc[
                        year
                    ] = municipal_water_withdrawal_m3_per_capita_per_day_donor.loc[year]

            assert not municipal_water_withdrawal_m3_per_capita_per_day_country.isna().any(), (
                f"Missing municipal water demand data for {ISO3} after donor filling"
            )

            municipal_water_withdrawal_m3_per_capita_per_day: pd.DataFrame = (
                municipal_water_withdrawal_m3_per_capita_per_day_country.reindex(
                    list(
                        range(
                            self.start_date.year,
                            self.end_date.year + 1,
                        )
                    )
                )
                .interpolate(method="linear")
                .bfill()
            )  # interpolate also extrapolates forward with constant values

            assert municipal_water_withdrawal_m3_per_capita_per_day.max() < 10, (
                f"Too large water demand data for {ISO3}"
            )

            # set baseline year for municipal water demand
            if 2000 not in municipal_water_withdrawal_m3_per_capita_per_day.index:
                # get first year with data
                first_year = municipal_water_withdrawal_m3_per_capita_per_day.index[0]
                self.logger.warning(
                    f"Missing 2000 data for {ISO3}, using first year {first_year} as baseline"
                )
                municipal_water_withdrawal_baseline_m3_per_capita_per_day = (
                    municipal_water_withdrawal_m3_per_capita_per_day.loc[
                        first_year
                    ].item()
                )
            # use the 2000 as baseline (default)
            municipal_water_withdrawal_baseline_m3_per_capita_per_day: pd.DataFrame = (
                municipal_water_withdrawal_m3_per_capita_per_day.loc[2000].item()
            )

            municipal_water_withdrawal_per_capita[
                self.array["agents/households/region_id"] == region_id
            ] = municipal_water_withdrawal_baseline_m3_per_capita_per_day

            # scale municipal water demand table to use baseline as 1.00 and scale other values
            # relatively
            municipal_water_withdrawal_m3_per_capita_per_day_multiplier[region_id] = (
                municipal_water_withdrawal_m3_per_capita_per_day
                / municipal_water_withdrawal_baseline_m3_per_capita_per_day
            )

        # we don't want to calculate the water demand for every year,
        # so instead we use a baseline (2000 for easy reasoning), and scale
        # the other years relatively to the baseline
        self.set_table(
            municipal_water_withdrawal_m3_per_capita_per_day_multiplier,
            name="municipal_water_withdrawal_m3_per_capita_per_day_multiplier",
        )

        assert not np.isnan(municipal_water_withdrawal_per_capita).any(), (
            "Missing municipal water demand per capita data"
        )
        self.set_array(
            municipal_water_withdrawal_per_capita,
            name="agents/households/municipal_water_withdrawal_per_capita_m3_baseline",
        )

        self.logger.info("Setting up other water demands")

        def parse_demand(file: str, variable: str, ssp: str) -> xr.DataArray:
            """Sets up the water demand data for a given demand type.

            Args:
                file: The file name of the dataset.
                variable: The variable name in the dataset.
                ssp: The SSP scenario to use.

            Returns:
                An xarray DataArray containing the water demand data for the specified demand type and SSP scenario.
            """
            ds_historic = self.data_catalog.fetch(
                f"cwatm_{file}_historical_year"
            ).read()
            da_historic = ds_historic.isel(
                get_window(ds_historic.x, ds_historic.y, self.bounds, buffer=2)
            )[variable]

            ds_future = self.data_catalog.fetch(f"cwatm_{file}_{ssp}_year").read()
            da_future = ds_future.isel(
                get_window(ds_future.x, ds_future.y, self.bounds, buffer=2)
            )[variable]

            da_future = da_future.sel(
                time=slice(da_historic.time[-1] + 1, da_future.time[-1])
            )

            da = xr.concat([da_historic, da_future], dim="time")
            # assert dataset is monotonically increasing
            assert (da.time.diff("time") == 1).all(), "not all years are there"

            da["time"] = pd.date_range(
                start=datetime(1901, 1, 1)
                + relativedelta(years=int(da.time[0].data.item())),
                periods=len(da.time),
                freq="YS",
            )

            assert (da.time.dt.year.diff("time") == 1).all(), "not all years are there"

            # Reindex to the model time range, filling missing years with backward/forward fill
            time_range: pd.DatetimeIndex = pd.date_range(
                start=datetime(self.start_date.year, 1, 1),
                end=datetime(self.end_date.year, 1, 1),
                freq="YS",
            )
            da = da.reindex(time=time_range).ffill("time").bfill("time")
            da.attrs["_FillValue"] = np.nan
            return da

        da = parse_demand(
            "industry_water_demand",
            "indWW",
            self.ssp,
        )

        self.set_other(da, name="water_demand/industry_water_demand")
        da = parse_demand(
            "industry_water_demand",
            "indCon",
            self.ssp,
        )
        self.set_other(da, name="water_demand/industry_water_consumption")
        da = parse_demand(
            "livestock_water_demand",
            "livestockConsumption",
            "ssp2",
        )
        self.set_other(da, name="water_demand/livestock_water_consumption")

    @build_method(required=True, depends_on=["setup_regions_and_land_use"])
    def setup_income_distribution_parameters(self) -> None:
        """Sets up the income distributions for GEB.

        Notes:
            This function used WID data on income distributions to generate income distribution profiles for each region.
            It retrieves the OECD income distribution data, processes it to extract mean and median income values for each country,
            and generates synthetic income distributions based on these parameters assuming a log-normal distribution. The resulting
            distributions are set as tables (be mindful for models outside of the EU that we do not yet account for currencies).
        """
        income_distribution_parameters = {}
        income_distributions = {}

        oecd_idd = self.data_catalog.fetch("oecd_idd").read()

        # clean data
        oecd_idd = oecd_idd[
            ["REF_AREA", "STATISTICAL_OPERATION", "TIME_PERIOD", "OBS_VALUE"]
        ]
        # get GDL regions to use their iso_code
        GDL_regions = self.data_catalog.fetch("GDL_regions_v4").read(
            geom=self.region.union_all()
        )

        gdl_countries = GDL_regions["iso_code"].unique().tolist()

        # setup donor countries for country missing in oecd data
        donor_countries = setup_donor_countries(
            self.data_catalog,
            self.geom["global_countries"],
            oecd_idd["REF_AREA"],
            alternative_countries=self.geom["regions"]["ISO3"].unique().tolist(),
        )

        for country in gdl_countries:
            income_distribution_parameters[country] = {}
            income_distributions[country] = {}
            if country not in oecd_idd["REF_AREA"].values:
                donor = donor_countries[country]
                self.logger.info(
                    f"Missing income distribution data for {country}, using donor country {donor}"
                )
                oecd_widd_country = oecd_idd[oecd_idd["REF_AREA"] == donor]
            else:
                oecd_widd_country = oecd_idd[oecd_idd["REF_AREA"] == country]

            # take the most recent year for each statistical operation separately
            mean_data = oecd_widd_country[
                oecd_widd_country["STATISTICAL_OPERATION"] == "MEAN"
            ]
            median_data = oecd_widd_country[
                oecd_widd_country["STATISTICAL_OPERATION"] == "MEDIAN"
            ]

            most_recent_mean = mean_data[
                mean_data["TIME_PERIOD"] == np.max(mean_data["TIME_PERIOD"])
            ]
            most_recent_median = median_data[
                median_data["TIME_PERIOD"] == np.max(median_data["TIME_PERIOD"])
            ]

            income_distribution_parameters[country]["MEAN"] = most_recent_mean[
                "OBS_VALUE"
            ].iloc[0]
            income_distribution_parameters[country]["MEDIAN"] = most_recent_median[
                "OBS_VALUE"
            ].iloc[0]

            # now also create national income distribution
            mu = np.log(income_distribution_parameters[country]["MEDIAN"])
            sd = np.sqrt(
                2
                * np.log(
                    income_distribution_parameters[country]["MEAN"]
                    / income_distribution_parameters[country]["MEDIAN"]
                )
            )
            income_distribution = np.sort(
                np.random.lognormal(mu, sd, 15_000).astype(np.int32)
            )
            income_distributions[country] = income_distribution

        # store to model table
        income_distribution_parameters_pd = pd.DataFrame(income_distribution_parameters)
        income_distributions_pd = pd.DataFrame(income_distributions)
        self.set_table(
            income_distribution_parameters_pd, "income/distribution_parameters"
        )
        self.set_table(income_distributions_pd, "income/national_distribution")

    @build_method(
        depends_on=["setup_regions_and_land_use", "set_time_range"], required=True
    )
    def setup_economic_data(self) -> None:
        """Sets up the economic data for GEB.

        Notes:
            This method sets up the lending rates and inflation rates data for GEB. It first retrieves the
            lending rates and inflation rates data from the World Bank dataset using the `get_geodataframe` method of the
            `data_catalog` object. It then creates dictionaries to store the data for each region, with the years as the time
            dimension and the lending rates or inflation rates as the data dimension.

            The lending rates and inflation rates data are converted from percentage to rate by dividing by 100 and adding 1.
            The data is then stored in the dictionaries with the region ID as the key.

            The resulting lending rates and inflation rates data are set as forcing data in the model with names of the form
            'socioeconomics/lending_rates' and 'socioeconomics/inflation_rates', respectively.
        """
        inflation_rates = self.data_catalog.fetch("wb_inflation_rate").read()
        inflation_rates_country_index = inflation_rates.set_index("Country Code")
        price_ratio = self.data_catalog.fetch("world_bank_price_ratio").read()
        LCU_per_USD = self.data_catalog.fetch("world_bank_LCU_per_USD").read()

        def select_years_from_df(
            df: pd.DataFrame, additional_cols: list[str]
        ) -> pd.DataFrame:
            """Selects columns corresponding to years and additional specified columns from a DataFrame.

            Args:
                df: The input DataFrame.
                additional_cols: A list of additional column names to retain.

            Returns:
                A DataFrame containing only the specified columns.
            """
            # Select columns: 'Country Name', 'Country Code', and columns containing "YR"
            columns_to_keep = additional_cols + [
                col
                for col in df.columns
                if col.isnumeric() and 1900 <= int(col) <= 3000
            ]
            filtered_df = df[columns_to_keep]
            return filtered_df

        def extract_years(df: pd.DataFrame) -> list[str]:
            """Extracts year columns from a DataFrame.

            Args:
                df: The input DataFrame.

            Returns:
                A list of year columns.
            """
            # Extract years that are numerically valid between 1900 and 3000
            return [
                col
                for col in df.columns
                if col.isnumeric() and 1900 <= int(col) <= 3000
            ]

        # Assuming dataframes for PPP and LCU per USD have been initialized
        price_ratio_filtered = select_years_from_df(
            price_ratio, ["Country Name", "Country Code"]
        )
        years_price_ratio = extract_years(price_ratio_filtered)
        price_ratio_dict: dict[str, Any] = {
            "time": years_price_ratio,
            "data": {},
        }  # price ratio

        lcu_filtered = select_years_from_df(
            LCU_per_USD, ["Country Name", "Country Code"]
        )

        years_lcu: list[str] = extract_years(lcu_filtered)
        lcu_dict: dict[str, Any] = {"time": years_lcu, "data": {}}  # LCU per USD

        # Assume lending_rates and inflation_rates are available
        # years_lending_rates = extract_years(lending_rates)
        years_inflation_rates = extract_years(inflation_rates)

        # lending_rates_dict = {"time": years_lending_rates, "data": {}}

        inflation_rates_dict: dict[str, Any] = {
            "time": years_inflation_rates,
            "data": {},
        }

        # Create a helper to process rates and assert single row data
        def retrieve_inflation_rates(
            df: pd.DataFrame,
            inflation_rate_columns: list[str],
            ISO3: str,
            convert_percent_to_ratio: bool = False,
        ) -> list[float]:
            """Processes rates for a given country code from a DataFrame.

            Args:
                df: The input DataFrame containing rate data.
                inflation_rate_columns: A list of columns corresponding to years.
                ISO3: The ISO3 country code to filter the data.
                convert_percent_to_ratio: Whether to convert percentage rates to ratios.

            Returns:
                A list of processed rates for the specified country code.
            """
            filtered_data = df.loc[df["Country Code"] == ISO3, inflation_rate_columns]
            if len(filtered_data) == 0:
                return list(
                    np.full(len(inflation_rate_columns), np.nan, dtype=np.float32)
                )
            if convert_percent_to_ratio:
                return (filtered_data.iloc[0] / 100 + 1).tolist()
            return filtered_data.iloc[0].tolist()

        USA_inflation_rates = retrieve_inflation_rates(
            inflation_rates,
            years_inflation_rates,
            "USA",
            convert_percent_to_ratio=True,
        )

        for _, region in self.geom["regions"].iterrows():
            region_id = str(region["region_id"])
            ISO3 = region["ISO3"]

            local_inflation_rates = retrieve_inflation_rates(
                inflation_rates,
                years_inflation_rates,
                ISO3,
                convert_percent_to_ratio=True,
            )

            if np.isnan(local_inflation_rates).any():
                # get index of nans in local_inflation_rates
                nan_indices = np.where(np.isnan(local_inflation_rates))[0]
                nan_years = [years_inflation_rates[i] for i in nan_indices]

                countries_with_data = (
                    inflation_rates_country_index[nan_years]
                    .dropna(axis=0, how="any")
                    .index.tolist()
                )

                ## get all the donor countries for countries in the dataset
                donor_countries = setup_donor_countries(
                    self.data_catalog,
                    self.geom["global_countries"],
                    countries_with_data,
                    alternative_countries=self.geom["regions"]["ISO3"]
                    .unique()
                    .tolist(),
                )
                donor_country = donor_countries[ISO3]

                self.logger.info(
                    f"Missing inflation rates for {ISO3}, using donor country {donor_country}"
                )
                donor_country_inflation_rates = retrieve_inflation_rates(
                    inflation_rates,
                    years_inflation_rates,
                    donor_country,
                    convert_percent_to_ratio=True,
                )

                # Replace NaN values in local_inflation_rates with values from similar_country_average_inflation
                for idx, value in zip(nan_indices, donor_country_inflation_rates):
                    local_inflation_rates[idx] = value

            assert not np.isnan(local_inflation_rates).any(), (
                f"Missing inflation rates for {region['ISO3']}"
            )
            inflation_rates_dict["data"][region_id] = (
                np.array(local_inflation_rates) / np.array(USA_inflation_rates)
            ).tolist()

            price_ratio_dict["data"][region_id] = retrieve_inflation_rates(
                price_ratio_filtered, years_price_ratio, region["ISO3"]
            )

            if np.all(np.isnan(price_ratio_dict["data"][region_id])):
                # check which countries are NOT fully nan
                price_ratio_filter_index = price_ratio_filtered.set_index(
                    "Country Code"
                )
                countries_with_price_ratio_data = (
                    price_ratio_filter_index[years_price_ratio]
                    .dropna(axis=0, how="all")
                    .index.unique()
                    .tolist()
                )
                donor_countries = setup_donor_countries(
                    self.data_catalog,
                    self.geom["global_countries"],
                    countries_with_price_ratio_data,
                    alternative_countries=self.geom["regions"]["ISO3"]
                    .unique()
                    .tolist(),
                )
                donor_country = donor_countries[ISO3]
                price_ratio_dict["data"][region_id] = retrieve_inflation_rates(
                    price_ratio_filtered,
                    years_price_ratio,
                    donor_country,
                )

                self.logger.info(
                    f"Missing price ratio data for {ISO3}, using donor country {donor_country}"
                )

            lcu_dict["data"][region_id] = retrieve_inflation_rates(
                lcu_filtered, years_lcu, region["ISO3"]
            )

            if np.all(np.isnan(lcu_dict["data"][region_id])):
                # check which countries are NOT fully nan
                lcu_dict_filter_index = lcu_filtered.set_index("Country Code")
                countries_with_lcu_data = (
                    lcu_dict_filter_index[years_lcu]
                    .dropna(axis=0, how="all")
                    .index.unique()
                    .tolist()
                )
                donor_countries = setup_donor_countries(
                    self.data_catalog,
                    self.geom["global_countries"],
                    countries_with_lcu_data,
                    alternative_countries=self.geom["regions"]["ISO3"]
                    .unique()
                    .tolist(),
                )
                donor_country = donor_countries[ISO3]
                lcu_dict["data"][region_id] = retrieve_inflation_rates(
                    lcu_filtered,
                    years_lcu,
                    donor_country,
                )

                self.logger.info(
                    f"Missing LCU (currency conversion) data for {ISO3}, using donor country {donor_country}"
                )

        for d in (
            inflation_rates_dict,
            price_ratio_dict,
            lcu_dict,
        ):
            # convert to pandas dataframe
            df = pd.DataFrame(d["data"], index=d["time"])
            df.index = df.index.astype(int)

            # re-index the inflation rates to ensure that at least all years from
            # model start to end are present. In addition, we add 10 years
            # to the beginning, since this is used in some of the model spinup.
            df = df.reindex(
                list(
                    range(
                        min(self.start_date.year - 10, df.index[0]),
                        max(self.end_date.year, df.index[-1]) + 1,
                    )
                )
            )
            # interpolate missing values in inflation rates. For extrapolation
            # linear interpolation uses the first and last value
            for column in df.columns:
                df[column] = df[column].interpolate(method="linear").bfill()

            d["time"] = df.index.astype(str).tolist()
            d["data"] = df.to_dict(orient="list")

        # lending_rates.index = lending_rates.index.astype(int)
        # extend lending rates to future
        # mean_lending_rate_since_reference_year = lending_rates.loc[
        #     reference_start_year:
        # ].mean(axis=0)
        # lending_rates = lending_rates.reindex(
        #     range(lending_rates.index.min(), project_future_until_year + 1)
        # ).fillna(mean_lending_rate_since_reference_year)

        # # convert back to dictionary
        # lending_rates_dict["time"] = lending_rates.index.astype(str).tolist()
        # lending_rates_dict["data"] = lending_rates.to_dict(orient="list")

        self.set_params(inflation_rates_dict, name="socioeconomics/inflation_rates")
        # self.set_params(lending_rates_dict, name="socioeconomics/lending_rates")
        self.set_params(price_ratio_dict, name="socioeconomics/price_ratio")
        self.set_params(lcu_dict, name="socioeconomics/LCU_per_USD")

    @build_method(required=True)
    def setup_irrigation_sources(self, irrigation_sources: dict[str, int]) -> None:
        """Sets up the irrigation sources for GEB.

        Args:
            irrigation_sources: A dictionary mapping irrigation source names to their corresponding IDs.
        """
        self.set_params(irrigation_sources, name="agents/farmers/irrigation_sources")

    @build_method(depends_on=["set_time_range", "setup_economic_data"], required=False)
    def setup_irrigation_prices_by_reference_year(
        self,
        operation_surface: float,
        operation_sprinkler: float,
        operation_drip: float,
        capital_cost_surface: float,
        capital_cost_sprinkler: float,
        capital_cost_drip: float,
        reference_year: int,
    ) -> None:
        """Sets up the well prices and upkeep prices for the hydrological model based on a reference year.

        Args:
            operation_surface: The operation cost for surface irrigation in the reference year.
            operation_sprinkler: The operation cost for sprinkler irrigation in the reference year.
            operation_drip: The operation cost for drip irrigation in the reference year.
            capital_cost_surface: The capital cost for surface irrigation in the reference year.
            capital_cost_sprinkler: The capital cost for sprinkler irrigation in the reference year.
            capital_cost_drip: The capital cost for drip irrigation in the reference year.
            reference_year: The reference year for the well prices and upkeep prices.

        Notes:
            This method sets up the well prices and upkeep prices for the hydrological model based on a reference year. It first
            retrieves the inflation rates data from the `socioeconomics/inflation_rates` dictionary. It then creates dictionaries to
            store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
            data dimension.

            The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
            resulting prices are stored in the dictionaries with the region ID as the key.

            The resulting well prices and upkeep prices data are set as dictionary with names of the form
            'socioeconomics/well_prices' and 'socioeconomics/upkeep_prices_well_per_m2', respectively.
        """
        # Retrieve the inflation rates data
        inflation_rates = self.params["socioeconomics/inflation_rates"]
        regions = list(inflation_rates["data"].keys())

        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "operation_cost_surface": operation_surface,
            "operation_cost_sprinkler": operation_sprinkler,
            "operation_cost_drip": operation_drip,
            "capital_cost_surface": capital_cost_surface,
            "capital_cost_sprinkler": capital_cost_sprinkler,
            "capital_cost_drip": capital_cost_drip,
        }

        start_year = self.start_date.year
        end_year = self.end_date.year

        # Iterate over each price type and calculate the prices across years for each region
        for price_type, initial_price in price_types.items():
            prices_dict = {"time": list(range(start_year, end_year + 1)), "data": {}}

            for region in regions:
                prices = pd.Series(index=range(start_year, end_year + 1))
                prices.loc[reference_year] = initial_price

                # Forward calculation from the reference year
                for year in range(reference_year + 1, end_year + 1):
                    prices.loc[year] = (
                        prices[year - 1]
                        * inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year))
                        ]
                    )
                # Backward calculation from the reference year
                for year in range(reference_year - 1, start_year - 1, -1):
                    prices.loc[year] = (
                        prices[year + 1]
                        / inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year + 1))
                        ]
                    )

                prices_dict["data"][region] = prices.tolist()

            # Set the calculated prices in the appropriate dictionary
            self.set_params(prices_dict, name=f"socioeconomics/{price_type}")

    @build_method(depends_on=["setup_economic_data"], required=True)
    def setup_well_prices_by_reference_year_global(
        self,
        WHY_10: float,
        WHY_20: float,
        WHY_30: float,
        reference_year: int,
    ) -> None:
        """Sets up the well prices and upkeep prices for the hydrological model based on a reference year.

        Args:
            WHY_10: the price for a well in WHY_10 in the reference year.
            WHY_20: the price for a well in WHY_20 in the reference year.
            WHY_30: the price for a well in WHY_30 in the reference year.
            reference_year: The reference year for the well prices and upkeep prices.

        Notes:
            This method sets up the well prices and upkeep prices for the hydrological model based on a reference year. It first
            retrieves the inflation rates data from the `socioeconomics/inflation_rates` dictionary. It then creates dictionaries to
            store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
            data dimension.

            The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
            resulting prices are stored in the dictionaries with the region ID as the key.

            The resulting well prices and upkeep prices data are set as dictionary with names of the form
            'socioeconomics/well_prices' and 'socioeconomics/upkeep_prices_well_per_m2', respectively.
        """
        # Retrieve the inflation rates data
        inflation_rates = self.params["socioeconomics/inflation_rates"]
        price_ratio = self.params["socioeconomics/price_ratio"]

        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "why_10": WHY_10,
            "why_20": WHY_20,
            "why_30": WHY_30,
        }

        start_year = self.start_date.year
        end_year = self.end_date.year

        # Iterate over each price type and calculate the prices across years for each region
        for price_type, initial_price in price_types.items():
            prices_dict = {
                "time": list(range(start_year, end_year + 1)),
            }
            prices_dict_data: dict[str, list] = {}

            for _, region in self.geom["regions"].iterrows():
                region_id = str(region["region_id"])

                prices: pd.Series = pd.Series(index=range(start_year, end_year + 1))
                price_ratio_region_year = price_ratio["data"][region_id][
                    price_ratio["time"].index(str(reference_year))
                ]

                prices.loc[reference_year] = price_ratio_region_year * initial_price

                # Forward calculation from the reference year
                for year in range(reference_year + 1, end_year + 1):
                    prices.loc[year] = (
                        prices[year - 1]
                        * inflation_rates["data"][region_id][
                            inflation_rates["time"].index(str(year))
                        ]
                    )

                # Backward calculation from the reference year
                for year in range(reference_year - 1, start_year - 1, -1):
                    prices.loc[year] = (
                        prices[year + 1]
                        / inflation_rates["data"][region_id][
                            inflation_rates["time"].index(str(year + 1))
                        ]
                    )

                prices_dict_data[region_id] = prices.tolist()

            prices_dict["data"] = prices_dict_data

            # Set the calculated prices in the appropriate dictionary
            self.set_params(prices_dict, name=f"socioeconomics/{price_type}")

        electricity_rates = self.data_catalog.fetch("gcam_electricity_rates").read()

        electricity_rates_dict = {
            "time": list(range(start_year, end_year + 1)),
        }
        electricity_rates_dict_data: dict[str, list] = {}

        for _, region in self.geom["regions"].iterrows():
            region_id = str(region["region_id"])
            prices = pd.Series(index=range(start_year, end_year + 1))
            country = region["ISO3"]

            # implement donors
            if country not in electricity_rates:
                countries_with_data = list(electricity_rates.keys())
                donor_countries = setup_donor_countries(
                    self.data_catalog,
                    self.geom["global_countries"],
                    countries_with_data,
                    alternative_countries=self.geom["regions"]["ISO3"]
                    .unique()
                    .tolist(),
                )
                donor_country = donor_countries.get(country, None)
                self.logger.info(
                    f"Missing electricity rates for {region['ISO3']}, using donor country {donor_country}"
                )
                country = donor_country

            prices.loc[reference_year] = electricity_rates[
                country
            ]  # use country or donor country

            # Forward calculation from the reference year
            for year in range(reference_year + 1, end_year + 1):
                prices.loc[year] = (
                    prices[year - 1]
                    * inflation_rates["data"][region_id][
                        inflation_rates["time"].index(str(year))
                    ]
                )

            # Backward calculation from the reference year
            for year in range(reference_year - 1, start_year - 1, -1):
                prices.loc[year] = (
                    prices[year + 1]
                    / inflation_rates["data"][region_id][
                        inflation_rates["time"].index(str(year + 1))
                    ]
                )

            electricity_rates_dict_data[region_id] = prices.tolist()

        electricity_rates_dict["data"] = electricity_rates_dict_data

        # Set the calculated prices in the appropriate dictionary
        self.set_params(electricity_rates_dict, name="socioeconomics/electricity_cost")

    def setup_drip_irrigation_prices_by_reference_year(
        self,
        drip_irrigation_price: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ) -> None:
        """Sets up the drip_irrigation prices and upkeep prices for the hydrological model based on a reference year.

        Args:
            drip_irrigation_price: The price of a drip_irrigation in the reference year.

            reference_year: The reference year for the drip_irrigation prices and upkeep prices.
            start_year: The start year for the drip_irrigation prices and upkeep prices.
            end_year: The end year for the drip_irrigation prices and upkeep prices.

        Notes:
            The drip_irrigation prices are calculated by applying the inflation rates to the reference year prices. The
            resulting prices are stored in the dictionaries with the region ID as the key.
        """
        # Retrieve the inflation rates data
        inflation_rates = self.params["socioeconomics/inflation_rates"]
        regions = list(inflation_rates["data"].keys())

        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "drip_irrigation_price": drip_irrigation_price,
        }

        # Iterate over each price type and calculate the prices across years for each region
        for price_type, initial_price in price_types.items():
            prices_dict: dict[str, Any] = {
                "time": list(range(start_year, end_year + 1)),
                "data": {},
            }

            for region in regions:
                prices = pd.Series(index=range(start_year, end_year + 1))
                prices.loc[reference_year] = initial_price

                # Forward calculation from the reference year
                for year in range(reference_year + 1, end_year + 1):
                    prices.loc[year] = (
                        prices[year - 1]
                        * inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year))
                        ]
                    )
                # Backward calculation from the reference year
                for year in range(reference_year - 1, start_year - 1, -1):
                    prices.loc[year] = (
                        prices[year + 1]
                        / inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year + 1))
                        ]
                    )

                prices_dict["data"][region] = prices.tolist()

            # Set the calculated prices in the appropriate dictionary
            self.set_params(prices_dict, name=f"socioeconomics/{price_type}")

    def set_farmers_and_create_farms(self, farmers: pd.DataFrame) -> None:
        """Sets up the farmers data for GEB.

        Args:
            farmers: A DataFrame containing the farmer data.
        """
        regions: gpd.GeoDataFrame = self.geom["regions"]
        region_ids: xr.DataArray = self.subgrid["region_ids"]
        cultivated_land: xr.DataArray = self.subgrid["landsurface/cultivated_land"]

        farms: xr.DataArray = self.full_like(
            region_ids, fill_value=-1, nodata=-1, dtype=np.int32
        )
        for region_id in tqdm(regions["region_id"]):
            region: xr.DataArray = region_ids == region_id
            region_clip, bounds = clip_with_grid(region, region)

            cultivated_land_region: xr.DataArray = cultivated_land.isel(bounds)
            cultivated_land_region: xr.DataArray = xr.where(
                region_clip, x=cultivated_land_region, y=False, keep_attrs=True
            )
            cultivated_land_region_values: TwoDArrayBool = cultivated_land_region.values

            farmers_region: pd.DataFrame = farmers[farmers["region_id"] == region_id]
            farms_region: TwoDArrayInt32 = create_farms(
                farmers_region,
                cultivated_land_region_values,
                farm_size_key="farm_size_cells",
            )
            assert (
                farms_region.min() >= -1
            )  # -1 is nodata value, all farms should be positive
            farms[bounds] = xr.where(
                region_clip, farms_region, farms.isel(bounds), keep_attrs=True
            )
            farms: xr.DataArray = farms.compute()

        farmers: pd.DataFrame = farmers.drop("farm_size_cells", axis=1)

        assert farms.min() >= -1  # -1 is nodata value, all farms should be positive
        assert farms.max().item() == len(farmers) - 1

        self.set_subgrid(farms, name="agents/farmers/farms")
        self.set_array(farmers["region_id"].values, name="agents/farmers/region_id")

    @build_method(
        depends_on=["setup_regions_and_land_use", "setup_cell_area"], required=True
    )
    def setup_create_farms(
        self,
        region_id_column: str = "region_id",
        country_iso3_column: str = "ISO3",
        data_source: Literal["lowder"] = "lowder",
        size_class_boundaries: dict[str, tuple[int | float, int | float]] | None = None,
    ) -> None:
        """Sets up the farmers for GEB.

        This method sets up the farmers for GEB. This is a simplified method that generates an example set of agent data.
        It first calculates the number of farmers and their farm sizes for each region based on the agricultural data for
        that region based on the amount of farm land and data from a global database on farm sizes per country. It then
        randomly assigns crops, irrigation sources, household sizes, and daily incomes and consumption levels to each farmer.

        A paper that reports risk aversion values for 75 countries is this one: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2646134

        Args:
            region_id_column: The name of the column in the region database that contains the region IDs. Default is 'UID'.
            country_iso3_column: The name of the column in the region database that contains the country ISO3 codes. Default is 'ISO3'.
            data_source: The source of the farm size data. Default is 'lowder', which uses the Lowder et al. (2016) dataset.
            size_class_boundaries: The boundaries for the size classes of farms. For the Lowder et al. (2016) dataset, this must be None
                because the boundaries are defined in the dataset itself.
        """
        assert data_source == "lowder", (
            "Currently, only the Lowder et al. (2016) dataset is supported as data source for farm sizes"
        )
        assert size_class_boundaries is None, (
            "size_class_boundaries must be None when using Lowder et al. (2016) dataset"
        )
        size_class_boundaries = {
            "< 1 Ha": (0, 10000),
            "1 - 2 Ha": (10000, 20000),
            "2 - 5 Ha": (20000, 50000),
            "5 - 10 Ha": (50000, 100000),
            "10 - 20 Ha": (100000, 200000),
            "20 - 50 Ha": (200000, 500000),
            "50 - 100 Ha": (500000, 1000000),
            "100 - 200 Ha": (1000000, 2000000),
            "200 - 500 Ha": (2000000, 5000000),
            "500 - 1000 Ha": (5000000, 10000000),
            "> 1000 Ha": (10000000, np.inf),
        }

        cultivated_land: xr.DataArray = self.subgrid[
            "landsurface/cultivated_land"
        ].compute()
        assert cultivated_land.dtype == bool, "Cultivated land must be boolean"
        region_ids: xr.DataArray = self.subgrid["region_ids"].compute()
        cell_area: xr.DataArray = self.subgrid["cell_area"].compute()

        regions_shapes = self.geom["regions"]
        assert country_iso3_column in regions_shapes.columns, (
            f"Region database must contain {country_iso3_column} column"
        )

        farm_sizes_per_region = self.data_catalog.fetch(
            "lowder_farm_size_distribution"
        ).read()

        # Remove countries with average farm size below subgrid resolution in smallest class
        subgrid_cell_area_ha: float = (1e6 / (self.subgrid_factor**2)) / 1e4
        subgrid_cell_area_ha *= 0.95  # substract 5 %, just so that Lithuania stays inside the dataset. No problems there, as the subgrid is not exactly 1km2 (but smaller due to high latitude)
        countries_to_remove: list[str] = []

        for iso3, country_data in farm_sizes_per_region.groupby(
            "ISO3"
        ):  # start removal of countries with small farms
            holdings = country_data[
                country_data["Holdings/ agricultural area"] == "Holdings"
            ]
            area = country_data[
                country_data["Holdings/ agricultural area"] == "Agricultural area (Ha)"
            ]

            if len(holdings) == 1 and len(area) == 1:
                n_holdings = (
                    holdings["< 1 Ha"].replace("..", np.nan).astype(np.float64).iloc[0]
                )
                area_ha = (
                    area["< 1 Ha"].replace("..", np.nan).astype(np.float64).iloc[0]
                )

                if pd.notna(n_holdings) and n_holdings > 0 and pd.notna(area_ha):
                    if (area_ha / n_holdings) < subgrid_cell_area_ha:
                        countries_to_remove.append(iso3)

        if countries_to_remove:
            self.logger.warning(
                f"Removed {len(countries_to_remove)} countries with avg farm size < {subgrid_cell_area_ha:.2f} ha: {countries_to_remove}"
            )
            farm_sizes_per_region = farm_sizes_per_region[
                ~farm_sizes_per_region["ISO3"].isin(countries_to_remove)
            ]

        farm_countries_list = list(farm_sizes_per_region["ISO3"].unique())
        farm_size_donor_country = setup_donor_countries(
            self.data_catalog,
            self.geom["global_countries"],
            farm_countries_list,
            alternative_countries=self.geom["regions"]["ISO3"].unique().tolist(),
        )

        all_agents = []

        self.logger.info(f"Starting processing of {len(regions_shapes)} regions")

        for i, (_, region) in enumerate(regions_shapes.iterrows()):
            UID = region[region_id_column]
            ISO3 = region[country_iso3_column]
            self.logger.info(
                f"Processing region ({i + 1}/{len(regions_shapes)}) with ISO3 {ISO3}"
            )

            if ISO3 in farm_size_donor_country.keys():
                ISO3 = farm_size_donor_country[ISO3]
                self.logger.info(
                    f"Missing farm sizes for {region[country_iso3_column]}, using donor country {ISO3}"
                )
            cultivated_land_region_total_cells = (
                ((region_ids == UID) & (cultivated_land)).sum().compute()
            ).item()

            # in the later corrections, it is important that the total cultivated land is
            # quite precise, so we first convert to float64 before summing
            cultivated_land_area_region_m2: np.float64 = (
                (((region_ids == UID) & (cultivated_land)) * cell_area)
                .astype(np.float64)
                .sum()
                .compute()
                .item()
            )
            if (
                cultivated_land_area_region_m2 == 0
            ):  # when no agricultural area, just continue as there will be no farmers. Also avoiding some division by 0 errors.
                continue

            # in later corrections, it is important that the average subgrid area is quite precise,
            # so we first convert to float64 before calculating the mean
            average_subgrid_area_region: np.float64 = (
                cell_area.where(((region_ids == UID) & (cultivated_land)))
                .astype(np.float64)
                .mean()
                .compute()
                .item()
            )

            region_farm_sizes = farm_sizes_per_region.loc[
                (farm_sizes_per_region["ISO3"] == ISO3)
            ].drop(["Country", "Census Year", "Total"], axis=1)
            assert len(region_farm_sizes) == 2, (
                f"Found {len(region_farm_sizes) / 2} region_farm_sizes for {ISO3}"
            )
            region_agents: pd.DataFrame = create_farm_distributions(
                region_farm_sizes,
                size_class_boundaries,
                cultivated_land_area_region_m2,
                average_subgrid_area_region,
                cultivated_land_region_total_cells,
                UID,
                ISO3,
                self.logger,
            )

            all_agents.append(region_agents)

        farmers = pd.concat(all_agents, ignore_index=True)
        self.set_farmers_and_create_farms(farmers)

    def canon(self, string_to_normalize: str) -> str:
        """Canonicalizes a string by normalizing it to ASCII and stripping whitespace.

        Args:
            string_to_normalize: The string to canonicalize.
        Returns:
            The canonicalized string.
        """
        return (
            unicodedata.normalize("NFKD", string_to_normalize)
            .encode("ascii", "ignore")
            .decode("ascii")
            .strip()
        )

    def setup_building_reconstruction_costs(
        self, buildings: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Assigns reconstruction costs (in USD) to buildings based on the global exposure model.

        Args:
            buildings: A GeoDataFrame containing building data within the model domain.
        Returns:
            A GeoDataFrame with reconstruction costs assigned to each building.
        Raises:
            ValueError: If a region in GADM level 1 is not found in the global exposure model or
                        if some buildings do not have reconstruction costs assigned.
        """
        # load GADM level 1 within model domain (older version for compatibility with global exposure model)
        gadm_level1 = self.data_catalog.fetch("gadm_28").read(
            geom=self.region.union_all().buffer(0.1),
        )
        countries_in_model = gadm_level1["NAME_0"].unique().tolist()

        global_exposure_model = self.data_catalog.fetch(
            "global_exposure_model",
            countries=countries_in_model,
        ).read()

        # append the NAME_1 column to the buildings
        buildings["NAME_1"] = gpd.sjoin(
            buildings,
            gadm_level1[["NAME_1", "geometry"]],
            how="left",
            predicate="within",
        )["NAME_1"].values

        # assert each building has a NAME_1 value
        if buildings["NAME_1"].isnull().any():
            # For buildings without NAME_1 we assign them to the nearest GADM level 1 region.
            # This happens when buildings are just outside the polygons (e.g., near coastlines).
            # Use a spatial index to avoid an O(n*m) distance calculation over all regions.
            buildings_no_name1 = buildings[buildings["NAME_1"].isnull()]
            if not buildings_no_name1.empty:
                # Precompute centroids for unmatched buildings
                # because the buildings are very small, we can ignore the warning about calculations
                # of centroids in a geographic coordinate system
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    unmatched_centroids = buildings_no_name1.geometry.centroid

                # Build spatial index over GADM level 1 geometries once
                gadm_sindex = gadm_level1.sindex
                for building_idx, centroid in zip(
                    buildings_no_name1.index, unmatched_centroids
                ):
                    # Query nearest polygon via its bounding box to limit candidate search
                    nearest_pos = gadm_sindex.nearest(centroid)[0]
                    nearest_region = gadm_level1.iloc[nearest_pos]
                    buildings.at[building_idx, "NAME_1"] = nearest_region[
                        "NAME_1"
                    ].values[0]

        # Iterate over unique admin-1 region names to avoid redundant checks and assignments
        buildings["NAME_1"] = buildings["NAME_1"].apply(self.canon)
        for name_1 in gadm_level1["NAME_1"].dropna().unique():
            # clean up name
            name_1 = self.canon(name_1)
            # check if region is in global exposure model
            if name_1 not in global_exposure_model:
                raise ValueError(
                    f"Region {name_1} not found in global exposure model. Please check if the region name has changed."
                )
            exposure_model_region = global_exposure_model[name_1]
            for reconstruction_type in exposure_model_region:
                buildings.loc[buildings["NAME_1"] == name_1, reconstruction_type] = (
                    float(exposure_model_region[reconstruction_type])
                )
        # assert all buildings have reconstruction costs assigned (i.e., no null values in the reconstruction cost columns)
        reconstruction_cost_columns = list(exposure_model_region.keys())
        if buildings[reconstruction_cost_columns].isnull().any().any():
            # get NAME_1 values for buildings with null reconstruction costs
            buildings_with_null_costs = buildings[
                buildings[reconstruction_cost_columns].isnull().any(axis=1)
            ]
            missing_name_1_values = buildings_with_null_costs["NAME_1"].unique()
            raise ValueError(
                f"Some buildings with NAME_1 values {missing_name_1_values} do not have reconstruction costs assigned. Please check the global exposure model and the region names."
            )
        # rename columns to match expected names in model
        buildings = buildings.rename(
            columns={
                "COST_STRUCTURAL_USD_SQM": "maximum_damage_structure",
                "COST_CONTENTS_USD_SQM": "maximum_damage_content",
            }
        )

        return buildings

    @build_method(required=True)
    def setup_buildings(self) -> None:
        """Gets buildings per GDL region within the model domain and assigns grid indices from GLOPOP-S grid."""
        # load region mask
        mask = self.region.union_all()
        buildings = self.data_catalog.fetch(
            "open_building_map",
            geom=mask,
            prefix="assets",
        ).read()
        buildings = self.setup_building_reconstruction_costs(buildings)

        # reset id column to avoid issues with duplicate ids
        buildings["id"] = np.arange(len(buildings))

        # write to disk
        self.set_geom(buildings, name="assets/open_building_map")

    @build_method(required=True)
    def setup_local_damage_model(
        self,
    ) -> None:
        """Sets up damage parameters for different hazards and asset types."""
        parameters = self.data_catalog.fetch("local_damage_model").read()
        for hazard, hazard_parameters in parameters.items():
            for asset_type, asset_parameters in hazard_parameters.items():
                for component, asset_components in asset_parameters.items():
                    curve = pd.DataFrame(
                        asset_components["curve"],
                        columns=np.array(["severity", "damage_ratio"]),
                    )

                    self.set_table(
                        curve,
                        name=f"damage_model/local/{hazard}/{asset_type}/{component}/curve",
                    )

                    maximum_damage = {
                        "maximum_damage": asset_components["maximum_damage"]
                    }

                    self.set_params(
                        maximum_damage,
                        name=f"damage_model/local/{hazard}/{asset_type}/{component}/maximum_damage",
                    )

    @build_method(required=True)
    def setup_global_damage_model(self, region: str = "global") -> None:
        """This method sets up the damage functions for flood events for the specified region.

        It retrieves the damage functions from the data catalog, processes them, and saves them as
        parquet files for use in the model.

        Args:
            region: The region for which to set up the damage functions. Default is 'global'; the
                accepted region identifiers are determined by the underlying 'global_damage_model'
                dataset.
        """
        damage_functions = self.data_catalog.fetch("global_damage_model").read(
            region=region
        )
        # save the cleaned dataframe as parquet
        for damage_class, df_damage_class in damage_functions.items():
            self.set_table(
                df_damage_class,
                name=f"damage_model/global/flood/{damage_class}",
            )

    def assign_buildings_to_grid_cells(
        self, GDL_regions: gpd.GeoDataFrame
    ) -> dict[str, gpd.GeoDataFrame]:
        """Assigns buildings to grid cells from GLOPOP-S grid for each GDL region.

        Args:
            GDL_regions: A GeoDataFrame containing GDL regions within the model domain.
        Returns:
            A dictionary with GDLcode as keys and GeoDataFrames of buildings with grid indices as values.
        """
        output = {}
        buildings = self.geom["assets/open_building_map"]

        # Vectorized centroid extraction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            centroids = buildings.geometry.centroid
        buildings["lon"] = centroids.x
        buildings["lat"] = centroids.y

        for _, GDL_region in GDL_regions.iterrows():
            _, GLOPOP_GRID_region = self.data_catalog.fetch(
                "glopop-sg", region=GDL_region["GDLcode"]
            ).read(GDL_region["GDLcode"])
            GLOPOP_GRID_region = GLOPOP_GRID_region.rio.clip_box(*self.bounds)

            # subset buildings to those within the GLOPOP_GRID_region
            buildings_gdl = buildings.cx[
                GLOPOP_GRID_region.x.min() : GLOPOP_GRID_region.x.max(),
                GLOPOP_GRID_region.y.min() : GLOPOP_GRID_region.y.max(),
            ]

            # Vectorized assignment of grid cells
            cells = GLOPOP_GRID_region.sel(
                x=xr.DataArray(buildings_gdl["lon"].values, dims="points"),
                y=xr.DataArray(buildings_gdl["lat"].values, dims="points"),
                method="nearest",
            )

            buildings_gdl["grid_idx"] = cells.values[0]
            # drop buildings without grid_idx
            buildings_gdl = buildings_gdl[buildings_gdl["grid_idx"] != 0]

            gdl_name = GDL_region["GDLcode"]
            output[gdl_name] = buildings_gdl
        return output

    @build_method(
        depends_on=[
            "setup_assets",
            "setup_buildings",
            "setup_income_distribution_parameters",
        ],
        required=True,
    )
    def setup_household_characteristics(
        self,
        maximum_age: int = 85,
        skip_countries_ISO3: list[str] = [],
        single_household_per_building: bool = False,
        redundancy_array_size: int = 20_000_000,
    ) -> None:
        """New method to set up household characteristics for agents using GLOPOP-S data. This method is still under development and may not be fully functional.

        Args:
            maximum_age: The maximum age for the head of household. Default is 85.
            skip_countries_ISO3: A list of ISO3 country codes to skip when setting up household characteristics.
            single_household_per_building: If True, only one household will be allocated per building. Default is False.
            redundancy_array_size: The size of the redundancy array used for preallocating region arrays of household characteristics. Default is 20 million, which should be sufficient for most regions. Adjust if you encounter memory issues or if you have very large regions.
        Raises:
            ValueError: If there are more buildings in the GDL region than the specified redundancy_array_size, which is used for preallocating arrays of household characteristics. In this case, consider increasing the redundancy_array_size parameter.
        """
        # create list of attibutes to include (and include name to store to)
        rename = {
            "HHSIZE_CAT": "household_type",
            "HHSIZE": "size",
            "AGE_HH_HEAD": "age_household_head",
            "EDUC": "education_level",
            "WEALTH_INDEX": "wealth_index",
            "RURAL": "rural",
        }
        region_results = {}

        # create income percentile based on wealth index mapping
        wealth_index_to_income_percentile = {
            1: (1, 19),
            2: (20, 39),
            3: (40, 59),
            4: (60, 79),
            5: (80, 100),
        }

        # get age class to age (head of household) mapping
        age_class_to_age = {
            1: (0, 4),
            2: (5, 14),
            3: (15, 24),
            4: (25, 34),
            5: (35, 44),
            6: (45, 54),
            7: (55, 64),
            8: (65, maximum_age + 1),
        }

        # load table with income distribution data
        national_income_distribution = self.table["income/national_distribution"]

        # load GDL region within model domain
        GDL_regions = self.data_catalog.fetch("GDL_regions_v4").read(
            geom=self.region.union_all(), columns=["GDLcode", "iso_code", "geometry"]
        )

        # setup buildings in region for household allocation
        all_buildings_model_region = self.assign_buildings_to_grid_cells(GDL_regions)

        # collect household characteristics for all regions; initialized once to avoid
        # overwriting results for earlier regions during the loop
        household_characteristics_region = {}  # type: dict[str, Any]

        for GDL_code in all_buildings_model_region:
            self.logger.info(f"Setting up household characteristics for {GDL_code}...")
            if GDL_code[:3] in skip_countries_ISO3:
                self.logger.info(f"Skipping {GDL_code[:3]}")

            buildings = all_buildings_model_region[GDL_code]
            # filter to residential buildings
            # check if occupancy column contains RES or UNK string (unknown occupancy assumed residential)
            residential_buildings_model_region = buildings[
                buildings["occupancy"].str.contains("RES|UNK", na=False)
            ]
            if residential_buildings_model_region.empty:
                self.logger.info(
                    f"No residential buildings found for GDL code: {GDL_code}"
                )
                continue

            GLOPOP_S_region, GLOPOP_GRID_region = self.data_catalog.fetch(
                "glopop-sg", region=GDL_code
            ).read(GDL_code)

            # get size of household
            HH_SIZE = GLOPOP_S_region["HID"].value_counts()

            # only select household heads
            GLOPOP_S_region = GLOPOP_S_region[GLOPOP_S_region["RELATE_HEAD"] == 1]

            # add household sizes to household df
            GLOPOP_S_region = GLOPOP_S_region.merge(HH_SIZE, on="HID", how="left")
            GLOPOP_S_region = GLOPOP_S_region.rename(
                columns={"count": "HHSIZE"}
            ).reset_index(drop=True)

            # rename
            GLOPOP_S_region = GLOPOP_S_region.rename(columns=rename)

            # clip grid to model bounds
            GLOPOP_GRID_region = GLOPOP_GRID_region.rio.clip_box(*self.bounds)

            # get unique cells in grid
            unique_grid_cells = np.unique(GLOPOP_GRID_region.values)

            # subset GLOPOP_households_region
            GLOPOP_S_region = GLOPOP_S_region[
                GLOPOP_S_region["GRID_CELL"].isin(unique_grid_cells)
            ]

            # create column WEALTH_INDEX (GLOPOP-S contains either INCOME or WEALTH data, depending on the region. Therefore, we combine these.)
            GLOPOP_S_region["wealth_index"] = (
                GLOPOP_S_region["WEALTH"] + GLOPOP_S_region["INCOME"] + 1
            )

            # sample income percentile
            GLOPOP_S_region["income_percentile"] = np.uint16(np.iinfo(np.uint16).max)
            for wealth_index in wealth_index_to_income_percentile:
                percentile_range = wealth_index_to_income_percentile[wealth_index]

                GLOPOP_S_region.loc[
                    GLOPOP_S_region["wealth_index"] == wealth_index,
                    "income_percentile",
                ] = np.random.randint(
                    percentile_range[0],
                    percentile_range[1],
                    size=len(
                        GLOPOP_S_region.loc[
                            GLOPOP_S_region["wealth_index"] == wealth_index
                        ]
                    ),
                ).astype(np.uint16)
            assert not (
                GLOPOP_S_region["income_percentile"] == np.iinfo(np.uint16).max
            ).any()

            # sample income from national distribution
            GLOPOP_S_region["disp_income"] = np.percentile(
                np.array(national_income_distribution[GDL_code[:3]]),
                np.array(GLOPOP_S_region["income_percentile"]),
            )

            GLOPOP_S_region["age_household_head"] = np.uint16(np.iinfo(np.uint16).max)
            for age_class in age_class_to_age:
                age_range = age_class_to_age[age_class]

                GLOPOP_S_region.loc[
                    GLOPOP_S_region["AGE"] == age_class, "age_household_head"
                ] = np.random.randint(
                    age_range[0],
                    age_range[1],
                    size=len(GLOPOP_S_region.loc[GLOPOP_S_region["AGE"] == age_class]),
                ).astype(np.uint16)
            assert not (
                GLOPOP_S_region["age_household_head"] == np.iinfo(np.uint16).max
            ).any()

            # pre-group households by grid cell
            households_by_cell = GLOPOP_S_region.groupby("GRID_CELL")["HID"].unique()

            # pre-group buildings by grid cell
            buildings_by_cell = residential_buildings_model_region.groupby("grid_idx")

            n_agents_allocated = 0
            household_ids = np.full(int(redundancy_array_size), -1, dtype=np.int32)
            building_ids = np.full(int(redundancy_array_size), -1, dtype=np.int32)

            for grid_cell, households_in_cell in households_by_cell.items():
                if grid_cell not in buildings_by_cell.groups:
                    continue

                buildings_in_cell = buildings_by_cell.get_group(grid_cell)
                n_buildings = buildings_in_cell.shape[0]

                sampled_households = np.random.choice(
                    households_in_cell,
                    size=n_buildings
                    if single_household_per_building
                    else np.max([n_buildings, households_in_cell.size]),
                    replace=households_in_cell.size < n_buildings,
                )

                end = n_agents_allocated + sampled_households.size

                household_ids[n_agents_allocated:end] = sampled_households
                building_ids_in_cell = buildings_in_cell["id"].values

                if sampled_households.size <= n_buildings:
                    # When the number of sampled households does not exceed the
                    # number of buildings, we can sample buildings without
                    # replacement while still allowing some buildings to host
                    # zero households, matching the previous behaviour.
                    building_ids_sampled = np.random.choice(
                        building_ids_in_cell,
                        size=sampled_households.size,
                        replace=False,
                    )
                else:
                    # When more households than buildings are allocated in a cell,
                    # first assign one household to every building (ensuring that
                    # no building is left without households), then distribute the
                    # remaining households across buildings with replacement.
                    first_building_ids = np.random.permutation(building_ids_in_cell)
                    n_remaining = sampled_households.size - n_buildings
                    additional_building_ids = np.random.choice(
                        building_ids_in_cell,
                        size=n_remaining,
                        replace=True,
                    )
                    building_ids_sampled = np.concatenate(
                        [first_building_ids, additional_building_ids]
                    )

                building_ids[n_agents_allocated:end] = building_ids_sampled
                n_agents_allocated = end
                if end > redundancy_array_size:
                    raise ValueError(
                        "Number of buildings in region exceeds redundancy array size, consider increasing redundancy_array_size parameter."
                    )

            household_ids = household_ids[:n_agents_allocated]
            building_ids = building_ids[:n_agents_allocated]
            # set locations
            locations = (
                residential_buildings_model_region.set_index("id")
                .loc[building_ids][["lon", "lat"]]
                .values
            )
            region_ids = sample_from_map(
                self.subgrid["region_ids"].values,
                locations,
                self.subgrid["region_ids"].rio.transform(recalc=True).to_gdal(),
            )

            # subset to only include households with a region (some buildings are located outside land masks)
            households_with_region = np.where(region_ids != -1)[0]
            household_ids = household_ids[households_with_region]
            building_ids = building_ids[households_with_region]
            region_ids = region_ids[households_with_region]

            # now fill the household attributes
            household_characteristics = {}
            GLOPOP_S_region = GLOPOP_S_region.set_index("HID", drop=True)
            for column in (
                "household_type",
                "age_household_head",
                "education_level",
                "wealth_index",
                "rural",
                "disp_income",
                "income_percentile",
                "size",
            ):
                household_characteristics[column] = np.array(
                    GLOPOP_S_region.loc[household_ids][column]
                )
            household_characteristics["building_id_of_household"] = building_ids
            household_characteristics["location"] = np.round(
                locations[households_with_region].astype(np.float32),
                5,
            )
            household_characteristics["region_id"] = region_ids

            # ensure that all households have a region assigned
            assert not (household_characteristics["region_id"] == -1).any()
            household_characteristics_region[GDL_code] = household_characteristics
        # now export all household characteristics for all regions
        for characteristic in next(
            iter(household_characteristics_region.values())
        ).keys():
            array_to_store = np.concatenate(
                [
                    region_data[characteristic]
                    for region_data in household_characteristics_region.values()
                ]
            )
            self.set_array(
                array_to_store,
                name=f"agents/households/{characteristic}",
            )

    @build_method(depends_on=["setup_create_farms"], required=True)
    def setup_farmer_household_characteristics(self, maximum_age: int = 85) -> None:
        """Sets up farmer household characteristics for farmers using GLOPOP-S data.

        Args:
            maximum_age: The maximum age for the head of household. Default is 85.
        """
        n_farmers = self.array["agents/farmers/region_id"].size
        farms = self.subgrid["agents/farmers/farms"]

        # get farmer locations
        vertical_index = (
            np.arange(farms.shape[0])
            .repeat(farms.shape[1])
            .reshape(farms.shape)[farms != -1]
        )
        horizontal_index = np.tile(np.arange(farms.shape[1]), farms.shape[0]).reshape(
            farms.shape
        )[farms != -1]
        farms_flattened = farms.values[farms.values != -1]

        pixels = np.zeros((n_farmers, 2), dtype=np.int32)
        pixels[:, 0] = np.round(
            np.bincount(farms_flattened, horizontal_index)
            / np.bincount(farms_flattened)
        ).astype(int)
        pixels[:, 1] = np.round(
            np.bincount(farms_flattened, vertical_index) / np.bincount(farms_flattened)
        ).astype(int)

        locations = pixels_to_coords(
            pixels + 0.5, farms.rio.transform(recalc=True).to_gdal()
        )
        locations = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(locations[:, 0], locations[:, 1]),
            crs="EPSG:4326",
        )  # convert locations to geodataframe

        # GLOPOP-S uses the GDL regions. So we need to get the GDL region for each farmer using their location
        GDL_regions = self.data_catalog.fetch("GDL_regions_v4").read(
            geom=self.region.union_all(), columns=["GDLcode", "geometry"]
        )

        # assign GDL region to each farmer based on their location. This is a heavy operation, so we include a progress bar to monitor progress.
        self.logger.info("Assigning GDL region to each farmer based on their location.")
        chunk_size = 5000
        n_farmers = len(locations)
        chunks = []

        self.logger.info(f"Processing {n_farmers} farmers in chunks of {chunk_size}")

        for i in tqdm(range(0, n_farmers, chunk_size), desc="Processing farmer chunks"):
            chunk_locations = locations.iloc[i : i + chunk_size]
            chunk_result = gpd.sjoin_nearest(chunk_locations, GDL_regions, how="left")
            chunks.append(chunk_result)

        GDL_region_per_farmer = pd.concat(chunks, ignore_index=True)

        # ensure that each farmer has a region
        assert GDL_region_per_farmer["GDLcode"].notna().all()

        # Get list of unique GDL codes from farmer dataframe
        attributes_to_include = ["HHSIZE_CAT", "AGE", "EDUC", "WEALTH"]

        for column in attributes_to_include:
            GDL_region_per_farmer[column] = np.full(
                len(GDL_region_per_farmer), -1, dtype=np.int32
            )

        for GDL_idx, (GDL_region, farmers_GDL_region) in enumerate(
            GDL_region_per_farmer.groupby("GDLcode")
        ):
            self.logger.info(
                f"Setting up farmer household characteristics for {GDL_region} ({GDL_idx + 1}/{len(GDL_regions)})"
            )
            GLOPOP_S_region, _ = self.data_catalog.fetch(
                "glopop-sg", region=GDL_region
            ).read(GDL_region)

            # select farmers only
            GLOPOP_S_region = GLOPOP_S_region[GLOPOP_S_region["RURAL"] == 1].drop(
                "RURAL", axis=1
            )

            # shuffle GLOPOP-S data to avoid biases in that regard
            GLOPOP_S_household_IDs = GLOPOP_S_region["HID"].unique()
            np.random.shuffle(GLOPOP_S_household_IDs)  # shuffle array in-place
            GLOPOP_S_region = (
                GLOPOP_S_region.set_index("HID")
                .loc[GLOPOP_S_household_IDs]
                .reset_index()
            )

            # Select a sample of farmers from the database. Because the households were
            # shuflled there is no need to pick random households, we can just take the first n_farmers.
            # If there are not enough farmers in the region, we need to upsample the data. In this case
            # we will just take the same farmers multiple times starting from the top.
            GLOPOP_S_household_IDs = GLOPOP_S_region["HID"].values

            # first we mask out all consecutive duplicates
            mask = np.concatenate(
                ([True], GLOPOP_S_household_IDs[1:] != GLOPOP_S_household_IDs[:-1])
            )
            GLOPOP_S_household_IDs = GLOPOP_S_household_IDs[mask]

            GLOPOP_S_region_sampled = []
            if GLOPOP_S_household_IDs.size < len(farmers_GDL_region):
                n_repetitions = len(farmers_GDL_region) // GLOPOP_S_household_IDs.size
                max_household_ID = GLOPOP_S_household_IDs.max()
                for i in range(n_repetitions):
                    GLOPOP_S_region_copy = GLOPOP_S_region.copy()
                    # increase the household ID to avoid duplicate household IDs. Using (i + 1) so that the original household IDs are not changed
                    # so that they can be used in the final "topping up" below.
                    GLOPOP_S_region_copy["HID"] = GLOPOP_S_region_copy["HID"] + (
                        (i + 1) * max_household_ID
                    )
                    GLOPOP_S_region_sampled.append(GLOPOP_S_region_copy)
                requested_farmers = (
                    len(farmers_GDL_region) % GLOPOP_S_household_IDs.size
                )
            else:
                requested_farmers = len(farmers_GDL_region)

            GLOPOP_S_household_IDs = GLOPOP_S_household_IDs[:requested_farmers]
            GLOPOP_S_region_sampled.append(
                GLOPOP_S_region[GLOPOP_S_region["HID"].isin(GLOPOP_S_household_IDs)]
            )

            GLOPOP_S_region_sampled = pd.concat(
                GLOPOP_S_region_sampled, ignore_index=True
            )
            assert GLOPOP_S_region_sampled["HID"].unique().size == len(
                farmers_GDL_region
            )
            households_region = GLOPOP_S_region_sampled.groupby("HID")

            idx = GLOPOP_S_region_sampled.groupby("HID")["RELATE_HEAD"].idxmax()
            household_heads = GLOPOP_S_region_sampled.loc[idx].reset_index(drop=True)
            assert len(household_heads) == len(farmers_GDL_region)

            # # age
            # household_heads["AGE_continuous"] = np.full(
            #     len(household_heads), -1, dtype=np.int32
            # )
            age_class_to_age = {
                1: (0, 16),
                2: (16, 26),
                3: (26, 36),
                4: (36, 46),
                5: (46, 56),
                6: (56, 66),
                7: (66, maximum_age + 1),
            }  # exclusive
            for age_class, age_range in age_class_to_age.items():
                household_heads_age_class = household_heads[
                    household_heads["AGE"] == age_class
                ]
                household_heads.loc[household_heads_age_class.index, "AGE"] = (
                    np.random.randint(
                        age_range[0],
                        age_range[1],
                        size=len(household_heads_age_class),
                        dtype=GDL_region_per_farmer["AGE"].dtype,
                    )
                )
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "AGE"] = (
                household_heads["AGE"].values
            )

            # education level
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "EDUC"] = (
                household_heads["EDUC"].values
            )

            # household size
            household_sizes_region = households_region.size().values.astype(np.int32)
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "HHSIZE_CAT"] = (
                household_sizes_region
            )

        # assert none of the household sizes are placeholder value -1
        assert (GDL_region_per_farmer["HHSIZE_CAT"] != -1).all()
        assert (GDL_region_per_farmer["AGE"] != -1).all()
        assert (GDL_region_per_farmer["EDUC"] != -1).all()

        self.set_array(
            GDL_region_per_farmer["HHSIZE_CAT"].values,
            name="agents/farmers/household_size",
        )
        self.set_array(
            GDL_region_per_farmer["AGE"].values,
            name="agents/farmers/age_household_head",
        )
        self.set_array(
            GDL_region_per_farmer["EDUC"].values,
            name="agents/farmers/education_level",
        )

    def create_behavioural_parameters(self) -> pd.DataFrame:
        """Creates behavioural parameters for agents based on country-level and individual-level preferences.

        Returns:
            A DataFrame containing behavioural parameters for each country, including risk aversion and discount factors.
        """
        # Risk aversion
        preferences_country_level = self.data_catalog.fetch(
            "global_preferences_survey_country"
        ).read()[["country", "isocode", "patience", "risktaking"]]
        preferences_individual_level = (
            self.data_catalog.fetch("global_preferences_survey_individual")
            .read()[["country", "isocode", "patience", "risktaking"]]
            .dropna()
        )

        def scale_to_range(x: pd.Series, new_min: float, new_max: float) -> pd.Series:
            """Scales a pandas Series to a new range [new_min, new_max].

            Args:
                x: The pandas Series to be scaled.
                new_min: The minimum value of the new range.
                new_max: The maximum value of the new range.

            Returns:
                A pandas Series scaled to the new range.
            """
            x_min = x.min()
            x_max = x.max()
            # Avoid division by zero
            if x_max == x_min:
                return pd.Series(new_min, index=x.index)
            scaled_x = (x - x_min) / (x_max - x_min) * (new_max - new_min) + new_min
            return scaled_x

        # Create 'risktaking_losses'
        preferences_individual_level["risktaking_losses"] = scale_to_range(
            preferences_individual_level["risktaking"] * -1,
            new_min=-0.88,
            new_max=-0.02,
        )

        # Create 'risktaking_gains'
        preferences_individual_level["risktaking_gains"] = scale_to_range(
            preferences_individual_level["risktaking"] * -1,
            new_min=0.04,
            new_max=0.94,
        )

        # Create 'discount'
        preferences_individual_level["discount"] = scale_to_range(
            preferences_individual_level["patience"] * -1,
            new_min=0.15,
            new_max=0.73,
        )

        # Create 'risktaking_losses'
        preferences_country_level["risktaking_losses"] = scale_to_range(
            preferences_country_level["risktaking"] * -1,
            new_min=-0.88,
            new_max=-0.02,
        )

        # Create 'risktaking_gains'
        preferences_country_level["risktaking_gains"] = scale_to_range(
            preferences_country_level["risktaking"] * -1,
            new_min=0.04,
            new_max=0.94,
        )

        # Create 'discount'
        preferences_country_level["discount"] = scale_to_range(
            preferences_country_level["patience"] * -1,
            new_min=0.15,
            new_max=0.73,
        )

        # List of variables for which to calculate the standard deviation
        variables: list[str] = ["discount", "risktaking_gains", "risktaking_losses"]

        # Convert the variables to numeric, coercing errors to NaN to handle non-numeric entries
        for var in variables:
            preferences_individual_level[var] = pd.to_numeric(
                preferences_individual_level[var], errors="coerce"
            )

        # Group by 'isocode' and calculate the standard deviation for each variable
        std_devs = preferences_individual_level.groupby("isocode")[variables].std()

        # Add a suffix '_std' to the column names to indicate standard deviation
        std_devs = std_devs.add_suffix("_std").reset_index()

        # Merge the standard deviation data into 'preferences_country_level' on 'isocode'
        preferences_country_level = preferences_country_level.merge(
            std_devs, on="isocode", how="left"
        )

        preferences_country_level = preferences_country_level.drop(
            ["patience", "risktaking"], axis=1
        )

        return preferences_country_level

    @build_method(
        depends_on=["setup_create_farms", "setup_farmer_household_characteristics"],
        required=True,
    )
    def setup_farmer_characteristics(
        self,
        interest_rate: float = 0.05,
    ) -> None:
        """Sets up farmer characteristics including behavioural parameters.

        Args:
            interest_rate: The interest rate. Value between 0 and 1. Default is 0.05.
        """
        n_farmers = self.array["agents/farmers/region_id"].size

        preferences_global = self.create_behavioural_parameters()
        preferences_global.rename(
            columns={
                "country": "Country",
                "isocode": "ISO3",
                "risktaking_losses": "Losses",
                "risktaking_gains": "Gains",
                "discount": "Discount",
                "discount_std": "Discount_std",
                "risktaking_losses_std": "Losses_std",
                "risktaking_gains_std": "Gains_std",
            },
            inplace=True,
        )

        ISO3_codes_region: set[str] = set(self.geom["regions"]["ISO3"].unique())
        relevant_trade_regions: dict[str, str] = {
            ISO3: TRADE_REGIONS[ISO3]
            for ISO3 in ISO3_codes_region
            if ISO3 in TRADE_REGIONS
        }
        all_ISO3_across_relevant_regions: set[str] = set(relevant_trade_regions.keys())

        # Only ISO3 codes that both lie within the model domain and appear in the trade
        # region dataset are considered here. For those ISO3 codes, missing preference
        # data is filled using donor countries. ISO3 codes that are not present in the
        # trade region dataset at all (e.g. some partially recognized states) are
        # excluded from this step, even if separate preference data might exist for them.
        donor_data = {}
        for ISO3 in all_ISO3_across_relevant_regions:
            region_risk_aversion_data = preferences_global[
                preferences_global["ISO3"] == ISO3
            ]
            if region_risk_aversion_data.empty:  # country NOT in preferences dataset
                countries_with_preferences_data = (
                    preferences_global["ISO3"].unique().tolist()
                )
                donor_countries = setup_donor_countries(
                    self.data_catalog,
                    self.geom["global_countries"],
                    countries_with_preferences_data,
                    list(all_ISO3_across_relevant_regions),
                )

                donor_country = donor_countries.get(ISO3, None)
                assert donor_country is not None, f"No donor country found for {ISO3}"

                region_risk_aversion_data = preferences_global[
                    preferences_global["ISO3"] == donor_country
                ].copy()

                self.logger.info(
                    f"Missing risk aversion data for {ISO3}, filling with {donor_country} instead."
                )
                # ensure that the country and ISO3 represent the original country, not the donor country
                region_risk_aversion_data.loc[:, "Country"] = [
                    key for key, val in COUNTRY_NAME_TO_ISO3.items() if val == ISO3
                ]
                region_risk_aversion_data.loc[:, "ISO3"] = ISO3

            region_risk_aversion_data = region_risk_aversion_data[
                [
                    "Country",
                    "ISO3",
                    "Gains",
                    "Losses",
                    "Gains_std",
                    "Losses_std",
                    "Discount",
                    "Discount_std",
                ]
            ]
            region_risk_aversion_data.reset_index(drop=True, inplace=True)
            # Store pivoted data in dictionary with region_id as key
            donor_data[ISO3] = region_risk_aversion_data

        # Concatenate all regional data into a single DataFrame with MultiIndex
        donor_data = pd.concat(donor_data, names=["ISO3"])

        # Drop crops with no data at all for these regions
        donor_data = donor_data.dropna(axis=1, how="all")

        unique_regions = self.geom["regions"]

        data = donate_and_receive_crop_prices(
            donor_data,
            unique_regions,
            TRADE_REGIONS,
            self.data_catalog,
            self.geom["global_countries"],
            self.geom["regions"],
        )

        # Map to corresponding region
        data_reset = data.reset_index(level="region_id")
        data = data_reset.set_index("region_id")
        region_ids = self.array["agents/farmers/region_id"]

        # Set gains and losses
        gains_array = pd.Series(region_ids).map(data["Gains"]).to_numpy()
        gains_std = pd.Series(region_ids).map(data["Gains_std"]).to_numpy()
        losses_array = pd.Series(region_ids).map(data["Losses"]).to_numpy()
        losses_std = pd.Series(region_ids).map(data["Losses_std"]).to_numpy()
        discount_array = pd.Series(region_ids).map(data["Discount"]).to_numpy()
        discount_std = pd.Series(region_ids).map(data["Discount_std"]).to_numpy()

        education_levels = self.array["agents/farmers/education_level"]
        age = self.array["agents/farmers/age_household_head"]

        def normalize(array: np.ndarray) -> np.ndarray:
            return (array - np.min(array)) / (np.max(array) - np.min(array))

        combined_deviation_risk_aversion = ((2 / 6) * normalize(education_levels)) + (
            (4 / 6) * normalize(age)
        )

        # Generate random noise, positively correlated with education levels and age
        # (Higher age / education level means more risk averse)
        z = np.random.normal(
            loc=0, scale=1, size=combined_deviation_risk_aversion.shape
        )

        # Initial gains_variation proportional to risk aversion
        gains_variation = combined_deviation_risk_aversion * z

        current_std = np.std(gains_variation)
        gains_variation = gains_variation * (gains_std / current_std)

        # Similarly for losses_variation
        losses_variation = combined_deviation_risk_aversion * z
        current_std_losses = np.std(losses_variation)
        losses_variation = losses_variation * (losses_std / current_std_losses)

        # Add the generated noise to the original gains and losses arrays
        gains_array_with_variation = np.clip(gains_array + gains_variation, -0.99, 0.99)
        losses_array_with_variation = np.clip(
            losses_array + losses_variation, -0.99, 0.99
        )

        # Calculate intention factor based on age and education
        # Intention factor scales negatively with age and positively with education level
        intention_factor = normalize(education_levels) - normalize(age)

        # Adjust the intention factor to center it around a mean of 0.3
        # The total intention of age, education and neighbor effects can scale to 1
        intention_factor = intention_factor * 0.333 + 0.333

        neutral_risk_aversion = np.mean(
            [gains_array_with_variation, losses_array_with_variation], axis=0
        )

        self.set_array(neutral_risk_aversion, name="agents/farmers/risk_aversion")
        self.set_array(
            gains_array_with_variation, name="agents/farmers/risk_aversion_gains"
        )
        self.set_array(
            losses_array_with_variation, name="agents/farmers/risk_aversion_losses"
        )
        self.set_array(intention_factor, name="agents/farmers/intention_factor")

        # discount rate
        discount_rate_variation_factor = ((2 / 6) * normalize(education_levels)) + (
            (4 / 6) * normalize(age)
        )
        discount_rate_variation = discount_rate_variation_factor * z * -1

        # Adjust the variations to have the desired overall standard deviation
        current_std = np.std(discount_rate_variation)
        discount_rate_variation = discount_rate_variation * (discount_std / current_std)

        # Apply the variations to the original discount rates and clip the values
        discount_rate_with_variation = np.clip(
            discount_array + discount_rate_variation, 0, 2
        )
        self.set_array(
            discount_rate_with_variation,
            name="agents/farmers/discount_rate",
        )

        interest_rate = np.full(n_farmers, interest_rate, dtype=np.float32)
        self.set_array(interest_rate, name="agents/farmers/interest_rate")

    @build_method(depends_on=[], required=True)
    def setup_assets(
        self,
        feature_types: str | list[str],
        source: str = "geofabrik",
        use_cache: bool = True,
    ) -> None:
        """Get assets from OpenStreetMap (OSM) data.

        Args:
            feature_types: The types of features to download from OSM. Available feature types are 'buildings', 'rails' and 'roads'.
            source: The source of the OSM data. Options are 'geofabrik' or 'movisda'. Default is 'geofabrik'.
            use_cache: If True, the data will be cached in the preprocessing directory. Default is True.
        """
        if isinstance(feature_types, str):
            feature_types: list[str] = [feature_types]

        all_features: dict[str, gpd.GeoDataFrame] = self.data_catalog.fetch(
            "open_street_map"
        ).read(
            self.region.union_all(),
            feature_types=feature_types,
        )

        for feature_type, features in all_features.items():
            self.set_geom(features, name=f"assets/{feature_type}")
