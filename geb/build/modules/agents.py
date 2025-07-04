import json
import math
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
from dateutil.relativedelta import relativedelta
from honeybees.library.raster import pixels_to_coords, sample_from_map
from tqdm import tqdm

from geb.agents.crop_farmers import (
    FIELD_EXPANSION_ADAPTATION,
    INDEX_INSURANCE_ADAPTATION,
    IRRIGATION_EFFICIENCY_ADAPTATION,
    PERSONAL_INSURANCE_ADAPTATION,
    SURFACE_IRRIGATION_EQUIPMENT,
    WELL_ADAPTATION,
)
from geb.workflows.io import fetch_and_save, get_window

from ..workflows.conversions import (
    AQUASTAT_NAME_TO_ISO3,
    COUNTRY_NAME_TO_ISO3,
    GLOBIOM_NAME_TO_ISO3,
    SUPERWELL_NAME_TO_ISO3,
)
from ..workflows.farmers import create_farms, get_farm_distribution, get_farm_locations
from ..workflows.general import (
    clip_with_grid,
)
from ..workflows.population import load_GLOPOP_S


class Agents:
    def __init__(self):
        pass

    def setup_water_demand(self):
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
        self.logger.info("Setting up municipal water demands")

        municipal_water_demand = self.data_catalog.get_dataframe(
            "AQUASTAT_municipal_withdrawal"
        )
        municipal_water_demand["ISO3"] = municipal_water_demand["Area"].map(
            AQUASTAT_NAME_TO_ISO3
        )
        municipal_water_demand = municipal_water_demand.set_index("ISO3")

        municipal_water_demand_per_capita = np.full_like(
            self.array["agents/households/region_id"],
            np.nan,
            dtype=np.float32,
        )

        municipal_water_withdrawal_m3_per_capita_per_day_multiplier = pd.DataFrame()
        for _, region in self.geoms["regions"].iterrows():
            ISO3 = region["ISO3"]

            if (
                ISO3 == "AND"
            ):  # for Andorra (not available in World Bank data), use Spain's data
                self.logger.warning(
                    "Andorra's economic data not available, using Spain's data"
                )
                ISO3 = "ESP"
            elif ISO3 == "LIE":  # for Liechtenstein, use Switzerland's data
                self.logger.warning(
                    "Liechtenstein's economic data not available, using Switzerland's data"
                )
                ISO3 = "CHE"

            region_id = region["region_id"]

            # select domestic water demand for the region
            municipal_water_demand_region = municipal_water_demand.loc[ISO3]
            population = municipal_water_demand_region[
                municipal_water_demand_region["Variable"] == "Total population"
            ]
            population = population.set_index("Year")
            population = population["Value"] * 1000

            municipal_water_withdrawal = municipal_water_demand_region[
                municipal_water_demand_region["Variable"]
                == "Municipal water withdrawal"
            ]
            assert len(municipal_water_withdrawal) > 0, (
                f"Missing municipal water withdrawal data for {ISO3}"
            )

            municipal_water_withdrawal = municipal_water_withdrawal.set_index("Year")
            municipal_water_withdrawal = municipal_water_withdrawal["Value"] * 10e9

            municipal_water_withdrawal_m3_per_capita_per_day = (
                municipal_water_withdrawal / population / 365.2425
            )
            municipal_water_withdrawal_m3_per_capita_per_day: pd.DataFrame = (
                municipal_water_withdrawal_m3_per_capita_per_day
            ).dropna()

            municipal_water_withdrawal_m3_per_capita_per_day: pd.DataFrame = (
                municipal_water_withdrawal_m3_per_capita_per_day.reindex(
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
                f"Too large water withdrawal data for {ISO3}"
            )

            municipal_water_demand_2000_m3_per_capita_per_day: pd.DataFrame = (
                municipal_water_withdrawal_m3_per_capita_per_day.loc[2000].item()
            )

            municipal_water_demand_per_capita[
                self.array["agents/households/region_id"] == region_id
            ] = municipal_water_demand_2000_m3_per_capita_per_day

            # scale municipal water demand table to use baseline as 1.00 and scale other values
            # relatively
            municipal_water_withdrawal_m3_per_capita_per_day_multiplier[region_id] = (
                municipal_water_withdrawal_m3_per_capita_per_day
                / municipal_water_demand_2000_m3_per_capita_per_day
            )

        # we don't want to calculate the water demand for every year,
        # so instead we use a baseline (2000 for easy reasoning), and scale
        # the other years relatively to the baseline
        self.set_table(
            municipal_water_withdrawal_m3_per_capita_per_day_multiplier,
            name="municipal_water_withdrawal_m3_per_capita_per_day_multiplier",
        )

        assert not np.isnan(municipal_water_demand_per_capita).any(), (
            "Missing municipal water demand per capita data"
        )
        self.set_array(
            municipal_water_demand_per_capita,
            name="agents/households/municipal_water_demand_per_capita_m3_baseline",
        )

        self.logger.info("Setting up other water demands")

        def set_demand(file, variable, name, ssp):
            ds_historic = xr.open_dataset(
                self.data_catalog.get_source(f"cwatm_{file}_historical_year").path,
                decode_times=False,
            ).rename({"lat": "y", "lon": "x"})
            ds_historic = ds_historic.isel(
                get_window(ds_historic.x, ds_historic.y, self.bounds, buffer=2)
            )[variable]

            ds_future = xr.open_dataset(
                self.data_catalog.get_source(f"cwatm_{file}_{ssp}_year").path,
                decode_times=False,
            ).rename({"lat": "y", "lon": "x"})
            ds_future = ds_future.isel(
                get_window(ds_future.x, ds_future.y, self.bounds, buffer=2)
            )[variable]

            ds_future = ds_future.sel(
                time=slice(ds_historic.time[-1] + 1, ds_future.time[-1])
            )

            ds = xr.concat([ds_historic, ds_future], dim="time")
            ds = ds.rio.write_crs(4326)
            # assert dataset in monotonicically increasing
            assert (ds.time.diff("time") == 1).all(), "not all years are there"

            ds["time"] = pd.date_range(
                start=datetime(1901, 1, 1)
                + relativedelta(years=int(ds.time[0].data.item())),
                periods=len(ds.time),
                freq="YS",
            )

            assert (ds.time.dt.year.diff("time") == 1).all(), "not all years are there"
            ds = ds.sel(time=slice(self.start_date, self.end_date))
            ds.attrs["_FillValue"] = np.nan
            self.set_other(ds, name=f"water_demand/{name}")

        set_demand(
            "industry_water_demand",
            "indWW",
            "industry_water_demand",
            self.ssp,
        )
        set_demand(
            "industry_water_demand",
            "indCon",
            "industry_water_consumption",
            self.ssp,
        )
        set_demand(
            "livestock_water_demand",
            "livestockConsumption",
            "livestock_water_consumption",
            "ssp2",
        )

    def setup_economic_data(self):
        """Sets up the economic data for GEB.

        Notes:
        -----
        This method sets up the lending rates and inflation rates data for GEB. It first retrieves the
        lending rates and inflation rates data from the World Bank dataset using the `get_geodataframe` method of the
        `data_catalog` object. It then creates dictionaries to store the data for each region, with the years as the time
        dimension and the lending rates or inflation rates as the data dimension.

        The lending rates and inflation rates data are converted from percentage to rate by dividing by 100 and adding 1.
        The data is then stored in the dictionaries with the region ID as the key.

        The resulting lending rates and inflation rates data are set as forcing data in the model with names of the form
        'socioeconomics/lending_rates' and 'socioeconomics/inflation_rates', respectively.
        """
        self.logger.info("Setting up economic data")

        # lending_rates = self.data_catalog.get_dataframe("wb_lending_rate")
        inflation_rates = self.data_catalog.get_dataframe("wb_inflation_rate")
        price_ratio = self.data_catalog.get_dataframe("world_bank_price_ratio")
        LCU_per_USD = self.data_catalog.get_dataframe("wb_LCU_per_USD")

        def filter_and_rename(df, additional_cols):
            # Select columns: 'Country Name', 'Country Code', and columns containing "YR"
            columns_to_keep = additional_cols + [
                col
                for col in df.columns
                if col.isnumeric() and 1900 <= int(col) <= 3000
            ]
            filtered_df = df[columns_to_keep]
            return filtered_df

        def extract_years(df):
            # Extract years that are numerically valid between 1900 and 3000
            return [
                col
                for col in df.columns
                if col.isnumeric() and 1900 <= int(col) <= 3000
            ]

        # Assuming dataframes for PPP and LCU per USD have been initialized
        price_ratio_filtered = filter_and_rename(
            price_ratio, ["Country Name", "Country Code"]
        )
        years_price_ratio = extract_years(price_ratio_filtered)
        price_ratio_dict = {"time": years_price_ratio, "data": {}}  # price ratio

        lcu_filtered = filter_and_rename(LCU_per_USD, ["Country Name", "Country Code"])
        years_lcu = extract_years(lcu_filtered)
        lcu_dict = {"time": years_lcu, "data": {}}  # LCU per USD

        # Assume lending_rates and inflation_rates are available
        # years_lending_rates = extract_years(lending_rates)
        years_inflation_rates = extract_years(inflation_rates)

        # lending_rates_dict = {"time": years_lending_rates, "data": {}}
        inflation_rates_dict = {"time": years_inflation_rates, "data": {}}

        # Create a helper to process rates and assert single row data
        def process_rates(df, rate_cols, ISO3, convert_percent_to_ratio=False):
            filtered_data = df.loc[df["Country Code"] == ISO3, rate_cols]
            assert len(filtered_data) == 1, (
                f"Expected one row for {ISO3}, got {len(filtered_data)}"
            )
            if convert_percent_to_ratio:
                return (filtered_data.iloc[0] / 100 + 1).tolist()
            return filtered_data.iloc[0].tolist()

        USA_inflation_rates = process_rates(
            inflation_rates,
            years_inflation_rates,
            "USA",
            convert_percent_to_ratio=True,
        )

        for _, region in self.geoms["regions"].iterrows():
            region_id = str(region["region_id"])

            # Store data in dictionaries
            # lending_rates_dict["data"][region_id] = process_rates(
            #     lending_rates,
            #     years_lending_rates,
            #     region["ISO3"],
            #     convert_percent_to_ratio=True,
            # )
            ISO3 = region["ISO3"]
            if (
                ISO3 == "AND"
            ):  # for Andorra (not available in World Bank data), use Spain's data
                self.logger.warning(
                    "Andorra's economic data not available, using Spain's data"
                )
                ISO3 = "ESP"
            elif ISO3 == "LIE":  # for Liechtenstein, use Switzerland's data
                self.logger.warning(
                    "Liechtenstein's economic data not available, using Switzerland's data"
                )
                ISO3 = "CHE"

            local_inflation_rates = process_rates(
                inflation_rates,
                years_inflation_rates,
                ISO3,
                convert_percent_to_ratio=True,
            )
            assert not np.isnan(local_inflation_rates).any(), (
                f"Missing inflation rates for {region['ISO3']}"
            )
            inflation_rates_dict["data"][region_id] = (
                np.array(local_inflation_rates) / np.array(USA_inflation_rates)
            ).tolist()

            price_ratio_dict["data"][region_id] = process_rates(
                price_ratio_filtered, years_price_ratio, region["ISO3"]
            )

            lcu_dict["data"][region_id] = process_rates(
                lcu_filtered, years_lcu, region["ISO3"]
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

        self.set_dict(inflation_rates_dict, name="socioeconomics/inflation_rates")
        # self.set_dict(lending_rates_dict, name="socioeconomics/lending_rates")
        self.set_dict(price_ratio_dict, name="socioeconomics/price_ratio")
        self.set_dict(lcu_dict, name="socioeconomics/LCU_per_USD")

    def setup_irrigation_sources(self, irrigation_sources):
        self.set_dict(irrigation_sources, name="agents/farmers/irrigation_sources")

    def setup_irrigation_prices_by_reference_year(
        self,
        operation_surface: float,
        operation_sprinkler: float,
        operation_drip: float,
        capital_cost_surface: float,
        capital_cost_sprinkler: float,
        capital_cost_drip: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ):
        """Sets up the well prices and upkeep prices for the hydrological model based on a reference year.

        Parameters
        ----------
        well_price : float
            The price of a well in the reference year.
        upkeep_price_per_m2 : float
            The upkeep price per square meter of a well in the reference year.
        reference_year : int
            The reference year for the well prices and upkeep prices.
        start_year : int
            The start year for the well prices and upkeep prices.
        end_year : int
            The end year for the well prices and upkeep prices.

        Notes:
        -----
        This method sets up the well prices and upkeep prices for the hydrological model based on a reference year. It first
        retrieves the inflation rates data from the `socioeconomics/inflation_rates` dictionary. It then creates dictionaries to
        store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
        data dimension.

        The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        The resulting well prices and upkeep prices data are set as dictionary with names of the form
        'socioeconomics/well_prices' and 'socioeconomics/upkeep_prices_well_per_m2', respectively.
        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["socioeconomics/inflation_rates"]
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
            self.set_dict(prices_dict, name=f"socioeconomics/{price_type}")

    def setup_well_prices_by_reference_year_global(
        self,
        WHY_10: float,
        WHY_20: float,
        WHY_30: float,
        reference_year: int,
    ):
        """Sets up the well prices and upkeep prices for the hydrological model based on a reference year.

        Parameters
        ----------
        well_price : float
            The price of a well in the reference year.
        upkeep_price_per_m2 : float
            The upkeep price per square meter of a well in the reference year.
        reference_year : int
            The reference year for the well prices and upkeep prices.

        Notes:
        -----
        This method sets up the well prices and upkeep prices for the hydrological model based on a reference year. It first
        retrieves the inflation rates data from the `socioeconomics/inflation_rates` dictionary. It then creates dictionaries to
        store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
        data dimension.

        The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        The resulting well prices and upkeep prices data are set as dictionary with names of the form
        'socioeconomics/well_prices' and 'socioeconomics/upkeep_prices_well_per_m2', respectively.
        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["socioeconomics/inflation_rates"]
        price_ratio = self.dict["socioeconomics/price_ratio"]

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
            prices_dict = {"time": list(range(start_year, end_year + 1)), "data": {}}

            for _, region in self.geoms["regions"].iterrows():
                region_id = str(region["region_id"])

                prices = pd.Series(index=range(start_year, end_year + 1))
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

                prices_dict["data"][region_id] = prices.tolist()

            # Set the calculated prices in the appropriate dictionary
            self.set_dict(prices_dict, name=f"socioeconomics/{price_type}")

        electricity_rates = self.data_catalog.get_dataframe("gcam_electricity_rates")
        electricity_rates["ISO3"] = electricity_rates["Country"].map(
            SUPERWELL_NAME_TO_ISO3
        )
        electricity_rates = electricity_rates.set_index("ISO3")["Rate"].to_dict()

        electricity_rates_dict = {
            "time": list(range(start_year, end_year + 1)),
            "data": {},
        }

        for _, region in self.geoms["regions"].iterrows():
            region_id = str(region["region_id"])

            prices = pd.Series(index=range(start_year, end_year + 1))
            prices.loc[reference_year] = electricity_rates[region["ISO3"]]

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

            electricity_rates_dict["data"][region_id] = prices.tolist()

        # Set the calculated prices in the appropriate dictionary
        self.set_dict(electricity_rates_dict, name="socioeconomics/electricity_cost")

    def setup_drip_irrigation_prices_by_reference_year(
        self,
        drip_irrigation_price: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ):
        """Sets up the drip_irrigation prices and upkeep prices for the hydrological model based on a reference year.

        Parameters
        ----------
        drip_irrigation_price : float
            The price of a drip_irrigation in the reference year.

        reference_year : int
            The reference year for the drip_irrigation prices and upkeep prices.
        start_year : int
            The start year for the drip_irrigation prices and upkeep prices.
        end_year : int
            The end year for the drip_irrigation prices and upkeep prices.

        Notes:
        -----
        The drip_irrigation prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["socioeconomics/inflation_rates"]
        regions = list(inflation_rates["data"].keys())

        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "drip_irrigation_price": drip_irrigation_price,
        }

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
            self.set_dict(prices_dict, name=f"socioeconomics/{price_type}")

    def setup_farmers(self, farmers):
        """Sets up the farmers data for GEB.

        Parameters
        ----------
        farmers : pandas.DataFrame
            A DataFrame containing the farmer data.
        irrigation_sources : dict, optional
            A dictionary mapping irrigation source names to IDs.
        n_seasons : int, optional
            The number of seasons to simulate.

        Notes:
        -----
        This method sets up the farmers data for GEB. It first retrieves the region data from the
        `regions` and `subgrid` grids. It then creates a `farms` grid with the same shape as the
        `region_subgrid` grid, with a value of -1 for each cell.

        For each region, the method clips the `cultivated_land` grid to the region and creates farms for the region using
        the `create_farms` function, using these farmlands as well as the dataframe of farmer agents. The resulting farms
        whose IDs correspondd to the IDs in the farmer dataframe are added to the `farms` grid for the region.

        The method then removes any farms that are outside the study area by using the `region_mask` grid. It then remaps
        the farmer IDs to a contiguous range of integers starting from 0.

        The resulting farms data is set as agents data in the model with names of the form 'agents/farmers/farms'. The
        crop names are mapped to IDs using the `crop_name_to_id` dictionary that was previously created. The resulting
        crop IDs are stored in the `season_#_crop` columns of the `farmers` DataFrame.

        If `irrigation_sources` is provided, the method sets the `irrigation_source` column of the `farmers` DataFrame to
        the corresponding IDs.

        Finally, the method sets the array data for each column of the `farmers` DataFrame as agents data in the model
        with names of the form 'agents/farmers/{column}'.
        """
        regions = self.geoms["regions"]
        region_ids = self.region_subgrid["region_ids"]
        full_region_cultivated_land = self.region_subgrid[
            "landsurface/full_region_cultivated_land"
        ]

        farms = self.full_like(region_ids, fill_value=-1, nodata=-1)
        for region_id in regions["region_id"]:
            self.logger.info(f"Creating farms for region {region_id}")
            region = region_ids == region_id
            region_clip, bounds = clip_with_grid(region, region)

            cultivated_land_region = full_region_cultivated_land.isel(bounds)
            cultivated_land_region = xr.where(
                region_clip, cultivated_land_region, 0, keep_attrs=True
            )
            # TODO: Why does nodata value disappear?
            # self.dict['areamaps/region_id_mapping'][farmers['region_id']]
            farmer_region_ids = farmers["region_id"]
            farmers_region = farmers[farmer_region_ids == region_id]
            farms_region = create_farms(
                farmers_region, cultivated_land_region, farm_size_key="area_n_cells"
            )
            assert (
                farms_region.min() >= -1
            )  # -1 is nodata value, all farms should be positive
            farms[bounds] = xr.where(
                region_clip, farms_region, farms.isel(bounds), keep_attrs=True
            )
            # farms = farms.compute()  # perhaps this helps with memory issues?

        farmers = farmers.drop("area_n_cells", axis=1)

        cut_farms = np.unique(
            xr.where(
                self.region_subgrid["mask"],
                farms.copy().values,
                -1,
                keep_attrs=True,
            )
        )
        cut_farm_indices = cut_farms[cut_farms != -1]

        assert farms.min() >= -1  # -1 is nodata value, all farms should be positive
        subgrid_farms = farms.raster.clip_bbox(self.subgrid["mask"].raster.bounds)

        subgrid_farms_in_study_area = xr.where(
            np.isin(subgrid_farms, cut_farm_indices), -1, subgrid_farms, keep_attrs=True
        )
        farmers = farmers[~farmers.index.isin(cut_farm_indices)]

        remap_farmer_ids = np.full(
            farmers.index.max() + 2, -1, dtype=np.int32
        )  # +1 because 0 is also a farm, +1 because no farm is -1, set to -1 in next step
        remap_farmer_ids[farmers.index] = np.arange(len(farmers))
        subgrid_farms_in_study_area = remap_farmer_ids[
            subgrid_farms_in_study_area.values
        ]

        farmers = farmers.reset_index(drop=True)

        assert np.setdiff1d(np.unique(subgrid_farms_in_study_area), -1).size == len(
            farmers
        )
        assert farmers.iloc[-1].name == subgrid_farms_in_study_area.max()

        subgrid_farms_in_study_area_ = self.full_like(
            self.subgrid["mask"],
            fill_value=-1,
            nodata=-1,
            dtype=np.int32,
        )
        subgrid_farms_in_study_area_[:] = subgrid_farms_in_study_area

        self.set_subgrid(subgrid_farms_in_study_area_, name="agents/farmers/farms")
        self.set_array(farmers.index.values, name="agents/farmers/id")
        self.set_array(farmers["region_id"].values, name="agents/farmers/region_id")

    def setup_farmers_from_csv(self, path=None):
        """Sets up the farmers data for GEB from a CSV file.

        Parameters
        ----------
        path : str
            The path to the CSV file containing the farmer data.

        Notes:
        -----
        This method sets up the farmers data for GEB from a CSV file. It first reads the farmer data from
        the CSV file using the `pandas.read_csv` method.

        See the `setup_farmers` method for more information on how the farmer data is set up in the model.
        """
        if path is None:
            path = self.preprocessing_dir / "agents" / "farmers" / "farmers.csv"
        farmers = pd.read_csv(path, index_col=0)
        self.setup_farmers(farmers)

    def setup_create_farms(
        self,
        region_id_column="region_id",
        country_iso3_column="ISO3",
        farm_size_donor_countries=None,
        data_source="lowder",
        size_class_boundaries=None,
    ):
        """Sets up the farmers for GEB.

        Parameters
        ----------
        region_id_column : str, optional
            The name of the column in the region database that contains the region IDs. Default is 'UID'.
        country_iso3_column : str, optional
            The name of the column in the region database that contains the country ISO3 codes. Default is 'ISO3'.
        farm_size_donor_countries : dict, optional
            Dictionary with key, value pairs of ISO3 codes. The value-country is used as donor for the key-country.
            Default is None.


        Notes:
        -----
        This method sets up the farmers for GEB. This is a simplified method that generates an example set of agent data.
        It first calculates the number of farmers and their farm sizes for each region based on the agricultural data for
        that region based on theamount of farm land and data from a global database on farm sizes per country. It then
        randomly assigns crops, irrigation sources, household sizes, and daily incomes and consumption levels to each farmer.

        A paper that reports risk aversion values for 75 countries is this one: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2646134
        """
        if data_source == "lowder":
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
        else:
            assert size_class_boundaries is not None
            assert farm_size_donor_countries is None, (
                "farm_size_donor_countries is only used for lowder data"
            )

        cultivated_land = self.region_subgrid["landsurface/full_region_cultivated_land"]
        assert cultivated_land.dtype == bool, "Cultivated land must be boolean"
        region_ids = self.region_subgrid["region_ids"]
        cell_area = self.region_subgrid["cell_area"]

        regions_shapes = self.geoms["regions"]
        if data_source == "lowder":
            assert country_iso3_column in regions_shapes.columns, (
                f"Region database must contain {country_iso3_column} column ({self.data_catalog['GADM_level1'].path})"
            )

            farm_sizes_per_region = (
                self.data_catalog.get_dataframe("lowder_farm_sizes")
                .dropna(subset=["Total"], axis=0)
                .drop(["empty", "income class"], axis=1)
            )
            farm_sizes_per_region["Country"] = farm_sizes_per_region["Country"].ffill()
            # Remove preceding and trailing white space from country names
            farm_sizes_per_region["Country"] = farm_sizes_per_region[
                "Country"
            ].str.strip()
            farm_sizes_per_region["Census Year"] = farm_sizes_per_region[
                "Country"
            ].ffill()

            farm_sizes_per_region["ISO3"] = farm_sizes_per_region["Country"].map(
                COUNTRY_NAME_TO_ISO3
            )
            assert not farm_sizes_per_region["ISO3"].isna().any(), (
                f"Found {farm_sizes_per_region['ISO3'].isna().sum()} countries without ISO3 code"
            )
        else:
            # load data source
            farm_sizes_per_region = pd.read_excel(
                data_source["farm_size"], index_col=(0, 1, 2)
            )
            n_farms_per_region = pd.read_excel(
                data_source["n_farms"],
                index_col=(0, 1, 2),
            )

        all_agents = []
        self.logger.info(f"Starting processing of {len(regions_shapes)} regions")
        for i, (_, region) in enumerate(regions_shapes.iterrows()):
            UID = region[region_id_column]
            if data_source == "lowder":
                ISO3 = region[country_iso3_column]
                if farm_size_donor_countries:
                    assert isinstance(farm_size_donor_countries, dict)
                    ISO3 = farm_size_donor_countries.get(ISO3, ISO3)
                self.logger.info(
                    f"Processing region ({i + 1}/{len(regions_shapes)}) with ISO3 {ISO3}"
                )
            else:
                state, district, tehsil = (
                    region["state_name"],
                    region["district_n"],
                    region["sub_dist_1"],
                )
                self.logger.info(f"Processing region ({i + 1}/{len(regions_shapes)})")

            cultivated_land_region_total_cells = (
                ((region_ids == UID) & (cultivated_land)).sum().compute()
            )
            total_cultivated_land_area_lu = (
                (((region_ids == UID) & (cultivated_land)) * cell_area).sum().compute()
            )
            if (
                total_cultivated_land_area_lu == 0
            ):  # when no agricultural area, just continue as there will be no farmers. Also avoiding some division by 0 errors.
                continue

            average_subgrid_area_region = (
                cell_area.where(((region_ids == UID) & (cultivated_land)))
                .mean()
                .compute()
            )

            if data_source == "lowder":
                region_farm_sizes = farm_sizes_per_region.loc[
                    (farm_sizes_per_region["ISO3"] == ISO3)
                ].drop(["Country", "Census Year", "Total"], axis=1)
                assert len(region_farm_sizes) == 2, (
                    f"Found {len(region_farm_sizes) / 2} region_farm_sizes for {ISO3}"
                )

                # Extract holdings and agricultural area data
                region_n_holdings = (
                    region_farm_sizes.loc[
                        region_farm_sizes["Holdings/ agricultural area"] == "Holdings"
                    ]
                    .iloc[0]
                    .drop(["Holdings/ agricultural area", "ISO3"])
                    .replace("..", np.nan)
                    .astype(float)
                )
                agricultural_area_db_ha = (
                    region_farm_sizes.loc[
                        region_farm_sizes["Holdings/ agricultural area"]
                        == "Agricultural area (Ha) "
                    ]
                    .iloc[0]
                    .drop(["Holdings/ agricultural area", "ISO3"])
                    .replace("..", np.nan)
                    .astype(float)
                )

                # Calculate average sizes for each bin
                average_sizes = {}
                for bin_name in agricultural_area_db_ha.index:
                    bin_name = bin_name.strip()
                    if bin_name.startswith("<"):
                        # For '< 1 Ha', average is 0.5 Ha
                        average_size = 0.5
                    elif bin_name.startswith(">"):
                        # For '> 1000 Ha', assume average is 1500 Ha
                        average_size = 1500
                    else:
                        # For ranges like '5 - 10 Ha', calculate the midpoint
                        try:
                            min_size, max_size = bin_name.replace("Ha", "").split("-")
                            min_size = float(min_size.strip())
                            max_size = float(max_size.strip())
                            average_size = (min_size + max_size) / 2
                        except ValueError:
                            # Default average size if parsing fails
                            average_size = 1
                    average_sizes[bin_name] = average_size

                # Convert average sizes to a pandas Series
                average_sizes_series = pd.Series(average_sizes)

                # Handle cases where entries are zero or missing
                agricultural_area_db_ha_zero_or_nan = (
                    agricultural_area_db_ha.isnull() | (agricultural_area_db_ha == 0)
                )
                region_n_holdings_zero_or_nan = region_n_holdings.isnull() | (
                    region_n_holdings == 0
                )

                if agricultural_area_db_ha_zero_or_nan.all():
                    # All entries in agricultural_area_db_ha are zero or NaN
                    if not region_n_holdings_zero_or_nan.all():
                        # Calculate agricultural_area_db_ha using average sizes and region_n_holdings
                        region_n_holdings = region_n_holdings.fillna(1).replace(0, 1)
                        agricultural_area_db_ha = (
                            average_sizes_series * region_n_holdings
                        )
                    else:
                        raise ValueError(
                            "Cannot calculate agricultural_area_db_ha: both datasets are zero or missing."
                        )
                elif region_n_holdings_zero_or_nan.all():
                    # All entries in region_n_holdings are zero or NaN
                    if not agricultural_area_db_ha_zero_or_nan.all():
                        # Calculate region_n_holdings using agricultural_area_db_ha and average sizes
                        agricultural_area_db_ha = agricultural_area_db_ha.fillna(
                            1
                        ).replace(0, 1)
                        region_n_holdings = (
                            agricultural_area_db_ha / average_sizes_series
                        )
                    else:
                        raise ValueError(
                            "Cannot calculate region_n_holdings: both datasets are zero or missing."
                        )
                else:
                    # Replace zeros and NaNs in both datasets to avoid division by zero
                    region_n_holdings = region_n_holdings.fillna(1).replace(0, 1)
                    agricultural_area_db_ha = agricultural_area_db_ha.fillna(1).replace(
                        0, 1
                    )

                # Calculate total agricultural area in square meters
                agricultural_area_db = (
                    agricultural_area_db_ha * 10000
                )  # Convert Ha to m^2

                # Calculate region farm sizes
                region_farm_sizes = agricultural_area_db / region_n_holdings

            else:
                region_farm_sizes = farm_sizes_per_region.loc[(state, district, tehsil)]
                region_n_holdings = n_farms_per_region.loc[(state, district, tehsil)]
                agricultural_area_db = region_farm_sizes * region_n_holdings

            total_cultivated_land_area_db = agricultural_area_db.sum()

            n_cells_per_size_class = pd.Series(0, index=region_n_holdings.index)

            for size_class in agricultural_area_db.index:
                if (
                    region_n_holdings[size_class] > 0
                ):  # if no holdings, no need to calculate
                    region_n_holdings[size_class] = region_n_holdings[size_class] * (
                        total_cultivated_land_area_lu / total_cultivated_land_area_db
                    )
                    n_cells_per_size_class.loc[size_class] = (
                        region_n_holdings[size_class]
                        * region_farm_sizes[size_class]
                        / average_subgrid_area_region
                    ).item()
                    assert not np.isnan(n_cells_per_size_class.loc[size_class])
            assert math.isclose(
                cultivated_land_region_total_cells,
                n_cells_per_size_class.sum().item(),
                abs_tol=1,
            ), (
                f"{cultivated_land_region_total_cells}, {n_cells_per_size_class.sum().item()}"
            )

            whole_cells_per_size_class = (n_cells_per_size_class // 1).astype(int)
            leftover_cells_per_size_class = n_cells_per_size_class % 1
            whole_cells = whole_cells_per_size_class.sum()
            n_missing_cells = cultivated_land_region_total_cells - whole_cells
            assert n_missing_cells <= len(agricultural_area_db)

            index = list(
                zip(
                    leftover_cells_per_size_class.index,
                    leftover_cells_per_size_class % 1,
                )
            )
            n_cells_to_add = sorted(index, key=lambda x: x[1], reverse=True)[
                : n_missing_cells.compute().item()
            ]
            whole_cells_per_size_class.loc[[p[0] for p in n_cells_to_add]] += 1

            region_agents = []
            for size_class in whole_cells_per_size_class.index:
                # if no cells for this size class, just continue
                if whole_cells_per_size_class.loc[size_class] == 0:
                    continue

                number_of_agents_size_class = round(
                    region_n_holdings[size_class].item()
                )
                # if there is agricultural land, but there are no agents rounded down, we assume there is one agent
                if (
                    number_of_agents_size_class == 0
                    and whole_cells_per_size_class[size_class] > 0
                ):
                    number_of_agents_size_class = 1

                min_size_m2, max_size_m2 = size_class_boundaries[size_class]
                if max_size_m2 in (np.inf, "inf", "infinity", "Infinity"):
                    max_size_m2 = region_farm_sizes[size_class] * 2

                min_size_cells = int(min_size_m2 / average_subgrid_area_region)
                min_size_cells = max(
                    min_size_cells, 1
                )  # farm can never be smaller than one cell
                max_size_cells = (
                    int(max_size_m2 / average_subgrid_area_region) - 1
                )  # otherwise they overlap with next size class
                mean_cells_per_agent = int(
                    region_farm_sizes[size_class] / average_subgrid_area_region
                )

                assert mean_cells_per_agent >= 1, (
                    f"Mean cells per agent must be at least 1, but got {mean_cells_per_agent}, consider increasing the number of subgrids"
                )

                if (
                    mean_cells_per_agent < min_size_cells
                    or mean_cells_per_agent > max_size_cells
                ):  # there must be an error in the data, thus assume centred
                    mean_cells_per_agent = (min_size_cells + max_size_cells) // 2

                population = pd.DataFrame(index=range(number_of_agents_size_class))

                offset = (
                    whole_cells_per_size_class[size_class]
                    - number_of_agents_size_class * mean_cells_per_agent
                )

                if (
                    number_of_agents_size_class * mean_cells_per_agent + offset
                    < min_size_cells * number_of_agents_size_class
                ):
                    min_size_cells = (
                        number_of_agents_size_class * mean_cells_per_agent + offset
                    ) // number_of_agents_size_class
                if (
                    number_of_agents_size_class * mean_cells_per_agent + offset
                    > max_size_cells * number_of_agents_size_class
                ):
                    max_size_cells = (
                        number_of_agents_size_class * mean_cells_per_agent + offset
                    ) // number_of_agents_size_class + 1

                n_farms_size_class, farm_sizes_size_class = get_farm_distribution(
                    number_of_agents_size_class,
                    min_size_cells,
                    max_size_cells,
                    mean_cells_per_agent,
                    offset,
                    self.logger,
                )
                assert n_farms_size_class.sum() == number_of_agents_size_class
                assert (farm_sizes_size_class >= 1).all()
                assert (
                    n_farms_size_class * farm_sizes_size_class
                ).sum() == whole_cells_per_size_class[size_class]
                farm_sizes = farm_sizes_size_class.repeat(n_farms_size_class)
                np.random.shuffle(farm_sizes)
                population["area_n_cells"] = farm_sizes
                region_agents.append(population)

                assert (
                    population["area_n_cells"].sum()
                    == whole_cells_per_size_class[size_class]
                )

            region_agents = pd.concat(region_agents, ignore_index=True)
            region_agents["region_id"] = UID
            all_agents.append(region_agents)

        farmers = pd.concat(all_agents, ignore_index=True)
        self.setup_farmers(farmers)

    def setup_household_characteristics(self, maximum_age=85, skip_countries_ISO3=[]):
        # load GDL region within model domain
        GDL_regions = self.data_catalog.get_geodataframe(
            "GDL_regions_v4",
            geom=self.region,
            variables=["GDLcode", "iso_code"],
        )
        GDL_regions = GDL_regions[
            GDL_regions["GDLcode"] != "NA"
        ]  # remove regions without GDL code

        # create list of attibutes to include (and include name to store to)
        rename = {
            "HHSIZE_CAT": "household_type",
            "AGE_HH_HEAD": "age_household_head",
            "EDUC": "education_level",
            "WEALTH_INDEX": "wealth_index",
            "RURAL": "rural",
        }
        region_results = {}

        # get age class to age (head of household) mapping
        age_class_to_age = {
            1: (0, 4),
            2: (5, 14),
            3: (15, 24),
            4: (25, 34),
            5: (35, 44),
            6: (45, 54),
            7: (55, 64),
            8: (66, maximum_age + 1),
        }

        # iterate over regions and sample agents from GLOPOP-S
        for i, (_, GDL_region) in enumerate(GDL_regions.iterrows()):
            GDL_code = GDL_region["GDLcode"]
            self.logger.info(
                f"Setting up household characteristics for {GDL_region['GDLcode']} ({i + 1}/{len(GDL_regions)})"
            )

            if GDL_region["iso_code"] in skip_countries_ISO3:
                self.logger.info(
                    f"Skipping setting up household characteristics for {GDL_region['GDLcode']}"
                )
                continue

            GLOPOP_S_region, GLOPOP_GRID_region = load_GLOPOP_S(
                self.data_catalog, GDL_code
            )

            GLOPOP_S_region = GLOPOP_S_region.rename(columns=rename)

            # get size of household
            HH_SIZE = GLOPOP_S_region["HID"].value_counts()

            # only select household heads
            GLOPOP_S_region = GLOPOP_S_region[GLOPOP_S_region["RELATE_HEAD"] == 1]

            # add household sizes to household df
            GLOPOP_S_region = GLOPOP_S_region.merge(HH_SIZE, on="HID", how="left")
            GLOPOP_S_region = GLOPOP_S_region.rename(
                columns={"count": "HHSIZE"}
            ).reset_index(drop=True)

            # clip grid to model bounds
            GLOPOP_GRID_region = GLOPOP_GRID_region.rio.clip_box(*self.bounds)

            # get unique cells in grid
            unique_grid_cells = np.unique(GLOPOP_GRID_region.values)

            # subset GLOPOP_households_region
            GLOPOP_S_region = GLOPOP_S_region[
                GLOPOP_S_region["GRID_CELL"].isin(unique_grid_cells)
            ]

            # create column WEALTH_INDEX (GLOPOP-S contains either INCOME or WEALTH data, depending on the region. Therefor we combine these.)
            GLOPOP_S_region["wealth_index"] = (
                GLOPOP_S_region["WEALTH"] + GLOPOP_S_region["INCOME"] + 1
            )

            # calculate age:
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

            # create all households
            GLOPOP_households_region = np.unique(GLOPOP_S_region["HID"])
            n_households = GLOPOP_households_region.size

            # iterate over unique housholds and extract the variables we want
            household_characteristics = {}
            household_characteristics["size"] = np.full(
                n_households, -1, dtype=np.int32
            )

            household_characteristics["location"] = np.full(
                (n_households, 2), -1, dtype=np.float32
            )

            for column in (
                "household_type",
                "age_household_head",
                "education_level",
                "wealth_index",
                "rural",
            ):
                household_characteristics[column] = np.array(GLOPOP_S_region[column])

            household_characteristics["size"] = np.array(GLOPOP_S_region["HHSIZE"])
            # now find location of household
            # get x and y from df

            res_x, res_y = GLOPOP_GRID_region.rio.resolution()
            n = len(GLOPOP_S_region)
            x_y = np.stack(
                [
                    GLOPOP_S_region["coord_X"].astype(np.float32)
                    + (np.random.random(n).astype(np.float32) * abs(res_x))
                    - 0.5 * res_x,
                    GLOPOP_S_region["coord_Y"].astype(np.float32)
                    + (np.random.random(n).astype(np.float32) * abs(res_y))
                    - 0.5 * res_y,
                ],
                axis=1,
            )
            # round to precision of ~0.11 m for lat/lon to reduce compressed file size
            household_characteristics["location"] = np.round(x_y, 6)

            household_characteristics["region_id"] = sample_from_map(
                self.region_subgrid["region_ids"].values,
                household_characteristics["location"],
                self.region_subgrid["region_ids"].rio.transform(recalc=True).to_gdal(),
            )

            households_with_region = household_characteristics["region_id"] != -1

            for column, data in household_characteristics.items():
                # only keep households with region
                household_characteristics[column] = data[households_with_region]

            # ensure that all households have a region assigned
            assert not (household_characteristics["region_id"] == -1).any()

            region_results[GDL_code] = household_characteristics

        # concatenate all data
        for household_attribute in household_characteristics:
            data_concatenated = np.concatenate(
                [
                    region_results[GDL_code][household_attribute]
                    for GDL_code in region_results
                ]
            )

            # and store to array
            self.set_array(
                data_concatenated,
                name=f"agents/households/{household_attribute}",
            )

    def setup_farmer_household_characteristics(self, maximum_age=85):
        n_farmers = self.array["agents/farmers/id"].size
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
        GDL_regions = self.data_catalog.get_geodataframe(
            "GDL_regions_v4", geom=self.region, variables=["GDLcode"]
        )
        if (GDL_regions["GDLcode"] == "NA").any():
            self.logger.warning("GDL region has a 'NA', these rows will be deleted.")
            GDL_regions = GDL_regions[GDL_regions["GDLcode"] != "NA"]

        GDL_region_per_farmer = gpd.sjoin_nearest(locations, GDL_regions, how="left")

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
            if GDL_region == "ANDt":
                GDL_region = "ESPr112"
            if GDL_region == "LIEt":
                GDL_region = "CHEr105"

            GLOPOP_S_region, _ = load_GLOPOP_S(self.data_catalog, GDL_region)

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
            # select only household heads
            household_heads = households_region.apply(
                lambda x: x[x["RELATE_HEAD"] == 1]
            )
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

    def create_preferences(self) -> pd.DataFrame:
        # Risk aversion
        preferences_country_level: pd.DataFrame = self.data_catalog.get_dataframe(
            "preferences_country",
            variables=["country", "isocode", "patience", "risktaking"],
        ).dropna()

        preferences_individual_level: pd.DataFrame = self.data_catalog.get_dataframe(
            "preferences_individual",
            variables=["country", "isocode", "patience", "risktaking"],
        ).dropna()

        def scale_to_range(x: pd.Series, new_min: float, new_max: float):
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

    def setup_farmer_characteristics(
        self,
        interest_rate=0.05,
    ):
        n_farmers = self.array["agents/farmers/id"].size

        preferences_global = self.create_preferences()
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

        GLOBIOM_regions = self.data_catalog.get_dataframe("GLOBIOM_regions_37")
        GLOBIOM_regions["ISO3"] = GLOBIOM_regions["Country"].map(GLOBIOM_NAME_TO_ISO3)
        # For my personal branch
        GLOBIOM_regions.loc[GLOBIOM_regions["Country"] == "Switzerland", "Region37"] = (
            "EU_MidWest"
        )
        assert not np.any(GLOBIOM_regions["ISO3"].isna()), "Missing ISO3 codes"

        ISO3_codes_region = self.geoms["regions"]["ISO3"].unique()
        GLOBIOM_regions_region = GLOBIOM_regions[
            GLOBIOM_regions["ISO3"].isin(ISO3_codes_region)
        ]["Region37"].unique()
        ISO3_codes_GLOBIOM_region = GLOBIOM_regions[
            GLOBIOM_regions["Region37"].isin(GLOBIOM_regions_region)
        ]["ISO3"]

        donor_data = {}
        for ISO3 in ISO3_codes_GLOBIOM_region:
            if ISO3 == "AND":
                ISO3 = "ESP"
            elif ISO3 == "LIE":
                ISO3 = "CHE"
            region_risk_aversion_data = preferences_global[
                preferences_global["ISO3"] == ISO3
            ]

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

        unique_regions = self.geoms["regions"]

        data = self.donate_and_receive_crop_prices(
            donor_data, unique_regions, GLOBIOM_regions
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

        def normalize(array):
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

    def setup_farmer_irrigation_source(self, irrigating_farmers, year):
        fraction_sw_irrigation = "aeisw"

        fraction_sw_irrigation_data = xr.open_dataarray(
            self.data_catalog.get_source(
                f"global_irrigation_area_{fraction_sw_irrigation}",
            ).path
        )
        fraction_sw_irrigation_data = fraction_sw_irrigation_data.isel(
            band=0,
            **get_window(
                fraction_sw_irrigation_data.x,
                fraction_sw_irrigation_data.y,
                self.bounds,
                buffer=5,
            ),
        ).raster.interpolate_na()

        fraction_gw_irrigation = "aeigw"
        fraction_gw_irrigation_data = xr.open_dataarray(
            self.data_catalog.get_source(
                f"global_irrigation_area_{fraction_gw_irrigation}",
            ).path
        )
        fraction_gw_irrigation_data = fraction_gw_irrigation_data.isel(
            band=0,
            **get_window(
                fraction_gw_irrigation_data.x,
                fraction_gw_irrigation_data.y,
                self.bounds,
                buffer=5,
            ),
        ).raster.interpolate_na()

        farmer_locations = get_farm_locations(
            self.subgrid["agents/farmers/farms"], method="centroid"
        )

        # Determine which farmers are irrigating
        grid_id_da = self.get_linear_indices(fraction_sw_irrigation_data)
        ny, nx = (
            fraction_sw_irrigation_data.sizes["y"],
            fraction_sw_irrigation_data.sizes["x"],
        )

        n_cells = grid_id_da.max().item()
        n_farmers = self.array["agents/farmers/id"].size

        farmer_cells = sample_from_map(
            grid_id_da.values,
            farmer_locations,
            grid_id_da.rio.transform(recalc=True).to_gdal(),
        )
        fraction_sw_irrigation_farmers = sample_from_map(
            fraction_sw_irrigation_data.values,
            farmer_locations,
            fraction_sw_irrigation_data.rio.transform(recalc=True).to_gdal(),
        )
        fraction_gw_irrigation_farmers = sample_from_map(
            fraction_gw_irrigation_data.values,
            farmer_locations,
            fraction_gw_irrigation_data.rio.transform(recalc=True).to_gdal(),
        )

        adaptations = np.full(
            (
                n_farmers,
                max(
                    [
                        SURFACE_IRRIGATION_EQUIPMENT,
                        WELL_ADAPTATION,
                        IRRIGATION_EFFICIENCY_ADAPTATION,
                        FIELD_EXPANSION_ADAPTATION,
                        PERSONAL_INSURANCE_ADAPTATION,
                        INDEX_INSURANCE_ADAPTATION,
                    ]
                )
                + 1,
            ),
            -1,
            dtype=np.int32,
        )

        for i in range(n_cells):
            farmers_cell_mask = farmer_cells == i  # Boolean mask for farmers in cell i
            farmers_cell_indices = np.where(farmers_cell_mask)[0]  # Absolute indices

            irrigating_farmers_mask = irrigating_farmers[farmers_cell_mask]
            num_irrigating_farmers = np.sum(irrigating_farmers_mask)

            if num_irrigating_farmers > 0:
                fraction_sw = fraction_sw_irrigation_farmers[farmers_cell_mask][0]
                fraction_gw = fraction_gw_irrigation_farmers[farmers_cell_mask][0]

                # Normalize fractions
                total_fraction = fraction_sw + fraction_gw

                # Handle edge cases if there are irrigating farmers but no data on sw/gw
                if total_fraction == 0:
                    # Find neighboring cells with valid data
                    neighbor_ids = self.get_neighbor_cell_ids_for_linear_indices(
                        i, nx, ny
                    )
                    found_valid_neighbor = False

                    for neighbor_id in neighbor_ids:
                        if neighbor_id not in np.unique(farmer_cells):
                            continue

                        neighbor_mask = farmer_cells == neighbor_id
                        fraction_sw_neighbor = fraction_sw_irrigation_farmers[
                            neighbor_mask
                        ][0]
                        fraction_gw_neighbor = fraction_gw_irrigation_farmers[
                            neighbor_mask
                        ][0]
                        neighbor_total_fraction = (
                            fraction_sw_neighbor + fraction_gw_neighbor
                        )

                        if neighbor_total_fraction > 0:
                            # Found valid neighbor
                            fraction_sw = fraction_sw_neighbor
                            fraction_gw = fraction_gw_neighbor
                            total_fraction = neighbor_total_fraction

                            found_valid_neighbor = True
                            break
                    if not found_valid_neighbor:
                        # No valid neighboring cells found, handle accordingly
                        print(f"No valid data found for cell {i} and its neighbors.")
                        continue  # Skip this cell

                # Normalize fractions
                probabilities = np.array([fraction_sw, fraction_gw], dtype=np.float64)
                probabilities_sum = probabilities.sum()
                probabilities /= probabilities_sum

                # Indices of irrigating farmers in the region (absolute indices)
                farmer_indices_in_region = farmers_cell_indices[irrigating_farmers_mask]

                # Assign irrigation sources using np.random.choice
                irrigation_equipment_per_farmer = np.random.choice(
                    [SURFACE_IRRIGATION_EQUIPMENT, WELL_ADAPTATION],
                    size=len(farmer_indices_in_region),
                    p=probabilities,
                )

                adaptations[
                    farmer_indices_in_region, irrigation_equipment_per_farmer
                ] = 1

        self.set_array(adaptations, name="agents/farmers/adaptations")

    def setup_assets(self, feature_types, source="geofabrik", overwrite=False):
        """Get assets from OpenStreetMap (OSM) data.

        Parameters
        ----------
        feature_types : str or list of str
            The types of features to download from OSM. Available feature types are 'buildings', 'rails' and 'roads'.
        source : str, optional
            The source of the OSM data. Options are 'geofabrik' or 'movisda'. Default is 'geofabrik'.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
        """
        if isinstance(feature_types, str):
            feature_types = [feature_types]

        OSM_data_dir = self.preprocessing_dir / "osm"
        OSM_data_dir.mkdir(exist_ok=True, parents=True)

        if source == "geofabrik":
            index_file = OSM_data_dir / "geofabrik_region_index.geojson"
            fetch_and_save(
                "https://download.geofabrik.de/index-v1.json",
                index_file,
                overwrite=overwrite,
            )

            index = gpd.read_file(index_file)
            # remove Dach region as all individual regions within dach countries are also in the index
            index = index[index["id"] != "dach"]

            # find all regions that intersect with the bbox
            intersecting_regions = index[index.intersects(self.region.geometry[0])]

            def filter_regions(ID, parents):
                return ID not in parents

            intersecting_regions = intersecting_regions[
                intersecting_regions["id"].apply(
                    lambda x: filter_regions(x, intersecting_regions["parent"].tolist())
                )
            ]

            urls = (
                intersecting_regions["urls"]
                .apply(lambda x: json.loads(x)["pbf"])
                .tolist()
            )

        elif source == "movisda":
            minx, miny, maxx, maxy = self.bounds

            urls = []
            for x in range(int(minx), int(maxx) + 1):
                # Movisda seems to switch the W and E for the x coordinate
                EW_code = f"E{-x:03d}" if x < 0 else f"W{x:03d}"
                for y in range(int(miny), int(maxy) + 1):
                    NS_code = f"N{y:02d}" if y >= 0 else f"S{-y:02d}"
                    url = f"https://osm.download.movisda.io/grid/{NS_code}{EW_code}-latest.osm.pbf"

                    # some tiles do not exists because they are in the ocean. Therefore we check if they exist
                    # before adding the url
                    response = requests.head(url, allow_redirects=True)
                    if response.status_code != 404:
                        urls.append(url)

        else:
            raise ValueError(f"Unknown source {source}")

        # download all regions
        all_features = {}
        for url in tqdm(urls):
            filepath = OSM_data_dir / url.split("/")[-1]
            fetch_and_save(url, filepath, overwrite=overwrite)
            for feature_type in feature_types:
                if feature_type not in all_features:
                    all_features[feature_type] = []

                if feature_type == "buildings":
                    features = gpd.read_file(
                        filepath,
                        mask=self.region,
                        layer="multipolygons",
                        use_arrow=True,
                    )
                    features = features[features["building"].notna()]
                elif feature_type == "rails":
                    features = gpd.read_file(
                        filepath,
                        mask=self.region,
                        layer="lines",
                        use_arrow=True,
                    )
                    features = features[
                        features["railway"].isin(
                            ["rail", "tram", "subway", "light_rail", "narrow_gauge"]
                        )
                    ]
                elif feature_type == "roads":
                    features = gpd.read_file(
                        filepath,
                        mask=self.region,
                        layer="lines",
                        use_arrow=True,
                    )
                    features = features[
                        features["highway"].isin(
                            [
                                "motorway",
                                "trunk",
                                "primary",
                                "secondary",
                                "tertiary",
                                "unclassified",
                                "residential",
                                "motorway_link",
                                "trunk_link",
                                "primary_link",
                                "secondary_link",
                                "tertiary_link",
                            ]
                        )
                    ]
                else:
                    raise ValueError(f"Unknown feature type {feature_type}")

                all_features[feature_type].append(features)

        for feature_type in feature_types:
            features = pd.concat(all_features[feature_type], ignore_index=True)
            self.set_geoms(features, name=f"assets/{feature_type}")
