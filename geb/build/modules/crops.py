"""Crops data processing and setup methods for GEB."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from geb.build.methods import build_method
from geb.workflows.io import get_window
from geb.workflows.raster import sample_from_map

from ..workflows.conversions import (
    GLOBIOM_NAME_TO_ISO3,
    M49_to_ISO3,
    setup_donor_countries,
)
from ..workflows.crop_calendars import parse_MIRCA2000_crop_calendar
from ..workflows.farmers import get_farm_locations


class Crops:
    """Contains all build methods for setting up crops for GEB."""

    def __init__(self) -> None:
        """Initialize the Crops module."""
        pass

    @build_method(depends_on=[])
    def setup_crops(
        self,
        crop_data: dict,
        type: str = "MIRCA2000",
    ) -> None:
        """Validate and set crop data used by the model.

        Requires crop data to contain specific fields depending on the type.

        For both types, the following fields are required:
        - name
        - reference_yield_kg_m2
        - is_paddy (whether the crop is a paddy irrigated)
        - rd_rain (maximum root depth for rainfed crops)
        - rd_irr (maximum root depth for irrigated crops)
        - crop_group_number (adaptation level to drought, 0-5). See WOFOST documentation.

        For 'GAEZ' type, the following additional fields are required:
        - d1, d2a, d2b, d3a, d3b, d4 (lengths of growth stages)
        - KyT, Ky1, Ky2a, Ky2b, Ky3a, Ky3b, Ky4 (crop coefficients for growth stages)

        For 'MIRCA2000' type, the following additional fields are required:
        - a, b, P0, P1 (parameters for yield response to water)
        - l_ini, l_dev, l_mid, l_late (lengths of growth stages)
        - kc_initial, kc_mid, kc_end (crop coefficients for growth stages)

        Args:
            crop_data: Dictionary keyed by crop id with metadata for each crop.
            type: Source/type of crop parameters ('MIRCA2000' or 'GAEZ').

        """
        assert type in ("MIRCA2000", "GAEZ")
        for crop_id, crop_values in crop_data.items():
            assert "name" in crop_values
            assert "reference_yield_kg_m2" in crop_values
            assert "is_paddy" in crop_values
            assert "rd_rain" in crop_values  # root depth rainfed crops
            assert "rd_irr" in crop_values  # root depth irrigated crops
            assert (
                "crop_group_number" in crop_values
            )  # adaptation level to drought (see WOFOST: https://wofost.readthedocs.io/en/7.2/)
            assert 5 >= crop_values["crop_group_number"] >= 0
            assert (
                crop_values["rd_rain"] >= crop_values["rd_irr"]
            )  # root depth rainfed crops should be larger than irrigated crops

            if type == "GAEZ":
                crop_values["l_ini"] = crop_values["d1"]
                crop_values["l_dev"] = crop_values["d2a"] + crop_values["d2b"]
                crop_values["l_mid"] = crop_values["d3a"] + crop_values["d3b"]
                crop_values["l_late"] = crop_values["d4"]

                assert "KyT" in crop_values
                assert "Ky1" in crop_values
                assert "Ky2a" in crop_values
                assert "Ky2b" in crop_values
                assert "Ky3a" in crop_values
                assert "Ky3b" in crop_values
                assert "Ky4" in crop_values

            elif type == "MIRCA2000":
                assert "a" in crop_values
                assert "b" in crop_values
                assert "P0" in crop_values
                assert "P1" in crop_values
                assert "l_ini" in crop_values
                assert "l_dev" in crop_values
                assert "l_mid" in crop_values
                assert "l_late" in crop_values
                assert "kc_initial" in crop_values
                assert "kc_mid" in crop_values
                assert "kc_end" in crop_values

            assert (
                crop_values["l_ini"]
                + crop_values["l_dev"]
                + crop_values["l_mid"]
                + crop_values["l_late"]
                == 100
            ), "Sum of l_ini, l_dev, l_mid, and l_late must be 100[%]"

        crop_data = {
            "data": crop_data,
            "type": type,
        }

        self.set_dict(crop_data, name="crops/crop_data")

    @build_method(depends_on=[])
    def setup_crops_from_source(
        self,
        source: str | None = "MIRCA2000",
        crop_specifier: str | None = None,
    ) -> None:
        """Sets up the crops data for the model."""
        self.logger.info("Preparing crops data")

        assert source in ("MIRCA2000",), (
            f"crop_variables_source {source} not understood, must be 'MIRCA2000'"
        )
        if crop_specifier is None:
            crop_data = {
                "data": (
                    self.data_catalog.get_dataframe("MIRCA2000_crop_data")
                    .set_index("id")
                    .to_dict(orient="index")
                ),
                "type": "MIRCA2000",
            }
        else:
            crop_data = {
                "data": (
                    self.data_catalog.get_dataframe(
                        f"MIRCA2000_crop_data_{crop_specifier}"
                    )
                    .set_index("id")
                    .to_dict(orient="index")
                ),
                "type": "MIRCA2000",
            }

        self.set_dict(crop_data, name="crops/crop_data")

    def process_crop_data(
        self,
        crop_prices: str | int | float,
        translate_crop_names: dict | None = None,
        adjust_currency: bool = False,
    ) -> dict[str, Any]:
        """Process crop price inputs into model-ready time series or constants.

        Args:
            crop_prices: Either 'FAO_stat' to fetch FAO data, a path to JSON prices,
                or a constant numeric price (USD/kg, nominal for the years in question).
            translate_crop_names: Optional mapping from model crop name to list/alias used in source.
            adjust_currency: Whether to convert to USD using currency conversion when available.

        Returns:
            A dictionary with either type='time_series' and per-region series or type='constant'.

        Raises:
            ValueError: If crop_prices is not a valid path, number, or 'FAO_stat'.

        Notes:
            The function performs the following steps:
            1. Fetches and processes crop data from FAO statistics if crop_prices is 'FAO_stat'.
            2. Adjusts the data for countries with missing values using PPP conversion rates.
            3. Determines price variability and performs interpolation/extrapolation of crop prices.
            4. Formats the processed data into a nested dictionary structure.
        """
        if crop_prices == "FAO_stat":
            crop_data = self.data_catalog.get_dataframe(
                "FAO_crop_price",
                variables=["Area Code (M49)", "year", "crop", "price_per_kg"],
            )

            # Dropping 58 (Belgium-Luxembourg combined), 200 (former Czechoslovakia),
            # 230 (old code Ethiopia), 891 (Serbia and Montenegro), 736 (former Sudan)
            crop_data = crop_data[
                ~crop_data["Area Code (M49)"].isin([58, 200, 230, 891, 736])
            ]

            crop_data["ISO3"] = crop_data["Area Code (M49)"].map(M49_to_ISO3)
            crop_data = crop_data.drop(columns=["Area Code (M49)"])

            crop_data["crop"] = crop_data["crop"].str.lower()

            assert not crop_data["ISO3"].isna().any(), "Missing ISO3 codes"

            all_years = crop_data["year"].unique()
            all_years.sort()
            all_crops = crop_data["crop"].unique()

            GLOBIOM_regions = self.data_catalog.get_dataframe("GLOBIOM_regions_37")
            GLOBIOM_regions["ISO3"] = GLOBIOM_regions["Country"].map(
                GLOBIOM_NAME_TO_ISO3
            )
            assert not np.any(GLOBIOM_regions["ISO3"].isna()), "Missing ISO3 codes"

            ISO3_codes_region = self.geom["regions"]["ISO3"].unique()
            GLOBIOM_regions_region = GLOBIOM_regions[
                GLOBIOM_regions["ISO3"].isin(ISO3_codes_region)
            ]["Region37"].unique()
            ISO3_codes_GLOBIOM_region = GLOBIOM_regions[
                GLOBIOM_regions["Region37"].isin(GLOBIOM_regions_region)
            ]["ISO3"]

            missing_regions_in_GLOBIOM = set(ISO3_codes_region) - set(
                ISO3_codes_GLOBIOM_region
            )
            if len(missing_regions_in_GLOBIOM) > 0:
                self.logger.info(
                    f"Regions in the model not present in GLOBIOM: {list(missing_regions_in_GLOBIOM)}"
                )
            for region in missing_regions_in_GLOBIOM:
                if not crop_data[crop_data["ISO3"] == region].empty:
                    raise ValueError(
                        f"Region {region} is not present in GLOBIOM, but it has crop data. This situation gives problems in the donate_and_receive_crop_prices function, because it will substitute the region's data for donor data. Please consult Tim to change the function"
                    )

            # Setup dataFrame for further data corrections
            donor_data = {}
            for ISO3 in ISO3_codes_GLOBIOM_region:
                region_crop_data = crop_data[crop_data["ISO3"] == ISO3]
                region_pivot = region_crop_data.pivot_table(
                    index="year",
                    columns="crop",
                    values="price_per_kg",
                    aggfunc="first",
                ).reindex(index=all_years, columns=all_crops)

                region_pivot["ISO3"] = ISO3
                # Store pivoted data in dictionary with region_id as key
                donor_data[ISO3] = region_pivot

            # Concatenate all regional data into a single DataFrame with MultiIndex
            donor_data = pd.concat(donor_data, names=["ISO3", "year"])

            # Drop crops with no data at all for these regions
            donor_data = donor_data.dropna(axis=1, how="all")

            # Filter out columns that contain the word 'meat'
            donor_data = donor_data[
                [
                    column
                    for column in donor_data.columns
                    if "meat" not in column.lower()
                ]
            ]

            national_data = False
            # Check whether there is national or subnational data
            duplicates = donor_data.index.duplicated(keep=False)
            if duplicates.any():
                # Data is subnational
                unique_regions = self.geom["regions"]
            else:
                # Data is national
                unique_regions = (
                    self.geom["regions"].groupby("ISO3").first().reset_index()
                )
                national_data = True

            # filter for model start and end year (important to do this before donation)
            donor_data = donor_data.loc[
                (slice(None), slice(self.start_date.year, self.end_date.year)), :
            ]

            data = self.donate_and_receive_crop_prices(
                donor_data, unique_regions, GLOBIOM_regions
            )

            # exand data to include all data empty rows from start to end year
            data = data.reindex(
                pd.MultiIndex.from_product(
                    [
                        unique_regions["region_id"],
                        range(self.start_date.year, self.end_date.year + 1),
                    ],
                    names=["region_id", "year"],
                )
            )

            data = self.assign_crop_price_inflation(data, unique_regions)

            # combine and rename crops
            all_crop_names_model = [
                d["name"] for d in self.dict["crops/crop_data"]["data"].values()
            ]
            for crop_name in all_crop_names_model:
                if (
                    translate_crop_names is not None
                    and crop_name in translate_crop_names
                ):
                    sub_crops = [
                        crop
                        for crop in translate_crop_names[crop_name]
                        if crop in data.columns
                    ]
                    if sub_crops:
                        data[crop_name] = data[sub_crops].mean(axis=1, skipna=True)
                    else:
                        data[crop_name] = np.nan
                else:
                    if crop_name not in data.columns:
                        data[crop_name] = np.nan

            # Extract the crop names from the dictionary and convert them to lowercase
            crop_names = [
                crop["name"].lower()
                for crop in self.dict["crops/crop_data"]["data"].values()
            ]

            # Filter the columns of the data DataFrame
            data = data[
                [
                    col
                    for col in data.columns
                    if col.lower() in crop_names
                    or col in ("_crop_price_inflation", "_crop_price_LCU_USD")
                ]
            ]

            data = self.inter_and_extrapolate_prices(data, unique_regions)

            # Create a dictionary structure with regions as keys and crops as nested dictionaries
            # This is the required format for crop_farmers.py
            crop_data = self.dict["crops/crop_data"]["data"]
            formatted_data = {
                "type": "time_series",
                "data": {},
                "time": data.index.get_level_values("year")
                .unique()
                .tolist(),  # Extract unique years for the time key
            }

            # If national_data is True, create a mapping from ISO3 code to representative region_id
            if national_data:
                unique_regions = data.index.get_level_values("region_id").unique()
                iso3_codes = (
                    self.geom["regions"]
                    .set_index("region_id")
                    .loc[unique_regions]["ISO3"]
                )
                iso3_to_representative_region_id = dict(zip(iso3_codes, unique_regions))

            for _, region in self.geom["regions"].iterrows():
                region_dict = {}
                region_id = region["region_id"]
                region_iso3 = region["ISO3"]

                # Determine the region_id to use based on national_data
                if national_data:
                    # Use the representative region_id for this ISO3 code
                    selected_region_id = iso3_to_representative_region_id.get(
                        region_iso3
                    )
                else:
                    # Use the actual region_id
                    selected_region_id = region_id

                # Fetch the data for the selected region_id
                if selected_region_id in data.index.get_level_values("region_id"):
                    region_data = data.loc[selected_region_id]
                else:
                    # If data is not available for the region, fill with NaNs
                    region_data = pd.DataFrame(
                        np.nan, index=formatted_data["time"], columns=data.columns
                    )

                region_data.index.name = "year"  # Ensure index name is 'year'

                crop_calendars_in_region = self.array["agents/farmers/crop_calendar"][
                    self.array["agents/farmers/region_id"] == region_id
                ]
                crops_in_region = crop_calendars_in_region[..., 0].ravel()
                crops_in_region = np.unique(crops_in_region[crops_in_region != -1])

                # Ensuring all crops are present according to the crop_data keys
                for crop_id, crop_info in crop_data.items():
                    crop_name = crop_info["name"]

                    if crop_name.endswith("_flood") or crop_name.endswith("_drought"):
                        crop_name = crop_name.rsplit("_", 1)[0]

                    if crop_name in region_data.columns:
                        # raise an error if the crop is in the crop calendar and has NaN values
                        if (
                            float(crop_id) in crops_in_region
                            and np.isnan(region_data[crop_name]).any()
                        ):
                            raise ValueError(
                                f"Crop {crop_name} has NaN values in region {region_id} data."
                            )
                        region_dict[str(crop_id)] = region_data[crop_name].tolist()
                    # check if crop is in the crop calendar, if is raise an error because it must be
                    elif crop_id in crops_in_region:
                        raise ValueError(
                            f"Crop {crop_name} not found in region {region_id} data, but is in crop calendar."
                        )
                    else:
                        # If data is not available for the crop, but is not in the crop calendar, it
                        # is no issue, so we can fill with NaNs
                        region_dict[str(crop_id)] = [np.nan] * len(
                            formatted_data["time"]
                        )

                formatted_data["data"][str(region_id)] = region_dict

            data = formatted_data.copy()

        # data is a file path
        elif isinstance(crop_prices, str):
            crop_prices = Path(crop_prices)
            if not crop_prices.exists():
                raise ValueError(f"file {crop_prices.resolve()} does not exist")
            with open(crop_prices) as f:
                data = json.load(f)
            data = pd.DataFrame(
                {
                    crop_id: data["crops"][crop_data["name"]]
                    for crop_id, crop_data in self.dict["crops/crop_data"][
                        "data"
                    ].items()
                },
                index=pd.to_datetime(data["time"]),
            )
            # compute mean price per year, using start day as index
            data = data.resample("AS").mean()
            # extend dataframe to include start and end years
            data = data.reindex(
                index=pd.date_range(
                    start=self.start_date,
                    end=self.end_date,
                    freq="YS",
                )
            )
            # only use year identifier as index
            data.index = data.index.year

            data = data.reindex(
                index=pd.MultiIndex.from_product(
                    [
                        self.geom["regions"]["region_id"],
                        data.index,
                    ],
                    names=["region_id", "date"],
                ),
                level=1,
            )

            data = self.assign_crop_price_inflation(data, self.geom["regions"])
            data = self.inter_and_extrapolate_prices(
                data, self.geom["regions"], adjust_currency
            )

            data = {
                "type": "time_series",
                "time": data.xs(
                    data.index.get_level_values(0)[0], level=0
                ).index.tolist(),
                "data": {
                    str(region_id): data.loc[region_id].to_dict(orient="list")
                    for region_id in self.geom["regions"]["region_id"]
                },
            }

        elif isinstance(crop_prices, (int, float)):
            data = {
                "type": "constant",
                "data": crop_prices,
            }
        else:
            raise ValueError(
                f"must be a file path or an integer, got {type(crop_prices)}"
            )

        return data

    def donate_and_receive_crop_prices(
        self,
        donor_data: pd.DataFrame,
        recipient_regions: pd.DataFrame,
        GLOBIOM_regions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Gets crop prices from other regions to fill missing data.

        If there are multiple countries in one selected basin, where one country has prices for a certain crop,
        but the other does not, this gives issues. This function adjusts crop data for those countries by
        filling in missing values using data from nearby regions and PPP conversion rates. In case crop data
        is missing for a country, and is also not in countries in the same GLOBIOM dataset, it uses the prices
        for that crop from the country in the model region with least nan values.

        Args:
            donor_data: A DataFrame containing crop data with a 'ISO3' column and indexed by 'region_id'.
                The DataFrame contains crop prices for different regions.
            recipient_regions: DataFrame containing recipient region information with 'region_id' and 'ISO3' columns.
            GLOBIOM_regions: DataFrame containing GLOBIOM region mapping with 'ISO3' and 'Region37' columns.

        Returns:
            The updated DataFrame with missing crop data filled in using PPP conversion rates from nearby regions.

        Notes:
            The function performs the following steps:
            1. Identifies columns where all values are NaN for each country and stores this information.
            2. For each country and column with missing values, finds a country/region within that study area that has data for that column.
            3. Uses PPP conversion rates to adjust and fill in missing values for regions without data.
            4. Drops the 'ISO3' column before returning the updated DataFrame.

            Some countries without data are also not in the GLOBIOM dataset (e.g Liechtenstein (LIE)).
            For these countries, we cannot assess which donor we should take, and the country_data will be
            empty for these countries. Therefore, we first estimate the most similar country based on the
            setup_donor_countries function. The ISO3 of these countries will be replaced by the ISO3 of the donor.
            However, the region_id will remain the same, so that only the data is used from the donor,
            but still the original region is used.
        """
        # create a copy of the data to avoid using data that was adjusted in this function
        data_out = None

        for _, region in recipient_regions.iterrows():
            ISO3 = region["ISO3"]
            region_id = region["region_id"]
            self.logger.info(f"Processing region {region_id}")

            # Filter the data for the current country
            country_data = donor_data[donor_data["ISO3"] == ISO3]

            if country_data.empty:  # happens if country is not in GLOBIOM regions dataset (e.g. Kosovo). Fill these countries using data from a country that is in the GLOBIOM regions dataset, using the regular donor countries setup.
                countries_with_donor_data = donor_data.ISO3.unique().tolist()
                donor_countries = setup_donor_countries(
                    self.data_catalog,
                    self.geom["global_countries"],
                    countries_with_donor_data,
                    alternative_countries=self.geom["regions"]["ISO3"]
                    .unique()
                    .tolist(),
                )
                ISO3 = donor_countries.get(ISO3, None)
                self.logger.info(
                    f"Missing price donor data for {region['ISO3']}, using donor country {ISO3}. This country is NOT in the GLOBIOM regions dataset"
                )
                assert ISO3 is not None, (
                    f"Could not find a donor country for {region['ISO3']}. Please check the donor countries setup."
                )

                country_data = donor_data[donor_data["ISO3"] == ISO3]

                assert not country_data.empty, (
                    f"Donor country {ISO3} has no data for {region['ISO3']}. Please check the donor countries setup."
                )
                # note: it can be that a country is donor for another in the first donor step (outside this function) (e.g. Isreal for cyprus), and that here cyprus is again selected as a donor country for another country (e.g. Liechtenstein)

            GLOBIOM_region = GLOBIOM_regions.loc[
                GLOBIOM_regions["ISO3"] == ISO3, "Region37"
            ].item()

            assert len(GLOBIOM_region) > 0, (
                f"GLOBIOM region for {ISO3} is empty. Please check the GLOBIOM regions setup."
            )

            GLOBIOM_region_countries = GLOBIOM_regions.loc[
                GLOBIOM_regions["Region37"] == GLOBIOM_region, "ISO3"
            ]

            for column in country_data.columns:
                if country_data[column].isna().all():
                    donor_data_region = donor_data.loc[
                        donor_data["ISO3"].isin(GLOBIOM_region_countries), column
                    ]

                    # Check if data is available within the GLOBIOM region
                    non_na_values = donor_data_region.groupby("ISO3").count()

                    if (
                        non_na_values.max() > 0
                    ):  # if there is at least one non-NaN value
                        donor_country = non_na_values.idxmax()
                        donor_data_country = donor_data_region[donor_country]

                    else:
                        # if no data is available, take the country with most non-nan values
                        donor_data_crop = donor_data[column]
                        donor_data_crop = donor_data_crop.reset_index()
                        donor_data_crop = donor_data_crop.set_index("year")
                        amount_of_non_na = donor_data_crop.groupby("ISO3").count()
                        donor_country = amount_of_non_na[column].idxmax()
                        donor_data_country = donor_data_crop.loc[
                            donor_data_crop["ISO3"] == donor_country, column
                        ]

                    new_data = pd.DataFrame(
                        donor_data_country.values,
                        index=pd.MultiIndex.from_product(
                            [[region["region_id"]], donor_data_country.index],
                            names=["region_id", "year"],
                        ),
                        columns=[donor_data_country.name],
                    )
                    if data_out is None:
                        data_out = new_data.copy()
                    else:
                        data_out = data_out.combine_first(new_data)

                else:
                    new_data = pd.DataFrame(
                        country_data[column].values,
                        index=pd.MultiIndex.from_product(
                            [
                                [region["region_id"]],
                                country_data.droplevel(level=0).index,
                            ],
                            names=["region_id", "year"],
                        ),
                        columns=[column],
                    )
                    if data_out is None:
                        data_out = new_data.copy()
                    else:
                        data_out = data_out.combine_first(new_data)

        data_out = data_out.drop(columns=["ISO3"])
        data_out = data_out.dropna(axis=1, how="all")
        data_out = data_out.dropna(axis=0, how="all")

        return data_out

    def assign_crop_price_inflation(
        self, costs: pd.DataFrame, unique_regions: pd.DataFrame
    ) -> pd.DataFrame:
        """Determines the price inflation of all crops in the region and adds a column that describes this inflation.

        If there is no data for a certain year, the inflation rate is taken from the socioeconomics data.

        Args:
            costs: A DataFrame containing crop prices for different regions. The DataFrame should be indexed by region IDs.
            unique_regions: A DataFrame containing unique regions with their IDs and other attributes.

        Returns:
            The updated DataFrame with a new column 'changes' that contains the average price changes for each region.

        To Do:
            Is it possible to use the regions from the costs DataFrame instead of the unique_regions DataFrame?

        """
        costs["_crop_price_inflation"] = np.nan
        costs["_crop_price_LCU_USD"] = np.nan

        # Determine the average changes of price of all crops in the region and add it to the data
        for _, region in unique_regions.iterrows():
            region_id = region["region_id"]
            region_data = costs.loc[region_id]
            changes = np.nanmean(
                region_data[1:].to_numpy() / region_data[:-1].to_numpy(), axis=1
            )

            changes = np.insert(changes, 0, np.nan)
            costs.at[region_id, "_crop_price_inflation"] = changes

            years_with_no_crop_inflation_data = costs.loc[
                region_id, "_crop_price_inflation"
            ]
            years_with_no_crop_inflation_data = costs.loc[
                region_id, "_crop_price_inflation"
            ]
            region_inflation_rates = self.dict["socioeconomics/inflation_rates"][
                "data"
            ][str(region["region_id"])]
            region_currency_conversion_rates = self.dict["socioeconomics/LCU_per_USD"][
                "data"
            ][str(region["region_id"])]

            for year, crop_inflation_rate in years_with_no_crop_inflation_data.items():
                year_currency_conversion = region_currency_conversion_rates[
                    self.dict["socioeconomics/LCU_per_USD"]["time"].index(str(year))
                ]
                costs.at[(region_id, year), "_crop_price_LCU_USD"] = (
                    year_currency_conversion
                )
                if np.isnan(crop_inflation_rate):
                    year_inflation_rate = region_inflation_rates[
                        self.dict["socioeconomics/inflation_rates"]["time"].index(
                            str(year)
                        )
                    ]
                    costs.at[(region_id, year), "_crop_price_inflation"] = (
                        year_inflation_rate
                    )

        return costs

    def inter_and_extrapolate_prices(
        self,
        data: pd.DataFrame,
        unique_regions: pd.DataFrame,
        adjust_currency: bool = False,
    ) -> pd.DataFrame:
        """Interpolates and extrapolates crop prices for different regions based on the given data and predefined crop categories.

        Args:
            data: A DataFrame containing crop price data for different regions. The DataFrame should be indexed by region IDs
            and have columns corresponding to different crops.
            unique_regions: A DataFrame containing unique regions with their IDs and other attributes.
            adjust_currency: If True, adjusts the crop prices based on currency conversion rates.

        Returns:
            Updated DataFrame with interpolated and extrapolated crop prices. Columns for 'others perennial' and 'others annual'
                crops are also added.

        Notes:
            The function performs the following steps:
                1. Extracts crop names from the internal crop data dictionary.
                2. Defines additional crops that fall under 'others perennial' and 'others annual' categories.
                3. Processes the data to compute average prices for these additional crops.
                4. Filters and updates the original data with the computed averages.
                5. Interpolates and extrapolates missing prices for each crop in each region based on the 'changes' column.

        To Do:
            Ensure adjust_currency is better explained and used correctly.
        """
        # Interpolate and extrapolate missing prices for each crop in each region based on the 'changes' column
        for _, region in unique_regions.iterrows():
            region_id = region["region_id"]
            region_data = data.loc[region_id]

            n = len(region_data)

            for crop in region_data.columns:
                if crop == "_crop_price_inflation":
                    continue
                crop_data = region_data[crop].to_numpy()
                if np.isnan(crop_data).all():
                    continue
                changes_data = region_data["_crop_price_inflation"].to_numpy()
                k = -1
                while np.isnan(crop_data[k]):
                    k -= 1
                for i in range(k + 1, 0, 1):
                    crop_data[i] = crop_data[i - 1] * changes_data[i]
                k = 0
                while np.isnan(crop_data[k]):
                    k += 1
                for i in range(k - 1, -1, -1):
                    crop_data[i] = crop_data[i + 1] / changes_data[i + 1]
                for j in range(0, n):
                    if np.isnan(crop_data[j]):
                        k = j
                        while np.isnan(crop_data[k]):
                            k += 1
                        empty_size = k - j
                        step_crop_price_inflation = changes_data[j : k + 1]
                        total_crop_price_inflation = np.prod(step_crop_price_inflation)
                        real_crop_price_inflation = crop_data[k] / crop_data[j - 1]
                        scaled_crop_price_inflation = (
                            step_crop_price_inflation
                            * (real_crop_price_inflation ** (1 / empty_size))
                            / (total_crop_price_inflation ** (1 / empty_size))
                        )
                        for i, change in zip(range(j, k), scaled_crop_price_inflation):
                            crop_data[i] = crop_data[i - 1] * change
                if adjust_currency and not crop == "_crop_price_LCU_USD":
                    conversion_data = region_data["_crop_price_LCU_USD"].to_numpy()
                    data.loc[region_id, crop] = crop_data / conversion_data
                else:
                    data.loc[region_id, crop] = crop_data

        # remove columns that are not needed anymore
        data = data.drop(columns=["_crop_price_inflation"])
        data = data.drop(columns=["_crop_price_LCU_USD"])

        return data

    @build_method(depends_on=["set_time_range"])
    def setup_cultivation_costs(
        self,
        cultivation_costs: str | int | float = 0,
        translate_crop_names: dict[str, str] | None = None,
        adjust_currency: bool = False,
    ) -> None:
        """Set cultivation costs per crop and region for the model run.

        Args:
            cultivation_costs: 'FAO_stat', file path, or constant (USD/kg, nominal).
            translate_crop_names: Optional mapping to aggregate/rename source crop columns.
            adjust_currency: Whether to convert to USD using currency conversion when available.
        """
        cultivation_costs = self.process_crop_data(
            crop_prices=cultivation_costs,
            translate_crop_names=translate_crop_names,
            adjust_currency=adjust_currency,
        )
        self.set_dict(cultivation_costs, name="crops/cultivation_costs")

    @build_method(
        depends_on=[
            "set_time_range",
            "setup_regions_and_land_use",
            "setup_economic_data",
            "setup_crops_from_source",
            "setup_farmer_crop_calendar",
        ]
    )
    def setup_crop_prices(
        self,
        crop_prices: str | int | float = "FAO_stat",
        translate_crop_names: dict[str, str] | None = None,
        adjust_currency: bool = False,
    ) -> None:
        """Set crop prices per crop and region for the model run.

        Args:
            crop_prices: 'FAO_stat', file path, or constant (USD/kg, nominal).
            translate_crop_names: Optional mapping to aggregate/rename source crop columns.
            adjust_currency: Whether to convert to USD using currency conversion when available.
        """
        crop_prices = self.process_crop_data(
            crop_prices=crop_prices,
            translate_crop_names=translate_crop_names,
            adjust_currency=adjust_currency,
        )
        self.set_dict(crop_prices, name="crops/crop_prices")
        self.set_dict(crop_prices, name="crops/cultivation_costs")

    @build_method(depends_on=[])
    def determine_crop_area_fractions(self, resolution: str = "5-arcminute") -> None:
        """Compute MIRCA crop area fractions and summarize per region.

        Args:
            resolution: Resolution tag for plotting/output naming.
        """
        output_folder = "plot/mirca_crops"
        os.makedirs(output_folder, exist_ok=True)

        crops = [
            "Wheat",  # 0
            "Maize",  # 1
            "Rice",  # 2
            "Barley",  # 3
            "Rye",  # 4
            "Millet",  # 5
            "Sorghum",  # 6
            "Soybeans",  # 7
            "Sunflower",  # 8
            "Potatoes",  # 9
            "Cassava",  # 10
            "Sugar_cane",  # 11
            "Sugar_beet",  # 12
            "Oil_palm",  # 13
            "Rapeseed",  # 14
            "Groundnuts",  # 15
            "Others_perennial",  # 23
            "Fodder",  # 24
            "Others_annual",  # 25,
        ]

        years = ["2000", "2005", "2010", "2015"]
        irrigation_types = ["ir", "rf"]

        # Initialize lists to collect DataArrays across years
        fraction_da_list = []
        irrigated_fraction_da_list = []

        # Initialize a dictionary to store datasets
        crop_data = {}

        for year in years:
            crop_data[year] = {}
            for crop in crops:
                crop_data[year][crop] = {}
                for irrigation in irrigation_types:
                    dataset_name = f"MIRCA-OS_cropping_area_{year}_{resolution}_{crop}_{irrigation}"

                    crop_map = xr.open_dataarray(
                        self.data_catalog.get_source(dataset_name).path
                    )
                    crop_map = crop_map.isel(
                        **get_window(crop_map.x, crop_map.y, self.bounds, buffer=2),
                        band=0,
                    )

                    crop_map = crop_map.fillna(0)

                    crop_data[year][crop][irrigation] = crop_map.assign_coords(
                        x=np.round(crop_map.coords["x"].values, decimals=6),
                        y=np.round(crop_map.coords["y"].values, decimals=6),
                    )

            # Initialize variables for total calculations
            total_cropped_area = None
            total_crop_areas = {}

            # Calculate total crop areas and total cropped area
            for crop in crops:
                irrigated = crop_data[year][crop]["ir"]
                rainfed = crop_data[year][crop]["rf"]

                total_crop = irrigated + rainfed
                total_crop_areas[crop] = total_crop

                if total_cropped_area is None:
                    total_cropped_area = total_crop.copy()
                else:
                    total_cropped_area += total_crop

            # Initialize lists to collect DataArrays for this year
            fraction_list = []
            irrigated_fraction_list = []

            # Calculate the fraction of each crop to the total cropped area
            for crop in crops:
                fraction = total_crop_areas[crop] / total_cropped_area

                # Assign 'crop' as a coordinate
                fraction = fraction.assign_coords(crop=crop)

                # Append to the list
                fraction_list.append(fraction)

            # Concatenate the list of fractions into a single DataArray along the 'crop' dimension
            fraction_da = xr.concat(fraction_list, dim="crop")

            # Assign the 'year' coordinate and expand dimensions to include 'year'
            fraction_da = fraction_da.assign_coords(year=year).expand_dims(dim="year")

            # Append to the list of all years
            fraction_da_list.append(fraction_da)

            # Calculate irrigated fractions for each crop and collect them
            for crop in crops:
                irrigated = crop_data[year][crop]["ir"].compute()
                total_crop = total_crop_areas[crop]
                irrigated_fraction = irrigated / total_crop

                # Assign 'crop' as a coordinate
                irrigated_fraction = irrigated_fraction.assign_coords(crop=crop)

                # Append to the list
                irrigated_fraction_list.append(irrigated_fraction)

            # Concatenate the list of irrigated fractions into a single DataArray along the 'crop' dimension
            irrigated_fraction_da = xr.concat(irrigated_fraction_list, dim="crop")

            # Assign the 'year' coordinate and expand dimensions to include 'year'
            irrigated_fraction_da = irrigated_fraction_da.assign_coords(
                year=year
            ).expand_dims(dim="year")

            # Append to the list of all years
            irrigated_fraction_da_list.append(irrigated_fraction_da)

        # After processing all years, concatenate along the 'year' dimension
        all_years_fraction_da = xr.concat(fraction_da_list, dim="year")
        all_years_irrigated_fraction_da = xr.concat(
            irrigated_fraction_da_list, dim="year"
        )

        # Save the concatenated DataArrays as NetCDF files
        save_dir = self.preprocessing_dir / "crops" / "MIRCA2000"
        save_dir.mkdir(parents=True, exist_ok=True)

        all_years_fraction_da.to_netcdf(save_dir / "crop_area_fraction_all_years.nc")
        all_years_irrigated_fraction_da.to_netcdf(
            save_dir / "crop_irrigated_fraction_all_years.nc"
        )

    @build_method(depends_on=[])
    def setup_farmer_crop_calendar_multirun(
        self,
        reduce_crops: bool = False,
        replace_base: bool = False,
        export: bool = False,
    ) -> None:
        """Generate crop calendars for multiple years for multirun scenarios."""
        years = [2000, 2005, 2010, 2015]
        nr_runs = 20

        for year_nr in years:
            for run in range(nr_runs):
                self.setup_farmer_crop_calendar(
                    year_nr, reduce_crops, replace_base, export
                )

    @build_method(depends_on=["setup_create_farms"])
    def setup_farmer_crop_calendar(
        self,
        year: int = 2000,
        reduce_crops: bool = False,
        replace_base: bool = False,
        minimum_area_ratio: float = 0.01,
        replace_crop_calendar_unit_code: dict = {},
    ) -> None:
        """Build per-farmer crop calendars for a single reference year.

        Args:
            year: Reference year (calendar year).
            reduce_crops: If True, reduce the number of crops per calendar based on area.
            replace_base: If True, replace base crop definitions with alternatives.
            minimum_area_ratio: Threshold for considering a crop present in a unit.
            replace_crop_calendar_unit_code: Optional mapping to replace MIRCA unit codes.

        Raises:
            ValueError: If no rotations are found for a crop in a unit or no valid neighbor data is found.
        """
        n_farmers = self.array["agents/farmers/id"].size

        MIRCA_unit_grid = xr.open_dataarray(
            self.data_catalog.get_source("MIRCA2000_unit_grid").path
        )

        MIRCA_unit_grid = MIRCA_unit_grid.isel(
            **get_window(MIRCA_unit_grid.x, MIRCA_unit_grid.y, self.bounds, buffer=2),
            band=0,
        )

        crop_calendar = parse_MIRCA2000_crop_calendar(
            self.data_catalog,
            MIRCA_units=np.unique(MIRCA_unit_grid.values),
        )

        def fix_365_in_crop_calendar(
            crop_calendar: dict[str, list[tuple[float, np.ndarray]]],
        ) -> None:
            """Replace any 365 day-of-year values with 364 in the 4th column.

            Scans each (area, arr) pair in every dictionary entry. If a value 365 is
            found, it asserts that it appears only in column index 3 and then rewrites
            it to 364. Increments a running count of replacements and raises a
            ValueError if a 365 is found outside column 3.

            Raises:
                ValueError: If any 365 is found outside column index 3 (the 4th column).

            Returns:
                A dictionary of crop calendars where the 365 length crops are now 364 days.
            """
            total_replacements = 0

            crop_calendar_adjusted = crop_calendar.copy()

            for key, entries in crop_calendar_adjusted.items():
                for i, (area, arr) in enumerate(entries):
                    rows, cols = np.where(arr == 365)

                    if rows.size == 0:
                        continue  # nothing to change in this array

                    # Safety: all 365s must be in column index 3 (4th column)
                    if not np.all(cols == 3):
                        raise ValueError(
                            f"Found 365 outside column 3 for key={key}, index={i}: "
                            f"indices={list(zip(rows, cols))}"
                        )

                    # Do the replacement
                    arr[rows, 3] = 364
                    entries[i] = (area, arr)
                    total_replacements += rows.size

            return crop_calendar_adjusted

        # Replace crop growth time of 365 with 364 as 365 leads to many issues
        crop_calendar = fix_365_in_crop_calendar(crop_calendar)

        if any(value in [None, "", [], {}] for value in crop_calendar.values()):
            missing_mirca_unit = [
                unit for unit, calendars in crop_calendar.items() if not calendars
            ]
            self.logger.warning(
                f"Missing crop calendar for MIRCA unit(s): {missing_mirca_unit}"
            )

            for mirca_unit in missing_mirca_unit:
                # Filter out the current mirca_unit from crop_calendar.keys()
                valid_keys = [key for key in crop_calendar.keys() if key != mirca_unit]

                # Find the closest MIRCA unit with a crop calendar
                if valid_keys:  # Ensure there are valid keys to process
                    closest_mirca_unit = min(
                        valid_keys, key=lambda x: abs(x - mirca_unit)
                    )
                else:
                    raise ValueError(
                        f"No valid MIRCA units found to replace missing crop calendar for {mirca_unit}."
                    )

                # use this closest_mirca_unit to fill the missing crop calendar
                crop_calendar[mirca_unit] = crop_calendar[closest_mirca_unit]
                self.logger.info(
                    f"Filling missing crop calendar for MIRCA unit {mirca_unit} with data from {closest_mirca_unit}."
                )

        else:
            self.logger.debug("All keys have valid values.")

        farmer_locations = get_farm_locations(
            self.subgrid["agents/farmers/farms"], method="centroid"
        )

        farmer_mirca_units = sample_from_map(
            MIRCA_unit_grid.values,
            farmer_locations,
            MIRCA_unit_grid.rio.transform(recalc=True).to_gdal(),
        )

        farmer_crops, is_irrigated = self.assign_crops(
            crop_calendar,
            farmer_locations,
            farmer_mirca_units,
            year,
            MIRCA_unit_grid,
            minimum_area_ratio=minimum_area_ratio,
            replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
        )
        self.setup_farmer_irrigation_source(is_irrigated, year)

        all_farmers_assigned = []

        crop_calendar_per_farmer = np.full((n_farmers, 3, 4), -1, dtype=np.int32)
        for mirca_unit in np.unique(farmer_mirca_units):
            farmers_in_unit = np.where(farmer_mirca_units == mirca_unit)[0]

            area_per_crop_rotation = []
            cropping_calenders_crop_rotation = []
            for crop_rotation in crop_calendar[
                replace_crop_calendar_unit_code.get(mirca_unit, mirca_unit)
            ]:
                area_per_crop_rotation.append(crop_rotation[0])
                crop_rotation_matrix = crop_rotation[1]
                starting_days = crop_rotation_matrix[:, 2]
                starting_days = starting_days[starting_days != -1]
                assert np.unique(starting_days).size == starting_days.size, (
                    "ensure all starting days are unique"
                )
                # TODO: Add check to ensure crop calendars are not overlapping.
                cropping_calenders_crop_rotation.append(crop_rotation_matrix)
            area_per_crop_rotation = np.array(area_per_crop_rotation)
            cropping_calenders_crop_rotation = np.stack(
                cropping_calenders_crop_rotation
            )

            crops_in_unit = np.unique(farmer_crops[farmers_in_unit])
            for crop_id in crops_in_unit:
                # Find rotations that include this crop
                rotations_with_crop_idx = []
                for idx, rotation in enumerate(cropping_calenders_crop_rotation):
                    # Get crop IDs in the rotation, excluding -1 entries
                    crop_ids_in_rotation = rotation[:, 0]
                    crop_ids_in_rotation = crop_ids_in_rotation[
                        crop_ids_in_rotation != -1
                    ]
                    if crop_id in crop_ids_in_rotation:
                        rotations_with_crop_idx.append(idx)

                if not rotations_with_crop_idx:
                    raise ValueError(
                        f"No rotations found for crop ID {crop_id} in mirca unit {mirca_unit}"
                    )

                # Get the area fractions and rotations for these indices
                areas_with_crop = area_per_crop_rotation[rotations_with_crop_idx]
                rotations_with_crop = cropping_calenders_crop_rotation[
                    rotations_with_crop_idx
                ]

                # Normalize the area fractions
                total_area_for_crop = areas_with_crop.sum()
                fractions = areas_with_crop / total_area_for_crop

                # Get farmers with this crop in the mirca_unit
                farmers_with_crop_in_unit = farmers_in_unit[
                    farmer_crops[farmers_in_unit] == crop_id
                ]

                # Assign crop rotations to these farmers
                assigned_rotation_indices = np.random.choice(
                    np.arange(len(rotations_with_crop)),
                    size=len(farmers_with_crop_in_unit),
                    replace=True,
                    p=fractions,
                )

                # Assign the crop calendars to the farmers
                for farmer_idx, rotation_idx in zip(
                    farmers_with_crop_in_unit, assigned_rotation_indices
                ):
                    assigned_rotation = rotations_with_crop[rotation_idx]
                    # Assign to farmer's crop calendar, taking columns [0, 2, 3, 4]
                    # Columns: [crop_id, planting_date, harvest_date, additional_attribute]
                    crop_calendar_per_farmer[farmer_idx] = assigned_rotation[
                        :, [0, 2, 3, 4]
                    ]
                    all_farmers_assigned.append(farmer_idx)

        def check_crop_calendar(crop_calendar_per_farmer: np.ndarray) -> None:
            """Validate that no overlapping crops exist per farmer calendar."""
            # this part asserts that the crop calendar is correctly set up
            # particularly that no two crops are planted at the same time
            for farmer_crop_calender in crop_calendar_per_farmer:
                farmer_crop_calender = farmer_crop_calender[
                    farmer_crop_calender[:, -1] != -1
                ]
                if farmer_crop_calender.shape[0] > 1:
                    assert (
                        np.unique(farmer_crop_calender[:, [1, 3]], axis=0).shape[0]
                        == farmer_crop_calender.shape[0]
                    )

        check_crop_calendar(crop_calendar_per_farmer)

        # Define constants for crop IDs
        WHEAT = 0
        MAIZE = 1
        RICE = 2
        BARLEY = 3
        RYE = 4
        MILLET = 5
        SORGHUM = 6
        SOYBEANS = 7
        SUNFLOWER = 8
        POTATOES = 9
        CASSAVA = 10
        SUGAR_CANE = 11
        SUGAR_BEETS = 12
        OIL_PALM = 13
        RAPESEED = 14
        GROUNDNUTS = 15
        # PULSES = 16
        # CITRUS = 17
        # # DATE_PALM = 18
        # # GRAPES = 19
        # COTTON = 20
        COCOA = 21
        COFFEE = 22
        OTHERS_PERENNIAL = 23
        FODDER_GRASSES = 24
        OTHERS_ANNUAL = 25
        WHEAT_DROUGHT = 26
        WHEAT_FLOOD = 27
        MAIZE_DROUGHT = 28
        MAIZE_FLOOD = 29
        RICE_DROUGHT = 30
        RICE_FLOOD = 31
        SOYBEANS_DROUGHT = 32
        SOYBEANS_FLOOD = 33
        POTATOES_DROUGHT = 34
        POTATOES_FLOOD = 35

        # Manual replacement of certain crops
        def replace_crop(
            crop_calendar_per_farmer: np.ndarray,
            crop_values: np.ndarray | list[int],
            replaced_crop_values: np.ndarray | list[int],
        ) -> np.ndarray:
            # Find the most common crop value among the given crop_values
            crop_instances = crop_calendar_per_farmer[:, :, 0][
                np.isin(crop_calendar_per_farmer[:, :, 0], crop_values)
            ]

            # if none of the crops are present, no need to replace anything
            if crop_instances.size == 0:
                return crop_calendar_per_farmer

            crops, crop_counts = np.unique(crop_instances, return_counts=True)
            most_common_crop = crops[np.argmax(crop_counts)]

            # Determine if there are multiple cropping versions of this crop and assign it to the most common
            new_crop_types = crop_calendar_per_farmer[
                (crop_calendar_per_farmer[:, :, 0] == most_common_crop).any(axis=1),
                :,
                :,
            ]
            unique_rows, counts = np.unique(new_crop_types, axis=0, return_counts=True)
            max_index = np.argmax(counts)
            crop_replacement = unique_rows[max_index]

            crop_replacement_only_crops = crop_replacement[
                crop_replacement[:, -1] != -1
            ]
            if crop_replacement_only_crops.shape[0] > 1:
                assert (
                    np.unique(crop_replacement_only_crops[:, [1, 3]], axis=0).shape[0]
                    == crop_replacement_only_crops.shape[0]
                )

            for replaced_crop in replaced_crop_values:
                # Check where to be replaced crop is
                crop_mask = (crop_calendar_per_farmer[:, :, 0] == replaced_crop).any(
                    axis=1
                )
                # Replace the crop
                crop_calendar_per_farmer[crop_mask] = crop_replacement

            return crop_calendar_per_farmer

        def unify_crop_variants(
            crop_calendar_per_farmer: np.ndarray, target_crop: int
        ) -> np.ndarray:
            # Create a mask for all entries whose first value == target_crop
            mask = crop_calendar_per_farmer[..., 0] == target_crop

            # If the crop does not appear at all, nothing to do
            if not np.any(mask):
                return crop_calendar_per_farmer

            # Extract only the rows/entries that match the target crop
            crop_entries = crop_calendar_per_farmer[mask]

            # Among these crop rows, find unique variants and their counts
            # (axis=0 ensures we treat each row/entry as a unit)
            unique_variants, variant_counts = np.unique(
                crop_entries, axis=0, return_counts=True
            )

            # The most common variant is the unique variant with the highest count
            most_common_variant = unique_variants[np.argmax(variant_counts)]

            # Replace all the target_crop rows with the most common variant
            crop_calendar_per_farmer[mask] = most_common_variant

            return crop_calendar_per_farmer

        def insert_other_variant_crop(
            crop_calendar_per_farmer: np.ndarray,
            base_crops: int | list[int],
            resistant_crops: tuple[int, int] | list[int] | np.ndarray,
        ) -> np.ndarray:
            # find crop rotation mask
            base_crop_rotation_mask = (
                crop_calendar_per_farmer[:, :, 0] == base_crops
            ).any(axis=1)

            # Find the indices of the crops to be replaced
            indices = np.where(base_crop_rotation_mask)[0]

            # Shuffle the indices to randomize the selection
            np.random.shuffle(indices)

            # Determine the number of crops for each category (stay same, first resistant, last resistant)
            n = len(indices)
            n_same = n // 3
            n_first_resistant = (n // 3) + (
                n % 3 > 0
            )  # Ensuring we account for rounding issues

            # Assign the new values
            crop_calendar_per_farmer[indices[:n_same], 0, 0] = base_crops
            crop_calendar_per_farmer[
                indices[n_same : n_same + n_first_resistant], 0, 0
            ] = resistant_crops[0]
            crop_calendar_per_farmer[indices[n_same + n_first_resistant :], 0, 0] = (
                resistant_crops[1]
            )

            return crop_calendar_per_farmer

        check_crop_calendar(crop_calendar_per_farmer)

        # Reduces certain crops of the same GCAM category to the one that is most common in that region
        # First line checks which crop is most common, second denotes which crops will be replaced by the most common one
        if reduce_crops:
            # Conversion based on the classification in table S1 by Yoon, J., Voisin, N., Klassert, C., Thurber, T., & Xu, W. (2024).
            # Representing farmer irrigated crop area adaptation in a large-scale hydrological model. Hydrology and Earth
            # System Sciences, 28(4), 899–916. https://doi.org/10.5194/hess-28-899-2024

            # Replace fodder with the most common grain crop
            most_common_check = [BARLEY, RYE, MILLET, SORGHUM]
            replaced_value = [FODDER_GRASSES]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change the grain crops to one
            most_common_check = [BARLEY, RYE, MILLET, SORGHUM]
            replaced_value = [BARLEY, RYE, MILLET, SORGHUM]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change other annual / misc to one
            most_common_check = [GROUNDNUTS, COCOA, COFFEE, OTHERS_ANNUAL]
            replaced_value = [GROUNDNUTS, COCOA, COFFEE, OTHERS_ANNUAL]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change oils to one
            most_common_check = [SOYBEANS, SUNFLOWER, RAPESEED]
            replaced_value = [SOYBEANS, SUNFLOWER, RAPESEED]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change tubers to one
            most_common_check = [POTATOES, CASSAVA]
            replaced_value = [POTATOES, CASSAVA]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Reduce sugar crops to one
            most_common_check = [SUGAR_CANE, SUGAR_BEETS]
            replaced_value = [SUGAR_CANE, SUGAR_BEETS]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change perennial to annual, otherwise counted double in esa dataset
            most_common_check = [OIL_PALM, OTHERS_PERENNIAL]
            replaced_value = [OIL_PALM, OTHERS_PERENNIAL]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            unique_rows = np.unique(crop_calendar_per_farmer, axis=0)
            values = unique_rows[:, 0, 0]
            unique_values, counts = np.unique(values, return_counts=True)

            # this part asserts that the crop calendar is correctly set up
            # particularly that no two crops are planted at the same time
            for farmer_crop_calender in crop_calendar_per_farmer:
                farmer_crop_calender = farmer_crop_calender[
                    farmer_crop_calender[:, -1] != -1
                ]
                if farmer_crop_calender.shape[0] > 1:
                    assert (
                        np.unique(farmer_crop_calender[:, [1, 3]], axis=0).shape[0]
                        == farmer_crop_calender.shape[0]
                    )

            # duplicates = unique_values[counts > 1]
            # if len(duplicates) > 0:
            #     for duplicate in duplicates:
            #         crop_calendar_per_farmer = unify_crop_variants(
            #             crop_calendar_per_farmer, duplicate
            #         )

        check_crop_calendar(crop_calendar_per_farmer)

        if replace_base:
            base_crops = [WHEAT]
            resistant_crops = [WHEAT_DROUGHT, WHEAT_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [MAIZE]
            resistant_crops = [MAIZE_DROUGHT, MAIZE_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [RICE]
            resistant_crops = [RICE_DROUGHT, RICE_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [SOYBEANS]
            resistant_crops = [SOYBEANS_DROUGHT, SOYBEANS_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [POTATOES]
            resistant_crops = [POTATOES_DROUGHT, POTATOES_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

        assert crop_calendar_per_farmer[:, :, 3].max() == 0

        check_crop_calendar(crop_calendar_per_farmer)

        self.set_array(crop_calendar_per_farmer, name="agents/farmers/crop_calendar")
        self.set_array(
            np.full_like(is_irrigated, 1, dtype=np.int32),
            name="agents/farmers/crop_calendar_rotation_years",
        )

    def assign_crops(
        self,
        crop_calendar: dict,
        farmer_locations: np.ndarray,
        farmer_mirca_units: np.ndarray,
        year: int,
        MIRCA_unit_grid: xr.DataArray,
        minimum_area_ratio: float,
        replace_crop_calendar_unit_code: dict = {},
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign crops and irrigation status to farmers for a given year.

        Args:
            crop_calendar: Mapping from MIRCA unit to list of rotations (fraction, matrix).
            farmer_locations: Array of farmer pixel coordinates (x, y) order.
            farmer_mirca_units: Array mapping farmer index to MIRCA unit id.
            year: Year to select fractions from raster inputs.
            MIRCA_unit_grid: Grid of MIRCA unit ids aligned with fraction rasters.
            minimum_area_ratio: Minimum fraction for a crop to be considered when sampling.
            replace_crop_calendar_unit_code: Optional remapping for MIRCA unit ids.

        Returns:
            A tuple of (farmer_crops, farmer_irrigated) arrays.

        Raises:
            ValueError: If no valid neighbor data is found while assigning crops.
        """
        # Define the directory and file paths
        data_dir = self.preprocessing_dir / "crops" / "MIRCA2000"
        # Load the DataArrays
        all_years_fraction_da = xr.open_dataarray(
            data_dir / "crop_area_fraction_all_years.nc"
        )
        all_years_irrigated_fraction_da = xr.open_dataarray(
            data_dir / "crop_irrigated_fraction_all_years.nc"
        )

        crop_dict = {
            "Wheat": 0,
            "Maize": 1,
            "Rice": 2,
            "Barley": 3,
            "Rye": 4,
            "Millet": 5,
            "Sorghum": 6,
            "Soybeans": 7,
            "Sunflower": 8,
            "Potatoes": 9,
            "Cassava": 10,
            "Sugar_cane": 11,
            "Sugar_beet": 12,
            "Oil_palm": 13,
            "Rapeseed": 14,
            "Groundnuts": 15,
            "Pulses": 16,
            "Cotton": 20,
            "Cocoa": 21,
            "Coffee": 22,
            "Others_perennial": 23,
            "Fodder": 24,
            "Others_annual": 25,
        }

        area_fraction_2000 = all_years_fraction_da.sel(year=str(year))
        irrigated_fraction_2000 = all_years_irrigated_fraction_da.sel(year=str(year))
        # Fill nas as there is no diff between 0 or na in code and can cause issues
        area_fraction_2000 = area_fraction_2000.fillna(0)
        irrigated_fraction_2000 = irrigated_fraction_2000.fillna(0)

        crop_ids_in_dataarray = np.array(
            [
                crop_dict[crop_name]
                for crop_name in area_fraction_2000.coords["crop"].values
            ]
        )

        mirca_crops_19_to_26 = np.full(26, -1, dtype=np.int32)
        mirca_crops_19_to_26[crop_ids_in_dataarray] = np.arange(
            len(crop_ids_in_dataarray)
        )

        grid_id_da = self.get_linear_indices(all_years_fraction_da)

        ny, nx = area_fraction_2000.sizes["y"], area_fraction_2000.sizes["x"]

        n_cells = grid_id_da.max().item()

        farmer_cells = sample_from_map(
            grid_id_da.values,
            farmer_locations,
            grid_id_da.rio.transform(recalc=True).to_gdal(),
        )

        crop_area_fractions = sample_from_map(
            area_fraction_2000.values,
            farmer_locations,
            area_fraction_2000.rio.transform(recalc=True).to_gdal(),
        )
        crop_irrigated_fractions = sample_from_map(
            irrigated_fraction_2000.values,
            farmer_locations,
            irrigated_fraction_2000.rio.transform(recalc=True).to_gdal(),
        )

        n_farmers = farmer_mirca_units.size

        # Prepare empty crop arrays
        farmer_crops = np.full(n_farmers, -1, dtype=np.int32)
        farmer_irrigated = np.full(n_farmers, 0, dtype=np.bool_)

        for cell_idx in range(n_cells):
            farmers_cell_mask = farmer_cells == cell_idx
            nr_farmers_cell = np.count_nonzero(farmers_cell_mask)
            if nr_farmers_cell == 0:
                continue

            crop_area_fraction = crop_area_fractions[farmer_cells == cell_idx][0]

            MIRCA_unit_cell = MIRCA_unit_grid.values.ravel()[cell_idx]
            MIRCA_unit_cell = replace_crop_calendar_unit_code.get(
                MIRCA_unit_cell, MIRCA_unit_cell
            )

            assert len(crop_calendar[MIRCA_unit_cell]) > 0, (
                f"Error: No crop calendar found for cell {cell_idx} with MIRCA unit {MIRCA_unit_cell}."
            )

            available_crops = np.unique(
                np.concat([crop for _, crop in crop_calendar[MIRCA_unit_cell]])[
                    :, 0, ...
                ]
            )
            available_crops = available_crops[available_crops != -1]

            if crop_area_fraction.sum() == 0:
                # Expand the search radius until valid data is found
                found_valid_neighbor = False
                max_radius = max(nx, ny)  # Maximum possible radius
                radius = 1
                while not found_valid_neighbor and radius <= max_radius:
                    neighbor_ids = self.get_neighbor_cell_ids_for_linear_indices(
                        cell_idx, nx, ny, radius
                    )
                    for neighbor_id in neighbor_ids:
                        if neighbor_id not in farmer_cells:
                            continue

                        neighbor_crop_area_fraction = crop_area_fractions[
                            farmer_cells == neighbor_id
                        ][0]
                        if neighbor_crop_area_fraction.sum() != 0:
                            # Found valid neighbor
                            crop_area_fraction = neighbor_crop_area_fraction
                            found_valid_neighbor = True
                            break
                    if not found_valid_neighbor:
                        radius += 1  # Increase the search radius
                if not found_valid_neighbor:
                    # No valid data found even after expanding radius
                    raise ValueError(
                        f"No valid data found for cell {cell_idx} after searching up to radius {radius - 1}."
                    )

            # ensure fractions sum to 1
            area_per_crop_rotation_26 = crop_area_fraction[mirca_crops_19_to_26]
            area_per_crop_rotation_26[mirca_crops_19_to_26 == -1] = 0

            available_crops_mask = np.zeros_like(area_per_crop_rotation_26, dtype=bool)
            available_crops_mask[available_crops] = True
            area_per_crop_rotation_26[~available_crops_mask] = 0

            assert area_per_crop_rotation_26.sum() > 0, (
                "Error: No crops available for this cell"
            )

            # normalize the area fractions
            area_per_crop_rotation_26 = (
                area_per_crop_rotation_26 / area_per_crop_rotation_26.sum()
            )

            # discard crops with area smaller than minimum_area_ratio
            area_per_crop_rotation_26[
                area_per_crop_rotation_26 < minimum_area_ratio
            ] = 0

            # normalize the area fractions again
            area_per_crop_rotation_26 = (
                area_per_crop_rotation_26 / area_per_crop_rotation_26.sum()
            )

            farmer_indices_in_cell = np.where(farmers_cell_mask)[0]
            farmer_crop_rotations = np.random.choice(
                area_per_crop_rotation_26.size,
                size=len(farmer_indices_in_cell),
                replace=True,
                p=area_per_crop_rotation_26,
            )

            # assign to farmers
            farmer_crops[farmer_indices_in_cell] = farmer_crop_rotations

            # Determine irrigating farmers
            chosen_crops = np.unique(farmer_crop_rotations)

            crop_irrigated_fraction_19 = crop_irrigated_fractions[
                farmer_cells == cell_idx
            ][0]
            crop_irrigated_fraction_26 = crop_irrigated_fraction_19[
                mirca_crops_19_to_26
            ]
            crop_irrigated_fraction_26[mirca_crops_19_to_26 == -1] = np.nan

            for c in chosen_crops:
                # Indices of farmers in the cell assigned to crop c
                farmers_with_crop_c_in_cell = np.where(farmer_crop_rotations == c)[0]
                N_c = len(farmers_with_crop_c_in_cell)
                f_c = crop_irrigated_fraction_26[c]
                if np.isnan(f_c) or f_c <= 0:
                    continue  # No irrigation for this crop
                N_irrigated = int(round(N_c * f_c))
                if N_irrigated > 0:
                    # Randomly select N_irrigated farmers from the N_c farmers
                    irrigated_indices_in_cell = np.random.choice(
                        farmers_with_crop_c_in_cell, size=N_irrigated, replace=False
                    )
                    # Get the overall farmer indices
                    overall_farmer_indices = farmer_indices_in_cell[
                        irrigated_indices_in_cell
                    ]
                    # Set irrigation status to True for these farmers
                    farmer_irrigated[overall_farmer_indices] = True

        assert not (farmer_crops == -1).any(), (
            "Error: some farmers have no crops assigned"
        )

        return farmer_crops, farmer_irrigated
