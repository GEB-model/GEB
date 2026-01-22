"""Functions to parse crop calendars from MIRCA2000 data."""

import calendar
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from hydromt.data_catalog import DataCatalog

from geb.build.workflows.conversions import setup_donor_countries
from geb.geb_types import TwoDArrayInt32


def get_day_index(date: date) -> int:
    """Get the day index (0-364) for a given date.

    Args:
        date: The date for which to get the day index.

    Returns:
        The day index (0-364).
    """
    return date.timetuple().tm_yday - 1  # 0-indexed


def get_growing_season_length(start_day_index: int, end_day_index: int) -> int:
    """Calculate the length of the growing season in days.

    Essentially calculates (end_day_index - start_day_index) mod 365, thus
    wrapping around the year if necessary. If start and end are the same,
    we assume the growing season lasts the entire year (365 days) rather
    than 0 days.

    Args:
        start_day_index: The starting day index (0-364).
        end_day_index: The ending day index (0-364).

    Returns:
        The length of the growing season in days.
    """
    length = (end_day_index - start_day_index) % 365
    if length == 0:
        return 365
    else:
        return length


def parse_MIRCA_file(
    parsed_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    crop_calendar: Path,
    MIRCA_units: list[int],
    is_irrigated: bool,
) -> dict[int, list[tuple[float, TwoDArrayInt32]]]:
    """Parse a MIRCA2000 crop calendar file.

    Args:
        parsed_calendar: The dictionary to store the parsed calendar in.
        crop_calendar: The path to the MIRCA2000 crop calendar file.
        MIRCA_units: The list of MIRCA unit codes to parse.
        is_irrigated: Whether the calendar is for irrigated crops.

    Returns:
        The updated parsed_calendar dictionary.
    """
    with open(crop_calendar, "r") as f:
        lines = f.readlines()
        # remove all empty lines
        lines = [line.strip() for line in lines if line.strip()]
        # skip header
        lines = lines[4:]
        for line in lines:
            line = line.replace("  ", " ").split(" ")
            unit_code = int(line[0])
            if unit_code not in MIRCA_units:
                continue
            if unit_code not in parsed_calendar:
                parsed_calendar[unit_code] = []
            crop_class = int(line[1]) - 1  # minus one to make it zero based
            number_of_rotations = int(line[2])
            if number_of_rotations == 0:
                continue
            crops = line[3:]
            crop_rotations = []
            for rotation in range(number_of_rotations):
                area = float(crops[rotation * 3])
                if area == 0:
                    continue
                start_month = int(crops[rotation * 3 + 1])
                end_month = int(crops[rotation * 3 + 2])
                start_day_index = get_day_index(date(2000, start_month, 1))
                end_day_index = get_day_index(
                    date(2000, end_month, calendar.monthrange(2000, end_month)[1])
                )
                growth_length = get_growing_season_length(
                    start_day_index, end_day_index
                )
                crop_rotations.append((start_day_index, growth_length, area))

            del start_month
            del end_month
            del start_day_index
            del end_day_index
            del growth_length

            # discard crop rotations with zero area
            crop_rotations = [
                crop_rotation
                for crop_rotation in crop_rotations
                if crop_rotation[2] > 0
            ]

            crop_rotations = sorted(crop_rotations, key=lambda x: x[2])  # sort by area
            if len(crop_rotations) > 2:
                crop_rotations = crop_rotations[-2:]
                import warnings

                warnings.warn(
                    "More than 2 crop rotations found, discarding the one with the lowest area. This should be fixed later."
                )
            if len(crop_rotations) == 1:
                start_day_index, growth_length, area = crop_rotations[0]
                crop_rotation = (
                    area,
                    np.array(
                        (
                            (
                                crop_class,
                                is_irrigated,
                                start_day_index,
                                growth_length,
                                0,
                            ),
                            (-1, -1, -1, -1, -1),
                            (-1, -1, -1, -1, -1),
                        )
                    ),
                )  # -1 means no crop
                parsed_calendar[unit_code].append(crop_rotation)
            elif len(crop_rotations) == 2:
                # if crop rotations start on the same day, they cannot be implemented
                # by the same farmer, so we split them
                # TODO: Ensure that this only happens when the crop rotations cannot overlap.
                if crop_rotations[0][0] == crop_rotations[1][0]:
                    for crop_rotation in crop_rotations:
                        start_day_index, growth_length, area = crop_rotation
                        crop_rotation = (
                            area,
                            np.array(
                                (
                                    (
                                        crop_class,
                                        is_irrigated,
                                        start_day_index,
                                        growth_length,
                                        0,
                                    ),
                                    (-1, -1, -1, -1, -1),
                                    (-1, -1, -1, -1, -1),
                                ),
                                dtype=np.int32,
                            ),
                        )
                        parsed_calendar[unit_code].append(crop_rotation)
                # if the crop rotations are consecutive, we assume multi-cropping.
                else:
                    crop_rotation = (
                        crop_rotations[1][2] - crop_rotations[0][2],
                        np.array(
                            (
                                (
                                    crop_class,
                                    is_irrigated,
                                    crop_rotations[1][0],
                                    crop_rotations[1][1],
                                    0,
                                ),
                                (-1, -1, -1, -1, -1),
                                (-1, -1, -1, -1, -1),
                            ),
                            dtype=np.int32,
                        ),  # -1 means no crop
                    )
                    parsed_calendar[unit_code].append(crop_rotation)
                    crop_rotation = (
                        crop_rotations[0][2],
                        np.array(
                            (
                                (
                                    crop_class,
                                    is_irrigated,
                                    crop_rotations[0][0],
                                    crop_rotations[0][1],
                                    0,
                                ),
                                (
                                    crop_class,
                                    is_irrigated,
                                    crop_rotations[1][0],
                                    crop_rotations[1][1],
                                    0,
                                ),
                                (-1, -1, -1, -1, -1),
                            ),
                            dtype=np.int32,
                        ),
                    )
                parsed_calendar[unit_code].append(crop_rotation)
                assert crop_rotation[1][0][2] != crop_rotation[1][1][2]
            else:
                raise NotImplementedError
        return parsed_calendar


def parse_MIRCA2000_crop_calendar(
    data_catalog: DataCatalog, MIRCA_units: list[int]
) -> dict[int, list[tuple[float, TwoDArrayInt32]]]:
    """Parse MIRCA2000 crop calendars for given MIRCA units.

    Args:
        data_catalog: The data catalog containing the MIRCA2000 files.
        MIRCA_units: The list of MIRCA unit codes to parse.

    Returns:
        A dictionary containing the parsed crop calendars.
    """
    rainfed_crop_calendar_fp = Path(
        data_catalog.get_source("MIRCA2000_cropping_calendar_rainfed").path
    )
    irrigated_crop_calendar_fp = Path(
        data_catalog.get_source("MIRCA2000_cropping_calendar_irrigated").path
    )

    MIRCA2000_data = {}

    MIRCA2000_data = parse_MIRCA_file(
        MIRCA2000_data,
        rainfed_crop_calendar_fp,
        MIRCA_units,
        is_irrigated=False,
    )
    MIRCA2000_data = parse_MIRCA_file(
        MIRCA2000_data,
        irrigated_crop_calendar_fp,
        MIRCA_units,
        is_irrigated=True,
    )

    return MIRCA2000_data


def donate_and_receive_crop_prices(
    donor_data: pd.DataFrame,
    recipient_regions: pd.DataFrame,
    GLOBIOM_regions: pd.DataFrame,
    data_catalog: DataCatalog,
    global_countries: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
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
        data_catalog: The data catalog containing necessary data sources.
        global_countries: A GeoDataFrame containing global country information.
        regions: A GeoDataFrame containing region information.

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
    data_out = pd.DataFrame()

    for _, region in recipient_regions.iterrows():
        ISO3 = region["ISO3"]
        region_id = region["region_id"]
        print(f"Processing region {region_id}")

        # Filter the data for the current country
        country_data = donor_data[donor_data["ISO3"] == ISO3]

        if country_data.empty:  # happens if country is not in GLOBIOM regions dataset (e.g. Kosovo). Fill these countries using data from a country that is in the GLOBIOM regions dataset, using the regular donor countries setup.
            countries_with_donor_data = donor_data.ISO3.unique().tolist()
            donor_countries = setup_donor_countries(
                data_catalog,
                global_countries,
                countries_with_donor_data,
                alternative_countries=regions["ISO3"].unique().tolist(),
            )
            ISO3 = donor_countries.get(ISO3, None)
            print(
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

                if non_na_values.max() > 0:  # if there is at least one non-NaN value
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
                    columns=np.array([donor_data_country.name]),
                )

                if data_out.empty:
                    data_out = new_data
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
                    columns=np.array([column]),
                )

                if data_out.empty:
                    data_out = new_data
                else:
                    data_out = data_out.combine_first(new_data)

    data_out = data_out.drop(columns=["ISO3"])
    data_out = data_out.dropna(axis=1, how="all")
    data_out = data_out.dropna(axis=0, how="all")

    return data_out
