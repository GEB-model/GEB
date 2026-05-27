"""Functions to parse crop calendars from MIRCA2000 data."""

import calendar
import warnings
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd

from geb.build.workflows.conversions import setup_donor_countries
from geb.geb_types import TwoDArrayInt32

from ..data_catalog import DataCatalog

MIRCA_OS_CROP_CLASS_MAP: dict[str, int] = {
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
    "Sugar cane": 11,
    "Sugar beet": 12,
    "Oil palm": 13,
    "Rapeseed": 14,
    "Groundnuts": 15,
    "Pulses": 16,
    "Cotton": 20,
    "Cocoa": 21,
    "Coffee": 22,
    "Others perennial": 23,
    "Fodder": 24,
    "Others annual": 25,
}


def _build_crop_rotation_array(
    crop_class: int,
    is_irrigated: bool,
    crop_rotations: list[tuple[int, int]],
) -> TwoDArrayInt32:
    """Create a fixed-size crop rotation array from one or two rotations.

    Args:
        crop_class: Crop class id matching the model crop id convention.
        is_irrigated: Whether this crop calendar comes from irrigated area.
        crop_rotations: Sequence of (start_day_index, growth_length) tuples.

    Returns:
        Crop rotation matrix with shape (3, 5).
    """
    rotation_array: TwoDArrayInt32 = np.full((3, 5), -1, dtype=np.int32)
    for rotation_index, (start_day_index, growth_length) in enumerate(crop_rotations):
        rotation_array[rotation_index] = (
            crop_class,
            int(is_irrigated),
            start_day_index,
            growth_length,
            0,
        )

    return rotation_array


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


def parse_MIRCA_crop_calendar(
    parsed_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    crop_calendar_data: pd.DataFrame,
    MIRCA_units: list[int],
    is_irrigated: bool,
) -> dict[int, list[tuple[float, TwoDArrayInt32]]]:
    """Parse a MIRCA-OS crop calendar table.

    Args:
        parsed_calendar: The dictionary to store the parsed calendar in.
        crop_calendar_data: MIRCA-OS crop calendar table.
        MIRCA_units: The list of MIRCA unit codes to parse.
        is_irrigated: Whether the calendar is for irrigated crops.

    Returns:
        The updated parsed_calendar dictionary.

    Raises:
        ValueError: If  crop names are unknown, or months are outside the valid 1-12 range.
    """
    calendar_data = crop_calendar_data.loc[
        crop_calendar_data["unit_code"].isin(MIRCA_units)
    ].copy()
    if calendar_data.empty:
        return parsed_calendar

    calendar_data["Crop"] = calendar_data["Crop"].astype(str).str.strip()

    # Strip any trailing digits so numbered variants (e.g. "Wheat1", "Rice2",
    # "Others annual3") all resolve to their base name before the map lookup.
    calendar_data["Crop"] = (
        calendar_data["Crop"].str.replace(r"\d+$", "", regex=True).str.strip()
    )

    calendar_data["crop_class"] = calendar_data["Crop"].map(MIRCA_OS_CROP_CLASS_MAP)

    unknown_crops = sorted(
        calendar_data.loc[calendar_data["crop_class"].isna(), "Crop"].unique().tolist()
    )
    if unknown_crops:
        raise ValueError(
            "Encountered unsupported MIRCA-OS crop name(s): " + ", ".join(unknown_crops)
        )

    calendar_data["unit_code"] = pd.to_numeric(calendar_data["unit_code"])
    calendar_data["Growing_area"] = pd.to_numeric(calendar_data["Growing_area"])
    calendar_data["Planting_Month"] = pd.to_numeric(calendar_data["Planting_Month"])
    calendar_data["Maturity_Month"] = pd.to_numeric(calendar_data["Maturity_Month"])
    calendar_data = calendar_data.dropna(
        subset=["unit_code", "Growing_area", "Planting_Month", "Maturity_Month"]
    )
    calendar_data["unit_code"] = calendar_data["unit_code"].astype(np.int64)
    calendar_data["crop_class"] = calendar_data["crop_class"].astype(np.int64)
    calendar_data["Planting_Month"] = calendar_data["Planting_Month"].astype(np.int64)
    calendar_data["Maturity_Month"] = calendar_data["Maturity_Month"].astype(np.int64)

    for unit_code, unit_rows in calendar_data.groupby("unit_code", sort=False):
        if unit_code not in parsed_calendar:
            parsed_calendar[unit_code] = []

        for crop_class, crop_rows in unit_rows.groupby("crop_class", sort=False):
            crop_rotations: list[tuple[int, int, float]] = []

            for row in crop_rows.itertuples(index=False):
                area: float = float(row.Growing_area)
                if area <= 0:
                    continue

                start_month: int = int(row.Planting_Month)
                end_month: int = int(row.Maturity_Month)
                if not (1 <= start_month <= 12) or not (1 <= end_month <= 12):
                    raise ValueError(
                        "MIRCA-OS planting and maturity months must be in [1, 12]."
                    )

                start_day_index = get_day_index(date(2000, start_month, 1))
                end_day_index = get_day_index(
                    date(2000, end_month, calendar.monthrange(2000, end_month)[1])
                )
                growth_length = get_growing_season_length(
                    start_day_index, end_day_index
                )

                crop_rotations.append((start_day_index, growth_length, area))

            if not crop_rotations:
                continue

            crop_rotations = sorted(crop_rotations, key=lambda x: x[2])
            if len(crop_rotations) > 2:
                crop_rotations = crop_rotations[-2:]
                warnings.warn(
                    "More than 2 crop rotations found, discarding the one with the lowest area. This should be fixed later."
                )

            if len(crop_rotations) == 1:
                start_day_index, growth_length, area = crop_rotations[0]
                crop_rotation: tuple[float, TwoDArrayInt32] = (
                    area,
                    _build_crop_rotation_array(
                        crop_class=int(crop_class),
                        is_irrigated=is_irrigated,
                        crop_rotations=[(start_day_index, growth_length)],
                    ),
                )
                parsed_calendar[unit_code].append(crop_rotation)

            elif len(crop_rotations) == 2:
                s0, l0, _ = crop_rotations[0]
                s1, l1, _ = crop_rotations[1]
                # If the growing seasons overlap (including same start day) the two
                # rotations cannot be practiced by the same farmer, so we split them
                # into separate entries. Overlap is tested on the circular 365-day
                # calendar: season [s, s+l) wraps around the year end when s+l > 365.
                seasons_overlap = ((s1 - s0) % 365) < l0 or ((s0 - s1) % 365) < l1
                if seasons_overlap:
                    for start_day_index, growth_length, area in crop_rotations:
                        crop_rotation_entry: tuple[float, TwoDArrayInt32] = (
                            area,
                            _build_crop_rotation_array(
                                crop_class=int(crop_class),
                                is_irrigated=is_irrigated,
                                crop_rotations=[(start_day_index, growth_length)],
                            ),
                        )
                        parsed_calendar[unit_code].append(crop_rotation_entry)
                else:
                    crop_rotation_entry = (
                        crop_rotations[1][2] - crop_rotations[0][2],
                        _build_crop_rotation_array(
                            crop_class=int(crop_class),
                            is_irrigated=is_irrigated,
                            crop_rotations=[
                                (crop_rotations[1][0], crop_rotations[1][1])
                            ],
                        ),
                    )
                    parsed_calendar[unit_code].append(crop_rotation_entry)

                    crop_rotation_entry = (
                        crop_rotations[0][2],
                        _build_crop_rotation_array(
                            crop_class=int(crop_class),
                            is_irrigated=is_irrigated,
                            crop_rotations=[
                                (crop_rotations[0][0], crop_rotations[0][1]),
                                (crop_rotations[1][0], crop_rotations[1][1]),
                            ],
                        ),
                    )

                    parsed_calendar[unit_code].append(crop_rotation_entry)
                    assert crop_rotation_entry[1][0][2] != crop_rotation_entry[1][1][2]

            else:
                raise NotImplementedError

    return parsed_calendar


def parse_MIRCA2000_crop_calendar(
    data_catalog: DataCatalog,
    MIRCA_units: list[int],
    year: int = 2000,
) -> dict[int, list[tuple[float, TwoDArrayInt32]]]:
    """Parse MIRCA-OS crop calendars for given MIRCA units.

    Notes:
        This function keeps a legacy name for backward compatibility, but now
        reads MIRCA-OS crop calendar CSV sources.

    Args:
        data_catalog: The data catalog containing MIRCA-OS files.
        MIRCA_units: The list of MIRCA unit codes to parse.
        year: The MIRCA-OS reference year.

    Returns:
        A dictionary containing the parsed crop calendars.

    Raises:
        TypeError: If the calendar data is not provided as a DataFrame.
    """
    rainfed_source = data_catalog.fetch(f"mirca_os_crop_calendar_{year}_rf").read()
    irrigated_source = data_catalog.fetch(f"mirca_os_crop_calendar_{year}_ir").read()

    if not isinstance(rainfed_source, pd.DataFrame) or not isinstance(
        irrigated_source, pd.DataFrame
    ):
        raise TypeError("Expected MIRCA-OS calendar data as pandas DataFrames.")

    mirca2000_data: dict[int, list[tuple[float, TwoDArrayInt32]]] = {}

    mirca2000_data = parse_MIRCA_crop_calendar(
        mirca2000_data,
        rainfed_source,
        MIRCA_units,
        is_irrigated=False,
    )
    mirca2000_data = parse_MIRCA_crop_calendar(
        mirca2000_data,
        irrigated_source,
        MIRCA_units,
        is_irrigated=True,
    )

    return mirca2000_data


def donate_and_receive_crop_prices(
    donor_data: pd.DataFrame,
    recipient_regions: pd.DataFrame,
    trade_regions: dict[str, str],
    data_catalog: DataCatalog,
    global_countries: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Gets crop prices from other regions to fill missing data.

    If there are multiple countries in one selected basin, where one country has prices for a certain crop,
    but the other does not, this gives issues. This function adjusts crop data for those countries by
    filling in missing values using data from nearby regions and PPP conversion rates. In case crop data
    is missing for a country, and is also not in countries in the same trade_regions dataset, it uses the prices
    for that crop from the country in the model region with least nan values.

    Args:
        donor_data: A DataFrame containing crop data with a 'ISO3' column and indexed by 'region_id'.
            The DataFrame contains crop prices for different regions.
        recipient_regions: DataFrame containing recipient region information with 'region_id' and 'ISO3' columns.
        trade_regions: A dictionary mapping ISO3 country codes to their respective trade regions.
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

        Some countries without data are also not in the trade_regions dataset (e.g Liechtenstein (LIE)).
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

        # Filter the data for the current country
        country_data = donor_data[donor_data["ISO3"] == ISO3]

        if country_data.empty:  # happens if country is not in trade_regions regions dataset (e.g. Kosovo). Fill these countries using data from a country that is in the trade_regions regions dataset, using the regular donor countries setup.
            countries_with_donor_data = donor_data.ISO3.unique().tolist()
            donor_countries = setup_donor_countries(
                data_catalog,
                global_countries,
                countries_with_donor_data,
                alternative_countries=regions["ISO3"].unique().tolist(),
            )
            ISO3 = donor_countries.get(ISO3, None)
            print(
                f"Missing price donor data for {region['ISO3']}, using donor country {ISO3}. This country is NOT in the trade_regions regions dataset"
            )
            assert ISO3 is not None, (
                f"Could not find a donor country for {region['ISO3']}. Please check the donor countries setup."
            )

            country_data = donor_data[donor_data["ISO3"] == ISO3]

            assert not country_data.empty, (
                f"Donor country {ISO3} has no data for {region['ISO3']}. Please check the donor countries setup."
            )
            # note: it can be that a country is donor for another in the first donor step (outside this function) (e.g. Isreal for cyprus), and that here cyprus is again selected as a donor country for another country (e.g. Liechtenstein)

        trade_regions_region = trade_regions[ISO3]

        assert len(trade_regions_region) > 0, (
            f"trade_regions region for {ISO3} is empty. Please check the trade_regions regions setup."
        )

        trade_regions_region_countries: list[str] = [
            ISO3
            for ISO3, trade_region in trade_regions.items()
            if trade_region == trade_regions_region
        ]

        for column in country_data.columns:
            if country_data[column].isna().all():
                donor_data_region = donor_data.loc[
                    donor_data["ISO3"].isin(trade_regions_region_countries), column
                ]

                # Check if data is available within the trade_regions region
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
