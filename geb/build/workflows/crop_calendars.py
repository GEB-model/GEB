"""Functions to parse crop calendars from MIRCA2000 data."""

import calendar
import warnings
from datetime import date

import numpy as np
import pandas as pd

from geb.geb_types import TwoDArrayInt32

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

    # Collapse exact duplicates: rows sharing the same unit, crop class, and
    # growing season (identical planting and maturity month) have their areas
    # summed for simplication of further logic.
    #
    # Rows with a different planting or maturity month are kept as
    # separate rotation candidates.
    calendar_data = (
        calendar_data.groupby(
            ["unit_code", "crop_class", "Planting_Month", "Maturity_Month"],
            sort=False,
        )
        .agg(Growing_area=("Growing_area", "sum"))
        .reset_index()
    )

    for unit_code, unit_rows in calendar_data.groupby("unit_code", sort=False):
        if unit_code not in parsed_calendar:
            parsed_calendar[unit_code] = []

        for crop_class, crop_rows in unit_rows.groupby("crop_class", sort=False):
            crop_rotations: list[tuple[int, int, float]] = []

            for row in crop_rows.itertuples(index=False):
                area: float = float(row.Growing_area)
                if area > 0.0:
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

            if crop_rotations:
                crop_rotations = sorted(crop_rotations, key=lambda x: x[2])
                if len(crop_rotations) > 2:
                    # Three or more distinct named variants exist for the same crop
                    # class in this unit (e.g. Wheat1, Wheat2, Wheat3). Only two
                    # rotation windows can be represented per farmer group, so we
                    # keep the two largest-area seasons and discard the rest.
                    crop_rotations: list[tuple[int, int, int | float]] = crop_rotations[
                        -2:
                    ]
                    warnings.warn(
                        "More than 2 distinct crop rotations found for the same crop "
                        "class in one MIRCA unit; discarding the rotation(s) with the "
                        "lowest area."
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
                        area = crop_rotations[1][2] - crop_rotations[0][2]
                        if area > 0:
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
                            assert (
                                crop_rotation_entry[1][0][2]
                                != crop_rotation_entry[1][1][2]
                            )

                else:
                    raise NotImplementedError

    return parsed_calendar


def donate_and_receive_crop_prices(
    donor_data: pd.DataFrame,
    recipient_regions: pd.DataFrame,
    trade_regions: dict[str, str],
) -> pd.DataFrame:
    """Gets crop prices from other countries in the same trade region to fill missing data.

    For each recipient region and each crop without any price in its country, the prices are
    filled with the median price per year over all countries in the same trade region that have
    data for that crop. ``donor_data`` should therefore contain all countries of the relevant
    trade regions, not only the countries in the model region. If no country in the trade region
    has data for a crop, the crop is left missing; ``process_crop_data`` then raises an error if
    farmers grow that crop. Prices are in USD, so no currency conversion is needed.

    Args:
        donor_data: Crop prices (one column per crop, plus an 'ISO3' column), indexed by
            (ISO3, year).
        recipient_regions: Regions to fill, with 'region_id' and 'ISO3' columns.
        trade_regions: Mapping of ISO3 country codes to their trade region.

    Returns:
        Crop prices per (region_id, year), with missing crops filled from the trade-region median.

    Raises:
        ValueError: If a recipient country has no price data or is not in any trade region.
    """
    data_out = pd.DataFrame()

    for _, region in recipient_regions.iterrows():
        ISO3 = region["ISO3"]
        country_data = donor_data[donor_data["ISO3"] == ISO3]
        if country_data.empty or ISO3 not in trade_regions:
            raise ValueError(
                f"No crop price data for {ISO3}: the country is not in the trade regions "
                "dataset, so prices cannot be filled from its trade region."
            )
        trade_regions_region = trade_regions[ISO3]

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

                if donor_data_region.notna().any():
                    # median price per year over all trade-region countries with data
                    donor_data_country = donor_data_region.groupby(
                        level="year"
                    ).median()
                else:
                    # no price in the trade region: leave missing; process_crop_data raises
                    # a clear error if farmers in this region grow the crop
                    continue

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
