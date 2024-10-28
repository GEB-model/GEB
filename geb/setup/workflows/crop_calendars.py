import numpy as np
from datetime import date
import calendar
import pandas as pd
import csv


def get_day_index(date):
    return date.timetuple().tm_yday - 1  # 0-indexed


def get_growing_season_length(start_day_index, end_day_index):
    length = (end_day_index - start_day_index) % 365
    if length == 0:
        return 365
    else:
        return length


def parse_MIRCA_file_test(parsed_calendar, crop_calendar, MIRCA_units, is_irrigated):
    # Create mapping from Crop names to crop_class integers
    crop_name_to_class = {
        "Wheat1": 0,
        "Wheat2": 0,
        "Maize": 1,
        "Rice1": 2,
        "Rice2": 2,
        "Rice3": 2,
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
        "Others annual1": 25,
        "Others annual2": 25,
        "Others annual3": 25,
        "Others annual4": 25,
    }

    # Initialize data structure
    data = {}  # data[unit_code][crop_class] = list of rotations

    with open(crop_calendar, "r", newline="", encoding="latin1") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract data from the row
            unit_code = int(row["unit_code"])

            if unit_code not in MIRCA_units:
                continue

            if unit_code not in data:
                data[unit_code] = {}

            crop_name = row["Crop"]
            subcrop = int(row["Subcrop"])  # May be used if needed
            # crop_type = row['Type']  # Not used in current context
            growing_area = float(row["Growing_area"])
            planting_month = int(row["Planting_Month"])
            maturity_month = int(row["Maturity_Month"])

            if growing_area == 0:
                continue  # Ignore crops with zero area

            crop_class = crop_name_to_class.get(crop_name)
            if crop_class is None:
                # Handle unknown crop
                continue

            crop_class -= 1  # Adjust to zero-based indexing

            if crop_class not in data[unit_code]:
                data[unit_code][crop_class] = []

            data[unit_code][crop_class].append(
                (growing_area, planting_month, maturity_month)
            )

    # Now process the data as in the original code
    for unit_code in data:
        if unit_code not in parsed_calendar:
            parsed_calendar[unit_code] = []

        for crop_class in data[unit_code]:
            crop_rotations = []
            rotations = data[unit_code][crop_class]

            number_of_rotations = len(rotations)

            for rotation in rotations:
                area, start_month, end_month = rotation

                start_day_index = get_day_index(date(2000, start_month, 1))
                end_day_index = get_day_index(
                    date(2000, end_month, calendar.monthrange(2000, end_month)[1])
                )
                growth_length = get_growing_season_length(
                    start_day_index, end_day_index
                )
                crop_rotations.append((start_day_index, growth_length, area))

            # Discard crop rotations with zero area
            crop_rotations = [
                crop_rotation
                for crop_rotation in crop_rotations
                if crop_rotation[2] > 0
            ]

            # Sort by area
            crop_rotations = sorted(crop_rotations, key=lambda x: x[2])

            if len(crop_rotations) == 3:
                crop_rotations = crop_rotations[1:]
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
                        ),
                        dtype=np.int32,
                    ),
                )  # -1 means no crop
                parsed_calendar[unit_code].append(crop_rotation)
            elif len(crop_rotations) == 2:
                # Check if crop rotations start on the same day
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
                else:
                    # Assume multi-cropping if crop rotations are consecutive
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
            else:
                raise NotImplementedError(
                    "Handling more than two crop rotations is not implemented."
                )
    return parsed_calendar


def parse_MIRCA_file(parsed_calendar, crop_calendar, MIRCA_units, is_irrigated):
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
            if len(crop_rotations) == 3:
                crop_rotations = crop_rotations[1:]
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
            else:
                raise NotImplementedError
        return parsed_calendar


def parse_MIRCA2000_crop_calendar(data_catalog, MIRCA_units):
    rainfed_crop_calendar_fp_2000 = data_catalog.get_source(
        "MIRCA2000_cropping_calendar_rainfed_2000"
    ).path
    rainfed_crop_calendar_fp = data_catalog.get_source(
        "MIRCA2000_cropping_calendar_rainfed"
    ).path
    irrigated_crop_calendar_fp = data_catalog.get_source(
        "MIRCA2000_cropping_calendar_irrigated"
    ).path

    MIRCA2000_data = {}

    MIRCA2000_data = parse_MIRCA_file_test(
        MIRCA2000_data,
        rainfed_crop_calendar_fp_2000,
        MIRCA_units,
        is_irrigated=False,
    )
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
