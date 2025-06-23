import json
from datetime import date, datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class DateIndex:
    def __init__(self, dates: list[date | datetime]) -> None:
        """Create a DateIndex object that allows for fast lookup of dates.

        This class takes a list of dates and creates an index that allows for fast lookup of the index of a date in the list.
        It also extrapolates the last date to allow for future dates.

        Args:
            dates: a list of dates in datetime format. The dates should be sorted in ascending order.
        """
        self.dates = np.array(dates)

        self.last_valid_date = self.dates[-1] + relativedelta(
            self.dates[-1], self.dates[-2]
        )  # extrapolate last date.

    def get(self, date: date | datetime) -> int:
        """Get the index of a date in the list of dates.

        This method returns the index of the date in the list of dates. If the date is before the first date or after the last date, it raises a ValueError.
        If the date is not in the list, it returns the index of the last date that is smaller than the given date.

        Args:
            date: a date in datetime format. The date should be larger than the first date and smaller than the last date.

        Raises:
            ValueError: If the date is before the first date or after the last date.
            ValueError: If the date is not in the list and no extrapolation is possible.

        Returns:
            int: the index of the date in the list of dates. If the date is not in the list, it returns the index of the last date that is smaller than the given date.
        """
        # find first date where date is larger or equal to date in self.dates
        if date < self.dates[0]:
            raise ValueError(f"Date {date} is before first valid date {self.dates[0]}")
        if date > self.last_valid_date:
            raise ValueError(
                f"Date {date} is after last valid date {self.last_valid_date}"
            )

        return np.searchsorted(self.dates, date, side="right").item() - 1

    def __len__(self):
        """Return the number of dates in the index."""
        return self.dates.size


def load_regional_crop_data_from_dict(
    model, name
) -> tuple[dict[dict[date, int]], dict[str, np.ndarray]]:
    """Load crop prices per state from the input data and return a dictionary of states containing 2D array of prices.

    Returns:
        date_index: Dictionary of states containing a dictionary of dates and their index in the 2D array.
        crop_prices: Dictionary of states containing a 2D array of crop prices. First index is for date, second index is for crop.
    """
    with open(model.files["dict"][name], "r") as f:
        timedata = json.load(f)

    if timedata["type"] == "constant":
        return None, timedata["data"]
    elif timedata["type"] == "time_series":
        dates = parse_dates(timedata["time"])
        date_index = DateIndex(dates)

        data = timedata["data"]

        d = np.full(
            (len(date_index), len(model.regions), len(data["0"])),
            np.nan,
            dtype=np.float32,
        )  # all lengths should be the same, so just taking data from region 0.
        for region_id, region_data in data.items():
            for ID, region_crop_data in region_data.items():
                d[:, int(region_id), int(ID)] = region_crop_data

        # assert not np.isnan(d).any()
        return date_index, d
    else:
        raise ValueError(f"Unknown type: {timedata['type']}")


def load_crop_data(files) -> dict[np.ndarray]:
    """Read csv-file of values for crop water depletion.

    Returns:
        yield_factors: dictonary with np.ndarray of values per crop for each variable.
    """
    with open(files["dict"]["crops/crop_data"], "r") as f:
        crop_data = json.load(f)
    data = pd.DataFrame.from_dict(crop_data["data"], orient="index")
    data.index = data.index.astype(int)
    return crop_data["type"], data


def parse_dates(date_strings, date_formats=["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y"]):
    for date_format in date_formats:
        try:
            return [datetime.strptime(str(d), date_format) for d in date_strings]
        except ValueError:
            pass
    else:
        raise ValueError(
            "No valid date format found for date strings: {}".format(date_strings[0])
        )


def load_economic_data(fp: str) -> tuple[DateIndex, dict[int, np.ndarray]]:
    with open(fp, "r") as f:
        data = json.load(f)
    dates = parse_dates(data["time"])
    date_index = DateIndex(dates)
    d = {int(region_id): values for region_id, values in data["data"].items()}
    return (date_index, d)


# def load_sprinkler_prices(self, inflation_rates_per_year):
#     sprinkler_price_2008 = self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['adaptation_cost']
#     #upkeep_price_2008_m2 = 3000 / 10_000  # ha to m2
#     # create dictory with prices for well_prices per year by applying inflation rates
#     sprinkler_prices = {2008: sprinkler_price_2008}
#     for year in range(2009, 2022):
#         sprinkler_prices[year] = sprinkler_prices[year-1] * inflation_rates_per_year[year]
#     for year in range(2007, 1960, -1):
#         sprinkler_prices[year] = sprinkler_prices[year+1] / inflation_rates_per_year[year+1]
#     return sprinkler_prices
