import os
from datetime import date, datetime
import numpy as np
import pandas as pd
import json
from pathlib import Path

from config import INPUT, ORIGINAL_DATA

class DateIndex:
    def __init__(self, dates):
        self.dates = np.array(dates)

    def get(self, date):
        # find first date where date is larger or equal to date in self.dates
        return np.searchsorted(self.dates, date, side='right') - 1
    
    def __len__(self):
        return self.dates.size

def load_cultivation_costs():
    fp = os.path.join(INPUT, 'crops', 'cultivation_costs.json')
    with open(fp, 'r') as f:
        costs = json.load(f)
    dates = parse_dates(costs['time'])
    date_index = DateIndex(dates)
    crops = costs['crops']

    cultivation_costs = np.full((len(date_index), len(crops)), np.nan, dtype=np.float32)  # first index for date, second index for crops
    for ID, data in crops.items():
        cultivation_costs[:, int(ID)] = data
    assert not np.isnan(cultivation_costs).any()
    return date_index, cultivation_costs

def load_crop_prices() -> tuple[dict[dict[date, int]], dict[str, np.ndarray]]:
    """Load crop prices per state from the input data and return a dictionary of states containing 2D array of prices.
    
    Returns:
        date_index: Dictionary of states containing a dictionary of dates and their index in the 2D array.
        crop_prices: Dictionary of states containing a 2D array of crop prices. First index is for date, second index is for crop."""
    
    fp = Path(INPUT, 'crops', 'crop_prices.json')
    with open(fp, 'r') as f:
        crop_prices = json.load(f)

    dates = parse_dates(crop_prices['time'])
    date_index = DateIndex(dates)

    crops = crop_prices['crops']

    crop_prices_array = np.full((len(date_index), len(crops)), np.nan, dtype=np.float32)  # first index for date, second index for crops
    for ID, data in crops.items():
        crop_prices_array[:, int(ID)] = data
    assert not np.isnan(crop_prices_array).any()
    
    return date_index, crop_prices_array

def load_crop_variables() -> dict[np.ndarray]:
    """Read csv-file of values for crop water depletion.
    
    Returns:
        yield_factors: dictonary with np.ndarray of values per crop for each variable.
    """
    with open(os.path.join(INPUT, 'crops', 'crop_variables.json'), 'r') as f:
        crop_variables = json.load(f)
    return pd.DataFrame.from_dict(crop_variables, orient='index')

    
    # growth_length = np.full((len(crops), 3), np.nan, dtype=np.float32)
    # growth_length[:, 0] = df['kharif_d']
    # growth_length[:, 1] = df['rabi_d']
    # growth_length[:, 2] = df['summer_d']
    # assert not np.isnan(growth_length).any()
    
    # stage_lengths = np.full((len(crops), 4), np.nan, dtype=np.float32)
    # stage_lengths[:,0] = df['d1']
    # stage_lengths[:,1] = df['d2a'] + df['d2b']
    # stage_lengths[:,2] = df['d3a'] + df['d3b']
    # stage_lengths[:,3] = df['d4']
    # assert not np.isnan(stage_lengths).any()

    # crop_factors = np.full((len(crops), 3), np.nan, dtype=np.float32)
    # crop_factors[:,0] = df['Kc1']
    # crop_factors[:,1] = df['Kc3']
    # crop_factors[:,2] = df['Kc5']
    # assert not np.isnan(crop_factors).any()

    # yield_factors = {
    #     # 'Ky1': df['Ky1'].to_numpy(),
    #     # 'Ky2': ((df['Ky2a'] * df['d2a'] + df['Ky2b'] * df['d2b']) / (df['d2a'] + df['d2b'])).to_numpy(),
    #     # 'Ky3': ((df['Ky3a'] * df['d3a'] + df['Ky3b'] * df['d3b']) / (df['d3a'] + df['d3b'])).to_numpy(),
    #     # 'Ky4': df['Ky4'].to_numpy(),
    #     'KyT': df['KyT'].to_numpy(),
    # }

    # # MIRCA2000 reference yields
    # reference_yield = df['reference_yield_gr_m2'].to_numpy()
    # assert not np.isnan(reference_yield).any()
    
    # return growth_length, stage_lengths, crop_factors, yield_factors, reference_yield

def parse_dates(date_strings, date_formats = ['%Y-%m-%dT%H%M%S', '%Y-%m-%d', '%Y']):
    for date_format in date_formats:
        try:
            return [datetime.strptime(str(d), date_format) for d in date_strings]
        except ValueError:
            pass
    else:
        raise ValueError('No valid date format found for date strings: {}'.format(date_strings[0]))

def load_crop_ids():
    with open(os.path.join(INPUT, 'crops', 'crop_ids.json'), 'r') as f:
        crop_ids = json.load(f)
    # convert keys to int
    crop_ids = {int(key): value for key, value in crop_ids.items()}
    return crop_ids

def load_economic_data(fp: str) -> tuple[DateIndex, dict[int, np.ndarray]]:
    with open(INPUT / fp, 'r') as f:
        data = json.load(f)
    dates = parse_dates(data['time'])
    date_index = DateIndex(dates)
    d = {
        int(region_id): values
        for region_id, values in data['data'].items()
    }
    return (date_index, d)

def load_sprinkler_prices(self, inflation_rates_per_year):
    sprinkler_price_2008 = self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['adaptation_cost']
    #upkeep_price_2008_m2 = 3000 / 10_000  # ha to m2
    # create dictory with prices for well_prices per year by applying inflation rates
    sprinkler_prices = {2008: sprinkler_price_2008}
    for year in range(2009, 2022):
        sprinkler_prices[year] = sprinkler_prices[year-1] * inflation_rates_per_year[year]
    for year in range(2007, 1960, -1):
        sprinkler_prices[year] = sprinkler_prices[year+1] / inflation_rates_per_year[year+1]
    return sprinkler_prices
