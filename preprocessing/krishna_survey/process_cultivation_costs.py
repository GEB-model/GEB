import os
from datetime import datetime
import json

import numpy as np
import pandas as pd

from preconfig import ORIGINAL_DATA, PREPROCESSING_FOLDER

STATES = ['Maharashtra']
YEARS = list(range(2004, 2019))

# The agricultural crop year in India is from July to June.
input_folder = os.path.join(ORIGINAL_DATA, 'cultivation_costs')

def parse():
    costs = {}
    for year in YEARS:
        cropping_season = f"{year}-{year+1}"
        costs[cropping_season] = {}
        year_end = year + 1
        if year < 2017:
            extension = 'xls'
        else:
            extension = 'xlsx'
        fn = f"{year}-{str(year_end)[2:]}.{extension}"
        xl = pd.ExcelFile(os.path.join(input_folder, fn))
        crops = xl.sheet_names
        for state in STATES:
            # costs[(year, state)] = {}
            for crop in crops:
                df = xl.parse(crop, skiprows=3)
                if 'Sl no' in df.columns:
                    index_col = 'Sl no'
                elif 'Sl No' in df.columns:
                    index_col = 'Sl No'
                else:
                    raise IndexError
                df = df.astype({index_col: str})
                df = df.set_index(index_col)
                if state in df.columns:
                    split_costs = df[state]
                    animal_labour = split_costs['11.2.3']
                    machine_labour = split_costs['11.3.3']
                    seeds = split_costs['11.4']
                    fertilizer = split_costs['11.5.1']
                    manure = split_costs['11.5.2']
                    insecticides = split_costs['11.6']
                    farm_deprecation = split_costs['12.4']

                    costs[cropping_season][(state, crop)] = (animal_labour + machine_labour + seeds + fertilizer + manure + insecticides + farm_deprecation) / 10_000  # rs / ha -> rs / m2

    return pd.DataFrame.from_dict(costs, orient='index')


def get_changes(costs):
    for state in STATES:
        state_data = costs[state]
        changes = np.nanmean(state_data[1:].to_numpy() / state_data[:-1].to_numpy(), axis=1)
        changes = np.insert(changes, 0, np.nan)
        costs[(state, 'changes')] = changes
    return costs

def inter_and_extrapolate(costs):
    for state in STATES:
        state_data = costs[state]

        n = len(state_data)
        for crop in state_data.columns:
            if crop == 'changes':
                continue
            crop_data = state_data[crop].to_numpy()
            k = -1
            while np.isnan(crop_data[k]):
                k -= 1
            for i in range(k+1, 0, 1):
                crop_data[i] = crop_data[i-1] * costs[(state, 'changes')][i]
            k = 0
            while np.isnan(crop_data[k]):
                k += 1
            for i in range(k-1, -1, -1):
                crop_data[i] = crop_data[i+1] / costs[(state, 'changes')][i+1]
            for j in range(0, n):
                if np.isnan(crop_data[j]):
                    k = j
                    while np.isnan(crop_data[k]):
                        k += 1
                    empty_size = k - j
                    step_changes = costs[(state, 'changes')][j:k+1]
                    total_changes = np.prod(step_changes)
                    real_changes = crop_data[k] / crop_data[j-1]
                    scaled_changes = step_changes * (real_changes ** (1 / empty_size)) / (total_changes ** (1 / empty_size))
                    for i, change in zip(range(j, k), scaled_changes):
                        crop_data[i] = crop_data[i-1] * change
            costs[(state, crop)] = crop_data
        costs = costs.drop((state, 'changes'), axis=1)
    
    # assert no nan values in costs
    assert not costs.isnull().values.any()
    return costs

def load_inflation_rates(country):
    fp = os.path.join(ORIGINAL_DATA, 'economics', 'WB inflation rates', 'API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_5551656.csv')
    inflation_series = pd.read_csv(fp, index_col=0, skiprows=4).loc[country]
    inflation = {}
    for year in range(1960, 2022):
        inflation[year] = 1 + inflation_series[str(year)] / 100
    return inflation

def process_additional_years(costs, lower_bound):
    inflation = load_inflation_rates('India')
    for year in range(YEARS[0], lower_bound[0], -1):
        costs.loc[f"{lower_bound[0]}-{lower_bound[1]}"] = costs.loc[f"{lower_bound[0]+1}-{lower_bound[1]+1}"] / inflation[year]
    return costs

if __name__ == '__main__':
    costs = parse()
    costs = get_changes(costs)
    costs = inter_and_extrapolate(costs)
    for year in range(2003, 1959, -1):
        costs = process_additional_years(costs, lower_bound=(year, year+1))
    # sort by index
    costs = costs.sort_index()
    costs = costs['Maharashtra']
    
    # replace index by datetime index with starting date of july
    costs.index = [datetime(year=int(cropping_season.split('-')[0]), month=7, day=1) for cropping_season in costs.index]

    conversion_dict = {
        'Bajra': 'Bajra',
        'Groundnut': 'Groundnut',
        'Jowar': 'Jowar',
        'Paddy': 'Paddy',
        'Sugarcane': 'Sugarcane',
        'Wheat': 'Wheat',
        'Cotton': 'Cotton',
        'Gram': 'Gram',
        'Maize': 'Maize',
        'Moong': 'Moong',
        'Ragi': 'Ragi',
        'Sunflower': 'Sunflower',
        'Arhar': 'Tur'
    }

    costs_dict = {
        'time': costs.index.strftime('%Y-%m-%d').tolist(),
        'crops': {
            conversion_dict[column]: costs[column].tolist()
            for column in conversion_dict.keys()
        },
    }

    folder = os.path.join(PREPROCESSING_FOLDER, 'crops')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'cultivation_costs.json'), 'w') as f:
        json.dump(costs_dict, f)

