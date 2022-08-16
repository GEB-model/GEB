import os
import math
import statistics

import numpy as np
import pandas as pd

from config import ORIGINAL_DATA, INPUT

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

    return costs

def add_tomatoes_maharashtra(costs):
    # https://www.chemijournal.com/archives/2019/vol7issue4/PartW/7-4-55-593.pdf
    # http://iasir.net/AIJRFANSpapers/AIJRFANS15-342.pdf
    # https://www.phytojournal.com/archives/2021/vol10issue1S/PartH/S-10-1-68-438.pdf
    # https://www.thepharmajournal.com/archives/2021/vol10issue7S/PartH/S-10-6-129-886.pdf
    state = 'Maharashtra'
    costs.insert(0, (state, 'Tomato'), np.nan)
    tomatoes = pd.read_excel(os.path.join(input_folder, 'tomato.xlsx'), skiprows=3)
    index_col = 'Sl no'
    tomatoes = tomatoes.astype({index_col: str})
    tomatoes = tomatoes.set_index(index_col)
    tomatoes = tomatoes[[column for column in tomatoes.columns if column.startswith('#')]]
    years = tomatoes.loc['Year']
    animal_labour = tomatoes.loc['11.2.3']
    machine_labour = tomatoes.loc['11.3.3']
    seeds = tomatoes.loc['11.4']
    fertilizer = tomatoes.loc['11.5.1']
    manure = tomatoes.loc['11.5.2']
    insecticides = tomatoes.loc['11.6']
    farm_deprecation = tomatoes.loc['12.4']

    tomato_total_costs = (animal_labour + machine_labour + seeds + fertilizer + manure + insecticides + farm_deprecation) / 10_000  # rs / ha -> rs / m2

    costs_base_year = []
    for study in tomato_total_costs.index:
        year = years.at[study]
        cost = tomato_total_costs.at[study]
        changes_per_year = costs[(state, 'changes')].iloc[1:costs.index.get_loc(year)+1]
        total_change = math.prod(changes_per_year)
        cost_base_year = cost / total_change
        costs_base_year.append(cost_base_year)

    cost_base_year = statistics.mean(costs_base_year)
    costs.loc[costs.index[0], (state, 'Tomato')] = cost_base_year
    for i in range(1, len(costs)):
        costs.loc[costs.index[i], (state, 'Tomato')] = costs.loc[costs.index[i-1], (state, 'Tomato')] * costs.loc[costs.index[i], (state, 'changes')]

    return costs


if __name__ == '__main__':
    costs = parse()
    costs = get_changes(costs)
    costs = inter_and_extrapolate(costs)
    costs = add_tomatoes_maharashtra(costs)

    print(costs)
    folder = os.path.join(INPUT, 'crops')
    os.makedirs(folder, exist_ok=True)
    fp = os.path.join(folder, 'cultivation_costs.xlsx')
    costs.to_excel(fp)
    # print(pd.read_excel(fp, header=(0, 1), index_col=0))
