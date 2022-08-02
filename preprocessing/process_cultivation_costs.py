import os

import numpy as np
import pandas as pd

from config import ORIGINAL_DATA, INPUT

STATES = ['Maharashtra']
YEARS = list(range(2004, 2019))

# The agricultural crop year in India is from July to June.
folder = os.path.join(ORIGINAL_DATA, 'cultivation_costs')
output_folder = os.path.join(INPUT, 'cultivation_costs')
os.makedirs(output_folder, exist_ok=True)

def parse():
    costs = {}
    for year in YEARS:
        cropping_season = f"{year}_{year+1}"
        costs[cropping_season] = {}
        year_end = year + 1
        if year < 2017:
            extension = 'xls'
        else:
            extension = 'xlsx'
        fn = f"{year}-{str(year_end)[2:]}.{extension}"
        xl = pd.ExcelFile(os.path.join(folder, fn))
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


def inter_and_extrapolate(costs):
    for state in STATES:
        state_data = costs[state]
        n = len(state_data)
        changes = np.nanmean(state_data[1:].to_numpy() / state_data[:-1].to_numpy(), axis=1)
        changes = np.insert(changes, 0, np.nan)
        costs['changes'] = changes

        for crop in state_data.columns:
            crop_data = state_data[crop].to_numpy()
            k = -1
            while np.isnan(crop_data[k]):
                k -= 1
            for i in range(k+1, 0, 1):
                crop_data[i] = crop_data[i-1] * changes[i]
            k = 0
            while np.isnan(crop_data[k]):
                k += 1
            for i in range(k-1, -1, -1):
                crop_data[i] = crop_data[i+1] / changes[i+1]
            for j in range(0, n):
                if np.isnan(crop_data[j]):
                    k = j
                    while np.isnan(crop_data[k]):
                        k += 1
                    empty_size = k - j
                    step_changes = changes[j:k+1]
                    total_changes = np.prod(step_changes)
                    real_changes = crop_data[k] / crop_data[j-1]
                    scaled_changes = step_changes * (real_changes ** (1 / empty_size)) / (total_changes ** (1 / empty_size))
                    for i, change in zip(range(j, k), scaled_changes):
                        crop_data[i] = crop_data[i-1] * change

    costs = costs.drop('changes', axis=1)
    return costs

if __name__ == '__main__':
    parsed_costs = parse()
    costs = inter_and_extrapolate(parsed_costs)
    print(costs)
    folder = os.path.join(INPUT, 'cultivation_costs')
    os.makedirs(folder, exist_ok=True)
    fp = os.path.join(folder, 'costs.xlsx')
    costs.to_excel(fp)
    # print(pd.read_excel(fp, header=(0, 1), index_col=0))
