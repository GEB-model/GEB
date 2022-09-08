import os

import numpy as np
import pandas as pd

from config import INPUT

def load_cultivation_costs():
    crops = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx')).set_index('ID')['CULTIVATION_COST'].to_dict()
    
    fp = os.path.join(INPUT, 'crops', 'cultivation_costs.xlsx')
    df = pd.read_excel(fp, index_col=0, header=(0, 1))['Maharashtra']
    date_index = dict(((year, i) for i, year in enumerate(df.index)))

    cultivation_costs = np.full((len(date_index), len(crops)), np.nan, dtype=np.float32)  # first index for date, second index for crops
    for ID, name in crops.items():
        cultivation_costs[:, ID] = df[name]

    return date_index, cultivation_costs

def load_crop_prices():
    fp = os.path.join(INPUT, 'crops', 'crop_prices_rs_per_g.xlsx')
    df = pd.read_excel(fp, index_col=0)
    date_index = dict(((date, i) for i, date in enumerate(df.index.date)))
    crop_prices = np.full((len(date_index), 2), np.nan, dtype=np.float32)  # first index for date, second index for crops
    crop_prices[:, 0] = df['Wheat']
    print("Set proper prices based on mill price for Sugar cane")
    crop_prices[:, 1] = 0.04 # df['Sugarcane']
    return date_index, crop_prices

def load_crop_yield_factors() -> dict[np.ndarray]:
    """Read csv-file of values for crop water depletion. Obtained from Table 2 of this paper: https://doi.org/10.1016/j.jhydrol.2009.07.031
    
    Returns:
        yield_factors: dictonary with np.ndarray of values per crop for each variable.
    """
    # df = pd.read_csv(os.path.join(INPUT, 'crops', 'yield_ratios.csv'))
    # yield_factors = df[['alpha', 'beta', 'P0', 'P1', 'yield_gr_m2']].to_dict(orient='list')
    # crop_yield_factors = {
    #     key: np.array([value[0], value[11]])
    #     for key, value in yield_factors.items()
    # }
    print("get crop yield factors from GAEZ")
    return None

def load_crop_names():
    return pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx')).set_index('CENSUS')['ID'].to_dict()

if __name__ == '__main__':
    # cultivation_costs = load_cultivation_costs()
    # crop_prices = load_crop_prices()
    crop_yield_factors = load_crop_yield_factors()
    print(crop_yield_factors)