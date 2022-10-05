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
    crops = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx')).set_index('ID')['PRICE'].to_dict()
    fp = os.path.join(INPUT, 'crops', 'crop_prices_rs_per_g.xlsx')
    # TODO: Could do more sophisticated interpolation or obtain data from other states.
    df = pd.read_excel(fp, index_col=0).fillna(method='ffill').fillna(method='bfill')
    date_index = dict(((date, i) for i, date in enumerate(df.index.date)))
    crop_prices = np.full((len(date_index), len(crops)), np.nan, dtype=np.float32)  # first index for date, second index for crops
    print("Deal with sugarcane prices")
    for ID, name in crops.items():
        crop_prices[:, ID] = df[name]
    return date_index, crop_prices

def load_crop_factors() -> dict[np.ndarray]:
    """Read csv-file of values for crop water depletion.
    
    Returns:
        yield_factors: dictonary with np.ndarray of values per crop for each variable.
    """
    crops = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx')).set_index('ID')['GAEZ'].to_dict()
    df = pd.read_excel(os.path.join(INPUT, 'crops', 'GAEZ.xlsx'), index_col=0).loc[crops.values()]
    
    growth_length = np.full((len(crops), 3), np.nan, dtype=np.float32)
    growth_length[:, 0] = df['kharif_d']
    growth_length[:, 1] = df['rabi_d']
    growth_length[:, 2] = df['summer_d']
    assert not np.isnan(growth_length).any()
    
    stage_lengths = np.full((len(crops), 4), np.nan, dtype=np.float32)
    stage_lengths[:,0] = df['d1']
    stage_lengths[:,1] = df['d2a'] + df['d2b']
    stage_lengths[:,2] = df['d3a'] + df['d3b']
    stage_lengths[:,3] = df['d4']
    assert not np.isnan(stage_lengths).any()

    crop_factors = np.full((len(crops), 3), np.nan, dtype=np.float32)
    crop_factors[:,0] = df['Kc1']
    crop_factors[:,1] = df['Kc3']
    crop_factors[:,2] = df['Kc5']
    assert not np.isnan(crop_factors).any()

    yield_factors = {
        'Ky1': df['Ky1'].to_numpy(),
        'Ky2': ((df['Ky2a'] * df['d2a'] + df['Ky2b'] * df['d2b']) / (df['d2a'] + df['d2b'])).to_numpy(),
        'Ky3': ((df['Ky3a'] * df['d3a'] + df['Ky3b'] * df['d3b']) / (df['d3a'] + df['d3b'])).to_numpy(),
        'Ky4': df['Ky4'].to_numpy(),
        'KyT': df['KyT'].to_numpy(),
    }

    print('replace with GAEZ yields, currently MIRCA2000')
    reference_yield = np.array([800, 850, 1500, 1200, 15000, 4000, 1000])  # gr / m2
    
    return growth_length, stage_lengths, crop_factors, yield_factors, reference_yield

def load_crop_names():
    return pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx')).set_index('CENSUS')['ID'].to_dict()

def load_inflation_rates(country):
    fp = os.path.join(INPUT, 'economics', 'WB inflation rates', 'API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_4570810.csv')
    inflation_series = pd.read_csv(fp, index_col=0, skiprows=4).loc[country]
    inflation = {}
    for year in range(1960, 2022):
        inflation[year] = 1 + inflation_series[str(year)] / 100
    return inflation

if __name__ == '__main__':
    # cultivation_costs = load_cultivation_costs()
    # crop_prices = load_crop_prices()
    # crop_yield_factors = load_crop_factors()
    load_inflation_rates('India')