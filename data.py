import os
from datetime import date
import numpy as np
import pandas as pd

from config import INPUT, ORIGINAL_DATA, DATA_FOLDER

def load_cultivation_costs(inflation_rates):
    start_year = min(inflation_rates.keys())
    end_year = max(inflation_rates.keys())
    
    crops = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx')).set_index('ID')['CULTIVATION_COST'].to_dict()

    fp = os.path.join(DATA_FOLDER, 'GEB', 'input', 'crops', 'cultivation_costs.xlsx')
    df = pd.read_excel(fp, index_col=0, header=(0, 1))['Maharashtra']

    # Convert the first index to a datetime index 
    df.index = pd.to_datetime(df.index.str.split('-').str[0], format='%Y')
    # select only dates before year 2020
    df = df[df.index.year <= end_year]
    # reindex to include all years starting from start_year
    df = df.reindex(pd.date_range(start=date(start_year, 1, 1), end=df.index[-1], freq='YS'))
    
    # interpolate missing values
    df = df.interpolate(method='linear', axis=0, limit_area='inside')

    # fill missing values, while correcting for inflation, historically
    for column_idx, column in enumerate(df.columns):
        # find first non-missing index
        first_non_missing_idx = df.index.get_loc(df[column].first_valid_index())
        # find index value before first missing value
        for idx in range(first_non_missing_idx, -1, -1):
            df.iloc[idx, column_idx] = df.iloc[idx + 12, column_idx] / inflation_rates[df.index[idx].year+1]
   
   # fill missing values, while correcting for inflation, future part
    for column_idx, column in enumerate(df.columns):
        # find first non-missing index
        last_non_missing_idx = df.index.get_loc(df[column].last_valid_index())
        # find index value before first missing value
        for idx in range(last_non_missing_idx, len(df)):
            df.iloc[idx, column_idx] = df.iloc[idx - 12, column_idx] * inflation_rates[df.index[idx].year]
    
    # This changes it into a timestamp object for some reason. TO DO: change to datetime 
    date_index = dict(((year, i) for i, year in enumerate(df.index)))

    cultivation_costs = np.full((len(date_index), len(crops)), np.nan, dtype=np.float32)  # first index for date, second index for crops
    for ID, name in crops.items():
        cultivation_costs[:, ID] = df[name]

    return date_index, cultivation_costs

def load_crop_prices(state2int: dict, inflation_rates) -> tuple[dict[dict[date, int]], dict[str, np.ndarray]]:
    """Load crop prices per state from the input data and return a dictionary of states containing 2D array of prices.
    
    Returns:
        date_index: Dictionary of states containing a dictionary of dates and their index in the 2D array.
        crop_prices: Dictionary of states containing a 2D array of crop prices. First index is for date, second index is for crop."""
    sugarcane_FRP = pd.read_excel(os.path.join(ORIGINAL_DATA, 'crop_prices', 'FRP.xlsx')).set_index('Year')  # Fair and Remunerative Price

    start_year = min(inflation_rates.keys())
    end_year = max(inflation_rates.keys())

    crops = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx')).set_index('ID')['PRICE'].to_dict()
    folder = os.path.join(INPUT, 'crops', 'crop_prices_rs_per_g')
    crop_prices = None
    date_index = None
    for fn in os.listdir(folder):
        assert fn.endswith('.xlsx')
        state = fn.replace('.xlsx', '')
        if state in state2int:
            state_index = state2int[state]
            # TODO: Could do more sophisticated interpolation or obtain data from other states.
            agmarknet_prices = pd.read_excel(os.path.join(folder, fn), index_col=0)
            # select only dates before year 2020
            agmarknet_prices = agmarknet_prices[agmarknet_prices.index.year <= end_year]
            # reindex to include all years starting from start_year
            agmarknet_prices = agmarknet_prices.reindex(pd.date_range(start=date(start_year, 1, 1), end=agmarknet_prices.index[-1], freq='MS'))
            
            sugarcane_prices = sugarcane_FRP.copy()
            start_year_sugarcane = int(sugarcane_FRP.index[0][:4])
            end_year_sugarcane = int(sugarcane_FRP.index[-1][-4:])
            
            sugarcane_prices.index = pd.date_range(start=date(start_year_sugarcane, 1, 1), end=date(end_year_sugarcane, 1, 1), freq='YS', inclusive='left') + pd.DateOffset(months=6)
            sugarcane_prices = sugarcane_prices.reindex(pd.date_range(start=date(start_year_sugarcane, 7, 1), end=date(end_year_sugarcane, 7, 1), freq='MS', inclusive='left'))
            sugarcane_prices = sugarcane_prices[sugarcane_prices.index.year <= end_year]
            sugarcane_prices = sugarcane_prices.interpolate(method='ffill', axis=0)
            # set sugarcane prices in agmarknet_prices
            agmarknet_prices = agmarknet_prices.assign(Sugarcane=sugarcane_prices['Sugarcane'])

            # interpolate missing values
            agmarknet_prices = agmarknet_prices.interpolate(method='linear', axis=0, limit_area='inside')
            # ensure each column has at least 12 non-missing values
            for column in agmarknet_prices.columns:
                if agmarknet_prices[column].count() < 12:
                    raise ValueError(f"Column {column} has less than 12 non-missing values")
            
            # fill missing values, while correcting for inflation, historically
            for column_idx, column in enumerate(agmarknet_prices.columns):
                # find first non-missing index
                first_non_missing_idx = agmarknet_prices.index.get_loc(agmarknet_prices[column].first_valid_index())
                # find index value before first missing value
                for idx in range(first_non_missing_idx, -1, -1):
                    agmarknet_prices.iloc[idx, column_idx] = agmarknet_prices.iloc[idx + 12, column_idx] / inflation_rates[agmarknet_prices.index[idx].year+1]

            # fill missing values, while correcting for inflation, future part
            for column_idx, column in enumerate(agmarknet_prices.columns):
                # find first non-missing index
                last_non_missing_idx = agmarknet_prices.index.get_loc(agmarknet_prices[column].last_valid_index())
                # find index value before first missing value
                for idx in range(last_non_missing_idx, len(agmarknet_prices)):
                    agmarknet_prices.iloc[idx, column_idx] = agmarknet_prices.iloc[idx - 12, column_idx] * inflation_rates[agmarknet_prices.index[idx].year]

            if not date_index:
                date_index = dict(((date, i) for i, date in enumerate(agmarknet_prices.index.date)))
            else:
                assert date_index == dict(((date, i) for i, date in enumerate(agmarknet_prices.index.date)))
            if crop_prices is None:
                crop_prices = np.full((len(date_index), len(state2int), len(crops)), np.nan, dtype=np.float32)  # first index for date, second for state, third index for crops
            
            for ID, name in crops.items():
                crop_prices[:, state_index, ID] = agmarknet_prices[name]
    
    assert not np.isnan(crop_prices).any()
    return date_index, crop_prices

def load_crop_factors() -> dict[np.ndarray]:
    """Read csv-file of values for crop water depletion.
    
    Returns:
        yield_factors: dictonary with np.ndarray of values per crop for each variable.
    """
    crops = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx')).set_index('ID')['GAEZ'].to_dict()
    df = pd.read_excel(os.path.join(ORIGINAL_DATA, 'crops', 'GAEZ.xlsx'), index_col=0).loc[crops.values()]
    
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

    # MIRCA2000 reference yields
    reference_yield = df['reference_yield_gr_m2'].to_numpy()
    assert not np.isnan(reference_yield).any()
    
    return growth_length, stage_lengths, crop_factors, yield_factors, reference_yield

def load_crop_names():
    return pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx')).set_index('CENSUS')['ID'].to_dict()

def load_inflation_rates(country):
    fp = os.path.join(ORIGINAL_DATA, 'economics', 'WB inflation rates', 'API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_4570810.csv')
    inflation_series = pd.read_csv(fp, index_col=0, skiprows=4).loc[country]
    inflation = {}
    for year in range(1960, 2022):
        inflation[year] = 1 + inflation_series[str(year)] / 100
    return inflation

def load_lending_rates(country):
    fp = os.path.join(ORIGINAL_DATA, 'economics', 'WB lending interest rates', 'API_FR.INR.LEND_DS2_en_csv_v2_4772904.csv')
    lending_series = pd.read_csv(fp, index_col=0, skiprows=4).loc[country]
    lending = {}
    for year in range(1960, 2022):
        lending[year] = lending_series[str(year)] / 100
    return lending

def load_well_prices(self, inflation_rates_per_year):
    well_price_2008 = self.model.config['agent_settings']['expected_utility']['adaptation_well']['adaptation_cost']
    upkeep_price_2008_m2 = self.model.config['agent_settings']['expected_utility']['adaptation_well']['upkeep_costs'] / 10_000  # ha to m2
    # create dictory with prices for well_prices per year by applying inflation rates 
    well_prices = {2008: well_price_2008}
    for year in range(2009, 2022):
        well_prices[year] = well_prices[year-1] * inflation_rates_per_year[year]
    for year in range(2007, 1960, -1):
        well_prices[year] = well_prices[year+1] / inflation_rates_per_year[year+1]
    # do the same for upkeep price
    upkeep_prices = {2008: upkeep_price_2008_m2}
    for year in range(2009, 2022):
        upkeep_prices[year] = upkeep_prices[year-1] * inflation_rates_per_year[year]
    for year in range(2007, 1960, -1):
        upkeep_prices[year] = upkeep_prices[year+1] / inflation_rates_per_year[year+1]
    return well_prices, upkeep_prices

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



if __name__ == '__main__':
    # cultivation_costs = load_cultivation_costs()
    # crop_prices = load_crop_prices()
    # crop_yield_factors = load_crop_factors()
    inflation_rates = load_inflation_rates('India')
    well_prices = load_well_prices(inflation_rates)
    load_lending_rates('India')