import os

import numpy as np
import pandas as pd
import geopandas as gpd

from config import ORIGINAL_DATA, INPUT

YEAR = '2000-01'
SIZE_CLASSES = (
    'Below 0.5',
    '0.5-1.0',
    '1.0-2.0',
    '2.0-3.0',
    '3.0-4.0',
    '4.0-5.0',
    '5.0-7.5',
    '7.5-10.0',
    '10.0-20.0',
    '20.0 & ABOVE',
)
CROPS = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx'))['CENSUS']

fn = os.path.join(ORIGINAL_DATA, 'census', 'output', 'crops', 'crops_WHEAT_2000-01.geojson')
census_data = gpd.read_file(fn)
df = census_data[['State', 'District', 'Tehsil']]

n = len(df)
df = df.loc[df.index.repeat(len(SIZE_CLASSES))]
df['size_class'] = SIZE_CLASSES * n
df = df.set_index(['State', 'District', 'Tehsil', 'size_class']).rename(columns={'size_class': 'size class'})

# for crop in CROPS:
#     fn = os.path.join(ORIGINAL_DATA, 'census', 'output', 'crops', f'crops_{crop.upper()}_{YEAR}.geojson')
#     census_data = gpd.read_file(fn)
#     for _, row in census_data.iterrows():
#         state, district, tehsil = row['State'], row['District'], row['Tehsil']
#         for size_class in SIZE_CLASSES:
#             total_holdings = row[f'{size_class}_total_holdings']
#             if np.isnan(total_holdings):
#                 total_holdings = 0
#             df.at[(state, district, tehsil, size_class), crop] = total_holdings

fn = os.path.join(ORIGINAL_DATA, 'census', 'output', f'farm_size_{YEAR}.geojson')
census_data = gpd.read_file(fn)
for _, row in census_data.iterrows():
    state, district, tehsil = row['State'], row['District'], row['Tehsil']
    for size_class in SIZE_CLASSES:
        total_holdings = row[f'{size_class}_n_total']
        if np.isnan(total_holdings):
            total_holdings = 0
        df.at[(state, district, tehsil, size_class), 'n'] = total_holdings

df.to_excel(os.path.join(INPUT, 'agents', 'tehsil_data.xlsx'))