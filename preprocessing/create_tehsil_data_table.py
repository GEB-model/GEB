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
CROPS = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx'))['CENSUS'].tolist()

fn = os.path.join(ORIGINAL_DATA, 'census', 'output', 'crops', 'crops_WHEAT_2000-01.geojson')
census_data = gpd.read_file(fn)
df = census_data[['State', 'District', 'Tehsil']]

n = len(df)
df = df.loc[df.index.repeat(len(SIZE_CLASSES))]
df['size_class'] = SIZE_CLASSES * n
# crops = CROPS * n * len(SIZE_CLASSES)
# crops_irr_rain = []
# for crop in crops:
#     crops_irr_rain.append(crop + '_irr')
#     crops_irr_rain.append(crop + '_rain')
# df['crop'] = crops_irr_rain
df = df.set_index(['State', 'District', 'Tehsil', 'size_class']).rename(columns={'size_class': 'size class'})

for crop in CROPS:
    fn = os.path.join(ORIGINAL_DATA, 'census', 'output', 'crops', f'crops_{crop.upper()}_{YEAR}.geojson')
    census_data = gpd.read_file(fn)
    for _, row in census_data.iterrows():
        state, district, tehsil = row['State'], row['District'], row['Tehsil']
        for size_class in SIZE_CLASSES:
            total_holdings = row[f'{size_class}_total_holdings']
            if np.isnan(total_holdings) or total_holdings == 0:
                rainfed_holdings = 0
                irrigated_holdings = 0
            else:
                irrigated_area = row[f'{size_class}_irrigated_area']
                rainfed_area = row[f'{size_class}_unirrigated_area']
                if isinstance(irrigated_area, str) and irrigated_area.isdigit():
                    irrigated_area = int(irrigated_area)
                if isinstance(rainfed_area, str) and rainfed_area.isdigit():
                    rainfed_area = int(rainfed_area)
                if isinstance(rainfed_area, str) and isinstance(irrigated_area, str):
                    rainfed_holdings = total_holdings
                    irrigated_holdings = 0
                elif isinstance(rainfed_area, str):
                    if rainfed_area == 'Neg' and irrigated_area == 0:
                        assert total_holdings < 20
                        rainfed_holdings = total_holdings
                        irrigated_holdings = 0
                    elif rainfed_area == 'Neg' and irrigated_area > 0:
                        irrigated_holdings = total_holdings
                        rainfed_holdings = 0
                    else:
                        raise ValueError
                elif isinstance(irrigated_area, str):
                    if irrigated_area == 'Neg' and rainfed_area == 0:
                        assert total_holdings < 20
                        irrigated_holdings = total_holdings
                        rainfed_holdings = 0
                    elif irrigated_area == 'Neg' and rainfed_area > 0:
                        rainfed_holdings = total_holdings
                        irrigated_holdings = 0
                    else:
                        raise ValueError
                else:
                    irrigated_ratio = irrigated_area / (irrigated_area + rainfed_area)
                    irrigated_holdings = round(total_holdings * irrigated_ratio)
                    rainfed_holdings = total_holdings - irrigated_holdings
            df.at[(state, district, tehsil, size_class), crop + '_irr'] = irrigated_holdings
            df.at[(state, district, tehsil, size_class), crop + '_rain'] = rainfed_holdings
            del irrigated_holdings
            del rainfed_holdings

# fn = os.path.join(ORIGINAL_DATA, 'census', 'output', f'farm_size_{YEAR}.geojson')
# census_data = gpd.read_file(fn)
# for _, row in census_data.iterrows():
#     state, district, tehsil = row['State'], row['District'], row['Tehsil']
#     for size_class in SIZE_CLASSES:
#         total_holdings = row[f'{size_class}_n_total']
#         if np.isnan(total_holdings):
#             total_holdings = 0
#         df.at[(state, district, tehsil, size_class), 'n'] = total_holdings

os.makedirs(os.path.join(INPUT, 'census'), exist_ok=True)
df.to_excel(os.path.join(INPUT, 'census', 'crop_data.xlsx'))