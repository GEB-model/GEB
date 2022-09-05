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

def get_crop_table(census_df):
    CROPS = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx'))['CENSUS'].tolist() + ['Other fodder crops'] + ['All crops']
    df = census_df.copy()
    n = len(df)
    df = df.loc[df.index.repeat(len(SIZE_CLASSES))]
    df['size_class'] = SIZE_CLASSES * n
    df = df.set_index(['State', 'District', 'Tehsil', 'size_class'])

    for crop in CROPS:
        fn = os.path.join(ORIGINAL_DATA, 'census', 'output', 'crops', f'crops_{crop.upper()}_{YEAR}.geojson')
        census_data = gpd.read_file(fn)
        for _, row in census_data.iterrows():
            state, district, tehsil = row['State'], row['District'], row['Tehsil']
            for size_class in SIZE_CLASSES:
                total_holdings = row[f'{size_class}_total_holdings']
                irrigated_area = row[f'{size_class}_irrigated_area']
                rainfed_area = row[f'{size_class}_unirrigated_area']

                if np.isnan(total_holdings) or total_holdings == 0:
                    rainfed_holdings = 0
                    irrigated_holdings = 0
                else:
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
                    
                    assert irrigated_holdings + rainfed_holdings == total_holdings
                
                df.at[(state, district, tehsil, size_class), crop + '_irr_holdings'] = irrigated_holdings
                df.at[(state, district, tehsil, size_class), crop + '_rain_holdings'] = rainfed_holdings

                if irrigated_area is None:
                    irrigated_area = 0
                elif isinstance(irrigated_area, float):
                    if np.isnan(irrigated_area):
                        irrigated_area = 0
                elif isinstance(irrigated_area, str):
                    if irrigated_area.isdigit():
                        irrigated_area = int(irrigated_area)
                    elif irrigated_area == 'Neg':
                        irrigated_area = 0
                    else:
                        raise ValueError

                if rainfed_area is None:
                    rainfed_area = 0
                elif isinstance(rainfed_area, float):
                    if np.isnan(rainfed_area):
                        rainfed_area = 0
                elif isinstance(rainfed_area, str):
                    if rainfed_area.isdigit():
                        rainfed_area = int(rainfed_area)
                    elif rainfed_area == 'Neg':
                        rainfed_area = 0
                    else:
                        raise ValueError
                
                df.at[(state, district, tehsil, size_class), crop + '_irr_area'] = irrigated_area
                df.at[(state, district, tehsil, size_class), crop + '_rain_area'] = rainfed_area

                del irrigated_holdings
                del rainfed_holdings

    # threat other fodder crops as bajra
    df['Bajra_irr_holdings'] = df['Bajra_irr_holdings'] + df['Other fodder crops_irr_holdings']
    df['Bajra_rain_holdings'] = df['Bajra_rain_holdings'] + df['Other fodder crops_rain_holdings']
    df['Bajra_irr_area'] = df['Bajra_irr_area'] + df['Other fodder crops_irr_area']
    df['Bajra_rain_area'] = df['Bajra_rain_area'] + df['Other fodder crops_rain_area']
    df = df.drop(['Other fodder crops_irr_holdings', 'Other fodder crops_rain_holdings', 'Other fodder crops_irr_area', 'Other fodder crops_rain_area'], axis=1)

    df['holdings_total'] = df[[column for column in df.columns if column.endswith('_holdings')]].sum(axis=1)
    df['area_total'] = df[[column for column in df.columns if column.endswith('_area')]].sum(axis=1)

    df.to_excel(os.path.join(INPUT, 'census', 'crop_data.xlsx'))

def get_farm_size_table(census_df):
    df = census_df.copy()
    df = df.set_index(['State', 'District', 'Tehsil'])
    fn = os.path.join(ORIGINAL_DATA, 'census', 'output', f'farm_size_{YEAR}.geojson')
    census_data = gpd.read_file(fn)
    
    for _, row in census_data.iterrows():
        state, district, tehsil = row['State'], row['District'], row['Tehsil']

        for size_class in SIZE_CLASSES:
            avg_area = row[f'{size_class}_area_total'] / row[f'{size_class}_n_total'] * 10_000  # ha -> m2
            df.at[(state, district, tehsil), size_class] = avg_area

    df.to_excel(os.path.join(INPUT, 'census', 'avg_farm_size.xlsx'))


def get_farm_count_table(census_df):
    df = census_df.copy()
    df = df.set_index(['State', 'District', 'Tehsil'])
    fn = os.path.join(ORIGINAL_DATA, 'census', 'output', f'farm_size_{YEAR}.geojson')
    census_data = gpd.read_file(fn)
    
    for _, row in census_data.iterrows():
        state, district, tehsil = row['State'], row['District'], row['Tehsil']

        for size_class in SIZE_CLASSES:
            df.at[(state, district, tehsil), size_class] = row[f'{size_class}_n_total']

    df.to_excel(os.path.join(INPUT, 'census', 'n_farms.xlsx'))


if __name__ == '__main__':
    os.makedirs(os.path.join(INPUT, 'census'), exist_ok=True)
    fn = os.path.join(INPUT, 'tehsils.geojson')
    census_data = gpd.read_file(fn)
    census_df = census_data[['State', 'District', 'Tehsil']]
    get_farm_size_table(census_df)
    get_farm_count_table(census_df)
    get_crop_table(census_df)