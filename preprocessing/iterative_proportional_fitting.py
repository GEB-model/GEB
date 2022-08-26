import os

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from methods import create_cell_area_map
from ipl import IPL

from config import INPUT

SEASONS = ['Kharif', 'Rabi', 'Summer']
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
CROPS = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx'))

with rasterio.open(os.path.join(INPUT, 'tehsils.tif'), 'r') as src:
    tehsils_tif = src.read(1)
    cell_area = create_cell_area_map(src.profile, write_to_disk=True)

tehsils_shape = gpd.read_file(os.path.join(INPUT, 'tehsils.geojson')).set_index(['State', 'District', 'Tehsil'])
avg_farm_size = pd.read_excel(os.path.join(INPUT, 'census', 'avg_farm_size.xlsx'), index_col=(0, 1, 2))
crop_data = pd.read_excel(os.path.join(INPUT, 'census', 'crop_data.xlsx'), index_col=(0, 1, 2, 3))
for (state, district, tehsil), tehsil_crop_data in crop_data.groupby(level=[0, 1, 2]):
    # tehsil_farm_size = avg_farm_size.loc[(state, district, tehsil)]
    farms_per_size_class = tehsil_crop_data.droplevel([0, 1, 2]).sum(axis=1)

    # assert (tehsil_farm_size.index == farms_per_size_class.index).all()

    # area_per_size_class = tehsil_farm_size * farms_per_size_class
    # census_farm_area = area_per_size_class.sum()

    tehsil_ID = tehsils_shape.at[(state, district, tehsil), 'ID']
    tehsil_area = cell_area[tehsils_tif == tehsil_ID].sum()

columns = [f'{crop}_irr_holdings' for crop in CROPS['CENSUS'].tolist() + ['All crops']] + [f'{crop}_rain_holdings' for crop in CROPS['CENSUS'].tolist() + ['All crops']]
crop_data = crop_data[columns]
crop_data = crop_data.rename(columns={
    column: column.replace('_holdings', '')
    for column in columns
})

n_farms = pd.read_excel(os.path.join(INPUT, 'census', 'n_farms.xlsx'), index_col=(0, 1, 2))

size_class_convert = {
    "Below 0.5": 0,
    "0.5-1.0": 1,
    "1.0-2.0": 2,
    "2.0-3.0": 3,
    "3.0-4.0": 4,
    "4.0-5.0": 5,
    "5.0-7.5": 6,
    "7.5-10.0": 7,
    "10.0-20.0": 8,
    "20.0 & ABOVE": 9,
}

def assign_size_classes(survey):
    size_classes = (
        ('Below 0.5', 0.5),
        ('0.5-1.0', 1),
        ('1.0-2.0', 2),
        ('2.0-3.0', 3),
        ('3.0-4.0', 4),
        ('4.0-5.0', 5),
        ('5.0-7.5', 7.5),
        ('7.5-10.0', 10),
        ('10.0-20.0', 20),
        ('20.0 & ABOVE', np.inf),
    )
    for idx, household in survey.iterrows():
        area = household['area owned & cultivated']
        for size_class_name, size in size_classes:
            if area < size:
                survey.loc[idx, 'size_class'] = size_class_name
                break
    return survey

survey_data = pd.read_csv(os.path.join(INPUT, 'agents', 'IHDS_I.csv'))
survey_data = assign_size_classes(survey_data)
survey_data[~(survey_data['Kharif: Crop: Name'].isnull() & survey_data['Rabi: Crop: Name'].isnull() & survey_data['Summer: Crop: Name'].isnull())]

for season in SEASONS:
    survey_data[f'{season}: Crop: Irrigation'] = survey_data[f'{season}: Crop: Irrigation'].map({'Yes': 'irr', 'No': 'rain'})

crop_convert = CROPS.set_index('IHDS')['CENSUS'].to_dict()
for season in SEASONS:
    survey_data[f'{season}: Crop: Name'] = survey_data[f'{season}: Crop: Name'].map(crop_convert)

print("Also remove households where other crops are grown?")
survey_data = survey_data[~(survey_data['Kharif: Crop: Name'].isnull() & survey_data['Rabi: Crop: Name'].isnull() & survey_data['Summer: Crop: Name'].isnull())]

# Check if irrigation is assigned to all crops
for season in SEASONS:
    assert (survey_data[~(survey_data[f'{season}: Crop: Name'].isnull())][f'{season}: Crop: Irrigation'].isnull() == False).all()

survey_data['crops'] = list(zip(
    list(np.where(survey_data['Kharif: Crop: Name'].isnull(), None, survey_data['Kharif: Crop: Name'] + '_' + survey_data['Kharif: Crop: Irrigation'])),
    list(np.where(survey_data['Rabi: Crop: Name'].isnull(), None, survey_data['Rabi: Crop: Name'] + '_' + survey_data['Rabi: Crop: Irrigation'])),
    list(np.where(survey_data['Summer: Crop: Name'].isnull(), None, survey_data['Summer: Crop: Name'] + '_' + survey_data['Summer: Crop: Irrigation'])),
))
# survey_data['size_class'] = survey_data['size_class'].map(size_class_convert)

folder = os.path.join(INPUT, 'agents', 'population')
os.makedirs(folder, exist_ok=True)

for (state, district, tehsil, size_class), crop_frequencies in crop_data.groupby(crop_data.index):
    print(state, district, tehsil, size_class)
    # survey_data_size_class = survey_data[survey_data['size_class'] == size_class]
    print('split by size class')
    survey_data_size_class = survey_data
    survey_data_size_class = survey_data_size_class.reset_index(drop=True)

    crops = crop_frequencies.iloc[0]
    crops.name = 'crops'
    aggregates = [crops]

    n = int(n_farms.loc[(state, district, tehsil), size_class])
    ipl = IPL(
        original=survey_data_size_class,
        aggregates=aggregates,
        n=n,
        learning_rate=1
    ).iteration()
    ipl['n'] = ipl['weight'] // 1
    missing = n - int(ipl['n'].sum())
    
    index = list(zip(ipl.index, ipl['weight'] % 1))
    missing_pop = sorted(index, key=lambda x: x[1], reverse=True)[:missing]
    missing_pop = [p[0] for p in missing_pop]
    ipl.loc[missing_pop, 'n'] += 1

    assert ipl['n'].sum() == n

    cutoff_value = np.sort(ipl['weight'] % 1)[::-1][missing]
    missing_pop = np.where(ipl['weight'] > cutoff_value)[0]

    population = ipl.loc[ipl.index.repeat(ipl['n'])]
    population = population.drop(['crops', 'weight'], axis=1)

    fp = os.path.join(folder, f"{state}_{district}_{tehsil}_{size_class}.csv")
    population.to_csv(fp, index=False)