import os
import pandas as pd
import geopandas as gpd

from preconfig import INPUT, ORIGINAL_DATA

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

census_folder = os.path.join(ORIGINAL_DATA, 'census')

def parse(x):
    if x is None:
        return 0
    elif pd.isnull(x):
        return 0
    elif isinstance(x, float):
        return x
    elif x.isdigit():
        return int(x)
    elif x == 'Neg':
        return 0
    else:
        print(x)
        exit()

crop_area_csv = os.path.join(INPUT, 'crops', 'area.csv')
if not os.path.exists(crop_area_csv):
    os.makedirs(os.path.join(INPUT, 'crops'), exist_ok=True)
    with open(crop_area_csv, 'w') as f:
        f.write('crop,area\n')
        output_folder = os.path.join(census_folder, 'output', 'crops')
        for fn in os.listdir(output_folder):
            if YEAR not in fn:
                continue
            crop = fn[6:-16]
            if crop.startswith('ALL CROPS'):
                continue
            if crop.startswith('ALL VEGETABLES'):
                continue
            if crop.startswith('TOTAL'):
                continue
            if crop == 'FODDER & GREEN MANURES': # Same as other fodder crops
                continue
            fp = os.path.join(output_folder, fn)
            gdf = gpd.read_file(fp)
            crop_area = 0
            for size_class in SIZE_CLASSES:
                column = f"{size_class}_total_area"
                if column not in gdf.columns:
                    continue
                crop_area += gdf[column].apply(lambda x: parse(x)).sum()
            f.write(f"{crop},{crop_area}\n")

df = pd.read_csv(crop_area_csv)
print(df[df['area'] != 0])
total_area = df['area'].sum()
print('total area', total_area)
df['area_percentage'] = df['area'] / total_area * 100
more_than_2_percent = df[df['area_percentage'] > 2]
print(more_than_2_percent)
print(len(more_than_2_percent))
print(more_than_2_percent['area_percentage'].sum())

