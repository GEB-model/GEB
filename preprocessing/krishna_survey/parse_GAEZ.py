import os
import pandas as pd
import yaml

from preconfig import ORIGINAL_DATA

df = pd.read_excel(os.path.join(ORIGINAL_DATA, 'crops', 'GAEZ.xlsx')).set_index('ID').drop('Name', axis=1)
# drop all rows where index is NaN
df = df[~df.index.isna()]
# split rows where multiple IDs are listed in the index
new_rows = []
for index, row in df.iterrows():
    if isinstance(index, str):
        for ID in index.split(','):
            new_row = row.copy()
            new_row.name = int(ID)
            new_rows.append(new_row)
    else:
        new_row = row.copy()
        new_row.name = index
        new_rows.append(new_row)

df = pd.DataFrame(new_rows).sort_index()

assert len(df) == max(df.index) + 1
df = df.drop(['kcT', 'Ky1', 'Ky2a', 'Ky2b', 'Ky3a', 'Ky3b', 'Ky4'], axis=1)

crop_data = df.to_dict(orient='index')
# export to yml
with open(os.path.join(ORIGINAL_DATA, 'crops', 'crop_data.yml'), 'w') as f:
    yaml.dump(crop_data, f, default_flow_style=False)
