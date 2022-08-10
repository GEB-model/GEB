import os

import pandas as pd
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from config import INPUT

base = importr('base')
ipfr = importr('ipfr')
dplyr = importr('dplyr')
rtibble = importr('tibble')

base.set_seed(42)

def df_to_tibble(df):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_dataframe = ro.conversion.py2rpy(df)
    return rtibble.as_tibble(r_dataframe)

def tibble_to_df(tibble):
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(tibble)

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

survey_data = pd.read_csv(os.path.join(INPUT, 'agents', 'IHDS_I.csv')).rename(columns={'size class': 'size_class'})
survey_data['size_class'] = survey_data['size_class'].map(size_class_convert)
survey = df_to_tibble(survey_data[['size_class', 'weight']])

tehsil_data = pd.read_excel(os.path.join(INPUT, 'agents', 'tehsil_data.xlsx'), index_col=(0, 1, 2))
tehsil_data.index = tehsil_data.index.to_flat_index()

folder = os.path.join(INPUT, 'agents', 'population')
os.makedirs(folder, exist_ok=True)

for tehsil, tehsil_data in tehsil_data.groupby(tehsil_data.index):
    targets = ro.ListVector({
        'size_class': df_to_tibble(pd.DataFrame([
            tehsil_data['n'].tolist(),
        ], columns=tehsil_data['size_class'].map(size_class_convert).tolist()))
    })

    result = ipfr.ipu(survey, targets)

    base.set_seed(42)

    population = tibble_to_df(ipfr.synthesize(result.rx2('weight_tbl'))).drop(['new_id', 'size_class'], axis=1)
    population['id'] -= 1  # R is 1-indexed and Python is 0-indexed.
    population = population.merge(survey_data, how='left', left_on='id', right_index=True)
    population = population.drop(['id', 'weight', 'State code', 'District code', 'PSU: village/neighborhood code', 'Household ID', 'Split household ID', 'Full household ID'], axis=1)

    fp = os.path.join(folder, "_".join(tehsil) + '.csv')
    population.to_csv(fp, index=False)