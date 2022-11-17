import os

import pandas as pd

from preconfig import ORIGINAL_DATA, INPUT

def prefilter():
    fp = os.path.join(INPUT, 'IHDS.csv')
    if not os.path.exists(fp):
        df = pd.read_csv(os.path.join(ORIGINAL_DATA, 'ICPSR_36151', 'DS0002', '36151-0002-Data.tsv'), delimiter='\t')
        df = df[df['STATEID'] == 27]  # select households from Maharashtra
        df = df[df['FM1'] == 1]  # select only households where primary income is farm.
        df.to_csv(fp)
        print(f'preselected {len(df)} people')
    else:
        df = pd.read_csv(fp, index_col=0)
    return df


def process(df):
    select = {
        'WT': 'weight',
        'DISTID': 'district code',
        'PSUID': 'village/neighborhood code',
        'HQ4': 'household size',
        'FM4A': 'owned kharif',
        'FM4B': 'owned rabi',
        'FM4C': 'owned summer',
        'FM12A': 'irrigated kharif',
        'FM12B': 'irrigated rabi',
        'FM12C': 'irrigated summer',
        'FM13A': 'irrigation type 1',
        'FM13B': 'irrigation type 2',
        'FM27B': 'hired farm labor Rs',
        'FM28A': 'seeds Rs',
        'FM28B': 'seeds homegrown',
        'FM29': 'fertilizers Rs',
        'FM30': 'pesticides Rs',
        'FM31': 'irrigation water Rs',
        'FM32': 'hired equipment/animals rs',
        'FM33': 'Agriculture loan repayment Rs',
        'FM34': 'Farm miscellaneous Rs',
        'FM39B': 'How land acquired',
        'FM40A': 'Own: Tubewells',
        'FM40B': 'Own: Electric Pumps',
        'FM40C': 'Own: Diesel Pumps',
        'FM40I': 'Own: Drip irrigation',
        'FM40J': 'Own: Sprinkler set',
        'FM41A': 'New farm equipt Rs',
        'FM41C': 'Other farm income (rupees)',
        'COTOTAL': 'Total household consumption expenditure',
        'ASSETS': 'Total household assets (0-33)',
        'FM22RSHH': 'Crop income',
    }
    df_selected = df[select.keys()].rename(columns=select)
    df_selected.to_csv(os.path.join(INPUT, 'IHDS_selected.csv'))


if __name__ == '__main__':
    df = prefilter()
    process(df)