import os

import numpy as np
import pandas as pd

from config import ORIGINAL_DATA, INPUT

def prefilter():
    df = pd.read_csv(os.path.join(ORIGINAL_DATA, 'ICPSR_22626', 'DS0002', '22626-0002-Data.tsv'), delimiter='\t')
    df = df[df['STATEID'] == 27]  # select households from Maharashtra
    df = df[df['FM3'] == 1]  # select only households where primary income is farm.
    print(f'preselected {len(df)} households')
    return df

def read_crops():
    df = pd.read_csv(os.path.join(ORIGINAL_DATA, 'ICPSR_22626', 'DS0008', '22626-0008-Data.tsv'), delimiter='\t')
    return df


def process(households, crops):
    select_households = {
        'SWEIGHT': 'weight',
        'STATEID': 'State code',
        'DISTID': 'District code',
        'PSUID': 'PSU: village/neighborhood code',
        'HHID': 'Household ID',
        'HHSPLITID': 'Split household ID',
        'IDHH': 'Full household ID',
        'NPERSONS': 'household size',
        'FM5': 'area owned & cultivated',
        'FM9A': 'Irrigation',
        'FM9B': 'Irrigation type 1',
        'FM9C': 'Irrigation type 2',
        'FM20B': 'Hired farm labour Rs',
        'FM21A': 'Seeds Rs',
        'FM21B': 'Seeds Homegrown',
        'FM22': 'Fertilizers Rs',
        'FM23': 'Pesticides Rs',
        'FM24': 'Irrigation water Rs',
        'FM25': 'Hired Equipt/Animals Rs',
        'FM26': 'Ag loan repayment Rs',
        'FM27': 'Farm miscellaneous Rs',
        'FM32A': 'Own: Tubewells',
        'FM32B': 'Own: Electric Pumps',
        'FM32C': 'Own: Diesel Pumps',
        'FM32H': 'New farm equipt Rs',
    }
    households = households[select_households.keys()].rename(columns=select_households)
    households = households[households['area owned & cultivated'] != ' ']
    households['area owned & cultivated'] = households['area owned & cultivated'].astype(float)
    households = households[households['area owned & cultivated'] > 0]
    
    select_crops = {
        'idhfp': 'Full household ID',
        'fm10': 'Crop: Season',
        'fm12': 'Crop: Name',
        'fm13': 'Crop: Irrigation'
    }
    crops = crops[select_crops.keys()].rename(columns=select_crops)

    def rename_and_drop(df, season):
        return df.rename(columns={
            'Crop: Name': f'{season}: Crop: Name',
            'Crop: Irrigation': f'{season}: Crop: Irrigation',
        }).drop('Crop: Season', axis=1)
    
    kharif_crops = rename_and_drop(crops[crops['Crop: Season'] == 1], 'Kharif')
    rabi_crops = rename_and_drop(crops[crops['Crop: Season'] == 2], 'Rabi')
    summer_crops = rename_and_drop(crops[crops['Crop: Season'] == 3], 'Summer')

    households = households.merge(kharif_crops, how='left', on='Full household ID')
    households = households.merge(rabi_crops, how='left', on='Full household ID')
    households = households.merge(summer_crops, how='left', on='Full household ID')

    households = households[~(households['Kharif: Crop: Name'].isnull() & households['Rabi: Crop: Name'].isnull() & households['Summer: Crop: Name'].isnull())]

    return households

def rename_parameters(households):
    households['Irrigation type 1'] = households['Irrigation type 1'].map({
        '1': 'Tubewell',
        '2': 'Other well',
        '3': 'Government',
        '4': 'Private canal',
        '5': 'Tank/pond/nala',
        '6': 'Other',
    })
    households['Irrigation type 2'] = households['Irrigation type 2'].map({
        '1': 'Tubewell',
        '2': 'Other well',
        '3': 'Government',
        '4': 'Private canal',
        '5': 'Tank/pond/nala',
        '6': 'Other',
    })
    households['Seeds Homegrown'] = households['Seeds Homegrown'].map({
        '1': 'Home grown',
        '2': 'Purchased',
        '3': 'Both',
    })
    irrigation_map = {
        0: 'Yes',
        1: 'No',
    }
    households['Kharif: Crop: Irrigation'] = households['Kharif: Crop: Irrigation'].map(irrigation_map)
    households['Rabi: Crop: Irrigation'] = households['Rabi: Crop: Irrigation'].map(irrigation_map)
    households['Summer: Crop: Irrigation'] = households['Summer: Crop: Irrigation'].map(irrigation_map)
    crop_map = {
        1: "Rice/ Paddy",
        2: "Jowar",
        3: "Bajra",
        4: "Maize",
        5: "Ragi",
        6: "Wheat",
        7: "Barley",
        8: "Other cereals",
        9: "Gram",
        10: "Tur (arhar)",
        11: "Urad",
        12: "Moong",
        13: "Kulthi",
        14: "Masur",
        15: "Lask/ Khesari",
        16: "Moth",
        17: "Other pulses",
        18: "Sugarcane",
        19: "Groundnut",
        20: "Castorseed",
        21: "Sesamum",
        22: "Mustard seed",
        23: "Linseed",
        24: "Safflower",
        25: "Nigerseed",
        26: "Sunflower",
        27: "Soyabean",
        28: "Other oilseeds",
        29: "Cottonseed",
        30: "Coconut",
        31: "Black pepper",
        32: "Chilis",
        33: "Ginger",
        34: "Turmeric",
        35: "Coriander",
        36: "Cardamom",
        37: "Garlic",
        38: "Arecanuts",
        39: "Other spice",
        40: "Guarseed",
        41: "Bananas",
        42: "Mangoes",
        43: "Citrus fruit",
        44: "Grapes",
        45: "Apples, etc.",
        46: "Papaya",
        47: "Other fruits",
        48: "Cashews",
        49: "Other nuts",
        50: "Potatoes",
        51: "Sweet potato",
        52: "Tapioca",
        53: "Onion",
        54: "Other veg",
        55: "Other food",
        56: "Cotton",
        57: "Jute",
        61: "Tobacco",
        63: "Other dyes",
        67: "Tea",
        68: "Coffee",
        69: "Rubber",
        70: "Other pl.",
        71: "Oats",
        72: "Fodder",
        73: "Green manure",
        74: "Other nonfood",
        75: "-",
    }
    households['Kharif: Crop: Name'] = households['Kharif: Crop: Name'].map(crop_map)
    households['Rabi: Crop: Name'] = households['Rabi: Crop: Name'].map(crop_map)
    households['Summer: Crop: Name'] = households['Summer: Crop: Name'].map(crop_map)

    return households

def get_size_class(households):
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
    for idx, household in households.iterrows():
        area = household['area owned & cultivated']
        for size_class_name, size in size_classes:
            if area < size:
                households.loc[idx, 'size class'] = size_class_name
                break
    return households

if __name__ == '__main__':
    households = prefilter()
    crops = read_crops()
    households = process(households, crops)
    households = rename_parameters(households)
    households = get_size_class(households)
    households.to_csv(os.path.join(INPUT, 'agents', 'IHDS_I.csv'), index=False)