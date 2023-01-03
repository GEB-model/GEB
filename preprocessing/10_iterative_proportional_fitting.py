import os
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from itertools import product, chain

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from tqdm import tqdm

from methods import create_cell_area_map

from preconfig import INPUT

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', '-n', type=int, default=1)
args = parser.parse_known_args()[0]

class IPL(object):
    def __init__(self, original, aggregates, weight_col='weight', n=None, learning_rate=1,
                 convergence_rate=1e-5, max_iteration=500, verbose=0, rate_tolerance=1e-8):
        """
        Initialize the ipfn class

        original: numpy darray matrix or dataframe to perform the ipfn on.

        aggregates: list of numpy array or darray or pandas dataframe/series. The aggregates are the same as the marginals.
        They are the target values that we want along one or several axis when aggregating along one or several axes.

        dimensions: list of lists with integers if working with numpy objects, or column names if working with pandas objects.
        Preserved dimensions along which we sum to get the corresponding aggregates.

        convergence_rate: if there are many aggregates/marginal, it could be useful to loosen the convergence criterion.

        max_iteration: Integer. Maximum number of iterations allowed.

        verbose: integer 0, 1 or 2. Each case number includes the outputs of the previous case numbers.
        0: Updated matrix returned.
        1: Flag with the output status (0 for failure and 1 for success).
        2: dataframe with iteration numbers and convergence rate information at all steps.

        rate_tolerance: float value. If above 0.0, like 0.001, the algorithm will stop once the difference between the conv_rate variable of 2 consecutive iterations is below that specified value

        For examples, please open the ipfn script or look for help on functions ipfn_np and ipfn_df
        """
        self.original = original
        self.aggregates = aggregates
        self.weight_col = weight_col
        self.n = n or sum(aggregates[0])
        self.learning_rate = learning_rate
        self.conv_rate = convergence_rate
        self.max_itr = max_iteration
        if verbose not in [0, 1, 2]:
            raise(ValueError(f"wrong verbose input, must be either 0, 1 or 2 but got {verbose}"))
        self.verbose = verbose
        self.rate_tolerance = rate_tolerance

    @staticmethod
    def index_axis_elem(dims, axes, elems):
        inc_axis = 0
        idx = ()
        for dim in range(dims):
            if (inc_axis < len(axes)):
                if (dim == axes[inc_axis]):
                    idx += (elems[inc_axis],)
                    inc_axis += 1
                else:
                    idx += (np.s_[:],)
        return idx

    def ipfn_df(self, df, aggregates, weight_col='weight'):
        steps = len(aggregates)
        tables = [df]
        for inc in range(steps - 1):
            tables.append(df.copy())

        # Calculate the new weights for each dimension
        inc = 0
        for aggregate in aggregates:
            feature = aggregate.name
            if inc == (steps - 1):
                table_update = df
                table_current = tables[inc].copy()
            else:
                table_update = tables[inc + 1]
                table_current = tables[inc]

            count_feature = isinstance(table_current[feature].iloc[0], tuple)

            xijk = aggregates[inc]

            feat_l = []
            if count_feature:
                table_sel = table_current[feature]
                items = [item for item in list(chain(*table_sel)) if item is not None]
                unique_items = np.unique(items)
                feat_l.append(unique_items)
                table_update.set_index(feature, inplace=True)
                table_current.set_index(feature, inplace=True)

                tmp = pd.Series(
                    0,
                    index=unique_items,
                    dtype=np.float64
                )
                for idx, row in table_update.iterrows():
                    for item in idx:
                        if item:
                            tmp.loc[item] += row[weight_col]
                
            else:
                feat_l.append(np.unique(table_current[feature]))

                table_update.set_index(feature, inplace=True)
                table_current.set_index(feature, inplace=True)

                tmp = table_current.groupby(feature)[weight_col].sum()

            if count_feature:
                update_table = pd.DataFrame(1, index=table_update.index, columns=list(product(*feat_l)))
                for characteristic in product(*feat_l):
                    den = tmp.loc[characteristic]
                    if den == 0:
                        den = 1

                    mask = np.array([characteristic[0] in idx for idx in table_update.index])
                    update_table.loc[mask, (characteristic, )] = (table_current[weight_col].astype(float) * xijk.loc[characteristic] / den)[mask == True] / table_current[weight_col].astype(float)[mask == True]
                    
                old_value = table_current[weight_col]
                new_value = update_table.mean(axis=1) * table_current[weight_col].astype(float)
                table_update[weight_col] = old_value + (new_value - old_value) * self.learning_rate
            
            else:
                for characteristic in product(*feat_l):
                    den = tmp.loc[characteristic]
                    if den == 0:
                        den = 1
                    # calculate new weight for this iteration
                    mask = table_update.index == characteristic[0]

                    table_update.loc[mask, weight_col] = \
                    old_value = table_current.loc[mask, weight_col].astype(float)
                    new_value = old_value * xijk.loc[characteristic] / den
                    table_current.loc[mask, weight_col] = old_value + (new_value - old_value) * self.learning_rate

            table_update.reset_index(inplace=True)
            table_current.reset_index(inplace=True)
            inc += 1
            feat_l = []

        table_update[weight_col] = table_update[weight_col] / table_update[weight_col].sum() * self.n

        # Calculate the max convergence rate
        max_conv = 0
        inc = 0
        for aggregate in aggregates:
            feature = aggregate.name
            count_feature = isinstance(table_current[feature].iloc[0], tuple)
            if count_feature:
                tmp = pd.Series(
                    0,
                    index=unique_items,
                    dtype=np.float64
                )
                for idx, row in table_update.set_index(feature).iterrows():
                    for item in idx:
                        if item:
                            tmp.loc[item] += row[weight_col]
                
            else:
                tmp = table_update.groupby(feature)[weight_col].sum()
            ori_ijk = aggregates[inc]
            temp_conv = max(abs(tmp / ori_ijk - 1))
            if temp_conv > max_conv:
                max_conv = temp_conv
            inc += 1
        
        return table_update, max_conv

    def iteration(self):
        """
        Runs the ipfn algorithm. Automatically detects of working with numpy ndarray or pandas dataframes.
        """

        i = 0
        conv = np.inf
        old_conv = -np.inf
        conv_list = []
        m = self.original

        # If the original data input is in pandas DataFrame format
        while ((i <= self.max_itr and conv > self.conv_rate) and (i <= self.max_itr and abs(conv - old_conv) > self.rate_tolerance)):
            old_conv = conv
            m, conv = self.ipfn_df(m, self.aggregates, self.weight_col)
            conv_list.append(conv)
            i += 1
        converged = 1
        if i <= self.max_itr:
            if (not conv > self.conv_rate) & (self.verbose > 1):
                print('ipfn converged: convergence_rate below threshold')
            elif not abs(conv - old_conv) > self.rate_tolerance:
                print('ipfn converged: convergence_rate not updating or below rate_tolerance')
        else:
            print('Maximum iterations reached')
            converged = 0

        # Handle the verbose
        if self.verbose == 0:
            return m
        elif self.verbose == 1:
            return m, converged
        elif self.verbose == 2:
            return m, converged, pd.DataFrame({'iteration': range(i), 'conv': conv_list}).set_index('iteration')
        else:
            raise(ValueError(f'wrong verbose input, must be either 0, 1 or 2 but got {self.verbose}'))

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
SIZE_GROUP = {
    'Below 0.5': ['Below 0.5', '0.5-1.0'],
    '0.5-1.0': ['Below 0.5', '0.5-1.0'],
    '1.0-2.0': ['1.0-2.0'],
    '2.0-3.0': ['2.0-3.0', '3.0-4.0'],
    '3.0-4.0': ['2.0-3.0', '3.0-4.0'],
    '4.0-5.0': ['4.0-5.0', '5.0-7.5', '7.5-10.0'],
    '5.0-7.5': ['4.0-5.0', '5.0-7.5', '7.5-10.0'],
    '7.5-10.0': ['4.0-5.0', '5.0-7.5', '7.5-10.0'],
    '10.0-20.0': ['10.0-20.0', '20.0 & ABOVE'],
    '20.0 & ABOVE': ['10.0-20.0', '20.0 & ABOVE'],
}

with rasterio.open(os.path.join(INPUT, 'areamaps', 'tehsils.tif'), 'r') as src:
    tehsils_tif = src.read(1)
    cell_area = create_cell_area_map(src.profile, write_to_disk=False)

tehsils_shape = gpd.read_file(os.path.join(INPUT, 'areamaps', 'subdistricts.shp')).set_index(['state_name', 'district_n', 'sub_dist_1'])
avg_farm_size = pd.read_excel(os.path.join(INPUT, 'census', 'avg_farm_size.xlsx'), index_col=(0, 1, 2))
crop_data = pd.read_excel(os.path.join(INPUT, 'census', 'crop_data.xlsx'), index_col=(0, 1, 2, 3))
print("Getting tehsil areas")
for (state, district, tehsil), tehsil_crop_data in tqdm(crop_data.groupby(level=[0, 1, 2])):
    # tehsil_farm_size = avg_farm_size.loc[(state, district, tehsil)]
    farms_per_size_class = tehsil_crop_data.droplevel([0, 1, 2]).sum(axis=1)

    # assert (tehsil_farm_size.index == farms_per_size_class.index).all()

    # area_per_size_class = tehsil_farm_size * farms_per_size_class
    # census_farm_area = area_per_size_class.sum()

    tehsil_ID = tehsils_shape.loc[(state, district, tehsil), 'ID']
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

folder = os.path.join(INPUT, 'agents', 'ipl')
os.makedirs(folder, exist_ok=True)

def fit(ipl_group):
    (state, district, tehsil, size_class), crop_frequencies = ipl_group
    if tehsil == '0':
        return
    fp = os.path.join(folder, f"{state}_{district}_{tehsil}_{size_class}.csv")
    # if os.path.exists(fp):
    #     return
    # print(state, district, tehsil, size_class)
    survey_data_size_class = survey_data[survey_data['size_class'].isin(SIZE_GROUP[size_class])]
    survey_data_size_class = survey_data_size_class.reset_index(drop=True)

    crops = crop_frequencies.iloc[0]
    crops.name = 'crops'
    aggregates = [crops]

    n = n_farms.loc[(state, district, tehsil), size_class]
    if np.isnan(n):
        n = 0
    else:
        n = int(n)
    ipl = IPL(
        original=survey_data_size_class,
        aggregates=aggregates,
        n=n,
        learning_rate=.1
    ).iteration()

    ipl.to_csv(fp, index=False)

ipl_groups = crop_data.groupby(crop_data.index)

if args.n_jobs == 1:
    for ipl_group in tqdm(ipl_groups):
        fit(ipl_group)
else:
    from tqdm.contrib.concurrent import process_map
    process_map(fit, ipl_groups, max_workers=args.n_jobs)

# create a list of all the files in the folder
files = os.listdir(folder)
#iterate through the list of files
for file in files:
    # open files in pandas
    df = pd.read_csv(os.path.join(folder, file))
    