import calendar
import os
import re
import pandas as pd
import numpy as np
import json
from datetime import timedelta, date, datetime
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio

plt.rcParams["font.family"] = "monospace"

from plotconfig import config
TIMEDELTA = timedelta(days=1)

FARMER_MULTIPLIER = 15657
MONEY_MULTIPLIER = 1_000_000_000
WATER_MULTIPLIER = 1_000_000

with rasterio.open(os.path.join(config['general']['input_folder'], 'areamaps', 'sub_cell_area.tif'), 'r') as src_cell_area:
    cell_area = src_cell_area.read(1)
with rasterio.open(os.path.join(config['general']['input_folder'], 'agents', 'farms.tif'), 'r') as src:
    farms = src.read(1)


## Field sizes are loaded in, however, the field size reports are not used 
is_field = np.where(farms != -1)
field_size = np.bincount(farms[is_field], weights=cell_area[is_field])

# field_size_test = np.load(os.path.join(config['general']['report_folder'], 'irrigation_source', '20110101.npz'))

def sum_all(x):
    return np.sum(x) / FARMER_MULTIPLIER

def sum_irrigation(x):
    filtered_values = [val for val in x if val == 2 or val == 3]
    return np.sum(filtered_values) / FARMER_MULTIPLIER

def get_farmer_states():
    tehsil = np.load(os.path.join(config['general']['input_folder'], 'agents', 'attributes', 'tehsil_code.npy'))

    with open(os.path.join(config['general']['input_folder'], 'areamaps', 'subdistrict2state.json'), 'r') as f:
        subdistrict2state = json.load(f)
        subdistrict2state = {int(subdistrict): state for subdistrict, state in subdistrict2state.items()}
        # assert that all subdistricts keys are integers
        assert all([isinstance(subdistrict, int) for subdistrict in subdistrict2state.keys()])
        # make sure all keys are consecutive integers starting at 0
        assert min(subdistrict2state.keys()) == 0
        assert max(subdistrict2state.keys()) == len(subdistrict2state) - 1
        # load unique states
        state_index = list(set(subdistrict2state.values()))
        # create numpy array mapping subdistricts to states
        state2int = {state: i for i, state in enumerate(state_index)}
        subdistrict2state_arr = np.zeros(len(subdistrict2state), dtype=np.int32)
        for subdistrict, state in subdistrict2state.items():
            subdistrict2state_arr[subdistrict] = state2int[state]

    return subdistrict2state_arr[tehsil], state_index

def read_npz(scenario, name, dt):
    fn = os.path.join(config['general']['report_folder'], scenario, name, dt.isoformat().replace(':', '').replace('-', '') + '.npz')
    array = np.load(fn)['data']
    return array[~np.isnan(array)]

def read_arrays(scenario, year, name, mode='full_year'):
    if mode == 'full_year':
        day = date(year, 1, 1)
        # loop through all days in year
        total = None
        n = 0
        while day < date(year + 1, 1, 1):
            # read the data
            array = read_npz(scenario, name, day)
            if total is None:
                total = array
            else:
                total += array
            # go to next day
            day += timedelta(days=1)
            n += 1
        total /= n
    elif mode == 'first_day_of_year':
        day = date(year, 1, 1)
        total = read_npz(scenario, name, day)
    else:
        raise ValueError(f'Unknown mode {mode}')
    return total


def get_discharge(scenario):
    values = []
    dates = []
    with open(os.path.join(config['general']['report_folder'], scenario, 'var.discharge_daily.tss'), 'r') as f:
        for i, line in enumerate(f):
            if i < 4:
                continue
            if len(dates) == 0:
                dt = config['general']['start_time']
            else:
                dt = dates[-1] + TIMEDELTA
            dates.append(dt)
            match = re.match(r"\s+([0-9]+)\s+([0-9\.e-]+)\s*$", line)
            value = float(match.group(2))
            values.append(value)
    assert len(dates) == len(values)
    discharge = pd.Series(values, index=dates)
    # convert index to datetime
    discharge.index = pd.to_datetime(discharge.index)
    return discharge

def get_honeybees_data(scenario, varname, start_year, end_year, fileformat='csv'):
    df = pd.read_csv(os.path.join(config['general']['report_folder'], scenario, varname + '.' + fileformat), index_col=0)
    dates = df.index.tolist()
    dates = [datetime.strptime(dt, "%Y-%m-%d") for dt in dates]
    df.index = dates
    # filter on start and end year
    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
    # get mean dataframe by year
    df = df.groupby(df.index.year).mean()
    # return hydraulic head as numpy array and years as list
    return np.array(df[varname].tolist()), df.index.tolist()

def get_values_small_vs_large(scenario, year, fn, name, correct_for_field_size=False, mode='full_year'):
    total = read_arrays(scenario, year, name, mode=mode)
    if correct_for_field_size:
        total /= field_size
    # get median field size
    median_field_size = np.percentile(field_size, 50)
    small_fields = field_size < median_field_size
    by_field_size = {
        'small': fn(total[small_fields]),
        'large': fn(total[~small_fields]),
    }
    return by_field_size

if __name__ == '__main__':
    # set x and y axis fontsize for all plots
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
    ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = axes
    # use tight layout
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    farmer_states, state_index = get_farmer_states()
    linestyles = ['-', '--', '-.']
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    scenarios = ['base', 'sprinkler', 'noadaptation']
    # scenarios = ['base']
    # scenarios = ['sprinkler']

    fn = os.path.join(config['general']['input_folder'], 'routing', 'lakesreservoirs', 'basin_lakes_data.xlsx')
    df = pd.read_excel(fn).set_index('Hylak_id')
    reservoirs = df[df['Lake_type'] == 2].copy()
    active_reservoirs = np.load(os.path.join(config['general']['report_folder'], 'base', 'active_reservoirs_waterBodyIDs.npy'))
    reservoirs = reservoirs[reservoirs.index.isin(active_reservoirs)]
    # create geodataframe from reservoirs
    reservoirs = gpd.GeoDataFrame(reservoirs, geometry=gpd.points_from_xy(reservoirs['Pour_long'], reservoirs['Pour_lat']), crs='epsg:4326')
    # load states of India
    states = gpd.read_file(os.path.join(config['general']['original_data'], 'census', 'tehsils.geojson')).to_crs('epsg:4326')
    # find state of each reservoir
    reservoirs = gpd.sjoin(reservoirs, states[['15_state', 'geometry']], how='left', predicate="within")

    # export to disk
    # reservoirs.to_file(os.path.join(config['general']['report_folder'], 'reservoirs.geojson'), driver='GeoJSON')

    inflation = np.array([
        0,
        8.34926704907581,
        10.8823529411765,
        11.9893899204243,
        8.9117933648337,
        9.47899691419793,
        10.0178784746103,
        6.665656718679,
        4.90697344127254,
        4.94821634062142,
        3.32817337461301,
        3.93882646691634,
    ])
    cum_inflation = np.cumprod(1 + inflation / 100)
    
    for i, scenario in enumerate(scenarios):
        discharge = get_discharge(scenario)
        # sum discharge per year
        discharge_per_year = discharge.resample('A', label='right').mean()
        discharge_per_year.index = [discharge_per_year.index[i].year for i in range(len(discharge_per_year))]
        # remove first year from index
        discharge_per_year = discharge_per_year[1:]
        to_plot = pd.DataFrame(discharge_per_year, columns=['discharge'])
        

        ## change the irrigation_source so that it only counts the well_irrigated farms 
        for year in discharge_per_year.index:
            small_vs_large = get_values_small_vs_large(scenario, year, sum_irrigation, 'irrigation_source', correct_for_field_size=False, mode='first_day_of_year')
            to_plot.loc[year, 'well_irrigated_small'] = small_vs_large['small']
            to_plot.loc[year, 'well_irrigated_large'] = small_vs_large['large']

        # reservoirs
        reservoir_storage = pd.read_csv(os.path.join(config['general']['report_folder'], scenario, 'reservoir storage.csv'), index_col=0)
        # convert index to datetime
        reservoir_storage.index = pd.to_datetime(reservoir_storage.index)
        for year in discharge_per_year.index:
            reservoir_storage_year = reservoir_storage[reservoir_storage.index.year == year].mean()
            to_plot.loc[year, 'reservoir_storage'] = reservoir_storage[reservoir_storage.index.year == year].mean().item()

        for year in discharge_per_year.index:
            crops = read_arrays(scenario, year, 'crops_kharif', mode='first_day_of_year')
            farmers_sugarcane = (crops == 4)
            for state_idx, state_name in enumerate(state_index):
                # set n_farmers_sugarcane in to_plot
                to_plot.loc[year, f'n_farmers_sugarcane_{state_name}'] = farmers_sugarcane[farmer_states == state_idx].sum()
        
        for year in discharge_per_year.index:
            has_well = sum_irrigation(read_arrays(scenario, year, 'irrigation_source', mode='first_day_of_year'))

            for state_idx, state_name in enumerate(state_index):
                # set
                to_plot.loc[year, f'n_farmers_irrigation_{state_name}'] = has_well

        for year in discharge_per_year.index:
            # get days in year
            days_in_year = np.array([calendar.monthrange(year, i)[1] for i in range(1, 13)]).sum()
            profit = read_arrays(scenario, year, 'profit', mode='full_year') * days_in_year / MONEY_MULTIPLIER
            for state_idx, state_name in enumerate(state_index):
                # set n_farmers_sugarcane in to_plot
                to_plot.loc[year, f'profit_{state_name}'] = profit[farmer_states == state_idx].sum()

        for year in discharge_per_year.index:
            small_vs_large = get_values_small_vs_large(scenario, year, sum_all, 'profit', correct_for_field_size=False, mode='full_year')
            to_plot.loc[year, 'profit_small'] = small_vs_large['small']
            to_plot.loc[year, 'profit_large'] = small_vs_large['large']

        for year in discharge_per_year.index:
            channel_irrigation = read_arrays(scenario, year, 'channel irrigation', mode='full_year')
            reservoir_irrigation = read_arrays(scenario, year, 'reservoir irrigation', mode='full_year')
            groundwater_irrigation = read_arrays(scenario, year, 'groundwater irrigation', mode='full_year')
            total_irrigation = channel_irrigation + reservoir_irrigation + groundwater_irrigation
            for state_idx, state_name in enumerate(state_index):
                to_plot.loc[year, f'groundwater_irrigation_{state_name}'] = groundwater_irrigation[farmer_states == state_idx].sum()
                to_plot.loc[year, f'total_irrigation_{state_name}'] = total_irrigation[farmer_states == state_idx].sum()

        for year in discharge_per_year.index:
            groundwater_depth = read_arrays(scenario, year, 'groundwater depth', mode='first_day_of_year')
            for state_idx, state_name in enumerate(state_index):
                # set n_farmers_sugarcane in to_plot
                to_plot.loc[year, f'groundwater_depth_{state_name}'] = groundwater_depth[farmer_states == state_idx].mean()

        # plotting
        # set linestyle
        linestyle = linestyles[i]

        if scenario == 'base':
            scenario = 'baseline'
        if scenario == 'sprinkler':
            scenario = 'drip'
        if scenario == 'noadaptation':
            scenario = 'No Adaptation'

        title_fontsize = 9

        # discharge
        ax0.plot(to_plot.index, to_plot['discharge'], label=scenario, linestyle=linestyle, color='#1f77b4')
        ax0.set_title('A - Average yearly discharge (m/s)', fontsize=title_fontsize)
        ax0.set_xlim(to_plot.index[0], to_plot.index[-1])

        # sugarcane farmers
        for j, state in enumerate(state_index):
            color = colors[j]
            ax1.plot(to_plot.index, to_plot[f'n_farmers_sugarcane_{state}'] / FARMER_MULTIPLIER, label=f'{scenario} {state.title()}', color=color, linestyle=linestyle)
        ax1.set_title(f'B - Sugarcane farmers (×{FARMER_MULTIPLIER})', fontsize=title_fontsize)
        ax1.set_ylim(0, 3)

        # well-irrigated farmers
        for j, state in enumerate(state_index):
            color = colors[j]
            ax2.plot(to_plot.index, to_plot[f'n_farmers_irrigation_{state}'], label=f'{scenario} {state.title()}', color=color, linestyle=linestyle)
        ax2.set_title(f'C - Well irrigating farmers (×{FARMER_MULTIPLIER})', fontsize=title_fontsize)
        ax2.set_ylim(0, 3)

        # groundwater irrigation
        for j, state in enumerate(state_index):
            color = colors[j]
            ax3.plot(to_plot.index, to_plot[f'groundwater_irrigation_{state}'] / WATER_MULTIPLIER, label=f'{scenario} {state.title()}', color=color, linestyle=linestyle)
        ax3.set_title(f'D - Groundwater consumption farmers ({WATER_MULTIPLIER}' + r'$m^3$/day)', fontsize=title_fontsize)

        # total irrigation
        for j, state in enumerate(state_index):
            color = colors[j]
            ax4.plot(to_plot.index, to_plot[f'total_irrigation_{state}'] / WATER_MULTIPLIER, label=f'{scenario} {state.title()}', color=color, linestyle=linestyle)
        ax4.set_title(f'E - Irrigation consumption farmers ({WATER_MULTIPLIER}' + r'$m^3$/day)', fontsize=title_fontsize)

        # groundwater depth
        for j, state in enumerate(state_index):
            color = colors[j]
            ax5.plot(to_plot.index, to_plot[f'groundwater_depth_{state}'], label=f'{scenario} {state.title()}', color=color, linestyle=linestyle)
        ax5.set_title('F - Groundwater depth (m)', fontsize=title_fontsize)

        # reservoirs
        ax6.plot(to_plot.index, to_plot['reservoir_storage'] / 1_000_000_000, label=scenario, linestyle=linestyle, color='#1f77b4')
        ax6.set_title('G - Mean reservoir storage (billion m$3$)', fontsize=title_fontsize)

        # profit per state
        for j, state in enumerate(state_index):
            color = colors[j]
            ax7.plot(to_plot.index, to_plot[f'profit_{state}'] / cum_inflation[0:4,], label=f'{scenario} {state.title()}', color=color, linestyle=linestyle)
        ax7.set_title('H - Profit farmers (billion 2007 rupees)', fontsize=title_fontsize)
        
        # small vs large farmers
        for j, size in enumerate(['small', 'large']):
            color = colors[j + len(state_index)]  # use different colors than for states
            ax8.plot(to_plot.index, to_plot[f'well_irrigated_{size}'], label=f'{scenario} {size}', color=color, linestyle=linestyle)
        ax8.set_title(f'I - Well irrigating farmers (×{FARMER_MULTIPLIER})', fontsize=title_fontsize)
        # ax1.set_ylim(0, 3)
        
        # profit small vs large
        for j, size in enumerate(['small', 'large']):
            color = colors[j + len(state_index)]
            ax9.plot(to_plot.index, to_plot[f'profit_{size}'] / cum_inflation[0:4,], label=f'{scenario} {size}', color=color, linestyle=linestyle)
        ax9.set_title('J - Profit farmers (billion 2007 rupees)', fontsize=title_fontsize)

    to_plot.to_csv('plot/sanctuary/total_discharge_per_year.csv')

    # invert y axis
    ax5.invert_yaxis()
    for horizontal_axes in axes:
        for ax in horizontal_axes:
            ax.legend()
    
    plt.savefig('plot/sanctuary/total_discharge_per_year.png')
    plt.savefig('plot/sanctuary/total_discharge_per_year.svg')
