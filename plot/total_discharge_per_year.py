import os
import re
import pandas as pd
import numpy as np
import json
from datetime import timedelta, date, datetime
import matplotlib.pyplot as plt

from plotconfig import config
TIMEDELTA = timedelta(days=1)

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

def read_npy(scenario, name, dt):
    dt -= timedelta(days=1)
    fn = os.path.join(config['general']['report_folder'], scenario, name, dt.isoformat().replace(':', '').replace('-', '') + '.npy')
    return np.load(fn)

def read_arrays(scenario, year, name, mode='full_year'):
    if mode == 'full_year':
        day = date(year, 1, 1)
        # loop through all days in year
        total = None
        n = 0
        while day < date(year + 1, 1, 1):
            # read the data
            array = read_npy(scenario, name, day)
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
        total = read_npy(scenario, name, day)
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

if __name__ == '__main__':
    fig, axes = plt.subplots(1, 4, figsize=(20, 8), sharex=True)
    (ax0, ax1, ax2, ax3) = axes
    # use tight layout
    fig.tight_layout()
    farmer_states, state_index = get_farmer_states()
    linestyles = ['-', '--']
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    # scenarios = ['base', 'sprinkler']
    scenarios = ['base']
    
    for i, scenario in enumerate(scenarios):
        discharge = get_discharge(scenario)
        # sum discharge per year
        discharge_per_year = discharge.resample('A', label='right').sum()
        discharge_per_year.index = [discharge_per_year.index[i].year for i in range(len(discharge_per_year))]
        # remove first year from index
        discharge_per_year = discharge_per_year[1:]
        to_plot = pd.DataFrame(discharge_per_year, columns=['discharge'])
        
        for year in discharge_per_year.index:
            crops = read_arrays(scenario, year, 'crops_kharif', mode='first_day_of_year')
            farmers_sugarcane = (crops == 4)
            for state_idx, state_name in enumerate(state_index):
                # set n_farmers_sugarcane in to_plot
                to_plot.loc[year, f'n_farmers_sugarcane_{state_name}'] = farmers_sugarcane[farmer_states == state_idx].sum()
        
        for year in discharge_per_year.index:
            has_well = read_arrays(scenario, year, 'well_irrigated', mode='first_day_of_year')
            for state_idx, state_name in enumerate(state_index):
                # set n_farmers_sugarcane in to_plot
                to_plot.loc[year, f'n_farmers_irrigation_{state_name}'] = has_well[farmer_states == state_idx].sum()

        for year in discharge_per_year.index:
            groundwater_depth = read_arrays(scenario, year, 'groundwater depth', mode='first_day_of_year')
            for state_idx, state_name in enumerate(state_index):
                # set n_farmers_sugarcane in to_plot
                to_plot.loc[year, f'groundwater_depth_{state_name}'] = groundwater_depth[farmer_states == state_idx].mean()

        hydraulic_head = get_honeybees_data(scenario, 'hydraulic head', to_plot.index[0], to_plot.index[-1], fileformat='csv')
        to_plot['hydraulic_head'] = hydraulic_head[0]
        
        # plotting
        # set linestyle
        linestyle = linestyles[i]

        # discharge
        ax0.plot(to_plot.index, to_plot['discharge'], label=scenario, linestyle=linestyle, color='#1f77b4')
        ax0.set_title('Discharge')
        ax0.set_xlim(to_plot.index[0], to_plot.index[-1])
        
        # sugarcane farmers
        for j, state in enumerate(state_index):
            color = colors[j]
            ax1.plot(to_plot.index, to_plot[f'n_farmers_sugarcane_{state}'], label=f'{scenario} {state.title()}', color=color, linestyle=linestyle)
        ax1.set_title('Sugarcane farmers')
        
        # well-irrigated farmers
        for j, state in enumerate(state_index):
            color = colors[j]
            ax2.plot(to_plot.index, to_plot[f'n_farmers_irrigation_{state}'], label=f'{scenario} {state.title()}', color=color, linestyle=linestyle)
        ax2.set_title('Well irrigating farmers')

        # groundwater depth
        for j, state in enumerate(state_index):
            color = colors[j]
            ax3.plot(to_plot.index, to_plot[f'groundwater_depth_{state}'], label=f'{scenario} {state.title()}', color=color, linestyle=linestyle)
        # invert y axis
        ax3.invert_yaxis()
        ax3.set_title('Groundwater depth farmers')

        # # hydraulic head
        # ax3.plot(to_plot.index, to_plot['hydraulic_head'], label=scenario, linestyle=linestyle, color='#1f77b4')
        # ax3.set_title('Hydraulic head')
    
    for ax in axes:
        ax.legend()
    
    plt.savefig('plot/output/total_discharge_per_year.png')
    plt.savefig('plot/output/total_discharge_per_year.svg')
