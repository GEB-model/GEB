import os
import re
import pandas as pd
import numpy as np
import json
from datetime import timedelta, date
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


if __name__ == '__main__':
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 20))
    farmer_states, state_index = get_farmer_states()
    scenarios = ['base', 'sprinkler']
    for i, scenario in enumerate(scenarios):
        discharge = get_discharge(scenario)
        # sum discharge per year
        discharge_per_year = discharge.resample('A', label='right').sum()
        discharge_per_year.index = [discharge_per_year.index[i].year for i in range(len(discharge_per_year))]
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
                to_plot.loc[year, f'n_farmers_irrigation_{state_name}'] = farmers_sugarcane[farmer_states == state_idx].sum()
        # plot
        ax0.bar(to_plot.index, to_plot['discharge'], label=scenario)
        bottom = np.zeros(len(to_plot.index))
        for state in state_index:
            ax1.plot(to_plot.index, to_plot[f'n_farmers_sugarcane_{state}'], label=f'{scenario} {state.title()}')#, bottom=bottom)
            bottom += to_plot[f'n_farmers_sugarcane_{state}']
        bottom = np.zeros(len(to_plot.index))
        for state in state_index:
            ax2.plot(to_plot.index, to_plot[f'n_farmers_irrigation_{state}'], label=f'{scenario} {state.title()}')#, bottom=bottom)
            bottom += to_plot[f'n_farmers_irrigation_{state}']
    ax0.legend()
    ax1.legend()
    ax2.legend()
    plt.savefig('plot/output/total_discharge_per_year.png')
    plt.savefig('plot/output/total_discharge_per_year.svg')
