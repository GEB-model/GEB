# -*- coding: utf-8 -*-
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from numpy.core.fromnumeric import size
import yaml
import pandas as pd
import sys
import numpy as np
import matplotlib.patches as mpatches

sys.path.insert(1, os.path.join(sys.path[0], '..'))

TIMEDELTA = timedelta(days=1)
OUTPUT_FOLDER = os.path.join('DataDrive', 'GEB', 'report')
with open('GEB.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def KGE(s, o):
    """
    Kling Gupta Efficiency (Kling et al., 2012, http://dx.doi.org/10.1016/j.jhydrol.2012.01.011)
    input:
        s: simulated
        o: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    B = np.mean(s) / np.mean(o)
    y = (np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o))
    r = np.corrcoef(o, s)[0,1]

    KGE = 1 - np.sqrt((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2)

    return KGE

def correlation(s,o):
    """
    correlation coefficient
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """

    if s.size == 0:
        corr = np.NaN
    else:
        corr = np.corrcoef(o, s)[0,1]
        
    return corr


def get_discharge(scenario):
    values = []
    dates = []
    with open(os.path.join(OUTPUT_FOLDER, scenario, 'var.discharge_daily.tss'), 'r') as f:
        for i, line in enumerate(f):
            if i < 4:
                continue
            if len(dates) == 0:
                dates.append(config['general']['start_time'])
            else:
                dates.append(dates[-1] + TIMEDELTA)
            match = re.match(r"\s+([0-9]+)\s+([0-9\.e-]+)\s*$", line)
            value = float(match.group(2))
            values.append(value)
    return dates, np.array(values)

def get_hyve_data(varname, scenario, fileformat='csv'):
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, scenario, varname + '.' + fileformat), index_col=0)
    dates = df.index.tolist()
    dates = [datetime.strptime(dt, "%Y-%m-%d") for dt in dates]
    return dates, np.array(df[varname].tolist())

def add_patches_legend(ax, labels, colors, ncol, legend_fontsize=8, legend_title=None):
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors)) if labels[i] is not None]
    legend = ax.legend(
        handles=patches,
        loc='upper left',
        bbox_to_anchor=(0, -0.13),
        borderaxespad=0,
        ncol=ncol,
        columnspacing=1,
        fontsize=legend_fontsize,
        frameon=False,
        handlelength=1,
        borderpad=0
    )
    if legend_title:
        legend.set_title(legend_title)

def get_observed_discharge(dates):
    df = pd.read_csv('DataDrive/GEB/calibration/observations.csv', parse_dates=['Dates'])
    return df[df['Dates'].isin(dates)]['flow'].to_numpy()

def main():
    title_formatter = {'size': 'small', 'fontweight': 'bold', 'pad': 3}
    scenarios = ('base', 'self_investment', 'government_subsidies', 'ngo_training')
    scenarios = ('base', 'government_subsidies', 'ngo_training')
    colors = ['black', 'blue', 'orange', 'red']
    colors = colors[:len(scenarios) + 1]
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(8, 3), dpi=300)
    plt.subplots_adjust(left=0.055, right=0.99, bottom=0.17, top=0.92, wspace=0.2)
    ax0, ax1, ax2 = axes

    add_patches_legend(
        ax0,
        labels=['observed'] + [s.replace('_', ' ') for s in scenarios],
        colors=colors,
        ncol=len(scenarios) + 1
    )

    discharges = []
    for i, scenario in enumerate(scenarios):
        dates, discharge = get_discharge(scenario)
        ax0.plot(dates, discharge, label=scenario, color=colors[i+1])
        discharges.append(discharge)
    
    observed_discharge = get_observed_discharge(dates)
    ax0.plot(dates, observed_discharge, color=colors[0], linestyle='dashed')

    for scenario, discharge in zip(scenarios, discharges):
        print('scenario')
        print('\tKGE:', KGE(discharge, observed_discharge))
        print('\tcorrelation:', correlation(discharge, observed_discharge))
    
    ax0.set_ylim(0, ax0.get_ylim()[1])

    ax0.text(0.1, 0.9, f'KGE: {round(KGE(discharge, observed_discharge), 3)}\ncorr: {round(correlation(discharge, observed_discharge), 3)}',
     horizontalalignment='left',
     verticalalignment='top',
     transform = ax0.transAxes)
     

    ax0.set_title('discharge $(m^3s^{-1})$', **title_formatter)
    # ax0.set_ylabel('$m^3/s$', **label_formatter)
    for i, scenario in enumerate(scenarios):
        dates, head = get_hyve_data('hydraulic head', scenario=scenario)
        ax1.plot(dates, head, color=colors[i+1])  # observed is 0
    ax1.set_title('mean hydraulic head $(m)$', **title_formatter)
    # ax1.set_ylabel('$m$', **label_formatter)
    for i, scenario in enumerate(scenarios):
        dates, reservoir_storage = get_hyve_data('reservoir storage', scenario=scenario)
        reservoir_storage /= 1e9
        ax2.plot(dates, reservoir_storage, color=colors[i+1])  # observed is 0
    ax2.set_title('reservoir storage $(billion\ m^3)$', **title_formatter)
    # ax2.set_ylabel('', **label_formatter)
    for ax in axes:
        ax.ticklabel_format(useOffset=False, axis='y')
        ax.tick_params(axis='both', labelsize='x-small', pad=1)
    # plt.savefig('plot/output/hydro_stats_per_scenario.png')
    plt.savefig('plot/output/hydro_stats_per_scenario.svg')
    plt.show()


if __name__ == '__main__':
    main()
