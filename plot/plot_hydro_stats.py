# -*- coding: utf-8 -*-
import os
import re
from isort import stream
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import yaml
import pandas as pd
import sys
import numpy as np
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from plot import read_npy
import matplotlib.transforms as transforms

sys.path.insert(1, os.path.join(sys.path[0], '..'))

TIMEDELTA = timedelta(days=1)
OUTPUT_FOLDER = os.path.join('DataDrive', 'GEB', 'report')
with open('GEB.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
LINEWIDTH = .5
TITLE_FORMATTER = {'size': 5, 'fontweight': 'bold', 'pad': 2}


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


def get_discharge(scenario, switch_crop):
    values = []
    dates = []
    subfolder = scenario
    if switch_crop:
        subfolder += '_switch_crops'
    try:
        with open(os.path.join(OUTPUT_FOLDER, subfolder, 'var.discharge_daily.tss'), 'r') as f:
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
    except FileNotFoundError:
        return None

def get_hyve_data(varname, scenario, switch_crop, fileformat='csv'):
    subfolder = scenario
    if switch_crop:
        subfolder += '_switch_crops'
    try:
        df = pd.read_csv(os.path.join(OUTPUT_FOLDER, subfolder, varname + '.' + fileformat), index_col=0)
    except FileNotFoundError:
        print(f"WARNING: {varname} for {scenario} {'with' if switch_crop else 'without'} not found.")
        return None
    dates = df.index.tolist()
    dates = [datetime.strptime(dt, "%Y-%m-%d") for dt in dates]
    return dates, np.array(df[varname].tolist())

def add_patches_legend(ax, labels, colors, ncol):
    offset = 0.10

    patches = [
        Line2D([0], [0], color=colors[i], label=labels[i], linestyle='--', marker='v', markersize=1, linewidth=0.5)
        for i in range(len(labels)) if labels[i] is not None
    ]
    
    for _ in range(len(patches)):
        patches.insert(0, Line2D([],[],linestyle='', label=''))
    
    legend = ax.legend(
        title=r'$\ \ \ \ \ \ \bf{With\ crop\ switching}$',
        title_fontsize=5,
        handles=patches,
        loc='upper left',
        bbox_to_anchor=(offset, 1.65),
        borderaxespad=0,
        ncol=ncol+1,
        columnspacing=1,
        fontsize=4,
        frameon=False,
        handlelength=3,
        borderpad=0
    )
    legend._legend_box.align = "left"
    
    ax.add_artist(legend)

    patches = [
        Line2D([0], [0], color=colors[i], label=labels[i], linestyle='-', marker='^', markersize=1, linewidth=0.5)
        for i in range(len(labels)) if labels[i] is not None
    ]
    for _ in range(len(patches)):
        patches.insert(0, Line2D([],[],linestyle='', label=''))
    
    legend = ax.legend(
        title=r'$\ \ \ \ \ \ \bf{No\ crop\ switching}$',
        title_fontsize=5,
        handles=patches,
        loc='upper left',
        bbox_to_anchor=(offset, 2.05),
        borderaxespad=0,
        ncol=ncol+1,
        columnspacing=1,
        fontsize=4,
        frameon=False,
        handlelength=3,
        borderpad=0
    )
    legend._legend_box.align = "left"

def get_observed_discharge(dates):
    df = pd.read_csv('DataDrive/GEB/calibration/observations.csv', parse_dates=['Dates'])
    df = df[df['Dates'].isin(dates)]
    df = df.set_index('Dates').resample('1D').mean()
    return df['flow'].to_numpy()

def read_crop_data(dates, scenario, switch_crop):
    surgar_cane = np.zeros(len(dates), dtype=np.float32)
    if switch_crop:
        scenario += '_switch_crops'
    for i, date in enumerate(dates):
        try:
            crops = read_npy(os.path.join('DataDrive', 'GEB', 'report'), 'crop', date.date(), scenario=scenario)
        except FileNotFoundError:
            print(f"WARNING: crops for {scenario} {'with' if switch_crop else 'without'} not found.")
            return None
        surgar_cane[i] = np.count_nonzero(crops == 11)

    return dates, surgar_cane

def scenarios():
    n_agents = np.load(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'farmer_locations.npy')).shape[0]

    scenarios = ('base', 'ngo_training', 'government_subsidies')
    labels = ('No irrigation adaptation', 'NGO adaptation', 'Government subsidies')
    colors = ['black', 'blue', 'orange', 'red']
    colors = colors[:len(scenarios) + 1]
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(5, 6), dpi=300)
    plt.subplots_adjust(left=0.065, right=0.98, bottom=0.08, top=0.96, wspace=0.15, hspace=0.15)
    ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = axes
    fig.delaxes(ax1)
    axes = (ax0, ax2, ax3, ax4, ax5)

    add_patches_legend(
        ax3,
        labels=labels,
        colors=colors,
        ncol=1
    )

    PLOT_STYLE = {'markersize': 1}
    MARKEVERY = 100
    MARKSTARTINT = int(MARKEVERY / 10 / len(scenarios))

    n = 0
    for switch_crop in (False, True):
        if switch_crop:
            PLOT_STYLE['marker'] = 'v'
        else:
            PLOT_STYLE['marker'] = '^'
        PLOT_STYLE['markevery'] = (n * MARKSTARTINT, MARKEVERY)
        linestyle = '--' if switch_crop else '-'

        discharges = []
        for i, scenario in enumerate(scenarios):
            res = get_discharge(scenario, switch_crop=switch_crop)
            if res:
                dates, discharge = res
                ax0.plot(dates, discharge, label=scenario, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)
                discharges.append(discharge)
            n += 1
            PLOT_STYLE['markevery'] = (n * MARKSTARTINT, MARKEVERY)

        for i, scenario in enumerate(scenarios):
            res = get_hyve_data('hydraulic head', scenario=scenario, switch_crop=switch_crop)
            if res:
                dates, head = res
                ax2.plot(dates, head, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)  # observed is 0
            n += 1
            PLOT_STYLE['markevery'] = (n * MARKSTARTINT, MARKEVERY)

        for i, scenario in enumerate(scenarios):
            res = get_hyve_data('reservoir storage', scenario=scenario, switch_crop=switch_crop)
            if res:
                dates, reservoir_storage = res
                reservoir_storage /= 1e9
                ax3.plot(dates, reservoir_storage, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)  # observed is 0
            n += 1
            PLOT_STYLE['markevery'] = (n * MARKSTARTINT, MARKEVERY)

        for i, scenario in enumerate(scenarios):
            res = get_hyve_data('is water aware', scenario=scenario, switch_crop=switch_crop)
            if res:
                dates, efficient = res
                efficient = efficient.astype(np.float64)
                efficient /= (n_agents / 100)
                ax4.plot(dates, efficient, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)  # observed is 0
            n += 1
            PLOT_STYLE['markevery'] = (n * MARKSTARTINT, MARKEVERY)

        for i, scenario in enumerate(scenarios):
            res = read_crop_data(dates, scenario=scenario, switch_crop=switch_crop)
            if res is not None:
                dates, sugar_cane = res
                sugar_cane /= (n_agents / 100)
                ax5.plot(dates, sugar_cane, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)  # observed is 0
            n += 1
            PLOT_STYLE['markevery'] = (n * MARKSTARTINT, MARKEVERY)
    
    ax0.set_title('discharge $(m^3s^{-1})$', **TITLE_FORMATTER)
    ax2.set_title('mean hydraulic head $(m)$', **TITLE_FORMATTER)
    ax3.set_title('reservoir storage $(billion\ m^3)$', **TITLE_FORMATTER)
    ax4.set_title('farmers with high irrigation efficiency $(\%)$', **TITLE_FORMATTER)
    ax5.set_title('farmers with sugar cane $(\%)$', **TITLE_FORMATTER)
    ax0.set_ylim(0, ax0.get_ylim()[1])
    ax4.set_ylim(0, 100)
    ax5.set_ylim(0, 100)
    for ax in axes:
        ax.set_xlim(dates[0], dates[-1] + timedelta(days=1))
        ax.ticklabel_format(useOffset=False, axis='y')
        ax.tick_params(axis='both', labelsize=5, pad=1)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.savefig('plot/output/hydro_stats_per_scenario.png')
    plt.savefig('plot/output/hydro_stats_per_scenario.svg')
    plt.show()

def obs_vs_sim(scenario, calibration_line, monthly=False, start_date=None):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    plt.subplots_adjust(left=0.1, right=0.97, bottom=0.07, top=0.92)
    dates, simulated_discharge = get_discharge(scenario, False)
    
    observed_discharge = get_observed_discharge(dates)

    ax.plot(dates, observed_discharge, color='black', linestyle='-', linewidth=LINEWIDTH, label='observed')
    ax.plot(dates, simulated_discharge, color='blue', linestyle='-', linewidth=LINEWIDTH, label='simulated')
    ax.set_xlim(dates[0], dates[-1] + timedelta(days=2))
    ax.set_title('Observed vs simulated discharge $(m^3s^{-1})$', **TITLE_FORMATTER)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.ticklabel_format(useOffset=False, axis='y')
    ax.tick_params(axis='both', labelsize='xx-small', pad=1)
    ax.tick_params(axis='y', labelsize='xx-small', pad=5, rotation=90)
    ax.legend(fontsize=6, frameon=False, handlelength=2, borderpad=1, loc=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # ax.set_yticklabels(ax.get_yticklabels())#, rotation=40)#, ha=ha[n])
    plt.setp(ax.get_yticklabels(), rotation=90, 
         ha="center", rotation_mode="anchor")
    ax.set_ylabel('$m^3/s$', size='x-small')
    
    ax.axvline(x=calibration_line, ymin=0, ymax=1, color='black', linestyle='--', linewidth=.6)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    offset = timedelta(days=50)
    ax.text(calibration_line - offset, .98, 'calibration', fontsize='xx-small', rotation=90, ha='center', va='top', transform=trans)
    ax.text(calibration_line + offset, .98, 'test', fontsize='xx-small', rotation=90, ha='center', va='top', transform=trans)

    streamflows = pd.DataFrame({'simulated': simulated_discharge, 'observed': observed_discharge}, index=[pd.Timestamp(d) for d in dates])
    streamflows = streamflows[~np.isnan(streamflows['observed'])]
    streamflows['simulated'] += 0.0001
    if monthly:
        streamflows['date'] = streamflows.index
        streamflows = streamflows.resample('M', on='date').mean()
    if calibration_line:
        streamflows = streamflows[streamflows.index > calibration_line]
    KGE_score = round(KGE(streamflows['simulated'], streamflows['observed']), 3)
    corr_score = round(correlation(streamflows['simulated'], streamflows['observed']), 3)

    ax.text(
        0.85,
        0.95,
        r'$\bf{KGE}$: '+ str(KGE_score) + '\n' + r'$\bf{corr}$: ' + str(corr_score),
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes,
        fontsize=5
    )
    plt.savefig('plot/output/obs_vs_sim.png')
    plt.savefig('plot/output/obs_vs_sim.svg')
    plt.show()


if __name__ == '__main__':
    # scenarios()
    obs_vs_sim('base', calibration_line=datetime(2012, 1, 1), monthly=True)
    # obs_vs_sim('base', calibration_line=date(2016, 1, 1), monthly=True, start_date=None)