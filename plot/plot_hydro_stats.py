# -*- coding: utf-8 -*-
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import pandas as pd
import sys
import numpy as np
from matplotlib import transforms
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from plot import read_npy

from plotconfig import config, ORIGINAL_DATA, INPUT

sys.path.insert(1, os.path.join(sys.path[0], '..'))

TIMEDELTA = timedelta(days=1)
OUTPUT_FOLDER = config['general']['report_folder']
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


def get_discharge(scenario):
    values = []
    dates = []
    try:
        with open(os.path.join(OUTPUT_FOLDER, scenario, 'var.discharge_daily.tss'), 'r') as f:
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
        return dates, np.array(values)
    except FileNotFoundError:
        return None

def get_honeybees_data(varname, scenario, switch_crop, fileformat='csv'):
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
    if 'gauges' in config['general']:
        gauges = config['general']['gauges']
    else:
        gauges = config['general']['poor_point']
    streamflow_path = os.path.join(config['general']['original_data'], 'calibration', 'streamflow', f"{gauges['lon']} {gauges['lat']}.csv")
    df = pd.read_csv(streamflow_path, parse_dates=['Dates'])
    df = df[df['Dates'].isin(dates)]
    erroneous_dates = [date(2017,12,30), date(2018,1,8), date(2018,1,9), date(2018,1,10), date(2018,1,11), date(2018,1,13), date(2018,1,16), date(2018,1,17), date(2018,1,18), date(2018,1,19)]
    df = df[~df['Dates'].isin(erroneous_dates)]
    # re-index dataframe to fill missing dates
    df = df.set_index('Dates').reindex(dates)
    assert len(df) == len(dates)
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

def plot_scenarios(scenarios):
    n_agents = np.load(os.path.join(INPUT, 'agents',  'attributes', 'household size.npy')).shape[0]

    # labels = ('No irrigation adaptation', 'NGO adaptation', 'Government subsidies')
    labels = ('No irrigation adaptation', )
    colors = ['black', 'blue', 'orange', 'red']
    colors = colors[:len(scenarios) + 1]
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(5, 6), dpi=300)
    plt.subplots_adjust(left=0.065, right=0.98, bottom=0.08, top=0.96, wspace=0.15, hspace=0.15)
    ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = axes
    # ax1_position = ax1.get_position()
    fig.delaxes(ax1)
    # ax3_2 = fig.add_subplot(3, 2, 4, facecolor='none')
    # ax3_2.get_shared_x_axes().join(ax1, ax3_2)
    axes = (ax0, ax2, ax3, ax4, ax5)
    # ax3_position = ax3.get_position()
    # ax3_position.y1 = ax1_position.y1
    # ax3.set_position(ax3_position)

    ax0.set_ylim(0, 10_000)
    # start_time = mdates.date2num(date(2007, 1, 1))
    # end_time = mdates.date2num(date(2007, 6, 1))
    ax3.set_ylim(0, 41)
    
    axins1 = ax0.inset_axes([.74, .20, .13, .70], transform=ax0.transAxes)
    axins1.set_xlim(date(2015, 1, 1), date(2015, 8, 1)) # apply the x-limits
    axins1.set_ylim(0, 250) # apply the y-limits
    axins1.get_xaxis().set_visible(False)
    axins1.get_yaxis().set_visible(False)
    for axis in ['top','bottom','left','right']:
        axins1.spines[axis].set_linewidth(.5)
    mark_inset(ax0, axins1, loc1=3, loc2=4, fc="none", ec="black", linewidth=.5)

    axins4 = ax0.inset_axes([.50, .45, .10, .50], transform=ax0.transAxes)
    axins4.set_xlim(date(2011, 9, 10), date(2011, 9, 13)) # apply the x-limits
    axins4.set_ylim(7000, 7800) # apply the y-limits
    axins4.get_xaxis().set_visible(False)
    axins4.get_yaxis().set_visible(False)
    for axis in ['top','bottom','left','right']:
        axins4.spines[axis].set_linewidth(.5)
    mark_inset(ax0, axins4, loc1=3, loc2=2, fc="none", ec="black", linewidth=.5)

    axins = ax3.inset_axes([.03, .01, .2, .25], transform=ax3.transAxes)
    axins.set_xlim(date(2009, 3, 1), date(2009, 8, 1)) # apply the x-limits
    axins.set_ylim(8, 15) # apply the y-limits
    axins.get_xaxis().set_visible(False)
    axins.get_yaxis().set_visible(False)
    for axis in ['top','bottom','left','right']:
        axins.spines[axis].set_linewidth(.5)
    mark_inset(ax3, axins, loc1=2, loc2=4, fc="none", ec="black", linewidth=.5)

    axins2 = ax3.inset_axes([.33, .01, .2, .22], transform=ax3.transAxes)
    axins2.set_xlim(date(2012, 3, 1), date(2012, 8, 1)) # apply the x-limits
    axins2.set_ylim(7, 15) # apply the y-limits
    axins2.get_xaxis().set_visible(False)
    axins2.get_yaxis().set_visible(False)
    for axis in ['top','bottom','left','right']:
        axins2.spines[axis].set_linewidth(.5)
    mark_inset(ax3, axins2, loc1=2, loc2=4, fc="none", ec="black", linewidth=.5)
    # ax0.set_yscale('log')

    # space = .05
    # upper_box = ax3.get_position()
    # size = upper_box.y1 - upper_box.y0
    # upper_box.y0 = upper_box.y0 + size * (.75 + space / 2)
    # lower_box = ax3_2.get_position()
    # lower_box.y1 = lower_box.y1 - size * (.25 + space / 2)
    # ax3.set_position(upper_box)
    # ax3_2.set_position(lower_box)
    
    # ax3.set_ylim(35, 40)
    # ax3_2.set_ylim(2.5, 15)
    # ax3.spines.bottom.set_visible(False)
    # ax3_2.spines.top.set_visible(False)
    # ax3.xaxis.tick_top()
    # ax3.tick_params(labeltop=False)  # don't put tick labels at the top
    # ax3_2.xaxis.tick_bottom()

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
            res = get_discharge(scenario)
            if res:
                dates, discharge = res
                ax0.plot(dates, discharge, label=scenario, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)
                axins1.plot(dates, discharge, label=scenario, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)
                axins4.plot(dates, discharge, label=scenario, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)
                discharges.append(discharge)
            n += 1
            PLOT_STYLE['markevery'] = (n * MARKSTARTINT, MARKEVERY)

        for i, scenario in enumerate(scenarios):
            res = get_honeybees_data('hydraulic head', scenario=scenario, switch_crop=switch_crop)
            if res:
                dates, head = res
                ax2.plot(dates, head, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)  # observed is 0
            n += 1
            PLOT_STYLE['markevery'] = (n * MARKSTARTINT, MARKEVERY)

        for i, scenario in enumerate(scenarios):
            res = get_honeybees_data('reservoir storage', scenario=scenario, switch_crop=switch_crop)
            if res:
                dates, reservoir_storage = res
                reservoir_storage /= 1e9
                ax3.plot(dates, reservoir_storage, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)  # observed is 0
                axins.plot(dates, reservoir_storage, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)  # observed is 0
                axins2.plot(dates, reservoir_storage, color=colors[i], linestyle=linestyle, linewidth=LINEWIDTH, **PLOT_STYLE)  # observed is 0
            n += 1
            PLOT_STYLE['markevery'] = (n * MARKSTARTINT, MARKEVERY)

        for i, scenario in enumerate(scenarios):
            res = get_honeybees_data('well_irrigated', scenario=scenario, switch_crop=switch_crop)
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

    
    ax0.set_title('A - discharge $(m^3s^{-1})$', **TITLE_FORMATTER)
    ax2.set_title('B - mean hydraulic head $(m)$', **TITLE_FORMATTER)
    ax3.set_title('C - reservoir storage $(billion\ m^3)$', **TITLE_FORMATTER)
    ax4.set_title('D - farmers with high irrigation efficiency $(\%)$', **TITLE_FORMATTER)
    ax5.set_title('E - farmers with sugar cane $(\%)$', **TITLE_FORMATTER)
    ax0.set_ylim(0, ax0.get_ylim()[1])
    ax4.set_ylim(0, 100)
    ax5.set_ylim(0, 100)
    for ax in axes:
        ax.set_xlim(dates[0], dates[-1] + timedelta(days=1))
        ax.ticklabel_format(useOffset=False, axis='y')
        ax.tick_params(axis='both', labelsize=5, pad=1)
        # ax.xaxis.set_tick_params(labelsize=5)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # ax3_2.set_xticklabels([])

    plt.savefig('hydro_stats_per_scenario.png')
    plt.savefig('hydro_stats_per_scenario.svg')
    # plt.show()

def obs_vs_sim(scenario, calibration_line, monthly=False, start_time=None):
    output_folder = 'plot/output'
    os.makedirs(output_folder, exist_ok=True)
    if isinstance(calibration_line, date):
        calibration_line = datetime.combine(calibration_line, datetime.min.time())

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    plt.subplots_adjust(left=0.1, right=0.97, bottom=0.07, top=0.92)
    dates, simulated_discharge = get_discharge(scenario)
    
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
    ax.text(calibration_line + offset + timedelta(days=15), .98, 'test', fontsize='xx-small', rotation=90, ha='center', va='top', transform=trans)

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
    plt.savefig(os.path.join(output_folder, 'obs_vs_sim.png'))
    plt.savefig(os.path.join(output_folder, 'obs_vs_sim.svg'))
    # plt.show()


if __name__ == '__main__':
    # scenarios()
    obs_vs_sim('base', calibration_line=config['calibration']['end_time'], monthly=True)
    # obs_vs_sim('base', calibration_line=date(2016, 1, 1), monthly=True, start_time=None)