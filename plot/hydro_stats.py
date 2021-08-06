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
OUTPUT_FOLDER = os.path.join('DataDrive/GEB/output')
with open('GEB.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


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
    return dates, values

def get_polygene_data(varname, scenario, fileformat='csv'):
    df = pd.read_csv(os.path.join('report', scenario, varname + '.' + fileformat), index_col=0)
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

def main():
    title_formatter = {'size': 'small', 'fontweight': 'bold', 'pad': 3}
    colors = ['black', 'blue', 'orange', 'red']
    scenarios = ('base', 'self_investment', 'government_subsidies', 'ngo_training')
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=300)
    plt.subplots_adjust(left=0.04, right=0.99, bottom=0.17, top=0.92, wspace=0.2)
    ax0, ax1, ax2 = axes

    add_patches_legend(
        ax0,
        labels=[s.replace('_', ' ') for s in scenarios],
        colors=colors,
        ncol=4
    )

    for i, scenario in enumerate(scenarios):
        dates, discharge = get_discharge(scenario)
        ax0.plot(dates, discharge, label=scenario, color=colors[i])
    ax0.set_title('discharge $(m^3s^{-1})$', **title_formatter)
    # ax0.set_ylabel('$m^3/s$', **label_formatter)
    for i, scenario in enumerate(scenarios):
        dates, head = get_polygene_data('hydraulic head', scenario=scenario)
        ax1.plot(dates, head, color=colors[i])
    ax1.set_title('mean hydraulic head $(m)$', **title_formatter)
    # ax1.set_ylabel('$m$', **label_formatter)
    for i, scenario in enumerate(scenarios):
        dates, reservoir_storage = get_polygene_data('reservoir storage', scenario=scenario)
        reservoir_storage /= 1e9
        ax2.plot(dates, reservoir_storage, color=colors[i])
    ax2.set_title('reservoir storage $(billion\ m^3)$', **title_formatter)
    # ax2.set_ylabel('', **label_formatter)
    for ax in axes:
        ax.ticklabel_format(useOffset=False, axis='y')
        ax.tick_params(axis='both', labelsize='x-small', pad=1)
    plt.savefig('plot/output/hydro_stats_per_scenario.png')
    plt.savefig('plot/output/hydro_stats_per_scenario.svg')
    # plt.show()


if __name__ == '__main__':
    main()
