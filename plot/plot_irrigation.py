# -*- coding: utf-8 -*-
import os
import sys
from copy import copy
from datetime import timedelta, datetime, date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import cm
from numba import njit
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import shapely
from tqdm import tqdm
import yaml
from jplot import plot_raster
from plot import read_npy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from plotconfig import config, ORIGINAL_DATA, INPUT

@njit(cache=True)
def _set_fields(land_owners):
    agents = np.unique(land_owners)
    if agents[0] == -1:
        n_agents = agents.size -1
    else:
        n_agents = agents.size
    field_indices = np.full((n_agents, 2), -1, dtype=np.int32)
    fields_per_farmer = np.full(land_owners.size, -1, dtype=np.int32)

    land_owners_sort_idx = np.argsort(land_owners)
    land_owners_sorted = land_owners[land_owners_sort_idx]

    last_not_owned = np.searchsorted(land_owners_sorted, -1, side='right')

    prev_land_owner = -1
    for i in range(last_not_owned, land_owners.size):
        land_owner = land_owners[land_owners_sort_idx[i]]
        if land_owner != -1:
            if land_owner != prev_land_owner:
                field_indices[land_owner, 0] = i - last_not_owned
            field_indices[land_owner, 1] = i + 1 - last_not_owned
            fields_per_farmer[i - last_not_owned] = land_owners_sort_idx[i]
            prev_land_owner = land_owner
    fields_per_farmer = fields_per_farmer[:-last_not_owned]
    return field_indices, fields_per_farmer


mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['ytick.minor.width'] = 0.5

START_DATE = config['general']['start_time']
TIMEDELTA = timedelta(days=1)
END_DATE = config['general']['end_time'] - TIMEDELTA
REPORT_FOLDER = config['general']['report_folder']

class Plot:
    def __init__(self, scenario):
        report_folder = os.path.join(config['general']['report_folder'], scenario)
        self.land_owners = np.load(os.path.join(report_folder, 'land_owners.npy'))
        self.field_indices, self.land_owners_per_farmer = _set_fields(self.land_owners)

        with rasterio.open(os.path.join(INPUT, 'areamaps', 'mask.tif'), 'r') as src:
            self.mask = src.read(1)
        with rasterio.open(os.path.join(INPUT, 'areamaps', 'submask.tif'), 'r') as src:
            self.submask = src.read(1)
            self.submask_transform = src.profile['transform']
        
        with rasterio.open(os.path.join(INPUT, 'areamaps', 'sub_cell_area.tif'), 'r') as src_cell_area:
            self.cell_area = src_cell_area.read(1)
        
        self.unmerged_HRU_indices = np.load(os.path.join(report_folder, 'unmerged_HRU_indices.npy'))
        self.scaling = np.load(os.path.join(report_folder, 'scaling.npy')).item()

    def decompress_HRU(self, array):
        if np.issubdtype(array.dtype, np.integer):
            nanvalue = -1
        else:
            nanvalue = np.nan
        outarray = array[self.unmerged_HRU_indices]
        outarray[self.submask] = nanvalue
        return outarray

    def read_gadm3(self):
        india_shp = os.path.join('plot', 'cache', 'gadm36_3_krishna.shp')
        if not os.path.exists(india_shp):
            gdf = gpd.GeoDataFrame.from_file(os.path.join('DataDrive/GADM/gadm36_3.shp'))
            gdf = gdf[gdf['GID_0'] == 'IND']
            mask = gpd.GeoDataFrame.from_file(os.path.join(INPUT, 'areamaps', 'mask.shp'))
            gdf = gpd.clip(gdf, mask)
            gdf.to_file(india_shp)
        else:
            gdf = gpd.GeoDataFrame.from_file(india_shp)
        return gdf.reset_index()

    def read_command_areas(self):
        fp = os.path.join(INPUT, 'routing', 'lakesreservoirs', 'subcommand_areas.tif')
        with rasterio.open(fp, 'r') as src:
            command_areas = src.read(1)
        return command_areas

    def design_plot(self, ax, title):
        if title:
            ax.set_title(title, size=6, pad=2, fontweight='bold')

    def farmer_array_to_fields(self, array, nofieldvalue, correct_for_field_size=True):
        fields_decompressed = self.decompress_HRU(self.land_owners)
        fields_decompressed = fields_decompressed[self.submask == 0]
        is_field = np.where(fields_decompressed != -1)
        cell_area = self.cell_area[self.submask == 0]

        if correct_for_field_size:
            field_size = np.bincount(fields_decompressed[is_field], weights=cell_area[is_field])
            assert field_size.size == array.size
            array /= field_size

        array = np.take(array, self.land_owners)
        array[self.land_owners == -1] = nofieldvalue
        array = self.decompress_HRU(array)
        return array

    def plot_by_area(self, array, ax=None, title=None):
        if not hasattr(self, "gadm_3"):
            self.gadm_3 = self.read_gadm3()
            geometries = [(shapely.geometry.mapping(geom), value) for value, geom in zip(self.gadm_3.index.tolist(), self.gadm_3['geometry'].tolist())]
            self.area_geometries = rasterize(geometries, out_shape=self.submask.shape, fill=-1, transform=self.submask_transform, dtype=np.int32, all_touched=True)
        
        array = self.farmer_array_to_fields(array, 0)
        array = array[self.area_geometries != -1]
        areas = self.area_geometries[self.area_geometries != -1]

        areas = areas[~np.isnan(array)]
        array = array[~np.isnan(array)]

        values = np.bincount(areas, weights=array, minlength=len(self.gadm_3))

        self.gadm_3['value'] = values

        if not ax:
            fig, ax = plt.subplots()

        self.gadm_3.plot(column='value', ax=ax, cmap='hot', legend=True)

        self.design_plot(ax, title)

    def plot_array(self, array, fig=None, ax=None, title=None, nofieldvalue=-1, vmin=None, vmax=None, cmap='Blues'):
        if not ax:
            fig, ax = plt.subplots()
        plot_raster(
            array,
            self.submask_transform.to_gdal(),
            proj=4326,
            bbox=None,
            ax=ax,
            fig=fig,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            x_display=3,
            y_display=3,
            realign_tick_labels=True,
            round_ticks=2,
            show=False,
            ticks_fontsize=4,
            # legend='colorbar'
        )
        axins = inset_axes(ax, .45, .45, loc='lower right', bbox_to_anchor=(.85, -0.03), bbox_transform=ax.transAxes)
        plot_raster(
            array,
            self.submask_transform.to_gdal(),
            proj=4326,
            bbox=None,
            ax=axins,
            fig=fig,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            show=False
        )
        axins.set_xlim(2050, 3300) # apply the x-limits
        axins.set_ylim(7250, 6000) # apply the y-limits
        axins.axes.xaxis.set_visible(False)
        axins.axes.yaxis.set_visible(False)
        _, pp1, pp2 = mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="black", linewidth=0.3)
        pp1.loc1 = 1
        pp1.loc2 = 4
        pp2.loc1 = 3
        pp2.loc2 = 2
        
        axins2 = inset_axes(ax, .30, .30, loc='lower right', bbox_to_anchor=(1.05, 0.05), bbox_transform=ax.transAxes)
        plot_raster(
            array,
            self.submask_transform.to_gdal(),
            proj=4326,
            bbox=None,
            ax=axins2,
            fig=fig,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            show=False
        )
        axins2.axes.xaxis.set_visible(False)
        axins2.axes.yaxis.set_visible(False)
        axins2.set_xlim(2050+910-25, 2050+960+25) # apply the x-limits
        axins2.set_ylim(6000+550+25, 6000+500-25) # apply the y-limits
        _, pp3, pp4 = mark_inset(axins, axins2, loc1=1, loc2=3, fc="none", ec="black", linewidth=0.3)
        pp3.loc1 = 2
        pp3.loc2 = 3
        pp4.loc1 = 3
        pp4.loc2 = 2
        
        self.design_plot(ax, title)

    @staticmethod
    @njit(cache=True)
    def _plot_by_activation_order(areas, inverse, counts, activation_order):
        activation_order_per_area = {}
        for i in range(areas.size):
            area = areas[i]
            if area == -1:
                continue
            activation_order_per_area[area] = np.full(counts[i], -1, dtype=np.int32)

        counts.fill(0)
        
        for i in range(inverse.size):
            index = inverse[i]
            area = areas[index]
            if area == -1:
                continue
            activation_order_per_area[area][counts[area]] = activation_order[i]
            counts[area] += 1

        return activation_order_per_area

    @staticmethod
    @njit(cache=True)
    def is_upper_part_area(activation_order, area_geometries, mean_activation_order_per_area):
        is_upper_part = np.full(activation_order.size, -1, dtype=np.int32)
        for i in range(activation_order.size):
            area = area_geometries[i]
            if area == -1:
                continue
            if activation_order[i] < mean_activation_order_per_area[area]:
                is_upper_part[i] = 1
            else:
                is_upper_part[i] = 0
        return is_upper_part
            
            
    def get_irrigation_by_head_and_tail_end(self, array, activation_order):
        command_areas = self.read_command_areas()
        mapping = np.full(command_areas.max() + 1, 0, dtype=np.int32)
        command_area_ids = np.unique(command_areas)
        assert command_area_ids[0] == -1
        command_area_ids = command_area_ids[1:]
        mapping[command_area_ids] = np.arange(0, command_area_ids.size, dtype=np.int32)
        command_areas_mapped = mapping[command_areas]
        command_areas_mapped[command_areas == -1] = -1
        
        array = self.farmer_array_to_fields(array, 0)
        activation_order = self.farmer_array_to_fields(activation_order, -1, correct_for_field_size=False)
        # activation_order[activation_order < 0] = -1
        # area_geometries = command_areas_mapped.flatten()
        # assert array.size == activation_order.size == area_geometries.size
        # areas, area_inverse, area_counts = np.unique(area_geometries, return_inverse=True, return_counts=True)
        # activation_order_per_area = self._plot_by_activation_order(areas, area_inverse, area_counts, activation_order)
        # activation_order_per_area = {
        #     key: value[value > 0]
        #     for key, value in activation_order_per_area.items()
        # }

        head_ends, tail_ends = [], []
        for command_area_id in command_area_ids:
            command_area = command_areas == command_area_id
            activation_order_area = activation_order[command_area]
            if (activation_order_area == -1).all():
                continue
            activation_order_area_filtered = activation_order_area[activation_order_area != -1]
            array_area = array[command_area][activation_order_area != -1]
            # get median of activation order
            activation_order_median = np.percentile(activation_order_area_filtered, 50)
            
            head_end = array_area[activation_order_area_filtered < activation_order_median].sum()
            tail_end = array_area[activation_order_area_filtered >= activation_order_median].sum()
            head_ends.append(head_end)
            tail_ends.append(tail_end)
        
        # if ax is None:
        #     fig, ax = plt.subplots(1)
        
        # # for each couple of tail and head end, plot a line with a circle at data point
        # for i in range(len(head_ends)):
        #     ax.plot([0, 1], [head_ends[i], tail_ends[i]], color='black', linewidth=1, marker='o', markersize=5)
        
        # # visualize x axis betwwen 0 and 1
        # ax.set_xlim(0, 1)
        # # visualize y axis between 0 and max of head and tail ends + 10%
        # ax.set_ylim(0, max(max(head_ends), max(tail_ends)) * 1.1)
        # # do not show x axis
        # ax.xaxis.set_visible(False)
        # plt.show()


def read_irrigation_data(scenario):
    dt = copy(START_DATE)
    timesteps = (END_DATE - START_DATE) // TIMEDELTA + 1
    print(END_DATE)
    for i in range(timesteps):
        print(dt)
        if not i % 10:
            print(i)
        if i == 0:
            channel_irrigation = read_npy(REPORT_FOLDER, 'channel irrigation', dt, scenario=scenario)
            groundwater_irrigation = read_npy(REPORT_FOLDER, 'groundwater irrigation', dt, scenario=scenario)
            reservoir_irrigation = read_npy(REPORT_FOLDER, 'reservoir irrigation', dt, scenario=scenario)
        else:
            channel_irrigation += read_npy(REPORT_FOLDER, 'channel irrigation', dt, scenario=scenario)
            groundwater_irrigation += read_npy(REPORT_FOLDER, 'groundwater irrigation', dt, scenario=scenario)
            reservoir_irrigation += read_npy(REPORT_FOLDER, 'reservoir irrigation', dt, scenario=scenario)
        dt += TIMEDELTA

    channel_irrigation /= timesteps
    groundwater_irrigation /= timesteps
    reservoir_irrigation /= timesteps

    return channel_irrigation, groundwater_irrigation, reservoir_irrigation

def add_colorbar_legend(
    fig,
    axes,
    cmap,
    vmin=None,
    vmax=None,
    legend_text_format=None,
    realign_colorbar_tick_labels=False,
    legend_title=None,
    legend_offset=0.10,
    legend_title_fontsize=5,
    extend_colorbar='neither',
    legend_location='bottom',
    labelsize='x-small'
):
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph

    assert len(axes) == 2
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])

    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()

    width = pos1.xmax - pos0.xmin
    reduce_width = width * 0.9
    move_right = (width - reduce_width) / 2
    cb_ax = fig.add_axes(
        [
            pos0.xmin + move_right,
            pos0.ymin - (pos0.ymax - pos0.ymin) * legend_offset,
            reduce_width,
            (pos0.ymax - pos0.ymin) * 0.03
        ]
    )
    orientation = 'horizontal'
    
    colorbar = plt.colorbar(
        mappable,
        cax=cb_ax,
        orientation=orientation,
        extend=extend_colorbar,
        format=legend_text_format,
        drawedges=False
    )
    
    colorbar.ax.tick_params(labelsize=labelsize, length=2, pad=1)

    if legend_title:
        # using x-label such that the title is displayed below the colorbar
        colorbar.ax.set_xlabel(legend_title, fontsize=legend_title_fontsize, labelpad=2)

def plot_irrigation(scenario):
    plotter = Plot(scenario=scenario)

    channel_irrigation_by_farm, groundwater_irrigation, reservoir_irrigation = read_irrigation_data(scenario)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, dpi=300, figsize=(6, 2))
    plt.subplots_adjust(wspace=0.15, left=0.03, right=0.98, bottom=0.15, top=0.99)

    cmap = 'Blues'
    # vmax = np.nanmax(np.maximum(np.maximum(channel_irrigation, groundwater_irrigation), reservoir_irrigation))
    vmin = 0
    vmax = 0.02

    fields = np.ones_like(channel_irrigation_by_farm)
    fields = plotter.farmer_array_to_fields(fields, 0)
    # fields = fields[:1000, :1000]

    def get_plt(array):
        array /= np.nanmax(vmax)
        array[array > vmax] = 1
        array[array < 0] = 0
        array = cm.Blues(array)
        array[fields == 0,:] = .9
        array[np.isnan(fields), :] = 1
        return array

    channel_irrigation = plotter.farmer_array_to_fields(channel_irrigation_by_farm, 0)
    # channel_irrigation = channel_irrigation[:1000, :1000]
    channel_irrigation = get_plt(channel_irrigation)
    groundwater_irrigation = plotter.farmer_array_to_fields(groundwater_irrigation, 0)
    groundwater_irrigation = get_plt(groundwater_irrigation)
    reservoir_irrigation = plotter.farmer_array_to_fields(reservoir_irrigation, 0)
    reservoir_irrigation = get_plt(reservoir_irrigation)


    plotter.plot_array(
        channel_irrigation,
        nofieldvalue=0,
        fig=fig,
        ax=ax0,
        title='channel irrigation',
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )

    plotter.plot_array(
        reservoir_irrigation,
        nofieldvalue=0,
        fig=fig,
        ax=ax1,
        title='reservoir irrigation',
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )
    
    plotter.plot_array(
        groundwater_irrigation,
        nofieldvalue=0,
        fig=fig,
        ax=ax2,
        title='groundwater irrigation',
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )

    add_colorbar_legend(
        fig,
        cmap=cmap,
        axes=[ax0, ax2],
        vmin=vmin,
        vmax=vmax,
        legend_offset=0.15,
        labelsize=4,
        legend_title_fontsize=4,
        legend_title='$m\ day^{-1}$'
    )

    output_folder = 'plot/output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.savefig(os.path.join(output_folder, 'irrigation_per_source.png'))
    plt.savefig(os.path.join(output_folder, 'irrigation_per_source.svg'))
    # plt.show()

if __name__ == '__main__':
    plot_irrigation('base')
