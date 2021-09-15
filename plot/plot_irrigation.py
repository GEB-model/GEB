# -*- coding: utf-8 -*-
import matplotlib.colors as mcolors

import os
from copy import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
from numba import njit
import rasterio
from rasterio.features import rasterize
import shapely
import yaml
from jplot import plot_raster
from plot import read_npy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib as mpl

with open('GEB.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

@njit(cache=True)
def _decompress_landunit(mixed_array, unmerged_landunit_indices, scaling, mask):
    ysize, xsize = mask.shape
    subarray = np.full((ysize * scaling, xsize * scaling), np.nan, dtype=mixed_array.dtype)
    
    i = 0
    
    for y in range(ysize):
        for x in range(xsize):
            is_masked = mask[y, x]
            if not is_masked:
                for ys in range(scaling):
                    for xs in range(scaling):
                        subarray[y * scaling + ys, x * scaling + xs] = mixed_array[unmerged_landunit_indices[i]]
                        i += 1

    return subarray

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
END_DATE = config['general']['end_time']
TIMEDELTA = timedelta(days=1)
REPORT_FOLDER = config['general']['report_folder']

class Plot:
    def __init__(self):
        areamaps_folder = os.path.join(config['general']['report_folder'], 'base', 'areamaps')
        self.land_owners = np.load(os.path.join(areamaps_folder, 'land_owners.npy'))
        self.field_indices, self.land_owners_per_farmer = _set_fields(self.land_owners)

        # self.mask = np.load('report/mask.npy')
        with rasterio.open(os.path.join('DataDrive', 'GEB', 'input', 'areamaps', 'mask.tif'), 'r') as src:
            self.mask = src.read(1)
        with rasterio.open(os.path.join('DataDrive', 'GEB', 'input', 'areamaps', 'submask.tif'), 'r') as src:
            self.submask = src.read(1)
            self.submask_transform = src.profile['transform']
        
        with rasterio.open(os.path.join('DataDrive', 'GEB', 'input', 'areamaps', 'sub_cell_area.tif'), 'r') as src_cell_area:
            self.cell_area = src_cell_area.read(1)
        
        self.unmerged_landunit_indices = np.load(os.path.join(areamaps_folder, 'unmerged_landunit_indices.npy'))
        self.scaling = np.load(os.path.join(areamaps_folder, 'scaling.npy')).item()

    def read_gadm3(self):
        india_shp = os.path.join('plot', 'cache', 'gadm36_3_krishna.shp')
        if not os.path.exists(india_shp):
            gdf = gpd.GeoDataFrame.from_file(os.path.join('DataDrive/GADM/gadm36_3.shp'))
            gdf = gdf[gdf['GID_0'] == 'IND']
            mask = gpd.GeoDataFrame.from_file(os.path.join('DataDrive', 'GEB', 'input', 'areamaps', 'mask.shp'))
            gdf = gpd.clip(gdf, mask)
            gdf.to_file(india_shp)
        else:
            gdf = gpd.GeoDataFrame.from_file(india_shp)
        return gdf.reset_index()

    def read_command_areas(self):
        fp = os.path.join('DataDrive', 'GEB', 'input', 'routing', 'lakesreservoirs', 'subcommand_areas.tif')
        with rasterio.open(fp, 'r') as src:
            command_areas = src.read(1)
        return command_areas

    def design_plot(self, ax, title):
        if title:
            ax.set_title(title, size=4, pad=2, fontweight='bold')

    def farmer_array_to_fields(self, array, nofieldvalue):
        fields_decompressed = _decompress_landunit(self.land_owners, self.unmerged_landunit_indices, self.scaling, self.mask)
        fields_decompressed = fields_decompressed[self.submask == 0]
        is_field = np.where(fields_decompressed != -1)
        cell_area = self.cell_area[self.submask == 0]
        field_size = np.bincount(fields_decompressed[is_field], weights=cell_area[is_field])

        assert field_size.size == array.size
        array /= field_size

        array = np.take(array, self.land_owners)
        array[self.land_owners == -1] = nofieldvalue
        array = _decompress_landunit(array, self.unmerged_landunit_indices, self.scaling, self.mask)
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
        axins = inset_axes(ax, .6, .6, loc='lower right', bbox_to_anchor=(1.1, -0.03), bbox_transform=ax.transAxes)
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
        axins.set_xlim(2300, 3300) # apply the x-limits
        axins.set_ylim(6500, 7500) # apply the x-limits
        axins.axes.xaxis.set_visible(False)
        axins.axes.yaxis.set_visible(False)
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="black", linewidth=0.3)
        self.design_plot(ax, title)

    @staticmethod
    @njit
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
    @njit
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
            
            
    def plot_by_activation_order(self, array, activation_order, name, ax=None):
        command_areas = self.read_command_areas()
        mapping = np.full(command_areas.max() + 1, 0, dtype=np.int32)
        command_area_ids = np.unique(command_areas)
        assert command_area_ids[0] == -1
        command_area_ids = command_area_ids[1:]
        mapping[command_area_ids] = np.arange(0, command_area_ids.size, dtype=np.int32)
        command_areas_mapped = mapping[command_areas]
        command_areas_mapped[command_areas == -1] = -1
        
        array = self.farmer_array_to_fields(array, 0).ravel()
        activation_order = self.farmer_array_to_fields(activation_order, -1).ravel()
        activation_order[activation_order < 0] = -1
        area_geometries = command_areas_mapped.flatten()
        assert array.size == activation_order.size == area_geometries.size
        areas, area_inverse, area_counts = np.unique(area_geometries, return_inverse=True, return_counts=True)
        activation_order_per_area = self._plot_by_activation_order(areas, area_inverse, area_counts, activation_order)
        activation_order_per_area = {
            key: value[value > 0]
            for key, value in activation_order_per_area.items()
        }

        percentile_values = np.arange(10, 100, 10)
        upper_values, lower_values = [], []
        for percentile in percentile_values:
            mean_activation_order_per_area = {
                key: np.percentile(value, percentile) if value.size > 0 else np.nan
                for key, value in activation_order_per_area.items()
            }
            mean_activation_order_per_area = np.array(list(mean_activation_order_per_area.values()))
            # mean_activation_order_per_area = np.insert(mean_activation_order_per_area, 0, 0)

            is_upper_part = self.is_upper_part_area(activation_order, area_geometries, mean_activation_order_per_area)

            # -1 is neither
            upper_array = array[(is_upper_part == 1) & (self.submask.ravel() == False)]
            lower_array = array[(is_upper_part == 0) & (self.submask.ravel() == False)]

            print('why not nice ratios?')
            print(percentile, upper_array.size, lower_array.size)

            upper_values.append(np.mean(upper_array))
            lower_values.append(np.mean(lower_array))
        
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(percentile_values, lower_values, label=f'lower {name}')
        ax.plot(percentile_values, upper_values, label=f'upper {name}')
        ax.set_xlabel('Basin percentile counted as upper')
        ax.legend()

def read_irrigation_data(scenario):
    dt = copy(START_DATE)
    timesteps = (END_DATE - START_DATE) // TIMEDELTA + 1
    for i in range(timesteps):
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

def plot_irrigation():
    plotter = Plot()

    channel_irrigation, groundwater_irrigation, reservoir_irrigation = read_irrigation_data('base')
    total_irrigation = channel_irrigation + groundwater_irrigation + reservoir_irrigation

    # fig, ax = plt.subplots(1)
    # plotter.plot_by_activation_order(reservoir_irrigation, activation_order, name='reservoir irrigation', ax=ax)
    # plotter.plot_by_activation_order(groundwater_irrigation, activation_order, name='groundwater irrigation', ax=ax)
    # plotter.plot_by_activation_order(channel_irrigation, activation_order, name='channel irrigation', ax=ax)

    # plt.show()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, dpi=300, figsize=(6, 2))
    plt.subplots_adjust(wspace=0.15, left=0.03, right=0.98, bottom=0.15, top=0.99)

    channel_irrigation = plotter.farmer_array_to_fields(channel_irrigation, 0)
    groundwater_irrigation = plotter.farmer_array_to_fields(groundwater_irrigation, 0)
    reservoir_irrigation = plotter.farmer_array_to_fields(reservoir_irrigation, 0)

    cmap = 'Blues'
    vmin = 0
    # vmax = np.nanmax(np.maximum(np.maximum(channel_irrigation, groundwater_irrigation), reservoir_irrigation))
    vmax = 0.01

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
        groundwater_irrigation,
        nofieldvalue=0,
        fig=fig,
        ax=ax1,
        title='groundwater irrigation',
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )

    plotter.plot_array(
        reservoir_irrigation,
        nofieldvalue=0,
        fig=fig,
        ax=ax2,
        title='reservoir irrigation',
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
    plot_irrigation()
