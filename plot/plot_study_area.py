# -*- coding: utf-8 -*-
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    raise ImportError("Matplotlib not found, could not create plot")

from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from cartopy.io.img_tiles import OSM
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import cartopy.feature as cfeature

import geopandas as gpd
import cartopy.crs as ccrs
import numpy as np
import os

from shapely.geometry import Polygon

from plotconfig import INPUT

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
# mpl.rcParams['image.interpolation'] = 'spline36'

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly, approx=False)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')


def create_figure(outline):
    crs = ccrs.PlateCarree()
    ax = plt.axes(projection=crs)
    ax.add_geometries([outline], crs=crs, facecolor='none', linewidth=1, edgecolor='black')   
    return ax, crs


def plot_location(outline):
    minx, miny, maxx, maxy = outline.bounds
    imagery = OSM()

    # fig = plt.figure(figsize=(50, 50), dpi=300)

    ax, crs = create_figure(outline)

    border = 1
    ax.set_extent((minx - border, maxx + border, miny - border * 0.3, maxy + border))
    ax.add_image(imagery, 8, interpolation='spline36', regrid_shape=5_000)

    ip = InsetPosition(ax, [0.6, -0.13, 0.6, 0.5])
    ax2 = plt.axes([0,0,1,1], projection=crs)
    ax2.set_axes_locator(ip)
    ax2.add_geometries([outline], crs=crs, facecolor='none', linewidth=1, edgecolor='black')

    ax2.set_extent((67.08, 98.13, 4.69, 36.46))
    ax2.add_image(imagery, 5, interpolation='spline36', regrid_shape=2000)

    # scale_bar(ax, (minx, miny), 1)

    # ax.set_title(f"lon: {lon}, lat: {lat}, radius: {radius}, bits: {bits}")

    scale_bar(ax, length=None, location=(0.15, 0.05), linewidth=3)

    # plt.show()
    plt.savefig('plot/output/study_area.tif', dpi=300)


def plot_basins(outline):
    ax, crs = create_figure(outline)
    plt.subplots_adjust(bottom=0.15, top=1)

    minx, miny, maxx, maxy = outline.bounds
    border = .5
    ax.set_extent((minx - border, maxx + border, miny - border, maxy + border))

    lakes = gpd.GeoDataFrame.from_file("DataDrive/GEB/input/routing/lakesreservoirs/hydrolakes.shp")
    lakes.plot(ax=ax, color='#3B94DB')

    command_areas = gpd.GeoDataFrame.from_file("DataDrive/GEB/input/routing/lakesreservoirs/command_areas.shp")
    command_areas.plot(ax=ax, legend=True, color="#D1A106")

    patches = [
        mpatches.Patch(color='#3B94DB', label="Water bodies"),
        mpatches.Patch(color="#D1A106", label="Reservoir command areas"),
    ]
    legend = ax.legend(
        handles=patches,
        loc='upper left',
        bbox_to_anchor=(0, -0.03),
        borderaxespad=0,
        ncol=1,
        columnspacing=1,
        fontsize=6,
        frameon=False,
        handlelength=1,
        borderpad=0
    )
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                                        edgecolor='face',
                                        facecolor="#0099FF")
    ax.add_feature(ocean)

    scale_bar(ax, length=None, location=(0.15, 0.05), linewidth=3)
    plt.savefig('plot/output/bhima_basin_water_bodies_command_areas.png', dpi=300)
    plt.savefig('plot/output/bhima_basin_water_bodies_command_areas.eps')
    plt.show()

def plot_cutout(outline):
    minx, miny, maxx, maxy = outline.bounds
    imagery = OSM()

    ax, crs = create_figure(outline)

    border = 1
    minx -= border
    maxx += border
    miny -= border
    maxy += border
    ax.set_extent((minx, maxx, miny, maxy))

    plot_border = Polygon([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny], [minx, miny]])
    plot_border = gpd.GeoDataFrame(geometry=[plot_border])
    mask = plot_border.overlay(gpd.GeoDataFrame(geometry=[outline]), how='difference')
    mask.plot(ax=ax, edgecolor='none', facecolor='white')

    ax.add_image(imagery, 8, interpolation='spline36', regrid_shape=5_000)

    plt.show()
    plt.savefig('plot/output/cutout.png', dpi=300)


if __name__ == '__main__':
    study_area = os.path.join(INPUT, 'areamaps', 'mask.shp')
    outline = gpd.GeoDataFrame.from_file(study_area).iloc[2].geometry
    plot_location(outline)
    # plot_cutout(outline)

    # plot_basins(outline)