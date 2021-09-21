# -*- coding: utf-8 -*-
import rasterio
import os
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize

original_data = os.path.join('DataDrive', 'GEB', 'original_data')
output_folder = os.path.join('DataDrive', 'GEB', 'input', 'routing', 'lakesreservoirs')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def cut_hydrolakes() -> None:
    """This function cuts the hydrolakes dataset using the basin mask such that is easier to handle."""
    outfile = os.path.join(output_folder, 'hydrolakes.shp')
    if not os.path.exists(outfile):
        with rasterio.open('DataDrive/GEB/input/areamaps/mask.tif', 'r') as src:
            bounds = src.bounds

        gdf = gpd.GeoDataFrame.from_file(os.path.join(original_data, "HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp"))
        basin = gdf.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
        basin.to_file(outfile)

def lakeResIDs2raster() -> None:
    """Creates a raster from hydrolakes dataset, both at the resolution of the basin mask and submask"""
    shpfile = os.path.join(output_folder, 'hydrolakes.shp')
    basin_lakes = gpd.GeoDataFrame.from_file(shpfile)
    geometries = [(shapely.geometry.mapping(geom), value) for value, geom in zip(basin_lakes['Hylak_id'].tolist(), basin_lakes['geometry'].tolist())]
    for type_ in ('', 'sub'):
        with rasterio.open(f'DataDrive/GEB/input/areamaps/{type_}mask.tif') as src:
            profile = src.profile
            transform = src.profile['transform']
            shape = src.profile['height'], src.profile['width']

            lake_ids = rasterize(geometries, out_shape=shape, fill=0, transform=transform, dtype=np.int32, all_touched=True)

            with rasterio.open(os.path.join(output_folder, f'{type_}lakesResID.tif'), 'w', **profile) as dst:
                dst.write(lake_ids, 1)

def export_other_lake_data(reservoirs: list[int], area_in_study_area: pd.Series) -> None:
    """
    This function exports several data from the hydrolakes dataset to raster for use in CWatM; year of construction, lake type, area and volume.

    All lakes in the hydrolakes dataset are set as reservoir when coupled to a command area.

    1 = lake
    2 = reservoir
    3 = natural lake with regulation structure

    Args:
        reservoirs: list of reservoir ids.
    """
    shpfile = os.path.join(output_folder, 'hydrolakes.shp')
    basin_lakes = gpd.GeoDataFrame.from_file(shpfile)
    # set all lakes with command area as reservoir
    basin_lakes.loc[basin_lakes['Hylak_id'].isin(reservoirs), 'Lake_type'] = 2
    with rasterio.open(os.path.join(output_folder, 'lakesResID.tif'), 'r') as src:
        lake_ids = src.read(1)
        reservoir_construction_year = np.where(lake_ids > 0, 0, -1)
        with rasterio.open(os.path.join(output_folder, 'reservoir_year_construction.tif'), 'w', **src.profile) as dst:
            dst.write(reservoir_construction_year, 1)

        lake_type_array = np.full(basin_lakes['Hylak_id'].max() + 1, 0, dtype=np.int8)
        lake_type_array[basin_lakes['Hylak_id']] = basin_lakes['Lake_type']
        lake_type = np.take(lake_type_array, lake_ids, mode='clip')
        with rasterio.open(
            os.path.join(output_folder, 'lakesResType.tif'), 'w',
            **{**src.profile, **{'dtype': lake_type.dtype}}
        ) as dst:
            dst.write(lake_type, 1)

        lake_command_area_in_study_area_array = np.full(basin_lakes['Hylak_id'].max() + 1, 0, dtype=np.float32)
        lake_command_area_in_study_area_array[area_in_study_area.index] = area_in_study_area
        lake_command_area_in_study_area = np.take(lake_command_area_in_study_area_array, lake_ids, mode='clip')
        with rasterio.open(
            os.path.join(output_folder, 'area_command_area_in_study_area.tif'), 'w',
            **{**src.profile, **{'dtype': lake_command_area_in_study_area.dtype}}
        ) as dst:
            dst.write(lake_command_area_in_study_area, 1)

        lake_area_array = np.full(basin_lakes['Hylak_id'].max() + 1, -1, dtype=np.float32)
        lake_area_array[basin_lakes['Hylak_id']] = basin_lakes['Lake_area']  # in km2
        lake_area = np.take(lake_area_array, lake_ids, mode='clip')
        with rasterio.open(
            os.path.join(output_folder, 'lakesResArea.tif'), 'w',
            **{**src.profile, **{'dtype': lake_area.dtype}}
        ) as dst:
            dst.write(lake_area, 1)

        lake_dis_array = np.full(basin_lakes['Hylak_id'].max() + 1, -1, dtype=np.float32)
        lake_dis_array[basin_lakes['Hylak_id']] = basin_lakes['Dis_avg']
        lake_dis = np.take(lake_dis_array, lake_ids, mode='clip')
        with rasterio.open(
            os.path.join(output_folder, 'lakesResDis.tif'), 'w',
            **{**src.profile, **{'dtype': lake_dis.dtype}}
        ) as dst:
            dst.write(lake_dis, 1)

        df = pd.read_excel('DataDrive/GEB/original_data/reservoir_capacity.xlsx')
        df = df[df['Hylak_id'] != -1].set_index('Hylak_id')

        reservoir_volumes = basin_lakes.set_index('Hylak_id')['Vol_total'].copy()
        for hylak_id, capacity in df['Gross_capacity_BCM'].items():
            capacity *= 1000  # BCM to MCM
            reservoir_volumes.loc[hylak_id] = capacity
        
        res_vol_array = np.full(basin_lakes['Hylak_id'].max() + 1, -1, dtype=np.float32)
        res_vol_array[basin_lakes['Hylak_id']] = reservoir_volumes
        res_vol = np.take(res_vol_array, lake_ids, mode='clip')
        with rasterio.open(
            os.path.join(output_folder, 'waterBodyVolRes.tif'), 'w',
            **{**src.profile, **{'dtype': res_vol.dtype}}
        ) as dst:
            dst.write(res_vol, 1)

        reservoir_FLR = reservoir_volumes.copy()
        for hylak_id, capacity in df['Capacity_FLR_BCM'].items():
            capacity *= 1000  # BCM to MCM
            reservoir_FLR.loc[hylak_id] = capacity

        res_vol_array = np.full(basin_lakes['Hylak_id'].max() + 1, -1, dtype=np.float32)
        res_vol_array[basin_lakes['Hylak_id']] = reservoir_FLR
        res_vol = np.take(res_vol_array, lake_ids, mode='clip')
        with rasterio.open(
            os.path.join(output_folder, 'waterBodyVolResFLR.tif'), 'w',
            **{**src.profile, **{'dtype': res_vol.dtype}}
        ) as dst:
            dst.write(res_vol, 1)

def export_variables_to_csv() -> None:
    """Exports the data from the hydrolakes shapefile to a csv file, while dropping the geometry columns. This simply allows the model to more rapidly read the sheet."""
    shpfile = os.path.join(output_folder, 'hydrolakes.shp')
    basin_lakes = gpd.GeoDataFrame.from_file(shpfile)
    df = pd.DataFrame(basin_lakes.drop(columns='geometry'))
    df.to_csv('DataDrive/GEB/input/routing/lakesreservoirs/hydrolakes.csv', index=False)

def create_command_area_raster() -> list[int]:
    """Create a command area raster for those command areas that were linked to the hydrolakes dataset. Command areas that are not 'complete' according to the dataset are dropped.
    
    Returns:
        hydrolake_ids: A list of hydrolake ids which link to a command area."""
    shpfile = os.path.join(output_folder, 'command_areas.shp')
    command_areas = gpd.GeoDataFrame.from_file(shpfile)
    # remove command areas not associated with a reservoir
    command_areas = command_areas[~command_areas['Hylak_id'].isnull()]
    command_areas['Hylak_id'] = command_areas['Hylak_id'].astype(np.int32)

    mask = gpd.read_file(os.path.join('DataDrive/GEB/input/areamaps/mask.shp'))

    command_areas_in_study_area = gpd.overlay(command_areas, mask, how='intersection')
    command_areas_in_study_area['area'] = command_areas_in_study_area.area
    area_of_command_area_in_study_area_by_hylak_id = command_areas_in_study_area.groupby('Hylak_id')['area'].sum()

    command_areas['area'] = command_areas.area
    area_of_command_area_by_hylak_id = command_areas.groupby('Hylak_id')['area'].sum()

    merge = pd.merge(left=area_of_command_area_in_study_area_by_hylak_id, right=area_of_command_area_by_hylak_id, left_index=True, right_index=True)
    area_in_study_area = merge['area_x'] / merge['area_y']
    
    geometries = [(shapely.geometry.mapping(geom), value) for value, geom in zip(command_areas['Hylak_id'].tolist(), command_areas['geometry'].tolist())]
    for type_ in ('', 'sub'):
        with rasterio.open(f'DataDrive/GEB/input/areamaps/{type_}mask.tif') as src:
            profile = src.profile
            transform = src.profile['transform']
            shape = src.profile['height'], src.profile['width']
            profile['nodata'] = None

            command_areas_raster = rasterize(geometries, out_shape=shape, fill=-1, transform=transform, dtype=np.int32, all_touched=True)
            output_file = f'DataDrive/GEB/input/routing/lakesreservoirs/{type_}command_areas.tif'
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(command_areas_raster, 1)

    return command_areas['Hylak_id'].to_list(), area_in_study_area


if __name__ == '__main__':
    cut_hydrolakes()
    lakeResIDs2raster()
    export_variables_to_csv()
    reservoirs, area_in_study_area = create_command_area_raster()
    export_other_lake_data(reservoirs, area_in_study_area)
    