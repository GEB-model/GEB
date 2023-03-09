# -*- coding: utf-8 -*-
import os
import rasterio
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
import geopandas as gpd

from honeybees.library.raster import clip_to_xy_bounds, clip_to_other, upscale

from methods import create_cell_area_map

from preconfig import config, ORIGINAL_DATA, INPUT

UPSCALE_FACTOR = config['general']['upscale_factor']
BASIN_ID = config['general']['basin_id']
if 'poor_point' in config['general']:
    POOR_POINT = config['general']['poor_point']['lon'], config['general']['poor_point']['lat']
else:
    POOR_POINT = None
from typing import Union

os.makedirs(os.path.join(INPUT, 'areamaps'), exist_ok=True)
os.makedirs(os.path.join(INPUT, 'routing', 'kinematic'), exist_ok=True)
os.makedirs(os.path.join(INPUT, 'landsurface', 'topo'), exist_ok=True)

    
def create_mask(basin_id: int, upscale_factor: int, poor_point: Union[None, tuple[float, float]]=None) -> tuple[rasterio.profiles.Profile, rasterio.profiles.Profile]:
    """This function creates a mask and a submask (using global UPSCALE_FACTOR), and returns rasterio profile for both

    Args:
        basin_id: The basin id of the `merit_hydro_30sec/30sec_basids.tif`
        upscale_factor: The size of the subcell mask relative to the basin mask.
        poor_point: Optional poor point of subbasin. pyflwdir must be installed to use this option.

    Returns:
        mask_profile: rasterio profile for mask
        submask_profile: rasterio profile for submask
    """
    mask_fn = os.path.join(INPUT, 'areamaps', 'mask.tif')
    submask_fn = os.path.join(INPUT, 'areamaps', 'submask.tif')
    with rasterio.open(os.path.join(ORIGINAL_DATA, 'merit_hydro_30sec/30sec_basids.tif'), 'r') as src:
        basins = src.read(1)
        mask = basins != basin_id  # mask anything that is not basin id

        nonmaskedx = np.where(mask.all(axis=0)==False)[0]
        xmin, xmax = nonmaskedx[0], nonmaskedx[-1] + 1
        nonmaskedy = np.where(mask.all(axis=1)==False)[0]
        ymin, ymax = nonmaskedy[0], nonmaskedy[-1] + 1

        mask_profile_org = src.profile
        mask_profile, mask = clip_to_xy_bounds(src, mask_profile_org, mask, xmin, xmax, ymin, ymax)
        mask_profile['nodata'] = -1

        if poor_point:
            import pyflwdir
            with rasterio.open(os.path.join(ORIGINAL_DATA, 'merit_hydro_30sec/30sec_flwdir.tif'), 'r') as src:
                ldd_profile, ldd = clip_to_xy_bounds(src, mask_profile, src.read(1), xmin, xmax, ymin, ymax)

            flw = pyflwdir.from_array(
                ldd,
                ftype="d8",
                transform=ldd_profile['transform'],
                latlon=ldd_profile['crs'].is_geographic,
                cache=True,
            )
            subbasin_mask = ~flw.basins(xy=poor_point).astype(bool)
            nonmaskedx = np.where(subbasin_mask.all(axis=0)==False)[0]
            subxmin, subxmax = nonmaskedx[0], nonmaskedx[-1] + 1
            nonmaskedy = np.where(subbasin_mask.all(axis=1)==False)[0]
            subymin, subymax = nonmaskedy[0], nonmaskedy[-1] + 1

            mask_profile, _ = clip_to_xy_bounds(src, mask_profile_org, mask, subxmin + xmin, subxmax + xmin, subymin + ymin, subymax + ymin)
            mask = subbasin_mask[subymin: subymax, subxmin: subxmax]
            mask_profile['nodata'] = -1

        with rasterio.open(mask_fn, 'w', **mask_profile) as mask_clipped_src:
            mask_clipped_src.write(mask.astype(mask_profile['dtype']), 1)

        submask, submask_profile = upscale(mask, mask_profile, upscale_factor)
        with rasterio.open(submask_fn, 'w', **submask_profile) as submask_clipped_src:
            submask_clipped_src.write(submask.astype(submask_profile['dtype']), 1)

        return mask_profile, submask_profile

def create_ldd(mask_profile: rasterio.profiles.Profile) -> None:
    """Clip ldd, and convert ArcGIS D8 convention to pcraster LDD convention. 
    
    Args:
        mask_profile: Rasterio profile of basin mask
    """
    output_fn = os.path.join(INPUT, 'routing/kinematic/ldd.tif')
    with rasterio.open(os.path.join(ORIGINAL_DATA, 'merit_hydro_30sec/30sec_flwdir.tif'), 'r') as src:
        dir_map = src.read(1)
        src_profile = src.profile

        dir_map, profile = clip_to_other(dir_map, src_profile, mask_profile)

        ldd = np.zeros_like(dir_map)
        ldd[dir_map == 32] = 7
        ldd[dir_map == 64] = 8
        ldd[dir_map == 128] = 9
        ldd[dir_map == 16] = 4
        ldd[dir_map == 0] = 5
        ldd[dir_map == 1] = 6
        ldd[dir_map == 8] = 1
        ldd[dir_map == 4] = 2
        ldd[dir_map == 2] = 3
        
        with rasterio.open(output_fn, 'w', **profile) as dst:
            dst.write(ldd, 1)

def get_channel_manning_and_width(mask_profile: rasterio.profiles.Profile) -> None:
    """Estimate channel Manning's roughness coefficient, and channel width. 
    
    Args:
        mask_profile: Rasterio profile of basin mask
    """
    with rasterio.open(os.path.join(ORIGINAL_DATA, 'merit_hydro_30sec/30sec_elevtn.tif'), 'r') as DEM_src:
        DEM = DEM_src.read(1)
        DEM, DEM_profile = clip_to_other(DEM, DEM_src.profile, mask_profile)
    
    cell_area_path = os.path.join(INPUT, 'areamaps', 'cell_area.tif')
    with rasterio.open(cell_area_path, 'r') as cell_area_src:
        area_m = cell_area_src.read(1)

    with rasterio.open(os.path.join(ORIGINAL_DATA, 'merit_hydro_30sec/30sec_uparea.tif'), 'r') as uparea_src:
        upstream_area = uparea_src.read(1) #* 1e6  # km2 to m2
        upstream_area, upstream_profile = clip_to_other(upstream_area, uparea_src.profile, mask_profile)

        with rasterio.open(os.path.join(INPUT, 'routing/kinematic/upstream_area.tif'), 'w', **upstream_profile) as upstream_dst:
            upstream_dst.write(upstream_area, 1)

    a = (2 * area_m) / upstream_area
    a[a > 1] = 1
    b = DEM / 2000
    b[b > 1] = 1
    manning = 0.025 + 0.015 * a + 0.030 * b

    with rasterio.open(os.path.join(INPUT, 'routing/kinematic/manning.tif'), 'w', **DEM_profile) as manning_dst:
        manning_dst.write(manning.astype(np.float32), 1)

    chanwidth = upstream_area / 500.
    chanwidth[chanwidth < 3] = 3

    with rasterio.open(os.path.join(INPUT, 'routing/kinematic/chanwidth.tif'), 'w', **DEM_profile) as chanwidth_dst:
        chanwidth_dst.write(chanwidth.astype(np.float32), 1)

    upstream_area_no_negative = np.array(upstream_area)
    upstream_area_no_negative[upstream_area_no_negative < 0] = 0
    channel_bankfull = 0.27 * upstream_area_no_negative ** 0.26
    with rasterio.open(os.path.join(INPUT, 'routing/kinematic/chandepth.tif'), 'w', **DEM_profile) as channel_bankfull_dst:
        channel_bankfull_dst.write(channel_bankfull.astype(np.float32), 1)


def get_river_length_and_channelratio(mask_profile: rasterio.profiles.Profile) -> None:
    """Clips channel length to mask, and calculates channel ratio from cell area, channel length and channel width
    
    Args:
        mask_profile: Rasterio profile of basin mask
    """
    with rasterio.open(os.path.join(ORIGINAL_DATA, 'merit_hydro_30sec/30sec_rivlen_ds.tif'), 'r') as rivlen_src:
        rivlen = rivlen_src.read(1)
        rivlen, profile = clip_to_other(rivlen, rivlen_src.profile, mask_profile)
        rivlen[rivlen == -9999] = np.nan

        with rasterio.open(os.path.join(INPUT, 'routing/kinematic/chanleng.tif'), 'w', **profile) as chanleng_dst:
            chanleng_dst.write(rivlen, 1)

        with rasterio.open(os.path.join(INPUT, 'areamaps', 'cell_area.tif'), 'r') as cell_area_src:
            cell_area = cell_area_src.read(1)

        with rasterio.open(os.path.join(INPUT, 'routing/kinematic/chanwidth.tif'), 'r') as chanwidth_src:
            chanwidth = chanwidth_src.read(1)
        
        rivlen[rivlen < 0] = 0
        river_area = chanwidth * rivlen
        river_ratio = river_area / cell_area
        river_ratio[river_ratio > 1] = 1
        assert not (river_ratio < 0).any()

        with rasterio.open(os.path.join(INPUT, 'routing/kinematic/chanratio.tif'), 'w', **profile) as river_ratio_dst:
            river_ratio_dst.write(river_ratio, 1)

def get_river_slope(mask_profile: rasterio.profiles.Profile) -> None:
    """Clips riverslope map to mask, and outputs as channelgradient
    
    Args:
        mask_profile: Rasterio profile of basin mask
    """
    with rasterio.open(os.path.join(ORIGINAL_DATA, 'merit_hydro_30sec/30sec_rivslp.tif'), 'r') as rivslp_src:
        rivslp = rivslp_src.read(1)
        rivslp, profile = clip_to_other(rivslp, rivslp_src.profile, mask_profile)

        with rasterio.open(os.path.join(INPUT, 'routing/kinematic/changrad.tif'), 'w', **profile) as chanslp_dst:
            chanslp_dst.write(rivslp, 1)

def get_elevation_std(mask_profile: rasterio.profiles.Profile) -> None:
    """Clips 30 arcsecond elevation map to mask, and creates map of standard deviation in elevation map using the 3 arcsecond elevation map.

    All MERIT Hydro elevation files should be in `ROOT/original_data/merit_hydro_03sec`.
    
        Args:
            mask_profile: Rasterio profile of basin mask
    """
    with rasterio.open(os.path.join(ORIGINAL_DATA, 'merit_hydro_30sec/30sec_elevtn.tif'), 'r') as DEM_src:
        DEM_profile = DEM_src.profile
        DEM = DEM_src.read(1)

    merit_hydro_03sec_folder = os.path.join(ORIGINAL_DATA, 'merit_hydro_03sec')
    elv_maps = []
    for fn in os.listdir(merit_hydro_03sec_folder):
        fp = os.path.join(merit_hydro_03sec_folder, fn)
        if os.path.splitext(fp)[0].endswith('_elv'):
            src = rasterio.open(fp)
            High_res_DEM_profile_org = src.profile
            elv_maps.append(rasterio.open(fp))
    
    High_res_DEM, High_res_DEM_transform_org = merge(elv_maps)
    High_res_DEM = High_res_DEM[0, :, :]
    High_res_DEM_profile_org.update({
        'transform': High_res_DEM_transform_org,
        'width': High_res_DEM.shape[1],
        'height': High_res_DEM.shape[0],
    })

    scaling = 10
    DEM, DEM_profile = clip_to_other(DEM, DEM_profile, mask_profile)
    with rasterio.open(os.path.join(INPUT, 'landsurface/topo/elv.tif'), 'w', **DEM_profile) as dst:
        dst.write(DEM, 1)
    _, high_res_dem_profile_target = upscale(DEM, mask_profile, scaling)
    High_res_DEM, High_res_DEM_profile = clip_to_other(High_res_DEM, High_res_DEM_profile_org, high_res_dem_profile_target)
    High_res_DEM[High_res_DEM < 0] = 0

    with rasterio.open(os.path.join(INPUT, 'landsurface/topo/subelv.tif'), 'w', **High_res_DEM_profile) as dst:
        dst.write(High_res_DEM, 1)
    
    elevation_per_cell = (High_res_DEM.reshape(High_res_DEM.shape[0] // scaling, scaling, -1, scaling).swapaxes(1, 2).reshape(-1, scaling, scaling))
    elevationSTD = np.std(elevation_per_cell, axis=(1,2)).reshape(DEM.shape)

    with rasterio.open(os.path.join(INPUT, 'landsurface/topo/elvstd.tif'), 'w', **DEM_profile) as dst:
        dst.write(elevationSTD, 1)

def create_mask_shapefile() -> None:
    """Creates a shapefile from the basin mask.
    """
    mask_file = os.path.join(INPUT, 'areamaps', 'mask.tif')
    with rasterio.open(mask_file, 'r') as src:
        transform = src.transform
        mask = src.read(1)
        profile = src.profile
        profile['nodata'] = 1

    geoms = list({'geometry': geom[0], 'properties': {}} for geom in shapes(mask, transform=transform, connectivity=4) if geom[1] == 0)
    gdf = gpd.GeoDataFrame.from_features(geoms).buffer(0)  # Invalid polygons are sometimes returned. Buffer(0) helps solve this issue.
    gdf = gdf.set_crs("EPSG:4326")
    gdf.to_file(mask_file.replace('.tif', '.geojson'), driver='GeoJSON')

if __name__ == '__main__':
    
    # mask_profile, submask_profile = create_mask(450000005, UPSCALE_FACTOR, poor_point=(75.896042,17.370451))  # Bhima
    mask_profile, submask_profile = create_mask(450000005, UPSCALE_FACTOR, poor_point=POOR_POINT)  # Bhimashankar north
    # mask_profile, submask_profile = create_mask(450000005, UPSCALE_FACTOR, poor_point=(73.86242,18.87037))  # Bhimashankar south
    print("creating mask shapefile")
    create_mask_shapefile()
    print("creating cell area map")
    create_cell_area_map(mask_profile)
    print("creating cell area map for submask")
    create_cell_area_map(submask_profile, prefix='sub_')
    print("creating ldd")
    create_ldd(mask_profile)
    print("creating mannings coefficient and channel width")
    get_channel_manning_and_width(mask_profile)
    print("get river length and channel ratio")
    get_river_length_and_channelratio(mask_profile)
    print("geter river slope")
    get_river_slope(mask_profile)
    print("get elevation standard deviation")
    get_elevation_std(mask_profile)
