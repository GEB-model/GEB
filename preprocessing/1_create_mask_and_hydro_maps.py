import os
import rasterio
import numpy as np
from polygene.library.raster import clip_to_xy_bounds, clip_to_mask, upscale
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
import geopandas as gpd

original_data_folder = 'DataDrive/GEB/original_data'
input_folder = 'DataDrive/GEB/input'

if not os.path.exists(os.path.join(input_folder, 'areamaps')):
    os.makedirs(os.path.join(input_folder, 'areamaps'))
if not os.path.exists(os.path.join(input_folder, 'routing', 'kinematic')):
    os.makedirs(os.path.join(input_folder, 'routing', 'kinematic'))
if not os.path.exists(os.path.join(input_folder, 'landsurface', 'topo')):
    os.makedirs(os.path.join(input_folder, 'landsurface', 'topo'))

    
def create_mask(basin_id: int, upscale_factor: int) -> tuple[rasterio.profiles.Profile, rasterio.profiles.Profile]:
    """This function creates a mask and a submask (using global UPSCALE_FACTOR), and returns rasterio profile for both

    Args:
        basin_id: The basin id of the `merit_hydro_30sec/30sec_basids.tif`
        upscale_factor: The size of the subcell mask relative to the basin mask.

    Returns:
        mask_profile: rasterio profile for mask
        submask_profile: rasterio profile for submask
    """
    mask_fn = os.path.join(input_folder, 'areamaps', 'mask.tif')
    submask_fn = os.path.join(input_folder, 'areamaps', 'submask.tif')
    with rasterio.open(os.path.join(original_data_folder, 'merit_hydro_30sec/30sec_basids.tif'), 'r') as src:
        basins = src.read(1)
        mask = basins != basin_id

        xmin = mask.shape[1] - np.searchsorted(mask.all(axis=0)[::-1], False, side='right')
        xmax = np.searchsorted(mask.all(axis=0), False, side='right')
        ymin = mask.shape[0] - np.searchsorted(mask.all(axis=1)[::-1], False, side='right')
        ymax = np.searchsorted(mask.all(axis=1), False, side='right')

        mask_profile = src.profile
        mask = clip_to_xy_bounds(src, mask_profile, mask, xmin, xmax, ymin, ymax)
        mask_profile['nodata'] = -1

        with rasterio.open(mask_fn, 'w', **mask_profile) as mask_clipped_src:
            mask_clipped_src.write(mask.astype(mask_profile['dtype']), 1)

        submask, submask_profile = upscale(mask, mask_profile, upscale_factor)
        with rasterio.open(submask_fn, 'w', **submask_profile) as submask_clipped_src:
            submask_clipped_src.write(submask.astype(submask_profile['dtype']), 1)

        return mask_profile, submask_profile


def create_cell_area_map(mask_profile: rasterio.profiles.Profile, prefix: str='') -> None:
    """Create cell area map for given rasterio profile. 
    
    Args:
        mask_profile: Rasterio profile of basin mask
        prefix: Filename prefix
    """
    cell_area_path = os.path.join(input_folder, 'areamaps', f'{prefix}cell_area.tif')
    RADIUS_EARTH_EQUATOR = 40075017  # m
    distance_1_degree_latitude = RADIUS_EARTH_EQUATOR / 360

    profile = dict(mask_profile)

    affine = profile['transform']

    lat_idx = np.arange(0, profile['width']).repeat(profile['height']).reshape((profile['width'], profile['height']))
    lat = (lat_idx + 0.5) * affine.e + affine.f
    width_m = distance_1_degree_latitude * np.cos(np.radians(lat)) * abs(affine.a)
    height_m = distance_1_degree_latitude * abs(affine.e)

    area_m = width_m * height_m

    profile['dtype'] = np.float32

    with rasterio.open(cell_area_path, 'w', **profile) as cell_area_dst:
        cell_area_dst.write(area_m.astype(np.float32), 1)

def create_ldd(mask_profile: rasterio.profiles.Profile) -> None:
    """Clip ldd, and convert ArcGIS D8 convention to pcraster LDD convention. 
    
    Args:
        mask_profile: Rasterio profile of basin mask
    """
    output_fn = os.path.join(input_folder, 'routing/kinematic/ldd.tif')
    with rasterio.open(os.path.join(original_data_folder, 'merit_hydro_30sec/30sec_flwdir.tif'), 'r') as src:
        dir_map = src.read(1)
        src_profile = src.profile

        dir_map, profile = clip_to_mask(dir_map, src_profile, mask_profile)

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
    with rasterio.open(os.path.join(original_data_folder, 'merit_hydro_30sec/30sec_elevtn.tif'), 'r') as DEM_src:
        DEM = DEM_src.read(1)
        DEM, DEM_profile = clip_to_mask(DEM, DEM_src.profile, mask_profile)
    
    cell_area_path = os.path.join(input_folder, 'areamaps', 'cell_area.tif')
    with rasterio.open(cell_area_path, 'r') as cell_area_src:
        area_m = cell_area_src.read(1)

    with rasterio.open(os.path.join(original_data_folder, 'merit_hydro_30sec/30sec_uparea.tif'), 'r') as uparea_src:
        upstream_area = uparea_src.read(1) #* 1e6  # km2 to m2
        upstream_area, upstream_profile = clip_to_mask(upstream_area, uparea_src.profile, mask_profile)

        with rasterio.open(os.path.join(input_folder, 'routing/kinematic/upstream_area.tif'), 'w', **upstream_profile) as upstream_dst:
            upstream_dst.write(upstream_area, 1)

    a = (2 * area_m) / upstream_area
    a[a > 1] = 1
    b = DEM / 2000
    b[b > 1] = 1
    manning = 0.025 + 0.015 * a + 0.030 * b

    with rasterio.open(os.path.join(input_folder, 'routing/kinematic/manning.tif'), 'w', **DEM_profile) as manning_dst:
        manning_dst.write(manning.astype(np.float32), 1)

    chanwidth = upstream_area / 500.
    chanwidth[chanwidth < 3] = 3

    with rasterio.open(os.path.join(input_folder, 'routing/kinematic/chanwidth.tif'), 'w', **DEM_profile) as chanwidth_dst:
        chanwidth_dst.write(chanwidth.astype(np.float32), 1)

    upstream_area_no_negative = np.array(upstream_area)
    upstream_area_no_negative[upstream_area_no_negative < 0] = 0
    channel_bankfull = 0.27 * upstream_area_no_negative ** 0.26
    with rasterio.open(os.path.join(input_folder, 'routing/kinematic/chandepth.tif'), 'w', **DEM_profile) as channel_bankfull_dst:
        channel_bankfull_dst.write(channel_bankfull.astype(np.float32), 1)


def get_river_length_and_channelratio(mask_profile: rasterio.profiles.Profile) -> None:
    """Clips channel length to mask, and calculates channel ratio from cell area, channel length and channel width
    
    Args:
        mask_profile: Rasterio profile of basin mask
    """
    with rasterio.open(os.path.join(original_data_folder, 'merit_hydro_30sec/30sec_rivlen_ds.tif'), 'r') as rivlen_src:
        rivlen = rivlen_src.read(1)
        rivlen, profile = clip_to_mask(rivlen, rivlen_src.profile, mask_profile)
        rivlen[rivlen == -9999] = np.nan

        with rasterio.open(os.path.join(input_folder, 'routing/kinematic/chanleng.tif'), 'w', **profile) as chanleng_dst:
            chanleng_dst.write(rivlen, 1)

        with rasterio.open(os.path.join(input_folder, 'areamaps', 'cell_area.tif'), 'r') as cell_area_src:
            cell_area = cell_area_src.read(1)

        with rasterio.open(os.path.join(input_folder, 'routing/kinematic/chanwidth.tif'), 'r') as chanwidth_src:
            chanwidth = chanwidth_src.read(1)
        
        rivlen[rivlen < 0] = 0
        river_area = chanwidth * rivlen
        river_ratio = river_area / cell_area
        river_ratio[river_ratio > 1] = 1
        assert not (river_ratio < 0).any()

        with rasterio.open(os.path.join(input_folder, 'routing/kinematic/chanratio.tif'), 'w', **profile) as river_ratio_dst:
            river_ratio_dst.write(river_ratio, 1)

def get_river_slope(mask_profile: rasterio.profiles.Profile) -> None:
    """Clips riverslope map to mask, and outputs as channelgradient
    
    Args:
        mask_profile: Rasterio profile of basin mask
    """
    with rasterio.open(os.path.join(original_data_folder, 'merit_hydro_30sec/30sec_rivslp.tif'), 'r') as rivslp_src:
        rivslp = rivslp_src.read(1)
        rivslp, profile = clip_to_mask(rivslp, rivslp_src.profile, mask_profile)

        with rasterio.open(os.path.join(input_folder, 'routing/kinematic/changrad.tif'), 'w', **profile) as chanslp_dst:
            chanslp_dst.write(rivslp, 1)

def get_elevation_std(mask_profile: rasterio.profiles.Profile) -> None:
    """Clips 30 arcsecond elevation map to mask, and creates map of standard deviation in elevation map using the 3 arcsecond elevation map.

    All MERIT Hydro elevation files should be in `DataDrive/GEB/original_data/merit_hydro_03sec`.
    
        Args:
            mask_profile: Rasterio profile of basin mask
    """
    with rasterio.open(os.path.join(original_data_folder, 'merit_hydro_30sec/30sec_elevtn.tif'), 'r') as DEM_src:
        DEM_profile = DEM_src.profile
        DEM = DEM_src.read(1)

    merit_hydro_03sec_folder = os.path.join(original_data_folder, 'merit_hydro_03sec')
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
    DEM, DEM_profile = clip_to_mask(DEM, DEM_profile, mask_profile)
    _, high_res_dem_profile_target = upscale(DEM, mask_profile, scaling)
    High_res_DEM, High_res_DEM_profile = clip_to_mask(High_res_DEM, High_res_DEM_profile_org, high_res_dem_profile_target)
    High_res_DEM[High_res_DEM < 0] = 0

    with rasterio.open(os.path.join(input_folder, 'landsurface/topo/subelv.tif'), 'w', **High_res_DEM_profile) as dst:
        dst.write(High_res_DEM, 1)
    
    elevation_per_cell = (High_res_DEM.reshape(High_res_DEM.shape[0] // scaling, scaling, -1, scaling).swapaxes(1, 2).reshape(-1, scaling, scaling))
    elevationSTD = np.std(elevation_per_cell, axis=(1,2)).reshape(DEM.shape)

    with rasterio.open(os.path.join(input_folder, 'landsurface/topo/elvstd.tif'), 'w', **DEM_profile) as dst:
        dst.write(elevationSTD, 1)

def create_mask_shapefile() -> None:
    """Creates a shapefile from the basin mask.
    """
    mask_file = 'DataDrive/GEB/input/areamaps/mask.tif'
    with rasterio.open(mask_file, 'r') as src:
        transform = src.transform
        mask = src.read(1)
        profile = src.profile
        profile['nodata'] = 1

    geoms = list({'geometry': geom[0], 'properties': {}} for geom in shapes(mask, transform=transform) if geom[1] == 0)
    gdf = gpd.GeoDataFrame.from_features(geoms)
    gdf.to_file('DataDrive/GEB/input/areamaps/mask.shp')

if __name__ == '__main__':
    UPSCALE_FACTOR = 20
    mask_profile, submask_profile = create_mask(450000005, UPSCALE_FACTOR)
    create_mask_shapefile()
    create_cell_area_map(mask_profile)
    create_cell_area_map(submask_profile, prefix='sub_')
    create_ldd(mask_profile)
    get_channel_manning_and_width(mask_profile)
    get_river_length_and_channelratio(mask_profile)
    get_river_slope(mask_profile)
    get_elevation_std(mask_profile)
