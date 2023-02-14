import os

import rasterio
import numpy as np

from preconfig import INPUT

def create_cell_area_map(mask_profile: rasterio.profiles.Profile, prefix: str='', write_to_disk=True) -> None:
    """Create cell area map for given rasterio profile. 
    
    Args:
        mask_profile: Rasterio profile of basin mask
        prefix: Filename prefix
    """
    cell_area_path = os.path.join(INPUT, 'areamaps', f'{prefix}cell_area.tif')
    RADIUS_EARTH_EQUATOR = 40075017  # m
    distance_1_degree_latitude = RADIUS_EARTH_EQUATOR / 360

    profile = dict(mask_profile)

    affine = profile['transform']

    lat_idx = np.arange(0, profile['height']).repeat(profile['width']).reshape((profile['height'], profile['width']))
    lat = (lat_idx + 0.5) * affine.e + affine.f
    width_m = distance_1_degree_latitude * np.cos(np.radians(lat)) * abs(affine.a)
    height_m = distance_1_degree_latitude * abs(affine.e)


    profile['dtype'] = np.float32
    area_m = (width_m * height_m).astype(profile['dtype'])

    if write_to_disk:
        with rasterio.open(cell_area_path, 'w', **profile) as cell_area_dst:
            cell_area_dst.write(area_m, 1)

    return area_m