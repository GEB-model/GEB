# -*- coding: utf-8 -*-
import os
import rasterio
from honeybees.library.raster import clip_to_other, upscale
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
import matplotlib.pyplot as plt

from preconfig import ORIGINAL_DATA, INPUT

def get_rivers(threshold: int, template: str):
    """Creates a river map at the resolution of the submask. All cells with at least `threshold` number of upstream cells are considered river. The function loads number of upstream cells from files that end with `_upg` from `DataDrive/GEB/original_data/merit_hydro_03sec`.
    
    Args:
        threshold: cells with more upstream cells than threshold are classified as river.
        
    Return:
        rivers: raster with rivers
    """
    merit_hydro_03sec_folder = os.path.join(ORIGINAL_DATA, 'merit_hydro_03sec')
    upcell_maps = []
    for fn in os.listdir(merit_hydro_03sec_folder):
        fp = os.path.join(merit_hydro_03sec_folder, fn)
        if os.path.splitext(fp)[0].endswith('_upg'):
            src = rasterio.open(fp)
            up_cells_profile_org = src.profile
            upcell_maps.append(rasterio.open(fp))
    
    upcells, upcells_transform = merge(upcell_maps)
    upcells = upcells[0, :, :]
    up_cells_profile_org.update({
        'transform': upcells_transform,
        'width': upcells.shape[1],
        'height': upcells.shape[0],
    })

    rivers = upcells > threshold

    with rasterio.open(template) as submask_src:
        rivers, riverprofile = upscale(rivers, up_cells_profile_org, 2)
        rivers, riverprofile = clip_to_other(rivers, riverprofile, submask_src.profile)

        return rivers
            

def merge_GLC30(template) -> np.ndarray:
    """Merges GLC30 data and scales to the resolution of the submask, using nearest sampling
    
    Returns:
        GLC30: Raster map of the land use type per GLC30 at the resolution of the submask.
    """
    with rasterio.open(template, 'r') as mask_src:
        mask_transform = mask_src.transform
        mask_crs = mask_src.profile['crs']
        mask = mask_src.read(1)

    GLC30 = None
    GLC30_folder = os.path.join(ORIGINAL_DATA, 'GLC30')
    for folder in os.listdir(GLC30_folder):
        folder = os.path.join(GLC30_folder, folder)
        if os.path.isfile(folder):
            continue
        for file in os.listdir(folder):
            file = os.path.join(folder, file)
            if os.path.splitext(file)[1] == '.tif':

                with rasterio.open(file, 'r') as src:
                    source = src.read(1)
                    src_transform = src.transform
                    src_crs = src.profile['crs']
                    src_profile = src.profile

                destination = np.zeros_like(mask, dtype=src_profile['dtype'])

                reproject(
                    source,
                    destination,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=mask_transform,
                    dst_crs=mask_crs,
                    resampling=Resampling.nearest
                )

                if GLC30 is None:
                    GLC30 = destination
                else:
                    GLC30[GLC30 == 0] = destination[GLC30 == 0]

    return GLC30


def create_cwatm_land_use(GLC30: np.ndarray, rivers: np.ndarray, template: str, prefix: str=None) -> None:
    """Converts GLC30 data to 6 land use classes of CWatM. Rivers are also burned in. In addition a map of cultivated land is created.

    Args:
        GLC30: ndarray of GLC30 land use types.
        rivers: ndarray of rivers.
    """
    CWatM = np.full_like(GLC30, -1, dtype=np.int32)
    CWatM[GLC30 == 0] = 1
    CWatM[GLC30 == 10] = 1
    CWatM[GLC30 == 20] = 0
    CWatM[GLC30 == 30] = 1
    CWatM[GLC30 == 40] = 1
    CWatM[GLC30 == 50] = 1
    CWatM[GLC30 == 60] = 5
    CWatM[GLC30 == 70] = 1
    CWatM[GLC30 == 80] = 4
    CWatM[GLC30 == 90] = 1
    CWatM[GLC30 == 100] = 1
    CWatM[GLC30 == 255] = 1

    CWatM[rivers == True] = 5

    with rasterio.open(template) as submask_src:
        submask = submask_src.read(1)
        submask_profile = submask_src.profile
    
    CWatM[submask == -1] = -1

    assert ((CWatM[submask == False] == 0) | (CWatM[submask == False] == 1) | (CWatM[submask == False] == 4) | (CWatM[submask == False] == 5)).all()

    CWatM_land_use_profile = dict(submask_profile)
    CWatM_land_use_profile['dtype'] = CWatM.dtype
    CWatM_land_use_profile['nodata'] = -1
    with rasterio.open(os.path.join(INPUT, 'landsurface', f'{prefix}land_use_classes.tif'), 'w', **CWatM_land_use_profile) as dst:
        dst.write(CWatM, 1)

    cultivated_land = np.zeros_like(GLC30, dtype=np.int8)
    cultivated_land[(GLC30 == 10) & (CWatM == 1)] = True
    cultivated_land_profile = dict(CWatM_land_use_profile)
    cultivated_land_profile['dtype'] = cultivated_land.dtype
    with rasterio.open(os.path.join(INPUT, "landsurface", f"{prefix}cultivated_land.tif"), 'w', **cultivated_land_profile) as dst:
        dst.write(cultivated_land, 1)

    return ((CWatM_land_use_profile, CWatM), (cultivated_land_profile, cultivated_land))

if __name__ == '__main__':
    prefix = "full_tehsils_"
    template = os.path.join(INPUT, 'tehsils.tif')
    rivers = get_rivers(100, template)
    GLC30 = merge_GLC30(template)
    ((CWatM_land_use_profile, CWatM), (cultivated_land_profile, cultivated_land)) = create_cwatm_land_use(GLC30, rivers, template, prefix)

    with rasterio.open(os.path.join(INPUT, 'areamaps', 'submask.tif'), 'r') as submask_src:
        submask_profile = submask_src.profile

    CWatM_cut, CWatM_cut_profile = clip_to_other(CWatM, CWatM_land_use_profile, submask_profile)
    with rasterio.open(os.path.join(INPUT, 'landsurface', 'land_use_classes.tif'), 'w', **CWatM_cut_profile) as dst:
        dst.write(CWatM_cut, 1)

    cultivated_land_cut, cultivated_land_cut_profile = clip_to_other(cultivated_land, cultivated_land_profile, submask_profile)
    with rasterio.open(os.path.join(INPUT, 'landsurface', 'cultivated_land.tif'), 'w', **cultivated_land_cut_profile) as dst:
        dst.write(cultivated_land_cut, 1)
