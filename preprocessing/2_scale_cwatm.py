# -*- coding: utf-8 -*-
import os
import xarray as xr
import numpy as np
import rasterio

from preconfig import INPUT, ORIGINAL_DATA

def list_files(dir: str) -> list[str]:
    """This function gets a list of all files in all subdirectories of given folder.

    Args:
        dir: path of parent directory

    Returns:
        files: list of all files
    
    """
    files = []
    for root, _, filenames in os.walk(dir):
        for filename in filenames:
            files.append(os.path.join(root.replace(dir + '\\', ''), filename))
    return files

def scale(files: list[str]) -> None:
    """Upscales resolution of files to the resultion of the basin mask
    
    Args:
        files: list of files to convert
    """
    with rasterio.open(os.path.join(INPUT, 'areamaps', 'mask.tif')) as src:
        profile = src.profile
        transform = profile['transform']

    # +.5 * cell size because NetCDF uses center of cell, while transform uses upper left corner
    newlon = np.linspace(transform.c + .5 * transform.a, transform.c + (profile['width'] - .5) * transform.a, profile['width'])
    newlat = np.linspace(transform.f + .5 * transform.e, transform.f + (profile['height'] - .5) * transform.e, profile['height'])

    assert newlon.size == profile['width']
    assert newlat.size == profile['height']

    for f in files:
        print(f"Converting {f}")
        extension = os.path.splitext(f)[1]
        if extension != '.nc':
            continue
        ds = xr.open_dataset(os.path.join(os.path.join(ORIGINAL_DATA, 'input_5min'), f))

        # Interpolate
        data_set_interp = ds.interp(lat=newlat, lon=newlon, method='nearest')
        output_file = os.path.join(INPUT, f)
        output_folder = os.path.dirname(output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        data_set_interp.to_netcdf(output_file, mode='w')

if __name__ == '__main__':
    input_5min = os.path.join(ORIGINAL_DATA, 'input_5min')
    files = list_files(input_5min)
    scale(files)