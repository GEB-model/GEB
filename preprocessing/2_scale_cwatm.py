import os
import xarray as xr
import numpy as np
import rasterio

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
            files.append(os.path.join(root, filename))
    return files

def scale(files: list[str]) -> None:
    """Upscales resolution of files to the resultion of the basin mask
    
    Args:
        files: list of files to convert
    """
    with rasterio.open('DataDrive/GEB/input/areamaps/mask.tif') as src:
        profile = src.profile
        transform = profile['transform']

    # +.5 * cell size because NetCDF uses center of cell, while transform uses upper left corner
    newlon = np.linspace(transform.c + .5 * transform.a, transform.c + (profile['width'] + .5) * transform.a, profile['width'])
    newlat = np.linspace(transform.f + .5 * transform.e, transform.f + (profile['height'] + .5) * transform.e, profile['height'])

    assert newlon.size == profile['width']
    assert newlat.size == profile['height']

    for f in files:
        extension = os.path.splitext(f)[1]
        if extension != '.nc':
            continue
        ds = xr.open_dataset(f)

        # Interpolate
        data_set_interp = ds.interp(lat=newlat, lon=newlon, method='nearest')
        output_file = f.replace('input_5min', 'input')
        output_folder = os.path.dirname(output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        data_set_interp.to_netcdf(output_file, mode='w')

if __name__ == '__main__':
    files = list_files('DataDrive/GEB/input_5min')
    scale(files)