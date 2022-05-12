# -*- coding: utf-8 -*-
from honeybees.library.raster import pixel_to_coord
import rasterio
import os
import numpy as np
from numba import njit
from random import random

from config import INPUT, ORIGINAL_DATA

import faulthandler
faulthandler.enable()

@njit(cache=True)
def create_farms(cultivated_land: np.ndarray, gt: tuple[float], farm_size_probabilities: np.ndarray, farm_size_choices: np.ndarray, cell_area: np.ndarray):
    """
    Creates random farms considering the farm size distribution.

    Args:
        cultivated_land: map of cultivated land.
        gt: geotransformation of cultivated land map.
        farm_size_probabilities: map of the probabilities for the various farm sizes to exist in a specific cell.
        farm_size_choices: Lower and upper bound of the farm size correlating to the farm size probabilities. First dimension must be equal to number of layers of farm_size_probabilities. Size of the second dimension is 2, to represent the lower and upper bound.
        cell_area: map of cell areas for all cells.

    Returns:
        farms: map of farms. Each unique ID is land owned by a single farmer. Non-cultivated land is represented by -1.
        farmer_coords: 2 dimensional numpy array of farmer locations. First dimension corresponds to the IDs of `farms`, and the second dimension are longitude and latitude.
    """

    assert farm_size_choices.shape[1] == 2
    assert farm_size_choices.shape[0] == farm_size_probabilities.shape[0]
    assert cultivated_land.shape[0] == farm_size_probabilities.shape[1]
    assert cultivated_land.shape[1] == farm_size_probabilities.shape[2]
   
    cur_farm_count = 0
    farms = np.where(cultivated_land == True, -1, -2).astype(np.int32)
    # pre-allocate an array of size 1000. This array is later expanded when more farms are created.
    farmer_coords = np.empty((1000, 2), dtype=np.float32)
    farmer_coords_length = farmer_coords.shape[0]
    farmer_pixels = np.empty((1000, 2), dtype=np.int32)
    ysize, xsize = farms.shape
    for y in range(farms.shape[0]):
        for x in range(farms.shape[1]):
            f = farms[y, x]
            if f == -1:
                
                farm_size_probabilities_cell = farm_size_probabilities[:, y, x]
                max_farm_size_range = farm_size_choices[(farm_size_probabilities_cell.cumsum() > np.random.rand()).argmax()]
                max_farm_size = int(np.round(np.random.uniform(max_farm_size_range[0], max_farm_size_range[1])))
                
                cur_farm_size = 0
                cur_farm_pixel_count = 0
                farm_done = False
                xmin, xmax, ymin, ymax = 1e6, -1e6, 1e6, -1e6
                xlow, xhigh, ylow, yhigh = x, x+1, y, y+1

                xsearch, ysearch = 0, 0
                
                while True:
                    if not np.count_nonzero(farms[ylow:yhigh+1+ysearch, xlow:xhigh+1+xsearch] == -1):
                        break

                    for yf in range(ylow, yhigh+1):
                        for xf in range(xlow, xhigh+1):
                            if xf < xsize and yf < ysize and farms[yf, xf] == -1:
                                if xf > xmax:
                                    xmax = xf
                                if xf < xmin:
                                    xmin = xf
                                if yf > ymax:
                                    ymax = yf
                                if yf < ymin:
                                    ymin = yf
                                farms[yf, xf] = cur_farm_count
                                cur_farm_size += cell_area[yf, xf]
                                cur_farm_pixel_count += 1
                                if cur_farm_size >= max_farm_size:
                                    farm_done = True
                                    break
                        
                        if farm_done is True:
                            break

                    if farm_done is True:
                        break

                    if random() < 0.5:
                        ylow -=1
                        ysearch = 1
                    else:
                        yhigh +=1
                        ysearch = 0

                    if random() < 0.5:
                        xlow -= 1
                        xsearch = 1
                    else:
                        xhigh += 1
                        xsearch = 0

                px, py = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
                farmer_pixels[cur_farm_count] = [py, px]
                lon_min, lat_max = pixel_to_coord(px, py, gt)
                lon_max, lat_min = pixel_to_coord(px+1, py+1, gt)
                farmer_coords[cur_farm_count, 0] = np.random.uniform(lon_min, lon_max)
                farmer_coords[cur_farm_count, 1] = np.random.uniform(lat_min, lat_max)
                

                cur_farm_count += 1

                # procedure to increase the size of the pre-alloated farmer_coords array
                if cur_farm_count >= farmer_coords_length:
                    farmer_coords_new = np.empty((int(farmer_coords_length * 1.5), 2), dtype=farmer_coords.dtype)  # increase size by 50%
                    farmer_coords_new[:farmer_coords_length] = farmer_coords
                    farmer_pixels_new = np.empty((int(farmer_coords_length * 1.5), 2), dtype=farmer_pixels.dtype)  # increase size by 50%
                    farmer_pixels_new[:farmer_coords_length] = farmer_pixels
                    farmer_coords_length = farmer_coords_new.shape[0]
                    farmer_coords = farmer_coords_new
                    farmer_pixels = farmer_pixels_new

    farmer_coords = farmer_coords[:cur_farm_count]
    farmer_pixels = farmer_pixels[:cur_farm_count]
    assert np.count_nonzero(farms == -1) == 0
    farms = np.where(farms != -2, farms, -1)
    return farms, farmer_pixels, farmer_coords

def create_farmers(farm_size_probabilities: np.ndarray, farm_size_choices_m2: np.ndarray) -> None:
    """
    Takes as input an map of farm sizes of relative distribution of given farm sizes. The first band is the probability that a farm size is of size as given by the farm_size_choices_m2 argumement. The function then writes a raster of farms, where each unique ID is a field owned by a different farmer. Farms only exist on cultivated land. In addition a numpy array of the farmer locations (center of the field) is outputted.
    
    Args:
        farm_size_probabilities: map of indices of farm size
        farm_size_choices_m2: size of 
    """
    with rasterio.open(os.path.join(INPUT, 'areamaps', 'sub_cell_area.tif'), 'r') as src_cell_area:
        cell_area = src_cell_area.read(1)
        cell_area_profile = src_cell_area.profile
        gt = src_cell_area.transform.to_gdal()

    with rasterio.open(os.path.join(INPUT, 'landsurface', "cultivated_land.tif"), 'r') as src_cultivated_land:
        cultivated_land = src_cultivated_land.read(1)

    farms, farmer_pixels, farmer_coords = create_farms(cultivated_land, gt, farm_size_probabilities, farm_size_choices_m2, cell_area)
    farms = farms.astype(np.int32)

    assert ((farms >= 0) == (cultivated_land == 1)).all()

    with rasterio.open(os.path.join(INPUT, 'tehsils.tif'), 'r') as src:
        subdistricts = src.read(1)

    farmer_subdistricts = subdistricts[(farmer_pixels[:,0], farmer_pixels[:,1])]
    
    farmer_folder = os.path.join(INPUT, 'agents')
    np.save(os.path.join(farmer_folder, 'farmer_locations.npy'), farmer_coords)
    np.save(os.path.join(farmer_folder, 'farmer_tehsils.npy'), farmer_subdistricts)

    profile = dict(cell_area_profile)
    profile['dtype'] = farms.dtype
    with rasterio.open(os.path.join(farmer_folder, 'farms.tif'), 'w', **profile) as dst:
        dst.write(farms, 1)

if __name__ == '__main__':
    FARM_SIZE_CHOICES_M2 = np.array([
        [0.25, 0.5],
        [0.5, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 7.5],
        [7.5, 10],
        [10, 20],
        [20, 40],
    ]) * 10_000  # Ha to m2
    
    with rasterio.open(os.path.join(INPUT, 'agents', 'farm_size', '2010-11', 'farmsize.tif'), 'r') as src_farm_size:
        FARM_SIZE_PROBABILITIES = src_farm_size.read()

    create_farmers(FARM_SIZE_PROBABILITIES, FARM_SIZE_CHOICES_M2)