# -*- coding: utf-8 -*-
from numba.core.decorators import njit
import numpy as np
from datetime import datetime
import os
import rasterio
import scipy.interpolate
from itertools import combinations
from scipy.ndimage import zoom

from hyve.library.raster import sample_from_map
from hyve.library.mapIO import ArrayReader, NetCDFReader
from random import choices

from config import INPUT, ORIGINAL_DATA

OUTPUT_FOLDER = os.path.join(INPUT, 'agents')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

MIRCA2000_FOLDER = os.path.join(ORIGINAL_DATA, 'MIRCA2000')
FARMER_LOCATIONS = os.path.join(OUTPUT_FOLDER, 'farmer_locations.npy')
ZOOM_FACTOR = 20

def get_crop_calendar(unit_codes: np.ndarray) -> tuple[dict, np.ndarray]:
    """This function parses MIRCA2000 crop calanders from downloaded format to a convenient dictionary. In addition, the unit code map will be mapped to lower values.

    Args:
        unit_codes: Array of unit codes for study area.

    Returns:
        crop_calendar: Dictionary of crop calendars for set of attributes.
        map_unit_codes: Array of reduced unit codes for study area.
    """
    map_unit_codes = np.full(max(unit_codes) + 1, -1, dtype=np.int32)
    map_unit_codes[unit_codes] = np.arange(len(unit_codes))
    crop_calendar_dict = {}
    for kind in ('irrigated', 'rainfed'):
        crop_calendar_dict[kind] = {}
        fn = os.path.join(MIRCA2000_FOLDER, f"cropping_calendar_{kind}.txt")
        with open(fn, 'r') as f:
            for i, line in enumerate(f):
                if i < 8:
                    continue
                line = line.strip()
                if not line:
                    continue
                data = line.split(' ')
                data = [d for d in data if d]
                unit_code = int(data[0])
                if unit_code not in unit_codes:
                    continue
                mapped_unit_code = map_unit_codes[unit_code]
                if mapped_unit_code not in crop_calendar_dict[kind]:
                    crop_calendar_dict[kind][mapped_unit_code] = {}
                crop = int(data[1]) - 1
                cropping_calendar = data[3:]
                if cropping_calendar:
                    if crop not in crop_calendar_dict[kind][mapped_unit_code]:
                        crop_calendar_dict[kind][mapped_unit_code][crop] = []
                    for i in range(len(cropping_calendar) // 3):
                        crop_calendar = cropping_calendar[i * 3: (i + 1) * 3]
                        crop_calendar = [float(crop_calendar[0]), int(crop_calendar[1]), int(crop_calendar[2])]
                        if crop_calendar[0]:
                            crop_calendar_dict[kind][mapped_unit_code][crop].append({
                                "area": crop_calendar[0],
                                "start": crop_calendar[1],
                                "end": crop_calendar[2],
                            })
    
    return crop_calendar_dict, map_unit_codes


def get_farm_sizes():
    with rasterio.open(os.path.join(INPUT, 'agents', 'farms.tif'), 'r') as src:
        farms = src.read(1)

    with rasterio.open(os.path.join(INPUT, 'areamaps', 'sub_cell_area.tif'), 'r') as src:
        cell_area = src.read(1)

    assert cell_area.shape == farms.shape

    cell_area = cell_area[farms != -1]
    farms = farms[farms != -1]

    farm_sizes = np.bincount(farms, weights=cell_area)
    assert (farm_sizes > 0).all()
    return farm_sizes

@njit
def is_groundwater_irrigating(irrigating_farmers, groundwater_irrigation_probabilities, farms_size_ha):
    """
    Below 0.5
    0.5-1.0
    1.0-2.0
    2.0-3.0
    3.0-4.0
    4.0-5.0
    5.0-7.5
    7.5-10.0
    10.0-20.0
    20.0 & ABOVE
    """

    has_well = np.zeros(irrigating_farmers.size, dtype=np.float32)
    for i in range(irrigating_farmers.size):
        is_irrigating = irrigating_farmers[i]
        if is_irrigating:
            farm_size_ha = farms_size_ha[i]
            if farm_size_ha < 0.5:
                has_well[i] = groundwater_irrigation_probabilities[i, 0]
            elif farm_size_ha < 1:
                has_well[i] = groundwater_irrigation_probabilities[i, 1]
            elif farm_size_ha < 2:
                has_well[i] = groundwater_irrigation_probabilities[i, 2]
            elif farm_size_ha < 3:
                has_well[i] = groundwater_irrigation_probabilities[i, 3]
            elif farm_size_ha < 4:
                has_well[i] = groundwater_irrigation_probabilities[i, 4]
            elif farm_size_ha < 5:
                has_well[i] = groundwater_irrigation_probabilities[i, 5]
            elif farm_size_ha < 7.5:
                has_well[i] = groundwater_irrigation_probabilities[i, 6]
            elif farm_size_ha < 10:
                has_well[i] = groundwater_irrigation_probabilities[i, 7]
            elif farm_size_ha < 20:
                has_well[i] = groundwater_irrigation_probabilities[i, 8]
            else:
                has_well[i] = groundwater_irrigation_probabilities[i, 9]
    return has_well

def get_crop_and_irrigation_per_farmer() -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get crop type, irrigation type and MIRCA2000 unit code for each farmer. First, the farmer locations are loaded (as previously generated), then a map of farmer irrigation is loaded to retrieve irrigating farmers. Next, the MIRCA2000 unit codes for each farmer are retrieved. Subsequently, MIRCA2000 crop data is read and linearly interpolated to support the higher resolution grid of the model. Using the farmer locations, a crop is then randomly chosen for each farmer while considering the probablity that a certain crop is growing in that space according to the MIRCA2000 specification. Crop choice, irrigation type and unit codes per farmer are saved to the disk in NumPy arrays.

    Args:
        crop_calendar: Dictionary of crop calendars for set of attributes.
        map_unit_codes: Array of reduced unit codes for study area.

    Returns:
        crop_per_farmer: Chosen crop code per farmer.
        irrigating_farmers: Array with whether farmer is irrigating or not (Irrigating == True).
        farmer_unit_codes: Reduced MIRCA2000 unit code per farmer.
    """
    locations = np.load(FARMER_LOCATIONS)
    n = locations.shape[0]

    with rasterio.open(os.path.join(INPUT, 'areamaps', 'mask.tif')) as src:
        bounds = src.bounds
    bounds = bounds.left, bounds.right, bounds.bottom, bounds.top

    irrigated_land = ArrayReader(
        fp=os.path.join(ORIGINAL_DATA, 'india_irrigated_land', '2010-2011.tif'),
        bounds=bounds
    )
    irrigating_farmers = irrigated_land.sample_coords(locations)

    groundwater_irrigated = ArrayReader(
        fp=os.path.join(INPUT, 'agents', 'irrigation_source', '2010-11', 'irrigation_source.tif'),
        bounds=bounds
    )

    farm_sizes = get_farm_sizes()
    farm_sizes_ha = farm_sizes / 10_000

    crop_per_farmer = np.full(n, -1, dtype=np.int8)

    groundwater_irrigation_probabilities = groundwater_irrigated.sample_coords(locations)
    irrigation_correction_factor = irrigating_farmers.sum() / irrigating_farmers.size
    groundwater_irrigation_probabilities /= irrigation_correction_factor
    groundwater_irrigation_probabilities[groundwater_irrigation_probabilities > 1] = 1

    groundwater_irrigated_farmers = is_groundwater_irrigating(irrigating_farmers, groundwater_irrigation_probabilities, farm_sizes_ha)
    groundwater_irrigated_farmers = np.random.binomial(1, groundwater_irrigated_farmers)

    np.save(os.path.join(OUTPUT_FOLDER, 'is_groundwater_irrigating.npy'), groundwater_irrigated_farmers)

    crop_per_farmer[:] = 0  # all farmers grow wheat
    # just make sure all farmers were assigned a crop
    assert not (crop_per_farmer == -1).any()
    np.save(os.path.join(OUTPUT_FOLDER, 'crop.npy'), crop_per_farmer)
    np.save(os.path.join(OUTPUT_FOLDER, 'irrigating.npy'), irrigating_farmers)

    return crop_per_farmer, irrigating_farmers

if __name__ == '__main__':
    crop_per_farmer, irrigating_farmers = get_crop_and_irrigation_per_farmer()
