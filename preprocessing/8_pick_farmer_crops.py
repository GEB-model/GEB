# -*- coding: utf-8 -*-
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

OUTPUT_FOLDER = os.path.join('DataDrive', 'GEB', 'input', 'agents')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

MIRCA2000_FOLDER = os.path.join('DataDrive', 'GEB', 'original_data', 'MIRCA2000')
FARMER_LOCATIONS = 'DataDrive/GEB/input/agents/farmer_locations.npy'
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

def get_crop_and_irrigation_per_farmer(crop_calendar: dict, map_unit_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    with rasterio.open('DataDrive/GEB/input/areamaps/mask.tif') as src:
        bounds = src.bounds
    bounds = bounds.left, bounds.right, bounds.bottom, bounds.top

    irrigated_land = ArrayReader(
        fp=os.path.join('DataDrive', 'IRRIGATED_LAND_INDIA', '2014-2015.tif'),
        bounds=bounds
    )
    
    crop_per_farmer = np.full(n, -1, dtype=np.int8)
    irrigating_farmers = irrigated_land.sample_coords(locations)

    MIRCA2000_unit_code = ArrayReader(
        fp=os.path.join(MIRCA2000_FOLDER, 'unit_code.asc'),
        bounds=bounds
    )
    MIRCA2000_unit_array = map_unit_codes[MIRCA2000_unit_code.get_data_array().repeat(ZOOM_FACTOR, axis=0).repeat(ZOOM_FACTOR, axis=1)]

    crop_data = {}
    for kind in ('rainfed', 'irrigated'):
        crop_data[kind] = {}
        for crop in range(26):
            crop_data[kind][crop] = NetCDFReader(
                fp=os.path.join(MIRCA2000_FOLDER, f'{kind}_{"%02d" % crop}.nc'),
                varname="cropland",
                bounds=bounds
            )

    ysize = crop_data['rainfed'][0].rowslice.stop - crop_data['rainfed'][0].rowslice.start
    xsize = crop_data['rainfed'][0].colslice.stop - crop_data['rainfed'][0].colslice.start
    gt = crop_data['rainfed'][0].gt
    gt = list(gt)
    gt[1] = gt[1] / ZOOM_FACTOR
    gt[5] = gt[5] / ZOOM_FACTOR
    gt = tuple(gt)

    all_crops = set(range(26))
    for kind in ('rainfed', 'irrigated'):

        excluded_crops = {}
        for unit_code, crops in crop_calendar[kind].items():
            excluded_crops[unit_code] = list(all_crops - set(crops.keys()))

        if kind == 'rainfed':
            farmers_kind = np.where(irrigating_farmers != 1)[0]
        else:
            farmers_kind = np.where(irrigating_farmers == 1)[0]

        growing_areas = np.zeros((26, ysize, xsize), dtype=np.float32)
        for crop in range(26):
            for month in range(1, 13):
                growing_area_crop = np.maximum(
                    crop_data[kind][crop].get_data_array(datetime(2000, month, 1)),
                    growing_areas[crop]
                )
                growing_areas[crop] = growing_area_crop

        # interpolate nan values
        mask = ~(np.sum(growing_areas, axis=0) == 0)

        xx, yy = np.meshgrid(np.arange(growing_areas.shape[2]), np.arange(growing_areas.shape[1]))
        xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

        growing_areas_list = []
        for i in range(26):
            growing_areas_crop = scipy.interpolate.NearestNDInterpolator(xym, np.ravel(growing_areas[i, ...][mask]))(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
            growing_areas_crop[growing_areas_crop < 0] = 0
            growing_areas_list.append(growing_areas_crop)

        growing_areas = np.stack(growing_areas_list)
        growing_areas = growing_areas / growing_areas.sum(axis=0)
        # growing_areas = zoom(growing_areas, (1, ZOOM_FACTOR, ZOOM_FACTOR), order=1, grid_mode=False)
        growing_areas_zoomed = []
        for crop in range(26):
            growing_areas_zoomed.append(
                zoom(growing_areas[crop], ZOOM_FACTOR, order=1, grid_mode=False)
            )
        growing_areas = np.stack(growing_areas_zoomed)
        
        def filter_crops(MIRCA2000_unit_array, excluded_crops, growing_areas):
            ysize, xsize = MIRCA2000_unit_array.shape
            for y in range(ysize):
                for x in range(xsize):
                    unit_code = MIRCA2000_unit_array[y, x]
                    excluded_crops_cell = excluded_crops[unit_code]
                    growing_areas[:, y, x][excluded_crops_cell] = 0
            return growing_areas
        
        growing_areas = filter_crops(MIRCA2000_unit_array, excluded_crops, growing_areas)
        
        growing_areas = growing_areas / growing_areas.sum(axis=0)
        
        # sample crop options
        planting_schemes = sample_from_map(growing_areas, locations[farmers_kind], gt)

        # randomly select crop
        crop_choice = (planting_schemes.cumsum(1) > np.random.rand(planting_schemes.shape[0])[:,None]).argmax(1)

        # assign crop choice to self
        crop_per_farmer[farmers_kind] = crop_choice

    # just make sure all farmers were assigned a crop
    assert not (crop_per_farmer == -1).any()
    np.save(os.path.join(OUTPUT_FOLDER, 'crop.npy'), crop_per_farmer)
    np.save(os.path.join(OUTPUT_FOLDER, 'irrigating.npy'), irrigating_farmers)

    farmer_unit_codes = map_unit_codes[MIRCA2000_unit_code.sample_coords(locations)]
    np.save(os.path.join(OUTPUT_FOLDER, 'farmer_unit_codes.npy'), farmer_unit_codes)
    
    return crop_per_farmer, irrigating_farmers, farmer_unit_codes

def overlap(r1: set[int], r2: set[int]) -> bool:
    """Check whether 2 sets overlap. Used to check whether 2 crops overlap in growth period.
    
    Args:
        r1: Set of growing months for first crop.
        r2: Set of growing months for second crop.

    Returns:
        overlap: True if crop growth periods overlap, False otherwise.
    """
    return len(r1 | r2) < len(r1) + len(r2)

def get_range_months(potential_crops: list[dict]) -> list[set]:
    """Gets growing months for list of crops.

    Args:
        potential_crops: List of crops, specifying start and end of growing period in dict as follows: [{'start': ..., 'end': ...}, ..., {...}]

    Returns:
        growth_range_months: List of crops (same as function input) and their growing months as a set.
    """
    growth_range_months = []
    for crop in potential_crops:
        start, end = crop['start'], crop['end']
        if start > end:
            end += 12
            months = np.array(range(start, end+1)) % 12
            months[months == 0] = 12
        else:
            months = np.array(range(start, end+1))
        growth_range_months.append(set(months))
    return growth_range_months

def get_planting_schemes(crop_calendars: dict, crop_per_farmer: np.ndarray, farmer_unit_codes: np.ndarray) -> np.ndarray:
    """Multiple planting schemes (e.g., single, multi-cropping) exist for some crops in MIRCA2000. This function is used to create the cropping options. Results are saved as a 6-dimensional NumPy arrays, specifying all the different options. While not very easy to work with, it allows the model to be much better than for example with a JSON-file.

    Args:
        crop_calendars: Dictionary of crop calendars for set of attributes.
        crop_per_farmer: Chosen crop code per farmer.
        farmer_unit_codes: Reduced MIRCA2000 unit code per farmer.

    Returns:
        planting_schemes: A 6-dimensional array which contains the planing schemes.
            1. Whether the crop is irrigated or not.
            2. The unit code for the crop.
            3. The crop.
            4. The planting scheme (e.g., single or double cropping).
            5. If multicropping, the various crop stages.
            6. The planting month and the harvest month.
    """
    planting_schemes = np.full((2, farmer_unit_codes.max() + 1, 26, 3, 2, 3), -1, dtype=np.int32)  # is_irrigating, unit_code, crop, crop_selection, max_multicrop, start, end, weight
    unique_unit_codes = np.unique(farmer_unit_codes)
    unique_crops = np.unique(crop_per_farmer)
    for is_irrigating, is_irrigating_name in ((0, 'rainfed'), (1, 'irrigated')):
        for unit_code in unique_unit_codes:
            for crop in unique_crops:
                if crop not in crop_calendars[is_irrigating_name][unit_code]:
                    continue
                potential_crops = crop_calendars[is_irrigating_name][unit_code][crop]
                if not potential_crops:
                    raise ValueError
                elif len(potential_crops) == 1:
                    planting_schemes[is_irrigating, unit_code, crop, 0, 0] = [potential_crops[0]['start'], potential_crops[0]['end'], 1]
                elif len(potential_crops) == 2:
                    ranges = get_range_months(potential_crops)
                    if overlap(ranges[0], ranges[1]):  # no multicropping
                        for i in range(2):
                            planting_schemes[is_irrigating, unit_code, crop, i, 0] = [potential_crops[i]['start'], potential_crops[i]['end'], potential_crops[i]['area']]
                    else: # multicropping
                        assert potential_crops[0]['area'] == potential_crops[1]['area']
                        planting_schemes[is_irrigating, unit_code, crop, 0, 0] = [potential_crops[0]['start'], potential_crops[0]['end'], 1]
                        planting_schemes[is_irrigating, unit_code, crop, 0, 1] = [potential_crops[1]['start'], potential_crops[1]['end'], 1]
                elif len(potential_crops) == 3:
                    ranges = get_range_months(potential_crops)
                    crops_overlap = all([overlap(left, right) for left, right in combinations(ranges, 2)])
                    if crops_overlap:  # no multicropping
                        for i in range(3):
                            planting_schemes[is_irrigating, unit_code, crop, i, 0] = [potential_crops[i]['start'], potential_crops[i]['end'], potential_crops[i]['area']]
                    else:
                        raise NotImplementedError('multicropping for 3 crops is not implemented')
                else:
                    raise NotImplementedError('Only max of 3 crops is currently implemented')

    np.save(os.path.join(OUTPUT_FOLDER, 'planting_schemes.npy'), planting_schemes)
    return planting_schemes

def pick_planting_scheme(planting_schemes: np.ndarray, irrigating_farmers: np.ndarray, crop_per_farmer: np.ndarray, farmer_unit_codes: np.ndarray) -> None:
    """Here the farmers pick one of the planting schemes relative to the growing area of their crop and irrigation type in the area. Saves the planting scheme to a numpy-array.

    Args:
        planting_schemes: A 6-dimensional array which contains the planing schemes.
            1. Whether the crop is irrigated or not.
            2. The unit code for the crop.
            3. The crop.
            4. The planting scheme (e.g., single or double cropping).
            5. If multicropping, the various crop stages.
            6. The planting month and the harvest month.
        irrigating_farmers: Array with whether farmer is irrigating or not (Irrigating == True).
        crop_per_farmer: Chosen crop code per farmer.
    """
    planting_scheme = np.full_like(crop_per_farmer, -1)
    for i in range(irrigating_farmers.size):
        is_irrigating = irrigating_farmers[i]
        unit_code = farmer_unit_codes[i]
        crop = crop_per_farmer[i]
        crop_options = planting_schemes[is_irrigating][unit_code][crop]
        n_options = (crop_options[:, 0, 0] != -1).sum()
        if n_options == 1:
            planting_scheme[i] = 0
        else:
            crop_options[:n_options, 0, 2]
            planting_scheme[i] = choices(range(n_options), weights=crop_options[:n_options, 0, 2])[0]

    np.save(os.path.join(OUTPUT_FOLDER, 'planting_scheme.npy'), planting_scheme)

if __name__ == '__main__':
    crop_calendar, map_unit_codes = get_crop_calendar(unit_codes=[
        356001, 356002, 356003, 356004, 356005,
        356006, 356007, 356008, 356009, 356010,
        356011, 356012, 356013, 356014, 356015,
        356016, 356017, 356018, 356019, 356020,
        356021, 356022, 356023, 356024, 356025,
        356026, 356027, 356028, 356029, 356030,
        356031, 356032, 356033, 356034, 356035
    ])
    crop_per_farmer, irrigating_farmers, farmer_unit_codes = get_crop_and_irrigation_per_farmer(crop_calendar, map_unit_codes)
    planting_schemes = get_planting_schemes(crop_calendar, crop_per_farmer, farmer_unit_codes)
    pick_planting_scheme(planting_schemes, irrigating_farmers, crop_per_farmer, farmer_unit_codes)
