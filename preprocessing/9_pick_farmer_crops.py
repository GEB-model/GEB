# -*- coding: utf-8 -*-
import os
import numpy as np

import rasterio
import geopandas as gpd

from config import INPUT, ORIGINAL_DATA

OUTPUT_FOLDER = os.path.join(INPUT, 'agents')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

FARMER_LOCATIONS = os.path.join(OUTPUT_FOLDER, 'farmer_locations.npy')

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

    with rasterio.open(os.path.join(INPUT, 'areamaps', 'mask.tif')) as src:
        bounds = src.bounds
    bounds = bounds.left, bounds.right, bounds.bottom, bounds.top

    census_irrigated_holdings = gpd.read_file(os.path.join(ORIGINAL_DATA, 'census', 'output', 'irrigation_source_2010-11.geojson'))
    farmer_locations = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=locations[:,0], y=locations[:,1], crs=census_irrigated_holdings.crs))
    tehsil_indices = gpd.sjoin(farmer_locations, census_irrigated_holdings[['geometry']], op='within', how='left')['index_right'].to_numpy()
    assert tehsil_indices.size == len(farmer_locations)

    farm_size_classes = (
        "Below 0.5",
        "0.5-1.0",
        "1.0-2.0",
        "2.0-3.0",
        "3.0-4.0",
        "4.0-5.0",
        "5.0-7.5",
        "7.5-10.0",
        "10.0-20.0",
        "20.0 & ABOVE",
    )

    farm_size_class_bounds = np.array([.5, 1, 2, 3, 4, 5, 7.4, 10, 20, np.inf])
    assert len(farm_size_classes) == farm_size_class_bounds.size
    
    irrigation_probabilities = np.full((len(farm_size_classes), len(census_irrigated_holdings), 3), -1, dtype=np.float32)
    for i, farm_size_class in enumerate(farm_size_classes):
        total_holdings = census_irrigated_holdings[f'{farm_size_class}_total_holdings']
        canal_holdings = census_irrigated_holdings[f'{farm_size_class}_canals_holdings']
        irrigation_probabilities[i, :, 1] = canal_holdings / total_holdings
        well_holdings = census_irrigated_holdings[f'{farm_size_class}_well_holdings'] + census_irrigated_holdings[f'{farm_size_class}_tubewell_holdings']
        irrigation_probabilities[i, :, 2] = well_holdings / total_holdings
        # tank_holdings = census_irrigated_holdings[f'{farm_size_class}_tank_holdings']
        # other_irrigation_sources_holdings = census_irrigated_holdings[f'{farm_size_class}_other_holdings']
        irrigation_probabilities[i, :, 0] = 1 - np.sum(irrigation_probabilities[i, :, 1:], axis=1)  # no irrigation is all others minus 1

    
    irrigation_probabilities[np.isnan(irrigation_probabilities)] = 0
    assert (irrigation_probabilities != -1).all()

    farm_sizes = get_farm_sizes()
    farm_sizes_ha = farm_sizes / 10_000

    canal_irrigated = np.zeros(farmer_locations.shape[0], dtype=bool)
    well_irrigated = np.zeros(farmer_locations.shape[0], dtype=bool)

    choices = np.array([0, 1, 2])
    for i in range(farmer_locations.shape[0]):
        tehsil_idx = tehsil_indices[i]
        farm_size_ha = farm_sizes_ha[i]
        farm_size_class = np.searchsorted(farm_size_class_bounds, farm_size_ha, side='right')
        irrigation_probabilities_farmer = irrigation_probabilities[farm_size_class, tehsil_idx]
        irrigation_type = np.random.choice(choices, p=irrigation_probabilities_farmer)
        if irrigation_type == 1:
            canal_irrigated[i] = True
        elif irrigation_type == 2:
            well_irrigated[i] = True

    crops = np.full(locations.shape[0], 2, dtype=np.int32)

    np.save(os.path.join(OUTPUT_FOLDER, 'crop.npy'), crops)
    np.save(os.path.join(OUTPUT_FOLDER, 'canal_irrigated.npy'), canal_irrigated)
    np.save(os.path.join(OUTPUT_FOLDER, 'well_irrigated.npy'), well_irrigated)
    return


if __name__ == '__main__':
    get_crop_and_irrigation_per_farmer()
