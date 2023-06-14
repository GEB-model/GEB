import math
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from preconfig import INPUT, ORIGINAL_DATA, PREPROCESSING_FOLDER

def fits(n, estimate, farm_sizes, mean, offset, stalled=False):
    target_area = n * mean + offset
    n_farms = (estimate // 1).astype(int)
    estimated_area_int = (n_farms * farm_sizes).sum()

    extra = np.zeros_like(estimate, dtype=n_farms.dtype)
    leftover_estimate = estimate % 1
    for i in range(len(leftover_estimate)):
        v = leftover_estimate[i]
        if v > .5:
            extra[i] += 1
            if i < len(leftover_estimate) - 1:
                leftover_estimate[i+1] -= (1 - v) / farm_sizes[i+1] * farm_sizes[i]
        else:
            if i < len(leftover_estimate) - 1:
                leftover_estimate[i+1] += v / farm_sizes[i+1] * farm_sizes[i]
    
    n_farms = n_farms + extra
    estimated_area_int = (n_farms * farm_sizes).sum()
    
    if estimated_area_int == target_area:
        return n_farms, farm_sizes
    
    elif stalled and abs(estimated_area_int - target_area) < farm_sizes.size:
        while True:
            difference = target_area - estimated_area_int
            if difference > 0:
                for i in range(len(n_farms)):
                    if n_farms[i] > 0:
                        n_farms[i] -= 1
                        if i == n_farms.size - 1:
                            farm_sizes = np.append(farm_sizes, farm_sizes[i] + 1)
                            n_farms = np.append(n_farms, 1)
                        else:
                            n_farms[min(i+difference, len(n_farms)-1)] += 1
                        break
            else:
                for i in range(len(n_farms)-1, -1, -1):
                    if n_farms[i] > 0:
                        n_farms[i] -= 1
                        n_farms[max(i+difference, 0)] += 1
                        break
            estimated_area_int = (n_farms * farm_sizes).sum()
            if estimated_area_int == target_area:
                break
            elif n_farms[0] > 0 and (n_farms[1:] == 0).all():
                n_farms[0] -= 1
                n_farms = np.insert(n_farms, 0, 1)
                farm_sizes = np.insert(farm_sizes, 0, max(farm_sizes[0] + target_area - estimated_area_int, 0))
                break
        return n_farms, farm_sizes

    else:
        return None

def get_farm_distribution(n, x0, x1, mean, offset):
    assert mean >= x0
    assert mean <= x1

    target_area = n * mean + offset
    farm_sizes = np.arange(x0, x1+1)
    n_farm_sizes = farm_sizes.size

    if n == 0:
        n_farms = np.zeros(n_farm_sizes, dtype=np.int32)
        assert target_area == (n_farms * farm_sizes).sum()
    
    elif n == 1:
        farm_sizes = np.array([mean + offset])
        n_farms = np.array([1])
        assert target_area == (n_farms * farm_sizes).sum()
    
    elif mean == x0:
        n_farms = np.zeros(n_farm_sizes, dtype=np.int32)
        n_farms[0] = n
        if offset > 0:
            if offset < n_farms[0]:
                n_farms[0] -= offset
                n_farms[1] += offset
            else:
                raise NotImplementedError
        elif offset < 0:
            n_farms[0] -= 1
            n_farms = np.insert(n_farms, 0, 1)
            farm_sizes = np.insert(farm_sizes, 0, farm_sizes[0] + offset)
            assert (farm_sizes > 0).all()
        assert target_area == (n_farms * farm_sizes).sum()

    elif mean == x1:
        n_farms = np.zeros(n_farm_sizes, dtype=np.int32)
        n_farms[-1] = n
        if offset < 0:
            if n_farms[-1] > -offset:
                n_farms[-1] += offset
                n_farms[-2] -= offset
            else:
                raise NotImplementedError
        elif offset > 0:
            n_farms[-1] -= 1
            n_farms = np.insert(n_farms, 0, 1)
            farm_sizes = np.insert(farm_sizes, 0, farm_sizes[-1] + offset)
            assert (farm_sizes > 0).all()
        assert target_area == (n_farms * farm_sizes).sum()
    
    else:
        growth_factor = 1

        while True:
            estimate = np.zeros(n_farm_sizes, dtype=np.float64)
            estimate[0] = 1
            for i in range(1, estimate.size):
                estimate[i] = estimate[i-1] * growth_factor
            estimate /= (estimate.sum() / n)
            assert (estimate >= 0).all()

            estimated_area = (estimate * farm_sizes).sum()
            
            res = fits(n, estimate, farm_sizes, mean, offset, stalled=True)
            
            if res is not None:
                n_farms, farm_sizes = res
                estimated_area_int = (n_farms * farm_sizes).sum()
                assert estimated_area_int == target_area
                assert (n_farms >= 0).all()
                assert target_area == (n_farms * farm_sizes).sum()
                break
            
            difference = (target_area / estimated_area) ** (1 / (n_farm_sizes - 1))
            growth_factor *= difference

    return n_farms, farm_sizes


def main():
    SIZE_CLASSES = (
        'Below 0.5',
        '0.5-1.0',
        '1.0-2.0',
        '2.0-3.0',
        '3.0-4.0',
        '4.0-5.0',
        '5.0-7.5',
        '7.5-10.0',
        '10.0-20.0',
        '20.0 & ABOVE',
    )

    SIZE_CLASSES_BOUNDARIES = {
        'Below 0.5': (2_500, 5_000),  # farm is assumed to be at least 2500 m2
        '0.5-1.0': (5_000, 10_000),
        '1.0-2.0': (10_000, 20_000),
        '2.0-3.0': (20_000, 30_000),
        '3.0-4.0': (30_000, 40_000),
        '4.0-5.0': (40_000, 50_000),
        '5.0-7.5': (50_000, 75_000),
        '7.5-10.0': (75_000, 100_000),
        '10.0-20.0': (100_000, 200_000),
        '20.0 & ABOVE': (200_000, 400_000),
    }

    with rasterio.open(Path(INPUT, 'landsurface', 'full_region_cultivated_land.tif'), 'r') as src:
        cultivated_land = src.read(1)

    with rasterio.open(Path(INPUT, 'areamaps', 'region_subgrid.tif'), 'r') as src:
        regions_grid = src.read(1)

    with rasterio.open(Path(INPUT, 'areamaps', 'region_cell_area_subgrid.tif'), 'r') as src:
        cell_area = src.read(1)

    regions_shapes = gpd.read_file(Path(INPUT, 'areamaps', 'regions.geojson'))
    avg_farm_size = pd.read_excel(Path(PREPROCESSING_FOLDER, 'census', 'avg_farm_size.xlsx'), index_col=(0, 1, 2))

    all_agents = []
    for _, region in regions_shapes.iterrows():
        region_id = region['region_id']
        region_name = region['sub_dist_1']
        print(f'Processing region {region_id} in {region_name}')

        state, district, tehsil = region["state_name"], region["district_n"], region["sub_dist_1"]

        ipfs = {}
        ipf_area = {}
        for size_class in SIZE_CLASSES:
            ipfs[size_class] = pd.read_csv(
                Path(
                    PREPROCESSING_FOLDER,
                    'agents',
                    'farmers',
                    'ipf',
                    f'{state}_{district}_{tehsil}_{size_class}.csv'),
                low_memory=False
            )

            avg_farm_size_size_class = avg_farm_size.loc[(state, district, tehsil), size_class]
            ipf_n_farms = ipfs[size_class]['weight'].sum()
            if np.isnan(avg_farm_size_size_class):
                assert ipf_n_farms == 0
                avg_farm_size_size_class = 0
            ipf_area[size_class] = avg_farm_size_size_class * ipf_n_farms
            del avg_farm_size_size_class

        cultivated_land_ipf = sum(ipf_area.values())
        average_cell_area_region = cell_area[(regions_grid == region_id) & (cultivated_land == True)].mean()
        region_cultivated_land_area_lu = cell_area[(regions_grid == region_id) & (cultivated_land == True)].sum()
        cultivated_land_region = cultivated_land[regions_grid == region_id]

        total_ipf_weight = 0
        farm_cells_size_class = pd.DataFrame(index=SIZE_CLASSES)
        for size_class in SIZE_CLASSES:
            ipfs[size_class]['weight'] = ipfs[size_class]['weight'] / cultivated_land_ipf * region_cultivated_land_area_lu
            total_size_class_weight = ipfs[size_class]['weight'].sum()
            total_ipf_weight += total_size_class_weight
            if total_size_class_weight == 0:
                farm_cells_size_class.loc[size_class, 'n_cells'] = 0
            else:
                n_cells = total_size_class_weight * avg_farm_size.loc[(state, district, tehsil), size_class] / average_cell_area_region
                assert not np.isnan(n_cells)
                farm_cells_size_class.loc[size_class, 'n_cells'] = n_cells
        
        assert math.isclose(region_cultivated_land_area_lu, farm_cells_size_class.sum() * average_cell_area_region.item())

        n_holdings_per_size_class = pd.Series(0, index=SIZE_CLASSES)
        n_cells_per_size_class = pd.Series(0, index=SIZE_CLASSES)
        for size_class in SIZE_CLASSES:
            n_holdings_per_size_class[size_class] = ipfs[size_class]['weight'].sum()
            if n_holdings_per_size_class[size_class] > 0:
                n_cells_per_size_class.loc[size_class] = n_holdings_per_size_class[size_class] * avg_farm_size.loc[(state, district, tehsil), size_class] / average_cell_area_region
                assert not np.isnan(n_cells_per_size_class.loc[size_class])

        assert math.isclose(cultivated_land_region.sum(), n_cells_per_size_class.sum())
        
        whole_cells_per_size_class = (n_cells_per_size_class // 1).astype(int)
        leftover_cells_per_size_class = n_cells_per_size_class % 1
        whole_cells = whole_cells_per_size_class.sum()
        n_missing_cells = cultivated_land_region.sum() - whole_cells
        assert n_missing_cells <= len(SIZE_CLASSES)

        index = list(zip(leftover_cells_per_size_class.index, leftover_cells_per_size_class % 1))
        n_cells_to_add = sorted(index, key=lambda x: x[1], reverse=True)[:n_missing_cells]
        whole_cells_per_size_class.loc[[p[0] for p in n_cells_to_add]] += 1

        assert whole_cells_per_size_class.sum() == cultivated_land_region.sum()

        region_agents = []
        for size_class in whole_cells_per_size_class.index:
            
            # if no cells for this size class, just continue
            if whole_cells_per_size_class.loc[size_class] == 0:
                continue
            
            min_size_m2, max_size_m2 = SIZE_CLASSES_BOUNDARIES[size_class]

            min_size_cells = int(min_size_m2 / average_cell_area_region)
            min_size_cells = max(min_size_cells, 1)  # farm can never be smaller than one cell
            max_size_cells = int(max_size_m2 / average_cell_area_region) - 1  # otherwise they overlap with next size class
            mean_cells_per_agent = int(avg_farm_size.loc[(state, district, tehsil), size_class] / average_cell_area_region)

            if mean_cells_per_agent < min_size_cells or mean_cells_per_agent > max_size_cells:  # there must be an error in the data, thus assume centred
                mean_cells_per_agent = (min_size_cells + max_size_cells) // 2

            number_of_agents_size_class = round(n_holdings_per_size_class[size_class])
            # if there is agricultural land, but there are no agents rounded down, we assume there is one agent
            if number_of_agents_size_class == 0 and whole_cells_per_size_class[size_class] > 0:
                number_of_agents_size_class = 1

            ipfs[size_class]['adjusted_weight'] = ipfs[size_class]['weight'] / ipfs[size_class]['weight'].sum() * number_of_agents_size_class
            ipfs[size_class]['n'] = (ipfs[size_class]['adjusted_weight'] // 1).astype(int)

            # because of the rounding, there might be a few agents missing. Here we increase n for the rows where the decimal part of the weight is the largest
            n_missing = number_of_agents_size_class - int(ipfs[size_class]['n'].sum())
            assert n_missing >= 0
            index = list(zip(ipfs[size_class].index, ipfs[size_class]['adjusted_weight'] % 1))
            sorted_index = sorted(index, key=lambda x: x[1], reverse=True)
            missing_pop_indices = [p[0] for p in sorted_index[:n_missing]]
            ipfs[size_class].loc[missing_pop_indices, 'n'] += 1
            assert ipfs[size_class]['n'].sum() == number_of_agents_size_class

            population = ipfs[size_class].loc[ipfs[size_class].index.repeat(ipfs[size_class]['n'])]
            population = population.drop(['crops', 'weight', 'adjusted_weight', 'size_class', 'n'], axis=1)            
            offset = whole_cells_per_size_class[size_class] - number_of_agents_size_class * mean_cells_per_agent

            n_farms_size_class, farm_sizes_size_class = get_farm_distribution(number_of_agents_size_class, min_size_cells, max_size_cells, mean_cells_per_agent, offset)
            assert n_farms_size_class.sum() == number_of_agents_size_class
            assert (farm_sizes_size_class > 0).all()
            assert (n_farms_size_class * farm_sizes_size_class).sum() == whole_cells_per_size_class[size_class]
            farm_sizes = farm_sizes_size_class.repeat(n_farms_size_class)
            np.random.shuffle(farm_sizes)
            population['area_n_cells'] = farm_sizes
            region_agents.append(population)

            assert population['area_n_cells'].sum() == whole_cells_per_size_class[size_class]

        region_agents = pd.concat(region_agents, ignore_index=True)
        region_agents['region_id'] = region_id
        all_agents.append(region_agents)

    all_agents = pd.concat(all_agents, ignore_index=True)
    return all_agents

def select_and_rename_columns(agents):
    select_and_rename = {
        'region_id': 'region_id',
        'household size': 'household_size',
        'irrigation_source': 'irrigation_source',
        'Kharif: Crop: Name': 'season #1 crop',
        'Rabi: Crop: Name': 'season #2 crop',
        'Summer: Crop: Name': 'season #3 crop',
        'daily_non_farm_income_family': 'daily_non_farm_income_family',
        'daily_consumption_per_capita': 'daily_consumption_per_capita',
    }

    agents = agents[list(select_and_rename.keys())]
    return agents.rename(columns=select_and_rename)

if __name__ == '__main__':
    all_agents = main()
    all_agents['daily_non_farm_income_family'] = (all_agents['Salaried income Rs'] + all_agents['Business income Rs'] + all_agents['Government benefits Rs'] + all_agents['Income property Rs'] + all_agents['Other income Rs']) / 365.25
    all_agents['daily_consumption_per_capita'] = all_agents['Monthly consumption per capita Rs'] / 12
    all_agents = select_and_rename_columns(all_agents)
    
    folder = Path(PREPROCESSING_FOLDER,  'agents', 'farmers')
    folder.mkdir(parents=True, exist_ok=True)
    
    all_agents.to_csv(Path(folder, 'farmers.csv'), index=True)