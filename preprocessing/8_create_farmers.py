# -*- coding: utf-8 -*-
import os
from random import random
import faulthandler
import math
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from numba import njit

from honeybees.library.raster import clip_to_xy_bounds, clip_to_other

from config import INPUT

faulthandler.enable()

@njit(cache=True)
def create_farms(cultivated_land, ids, farm_sizes):
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

    current_farm_counter = 0
    cur_farm_size = 0
    farm_done = False

    farm_id = ids[current_farm_counter]
    farm_size = farm_sizes[current_farm_counter]
    farms = np.where(cultivated_land == True, -1, -2).astype(np.int32)
    ysize, xsize = farms.shape
    for y in range(farms.shape[0]):
        for x in range(farms.shape[1]):
            f = farms[y, x]
            if f == -1:
                assert farm_size > 0
                                
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
                                farms[yf, xf] = farm_id
                                cur_farm_size += 1
                                if cur_farm_size == farm_size:
                                    cur_farm_size = 0
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

                if farm_done:
                    farm_done = False
                    current_farm_counter += 1
                    farm_id = ids[current_farm_counter]
                    farm_size = farm_sizes[current_farm_counter]

    assert np.count_nonzero(farms == -1) == 0
    farms = np.where(farms != -2, farms, -1)
    return farms

def create_farmers(agents: pd.DataFrame, cultivated_land_tehsil: np.ndarray, tehsil: np.ndarray, profile) -> None:
    """
    Takes as input an map of farm sizes of relative distribution of given farm sizes. The first band is the probability that a farm size is of size as given by the farm_size_choices_m2 argumement. The function then writes a raster of farms, where each unique ID is a field owned by a different farmer. Farms only exist on cultivated land. In addition a numpy array of the farmer locations (center of the field) is outputted.
    
    Args:
        farm_size_probabilities: map of indices of farm size
        farm_size_choices_m2: size of 
    """

    assert np.count_nonzero(cultivated_land_tehsil) == agents['farm_size'].sum()

    shuffled_agents = agents.sample(frac=1)

    farms = create_farms(
        cultivated_land_tehsil,
        ids=shuffled_agents.index.to_numpy(),
        farm_sizes=shuffled_agents['farm_size'].to_numpy(),
    ).astype(np.int32)
    unique_farms = np.unique(farms)
    unique_farms = unique_farms[unique_farms != -1]
    if unique_farms.size > 0:
        assert unique_farms.size == unique_farms.max() + 1
    assert shuffled_agents['farm_size'].sum() == np.count_nonzero(farms != -1)
    assert farms.max() + 1 == len(shuffled_agents)
    assert ((farms >= 0) == (cultivated_land_tehsil == 1)).all()
    return farms

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

    farm_sizes = np.arange(x0, x1+1)
    n_farm_sizes = farm_sizes.size

    if n == 0:
        n_farms = np.zeros(n_farm_sizes, dtype=np.int32)
    
    elif mean == x0:
        n_farms = np.zeros(n_farm_sizes, dtype=np.int32)
        n_farms[0] = n
        if offset:
            n_farms[0] -= 1
            n_farms = np.insert(n_farms, 0, 1)
            farm_sizes = np.insert(farm_sizes, 0, farm_sizes[0] + offset)
            assert (farm_sizes > 0).all()

    elif mean == x1:
        n_farms = np.zeros(n_farm_sizes, dtype=np.int32)
        n_farms[-1] = n
        if offset:
            n_farms[-1] -= 1
            n_farms = np.insert(n_farms, 0, 1)
            farm_sizes = np.insert(farm_sizes, 0, farm_sizes[0] + offset)
            assert (farm_sizes > 0).all()
    
    else:
        total_area = n * mean + offset
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
                target_area = n * mean + offset
                estimated_area_int = (n_farms * farm_sizes).sum()
                assert estimated_area_int == target_area
                assert (n_farms >= 0).all()
                break
            
            difference = (total_area / estimated_area) ** (1 / (n_farm_sizes - 1))
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

    tehsils_shapefile = gpd.read_file(os.path.join(INPUT, 'tehsils.geojson')).set_index('ID')

    avg_farm_size = pd.read_excel(os.path.join(INPUT, 'census', 'avg_farm_size.xlsx'), index_col=(0, 1, 2))

    with rasterio.open(os.path.join(INPUT, 'areamaps', 'sub_cell_area.tif'), 'r') as src:
        avg_cell_area = np.mean(src.read(1))


    with rasterio.open(os.path.join(INPUT, "tehsils.tif")) as src_tehsils, rasterio.open(os.path.join(INPUT, 'landsurface', "full_tehsils_cultivated_land.tif"), 'r') as src_cultivated_land:
        profile = src_tehsils.profile
        tehsils = src_tehsils.read(1)
        cultivated_land = src_cultivated_land.read(1)

        assert src_tehsils.profile['transform'] == src_cultivated_land.profile['transform']
        assert src_tehsils.profile['height'] == src_cultivated_land.profile['height']
        assert src_tehsils.profile['width'] == src_cultivated_land.profile['width']

        tehsil_codes = np.unique(tehsils)
        tehsil_codes = list(tehsil_codes[tehsil_codes != -1])
        
        farms = np.full_like(tehsils, -1, dtype=np.int32)
        all_agents = []
        agent_count = 0

        for tehsil_code in tehsil_codes:
            tehsil_name = tehsils_shapefile.loc[tehsil_code]
            state, district, tehsil = tehsil_name["State"], tehsil_name["District"], tehsil_name["Tehsil"]
            print(state, district, tehsil)

            ipls = {}
            ipl_area = {}
            for size_class in SIZE_CLASSES:
                ipls[size_class] = pd.read_csv(
                    os.path.join(
                        INPUT,
                        'agents',
                        'ipl',
                        f'{state}_{district}_{tehsil}_{size_class}.csv'),
                    low_memory=False
                )

                avg_farm_size_size_class = avg_farm_size.loc[(state, district, tehsil), size_class]
                ipl_n_farms = ipls[size_class]['weight'].sum()
                ipl_area[size_class] = avg_farm_size_size_class * ipl_n_farms

            cultivated_land_ipl = sum(ipl_area.values())

            nontehsilx = np.where(~(tehsils != tehsil_code).all(axis=0))[0]
            xmin, xmax = nontehsilx[0], nontehsilx[-1] + 1
            nontehsily = np.where(~(tehsils != tehsil_code).all(axis=1))[0]
            ymin, ymax = nontehsily[0], nontehsily[-1] + 1

            tehsil_profile, tehsils_clipped = clip_to_xy_bounds(src_tehsils, dict(profile), tehsils, xmin, xmax, ymin, ymax)
            tehsil_mask = tehsils_clipped == tehsil_code

            _, cultivated_land_clipped = clip_to_xy_bounds(src_cultivated_land, dict(profile), cultivated_land, xmin, xmax, ymin, ymax)
            cultivated_land_clipped = cultivated_land_clipped.astype(bool)
            cultivated_land_tehsil = cultivated_land_clipped & tehsil_mask

            cultivated_land_map = cultivated_land_tehsil.sum() * avg_cell_area

            farm_cells_size_class = pd.DataFrame(index=SIZE_CLASSES)
            for size_class in SIZE_CLASSES:
                ipls[size_class]['weight'] = ipls[size_class]['weight'] / cultivated_land_ipl * cultivated_land_map
                farm_cells_size_class.loc[size_class, 'n_cells'] = ipls[size_class]['weight'].sum() * avg_farm_size.loc[(state, district, tehsil), size_class] / avg_cell_area

            assert math.isclose(cultivated_land_tehsil.sum(), farm_cells_size_class['n_cells'].sum())
            farm_cells_size_class['whole_cells'] = (farm_cells_size_class['n_cells'] // 1).astype(int)
            farm_cells_size_class['leftover'] = farm_cells_size_class['n_cells'] % 1
            whole_cells = farm_cells_size_class['whole_cells'].sum()
            missing = cultivated_land_tehsil.sum() - whole_cells
            assert missing <= len(SIZE_CLASSES)

            index = list(zip(farm_cells_size_class.index, farm_cells_size_class['n_cells'] % 1))
            add_cells = sorted(index, key=lambda x: x[1], reverse=True)[:missing]
            add_cells = [p[0] for p in add_cells]
            farm_cells_size_class.loc[add_cells, 'whole_cells'] += 1

            assert farm_cells_size_class['whole_cells'].sum() == cultivated_land_tehsil.sum()

            agents = {}
            for size_class in SIZE_CLASSES:
                min_size_m2, max_size_m2 = SIZE_CLASSES_BOUNDARIES[size_class]

                min_size_cells = int(min_size_m2 // avg_cell_area)
                max_size_cells = int(max_size_m2 // avg_cell_area) - 1  # otherwise they overlap with next size class
                mean_cells = int(avg_farm_size_size_class // avg_cell_area)

                if mean_cells < min_size_cells or mean_cells > max_size_cells:  # there must be an error in the data, thus assume centred
                    mean_cells = (min_size_cells + max_size_cells) // 2

                avg_farm_size_size_class = avg_farm_size.loc[(state, district, tehsil), size_class]

                n = round(farm_cells_size_class.at[size_class, 'whole_cells'] / mean_cells)
                offset = farm_cells_size_class.at[size_class, 'whole_cells'] - n * mean_cells

                ipls[size_class]['n'] = (ipls[size_class]['weight'] // 1).astype(int)

                missing = n - int(ipls[size_class]['n'].sum())
                
                index = list(zip(ipls[size_class].index, ipls[size_class]['weight'] % 1))
                missing_pop = sorted(index, key=lambda x: x[1], reverse=True)[:missing]
                missing_pop = [p[0] for p in missing_pop]
                ipls[size_class].loc[missing_pop, 'n'] += 1

                assert ipls[size_class]['n'].sum() == n

                population = ipls[size_class].loc[ipls[size_class].index.repeat(ipls[size_class]['n'])]
                population = population.drop(['crops', 'weight', 'size_class', 'n'], axis=1)

                n_farms_size_class, farm_sizes_size_class = get_farm_distribution(n, min_size_cells, max_size_cells, mean_cells, offset)
                assert (farm_sizes_size_class > 0).all()
                assert (n_farms_size_class * farm_sizes_size_class).sum() == farm_cells_size_class.at[size_class, 'whole_cells']
                farm_sizes = farm_sizes_size_class.repeat(n_farms_size_class)
                np.random.shuffle(farm_sizes)
                population['farm_size'] = farm_sizes
                agents[size_class] = population

                assert population['farm_size'].sum() == farm_cells_size_class.at[size_class, 'whole_cells']

            agents = pd.concat(agents.values(), ignore_index=True)
            
            farms_tehsil = create_farmers(agents, cultivated_land_tehsil, tehsil, tehsil_profile)

            assert farms_tehsil.max() + 1 == len(agents)

            agents.index += agent_count
            farms_tehsil[farms_tehsil != -1] += agent_count

            all_agents.append(agents)
            farms[ymin: ymax, xmin: xmax][tehsil_mask] = farms_tehsil[tehsil_mask]

            agent_count += len(agents)

        all_agents = pd.concat(all_agents, ignore_index=True)
        assert len(all_agents) == all_agents.index.max() + 1 == farms.max() + 1

        all_agents = all_agents.drop(['State code', 'District code', 'PSU: village/neighborhood code', 'Household ID', 'Split household ID', 'Full household ID', 'area owned & cultivated'], axis=1)

        profile = dict(profile)
        profile['dtype'] = farms.dtype
        with rasterio.open(os.path.join(INPUT, 'areamaps', 'submask.tif'), 'r') as submask_src:
            farms_clipped, farms_clipped_profile = clip_to_other(farms, profile, submask_src.profile)

        # remove farms that have been reduced in size because they were cut off
        clipped_farms_sizes = np.unique(farms_clipped, return_counts=True)
        farms_sizes = np.unique(farms[np.isin(farms, clipped_farms_sizes[0])], return_counts=True)
        clipped_farms = clipped_farms_sizes[0][(farms_sizes[1] > clipped_farms_sizes[1])]
        farms_clipped[np.isin(farms_clipped, clipped_farms)] = -1

        unique_farms = np.unique(farms_clipped)
        unique_farms = unique_farms[unique_farms != -1]
        ids = np.arange(unique_farms.size)
        converter = np.full(unique_farms.max() + 2, -1, dtype=np.int32) # +1 because 0 is also a farm, +1 because no farm is -1, set to -1 in next step
        converter[unique_farms] = ids
        converter[-1] = -1
        farms = converter[farms_clipped]

        all_agents = all_agents[np.isin(all_agents.index, unique_farms)]
        all_agents.index = ids

        farm_ids, farm_sizes = np.unique(farms, return_counts=True)
        farm_sizes = farm_sizes[farm_ids != -1]

        assert np.array_equal(farm_sizes, all_agents['farm_size'])

        farmer_folder = os.path.join(INPUT, 'agents')
        attribute_folder = os.path.join(farmer_folder, 'attributes')
        os.makedirs(attribute_folder, exist_ok=True)

        np.save(os.path.join(attribute_folder, 'household_size.npy'), all_agents['household size'])
        np.save(os.path.join(attribute_folder, 'tubewell.npy'), all_agents['Own: Tubewells'])
        np.save(os.path.join(attribute_folder, 'electric pump.npy'), all_agents['Own: Electric Pumps'])
        np.save(os.path.join(attribute_folder, 'diesel pump.npy'), all_agents['Own: Diesel Pumps'])
        np.save(os.path.join(attribute_folder, 'kharif crop.npy'), all_agents['Kharif: Crop: Name'])
        np.save(os.path.join(attribute_folder, 'kharif irrigation.npy'), all_agents['Kharif: Crop: Irrigation'])
        np.save(os.path.join(attribute_folder, 'rabi crop.npy'), all_agents['Rabi: Crop: Name'])
        np.save(os.path.join(attribute_folder, 'rabi irrigation.npy'), all_agents['Rabi: Crop: Irrigation'])
        np.save(os.path.join(attribute_folder, 'summer crop.npy'), all_agents['Summer: Crop: Name'])
        np.save(os.path.join(attribute_folder, 'summer irrigation.npy'), all_agents['Summer: Crop: Irrigation'])

        with rasterio.open(os.path.join(farmer_folder, 'farms.tif'), 'w', **farms_clipped_profile) as dst:
            dst.write(farms, 1)


if __name__ == '__main__':
    main()