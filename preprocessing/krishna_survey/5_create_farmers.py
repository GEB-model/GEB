import math
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State

from hydromt_geb.workflows import get_farm_distribution

from preconfig import INPUT, PREPROCESSING_FOLDER

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

    SIZE_GROUPS = {
        'Below 0.5': '0-1',
        '0.5-1.0': '0-1',
        '1.0-2.0': '1-2',
        '2.0-3.0': '2-4',
        '3.0-4.0': '2-4',
        '4.0-5.0': '>4',
        '5.0-7.5': '>4',
        '7.5-10.0': '>4',
        '10.0-20.0': '>4',
        '20.0 & ABOVE': '>4',
    }

    SIZE_CLASSES_BOUNDARIES = {
        'Below 0.5': (0, 5_000),  # farm is assumed to be at least 2500 m2
        '0.5-1.0': (5_000, 10_000),
        '1.0-2.0': (10_000, 20_000),
        '2.0-3.0': (20_000, 30_000),
        '3.0-4.0': (30_000, 40_000),
        '4.0-5.0': (40_000, 50_000),
        '5.0-7.5': (50_000, 75_000),
        '7.5-10.0': (75_000, 100_000),
        '10.0-20.0': (100_000, 200_000),
        '20.0 & ABOVE': (200_000, np.inf),
    }

    with rasterio.open(Path(INPUT, 'landsurface', 'full_region_cultivated_land.tif'), 'r') as src:
        cultivated_land = src.read(1)

    with rasterio.open(Path(INPUT, 'areamaps', 'region_subgrid.tif'), 'r') as src:
        regions_grid = src.read(1)

    with rasterio.open(Path(INPUT, 'areamaps', 'region_cell_area_subgrid.tif'), 'r') as src:
        cell_area = src.read(1)

    regions_shapes = gpd.read_file(Path(INPUT, 'areamaps', 'regions.geojson'))
    avg_farm_size = pd.read_excel(Path(PREPROCESSING_FOLDER, 'census', 'avg_farm_size.xlsx'), index_col=(0, 1, 2))
    n_farms = pd.read_excel(Path(PREPROCESSING_FOLDER, 'census', 'n_farms.xlsx'), index_col=(0, 1, 2))

    bayesian_network = BayesianNetwork.load(str(PREPROCESSING_FOLDER / 'bayesian_net' / 'bayesian_net.bif'), filetype='bif')
    bayesian_sampler = BayesianModelSampling(bayesian_network)

    all_agents = []
    for _, region in regions_shapes.iterrows():
        region_id = region['region_id']
        region_name = region['sub_dist_1']
        print(f'Processing region {region_id} in {region_name}')

        state, district, tehsil = region["state_name"], region["district_n"], region["sub_dist_1"]

        n_farms_region = n_farms.loc[(state, district, tehsil)]
        avg_farm_size_region = avg_farm_size.loc[(state, district, tehsil)]

        average_cell_area_region = cell_area[(regions_grid == region_id) & (cultivated_land == True)].mean()
        region_cultivated_land_area_lu = cell_area[(regions_grid == region_id) & (cultivated_land == True)].sum()
        cultivated_land_region = cultivated_land[regions_grid == region_id]
        print('Cultivated land area in region:', cultivated_land_region.sum())

        census_farmed_area = (n_farms_region * avg_farm_size_region).sum()
        correction_factor = region_cultivated_land_area_lu / census_farmed_area

        n_holdings_per_size_class = pd.Series(0, index=SIZE_CLASSES)
        n_cells_per_size_class = pd.Series(0, index=SIZE_CLASSES)
        for size_class in SIZE_CLASSES:
            n_holdings_for_size_class = n_farms_region[size_class] * correction_factor
            if n_holdings_for_size_class > 0:
                n_cells_per_size_class.loc[size_class] = n_holdings_for_size_class * avg_farm_size_region[size_class] / average_cell_area_region
                assert not np.isnan(n_cells_per_size_class.loc[size_class])
            else:
                n_cells_per_size_class.loc[size_class] = 0
            n_holdings_per_size_class[size_class] = n_holdings_for_size_class

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
            assert max_size_m2 >= avg_farm_size.loc[(state, district, tehsil), size_class] >= min_size_m2, "average farm size is outside of the size class boundaries"
            
            # if minimum farm size is 0, assume it is half of the average farm size for that class
            if min_size_m2 == 0:
                min_size_m2 = avg_farm_size.loc[(state, district, tehsil), size_class] / 2
            # if maximum farm size is inf, assume it is twice the average farm size for that class
            if max_size_m2 == np.inf:
                max_size_m2 = avg_farm_size.loc[(state, district, tehsil), size_class] * 2

            min_size_cells = math.floor(min_size_m2 / average_cell_area_region)
            min_size_cells = max(min_size_cells, 1)  # farm can never be smaller than one cell
            max_size_cells = math.ceil(max_size_m2 / average_cell_area_region)  # otherwise they overlap with next size class
            
            mean_cells_per_agent = int(avg_farm_size.loc[(state, district, tehsil), size_class] / average_cell_area_region)

            if mean_cells_per_agent < min_size_cells or mean_cells_per_agent > max_size_cells:  # there must be an error in the data, thus assume centred
                mean_cells_per_agent = (min_size_cells + max_size_cells) // 2

            number_of_agents_size_class = round(n_holdings_per_size_class[size_class])
            # if there is agricultural land, but there are no agents rounded down, we assume there is one agent
            if number_of_agents_size_class == 0 and whole_cells_per_size_class[size_class] > 0:
                number_of_agents_size_class = 1

            offset = whole_cells_per_size_class[size_class] - number_of_agents_size_class * mean_cells_per_agent

            n_farms_size_class, farm_sizes_size_class = get_farm_distribution(number_of_agents_size_class, min_size_cells, max_size_cells, mean_cells_per_agent, offset)
            assert n_farms_size_class.sum() == number_of_agents_size_class
            assert (farm_sizes_size_class > 0).all()
            assert (n_farms_size_class * farm_sizes_size_class).sum() == whole_cells_per_size_class[size_class]
            farm_sizes = farm_sizes_size_class.repeat(n_farms_size_class)
            
            np.random.shuffle(farm_sizes)

            evidence = [
                State(
                    var="How_large_is_the_area_you_grow_crops_on_in_hectares",
                    state=SIZE_GROUPS[size_class]
                ),
            ]
            population = bayesian_sampler.rejection_sample(evidence=evidence, size=number_of_agents_size_class, show_progress=False)
            
            population['area_n_cells'] = farm_sizes
            
            region_agents.append(population)

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
        'Kharif: Crop: Name': 'season_#1_crop',
        'Rabi: Crop: Name': 'season_#2_crop',
        'Summer: Crop: Name': 'season_#3_crop',
        'daily_non_farm_income_family': 'daily_non_farm_income_family',
        'daily_consumption_per_capita': 'daily_consumption_per_capita',
        'area_n_cells': 'area_n_cells',
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