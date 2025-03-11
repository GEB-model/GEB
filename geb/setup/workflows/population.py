import gzip

import numpy as np
import pandas as pd
from numba import njit
import rioxarray


@njit(cache=True)
def generate_locations(
    population: np.ndarray,
    geotransform: tuple[float, float, float, float, float, float],
    mean_household_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function is used to create the locations of the household agent and household sizes. It generates households by sampling from the gridded population map using random household sizes.

    Args:
        population: array constructed from gridded population data.
        can_flood: array representing a global raster. It is a boolean contructed using the 1/100 year flood map of 2080. 1 = inundated, 0 = not inundated.
        x_offset: geotransformation
        y_offset: geotransformation
        x_step: x-dimension of cell size in degrees
        y_step: y-dimension of cell size in degrees
        max_household_size: the maximimum household size used in sampling household agents.

    Returns:
        household_locations: array containing coordinates of each generated household.
        household_sizes: array containing the household size of each generated household.

    """
    x_offset, x_step, _, y_offset, _, y_step = geotransform
    household_locations = np.full((population.sum(), 2), -1, dtype=np.float32)
    household_sizes = np.full(population.sum(), -1, dtype=np.int32)

    total_household_count = 0
    for row in range(0, population.shape[0]):
        for col in range(0, population.shape[1]):
            cell_population = population[row, col]
            ymax = y_offset + row * y_step
            ymin = ymax + y_step

            xmin = x_offset + col * x_step
            xmax = xmin + x_step

            n_households = 0
            while cell_population > 0:
                household_size = min(
                    mean_household_size, cell_population
                )  # cap household size to current population left in cell
                household_sizes[total_household_count + n_households] = household_size
                cell_population -= household_size
                n_households += 1

            household_locations[
                total_household_count : total_household_count + n_households, 0
            ] = np.random.uniform(xmin, xmax, size=n_households)
            household_locations[
                total_household_count : total_household_count + n_households, 1
            ] = np.random.uniform(ymin, ymax, size=n_households)

            total_household_count += n_households

    return (
        household_locations[:total_household_count],
        household_sizes[:total_household_count],
    )


def load_GLOPOP_S(data_catalog, GDL_region):
    # Load GLOPOP-S data. This is a binary file and has no proper loading in hydromt. So we use the data catalog to get the path and format the path with the regions and load it with NumPy
    GLOPOP_S_attribute_names = [
        "HID",
        "RELATE_HEAD",
        "INCOME",
        "WEALTH",
        "RURAL",
        "AGE",
        "GENDER",
        "EDUC",
        "HHTYPE",
        "HHSIZE_CAT",
        "AGRI_OWNERSHIP",
        "FLOOR",
        "WALL",
        "ROOF",
        "SOURCE",
        "GRID_CELL",
    ]

    GLOPOP_S = data_catalog.get_source("GLOPOP-S")
    # get path to GLOPOP grid
    GLOPOP_S_GRID = data_catalog.get_source("GLOPOP-S_grid")

    with gzip.open(GLOPOP_S.path.format(region=GDL_region), "rb") as f:
        GLOPOP_S_region = np.frombuffer(f.read(), dtype=np.int32)

    n_people = GLOPOP_S_region.size // len(GLOPOP_S_attribute_names)
    GLOPOP_S_region = pd.DataFrame(
        np.reshape(
            GLOPOP_S_region, (len(GLOPOP_S_attribute_names), n_people)
        ).transpose(),
        columns=GLOPOP_S_attribute_names,
    )

    # load grid
    GLOPOP_GRID_region = rioxarray.open_rasterio(
        GLOPOP_S_GRID.path.format(region=GDL_region)
    )

    return GLOPOP_S_region, GLOPOP_GRID_region
