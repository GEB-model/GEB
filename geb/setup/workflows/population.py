from numba import njit
import numpy as np


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
