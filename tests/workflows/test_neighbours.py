"""Tests for neighbor finding functionality."""

from typing import Any

import matplotlib.patches as mpatches
import numpy as np
import pytest
from numba import njit

from geb.workflows import geohash
from geb.workflows.neighbors import find_neighbors


@pytest.fixture(params=[1000])
def n_locations(benchmark: Any, request: Any) -> int:
    """Fixture providing number of locations for testing.

    Returns:
        Number of locations to use in tests.
    """
    return request.param


@pytest.fixture(params=[29, 31, 33])
def bits(benchmark: Any, request: Any) -> int:
    """Fixture providing geohash precision bits for testing.

    Returns:
        Geohash precision bits for testing.
    """
    benchmark.group = "%s bits" % request.param
    return request.param


def test_neighbors_speed(
    benchmark: Any, n_locations: int, bits: int, pytestconfig: Any
) -> None:
    """Test the speed of finding neighbors using benchmark.

    Args:
        benchmark: Pytest benchmark fixture for performance testing.
        n_locations: Number of locations to generate for testing.
        bits: Geohash precision bits.
        pytestconfig: Pytest configuration object.
    """
    radius = 5000
    n_neighbors = 5

    locations = np.c_[
        np.random.uniform(33, 34, n_locations), np.random.uniform(-6, -5, n_locations)
    ]
    search_ids = np.random.choice(np.arange(0, n_locations), 50, replace=False)

    benchmark.pedantic(
        find_neighbors,
        args=(locations, radius, n_neighbors, bits),
        kwargs={"search_ids": search_ids},
        warmup_rounds=1,
        iterations=10,
        rounds=10,
    )


# @pytest.fixture(params=[
#     ('Amsterdam Zuid', (4.816448743767209, 52.32154672402562, 4.914774420445526, 52.36205151188381)),
#     ('Amsterdam', (4.731224060058707, 52.27829742431646, 5.071867942810172, 52.43067169189453)),
#     # ('Noord Holland', (4.4941659, 52.16555405, 5.32160902, 53.18319321)),
#     # ('Nederland', (3.36078191, 50.72349167, 7.22709513, 53.5545845)),
#     # ('Europe', (-31.289030075073242, 34.93054962158203, 68.93136596679688, 81.8519287109375))
# ])
# def real_world_area(benchmark, request, pytestconfig):
#     if not pytestconfig.getoption('extended'):
#         pytest.skip("Only runs in extended mode")
#     gpw_v4 = 'files/population/gpw_v4_population_count_30_sec.tif'
#     if not os.path.exists(gpw_v4):
#         raise FileNotFoundError(
#             f'{gpw_v4} not found \n solution: download gridded population of the world and place in tests/{gpw_v4} \n also see tests/files/tree.txt'
#         )

#     ds = gdal.Open(gpw_v4)
#     gt = ds.GetGeoTransform()

#     coordinates = request.param[1]
#     ul = coord_to_pixel(coordinates[:2], gt)
#     lr = coord_to_pixel(coordinates[2:], gt)

#     band = ds.GetRasterBand(1)

#     population = band.ReadAsArray(ul[0], lr[1], lr[0] - ul[0] + 1, ul[1] - lr[1] + 1)
#     population[population<0] = 0
#     population = population.astype(np.int32)

#     return generate_locations(population, gt[0] + ul[0] * gt[1], gt[3] + lr[1] * gt[5], gt[1], -gt[5]), population.sum()


@njit(cache=True)
def generate_locations(
    population: np.ndarray,
    x_offset: float,
    y_offset: float,
    x_step: float,
    y_step: float,
) -> np.ndarray:
    """Generate agent locations based on population grid.

    Args:
        population: 2D array of population counts per grid cell.
        x_offset: Minimum x coordinate of the grid.
        y_offset: Minimum y coordinate of the grid.
        x_step: Step size in x direction.
        y_step: Step size in y direction.

    Returns:
        Array of agent locations as (x, y) coordinates.
    """
    agent_locations = np.empty((population.sum(), 2), dtype=np.float32)
    count = 0
    for row in range(0, population.shape[0]):
        for col in range(0, population.shape[1]):
            cell_population = population[row, col]
            if cell_population != 0:
                ymax = y_offset + row * y_step
                ymin = ymax + y_step

                xmin = x_offset + col * x_step
                xmax = xmin + x_step

                agent_locations[count : count + cell_population, 0] = np.random.uniform(
                    xmin, xmax, size=cell_population
                )
                agent_locations[count : count + cell_population, 1] = np.random.uniform(
                    ymin, ymax, size=cell_population
                )

                count += cell_population

    return agent_locations


@pytest.fixture(
    params=[
        (
            "Amsterdam Zuid",
            (
                4.816448743767209,
                52.32154672402562,
                4.914774420445526,
                52.36205151188381,
            ),
        ),
        (
            "Amsterdam",
            (
                4.731224060058707,
                52.27829742431646,
                5.071867942810172,
                52.43067169189453,
            ),
        ),
        # ('Noord Holland', (4.4941659, 52.16555405, 5.32160902, 53.18319321)),
        # ('Nederland', (3.36078191, 50.72349167, 7.22709513, 53.5545845)),
        # ('Europe', (-31.289030075073242, 34.93054962158203, 68.93136596679688, 81.8519287109375))
    ]
)
def test_neighbors_real_world_speed(
    benchmark: Any,
    real_world_area: tuple[np.ndarray, int],
    bits: int,
    pytestconfig: Any,
) -> None:
    """Test neighbor finding speed with real-world population data.

    Args:
        benchmark: Pytest benchmark fixture for performance testing.
        real_world_area: Tuple of (locations, population_count) from real world data.
        bits: Geohash precision bits.
        pytestconfig: Pytest configuration object.
    """
    if not pytestconfig.getoption("extended"):
        pytest.skip("Only runs in extended mode")

    locations, population_count = real_world_area

    radius = 5000
    n_neighbors = 5

    find_agent_ids = np.arange(0, locations.shape[0])[::100]

    benchmark.pedantic(
        find_neighbors,
        args=(
            locations,
            radius,
            n_neighbors,
            bits,
            -180,
            180,
            -90,
            90,
            "longlat",
            find_agent_ids,
            None,
        ),
        warmup_rounds=1,
        iterations=1,
        rounds=1,
    )
    benchmark.extra_info["population_count"] = int(population_count)
    benchmark.extra_info["radius"] = radius
    benchmark.extra_info["n_neighbors"] = n_neighbors


# def test_neighbors_real_world_speed_mesa(benchmark, real_world_area, bits, pytestconfig):
#     if not pytestconfig.getoption('extended'):
#         pytest.skip("Only runs in extended mode")

#     if not pytestconfig.getoption('compare'):
#         pytest.skip("Only runs in compare mode")

#     locations = real_world_area

#     radius = 500
#     n_neighbors = 5

#     # print(locations.shape[0])
#     # find_agent_ids = np.random.choice(np.arange(0, locations.shape[0]), 50, replace=False)
#     find_agent_ids = np.arange(0, locations.shape[0])

#     benchmark.pedantic(find_neighbors, args=(locations, find_agent_ids, radius, n_neighbors, bits), warmup_rounds=1, iterations=5, rounds=1)


def test_find_neighbors_coordinates_plot(plt: Any) -> None:
    """Test and visualize neighbor finding in coordinate space.

    Args:
        plt: Matplotlib pyplot fixture for plotting.
    """
    n_locations = 1000
    radius = 5000
    n_neighbor = 5
    bits = 29

    locations = np.c_[
        np.random.uniform(33, 33.5, n_locations),
        np.random.uniform(-6, -5.5, n_locations),
    ]
    search_ids = np.random.choice(np.arange(0, n_locations), 50, replace=False)

    counts = np.zeros(n_locations, dtype=np.int32)
    for i in range(1000):
        neighbors = find_neighbors(
            locations, radius, n_neighbor, bits, search_ids=search_ids
        )
        for neighbor in neighbors.reshape(neighbors.size):
            if neighbor != 4294967295:
                counts[neighbor] += 1

    window_width, window_height = geohash.window(bits)

    _, ax = plt.subplots(figsize=(10, 10))

    for agent, agent_neighbors in zip(search_ids, neighbors):
        agent_neighbors = agent_neighbors[agent_neighbors != 4294967295]
        assert np.unique(agent_neighbors).size == agent_neighbors.size
        geohash_coord = geohash.decode(
            geohash.encode_precision(*locations[agent], bits), bits
        )
        shifts = geohash.get_shifts(*geohash_coord, radius, bits)
        neighbor_geohashes = geohash.shift_multiple(
            geohash.encode_precision(*locations[agent], bits), bits, shifts
        )
        neighbor_geohashes = np.sort(neighbor_geohashes)
        for j, neighbor_geohash in enumerate(neighbor_geohashes):
            patch = mpatches.Rectangle(
                geohash.decode(neighbor_geohash, bits),
                window_width,
                window_height,
                facecolor="orange",
                edgecolor="black",
                alpha=j / neighbor_geohashes.size * 0.5 + 0.1,
            )
            ax.add_patch(patch)

        loc = locations[agent]
        for neighbor in agent_neighbors:
            assert agent != neighbor
            neighborloc = locations[neighbor]
            ax.plot([loc[0], neighborloc[0]], [loc[1], neighborloc[1]])

    for x, y, count in zip(locations[:, 0], locations[:, 1], counts):
        if count != 0:
            ax.annotate(str(count), (x, y))

    ax.scatter(locations[:, 0], locations[:, 1], s=4)


def test_find_neighbors_meters_plot(plt: Any) -> None:
    """Test and visualize neighbor finding in meter-based coordinate space.

    Args:
        plt: Matplotlib pyplot fixture for plotting.
    """
    n_locations = 10000
    radius = 400
    n_neighbor = 5
    bits = 18

    minx = 0
    maxx = 20000
    miny = 0
    maxy = 40000

    locations = np.c_[
        np.random.uniform(0, 20000, n_locations),
        np.random.uniform(0, 40000, n_locations),
    ]
    search_ids = np.random.choice(np.arange(0, n_locations), 50, replace=False)

    counts = np.zeros(n_locations, dtype=np.int32)
    for _ in range(100):
        neighbors = find_neighbors(
            locations,
            radius,
            n_neighbor,
            bits,
            minx=minx,
            maxx=maxx,
            miny=miny,
            maxy=maxy,
            grid="orthogonal",
            search_ids=search_ids,
        )
        for neighbor in neighbors.reshape(neighbors.size):
            if neighbor != 4294967295:
                counts[neighbor] += 1

    window_width, window_height = geohash.window(bits, minx, maxx, miny, maxy)

    _, ax = plt.subplots(figsize=(20, 20))

    for agent, agent_neighbors in zip(search_ids, neighbors):
        agent_neighbors = agent_neighbors[agent_neighbors != 4294967295]
        assert np.unique(agent_neighbors).size == agent_neighbors.size
        geohash_coord = geohash.decode(
            geohash.encode_precision(
                *locations[agent], bits, minx=minx, maxx=maxx, miny=miny, maxy=maxy
            ),
            bits,
        )
        shifts = geohash.get_shifts(
            geohash_coord[0],
            geohash_coord[1],
            radius,
            bits,
            minx,
            maxx,
            miny,
            maxy,
            grid="orthogonal",
        )
        neighbor_geohashes = geohash.shift_multiple(
            geohash.encode_precision(
                *locations[agent], bits, minx=minx, maxx=maxx, miny=miny, maxy=maxy
            ),
            bits,
            shifts,
        )
        neighbor_geohashes = np.sort(neighbor_geohashes)
        for j, neighbor_geohash in enumerate(neighbor_geohashes):
            patch = mpatches.Rectangle(
                geohash.decode(neighbor_geohash, bits, minx, maxx, miny, maxy),
                window_width,
                window_height,
                facecolor="orange",
                edgecolor="black",
                alpha=j / neighbor_geohashes.size * 0.5 + 0.1,
            )
            ax.add_patch(patch)

        loc = locations[agent]
        for neighbor in agent_neighbors:
            assert agent != neighbor
            neighborloc = locations[neighbor]
            ax.plot([loc[0], neighborloc[0]], [loc[1], neighborloc[1]])

    for x, y, count in zip(locations[:, 0], locations[:, 1], counts):
        if count != 0:
            ax.annotate(str(count), (x, y))

    ax.scatter(locations[:, 0], locations[:, 1], s=4)
