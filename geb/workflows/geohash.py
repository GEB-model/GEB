"""Submodule that implements several functions for geohashing. Inspired by https://www.factual.com/blog/how-geohashes-work/."""

from math import cos, radians

import numba as nb
import numpy as np
from numba import njit, prange

RADIUS_EARTH_EQUATOR = 40075017  # m
DISTANCE_1_DEGREE_LATITUDE = RADIUS_EARTH_EQUATOR / 360  # m


@njit(parallel=True, cache=True)
def encode_locations(
    locations: np.ndarray, minx: float, maxx: float, miny: float, maxy: float
) -> np.ndarray:
    """Geohash 2d-array of locations.

    Args:
        locations: 2d array of locations x and y locations.
        minx: Minimum x-value of the entire relevant space.
        maxx: Maximum x-value of the entire relevant space.
        miny: Minimum y-value of the entire relevant space.
        maxy: Maximum y-value of the entire relevant space.

    Returns:
        hashcodes: Geohash codes array for all locations.
    """
    hashcodes = np.empty(locations.shape[0], np.int64)
    for i in nb.prange(locations.shape[0]):  # ty: ignore[not-iterable]
        hashcodes[i] = encode(locations[i, 0], locations[i, 1], minx, maxx, miny, maxy)
    return hashcodes


@njit(cache=True)
def reduce_precision(
    geohashes: np.ndarray, bits: int, inplace: bool = False
) -> np.ndarray:
    """Reduces the precision of input geohashes.

    Args:
        geohashes: Array of geohashes.
        bits: Required number of bits for output geohashes
        inplace: If true, the operation is executed in place, otherwise a new array is returned.

    Returns:
        Array of geohashes with reduced precision.
    """
    precision_tag = get_precision_tag(bits)
    if inplace:
        geohashes_reduced_precision = geohashes
    else:
        geohashes_reduced_precision = np.zeros(geohashes.size, dtype=np.int64)
    for i in nb.prange(geohashes.size):  # ty: ignore[not-iterable]
        geohashes_reduced_precision[i] = geohashes[i] >> (61 - bits) | precision_tag
    return geohashes_reduced_precision


@njit(cache=True)
def get_precision_tag(bits: int) -> np.int64:
    """Get a 64-bit integer that can be ORed with a geohash. This is done by setting the most-significant bit to 1.

    Args:
        bits: Number of bits wanted for resulting geohash.

    Returns:
        precision_tag: 64-bit integer that can be ORed with geohash to set precision.
    """
    return np.int64(0x4000000000000000) | np.int64(0x1) << bits


@njit(cache=True)
def widen(bitstring: np.int64) -> np.int64:
    """To interleave the x and y pair of the geohash we `widen` the bit-values by inserting a 0 value to the left of each of the bits.

    Args:
        bitstring: Geohash of x or y only.

    Returns:
        interleaved_bitstring: Interleaved geohash of x or y.
    """
    bitstring |= bitstring << 16
    bitstring &= np.int64(0x0000FFFF0000FFFF)
    bitstring |= bitstring << 8
    bitstring &= np.int64(0x00FF00FF00FF00FF)
    bitstring |= bitstring << 4
    bitstring &= np.int64(0x0F0F0F0F0F0F0F0F)
    bitstring |= bitstring << 2
    bitstring &= np.int64(0x3333333333333333)
    bitstring |= bitstring << 1
    bitstring &= np.int64(0x5555555555555555)
    return bitstring


@njit(cache=True)
def unwiden(bitstring: np.int64) -> np.int64:
    """To unpack a geohash in its x and y pair, the inverse of widen needs to happen. This function removes a bit from the left of each other bit.

    Args:
        bitstring: Widened bitstring.

    Returns:
        unwidened_bitstring: Unwidened bitstring.
    """
    bitstring &= np.int64(0x5555555555555555)
    bitstring ^= bitstring >> 1
    bitstring &= np.int64(0x3333333333333333)
    bitstring ^= bitstring >> 2
    bitstring &= np.int64(0x0F0F0F0F0F0F0F0F)
    bitstring ^= bitstring >> 4
    bitstring &= np.int64(0x00FF00FF00FF00FF)
    bitstring ^= bitstring >> 8
    bitstring &= np.int64(0x0000FFFF0000FFFF)
    bitstring ^= bitstring >> 16
    bitstring &= np.int64(0x00000000FFFFFFFF)
    return bitstring


@njit(cache=True, locals={"y": nb.float64, "x": nb.float64, "bits": nb.int16})
def encode_precision(
    x: float,
    y: float,
    bits: int,
    minx: int | float = -180,
    maxx: int | float = 180,
    miny: int | float = -90,
    maxy: int | float = 90,
) -> np.int64:
    """Geohashes a x/y pair for given precision in bits.

    Args:
        x: x-value.
        y: y-value.
        bits: Precision for geohash.
        minx: Minimum x-value of the entire relevant space.
        maxx: Maximum x-value of the entire relevant space.
        miny: Minimum y-value of the entire relevant space.
        maxy: Maximum y-value of the entire relevant space.

    Returns:
        geohash: Geohash for x/y pair.
    """
    return encode(x, y, minx, maxx, miny, maxy) >> (61 - bits) | get_precision_tag(bits)


@njit(cache=True, locals={"y": nb.float64, "x": nb.float64})
def encode(
    x: float,
    y: float,
    minx: int | float = -180,
    maxx: int | float = 180,
    miny: int | float = -90,
    maxy: int | float = 90,
) -> np.int64:
    """Geohashes a x/y pair.

    Args:
        x: x-value.
        y: y-value.
        minx: Minimum x-value of the entire relevant space.
        maxx: Maximum x-value of the entire relevant space.
        miny: Minimum y-value of the entire relevant space.
        maxy: Maximum y-value of the entire relevant space.

    Returns:
        geohash: Geohash for x/y pair.
    """
    xs = widen(
        np.int64(
            np.float64(
                (x + np.int64(maxx)) * np.int64(0x80000000) / np.float64(maxx - minx)
            )
        )
        & np.int64(0x7FFFFFFF)
    )
    ys = widen(
        np.int64(
            np.float64(
                (y + np.int64(maxy)) * np.int64(0x80000000) / np.float64(maxy - miny)
            )
        )
        & np.int64(0x7FFFFFFF)
    )
    return ys >> 1 | xs


@njit(cache=True, locals={"shifted": nb.int64})
def decode(
    gh: np.int64,
    bits: int,
    minx: int | float = -180,
    maxx: int | float = 180,
    miny: int | float = -90,
    maxy: int | float = 90,
) -> tuple[float, float]:
    """Decodes a geohashes into x/y pair for given precision in bits.

    Args:
        gh: Geohash
        bits: Precision for geohash.
        minx: Minimum x-value of the entire relevant space.
        maxx: Maximum x-value of the entire relevant space.
        miny: Minimum y-value of the entire relevant space.
        maxy: Maximum y-value of the entire relevant space.

    Returns:
        x: Decoded x-value
        y: Decoded y-value
    """
    shifted = gh << (61 - bits)
    y = (
        np.float64(unwiden(shifted >> 1) & np.int64(0x3FFFFFFF))
        / np.int64(0x40000000)
        * (maxy - miny)
        + miny
    )
    x = (
        np.float64(unwiden(shifted) & np.int64(0x7FFFFFFF))
        / np.int64(0x80000000)
        * (maxx - minx)
        + minx
    )
    return x, y


@njit(cache=True)
def shift(gh: np.int64, bits: int, dx: int, dy: int) -> np.int64:
    """Shifts a geohash by number of windows in given x and y-direction.

    Args:
        gh: Geohash
        bits: Precision for geohash.
        dx: Number of windows to shift in x-direction. Can be negative.
        dy: Number of windows to shift in y-direction. Can be negative.

    Returns:
        gh: Shifted geohash.
    """
    if (bits & 1) == 0:
        sx = dy
        sy = dx
    else:
        sx = dx
        sy = dy
    return (widen(unwiden(gh >> 1) + sy) << 1 | widen(unwiden(gh) + sx)) & ~(
        -np.int64(0x1) << bits
    ) | get_precision_tag(bits)


@njit(cache=True)
def shift_multiple(gh: np.int64, bits: int, shifts: np.ndarray) -> np.ndarray:
    """Shifts a geohash by number of windows in given x and y-directions.

    Args:
        gh: Geohash
        bits: Precision for geohash.
        shifts: 2-dimensional array with number of windows too shift in x- and y-direction. Can be negative. The first dimension represents a number of shifts, while the second dimension is the x- and y-shift.

    Returns:
        ghs: Array of shifted geohashes.
    """
    ghs = np.empty(shifts.shape[0], dtype=np.int64)
    unwidened_gh = unwiden(gh)
    unwidened_shifted_gh = unwiden(gh >> 1)
    precision = get_precision_tag(bits)
    for i in prange(shifts.shape[0]):  # ty: ignore[not-iterable]
        if (bits & 1) == 0:
            sx = shifts[i, 1]
            sy = shifts[i, 0]
        else:
            sx = shifts[i, 0]
            sy = shifts[i, 1]
        ghs[i] = (widen(unwidened_shifted_gh + sy) << 1 | widen(unwidened_gh + sx)) & ~(
            -np.int64(0x1) << bits
        ) | precision
    return ghs


@njit(cache=True)
def window(
    bits: int,
    minx: int | float = -180,
    maxx: int | float = 180,
    miny: int | float = -90,
    maxy: int | float = 90,
) -> tuple[float, float]:
    """Gets the width and height of a geohash window for given precision and area size.

    Args:
        bits: Precision for geohash.
        minx: Minimum x-value of the entire relevant space.
        maxx: Maximum x-value of the entire relevant space.
        miny: Minimum y-value of the entire relevant space.
        maxy: Maximum y-value of the entire relevant space.

    Returns:
        window_width: Width of geohash window.
        window_height: Height of geohash window.
    """
    window_height = 0.5 ** (bits // 2) * (maxy - miny)
    ratio = (maxx - minx) / (maxy - miny)
    if (bits % 2) == 0:
        window_width = window_height * ratio
    else:
        window_width = window_height * ratio / 2
    return window_width, window_height


@njit
def get_shifts(
    x: float,
    y: float,
    radius: float | int,
    n_bits: int,
    minx: float | int = -180,
    maxx: float | int = 180,
    miny: float | int = -90,
    maxy: float | int = 90,
    grid: str = "longlat",
) -> np.ndarray:
    """Gets the geohash shifts required to cover a circle (x, y) with given radius for given number of bits.

    Args:
        x: x-coordinate of circle center.
        y: y-coordinate of circle center.
        radius: Circle radius. Specified in meters for grid="longlat", specified in map units for grid="orthogonal".
        n_bits: Precision for geohash.
        minx: Minimum x-value of the entire relevant space.
        maxx: Maximum x-value of the entire relevant space.
        miny: Minimum y-value of the entire relevant space.
        maxy: Maximum y-value of the entire relevant space.
        grid: The type of grid. Choose from `longlat` and `orthogonal`.

    Returns:
        shifts: Geohash shifts required to cover circle for given number of bits.

    Raises:
        ValueError: If grid is not 'longlat' or 'orthogonal'.
    """
    if grid == "longlat":
        width_deg, height_deg = window(n_bits, minx, maxx, miny, maxy)
        distance_1_degree_longitude = DISTANCE_1_DEGREE_LATITUDE * cos(radians(y))

        lon_distance_degrees = radius / distance_1_degree_longitude
        lat_distance_degrees = radius / DISTANCE_1_DEGREE_LATITUDE

        width_per_cell_m = width_deg * distance_1_degree_longitude
        height_per_cell_m = height_deg * DISTANCE_1_DEGREE_LATITUDE

        w_max = int(lon_distance_degrees / width_deg)
        h_max = int(lat_distance_degrees / height_deg)

    elif grid == "orthogonal":
        width_per_cell_m, height_per_cell_m = window(n_bits, minx, maxx, miny, maxy)

        w_max = int(radius / width_per_cell_m)
        h_max = int(radius / height_per_cell_m)
    else:
        raise ValueError("'grid' must be either 'longlat' or 'orthogonal'")

    shifts = np.empty(((w_max * 2 + 1) * (2 * h_max + 1), 2), dtype=np.int32)
    shifts[0] = [0, 0]
    i = 1
    for y in range(1, h_max + 1):
        shifts[i] = [0, y]
        i += 1
    for x in range(1, w_max + 1):
        shifts[i] = [x, 0]
        i += 1
    for y in range(-h_max, 0):
        shifts[i] = [0, y]
        i += 1
    for x in range(-w_max, 0):
        shifts[i] = [x, 0]
        i += 1
    for x in range(1, w_max + 1):
        for y in range(1, h_max + 1):
            if (
                (x * width_per_cell_m) ** 2 + (y * height_per_cell_m) ** 2
            ) ** 0.5 <= radius:
                shifts[i] = [x, y]
                i += 1
                shifts[i] = [-x, y]
                i += 1
                shifts[i] = [x, -y]
                i += 1
                shifts[i] = [-x, -y]
                i += 1
            else:
                h_max = y
                break

    return shifts[:i]


def plot_geohash_shifts(
    lon: float = 4.8945,
    lat: float = 52.3667,
    radius: float | int = 5000,
    bits: int = 31,
    show: bool = True,
) -> None:
    """This function can be used to explore how geohashes can be used to cover a given circle. There is a trade-off between precision of the geohash, speed and how well the shifts represent a circle. This function plots a circle and geohash windows required to cover the circle.

    Args:
        lon: Longitude of circle centre.
        lat: Latitude of circle centre.
        radius: Radius of circle.
        bits: Number of geohash bits used.
        show: Whether to show the resulting plot.

    Raises:
        ImportError: If matplotlib or cartopy are not available.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib not found, could not create plot")
    try:
        import cartopy.crs as ccrs
        from cartopy.io.img_tiles import OSM
    except ImportError:
        raise ImportError("Cartopy not found, could not create plot")

    imagery = OSM()

    gh = encode_precision(lon, lat, bits)
    shifts = get_shifts(lon, lat, radius, bits)

    ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea())

    window_width, window_height = window(bits)

    minlat, maxlat, minlon, maxlon = 90, -90, 180, -180
    patches = []
    for sh in shifts:
        neighbor = shift(gh, bits, sh[0], sh[1])
        gh_lon, gh_lat = decode(neighbor, bits)
        if gh_lon < minlon:
            minlon = gh_lon
        if gh_lon > maxlon:
            maxlon = gh_lon
        if gh_lat < minlat:
            minlat = gh_lat
        if gh_lat > maxlat:
            maxlat = gh_lat
        patches.append(
            mpatches.Rectangle(
                (gh_lon, gh_lat),
                window_width,
                window_height,
                transform=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="black",
            )
        )

    patches[0].set_facecolor("#ff000088")

    maxlon += window_width
    maxlat += window_height

    latd = (maxlat - minlat) / 10
    lond = (maxlon - minlon) / 10

    ax.set_extent((minlon - lond, maxlon + lond, minlat - latd, maxlat + latd))
    ax.add_image(imagery, 14)  # ty: ignore[too-many-positional-arguments,invalid-argument-type]  add image is enhanced by cartopy, but not in typeshed

    ax.set_title(
        f"lon: {lon}, lat: {lat}, radius: {radius}, bits: {bits}", size="x-small"
    )

    for patch in patches:
        ax.add_patch(patch)

    if show:
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plot_geohash_shifts(bits=31, show=False)
    plt.savefig("geohash_31bits.svg", bbox_inches="tight")
    plot_geohash_shifts(bits=32, show=False)
    plt.savefig("geohash_32bits.svg", bbox_inches="tight")
