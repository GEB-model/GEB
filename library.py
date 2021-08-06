from numba import njit, types
from numba.typed import List


@njit()
def remap_to_cwatm(from_array, from_gt, to_array, to_gt):
    for py_from in range(from_array.shape[0]):
        for px_from in range(from_array.shape[1]):
            px_to, py_to = remap_pixel(from_gt, to_gt, px_from, py_from)
            to_array[py_to, px_to] += from_array[py_from, px_from]
    assert from_array.sum() == to_array.sum()


@njit()
def get_array_remapper_large_cell_to_small_cell(from_array, from_gt, to_array, to_gt):
    from_array_xsize = from_array.shape[1]
    array_map = np.full(to_array.size, -1, dtype=np.int32)
    
    n = 0  # == py_to * to_array_xsize + px_to
    for py_to in range(to_array.shape[0]):
        for px_to in range(to_array.shape[1]):
            px_from, py_from = remap_pixel(to_gt, from_gt, px_to, py_to)
            array_map[n] = py_from * from_array_xsize + px_from
            n += 1

    pixel_counter = np.full(from_array.size, 0, dtype=np.int32)
    for v in array_map:
        pixel_counter[v] += 1
        assert v <= from_array.size

    array_map_n_pixels = np.full(to_array.size, -1, dtype=np.int32)
    for i in range(to_array.size):
        array_map_n_pixels[i] = pixel_counter[array_map[i]]

    return array_map, array_map_n_pixels

@njit()
def apply_array_remapper(large_cell, small_cell, array_map, array_map_n_pixels, f='mean'):
    small_cell_shape = small_cell.shape
    large_cell = large_cell.reshape(large_cell.size)
    small_cell = small_cell.reshape(small_cell.size)
    if f == 'mean':
        for i in range(array_map.size):
            small_cell[i] = large_cell[array_map[i]] / array_map_n_pixels[i]
    elif f == 'absolute':
        for i in range(array_map.size):
            small_cell[i] = large_cell[array_map[i]]
    else:
        raise ValueError        

    return small_cell.reshape(small_cell_shape)
