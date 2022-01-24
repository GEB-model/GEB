# -*- coding: utf-8 -*-
import math
from itertools import combinations
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as mpatches
from numba import njit
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import networkx as nx
from networkx.algorithms.coloring import greedy_color  # other graph coloring schemes are available too
import numpy as np
from scipy.ndimage import generate_binary_structure
from cwatm.management_modules import data_handling
from matplotlib.patches import Circle
import yaml


def cbinding_replace(name):
    return {
        "farms": "D:/DATA/GEB/input/agents/farms.tif",
        "land_use_classes": "D:/DATA/GEB/input/landsurface/land_use_classes.tif"
    }[name]

data_handling.cbinding = cbinding_replace

from HRUs import Data
LAND_USE_TYPE_COLORS = np.array([
    (0, 124, 8),  # Forest No.0
    (195, 255, 211),  # Grasland/non irrigated land
    (0, 222, 15),  # Irrigation
    # (118, 118, 118),  # Sealed area
    (0, 104, 222),  # Water covered area
])
LAND_USE_TYPE_LABELS = [
    'forest',
    'grassland / barren land',
    'cropland',
    # 'sealed area',
    'water'
]
UNIT_COLORS = [
    (0, 161, 157),
    (255, 248, 229),
    (255, 179, 68),
    (224, 93, 93)
]
COLORS = [
    tuple(int(h.strip('#')[i:i+2], 16) for i in (0, 2, 4))
    for h in [
        '#6F69AC', '#95DAC1', '#FFEBA1',
        '#FD6F96'
    ]
]
CELL_SIZE = 20


def cut(array):
    array = array[1540*2+60: 1610*2-20, 2480*2+40: 2550*2-40]
    return array

def neighbors(shape, conn=1):
    dim = len(shape)
    block = generate_binary_structure(dim, conn)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

@njit
def search_regions(img, nbs):
    line = img.ravel()
    rst = np.zeros((line.size, nbs.size), img.dtype)
    for i in range(len(line)):
        if line[i]==0:
            continue
        idx = 0
        for d in nbs:
            if line[i+d]==0:
                continue
            if line[i]==line[i+d]:
                rst[i, idx] = i + d
                idx += 1
    return rst

# @njit
def search(img, nbs):
    s, line = 0, img.ravel()
    rst = np.zeros((len(line) * nbs.size,2), img.dtype)
    for i in range(len(line)):
        if line[i]==0:continue
        for d in nbs:
            if line[i+d]==0: continue
            if line[i]==line[i+d]: continue
            rst[s,0] = line[i]
            rst[s,1] = line[i+d]
            s += 1
    return rst[:s]
                            
def connect_regions(img, conn=1):
    buf = np.pad(img, 1, 'constant')
    nbs = neighbors(buf.shape, conn)
    return search_regions(buf, nbs)

def create_graph(img, conn=1):
    buf = np.pad(img, 1, 'constant')
    nbs = neighbors(buf.shape, conn)
    graph = search(buf, nbs)
    return np.unique(graph, axis=0)

def create_regions(array, indices):
    regions_padded = np.pad(np.zeros_like(array, dtype=np.int32), 1, 'constant', constant_values=-1)
    regions = regions_padded.ravel()

    r = 1
    for i in range(indices.shape[0]):
        if regions[i] != 0:
            continue
        regions[i] = r
        search = [i]
        while search:
            for neighbor in indices[search.pop()]:
                if neighbor == 0:
                    break
                if regions[neighbor] == 0:
                    regions[neighbor] = r
                    search.append(neighbor)
        r += 1

    regions = regions.reshape(regions_padded.shape)
    return regions[1:-1, 1:-1]

def create_adjacency_matrix(graph):
    matrix = np.zeros((graph.max() + 1, graph.max() + 1), dtype=np.int32)
    for i in range(graph.shape[0]):
        x, y = graph[i]
        matrix[x, y] = 1
    assert np.array_equal(matrix.transpose(), matrix)
    return matrix


def add_patches_legend(ax, labels, colors, ncol, legend_fontsize=8, legend_title=None):
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors)) if labels[i] is not None]
    legend = ax.legend(
        handles=patches,
        loc='upper left',
        bbox_to_anchor=(0, -0.10),
        borderaxespad=0,
        ncol=ncol,
        columnspacing=1,
        fontsize=legend_fontsize,
        frameon=False,
        handlelength=1,
        borderpad=0
    )
    if legend_title:
        legend.set_title(legend_title)



def create_region_graph(array):
    array = array.astype(np.int64)
    unique_values = np.unique(array)
    values = np.arange(0, unique_values.size)
    replace = np.zeros(array.max() + 1, dtype=np.int64)
    replace[unique_values] = values
    array = replace[array]
    array += 1

    idx = connect_regions(array)

    regions = create_regions(array, idx)

    graph = create_graph(regions)

    matrix = create_adjacency_matrix(graph)

    G = nx.from_numpy_array(matrix)

    return G, regions

def color(G, regions, colors, strategy, maximum, contracted_nodes=None):

    coloring = greedy_color(G, strategy=strategy, interchange=True)
    assert max(coloring.values()) + 1 <= maximum

    coloring_array = np.full(regions.max() + 1, -1, dtype=np.int32)
    for idx, color in coloring.items():
        coloring_array[idx] = color

    if contracted_nodes:
        for contracted_node in contracted_nodes[::-1]:
            coloring_array[contracted_node[1]] = coloring_array[contracted_node[0]]

    if (coloring_array == -1).any():
        colors.append((255, 255, 255))

    color_indices = coloring_array[regions]
    return np.array(colors)[color_indices]


def create_grid(ax, high_res=False):
    step = 1 if high_res else CELL_SIZE
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, step))
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(end, start, step))
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.tick_params(axis='both', which='both', length=0)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])


def plot_land_use_type(ax, dummymodel, land_owners):
    land_use_type = dummymodel.data.HRU.land_use_type
    land_use_type = dummymodel.data.HRU.decompress(land_use_type)
    land_use_type = cut(land_use_type)
    land_use_type[(land_use_type == 1) & (land_owners != -1)] = 3
    land_use_type[land_use_type > 2] = land_use_type[land_use_type > 2] - 1
    land_use_type[land_use_type > 3] = land_use_type[land_use_type > 3] - 1
    land_use_type_colored = np.array(LAND_USE_TYPE_COLORS)[land_use_type.astype(np.int32)]
    ax.imshow(land_use_type_colored)
    ax.set_title("a - Land use type")

    patches = [
        mpatches.Patch(color=LAND_USE_TYPE_COLORS[i] / 255, label=LAND_USE_TYPE_LABELS[i])
        for i in range(len(LAND_USE_TYPE_COLORS))
    ]
    legend = ax.legend(
        handles=patches,
        loc='upper left',
        bbox_to_anchor=(0, -0.05),
        borderaxespad=0,
        ncol=2,
        columnspacing=1,
        fontsize=10,
        frameon=False,
        handlelength=1,
        borderpad=0
    )
    return land_use_type, land_use_type_colored


def get_dummy_model():
    class Args:
        def __init__(self):
            self.use_gpu = False

    class DummyModel:
        def __init__(self):
            self.args = Args()
            with open('GEB.yml', 'r') as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.data = Data(self)

    return DummyModel()

def main(include_circle=False, show=True, combine_units=True, high_res=False):
    dummymodel = get_dummy_model()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3,
        sharex=True, sharey=True, figsize=(12, 4.5))
    plt.tight_layout()
    plt.subplots_adjust(top=0.99)
    land_owners = dummymodel.data.HRU.land_owners
    land_owners = dummymodel.data.HRU.decompress(land_owners)
    land_owners = cut(land_owners)

    # land use type
    land_use_type, land_use_type_colored = plot_land_use_type(ax0, dummymodel, land_owners)
    create_grid(ax0, high_res=high_res)

    # land owners
    graph, regions = create_region_graph(land_owners)
    for v in np.unique(regions[land_owners == -1]):
        graph.remove_node(v)
    land_owners_colored = color(graph, regions, COLORS, "smallest_last", 4)
    land_owners_colored[land_owners == -1] = 255
    ax1.imshow(land_owners_colored)
    ax1.set_title("b - Crop farms (agents)")
    create_grid(ax1, high_res=high_res)

    # units
    if combine_units:
        units = dummymodel.data.HRU.full_compressed(0, dtype=np.int32)
        if dummymodel.args.use_gpu:
            units = cp.arange(0, units.size, dtype=np.int32)
        else:
            units = np.arange(0, units.size, dtype=np.int32)

        units = dummymodel.data.HRU.decompress(units)
        units = cut(units)
        graph, regions = create_region_graph(units)

        contracted_nodes = []
        for y in range(0, regions.shape[0], CELL_SIZE):
            for x in range(0, regions.shape[1], CELL_SIZE):
                cell = regions[x:x+CELL_SIZE, y:y+CELL_SIZE]
                land_use = land_use_type[x:x+CELL_SIZE, y:y+CELL_SIZE]

                for l in (0, 1, 3, 4):
                    cells_land_use_type = cell[land_use == l]
                    if cells_land_use_type.size > 1:
                        for left, right in zip(cells_land_use_type[1:], cells_land_use_type[:-1]):
                            graph = nx.algorithms.minors.contracted_nodes(graph, left, right, self_loops=False, copy=False)
                            contracted_nodes.append((left, right))
                
        units = color(graph, regions, UNIT_COLORS, "largest_first", 10, contracted_nodes=contracted_nodes)
        create_grid(ax2, high_res=high_res)

    else:
        units = land_owners_colored.copy()
        units[land_use_type != 2] = land_use_type_colored[land_use_type != 2]
        step = CELL_SIZE if not high_res else 1
        start, end = ax2.get_xlim()
        ax2.axes.xaxis.set_ticks(np.arange(start, end, step))
        start, end = ax2.get_ylim()
        ax2.axes.yaxis.set_ticks(np.arange(end, start, step))
        # ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax2.tick_params(axis='both', which='both', length=0)
        # ax2.axes.xaxis.set_ticklabels([])
        # ax2.axes.yaxis.set_ticklabels([])
    
    ax2.imshow(units)
    ax2.set_title("c - Hydrological response units")

    if include_circle:
        for ax in (ax1, ax2):
            circle = Circle((23.5, 19.5), 4.5, fill=False, edgecolor='red', linewidth=2.5)
            ax.add_patch(circle)

    plt.savefig(f'D:/OneDrive - IIASA/Paper/figures/subcells{"_with_circle" if include_circle else ""}{"_high_res" if high_res else ""}.png', dpi=300)
    plt.savefig(f'D:/OneDrive - IIASA/Paper/figures/subcells{"_with_circle" if include_circle else ""}{"_high_res" if high_res else ""}.eps')
    
    if show:
        plt.show()


def draw_pie(ax, ratios, X=0, Y=0, size=1000, colors=['red','blue','green','yellow','magenta','purple']):
    N = len(ratios)

    xy = []

    import matplotlib.path as mpath

    start = 0.
    plot_colors = []
    for ratio, color in zip(ratios, colors):
        if ratio:
            n = int(ratio * 100) + 2
            x = [0] + np.cos(np.linspace(2*math.pi*start,2*math.pi*(start+ratio), n)).tolist()
            y = [0] + np.sin(np.linspace(2*math.pi*start,2*math.pi*(start+ratio), n)).tolist()
            xy.append(list(zip(x, y)))
            start += ratio
            plot_colors.append(color)

    for color, xyi in zip(plot_colors, xy):
        marker = mpath.Path(xyi)
        ax.scatter([X],[Y], marker=marker, s=size, facecolor=tuple(color / 255), edgecolor='none')

def plot_pies():
    dummymodel = get_dummy_model()
    size = 70
    array = np.full((size, size, 3), 255, dtype=np.int32)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4.5))
    plt.tight_layout()
    plt.subplots_adjust(top=0.99)

    plot_land_use_type(ax0, dummymodel)
    create_grid(ax0)

    ax1.imshow(array)
    create_grid(ax1)

    land_use_type = dummymodel.HRU.land_use_type
    land_use_type = dummymodel.HRU.decompress(land_use_type)
    land_use_type = cut(land_use_type)
    land_use_type[land_use_type > 2] = land_use_type[land_use_type > 2] - 1

    for y in range(0, land_use_type.shape[0], CELL_SIZE):
        for x in range(0, land_use_type.shape[1], CELL_SIZE):
            land_use = land_use_type[x:x+CELL_SIZE, y:y+CELL_SIZE]
            land_use_counts = np.unique(land_use.astype(np.int32), return_counts=True)
            ratios = np.zeros(5, dtype=np.float32)
            for land_use_index, land_use_count in zip(land_use_counts[0], land_use_counts[1]):
                ratios[land_use_index] = land_use_count / CELL_SIZE ** 2
            draw_pie(ax1, ratios, Y=x + 4.5, X=y + 4.5, colors=LAND_USE_TYPE_COLORS, size=500)

    ax1.set_title('CWatM land cover')

    plt.savefig(f'plot/output/cwatm_pies.png', dpi=300)
    plt.savefig(f'plot/output/cwatm_pies.eps')

    plt.show()


if __name__ == '__main__':
    # main(include_circle=True, show=True, combine_units=False)
    # main(include_circle=False, show=True, combine_units=True)
    main(include_circle=True, show=True, combine_units=True, high_res=False)
    # main(include_circle=True, show=False, combine_units=True)
    # main(include_circle=True, show=False, combine_units=True)
    # plot_pies()