# --------------------------------------------------------------------------------
# This file contains code that has been adapted from an original source available
# in a public repository under the GNU General Public License. The original code
# has been modified to fit the specific needs of this project.
#
# Original source repository: https://github.com/iiasa/CWatM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------


import numpy as np
from numba import njit

BOTTOM_LEFT = 1
BOTTOM = 2
BOTTOM_RIGHT = 3
LEFT = 4
PIT = 5
RIGHT = 6
TOP_LEFT = 7
TOP = 8
TOP_RIGHT = 9


def Compress(map, mask):
    """
    compressing map from 2D to 1D without missing values

    :param map:  input map
    :param mask: mask map
    :return: compressed map
    """

    maskmap = np.ma.masked_array(map, mask)
    compmap = np.ma.compressed(maskmap)
    return compmap


def postorder(dirUp, catchment, node, catch, dirDown):
    """
    Routine to run a postorder tree traversal

    :param dirUp:
    :param catchment:
    :param node:
    :param catch:
    :param dirDown:
    :return: dirDown and catchment
    """

    import sys

    original_recursion_limit = sys.getrecursionlimit()

    sys.setrecursionlimit(10000)

    if dirUp[node] != []:
        postorder(dirUp, catchment, dirUp[node][0], catch, dirDown)
        catchment[dirUp[node][0]] = catch
        dirDown.append(dirUp[node][0])
        if len(dirUp[node]) > 1:
            postorder(dirUp, catchment, dirUp[node][1], catch, dirDown)
            catchment[dirUp[node][1]] = catch
            dirDown.append(dirUp[node][1])
            if len(dirUp[node]) > 2:
                postorder(dirUp, catchment, dirUp[node][2], catch, dirDown)
                catchment[dirUp[node][2]] = catch
                dirDown.append(dirUp[node][2])
                if len(dirUp[node]) > 3:
                    postorder(dirUp, catchment, dirUp[node][3], catch, dirDown)
                    catchment[dirUp[node][3]] = catch
                    dirDown.append(dirUp[node][3])
                    if len(dirUp[node]) > 4:
                        postorder(dirUp, catchment, dirUp[node][4], catch, dirDown)
                        catchment[dirUp[node][4]] = catch
                        dirDown.append(dirUp[node][4])
                        if len(dirUp[node]) > 5:
                            postorder(dirUp, catchment, dirUp[node][5], catch, dirDown)
                            catchment[dirUp[node][5]] = catch
                            dirDown.append(dirUp[node][5])
                            if len(dirUp[node]) > 6:
                                postorder(
                                    dirUp, catchment, dirUp[node][6], catch, dirDown
                                )
                                catchment[dirUp[node][6]] = catch
                                dirDown.append(dirUp[node][6])
                                if len(dirUp[node]) > 7:
                                    postorder(
                                        dirUp, catchment, dirUp[node][7], catch, dirDown
                                    )
                                    catchment[dirUp[node][7]] = catch
                                    dirDown.append(dirUp[node][7])

    sys.setrecursionlimit(original_recursion_limit)
    return


def dirUpstream(dirshort):
    """
    runs the network tree upstream from outlet to source

    :param dirshort:
    :return: direction upstream
    """

    # -- up direction
    dirUp = list([] for i in range(dirshort.shape[0]))
    for i in range(dirshort.shape[0]):
        value = dirshort[i]
        if value > -1:
            dirUp[value].append(i)

    dirupLen = [0]
    dirupID = []
    j = 0
    for i in range(dirshort.shape[0]):
        j += len(dirUp[i])
        dirupLen.append(j)
        for k in range(len(dirUp[i])):
            dirupID.append(dirUp[i][k])

    return (
        dirUp,
        np.array(dirupLen).astype(np.int64),
        np.array(dirupID).astype(np.int64),
    )


def dirDownstream(dirUp, lddcomp, dirDown):
    """
    runs the river network tree downstream - from source to outlet

    :param dirUp:
    :param lddcomp:
    :param dirDown:
    :return: direction downstream
    """

    catchment = np.zeros_like(
        lddcomp, dtype=np.int64
    )  # not sure whether int64 is necessary
    j = 0
    for pit in range(lddcomp.shape[0]):
        if lddcomp[pit] == PIT:
            j += 1
            postorder(dirUp, catchment, pit, j, dirDown)
            dirDown.append(pit)
            catchment[pit] = j
    return np.array(dirDown), np.array(catchment)


@njit(cache=True)
def upstreamArea(dirDown, dirshort, area):
    """
    calculates upstream area

    :param dirDown: array which point from each cell to the next downstream cell
    :param dirshort:
    :param area: area (can be any variable)
    :return: upstream area
    """

    upstream_area = area.copy()
    for i in range(dirDown.size):
        j = dirDown[i]
        k = dirshort[j]
        if k > -1:
            upstream_area[k] += upstream_area[j]

    return upstream_area


def upstream1(downstruct, weight):
    """
    Calculates 1 cell upstream

    :param downstruct:
    :param weight:
    :return: upstream 1cell
    """
    return np.bincount(downstruct, weights=weight)[:-1]


def downstream1(dirUp, weight):
    """
    calculated 1 cell downstream

    :param dirUp:
    :param weight:
    :return: dowmnstream 1 cell
    """

    downstream = weight.copy()
    k = 0
    for i in dirUp:
        for j in i:
            downstream[j] = weight[k]
        k += 1
    return downstream


def subcatchment1(dirUp, points, ups):
    """
    calculates subcatchments of points

    :param dirUp:
    :param points:
    :param ups:
    :return: subcatchment
    """

    subcatch = np.zeros_like(
        points, dtype=np.int64
    )  # not sure whether int64 is necessary
    # if subcatchment = true ->  calculation of subcatchment: every point is calculated
    # if false : calculation of catchment: only point calculated which are not inside a bigger catchment from another point

    subs = {}
    # sort waterbodies of reverse upstream area
    for cell in range(points.size):
        if points[cell] > 0:
            subs[points[cell]] = [cell, ups[cell]]
    subsort = sorted(list(subs.items()), key=lambda x: x[1][1], reverse=True)

    for sub in subsort:
        j = sub[0]
        cell = sub[1][0]
        dirDown = []
        postorder(dirUp, subcatch, cell, j, dirDown)
        subcatch[cell] = j

    return subcatch


def define_river_network(ldd2D, grid):
    """
    defines river network

    :param ldd: river network
    :return: ldd variables
    """
    # every cell gets an order starting from 0
    lddOrder = grid.decompress(np.arange(grid.compressed_size), fillvalue=-1)

    lddCompress, dirshort = lddrepair(ldd2D, lddOrder, grid)
    dirUp, dirupLen, dirupID = dirUpstream(dirshort)

    # for upstream calculation
    inAr = np.arange(grid.compressed_size, dtype=np.int64)
    # each upstream pixel gets the id of the downstream pixel
    downstruct = downstream1(dirUp, inAr).astype(np.int64)
    # all pits gets a high number
    downstruct[lddCompress == PIT] = grid.compressed_size

    # self.var.dirDown: direction downstream - from each cell the pointer to a downstream cell (can only be 1)
    # self.var.catchment: each catchment with a pit gets a own ID
    dirDown = []
    dirDown, catchment = dirDownstream(dirUp, lddCompress, dirDown)
    lendirDown = len(dirDown)

    return (
        lddCompress,
        dirshort,
        dirUp,
        dirupLen,
        dirupID,
        downstruct,
        catchment,
        dirDown,
        lendirDown,
    )


@njit(cache=True)
def repairLdd1(ldd):
    dirX = np.array([0, -1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
    dirY = np.array([0, 1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)

    sizei = ldd.shape[0]
    sizej = ldd.shape[1]

    for i in range(sizei):
        for j in range(sizej):
            lddvalue = ldd[i, j]
            assert lddvalue >= 0 and lddvalue < 10

            if lddvalue != 0 and lddvalue != PIT:
                y = i + dirY[lddvalue]  # y of outflow cell
                x = j + dirX[lddvalue]  # x of outflow cell
                if (
                    y < 0 or y == sizei
                ):  # if outflow cell is outside the domain, make it a pit
                    ldd[i, j] = PIT
                if (
                    x < 0 or x == sizej
                ):  # if outflow cell is outside the domain, make it a pit
                    ldd[i, j] = PIT
                if lddvalue != PIT:
                    if (
                        ldd[y, x] == 0
                    ):  # if outflow cell has no flow, make inflow cell a pit
                        ldd[i, j] = PIT
    return ldd


@njit(cache=True)
def dirID(lddorder, ldd):
    out_array = np.full_like(ldd, -1)  # Initialize out_array with -1, same shape as ldd
    dirX = np.array([0, -1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
    dirY = np.array([0, 1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)

    sizei = ldd.shape[0]
    sizej = ldd.shape[1]

    for i in range(sizei):
        for j in range(sizej):
            lddvalue = ldd[i, j]
            assert lddvalue >= 0 and lddvalue < 10

            if lddvalue != 0 and lddvalue != PIT:
                x = j + dirX[lddvalue]
                y = i + dirY[lddvalue]
                if 0 <= x < sizej and 0 <= y < sizei:
                    out_array[i, j] = lddorder[y, x]

    return out_array


@njit(cache=True)
def repairLdd2(ldd, dir):
    check = np.zeros(ldd.size, dtype=np.int64)
    for i in range(ldd.size):
        path = []
        k = 0
        j = i
        while True:
            if j in path:
                id = path[k - 1]
                ldd[id] = PIT
                dir[id] = -1
                break
            # if drainage direction is a pit or cell is already checked, break
            if ldd[j] == PIT or check[j] == 1:
                break
            path.append(j)
            k += 1
            j = dir[j]

        for id in path:
            check[id] = 1
    return ldd, dir


def lddrepair(lddnp, lddOrder, grid):
    """
    repairs a river network

    * eliminate unsound parts
    * add pits at points with no connections

    :param lddnp: rivernetwork as 1D array
    :param lddOrder:
    :return: repaired ldd
    """

    lddnp = repairLdd1(lddnp)
    dir = dirID(lddOrder, lddnp)

    dir_compressed = grid.compress(dir)
    ldd_compressed = grid.compress(lddnp)

    ldd_compressed, dir_compressed = repairLdd2(ldd_compressed, dir_compressed)

    return ldd_compressed, dir_compressed


MAX_ITERS = 10


@njit(cache=True)
def IterateToQnew(Qin, Qold, sideflow, alpha, beta, deltaT, deltaX):
    epsilon = np.float64(0.0001)

    # If deltaX (channel length) is 0 this should be a pit, handle inflow but no outflow
    if deltaX == 0:
        # Inflow into the pit occurs, so return Qin
        return max(Qin, 0)

    # If no input, then output = 0
    if (Qin + Qold + sideflow) == 0:
        return 0

    # Common terms
    ab_pQ = alpha * beta * ((Qold + Qin) / 2) ** (beta - 1)
    deltaTX = deltaT / deltaX
    C = deltaTX * Qin + alpha * Qold**beta + deltaT * sideflow

    # Initial guess for Qkx and iterative process
    Qkx = (deltaTX * Qin + Qold * ab_pQ + deltaT * sideflow) / (deltaTX + ab_pQ)
    fQkx = deltaTX * Qkx + alpha * Qkx**beta - C
    dfQkx = deltaTX + alpha * beta * Qkx ** (beta - 1)
    Qkx -= fQkx / dfQkx
    if np.isnan(Qkx):
        Qkx = 1e-30
    else:
        Qkx = max(Qkx, 1e-30)
    count = 0

    assert not np.isnan(Qkx)
    while np.abs(fQkx) > epsilon and count < MAX_ITERS:
        fQkx = deltaTX * Qkx + alpha * Qkx**beta - C
        dfQkx = deltaTX + alpha * beta * Qkx ** (beta - 1)
        Qkx -= fQkx / dfQkx
        count += 1
        assert not np.isnan(Qkx)

    if np.isnan(Qkx):
        Qkx = 0
    else:
        Qkx = max(Qkx, 0)
    return Qkx


@njit(cache=True)
def kinematic(Qold, sideflow, dirDown, dirUpLen, dirUpID, alpha, beta, deltaT, deltaX):
    """
    Kinematic wave routing

    Parameters
    ----------
    deltaT: float
        Time step, must be > 0
    deltaX: np.ndarray
        Array of floats containing the channel length, must be > 0
    """
    Qnew = np.zeros_like(Qold)
    for i in range(Qold.size):
        Qin = np.float32(0.0)
        down = dirDown[i]
        minID = dirUpLen[down]
        maxID = dirUpLen[down + 1]

        for j in range(minID, maxID):
            Qin += Qnew[dirUpID[j]]

        Qnew[down] = IterateToQnew(
            Qin, Qold[down], sideflow[down], alpha[down], beta, deltaT, deltaX[down]
        )
    return Qnew
