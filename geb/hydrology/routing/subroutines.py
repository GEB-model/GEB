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


MAX_ITERS = 1000


@njit(cache=True)
def IterateToQnew(Qin, Qold, sideflow, alpha, beta, deltaT, deltaX):
    epsilon = np.float64(0.000001)

    assert deltaX > 0, "channel length must be greater than 0"

    # If no input, then output = 0
    if (Qin + Qold + sideflow) == 0:
        return 0

    # Common terms
    ab_pQ = alpha * beta * ((Qold + Qin) / 2) ** (beta - 1)
    deltaTX = deltaT / deltaX
    C = deltaTX * Qin + alpha * Qold**beta + deltaT * sideflow

    # Initial guess for Qnew and iterative process
    Qnew = (deltaTX * Qin + Qold * ab_pQ + deltaT * sideflow) / (deltaTX + ab_pQ)
    fQnew = deltaTX * Qnew + alpha * Qnew**beta - C
    dfQnew = deltaTX + alpha * beta * Qnew ** (beta - 1)
    Qnew -= fQnew / dfQnew
    if np.isnan(Qnew):
        Qnew = 1e-30
    else:
        Qnew = max(Qnew, 1e-30)
    count = 0

    while np.abs(fQnew) > epsilon and count < MAX_ITERS:
        fQnew = deltaTX * Qnew + alpha * Qnew**beta - C
        dfQnew = deltaTX + alpha * beta * Qnew ** (beta - 1)
        Qnew -= fQnew / dfQnew
        count += 1

    return max(Qnew, 0)


@njit(cache=True)
def kinematic(
    Qold,
    sideflow,
    dirDown,
    dirUpLen,
    dirUpID,
    alpha,
    beta,
    deltaT,
    deltaX,
    is_waterbody,
    is_outflow,
    waterbody_id,
    waterbody_storage,
    outflow_per_waterbody_m3,
):
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
        down = dirDown[i]

        minID = dirUpLen[down]
        maxID = dirUpLen[down + 1]

        Qin = np.float32(0.0)
        sideflow_node = sideflow[down]
        for j in range(minID, maxID):
            upstream_ID = dirUpID[j]

            if is_outflow[upstream_ID]:
                # if upstream node is an outflow add the outflow of the waterbody
                # to the sideflow
                node_waterbody_id = waterbody_id[upstream_ID]

                # make sure that the waterbody ID is valid
                assert node_waterbody_id != -1
                waterbody_outflow_m3 = outflow_per_waterbody_m3[node_waterbody_id]

                waterbody_storage[node_waterbody_id] -= waterbody_outflow_m3

                # make sure that the waterbody storage does not go below 0
                assert waterbody_storage[node_waterbody_id] >= 0

                sideflow_node += waterbody_outflow_m3 / deltaT / deltaX[down]

            elif is_waterbody[
                upstream_ID
            ]:  # if upstream node is a waterbody, but not an outflow
                assert sideflow[upstream_ID] == 0
                assert Qold[upstream_ID] == 0

            else:  # in normal case, just take the inflow from upstream
                Qin += Qnew[upstream_ID]

        Qnew[down] = IterateToQnew(
            Qin, Qold[down], sideflow_node, alpha[down], beta, deltaT, deltaX[down]
        )
    return Qnew
