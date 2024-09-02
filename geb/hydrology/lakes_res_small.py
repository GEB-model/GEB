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


class SmallLakesReservoirs(object):
    """
    Small LAKES AND RESERVOIRS

    Note:

        Calculate water retention in lakes and reservoirs

        Using the **Modified Puls approach** to calculate retention of a lake
        See also: LISFLOOD manual Annex 3 (Burek et al. 2013)


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m
    load_initial
    waterbalance_module
    seconds_per_timestep                 number of seconds per timestep (default = 86400)                                  s
    lakeEvaFactor         a factor which increases evaporation from lake because of wind                    --
    Invseconds_per_timestep
    runoff
    cellArea              Cell area [m²] of each simulated mesh
    smallpart
    smalllakeArea
    smalllakeDis0
    smalllakeA
    smalllakeFactor
    smalllakeFactorSqr
    smalllakeInflowOld
    smalllakeVolumeM3
    smalllakeOutflow
    smalllakeLevel
    smalllakeStorage
    minsmalllakeVolumeM3
    preSmalllakeStorage
    smallLakeIn
    smallevapWaterBody
    smallLakeout
    smallLakeDiff
    smallrunoffDiff
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        """
        Initialize small lakes and reservoirs
        Read parameters from maps e.g
        area, location, initial average discharge, type: reservoir or lake) etc.
        """
        self.var = model.data.grid
        self.model = model

    def step(self):
        pass
