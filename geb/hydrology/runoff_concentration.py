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


class RunoffConcentration(object):
    r"""
    Runoff concentration

    this is the part between runoff generation and routing
    for each gridcell and for each land cover class the generated runoff is concentrated at a corner of a gridcell
    this concentration needs some lag-time (and peak time) and leads to diffusion
    lag-time/ peak time is calculated using slope, length and land cover class
    diffusion is calculated using a triangular-weighting-function

    :math:`Q(t) = \sum_{i=0}^{max} c(i) * Q_{\mathrm{GW}} (t - i + 1)`

    where :math:`c(i) = \int_{i-1}^{i} {2 \over{max}} - | u - {max \over {2}} | * {4 \over{max^2}} du`

    see also:

    http://stackoverflow.com/questions/24040984/transformation-using-triangular-weighting-function-in-python







    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    baseflow              simulated baseflow (= groundwater discharge to river)                             m
    coverTypes            land cover types - forest - grassland - irrPaddy - irrNonPaddy - water - sealed   --
    runoff
    fracVegCover          Fraction of area covered by the corresponding landcover type
    sum_interflow
    runoff_peak           peak time of runoff in seconds for each land use class                            s
    tpeak_interflow       peak time of interflow                                                            s
    tpeak_baseflow        peak time of baseflow                                                             s
    maxtime_runoff_conc   maximum time till all flow is at the outlet                                       s
    runoff_conc           runoff after concentration - triangular-weighting method                          m
    sum_landSurfaceRunof  Runoff concentration above the soil more interflow including all landcover types  m
    landSurfaceRunoff     Runoff concentration above the soil more interflow                                m
    directRunoff          Simulated surface runoff                                                          m
    interflow             Simulated flow reaching runoff instead of groundwater                             m
    ====================  ================================================================================  =========
    """

    def __init__(self, model, hydrology):
        self.model = model
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    def spinup(self):
        pass

    def step(self, interflow, baseflow, runoff):
        assert (runoff >= 0).all()
        assert (interflow >= 0).all()
        assert (baseflow >= 0).all()
        return interflow + baseflow + runoff
