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
from geb.module import Module


class RunoffConcentration(Module):
    """Combines all sources of runoff.

    Args:
        model: The GEB model instance.
        hydrology: The hydrology submodel instance.
    """

    def __init__(self, model, hydrology):
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self):
        return "hydrology.runoff_concentration"

    def spinup(self):
        pass

    def step(self, interflow, baseflow, runoff):
        assert (runoff >= 0).all()
        assert (interflow >= 0).all()
        assert (baseflow >= 0).all()
        self.report(self, locals())
        return interflow + baseflow + runoff
