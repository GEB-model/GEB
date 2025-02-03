# --------------------------------------------------------------------------------
# This file contains code that has been adapted from an original source available
# in a public repository under the GNU General Public License. The original code
# has been modified to fit the specific needs of this project.
#
# Original source repository: https://github.com/FutureWater/SPHY
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


def get_kinetic_energy_direct_throughfall(direct_throughfall, precipitation_intensity):
    """
    Calculate the kinetic energy of rainfall

    Parameters
    ----------
    direct_throughfall : np.array
        Direct throughfall [mm]
    precipitation_intensity : np.array
        Precipitation intensity [?]

    Returns
    -------
    kinetic_energy : np.array
        Kinetic energy of rainfall [J m-2]

    """
    kinetic_energy = (
        direct_throughfall
        * (0.29 * (1 - 0.72 * np.exp(-0.05 * precipitation_intensity)))
        * 100
    )  # Brown and Foster (1987)
    kinetic_energy[direct_throughfall == 0] = (
        0  # no kinetic energy if no direct_throughfall
    )
    return kinetic_energy


def get_kinetic_energy_leaf_drainage(leaf_drainage, plant_height):
    """
    Calculate the kinetic energy of leaf drainage

    Parameters
    ----------
    leaf_drainage : np.array
        Leaf drainage [mm/day]
    plant_height : np.array
        Plant height [m]

    Returns
    -------
    kinetic_energy : np.array
        Kinetic energy of leaf drainage [J m-2]
    """
    return np.where(
        plant_height >= 0.15, leaf_drainage * (15.8 * plant_height**0.5 - 5.87), 0
    )


def get_detachment_from_raindrops(
    K, texture_class_ratio, kinetic_energy, no_erosion, cover
):
    return (
        K
        * texture_class_ratio
        * np.maximum(0, 1 - (no_erosion + cover))
        * kinetic_energy
        * 1e-3
    )


def get_detachment_from_flow(
    DR, texture_class_ratio, direct_runoff, slope, no_erosion, cover
):
    return (
        DR
        * texture_class_ratio
        * direct_runoff**1.5
        * np.maximum(0, 1 - (no_erosion + cover))
        * np.sin(slope) ** (0.3)
        * 1e-3
    )


def get_particle_fall_number(
    delta, velocity, water_depth, rho_s, rho, eta, slope, cell_length
):
    v_s = (float(1) / 18 * (delta**2) * (rho_s - rho) * 9.81) / (eta)
    N_f = (cell_length / np.cos(slope) * v_s) / (velocity * water_depth)
    return N_f


def get_deposition(particle_fall_number):
    return np.minimum(44.1 * particle_fall_number**0.29, 100)


def get_material_transport(detachment_from_raindrops, detachment_from_flow, deposition):
    G = (detachment_from_raindrops + detachment_from_flow) * (1 - deposition / 100)
    D = (detachment_from_raindrops + detachment_from_flow) * deposition / 100
    return G, D


class HillSlopeErosion:
    def __init__(self, model):
        """The constructor erosion"""
        self.HRU = model.data.HRU
        self.model = model

        if self.model.spinup:
            self.spinup()

    def spinup(self):
        self.var = self.model.store.create_bucket("hillslope_erosion.var")
        self.HRU.var.canopy_cover = self.HRU.full_compressed(
            0.5, dtype=np.float32
        )  # using a constant for now
        self.HRU.var.ground_cover = self.HRU.full_compressed(
            0.5, dtype=np.float32
        )  # using a constant for now
        self.HRU.var.slope = self.HRU.full_compressed(0.1, dtype=np.float32)
        self.HRU.var.plant_height = self.HRU.full_compressed(1, dtype=np.float32)
        # need to see what this is, dependent on land use type probably.
        self.HRU.var.no_erosion = self.HRU.full_compressed(0, dtype=np.float32)
        # need to see what this is.
        self.HRU.var.cover = self.HRU.full_compressed(0, dtype=np.float32)

        self.HRU.var.cell_length = self.HRU.full_compressed(1, dtype=np.float32)

        self.var.alpha = 0.34  # using a constant for now
        self.var.K_clay = 0.0004  # using a constant for now
        self.var.K_silt = 0.0002  # using a constant for now
        self.var.K_sand = 0.0001  # using a constant for now

        self.var.DR_clay = 0.0004  # using a constant for now, detachment ratio?
        self.var.DR_silt = 0.0002  # using a constant for now
        self.var.DR_sand = 0.0001  # using a constant for now

        self.var.delta_clay = 0.000002  # using a constant for now
        self.var.delta_silt = 0.000001  # using a constant for now
        self.var.delta_sand = 0.0000005  # using a constant for now

        self.var.rho = 1000  # using a constant for now
        self.var.rho_s = 2650  # using a constant for now
        self.var.eta = 0.001  # using a constant for now

    def step(self):
        pr_mm_day = self.HRU.pr * (24 * 3600)  # # kg/m2/s to m/day
        effective_rainfall_mm_day = pr_mm_day * np.cos(self.HRU.var.slope)

        leaf_drainage = effective_rainfall_mm_day * self.HRU.var.canopy_cover

        direct_throughfall = effective_rainfall_mm_day - leaf_drainage

        precipitation_intensity = direct_throughfall * self.var.alpha

        kinetic_energy_direct_throughfall = get_kinetic_energy_direct_throughfall(
            direct_throughfall, precipitation_intensity
        )

        kinetic_energy_leaf_drainage = get_kinetic_energy_leaf_drainage(
            leaf_drainage, self.HRU.var.plant_height
        )

        kinetic_energy = (
            kinetic_energy_direct_throughfall + kinetic_energy_leaf_drainage
        )

        detachment_from_raindrops_clay = get_detachment_from_raindrops(
            self.var.K_clay,
            self.HRU.var.clay[0] / 100,  # only consider top layer
            kinetic_energy,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        detachment_from_raindrops_silt = get_detachment_from_raindrops(
            self.var.K_silt,
            self.HRU.var.silt[0] / 100,  # only consider top layer
            kinetic_energy,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        detachment_from_raindrops_sand = get_detachment_from_raindrops(
            self.var.K_sand,
            self.HRU.var.sand[0] / 100,  # only consider top layer
            kinetic_energy,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        # detachment_from_raindrops = (
        #     detachment_from_raindrops_clay
        #     + detachment_from_raindrops_silt
        #     + detachment_from_raindrops_sand
        # )

        detachment_from_flow_clay = get_detachment_from_flow(
            self.var.DR_clay,
            self.HRU.var.clay[0] / 100,  # only consider top layer
            self.HRU.var.direct_runoff,
            self.HRU.var.slope,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        detachment_from_flow_silt = get_detachment_from_flow(
            self.var.DR_silt,
            self.HRU.var.silt[0] / 100,  # only consider top layer
            self.HRU.var.direct_runoff,
            self.HRU.var.slope,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        detachment_from_flow_sand = get_detachment_from_flow(
            self.var.DR_sand,
            self.HRU.var.sand[0] / 100,  # only consider top layer
            self.HRU.var.direct_runoff,
            self.HRU.var.slope,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        # detachment_from_flow = (
        #     detachment_from_flow_clay
        #     + detachment_from_flow_silt
        #     + detachment_from_flow_sand
        # )

        velocity = self.HRU.full_compressed(
            0.0001, dtype=np.float32
        )  # constant for now
        water_depth = self.HRU.full_compressed(
            0.1, dtype=np.float32
        )  # constant for now

        particle_fall_number_clay = get_particle_fall_number(
            self.var.delta_clay,
            velocity,
            water_depth,
            self.var.rho_s,
            self.var.rho,
            self.var.eta,
            self.HRU.var.slope,
            self.HRU.var.cell_length,
        )

        particle_fall_number_silt = get_particle_fall_number(
            self.var.delta_silt,
            velocity,
            water_depth,
            self.var.rho_s,
            self.var.rho,
            self.var.eta,
            self.HRU.var.slope,
            self.HRU.var.cell_length,
        )

        particle_fall_number_sand = get_particle_fall_number(
            self.var.delta_sand,
            velocity,
            water_depth,
            self.var.rho_s,
            self.var.rho,
            self.var.eta,
            self.HRU.var.slope,
            self.HRU.var.cell_length,
        )

        deposition_clay = get_deposition(particle_fall_number_clay)
        deposition_silt = get_deposition(particle_fall_number_silt)
        deposition_sand = get_deposition(particle_fall_number_sand)

        G_clay, D_clay = get_material_transport(
            detachment_from_raindrops_clay, detachment_from_flow_clay, deposition_clay
        )
        G_silt, D_silt = get_material_transport(
            detachment_from_raindrops_silt, detachment_from_flow_silt, deposition_silt
        )
        G_sand, D_sand = get_material_transport(
            detachment_from_raindrops_sand, detachment_from_flow_sand, deposition_sand
        )

        G = G_clay + G_silt + G_sand  # Is G the total material transported away?
        # D = D_clay + D_silt + D_sand  # Is D the total material deposited?

        return G
