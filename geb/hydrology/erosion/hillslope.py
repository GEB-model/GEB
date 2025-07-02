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

from geb.module import Module


def get_kinetic_energy_direct_throughfall(direct_throughfall, precipitation_intensity):
    """Calculate the kinetic energy of rainfall.

    Parameters
    ----------
    direct_throughfall : np.array
        Direct throughfall [mm]
    precipitation_intensity : np.array
        Precipitation intensity [?]

    Returns:
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
    """Calculate the kinetic energy of leaf drainage.

    Parameters
    ----------
    leaf_drainage : np.array
        Leaf drainage [mm/day]
    plant_height : np.array
        Plant height [m]

    Returns:
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


def get_particle_fall_velocity(particle_diameter, rho_s, rho, eta, g=9.81):
    """See https://doi.org/10.1002/esp.1530 equation 31.

    Parameters
    ----------
    particle_diameter : float
        Particle diameter [m]
    rho_s : float
        Sediment density [kg m−3]
    rho : float
        Flow density [kg m−3]
    eta : float
        Fluid viscosity [kg m−1 s−1]

    Returns:
    -------
    np.array

    """
    return (1 / 18 * (particle_diameter**2) * (rho_s - rho) * g) / eta


def get_particle_fall_number(
    particle_diameter, velocity, water_depth, rho_s, rho, eta, slope, cell_length
):
    """See https://doi.org/10.1002/esp.1530 equations 28-30.

    Parameters
    ----------
    particle_diameter : float
        Particle diameter [m]
    velocity : np.array
        Flow velocity [m/s]
    water_depth : np.array
        Water depth [m]
    rho_s : float
        Sediment density [kg m−3]
    rho : float
        Flow density [kg m−3]
    eta : float
        Fluid viscosity [kg m−1 s−1]

    Returns:
    -------
    np.array

    """
    particle_fall_velocity = get_particle_fall_velocity(
        particle_diameter, rho_s, rho, eta
    )
    N_f = (cell_length / np.cos(slope) * particle_fall_velocity) / (
        velocity * water_depth
    )
    return N_f


def get_deposition(particle_fall_number: np.ndarray):
    """Gets the percentage of particles deposited on the soil surface.

    Args:
        particle_fall_number : np.array

    Returns:
        np.array: Percentage of particles deposited on the soil surface.
    """
    return np.minimum(44.1 * particle_fall_number**0.29, 100)


def get_material_transport(
    detachment_from_raindrops, detachment_from_flow, deposition_percentage
):
    transported_material = (detachment_from_raindrops + detachment_from_flow) * (
        1 - deposition_percentage / 100
    )
    redeposited_material = (
        (detachment_from_raindrops + detachment_from_flow) * deposition_percentage / 100
    )
    return transported_material, redeposited_material


def get_mannings_vegatation(water_depth, stem_diameter, number_of_stems):
    return (water_depth**0.67) / ((2 * 9.81) / (stem_diameter * number_of_stems)) ** 0.5


def get_mannings_tillaged_soil(surface_roughness_parameter_tillage):
    return np.exp(-2.1132 + 0.0349 * surface_roughness_parameter_tillage)


def get_flow_velocity(manning, water_depth, slope, minimum_slope=1e-5):
    """Use the Manning's equation to calculate the flow velocity.

    Parameters
    ----------
    manning : float
        Manning's coefficient
    water_depth : float
        Water depth
    slope : float
        Slope
    minimum_slope : float, optional
        Minimum slope, by default 1e-5
    """
    return (
        1 / manning * water_depth ** (2 / 3) * np.maximum(slope, minimum_slope) ** 0.5
    )


class HillSlopeErosion(Module):
    """Calculate soil erosion on hillslopes using the MMF model.

    The Morgan–Morgan–Finney (MMF) model is a process-based soil erosion model
    developed by Morgan, Morgan, and Finney (1984). It is designed to estimate annual
    or event-based soil loss by considering the detachment and transport of soil
    particles separately.
    """

    def __init__(self, model, hydrology):
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        self.simulate = self.model.config["hazards"]["erosion"]["simulate"]

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self):
        return "hydrology.hillslope_erosion"

    def spinup(self):
        if not self.simulate:
            return None

        self.var.total_erosion = 0

        self.HRU.var.canopy_cover = self.HRU.full_compressed(
            0.5, dtype=np.float32
        )  # using a constant for now
        self.HRU.var.ground_cover = self.HRU.full_compressed(
            0.5, dtype=np.float32
        )  # using a constant for now
        self.HRU.var.plant_height = self.HRU.full_compressed(1, dtype=np.float32)
        # need to see what this is, dependent on land use type probably.
        self.HRU.var.no_erosion = self.HRU.full_compressed(False, dtype=bool)
        # need to see what this is.
        self.HRU.var.cover = self.HRU.full_compressed(0, dtype=np.float32)

        self.HRU.var.slope = self.hydrology.to_HRU(
            data=self.hydrology.grid.load(
                self.model.files["grid"]["landsurface/slope"]
            ),
            fn=None,
        )

        # Is correct? -> Does not seem to be correct. Should be "length" of the element. But what is that?
        self.HRU.var.cell_length = np.sqrt(self.HRU.var.cell_area)

        # NOTE: water depth in field seems quite deep now, and is not variable.
        # perhaps this should be made more dynamic?
        # Depth is dicussed here, depends on type of rill: https://doi.org/10.1002/esp.1530
        self.HRU.var.water_depth_in_field = self.HRU.full_compressed(
            0.1, dtype=np.float32
        )
        self.HRU.var.stem_diameter = self.HRU.full_compressed(0.01, dtype=np.float32)
        self.HRU.var.stem_diameter_harvested = self.HRU.full_compressed(
            0.005, dtype=np.float32
        )

        self.var.mannings_bare_soil = 0.015
        self.var.surface_roughness_parameter_tillage = 6.0
        self.HRU.var.no_elements = self.HRU.full_compressed(1000, dtype=np.float32)
        self.HRU.var.no_elements_harvested = self.HRU.full_compressed(
            10000, dtype=np.float32
        )

        self.HRU.var.tillaged = self.HRU.full_compressed(True, dtype=bool)
        self.HRU.var.no_vegetation = self.HRU.full_compressed(False, dtype=bool)

        self.var.alpha = 0.34  # using a constant for now

        # Detachability (K) of the soil by raindrop impact (g J−1)
        self.var.K_clay = 0.1
        self.var.K_silt = 0.5
        self.var.K_sand = 0.3

        # Detachability of the soil by runoff (g mm−1)
        self.var.detachability_due_to_runoff_clay = 1.0
        self.var.detachability_due_to_runoff_silt = 1.6
        self.var.detachability_due_to_runoff_sand = 1.5

        # particle diameter (m)
        self.var.particle_diameter_clay = 2e-6
        self.var.particle_diameter_silt = 6e-5
        self.var.particle_diameter_sand = 2e-4

        self.var.rho = 1100  # Sediment density (kg m−3). Typically 2650 kg m−3.
        self.var.rho_s = 2650  # Flow density (kg m-3). Typically 1100 kg m−3 for runoff on hillslopes (Abrahams et al., 2001).
        self.var.eta = 0.0015  # Fluid viscosity (kg m−1 s−1). Nominally 0.001 kg m−1 s−1 but taken as 0.0015 to allow for the effects of the sediment in the flow.

    def get_velocity(self):
        mannings_vegated_field = get_mannings_vegatation(
            self.HRU.var.water_depth_in_field,
            self.HRU.var.stem_diameter,
            self.HRU.var.no_elements,
        )
        mannings_vegated_field[self.HRU.var.no_vegetation] = 0
        mannings_vegated_field[self.HRU.var.no_erosion] = 0

        # this one is not yet implemented
        # mannings_vegated_field = pcr.ifthenelse(
        #     self.n_table > 0, self.n_table, mannings_vegated_field
        # )

        mannings_tilled_soil = get_mannings_tillaged_soil(
            self.var.surface_roughness_parameter_tillage
        )
        mannings_soil = np.where(
            self.HRU.var.tillaged, mannings_tilled_soil, self.var.mannings_bare_soil
        )

        mannings_field = (mannings_soil**2 + mannings_vegated_field**2) ** 0.5
        flow_velocity_field = get_flow_velocity(
            mannings_field, self.HRU.var.water_depth_in_field, self.HRU.var.slope
        )

        mannings_vegated_field_harvested = get_mannings_vegatation(
            self.HRU.var.water_depth_in_field,
            self.HRU.var.stem_diameter_harvested,
            self.HRU.var.no_elements_harvested,
        )
        mannings_vegated_field_harvested[self.HRU.var.tillaged] = 0

        mannings_field_harvested = (
            mannings_soil**2 + mannings_vegated_field_harvested**2
        ) ** 0.5
        flow_velocity_field_harvested = get_flow_velocity(
            mannings_field_harvested,
            self.HRU.var.water_depth_in_field,
            self.HRU.var.slope,
        )

        # all cells without a crop (-1), but with a land owner (not -1) can be considered harvested
        harvested = (self.HRU.var.crop_map == -1) & (self.HRU.var.land_owners != -1)
        flow_velocity = np.where(
            harvested,
            flow_velocity_field_harvested,
            flow_velocity_field,
        )
        return flow_velocity

    def step(self):
        if not self.simulate:
            return None

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
            self.var.detachability_due_to_runoff_clay,
            self.HRU.var.clay[0] / 100,  # only consider top layer
            self.HRU.var.direct_runoff,
            self.HRU.var.slope,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        detachment_from_flow_silt = get_detachment_from_flow(
            self.var.detachability_due_to_runoff_silt,
            self.HRU.var.silt[0] / 100,  # only consider top layer
            self.HRU.var.direct_runoff,
            self.HRU.var.slope,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        detachment_from_flow_sand = get_detachment_from_flow(
            self.var.detachability_due_to_runoff_sand,
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

        velocity = self.get_velocity()

        particle_fall_number_clay = get_particle_fall_number(
            self.var.particle_diameter_clay,
            velocity,
            self.HRU.var.water_depth_in_field,
            self.var.rho_s,
            self.var.rho,
            self.var.eta,
            self.HRU.var.slope,
            self.HRU.var.cell_length,
        )

        particle_fall_number_silt = get_particle_fall_number(
            self.var.particle_diameter_silt,
            velocity,
            self.HRU.var.water_depth_in_field,
            self.var.rho_s,
            self.var.rho,
            self.var.eta,
            self.HRU.var.slope,
            self.HRU.var.cell_length,
        )

        particle_fall_number_sand = get_particle_fall_number(
            self.var.particle_diameter_sand,
            velocity,
            self.HRU.var.water_depth_in_field,
            self.var.rho_s,
            self.var.rho,
            self.var.eta,
            self.HRU.var.slope,
            self.HRU.var.cell_length,
        )

        percentage_deposition_clay = get_deposition(particle_fall_number_clay)
        percentage_deposition_silt = get_deposition(particle_fall_number_silt)
        percentage_deposition_sand = get_deposition(particle_fall_number_sand)

        transported_material_clay, redeposited_material_clay = get_material_transport(
            detachment_from_raindrops_clay,
            detachment_from_flow_clay,
            percentage_deposition_clay,
        )
        transported_material_silt, redeposited_material_silt = get_material_transport(
            detachment_from_raindrops_silt,
            detachment_from_flow_silt,
            percentage_deposition_silt,
        )
        transported_material_sand, redeposited_material_sand = get_material_transport(
            detachment_from_raindrops_sand,
            detachment_from_flow_sand,
            percentage_deposition_sand,
        )

        transported_material = (
            transported_material_clay
            + transported_material_silt
            + transported_material_sand
        )
        # redeposited_material = redeposited_material_clay + redeposited_material_silt + redeposited_material_sand  # Is D the total material deposited?

        transported_material_kg = transported_material * self.HRU.var.cell_area  # kg

        self.var.total_erosion += transported_material_kg.sum()

        self.report(self, locals())

        return transported_material
