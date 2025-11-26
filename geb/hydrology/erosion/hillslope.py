"""Implementation of the Morgan–Morgan–Finney (MMF) hillslope erosion model."""

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
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from geb.module import Module
from geb.types import ArrayFloat32

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


def get_kinetic_energy_direct_throughfall(
    direct_throughfall: npt.NDArray[np.float32],
    precipitation_intensity: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the kinetic energy of rainfall.

    Args:
        direct_throughfall: Direct throughfall (mm).
        precipitation_intensity: Precipitation intensity derived from direct throughfall and alpha fraction (mm).

    Returns:
        Kinetic energy of rainfall (J/m²).
    """
    kinetic_energy = (
        direct_throughfall
        * (
            np.float32(0.29)
            * (
                np.float32(1.0)
                - np.float32(0.72) * np.exp(np.float32(-0.05) * precipitation_intensity)
            )
        )
        * np.float32(100.0)
    )  # Brown and Foster (1987)
    kinetic_energy[direct_throughfall == np.float32(0.0)] = np.float32(
        0.0
    )  # no kinetic energy if no direct_throughfall
    return kinetic_energy


def get_kinetic_energy_leaf_drainage(
    leaf_drainage: npt.NDArray[np.float32], plant_height: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Calculate the kinetic energy of leaf drainage.

    Args:
        leaf_drainage: Leaf drainage (mm/day).
        plant_height: Plant height (m).

    Returns:
        Kinetic energy of leaf drainage (J/m²).
    """
    return np.where(
        plant_height >= np.float32(0.15),
        leaf_drainage
        * (np.float32(15.8) * plant_height ** np.float32(0.5) - np.float32(5.87)),
        np.float32(0.0),
    )


def get_detachment_from_raindrops(
    K: np.float32,
    texture_class_ratio: npt.NDArray[np.float32],
    kinetic_energy: npt.NDArray[np.float32],
    no_erosion: npt.NDArray[np.bool_],
    cover: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate soil detachment by raindrop impact.

    Args:
        K: Detachability of the soil by raindrop impact (g/J).
        texture_class_ratio: Ratio of texture class (clay, silt, or sand) (dimensionless).
        kinetic_energy: Kinetic energy of rainfall (J/m²).
        no_erosion: Mask indicating areas where erosion does not occur (dimensionless).
        cover: Ground cover fraction (dimensionless).

    Returns:
        Detachment from raindrops (kg/m²).
    """
    return (
        K
        * texture_class_ratio
        * np.maximum(
            np.float32(0.0), np.float32(1.0) - (no_erosion.astype(np.float32) + cover)
        )
        * kinetic_energy
        * np.float32(1e-3)
    )


def get_detachment_from_flow(
    DR: np.float32,
    texture_class_ratio: npt.NDArray[np.float32],
    direct_runoff: npt.NDArray[np.float32],
    slope: npt.NDArray[np.float32],
    no_erosion: npt.NDArray[np.bool_],
    cover: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate soil detachment by overland flow.

    Args:
        DR: Detachability of the soil by runoff (g/mm).
        texture_class_ratio: Ratio of texture class (clay, silt, or sand) (dimensionless).
        direct_runoff: Direct runoff (mm).
        slope: Slope angle (radians).
        no_erosion: Mask indicating areas where erosion does not occur (dimensionless).
        cover: Ground cover fraction (dimensionless).

    Returns:
        Detachment from flow (kg/m²).
    """
    return (
        DR
        * texture_class_ratio
        * direct_runoff ** np.float32(1.5)
        * np.maximum(
            np.float32(0.0), np.float32(1.0) - (no_erosion.astype(np.float32) + cover)
        )
        * np.sin(slope) ** np.float32(0.3)
        * np.float32(1e-3)
    )


def get_particle_fall_velocity(
    particle_diameter: np.float32,
    rho_s: np.float32,
    rho: np.float32,
    eta: np.float32,
    g: np.float32 = np.float32(9.81),
) -> np.float32:
    """Calculate particle fall velocity using Stokes' law.

    See https://doi.org/10.1002/esp.1530 equation 31.

    Args:
        particle_diameter: Particle diameter (m).
        rho_s: Sediment density (kg/m³).
        rho: Flow density (kg/m³).
        eta: Fluid viscosity (kg/(m·s)).
        g: Gravitational acceleration (m/s²), default is 9.81.

    Returns:
        Particle fall velocity (m/s).
    """
    return (
        np.float32(1.0 / 18.0)
        * (particle_diameter ** np.float32(2.0))
        * (rho_s - rho)
        * g
    ) / eta


def get_particle_fall_number(
    particle_diameter: np.float32,
    velocity: npt.NDArray[np.float32],
    water_depth: npt.NDArray[np.float32],
    rho_s: np.float32,
    rho: np.float32,
    eta: np.float32,
    slope: npt.NDArray[np.float32],
    cell_length: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the particle fall number.

    See https://doi.org/10.1002/esp.1530 equations 28-30.

    Notes:
        It seems that the cell_length should not be used like this. Have a good look
        at this before using this in production.

    Args:
        particle_diameter: Particle diameter (m).
        velocity: Flow velocity (m/s).
        water_depth: Water depth (m).
        rho_s: Sediment density (kg/m³).
        rho: Flow density (kg/m³).
        eta: Fluid viscosity (kg/(m·s)).
        slope: Slope angle (radians).
        cell_length: Length of the cell (m).

    Returns:
        Particle fall number (dimensionless).
    """
    particle_fall_velocity = get_particle_fall_velocity(
        particle_diameter, rho_s, rho, eta
    )
    N_f = (cell_length / np.cos(slope) * particle_fall_velocity) / (
        velocity * water_depth
    )
    return N_f


def get_deposition(
    particle_fall_number: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the percentage of particles deposited on the soil surface.

    Args:
        particle_fall_number: Particle fall number (dimensionless).

    Returns:
        Percentage of particles deposited on the soil surface (%).
    """
    return np.minimum(
        np.float32(44.1) * particle_fall_number ** np.float32(0.29), np.float32(100.0)
    )


def get_material_transport(
    detachment_from_raindrops: npt.NDArray[np.float32],
    detachment_from_flow: npt.NDArray[np.float32],
    deposition_percentage: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Calculate transported and redeposited material.

    Args:
        detachment_from_raindrops: Soil detached by raindrops (kg/m²).
        detachment_from_flow: Soil detached by flow (kg/m²).
        deposition_percentage: Percentage of material deposited (%).

    Returns:
        Tuple containing transported material (kg/m²) and redeposited material (kg/m²).
    """
    transported_material = (detachment_from_raindrops + detachment_from_flow) * (
        np.float32(1.0) - deposition_percentage / np.float32(100.0)
    )
    redeposited_material = (
        (detachment_from_raindrops + detachment_from_flow)
        * deposition_percentage
        / np.float32(100.0)
    )
    return transported_material, redeposited_material


def get_mannings_vegatation(
    water_depth: ArrayFloat32,
    stem_diameter: ArrayFloat32,
    number_of_stems: ArrayFloat32,
) -> ArrayFloat32:
    """Calculate the Manning's n for land with vegetation.

    Args:
        water_depth: Water depth (m).
        stem_diameter: Stem diameter (m).
        number_of_stems: Number of stems per m² (dimensionless).

    Returns:
        Manning's n for land with vegetation (dimensionless).
    """
    return (water_depth ** np.float32(0.67)) / (
        np.float32(2.0 * 9.81) / (stem_diameter * number_of_stems)
    ) ** np.float32(0.5)


def get_mannings_tillaged_soil(
    surface_roughness_parameter_tillage: np.float32,
) -> np.float32:
    """Calculate the Manning's n for tilled soil.

    Args:
        surface_roughness_parameter_tillage: Surface roughness parameter for tillage (dimensionless).

    Returns:
        Manning's n for tilled soil (dimensionless).
    """
    return np.exp(
        np.float32(-2.1132) + np.float32(0.0349) * surface_roughness_parameter_tillage
    )


def get_flow_velocity(
    manning: npt.NDArray[np.float32],
    water_depth: npt.NDArray[np.float32],
    slope: npt.NDArray[np.float32],
    minimum_slope: np.float32 = np.float32(1e-5),
) -> npt.NDArray[np.float32]:
    """Calculate the flow velocity using Manning's equation.

    Args:
        manning: Manning's coefficient (dimensionless).
        water_depth: Water depth (m).
        slope: Slope (m/m, dimensionless).
        minimum_slope: Minimum slope (m/m, dimensionless), default is 1e-5.

    Returns:
        Flow velocity (m/s).
    """
    return (
        np.float32(1.0)
        / manning
        * water_depth ** np.float32(2.0 / 3.0)
        * np.maximum(slope, minimum_slope) ** np.float32(0.5)
    )


class HillSlopeErosion(Module):
    """Calculate soil erosion on hillslopes using the MMF model.

    The Morgan–Morgan–Finney (MMF) model is a process-based soil erosion model
    developed by Morgan, Morgan, and Finney (1984). It is designed to estimate annual
    or event-based soil loss by considering the detachment and transport of soil
    particles separately.
    """

    def __init__(self, model: GEBModel, hydrology: Hydrology) -> None:
        """Initialize the hillslope erosion model.

        Currently in super alpha stage and uses a lot of constants and assumptions. Should NOT
        be used for science yet.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology module instance.
        """
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        self.simulate = self.model.config["hazards"]["erosion"]["simulate"]

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Return the name of the module.

        Returns:
            The name of the module.
        """
        return "hydrology.hillslope_erosion"

    def spinup(self) -> None:
        """Initialize variables needed for the hillslope erosion model.

        Currently this is in alpha stage and uses a lot of constants and assumptions. Should NOT
        be use for science yet.

        Raises:
            ValueError: For now always, until slope is checked.
        """
        if not self.simulate:
            return None

        self.var.total_erosion = np.float32(0.0)

        self.HRU.var.canopy_cover = self.HRU.full_compressed(
            np.float32(0.5), dtype=np.float32
        )  # using a constant for now
        self.HRU.var.ground_cover = self.HRU.full_compressed(
            np.float32(0.5), dtype=np.float32
        )  # using a constant for now
        self.HRU.var.plant_height = self.HRU.full_compressed(
            np.float32(1.0), dtype=np.float32
        )
        # need to see what this is, dependent on land use type probably.
        self.HRU.var.no_erosion = self.HRU.full_compressed(False, dtype=bool)
        # need to see what this is.
        self.HRU.var.cover = self.HRU.full_compressed(np.float32(0.0), dtype=np.float32)

        raise ValueError(
            "Check if slope is correctly implemented here. There seems to be some uncertainty whether slope should be in radians or m/m (or perhaps something else entirely)"
        )

        self.HRU.var.slope = self.hydrology.to_HRU(
            data=self.hydrology.grid.load(
                self.model.files["grid"]["landsurface/slope"]
            ),
            fn=None,
        )

        # Is correct? -> Does not seem to be correct. Should be "length" of the element. But what is that?
        raise ValueError(
            "Check if cell_length is correctly implemented here. Looking at the original code, cell length seems to implictly assume one hillslope element per cell. Which would have some implications for the dynamic cell size that we have in GEB"
        )
        self.HRU.var.cell_length = np.sqrt(self.HRU.var.cell_area)

        # NOTE: water depth in field seems quite deep now, and is not variable.
        # perhaps this should be made more dynamic?
        # Depth is dicussed here, depends on type of rill: https://doi.org/10.1002/esp.1530
        self.HRU.var.water_depth_in_field = self.HRU.full_compressed(
            np.float32(0.1), dtype=np.float32
        )
        self.HRU.var.stem_diameter = self.HRU.full_compressed(
            np.float32(0.01), dtype=np.float32
        )
        self.HRU.var.stem_diameter_harvested = self.HRU.full_compressed(
            np.float32(0.005), dtype=np.float32
        )

        self.var.mannings_bare_soil = np.float32(0.015)
        self.var.surface_roughness_parameter_tillage = np.float32(6.0)
        self.HRU.var.no_elements = self.HRU.full_compressed(
            np.float32(1000.0), dtype=np.float32
        )
        self.HRU.var.no_elements_harvested = self.HRU.full_compressed(
            np.float32(10000.0), dtype=np.float32
        )

        self.HRU.var.tillaged = self.HRU.full_compressed(True, dtype=bool)
        self.HRU.var.no_vegetation = self.HRU.full_compressed(False, dtype=bool)

        self.var.alpha = np.float32(0.34)  # using a constant for now

        # Detachability (K) of the soil by raindrop impact (g/J)
        self.var.K_clay = np.float32(0.1)
        self.var.K_silt = np.float32(0.5)
        self.var.K_sand = np.float32(0.3)

        # Detachability of the soil by runoff (g/mm)
        self.var.detachability_due_to_runoff_clay = np.float32(1.0)
        self.var.detachability_due_to_runoff_silt = np.float32(1.6)
        self.var.detachability_due_to_runoff_sand = np.float32(1.5)

        # particle diameter (m)
        self.var.particle_diameter_clay = np.float32(2e-6)
        self.var.particle_diameter_silt = np.float32(6e-5)
        self.var.particle_diameter_sand = np.float32(2e-4)

        self.var.rho = np.float32(
            1100.0
        )  # Flow density (kg/m³). Typically 1100 kg/m³ for runoff on hillslopes (Abrahams et al., 2001).
        self.var.rho_s = np.float32(
            2650.0
        )  # Sediment density (kg/m³). Typically 2650 kg/m³.
        self.var.eta = np.float32(
            0.0015
        )  # Fluid viscosity (kg/(m·s)). Nominally 0.001 kg/(m·s) but taken as 0.0015 to allow for the effects of the sediment in the flow.

    def get_flow_velocity(self) -> ArrayFloat32:
        """Calculate the flow velocity using Manning's equation.

        Returns:
            The flow velocity (m/s).
        """
        mannings_vegated_field = get_mannings_vegatation(
            self.HRU.var.water_depth_in_field,
            self.HRU.var.stem_diameter,
            self.HRU.var.no_elements,
        )
        mannings_vegated_field[self.HRU.var.no_vegetation] = np.float32(0.0)
        mannings_vegated_field[self.HRU.var.no_erosion] = np.float32(0.0)

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
        mannings_vegated_field_harvested[self.HRU.var.tillaged] = np.float32(0.0)

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

    def step(self) -> ArrayFloat32 | None:
        """Perform one timestep of the hillslope erosion model.

        Returns:
            transported_material_kg: The amount of material transported in kg.
        """
        if not self.simulate:
            return None

        pr_mm_day = self.HRU.pr_kg_per_m2_per_s.mean() * np.float32(
            24.0 * 3600.0
        )  # kg/m²/s to mm/day (assuming water density = 1000 kg/m³)
        effective_rainfall_mm_day = pr_mm_day * np.cos(
            self.HRU.var.slope
        )  # slope-corrected rainfall (mm/day)

        leaf_drainage = effective_rainfall_mm_day * self.HRU.var.canopy_cover

        direct_throughfall = effective_rainfall_mm_day - leaf_drainage

        precipitation_intensity = (
            direct_throughfall * self.var.alpha
        )  # alpha is the fraction of rain in the highest intensity class (dimensionless)

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
            self.HRU.var.clay[0] / np.float32(100.0),  # only consider top layer
            kinetic_energy,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        detachment_from_raindrops_silt = get_detachment_from_raindrops(
            self.var.K_silt,
            self.HRU.var.silt[0] / np.float32(100.0),  # only consider top layer
            kinetic_energy,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        detachment_from_raindrops_sand = get_detachment_from_raindrops(
            self.var.K_sand,
            self.HRU.var.sand[0] / np.float32(100.0),  # only consider top layer
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
            self.HRU.var.clay[0] / np.float32(100.0),  # only consider top layer
            self.HRU.var.direct_runoff,
            self.HRU.var.slope,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        detachment_from_flow_silt = get_detachment_from_flow(
            self.var.detachability_due_to_runoff_silt,
            self.HRU.var.silt[0] / np.float32(100.0),  # only consider top layer
            self.HRU.var.direct_runoff,
            self.HRU.var.slope,
            self.HRU.var.no_erosion,
            self.HRU.var.cover,
        )

        detachment_from_flow_sand = get_detachment_from_flow(
            self.var.detachability_due_to_runoff_sand,
            self.HRU.var.sand[0] / np.float32(100.0),  # only consider top layer
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

        velocity = self.get_flow_velocity()

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

        self.report(locals())

        return transported_material
