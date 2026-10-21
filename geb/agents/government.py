"""This module contains the Government agent class for GEB."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geb.geb_types import TwoDArrayFloat32
from geb.hydrology.landcovers import (
    ALL_LAND_COVER_TYPES,
    FOREST,
    OPEN_WATER,
    SEALED,
)
from geb.hydrology.landsurface.water import (
    calculate_soil_hydraulic_properties,
    get_is_top_soil_mask,
)
from geb.workflows.io import read_geom

from .general import AgentBaseClass

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel

logger = logging.getLogger(__name__)


class Government(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model: GEBModel, agents: Agents) -> None:
        """Initialize the Government agent.

        Args:
            model: The GEB model.
            agents: The class that includes all agent types (allowing easier communication between agents).
        """
        super().__init__(model)
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["government"]
            if "government" in self.model.config["agent_settings"]
            else {}
        )
        self.ratio_farmers_to_provide_subsidies_per_year = 0.05

    @property
    def name(self) -> str:
        """Name of the module.

        Returns:
            The name of the module.
        """
        return "agents.government"

    def spinup(self) -> None:
        """This function is called during model spinup."""
        # Set to True once forest is planted during spinup, so a later run with
        # plant_forest enabled can check that its spinup included the forest.
        self.var.forest_planted_in_spinup = False

    def set_irrigation_limit(self) -> None:
        """Set the irrigation limit for crop farmers based on the configuration.

        The irrigation limit can be set per capita, per area of fields, or per command area.
        """
        if "irrigation_limit" not in self.config:
            return None
        irrigation_limit = self.config["irrigation_limit"]
        if irrigation_limit["per"] == "capita":
            self.agents.crop_farmers.var.irrigation_limit_m3[:] = (
                self.agents.crop_farmers.var.household_size * irrigation_limit["limit"]
            )
        elif irrigation_limit["per"] == "area":  # limit per m2 of field
            self.agents.crop_farmers.var.irrigation_limit_m3[:] = (
                self.agents.crop_farmers.field_size_per_farmer
                * irrigation_limit["limit"]
            )
        elif irrigation_limit["per"] == "command_area":
            farmer_command_area = self.agents.crop_farmers.command_area
            farmers_per_command_area = np.bincount(
                farmer_command_area[farmer_command_area != -1],
                minlength=self.model.hydrology.waterbodies.n,
            )

            # get yearly usable release m3. We do not use the current year, as it
            # may not be complete yet, and we only use up to the history fill index
            yearly_usable_release_m3_per_command_area = np.full(
                self.model.hydrology.waterbodies.n, np.nan, dtype=np.float32
            )
            yearly_usable_release_m3_per_command_area[
                self.model.hydrology.waterbodies.is_reservoir
            ] = (self.agents.reservoir_operators.yearly_usuable_release_m3).mean(axis=1)

            irritation_limit_per_command_area = (
                yearly_usable_release_m3_per_command_area / farmers_per_command_area
            )

            # give all farmers there unique irrigation limit
            # all farmers without a command area get no irrigation limit (nan)
            irrigation_limit_per_farmer = irritation_limit_per_command_area[
                farmer_command_area
            ]
            irrigation_limit_per_farmer[farmer_command_area == -1] = np.nan

            # make sure all farmers in a command area have an irrigation limit
            assert not np.isnan(
                irrigation_limit_per_farmer[farmer_command_area != -1]
            ).any()

            self.agents.crop_farmers.var.irrigation_limit_m3[:] = (
                irrigation_limit_per_farmer
            )
        else:
            raise NotImplementedError(
                "Only 'capita' and 'area' are implemented for irrigation limit"
            )
        if "min" in irrigation_limit:
            self.agents.crop_farmers.var.irrigation_limit_m3[
                self.agents.crop_farmers.var.irrigation_limit_m3
                < irrigation_limit["min"]
            ] = irrigation_limit["min"]

    def export_flood_protection_standards(self) -> None:
        """Export flood protection standards to a geoparquet file."""
        subbasins = read_geom(self.model.files["geom"]["routing/subbasins"])[
            ["geometry"]
        ]
        subbasins["COMID"] = subbasins.index

        standards = pd.Series(
            self.flood_protection_standard_subbasins,
            name="flood_protection_standard",
        )

        # Align on COMID index
        subbasins["flood_protection_standard"] = standards.reindex(subbasins.index)

        # export to geoparquet
        output_path = self.model.output_folder / "flood_protection_standards.parquet"
        subbasins.to_parquet(output_path, index=False)

    def step(self) -> None:
        """This function is run each timestep.

        Raises:
            RuntimeError: If a run has plant_forest enabled but its spinup was
                made without planting forest.
        """
        adaptation_enabled = self.config["adaptation"]["enabled"]
        if (
            self.model.current_timestep == 0
            and self.config["plant_forest"]
            and not adaptation_enabled
        ):
            # The forest soils must evolve through the whole spinup, so a run
            # cannot start reforestation on top of a spinup made without it.
            # Spinups made before this flag existed are treated the same way.
            if not self.model.in_spinup and not getattr(
                self.var, "forest_planted_in_spinup", False
            ):
                message = (
                    "Reforestation (agent_settings.government.plant_forest) is "
                    "enabled for this run, but the spinup was made without "
                    "reforestation. Redo the spinup with plant_forest enabled "
                    "('geb spinup'), then start the run again."
                )
                print(message)
                raise RuntimeError(message)
            self.prepare_modified_soil_maps_for_forest()
            if self.model.in_spinup:
                self.var.forest_planted_in_spinup = True

        self.adaptation()
        self.set_irrigation_limit()

        self.report(locals())

        end_time: datetime = datetime.combine(
            self.model.config["general"]["end_time"], datetime.min.time()
        )

        if self.model.current_time.year == end_time.year - 1 and adaptation_enabled:
            self.export_flood_protection_standards()

    def prepare_modified_soil_maps_for_forest(self) -> None:
        """Plant forest: update soil properties in memory and remove displaced farmers.

        HRUs with a forest restoration potential of at least
        ``forest_restoration_potential_threshold`` become forest, except sealed,
        open water and existing forest HRUs. Their soil organic carbon and bulk
        density are adjusted towards the existing forest in the basin, and the soil
        hydraulic properties are recalculated.

        When ``increment_fraction`` is set, only that fraction of the remaining
        suitable HRUs is planted, highest potential first. Each call plants the
        next batch.

        Raises:
            ValueError: If ``increment_fraction`` is not in (0, 1].
        """
        hydrology = self.model.hydrology
        plant_forest_config = self.config["plant_forest"]
        if isinstance(plant_forest_config, dict):
            threshold = plant_forest_config.get(
                "forest_restoration_potential_threshold", 0.5
            )
            increment_fraction = plant_forest_config.get("increment_fraction", None)
        else:
            threshold = 0.5
            increment_fraction = None

        forest_potential = hydrology.grid.load2d(
            self.model.files["grid"]["landsurface/forest_restoration_potential_ratio"]
        )
        suitability_grid = forest_potential >= threshold
        suitability_HRU = hydrology.to_HRU(data=suitability_grid)

        # Sealed (built-up) and open water are not realistic reforestation
        # targets -- exclude them from eligibility regardless of what the
        # forest restoration potential grid says for that pixel. Existing
        # forest is excluded too, so its own soil is not overwritten with the
        # basin forest mean.
        land_use_type = hydrology.HRU.var.land_use_type
        excluded_by_land_use = suitability_HRU & np.isin(
            land_use_type, [SEALED, OPEN_WATER, FOREST]
        )
        self.model.logger.info(
            "Reforestation eligibility: excluded %d sealed (built-up), %d "
            "open water and %d already-forest HRUs that were otherwise "
            "potential-suitable (threshold %.2f).",
            int((excluded_by_land_use & (land_use_type == SEALED)).sum()),
            int((excluded_by_land_use & (land_use_type == OPEN_WATER)).sum()),
            int((excluded_by_land_use & (land_use_type == FOREST)).sum()),
            threshold,
        )
        suitability_HRU = suitability_HRU & ~excluded_by_land_use

        if increment_fraction is not None:
            if not (0.0 < increment_fraction <= 1.0):
                raise ValueError(
                    f"increment_fraction must be between 0 and 1, got {increment_fraction}"
                )
            # Sort suitable HRUs by potential value descending (best areas first).
            # Skip any already classified as FOREST (planted in prior years or
            # originally forest), then take the next chunk of
            # increment_fraction * remaining HRUs.
            forest_potential_HRU = hydrology.to_HRU(data=forest_potential)
            suitable_indices = np.where(suitability_HRU)[0]
            suitable_potentials = forest_potential_HRU[suitable_indices]
            sorted_indices = suitable_indices[np.argsort(suitable_potentials)[::-1]]

            n_suitable = len(sorted_indices)
            if n_suitable == 0:
                self.model.logger.warning(
                    "Incremental reforestation: no suitable HRUs found. No planting applied."
                )
                return

            already_forest = hydrology.HRU.var.land_use_type == FOREST
            remaining = sorted_indices[~already_forest[sorted_indices]]

            if len(remaining) == 0:
                self.model.logger.warning(
                    "Incremental reforestation: all %d suitable HRUs are already "
                    "forest. No planting applied.",
                    n_suitable,
                )
                return

            # chunk is 10% of what remains, not 10% of the fixed total
            chunk_size = max(1, int(np.round(increment_fraction * len(remaining))))
            chunk_indices = remaining[:chunk_size]
            n_already = n_suitable - len(remaining)
            self.model.logger.info(
                "Incremental reforestation: planting %d HRUs (rank %d–%d of %d "
                "suitable; %d already forest, %d remaining).",
                len(chunk_indices),
                n_already,
                n_already + len(chunk_indices) - 1,
                n_suitable,
                n_already,
                len(remaining),
            )
            suitability_HRU = np.zeros(suitability_HRU.shape[0], dtype=bool)
            suitability_HRU[chunk_indices] = True
        else:
            self.model.logger.info(
                "Reforestation (all at once): planting %d suitable HRUs "
                "(threshold %.2f).",
                int(suitability_HRU.sum()),
                threshold,
            )

        land_use_type_before = hydrology.HRU.var.land_use_type.copy()

        # Only the two primary soil properties that actually change under
        # afforestation are modified: soil organic carbon content and bulk
        # density. Sand, silt and clay content (soil texture) are left
        # untouched, so each converted HRU keeps its native soil type. All
        # soil hydraulic properties are then recalculated from these (partly
        # updated) primary properties, using the same pedotransfer functions
        # used at model initialization.
        if not suitability_HRU.any():
            self.model.logger.info(
                "Reforestation: no eligible HRUs, no planting applied."
            )
            return
        forest_mean_soc = self._compute_forest_mean_soc(land_use_type_before)
        bulk_density_ratio = self._compute_bulk_density_ratios(land_use_type_before)
        selected_land_use = land_use_type_before[suitability_HRU]

        soc = hydrology.HRU.var.soil_organic_carbon_percentage
        bulk_density = hydrology.HRU.var.bulk_density_kg_per_dm3
        soc_before = soc[:, suitability_HRU].copy()
        bulk_density_before = bulk_density[:, suitability_HRU].copy()
        # Snapshot the derived properties so the log can report before -> after
        # for everything this conversion touches, not just the two primaries.
        derived_before = {
            name: getattr(hydrology.HRU.var, name)[:, suitability_HRU].copy()
            for name in (
                "water_content_saturated_m",
                "water_content_field_capacity_m",
                "water_content_wilting_point_m",
                "water_content_residual_m",
                "saturated_hydraulic_conductivity_m_per_s",
                "bubbling_pressure_m_positive",
                "lambda_pore_size_distribution",
            )
        }

        # 1) Soil organic carbon: set to the basin forest mean per soil layer,
        # floored at the HRU's original value so conversion can never *decrease*
        # organic carbon (without the floor, HRUs already richer than the
        # forest mean would lose carbon).
        soc_after = np.maximum(forest_mean_soc[:, np.newaxis], soc_before)

        # Afforestation must not decrease soil organic carbon anywhere.
        assert (soc_after >= soc_before).all(), (
            "reforestation decreased soil organic carbon in at least one HRU"
        )

        # 2) Bulk density: the HRU's original value scaled by the measured forest to
        # previous-land-use ratio for its soil layer, bounded per layer by the
        # lowest and highest bulk density already present in the basin. The
        # bound keeps converted soils inside the range the pedotransfer
        # functions already handle at initialization (a very high bulk density
        # gives a negative saturated water content in thetas_toth).
        bulk_density_scaled = (
            bulk_density_before * bulk_density_ratio[:, selected_land_use]
        )
        bulk_density_after = np.clip(
            bulk_density_scaled,
            bulk_density.min(axis=1, keepdims=True),
            bulk_density.max(axis=1, keepdims=True),
        )
        n_bulk_density_bounded = int((bulk_density_after != bulk_density_scaled).sum())
        if n_bulk_density_bounded > 0:
            self.model.logger.warning(
                "Reforestation: bulk density of %d HRU-layers was bounded to the "
                "basin range for its soil layer.",
                n_bulk_density_bounded,
            )

        # 3) Sand, silt and clay content: intentionally left unchanged.

        # 4) Recalculate the soil hydraulic properties from the new SOC and
        # bulk density, using the unchanged sand/silt/clay.
        is_top_soil_subset = get_is_top_soil_mask(soc_after.shape)
        soil_hydraulic_properties = calculate_soil_hydraulic_properties(
            organic_carbon_percentage=soc_after,
            bulk_density_kg_per_dm3=bulk_density_after,
            sand_percentage=hydrology.HRU.var.sand_percentage[:, suitability_HRU],
            silt_percentage=hydrology.HRU.var.silt_percentage[:, suitability_HRU],
            clay_percentage=hydrology.HRU.var.clay_percentage[:, suitability_HRU],
            is_top_soil=is_top_soil_subset,
            soil_layer_height_m=hydrology.HRU.var.soil_layer_height_m[
                :, suitability_HRU
            ],
            saturated_hydraulic_conductivity_multiplier=self.model.config["parameters"][
                "saturated_hydraulic_conductivity_multiplier"
            ],
        )

        # 5) Soil water: the new saturated and residual water contents may no
        # longer bracket each HRU's current water content. Staged on copies,
        # so it can be checked before anything is written.
        cell_area = hydrology.HRU.var.cell_area[suitability_HRU]
        water_content = hydrology.HRU.var.water_content_m[:, suitability_HRU].copy()
        topwater = hydrology.HRU.var.topwater_m[suitability_HRU].copy()
        saturated = soil_hydraulic_properties.water_content_saturated_m
        residual = soil_hydraulic_properties.water_content_residual_m

        # Water above the new saturation is routed to topwater (water conserved).
        excess = np.maximum(0.0, water_content - saturated)
        water_content -= excess
        topwater += excess.sum(axis=0)

        # Water below the new residual is raised to the residual, paid for
        # first from topwater, then from whatever surplus above residual the
        # same HRU holds in its other layers.
        deficit_per_layer = np.maximum(0.0, residual - water_content)
        water_content += deficit_per_layer
        outstanding = deficit_per_layer.sum(axis=0)

        from_topwater = np.minimum(outstanding, topwater)
        topwater -= from_topwater
        outstanding -= from_topwater

        surplus_per_layer = np.maximum(0.0, water_content - residual)
        total_surplus = surplus_per_layer.sum(axis=0)
        from_other_layers = np.minimum(outstanding, total_surplus)
        # Take it out of the surplus layers in proportion to how much each holds.
        share = np.divide(
            surplus_per_layer,
            total_surplus,
            out=np.zeros_like(surplus_per_layer),
            where=total_surplus > 0,
        )
        water_content -= share * from_other_layers
        outstanding -= from_other_layers

        # HRUs that cannot pay for their new residual water content from
        # topwater or their other layers would need water created out of
        # nothing. They are not converted (this round); everything about them
        # stays as it was.
        funded = outstanding <= 0.0
        n_unfunded = int((~funded).sum())
        if n_unfunded > 0:
            self.model.logger.warning(
                "Reforestation: %d HRUs not converted because they hold too "
                "little water to reach the new residual water content of forest "
                "soil.",
                n_unfunded,
            )
        converted_indices = np.flatnonzero(suitability_HRU)[funded]
        suitability_HRU = np.zeros_like(suitability_HRU)
        suitability_HRU[converted_indices] = True

        # Write the funded HRUs to the model.
        soc[:, suitability_HRU] = soc_after[:, funded]
        bulk_density[:, suitability_HRU] = bulk_density_after[:, funded]
        for name in derived_before:
            getattr(hydrology.HRU.var, name)[:, suitability_HRU] = getattr(
                soil_hydraulic_properties, name
            )[:, funded]
        hydrology.HRU.var.water_content_m[:, suitability_HRU] = water_content[:, funded]
        hydrology.HRU.var.topwater_m[suitability_HRU] = topwater[funded]
        # Soil thermal properties (solid heat capacity, solid thermal
        # conductivity and the saturated unfrozen/frozen conductivities) are
        # deliberately left alone. They are not soil hydraulic properties, so
        # they keep the values calculated at model initialization.

        self._log_soil_change_summary(
            suitability_HRU=suitability_HRU,
            soc_before=soc_before[:, funded],
            soc_after=soc[:, suitability_HRU],
            bulk_density_before=bulk_density_before[:, funded],
            bulk_density_after=bulk_density[:, suitability_HRU],
            derived_before={
                name: before[:, funded] for name, before in derived_before.items()
            },
        )

        cell_area = cell_area[funded]
        self.model.logger.info(
            "Reforestation soil water adjustment: %.4f m3 routed from soil to "
            "topwater, %.4f m3 drawn from topwater and %.4f m3 from surplus in "
            "other layers to reach the new residual water content.",
            float((excess.sum(axis=0)[funded] * cell_area).sum()),
            float((from_topwater[funded] * cell_area).sum()),
            float((from_other_layers[funded] * cell_area).sum()),
        )

        self.remove_farmers_from_converted_forest_areas(suitability_HRU)

        # Explicitly mark all planted HRUs as FOREST so that future calls to
        # prepare_modified_soil_maps_for_forest can detect them via the
        # already_forest check and advance to the next increment.
        hydrology.HRU.var.land_use_type[suitability_HRU] = FOREST

        output_folder = self.model.output_folder / "forest_planting"
        output_folder.mkdir(parents=True, exist_ok=True)
        self._save_forest_planting_figure(
            land_use_type_before, suitability_HRU, output_folder, threshold
        )

    def _log_soil_change_summary(
        self,
        suitability_HRU: np.ndarray,
        soc_before: TwoDArrayFloat32,
        soc_after: TwoDArrayFloat32,
        bulk_density_before: TwoDArrayFloat32,
        bulk_density_after: TwoDArrayFloat32,
        derived_before: dict[str, TwoDArrayFloat32],
    ) -> None:
        """Log the before and after value of every soil property that reforestation touches.

        Reports area-weighted means over the converted HRUs, per soil layer,
        for the two primary properties that are changed and for every soil
        hydraulic property recalculated from them.

        Args:
            suitability_HRU: Boolean mask of the converted HRUs.
            soc_before: Organic carbon of the converted HRUs before the change.
            soc_after: Organic carbon of the converted HRUs after the change.
            bulk_density_before: Bulk density of the converted HRUs before the change.
            bulk_density_after: Bulk density of the converted HRUs after the change.
            derived_before: Snapshot of each derived property before the change,
                keyed by HRU variable name.
        """
        hydrology = self.model.hydrology
        n_converted = int(suitability_HRU.sum())
        if n_converted == 0:
            self.model.logger.info("Reforestation: no HRUs converted, soil unchanged.")
            return

        weights = hydrology.HRU.var.cell_area[suitability_HRU]

        def mean_per_layer(values: TwoDArrayFloat32) -> np.ndarray:
            return np.average(values, axis=1, weights=weights)

        self.model.logger.info(
            "Reforestation soil change summary over %d converted HRUs "
            "(area-weighted mean per soil layer, before -> after):",
            n_converted,
        )

        changed: list[tuple[str, TwoDArrayFloat32, TwoDArrayFloat32]] = [
            ("soil_organic_carbon_percentage [%] (CHANGED)", soc_before, soc_after),
            (
                "bulk_density_kg_per_dm3 [kg/dm3] (CHANGED)",
                bulk_density_before,
                bulk_density_after,
            ),
        ]
        for name, before in derived_before.items():
            changed.append(
                (
                    f"{name} (recalculated)",
                    before,
                    getattr(hydrology.HRU.var, name)[:, suitability_HRU],
                )
            )

        for label, before, after in changed:
            before_mean = mean_per_layer(before)
            after_mean = mean_per_layer(after)
            relative_change = np.divide(
                after_mean - before_mean,
                before_mean,
                out=np.zeros_like(before_mean),
                where=before_mean != 0,
            )
            self.model.logger.info(
                "  %-56s before %s -> after %s (%s %%)",
                label,
                np.array2string(before_mean, precision=5, suppress_small=False),
                np.array2string(after_mean, precision=5, suppress_small=False),
                np.array2string(relative_change * 100, precision=1),
            )

        # Properties that must NOT have moved, reported so the log proves it.
        for name in ("sand_percentage", "silt_percentage", "clay_percentage"):
            values = getattr(hydrology.HRU.var, name)[:, suitability_HRU]
            self.model.logger.info(
                "  %-56s unchanged at %s",
                f"{name} [%] (UNCHANGED)",
                np.array2string(mean_per_layer(values), precision=3),
            )

        # The "never decreases" invariant is asserted in
        # prepare_modified_soil_maps_for_forest before this is called.
        self.model.logger.info(
            "  check: soil organic carbon decreased in 0 of %d HRU-layers; "
            "sand/silt/clay untouched; soil thermal properties untouched.",
            int(soc_after.size),
        )

    def _compute_forest_mean_soc(self, land_use_type: np.ndarray) -> TwoDArrayFloat32:
        """Measure the area-weighted mean organic carbon over existing forest, per layer.

        Converted HRUs get this basin forest mean, floored so it never
        decreases an HRU's original value. See
        :meth:`prepare_modified_soil_maps_for_forest`.

        Args:
            land_use_type: Land use type per HRU, as it is just before conversion.

        Returns:
            Float32 array of shape (N_SOIL_LAYERS,): the forest mean organic
            carbon content per layer.

        Raises:
            ValueError: If there are no HRUs currently under forest.
        """
        hydrology = self.model.hydrology
        area = hydrology.HRU.var.cell_area
        soc = hydrology.HRU.var.soil_organic_carbon_percentage

        forest_mask = land_use_type == FOREST
        if not forest_mask.any():
            raise ValueError(
                "Reforestation requires at least one HRU currently under "
                "forest to measure the basin forest soil from, but none were "
                "found."
            )
        return np.average(
            soc[:, forest_mask], axis=1, weights=area[forest_mask]
        ).astype(np.float32)

    def _compute_bulk_density_ratios(
        self, land_use_type: np.ndarray
    ) -> TwoDArrayFloat32:
        """Measure how forest bulk density differs from each other land use, per soil layer.

        For every soil layer and every land use type, the ratio between the
        area-weighted mean bulk density over HRUs currently under forest and
        the area-weighted mean over HRUs currently under that land use is
        computed. Converting an HRU to forest then multiplies its own bulk
        density by the ratio for its layer and its previous land use, so the
        HRU keeps its own soil signal instead of being replaced by a catchment
        average.

        Deriving the ratios from the model's own soil maps means no change
        factor is hard-coded, and the ratios are specific to the basin being
        modelled.

        Land use types with no HRUs, and layers whose mean is zero, get a ratio
        of 1.0 (no change), since there is nothing to measure a change against.
        At least one forest HRU must exist; this is checked by
        :meth:`_compute_forest_mean_soc`, which is called first.

        Args:
            land_use_type: Land use type per HRU, as it is just before conversion.

        Returns:
            Bulk density ratio of shape (N_SOIL_LAYERS, len(ALL_LAND_COVER_TYPES)),
            indexable by land use type, e.g. ``ratio[:, GRASSLAND_LIKE]``.
        """
        hydrology = self.model.hydrology
        area = hydrology.HRU.var.cell_area
        bulk_density = hydrology.HRU.var.bulk_density_kg_per_dm3

        n_layers: int = bulk_density.shape[0]
        bulk_density_ratio = np.ones(
            (n_layers, len(ALL_LAND_COVER_TYPES)), dtype=np.float32
        )

        forest_mask = land_use_type == FOREST
        forest_bulk_density = np.average(
            bulk_density[:, forest_mask], axis=1, weights=area[forest_mask]
        )

        for source_land_use in ALL_LAND_COVER_TYPES:
            source_mask = land_use_type == source_land_use
            if not source_mask.any():
                continue
            source_bulk_density = np.average(
                bulk_density[:, source_mask], axis=1, weights=area[source_mask]
            )
            bulk_density_ratio[:, source_land_use] = np.divide(
                forest_bulk_density,
                source_bulk_density,
                out=np.ones_like(forest_bulk_density),
                where=source_bulk_density > 0,
            )
            self.model.logger.info(
                "Reforestation bulk density ratio from land use %d (%d HRUs): "
                "per layer %s.",
                source_land_use,
                int(source_mask.sum()),
                np.round(bulk_density_ratio[:, source_land_use], 3).tolist(),
            )

        return bulk_density_ratio

    def _save_forest_planting_figure(
        self,
        land_use_type_before: np.ndarray,
        suitability_HRU: np.ndarray,
        output_folder: Path,
        threshold: float = 0.5,
    ) -> None:
        """Save a 4-panel reforestation scenario figure."""
        hydrology = self.model.hydrology
        catchment_gdf = read_geom(self.model.files["geom"]["mask"])

        bounds = catchment_gdf.total_bounds  # [minx, miny, maxx, maxy]
        extent = [
            bounds[0],
            bounds[2],
            bounds[1],
            bounds[3],
        ]  # [left, right, bottom, top]

        current_2d = hydrology.HRU.decompress(land_use_type_before.astype(np.float32))
        future_2d = hydrology.HRU.decompress(
            hydrology.HRU.var.land_use_type.astype(np.float32)
        )
        suitability_2d = hydrology.HRU.decompress(suitability_HRU.astype(np.float32))
        change_2d = (future_2d != current_2d).astype(np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        im1 = axes[0, 0].imshow(
            current_2d, cmap="tab20", interpolation="nearest", extent=extent
        )
        axes[0, 0].set_title("Current Land Cover")
        catchment_gdf.boundary.plot(
            ax=axes[0, 0], color="black", linewidth=2, alpha=0.8
        )
        fig.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(
            future_2d, cmap="tab20", interpolation="nearest", extent=extent
        )
        axes[0, 1].set_title("Future Land Cover (with Reforestation)")
        catchment_gdf.boundary.plot(
            ax=axes[0, 1], color="black", linewidth=2, alpha=0.8
        )
        fig.colorbar(im2, ax=axes[0, 1])

        im3 = axes[1, 0].imshow(
            suitability_2d,
            cmap="Greens",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent,
        )
        axes[1, 0].set_title(f"Reforestation Suitability ({threshold:.0%} threshold)")
        catchment_gdf.boundary.plot(
            ax=axes[1, 0], color="black", linewidth=2, alpha=0.8
        )
        cbar3 = fig.colorbar(im3, ax=axes[1, 0])
        cbar3.set_ticks([0, 1])
        cbar3.set_ticklabels(["Unsuitable", "Suitable"])

        im4 = axes[1, 1].imshow(
            change_2d,
            cmap="Reds",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent,
        )
        axes[1, 1].set_title("Converted Areas")
        catchment_gdf.boundary.plot(
            ax=axes[1, 1], color="black", linewidth=2, alpha=0.8
        )
        cbar4 = fig.colorbar(im4, ax=axes[1, 1])
        cbar4.set_ticks([0, 1])
        cbar4.set_ticklabels(["No Change", "Converted"])

        plt.suptitle("Reforestation Scenario Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            output_folder / "reforestation_scenario.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    def remove_farmers_from_converted_forest_areas(
        self, suitability_HRU: np.ndarray
    ) -> None:
        """Remove farmers from HRUs that are suitable for reforestation.

        Args:
            suitability_HRU: Boolean array at HRU scale (True = suitable for forest).
        """
        if not hasattr(self.agents, "crop_farmers"):
            return

        crop_farmers = self.agents.crop_farmers
        converted_HRU_indices = np.where(suitability_HRU)[0]
        if len(converted_HRU_indices) == 0:
            return

        land_owners = crop_farmers.HRU.var.land_owners[converted_HRU_indices]
        farmer_indices = land_owners[land_owners != -1]
        if len(farmer_indices) == 0:
            print("No farmers found in suitable areas, none removed")
            return

        unique_farmer_indices = np.unique(farmer_indices)
        farmers_before = crop_farmers.n
        crop_farmers.remove_agents(
            farmer_indices=unique_farmer_indices,
            new_land_use_type=FOREST,
        )
        print(
            f"Farmers removed: {len(unique_farmer_indices):,} ({farmers_before:,} → {crop_farmers.n:,})"
        )

    def adaptation(self) -> None:
        """Decide whether adaptation is needed and apply appropriate adaptation measures.

        Checks if adaptation is enabled and if it is January 1st, then calculates EAD,
        equity, and ecosystem indicators. If any thresholds are crossed, applies the corresponding
        adaptation measures (building floodproofing, subsidies, or reforestation).

        Raises:
            ValueError: If an invalid adaptation mode is specified in the configuration.
        """
        # something to specify that this should only run when adaptation is turned on in the config file
        # should this step be skipped during spinup?
        if not self.config["adaptation"]["enabled"]:
            return  # exits because adaptation is not (enabled) in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return  # exits because it is not the first of January

        if self.config["adaptation"]["mode"] == "cba":
            # for easier access to the flood risk module of the households agent, we store it here as well
            self.flood_risk_module = self.agents.households.flood_risk_module
            self._cost_benefit_adaptation()
            return

        elif self.config["adaptation"]["mode"] != "threshold":
            raise ValueError(
                f"Invalid adaptation mode: {self.config['adaptation']['mode']}. "
                "Supported modes are 'cba' and 'threshold'."
            )
        # calculate the water risk, equity and ecosystem health for the current year (adaptation is enabled and it is january first)
        EAD_value = self.calculate_EAD()  # this is defined by the EAD
        equity_indicator_value = (
            self.calculate_equity_indicator()
        )  # this is defined by the exposure inequalty
        ecosystem_indicator_value = (
            self.calculate_ecosystem_indicator()
        )  # this is defined by the ecosystem indicator

        # Set the indicator value and the thresholds for the adaptation decision.
        # but they should be defined in such a way that they trigger adaptation when threshold is crossed

        thresholds = {
            # defines the threshold --> value in current situation, threshold
            "EAD": (EAD_value, self.config["adaptation"]["EAD_threshold"]),
            "equity_indicator": (
                equity_indicator_value,
                self.config["adaptation"]["equity_indicator_threshold"],
            ),
            "ecosystem_indicator": (
                ecosystem_indicator_value,
                self.config["adaptation"]["ecosystem_indicator_threshold"],
            ),
        }

        # if any of these the thresholds are crossed, we adapt the policies so a make a list of the thresholds that are crossed
        # skip indicators where the value or threshold is None (not yet implemented)
        triggered = [
            key
            for key, value in thresholds.items()
            if value[0] is not None and value[1] is not None and value[0] > value[1]
        ]

        # if one of the threshold is triggered something needs to be done
        if triggered:
            print(
                f"Adaptation needed, the following thresholds are crossed: {triggered}"
            )
            self.apply_adaptation(
                triggered
            )  # we now know what has caused the need for adaptation, so now we move on and actually implement the adaptation through another function, what caused the trigger is passed to this function
        else:
            print(
                "No adaptation needed, all thresholds are below the defined thresholds"
            )

    def _apply_cumulative_time_discounting(
        self, value_to_discount: float, discount_rate: float = 0.1, years: int = 35
    ) -> float:
        """Return the cumulative time discounted value.

        Args:
            value_to_discount: The value to be discounted.
            discount_rate: The discount rate (as a decimal).
            years: The number of years into the future.

        Returns:
            The discounted value.
        """
        # Calculate time discounted NPVs
        t_arr = np.arange(1, years + 1, dtype=np.float32)
        discounts = 1 / (1 + discount_rate) ** t_arr
        discounted_value = np.sum(discounts) * value_to_discount
        return discounted_value

    def _cost_benefit_adaptation(
        self,
        indirect_damages: float = 1.6,
        model_removal_flood_protection_standards: bool = False,
    ) -> None:
        """Evaluate flood protection standard upgrade using cost-benefit analysis.

        Compares expected annual damage (EAD) at current and next flood protection
        standard level. If damage reduction exceeds threshold, upgrades the standard.
        Args:
            indirect_damages: Multiplier for indirect damages (default: 1.6).
            model_removal_flood_protection_standards: Whether to consider model removal of flood protection standards (default: False).
        """
        # if not self.flood_risk_module.flood_in_last_year:
        #     return

        return_periods = self.agents.households.return_periods

        damages_do_not_adapt = self.flood_risk_module.damages_do_not_adapt
        damages_adapt = self.flood_risk_module.damages_adapt
        adapted = self.agents.households.var.adapted.data

        # iterate over each subbasin in the model and calculate the EAD for the current and next flood protection standard
        dike_heights = self.flood_risk_module.dike_heights()
        for subbasin in dike_heights[
            next(iter(dike_heights))
        ]:  # get return period of 10 years
            if (
                subbasin
                not in self.flood_risk_module.flood_protection_standard_subbasins
            ):
                continue  # Skip subbasins without a defined flood protection standard
            current_fps = self.flood_risk_module.flood_protection_standard_subbasins[
                subbasin
            ]
            idx_fps = np.where(return_periods == current_fps)[0]

            if len(idx_fps) == 0 or idx_fps[0] >= len(return_periods) - 1:
                continue  # Cannot upgrade further

            altered_fps = return_periods[idx_fps[0] + 1]
            households = np.where(
                self.agents.households.comid_of_household == subbasin
            )[0]

            # Calculate EAD at current and higher protection standard
            current_ead = self.flood_risk_module.calculate_ead(
                damages_do_not_adapt[:, households],
                damages_adapt[:, households],
                adapted[households],
            ).sum()

            altered_ead = self.flood_risk_module.calculate_ead(
                damages_do_not_adapt[:, households],
                damages_adapt[:, households],
                adapted[households],
                altered_fps,
            ).sum()

            # calculate the cost of raising the dike to the next flood protection standard
            dike_heights_current_fps = dike_heights[current_fps][subbasin]
            dike_heights_altered_fps = dike_heights[altered_fps][subbasin]
            damage_reduction = current_ead - altered_ead
            # account for indirect damages
            damage_reduction *= indirect_damages

            # get total length and height difference of the dikes that need to be raised
            height_difference = dike_heights_altered_fps - dike_heights_current_fps
            if height_difference.sum() == 0:
                continue
            cost_per_meter = self.config["adaptation"][
                "dike_elevation_cost_per_meter_usd"
            ]
            maintenance_cost_per_km_dike = self.config["adaptation"][
                "dike_maintenance_cost_per_year_usd"
            ]
            total_cost = (
                np.sum(height_difference * 100 * cost_per_meter) * 2
            )  # investment cost in euros; segments are roughly 100 meters long, double the cost to account for both sides of the dike

            maintenance_cost_per_year = (
                maintenance_cost_per_km_dike * height_difference.size * 100 * 2
            )  # maintenance cost per year in euros; segments are roughly 100 meters long, double the cost to account for both sides of the dike

            if model_removal_flood_protection_standards:
                ead_no_flood_protection_standard = self.flood_risk_module.calculate_ead(
                    damages_do_not_adapt[:, households],
                    damages_adapt[:, households],
                    adapted[households],
                    0,  # set flood protection standard to 0 to calculate EAD without any flood protection
                ).sum()
                # also calculate the damage reduction of current fps compared to no fps
                damage_reduction_no_fps = ead_no_flood_protection_standard - current_ead
                damage_reduction_no_fps *= indirect_damages

                if maintenance_cost_per_year > damage_reduction_no_fps:
                    self.model.logger.info(
                        "Maintenance cost per year for subbasin %s exceeds damage reduction of current flood protection standard. Removing flood protection.",
                        subbasin,
                    )
                    self.flood_protection_standard_subbasins[subbasin] = 0
                    continue

            if self._apply_cumulative_time_discounting(
                damage_reduction
            ) > total_cost + self._apply_cumulative_time_discounting(
                maintenance_cost_per_year
            ):
                self.flood_protection_standard_subbasins[subbasin] = altered_fps
                self.model.logger.info(
                    "Upgraded flood protection standard for subbasin %s from %d to %d",
                    subbasin,
                    current_fps,
                    altered_fps,
                )

    def calculate_EAD(self) -> None | float:
        """Calculate the expected annual damage (EAD) for the current year.

        EAD is computed by integrating total flood damage over the exceedance
        probability curve (trapezoid rule across return periods).

        Returns:
            EAD in euros, or None if adaptation is disabled, it is not January 1st,
            or flood risk functionality is not available.
        """
        # should also only be calculated if adaptation is turned on in the config file and it is the first of jan otherwise it is not needed
        if not self.config["adaptation"]["enabled"]:
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return

        households = self.agents.households
        if not hasattr(households, "flood_risk_module"):
            return None
        return_periods = households.return_periods

        total_damage_per_rp = np.zeros(len(return_periods), dtype=np.float64)
        for i, return_period in enumerate(return_periods):
            flood_map = households.flood_maps[return_period]
            total_damage_per_rp[i] = households.flood_risk_module.flood(flood_map)

        # sort by descending return period → ascending exceedance probability for trapezoid
        sorted_rp = np.argsort(return_periods)[::-1]
        exceedance_probabilities = 1.0 / return_periods[sorted_rp]
        sorted_damages = total_damage_per_rp[sorted_rp]

        EAD = np.trapezoid(sorted_damages, x=exceedance_probabilities)
        return EAD

    def calculate_equity_indicator(self) -> None | float:
        # should also only be calculated if adaptation is turned on in the config file otherwise it is not needed
        """Calculate the equity for the current year.

        Returns:
         the equity indicator value.
        """
        if not self.config["adaptation"]["enabled"]:
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return
        # this is defined by the exposure inequalty
        equity_index = 0.7  # this should become an actual calculation
        return equity_index

    def calculate_ecosystem_indicator(self) -> None | float:
        # should also only be calculated if adaptation is turned on in the config file otherwise it is not needed
        """Calculate the ecosystem health for the current year.

        Returns:
        the ecosystem indicator value.
        """
        if not self.config["adaptation"]["enabled"]:
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return

        # assign values per landuse type (maybe store in dictionary) (what do we base this on, tbd find literature)
        from geb.hydrology.landcovers import (
            FOREST,
            GRASSLAND_LIKE,
            NON_PADDY_IRRIGATED,
            OPEN_WATER,
            PADDY_IRRIGATED,
            SEALED,
        )

        values_per_land_use_type = {
            FOREST: 1.0,  # highest ecosystem quality
            GRASSLAND_LIKE: 0.7,
            OPEN_WATER: 0.6,
            PADDY_IRRIGATED: 0.3,
            NON_PADDY_IRRIGATED: 0.3,
            SEALED: 0.0,  # lowest ecosystem quality
        }

        # land use type per HRU and also area
        land_use_type_per_HRU = self.model.hydrology.HRU.var.land_use_type
        area_per_HRU = self.model.hydrology.HRU.var.cell_area

        # multiply the values with the area of each land use type
        # sum these
        weighted_sum = sum(
            np.sum(area_per_HRU[land_use_type_per_HRU == land_use_type]) * value
            for land_use_type, value in values_per_land_use_type.items()
        )

        # divide by total catchment area
        total_catchment_area = area_per_HRU.sum()
        ecosystem_indicator = weighted_sum / total_catchment_area

        return ecosystem_indicator

    def apply_adaptation(self, triggered: list) -> None:
        """Apply the adaptation measures decided in the adaptation function.

        Args:
            triggered: List of adaptation triggers that were activated.

        Raises:
            ValueError: If ``adaptation_fraction`` is not in [0, 1].
        """
        if not self.config["adaptation"]["enabled"]:
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return  # exits because it is not the first of January

        if "EAD" in triggered:
            # the government decides which measure to apply based on what triggered the need for adaptation
            # apply updating the building structure but this takes the number of households that are adapting as input so we fist need to define that
            adaptation_fraction = self.config["adaptation"]["adaptation_fraction"]
            if not (0.0 <= adaptation_fraction <= 1.0):
                raise ValueError(
                    f"adaptation_fraction must be in [0, 1], got {adaptation_fraction}"
                )
            # figure out which buildings are marked as flooded in this year
            household_agents = self.agents.households
            flooded_buildings_mask = set(
                household_agents.buildings.loc[
                    household_agents.buildings["flooded"] == True, "id"
                ].astype(int)
            )
            # figure out which households are in these flooded buildings
            households_in_flooded_buildings = np.where(
                pd.Series(household_agents.var.building_id_of_household.data).isin(
                    flooded_buildings_mask
                )
            )[0]
            # the government decides who of those are adapting, we only pick a fraction as specified in the config file
            n_to_adapt = min(
                int(len(households_in_flooded_buildings) * adaptation_fraction),
                len(households_in_flooded_buildings),
            )
            adapting_households_sample = np.random.choice(
                households_in_flooded_buildings, size=n_to_adapt, replace=False
            )
            # use the function to floodproof the buildings of the households who are selected to adapt.
            household_agents.update_building_adaptation_status(
                adapting_households_sample
            )
            print(
                f"the government adapted {n_to_adapt} of the {len(households_in_flooded_buildings)} households in the floodzone by floodproofing their buildings"
            )

        # if "equity_indicator" in triggered:
        #     # apply subsidies" --> maybe we can change who the subsidies are applied to but idk if that would make it better
        #     # this piece of code is still in Veerle's branch so can be uncommented when merged
        #     self.provide_subsidies()

        if "ecosystem_indicator" in triggered:
            # apply reforestation --> to do: but maybe a percentage of the suitable areas instead of all suitable areas
            # the threshold for the reforestation amount can be set in the config file under forest_reforestation_potential_threshold, set to 0.9 to convert only the top 10% suitable areas in the model.yml
            self.prepare_modified_soil_maps_for_forest()

    @property
    def flood_protection_standard_subbasins(self) -> dict[int, int]:
        """Get the flood protection standard for each subbasin.

        Returns:
            A dictionary mapping subbasin IDs to their flood protection standards.
        """
        return (
            self.agents.households.flood_risk_module.flood_protection_standard_subbasins
        )
