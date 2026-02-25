"""This module contains the Government agent class for GEB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from geb.hydrology.landcovers import FOREST
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
        """Initialize the government agent.

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

        # Flag to handle forest planting at timestep 0
        self.forest_planting_done = False

    @property
    def name(self) -> str:
        """Name of the module.

        Returns:
            The name of the module.
        """
        return "agents.government"

    def spinup(self) -> None:
        """This function is called during model spinup."""
        pass

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

    def step(self) -> None:
        """This function is run each timestep."""
        if (
            not self.forest_planting_done
            and self.config.get("plant_forest", False)
            and self.model.current_timestep == 0
        ):
            logger.info("FOREST PLANTING: Starting farmer removal at timestep 0...")

            logger.info("GOVERNMENT: Starting forest planting workflow at timestep 0")

            # Run the complete forest planting workflow at timestep 0
            # Capture before state
            farmers_before = self.model.agents.crop_farmers.n
            # Get current planting info by accessing crop map
            crop_map_before = self.model.hydrology.HRU.var.crop_map.copy()
            planted_fields_before = np.count_nonzero(crop_map_before >= 0)
            crops_before = (
                np.unique(crop_map_before[crop_map_before >= 0])
                if planted_fields_before > 0
                else np.array([])
            )

            self.prepare_modified_soil_maps_for_forest()

            # Capture after state
            farmers_after = self.model.agents.crop_farmers.n
            crop_map_after = self.model.hydrology.HRU.var.crop_map.copy()
            planted_fields_after = np.count_nonzero(crop_map_after >= 0)
            crops_after = (
                np.unique(crop_map_after[crop_map_after >= 0])
                if planted_fields_after > 0
                else np.array([])
            )

            # Summary
            logger.info("FOREST PLANTING WORKFLOW SUMMARY:")
            logger.info(
                f"FARMERS: {farmers_before} → {farmers_after} (removed: {farmers_before - farmers_after})"
            )
            logger.info(
                f"FIELDS PLANTED: {planted_fields_before} → {planted_fields_after} (change: {planted_fields_after - planted_fields_before})"
            )

            self.forest_planting_done = True

            logger.info("GOVERNMENT: Forest planting workflow complete")

        self.set_irrigation_limit()

        self.report(locals())

    def create_forest_suitability_from_potential(
        self,
        potential_path: str | Path,
        template_da: xr.DataArray,
        threshold_ratio: float = 0.5,
    ) -> xr.DataArray:
        """Create a binary suitability map from forest restoration potential dataset.

        Uses a simple threshold approach based on restoration potential ratio.

        Args:
            potential_path: Path to the forest_restoration_potential_ratio dataset file.
            template_da: Template DataArray defining target grid, CRS, and coordinates.
            threshold_ratio: Minimum restoration potential ratio to consider suitable (default: 0.5).

        Returns:
            Binary suitability map (1=suitable, 0=unsuitable) on the template grid.

        """
        logger.info("Creating Suitability Map from Forest Restoration Potential")

        potential_path = Path(potential_path)

        logger.info(f"Loading forest restoration potential from: {potential_path}")

        # Load dataset and compute it to avoid dask issues
        potential_ds = xr.open_zarr(potential_path, consolidated=False)
        potential_data = potential_ds["forest_restoration_potential_ratio"].compute()

        # Set CRS to match template if missing
        if potential_data.rio.crs is None:
            potential_data = potential_data.rio.write_crs(template_da.rio.crs)
            logger.info(f"  Set CRS to match template: {template_da.rio.crs}")

        logger.info(f"  Potential data shape: {potential_data.shape}")
        logger.info(f"  Potential data CRS: {potential_data.rio.crs}")

        # Compute min/max values safely
        potential_min = float(potential_data.min())
        potential_max = float(potential_data.max())
        logger.info(
            f"  Potential ratio range: [{potential_min:.4f}, {potential_max:.4f}]"
        )
        if (
            potential_data.rio.crs != template_da.rio.crs
            or potential_data.shape != template_da.shape
        ):
            logger.info(f"\nReprojecting to template CRS: {template_da.rio.crs}")
            potential_reprojected = potential_data.rio.reproject_match(template_da)
            logger.info(f"  Reprojected shape: {potential_reprojected.shape}")
        else:
            logger.info("\nNo reprojection needed - grids already match")
            potential_reprojected = potential_data

        # Apply threshold to create binary suitability map
        logger.info(f"\nApplying threshold: potential ratio >= {threshold_ratio}")
        suitability_binary = (potential_reprojected >= threshold_ratio).astype(np.int8)

        # Calculate statistics
        total_pixels = int((~np.isnan(potential_reprojected)).sum())
        suitable_pixels = int(suitability_binary.sum())
        suitability_percentage = (
            (suitable_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
        )

        logger.info(f"\nSuitability map statistics:")
        logger.info(f"  Total valid pixels: {total_pixels:,}")
        logger.info(f"  Suitable pixels: {suitable_pixels:,}")
        logger.info(f"  Suitability percentage: {suitability_percentage:.2f}%")

        return suitability_binary

    def modify_bulk_density_for_forest_relative(
        self,
        bulk_density_ds: xr.Dataset,
        suitability: xr.DataArray,
        reduction_factor: float = 0.85,
    ) -> xr.Dataset:
        """Modify bulk density using a relative reduction factor.

        Applies a multiplicative factor to reduce bulk density in suitable areas,
        matching the notebook's approach of 15% reduction (0.85x).

        Args:
            bulk_density_ds: Original bulk density dataset (kg/m³).
            suitability: Binary suitability map (1=suitable, 0=unsuitable).
            reduction_factor: Multiplicative factor for reduction (default: 0.85 for 15% reduction).

        Returns:
            Modified bulk density dataset with reduced values in suitable areas.
        """
        logger.info("\n" + "-" * 80)
        logger.info("BULK DENSITY: Applying RELATIVE modification")
        logger.info("-" * 80)

        modified_ds = bulk_density_ds.copy(deep=True)

        for layer_idx in range(bulk_density_ds.sizes["soil_layer"]):
            layer_data = bulk_density_ds["bulk_density_kg_per_dm3"].isel(
                soil_layer=layer_idx
            )

            original_mean = float(layer_data.mean())
            original_min = float(layer_data.min())
            original_max = float(layer_data.max())

            # Apply relative reduction: new = old * reduction_factor
            modified_layer = xr.where(
                suitability == 1, layer_data * reduction_factor, layer_data
            )

            # Use direct array assignment instead of .loc indexing
            modified_ds["bulk_density_kg_per_dm3"].values[layer_idx, :, :] = (
                modified_layer.values
            )

            modified_mean = float(modified_layer.mean())
            modified_min = float(modified_layer.min())
            modified_max = float(modified_layer.max())
            absolute_change = modified_mean - original_mean
            percent_change = (
                (absolute_change / original_mean * 100) if original_mean != 0 else 0.0
            )

            logger.info(f"\n  Layer {layer_idx}:")
            logger.info(
                f"    Reduction factor: {reduction_factor} ({(1 - reduction_factor) * 100:.0f}% reduction)"
            )
            logger.info(
                f"    Original  - Mean: {original_mean:.2f}, Min: {original_min:.2f}, Max: {original_max:.2f} kg/m³"
            )
            logger.info(
                f"    Modified  - Mean: {modified_mean:.2f}, Min: {modified_min:.2f}, Max: {modified_max:.2f} kg/m³"
            )
            logger.info(
                f"    Change    - Absolute: {absolute_change:.2f} kg/m³, Relative: {percent_change:.2f}%"
            )

        return modified_ds

    def analyze_soil_by_landcover_for_forest(
        self,
        soil_ds: xr.Dataset,
        landcover: xr.DataArray,
        variable_name: str,
        target_class: int = 10,
    ) -> dict:
        """Analyze soil characteristics grouped by land cover class.

        Args:
            soil_ds: Soil dataset containing the variable to analyze.
            landcover: Land cover classification array.
            variable_name: Name of the variable in soil_ds to analyze.
            target_class: Forest class value in ESA WorldCover (default: 10).

        Returns:
            Dictionary with statistics (mean, min, max, std) per land cover class.
        """
        logger.info(f"\nAnalyzing {variable_name} by land cover class:")

        stats_by_class = {}
        unique_classes = np.unique(landcover.values[~np.isnan(landcover.values)])

        for lc_class in unique_classes:
            lc_class_int = int(lc_class)
            mask = landcover == lc_class_int

            # Get stats across all soil layers
            masked_values = soil_ds[variable_name].where(mask, drop=False)

            stats_by_class[lc_class_int] = {
                "mean": float(masked_values.mean()),
                "min": float(masked_values.min()),
                "max": float(masked_values.max()),
                "std": float(masked_values.std()),
                "count": int(mask.sum()),
            }

            class_label = f"Class {lc_class_int}"
            if lc_class_int == target_class:
                class_label += " (FOREST TARGET)"

            logger.info(f"  {class_label}:")
            logger.info(f"    Mean: {stats_by_class[lc_class_int]['mean']:.4f}")
            logger.info(f"    Min: {stats_by_class[lc_class_int]['min']:.4f}")
            logger.info(f"    Max: {stats_by_class[lc_class_int]['max']:.4f}")
            logger.info(f"    Std: {stats_by_class[lc_class_int]['std']:.4f}")
            logger.info(f"    Cell count: {stats_by_class[lc_class_int]['count']:,}")

        return stats_by_class

    def modify_soil_characteristics_for_forest(
        self,
        soil_ds: xr.Dataset,
        landcover_resampled: xr.DataArray,
        suitability: xr.DataArray,
        stats_by_class: dict,
        variable_name: str,
        strategy: str = "mean",
        target_class: int = 10,
    ) -> xr.Dataset:
        """Modify soil characteristics in suitable areas based on forest class statistics.

        Args:
            soil_ds: Original soil dataset.
            landcover_resampled: Resampled land cover classification.
            suitability: Binary suitability map (1=suitable, 0=unsuitable).
            stats_by_class: Statistics dictionary from analyze_soil_by_landcover.
            variable_name: Name of the variable to modify.
            strategy: Modification strategy - "mean", "min", or "max" (default: "mean").
            target_class: Forest class value (default: 10).

        Returns:
            Modified soil dataset.

        Raises:
            ValueError: If target_class not found in stats or invalid strategy.
        """
        logger.info(
            f"\n{variable_name.upper()}: Applying {strategy.upper()} modification from forest class {target_class}"
        )

        if target_class not in stats_by_class:
            raise ValueError(
                f"Target class {target_class} not found in landcover statistics"
            )

        if strategy not in ["mean", "min", "max"]:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be 'mean', 'min', or 'max'"
            )

        forest_value = stats_by_class[target_class][strategy]
        logger.info(f"  Using forest {strategy} value: {forest_value:.4f}")

        modified_ds = soil_ds.copy(deep=True)

        for layer_idx in range(soil_ds.sizes["soil_layer"]):
            layer_data = soil_ds[variable_name].isel(soil_layer=layer_idx)

            original_mean = float(layer_data.mean())

            # Apply modification only in suitable areas
            modified_layer = xr.where(suitability == 1, forest_value, layer_data)

            # Use dict indexing like in the notebook instead of .loc
            modified_ds[variable_name].values[layer_idx, :, :] = modified_layer.values

            modified_mean = float(modified_layer.mean())
            change = modified_mean - original_mean
            pct_change = (change / original_mean * 100) if original_mean != 0 else 0.0

            logger.info(
                f"  Layer {layer_idx}: Original mean={original_mean:.4f}, Modified mean={modified_mean:.4f}, Change={change:.4f} ({pct_change:.2f}%)"
            )

        return modified_ds

    def create_future_landcover_with_forest(
        self, landcover: xr.DataArray, suitability: xr.DataArray, target_class: int = 10
    ) -> xr.DataArray:
        """Create future land cover scenario by converting suitable areas to forest.

        Args:
            landcover: Current land cover classification.
            suitability: Binary suitability map (1=suitable for conversion, 0=not suitable).
            target_class: Forest class value in ESA WorldCover (default: 10).

        Returns:
            Future land cover with suitable areas converted to forest class.
        """
        logger.info("\n" + "=" * 80)
        logger.info(
            f"Creating Future Land Cover Scenario (converting to class {target_class})"
        )
        logger.info("=" * 80)

        # Count conversions
        n_conversions = int((suitability == 1).sum())
        n_total = int((~np.isnan(landcover)).sum())
        pct_converted = (n_conversions / n_total * 100) if n_total > 0 else 0.0

        logger.info("Conversion statistics:")
        logger.info(f"  Total cells: {n_total:,}")
        logger.info(f"  Cells to convert: {n_conversions:,}")
        logger.info(f"  Percentage converted: {pct_converted:.2f}%")

        # Apply conversion
        future_landcover = xr.where(suitability == 1, target_class, landcover)

        # Verify conversion
        n_forest_before = int((landcover == target_class).sum())
        n_forest_after = int((future_landcover == target_class).sum())
        n_new_forest = n_forest_after - n_forest_before

        logger.info(f"\nForest class {target_class} statistics:")
        logger.info(f"  Before: {n_forest_before:,} cells")
        logger.info(f"  After: {n_forest_after:,} cells")
        logger.info(f"  New forest: {n_new_forest:,} cells")

        return future_landcover

    def plot_reforestation_scenario(
        self,
        current_landcover: xr.DataArray,
        future_landcover: xr.DataArray,
        suitability: xr.DataArray,
        output_path: str | Path | None = None,
    ) -> None:
        """Create visualization plots for the reforestation scenario.

        Generates a 2x2 grid showing:
        - Top left: Current land cover
        - Top right: Future land cover
        - Bottom left: Suitability map
        - Bottom right: Change map (converted areas)
        All plots include catchment boundary overlay.

        Args:
            current_landcover: Current land cover classification.
            future_landcover: Future land cover after forest conversion.
            suitability: Binary suitability map.
            output_path: Optional path to save the figure (default: None, shows plot).
        """
        # Load catchment boundary
        catchment_gdf = read_geom(self.model.files["geom"]["mask"])

        # Downsample arrays by factor of 10 for faster plotting
        current_downsampled = current_landcover.values[::10, ::10]
        future_downsampled = future_landcover.values[::10, ::10]
        suitability_downsampled = suitability.values[::10, ::10]

        # Calculate extent for proper georeferenced plotting
        x_min, x_max = (
            float(current_landcover.x.min()),
            float(current_landcover.x.max()),
        )
        y_min, y_max = (
            float(current_landcover.y.min()),
            float(current_landcover.y.max()),
        )
        extent = [x_min, x_max, y_min, y_max]

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Top left: Current landcover
        im1 = axes[0, 0].imshow(
            current_downsampled,
            cmap="tab20",
            interpolation="nearest",
            extent=extent,
            origin="upper",
        )
        axes[0, 0].set_title("Current Land Cover", fontsize=14, fontweight="bold")
        catchment_gdf.boundary.plot(
            ax=axes[0, 0], color="black", linewidth=2, alpha=0.8
        )
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # Top right: Future landcover
        im2 = axes[0, 1].imshow(
            future_downsampled,
            cmap="tab20",
            interpolation="nearest",
            extent=extent,
            origin="upper",
        )
        axes[0, 1].set_title(
            "Future Land Cover (with Reforestation)", fontsize=14, fontweight="bold"
        )
        catchment_gdf.boundary.plot(
            ax=axes[0, 1], color="black", linewidth=2, alpha=0.8
        )
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Bottom left: Suitability map
        im3 = axes[1, 0].imshow(
            suitability_downsampled,
            cmap="Greens",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            extent=extent,
            origin="upper",
        )
        axes[1, 0].set_title(
            "Reforestation Suitability (50% threshold)", fontsize=14, fontweight="bold"
        )
        catchment_gdf.boundary.plot(
            ax=axes[1, 0], color="black", linewidth=2, alpha=0.8
        )
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cbar3.set_ticks([0, 1])
        cbar3.set_ticklabels(["Unsuitable", "Suitable"])

        # Bottom right: Change map
        change_map = (future_downsampled != current_downsampled).astype(int)
        im4 = axes[1, 1].imshow(
            change_map,
            cmap="Reds",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            extent=extent,
            origin="upper",
        )
        axes[1, 1].set_title("Converted Areas", fontsize=14, fontweight="bold")
        catchment_gdf.boundary.plot(
            ax=axes[1, 1], color="black", linewidth=2, alpha=0.8
        )
        cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar4.set_ticks([0, 1])
        cbar4.set_ticklabels(["No Change", "Converted"])

        plt.suptitle(
            "Reforestation Scenario Analysis", fontsize=16, fontweight="bold", y=0.995
        )
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("  Saving plot (using reduced DPI for speed)...")
            plt.savefig(
                output_path, dpi=150, bbox_inches="tight"
            )  # Reduced from 300 for speed
            logger.info(f"Saved plot to: {output_path}")
        else:
            plt.show()

        plt.close()

    def reload_hydrology_soil_properties(self) -> None:
        """Reload soil properties in hydrology after modifying soil files.

        Re-read the modified soil maps.
        """
        logger.info("HYDROLOGY: Reloading soil properties from modified files...")

        # Call the setup_soil_properties method again to reload from files
        self.model.hydrology.landsurface.setup_soil_properties()

        logger.info("HYDROLOGY: Soil properties reloaded successfully!")
        logger.info("Hydrology will now use modified soil maps")

    def prepare_modified_soil_maps_for_forest(self) -> None:
        """Forest planting workflow - only modify files and reload hydrology.

        This approach is much simpler:
        1. Analyze soil characteristics by land cover
        2. Create suitability map
        3. Directly update hydrology soil arrays in memory
        4. Remove farmers from converted areas
        5. Create visualization (optional)
        6. Skip file saving and model file path updates for simplicity
        7. Skip future land cover scenario creation for simplicity
        """
        logger.info("WORKFLOW: Running complete forest planting workflow...")

        # Load restored forest potential from model files
        forest_potential_path = self.model.files["grid"][
            "landsurface/forest_restoration_potential_ratio"
        ]
        logger.info(f"Forest restoration potential path: {forest_potential_path}")

        # Load soil data using model.files paths
        bulk_density_path = self.model.files["subgrid"]["soil/bulk_density_kg_per_dm3"]
        soc_path = self.model.files["subgrid"]["soil/soil_organic_carbon_percentage"]
        bulk_density_ds = xr.open_zarr(bulk_density_path)
        soc_ds = xr.open_zarr(soc_path)

        # Load landcover
        landcover_path = self.model.files["other"]["landcover/classification"]
        landcover_ds = xr.open_zarr(landcover_path)
        landcover = landcover_ds["classification"]

        # Use first soil layer as template
        template_da = bulk_density_ds["bulk_density_kg_per_dm3"].isel(soil_layer=0)

        # Simple CRS handling for reprojection
        if landcover.rio.crs is None and hasattr(self.model, "hydrology"):
            if hasattr(self.model.hydrology, "grid") and hasattr(
                self.model.hydrology.grid, "crs"
            ):
                landcover = landcover.rio.write_crs(self.model.hydrology.grid.crs)
                template_da = template_da.rio.write_crs(self.model.hydrology.grid.crs)

        logger.info("Creating suitability map from forest restoration potential...")
        suitability = self.create_forest_suitability_from_potential(
            potential_path=forest_potential_path,
            template_da=template_da,
            threshold_ratio=0.5,
        )
        logger.info(
            f"Suitability map created. Shape: {suitability.shape}, Suitable pixels: {(suitability == 1).sum().item():,}"
        )

        logger.info("Resampling land cover to match soil grid...")
        landcover_resampled = landcover.rio.reproject_match(template_da, resampling=0)
        logger.info(
            f"Landcover resampled from {landcover.shape} to {landcover_resampled.shape}"
        )

        logger.info("Creating future landcover scenario with forest conversions...")
        future_landcover = self.create_future_landcover_with_forest(
            landcover=landcover_resampled, suitability=suitability, target_class=10
        )
        n_converted = (
            ((future_landcover == 10) & (landcover_resampled != 10)).sum().item()
        )
        logger.info(
            f"Future scenario created. Pixels converted to forest: {n_converted:,}"
        )

        # Store for farmer removal
        self._landcover_resampled = landcover_resampled

        logger.info("Analyzing soil characteristics by land cover class...")
        soc_stats = self.analyze_soil_by_landcover_for_forest(
            soil_ds=soc_ds,
            landcover=landcover_resampled,
            variable_name="soil_organic_carbon_percentage",
            target_class=10,
        )
        logger.info(f"SOC analysis complete. Forest mean: {soc_stats[10]['mean']:.4f}")

        bulk_density_stats = self.analyze_soil_by_landcover_for_forest(
            soil_ds=bulk_density_ds,
            landcover=landcover_resampled,
            variable_name="bulk_density_kg_per_dm3",
            target_class=10,
        )
        logger.info(
            f"Bulk density analysis complete. Forest mean: {bulk_density_stats[10]['mean']:.4f}"
        )

        logger.info("Modifying soil files for converted areas...")

        # Get forest soil characteristics from analysis
        forest_soc_mean = soc_stats[10]["mean"]  # Class 10 = forest
        forest_bd_mean = bulk_density_stats[10]["mean"]

        logger.info(f"Applying forest SOC: {forest_soc_mean:.4f}%")
        logger.info(f"Applying forest bulk density: {forest_bd_mean:.4f} kg/m³")

        # Create modified SOC map - replace areas where suitability = 1 with forest values
        soc_variable = soc_ds["soil_organic_carbon_percentage"]
        soc_modified = soc_variable.where(suitability != 1, forest_soc_mean)

        # Create modified bulk density map
        bd_variable = bulk_density_ds["bulk_density_kg_per_dm3"]
        bulk_density_modified = bd_variable.where(suitability != 1, forest_bd_mean)

        # Create modified file paths
        soc_modified_path = str(soc_path).replace(".zarr", "_modified.zarr")
        bd_modified_path = str(bulk_density_path).replace(".zarr", "_modified.zarr")

        # Save modified SOC - dataset key must match the filename stem
        soc_modified_ds = soc_ds.copy()
        soc_modified_ds = soc_modified_ds.drop_vars("soil_organic_carbon_percentage")
        soc_modified_ds["soil_organic_carbon_percentage_modified"] = soc_modified
        soc_modified_ds.to_zarr(soc_modified_path, mode="w")

        # Save modified bulk density - dataset key must match the filename stem
        bd_modified_ds = bulk_density_ds.copy()
        bd_modified_ds = bd_modified_ds.drop_vars("bulk_density_kg_per_dm3")
        bd_modified_ds["bulk_density_kg_per_dm3_modified"] = bulk_density_modified
        bd_modified_ds.to_zarr(bd_modified_path, mode="w")

        # Update model file paths to use modified versions
        self.model.files["subgrid"]["soil/soil_organic_carbon_percentage"] = Path(
            soc_modified_path
        )
        self.model.files["subgrid"]["soil/bulk_density_kg_per_dm3"] = Path(
            bd_modified_path
        )
        self.reload_hydrology_soil_properties()

        output_folder = self.model.output_folder / "forest_planting"
        output_folder.mkdir(parents=True, exist_ok=True)
        plot_output_path = output_folder / "reforestation_scenario.png"
        self.plot_reforestation_scenario(
            current_landcover=landcover_resampled,
            future_landcover=future_landcover,
            suitability=suitability,
            output_path=plot_output_path,
        )
        logger.info(f"Visualization saved to: {plot_output_path}")

        self.remove_farmers_from_converted_forest_areas(future_landcover)

    def remove_farmers_from_converted_forest_areas(
        self, future_landcover: xr.DataArray
    ) -> None:
        """Remove farmers from areas that have been converted to forest.

        Args:
            future_landcover: Future land cover scenario with forest conversions.

        Raises:
            RuntimeError: If no landcover_resampled is found in storage.
        """
        if not hasattr(self.agents, "crop_farmers"):
            logger.warning("No crop farmers agent found - skipping farmer removal")
            return

        logger.info("Analyzing converted areas and removing farmers where needed")
        logger.info("Removing farmers from converted forest areas...")

        crop_farmers = self.agents.crop_farmers

        logger.info(f"Total farmers in model: {crop_farmers.n}")

        # Use the resampled landcover that was used to create the future scenario
        # This ensures both arrays have the same shape and resolution
        if not hasattr(self, "_landcover_resampled"):
            raise RuntimeError(
                "Resampled landcover not found. Ensure prepare_modified_soil_maps() was called first."
            )

        original_landcover = self._landcover_resampled.values

        # Find locations where landcover changed to forest (class 10 = tree cover in ESA WorldCover)
        converted_mask = (future_landcover.values == 10) & (original_landcover != 10)

        if not converted_mask.any():
            logger.info("No areas converted to forest, no farmers to remove")
            return

        n_converted_cells = converted_mask.sum()
        logger.info(f"Found {n_converted_cells:,} grid cells converted to forest")

        # Convert subgrid mask to HRU scale
        # Use "last" method since we're working with a binary mask
        logger.info("Converting grid mask to HRU scale")
        converted_mask_HRU = crop_farmers.HRU.convert_subgrid_to_HRU(
            converted_mask.astype(np.int32), method="last"
        )

        # Find HRUs that were converted to forest
        converted_HRU_indices = np.where(converted_mask_HRU == 1)[0]

        if len(converted_HRU_indices) == 0:
            logger.info("No HRUs converted to forest")
            return

        # Get land owners for converted HRUs
        land_owners_in_converted = crop_farmers.HRU.var.land_owners[
            converted_HRU_indices
        ]

        # Filter out HRUs with no owner (-1)
        farmer_indices_in_converted = land_owners_in_converted[
            land_owners_in_converted != -1
        ]

        if len(farmer_indices_in_converted) == 0:
            logger.info("No farmers found in converted areas")
            return

        # Get unique farmer indices
        unique_farmer_indices = np.unique(farmer_indices_in_converted)
        n_farmers_to_remove = len(unique_farmer_indices)

        logger.info(f"Removing {n_farmers_to_remove:,} farmers from converted areas")

        # Remove farmers using crop_farmers' remove_agents method
        # The new land use type should be FOREST (hydrology module constant = 0)
        logger.info("Removing farmers (this may take a moment)")
        removed_HRUs = crop_farmers.remove_agents(
            farmer_indices=unique_farmer_indices,
            new_land_use_type=FOREST,
        )

        logger.info(f"Successfully removed {n_farmers_to_remove:,} farmers")
        logger.info(f"Total HRUs disowned: {len(removed_HRUs):,}")
        print(f"  Successfully removed {n_farmers_to_remove:,} farmers")
        print(f"  Total HRUs disowned: {len(removed_HRUs):,}")

        # Safe access to remaining farmer count
        if hasattr(crop_farmers.var, "n"):
            print(f"  Remaining farmers: {crop_farmers.n}")
        else:
            print("  Remaining farmers: (count unavailable after removal)")
