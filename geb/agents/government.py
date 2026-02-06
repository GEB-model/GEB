"""This module contains the Government agent class for GEB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rioxarray
import xarray as xr

from geb.hydrology.landcovers import FOREST

from .general import AgentBaseClass

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel

logger = logging.getLogger(__name__)


def prepare_modified_soil_maps_standalone(model: GEBModel) -> None:
    """Prepare modified soil maps for forest planting scenario (standalone version).

    This function is called during model initialization, BEFORE hydrology is created.
    It creates a temporary Government instance and calls the existing working method.

    Args:
        model: The GEB model instance.
    """
    logger.info("\n" + "=" * 80)
    logger.info("FOREST PLANTING SOIL MODIFICATION (PRE-HYDROLOGY)")
    logger.info("=" * 80)

    # Create a minimal temporary Government instance to reuse the existing working method
    class TempGovernment(Government):
        def __init__(self, model_inst: GEBModel) -> None:
            # Initialize without calling super().__init__ to avoid agent dependencies
            self.model = model_inst
            self.agents = None  # Won't be used during soil preparation
            self.config = (
                model_inst.config["agent_settings"]["government"]
                if "government" in model_inst.config["agent_settings"]
                else {}
            )

        def remove_farmers_from_converted_areas(
            self, future_landcover: xr.DataArray
        ) -> None:
            """Override to skip farmer removal - it will be done later in Government.__init__."""
            logger.info(
                "Skipping farmer removal in standalone mode - will be done after agents are created"
            )
            print(
                "  [INFO] Farmer removal deferred to Government initialization",
                flush=True,
            )
            # Store the future_landcover for later use
            self.model._future_landcover = future_landcover
            # Set a flag to indicate farmer removal was deferred
            self._farmer_removal_deferred = True

    print("Forest planting workflow begins...", flush=True)
    logger.info(
        "Creating temporary Government instance and calling prepare_modified_soil_maps()"
    )

    # Create temporary instance and call the proven working method
    temp_gov = TempGovernment(model)
    temp_gov.prepare_modified_soil_maps()  # This is the original working code!

    # Transfer the landcover data from instance to model for later farmer removal
    if hasattr(temp_gov, "_landcover_resampled"):
        model._landcover_resampled = temp_gov._landcover_resampled
        logger.info("Transferred _landcover_resampled to model instance")
    else:
        logger.warning("⚠️  _landcover_resampled not found on temp_gov instance")

    print(
        "MODEL: Modified soil maps ready for hydrology initialization",
        flush=True,
    )
    logger.info("=" * 80)
    logger.info("SOIL MAPS PREPARED - READY FOR HYDROLOGY INITIALIZATION")
    logger.info("=" * 80)


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

        # If forest planting was enabled, the soil maps were already prepared in model.__init__
        # Now we just need to remove farmers from converted areas
        if self.config.get("plant_forest", False) and hasattr(
            self.model, "_future_landcover"
        ):
            print("\n" + "=" * 80, flush=True)
            print(
                "FOREST PLANTING: Now removing farmers from converted areas...",
                flush=True,
            )
            print("=" * 80, flush=True)
            logger.info("\nGOVERNMENT: Removing farmers from forest-converted areas...")
            # Store landcover_resampled on self for the removal method to access
            self._landcover_resampled = self.model._landcover_resampled
            self.remove_farmers_from_converted_areas(self.model._future_landcover)
            print("FOREST PLANTING: Farmer removal complete!", flush=True)
            print("=" * 80, flush=True)
            logger.info("GOVERNMENT: Farmer removal complete\n")

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
        self.set_irrigation_limit()
        self.provide_subsidies()

        self.report(locals())

    def create_suitability_map_from_potential(
        self,
        global_potential_path: str | Path,
        template_da: xr.DataArray,
        threshold_percent: float = 50.0,
    ) -> xr.DataArray:
        """Create a binary suitability map from global restoration potential raster.

        Uses a simple threshold approach based on restoration potential percentage.

        Args:
            global_potential_path: Path to the Global_Tree_Restoration_Potential.tif file (in percent).
            template_da: Template DataArray defining target grid, CRS, and coordinates.
            threshold_percent: Minimum restoration potential (%) to consider suitable (default: 50.0).

        Returns:
            Binary suitability map (1=suitable, 0=unsuitable) on the template grid.

        Raises:
            FileNotFoundError: If the global potential raster file does not exist.
        """
        logger.info("\n" + "=" * 80)
        logger.info(
            "STEP 1: Creating Suitability Map from Global Restoration Potential"
        )
        logger.info("=" * 80)

        potential_path = Path(global_potential_path)
        if not potential_path.exists():
            raise FileNotFoundError(
                f"Global restoration potential file not found: {potential_path}"
            )

        logger.info(f"Loading global restoration potential from: {potential_path}")
        potential_raw = rioxarray.open_rasterio(potential_path, masked=True).squeeze()
        logger.info(f"  Raw potential shape: {potential_raw.shape}")
        logger.info(f"  Raw potential CRS: {potential_raw.rio.crs}")
        logger.info(
            f"  Raw potential range: [{float(potential_raw.min().item()):.2f}, {float(potential_raw.max().item()):.2f}]"
        )

        # Reproject to match template grid
        logger.info(f"\nReprojecting to template CRS: {template_da.rio.crs}")
        potential_reprojected = potential_raw.rio.reproject_match(template_da)
        logger.info(f"  Reprojected shape: {potential_reprojected.shape}")
        logger.info(
            f"  Reprojected range: [{float(potential_reprojected.min().item()):.2f}, {float(potential_reprojected.max().item()):.2f}]"
        )

        # Apply threshold to create binary suitability map
        logger.info(f"\nApplying threshold: potential >= {threshold_percent}%")
        suitability_binary = (potential_reprojected >= threshold_percent).astype(
            np.int8
        )

        n_suitable = int(suitability_binary.sum())
        n_total = int((~np.isnan(potential_reprojected)).sum())
        pct_suitable = (n_suitable / n_total * 100) if n_total > 0 else 0.0

        logger.info(f"\nSuitability Statistics:")
        logger.info(f"  Total valid cells: {n_total:,}")
        logger.info(f"  Suitable cells (1): {n_suitable:,}")
        logger.info(f"  Unsuitable cells (0): {n_total - n_suitable:,}")
        logger.info(f"  Percentage suitable: {pct_suitable:.2f}%")

        return suitability_binary

    def modify_bulk_density_relative(
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
            layer_data = bulk_density_ds["bulk_density"].isel(soil_layer=layer_idx)

            original_mean = float(layer_data.mean())
            original_min = float(layer_data.min())
            original_max = float(layer_data.max())

            # Apply relative reduction: new = old * reduction_factor
            modified_layer = xr.where(
                suitability == 1, layer_data * reduction_factor, layer_data
            )

            # Use direct array assignment instead of .loc indexing
            modified_ds["bulk_density"].values[layer_idx, :, :] = modified_layer.values

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

    def analyze_soil_by_landcover(
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

    def modify_soil_characteristics(
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

    def create_future_landcover_scenario(
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
            f"STEP 2: Creating Future Land Cover Scenario (converting to class {target_class})"
        )
        logger.info("=" * 80)

        # Count conversions
        n_conversions = int((suitability == 1).sum())
        n_total = int((~np.isnan(landcover)).sum())
        pct_converted = (n_conversions / n_total * 100) if n_total > 0 else 0.0

        logger.info(f"Conversion statistics:")
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

        Args:
            current_landcover: Current land cover classification.
            future_landcover: Future land cover after forest conversion.
            suitability: Binary suitability map.
            output_path: Optional path to save the figure (default: None, shows plot).
        """
        logger.info("\n" + "=" * 80)
        logger.info("CREATING VISUALIZATION PLOTS")
        logger.info("=" * 80)
        logger.info("  Downsampling arrays for faster visualization...")

        # Downsample arrays by factor of 10 for faster plotting
        current_downsampled = current_landcover.values[::10, ::10]
        future_downsampled = future_landcover.values[::10, ::10]
        suitability_downsampled = suitability.values[::10, ::10]

        logger.info(f"  Visualization array size: {current_downsampled.shape}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Top left: Current landcover
        im1 = axes[0, 0].imshow(
            current_downsampled, cmap="tab20", interpolation="nearest"
        )
        axes[0, 0].set_title("Current Land Cover", fontsize=14, fontweight="bold")
        axes[0, 0].axis("off")
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # Top right: Future landcover
        im2 = axes[0, 1].imshow(
            future_downsampled, cmap="tab20", interpolation="nearest"
        )
        axes[0, 1].set_title(
            "Future Land Cover (with Reforestation)", fontsize=14, fontweight="bold"
        )
        axes[0, 1].axis("off")
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Bottom left: Suitability map
        im3 = axes[1, 0].imshow(
            suitability_downsampled,
            cmap="Greens",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        axes[1, 0].set_title(
            "Reforestation Suitability (50% threshold)", fontsize=14, fontweight="bold"
        )
        axes[1, 0].axis("off")
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cbar3.set_ticks([0, 1])
        cbar3.set_ticklabels(["Unsuitable", "Suitable"])

        # Bottom right: Change map
        change_map = (future_downsampled != current_downsampled).astype(int)
        im4 = axes[1, 1].imshow(
            change_map, cmap="Reds", interpolation="nearest", vmin=0, vmax=1
        )
        axes[1, 1].set_title("Converted Areas", fontsize=14, fontweight="bold")
        axes[1, 1].axis("off")
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

    def prepare_modified_soil_maps(self) -> None:
        """Prepare modified soil maps for forest planting scenario.

        This is the main workflow that:
        1. Loads input soil data (bulk density, SOC) and land cover
        2. Creates suitability map from global restoration potential (50% threshold)
        3. Resamples land cover to match soil grid
        4. Creates future land cover scenario
        5. Analyzes soil by land cover class
        6. Modifies soil characteristics using forest class statistics
           - SOC: mean strategy (all converted pixels → forest mean SOC)
           - Bulk density: site-specific relative reduction based on forest vs non-forest ratio
        7. Saves modified soil maps
        8. Creates visualization plots
        9. Removes farmers from converted areas

        Raises:
            RuntimeError: If landcover classification contains unexpected values or if bulk density is invalid.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING FOREST PLANTING SOIL MODIFICATION WORKFLOW")
        logger.info("=" * 80)

        # Load global restoration potential from current model folder
        # TODO: Prepare from data_catalog instead of assuming everyone has this downloaded
        global_potential_path = Path.cwd() / "Global_Tree_Restoration_Potential.tif"
        print(
            f"Global restoration potential path: {global_potential_path}", flush=True
        )  # remove later, now just for checking

        # Load soil data using model.files paths
        logger.info("\nLoading soil datasets:")
        bulk_density_path = self.model.files["subgrid"]["soil/bulk_density"]
        soc_path = self.model.files["subgrid"]["soil/soil_organic_carbon"]

        logger.info(f"  Bulk density: {bulk_density_path}")
        bulk_density_ds = xr.open_zarr(bulk_density_path)
        logger.info(f"    Shape: {bulk_density_ds['bulk_density'].shape}")
        logger.info(f"    Dimensions: {list(bulk_density_ds['bulk_density'].dims)}")

        logger.info(f"  Soil organic carbon: {soc_path}")
        soc_ds = xr.open_zarr(soc_path)
        logger.info(f"    Shape: {soc_ds['soil_organic_carbon'].shape}")
        logger.info(f"    Dimensions: {list(soc_ds['soil_organic_carbon'].dims)}")

        # Load landcover
        landcover_path = self.model.files["other"]["landcover/classification"]

        logger.info(f"\nLoading land cover: {landcover_path}")
        landcover_ds = xr.open_zarr(landcover_path)
        landcover = landcover_ds["classification"]

        # Set CRS on landcover if not present (zarr files may not have rio CRS embedded)
        if landcover.rio.crs is None:
            if "_CRS" in landcover.attrs:
                # Use the _CRS attribute that's stored in the zarr file
                landcover = landcover.rio.write_crs(
                    pyproj.CRS(landcover.attrs["_CRS"]["wkt"])
                )
                logger.info(f"  Set CRS from zarr _CRS attribute: {landcover.rio.crs}")
            elif (
                hasattr(self.model, "hydrology")
                and hasattr(self.model.hydrology, "grid")
                and hasattr(self.model.hydrology.grid, "crs")
            ):
                # Fallback to model grid CRS
                landcover = landcover.rio.write_crs(self.model.hydrology.grid.crs)
                landcover = landcover.rio.write_transform(
                    self.model.hydrology.grid.transform
                )
                logger.info(f"  Set CRS from model grid: {landcover.rio.crs}")

        logger.info(f"  Shape: {landcover.shape}")
        logger.info(f"  Dimensions: {list(landcover.dims)}")
        unique_classes = np.unique(landcover.values[~np.isnan(landcover.values)])
        logger.info(f"  Unique classes: {sorted([int(c) for c in unique_classes])}")

        # Use first soil layer as template for reprojection
        # Need to copy CRS from model since zarr files may not have it embedded
        template_da = bulk_density_ds["bulk_density"].isel(soil_layer=0)

        # Get CRS and transform from model hydrology grid (most reliable source)
        if (
            hasattr(self.model, "hydrology")
            and hasattr(self.model.hydrology, "grid")
            and hasattr(self.model.hydrology.grid, "crs")
        ):
            model_crs = self.model.hydrology.grid.crs
            model_transform = self.model.hydrology.grid.transform
            template_da = template_da.rio.write_crs(model_crs)
            template_da = template_da.rio.write_transform(model_transform)
            logger.info(
                f"Set template CRS from model hydrology.grid: {template_da.rio.crs}"
            )
            logger.info(f"Set template transform: {template_da.rio.transform()}")
        elif landcover.rio.crs is not None:
            # Fallback to landcover if hydrology doesn't have CRS
            template_da = template_da.rio.write_crs(landcover.rio.crs)
            template_da = template_da.rio.write_transform(landcover.rio.transform())
            logger.info(f"Set template CRS from landcover: {template_da.rio.crs}")
        else:
            raise RuntimeError(
                "Cannot determine CRS for spatial operations. "
                "Neither model.hydrology.grid.crs nor landcover CRS are available."
            )

        # Step 1: Create suitability map from global restoration potential
        print(
            "\n[STEP 1] Creating suitability map from global restoration potential...",
            flush=True,
        )
        suitability = self.create_suitability_map_from_potential(
            global_potential_path=global_potential_path,
            template_da=template_da,
            threshold_percent=50.0,
        )
        print(
            f"[STEP 1] Suitability map created. Shape: {suitability.shape}, Suitable pixels: {(suitability == 1).sum().item():,}",
            flush=True,
        )

        # Step 2: Resample landcover to match soil grid
        print("\n[STEP 2] Resampling land cover to match soil grid...", flush=True)
        logger.info("\n" + "=" * 80)
        logger.info("Resampling land cover to soil grid")
        logger.info("=" * 80)
        landcover_resampled = landcover.rio.reproject_match(
            template_da, resampling=0
        )  # nearest neighbor
        logger.info(f"  Resampled shape: {landcover_resampled.shape}")
        print(
            f"[STEP 2] Landcover resampled from {landcover.shape} to {landcover_resampled.shape}",
            flush=True,
        )

        # Step 3: Create future landcover scenario
        print(
            "\n[STEP 3] Creating future landcover scenario with forest conversions...",
            flush=True,
        )
        future_landcover = self.create_future_landcover_scenario(
            landcover=landcover_resampled,
            suitability=suitability,
            target_class=10,  # ESA WorldCover forest class
        )
        n_converted = (
            ((future_landcover == 10) & (landcover_resampled != 10)).sum().item()
        )
        print(
            f"[STEP 3] Future scenario created. Pixels converted to forest: {n_converted:,}",
            flush=True,
        )

        # Store the resampled landcover for later use in farmer removal
        self._landcover_resampled = landcover_resampled

        # Step 4: Analyze soil by landcover class
        print(
            "\n[STEP 4] Analyzing soil characteristics by land cover class...",
            flush=True,
        )
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Analyzing Soil Characteristics by Land Cover")
        logger.info("=" * 80)

        soc_stats = self.analyze_soil_by_landcover(
            soil_ds=soc_ds,
            landcover=landcover_resampled,
            variable_name="soil_organic_carbon",
            target_class=10,
        )
        print(
            f"[STEP 4] SOC analysis complete. Forest mean: {soc_stats[10]['mean']:.4f}",
            flush=True,
        )

        bulk_density_stats = self.analyze_soil_by_landcover(
            soil_ds=bulk_density_ds,
            landcover=landcover_resampled,
            variable_name="bulk_density",
            target_class=10,
        )
        print(
            f"[STEP 4] Bulk density analysis complete. Forest mean: {bulk_density_stats[10]['mean']:.4f}",
            flush=True,
        )

        # Step 5: Modify soil characteristics
        print(
            "\n[STEP 5] Modifying soil characteristics for converted areas...",
            flush=True,
        )
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Modifying Soil Characteristics")
        logger.info("=" * 80)

        # SOC: Use MEAN strategy
        print(
            "[STEP 5] Modifying soil organic carbon (using mean from forest)...",
            flush=True,
        )
        modified_soc_ds = self.modify_soil_characteristics(
            soil_ds=soc_ds,
            landcover_resampled=landcover_resampled,
            suitability=suitability,
            stats_by_class=soc_stats,
            variable_name="soil_organic_carbon",
            strategy="mean",  # Use mean from forest class
            target_class=10,
        )
        print("[STEP 5] SOC modified", flush=True)

        # Bulk density: Calculate site-specific reduction factor from existing data
        print(
            "[STEP 5] Calculating bulk density reduction factor from existing forest vs non-forest...",
            flush=True,
        )

        # Get forest mean bulk density
        forest_mean_bd = bulk_density_stats[10]["mean"]

        # Get non-forest mean bulk density (average across all non-forest classes)
        non_forest_classes = [cls for cls in bulk_density_stats.keys() if cls != 10]
        non_forest_mean_bd = float(
            np.mean([bulk_density_stats[cls]["mean"] for cls in non_forest_classes])
        )

        # Calculate reduction factor (site-specific)
        calculated_reduction_factor = forest_mean_bd / non_forest_mean_bd
        reduction_percent = (1 - calculated_reduction_factor) * 100

        logger.info("\n" + "=" * 80)
        logger.info("BULK DENSITY REDUCTION FACTOR CALCULATION (DATA-DRIVEN)")
        logger.info("=" * 80)
        logger.info(f"Existing forest mean BD: {forest_mean_bd:.4f} kg/m³")
        logger.info(f"Non-forest mean BD: {non_forest_mean_bd:.4f} kg/m³")
        logger.info(
            f"Calculated reduction factor: {calculated_reduction_factor:.4f} ({reduction_percent:.1f}% reduction)"
        )

        print(
            f"[STEP 5]   Forest mean BD: {forest_mean_bd:.4f} kg/m³",
            flush=True,
        )
        print(
            f"[STEP 5]   Non-forest mean BD: {non_forest_mean_bd:.4f} kg/m³",
            flush=True,
        )
        print(
            f"[STEP 5]   Calculated reduction factor: {calculated_reduction_factor:.4f} ({reduction_percent:.1f}% reduction)",
            flush=True,
        )

        print(
            f"[STEP 5] Modifying bulk density ({reduction_percent:.1f}% reduction)...",
            flush=True,
        )
        modified_bulk_density_ds = self.modify_bulk_density_relative(
            bulk_density_ds=bulk_density_ds,
            suitability=suitability,
            reduction_factor=calculated_reduction_factor,
        )
        print("[STEP 5] Bulk density modified", flush=True)

        # Step 6: Save modified datasets to input folder (alongside originals)
        print("\n[STEP 6] Saving modified soil datasets...", flush=True)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Saving Modified Soil Datasets")
        logger.info("=" * 80)

        # Save to input folder (where original soil data is located)
        # Get the directory containing the original bulk density file
        original_bulk_density_path = Path(
            self.model.files["subgrid"]["soil/bulk_density"]
        )
        input_soil_folder = original_bulk_density_path.parent
        print(f"[STEP 6] Saving to input folder: {input_soil_folder}", flush=True)

        modified_soc_path = (
            input_soil_folder / "soil_organic_carbon_forest_modified.zarr"
        )
        modified_bulk_density_path = (
            input_soil_folder / "bulk_density_forest_modified.zarr"
        )

        # Save visualization to output folder
        output_folder = self.model.output_folder / "forest_planting"
        output_folder.mkdir(parents=True, exist_ok=True)
        future_landcover_path = output_folder / "landcover_future.zarr"

        logger.info(f"Saving modified SOC to: {modified_soc_path}")
        # Create new dataset with renamed variable to match the filename for proper zarr reading
        modified_soc_renamed = modified_soc_ds.rename(
            {"soil_organic_carbon": "soil_organic_carbon_forest_modified"}
        )
        modified_soc_renamed.to_zarr(modified_soc_path, mode="w")

        logger.info(f"Saving modified bulk density to: {modified_bulk_density_path}")
        # Create new dataset with renamed variable to match the filename for proper zarr reading
        modified_bulk_density_renamed = modified_bulk_density_ds.rename(
            {"bulk_density": "bulk_density_forest_modified"}
        )
        modified_bulk_density_renamed.to_zarr(modified_bulk_density_path, mode="w")

        logger.info(f"Saving future landcover to: {future_landcover_path}")
        future_landcover_ds = xr.Dataset({"classification": future_landcover})
        future_landcover_ds.to_zarr(future_landcover_path, mode="w")

        logger.info("\nAll modified datasets saved successfully!")
        print("[STEP 6] All modified datasets saved successfully!", flush=True)

        # Update model.files to point to modified soil maps
        logger.info("\n" + "=" * 80)
        logger.info("UPDATING MODEL FILE PATHS TO USE MODIFIED SOIL MAPS")
        logger.info("=" * 80)

        self.model.files["subgrid"]["soil/bulk_density"] = modified_bulk_density_path
        self.model.files["subgrid"]["soil/soil_organic_carbon"] = modified_soc_path

        logger.info(
            f"Updated bulk density path: {self.model.files['subgrid']['soil/bulk_density']}"
        )
        logger.info(
            f"Updated SOC path: {self.model.files['subgrid']['soil/soil_organic_carbon']}"
        )
        print("[STEP 6] Model file paths updated to use modified soil maps", flush=True)

        # Step 7: Create visualization
        print("\n[STEP 7] Creating visualization...", flush=True)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Creating Visualization")
        logger.info("=" * 80)

        plot_output_path = output_folder / "reforestation_scenario.png"
        self.plot_reforestation_scenario(
            current_landcover=landcover_resampled,
            future_landcover=future_landcover,
            suitability=suitability,
            output_path=plot_output_path,
        )
        print(f"[STEP 7] Visualization saved to: {plot_output_path}", flush=True)

        # Step 8: Remove farmers from converted areas
        print("\n[STEP 8] Removing farmers from converted areas...", flush=True)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: Removing Farmers from Converted Areas")
        logger.info("=" * 80)
        self.remove_farmers_from_converted_areas(future_landcover)

        # Check if farmer removal was deferred or completed
        if hasattr(self, "_farmer_removal_deferred") and self._farmer_removal_deferred:
            print(
                "[STEP 8] Farmer removal DEFERRED (will happen during Government agent initialization)",
                flush=True,
            )
        else:
            print("[STEP 8] Farmer removal complete", flush=True)

        logger.info("\n" + "=" * 80)
        logger.info("FOREST PLANTING WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    def remove_farmers_from_converted_areas(
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
            print(
                "  [WARNING] No crop farmers agent found - skipping farmer removal",
                flush=True,
            )
            return

        print("  Analyzing converted areas...", flush=True)
        logger.info("Removing farmers from converted forest areas...")

        crop_farmers = self.agents.crop_farmers
        print(f"  Total farmers in model: {crop_farmers.var.n}", flush=True)

        # Use the resampled landcover that was used to create the future scenario
        # This ensures both arrays have the same shape and resolution
        if not hasattr(self, "_landcover_resampled"):
            raise RuntimeError(
                "Resampled landcover not found. Ensure prepare_modified_soil_maps() was called first."
            )

        original_landcover = self._landcover_resampled.values

        # Find locations where landcover changed to forest (class 10 = tree cover in ESA WorldCover)
        print("  Comparing landcover scenarios...", flush=True)
        converted_mask = (future_landcover.values == 10) & (original_landcover != 10)

        logger.info(f"Future landcover shape: {future_landcover.shape}")
        logger.info(f"Original landcover shape: {original_landcover.shape}")
        logger.info(f"Converted mask shape: {converted_mask.shape}")
        print(
            f"  Landcover shapes match: {future_landcover.shape} == {original_landcover.shape}",
            flush=True,
        )

        if not converted_mask.any():
            logger.info("No areas converted to forest, no farmers to remove")
            print("  No areas converted to forest - nothing to do", flush=True)
            return

        n_converted_cells = converted_mask.sum()
        logger.info(f"Found {n_converted_cells:,} grid cells converted to forest")
        print(
            f"  Found {n_converted_cells:,} grid cells converted to forest", flush=True
        )

        # Convert subgrid mask to HRU scale
        # Use "last" method since we're working with a binary mask
        print("  Converting grid mask to HRU scale...", flush=True)
        converted_mask_HRU = crop_farmers.HRU.convert_subgrid_to_HRU(
            converted_mask.astype(np.int32), method="last"
        )

        # Find HRUs that were converted to forest
        converted_HRU_indices = np.where(converted_mask_HRU == 1)[0]

        if len(converted_HRU_indices) == 0:
            logger.info("No HRUs converted to forest")
            print("  No HRUs converted to forest after scaling", flush=True)
            return

        logger.info(f"Found {len(converted_HRU_indices):,} HRUs converted to forest")
        print(
            f"  Found {len(converted_HRU_indices):,} HRUs converted to forest",
            flush=True,
        )

        # Get land owners for converted HRUs
        print("  Identifying farmers in converted areas...", flush=True)
        land_owners_in_converted = crop_farmers.HRU.var.land_owners[
            converted_HRU_indices
        ]

        # Filter out HRUs with no owner (-1)
        farmer_indices_in_converted = land_owners_in_converted[
            land_owners_in_converted != -1
        ]
        print(f"  HRUs with farmers: {len(farmer_indices_in_converted):,}", flush=True)

        if len(farmer_indices_in_converted) == 0:
            logger.info("No farmers found in converted areas")
            print("  No farmers found in converted areas", flush=True)
            return

        # Get unique farmer indices
        unique_farmer_indices = np.unique(farmer_indices_in_converted)
        n_farmers_to_remove = len(unique_farmer_indices)

        logger.info(f"Removing {n_farmers_to_remove:,} farmers from converted areas")
        print(f"  Unique farmers to remove: {n_farmers_to_remove:,}", flush=True)
        logger.info(
            f"Farmer indices: {unique_farmer_indices[:10]}..."
            if n_farmers_to_remove > 10
            else f"Farmer indices: {unique_farmer_indices}"
        )

        # Remove farmers using crop_farmers' remove_agents method
        # The new land use type should be FOREST (hydrology module constant = 0)
        print(f"  Removing farmers (this may take a moment)...", flush=True)
        removed_HRUs = crop_farmers.remove_agents(
            farmer_indices=unique_farmer_indices,
            new_land_use_type=FOREST,
        )

        logger.info(f"Successfully removed {n_farmers_to_remove:,} farmers")
        logger.info(f"Total HRUs disowned: {len(removed_HRUs):,}")
        print(f"  Successfully removed {n_farmers_to_remove:,} farmers", flush=True)
        print(f"  Total HRUs disowned: {len(removed_HRUs):,}", flush=True)
        print(f"  Remaining farmers: {crop_farmers.var.n}", flush=True)
