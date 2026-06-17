"""This module contains the EarlyWarningModule, which provides early warning capabilities for the agents."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from rasterio.features import geometry_mask, rasterize

from geb.workflows.io import read_geom, read_zarr, write_zarr

from ..workflows.helpers import from_landuse_raster_to_polygon

if TYPE_CHECKING:
    from geb.agents.households import Households
    from geb.model import GEBModel


class EarlyWarningModule:
    """Module responsible for loading and managing Early Warnings for the households in the model."""

    def __init__(self, model: GEBModel, households: Households) -> None:
        """Initialize the EarlyWarningModule with the model and households, and load all necessary data.

        Args:
            model (GEBModel): The main model instance containing configuration and file paths.
            households (Households): The households agent instance where the loaded data will be stored.
        """
        self.model = model
        self.logger = self.model.logger
        self.households = households
        self.load_wlranges_and_measures()

    def load_wlranges_and_measures(self) -> None:
        """
        Loads the water level ranges and appropriate measures, and the implementation times for measures.

        Raises:
            ValueError: If the wl_ranges.json structure is invalid.
        """
        with open(self.model.files["dict"]["measures/implementation_times"], "r") as f:
            self.households.var.implementation_times = json.load(f)

        with open(self.model.files["dict"]["measures/wl_ranges"], "r") as f:
            wlranges_and_measures = json.load(f)

            # Check whether the old format is being used
            first_key = next(iter(wlranges_and_measures))

            if first_key.isdigit():
                raise ValueError(
                    "Invalid wl_ranges.json structure. Expected:\n"
                    "{\n"
                    '    "residential_buildings": {"1": {...}, "2": {...}},\n'
                    '    "energy_stations": {"1": {...}, "2": {...}}\n'
                    "}"
                )

            self.households.var.wlranges_and_measures = {
                asset_type: {
                    int(range_id): range_info for range_id, range_info in ranges.items()
                }
                for asset_type, ranges in wlranges_and_measures.items()
            }
        self.logger.info(
            "Loaded water level ranges with associated measures and their implementation times for %d asset types.",
            len(self.households.var.wlranges_and_measures),
        )

    def load_ensemble_flood_maps(self, date_time: datetime) -> xr.DataArray:
        """Load flood maps for all ensemble members for a specific forecast date time.

        River channels are removed from each member before concatenation to ensure
        probability calculations only consider floodable areas outside permanent water bodies.

        Args:
            date_time: The forecast date time for which to load the flood maps.

        Returns:
            An xarray DataArray containing the flood maps for all ensemble members
            with river channels masked out.

        Raises:
            FileNotFoundError: If no flood maps are found for the specified date time.
            FileExistsError: If multiple flood maps are found for the specified date time.
            ValueError: If rivers geometry is missing the 'width' column.
        """
        # Load and prepare river mask once (will be applied to all members)
        rivers_path = self.model.files["geom"]["routing/rivers"]
        river_mask_array: np.ndarray | None = None
        gdf_mercator = None

        if rivers_path.exists():
            print(f"Preparing river mask from {rivers_path}")
            rivers = gpd.read_parquet(rivers_path)

            if "width" not in rivers.columns:
                raise ValueError("Rivers geometry missing 'width' column")

            # Set CRS and buffer in Mercator projection (meters)
            crs_wgs84 = CRS.from_epsg(4326)
            crs_mercator = CRS.from_epsg(3857)
            rivers.set_crs(crs_wgs84, inplace=True)
            gdf_mercator = rivers.to_crs(crs_mercator)

            # Separate rivers with width values from those with NaN width
            rivers_with_width = gdf_mercator[gdf_mercator["width"].notna()].copy()
            rivers_no_width = gdf_mercator[gdf_mercator["width"].isna()].copy()

            print(f"Rivers with width data: {len(rivers_with_width)}")
            print(
                f"Rivers without width data (will mask grid cells only): {len(rivers_no_width)}"
            )

            # For rivers with width: create buffers based on width
            if len(rivers_with_width) > 0:
                rivers_with_width["geometry"] = rivers_with_width.buffer(
                    rivers_with_width["width"] / 2
                )

            # For rivers without width: keep original line geometry (will mask intersected grid cells)
            # No buffering needed - geometry_mask will handle line intersection with grid cells

            # Combine both geometries for final masking
            all_river_geometries = []
            if len(rivers_with_width) > 0:
                all_river_geometries.extend(rivers_with_width.geometry.tolist())
            if len(rivers_no_width) > 0:
                all_river_geometries.extend(rivers_no_width.geometry.tolist())

            # Create a single GeoDataFrame with all river geometries
            gdf_mercator = gpd.GeoDataFrame(
                {"geometry": all_river_geometries}, crs=crs_mercator
            )

            print("Created river buffers in Mercator projection for masking")
        else:
            print(f"Rivers file not found at {rivers_path} - skipping river masking")

        members = []
        rivers_gdf = None

        # TODO: get the number of members dynamically from the flood maps folder instead of hardcoding 50

        # Get the number of members from the flood maps folder
        flood_maps_folder = (
            self.model.output_folder
            / "flood_maps"
            / f"forecast_{date_time.strftime('%Y%m%dT%H%M%S')}"
        )
        member_folders = [
            f
            for f in flood_maps_folder.iterdir()
            if f.is_dir() and f.name.startswith("member_")
        ]
        num_members = len(member_folders)

        for member in range(num_members):
            member_folder = (
                self.model.output_folder
                / "flood_maps"
                / f"forecast_{date_time.strftime('%Y%m%dT%H%M%S')}"
                / f"member_{member}"
            )

            # Dynamically find the zarr file in the member folder
            zarr_files = list(member_folder.glob("*.zarr"))
            if not zarr_files:
                raise FileNotFoundError(f"No zarr files found in {member_folder}")
            elif len(zarr_files) > 1:
                raise FileExistsError(
                    f"Multiple zarr files found in {member_folder}: {[f.name for f in zarr_files]}"
                )

            flood_map_path = zarr_files[0]

            # open flood map for this ensemble member
            flood_map_da = read_zarr(flood_map_path)

            # TODO: check this hardcoded CRS
            flood_map_da.rio.write_crs("EPSG:4326", inplace=True)

            # Apply river mask to this member if available
            if gdf_mercator is not None:
                # Create river mask on first iteration (reuse for all members)
                if river_mask_array is None:
                    # Reproject buffered rivers to flood map CRS
                    gdf_buffered = gdf_mercator.to_crs(flood_map_da.rio.crs)

                    # Create mask using geometry_mask
                    # geometry_mask returns True for pixels OUTSIDE geometries (when invert=False)
                    # We want True for pixels INSIDE rivers (to mask them)
                    # So we use invert=True
                    river_mask_array = geometry_mask(
                        gdf_buffered.geometry,
                        out_shape=flood_map_da.rio.shape,
                        transform=flood_map_da.rio.transform(),
                        all_touched=True,
                        invert=True,  # True = inside rivers (to be masked)
                    )

                    # Store river mask as xarray with proper coordinates so it can be flipped along with data
                    self.river_mask_da = xr.DataArray(
                        data=river_mask_array,
                        coords={"y": flood_map_da.y.values, "x": flood_map_da.x.values},
                        dims=("y", "x"),
                    )

                    print(
                        f"River mask created: {np.sum(river_mask_array)} pixels will be masked"
                    )

                # Apply mask: set river pixels (where river_mask == True) to 0
                masked_flood_map_da = flood_map_da.where(~self.river_mask_da, 0)

            members.append(masked_flood_map_da)

        # Concatenate all masked members
        ensemble_flood_maps = xr.concat(members, dim="member")
        if river_mask_array is not None:
            for member_idx in range(ensemble_flood_maps.sizes["member"]):
                member_data = ensemble_flood_maps.isel(member=member_idx).values
                river_pixels = member_data[river_mask_array]
                n_nonzero = np.sum(river_pixels > 0)
                self.logger.debug(
                    f"Member {member_idx + 1}: {n_nonzero} non-zero pixels in river mask"
                )
                if n_nonzero > 0:
                    self.logger.warning(
                        f"Member {member_idx + 1} has non-zero values in river!"
                    )

        return ensemble_flood_maps

    def load_ensemble_damage_maps(self, date_time: datetime) -> pd.DataFrame:
        """Loads the damage maps for all ensemble members and aggregates them into a single dataframe. Work in standby for now.

        Args:
            date_time: The forecast date time for which to load the damage maps.
        Returns:
            A dataframe containing the aggregated damage maps for all ensemble members.
        """
        # Get number of members
        # open the damage maps folder to see the number of members
        damage_forecast_folder = (
            self.model.output_folder
            / "damage_maps"
            / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
        )
        n_ensemble_members = sum(1 for _ in damage_forecast_folder.glob("member_*"))
        print(f"Loading damage maps for {n_ensemble_members} ensemble members.")

        damage_maps = []
        # Load the damage maps for each ensemble member
        for member in range(1, n_ensemble_members + 1):
            file_path = (
                damage_forecast_folder
                / f"damage_map_buildings_content_{date_time.isoformat().replace(':', '').replace('-', '')}_member{member}.gpkg"
            )
            damage_map = gpd.read_file(file_path)

            # Add member and building_id columns
            damage_map["member"] = member
            damage_map["building_id"] = damage_map.index + 1

            # Aggregate all damage maps
            damage_maps.append(damage_map)

        # Concatenate all damage maps into a single dataframe
        ensemble_damage_maps = pd.concat(damage_maps)

        return ensemble_damage_maps

    def create_flood_probability_maps(
        self,
        date_time: datetime,
        strategy: str = "residential_buildings",
        exceedance: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """Creates flood probability maps based on the ensemble of flood maps for different warning strategies.

        Args:
            date_time: The forecast date time for which to create the probability maps.
            strategy: The warning strategy to use (e.g., 'residential_buildings', 'energy_stations').
            exceedance: Whether to calculate flood exceedance probability maps (instead of regular probability maps).

        Returns:
            Probability maps for each water level range OR Probability exceedance maps for each lower bound of the water level ranges
            for that forecast initialization date.

        Raises:
            ValueError: If the specified strategy is not recognized.
        """
        # Load the ensemble of flood maps for that specific date time
        ensemble_flood_maps = self.load_ensemble_flood_maps(date_time=date_time)
        crs = self.model.config["hazards"]["floods"]["crs"]
        # TODO: need to think on how to load all the flood maps for each date time, instead of for each strategy/and to create flood prob exceedance maps

        if exceedance:
            # Create output folder for exceedance probability maps
            prob_folder = (
                self.model.output_folder
                / "flood_prob_exceedance_maps"
                / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
            )
        else:
            # Create output folder for regular probability maps
            prob_folder = (
                self.model.output_folder
                / "flood_prob_maps"
                / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
            )
        if strategy not in self.households.var.wlranges_and_measures:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available strategies: {list(self.households.var.wlranges_and_measures.keys())}"
            )

        ranges = []

        for range_id, range_info in self.households.var.wlranges_and_measures[
            strategy
        ].items():
            if exceedance:
                ranges.append((range_id, range_info["min"], None))
            else:
                ranges.append((range_id, range_info["min"], range_info["max"]))

        probability_maps = []
        # Loop over water level ranges to calculate probability maps
        for range_id, wl_min, wl_max in ranges:
            daily_ensemble = ensemble_flood_maps
            self.logger.info(
                f"Creating probability map for date {date_time}, strategy {strategy}, range_id {range_id} (wl_min={wl_min}, wl_max={wl_max})"
            )
            if wl_max is not None:
                condition = (daily_ensemble >= wl_min) & (daily_ensemble <= wl_max)
                self.logger.info(f"Using range: {wl_min} <= x <= {wl_max}")
            else:
                condition = daily_ensemble >= wl_min
                self.logger.info(f"Using exceedance: x >= {wl_min}")

            probability = condition.sum(dim="member") / condition.sizes["member"]

            if probability.y.values[0] < probability.y.values[-1]:
                self.logger.info(
                    "flipping y axis so it can be used for zonal stats later"
                )
                probability: Unknown | xr.DataArray = probability.sortby(
                    "y", ascending=False
                )

            # Save probability map as a zarr file
            if exceedance:
                file_name = (
                    f"prob_exceedance_map_range{range_id}_strategy_{strategy}.zarr"
                )
            else:
                file_name = f"prob_map_range{range_id}_strategy_{strategy}.zarr"
            file_path = prob_folder / file_name
            file_path.mkdir(parents=True, exist_ok=True)

            probability = probability.astype(np.float32)
            probability = probability.rio.write_nodata(np.nan)
            probability = write_zarr(da=probability, path=file_path, crs=crs)
            # probability_maps.append(
            #     probability.expand_dims(time=[date_time], range_id=[range_id])
            # )
            probability_maps.append(probability)

        range_ids = [range_id for range_id, _, _ in ranges]
        probability_maps = xr.concat(
            probability_maps,
            dim=xr.DataArray(range_ids, dims="range_id", name="range_id"),
            join="exact",
        ).expand_dims(time=[date_time])

        return probability_maps

    def create_damage_probability_maps(self, date_time: datetime) -> None:
        """Creates an object-based (buildings) probability map based on the ensemble of damage maps. Work in standby for now.

        Args:
            date_time: The forecast date time for which to create the damage probability maps.
        """
        crs = self.model.config["hazards"]["floods"]["crs"]

        damage_prob_maps_folder = (
            self.model.output_folder
            / "damage_prob_maps"
            / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
        )
        damage_prob_maps_folder.mkdir(parents=True, exist_ok=True)

        # Damage ranges for the probability map
        damage_ranges = [
            (1, 0, 1),
            (2, 100, 15000),
            (3, 15000, 30000),
            (4, 30000, 45000),
            (5, 45000, 60000),
            (6, 60000, None),
        ]
        # TODO: need to create a json dictionary with the right damage ranges

        # Load the ensemble of damage maps
        damage_ensemble = self.load_ensemble_damage_maps(date_time)

        # Get buildings geometry from one ensemble member (needed for the merges later)
        building_geometry = damage_ensemble[damage_ensemble["member"] == 1][
            ["building_id", "geometry"]
        ].copy()

        # Total number of ensemble members
        n_members = damage_ensemble["member"].nunique()

        # Loop over damage ranges and calculate probabilities
        for range_id, min_d, max_d in damage_ranges:
            if max_d is not None:
                condition = damage_ensemble["damages"].between(
                    min_d, max_d, inclusive="left"
                )
            else:
                condition = damage_ensemble["damages"] >= min_d

            # Count how many times each building falls in the range
            range_counts = (
                damage_ensemble[condition]
                .groupby("building_id")
                .size()
                .rename("count")
                .reset_index()
            )

            # Include all buildings and calculate the probability for each building in the range
            range_counts = pd.merge(
                building_geometry[["building_id"]],
                range_counts,
                on="building_id",
                how="left",
            )
            range_counts["count"] = range_counts["count"].fillna(0)
            range_counts["range_id"] = range_id
            range_counts["probability"] = range_counts["count"] / n_members

            damage_probability_map = pd.merge(
                range_counts, building_geometry, on="building_id", how="left"
            )
            damage_probability_map = gpd.GeoDataFrame(
                damage_probability_map, geometry="geometry", crs=crs
            )

            output_path = (
                damage_prob_maps_folder
                / f"damage_prob_map_range{range_id}_forecast{date_time.isoformat().replace(':', '').replace('-', '')}.geoparquet"
            )
            damage_probability_map.to_parquet(output_path)

    def identify_flooded_buildings(
        self,
        buildings: gpd.GeoDataFrame,
        prob_map: xr.DataArray,
        prob_threshold: float,
        return_series: bool = False,
    ) -> gpd.GeoDataFrame | pd.Series:
        """
        Identifies which buildings are flooded based on the flood probability map and a specified probability threshold.

        Args:
            buildings: GeoDataFrame with building geometries.
            prob_map: GeoDataFrame with flood probability values.
            prob_threshold: Probability threshold for identifying flooded buildings.
            return_series: If True, returns a pandas Series; otherwise, returns a GeoDataFrame.

        Returns:
            GeoDataFrame or Series with flooded building information.
        """
        prob_map = prob_map >= prob_threshold  # convert to boolean mask

        buildings["flooded"] = False  # Initialize the flooded column

        # convert flood map to polygons
        prob_map_polygon = from_landuse_raster_to_polygon(
            prob_map.values,
            prob_map.rio.transform(recalc=True),
            prob_map.rio.crs,
        )

        prob_map_polygons_union = prob_map_polygon.union_all()

        # Create a mask for buildings that overlap with the flood map
        buildings_mask = buildings.geometry.intersects(prob_map_polygons_union)

        # Update the flood_proofed status for buildings that overlap with the flood map
        buildings.loc[buildings_mask, "flooded"] = True

        if return_series:
            return buildings["flooded"].copy()
        else:
            return buildings

    def get_weight(self, weights_table: pd.DataFrame, row: pd.Series) -> float:
        """
        Get the communication efficiency weight for a household based on its education and income.

        Args:
            weights_table: DataFrame with communication efficiency weights indexed by education and income categories.
            row: Series with the household's education and income information.

        Returns:
            The communication efficiency weight for the household.
        """
        edu = row["Education"]
        inc = row["Income_Category"]
        return weights_table.loc[edu, inc]

    def calculate_communication_efficiency_probability(
        self,
        target_households: gpd.GeoDataFrame,
    ) -> pd.Series:
        """
        Calculate communication efficiency probability based on education level and income.

        Higher education and income lead to better access to communication channels
        and faster response to warnings. Education and Income data are automatically
        retrieved from self.var.education_level and self.var.income arrays and mapped
        to the target households based on their indices.

        Args:
            target_households: GeoDataFrame with household data including education and income.

        Returns:
            Series with communication efficiency probabilities for each household.

        Raises:
            KeyError: If required columns are missing from target_households.
        """
        # 1. Check if required columns exist
        required_columns = ["Education", "Income"]
        missing_columns = [
            col for col in required_columns if col not in target_households.columns
        ]

        # If required columns are missing, use uniform random probabilities as fallback
        if missing_columns:
            raise KeyError(
                f"Missing required columns in target_households: {missing_columns}. "
                f"Education and Income data should have been added from self.var arrays."
            )

        # 2. Normalized Weights based on the regression of the WRP survey data
        # (https://documents1.worldbank.org/curated/en/099259309032538041/pdf/IDU-c6f56dc5-a0cb-4375-ac15-a91f1c202b09.pdf )
        # Rows = Education classes, Columns = Income quintiles
        weights_table = pd.DataFrame(
            data=[
                [0.0336, 0.0349, 0.0366, 0.0380, 0.0396],
                [0.0336, 0.0349, 0.0366, 0.0380, 0.0396],
                [0.0366, 0.0380, 0.0396, 0.0413, 0.0430],
                [0.0400, 0.0413, 0.0430, 0.0450, 0.0467],
                [0.0407, 0.0423, 0.0439, 0.0456, 0.0477],
            ],
            index=np.array([1, 2, 3, 4, 5]),  # Education classes
            columns=np.array([1, 2, 3, 4, 5]),  # Income quintiles
        )

        # 3. Compute total weight per agent
        # Convert Income from absolute values to categories (1-5)
        # Handle duplicates by using rank-based percentiles instead of qcut
        income_values = target_households["Income"]

        # Use percentile-based categorization that handles duplicates
        income_percentiles = income_values.rank(method="average", pct=True)

        # Map percentiles to categories 1-5
        income_categories = pd.cut(
            income_percentiles,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=[1, 2, 3, 4, 5],
            include_lowest=True,
        )
        target_households["Income_Category"] = income_categories.astype(int)

        target_households["weight"] = target_households.apply(
            lambda row: self.get_weight(weights_table=weights_table, row=row), axis=1
        )
        weights = target_households["weight"].values

        return weights

    def compute_lead_time(self) -> float:
        """Compute lead time in hours based on forecast start time and current model time.

        Returns:
            float: Lead time in hours.
        """
        moment_of_flooding = self.model.config["hazards"]["floods"][
            "moment_of_flooding"
        ]
        current_time = self.model.current_time
        lead_time = (moment_of_flooding - current_time).total_seconds() / 3600
        lead_time = max(lead_time, 0)  # Ensure non-negative lead time
        return lead_time

    def pick_recommendations(
        self, available_measures: list[str], evacuate: bool, lead_time: float
    ) -> list[str]:
        """
        Decide which recommendations to include in the warning based on the lead time available.

        Args:
            available_measures (set[str]): Set of possible in-place measures to recommend.
            evacuate (bool): Whether evacuation is requested.
            lead_time (float): Lead time available in hours.

        Returns:
            chosen (list[str]): List of measures to recommend.
        """
        # Get implementation times for measures
        implementation_times = self.households.var.implementation_times
        available_measures = [
            m for m in available_measures if m and m in implementation_times
        ]
        # TODO: available measures is huge, need to know why

        self.logger.debug(f"Available_measures: {available_measures}")
        self.logger.debug(
            f"Implementation_times keys: {list(implementation_times.keys())}"
        )

        # Filter out empty strings and invalid measures to prevent KeyError
        valid_measures = {
            m for m in available_measures if m and m in implementation_times
        }
        self.logger.debug(f"Valid_measures after filtering: {valid_measures}")

        # Sort the measures by their implementation times and alphabetically
        sorted_inplace_measures_per_time = sorted(
            valid_measures,
            key=lambda measure: (implementation_times[measure], measure),
        )

        # Get evacuation time
        evac_time = implementation_times["evacuate"]

        # Non-evacuation case
        # If all measures fit in the lead time, return all
        if not evacuate:
            total = sum(
                implementation_times[measure]
                for measure in sorted_inplace_measures_per_time
            )
            if lead_time >= total:
                return sorted_inplace_measures_per_time

            else:
                # If not, pick as many measures as possible within the lead time
                chosen_measures = []
                used_time = 0
                for measure in sorted_inplace_measures_per_time:
                    imp_time = implementation_times[measure]
                    if used_time + imp_time <= lead_time:
                        chosen_measures.append(measure)
                        used_time += imp_time
                if not chosen_measures:
                    print("No measures fit within the lead time")
                return chosen_measures
        else:
            # Evacuation case
            # If there is no time for evacuation, nothing to recommend
            if evac_time > lead_time:
                print("Not enough time for evacuation, cannot recommend it")
                return []
            else:
                # If there is time for evacuation, check if any in-place measures fit as well
                chosen_measures = ["evacuate"]
                used_time = evac_time
                for measure in sorted_inplace_measures_per_time:
                    imp_time = implementation_times[measure]
                    if used_time + imp_time <= lead_time:
                        chosen_measures.append(measure)
                        used_time += imp_time

                return chosen_measures

    def warning_communication(
        self,
        target_households: gpd.GeoDataFrame,
        measures: list[str],
        evacuate: bool,
        trigger: str,
        communication_efficiency: float = 1,
        evacuation_lead_time_threshold: int = 48,
        weight_by_socioeconomic_factors: bool = False,
    ) -> int:
        """Send warnings to a subset of target households according to communication_efficiency.

        Makes sure to send a single message and allows only upward escalation:
        none (0) -> measures (1) -> evacuate (2)
        Evacuation warning only sent if evacuate is True and lead_time <= threshold.

        Args:
            target_households: GeoDataFrame of household points targeted by the warning.
            measures: List of recommended protective measures to communicate (strings).
            evacuate: Whether evacuation should be advised for this warning.
            trigger: Identifier of the trigger that initiated the warning.
            communication_efficiency: Fraction of target households that successfully receive the warning (0 to 1).
            evacuation_lead_time_threshold: Maximum lead time in hours for which evacuation warnings can be effective.
            weight_by_socioeconomic_factors: Whether to weight the selection of households by socio-economic factors (education, income) or select randomly.

        Returns:
            int: Number of households that were successfully warned.

        Raises:
            ValueError: If communication_efficiency is too low to warn any households.
        """
        self.logger.info("Running the warning communication for households...")
        # Get the number of target households
        n_target_households = len(target_households)
        if n_target_households == 0:
            self.logger.info("No target households to warn.")
            return 0

        # Select exactly communication_efficiency fraction of households using socio-economic weights
        n_to_select = int(communication_efficiency * n_target_households)
        if n_to_select == 0:
            raise ValueError(
                "Communication efficiency is too low to warn any households. Please increase it or check the target households."
            )

        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        if weight_by_socioeconomic_factors:
            # Calculate individual communication efficiency probabilities based on socio-economic factors
            warning_weights = self.calculate_communication_efficiency_probability(
                target_households
            )
            # Normalize weights to ensure they sum to exactly 1.0 (avoid floating point precision errors)
            warning_weights = warning_weights / warning_weights.sum()
            # Use weighted random sampling to select exactly n_to_select households
            chosen_indices = rng.choice(
                target_households.index,
                size=n_to_select,
                replace=False,  # Each household can only be selected once for warning
                p=warning_weights,
            )
        else:
            chosen_indices = rng.choice(
                target_households.index,
                size=n_to_select,
                replace=False,  # Each household can only be selected once for warning
            )

        selected_households = target_households.loc[chosen_indices]
        n_feasible_warnings = len(selected_households)

        self.logger.info(
            f"Communication efficiency resulted in {n_feasible_warnings} out of {n_target_households} households receiving warning by using random sampling {'with' if weight_by_socioeconomic_factors else 'without'} socio-economic weighting."
        )

        # Get lead time in hours
        lead_time = self.compute_lead_time()
        self.logger.info(f"Lead time for warning: {lead_time:.0f} hours")

        # Pick the measures to recommend based on lead time
        recommended_measures = self.pick_recommendations(measures, evacuate, lead_time)

        # Policy check for evacuation (only recommend it if lead time <= threshold)
        evac_feasible = "evacuate" in recommended_measures
        evac_policy = lead_time <= evacuation_lead_time_threshold
        if evac_feasible and not evac_policy:
            self.logger.info(
                f"Evacuation feasible but not recommended because of >{evacuation_lead_time_threshold} lead time policy."
            )
            recommended_measures.remove("evacuate")
            evac_feasible = False

        # Determine the desired level of warning (0 = no warning, 1 = measures, 2 = evacuate)
        desired_level = 2 if evac_feasible else (1 if recommended_measures else 0)

        if desired_level == 0:
            return 0

        # Get possible measures and triggers (this is a list of all possible measures, not the recommended measures given by the warning strategies)
        # Used only to make sure the indices are correct when updating the arrays (self.var.recommended_measures, self.var.warning_trigger)
        # TODO: Include water level ranges in the log files
        possible_measures_to_recommend = self.households.var.possible_measures
        possible_warning_triggers = self.households.var.possible_warning_triggers

        n_warned_households = 0
        for household_id in selected_households.index:
            current_level = int(self.households.var.warning_level[household_id])

            # Only send a warning if the desired level is higher than current level
            if desired_level > current_level:
                for measure in recommended_measures:
                    measure_idx = possible_measures_to_recommend.index(measure)
                    self.households.var.recommended_measures[
                        household_id, measure_idx
                    ] = True
            else:
                continue
            self.households.var.warning_level[household_id] = desired_level
            self.households.var.warning_reached[household_id] = 1
            trigger_idx = possible_warning_triggers.index(trigger)
            self.households.var.warning_trigger[household_id, trigger_idx] = True
            n_warned_households += 1
        self.logger.info(f"Warning targeted to reach {n_target_households} households")
        self.logger.info(f"Warning reached {n_warned_households} households")
        return n_warned_households

    def water_level_warning_strategy(
        self,
        date_time: datetime,
        warning_type: str = "building_based",
        prob_threshold: float = 0.6,
        buildings_hit_threshold: float = 0.1,
        area_hit_threshold: float = 0.1,
        communication_efficiency: float = 1,
        evacuation_lead_time_threshold: int = 48,
        weight_by_socioeconomic_factors: bool = False,
        exceedance: bool = True,
    ) -> None:
        """
        Implements the water level warning strategy based on flood probability maps BUCKETS PER MEASURE.

        Args:
            date_time: The forecast date time for which to implement the warning strategy.
            warning_type: The type of warning strategy to implement.
            prob_threshold: The probability threshold above which a warning is issued.
            buildings_hit_threshold: The threshold (percentage of buildings hit) above which a warning is issued.
            area_hit_threshold: The threshold (percentage of area hit) above which a warning is issued.
            communication_efficiency: The efficiency of the warning communication.
            evacuation_lead_time_threshold: The threshold for the lead time required for evacuation.
            weight_by_socioeconomic_factors: Whether to consider the social economic factors in the distribution of warnings.
            exceedance: Whether to create a probability map based on exceedance of a critical water thresholds or a probability of a waterlevel range.
        """
        # Get the range ids and initialize the warning_log
        residential_wlranges = self.households.var.wlranges_and_measures[
            "residential_buildings"
        ]
        range_ids = list(residential_wlranges.keys())
        warning_log = []
        warnings_folder = self.model.output_folder / "warning_logs"
        warnings_folder.mkdir(exist_ok=True, parents=True)

        # Create probability maps
        # TODO: Only create flood probability maps if they do not exist yet
        probability_maps = self.create_flood_probability_maps(
            strategy="residential_buildings", date_time=date_time, exceedance=exceedance
        )

        # Load households and postal codes
        buildings = self.households.var.buildings.copy()
        households = self.households.var.households_with_postal_codes.copy()
        # Add education and income data to the households
        # IMPORTANT: We use the index of households to correctly map the data
        # because household_points.index corresponds to the arrays in self.var
        if "Education" not in households.columns:
            households["Education"] = households.index.map(
                lambda idx: (
                    self.households.var.education_level.data[idx]
                    if idx < len(self.households.var.education_level.data)
                    else np.nan
                )
            )
        if "Income" not in households.columns:
            households["Income"] = households.index.map(
                lambda idx: (
                    self.households.var.income.data[idx]
                    if idx < len(self.households.var.income.data)
                    else np.nan
                )
            )

        postal_codes = self.households.postal_codes
        # Maybe load this as a global var (?) instead of loading it each time

        # Check intersection between flood maps and buildings or postal codes
        for range_id in range_ids:
            flood_probability_map = probability_maps.sel(range_id=range_id)
            if warning_type == "building_based":
                hit = self.identify_flooded_buildings(
                    buildings, flood_probability_map, prob_threshold, return_series=True
                )
                buildings[f"hit_r{range_id}"] = hit

            elif warning_type == "area_based":
                if postal_codes.crs != flood_probability_map.rio.crs:
                    postal_codes = postal_codes.to_crs(flood_probability_map.rio.crs)

                # Rasterize postal codes to the same grid as the probability map
                pc_mask = rasterize(
                    ((geom, i) for i, geom in enumerate(postal_codes.geometry)),
                    out_shape=(
                        flood_probability_map.rio.height,
                        flood_probability_map.rio.width,
                    ),
                    transform=flood_probability_map.rio.transform(),
                    all_touched=True,
                    fill=-1,
                )

                # Iterate through each postal code and check how many pixels exceed the threshold
                for i, pc_row in postal_codes.iterrows():
                    postal_code = pc_row["postcode"]

                    # Get the values for this postal code from the probability map
                    valid_pixels = flood_probability_map.values.squeeze()[pc_mask == i]

                    n_total = len(valid_pixels)

                    if n_total == 0:
                        print(f"No valid pixels found for postal code {postal_code}")
                        percentage = 0
                    else:
                        n_above = np.sum(valid_pixels >= prob_threshold)
                        percentage = n_above / n_total

                        issue_warning = percentage >= area_hit_threshold
                        postal_codes.loc[i, f"hit_r{range_id}"] = issue_warning
                        postal_codes.loc[postal_code, f"issue_warning_r{range_id}"] = (
                            issue_warning
                        )

                        if issue_warning:
                            self.logger.info(
                                f"Warning issued to postal code {postal_code} on {date_time.isoformat()} for range {range_id}: {percentage:.0%} of pixels > {prob_threshold}"
                            )

        if warning_type == "building_based":
            # Calculate fraction of flooded buildings per postal code
            hit_cols = [f"hit_r{rid}" for rid in range_ids]
            building_fraction_hit_per_postal_code = buildings.groupby("postcode")[
                hit_cols
            ].mean()
            building_fraction_hit_per_postal_code_map = postal_codes.merge(
                building_fraction_hit_per_postal_code.reset_index(),
                on="postcode",
                how="right",
            )
            building_fraction_hit_per_postal_code_map.to_parquet(
                warnings_folder
                / f"building_fraction_hit_per_postcode_{date_time.isoformat().replace(':', '').replace('-', '')}.parquet"
            )
            self.logger.info(
                f"Building fraction of buildings hit per postal code for forecast initialization date {date_time.isoformat()} saved as a parquet file."
            )

            # Iterate through each postal code and check the fraction of flooded buildings
            for postal_code, row in building_fraction_hit_per_postal_code.iterrows():
                for range_id in range_ids:
                    percentage_flooded = row[f"hit_r{range_id}"]
                    issue_warning = percentage_flooded >= buildings_hit_threshold
                    postal_codes.loc[
                        postal_codes["postcode"] == postal_code,
                        f"issue_warning_r{range_id}",
                    ] = issue_warning
                    if issue_warning:
                        self.logger.info(
                            f"Warning issued to postal code {postal_code} on {date_time.isoformat()} for range {range_id}: {percentage_flooded:.0%} of buildings flooded"
                        )
                    # TODO: need to think on how to include warning recommendation straight after this, instead of in a separate loop

        # Filter only postal codes that need to be warned
        warning_cols = [f"issue_warning_r{range_id}" for range_id in range_ids]
        postal_codes_to_warn = postal_codes[postal_codes[warning_cols].any(axis=1)]

        # Communicate the warning to the households in the postal code and store the details in the warning log
        # Initialize measures and evacuate flag
        measures = []
        triggered_ranges = []
        evacuate = False
        for i, row in postal_codes_to_warn.iterrows():
            postal_code = row["postcode"]
            for range_id in range_ids:
                issue_warning = row[f"issue_warning_r{range_id}"]
                if issue_warning:
                    # Get the measures and evacuation flag from the json dictionary to use in the warning communication function
                    triggered_ranges.append(range_id)
                    recom_measure = residential_wlranges[range_id]["measure"]
                    evacuate = evacuate or residential_wlranges[range_id]["evacuate"]
                    if recom_measure:
                        measures.extend(recom_measure)

            if measures or evacuate:
                # Filter the affected households based on the postal code
                affected_households = households[households["postcode"] == postal_code]

                n_warned_households = self.warning_communication(
                    target_households=affected_households,
                    measures=measures,
                    evacuate=evacuate,
                    trigger="water_levels",
                    communication_efficiency=communication_efficiency,
                    evacuation_lead_time_threshold=evacuation_lead_time_threshold,
                    weight_by_socioeconomic_factors=weight_by_socioeconomic_factors,
                )

                warning_log.append(
                    {
                        "date_time": date_time.isoformat(),
                        "postcode": postal_code,
                        "warning_type": warning_type,
                        "measures": measures,
                        "triggered_ranges": triggered_ranges,
                        "n_affected_households": len(affected_households),
                        "n_warned_households": n_warned_households,
                    }
                )

                # Save the warning log to a csv file
                path = (
                    warnings_folder
                    / f"warning_log_water_levels_{date_time.isoformat().replace(':', '').replace('-', '')}.csv"
                )
                pd.DataFrame(warning_log).to_csv(path, index=False)

    def critical_infrastructure_warning_strategy(
        self,
        date_time: datetime,
        config_asset_type: str,
        prob_threshold: float,
        exceedance: bool = False,
    ) -> None:
        """
        This function implements an evacuation warning strategy based on critical infrastructure elements, such as energy substations, vulnerable and emergency facilities.

        Args:
            date_time: The forecast date time for which to implement the warning strategy.
            config_asset_type: The type of critical infrastructure for which to issue warnings.
            prob_threshold: The probability threshold for issuing warnings.
            exceedance: Whether to consider exceedance probabilities.

        Raises:
            ValueError: If no assets are found for the specified asset type.
        """
        # Get the household points, needed to issue warnings
        households = self.households.var.households_with_postal_codes.copy()
        affected_postcodes = []
        for asset_type in config_asset_type:
            self.logger.info(
                f"Running critical infrastructure based warning strategy for asset type {asset_type}..."
            )
            # Load substations and critical facilities
            assets = read_geom(self.model.files["geom"][f"assets/{asset_type}"])
            if assets.empty:
                raise ValueError(
                    f"No assets found for type {asset_type}. Please check the input/files.yml whether the reference of the asset is similar to the one defined in the config."
                )

            # Create flood probability maps associated to the critical hit of energy substations
            # Need to give the number (id) of strategy as argument
            probability_maps = self.create_flood_probability_maps(
                strategy=asset_type, date_time=date_time, exceedance=exceedance
            )
            flood_probability_map: xr.DataArray = probability_maps[
                list(probability_maps.data_vars)[0]
            ]

            # Ensure prob_map is 2D by squeezing out any singleton dimensions
            prob_array = flood_probability_map.values
            while prob_array.ndim > 2:
                prob_array = prob_array.squeeze()

            # Sample the probability map at the substations locations
            x = xr.DataArray(assets.geometry.x.values, dims="z")
            y = xr.DataArray(assets.geometry.y.values, dims="z")
            assets["probability"] = flood_probability_map.sel(
                x=x, y=y, method="nearest"
            ).values

            # Filter substations that have a flood hit probability > threshold
            critical_hits_CI = assets[assets["probability"] >= prob_threshold]

            postal_codes_with_critical_infrastructure = gpd.read_parquet(
                self.model.input_folder
                / "geom"
                / "assets"
                / f"postal_codes_with_critical_infrastructure_{asset_type}.geoparquet"
            )

            # If there are critical hits, issue warnings
            if not critical_hits_CI.empty:
                self.logger.info(
                    f"Critical hits for {asset_type}: {len(critical_hits_CI)}"
                )
                # Get the postcodes that will be indirect affected by the critical infrastructure assets
                affected_assets = postal_codes_with_critical_infrastructure[
                    postal_codes_with_critical_infrastructure["asset_id"].isin(
                        critical_hits_CI["asset_id"]
                    )
                ]
                affected_assets["asset_type"] = asset_type
                affected_postcodes.append(
                    affected_assets[["postcode", "asset_id", "asset_type"]]
                )

        # Combine affected_postcodes from all asset types and remove duplicates
        affected_postcodes = pd.concat(affected_postcodes, ignore_index=True)
        affected_postcodes = (
            affected_postcodes.groupby("postcode")
            .agg(
                {
                    "asset_type": lambda x: list(x.unique()),
                    "asset_id": lambda x: list(x.unique()),
                }
            )
            .reset_index()
        )

        # Create an empty log to store the warnings
        warning_log = []

        # Issue warnings to the households in the affected postcodes
        for postcode in affected_postcodes:
            asset_type = affected_postcodes.loc[
                affected_postcodes["postcode"] == postcode, "asset_type"
            ]
            asset_id = affected_postcodes.loc[
                affected_postcodes["postcode"] == postcode, "asset_id"
            ]
            affected_households = households[households["postcode"] == postcode]
            n_warned_households = self.warning_communication(
                target_households=affected_households,
                measures=[],
                evacuate=True,
                trigger="critical_infrastructure",
            )

            print(
                f"Evacuation warning issued to postal code {postcode} on {date_time.isoformat()} (trigger: {asset_type})"
            )
            warning_log.append(
                {
                    "date_time": date_time.isoformat(),
                    "postcode": postcode,
                    "n_warned_households": n_warned_households,
                    "trigger": asset_type,
                    "asset_id": asset_id,
                }
            )

        # Save warning log
        warnings_folder = self.model.output_folder / "warning_logs"
        warnings_folder.mkdir(exist_ok=True, parents=True)
        path = (
            warnings_folder
            / f"warning_log_critical_infrastructure_{date_time.isoformat()}.csv"
        )
        pd.DataFrame(warning_log).to_csv(path, index=False)

    def household_decision_making(
        self, date_time: datetime, warning_type: str, responsive_ratio: float
    ) -> None:
        """
        Simulate household emergency response decisions based on warnings and lead time.

        Args:
            date_time: The forecast date time for which to run the decision-making.
            warning_type: The type of warning strategy that was implemented (building_based or area_based).
            responsive_ratio: The ratio of responsive households.
        """
        self.logger.info("Running emergency response decision-making for households...")
        self.logger.info(f"Date/time: {date_time}")
        self.logger.info(f"Warning type: {warning_type}")
        self.logger.info(f"Responsive ratio: {responsive_ratio * 100:.2f}%")

        # Get lead time in hours
        lead_time = self.compute_lead_time()
        self.logger.info(f"Lead time: {lead_time:.1f} hours")

        # Filter households that did not evacuate, were warned and are responsive
        not_evacuated_ids = self.households.var.evacuated == 0
        warned_ids = self.households.var.warning_reached == 1
        responsive_ids = self.households.var.response_probability < responsive_ratio

        self.logger.info(
            f"Total households: {len(self.households.var.household_points)}"
        )
        self.logger.info(f"Non-evacuated households: {np.sum(not_evacuated_ids)}")
        self.logger.info(f"Warned households: {np.sum(warned_ids)}")
        self.logger.info(f"Responsive households: {np.sum(responsive_ids)}")

        # Combine the filters and apply it to household_points
        eligible_ids = np.asarray(warned_ids & not_evacuated_ids & responsive_ids)
        eligible_households = self.households.var.households_with_postal_codes.loc[
            eligible_ids
        ]

        self.logger.info(f"Eligible households for action: {len(eligible_households)}")

        # Get the list of possible actions for eligible households
        # (this is a list of all possible actions, not the recommended actions given by the warning strategies)
        possible_actions = self.households.var.possible_measures
        evac_idx = possible_actions.index("evacuate")

        # Initialize an empty list to log actions taken
        actions_log = []

        self.logger.info(
            "Processing eligible households for action based on warnings and lead time..."
        )

        # For every eligible household, do the recommended measures in the communicated warning
        for i, household_id in enumerate(eligible_households.index):
            # Check if elevated possessions action is being taken for the FIRST TIME
            # (before updating actions_taken array)
            elevated_possessions_idx = possible_actions.index("elevate possessions")
            newly_taking_elevated_possessions = (
                not self.households.var.actions_taken[
                    household_id, elevated_possessions_idx
                ]
                and self.households.var.recommended_measures[
                    household_id, elevated_possessions_idx
                ]
            )

            # For measure in recommended measures, add new measures to existing actions_taken array
            # Use an OR operation to preserve previously taken actions while adding new ones
            for i, measure_recommended in enumerate(
                self.households.var.recommended_measures[household_id]
            ):
                if measure_recommended:
                    self.households.var.actions_taken[household_id, i] = True

            # Only update action_lead_time if elevated possessions action is being taken for the first time
            if newly_taking_elevated_possessions:
                self.households.var.action_lead_time[household_id] = lead_time

            # If evacuation is among the actions taken, mark household as evacuated
            if self.households.var.actions_taken[household_id, evac_idx]:
                self.households.var.evacuated[household_id] = 1

            # Log the actions taken
            actions = []
            for j, action in enumerate(possible_actions):
                if self.households.var.actions_taken[household_id, j]:
                    actions.append(action)

            # Get the warning_level (range_id) for this household
            warning_level = int(self.households.var.warning_level[household_id])

            actions_log.append(
                {
                    "lead_time": lead_time,
                    "date_time": date_time.isoformat(),
                    "postal_code": self.households.var.households_with_postal_codes.loc[
                        household_id, "postcode"
                    ],
                    "household_id": household_id,
                    "actions": actions,
                    "warning_level": warning_level,
                }
            )
        self.logger.info(f"Total number of households acted: {len(actions_log)}")
        total_actions = sum(len(entry["actions"]) for entry in actions_log)
        self.logger.info(f"Total individual actions taken: {total_actions}")

        # Save actions log
        actions_log_folder = self.model.output_folder / "actions_logs"
        actions_log_folder.mkdir(exist_ok=True, parents=True)
        path = (
            actions_log_folder
            / f"actions_log_{date_time.isoformat().replace(':', '').replace('-', '')}.csv"
        )
        pd.DataFrame(actions_log).to_csv(path, index=False)

    def update_households_geodataframe_w_warning_variables(
        self, date_time: datetime
    ) -> None:
        """This function merges the global variables related to warnings to the households geodataframe for visualization purposes.

        Args:
            date_time: The forecast date time for which to update the households geodataframe.
        """
        household_points: gpd.GeoDataFrame = (
            self.households.var.households_with_postal_codes.copy()
        )

        action_maps_folder: Path = self.model.output_folder / "action_maps"
        action_maps_folder.mkdir(parents=True, exist_ok=True)

        global_vars = [
            "warning_reached",
            "warning_level",
            "response_probability",
            "evacuated",
            "recommended_measures",
            "warning_trigger",
            "actions_taken",
            "action_lead_time",
        ]

        # make sure household points and global variables have the same length
        for global_var in global_vars:
            global_array = getattr(self.households.var, global_var)
            assert len(household_points) == global_array.shape[0], (
                f"The size of household points and {global_var} do not match"
            )

        # add columns in the household points geodataframe
        for name in [
            "warning_reached",
            "warning_level",
            "response_probability",
            "evacuated",
            "warning_trigger",
            "action_lead_time",
        ]:
            if hasattr(self.households.var, name):
                household_points[name] = getattr(self.households.var, name)

        warning_triggers = self.households.var.possible_warning_triggers
        for i, _ in enumerate(warning_triggers):
            if warning_triggers[i] == "water_levels":
                household_points["trig_w_levels"] = self.households.var.warning_trigger[
                    :, i
                ]
            if warning_triggers[i] == "critical_infrastructure":
                household_points["trig_crit_infra"] = (
                    self.households.var.warning_trigger[:, i]
                )

        possible_measures_to_recommend = self.households.var.possible_measures
        for i, measure in enumerate(possible_measures_to_recommend):
            if measure == "elevate possessions":
                household_points["recom_elev_possessions"] = (
                    self.households.var.recommended_measures[:, i]
                )
            if measure == "sandbags":
                household_points["recom_sandbags"] = (
                    self.households.var.recommended_measures[:, i]
                )
            if measure == "evacuate":
                household_points["recom_evacuate"] = (
                    self.households.var.recommended_measures[:, i]
                )

        possible_actions = self.households.var.possible_measures
        for i, action in enumerate(possible_actions):
            if action == "elevate possessions":
                household_points["elevated_possessions"] = (
                    self.households.var.actions_taken[:, i]
                )
            if action == "sandbags":
                household_points["sandbags"] = self.households.var.actions_taken[:, i]

        # Create the action_maps directory if it doesn't exist
        action_maps_dir = self.model.output_folder / "action_maps"
        action_maps_dir.mkdir(parents=True, exist_ok=True)
        household_points.to_parquet(
            action_maps_dir
            / f"households_with_warning_parameters_{date_time.strftime('%Y%m%dT%H%M%S')}.geoparquet"
        )
