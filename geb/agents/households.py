"""This module contains the Households agent class for simulating household behavior in the GEB model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import shapes
from rasterstats import point_query, zonal_stats
from scipy import interpolate
from shapely.geometry import shape

from geb.workflows.io import load_dict

from ..hydrology.landcovers import (
    FOREST,
)
from ..store import DynamicArray
from ..workflows.damage_scanner import VectorScanner
from ..workflows.io import load_array, load_table, open_zarr, to_zarr
from .decision_module import DecisionModule
from .general import AgentBaseClass

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel


def from_landuse_raster_to_polygon(mask, transform, crs) -> gpd.GeoDataFrame:
    """Convert raster data into separate GeoDataFrames for specified land use values.

    Args:
        mask: A 2D numpy array representing the land use raster, where each unique value corresponds to a different land use type.
        transform: A rasterio Affine transform object that defines the spatial reference of the raster.
        crs: The coordinate reference system (CRS) to use for the resulting GeoDataFrame.

    Returns:
        A GeoDataFrame containing polygons for the specified land use values.
    """
    shapes_gen = shapes(mask.astype(np.uint8), mask=mask, transform=transform)

    polygons = []
    for geom, _ in shapes_gen:
        polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=crs)

    return gdf


class Households(AgentBaseClass):
    """This class implements the household agents."""

    def __init__(self, model: GEBModel, agents: Agents, reduncancy: float) -> None:
        """Initialize the Households agent module.

        Args:
            model: The GEB model.
            agents: The class that includes all agent types (allowing easier communication between agents).
            reduncancy: a lot of data is saved in pre-allocated NumPy arrays.
                While this allows much faster operation, it does mean that the number of agents cannot
                grow beyond the size of the pre-allocated arrays. This parameter allows you to specify
                how much redundancy should be used. A lower redundancy means less memory is used, but the
                model crashes if the redundancy is insufficient. The redundancy is specified as a fraction of
                the number of agents, e.g. 0.2 means 20% more space is allocated than the number of agents.
        """
        super().__init__(model)

        self.agents = agents
        self.reduncancy = reduncancy

        if self.model.simulate_hydrology:
            self.HRU = model.hydrology.HRU
            self.grid = model.hydrology.grid

        self.config = (
            self.model.config["agent_settings"]["households"]
            if "households" in self.model.config["agent_settings"]
            else {}
        )
        self.decision_module = DecisionModule()

        if self.config["adapt"]:
            self.load_flood_maps()

        self.load_objects()
        self.load_max_damage_values()
        self.load_damage_curves()
        if self.model.config["agent_settings"]["households"]["warning_response"]:
            # self.load_critical_infrastructure()
            self.load_wlranges_and_measures()

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        return "agents.households"

    def load_flood_maps(self) -> None:
        """Load flood maps for different return periods. This might be quite ineffecient for RAM, but faster then loading them each timestep for now."""
        self.return_periods = np.array(
            self.model.config["hazards"]["floods"]["return_periods"]
        )

        flood_maps = {}
        for return_period in self.return_periods:
            file_path = (
                self.model.output_folder / "flood_maps" / f"{return_period}.zarr"
            )
            flood_maps[return_period] = open_zarr(file_path)
        self.flood_maps = flood_maps

    def construct_income_distribution(self) -> None:
        """Construct a lognormal income distribution for the region."""
        # These values only work for a single country now. Should come from subnational datasets.
        distribution_parameters = load_table(
            self.model.files["table"]["income/distribution_parameters"]
        )
        country = self.model.regions["ISO3"].values[0]
        average_household_income = distribution_parameters[country]["MEAN"]
        median_income = distribution_parameters[country]["MEDIAN"]

        # construct lognormal income distribution
        mu = np.log(median_income)
        sd = np.sqrt(2 * np.log(average_household_income / median_income))
        income_distribution_region = np.sort(
            np.random.lognormal(mu, sd, 15_000).astype(np.int32)
        )
        # set var
        self.var.income_distribution = income_distribution_region

    def assign_household_wealth_and_income(self) -> None:
        """Assign household wealth and income attributes based on GLOPOP-S and OECD data."""
        # initiate array with wealth indices
        wealth_index = load_array(
            self.model.files["array"]["agents/households/wealth_index"]
        )
        self.var.wealth_index = DynamicArray(wealth_index, max_n=self.max_n)

        income_percentiles = load_array(
            self.model.files["array"]["agents/households/income_percentile"]
        )
        self.var.income_percentile = DynamicArray(income_percentiles, max_n=self.max_n)

        # assign household disposable income based on income percentile households
        income = load_array(self.model.files["array"]["agents/households/disp_income"])
        self.var.income = DynamicArray(income, max_n=self.max_n)

        # assign wealth based on income (dummy data, there are ratios available in literature)
        self.var.wealth = DynamicArray(2.5 * self.var.income.data, max_n=self.max_n)

    def update_building_attributes(self) -> None:
        """Update building attributes based on household data."""
        # Start by computing occupancy from the var.building_id array
        building_id_series = pd.Series(self.var.building_id.data)

        # Drop NaNs and convert to string format (matching building ID format)
        building_id_counts = building_id_series.dropna().astype(int).value_counts()
        # Initialize occupancy column
        self.buildings["occupancy"] = 0

        # Map the counts back to the buildings dataframe
        self.buildings["occupancy"] = (
            self.buildings["id"]
            .map(building_id_counts)
            .fillna(self.buildings["occupancy"])
        )

        # Initialize dry floodproofing status
        self.buildings["flood_proofed"] = False

        # check if building overlaps with the flood map
        # get highest return period
        highest_return_period = self.return_periods.max()
        flood_map = self.flood_maps[highest_return_period].copy()
        # check if building geometry overlaps with flood map
        flood_map = flood_map.rio.reproject(self.buildings.crs)
        flood_map = flood_map > 0  # convert to boolean mask

        # convert flood map to polygons
        flood_map_polygons = from_landuse_raster_to_polygon(
            flood_map.values,
            flood_map.rio.transform(recalc=True),
            flood_map.rio.crs,
        )

        flood_map_polygons_union = flood_map_polygons.union_all()

        # Create a mask for buildings that overlap with the flood map
        buildings_mask = self.buildings.geometry.intersects(flood_map_polygons_union)

        # Update the flood_proofed status for buildings that overlap with the flood map
        self.buildings.loc[buildings_mask, "flooded"] = True
        self.buildings["flooded"].fillna(False, inplace=True)

    def update_building_adaptation_status(self, household_adapting: np.ndarray) -> None:
        """Update the floodproofing status of buildings based on adapting households."""
        # Extract and clean OSM IDs from adapting households
        building_id = pd.DataFrame(
            np.unique(self.var.building_id.data[household_adapting])
        ).dropna()
        building_id = building_id.astype(int).astype(str)
        building_id["flood_proofed"] = True
        building_id = building_id.set_index(0)

        # Add/Update the flood_proofed status in buildings based on OSM way IDs
        self.buildings["flood_proofed"] = (
            self.buildings["id"].astype(str).map(building_id["flood_proofed"])
        )

        # Replace NaNs with False (i.e., buildings not in the adapting households list)
        self.buildings["flood_proofed"] = self.buildings["flood_proofed"].fillna(False)

    def assign_household_attributes(self) -> None:
        """Household locations are already sampled from population map in GEBModel.setup_population().

        These are loaded in the spinup() method.
        Here we assign additional attributes (dummy data) to the households that are used in the decision module.
        """
        # load household locations
        locations = load_array(self.model.files["array"]["agents/households/location"])
        self.max_n = int(locations.shape[0] * (1 + self.reduncancy) + 1)
        self.var.locations = DynamicArray(locations, max_n=self.max_n)

        self.var.region_id = load_array(
            self.model.files["array"]["agents/households/region_id"]
        )

        # load household sizes
        sizes = load_array(self.model.files["array"]["agents/households/size"])
        self.var.sizes = DynamicArray(sizes, max_n=self.max_n)

        self.var.municipal_water_demand_per_capita_m3_baseline = load_array(
            self.model.files["array"][
                "agents/households/municipal_water_demand_per_capita_m3_baseline"
            ]
        )

        # set municipal water demand efficiency to 1.0 for all households
        self.var.water_efficiency_per_household = np.full_like(
            self.var.municipal_water_demand_per_capita_m3_baseline, 1.0, np.float32
        )

        self.var.municipal_water_withdrawal_m3_per_capita_per_day_multiplier = (
            load_table(
                self.model.files["table"][
                    "municipal_water_withdrawal_m3_per_capita_per_day_multiplier"
                ]
            )
        )

        # load building id household
        building_id = load_array(
            self.model.files["array"]["agents/households/building_id"]
        )
        self.var.building_id = DynamicArray(building_id, max_n=self.max_n)

        # update building attributes based on household data
        if self.config["adapt"]:
            self.update_building_attributes()

        # load age household head
        age_household_head = load_array(
            self.model.files["array"]["agents/households/age_household_head"]
        )
        self.var.age_household_head = DynamicArray(age_household_head, max_n=self.max_n)

        # load education level household head
        education_level = load_array(
            self.model.files["array"]["agents/households/education_level"]
        )
        self.var.education_level = DynamicArray(education_level, max_n=self.max_n)

        # initiate array for adaptation status [0=not adapted, 1=dryfloodproofing implemented]
        self.var.adapted = DynamicArray(np.zeros(self.n, np.int32), max_n=self.max_n)

        # initiate array for warning state [0 = not warned, 1 = warned]
        self.var.warning_reached = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
        )

        # initiate array for warning level [0 = no warning, 1 = measures, 2 = evacuate]
        self.var.warning_level = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
        )

        # initiate array for storing the trigger of the warning
        self.var.possible_warning_triggers = ["critical_infrastructure", "water_levels"]
        self.var.warning_trigger = DynamicArray(
            np.zeros((self.n, len(self.var.possible_warning_triggers)), dtype=bool)
        )

        # initiate array for response probability (between 0 and 1)
        # using a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        self.var.response_probability = DynamicArray(
            rng.random(self.n), max_n=self.max_n
        )

        # initiate array for evacuation status [0=not evacuated, 1=evacuated]
        self.var.evacuated = DynamicArray(np.zeros(self.n, np.int32), max_n=self.max_n)

        # initiate array for storing the recommended measures received with the warnings
        self.var.possible_measures_to_recommend = [
            "elevate possessions",
            "evacuate",
        ]
        self.var.recommended_measures = DynamicArray(
            np.zeros((self.n, len(self.var.possible_measures_to_recommend)), dtype=bool)
        )

        # initiate array for storing the actions taken by the household
        self.var.possible_actions = ["elevate possessions", "evacuate"]
        self.var.actions_taken = DynamicArray(
            np.zeros((self.n, len(self.var.possible_actions)), dtype=bool)
        )

        # initiate array with risk perception [dummy data for now]
        self.var.risk_perc_min = self.model.config["agent_settings"]["households"][
            "expected_utility"
        ]["flood_risk_calculations"]["risk_perception"]["min"]
        self.var.risk_perc_max = self.model.config["agent_settings"]["households"][
            "expected_utility"
        ]["flood_risk_calculations"]["risk_perception"]["max"]
        self.var.risk_decr = self.model.config["agent_settings"]["households"][
            "expected_utility"
        ]["flood_risk_calculations"]["risk_perception"]["coef"]

        risk_perception = np.full(self.n, self.var.risk_perc_min)
        self.var.risk_perception = DynamicArray(risk_perception, max_n=self.max_n)

        # initiate array with risk aversion [fixed for now]
        self.var.risk_aversion = DynamicArray(np.full(self.n, 1), max_n=self.max_n)

        # initiate array with time adapted
        self.var.time_adapted = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
        )

        # initiate array with time since last flood
        self.var.years_since_last_flood = DynamicArray(
            np.full(self.n, 25, np.int32), max_n=self.max_n
        )

        # assign income and wealth attributes
        self.assign_household_wealth_and_income()

        # initiate array with property values (used as max damage) [dummy data for now, could use Huizinga combined with building footprint to calculate better values]
        self.var.property_value = DynamicArray(
            np.int64(self.var.wealth.data * 0.8), max_n=self.max_n
        )
        # initiate array with RANDOM annual adaptation costs [dummy data for now, values are available in literature]
        adaptation_costs = np.int64(
            np.maximum(self.var.property_value.data * 0.05, 10_800)
        )
        self.var.adaptation_costs = DynamicArray(adaptation_costs, max_n=self.max_n)

        # initiate array with amenity value [dummy data for now, use hedonic pricing studies to calculate actual values]
        amenity_premiums = np.random.uniform(0, 0.2, self.n)
        self.var.amenity_value = DynamicArray(
            amenity_premiums * self.var.wealth, max_n=self.max_n
        )

        # load household points (only in use for damagescanner, could be removed)
        household_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                self.var.locations.data[:, 0], self.var.locations.data[:, 1]
            ),
            crs="EPSG:4326",
        )

        self.var.household_points = household_points
        print(
            f"Household attributes assigned for {self.n} households with {self.population} people."
        )

    def _load_postal_codes_clipped(self) -> gpd.GeoDataFrame:
        """Load postal codes efficiently clipped to model region.

        This method loads the postal codes dataset and clips it to the model region
        to optimize memory usage and performance when the postal codes dataset
        is much larger than the catchment area.

        Returns:
            gpd.GeoDataFrame: Postal codes clipped to model region.
        """
        # Load full postal codes dataset
        postal_codes: gpd.GeoDataFrame = gpd.read_parquet(
            self.model.files["geom"]["postal_codes"]
        )

        # Get model region bounds for clipping
        if hasattr(self.model, "regions") and self.model.regions is not None:
            # Use model regions if available
            model_bounds = self.model.regions.total_bounds
            # Create bounding box geometry
            from shapely.geometry import box

            bbox = box(
                model_bounds[0], model_bounds[1], model_bounds[2], model_bounds[3]
            )
            bbox_gdf = gpd.GeoDataFrame(
                [1], geometry=[bbox], crs=self.model.regions.crs
            )

            # Ensure same CRS for intersection
            postal_codes = postal_codes.to_crs(bbox_gdf.crs)

            # Clip postal codes to model region with buffer for edge cases
            buffer_distance = (
                0.1  # degrees buffer to ensure no missing postcodes at edges
            )
            bbox_buffered = bbox_gdf.buffer(buffer_distance)
            postal_codes_clipped = postal_codes[
                postal_codes.intersects(bbox_buffered.geometry.iloc[0])
            ].copy()

            print(
                f"Postal codes clipped from {len(postal_codes)} to {len(postal_codes_clipped)} records"
            )
            return postal_codes_clipped
        else:
            # Fallback: return full dataset if no region bounds available
            print("No model region bounds available, loading full postal codes dataset")
            return postal_codes

    def change_household_locations(self) -> None:
        """Change the location of the household points to the centroid of the buildings.

        Also, it associates the household points with their postal codes.
        This is done to get the correct geometry for the warning function.
        """
        # crs: str = self.model.sfincs.crs
        crs = self.model.config["hazards"]["floods"]["crs"]

        locations = self.var.household_points.copy()
        locations.to_crs(
            crs, inplace=True
        )  # Change to a projected CRS to get a distance in meters

        buildings_centroid = self.buildings_centroid[["geometry"]].copy()
        buildings_centroid.to_crs(crs, inplace=True)  # Change to the same projected CRS

        # Copy the geometry to a new column otherwise it gets lost in the spatial join
        buildings_centroid["new_geometry"] = buildings_centroid.geometry

        # Create unique ids for household points and building centroids
        # This is done to avoid duplicates when doing the spatial join
        locations["pointid"] = range(locations.shape[0])

        new_locations = gpd.sjoin_nearest(
            locations,
            buildings_centroid,
            how="left",
            exclusive=True,
            distance_col="distance",
        )

        # Sort values by pointid and distance and drop duplicate pointids
        new_locations = new_locations.sort_values(
            by=["pointid", "distance"], ascending=True, na_position="last"
        )
        new_locations = new_locations.drop_duplicates(subset="pointid", keep="first")

        # Change the geometry of the household points to the geometry of the building centroid
        new_locations["geometry"] = new_locations["new_geometry"]
        new_locations.set_geometry("geometry", inplace=True)

        # Drop columns that are not needed
        new_locations.drop(
            columns={"index_right", "new_geometry", "distance"}, inplace=True
        )

        # Associate households with their postal codes to use it later in the warning function
        postal_codes: gpd.GeoDataFrame = self._load_postal_codes_clipped()
        postal_codes["postcode"] = postal_codes["CD_SETOR"].astype(str)

        new_locations = gpd.sjoin(
            new_locations,
            postal_codes[["postcode", "geometry"]],
            how="left",
            predicate="intersects",
        )

        # Drop columns that are not needed
        new_locations.drop(columns=["index_right"], inplace=True)

        self.var.household_points = new_locations

    def update_risk_perceptions(self) -> None:
        """Update the risk perceptions of households based on the latest flood data."""
        # update timer
        self.var.years_since_last_flood.data += 1

        # generate random flood (not based on actual modeled flood data, replace this later with events)
        if np.random.random() < 0.1:
            print("Flood event!")
            self.var.years_since_last_flood.data = 0

        self.var.risk_perception.data = (
            self.var.risk_perc_max
            * 1.6 ** (self.var.risk_decr * self.var.years_since_last_flood)
            + self.var.risk_perc_min
        )

    def load_ensemble_flood_maps(self, date_time) -> xr.DataArray:
        """Loads the flood maps for all ensemble members for a specific forecast date time.

        Args:
            date_time: The forecast date time for which to load the flood maps.
        Returns:
            An xarray containing the flood maps for all ensemble members.

        Raises:
            FileNotFoundError: If no flood maps are found for the specified date time.
            FileExistsError: If multiple flood maps are found for the specified date time.
        """
        # Load all the flood maps in an ensemble per each day
        flood_start_time = self.model.config["hazards"]["floods"]["events"][0][
            "start_time"
        ]
        flood_end_time = self.model.config["hazards"]["floods"]["events"][0]["end_time"]
        ensemble_per_time = []

        members = []  # Every time has its own ensemble
        for member in range(1, 50 + 1):  # Define the number of members here
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

            members.append(xr.open_dataarray(flood_map_path, engine="zarr"))

        # Concatenate the members for each time (This stacks the list of dataarrays along a new "member" dimension)
        members_stacked = xr.concat(members, dim="member")
        ensemble_per_time.append(members_stacked)

        ensemble_flood_maps = xr.concat(ensemble_per_time, dim="time")

        return ensemble_flood_maps

    def load_ensemble_damage_maps(self):
        start_time = (
            "2021-07-13"  # CHECK LATER IF YOU WANT THE START TIME OR THE FORECAST DAYS
        )

        damage_maps = []
        # Load the damage maps for each ensemble member -- CHANGE IT LATER TO THE ACTUAL NUMBER OF MEMBERS
        for member in range(1, 4):
            file_path = (
                self.model.output_folder
                / "damage_maps"
                / f"damage_map_buildings_content_{start_time}_member{member}.gpkg"
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
        self, date_time, strategy=1
    ) -> dict[tuple[pd.Timestamp, int], xr.DataArray]:
        """Creates flood probability maps based on the ensemble of flood maps for different warning strategies.

        Args:
            date_time: The forecast date time for which to create the probability maps.
            strategy: Identifier of the warning strategy (1 for water level ranges with measures,
                2 for energy substations, 3 for vulnerable/emergency facilities).

        Returns:
            Dict[Tuple[datetime, int], xr.DataArray]: Dictionary mapping (date_time, range_id) tuples
                to probability maps (xarray DataArray) for that forecast day and water level range.
        """
        # Load the ensemble of flood maps for that specific date time
        ensemble_flood_maps = self.load_ensemble_flood_maps(date_time=date_time)
        crs = self.model.config["hazards"]["floods"]["crs"]

        # Create output folder for probability maps
        prob_folder = (
            self.model.output_folder
            / "prob_maps"
            / f"forecast_{date_time.strftime('%Y%m%dT%H%M%S')}"
        )
        prob_folder.mkdir(exist_ok=True, parents=True)

        # Define water level ranges based on the chosen strategy
        if strategy == 1:
            # Water level ranges associated to specific measures (based on "impact" scale)
            ranges = []
            for key, value in self.var.wlranges_and_measures.items():
                ranges.append((key, value["min"], value["max"]))

        elif strategy == 2:
            # Water level range for energy substations (based on critical hit) -- need to make it nor hard coded
            ranges = [(1, 0.3, None)]

        else:
            # Water level range for vulnerable and emergency facilities (flooded or not)
            ranges = [(1, 0.1, None)]

        probability_maps = {}

        # Loop over water level ranges to calculate probability maps
        for range_id, min, max in ranges:
            daily_ensemble = ensemble_flood_maps
            if max is not None:
                condition = (daily_ensemble >= min) & (daily_ensemble <= max)
            else:
                condition = daily_ensemble >= min
            probability = condition.sum(dim="member") / condition.sizes["member"]

            # Save probability map as a zarr file
            file_name = f"prob_map_range{range_id}_strategy{strategy}.zarr"
            file_path = prob_folder / file_name
            file_path.mkdir(parents=True, exist_ok=True)

            # The y axis is flipped when writing to zarr, so fixing it here for now
            if probability.y.values[0] < probability.y.values[-1]:
                print("flipping y axis")
                probability = probability.sortby("y", ascending=False)

            probability = probability.astype(np.float32)
            probability = probability.rio.write_nodata(np.nan)
            probability = probability.isel(time=0)  # select the first time step
            probability = to_zarr(da=probability, path=file_path, crs=crs)

            probability_maps[(date_time, range_id)] = probability

        return probability_maps

    def create_flood_exceedance_probability_maps(
        self, date_time, strategy=1
    ) -> dict[tuple[pd.Timestamp, int], xr.DataArray]:
        """Creates flood exceedance probability maps based on the ensemble of flood maps for different warning strategies.

        Args:
            date_time: The forecast date time for which to create the probability maps.
            strategy: Identifier of the warning strategy (1 for water level ranges with measures,
                2 for energy substations, 3 for vulnerable/emergency facilities).

        Returns:
            Dict[Tuple[datetime, int], xr.DataArray]: Dictionary mapping (date_time, range_id) tuples
                to probability maps (xarray DataArray) for that forecast day and water level range.
        """
        # Load the ensemble of flood maps for that specific date time
        ensemble_flood_maps = self.load_ensemble_flood_maps(date_time=date_time)
        crs = self.model.config["hazards"]["floods"]["crs"]

        # Create output folder for probability maps
        prob_folder = (
            self.model.output_folder
            / "prob_exceedance_maps"
            / f"forecast_{date_time.strftime('%Y%m%dT%H%M%S')}"
        )
        prob_folder.mkdir(exist_ok=True, parents=True)

        # Define water level ranges based on the chosen strategy
        if strategy == 1:
            # Water level ranges associated to specific measures (based on "impact" scale)
            ranges = []
            for key, value in self.var.wlranges_and_measures.items():
                ranges.append((key, value["min"], None))

        elif strategy == 2:
            # Water level range for energy substations (based on critical hit) -- need to make it nor hard coded
            ranges = [(1, 0.3, None)]

        else:
            # Water level range for vulnerable and emergency facilities (flooded or not)
            ranges = [(1, 0.1, None)]

        probability_maps = {}

        # Loop over water level ranges to calculate probability maps
        for range_id, min, max in ranges:
            daily_ensemble = ensemble_flood_maps
            if max is not None:
                condition = (daily_ensemble >= min) & (daily_ensemble <= max)
            else:
                condition = daily_ensemble >= min
            probability = condition.sum(dim="member") / condition.sizes["member"]

            # Save probability map as a zarr file
            file_name = f"prob_exceedance_map_range{range_id}_strategy{strategy}.zarr"
            file_path = prob_folder / file_name
            file_path.mkdir(parents=True, exist_ok=True)

            # The y axis is flipped when writing to zarr, so fixing it here for now
            if probability.y.values[0] < probability.y.values[-1]:
                print("flipping y axis")
                probability = probability.sortby("y", ascending=False)

            probability = probability.astype(np.float32)
            probability = probability.rio.write_nodata(np.nan)
            probability = probability.isel(time=0)  # select the first time step
            probability = to_zarr(da=probability, path=file_path, crs=crs)

            probability_maps[(date_time, range_id)] = probability

        return probability_maps

    def create_damage_probability_maps(self) -> None:
        """Creates an object-based (buildings) probability map based on the ensemble of damage maps."""
        crs = self.model.config["hazards"]["floods"]["crs"]
        days = self.model.config["general"]["forecasts"]["days"]

        # Damage ranges for the probability map
        damage_ranges = [
            (1, 0, 1),
            (2, 0, 1500000),
            (3, 1500000, 7500000),
            (4, 7500000, None),
        ]
        # NEED TO CREATE A PARQUET FILE WITH THE RIGHT RANGES

        # Load the ensemble of damage maps
        damage_ensemble = self.load_ensemble_damage_maps()

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
                self.model.output_folder
                / "prob_maps"
                / f"damage_prob_map_range{range_id}_forecast{days[0].strftime('%Y-%m-%d')}.gpkg"
            )

            damage_probability_map.to_file(output_path)

    def water_level_warning_strategy(
        self, date_time, prob_threshold=0.6, area_threshold=0.1, strategy_id=1
    ) -> None:
        """Implements the water level warning strategy based on flood probability maps.

        Args:
            date_time: The forecast date time for which to implement the warning strategy.
            prob_threshold: The probability threshold above which a warning is issued.
            area_threshold: The area threshold (percentage of area) above which a warning is issued.
            strategy_id: Identifier of the warning strategy (1 for water level ranges with measures).
        """
        # Get the range ids and initialize the warning_log
        range_ids = list(self.var.wlranges_and_measures.keys())
        warning_log = []

        # Create probability maps
        self.create_flood_probability_maps(strategy=strategy_id, date_time=date_time)

        # Load households and postal codes
        households = self.var.household_points.copy()
        postal_codes = self._load_postal_codes_clipped()
        # Postal codes are now efficiently clipped to model region

        for range_id in range_ids:
            # prob_map = Path(
            #     self.model.output_folder
            #     / "prob_maps"
            #     / f"forecast_{date_time.strftime('%Y%m%dT%H%M%S')}"
            #     / f"prob_map_range{range_id}_strategy{strategy_id}.tif"
            # )

            # with rasterio.open(prob_map) as src:
            #     prob_map = src.read(1)
            #     affine = src.transform

            # # Get the pixel values for each postal code
            # stats = zonal_stats(
            #     postal_codes,
            #     prob_map,
            #     affine=affine,
            #     raster_out=True,
            #     all_touched=True,
            #     nodata=np.nan,
            # )

            # Build path to probability map
            prob_map = Path(
                self.model.output_folder
                / "prob_maps"
                / f"forecast_{date_time.strftime('%Y%m%dT%H%M%S')}"
                / f"prob_map_range{range_id}_strategy{strategy_id}.zarr"
            )

            # Open the probability map
            prob_map = open_zarr(prob_map)
            affine = prob_map.rio.transform()
            prob_map = np.asarray(prob_map.values)

            # Get the pixel values for each postal code
            stats = zonal_stats(
                postal_codes,
                prob_map,
                affine=affine,
                raster_out=True,
                all_touched=True,
                nodata=np.nan,
            )

            # Iterate through each postal code and check how many pixels exceed the threshold
            for i, postalcode in enumerate(stats):
                pixel_values = postalcode["mini_raster_array"]
                postal_code = postal_codes.iloc[i]["postcode"]

                # Only get the values that are within the postal code
                postalcode_pixels = pixel_values[~pixel_values.mask]

                # Calculate the number of pixels with flood prob above the threshold
                n_above = np.sum(postalcode_pixels >= prob_threshold)

                # Calculate the total number of pixels within the postal code
                n_total = len(postalcode_pixels)

                if n_total == 0:
                    print(f"No valid pixels found for postal code {postal_code}")
                    continue
                else:
                    percentage = n_above / n_total

                if percentage >= area_threshold:
                    print(
                        f"Warning issued to postal code {postal_code} on {date_time.strftime('%d-%m-%Y T%H:%M:%S')} for range {range_id}: {percentage:.0%} of pixels > {prob_threshold}"
                    )

                    # Filter the affected households based on the postal code
                    affected_households = households[
                        households["postcode"] == postal_code
                    ]

                    # Get the measures and evacuation flag from the json dictionary to use in the warning communication function
                    measures = self.var.wlranges_and_measures[range_id]["measure"]
                    evacuate = self.var.wlranges_and_measures[range_id]["evacuate"]

                    # Communicate the warning to the target households
                    # This function should return the number of households that were warned
                    n_warned_households = self.warning_communication(
                        target_households=affected_households,
                        measures=measures,
                        evacuate=evacuate,
                        trigger="water_levels",
                    )

                    warning_log.append(
                        {
                            "date_time": date_time.strftime("%Y%m%d T%H%M%S"),
                            "postcode": postal_code,
                            "range": range_id,
                            "n_affected_households": len(affected_households),
                            "n_warned_households": n_warned_households,
                            "percentage_above_threshold": f"{percentage:.2f}",
                        }
                    )

        # Save the warning log to a csv file
        warnings_folder = self.model.output_folder / "warning_logs"
        warnings_folder.mkdir(exist_ok=True, parents=True)
        path = (
            warnings_folder
            / f"warning_log_water_levels_{date_time.strftime('%Y%m%dT%H%M%S')}.csv"
        )
        pd.DataFrame(warning_log).to_csv(path, index=False)

    def infrastructure_warning_strategy(self, prob_threshold=0.6) -> None:
        """Infrastructure warning strategy based on critical infrastructure flooding.

        Args:
            prob_threshold: Probability threshold for issuing warnings (0-1).

        Note:
            Currently disabled for development - will be enabled later.
            This function will issue warnings when critical infrastructure
            (energy substations) have flood probability above threshold.
        """
        # TODO: Enable infrastructure warnings when ready
        print("Infrastructure warning strategy is currently disabled for development.")
        print("This functionality will be enabled in a future update.")
        return

    def critical_infrastructure_warning_strategy(
        self, date_time, prob_threshold=0.6
    ) -> None:
        """Alternative name for infrastructure_warning_strategy to match model.py calls.

        Args:
            date_time: The forecast date time (currently unused but required by model.py).
            prob_threshold: Probability threshold for issuing warnings (0-1).
        """
        return self.infrastructure_warning_strategy(prob_threshold=prob_threshold)

        # ===== INFRASTRUCTURE WARNING CODE (DISABLED FOR NOW) =====
        # Load postal codes and substations
        PC4 = self._load_postal_codes_clipped()
        substations = gpd.read_parquet(
            self.model.files["geom"]["assets/energy_substations"]
        )

        # Get the forecast start date from the config
        day = self.model.config["general"]["forecasts"]["days"][0]

        # Create flood probability maps for critical substation hit (config needs to be set as strategy 2)
        self.create_flood_probability_maps()

        # Assign substations to postal codes based on distance -- THIS NEEDS TO BE IMPROVED
        PC4_with_substations = gpd.sjoin_nearest(
            PC4, substations[["fid", "geometry"]], how="left", distance_col="distance"
        )
        # Rename the fid_right column for clarity
        PC4_with_substations.rename(
            columns={"fid_right": "substation_id"},
            inplace=True,
        )

        # Get the household points
        households = self.var.household_points.copy()

        # Get the probability map for the specific day and strategy
        prob_tif_path = (
            self.model.output_folder
            / "prob_maps"
            / f"prob_map_range1_forecast{day.strftime('%Y-%m-%d')}_strategy2.tif"
        )
        # prob_tif_path = "C:\\Users\\adq582\\Documents\\Programming\\GEB\\models\\geul\\base\\output\\prob_maps\\prob_map_range1_forecast2021-07-13_strategy2.tif"
        prob_map = rasterio.open(prob_tif_path)
        prob_array = prob_map.read(1)
        affine = prob_map.transform

        # Sample the probability map at the substations locations
        sampled_probs = point_query(
            substations, prob_array, affine=affine, interpolate="nearest"
        )
        substations["probability"] = sampled_probs

        # Filter substations that have a flood hit probability > threshold
        critical_hits = substations[substations["probability"] >= prob_threshold]

        # If there are critical hits, issue warnings
        if not critical_hits.empty:
            print(f"Critical hits found: {len(critical_hits)}")

            # Get the postcodes that will be affected by the critical hits
            affected_postcodes = PC4_with_substations[
                PC4_with_substations["substation_id"].isin(critical_hits["fid"])
            ]["postcode"].unique()

            warning_log = []

            # Issue warnings to the households in the affected postcodes
            for postal_code in affected_postcodes:
                affected_households = households[households["postcode"] == postal_code]
                n_warned_households = self.warning_communication(
                    target_households=affected_households
                )
                # Need to think about how to deal with the communication efficiency for the two different strategies

                print(
                    f"Warning issued to postal code {postal_code} on {day} for critical substation hit."
                )
                warning_log.append(
                    {
                        "day": day,
                        "postcode": postal_code,
                        "n_warned_households": n_warned_households,
                        "critical_substation_hits": len(critical_hits),
                    }
                )

            # Save log
            path = os.path.join(self.model.output_folder, "warning_log_energy.csv")
            pd.DataFrame(warning_log).to_csv(path, index=False)

    def warning_communication(
        self,
        target_households: gpd.GeoDataFrame,
        measures: set[str],
        evacuate: bool,
        trigger: str,
        communication_efficiency: float = 0.78,
        lt_threshold_evacuation: int = 48,
    ) -> int:
        """Send warnings to a subset of target households according to communication_efficiency.

        Makes sure to send a single message and allows only upward escalation:
        none (0) -> measures (1) -> evacuate (2)
        Evacuation warning only sent if evacuate is True and lead_time <= 48h.

        Args:
            target_households: GeoDataFrame of household points targeted by the warning, must
                contain a 'pointid' column with the household index.
            measures: Set of recommended protective measures to communicate (strings).
            evacuate: Whether evacuation should be advised for this warning.
            trigger: Identifier of the trigger that initiated the warning.
            communication_efficiency: Fraction (0-1) of targeted households that can be reached.
            lt_threshold_evacuation: Lead time threshold (hours) below which evacuation warnings
                are allowed.

        Returns:
            int: Number of households that were successfully warned.
        """
        print("Running the warning communication for households...")

        # DEBUG: Print detailed information about target_households structure
        print("\n" + "=" * 80)
        print("DEBUG: target_households GeoDataFrame structure")
        print("=" * 80)
        print(f"Shape: {target_households.shape}")
        print(f"Columns: {target_households.columns.tolist()}")
        print(f"Data types:\n{target_households.dtypes}")
        print(f"CRS: {target_households.crs}")
        print(f"Index type: {type(target_households.index)}")
        print(f"Index name: {target_households.index.name}")
        print(
            f"First 5 indices: {target_households.index[:5].tolist() if len(target_households) > 0 else 'Empty'}"
        )

        if len(target_households) > 0:
            print("\nFirst 3 rows of data:")
            print(target_households.head(3).to_string())

            print("\nGeometry sample (first 3):")
            for i, geom in enumerate(target_households.geometry.head(3)):
                print(f"  {i}: {geom}")

            # Check for specific columns that might be important
            important_cols = ["postcode", "pointid", "building_id"]
            for col in important_cols:
                if col in target_households.columns:
                    unique_vals = target_households[col].nunique()
                    sample_vals = target_households[col].head(5).tolist()
                    print(
                        f"\nColumn '{col}': {unique_vals} unique values, samples: {sample_vals}"
                    )

        print("=" * 80)
        print("")

        # Get the number of target households
        n_target_households = len(target_households)
        if n_target_households == 0:
            print("No target households to warn.")
            return 0

        # Calculate individual communication efficiency probabilities based on socio-economic factors
        warning_weights = self.calculate_communication_efficiency_probability(
            target_households
        )

        print(f"Communication efficiency statistics:")
        print(f"  Mean: {warning_weights.mean():.3f}")
        print(f"  Std:  {warning_weights.std():.3f}")
        print(f"  Min:  {warning_weights.min():.3f}")
        print(f"  Max:  {warning_weights.max():.3f}")

        # Select exactly communication_efficiency fraction of households using socio-economic weights
        n_total = len(target_households)
        n_to_select = int(communication_efficiency * n_total)

        if n_to_select >= n_total:
            # If we need to select all or more than available, select all
            selected_households = target_households
        elif n_to_select == 0:
            # If we need to select none, return empty
            selected_households = target_households.iloc[:0]
        else:
            # Use weighted random sampling to select exactly n_to_select households
            np.random.seed(42)  # Fixed seed for reproducibility
            chosen_indices = np.random.choice(
                target_households.index,
                size=n_to_select,
                replace=False,
                p=warning_weights,
            )
            selected_households = target_households.loc[chosen_indices]

        n_feasible_warnings = len(selected_households)
        if n_feasible_warnings == 0:
            print(
                "Based on the communication efficiency, no warning will reach households."
            )
            return 0

        print(
            f"Socio-economic efficiency resulted in {n_feasible_warnings} out of {n_target_households} households receiving warning"
        )

        # Get lead time in hours
        lead_time = self.compute_lead_time()
        print(f"Lead time for warning: {lead_time:.0f} hours")

        # Helper function to pick the measures to recommend based on lead time
        def pick_recommendations(
            available_measures: set[str], evacuate: bool, lead_time: float
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
            implementation_times = self.var.implementation_times

            # Filter out empty strings and invalid measures
            valid_measures = {
                measure
                for measure in available_measures
                if measure and measure.strip() and measure in implementation_times
            }

            # Sort the valid measures by their implementation times and alphabetically
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
                    return chosen_measures
            else:
                # Evacuation case
                # If there is no time for evacuation, nothing to recommend
                if evac_time > lead_time:
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

        # Pick the measures to recommend based on lead time
        # Convert measures list to set and ensure they are valid
        measures_set = set(measures) if isinstance(measures, list) else {measures}
        recommended_measures = pick_recommendations(measures_set, evacuate, lead_time)

        # Policy check for evacuation (only recommend it if lead time <= threshold)
        evac_feasible = "evacuate" in recommended_measures
        evac_policy = lead_time <= lt_threshold_evacuation
        if evac_feasible and not evac_policy:
            recommended_measures.remove("evacuate")
            evac_feasible = False

        # Determine the desired level of warning (0 = no warning, 1 = measures, 2 = evacuate)
        desired_level = 2 if evac_feasible else (1 if recommended_measures else 0)

        if desired_level == 0:
            return 0

        # Get possible measures and triggers (this is a list of all possible measures, not the recommended measures given by the warning strategies)
        # Used only to make sure the indices are correct when updating the arrays (self.var.recommended_measures, self.var.warning_trigger)
        possible_measures_to_recommend = self.var.possible_measures_to_recommend
        possible_warning_triggers = self.var.possible_warning_triggers

        n_warned_households = 0
        for _, row in selected_households.iterrows():
            household_id = row["pointid"]

            # Skip already evacuated
            if self.var.evacuated[household_id] == 1:
                continue

            # Get current warning level
            current_level = int(self.var.warning_level[household_id])

            # Only send a warning if the desired level is higher than current level
            if desired_level > current_level:
                # For every measure in the recommended measures, get the corresponding index and set the right column to True to store it
                for measure in recommended_measures:
                    measure_idx = possible_measures_to_recommend.index(measure)
                    self.var.recommended_measures[household_id, measure_idx] = True
            else:
                continue

            # Mark warning level and reached status
            self.var.warning_level[household_id] = desired_level
            self.var.warning_reached[household_id] = 1

            # Mark the trigger
            trigger_idx = possible_warning_triggers.index(trigger)
            self.var.warning_trigger[household_id, trigger_idx] = True
            n_warned_households += 1

        print(f"Warning targeted to reach {n_target_households} households")
        print(f"Warning reached {n_warned_households} households")

        return n_warned_households

    def calculate_communication_efficiency_probability(
        self,
        target_households: gpd.GeoDataFrame,
    ) -> pd.Series:
        """
        Calculate communication efficiency probability based on education level and income.

        Higher education and income lead to better access to communication channels
        and faster response to warnings.

        Args:
            target_households: GeoDataFrame with household data including education and income.
            base_efficiency: Base probability for communication efficiency (0-1).
            use_socioeconomic_factors: Whether to use socio-economic factors or fallback to random.

        Returns:
            Series with communication efficiency probabilities for each household.

        Raises:
            KeyError: If required columns are missing from target_households.
        """
        # 1. Check required columns
        required_columns = ["Education", "Income"]
        missing_columns = [
            col for col in required_columns if col not in target_households.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Missing required columns in target_households: {missing_columns}"
            )

        # 2. Weights based on the regression of the WRP survey data
        # (https://documents1.worldbank.org/curated/en/099259309032538041/pdf/IDU-c6f56dc5-a0cb-4375-ac15-a91f1c202b09.pdf )
        w_edu = {
            1: 1.0,  # less than primary
            2: 1.1765,  # complete primary
            3: 1.3530,  # incomplete secondary
            4: 1.5295,  # complete secondary / tertiary
            5: 1.7060,  # higher
        }

        w_income = {
            1: 1.0,  # lowest income
            2: 1.041,
            3: 1.082,
            4: 1.123,
            5: 1.164,  # highest income
        }

        # 3. Compute total weight per agent
        target_households["weight"] = target_households["Education"].map(
            w_edu
        ) + target_households["Income"].map(w_income)

        weights = target_households["weight"].to_numpy()
        weights = weights / weights.sum()  # normalize for sampling

        return weights

    def compute_lead_time(self) -> float:
        """Compute lead time in hours based on forecast start time and current model time.

        Returns:
            float: Lead time in hours.
        """
        start_time = self.model.config["hazards"]["floods"]["events"][0]["start_time"]
        current_time = self.model.current_time
        lead_time = (start_time - current_time).total_seconds() / 3600
        lead_time = max(lead_time, 0)  # Ensure non-negative lead time
        return lead_time

    def household_decision_making(self, date_time, responsive_ratio=0.7) -> None:
        """Simulate household emergency response decisions based on warnings and lead time.

        Args:
            date_time: The forecast date time for which to run the decision-making.
            responsive_ratio: Threshold for household responsiveness (0-1). Households with
                response_probability below this value are considered responsive and will
                act on warnings.
        """
        print("Running emergency response decision-making for households...")

        # Get lead time in hours
        lead_time = self.compute_lead_time()

        # Filter households that did not evacuate, were warned and are responsive
        not_evacuated_ids = self.var.evacuated == 0
        warned_ids = self.var.warning_reached == 1
        responsive_ids = self.var.response_probability < responsive_ratio

        # Combine the filters and apply it to household_points
        eligible_ids = np.asarray(warned_ids & not_evacuated_ids & responsive_ids)
        eligible_households = self.var.household_points.loc[eligible_ids]

        # Get the list of possible actions for eligible households
        # (this is a list of all possible actions, not the recommended actions given by the warning strategies)
        possible_actions = self.var.possible_actions
        evac_idx = possible_actions.index("evacuate")

        # Initialize an empty list to log actions taken
        actions_log = []

        # For every eligible household, do the recommended measures in the communicated warning
        for household_id, _ in eligible_households.iterrows():
            # For measure in recommended measures, change it to true in the actions_taken array
            self.var.actions_taken[household_id] = self.var.recommended_measures[
                household_id
            ].copy()

            # If evacuation is among the actions taken, mark household as evacuated
            if self.var.actions_taken[household_id, evac_idx]:
                self.var.evacuated[household_id] = 1

            # Log the actions taken
            actions = []
            for i, action in enumerate(possible_actions):
                if self.var.actions_taken[household_id, i]:
                    actions.append(action)

            actions_log.append(
                {
                    "lead_time": lead_time,
                    "date_time": date_time.strftime("%Y%m%d T%H%M%S"),
                    "postal_code": self.var.household_points.loc[
                        household_id, "postcode"
                    ],
                    "household_id": household_id,
                    "actions": actions,
                }
            )

        # Save actions log
        actions_log_folder = self.model.output_folder / "actions_logs"
        actions_log_folder.mkdir(exist_ok=True, parents=True)
        path = (
            actions_log_folder
            / f"actions_log_{date_time.strftime('%Y%m%dT%H%M%S')}.csv"
        )
        pd.DataFrame(actions_log).to_csv(path, index=False)

    def decide_household_strategy(self) -> None:
        """This function calculates the utility of adapting to flood risk for each household and decides whether to adapt or not."""
        # update risk perceptions
        self.update_risk_perceptions()

        # calculate damages for adapting and not adapting households based on building footprints
        damages_do_not_adapt, damages_adapt = self.calculate_building_flood_damages()

        # calculate expected utilities
        EU_adapt = self.decision_module.calcEU_adapt_flood(
            geom_id="NoID",
            n_agents=self.n,
            wealth=self.var.wealth.data,
            income=self.var.income.data,
            expendature_cap=1,
            amenity_value=self.var.amenity_value.data,
            amenity_weight=1,
            risk_perception=self.var.risk_perception.data,
            expected_damages_adapt=damages_adapt,
            adaptation_costs=self.var.adaptation_costs.data,
            time_adapted=self.var.time_adapted.data,
            loan_duration=20,
            p_floods=1 / self.return_periods,
            T=35,
            r=0.03,
            sigma=1,
        )

        EU_do_not_adapt = self.decision_module.calcEU_do_nothing_flood(
            geom_id="NoID",
            n_agents=self.n,
            wealth=self.var.wealth.data,
            income=self.var.income.data,
            expendature_cap=1,
            amenity_value=self.var.amenity_value.data,
            amenity_weight=1,
            risk_perception=self.var.risk_perception.data,
            expected_damages=damages_do_not_adapt,
            adapted=self.var.adapted.data,
            p_floods=1 / self.return_periods,
            T=35,
            r=0.03,
            sigma=1,
        )

        # execute strategy
        household_adapting = np.where(EU_adapt > EU_do_not_adapt)[0]
        self.var.adapted[household_adapting] = 1
        self.var.time_adapted[household_adapting] += 1

        # update column in buildings
        self.update_building_adaptation_status(household_adapting)

        # print percentage of households that adapted
        print(f"N households that adapted: {len(household_adapting)}")

    def load_objects(self) -> None:
        # Load buildings
        self.buildings = gpd.read_parquet(
            self.model.files["geom"]["assets/open_building_map"]
        )
        self.buildings["object_type"] = (
            "building_unprotected"  # before it was "building_structure"
        )
        self.buildings_centroid = gpd.GeoDataFrame(geometry=self.buildings.centroid)
        self.buildings_centroid["object_type"] = (
            "building_unprotected"  # before it was "building_content"
        )

        # Load roads
        self.roads = gpd.read_parquet(self.model.files["geom"]["assets/roads"]).rename(
            columns={"highway": "object_type"}
        )

        # Load rail
        self.rail = gpd.read_parquet(self.model.files["geom"]["assets/rails"])
        self.rail["object_type"] = "rail"

    def load_max_damage_values(self) -> None:
        # Load maximum damages
        self.var.max_dam_buildings_structure = float(
            load_dict(
                self.model.files["dict"][
                    "damage_parameters/flood/buildings/structure/maximum_damage"
                ]
            )["maximum_damage"]
        )
        self.buildings["maximum_damage_m2"] = self.var.max_dam_buildings_structure

        max_dam_buildings_content = load_dict(
            self.model.files["dict"][
                "damage_parameters/flood/buildings/content/maximum_damage"
            ]
        )
        self.var.max_dam_buildings_content = float(
            max_dam_buildings_content["maximum_damage"]
        )
        self.buildings_centroid["maximum_damage"] = self.var.max_dam_buildings_content

        self.var.max_dam_rail = float(
            load_dict(
                self.model.files["dict"][
                    "damage_parameters/flood/rail/main/maximum_damage"
                ]
            )["maximum_damage"]
        )
        self.rail["maximum_damage_m"] = self.var.max_dam_rail

        max_dam_road_m: dict[str, float] = {}
        road_types = [
            (
                "residential",
                "damage_parameters/flood/road/residential/maximum_damage",
            ),
            (
                "unclassified",
                "damage_parameters/flood/road/unclassified/maximum_damage",
            ),
            ("tertiary", "damage_parameters/flood/road/tertiary/maximum_damage"),
            ("primary", "damage_parameters/flood/road/primary/maximum_damage"),
            (
                "primary_link",
                "damage_parameters/flood/road/primary_link/maximum_damage",
            ),
            ("secondary", "damage_parameters/flood/road/secondary/maximum_damage"),
            (
                "secondary_link",
                "damage_parameters/flood/road/secondary_link/maximum_damage",
            ),
            ("motorway", "damage_parameters/flood/road/motorway/maximum_damage"),
            (
                "motorway_link",
                "damage_parameters/flood/road/motorway_link/maximum_damage",
            ),
            ("trunk", "damage_parameters/flood/road/trunk/maximum_damage"),
            ("trunk_link", "damage_parameters/flood/road/trunk_link/maximum_damage"),
        ]

        for road_type, path in road_types:
            max_dam_road_m[road_type] = load_dict(self.model.files["dict"][path])[
                "maximum_damage"
            ]

        self.roads["maximum_damage_m"] = self.roads["object_type"].map(max_dam_road_m)

        self.var.max_dam_forest_m2 = float(
            load_dict(
                self.model.files["dict"][
                    "damage_parameters/flood/land_use/forest/maximum_damage"
                ]
            )["maximum_damage"]
        )

        self.var.max_dam_agriculture_m2 = float(
            load_dict(
                self.model.files["dict"][
                    "damage_parameters/flood/land_use/agriculture/maximum_damage"
                ]
            )["maximum_damage"]
        )

    def load_damage_curves(self) -> None:
        # Load vulnerability curves [look into these curves, some only max out at 0.5 damage ratio]
        road_curves = []
        road_types = [
            ("residential", "damage_parameters/flood/road/residential/curve"),
            ("unclassified", "damage_parameters/flood/road/unclassified/curve"),
            ("tertiary", "damage_parameters/flood/road/tertiary/curve"),
            ("tertiary_link", "damage_parameters/flood/road/tertiary_link/curve"),
            ("primary", "damage_parameters/flood/road/primary/curve"),
            ("primary_link", "damage_parameters/flood/road/primary_link/curve"),
            ("secondary", "damage_parameters/flood/road/secondary/curve"),
            ("secondary_link", "damage_parameters/flood/road/secondary_link/curve"),
            ("motorway", "damage_parameters/flood/road/motorway/curve"),
            ("motorway_link", "damage_parameters/flood/road/motorway_link/curve"),
            ("trunk", "damage_parameters/flood/road/trunk/curve"),
            ("trunk_link", "damage_parameters/flood/road/trunk_link/curve"),
        ]

        severity_column = None
        for road_type, path in road_types:
            df = pd.read_parquet(self.model.files["table"][path])

            if severity_column is None:
                severity_column = df["severity"]

            df = df.rename(columns={"damage_ratio": road_type})

            road_curves.append(df[[road_type]])

        self.var.road_curves = pd.concat([severity_column] + road_curves, axis=1)
        self.var.road_curves.set_index("severity", inplace=True)

        self.var.forest_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/land_use/forest/curve"]
        )
        self.var.forest_curve.set_index("severity", inplace=True)
        self.var.forest_curve = self.var.forest_curve.rename(
            columns={"damage_ratio": "forest"}
        )
        self.var.agriculture_curve = pd.read_parquet(
            self.model.files["table"][
                "damage_parameters/flood/land_use/agriculture/curve"
            ]
        )
        self.var.agriculture_curve.set_index("severity", inplace=True)
        self.var.agriculture_curve = self.var.agriculture_curve.rename(
            columns={"damage_ratio": "agriculture"}
        )

        self.buildings_structure_curve = pd.read_parquet(
            self.model.files["table"][
                "damage_parameters/flood/buildings/structure/curve"
            ]
        )
        self.buildings_structure_curve.set_index("severity", inplace=True)
        self.buildings_structure_curve = self.buildings_structure_curve.rename(
            columns={"damage_ratio": "building_unprotected"}
        )

        # create another column (curve) in the buildings structure curve for protected buildings
        self.buildings_structure_curve["building_protected"] = (
            self.buildings_structure_curve["building_unprotected"] * 0.85
        )

        # create another column (curve) in the buildings structure curve for flood-proofed buildings
        self.buildings_structure_curve["building_flood_proofed"] = (
            self.buildings_structure_curve["building_unprotected"] * 0.85
        )
        self.buildings_structure_curve["building_flood_proofed"].loc[0:1] = 0.0

        self.buildings_content_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/buildings/content/curve"]
        )
        self.buildings_content_curve.set_index("severity", inplace=True)
        self.buildings_content_curve = self.buildings_content_curve.rename(
            columns={"damage_ratio": "building_unprotected"}
        )

        # create another column (curve) in the buildings content curve for protected buildings
        self.buildings_content_curve["building_protected"] = (
            self.buildings_content_curve["building_unprotected"] * 0.7
        )

        # create damage curves for adaptation
        buildings_content_curve_adapted = self.buildings_content_curve.copy()
        buildings_content_curve_adapted.loc[0:1] = (
            0  # assuming zero damages untill 1m water depth
        )
        buildings_content_curve_adapted.loc[1:] *= (
            0.8  # assuming 80% damages above 1m water depth
        )
        self.buildings_content_curve_adapted = buildings_content_curve_adapted

        self.var.rail_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/rail/main/curve"]
        )
        self.var.rail_curve.set_index("severity", inplace=True)
        self.var.rail_curve = self.var.rail_curve.rename(
            columns={"damage_ratio": "rail"}
        )

    def create_damage_interpolators(self) -> None:
        # create interpolation function for damage curves [interpolation objects cannot be stored in bucket]
        self.buildings_content_curve_interpolator = interpolate.interp1d(
            x=self.buildings_content_curve.index,
            y=self.buildings_content_curve["building_unprotected"],
            # fill_value="extrapolate",
        )
        self.buildings_content_curve_adapted_interpolator = interpolate.interp1d(
            x=self.buildings_content_curve_adapted.index,
            y=self.buildings_content_curve_adapted["building_unprotected"],
            # fill_value="extrapolate",
        )

    def spinup(self) -> None:
        self.construct_income_distribution()
        self.assign_household_attributes()
        if self.config["warning_response"]:
            self.change_household_locations()

    def calculate_building_flood_damages(self) -> tuple[np.ndarray, np.ndarray]:
        """This function calculates the flood damages for the households in the model.

        It iterates over the return periods and calculates the damages for each household
        based on the flood maps and the building footprints.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the damage arrays for unprotected and protected buildings.
        """
        damages_do_not_adapt = np.zeros((self.return_periods.size, self.n), np.float32)
        damages_adapt = np.zeros((self.return_periods.size, self.n), np.float32)
        buildings: gpd.GeoDataFrame = self.buildings.copy().to_crs(
            self.flood_maps[self.return_periods[0]].rio.crs
        )
        # subset building to those exposed to flooding
        buildings = buildings[buildings["flooded"]]

        for i, return_period in enumerate(self.return_periods):
            flood_map: xr.DataArray = self.flood_maps[return_period]

            # Calculate damages to building structure (unprotected buildings)
            damage_unprotected: pd.Series = VectorScanner(
                features=buildings.rename(
                    columns={"maximum_damage_m2": "maximum_damage"}
                ),
                hazard=flood_map,
                vulnerability_curves=self.buildings_structure_curve,
                disable_progress=True,
            )
            total_damage_structure = damage_unprotected.sum()
            print(
                f"damages to building unprotected structure rp{return_period} are: {round(total_damage_structure / 1e6, 2)} M€"
            )

            # Save the damages to the dataframe
            buildings_with_damages = buildings[["id"]]
            buildings_with_damages["damage"] = damage_unprotected

            # Calculate damages to building structure (floodproofed buildings)
            buildings_floodproofed = buildings.copy()
            buildings_floodproofed["object_type"] = "building_flood_proofed"
            damage_flood_proofed: pd.Series = VectorScanner(
                features=buildings_floodproofed.rename(
                    columns={"maximum_damage_m2": "maximum_damage"}
                ),
                hazard=flood_map,
                vulnerability_curves=self.buildings_structure_curve,
                disable_progress=True,
            )
            total_damage_structure = damage_flood_proofed.sum()
            print(
                f"damages to building flood-proofed structure rp{return_period} are: {round(total_damage_structure / 1e6, 2)} M€"
            )

            # Save the damages to the dataframe
            buildings_with_damages_floodproofed = buildings[["id"]]
            buildings_with_damages_floodproofed["damage"] = damage_flood_proofed

            # add damages to agents (unprotected buildings)
            for _, row in buildings_with_damages.iterrows():
                damage = row["damage"]
                building_id = int(row["id"])
                idx_agents_in_building = np.where(self.var.building_id == building_id)[
                    0
                ]
                damages_do_not_adapt[i, idx_agents_in_building] = damage

            # add damages to agents (flood-proofed buildings)
            for _, row in buildings_with_damages_floodproofed.iterrows():
                damage = row["damage"]
                building_id = int(row["id"])
                idx_agents_in_building = np.where(self.var.building_id == building_id)[
                    0
                ]
                damages_adapt[i, idx_agents_in_building] = damage

        return damages_do_not_adapt, damages_adapt

    def update_households_gdf(self, date_time) -> None:
        """This function merges the global variables related to warnings to the households geodataframe for visualization purposes.

        Args:
            date_time: The forecast date time for which to update the households geodataframe.
        """
        household_points: gpd.GeoDataFrame = self.var.household_points.copy()
        global_vars = [
            "warning_reached",
            "warning_level",
            "response_probability",
            "evacuated",
            "recommended_measures",
            "warning_trigger",
            "actions_taken",
        ]

        # make sure household points and global variables have the same length
        for global_var in global_vars:
            global_array = getattr(self.var, global_var)
            assert len(household_points) == global_array.shape[0], (
                f"The size of household points and {global_var} do not match"
            )

        # add columns in the household points geodataframe
        for name in [
            "warning_reached",
            "warning_level",
            "response_probability",
            "evacuated",
        ]:
            household_points[name] = getattr(self.var, name)

        warning_triggers = self.var.possible_warning_triggers
        for i, _ in enumerate(warning_triggers):
            if warning_triggers[i] == "water_levels":
                household_points["trig_w_levels"] = self.var.warning_trigger[:, i]
            if warning_triggers[i] == "critical_infrastructure":
                household_points["trig_crit_infra"] = self.var.warning_trigger[:, i]

        possible_measures_to_recommend = self.var.possible_measures_to_recommend
        for i, measure in enumerate(possible_measures_to_recommend):
            if measure == "elevate possessions":
                household_points["recom_elev_possessions"] = (
                    self.var.recommended_measures[:, i]
                )
            if measure == "evacuate":
                household_points["recom_evacuate"] = self.var.recommended_measures[:, i]

        possible_actions = self.var.possible_actions
        for i, action in enumerate(possible_actions):
            if action == "elevate possessions":
                household_points["elevated_possessions"] = self.var.actions_taken[:, i]

        household_points.to_parquet(
            self.model.output_folder
            / "action_maps"
            / f"households_with_warning_parameters_{date_time.strftime('%Y%m%dT%H%M%S')}.geoparquet"
        )

    def load_wlranges_and_measures(self):
        with open(self.model.files["dict"]["measures/implementation_times"], "r") as f:
            self.var.implementation_times = json.load(f)

        with open(self.model.files["dict"]["measures/wl_ranges"], "r") as f:
            wlranges_and_measures = json.load(f)
            # convert the keys (range ids) to integers and store them in a new dictionary
            self.var.wlranges_and_measures = {
                int(key): value for key, value in wlranges_and_measures.items()
            }

    def flood(self, flood_depth: xr.DataArray) -> float:
        """This function computes the damages for the assets and land use types in the model.

        Args:
            flood_depth: The flood map containing water levels for the flood event [m].

        Returns:
            The total flood damages for the event for all assets and land use types.

        """
        flood_depth: xr.DataArray = flood_depth.compute()
        # flood_map = flood_map.chunk({"x": 100, "y": 1000})

        buildings: gpd.GeoDataFrame = self.buildings.copy().to_crs(flood_depth.rio.crs)
        household_points: gpd.GeoDataFrame = self.var.household_points.copy().to_crs(
            flood_depth.rio.crs
        )

        assert len(household_points) == self.var.risk_perception.size

        household_points["protect_building"] = False
        household_points.loc[
            self.var.risk_perception.data >= 0.1, "protect_building"
        ] = True

        buildings: gpd.GeoDataFrame = gpd.sjoin_nearest(
            buildings, household_points, how="left", exclusive=True
        )

        buildings["object_type"] = "building_unprotected"
        buildings.loc[buildings["protect_building"], "object_type"] = (
            "building_protected"
        )

        # Create the folder to save damage maps if it doesn't exist
        damage_folder: Path = self.model.output_folder / "damage_maps"
        damage_folder.mkdir(parents=True, exist_ok=True)

        # Compute damages for buildings content and save it to a gpkg file
        category_name: str = "buildings_content"
        filename: str = f"damage_map_{category_name}.gpkg"

        buildings_centroid = household_points.to_crs(flood_depth.rio.crs)
        buildings_centroid["object_type"] = buildings_centroid[
            "protect_building"
        ].apply(lambda x: "building_protected" if x else "building_unprotected")
        buildings_centroid["maximum_damage"] = self.var.max_dam_buildings_content

        damages_buildings_content = VectorScanner(
            features=buildings_centroid,
            hazard=flood_depth,
            vulnerability_curves=self.buildings_content_curve,
        )

        total_damages_content = damages_buildings_content.sum()
        print(f"damages to building content are: {total_damages_content}")

        # Compute damages for buildings structure and save it to a gpkg file
        category_name: str = "buildings_structure"
        filename: str = f"damage_map_{category_name}.gpkg"

        damages_buildings_structure: pd.Series = VectorScanner(
            features=buildings.rename(columns={"maximum_damage_m2": "maximum_damage"}),
            hazard=flood_depth,
            vulnerability_curves=self.buildings_structure_curve,
        )

        total_damage_structure = damages_buildings_structure.sum()

        print(f"damages to building structure are: {total_damage_structure}")
        print(
            f"Total damages to buildings are: {total_damages_content + total_damage_structure}"
        )

        agriculture = from_landuse_raster_to_polygon(
            self.HRU.decompress(self.HRU.var.land_owners != -1),
            self.HRU.transform,
            self.model.crs,
        )
        agriculture["object_type"] = "agriculture"
        agriculture["maximum_damage"] = self.var.max_dam_agriculture_m2

        agriculture = agriculture.to_crs(flood_depth.rio.crs)

        damages_agriculture = VectorScanner(
            features=agriculture,
            hazard=flood_depth,
            vulnerability_curves=self.var.agriculture_curve,
        )
        total_damages_agriculture = damages_agriculture.sum()
        print(f"damages to agriculture are: {total_damages_agriculture}")

        # Load landuse and make turn into polygons
        forest = from_landuse_raster_to_polygon(
            self.HRU.decompress(self.HRU.var.land_use_type == FOREST),
            self.HRU.transform,
            self.model.crs,
        )
        forest["object_type"] = "forest"
        forest["maximum_damage"] = self.var.max_dam_forest_m2

        forest = forest.to_crs(flood_depth.rio.crs)

        damages_forest = VectorScanner(
            features=forest,
            hazard=flood_depth,
            vulnerability_curves=self.var.forest_curve,
        )
        total_damages_forest = damages_forest.sum()
        print(f"damages to forest are: {total_damages_forest}")

        roads = self.roads.to_crs(flood_depth.rio.crs)
        damages_roads = VectorScanner(
            features=roads.rename(columns={"maximum_damage_m": "maximum_damage"}),
            hazard=flood_depth,
            vulnerability_curves=self.var.road_curves,
        )
        total_damages_roads = damages_roads.sum()
        print(f"damages to roads are: {total_damages_roads} ")

        rail = self.rail.to_crs(flood_depth.rio.crs)
        damages_rail = VectorScanner(
            features=rail.rename(columns={"maximum_damage_m": "maximum_damage"}),
            hazard=flood_depth,
            vulnerability_curves=self.var.rail_curve,
        )
        total_damages_rail = damages_rail.sum()
        print(f"damages to rail are: {total_damages_rail}")

        total_flood_damages = (
            total_damage_structure
            + total_damages_content
            + total_damages_roads
            + total_damages_rail
            + total_damages_forest
            + total_damages_agriculture
        )
        print(f"the total flood damages are: {total_flood_damages}")

        return total_flood_damages

    def water_demand(
        self,
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        """Calculate the water demand per household in m3 per day.

        In the default option (see configuration), the water demand is calculated
        based on the municipal water demand per capita in the baseline year,
        the size of the household, and a water demand multiplier that varies
        by region and year.

        In the 'custom_value' option, all households are assigned the same
        water demand value specified in the configuration.

        Returns:
            Tuple containing:
                - water_demand_per_household_m3: Water demand per household in m3 per day.
                - water_efficiency_per_household: Water efficiency per household (0-1).
                    A factor of 1 means no water is wasted, while 0 means all water is wasted.
                - locations: Locations of the households (x, y coordinates).

        Raises:
            ValueError: If the water demand method in the configuration is invalid.
        """
        assert (self.var.water_efficiency_per_household == 1).all(), (
            "if not 1, code must be updated to account for water efficiency in water demand"
        )
        if self.config["water_demand"]["method"] == "default":
            # the water demand multiplier is a function of the year and region
            water_demand_multiplier_per_region = self.var.municipal_water_withdrawal_m3_per_capita_per_day_multiplier.loc[
                self.model.current_time.year
            ]
            assert (
                water_demand_multiplier_per_region.index
                == np.arange(len(water_demand_multiplier_per_region))
            ).all()
            water_demand_multiplier_per_household = (
                water_demand_multiplier_per_region.values[self.var.region_id]
            )

            # water demand is the per capita water demand in the household,
            # multiplied by the size of the household and the water demand multiplier
            # per region and year, relative to the baseline.
            self.var.water_demand_per_household_m3 = (
                self.var.municipal_water_demand_per_capita_m3_baseline
                * self.var.sizes
                * water_demand_multiplier_per_household
            )
        elif self.config["water_demand"]["method"] == "custom_value":
            # Function to set a custom_value for household water demand. All households have the same demand.
            custom_value = self.config["water_demand"]["custom_value"]["value"]
            self.var.water_demand_per_household_m3 = np.full(
                self.var.region_id.shape, custom_value, dtype=float
            )
        else:
            raise ValueError(
                "Invalid water demand method. Configuration must be 'default' or 'custom_value'."
            )

        return (
            self.var.water_demand_per_household_m3,
            self.var.water_efficiency_per_household,
            self.var.locations.data,
        )

    def step(self) -> None:
        if (
            self.config["adapt"]
            and self.model.current_time.month == 1
            and self.model.current_time.day == 1
        ):
            if "flooded" not in self.buildings.columns:
                self.update_building_attributes()

            print("Thinking about adapting...")
            self.decide_household_strategy()
        self.report(locals())

    @property
    def n(self):
        return self.var.locations.shape[0]

    @property
    def population(self):
        return self.var.sizes.data.sum()
