import json
import os
from pathlib import Path
from typing import Tuple

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

from ..hydrology.landcover import (
    FOREST,
)
from ..store import DynamicArray
from ..workflows.damage_scanner import VectorScanner
from ..workflows.io import load_array, load_table, open_zarr
from .decision_module import DecisionModule
from .general import AgentBaseClass


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

    def __init__(self, model, agents, reduncancy: float) -> None:
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
        self.decision_module = DecisionModule(self, model=None)

        if self.config["adapt"]:
            self.load_flood_maps()

        self.load_objects()
        self.load_max_damage_values()
        self.load_damage_curves()
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
        # Start by computing occupancy from the var.osm_id and var.osm_way_id arrays
        osm_id_series = pd.Series(self.var.osm_id.data)
        osm_way_id_series = pd.Series(self.var.osm_way_id.data)

        # Drop NaNs and convert to string format (matching building ID format)
        osm_id_counts = osm_id_series.dropna().astype(int).astype(str).value_counts()
        osm_way_id_counts = (
            osm_way_id_series.dropna().astype(int).astype(str).value_counts()
        )

        # Initialize occupancy column
        self.buildings["occupancy"] = 0

        # Map the counts back to the buildings dataframe
        self.buildings["occupancy"] = (
            self.buildings["osm_id"]
            .map(osm_id_counts)
            .fillna(self.buildings["occupancy"])
        )
        self.buildings["occupancy"] = (
            self.buildings["osm_way_id"]
            .map(osm_way_id_counts)
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
            flood_map.rio.transform(),
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
        osm_ids = pd.DataFrame(
            np.unique(self.var.osm_id.data[household_adapting])
        ).dropna()
        osm_ids = osm_ids.astype(int).astype(str)
        osm_ids["flood_proofed"] = True
        osm_ids = osm_ids.set_index(0)

        # Extract and clean OSM way IDs from adapting households
        osm_way_ids = pd.DataFrame(
            np.unique(self.var.osm_way_id.data[household_adapting])
        ).dropna()
        osm_way_ids = osm_way_ids.astype(int).astype(str)
        osm_way_ids["flood_proofed"] = True
        osm_way_ids = osm_way_ids.set_index(0)

        # Add/Update the flood_proofed status in buildings based on OSM way IDs
        self.buildings["flood_proofed"] = (
            self.buildings["osm_way_id"].astype(str).map(osm_way_ids["flood_proofed"])
        )
        self.buildings["flood_proofed"] = self.buildings["flood_proofed"].fillna(
            self.buildings["osm_id"].astype(str).map(osm_ids["flood_proofed"])
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
        osm_id = load_array(self.model.files["array"]["agents/households/osm_id"])
        self.var.osm_id = DynamicArray(osm_id, max_n=self.max_n)

        osm_way_id = load_array(
            self.model.files["array"]["agents/households/osm_way_id"]
        )
        self.var.osm_way_id = DynamicArray(osm_way_id, max_n=self.max_n)

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

        # initiate array for warning source (set of sources that issued the warning)
        # testing without being numpy array -- need to fix this later
        self.var.warning_source = [set() for _ in range(self.n)]

        # initiate array for response probability (between 0 and 1)
        # using a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        self.var.response_probability = DynamicArray(
            rng.random(self.n), max_n=self.max_n
        )

        # initiate array for evacuation status [0=not evacuated, 1=evacuated]
        self.var.evacuated = DynamicArray(np.zeros(self.n, np.int32), max_n=self.max_n)

        # initiate array for storing the recommended measures received with the warnings
        # self.var.recommended_measures = DynamicArray(
        #     np.empty(self.n, dtype=object), max_n=self.max_n
        # ) --> assert error, dtype cannot be object --> not dynamic array for now, need to fix it later
        self.var.recommended_measures = np.array([set() for _ in range(self.n)])

        # initiate array for storing the actions taken by the household
        # --> assert error, dtype cannot be object --> not dynamic for now, need to fix it later
        self.var.actions_taken = np.array([set() for _ in range(self.n)])

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

    def change_household_locations(self):
        """Change the location of the household points to the centroid of the buildings.

        Also, it associates the household points with their postal codes.
        This is done to get the correct geometry for the warning function
        """
        crs: str = self.model.sfincs.crs
        locations = self.var.household_points.copy()
        locations.to_crs(
            crs, inplace=True
        )  # Change to a projected CRS to get a distance in meters

        buildings_centroid = self.var.buildings_centroid[["geometry"]].copy()
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
        PC4: gpd.GeoDataFrame = gpd.read_parquet("data/postal_codes_4.parquet")
        PC4["postcode"] = PC4["postcode"].astype("int32")

        new_locations = gpd.sjoin(
            new_locations,
            PC4[["postcode", "geometry"]],
            how="left",
            predicate="intersects",
        )

        self.var.household_points = new_locations

    def get_flood_risk_information_honeybees(self):
        # preallocate array for damages
        damages_do_not_adapt = np.zeros((self.return_periods.size, self.n), np.float32)
        damages_adapt = np.zeros((self.return_periods.size, self.n), np.float32)

        # load damage interpolators (cannot be store in bucket, therefor outside spinup)
        if not hasattr(self, "buildings_content_curve_interpolator"):
            self.create_damage_interpolators()

        if not hasattr(self.var, "locations_reprojected_to_flood_map"):
            self.reproject_locations_to_floodmap_crs()

        # loop over return periods
        for i, return_period in enumerate(self.return_periods):
            # get flood map
            flood_map = self.flood_maps[return_period]

            # sample waterlevels for individual households
            water_levels = sample_from_map(
                flood_map.values,
                self.var.locations_reprojected_to_flood_map.data,
                self.flood_maps["gdal_geotransform"],
            )

            # cap water levels at damage curve max inundation
            water_levels = np.minimum(
                water_levels, self.var.buildings_content_curve_interpolator.x.max()
            )

            # interpolate damages
            damages_do_not_adapt[i, :] = (
                self.var.buildings_content_curve_interpolator(water_levels)
                * self.var.property_value.data
            )

            damages_adapt[i, :] = (
                self.var.buildings_content_curve_adapted_interpolator(water_levels)
                * self.var.property_value.data
            )

        return damages_do_not_adapt, damages_adapt

    def get_flood_risk_information_damage_scanner(self):
        """Initiate flood risk information for each household.

        This information is used in the decision module.
        For now also only dummy data is created.
        """
        # preallocate array for damages
        damages_do_not_adapt = np.zeros((self.return_periods.size, self.n), np.float32)
        damages_adapt = np.zeros((self.return_periods.size, self.n), np.float32)

        for i, return_period in enumerate(self.return_periods):
            # get flood map
            flood_map = self.flood_maps[return_period]
            # reproject_households_to_floodmap (should be done somewhere else, this is repetitive)

            # calculate damages household (assuming every household has its own building)
            damages_do_not_adapt[i, :] = np.array(
                object_scanner(
                    objects=self.var.household_points,
                    hazard=flood_map,
                    curves=self.var.buildings_content_curve,
                )
            )

            # calculate damages for adapted households
            damages_adapt[i, :] = np.array(
                object_scanner(
                    objects=self.var.household_points,
                    hazard=flood_map,
                    curves=self.var.buildings_content_curve_adapted,
                )
            )

        return damages_do_not_adapt, damages_adapt

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

    def load_ensemble_flood_maps(self):
        # Load all the flood maps in an ensemble per each day
        days = self.model.config["general"]["forecasts"]["days"]
        ensemble_per_day = []

        for day in days:
            members = []  # Every day has its own ensemble -- CHANGE IT LATER TO THE ACTUAL NUMBER OF MEMBERS
            for member in range(1, 4):
                file_path = (
                    self.model.output_folder
                    / "flood_maps"
                    / "ensemble"
                    / f"{day.strftime('%Y-%m-%d')}_member{member}.zarr"
                )  # Think later about which date info should be in the file: start time or forecast days?

                members.append(
                    xr.open_dataarray(file_path, engine="zarr")
                )  # Do I want to use this later?

            # Concatenate the members for each day (This stacks the list of dataarrays along a new "member" dimension)
            members_stacked = xr.concat(members, dim="member")
            ensemble_per_day.append(members_stacked)

        ensemble_flood_maps = xr.concat(ensemble_per_day, dim="day")

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

    def create_flood_probability_maps(self, strategy=1):
        """Creates flood probability maps based on the ensemble of flood maps for different warning strategies."""
        # Load the ensemble of flood maps
        ensemble_flood_maps = self.load_ensemble_flood_maps()

        days = self.model.config["general"]["forecasts"]["days"]
        crs = self.model.config["hazards"]["floods"]["crs"]
        # strategy = self.model.config["general"]["forecasts"]["strategy"]

        # Create output folder for probability maps
        prob_folder = self.model.output_folder / "prob_maps"
        prob_folder.mkdir(exist_ok=True, parents=True)

        # Define water level ranges based on the chosen strategy
        if strategy == 1:
            # Water level ranges associated to specific measures (based on "impact" scale)
            ranges = []
            for key, value in self.var.wlranges_and_measures.items():
                ranges.append((key, value["min"], value["max"]))

        elif strategy == 2:
            # Water level range for energy substations (based on critical hit) -- need to change it to a dictionary
            ranges = [(1, 0.3, None)]

        else:
            # Water level range for vulnerable and emergency facilities (flooded or not)
            ranges = [(1, 0.1, None)]

        probability_maps = {}

        # Loop over days and water level ranges to calculate probability maps
        for i, day in enumerate(days):
            for range_id, min, max in ranges:
                daily_ensemble = ensemble_flood_maps.isel(day=i)
                if max is not None:
                    condition = (daily_ensemble >= min) & (daily_ensemble <= max)
                else:
                    condition = daily_ensemble >= min
                probability = condition.sum(dim="member") / condition.sizes["member"]

                file_name = f"prob_map_range{range_id}_forecast{day.strftime('%Y-%m-%d')}_strategy{strategy}.tif"
                file_path = prob_folder / file_name

                # SOMETHING IS WRONG WHEN WRITING THE TIFF FILE, THE Y AXIS IS FLIPPED, IDWK WHY, so doing this to fix it for now -- need to fix it
                if probability.y.values[0] < probability.y.values[-1]:
                    print("flipping y axis")
                    probability = probability.sortby("y", ascending=False)

                probability = probability.rio.write_crs(crs)
                probability.rio.to_raster(file_path, driver="GTiff")

                probability_maps[(day, range_id)] = probability

        return probability_maps

    def create_damage_probability_maps(self) -> None:
        """Creates an object-based (buildings) probability map based on the ensemble of damage maps."""
        crs = self.model.config["hazards"]["floods"]["crs"]
        days = self.model.config["general"]["forecasts"]["days"]

        # Damage ranges for the probability map
        damage_ranges = [
            (1, 0, 1),
            (2, 0, 15000),
            (3, 15000, 30000),
            (4, 30000, 45000),
            (5, 45000, 60000),
            (6, 60000, None),
        ]
        # Later need to create a json dictionary with the right damage ranges

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
        self, prob_threshold=0.6, area_threshold=0.1
    ) -> None:
        # I probably should use the probability_maps as argument for this function instead of getting it inside the function
        # ideally add an option to choose the warning strategy

        days = self.model.config["general"]["forecasts"]["days"]
        range_ids = [1, 2, 3]
        # range_ids = list(self.var.wlranges_and_measures.keys()) activate it later when you have the prob maps for all right ranges
        warnings_log = []

        # Create probability maps
        self.create_flood_probability_maps()

        # Load households and postal codes
        households = self.var.household_points.copy()
        PC4 = gpd.read_parquet(self.model.files["geom"]["postal_codes"])
        # Maybe load this as a global var (?) instead of loading it each time

        for day in days:
            for range_id in range_ids:
                # Build path to probability map (is there a way of doing it with array instead of raster?)
                prob_map = Path(
                    self.model.output_folder
                    / "prob_maps"
                    / f"prob_map_range{range_id}_forecast{day.strftime('%Y-%m-%d')}.tif"
                )

                # Get the pixel values for each postal code
                stats = zonal_stats(PC4, prob_map, raster_out=True, nodata=np.nan)

                # Iterate through each postal code and check how many pixels exceed the threshold
                for i, postalcode in enumerate(stats):
                    pixel_values = postalcode["mini_raster_array"]
                    pc4_code = PC4.iloc[i]["postcode"]

                    # Only get the values that are within the postal code
                    postalcode_pixels = pixel_values[~pixel_values.mask]

                    # Calculate the number of pixels with flood prob above the threshold
                    n_above = np.sum(postalcode_pixels >= prob_threshold)

                    # Calculate the total number of pixels within the postal code
                    n_total = len(postalcode_pixels)

                    if n_total == 0:
                        print("No valid pixels found for postal code {pc4_code}")
                    else:
                        percentage = n_above / n_total

                    if percentage >= area_threshold:
                        print(
                            f"Warning issued to postal code {pc4_code} on {day} for range {range_id}: {percentage:.0%} of pixels > {prob_threshold}"
                        )

                        # Filter the affected households based on the postal code
                        affected_households = households[
                            households["postcode"] == pc4_code
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
                            source="water_levels",
                        )

                        warnings_log.append(
                            {
                                "day": day,
                                "postcode": pc4_code,
                                "range": range_id,
                                "n_affected_households": len(affected_households),
                                "n_warned_households": n_warned_households,
                                "percentage_above_threshold": percentage,
                            }
                        )
                        # Need to change day to time later

        # Save the warnings log to a csv file
        path = os.path.join(self.model.output_folder, "warnings_log_water_levels.csv")
        pd.DataFrame(warnings_log).to_csv(path, index=False)

    def get_critical_infrastructure(self):
        # I think this should only be run once, when loading the assets
        """Extract critical infrastructure elements (vulnerable and emergency facilities) from OSM using the catchment polygon as boundary."""
        catchment_boundary = gpd.read_parquet(
            self.model.files["geoms"]["catchment_boundary"]
        )

        # OSM needs a shapely geometry in EPSG:4326
        catchment_boundary = catchment_boundary.to_crs(epsg=4326)
        catchment_boundary = catchment_boundary.geometry.iloc[0]

        # Define the crs for the output data
        wanted_crs = self.model.config["hazards"]["floods"]["crs"]

        # Define the queries for vulnerable and emergency facilities
        # These queries are based on OSM tags, you can modify them as needed
        # OSMnx considers an AND statement across different tag keys, so we need to split queries and then combine them later
        queries = {
            "vulnerable_facilities_query": {
                "amenity": [
                    "hospital",
                    "school",
                    "kindergarten",
                    "nursing_home",
                    "childcare",
                ]
            },
            "emergency_facilities_query1": {"amenity": ["fire_station", "police"]},
            "emergency_facilities_query2": {"emergency": ["ambulance_station"]},
        }

        # Extract the wanted facilities from OSM using the queries
        gdf = {}
        for key, query in queries.items():
            gdf[key] = ox.features_from_polygon(catchment_boundary, query)

        # Combine emergency facilities from both queries
        emergency_facilities = gpd.GeoDataFrame(
            pd.concat(
                [
                    gdf["emergency_facilities_query1"],
                    gdf["emergency_facilities_query2"],
                ],
                ignore_index=True,
            )
        )
        vulnerable_facilities = gdf["vulnerable_facilities_query"]

        dataset = {
            "vulnerable_facilities": vulnerable_facilities,
            "emergency_facilities": emergency_facilities,
        }

        # Reproject the facilities to the wanted CRS and save them to gpkg files
        for name, facility in dataset.items():
            facility = facility.to_crs(wanted_crs)
            save_path = os.path.join(self.model.output_folder, f"{name}.gpkg")
            facility.to_file(save_path, driver="GPKG")

        critical_facilities = list(dataset.values())
        self.var.buildings = self.update_buildings_w_critical_infrastructure(
            critical_facilities
        )

        self.var.buildings.to_file(
            os.path.join(
                self.model.output_folder, "buildings_with_critical_infrastructure.gpkg"
            ),
            driver="GPKG",
        )

    def update_buildings_w_critical_infrastructure(self, critical_infrastructure):
        """Updates the buildings layer with the attributes from the critical infrastructure layer based on spatial intersection."""
        buildings = self.var.buildings.copy()

        for critical_infrastructure_type in critical_infrastructure:
            # Spatial join: find which facility features intersect which buildings
            joined = gpd.sjoin(
                buildings,
                critical_infrastructure_type,
                how="left",
                predicate="intersects",
                lsuffix="bld",
                rsuffix="fac",
            )

            # Take only the first match for each building
            # This is to avoid duplicating buildings if they intersect with multiple facilities
            joined = joined[~joined.index.duplicated(keep="first")]

            # Identify shared columns
            common_cols = buildings.columns.intersection(
                critical_infrastructure_type.columns
            )
            common_cols = [
                col for col in common_cols if col not in ("geometry", "index_right")
            ]

            # Replace only where there was a match in the column name
            for col in common_cols:
                col_fac = f"{col}_fac"
                if col_fac in joined.columns:
                    buildings[col] = joined[col_fac].combine_first(buildings[col])

        return buildings

    def assign_energy_substations_to_postal_codes(self, substations, pc4):
        # -- Need to improve this with Thiessen polygons or similar
        PC4_with_substations = gpd.sjoin_nearest(
            pc4, substations[["fid", "geometry"]], how="left", distance_col="distance"
        )
        # Rename the fid_right column for clarity
        PC4_with_substations.rename(
            columns={"fid_right": "substation_id"},
            inplace=True,
        )
        return PC4_with_substations

    def assign_critical_facilities_to_postal_codes(self, critical_facilities, pc4):
        # Use the centroid of the critical facilities to assign them to postal codes
        critical_facilities_centroid = critical_facilities.copy()
        critical_facilities_centroid["geometry"] = (
            critical_facilities_centroid.geometry.centroid
        )

        # Save the centroids to a file
        critical_facilities_centroid.to_file(
            os.path.join(self.model.output_folder, "critical_facilities_centroid.gpkg"),
            driver="GPKG",
        )

        # Spatial join to assign postal codes to critical facilities based on their centroid
        critical_facilities_with_PC6 = gpd.sjoin(
            critical_facilities_centroid[["id", "geometry"]],
            pc4,
            how="left",
            predicate="within",
        )

        # Rename the id column for clarity and drop fid so it can save to gpkg
        critical_facilities_with_PC6 = critical_facilities_with_PC6.rename(
            columns={"id": "facility_id"}
        ).drop(columns=["fid"])

        # Save the critical facilities with postal codes to a file
        critical_facilities_with_PC6.to_file(
            os.path.join(self.model.output_folder, "critical_facilities_with_PC6.gpkg")
        )

        return critical_facilities_with_PC6

    def critical_infrastructure_warning_strategy(self, prob_threshold=0.6):
        """This function implements an evacuation warning strategy based on critical infrastructure elements, such as energy substations, vulnerable and emergency facilities."""
        ## Load inputs -- This should only be done once when loading the assets in the model, need to fix it later
        # Get the household points, needed to issue warnings
        households = self.var.household_points.copy()

        # Get the forecast start date from the config (to know to which forecast day the warnings are associated to)
        # Before commiting NEED to change days to times and remove the filter for days[0]
        day = self.model.config["general"]["forecasts"]["days"][0]

        # Load postal codes, energy substations and vulnerable/emergency facilities
        substations = gpd.read_parquet(
            self.model.files["geoms"]["assets/energy_substations"]
        )
        PC4 = gpd.read_parquet(self.model.files["geoms"]["postal_codes"])
        critical_facilities = gpd.read_file(
            os.path.join(self.model.output_folder, "vulnerable_facilities.gpkg")
        )
        # need to get from the folder where get_critical_infrastructure (which should only run once) function saves it
        # need to combine critical with vulnerable facilities

        ## For energy substations:
        strategy_id = 2

        # Assign substations to postal codes based on distance  -- This should only be done once when loading the assets in the model
        PC4_with_substations = self.assign_energy_substations_to_postal_codes(
            substations, PC4
        )

        # Create flood probability maps associated to the critical hit of energy substations
        # Need to give the number (id) of strategy as argument
        self.create_flood_probability_maps(strategy=strategy_id)
        prob_maps_directory = self.model.output_folder / "prob_maps"

        # Get the probability map for the specific day and strategy
        prob_energy_hit_path = (
            prob_maps_directory
            / f"prob_map_range1_forecast{day.strftime('%Y-%m-%d')}_strategy{strategy_id}.tif"
        )

        # Should I open this with "with rasterio.open()..."? -- need to check
        prob_map = rasterio.open(prob_energy_hit_path)
        prob_array = prob_map.read(1)
        affine = prob_map.transform

        # Sample the probability map at the substations locations
        sampled_probs = point_query(
            substations, prob_array, affine=affine, interpolate="nearest"
        )
        substations["probability"] = sampled_probs

        # Filter substations that have a flood hit probability > threshold
        critical_hits_energy = substations[substations["probability"] >= prob_threshold]

        # Create an empty list to store the postcodes
        affected_postcodes_energy = []

        # If there are critical hits, issue warnings
        if not critical_hits_energy.empty:
            print(f"Critical hits for energy substations: {len(critical_hits_energy)}")

            # Get the postcodes that will be affected by the critical hits of energy substations
            affected_postcodes_energy = PC4_with_substations[
                PC4_with_substations["substation_id"].isin(critical_hits_energy["fid"])
            ]["postcode"].unique()

        ## For vulnerable and emergency facilities:
        strategy_id = 3

        # Assign critical facilities to postal codes based on intersection -- This should only be done once when loading the assets, need to fix it
        critical_facilities_with_PC6 = self.assign_critical_facilities_to_postal_codes(
            critical_facilities, PC4
        )

        # Create flood probability map for vulnerable and emergency facilities
        self.create_flood_probability_maps(strategy=strategy_id)

        # Get the probability map for the specific day and strategy
        prob_critical_facilities_hit_path = (
            prob_maps_directory
            / f"prob_map_range1_forecast{day.strftime('%Y-%m-%d')}_strategy{strategy_id}.tif"
        )

        # Should I open this with "with rasterio.open()..."? -- need to check
        prob_map = rasterio.open(prob_critical_facilities_hit_path)
        prob_array = prob_map.read(1)
        affine = prob_map.transform

        # Sample the probability map at the facilities locations using their centroid (can be improved later to use whole area of the polygon)
        critical_facilities = critical_facilities.copy()

        sampled_probs = point_query(
            critical_facilities.geometry.centroid,
            prob_array,
            affine=affine,
            interpolate="nearest",
        )
        critical_facilities["probability"] = sampled_probs

        # Filter facilities that have a flood hit probability > threshold
        critical_hits_facilities = critical_facilities[
            critical_facilities["probability"] >= prob_threshold
        ]
        # Create an empty list to store the postcodes
        affected_postcodes_facilities = []

        # If there are critical hits, issue warnings
        if not critical_hits_facilities.empty:
            print(
                f"Critical hits for vulnerable/emergency facilities: {len(critical_hits_facilities)}"
            )

            # Get the postcodes that will be affected by the critical hits of facilities
            affected_postcodes_facilities = critical_facilities_with_PC6[
                critical_facilities_with_PC6["facility_id"].isin(
                    critical_hits_facilities["id"]
                )
            ]["postcode"].unique()

        # Function to keep track of what triggered the warning for each postal code
        def trigger_label(postcode):
            e = postcode in affected_postcodes_energy
            f = postcode in affected_postcodes_facilities
            if e and f:
                return "energy_and_facilities"
            elif e:
                return "energy"
            elif f:
                return "facilities"

        # Combine affected_postcodes from both strategies and remove duplicates
        affected_postcodes = np.unique(
            np.concatenate(
                (affected_postcodes_energy, affected_postcodes_facilities), axis=0
            )
        )

        # Create an empty log to store the warnings
        warning_log = []

        # Issue warnings to the households in the affected postcodes
        for postcode in affected_postcodes:
            affected_households = households[households["postcode"] == postcode]
            n_warned_households = self.warning_communication(
                target_households=affected_households,
                measures=set(),
                evacuate=True,
                source="critical_infrastructure",
            )
            trigger = trigger_label(postcode)

            print(
                f"Evacuation warning issued to postal code {postcode} on {day.strftime('%d/%m/%Y')} (trigger: {trigger})"
            )
            warning_log.append(
                {
                    "day": day,
                    "postcode": postcode,
                    "n_warned_households": n_warned_households,
                    "trigger": trigger,
                }
            )

            # need to add this to the warning_log:
            # "critical_hits_energy": len(
            #             critical_hits_energy[
            #                 critical_hits_energy["postcode"] == postcode
            #             ]
            #         ),
            #         "critical_hits_facilities": len(
            #             critical_hits_facilities[
            #                 critical_hits_facilities["postcode"] == postcode
            #             ]

        # Save log
        path = os.path.join(
            self.model.output_folder, "warning_log_critical_infrastructure.csv"
        )
        pd.DataFrame(warning_log).to_csv(path, index=False)

    def warning_communication(
        self,
        target_households: gpd.GeoDataFrame,
        measures: set[str],
        evacuate: bool,
        source: str,
        communication_efficiency: float = 0.35,
        lt_threshold_evacuation: int = 48,
    ) -> int:
        """Send warnings to a subset of target households according to communication_efficiency.

        Enforces single-message rule and allows only upward escalation:
        none -> measures -> evacuate
        Evacuation only sent if evacuate is True *and* lead_time <= 48h.
        """
        n_target_households = len(target_households)
        if n_target_households == 0:
            return 0

        # Get lead time in hours -- need to check if this works correctly
        start_time = self.model.config["hazards"]["floods"]["events"][0][
            "start_time"
        ].date()
        current_time = self.model.current_time.date()
        lead_time = (start_time - current_time).total_seconds() / 3600

        # Determine the desired level of warning (0 = no warning, 1 = measures, 2 = evacuate)
        desired_level = 2 if (evacuate and lead_time <= lt_threshold_evacuation) else (1 if measures else 0)  # fmt: skip

        if desired_level == 0:
            return 0

        # Get household indices of those who can receive the warning based on comm. efficiency (randomly selected)
        n_feasible_warnings = int(n_target_households * communication_efficiency)
        if n_feasible_warnings == 0:
            return 0

        position_indices = np.random.choice(
            n_target_households, n_feasible_warnings, replace=False
        )
        selected_households = target_households.iloc[position_indices]

        n_warned_households = 0
        for _, row in selected_households.iterrows():
            household_id = row["pointid"]

            # Skip already evacuated
            if self.var.evacuated[household_id] == 1:
                continue

            current_level = int(self.var.warning_level[household_id])

            # Only send a warning if the warning level is lower than the desired level
            if desired_level > current_level:
                # Apply the new warning
                if desired_level >= 1 and measures:
                    self.var.recommended_measures[household_id].update(measures)

                if desired_level == 2:
                    # Add 'evacuate' to the set of recommendations
                    self.var.recommended_measures[household_id].add("evacuate")
            else:
                continue

            # Mark warning delivered & record provenance
            self.var.warning_level[household_id] = desired_level
            self.var.warning_reached[household_id] = 1
            self.var.warning_source[household_id] = source
            # -- need to figure out how to add this
            n_warned_households += 1

        print(f"Warning targeted to reach {n_target_households} households")
        print(f"Warning reached {n_warned_households} households")

        return n_warned_households

    def warning_communication2(self, target_households, range_id, warning_range=0.35):
        # need to get the range measures from the other functions
        """Communicates a warning to households based on the communication efficiency.

        changes risk perception --> to be moved to the update risk perception function;
        and return the number of households that were warned.

        Args:
            target_households: The households that are targeted to receive the warning.

        Returns:
            The number of households that received the warning.
        """
        print("Communicating the warning...")

        # Get the measures and evacuation advice from the dict with wl ranges and measures
        wlranges_and_measures = self.var.wlranges_and_measures
        measures = wlranges_and_measures[range_id]["measure"]
        evacuate = wlranges_and_measures[range_id]["evacuate"]

        # Total number of target households
        n_target_households = len(target_households)

        # Get lead time in hours
        start_time = self.model.config["hazards"]["floods"]["events"][0][
            "start_time"
        ].date()
        current_time = self.model.current_time.date()
        lead_time = (start_time - current_time).total_seconds() / 3600

        # If no target households, return 0 warned households
        if n_target_households == 0:
            return 0

        # Get random indices to change the warning state
        n_warned_households = int(n_target_households * warning_range)
        position_indices = np.random.choice(
            n_target_households, n_warned_households, replace=False
        )
        selected_households = target_households.iloc[position_indices]

        warned_ids = []
        for _, row in selected_households.iterrows():
            # Get the 'original' index of the household point
            household_id = row["pointid"]

            # If the household has already evacuated, do not communicate the warning
            if self.var.evacuated[household_id] == 1:
                continue

            # Get current recommended measures as a set
            received = self.var.recommended_measures[household_id]
            # this should be the set where you store the recommended measures

            # If the household did not get the warning yet, issue warning with recommended measures
            # (if received already contains the measures, do not warn again)
            if not received.issuperset(measures):
                self.var.warning_reached[household_id] = 1
                self.var.recommended_measures[household_id].update(measures)
                warned_ids.append(household_id)

            # If evacuation is advised and household did not get the evacuation warning yet, and lead time ≤ 48h, issue evacuation warning
            if evacuate and "evacuate" not in received and lead_time <= 48:
                self.var.warning_reached[household_id] = 1
                self.var.recommended_measures[household_id].add("evacuate")
                warned_ids.append(household_id)

        print(f"Warning targeted to reach {n_target_households} households.")
        print(f"Warning reached {len(set(warned_ids))} households.")

        return n_warned_households
        # Need to add the evacuation warning from second warning strategy in case the leadtime is <48h and there is no evacuation warning coming from the first strategy

    def household_decision_making(self, responsive_ratio=0.7):
        """Summary.

        This function simulates the decision-making process of households that received warnings,
        based on recommended measures, lead time, and implementation times.
        """
        print("Running emergency response decision-making for households...")
        # Need to fix description of functions

        # Initialize an empty list to log actions taken
        actions_log = []

        # Get lead time in hours -- need to create a helper function for this
        start_time = self.model.config["hazards"]["floods"]["events"][0][
            "start_time"
        ].date()
        current_time = self.model.current_time.date()
        lead_time = (start_time - current_time).total_seconds() / 3600

        implementation_times = self.var.implementation_times
        evacuation_time = implementation_times["evacuate"]

        # Filter households that did not evacuate, were warned and are responsive
        # (AND FILTER OUT WHO TOOK ACTION, BUT THERE IS THE PROBLEM WITH EVACUATION...)
        not_evacuated_ids = self.var.evacuated == 0
        warned_ids = self.var.warning_reached == 1
        responsive_ids = self.var.response_probability < responsive_ratio

        eligible_ids = warned_ids & not_evacuated_ids & responsive_ids

        # Apply the indices to household_points (maybe I can get the postal codes or add actions to this one)
        eligible_households = self.var.household_points[eligible_ids]

        # For every eligible household, check the measures and their time of implementation
        for household_id, _ in eligible_households.iterrows():
            actions = []
            possible_measures = self.var.recommended_measures[household_id]
            total_time_all = sum(
                implementation_times[measure] for measure in possible_measures
            )

            # Check if evacuation is in the recommended measures
            evacuate = "evacuate" in possible_measures

            # Check if there were advised any inplace measures
            inplace_measures = [
                measure for measure in possible_measures if measure != "evacuate"
            ]

            # If only inplace measures are advised...
            if not evacuate:
                if self.var.actions_taken[household_id]:
                    continue  # Skip if actions were already taken

                # If lead time is sufficient for all (inplace) measures, take all measures
                if lead_time >= total_time_all:
                    actions = list(possible_measures)
                else:
                    # If lead time is not sufficient for all measures, check if any measure can be done
                    actions = [
                        measure
                        for measure in possible_measures
                        if implementation_times[measure] <= lead_time
                    ]
                    # Take only the first measure from the list that fits the lead time
                    actions = actions[:1] if actions else []

            # If evacuation is advised...
            else:
                # Check if lead time is sufficient for all measures and evacuation
                if lead_time >= total_time_all:
                    actions = list(possible_measures)
                    self.var.evacuated[household_id] = 1
                elif inplace_measures:
                    # If lead time is not sufficient for all measures, check if any inplace measure can be done before evacuation
                    for m in inplace_measures:
                        measure_implementation_time = implementation_times[m]
                        if measure_implementation_time + evacuation_time <= lead_time:
                            actions = [m, "evacuate"]
                            self.var.evacuated[household_id] = 1
                            break
                    else:
                        if evacuation_time <= lead_time:
                            actions = ["evacuate"]
                            self.var.evacuated[household_id] = 1
                # elif evacuation_time <= lead_time:
                #     # If no inplace measures are advised and there is enough time, only evacuate
                #     actions = ["evacuate"]
                #     self.var.evacuated[household_id] = 1 --> Actually I think I dont need this

            # If there are actions to take, log them and update self.var.actions_taken
            if actions:
                self.var.actions_taken[household_id].update(actions)
                # self.var.household_points.at[household_id, "actions_taken"] = (
                #    self.var.actions_taken[household_id]
                # )
                actions_log.append(
                    {
                        "lead_time": lead_time,
                        "postal_code": self.var.household_points.loc[
                            household_id, "postcode"
                        ],
                        "household_id": household_id,
                        "actions": actions,
                    }
                )

        # Save actions log
        path = os.path.join(self.model.output_folder, "actions_log.csv")
        pd.DataFrame(actions_log).to_csv(path, index=False)

        # NEED TO INCLUDE 2ND STRATEGY HERE
        # NEED TO SAVE THE ACTIONS LOG
        # NEED TO SAVE THE ACTIONS TAKEN IN THE VAR OBJECT AS SET OR LIST

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
        self.buildings = gpd.read_parquet(self.model.files["geom"]["assets/buildings"])
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
        with open(
            self.model.files["dict"][
                "damage_parameters/flood/buildings/structure/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_buildings_structure = float(json.load(f)["maximum_damage"])
        self.buildings["maximum_damage_m2"] = self.var.max_dam_buildings_structure

        with open(
            self.model.files["dict"][
                "damage_parameters/flood/buildings/content/maximum_damage"
            ],
            "r",
        ) as f:
            max_dam_buildings_content = json.load(f)
        self.var.max_dam_buildings_content = float(
            max_dam_buildings_content["maximum_damage"]
        )
        self.buildings_centroid["maximum_damage"] = self.var.max_dam_buildings_content

        with open(
            self.model.files["dict"][
                "damage_parameters/flood/rail/main/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_rail = float(json.load(f)["maximum_damage"])
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
            with open(self.model.files["dict"][path], "r") as f:
                max_damage = json.load(f)
            max_dam_road_m[road_type] = max_damage["maximum_damage"]

        self.roads["maximum_damage_m"] = self.roads["object_type"].map(max_dam_road_m)

        with open(
            self.model.files["dict"][
                "damage_parameters/flood/land_use/forest/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_forest_m2 = float(json.load(f)["maximum_damage"])

        with open(
            self.model.files["dict"][
                "damage_parameters/flood/land_use/agriculture/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_agriculture_m2 = float(json.load(f)["maximum_damage"])

    def load_damage_curves(self) -> None:
        # Load vulnerability curves [look into these curves, some only max out at 0.5 damage ratio]
        road_curves = []
        road_types = [
            ("residential", "damage_parameters/flood/road/residential/curve"),
            ("unclassified", "damage_parameters/flood/road/unclassified/curve"),
            ("tertiary", "damage_parameters/flood/road/tertiary/curve"),
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

    def load_wlranges_and_measures(self):
        with open(self.model.files["dict"]["measures/implementation_times"], "r") as f:
            self.var.implementation_times = json.load(f)

        with open(self.model.files["dict"]["measures/wl_ranges"], "r") as f:
            dict = json.load(f)
            # convert the keys (range ids) to integers and store them in a new dictionary
            self.var.wlranges_and_measures = {int(k): v for k, v in dict.items()}

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
            self.change_household_locations()  # ideally this should be done in the setup_population when building the model
        self.get_critical_infrastructure()
        self.critical_infrastructure_warning_strategy()
        self.water_level_warning_strategy()
        self.household_decision_making()

    def calculate_building_flood_damages(self) -> Tuple[np.ndarray, np.ndarray]:
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
            buildings_with_damages = buildings[["osm_id", "osm_way_id"]]
            buildings_with_damages["damage"] = damage_unprotected
            # damage_unprotected = damage_unprotected[["osm_id", "osm_way_id", "damage"]]

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
            buildings_with_damages_floodproofed = buildings[["osm_id", "osm_way_id"]]
            buildings_with_damages_floodproofed["damage"] = damage_flood_proofed

            # add damages to agents (unprotected buildings)
            for _, row in buildings_with_damages.iterrows():
                damage = row["damage"]
                if row["osm_id"] is not None:
                    osm_id = int(row["osm_id"])
                    idx_agents_in_building = np.where(self.var.osm_id == osm_id)[0]
                    damages_do_not_adapt[i, idx_agents_in_building] = damage
                else:
                    osm_way_id = int(row["osm_way_id"])
                    idx_agents_in_building_way = np.where(
                        self.var.osm_way_id == osm_way_id
                    )[0]
                    damages_do_not_adapt[i, idx_agents_in_building_way] = damage

            # add damages to agents (flood-proofed buildings)
            for _, row in buildings_with_damages_floodproofed.iterrows():
                damage = row["damage"]
                if row["osm_id"] is not None:
                    osm_id = int(row["osm_id"])
                    idx_agents_in_building = np.where(self.var.osm_id == osm_id)[0]
                    damages_adapt[i, idx_agents_in_building] = damage
                else:
                    osm_way_id = int(row["osm_way_id"])
                    idx_agents_in_building_way = np.where(
                        self.var.osm_way_id == osm_way_id
                    )[0]
                    damages_adapt[i, idx_agents_in_building_way] = damage

        return damages_do_not_adapt, damages_adapt

    def flood(self, flood_depth: xr.DataArray) -> float:
        if self.config["warning_response"]:
            self.change_household_locations()  # ideally this should be done in the setup_population when building the model
        self.get_critical_infrastructure()
        self.critical_infrastructure_warning_strategy()
        self.water_level_warning_strategy()
        self.household_decision_making()

        print("test")

    def flood(self, flood_map: xr.DataArray) -> float:
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
        ] = True  # need to change this

        buildings: gpd.GeoDataFrame = gpd.sjoin_nearest(
            buildings, household_points, how="left", exclusive=True
        )
        # I need to check if this is working

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
    ) -> Tuple[
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
