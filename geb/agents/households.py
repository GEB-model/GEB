"""This module contains the Households agent class for simulating household behavior in the GEB model."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import xarray as xr
from pyproj import CRS
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterstats import point_query
from scipy import interpolate
from shapely.geometry import shape
from workflows.households.early_warning import EarlyWarningModule

from geb.geb_types import ArrayFloat32, TwoDArrayBool, TwoDArrayInt
from geb.workflows.io import read_params

from ..hydrology.landcovers import (
    FOREST,
)
from ..store import Bucket, DynamicArray
from ..workflows.damage_scanner import VectorScanner, VectorScannerMultiCurves
from ..workflows.io import read_array, read_table, read_zarr
from .decision_module import DecisionModule
from .general import AgentBaseClass

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel


def from_landuse_raster_to_polygon(
    mask: TwoDArrayBool | TwoDArrayInt, transform: Affine, crs: str | int | CRS
) -> gpd.GeoDataFrame:
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


class HouseholdVariables(Bucket):
    """Variables for the Households agent."""

    household_points: gpd.GeoDataFrame
    actions_taken: DynamicArray
    possible_measures: list[str]
    possible_warning_triggers: list[str]
    municipal_water_demand_per_capita_m3_baseline: ArrayFloat32
    water_demand_per_household_m3: ArrayFloat32
    income: DynamicArray
    building_id_of_household: DynamicArray
    wealth: DynamicArray
    property_value: DynamicArray
    locations: DynamicArray
    years_since_last_flood: DynamicArray
    risk_perception: DynamicArray
    sizes: DynamicArray
    water_efficiency_per_household: ArrayFloat32
    municipal_water_withdrawal_m3_per_capita_per_day_multiplier: pd.DataFrame
    risk_perc_max: float
    risk_perc_min: float
    risk_decr: float
    wlranges_and_measures: dict[int, Any]
    implementation_times: Any
    rail_curve: pd.DataFrame
    warning_trigger: DynamicArray
    evacuated: DynamicArray
    warning_level: DynamicArray
    warning_reached: DynamicArray
    response_probability: DynamicArray
    amenity_value: DynamicArray
    adaptation_costs: DynamicArray
    recommended_measures: DynamicArray
    time_adapted: DynamicArray
    max_dam_buildings_structure: float
    forest_curve: pd.DataFrame
    agriculture_curve: pd.DataFrame
    road_curves: pd.DataFrame
    adapted: DynamicArray
    max_dam_buildings_content: float
    max_dam_rail: float
    max_dam_forest_m2: float
    max_dam_agriculture_m2: float
    region_id: DynamicArray


class Households(AgentBaseClass):
    """This class implements the household agents."""

    var: HouseholdVariables

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
        self.early_warning_module = EarlyWarningModule(self)
        if self.config["adapt"]:
            self.load_flood_maps()
            self.flood_risk_perceptions = []  # Store the flood risk perceptions in here
            self.flood_risk_perceptions_statistics = []  # Store some statistics on flood risk perceptions here

        self.load_objects()
        self.load_max_damage_values()
        self.load_damage_curves()
        if self.model.config["agent_settings"]["households"]["warning_response"]:
            self.load_critical_infrastructure()  # ideally this should be done in the setup_assets when building the model
            self.load_wlranges_and_measures()

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Return the name of the agent type."""
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
            flood_maps[return_period] = read_zarr(file_path)
        self.flood_maps = flood_maps

    def construct_income_distribution(self) -> None:
        """Construct a lognormal income distribution for the region."""
        # These values only work for a single country now. Should come from subnational datasets.
        distribution_parameters = read_table(
            self.model.files["table"]["income/distribution_parameters"]
        )

        # Use first available country from distribution parameters (consistent with GDL regions used in build phase)
        # This is a simplification - in the future this should use proper subnational datasets
        available_countries = list(distribution_parameters.columns)
        country = available_countries[0]
        self.model.logger.info(
            "Using income distribution for country: %s (first available from GDL regions)",
            country,
        )

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
        wealth_index = read_array(
            self.model.files["array"]["agents/households/wealth_index"]
        )
        self.var.wealth_index = DynamicArray(wealth_index, max_n=self.max_n)

        income_percentiles = read_array(
            self.model.files["array"]["agents/households/income_percentile"]
        )
        self.var.income_percentile = DynamicArray(income_percentiles, max_n=self.max_n)

        # assign household disposable income based on income percentile households
        income = read_array(self.model.files["array"]["agents/households/disp_income"])
        self.var.income = DynamicArray(income, max_n=self.max_n)

        # assign wealth based on income (dummy data, there are ratios available in literature)
        self.var.wealth = DynamicArray(2.5 * self.var.income.data, max_n=self.max_n)

    def update_building_attributes(self) -> None:
        """Update building attributes based on household data."""
        # Start by computing n occupants from the var.building_id_of_household array
        building_id_of_household_series = pd.Series(
            self.var.building_id_of_household.data
        )

        # Drop NaNs and convert to string format (matching building ID format)
        building_id_of_household_counts = (
            building_id_of_household_series.dropna().astype(int).value_counts()
        )
        # Initialize occupants column
        self.buildings["n_occupants"] = 0

        # Map the counts back to the buildings dataframe
        self.buildings["n_occupants"] = (
            self.buildings["id"]
            .map(building_id_of_household_counts)
            .fillna(self.buildings["n_occupants"])
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
        building_id_of_household = pd.DataFrame(
            np.unique(self.var.building_id_of_household.data[household_adapting])
        ).dropna()
        building_id_of_household = building_id_of_household.astype(int).astype(str)
        building_id_of_household["flood_proofed"] = True
        building_id_of_household = building_id_of_household.set_index(0)

        # Add/Update the flood_proofed status in buildings based on OSM way IDs
        self.buildings["flood_proofed"] = (
            self.buildings["id"]
            .astype(str)
            .map(building_id_of_household["flood_proofed"])
        )

        # Replace NaNs with False (i.e., buildings not in the adapting households list)
        self.buildings["flood_proofed"] = self.buildings["flood_proofed"].fillna(False)

    def assign_household_attributes(self) -> None:
        """Household locations are already sampled from population map in GEBModel.setup_population().

        These are loaded in the spinup() method.
        Here we assign additional attributes (dummy data) to the households that are used in the decision module.
        """
        # load household locations
        locations = read_array(self.model.files["array"]["agents/households/location"])
        self.max_n = int(locations.shape[0] * (1 + self.reduncancy) + 1)
        self.var.locations = DynamicArray(
            locations, max_n=self.max_n, extra_dims_names=["lonlat"]
        )

        region_id = read_array(self.model.files["array"]["agents/households/region_id"])
        self.var.region_id = DynamicArray(region_id, max_n=self.max_n)

        # load household sizes
        sizes = read_array(self.model.files["array"]["agents/households/size"])
        self.var.sizes = DynamicArray(sizes, max_n=self.max_n)

        self.var.municipal_water_demand_per_capita_m3_baseline = read_array(
            self.model.files["array"][
                "agents/households/municipal_water_withdrawal_per_capita_m3_baseline"
            ]
        )

        # set municipal water demand efficiency to 1.0 for all households
        self.var.water_efficiency_per_household = np.full_like(
            self.var.municipal_water_demand_per_capita_m3_baseline, 1.0, np.float32
        )

        self.var.municipal_water_withdrawal_m3_per_capita_per_day_multiplier = (
            read_table(
                self.model.files["table"][
                    "municipal_water_withdrawal_m3_per_capita_per_day_multiplier"
                ]
            )
        )

        # load building id household
        building_id_of_household = read_array(
            self.model.files["array"]["agents/households/building_id_of_household"]
        )
        self.var.building_id_of_household = DynamicArray(
            building_id_of_household, max_n=self.max_n
        )

        # update building attributes based on household data
        if self.config["adapt"]:
            self.update_building_attributes()

        # load age household head
        age_household_head = read_array(
            self.model.files["array"]["agents/households/age_household_head"]
        )
        self.var.age_household_head = DynamicArray(age_household_head, max_n=self.max_n)

        # load education level household head
        education_level = read_array(
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
            np.zeros((self.n, len(self.var.possible_warning_triggers)), dtype=bool),
            extra_dims_names=["trigger_type"],
            max_n=self.max_n,
        )

        # initiate array for response probability (between 0 and 1)
        # using a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        self.var.response_probability = DynamicArray(
            rng.random(self.n), max_n=self.max_n
        )

        # initiate array for evacuation status [0=not evacuated, 1=evacuated]
        self.var.evacuated = DynamicArray(
            np.zeros(self.n, np.int32),
            max_n=self.max_n,
            extra_dims_names=["evacuation_status"],
        )

        # this list defines the possible measures that can be recommended to/taken by households
        self.var.possible_measures = [
            "elevate possessions",
            "sandbags",
            "evacuate",
        ]

        # initiate array for storing the recommended measures received with the warnings
        self.var.recommended_measures = DynamicArray(
            np.zeros((self.n, len(self.var.possible_measures)), dtype=bool),
            max_n=self.max_n,
            extra_dims_names=["measure_type"],
        )

        # initiate array for storing the actions taken by the household
        self.var.actions_taken = DynamicArray(
            np.zeros((self.n, len(self.var.possible_measures)), dtype=bool),
            max_n=self.max_n,
            extra_dims_names=["measure_type"],
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
            (self.var.wealth.data * 0.8).astype(np.int64), max_n=self.max_n
        )
        # initiate array with RANDOM annual adaptation costs [dummy data for now, values are available in literature]
        adaptation_costs = (
            np.maximum(self.var.property_value.data * 0.05, 10_800)
        ).astype(np.int64)
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

    def update_risk_perceptions(self) -> None:
        """Update the risk perceptions of households based on the latest flood data."""
        # update timer
        self.var.years_since_last_flood.data += 1

        # Here we update flood risk perception based on actual floods that have happened and whether a household was flooded (yes/no)
        if self.config["adapt_to_actual_floods"]:
            # Find the flood event that corresponds to the current time (in the model)
            for event in self.flood_events:
                end: datetime = event["end_time"]

                if (
                    self.model.current_time == end + timedelta(days=14)
                ):  # Households won't start to immediately think about adapting after a flood occurs. For now an assumption, that after 2 weeks they will start adapting
                    # Open the flood map
                    flood_map_name: str = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
                    flood_map_path: Path = (
                        self.model.output_folder / "flood_maps" / flood_map_name
                    )

                    flood_map: xr.DataArray = read_zarr(flood_map_path)

                    buildings = (
                        self.buildings.copy()
                    )  # Make copy of buildings to be safe

                    buildings_proj = buildings.to_crs(
                        flood_map.rio.crs
                    )  # Reproject buildings to flood map

                    # get building centroids
                    xs = buildings_proj.geometry.centroid.x.values
                    ys = buildings_proj.geometry.centroid.y.values

                    # Interpolate flood depths at building centroids
                    depths = flood_map.interp(x=("points", xs), y=("points", ys)).values
                    depths = np.nan_to_num(
                        depths, nan=0.0
                    )  # replace NaNs outside the raster with 0

                    # Identify flooded buildings (more than 5cm of water)
                    buildings_proj["flooded"] = depths > 0.05

                    # Find households located in flooded buildings
                    flooded_building_ids = set(
                        buildings_proj.loc[buildings_proj["flooded"], "id"]
                    )

                    flooded_households = np.isin(
                        self.var.building_id_of_household,
                        list(flooded_building_ids),
                    )

                    # Reset years_since_last_flood to 0 for flooded households
                    self.var.years_since_last_flood.data[flooded_households] = 0

                    # Some debugging print statements
                    n_buildings = len(buildings_proj)
                    n_flooded_buildings = buildings_proj["flooded"].sum()
                    n_flooded_households = flooded_households.sum()

                    print(
                        f"Flooded buildings: {n_flooded_buildings}/{n_buildings}, "
                        f"Flooded households: {n_flooded_households}"
                    )

        else:
            if (
                np.random.random() < 0.1
            ):  # generate random flood (not based on actual modeled flood data)
                print("Flood event!")
                self.var.years_since_last_flood[:] = 0

        self.var.risk_perception.data = (
            self.var.risk_perc_max
            * 1.6 ** (self.var.risk_decr * self.var.years_since_last_flood.data)
            + self.var.risk_perc_min
        )

        stats = {
            "time": self.model.current_time,
            "min_risk": np.min(self.var.risk_perception.data),
            "max_risk": np.max(self.var.risk_perception.data),
            "mean_risk": np.mean(self.var.risk_perception.data),
        }
        self.flood_risk_perceptions_statistics.append(stats)

        # Append and save floor risk perception data spatially
        df = pd.DataFrame(
            {
                "time": [self.model.current_time] * len(self.var.household_points),
                "x": self.var.household_points.geometry.x,
                "y": self.var.household_points.geometry.y,
                "risk_perception": self.var.risk_perception.data,
                "years_since_last_flood": self.var.years_since_last_flood.data,
            }
        )
        self.flood_risk_perceptions.append(df)

    def load_critical_infrastructure(self) -> None:
        """Load critical infrastructure elements (vulnerable and emergency facilities, energy substations) and assign them to postal codes."""
        # Load postal codes
        postal_codes = self.postal_codes.copy()

        # Get critical facilities (vulnerable and emergency) and update buildings with relevant attributes
        self.get_critical_facilities()

        # Assign critical facilities to postal codes
        critical_facilities = gpd.read_parquet(
            self.model.files["geom"]["assets/critical_facilities"]
        )
        self.assign_critical_facilities_to_postal_codes(
            critical_facilities, postal_codes
        )

        # Get energy substations and assign them to postal codes
        substations = gpd.read_parquet(
            self.model.files["geom"]["assets/energy_substations"]
        )
        self.assign_energy_substations_to_postal_codes(substations, postal_codes)

    def get_critical_facilities(self) -> None:
        """Extract critical infrastructure elements (vulnerable and emergency facilities) from OSM using the catchment polygon as boundary."""
        catchment_boundary = gpd.read_parquet(
            self.model.files["geom"]["catchment_boundary"]
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

        # Combine emergency facilities from all queries
        all_critical_facilities = gpd.GeoDataFrame(
            pd.concat(
                [
                    gdf["emergency_facilities_query1"],
                    gdf["emergency_facilities_query2"],
                    gdf["vulnerable_facilities_query"],
                ],
                ignore_index=True,
            )
        )

        # Reproject the facilities to the wanted CRS and save them
        all_critical_facilities.insert(0, "id", all_critical_facilities.index + 1)
        all_critical_facilities.to_crs(wanted_crs, inplace=True)
        save_path = (
            self.model.input_folder
            / "geom"
            / "assets"
            / "critical_facilities.geoparquet"
        )
        all_critical_facilities.to_parquet(save_path)
        print(f"Loaded {len(all_critical_facilities)} critical facilities.")

        # Update buildings with critical infrastructure attributes
        buildings = self.update_buildings_w_critical_infrastructure(
            all_critical_facilities
        )

        # Save updated buildings with critical infrastructure attributes
        save_path = (
            self.model.input_folder
            / "geom"
            / "assets"
            / "buildings_with_critical_facilities.geoparquet"
        )
        buildings.to_parquet(save_path)

        # Update the buildings (global variable) for later use
        self.buildings = buildings

    def update_buildings_w_critical_infrastructure(
        self, critical_infrastructure: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Update buildings layer with critical infrastructure attributes via spatial intersection.

        Args:
            critical_infrastructure: Iterable of GeoDataFrames representing different sets of
                critical infrastructure (e.g., vulnerable facilities, emergency facilities).

        Returns:
            Copy of buildings with updated attributes from critical infrastructure data where spatial intersections occurred.
        """
        buildings = self.buildings.copy()
        # TODO: check if this function is needed with the new OBM data

        # Spatial join: find which facility features intersect which buildings
        joined = gpd.sjoin(
            buildings,
            critical_infrastructure,
            how="left",
            predicate="intersects",
            lsuffix="bld",
            rsuffix="fac",
        )

        # Take only the first match for each building
        # This is to avoid duplicating buildings if they intersect with multiple facilities
        joined = joined[~joined.index.duplicated(keep="first")]

        # Identify shared columns
        common_cols = buildings.columns.intersection(critical_infrastructure.columns)
        common_cols = [
            col for col in common_cols if col not in ("geometry", "index_right")
        ]

        # Replace only where there was a match in the column name
        for col in common_cols:
            col_fac = f"{col}_fac"
            if col_fac in joined.columns:
                buildings[col] = joined[col_fac].combine_first(buildings[col])

        return buildings

    def assign_energy_substations_to_postal_codes(
        self, substations: gpd.GeoDataFrame, postal_codes: gpd.GeoDataFrame
    ) -> None:
        """Assign energy substations to postal codes based on spatial proximity. Every postal code gets assigned to the nearest substation.

        Args:
            substations: GeoDataFrame of energy substations.
            postal_codes: GeoDataFrame of postal codes.
        """
        # TODO: need to improve this with Thiessen polygons or similar
        postal_codes_with_substations = gpd.sjoin_nearest(
            postal_codes,
            substations[["fid", "geometry"]],
            how="left",
            distance_col="distance",
        )
        # Rename the fid_right column for clarity
        postal_codes_with_substations.rename(
            columns={"fid_right": "substation_id"},
            inplace=True,
        )

        # Save the postal codes with associated energy substations to a file
        path = (
            self.model.input_folder
            / "geom"
            / "assets"
            / "postal_codes_with_energy_substations.geoparquet"
        )
        postal_codes_with_substations.to_parquet(path)

    def assign_critical_facilities_to_postal_codes(
        self, critical_facilities: gpd.GeoDataFrame, postal_codes: gpd.GeoDataFrame
    ) -> None:
        """Assign critical facilities (vulnerable and emergency) to postal codes based on spatial intersection. Every facility gets assigned to the postal code it is located in.

        Args:
            critical_facilities: GeoDataFrame of critical facilities.
            postal_codes: GeoDataFrame of postal codes.
        """
        # Use the centroid of the critical facilities to assign them to postal codes
        critical_facilities_centroid = critical_facilities.copy()
        critical_facilities_centroid["geometry"] = (
            critical_facilities_centroid.geometry.centroid
        )

        # Spatial join to assign postal codes to critical facilities based on their centroid
        critical_facilities_with_postal_codes = gpd.sjoin(
            critical_facilities_centroid[
                ["id", "addr:city", "amenity", "name", "source", "geometry"]
            ],
            postal_codes[["postcode", "geometry"]],
            how="left",
            predicate="within",
        )
        critical_facilities_with_postal_codes.drop(
            columns=["index_right"], inplace=True
        )

        # Rename the id column for clarity and drop fid so it can save to gpkg
        critical_facilities_with_postal_codes = (
            critical_facilities_with_postal_codes.rename(columns={"id": "facility_id"})
        )

        # Save the critical facilities with postal codes to a file
        path = (
            self.model.input_folder
            / "geom"
            / "assets"
            / "critical_facilities_with_postal_codes.geoparquet"
        )
        critical_facilities_with_postal_codes.to_parquet(path)

    def critical_infrastructure_warning_strategy(
        self, date_time: datetime, prob_threshold: float = 0.6
    ) -> None:
        """This function implements an evacuation warning strategy based on critical infrastructure elements, such as energy substations, vulnerable and emergency facilities.

        Args:
            date_time: The forecast date time for which to implement the warning strategy.
            prob_threshold: The probability threshold above which a warning is issued.
        """
        # Get the household points, needed to issue warnings
        households = self.var.household_points.copy()

        # Load substations and critical facilities
        substations = gpd.read_parquet(
            self.model.files["geom"]["assets/energy_substations"]
        )
        critical_facilities = gpd.read_parquet(
            self.model.files["geom"]["assets/critical_facilities"]
        )

        # Load postal codes with associated substations and critical facilities
        path = self.model.input_folder / "geom" / "assets"
        postal_codes_with_substations = gpd.read_parquet(
            path / "postal_codes_with_energy_substations.geoparquet"
        )
        critical_facilities_with_postal_codes = gpd.read_parquet(
            path / "critical_facilities_with_postal_codes.geoparquet"
        )

        ## For energy substations:
        # The strategy id is used in the create_flood_probability_maps function to define the right water level range, so it makes sure you get the right probability map for that specific strategy
        # strategy_id = 2 means wl_range > 30 cm for energy substations
        strategy_id = 2

        # Create flood probability maps associated to the critical hit of energy substations
        # Need to give the number (id) of strategy as argument
        self.create_flood_probability_maps(strategy=strategy_id, date_time=date_time)

        # Get the probability map for the specific day and strategy
        prob_energy_hit_path = Path(
            self.model.output_folder
            / "prob_maps"
            / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
            / f"prob_map_range1_strategy{strategy_id}.zarr"
        )

        # Open the probability map
        prob_map = read_zarr(prob_energy_hit_path)
        affine = prob_map.rio.transform()
        prob_array = np.asarray(prob_map.values)

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
            affected_postcodes_energy = postal_codes_with_substations[
                postal_codes_with_substations["substation_id"].isin(
                    critical_hits_energy["fid"]
                )
            ]["postcode"].unique()

        ## For vulnerable and emergency facilities:
        # strategy_id = 3 means wl_range > 10 cm for vulnerable and emergency facilities (flooded or not)
        strategy_id = 3

        # Create flood probability map for vulnerable and emergency facilities
        self.create_flood_probability_maps(strategy=strategy_id, date_time=date_time)

        # Get the probability map for the specific day and strategy
        prob_critical_facilities_hit_path = Path(
            self.model.output_folder
            / "prob_maps"
            / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
            / f"prob_map_range1_strategy{strategy_id}.zarr"
        )

        # Open the probability map
        prob_map = read_zarr(prob_critical_facilities_hit_path)
        affine = prob_map.rio.transform()
        prob_array = np.asarray(prob_map.values)

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
            affected_postcodes_facilities = critical_facilities_with_postal_codes[
                critical_facilities_with_postal_codes["facility_id"].isin(
                    critical_hits_facilities["id"]
                )
            ]["postcode"].unique()

        # Function to keep track of what triggered the warning for each postal code
        def trigger_label(postcode: str) -> str:
            """Determine the trigger label for a given postcode based on affected postcodes from both strategies.

            Args:
                postcode: The postcode to evaluate.

            Returns:
                A string indicating the trigger type.

            Raises:
                ValueError: If the postcode is not found in either affected postcodes list.
            """
            e = postcode in affected_postcodes_energy
            f = postcode in affected_postcodes_facilities
            if e and f:
                return "energy_and_facilities"
            elif e:
                return "energy"
            elif f:
                return "facilities"
            else:
                raise ValueError(
                    f"Postcode {postcode} not found in either affected postcodes list."
                )

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
                trigger="critical_infrastructure",
            )
            trigger = trigger_label(postcode)

            print(
                f"Evacuation warning issued to postal code {postcode} on {date_time.isoformat()} (trigger: {trigger})"
            )
            warning_log.append(
                {
                    "date_time": date_time.isoformat(),
                    "postcode": postcode,
                    "n_warned_households": n_warned_households,
                    "trigger": trigger,
                }
            )

        # Save warning log
        path = (
            self.model.output_folder
            / "warning_logs"
            / f"warning_log_critical_infrastructure_{date_time.isoformat()}.csv"
        )
        pd.DataFrame(warning_log).to_csv(path, index=False)

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
        """Load buildings, roads, and rail geometries from model files."""
        # Load buildings
        self.buildings = gpd.read_parquet(
            self.model.files["geom"]["assets/open_building_map"]
        )
        self.buildings["object_type"] = (
            "building_unprotected"  # before it was "building_structure"
        )
        self.buildings_centroid = gpd.GeoDataFrame(
            geometry=self.buildings.to_crs(epsg=3857).centroid.to_crs(
                self.buildings.crs
            )
        )
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

        if self.model.config["general"]["forecasts"]["use"]:
            # Load postal codes --
            # TODO: maybe move it to another function? (not really an object)
            self.postal_codes = gpd.read_parquet(
                self.model.files["geom"]["postal_codes"]
            )

    def load_max_damage_values(self) -> None:
        """Load maximum damage values from model files and store them in the model variables."""
        # Load maximum damages
        self.var.max_dam_buildings_structure = float(
            read_params(
                self.model.files["dict"][
                    "damage_parameters/flood/buildings/structure/maximum_damage"
                ]
            )["maximum_damage"]
        )
        self.buildings["maximum_damage_m2"] = self.var.max_dam_buildings_structure

        max_dam_buildings_content = read_params(
            self.model.files["dict"][
                "damage_parameters/flood/buildings/content/maximum_damage"
            ]
        )
        self.var.max_dam_buildings_content = float(
            max_dam_buildings_content["maximum_damage"]
        )
        self.buildings_centroid["maximum_damage"] = self.var.max_dam_buildings_content

        self.var.max_dam_rail = float(
            read_params(
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
            max_dam_road_m[road_type] = read_params(self.model.files["dict"][path])[
                "maximum_damage"
            ]

        self.roads["maximum_damage_m"] = self.roads["object_type"].map(max_dam_road_m)

        self.var.max_dam_forest_m2 = float(
            read_params(
                self.model.files["dict"][
                    "damage_parameters/flood/land_use/forest/maximum_damage"
                ]
            )["maximum_damage"]
        )

        self.var.max_dam_agriculture_m2 = float(
            read_params(
                self.model.files["dict"][
                    "damage_parameters/flood/land_use/agriculture/maximum_damage"
                ]
            )["maximum_damage"]
        )

    def load_damage_curves(self) -> None:
        """Load damage curves from model files and store them in the model variables."""
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

        for road_type, path in road_types:
            df = pd.read_parquet(self.model.files["table"][path])
            df = df.rename(columns={"damage_ratio": road_type})

            road_curves.append(df[[road_type]])

        severity_column: pd.DataFrame = df[["severity"]]

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

        # TODO: Need to adjust the vulnerability curves
        # create another column (curve) in the buildings structure curve for
        # protected buildings with sandbags
        self.buildings_structure_curve["building_with_sandbags"] = (
            self.buildings_structure_curve["building_unprotected"] * 0.85
        )

        # create another column (curve) in the buildings structure curve for
        # protected buildings with elevated possessions -- no effect on structure
        self.buildings_structure_curve["building_elevated_possessions"] = (
            self.buildings_structure_curve["building_unprotected"]
        )

        # create another column (curve) in the buildings structure curve for
        # protected buildings with both sandbags and elevated possessions -- only sandbags have an effect on structure
        self.buildings_structure_curve["building_all_forecast_based"] = (
            self.buildings_structure_curve["building_with_sandbags"]
        )

        # create another column (curve) in the buildings structure curve for flood-proofed buildings
        self.buildings_structure_curve["building_flood_proofed"] = (
            self.buildings_structure_curve["building_unprotected"] * 0.85
        )
        self.buildings_structure_curve.loc[0:1, "building_flood_proofed"] = 0.0

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
        # create another column (curve) in the buildings content curve for flood-proofed buildings
        self.buildings_content_curve["building_flood_proofed"] = (
            self.buildings_content_curve["building_unprotected"] * 0.85
        )

        self.buildings_content_curve.loc[0:1, "building_flood_proofed"] = 0.0

        # TODO: need to adjust the vulnerability curves
        # create another column (curve) in the buildings content curve for
        # protected buildings with sandbags
        self.buildings_content_curve["building_with_sandbags"] = (
            self.buildings_content_curve["building_unprotected"] * 0.85
        )

        # create another column (curve) in the buildings content curve for
        # protected buildings with elevated possessions
        self.buildings_content_curve["building_elevated_possessions"] = (
            self.buildings_content_curve["building_unprotected"] * 0.85
        )

        # create another column (curve) in the buildings content curve for
        # protected buildings with both sandbags and elevated possessions
        self.buildings_content_curve["building_all_forecast_based"] = (
            self.buildings_content_curve["building_unprotected"] * 0.85
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

    def load_wlranges_and_measures(self) -> None:
        """Loads the water level ranges and appropriate measures, and the implementation times for measures."""
        with open(self.model.files["dict"]["measures/implementation_times"], "r") as f:
            self.var.implementation_times = json.load(f)

        with open(self.model.files["dict"]["measures/wl_ranges"], "r") as f:
            wlranges_and_measures = json.load(f)
            # convert the keys (range ids) to integers and store them in a new dictionary
            self.var.wlranges_and_measures = {
                int(key): value for key, value in wlranges_and_measures.items()
            }

    def create_damage_interpolators(self) -> None:
        """This function creates interpolation functions for the damage curves."""
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
        """This function runs the spin-up process for the household agents."""
        self.construct_income_distribution()
        self.assign_household_attributes()
        if self.model.config["agent_settings"]["households"]["warning_response"]:
            self.assign_households_to_postal_codes()

    def assign_damages_to_agents(
        self, agent_df: pd.DataFrame, buildings_with_damages: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """This function assigns the building damages calculated by the vector scanner to the corresponding households.

        Args:
            agent_df: Pandas dataframe that contains the open building map building id assigned to each agent.
            buildings_with_damages: Pandas dataframe constructed by the vector scanner that contains the damages for each building.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing damage arrays for (unprotected buildings, flood-proofed buildings).
        """
        merged = agent_df.merge(
            buildings_with_damages.rename(columns={"id": "building_id_of_household"}),
            on="building_id_of_household",
            how="left",
        ).fillna(0)
        damages_do_not_adapt = merged["damages"].to_numpy()
        damages_adapt = merged["damages_flood_proofed"].to_numpy()

        return damages_do_not_adapt, damages_adapt

    def calculate_building_flood_damages(
        self, verbose: bool = True, export_building_damages: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """This function calculates the flood damages for the households in the model.

        It iterates over the return periods and calculates the damages for each household
        based on the flood maps and the building footprints.

        Args:
            verbose: Verbosity flag.
            export_building_damages: Whether to export the building damages to parquet files.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the damage arrays for unprotected and protected buildings.
        """
        damages_do_not_adapt = np.zeros((self.return_periods.size, self.n), np.float32)
        damages_adapt = np.zeros((self.return_periods.size, self.n), np.float32)

        buildings: gpd.GeoDataFrame = self.buildings.copy().to_crs(
            self.flood_maps[self.return_periods[0]].rio.crs
        )

        # create a pandas data array for assigning damage to the agents:
        agent_df = pd.DataFrame(
            {"building_id_of_household": self.var.building_id_of_household}
        )

        # subset building to those exposed to flooding
        buildings = buildings[buildings["flooded"]]

        # only calculate damages for buildings with more than 0 occupant
        buildings = buildings[buildings["n_occupants"] > 0]

        for i, return_period in enumerate(self.return_periods):
            flood_map: xr.DataArray = self.flood_maps[return_period]

            building_multicurve = buildings.copy()
            multi_curves = {
                "damages_structure": self.buildings_structure_curve[
                    "building_unprotected"
                ],
                "damages_content": self.buildings_content_curve["building_unprotected"],
                "damages_structure_flood_proofed": self.buildings_structure_curve[
                    "building_flood_proofed"
                ],
                "damages_content_flood_proofed": self.buildings_content_curve[
                    "building_flood_proofed"
                ],
            }
            damage_buildings: pd.DataFrame = VectorScannerMultiCurves(
                features=building_multicurve.rename(
                    columns={
                        "COST_STRUCTURAL_USD_SQM": "maximum_damage_structure",
                        "COST_CONTENTS_USD_SQM": "maximum_damage_content",
                    }
                ),
                hazard=flood_map,
                multi_curves=multi_curves,
            )

            # sum structure and content damages
            damage_buildings["damages"] = (
                damage_buildings["damages_structure"]
                + damage_buildings["damages_content"]
            )
            damage_buildings["damages_flood_proofed"] = (
                damage_buildings["damages_structure_flood_proofed"]
                + damage_buildings["damages_content_flood_proofed"]
            )
            # concatenate damages to building_multicurve
            building_multicurve = pd.concat(
                [building_multicurve, damage_buildings], axis=1
            )

            if export_building_damages:
                fn_for_export = self.model.output_folder / "building_damages"
                fn_for_export.mkdir(parents=True, exist_ok=True)
                building_multicurve.to_parquet(
                    self.model.output_folder
                    / "building_damages"
                    / f"building_damages_rp{return_period}_{self.model.current_time.year}.parquet"
                )
            building_multicurve = building_multicurve[
                ["id", "damages", "damages_flood_proofed"]
            ]
            # merged["damage"] is aligned with agents
            damages_do_not_adapt[i], damages_adapt[i] = self.assign_damages_to_agents(
                agent_df,
                building_multicurve,
            )
            if verbose:
                print(
                    f"Damages rp{return_period}: {round(damages_do_not_adapt[i].sum() / 1e6)} million"
                )
                print(
                    f"Damages adapt rp{return_period}: {round(damages_adapt[i].sum() / 1e6)} million"
                )
        return damages_do_not_adapt, damages_adapt

    def flood(self, flood_depth: xr.DataArray) -> float:
        """This function computes the damages for the assets and land use types in the model.

        Args:
            flood_depth: The flood map containing water levels for the flood event [m].

        Returns:
            The total flood damages for the event for all assets and land use types.

        """
        flood_depth: xr.DataArray = flood_depth.compute()

        buildings: gpd.GeoDataFrame = self.buildings.copy().to_crs(flood_depth.rio.crs)

        household_points: gpd.GeoDataFrame = self.var.household_points.copy().to_crs(
            flood_depth.rio.crs
        )

        if self.model.config["agent_settings"]["households"]["warning_response"]:
            # make sure household points and actions taken have the same length
            assert len(household_points) == self.var.actions_taken.shape[0]

            # add columns for protective actions
            household_points["sandbags"] = False
            household_points["elevated_possessions"] = False

            # mark households that took protective actions
            household_points.loc[
                np.asarray(self.var.actions_taken)[:, 0] == 1, "elevated_possessions"
            ] = True
            household_points.loc[
                np.asarray(self.var.actions_taken)[:, 1] == 1, "sandbags"
            ] = True

            # spatial join to get household attributes to buildings
            buildings: gpd.GeoDataFrame = gpd.sjoin_nearest(
                buildings, household_points, how="left", exclusive=True
            )

            # Assign object types for buildings based on protective measures taken
            buildings["object_type"] = "building_unprotected"  # reset
            buildings.loc[buildings["elevated_possessions"], "object_type"] = (
                "building_elevated_possessions"
            )
            buildings.loc[buildings["sandbags"], "object_type"] = (
                "building_with_sandbags"
            )
            buildings.loc[
                buildings["elevated_possessions"] & buildings["sandbags"], "object_type"
            ] = "building_all_forecast_based"
            # TODO: need to move the update of the actions takens by households to outside the flood function

            # Save the buildings with actions taken
            buildings.to_parquet(
                self.model.output_folder
                / "action_maps"
                / "buildings_with_protective_measures.geoparquet"
            )

            # Assign object types for buildings centroid based on protective measures taken
            buildings_centroid = household_points.to_crs(flood_depth.rio.crs)
            buildings_centroid["object_type"] = np.select(
                [
                    (
                        buildings_centroid["elevated_possessions"]
                        & buildings_centroid["sandbags"]
                    ),
                    buildings_centroid["elevated_possessions"],
                    buildings_centroid["sandbags"],
                ],
                [
                    "building_all_forecast_based",
                    "building_elevated_possessions",
                    "building_with_sandbags",
                ],
                default="building_unprotected",
            )
            buildings_centroid["maximum_damage"] = self.var.max_dam_buildings_content

        if self.config["adapt"]:
            household_points["building_id"] = (
                self.var.building_id_of_household
            )  # first assign building id to household points gdf
            household_points = household_points.merge(
                buildings[["id", "flood_proofed"]],
                left_on="building_id",
                right_on="id",
                how="left",
            )  # now merge to get flood proofed status

            buildings_centroid = household_points.to_crs(flood_depth.rio.crs)

            buildings_centroid["maximum_damage"] = self.var.max_dam_buildings_content

            buildings["object_type"] = np.where(
                buildings["flood_proofed"],
                "building_flood_proofed",
                "building_unprotected",
            )

            buildings_centroid["object_type"] = np.where(
                buildings_centroid["flood_proofed"],
                "building_protected",
                "building_unprotected",
            )

        else:
            household_points["protect_building"] = False

            buildings: gpd.GeoDataFrame = gpd.sjoin_nearest(
                buildings, household_points, how="left", exclusive=True
            )

            buildings["object_type"] = "building_unprotected"

            # Right now there is no condition to make the households protect their buildings outside of the warning response
            buildings.loc[buildings["protect_building"], "object_type"] = (
                "building_protected"
            )

            buildings_centroid = household_points.to_crs(flood_depth.rio.crs)
            buildings_centroid["object_type"] = buildings_centroid[
                "protect_building"
            ].apply(lambda x: "building_protected" if x else "building_unprotected")
            buildings_centroid["maximum_damage"] = self.var.max_dam_buildings_content

        # Create the folder to save damage maps if it doesn't exist
        damage_folder: Path = self.model.output_folder / "damage_maps"
        damage_folder.mkdir(parents=True, exist_ok=True)

        damages_buildings_content = VectorScanner(
            features=buildings_centroid,
            hazard=flood_depth,
            vulnerability_curves=self.buildings_content_curve,
        )

        total_damages_content = damages_buildings_content.sum()

        # save it to a gpkg file
        gdf_content = buildings_centroid.copy()
        gdf_content["damage"] = damages_buildings_content
        category_name: str = "buildings_content"
        filename: str = f"damage_map_{category_name}.gpkg"
        gdf_content.to_file(damage_folder / filename, driver="GPKG")

        print(f"damages to building content are: {total_damages_content}")

        # Compute damages for buildings structure
        damages_buildings_structure: pd.Series = VectorScanner(
            features=buildings.rename(columns={"maximum_damage_m2": "maximum_damage"}),  # ty:ignore[invalid-argument-type]
            hazard=flood_depth,
            vulnerability_curves=self.buildings_structure_curve,
        )

        total_damage_structure = damages_buildings_structure.sum()

        print(f"damages to building structure are: {total_damage_structure}")

        # save it to a gpkg file
        gdf_structure = buildings.copy()
        gdf_structure["damage"] = damages_buildings_structure
        category_name: str = "buildings_structure"
        filename: str = f"damage_map_{category_name}.gpkg"
        gdf_structure.to_file(damage_folder / filename, driver="GPKG")

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
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
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
            ) * self.config["adjust_demand_factor"]
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
        """Advance the households by one time step."""
        if self.config["adapt"]:
            if self.config["adapt_to_actual_floods"]:
                self.flood_events: list[dict[str, datetime]] = self.model.config[
                    "hazards"
                ]["floods"]["events"]
                current_time: datetime = self.model.current_time

                # Check if a flood has recently happened by comparing the current time to the end of flood + 14 days
                # (assumption that people wait around 2 weeks before adapting)
                is_flood_triggered: bool = any(
                    current_time
                    == (
                        e["end_time"] + timedelta(days=14)
                        if isinstance(e["end_time"], datetime)
                        else datetime.strptime(e["end_time"], "%Y-%m-%d %H:%M:%S")
                        + timedelta(days=14)
                    )
                    for e in self.flood_events
                )

                # Households adapt on first day of the year or 2 weeks after flood happened
                if (
                    self.model.current_time.month == 1
                    and self.model.current_time.day == 1
                ) or is_flood_triggered:
                    if "flooded" not in self.buildings.columns:
                        self.update_building_attributes()
                    print(f"Thinking about adapting at {current_time}...")
                    self.decide_household_strategy()

                end_time: datetime = datetime.combine(
                    self.model.config["general"]["end_time"], datetime.min.time()
                )

                if self.model.current_time == end_time:
                    print("end of sim reached")
                    df_all: pd.DataFrame = pd.concat(
                        self.flood_risk_perceptions, ignore_index=True
                    )

                    gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
                        df_all,
                        geometry=gpd.points_from_xy(df_all.x, df_all.y),
                        crs=self.var.household_points.crs,
                    )

                    out_path: Path = (
                        Path(self.model.output_folder) / "risk_perceptions.gpkg"
                    )
                    print(f"saved risk perception here: {out_path}")
                    gdf.to_file(out_path, layer="perceptions", driver="GPKG")

                    df_stats: pd.DataFrame = pd.DataFrame(
                        self.flood_risk_perceptions_statistics
                    )
                    out_path: Path = (
                        Path(self.model.output_folder) / "risk_perception_stats.csv"
                    )
                    df_stats.to_csv(out_path, index=False)
                    print(f"Saved risk perception statistics to {out_path}")

            else:  # Household don't respond to actual floods, but make decision on the first day of the year. Decisions are based on random floods
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
    def n(self) -> int:
        """Number of households in the agent class.

        Returns:
            Number of households.
        """
        return self.var.locations.shape[0]

    @property
    def population(self) -> int:
        """Total total number of people in all households.

        Returns:
            Total population.
        """
        return self.var.sizes.data.sum()
