"""This module contains the Households agent class for simulating household behavior in the GEB model."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import xarray as xr
from rasterio.features import rasterize

from geb.geb_types import ArrayFloat32, TwoDArrayFloat32
from geb.workflows.io import read_geom
from geb.workflows.raster import sample_from_map

from ..store import Bucket, DynamicArray
from ..workflows.io import read_array, read_table, read_zarr, write_zarr
from .decision_module import DecisionModule
from .general import AgentBaseClass
from .modules.flood_risk import FloodRiskModule
from .modules.wind_risk import WindRiskModule
from .workflows.helpers import from_landuse_raster_to_polygon

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel


class HouseholdVariables(Bucket):
    """Variables for the Households agent."""

    household_points: gpd.GeoDataFrame
    actions_taken: DynamicArray
    possible_measures: list[str]
    insurance_scheme: str
    possible_warning_triggers: list[str]
    municipal_water_demand_per_capita_m3_baseline: ArrayFloat32
    water_demand_per_household_m3: ArrayFloat32
    income: DynamicArray
    building_id_of_household: DynamicArray
    household_building_area: DynamicArray
    wealth: DynamicArray
    property_value: DynamicArray
    locations: DynamicArray
    years_since_last_flood: DynamicArray
    years_since_last_windstorm: DynamicArray
    risk_perception: DynamicArray
    risk_perception_windstorm: DynamicArray
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
    water_demand_per_household_year: int
    water_demand_per_household_m3_gridded: ArrayFloat32


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

        if self.config["adapt"]:
            self.load_objects()
            self.flood_risk_module = FloodRiskModule(model=self.model, households=self)
            self.flood_risk_perceptions = []  # Store the flood risk perceptions in here
            self.flood_risk_perceptions_statistics = []  # Store some statistics on flood risk perceptions here
            if self.model.config["agent_settings"]["households"]["warning_response"]:
                self.load_critical_infrastructure()  # ideally this should be done in the setup_assets when building the model
                self.load_wlranges_and_measures()
            if self.model.config["agent_settings"]["households"]["wind_adaptation"]:
                self.wind_risk_module = WindRiskModule(model=self.model, households=self)
                self.wind_risk_perceptions = []  # Store the windstorm risk perceptions in here
                self.wind_risk_perceptions_statistics = []  # Store some statistics on windstorm risk perceptions here
        

        

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Return the name of the agent type."""
        return "agents.households"

    def load_objects(self) -> None:
        """Load buildings, roads, and rail geometries from model files."""
        # Load buildings
        columns_to_load = [
            "id",
            "floorspace",
            "occupancy",
            "height",
            "geometry",
            "x",
            "y",
            "NAME_1",
            "TOTAL_REPL_COST_USD_SQM",
            "COST_STRUCTURAL_USD_SQM",
            "COST_NONSTRUCTURAL_USD_SQM",
            "COST_CONTENTS_USD_SQM",
            #"TOTAL_AREA_SQM",
        ]
        self.buildings = read_table(
            self.model.files["geom"]["assets/open_building_map"],
            columns=columns_to_load,
        )

        self.buildings["object_type"] = (
            "building_unprotected"  # before it was "building_structure"
        )

        # Load roads
        self.roads = read_geom(self.model.files["geom"]["assets/roads"]).rename(
            columns={"highway": "object_type"}
        )

        # Load rail
        self.rail = read_geom(self.model.files["geom"]["assets/rails"])
        self.rail["object_type"] = "rail"

        if self.model.config["general"]["forecasts"]["use"]:
            # Load postal codes --
            # TODO: maybe move it to another function? (not really an object)
            self.postal_codes = read_geom(self.model.files["geom"]["postal_codes"])

    # not in current main version
    # def load_flood_maps(self) -> None:
    #     self.return_periods = np.array(
    #         self.model.config["hazards"]["flood"]["return_periods"]
    #     )

    #     flood_maps = {}
    #     for return_period in self.return_periods:
    #         file_path = (
    #             self.model.output_folder / "flood_maps" / f"return_level_rp{return_period}.zarr"
    #         )
    #         flood_maps[return_period] = read_zarr(file_path)
    #     self.flood_maps = flood_maps
    # Modification
    # def load_wind_maps(self):
    #     """Load wind maps in this case for the 50 and 100 return periods.This maps are created separately and copied to the "wind_maps" folder.Creating the folder and copying the wind maps should be done manually."""
    #     self.windstorm_return_periods = np.array(
    #         self.model.config["hazards"]["windstorm"]["return_periods"]
    #     )

    #     debug_maps = bool(
    #         self.model.config.get("hazards", {})
    #         .get("windstorm", {})
    #         .get("debug_maps", False)
    #     )

    #     windstorm_maps = {}
    #     windstorm_path = self.model.output_folder / "wind_maps"
    #     for return_period in self.windstorm_return_periods:
    #         file_path = (
    #             windstorm_path / f"return_level_rp{return_period}.tif"
    #         )  # adjust to file name
    #         windstorm_map = xr.open_dataarray(file_path, engine="rasterio")

    #         if debug_maps:
    #             try:
    #                 crs = windstorm_map.rio.crs
    #             except Exception:
    #                 crs = None
    #             print(
    #                 "Loaded windstorm map: "
    #                 f"rp={int(return_period)}, path={file_path}, "
    #                 f"shape={tuple(windstorm_map.shape)}, crs={crs}"
    #             )

    #         windstorm_maps[return_period] = windstorm_map

    #     self.windstorm_maps = windstorm_maps
    #     # print("Wind maps loaded for return periods:", self.windstorm_return_periods)

    

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
        self.model.logger.warning(
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

    def update_building_attributes(self, drop_not_flooded: bool = False) -> None:
        """Update building attributes based on household data.

        Args:
            drop_not_flooded: If True, drop buildings that are not flooded. This can save memory and speed up the model.
        """
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

        buildings_centroid = gpd.GeoDataFrame(
            self.buildings,
            geometry=gpd.points_from_xy(self.buildings["x"], self.buildings["y"]),
            crs="EPSG:4326",
        )
        buildings_centroid = buildings_centroid.to_crs(
            flood_map.rio.crs
        )  # Reproject building centroids to flood map CRS

        # # convert flood map to polygons
        flood_map = flood_map > 0  # convert to boolean mask
        flood_map_polygons = from_landuse_raster_to_polygon(
            flood_map.values,
            flood_map.rio.transform(recalc=True),
            flood_map.rio.crs,
        )

        flood_map_polygons_union: gpd.GeoDataFrame = gpd.GeoDataFrame(
            [flood_map_polygons.union_all()],
            columns=["geometry"],
            crs=buildings_centroid.crs,
        )

        # # Create a mask for buildings that overlap with the flood map
        flooded_buildings = gpd.sjoin(
            buildings_centroid,
            flood_map_polygons_union,
            predicate="intersects",
            how="left",
        )

        # Flooded if match exists
        self.buildings["flooded"] = flooded_buildings["index_right"].notna()

        # drop buildings which are not flooded
        if drop_not_flooded:
            self.buildings = self.buildings[self.buildings["flooded"]]

    def update_building_adaptation_status(
        self, household_adapting: np.ndarray, adaptation_type: str
    ) -> None:
        """Update buildings based on categorical adaptation array (1=dryfloodproofing, 2=shutters)."""
        col_name = f"adaptation_{adaptation_type}"

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

        # initiate array for adaptation status [0=not adapted, 1=dryfloodproofing implemented, 2=shutters installed]
        self.var.adapted = DynamicArray(np.zeros(self.n, np.int32), max_n=self.max_n)
        self.var.adapted_shutters = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
        )
        self.var.adapted_insurance = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
        )

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

        # insurance scheme
        insurance_cfg = self.model.config["agent_settings"]["households"]["insurance"]["scheme"]
        valid_schemes = {"catnat", "private", "reform", "no_insurance"}

        if insurance_cfg not in valid_schemes:
            raise ValueError(
                f"Unkwown insurance scheme: {insurance_cfg}."
                f"Must be one of {valid_schemes}"
            )
        
        self.var.insurance_scheme = insurance_cfg

        # ############### FIX FIX #################
        eu_cfg = self.model.config["agent_settings"]["households"]["expected_utility"]

        flood_rp_cfg = eu_cfg["flood_risk_calculations"]["risk_perception"]
        wind_rp_cfg = eu_cfg.get("windstorm_risk_calculations", {}).get(
            "risk_perception", flood_rp_cfg
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

        self.var.risk_perc_w_min = self.model.config["agent_settings"]["households"][
            "expected_utility"
        ]["windstorm_risk_calculations"]["risk_perception"]["min"]
        self.var.risk_perc_w_max = self.model.config["agent_settings"]["households"][
            "expected_utility"
        ]["windstorm_risk_calculations"]["risk_perception"]["max"]
        self.var.risk_w_decr = self.model.config["agent_settings"]["households"][
            "expected_utility"
        ]["windstorm_risk_calculations"]["risk_perception"]["coef"]
        self.var.risk_w_base = self.model.config["agent_settings"]["households"][
            "expected_utility"
        ]["windstorm_risk_calculations"]["risk_perception"]["base"]

        risk_perception = np.full(self.n, self.var.risk_perc_min)
        self.var.risk_perception = DynamicArray(risk_perception, max_n=self.max_n)

        risk_perception_w = np.full(self.n, self.var.risk_perc_w_min, dtype=np.float32)
        self.var.risk_perception_w = DynamicArray(risk_perception_w, max_n=self.max_n)

        self.var.risk_perception_windstorm = self.var.risk_perception_w

        # initiate array with risk aversion [fixed for now]
        self.var.risk_aversion = DynamicArray(np.full(self.n, 1), max_n=self.max_n)

        # initiate array with time adapted
        self.var.time_adapted = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
        )
        # initiate array with time adapted for shutters
        self.var.time_adapted_shutters = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
        )

        # initiate array with time with insurance
        self.var.time_with_insurance = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
        )

        # initiate array with time since last flood
        self.var.years_since_last_flood = DynamicArray(
            np.full(self.n, 25, np.int32), max_n=self.max_n
        )

        # initiate array with time since last flood
        self.var.years_since_last_windstorm = DynamicArray(
            np.full(self.n, 25, np.int32), max_n=self.max_n
        )

        # assign income and wealth attributes
        self.assign_household_wealth_and_income()

        # initiate array with property values (used as max damage) [dummy data for now, could use Huizinga combined with building footprint to calculate better values]
        self.var.property_value = DynamicArray(
            (self.var.wealth.data * 0.8).astype(np.int64), max_n=self.max_n
        )

        ### CARO BASED ON DAMAGE and BUILDING FOOTPRINT

        # # load household points (only in use for damagescanner, could be removed)
        # household_points = gpd.GeoDataFrame(
        #     geometry=gpd.points_from_xy(
        #         self.var.locations.data[:, 0], self.var.locations.data[:, 1]
        #     ),
        #     crs="EPSG:4326",
        # )
        # self.var.household_points = household_points

        # buildings = self.buildings.copy()

        # household_points_copy = self.var.household_points.copy()
        # household_points_copy["building_id"] = self.var.building_id_of_household

        # household_points_copy = household_points_copy.merge(
        #     buildings[["id", "geom", "occupancy"]],
        #     left_on="building_id",
        #     right_on="id",
        #     how="left",
        # )

        # projected_crs = buildings.estimate_utm_crs()
        # household_points_copy = household_points_copy.set_geometry("geometry_y")
        # household_points_copy = household_points_copy.to_crs(projected_crs)

        # household_points_copy["building_area"] = household_points_copy[
        #     "geometry_y"
        # ].area

        # self.var.household_building_area = DynamicArray(
        #  household_points_copy["building_area"].values.astype(np.float32),
        #     max_n=self.max_n
        # )

        # self.var.property_value = DynamicArray(
        #     (
        #         self.var.household_building_area.data
        #         * self.var.max_dam_buildings_structure
        #     ).astype(np.int64),
        #     max_n=self.max_n,
        # )

        # initiate array with RANDOM annual adaptation costs [dummy data for now, values are available in literature]
        # adaptation_costs = (
        #     np.maximum(self.var.property_value.data * 0.05, 10_800)
        # ).astype(np.int64)
        # self.var.adaptation_costs = DynamicArray(adaptation_costs, max_n=self.max_n)

        # loan_duration_w = self.model.config["agent_settings"]["households"][
        #     "loan_duration_years_w"
        # ]
        # shutters_total = np.maximum(self.var.property_value.data * 0.02, 2_000).astype(
        #     np.float32
        # )
        # # shutters_annual = shutters_total  # / np.float32(loan_duration_w)

        # self.var.adaptation_costs_shutters = DynamicArray(
        #     shutters_total, max_n=self.max_n
        # )


        #NEW CODE WITH RECONSTRUCTION VALUES
        self.var.household_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                self.var.locations.data[:, 0], self.var.locations.data[:, 1]
            ),
            crs="EPSG:4326",        
        )

        hh_bld_ids = pd.Series(self.var.building_id_of_household.data).astype("Int64")
        unique_bld_ids = hh_bld_ids.dropna().astype(int).unique()

        if unique_bld_ids.size > 0:
            bld = read_geom(
                self.model.files["geom"]["assets/open_building_map"],
                filters=[("id", "in", unique_bld_ids)],
                columns=["id","geometry","COST_STRUCTURAL_USD_SQM"]
            )
            projected_crs = bld.estimate_utm_crs()
            bld_m = bld.to_crs(projected_crs)

            bld["footprint_m2"] = bld_m.area.astype(np.float32)
            bld["structural_value_usd"] = (
                bld["footprint_m2"] * bld["COST_STRUCTURAL_USD_SQM"]   
            ).astype(np.float32)

            footprint_by_id = bld.set_index("id")["footprint_m2"]
            value_by_id = bld.set_index("id")["structural_value_usd"]

            hh_per_bld = hh_bld_ids.value_counts(dropna=True).astype(np.float32)
            value_per_hh_by_id = value_by_id.divide(hh_per_bld, fill_value=0.0)

            hh_footprint_m2 = (
                footprint_by_id.reindex(hh_bld_ids).fillna(0.0).to_numpy(np.float32)
            )
            hh_structural_value_usd = (
                value_per_hh_by_id.reindex(hh_bld_ids).fillna(0.0).to_numpy(np.float32)
            )
        else:
            hh_footprint_m2 = np.zeros(self.n, dtype=np.float32)
            hh_structural_value_usd = np.zeros(self.n, dtype=np.float32)

        # buildings = self.buildings.copy()

        # buildings = gpd.GeoDataFrame(
        #     buildings,
        #     geometry=gpd.points_from_xy(buildings["x"], buildings["y"]),
        #     crs="EPSG:4326",
        # )
        # adaptation_costs = (
        #     np.maximum(self.var.max_dam_buildings_structure *  self.var.household_building_area.data * 0.05, 10_800)
        # ).astype(np.int64)
        # self.var.adaptation_costs = DynamicArray(adaptation_costs, max_n=self.max_n)

        # loan_duration_w = self.model.config["agent_settings"]["households"]["loan_duration_years_w"]

        # shutters_cost = np.maximum(self.var.max_dam_buildings_structure * self.var.household_building_area.data * 0.02, 2_000).astype(np.float32)
        # self.var.adaptation_costs_shutters = DynamicArray(
        #     shutters_cost, max_n=self.max_n
        # )

        self.var.household_building_area = DynamicArray(hh_footprint_m2, max_n=self.max_n)

        adaptation_costs = np.maximum(hh_structural_value_usd * 0.05, 10_800).astype(np.int64)
        self.var.adaptation_costs = DynamicArray(adaptation_costs, max_n=self.max_n)

        shutters_cost = np.maximum(hh_structural_value_usd * 0.02, 2_000).astype(np.float32)
        self.var.adaptation_costs_shutters = DynamicArray(shutters_cost, max_n=self.max_n)

        # initiate array with amenity value [dummy data for now, use hedonic pricing studies to calculate actual values]
        amenity_premiums = np.random.uniform(0, 0.2, self.n)
        self.var.amenity_value = DynamicArray(
            amenity_premiums * self.var.wealth, max_n=self.max_n
        )

        self.model.logger.info(
            f"Household attributes assigned for {self.n} households with {self.population} people."
        )

    def assign_households_to_postal_codes(self) -> None:
        """This function associates the household points with their postal codes to get the correct geometry for the warning function."""
        households = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                self.var.locations.data[:, 0], self.var.locations.data[:, 1]
            ),
            crs="EPSG:4326",
        )

        # Associate households with their postal codes to use it later in the warning function
        postal_codes: gpd.GeoDataFrame = read_geom(
            self.model.files["geom"]["postal_codes"]
        )
        postal_codes["postcode"] = postal_codes["postcode"].astype(str)
        households.to_crs(
            postal_codes.crs, inplace=True
        )  # Change to the same CRS as the postal codes

        households_with_postal_codes = gpd.sjoin(
            households,
            postal_codes[["postcode", "geometry"]],
            how="left",
            predicate="intersects",
        )

        # Drop columns that are not needed
        households_with_postal_codes.drop(columns=["index_right"], inplace=True)

        households_with_postal_codes.to_parquet(
            self.model.output_folder / "household_points_w_postal_codes.geoparquet"
        )

        self.var.households_with_postal_codes = households_with_postal_codes

        print(
            f"{len(households_with_postal_codes[households_with_postal_codes['postcode'].notnull()])} households assigned to {households_with_postal_codes['postcode'].nunique()} postal codes."
        )

    def return_period_flood(self) -> np.ndarray:
        """Simulate a flood event based on return periods and determine which households are flooded.

        Returns:
            Array of indices of flooded households.
        """
        # draw a single random number
        p_random = np.random.random()
        # Work with a locally sorted copy of return periods to ensure correct event selection
        return_periods_arr = np.asarray(self.return_periods, dtype=float)
        sort_idx = np.argsort(return_periods_arr)  # ascending order
        sorted_return_periods = return_periods_arr[sort_idx]
        probabilities = 1.0 / sorted_return_periods

        if p_random >= probabilities.max():
            return np.array([], dtype=int)

        # find the event corresponding to the random draw
        event_idx = np.searchsorted(probabilities[::-1], p_random)
        event_idx = len(probabilities) - 1 - event_idx
        event = sorted_return_periods[event_idx]
        self.model.logger.info(
            "Return period flood event: %s years (p=%.4f, random draw=%.4f)",
            event,
            probabilities[event_idx],
            p_random,
        )

        # get the flood map for this event
        flood_map: xr.DataArray = self.flood_maps[event]

        # cache household coordinates in flood_map CRS (Nx2 numpy array)
        if not hasattr(self, "_household_xy"):
            import pyproj

            x, y = np.array(self.buildings.x), np.array(self.buildings.y)
            transformer = pyproj.Transformer.from_crs(
                "EPSG:4326", flood_map.rio.crs, always_xy=True
            )
            self._building_xy = np.array(transformer.transform(x, y)).T

        # sample flood map using clipped coordinates
        sampled_values = sample_from_map(
            array=flood_map.values,
            coords=self._building_xy,
            gt=flood_map.rio.transform(recalc=True).to_gdal(),
            out_of_bounds_value=np.nan,
        )
        # Use the same minimum flood depth threshold (0.05 m) as elsewhere in the model
        minimum_flood_depth_m = 0.05
        # np.where will return indices of flooded households relative to the original household array
        flooded_building_indices = np.where(sampled_values > minimum_flood_depth_m)[0]

        # get building IDs of flooded buildings
        flooded_building_ids = self.buildings.loc[
            flooded_building_indices, "id"
        ].values.astype(int)

        # get indices of households located in flooded buildings
        flooded_household_indices = np.where(
            np.isin(self.var.building_id_of_household.data, flooded_building_ids)
        )[0]

        return flooded_household_indices

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
            flooded_household_indices = self.return_period_flood()
            self.var.years_since_last_flood.data[flooded_household_indices] = 0

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
                "time": [self.model.current_time] * self.var.locations.data.shape[0],
                "x": self.var.locations.data[:, 0],
                "y": self.var.locations.data[:, 1],
                "risk_perception": self.var.risk_perception.data,
                "years_since_last_flood": self.var.years_since_last_flood.data,
            }
        )
        self.flood_risk_perceptions.append(df)

    def return_period_windstorm(self) -> np.ndarray:
        """Simulate a windstorm event based on return periods and determine which households are affected.

        Returns:
            Array of indices of affected households.
        """
        #draw a single random number
        p_random = np.random.random()
        # Work with a locally sorted copy of return periods to ensure correct event selection
        return_periods_arr = np.asarray(self.windstorm_return_periods, dtype=float)
        sort_idx = np.argsort(return_periods_arr)  # ascending order
        sorted_return_periods = return_periods_arr[sort_idx]
        probabilities = 1.0 / sorted_return_periods

        if p_random >= probabilities.max():
            return np.array([], dtype=int)
        
        # find the event corresponding to the random draw
        event_idx = np.searchsorted(probabilities[::-1], p_random)
        event_idx = len(probabilities) - 1 - event_idx
        event = sorted_return_periods[event_idx]
        self.model.logger.info(
            "Return period windstorm event: %s years (p=%.4f, random draw=%.4f)",
            event,
            probabilities[event_idx],
            p_random,
        )

        windstorm_map: xr.DataArray = self.windstorm_maps[event]

        # cache household coordinates in windstorm_map CRS (Nx2 numpy array)
        if not hasattr(self, "_household_xy_wind"):
            import pyproj

            x, y = np.array(self.buildings.x), np.array(self.buildings.y)
            transformer = pyproj.Transformer.from_crs(
                "EPSG:4326", windstorm_map.rio.crs, always_xy=True
            )
            self._building_xy_wind = np.array(transformer.transform(x, y)).T

        # sample windstorm map using clipped coordinates
        sampled_values = sample_from_map(
            array=windstorm_map.values,
            coords=self._building_xy_wind,
            gt=windstorm_map.rio.transform(recalc=True).to_gdal(),
            out_of_bounds_value=np.nan,
        )

        # Use a wind speed threshold to determine affected households (e.g., >30 m/s)
        minimum_wind_speed_ms = 30.0
        windstorm_building_indices = np.where(sampled_values > minimum_wind_speed_ms)[0]
        #get building IDs of affected buildings
        windstorm_building_ids = self.buildings.loc[
            windstorm_building_indices, "id"
        ].values.astype(int)

        #get indices of households located in affected buildings
        windstorm_household_indices = np.where(
            np.isin(self.var.building_id_of_household.data, windstorm_building_ids)
        )[0]

        return windstorm_household_indices

    def update_windstorm_risk_perceptions(self) -> None:
        """Update the risk perceptions of households based on the latest flood data."""
        self.var.years_since_last_windstorm.data += 1

        # wind_cfg = self.model.config.get("hazards", {}).get("windstorm", {})
        # adapt_to_actual_windstorms = bool(wind_cfg.get("adapt_to_actual_windstorms", True))

        # if adapt_to_actual_windstorms:
        if self.config["adapt_to_actual_windstorms"]:  # NEW
            for event in self.windstorm_events:  # NEW
                end: datetime = event["end_time"]  # NEW

                if self.model.current_time == end + timedelta(days=14):
                    windstorm_map_name: str = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}.tif"
                    windstorm_map_path: Path = (
                        self.model.output_folder / "wind_maps" / windstorm_map_name
                    )

                    windstorm_map: xr.DataArray = xr.open_dataarray(
                        windstorm_map_path, engine="rasterio"
                    ).squeeze()

                    buildings = self.buildings.copy()
                    buildings_proj = buildings.to_crs(windstorm_map.rio.crs)

                    xs = buildings_proj.geometry.centroid.x.values
                    ys = buildings_proj.geometry.centroid.y.values

                    speed = windstorm_map.interp(
                        x=("points", xs), y=("points", ys)
                    ).values
                    speed = np.nan_to_num(speed, nan=0.0)

                    buildings_proj["windstorm_hit"] = speed > 30.0

                    windstorm_hit_building_ids = set(
                        buildings_proj.loc[buildings_proj["windstorm_hit"], "id"]
                    )

                    hit_households = np.isin(
                        self.var.building_id_of_household,
                        list(windstorm_hit_building_ids),
                    )

                    self.var.years_since_last_windstorm.data[hit_households] = 0

                    n_buildings = len(buildings_proj)
                    n_hit_buildings = buildings_proj["windstorm_hit"].sum()
                    n_hit_households = hit_households.sum()

                    print(
                        f"Windstorm hit buildings: {n_hit_buildings}/{n_buildings}, "
                        f"Hit households: {n_hit_households}"
                    )
        else:
            windstorm_household_indices = self.return_period_windstorm()
            self.var.years_since_last_windstorm.data[windstorm_household_indices] = 0

        self.var.risk_perception_windstorm.data = (
            self.var.risk_perc_w_max
            * 1.6 ** (self.var.risk_w_decr * self.var.years_since_last_windstorm.data)
            + self.var.risk_perc_w_min
        )

        stats = {
            "time": self.model.current_time,
            "min_risk": np.min(self.var.risk_perception_windstorm.data),
            "max_risk": np.max(self.var.risk_perception_windstorm.data),
            "mean_risk": np.mean(self.var.risk_perception_windstorm.data),
        }

        self.wind_risk_perceptions_statistics.append(stats)

        df = pd.DataFrame(
            {
                "time": [self.model.current_time] * len(self.var.household_points),
                "x": self.var.household_points.geometry.x,
                "y": self.var.household_points.geometry.y,
                "risk_perception_windstorm": self.var.risk_perception_windstorm.data,
                "years_since_last_windstorm": self.var.years_since_last_windstorm.data,
            }
        )

        self.wind_risk_perceptions.append(df)


    def load_ensemble_flood_maps(self, date_time: datetime) -> xr.DataArray:
        """Loads the flood maps for all ensemble members for a specific forecast date time.

        Args:
            date_time: The forecast date time for which to load the flood maps.
        Returns:
            A dataarray containing the flood maps from all ensemble members stacked along a new "member" dimension for the specified forecast time.
        """
        # Get number of members
        # open the flood maps folder to see the number of members
        flood_forecast_folder = (
            self.model.output_folder
            / "flood_maps"
            / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
        )
        n_ensemble_members = sum(1 for _ in flood_forecast_folder.glob("member_*"))
        print(f"Loading flood maps for {n_ensemble_members} ensemble members")

        # Load all the flood maps in an ensemble per each day
        flood_start_time = self.model.config["hazards"]["floods"]["events"][0][
            "start_time"
        ]
        flood_end_time = self.model.config["hazards"]["floods"]["events"][0]["end_time"]

        members = []  # Every time has its own ensemble
        for member in range(
            1, n_ensemble_members + 1
        ):  # Define the number of members here
            file_path = (
                flood_forecast_folder
                / f"member_{member}"
                / f"{flood_start_time.strftime(format='%Y%m%dT%H%M%S')} - {flood_end_time.strftime(format='%Y%m%dT%H%M%S')}.zarr"
            )

            members.append(read_zarr(file_path))

        # Concatenate the members for each time (This stacks the list of dataarrays along a new "member" dimension)
        ensemble_flood_maps = xr.concat(members, dim="member")

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
        self, date_time: datetime, strategy: int = 1, exceedance: bool = False
    ) -> dict[tuple[datetime, int], xr.DataArray]:
        """Creates flood probability maps based on the ensemble of flood maps for different warning strategies.

        Args:
            date_time: The forecast date time for which to create the probability maps.
            strategy: Identifier of the warning strategy (1 for water level ranges with measures,
                2 for energy substations, 3 for vulnerable/emergency facilities).
            exceedance: Whether to calculate flood exceedance probability maps (instead of regular probability maps).

        Returns:
            Probability maps for each water level range OR Probability exceedance maps for each lower bound of the water level ranges
            for that forecast initialization date.
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

        # Define water level ranges based on the chosen strategy (check the arguments for more info)
        if strategy == 1:
            # Water level ranges associated to specific measures (based on "impact" scale)
            ranges = []
            if exceedance:
                for key, value in self.var.wlranges_and_measures.items():
                    ranges.append((key, value["min"], None))
            else:
                for key, value in self.var.wlranges_and_measures.items():
                    ranges.append((key, value["min"], value["max"]))
        elif strategy == 2:
            # Water level range for energy substations (based on critical hit > 30 cm of flood)
            # TODO: need to make it not hard coded
            ranges = [(1, 0.3, None)]

        else:
            # Water level range for vulnerable and emergency facilities (flooded or not)
            # TODO: need to make it not hard coded
            ranges = [(1, 0.05, None)]

        probability_maps = {}
        # Loop over water level ranges to calculate probability maps
        for range_id, wl_min, wl_max in ranges:
            daily_ensemble = ensemble_flood_maps
            if wl_max is not None:
                condition = (daily_ensemble >= wl_min) & (daily_ensemble <= wl_max)
            else:
                condition = daily_ensemble >= wl_min
            probability = condition.sum(dim="member") / condition.sizes["member"]

            # Save probability map as a zarr file
            if exceedance:
                file_name = (
                    f"prob_exceedance_map_range{range_id}_strategy{strategy}.zarr"
                )
            else:
                file_name = f"prob_map_range{range_id}_strategy{strategy}.zarr"
            file_path = prob_folder / file_name
            file_path.mkdir(parents=True, exist_ok=True)

            # The y axis is flipped when writing to zarr, so fixing it here so it can be used for zonal stats later
            # TODO: need do it in a more elegant way
            if probability.y.values[0] < probability.y.values[-1]:
                print("flipping y axis so it can be used for zonal stats later")
                probability = probability.sortby("y", ascending=False)

            probability = probability.astype(np.float32)
            probability = probability.rio.write_nodata(np.nan)
            probability = write_zarr(da=probability, path=file_path, crs=crs)

            probability_maps[(date_time, range_id)] = probability

        return probability_maps
        # Right now I am not using this for anything, but maybe useful later to replace the file loading

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

    def water_level_warning_strategy(
        self,
        date_time: datetime,
        prob_threshold: float = 0.6,
        area_threshold: float = 0.1,
        strategy_id: int = 1,
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
        households = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                self.var.locations.data[:, 0], self.var.locations.data[:, 1]
            ),
            crs="EPSG:4326",
        )
        postal_codes = self.postal_codes.copy()
        # Maybe load this as a global var (?) instead of loading it each time

        for range_id in range_ids:
            # Build path to probability map
            prob_path = Path(
                self.model.output_folder
                / "prob_maps"
                / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
                / f"prob_map_range{range_id}_strategy{strategy_id}.zarr"
            )

            # Open the probability map
            prob_da = read_zarr(prob_path)

            if postal_codes.crs != prob_da.rio.crs:
                postal_codes = postal_codes.to_crs(prob_da.rio.crs)

            # Rasterize postal codes to the same grid as the probability map
            pc_mask = rasterize(
                ((geom, i) for i, geom in enumerate(postal_codes.geometry)),
                out_shape=(prob_da.rio.height, prob_da.rio.width),
                transform=prob_da.rio.transform(),
                all_touched=True,
                fill=-1,
            )

            # Iterate through each postal code and check how many pixels exceed the threshold
            for i, pc_row in postal_codes.iterrows():
                postal_code = pc_row["postcode"]

                # Get the values for this postal code from the probability map
                valid_pixels = prob_da.values.squeeze()[pc_mask == i]

                n_total = len(valid_pixels)

                if n_total == 0:
                    print(f"No valid pixels found for postal code {postal_code}")
                    percentage = 0
                else:
                    n_above = np.sum(valid_pixels >= prob_threshold)
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
                            "date_time": date_time.isoformat(),
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
            / f"warning_log_water_levels_{date_time.isoformat().replace(':', '').replace('-', '')}.csv"
        )
        pd.DataFrame(warning_log).to_csv(path, index=False)

    def load_critical_infrastructure(self) -> None:
        """Load critical infrastructure elements (vulnerable and emergency facilities, energy substations) and assign them to postal codes."""
        # Load postal codes
        postal_codes = self.postal_codes.copy()

        # Get critical facilities (vulnerable and emergency) and update buildings with relevant attributes
        self.get_critical_facilities()

        # Assign critical facilities to postal codes
        critical_facilities = read_geom(
            self.model.files["geom"]["assets/critical_facilities"]
        )
        self.assign_critical_facilities_to_postal_codes(
            critical_facilities, postal_codes
        )

        # Get energy substations and assign them to postal codes
        substations = read_geom(self.model.files["geom"]["assets/energy_substations"])
        self.assign_energy_substations_to_postal_codes(substations, postal_codes)

    def get_critical_facilities(self) -> None:
        """Extract critical infrastructure elements (vulnerable and emergency facilities) from OSM using the catchment polygon as boundary."""
        catchment_boundary = read_geom(self.model.files["geom"]["catchment_boundary"])

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
        self.buildings = pd.DataFrame(buildings).drop("geometry", axis=1)

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
        buildings = gpd.GeoDataFrame(
            self.buildings,
            geometry=gpd.points_from_xy(self.buildings["x"], self.buildings["y"]),
            crs="EPSG:4326",
        )
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
        households = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                self.var.locations.data[:, 0], self.var.locations.data[:, 1]
            ),
            crs="EPSG:4326",
        )

        # Load substations and critical facilities
        substations = read_geom(self.model.files["geom"]["assets/energy_substations"])
        critical_facilities = read_geom(
            self.model.files["geom"]["assets/critical_facilities"]
        )

        # Load postal codes with associated substations and critical facilities
        path = self.model.input_folder / "geom" / "assets"
        postal_codes_with_substations = read_geom(
            path / "postal_codes_with_energy_substations.geoparquet"
        )
        critical_facilities_with_postal_codes = read_geom(
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

        # Sample the probability map at the substations locations
        x = xr.DataArray(substations.geometry.x.values, dims="z")
        y = xr.DataArray(substations.geometry.y.values, dims="z")
        substations["probability"] = prob_map.sel(x=x, y=y, method="nearest").values

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

        # Sample the probability map at the facilities locations using their centroid (can be improved later to use whole area of the polygon)
        critical_facilities = critical_facilities.copy()
        x = xr.DataArray(critical_facilities.geometry.centroid.x.values, dims="z")
        y = xr.DataArray(critical_facilities.geometry.centroid.y.values, dims="z")
        critical_facilities["probability"] = prob_map.sel(
            x=x, y=y, method="nearest"
        ).values

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

    def warning_communication(
        self,
        target_households: gpd.GeoDataFrame,
        measures: set[str],
        evacuate: bool,
        trigger: str,
        communication_efficiency: float = 0.35,
        lt_threshold_evacuation: int = 48,
    ) -> int:
        """Send warnings to a subset of target households according to communication_efficiency.

        Makes sure to send a single message and allows only upward escalation:
        none (0) -> measures (1) -> evacuate (2)
        Evacuation warning only sent if evacuate is True and lead_time <= 48h.

        Args:
            target_households: GeoDataFrame of household points targeted by the warning.
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

        # Get the number of target households
        n_target_households = len(target_households)
        if n_target_households == 0:
            print("No target households to warn.")
            return 0

        # Based on communication efficiency, get household indices of those who can receive the warning (randomly selected)
        n_feasible_warnings = int(n_target_households * communication_efficiency)
        if n_feasible_warnings == 0:
            print(
                "Based on the communication efficiency, no warning will reach households."
            )
            return 0

        rng = np.random.default_rng(42)  # Using a fixed seed for reproducibility
        position_indices = rng.choice(
            n_target_households, n_feasible_warnings, replace=False
        )
        selected_households = target_households.iloc[position_indices]

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

            # Sort the measures by their implementation times and alphabetically
            sorted_inplace_measures_per_time = sorted(
                available_measures,
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
        recommended_measures = pick_recommendations(measures, evacuate, lead_time)

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
        possible_measures_to_recommend = self.var.possible_measures
        possible_warning_triggers = self.var.possible_warning_triggers

        n_warned_households = 0
        for household_id in selected_households.index:
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

    def compute_lead_time(self) -> float:
        """Compute lead time in hours based on forecast start time and current model time.

        Returns:
            float: Lead time in hours.
        """
        flood_event_start_time = self.model.config["hazards"]["floods"]["events"][0][
            "start_time"
        ]
        current_time = self.model.current_time
        lead_time = (flood_event_start_time - current_time).total_seconds() / 3600
        lead_time = max(lead_time, 0)  # Ensure non-negative lead time
        return lead_time

    def household_decision_making(
        self, date_time: datetime, responsive_ratio: float = 0.7
    ) -> None:
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
        eligible_households = self.var.households_with_postal_codes.loc[eligible_ids]

        # Get the list of possible actions for eligible households
        # (this is a list of all possible actions, not the recommended actions given by the warning strategies)
        possible_actions = self.var.possible_measures
        evac_idx = possible_actions.index("evacuate")

        # Initialize an empty list to log actions taken
        actions_log = []

        # For every eligible household, do the recommended measures in the communicated warning
        for household_id in eligible_households.index:
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
                    "date_time": date_time.isoformat(),
                    "postal_code": self.var.households_with_postal_codes.loc[
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
            / f"actions_log_{date_time.isoformat().replace(':', '').replace('-', '')}.csv"
        )
        pd.DataFrame(actions_log).to_csv(path, index=False)

    def decide_household_strategy(self) -> None:
        """This function calculates the utility of adapting to flood risk for each household and decides whether to adapt or not."""
        # update risk perceptions
        self.update_risk_perceptions()
        
        # calculate damages for adapting and not adapting households based on building footprints
        damages_do_not_adapt, damages_adapt = (
            self.flood_risk_module.calculate_building_flood_damages()
        )
        damages_unprotected_w, damages_adapt_w = self.wind_risk_module.calculate_building_wind_damages()
        # update windstorm risk perceptions (use computed damages to avoid re-running scanners)
        self._last_damages_unprotected_w = damages_unprotected_w
        self._last_damages_adapt_w = damages_adapt_w

        self.update_windstorm_risk_perceptions()

        # risk_perception_multi = np.maximum(
        #     self.var.risk_perception.data, self.var.risk_perception_windstorm.data
        # )

        loan_duration_flood = 20
        loan_duration_wind = 20
        shared_cap_on = True
        eu_cap = 1  # 1e9 if shared_cap_on else 1.0
        # self.var.risk_perception_windstorm.data = 2        )

        # calculate expected utilities
        EU_adapt = self.decision_module.calcEU_adapt_flood(
            geom_id="NoID",
            n_agents=self.n,
            wealth=self.var.wealth.data,
            income=self.var.income.data,
            expendature_cap=eu_cap,
            amenity_value=self.var.amenity_value.data,
            amenity_weight=1,
            risk_perception=self.var.risk_perception.data,
            expected_damages_adapt=damages_adapt,
            adaptation_costs=self.var.adaptation_costs.data,
            time_adapted=self.var.time_adapted.data,
            loan_duration=loan_duration_flood,
            p_floods=1 / self.return_periods,
            T=35,
            r=0.03,
            sigma=1,
        )

        EU_adapt_shutters = self.decision_module.calcEU_shutters_windstorm(
            geom_id="NoID",
            n_agents=self.n,
            wealth=self.var.wealth.data,
            income=self.var.income.data,
            expendature_cap=eu_cap,
            amenity_value=self.var.amenity_value.data,
            amenity_weight=1,
            risk_perception=self.var.risk_perception_windstorm.data,  # + 10,
            expected_damages_adapt=damages_adapt_w,
            adaptation_costs=self.var.adaptation_costs_shutters.data,
            time_adapted=self.var.time_adapted_shutters.data,
            loan_duration=loan_duration_wind,
            p_windstorm=1 / self.windstorm_return_periods,
            T=35,
            r=0.03,
            sigma=1,
            # adapted_shutters=self.var.adapted_shutters.data == 1,
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
            adapted=self.var.adapted.data == 1,
            p_floods=1 / self.return_periods,
            T=35,
            r=0.03,
            sigma=1,
        )

        EU_unprotected_w = self.decision_module.calcEU_do_nothing_w(
            geom_id="NoID",
            n_agents=self.n,
            wealth=self.var.wealth.data,
            income=self.var.income.data,
            expendature_cap=eu_cap,
            amenity_value=self.var.amenity_value.data,
            amenity_weight=1,
            risk_perception=self.var.risk_perception_windstorm.data,
            expected_damages=damages_unprotected_w,
            adapted=self.var.adapted_shutters.data == 1,
            p_windstorm=1 / self.windstorm_return_periods,
            T=35,
            r=0.03,
            sigma=1,
        )

        EU_do_nothing = self.decision_module.calcEU_no_insure(
            n_agents=self.n,
            wealth=self.var.wealth.data,
            income=self.var.income.data,
            expenditure_cap=eu_cap,
            amenity_value=self.var.amenity_value.data,
            amenity_weight=1,
            risk_perception_flood=self.var.risk_perception.data,
            risk_perception_wind=self.var.risk_perception_windstorm.data,
            expected_damages_flood=damages_do_not_adapt,
            expected_damages_wind=damages_unprotected_w,
            p_flood=1 / self.return_periods,
            p_wind=1 / self.windstorm_return_periods,
            T=35,
            r=0.03,
            sigma=1,
            public_reinsurer=0.5,
        )

        EU_multirisk_insurance, premium, premium_private, premium_public = (
            self.decision_module.calcEU_insure_multirisk_residual(
                geom_id="NoID",
                n_agents=self.n,
                insurance_scheme=self.var.insurance_scheme,
                wealth=self.var.wealth.data,
                income=self.var.income.data,
                expenditure_cap=eu_cap,
                amenity_value=self.var.amenity_value.data,
                amenity_weight=1,
                risk_perception_flood=self.var.risk_perception.data,
                risk_perception_wind=self.var.risk_perception_windstorm.data,
                expected_damages_flood=damages_do_not_adapt,
                expected_damages_floodadapted=damages_adapt,
                expected_damages_wind=damages_unprotected_w,
                expected_damages_windadapted=damages_adapt_w,
                p_flood=1 / self.return_periods,  # dummy, should be multirisk damages
                p_wind=1
                / self.windstorm_return_periods,  # dummy, should be multirisk return period
                time_adapted=self.var.time_with_insurance.data,
                loan_duration=0,  # insurance premium to be calculated
                T=35,
                r=0.03,  # needs to be adapted for insurance
                sigma=1,
                deductible=0.1,
                operating_insurer=0.3,  # needs to be discussed for insurance
                public_reinsurer=0.5,
                adapted_floodproofing=self.var.adapted.data == 1,
                adapted_windshutters=self.var.adapted_shutters.data == 1,
                insured_value=self.var.property_value.data.astype(np.float32),
            )
        )


        ## CARO REPORTING
        self._last_premium = premium
        self._last_premium_private = premium_private
        self._last_premium_public = premium_public

        # CARO DEBUG: premium affordability
        inc = self.var.income.data.astype(np.float32)
        prem = np.asarray(premium, dtype=np.float32).reshape(-1)
        print(
            "[insurance] premium stats: "
            f"min={float(np.min(prem)):.2f}, p50={float(np.median(prem)):.2f},"
            f"p95={float(np.quantile(prem, 0.95)):.2f}, max={float(np.max(prem)):.2f}"
        )

        print(
            f"[insurance] affordable frac (premium < income): {float(np.mean(prem < inc)):.4f}"
        )
        # print(
        #     f"[insurance] mean premium/income (where income>0): {float(np.mean(prem[inc > 0] / inc[inc > 0])):.4f}"
        # )
        mask = inc > 0

        if np.any(mask):
            ratio = prem[mask] / inc[mask]
            print(f"[insurance] mean premium/income: {float(np.mean(ratio)):.4f}")
        else:
            print("[insurance] no positive income households")

        # CARO DEBUG: Income stats
        def q(a, p):
            return float(np.quantile(a[a > 0], p)) if np.any(a > 0) else float("nan")

        print(
            "[income] stats: "
            f"min={float(np.min(inc)):.2f}, p50={q(inc, 0.5):.2f}, p95={q(inc, 0.95):.2f}, max={float(np.max(inc)):.2f}"
        )
        print(
            "[insurance] ratio stats (premium/income, income>0): "
            f"p50={q(prem / inc, 0.5):.2f}, p95={q(prem / inc, 0.95):.2f}"
        )
        print(
            f"[insurance] affordable frac (premium < income): {float(np.mean((inc > 0) & (prem < inc))):.4f}"
        )

        # ---------------------------------------------------------------------
        # Shared affordability constraint across strategies (one income/wealth)
        # ---------------------------------------------------------------------
        exp_cap = (
            1.0  # currently hard-coded in your calls; consider pulling from config
        )

        inc = self.var.income.data.astype(np.float32)
        w = self.var.wealth.data.astype(np.float32)
        budget = 0.5 * inc * np.float32(exp_cap)  # + (0.1 * w * np.float32(exp_cap))

        flood_cost = self.var.adaptation_costs.data.astype(
            np.float32
        )  # annual payment in your model
        shutters_cost = self.var.adaptation_costs_shutters.data.astype(
            np.float32
        )  # one-time in your EU (still treated as cash-out)
        prem_cost = np.asarray(premium, dtype=np.float32).reshape(-1)


        # OLD CODE
        # initial choices (before shared-budget reconciliation)
        choose_flood = (EU_adapt > EU_do_not_adapt) | (self.var.adapted.data == 1)
        choose_shutters = (EU_adapt_shutters > EU_unprotected_w) | (self.var.adapted_shutters.data == 1)
        

        if self.var.insurance_scheme == "private":
            choose_ins = EU_multirisk_insurance > EU_do_nothing
        else:
            choose_ins = np.ones(self.n,dtype=bool)

        # if self.var.insurance_scheme == "catnat":
        #     choose_ins[:] = True

        # "benefit" of each choice (used to decide what to drop if over budget)
        # OLD CODE
        gain_flood = (EU_adapt - EU_do_not_adapt).astype(np.float32)
        gain_shutters = (EU_adapt_shutters - EU_unprotected_w).astype(np.float32)
        gain_ins = (EU_multirisk_insurance - EU_do_nothing).astype(np.float32)

        def total_cost() -> np.ndarray:
            return (
                choose_flood.astype(np.float32) * flood_cost
                + choose_shutters.astype(np.float32) * shutters_cost
                + choose_ins.astype(np.float32) * prem_cost
            )

        # OLD CODE
        # # drop least beneficial selected actions until within budget (max 3 drops)
        for _ in range(3):
            over = total_cost() > budget
            if not np.any(over):
                break

            # if self.var.insurance_scheme == "catnat":
            #     # Insurance is mandatory so only structural measures can be dropped when over budget

            #     gains = np.stack([gain_flood, gain_shutters, np.full_like(gain_ins, -np.inf)], axis=1)
            #     chosen = np.stack([choose_flood & (self.var.adapted.data == 0),
            #                    choose_shutters & (self.var.adapted_shutters.data == 0), np.zeros_like(choose_ins)], axis=1)

            # else:
            #     # All three actions can be dropped when over budget
            #     gains = np.stack([gain_flood, gain_shutters, gain_ins], axis=1)
            #     chosen = np.stack([choose_flood & (self.var.adapted.data == 0),
            #                    choose_shutters & (self.var.adapted_shutters.data == 0), choose_ins], axis=1)
            if self.var.insurance_scheme == "private":
                # All three actions can be dropped when over budget
                gains = np.stack([gain_flood, gain_shutters, gain_ins], axis=1)
                chosen = np.stack([choose_flood & (self.var.adapted.data == 0),
                               choose_shutters & (self.var.adapted_shutters.data == 0), choose_ins], axis=1)
            else: 
                #Insurance is mandatory so only structural measures can be dropped when over budget
                gains = np.stack([gain_flood, gain_shutters, np.full_like(gain_ins, -np.inf)], axis=1)
                chosen = np.stack([choose_flood & (self.var.adapted.data == 0),
                               choose_shutters & (self.var.adapted_shutters.data == 0), np.zeros_like(choose_ins)], axis=1)

            gains_masked = np.where(chosen, gains, np.inf)
            drop_idx = np.argmin(gains_masked, axis=1)

            drop_f = over & (drop_idx == 0) & choose_flood
            drop_s = over & (drop_idx == 1) & choose_shutters
            
            # if self.var.insurance_scheme == "catnat":
            #     drop_i = np.zeros_like(drop_f)  # insurance cannot be dropped
            # else:
            #     drop_i = over & (drop_idx == 2) & choose_ins

            if self.var.insurance_scheme == "private":
                drop_i = over & (drop_idx == 2) & choose_ins
            else:
                drop_i = np.zeros_like(drop_f)  # insurance cannot be dropped

            choose_flood[drop_f] = False
            choose_shutters[drop_s] = False
            choose_ins[drop_i] = False

        
        # -----------------------------
        # DEBUG: shared-cap diagnostics
        # -----------------------------
        pre_flood = gain_flood > 0
        pre_shut = gain_shutters > 0
        pre_ins = (EU_multirisk_insurance > EU_do_nothing
                   if self.var.insurance_scheme == "private"
                   else np.ones(self.n, dtype=bool))
        ## END NEW CODE

        # pre_flood = EU_adapt > EU_do_not_adapt
        # pre_shut = EU_adapt_shutters > EU_unprotected_w
        # pre_ins = EU_multirisk_insurance > EU_do_nothing

        pre_cost = (
            pre_flood.astype(np.float32) * flood_cost
            + pre_shut.astype(np.float32) * shutters_cost
            + pre_ins.astype(np.float32) * prem_cost
        )
        post_cost = total_cost()

        over_pre = pre_cost > budget
        over_post = post_cost > budget

        drop_flood = pre_flood & ~choose_flood
        drop_shut = pre_shut & ~choose_shutters
        drop_ins = pre_ins & ~choose_ins
        dropped_any = drop_flood | drop_shut | drop_ins

        print(
            "[shared cap] chosen (pre->post): "
            f"flood={int(pre_flood.sum())}->{int(choose_flood.sum())}, "
            f"shutters={int(pre_shut.sum())}->{int(choose_shutters.sum())}, "
            f"ins={int(pre_ins.sum())}->{int(choose_ins.sum())}"
        )
        print(
            "[shared cap] over-budget households (pre->post): "
            f"{int(over_pre.sum())}->{int(over_post.sum())}"
        )
        print(
            "[shared cap] dropped actions: "
            f"flood={int(drop_flood.sum())}, shutters={int(drop_shut.sum())}, ins={int(drop_ins.sum())}"
        )
        if np.any(dropped_any):
            ratio = pre_cost[dropped_any] / budget[dropped_any]
            print(
                "[shared cap] pre-cost/budget among affected: "
                f"p50={float(np.median(ratio)):.2f}, p95={float(np.quantile(ratio, 0.95)):.2f}, max={float(np.max(ratio)):.2f}"
            )

        ## CARO MORE DEBUG
        both_pre = pre_flood & pre_shut
        both_post = choose_flood & choose_shutters
        print(
            f"[shared cap] overlap flood&shutters (pre->post): {int(both_pre.sum())}->{int(both_post.sum())}"
        )

        if np.any(dropped_any):
            idx = dropped_any
            print(
                "[shared cap] affected median costs: "
                f"flood={float(np.median(flood_cost[idx])):.0f}, "
                f"shutters={float(np.median(shutters_cost[idx])):.0f}, "
                f"insurance={float(np.median(premium)):.0f}, "
                f"budget={float(np.median(budget[idx])):.0f}"
            )

        # ---------------------------------------------------------------------
        # Execute strategy with reconciled choices
        # ---------------------------------------------------------------------
        household_adapting_flood = np.where(choose_flood)[0]
        self.var.adapted[household_adapting_flood] = 1
        self.var.time_adapted[household_adapting_flood] += 1

        household_adapting_shutters = np.where(choose_shutters)[0]
        self.var.adapted_shutters[household_adapting_shutters] = 1
        self.var.time_adapted_shutters[household_adapting_shutters] += 1

        insurance_now = choose_ins
        households_insurance = np.where(insurance_now)[0]

        self.var.adapted_insurance.data[: self.n] = insurance_now.astype(np.int32)
        self.var.time_with_insurance.data[: self.n] = np.where(
            insurance_now,
            self.var.time_with_insurance.data[: self.n] + 1,
            0,
        ).astype(self.var.time_with_insurance.data.dtype, copy=False)

        # update column in buildings
        self.update_building_adaptation_status(
            household_adapting_flood, "floodproofing"
        )
        self.update_building_adaptation_status(household_adapting_shutters, "shutters")
        self.update_building_adaptation_status(households_insurance, "insurance")

        # Store premiums for research output
        self.var.premium = premium
        self.var.premium_private = premium_private
        self.var.premium_public = premium_public

        # self.buildings.to_file(
        #     "C:/Users/nxu279/GitHub/Data/buildings_adapted.gpkg", driver="GPKG"
        # )

        # ds = xr.open_zarr(
        #     "C:/Users/nxu279/GitHub/GEB_try/models/etaple/base/output/flood_maps/coastal_0500.zarr"
        # )
        # ds.rio.to_raster("C:/Users/nxu279/GitHub/Data/coastal_0500.tif")

        n_households = self.n
        print(f"Total N households: {n_households}")

        # print percentage of households that adapted
        print(f"N households that adapted: {len(household_adapting_flood)}")
        print(
            f"N households that adapted with Window Shutters: {len(household_adapting_shutters)}"
        )
        print(f"N households taking insurance: {len(households_insurance)}")

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

    def spinup(self) -> None:
        """This function runs the spin-up process for the household agents."""
        self.construct_income_distribution()
        self.assign_household_attributes()
        if self.model.config["agent_settings"]["households"]["warning_response"]:
            self.assign_households_to_postal_codes()

        self.var.water_demand_per_household_year = (
            -100_000
        )  # probably we will not simulate before the year -100k ;-)

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
        # damages_unprotected_w = merged["damages_unprotected"].to_numpy()
        # damages_shutters_w = merged["damages_wind_shutters"].to_numpy()

        # return (
        #     damages_do_not_adapt,
        #     damages_adapt,
        #     damages_unprotected_w,
        #     damages_shutters_w,

        return damages_do_not_adapt, damages_adapt

    def assign_wdamages_to_agents(
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

        damages_unprotected_w = merged["damages_unprotected"].to_numpy()
        damages_shutters_w = merged["damages_wind_shutters"].to_numpy()

        return damages_unprotected_w, damages_shutters_w

    # def calculate_building_flood_damages(
    #     self, verbose: bool = True, export_building_damages: bool = False
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     """This function calculates the flood damages for the households in the model.

    #     It iterates over the return periods and calculates the damages for each household
    #     based on the flood maps and the building footprints.

    #     Args:
    #         verbose: Verbosity flag.
    #         export_building_damages: Whether to export the building damages to parquet files.
    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: A tuple containing the damage arrays for unprotected and protected buildings.
    #     """
    #     damages_do_not_adapt = np.zeros((self.return_periods.size, self.n), np.float32)
    #     damages_adapt = np.zeros((self.return_periods.size, self.n), np.float32)

    #     buildings: gpd.GeoDataFrame = self.buildings.copy().to_crs(
    #         self.flood_maps[self.return_periods[0]].rio.crs
    #     )

    #     # create a pandas data array for assigning damage to the agents:
    #     agent_df = pd.DataFrame(
    #         {"building_id_of_household": self.var.building_id_of_household}
    #     )

    #     # subset building to those exposed to flooding
    #     buildings = buildings[buildings["flooded"]]

    #     # only calculate damages for buildings with more than 0 occupant
    #     buildings = buildings[buildings["n_occupants"] > 0]

    #     for i, return_period in enumerate(self.return_periods):
    #         flood_map: xr.DataArray = self.flood_maps[return_period]

    #         building_multicurve = buildings.copy()
    #         multi_curves = {
    #             "damages_structure": self.buildings_structure_curve[
    #                 "building_unprotected"
    #             ],
    #             "damages_content": self.buildings_content_curve["building_unprotected"],
    #             "damages_structure_flood_proofed": self.buildings_structure_curve[
    #                 "building_flood_proofed"
    #             ],
    #             "damages_content_flood_proofed": self.buildings_content_curve[
    #                 "building_flood_proofed"
    #             ],
    #         }
    #         damage_buildings: pd.DataFrame = VectorScannerMultiCurves(
    #             features=building_multicurve.rename(
    #                 columns={
    #                     "COST_STRUCTURAL_USD_SQM": "maximum_damage_structure",
    #                     "COST_CONTENTS_USD_SQM": "maximum_damage_content",
    #                 }
    #             ),
    #             hazard=flood_map,
    #             multi_curves=multi_curves,
    #         )

    #         # sum structure and content damages
    #         damage_buildings["damages"] = (
    #             damage_buildings["damages_structure"]
    #             + damage_buildings["damages_content"]
    #         )
    #         damage_buildings["damages_flood_proofed"] = (
    #             damage_buildings["damages_structure_flood_proofed"]
    #             + damage_buildings["damages_content_flood_proofed"]
    #         )
    #         # concatenate damages to building_multicurve
    #         building_multicurve = pd.concat(
    #             [building_multicurve, damage_buildings], axis=1
    #         )

    #         if export_building_damages:
    #             fn_for_export = self.model.output_folder / "building_damages"
    #             fn_for_export.mkdir(parents=True, exist_ok=True)
    #             building_multicurve.to_parquet(
    #                 self.model.output_folder
    #                 / "building_damages"
    #                 / f"building_damages_rp{return_period}_{self.model.current_time.year}.parquet"
    #             )
    #         building_multicurve = building_multicurve[
    #             ["id", "damages", "damages_flood_proofed"]
    #         ]
    #         # merged["damage"] is aligned with agents
    #         damages_do_not_adapt[i], damages_adapt[i] = self.assign_damages_to_agents(
    #             agent_df,
    #             building_multicurve,
    #         )
    #         if verbose:
    #             print(
    #                 f"Damages rp{return_period}: {round(damages_do_not_adapt[i].sum() / 1e6)} million"
    #             )
    #             print(
    #                 f"Damages adapt rp{return_period}: {round(damages_adapt[i].sum() / 1e6)} million"
    #             )
    #     return damages_do_not_adapt, damages_adapt

    def update_households_geodataframe_w_warning_variables(
        self, date_time: datetime
    ) -> None:
        """This function merges the global variables related to warnings to the households geodataframe for visualization purposes.

        Args:
            date_time: The forecast date time for which to update the households geodataframe.
        """
        household_points: gpd.GeoDataFrame = (
            self.var.households_with_postal_codes.copy()
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

        possible_measures_to_recommend = self.var.possible_measures
        for i, measure in enumerate(possible_measures_to_recommend):
            if measure == "sandbags":
                household_points["recom_sandbags"] = self.var.recommended_measures[:, i]
            if measure == "elevate possessions":
                household_points["recom_elev_possessions"] = (
                    self.var.recommended_measures[:, i]
                )
            if measure == "evacuate":
                household_points["recom_evacuate"] = self.var.recommended_measures[:, i]

        possible_actions = self.var.possible_measures
        for i, action in enumerate(possible_actions):
            if action == "sandbags":
                household_points["sandbags"] = self.var.actions_taken[:, i]
            if action == "elevate possessions":
                household_points["elevated_possessions"] = self.var.actions_taken[:, i]

        household_points.to_parquet(
            self.model.output_folder
            / "action_maps"
            / f"households_with_warning_parameters_{date_time.isoformat().replace(':', '').replace('-', '')}.geoparquet"
        )

    # def calculate_building_wind_damages(
    #     self, verbose: bool = True, export_building_damages: bool = False
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     """This function calculates the windstorm damages for the households in the model.

    #     It iterates over the return periods and calculates the damages for each household
    #     based on the windstorm maps and the building footprints.

    #     Args:
    #         verbose: Verbosity flag.
    #         export_building_damages: Whether to export the building damages to parquet files.
    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: A tuple containing the damage arrays for unprotected and protected buildings.
    #     """
    #     damages_unprotected_w = np.zeros(
    #         (self.windstorm_return_periods.size, self.n), np.float32
    #     )
    #     damages_adapt_w = np.zeros(
    #         (self.windstorm_return_periods.size, self.n), np.float32
    #     )

    #     # CRS alignment
    #     crs = self.flood_maps[self.return_periods[0]].rio.crs
    #     windstorm_map_crs = {
    #         rp: self.windstorm_maps[rp].rio.reproject(crs)
    #         for rp in self.windstorm_return_periods
    #     }

    #     debug_damage_stats = bool(
    #         self.model.config.get("hazards", {})
    #         .get("windstorm", {})
    #         .get("debug_damage_stats", False)
    #     )

    #     # create a pandas data array for assigning damage to the agents:
    #     agent_df = pd.DataFrame(
    #         {"building_id_of_household": self.var.building_id_of_household}
    #     )

    #     # subset building to those exposed to flooding (multi-hazard exposure)
    #     buildings: gpd.GeoDataFrame = self.buildings.copy().to_crs(crs)

    #     only_flooded_buildings = bool(
    #         self.model.config.get("hazards", {})
    #         .get("windstorm", {})
    #         .get("only_flooded_buildings", True)
    #     )
    #     if only_flooded_buildings:
    #         buildings = buildings[buildings["flooded"]]
    #     # only calculate damages for buildings with more than 0 occupant
    #     buildings = buildings[buildings["n_occupants"] > 0]

    #     for i, return_period in enumerate(self.windstorm_return_periods):
    #         wind_map = windstorm_map_crs[return_period]

    #         wind_threshold = 27.01
    #         wind_map_masked = wind_map.fillna(0.0)
    #         mind_map_masked = wind_map_masked.where(
    #             wind_map_masked >= wind_threshold, 0.0
    #         )

    #         building_multicurve = buildings.copy()

    #         multi_curves = {
    #             "damages_structure_unprotected": self.wind_buildings_structure_curve[
    #                 "building_unprotected"
    #             ],
    #             "damages_structure_wind_shutters": self.wind_buildings_structure_curve[
    #                 "building_window_shutters"
    #             ],
    #         }
    #         damage_buildings: pd.Series = VectorScannerMultiCurves(
    #             features=building_multicurve.rename(
    #                 columns={"maximum_damage_m2": "maximum_damage"}
    #             ),
    #             hazard=wind_map_masked,
    #             multi_curves=multi_curves,
    #         )

    #         damage_buildings["damages_unprotected"] = damage_buildings[
    #             "damages_structure_unprotected"
    #         ]
    #         damage_buildings["damages_wind_shutters"] = damage_buildings[
    #             "damages_structure_wind_shutters"
    #         ]

    #         building_multicurve = pd.concat(
    #             [building_multicurve, damage_buildings], axis=1
    #         )

    #         # Damage_threshold = 0.001

    #         # buildings_to_damage.loc[
    #         #     buildings_to_damage["damages_unprotected"] < Damage_threshold,
    #         #     "damages_unprotected",
    #         # ] = 0.0

    #         if export_building_damages:
    #             fn_export = self.model.output_folder / "building_wind_damages"
    #             fn_export.mkdir(parents=True, exist_ok=True)
    #             building_multicurve.to_parquet(
    #                 self.model.output_folder
    #                 / "building_wind_damages"
    #                 / f"building_wind_damages_rp{return_period}_{self.model.current_time.year}.parquet"
    #             )
    #         building_multicurve = building_multicurve[
    #             ["id", "damages_unprotected", "damages_wind_shutters"]
    #         ]
    #         # merged["damage"] is aligned with agents
    #         damages_unprotected_w[i], damages_adapt_w[i] = (
    #             self.assign_wdamages_to_agents(
    #                 agent_df,
    #                 building_multicurve,
    #             )
    #         )

    #         if debug_damage_stats:
    #             unprot = damages_unprotected_w[i]
    #             prot = damages_adapt_w[i]
    #             # Keep this cheap: simple stats + non-zero fraction.
    #             frac_nonzero = float(np.mean(unprot > 0))
    #             print(
    #                 "Wind damage stats "
    #                 f"rp={int(return_period)}: "
    #                 f"sum={float(unprot.sum()):.3e}, mean={float(np.mean(unprot)):.3e}, "
    #                 f"p95={float(np.quantile(unprot, 0.95)):.3e}, max={float(np.max(unprot)):.3e}, "
    #                 f"nonzero_frac={frac_nonzero:.3f}, "
    #                 f"adapt_mean={float(np.mean(prot)):.3e}"
    #             )
    #         if verbose:
    #             print(
    #                 f"Wind Damages rp{return_period}: {round(damages_unprotected_w[i].sum() / 1e6)} million"
    #             )
    #             print(
    #                 f"Wind Damages adapt rp{return_period}: {round(damages_adapt_w[i].sum() / 1e6)} million"
    #             )
    #     return damages_unprotected_w, damages_adapt_w

    # def get_max_wind_at_buildings(
    #     self, buildings_gdf: gpd.GeoDataFrame, wind_map: xr.DataArray
    # ) -> np.ndarray:
    #     """This function extracts the maximum wind speed at the location of each building.

    #     Args:
    #         buildings_gdf: GeoDataFrame containing building geometries.
    #         wind_map: xarray DataArray containing the wind speed map.

    #     Returns:
    #         A numpy array containing the maximum wind speed at each building location.
    #     """
    #     # Extract the maximum wind speed at the location of each building
    #     # by sampling the wind map at the building centroids
    #     max_winds = []
    #     for geom in buildings_gdf.geometry:
    #         masked = wind_map.rio.clip([geom], buildings_gdf.crs, drop=False)
    #         max_winds.append(float(masked.max()))

    #     return np.array(max_winds)

    # def flood(self, flood_depth: xr.DataArray) -> float:
    #     """This function computes the damages for the assets and land use types in the model.

    #     Args:
    #         flood_depth: The flood map containing water levels for the flood event [m].

    #     Returns:
    #         The total flood damages for the event for all assets and land use types.

    #     """
    #     flood_depth: xr.DataArray = flood_depth.compute()

    #     buildings: gpd.GeoDataFrame = self.buildings.copy().to_crs(flood_depth.rio.crs)

    #     household_points: gpd.GeoDataFrame = self.var.household_points.copy().to_crs(
    #         flood_depth.rio.crs
    #     )

    #     if self.model.config["agent_settings"]["households"]["warning_response"]:
    #         # make sure household points and actions taken have the same length
    #         assert len(household_points) == self.var.actions_taken.shape[0]

    #         # add columns for protective actions
    #         household_points["sandbags"] = False
    #         household_points["elevated_possessions"] = False

    #         # mark households that took protective actions
    #         household_points.loc[
    #             np.asarray(self.var.actions_taken)[:, 0] == 1, "elevated_possessions"
    #         ] = True
    #         household_points.loc[
    #             np.asarray(self.var.actions_taken)[:, 1] == 1, "sandbags"
    #         ] = True

    #         # spatial join to get household attributes to buildings
    #         buildings: gpd.GeoDataFrame = gpd.sjoin_nearest(
    #             buildings, household_points, how="left", exclusive=True
    #         )

    #         # Assign object types for buildings based on protective measures taken
    #         buildings["object_type"] = "building_unprotected"  # reset
    #         buildings.loc[buildings["elevated_possessions"], "object_type"] = (
    #             "building_elevated_possessions"
    #         )
    #         buildings.loc[buildings["sandbags"], "object_type"] = (
    #             "building_with_sandbags"
    #         )
    #         buildings.loc[
    #             buildings["elevated_possessions"] & buildings["sandbags"], "object_type"
    #         ] = "building_all_forecast_based"
    #         # TODO: need to move the update of the actions takens by households to outside the flood function

    #         # Save the buildings with actions taken
    #         buildings.to_parquet(
    #             self.model.output_folder
    #             / "action_maps"
    #             / "buildings_with_protective_measures.geoparquet"
    #         )

    #         # Assign object types for buildings centroid based on protective measures taken
    #         buildings_centroid = household_points.to_crs(flood_depth.rio.crs)
    #         buildings_centroid["object_type"] = np.select(
    #             [
    #                 (
    #                     buildings_centroid["elevated_possessions"]
    #                     & buildings_centroid["sandbags"]
    #                 ),
    #                 buildings_centroid["elevated_possessions"],
    #                 buildings_centroid["sandbags"],
    #             ],
    #             [
    #                 "building_all_forecast_based",
    #                 "building_elevated_possessions",
    #                 "building_with_sandbags",
    #             ],
    #             default="building_unprotected",
    #         )
    #         buildings_centroid["maximum_damage"] = self.var.max_dam_buildings_content

    #     if self.config["adapt"]:
    #         household_points["building_id"] = (
    #             self.var.building_id_of_household
    #         )  # first assign building id to household points gdf
    #         household_points = household_points.merge(
    #             buildings[["id", "flood_proofed"]],
    #             left_on="building_id",
    #             right_on="id",
    #             how="left",
    #         )  # now merge to get flood proofed status

    #         buildings_centroid = household_points.to_crs(flood_depth.rio.crs)

    #         buildings_centroid["maximum_damage"] = self.var.max_dam_buildings_content

    #         buildings["object_type"] = np.where(
    #             buildings["flood_proofed"],
    #             "building_flood_proofed",
    #             "building_unprotected",
    #         )

    #         buildings_centroid["object_type"] = np.where(
    #             buildings_centroid["flood_proofed"],
    #             "building_protected",
    #             "building_unprotected",
    #         )

    #     else:
    #         household_points["protect_building"] = False

    #         buildings: gpd.GeoDataFrame = gpd.sjoin_nearest(
    #             buildings, household_points, how="left", exclusive=True
    #         )

    #         buildings["object_type"] = "building_unprotected"

    #         # Right now there is no condition to make the households protect their buildings outside of the warning response
    #         buildings.loc[buildings["protect_building"], "object_type"] = (
    #             "building_protected"
    #         )

    #         buildings_centroid = household_points.to_crs(flood_depth.rio.crs)
    #         buildings_centroid["object_type"] = buildings_centroid[
    #             "protect_building"
    #         ].apply(lambda x: "building_protected" if x else "building_unprotected")
    #         buildings_centroid["maximum_damage"] = self.var.max_dam_buildings_content

    #     # Create the folder to save damage maps if it doesn't exist
    #     damage_folder: Path = self.model.output_folder / "damage_maps"
    #     damage_folder.mkdir(parents=True, exist_ok=True)

    #     damages_buildings_content = VectorScanner(
    #         features=buildings_centroid,
    #         hazard=flood_depth,
    #         vulnerability_curves=self.buildings_content_curve,
    #     )

    #     total_damages_content = damages_buildings_content.sum()

    #     # save it to a gpkg file
    #     gdf_content = buildings_centroid.copy()
    #     gdf_content["damage"] = damages_buildings_content
    #     category_name: str = "buildings_content"
    #     filename: str = f"damage_map_{category_name}.gpkg"
    #     gdf_content.to_file(damage_folder / filename, driver="GPKG")

    #     print(f"damages to building content are: {total_damages_content}")

    #     # Compute damages for buildings structure
    #     damages_buildings_structure: pd.Series = VectorScanner(
    #         features=buildings.rename(columns={"maximum_damage_m2": "maximum_damage"}),  # ty:ignore[invalid-argument-type]
    #         hazard=flood_depth,
    #         vulnerability_curves=self.buildings_structure_curve,
    #     )

    #     total_damage_structure = damages_buildings_structure.sum()

    #     print(f"damages to building structure are: {total_damage_structure}")

    #     # save it to a gpkg file
    #     gdf_structure = buildings.copy()
    #     gdf_structure["damage"] = damages_buildings_structure
    #     category_name: str = "buildings_structure"
    #     filename: str = f"damage_map_{category_name}.gpkg"
    #     gdf_structure.to_file(damage_folder / filename, driver="GPKG")

    #     print(
    #         f"Total damages to buildings are: {total_damages_content + total_damage_structure}"
    #     )

    #     agriculture = from_landuse_raster_to_polygon(
    #         self.HRU.decompress(self.HRU.var.land_owners != -1),
    #         self.HRU.transform,
    #         self.model.crs,
    #     )
    #     agriculture["object_type"] = "agriculture"
    #     agriculture["maximum_damage"] = self.var.max_dam_agriculture_m2

    #     agriculture = agriculture.to_crs(flood_depth.rio.crs)

    #     damages_agriculture = VectorScanner(
    #         features=agriculture,
    #         hazard=flood_depth,
    #         vulnerability_curves=self.var.agriculture_curve,
    #     )
    #     total_damages_agriculture = damages_agriculture.sum()
    #     print(f"damages to agriculture are: {total_damages_agriculture}")

    #     # Load landuse and make turn into polygons
    #     forest = from_landuse_raster_to_polygon(
    #         self.HRU.decompress(self.HRU.var.land_use_type == FOREST),
    #         self.HRU.transform,
    #         self.model.crs,
    #     )
    #     forest["object_type"] = "forest"
    #     forest["maximum_damage"] = self.var.max_dam_forest_m2

    #     forest = forest.to_crs(flood_depth.rio.crs)

    #     damages_forest = VectorScanner(
    #         features=forest,
    #         hazard=flood_depth,
    #         vulnerability_curves=self.var.forest_curve,
    #     )
    #     total_damages_forest = damages_forest.sum()
    #     print(f"damages to forest are: {total_damages_forest}")

    #     roads = self.roads.to_crs(flood_depth.rio.crs)
    #     damages_roads = VectorScanner(
    #         features=roads.rename(columns={"maximum_damage_m": "maximum_damage"}),
    #         hazard=flood_depth,
    #         vulnerability_curves=self.var.road_curves,
    #     )
    #     total_damages_roads = damages_roads.sum()
    #     print(f"damages to roads are: {total_damages_roads} ")

    #     rail = self.rail.to_crs(flood_depth.rio.crs)
    #     damages_rail = VectorScanner(
    #         features=rail.rename(columns={"maximum_damage_m": "maximum_damage"}),
    #         hazard=flood_depth,
    #         vulnerability_curves=self.var.rail_curve,
    #     )
    #     total_damages_rail = damages_rail.sum()
    #     print(f"damages to rail are: {total_damages_rail}")

    #     total_flood_damages = (
    #         total_damage_structure
    #         + total_damages_content
    #         + total_damages_roads
    #         + total_damages_rail
    #         + total_damages_forest
    #         + total_damages_agriculture
    #     )
    #     print(f"the total flood damages are: {total_flood_damages}")

    #     return total_flood_damages

    def water_demand(
        self,
        household_demand_to_grid_fn: Callable[
            [ArrayFloat32, TwoDArrayFloat32], ArrayFloat32
        ],
    ) -> tuple[
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

        Args:
            household_demand_to_grid_fn: A function that takes the water demand per
                household and their locations, and returns the water demand gridded to the model grid.
                This is used to convert the household-level water demand to a grid-level water demand
                that can be used in the hydrological model.

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
            current_year = self.model.current_time.year

            # the household water demand calculation is quite expensive,
            # so we only update it when the year changes
            if current_year != self.var.water_demand_per_household_year:
                # the water demand multiplier is a function of the year and region
                water_demand_multiplier_per_region = self.var.municipal_water_withdrawal_m3_per_capita_per_day_multiplier.loc[
                    current_year
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
                water_demand_per_household_m3 = (
                    self.var.municipal_water_demand_per_capita_m3_baseline
                    * self.var.sizes
                    * water_demand_multiplier_per_household
                ) * self.config["adjust_demand_factor"]
                self.var.water_demand_per_household_year = current_year
                self.var.water_demand_per_household_m3_gridded = (
                    household_demand_to_grid_fn(
                        water_demand_per_household_m3.data, self.var.locations.data
                    )
                )
                self.model.logger.debug(
                    f"Calculated water demand per household for year {current_year}."
                )

        elif self.config["water_demand"]["method"] == "custom_value":
            # Function to set a custom_value for household water demand. All households have the same demand.
            custom_value = self.config["water_demand"]["custom_value"]["value"]
            water_demand_per_household_m3 = np.full(
                self.var.region_id.shape, custom_value, dtype=float
            )
            self.var.water_demand_per_household_m3_gridded = (
                household_demand_to_grid_fn(
                    water_demand_per_household_m3, self.var.locations.data
                )
            )

        else:
            raise ValueError(
                "Invalid water demand method. Configuration must be 'default' or 'custom_value'."
            )

        return (
            self.var.water_demand_per_household_m3_gridded,
            self.var.water_efficiency_per_household,
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
                        crs=self.buildings.crs,
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
        out_dir = Path(self.model.output_folder) / "buildings_each_step"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = self.model.current_time.strftime("%Y%m%d")

        # self.buildings.to_file(out_dir / f"buildings_{ts}.gpkg", driver="GPKG")

        n = int(self.n)
        
        def _1d(x):
            a = np.asarray(x, dtype=np.float32).reshape(-1)
            if a.size == 1:
                return np.full(n, float(a[0]), dtype=np.float32)
            return a[:n]
        
        df = pd.DataFrame(
            {
                "time": self.model.current_time.isoformat(),
                "household_id": np.arange(n, dtype=np.int32),

                # risk perception
                "risk_perception_flood": _1d(self.var.risk_perception.data[:n]),
                "risk_perception_wind": _1d(self.var.risk_perception_windstorm.data[:n]),

                # uptake / adaptation
                "flood_adaptation": (np.asarray(self.var.adapted.data[:n]).reshape(-1) == 1),
                "wind_adaptation": (np.asarray(self.var.adapted_shutters.data[:n]).reshape(-1) == 1),
                "insurance_uptake": (np.asarray(self.var.adapted_insurance.data[:n]).reshape(-1) == 1),

                # premiums
                "premium": _1d(getattr(self.var, "premium", np.nan)),
                "premium_private": _1d(getattr(self.var, "premium_private", np.nan)),
                "premium_public": _1d(getattr(self.var, "premium_public", np.nan)),
            }
        )
            
                
        df.to_parquet(out_dir / f"household_data_{ts}.parquet", index=False)
            
        print("Saved household data")

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
