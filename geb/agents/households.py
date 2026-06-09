"""This module contains the Households agent class for simulating household behavior in the GEB model."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import xarray as xr
from pyproj import CRS
from rasterio.features import geometry_mask, rasterize

from geb.geb_types import ArrayFloat32, TwoDArrayFloat32
from geb.workflows.io import read_geom

from ..store import Bucket, DynamicArray
from ..workflows.io import read_array, read_table, read_zarr, write_zarr
from .decision_module import DecisionModule
from .general import AgentBaseClass
from .modules.flood_risk import FloodRiskModule
from .workflows.helpers import from_landuse_raster_to_polygon

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel


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
    water_demand_per_household_year: int
    water_demand_per_household_m3_gridded: ArrayFloat32
    households_with_postal_codes: gpd.GeoDataFrame


class Households(AgentBaseClass):
    """This class implements the household agents."""

    var: HouseholdVariables
    buildings: pd.DataFrame
    roads: gpd.GeoDataFrame
    rail: gpd.GeoDataFrame
    buildings_structure_curve: pd.DataFrame
    buildings_content_curve: pd.DataFrame
    flood_maps: dict[int, xr.DataArray]
    return_periods: np.ndarray
    config: dict

    def __init__(
        self,
        model: GEBModel,
        agents: Agents,
        reduncancy: float,
        logger: logging.Logger | None = None,
    ) -> None:
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

        self.logger = logger or logging.getLogger(__name__)  # model logger
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

        self.load_objects()
        self.load_max_damage_values()
        self.load_damage_curves()
        if self.model.config["agent_settings"]["households"]["warning_response"]:
            # TODO: Temporarily disabled - missing infrastructure data
            self.load_critical_infrastructure()
            self.load_wlranges_and_measures()

        if self.model.in_spinup:
            self.spinup()
        else:
            # In run mode, reload warning configuration data after store loading
            # to ensure JSON updates are reflected immediately
            if self.model.config["agent_settings"]["households"]["warning_response"]:
                self.load_wlranges_and_measures()

            # Check if sensitivity analysis should run
            self._check_and_run_sensitivity_analysis()

    def _check_and_run_sensitivity_analysis(self) -> None:
        """Check if sensitivity analysis is enabled and run it if configured.

        This method checks the model configuration for sensitivity analysis settings
        and runs the analysis if enabled. Results are saved to the output folder.
        """
        if not self.config.get("warning_response", False):
            return

        sensitivity_config = self.config.get("warning_system", {}).get(
            "sensitivity", {}
        )

        if not sensitivity_config.get("enabled", False):
            return

        # Only run if auto_run is enabled (default: False for manual control)
        if not sensitivity_config.get("auto_run", False):
            return

        print("\n" + "=" * 80)
        print("SENSITIVITY ANALYSIS: Auto-run enabled in configuration")
        print("=" * 80 + "\n")

        # Import here to avoid circular imports
        from geb.agents.sensitivity_analysis import SensitivityAnalyzer

        # Initialize and run sensitivity analysis
        analyzer = SensitivityAnalyzer(households=self, config=sensitivity_config)

        # Get forecast dates: either from config, auto-detect from flood maps, or use defaults
        forecast_dates = self._get_forecast_dates_for_sensitivity(sensitivity_config)

        if not forecast_dates:
            print("⚠️  No forecast dates found. Skipping sensitivity analysis.")
            return

        print(f"📅 Found {len(forecast_dates)} forecast dates to analyze:")
        for date in forecast_dates:
            print(f"    - {date.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run the analysis
        results_df = analyzer.run_sensitivity_analysis(forecast_dates=forecast_dates)

        # Create visualizations
        print("\nGenerating visualizations...")
        analyzer.create_visualizations(results_df)

        print(f"\n{'=' * 80}")
        print("SENSITIVITY ANALYSIS COMPLETE")
        print(f"Output folder: {analyzer.output_folder}")
        print(f"{'=' * 80}\n")

    def _get_forecast_dates_for_sensitivity(
        self, sensitivity_config: dict
    ) -> list[datetime.datetime]:
        """Auto-detect forecast dates from existing flood probability exceedance maps.

        Searches for all forecast folders in output/flood_prob_exceedance_maps/ and
        extracts the forecast dates from the folder names.

        Args:
            sensitivity_config: Sensitivity configuration dictionary (unused, kept for compatibility).

        Returns:
            List of datetime objects representing forecast dates found in flood maps.
            Empty list if no flood maps are found.
        """
        import datetime

        print(
            "🔍 Auto-detecting forecast dates from flood probability exceedance maps..."
        )

        exceedance_folder = self.model.output_folder / "flood_prob_exceedance_maps"

        if not exceedance_folder.exists():
            print(f"⚠️  Flood exceedance folder not found: {exceedance_folder}")
            print("    No forecast dates available for sensitivity analysis.")
            return []

        # Find all forecast_* folders
        forecast_folders = sorted(exceedance_folder.glob("forecast_*"))

        if not forecast_folders:
            print(f"⚠️  No forecast folders found in {exceedance_folder}")
            return []

        forecast_dates = []
        for folder in forecast_folders:
            # Extract date from folder name: forecast_20240115T000000
            folder_name = folder.name
            date_str = folder_name.replace("forecast_", "")

            try:
                # Parse date: 20240115T000000 -> 2024-01-15T00:00:00
                date = datetime.datetime.strptime(date_str, "%Y%m%dT%H%M%S")
                forecast_dates.append(date)
            except ValueError:
                print(f"⚠️  Could not parse date from folder: {folder_name}")
                continue

        if forecast_dates:
            print(
                f"✅ Auto-detected {len(forecast_dates)} forecast dates from flood maps"
            )
        else:
            print("⚠️  No valid forecast dates found in flood maps.")

        return forecast_dates

    def run_sensitivity_analysis(
        self,
        forecast_dates: list[datetime.datetime] | None = None,
        custom_config: dict | None = None,
    ) -> pd.DataFrame:
        """Manually run sensitivity analysis for warning system parameters.

        This method allows you to run sensitivity analysis programmatically,
        independent of the auto_run configuration setting.

        Args:
            forecast_dates: List of forecast dates to evaluate. If None, uses defaults.
            custom_config: Custom sensitivity configuration to override model.yml settings.
                If None, uses configuration from model.yml.

        Returns:
            DataFrame containing all sensitivity analysis results.

        Example:
            >>> import datetime
            >>> dates = [datetime.datetime(2024, 1, 15)]
            >>> results = households.run_sensitivity_analysis(forecast_dates=dates)
            >>> print(results.groupby('warning_type')['n_households_warned'].mean())
        """
        import datetime

        from geb.agents.sensitivity_analysis import SensitivityAnalyzer

        # Use custom config or get from model
        if custom_config is None:
            sensitivity_config = self.config.get("warning_system", {}).get(
                "sensitivity", {}
            )
        else:
            sensitivity_config = custom_config

        # Use provided dates or defaults
        if forecast_dates is None:
            forecast_dates = [
                datetime.datetime(2024, 1, 15, 0, 0, 0),
                datetime.datetime(2024, 2, 1, 0, 0, 0),
            ]

        print("\n" + "=" * 80)
        print("SENSITIVITY ANALYSIS: Manual run")
        print("=" * 80 + "\n")

        # Initialize and run
        analyzer = SensitivityAnalyzer(households=self, config=sensitivity_config)
        results_df = analyzer.run_sensitivity_analysis(forecast_dates=forecast_dates)

        # Create visualizations
        print("\nGenerating visualizations...")
        analyzer.create_visualizations(results_df)

        print(f"\n{'=' * 80}")
        print("SENSITIVITY ANALYSIS COMPLETE")
        print(f"Output folder: {analyzer.output_folder}")
        print(f"{'=' * 80}\n")

        return results_df

    @property
    def name(self) -> str:
        """Return the name of the agent type."""
        return "agents.households"

    def load_objects(self) -> None:
        """Load buildings, roads, and rail geometries from model files."""
        # Load buildings
        columns_to_load = [
            "id",
            "x",
            "y",
            "COST_STRUCTURAL_USD_SQM",
            "COST_CONTENTS_USD_SQM",
        ]
        self.buildings = read_table(
            self.model.files["geom"]["assets/open_building_map"],
            columns=columns_to_load,
        )

        self.buildings["object_type"] = (
            "building_unprotected"  # before it was "building_structure"
        )

        # Load roads
        self.roads = read_geom(self.model.files["geom"]["assets/roads"]).rename(  # ty:ignore[invalid-assignment]
            columns={"highway": "object_type"}
        )

        # Load rail
        self.rail = read_geom(self.model.files["geom"]["assets/rails"])
        self.rail["object_type"] = "rail"

        if self.model.config["general"]["forecasts"]["use"]:
            # Load postal codes --
            # TODO: maybe move it to another function? (not really an object)
            self.postal_codes = read_geom(self.model.files["geom"]["postal_codes"])

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

    def load_wlranges_and_measures(self) -> None:
        """Loads the water level ranges and appropriate measures, and the implementation times for measures."""
        with open(self.model.files["dict"]["measures/implementation_times"], "r") as f:
            self.var.implementation_times = json.load(f)

        with open(self.model.files["dict"]["measures/wl_ranges"], "r") as f:
            wlranges_and_measures = json.load(f)
            # convert the keys (range ids) to integers and store them in a new dictionary
            print("Loaded water level ranges and measures:")
            self.var.wlranges_and_measures = {
                int(key): value for key, value in wlranges_and_measures.items()
            }
        print(
            "Loaded water level ranges with associated measures and their implementation times."
        )

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
        # initiate array for storing the water level range that triggered the warning (if any)
        self.var.triggered_wlrange = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
        )
        # initiate array for storing the lead time of the households action
        self.var.action_lead_time = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
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

        # load household points
        household_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                self.var.locations.data[:, 0], self.var.locations.data[:, 1]
            ),
            crs="EPSG:4326",
        )
        self.var.household_points = household_points
        household_points.to_parquet(
            self.model.output_folder / "household_points.geoparquet"
        )
        # initiate an array with expected annual damages (EAD) for each household
        self.var.ead_usd_per_year = DynamicArray(
            np.zeros(self.n, np.float32), max_n=self.max_n
        )

        self.model.logger.info(
            f"Household attributes assigned for {self.n} households with {self.population} people."
        )

    def assign_households_and_buildings_to_postal_codes(self) -> None:
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

        # Buildings
        buildings = self.buildings.copy()
        # Associate buildings with their postal codes to use it later in the warning function
        buildings.to_crs(
            postal_codes.crs, inplace=True
        )  # Change to the same CRS as the postal codes

        # Use representative points of the building geometries to check which postal code they are in
        # This is done to avoid issues with sliver polygons and to ensure that each building is assigned to a single postal code
        representative_points = buildings.copy()
        representative_points["building_geometry"] = representative_points.geometry
        representative_points["geometry"] = (
            representative_points.geometry.representative_point()
        )

        buildings_with_postal_codes = gpd.sjoin(
            representative_points,
            postal_codes[["postcode", "geometry"]],
            how="left",
            predicate="within",
        )

        buildings_with_postal_codes["geometry"] = buildings_with_postal_codes[
            "building_geometry"
        ]
        buildings_with_postal_codes = buildings_with_postal_codes.set_geometry(
            "geometry"
        )
        buildings_with_postal_codes.drop(
            columns=["building_geometry", "index_right"], inplace=True
        )

        buildings_with_postal_codes.to_parquet(
            self.model.output_folder / "buildings_w_postal_codes.geoparquet"
        )
        # TODO: Understand why it does not work if it is just self.buildings
        self.var.buildings = buildings_with_postal_codes

        # Households
        # Associate households with their postal codes to use it later in the warning function
        households = self.var.household_points.copy()
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

        # TODO: Move this to build>agents
        # Change the location of households to the representative point of the building they are in
        # This is done to avoid issues with sliver polygons
        # rep_points_lookup = representative_points[["id", "geometry"]].copy()
        # rep_points_lookup.to_parquet(
        #     self.model.output_folder / "rep_points_lookup.geoparquet"
        # )
        # rep_points_lookup.rename(columns={"geometry": "rep_geometry"}, inplace=True)

        # households_with_postal_codes["b_id"] = self.var.building_id_of_household

        # # Merge based on the building id to get the representative point geometry for each household
        # households_with_postal_codes = households_with_postal_codes.merge(
        #     rep_points_lookup,
        #     left_on="b_id",
        #     right_on="id",
        #     how="left",
        # )

        # # Replace the geometry of the household with the representative point geometry of the building
        # households_with_postal_codes["geometry"] = households_with_postal_codes["rep_geometry"]
        # households_with_postal_codes = households_with_postal_codes.set_geometry("geometry")

        # # Drop columns that are not needed
        # households_with_postal_codes.drop(columns=["rep_geometry", "b_id"], inplace=True, errors="ignore")

        households_with_postal_codes.to_parquet(
            self.model.output_folder / "household_points_w_postal_codes.geoparquet"
        )

        self.var.households_with_postal_codes = households_with_postal_codes

        print(
            f"{len(households_with_postal_codes[households_with_postal_codes['postcode'].notnull()])} households assigned to {households_with_postal_codes['postcode'].nunique()} postal codes."
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
            flooded_household_indices = self.flood_risk_module.return_period_flood()
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

        for member in range(1, 50 + 1):
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
        self, date_time: datetime, strategy: int = 1, exceedance: bool = True
    ) -> xr.DataArray | xr.Dataset:
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

        probability_maps = []
        # Loop over water level ranges to calculate probability maps
        for range_id, wl_min, wl_max in ranges:
            daily_ensemble = ensemble_flood_maps
            self.logger.info(
                f"Creating probability map for date {date_time}, strategy {strategy}, range_id {range_id} (wl_min={wl_min}, wl_max={wl_max})"
            )
            if wl_max is not None:
                condition = (daily_ensemble >= wl_min) & (daily_ensemble <= wl_max)
                self.logger.debug(f"Using range: {wl_min} <= x <= {wl_max}")
            else:
                condition = daily_ensemble >= wl_min
                self.logger.debug(f"Using exceedance: x >= {wl_min}")

            probability = condition.sum(dim="member") / condition.sizes["member"]

            if probability.y.values[0] < probability.y.values[-1]:
                self.logger.info(
                    "flipping y axis so it can be used for zonal stats later"
                )
                probability: Unknown | DataArray = probability.sortby(
                    "y", ascending=False
                )

            # Save probability map as a zarr file
            if exceedance:
                file_name = (
                    f"prob_exceedance_map_range{range_id}_strategy{strategy}.zarr"
                )
            else:
                file_name = f"prob_map_range{range_id}_strategy{strategy}.zarr"
            file_path = prob_folder / file_name
            file_path.mkdir(parents=True, exist_ok=True)

            probability = probability.astype(np.float32)
            probability = probability.rio.write_nodata(np.nan)
            crs = (
                probability.rio.crs if probability.rio.crs is not None else "EPSG:4326"
            )
            probability = write_zarr(da=probability, path=file_path, crs=crs)
            probability_maps.append(
                probability.expand_dims(time=[date_time], range_id=[range_id])
            )

        probability_maps = xr.combine_by_coords(probability_maps)

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
        self, buildings, prob_map, prob_threshold, return_series=False
    ):
        """Identifies which buildings are flooded based on the flood probability map and a specified probability threshold."""
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
            base_efficiency: Base probability for communication efficiency (0-1).
            use_socioeconomic_factors: Whether to use socio-economic factors or fallback to random.

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
            index=[1, 2, 3, 4, 5],  # Education classes
            columns=[1, 2, 3, 4, 5],  # Income quintiles
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

        def get_weight(row):
            edu = row["Education"]
            inc = row["Income_Category"]
            return weights_table.loc[edu, inc]

        target_households["weight"] = target_households.apply(get_weight, axis=1)
        weights = target_households["weight"].values

        return weights

    def compute_lead_time(self) -> float:
        """Compute lead time in hours based on forecast start time and current model time.

        Returns:
            float: Lead time in hours.
        """
        moment_of_flooding = self.model.config["hazards"]["floods"]["events"][0][
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
        implementation_times = self.var.implementation_times
        available_measures = [
            m for m in available_measures if m and m in implementation_times
        ]

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
        triggered_wlrange: int | None = None,
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
            triggered_wlrange: The water level range that triggered the warning, if applicable.
            communication_efficiency: Fraction of target households that successfully receive the warning (0 to 1).
            evacuation_lead_time_threshold: Maximum lead time in hours for which evacuation warnings can be effective.
            weight_by_socioeconomic_factors: Whether to weight the selection of households by socio-economic factors (education, income) or select randomly.

        Returns:
            int: Number of households that were successfully warned.
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
        n_ranges = len(self.var.wlranges_and_measures)
        for household_id in selected_households.index:
            current_level = int(self.var.warning_level[household_id])
            self.var.action_per_range = np.zeros((self.n, n_ranges), dtype=int)
            if triggered_wlrange is not None:
                range_idx = list(self.var.wlranges_and_measures.keys()).index(
                    triggered_wlrange
                )
                self.var.action_per_range[household_id, range_idx] = desired_level
            # Only send a warning if the desired level is higher than current level
            if desired_level > current_level:
                for measure in recommended_measures:
                    measure_idx = possible_measures_to_recommend.index(measure)
                    self.var.recommended_measures[household_id, measure_idx] = True
            else:
                continue
            self.var.warning_level[household_id] = desired_level
            self.var.warning_reached[household_id] = 1
            trigger_idx = possible_warning_triggers.index(trigger)
            self.var.warning_trigger[household_id, trigger_idx] = True
            if triggered_wlrange is not None:
                old_val = self.var.triggered_wlrange[household_id]
                self.logger.debug(
                    f"DEBUG: household_id={household_id}, oude triggered_wlrange={old_val}, nieuwe triggered_wlrange={triggered_wlrange}, trigger={trigger}"
                )
                self.var.triggered_wlrange[household_id] = triggered_wlrange
            n_warned_households += 1
        self.logger.info(f"Warning targeted to reach {n_target_households} households")
        self.logger.info(f"Warning reached {n_warned_households} households")
        return n_warned_households

    def water_level_warning_strategy(
        self,
        date_time: datetime.datetime,
        warning_type: str = "building_based",
        prob_threshold: float = 0.6,
        buildings_hit_threshold: float = 0.1,
        area_hit_threshold: float = 0.1,
        communication_efficiency: float = 1,
        evacuation_lead_time_threshold: int = 48,
        weight_by_socioeconomic_factors: bool = False,
        exceedance: bool = True,
        strategy_id: int = 1,
    ) -> None:
        """Implements the water level warning strategy based on flood probability maps BUCKETS PER MEASURE.

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
            strategy_id: Identifier of the warning strategy (1 for water level ranges with measures).
        """

        # Get the range ids and initialize the warning_log
        range_ids = list(self.var.wlranges_and_measures.keys())
        warning_log = []
        warnings_folder = self.model.output_folder / "warning_logs"
        warnings_folder.mkdir(exist_ok=True, parents=True)

        # Create probability maps
        # TODO: Only create flood probability maps if they do not exist yet
        probability_maps = self.create_flood_probability_maps(
            strategy=strategy_id, date_time=date_time, exceedance=exceedance
        )

        # Load households and postal codes
        buildings = self.var.buildings.copy()
        households = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                self.var.locations.data[:, 0], self.var.locations.data[:, 1]
            ),
            crs="EPSG:4326",
        )
        # Add education and income data to the households
        # IMPORTANT: We use the index of households to correctly map the data
        # because household_points.index corresponds to the arrays in self.var
        if "Education" not in households.columns:
            households["Education"] = households.index.map(
                lambda idx: (
                    self.var.education_level.data[idx]
                    if idx < len(self.var.education_level.data)
                    else np.nan
                )
            )
        if "Income" not in households.columns:
            households["Income"] = households.index.map(
                lambda idx: (
                    self.var.income.data[idx]
                    if idx < len(self.var.income.data)
                    else np.nan
                )
            )

        postal_codes = self.postal_codes
        # Maybe load this as a global var (?) instead of loading it each time

        # Intersection flood maps with buildings or postal codes
        for range_id in range_ids:
            probability_map = probability_maps.sel(range_id=range_id)
            flood_probability_map: xr.DataArray = probability_map[
                list(probability_map.data_vars)[0]
            ]

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
                    postal_codes.loc[postal_code, f"issue_warning_r{range_id}"] = (
                        issue_warning
                    )
                    if issue_warning:
                        self.logger.info(
                            f"Warning issued to postal code {postal_code} on {date_time.isoformat()} for range {range_id}: {percentage_flooded:.0%} of buildings flooded"
                        )
        # Warning Communication - Communicate the warning to the households in the postal code and store the details in the warning log
        # Initialize measures and evacuate flag
        measures = []
        triggered_ranges = []
        evacuate = False
        for postal_code, row in postal_codes.iterrows():
            for range_id in range_ids:
                issue_warning = row[f"issue_warning_r{range_id}"]
                if issue_warning:
                    # Get the measures and evacuation flag from the json dictionary to use in the warning communication function
                    triggered_ranges.append(range_id)
                    recom_measure = self.var.wlranges_and_measures[range_id]["measure"]
                    evacuate = (
                        evacuate or self.var.wlranges_and_measures[range_id]["evacuate"]
                    )
                    if recom_measure:
                        measures.extend(recom_measure)

                    if measures or evacuate:
                        # Filter the affected households based on the postal code
                        affected_households = households[
                            households["postcode"] == postal_code
                        ]

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

    def load_critical_infrastructure(self) -> None:
        """Load critical infrastructure elements (vulnerable and emergency facilities, energy substations) and assign them to postal codes.

        NOTE: This function is currently disabled due to missing infrastructure data.
        TODO: Re-enable when infrastructure data becomes available.
        """
        # TODO: Temporarily disabled - missing infrastructure data
        print(
            "WARNING: load_critical_infrastructure is temporarily disabled (missing data)"
        )
        return
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
        queries: dict[str, dict[str, bool | str | list[str]]] = {
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
        self, date_time: datetime, exceedance: bool = False
    ) -> None:
        """This function implements an evacuation warning strategy based on critical infrastructure elements, such as energy substations, vulnerable and emergency facilities.

        Args:
            date_time: The forecast date time for which to implement the warning strategy.
            exceedance: Whether to consider exceedance probabilities.
        """
        # Check if critical infrastructure warnings are enabled in config
        warning_config = self.model.config["agent_settings"]["households"][
            "warning_system"
        ]
        if not warning_config["strategies"]["critical_infrastructure_warnings"]:
            print(
                f"Critical infrastructure warnings disabled in config for {date_time.isoformat()}"
            )
            return

        # Get probability threshold from config
        prob_threshold = warning_config["probability_threshold"]

        # Get the household points, needed to issue warnings
        households = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                self.var.locations.data[:, 0], self.var.locations.data[:, 1]
            ),
            crs="EPSG:4326",
        )

        # Add education and income data here as well for consistency
        if "Education" not in households.columns:
            households["Education"] = households.index.map(
                lambda idx: (
                    self.var.education_level.data[idx]
                    if idx < len(self.var.education_level.data)
                    else np.nan
                )
            )
        if "Income" not in households.columns:
            households["Income"] = households.index.map(
                lambda idx: (
                    self.var.income.data[idx]
                    if idx < len(self.var.income.data)
                    else np.nan
                )
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
        self.create_flood_probability_maps(
            strategy=strategy_id, date_time=date_time, exceedance=exceedance
        )

        # Get the probability map for the specific day and strategy using dynamic path building
        if exceedance:
            folder_name = "flood_prob_exceedance_maps"
            file_prefix = "prob_exceedance_map"
        else:
            folder_name = "flood_prob_maps"
            file_prefix = "prob_map"

        prob_energy_hit_path = Path(
            self.model.output_folder
            / folder_name
            / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
            / f"{file_prefix}_range1_strategy{strategy_id}.zarr"
        )

        # Open the probability map
        prob_map = read_zarr(prob_energy_hit_path)
        affine = prob_map.rio.transform()

        # Ensure prob_map is 2D by squeezing out any singleton dimensions
        prob_array = prob_map.values
        while prob_array.ndim > 2:
            prob_array = prob_array.squeeze()

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
        self.create_flood_probability_maps(
            strategy=strategy_id, date_time=date_time, exceedance=exceedance
        )

        # Get the probability map for the specific day and strategy using dynamic path building
        if exceedance:
            folder_name = "flood_prob_exceedance_maps"
            file_prefix = "prob_exceedance_map"
        else:
            folder_name = "flood_prob_maps"
            file_prefix = "prob_map"

        prob_critical_facilities_hit_path = Path(
            self.model.output_folder
            / folder_name
            / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
            / f"{file_prefix}_range1_strategy{strategy_id}.zarr"
        )

        # Open the probability map
        prob_map = read_zarr(prob_critical_facilities_hit_path)
        affine = prob_map.rio.transform()

        # Ensure prob_map is 2D by squeezing out any singleton dimensions
        prob_array = prob_map.values
        while prob_array.ndim > 2:
            prob_array = prob_array.squeeze()

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
        warning_config = self.model.config["agent_settings"]["households"][
            "warning_system"
        ]
        subfolder_name = warning_config["strategies"]["warning_type"]
        warnings_folder = self.model.output_folder / "warning_logs" / subfolder_name
        warnings_folder.mkdir(exist_ok=True, parents=True)
        path = (
            warnings_folder
            / f"warning_log_critical_infrastructure_{date_time.isoformat()}.csv"
        )
        pd.DataFrame(warning_log).to_csv(path, index=False)

    def household_decision_making(self, date_time: datetime) -> None:
        """Simulate household emergency response decisions based on warnings and lead time.

        Args:
            date_time: The forecast date time for which to run the decision-making.

        Raises:
            ValueError: If an unknown warning type is specified in the configuration.
        """
        self.logger.info("Running emergency response decision-making for households...")
        self.logger.info(f"Date/time: {date_time}")

        # Get warning system config settings
        warning_config = self.model.config["agent_settings"]["households"][
            "warning_system"
        ]
        warning_type = warning_config["strategies"]["warning_type"]
        self.logger.info(f"Warning type: {warning_type}")

        # Determine response rate based on warning type
        if warning_type == "building_based":
            responsive_ratio = warning_config["response_rates"][
                "building_based_warnings"
            ]

        elif warning_type == "area_based":
            responsive_ratio = warning_config["response_rates"]["area_based_warnings"]
        else:
            raise ValueError(
                f"Unknown warning type: {warning_type} selected in config, choose 'building_based' or 'area_based'."
            )

        self.logger.info(f"Responsive ratio: {responsive_ratio * 100:.2f}%")

        # Get lead time in hours
        lead_time = self.compute_lead_time()
        self.logger.info(f"Lead time: {lead_time:.1f} hours")

        # Filter households that did not evacuate, were warned and are responsive
        not_evacuated_ids = self.var.evacuated == 0
        warned_ids = self.var.warning_reached == 1
        responsive_ids = self.var.response_probability < responsive_ratio

        self.logger.info(f"Total households: {len(self.var.household_points)}")
        self.logger.info(f"Non-evacuated households: {np.sum(not_evacuated_ids)}")
        self.logger.info(f"Warned households: {np.sum(warned_ids)}")
        self.logger.info(f"Responsive households: {np.sum(responsive_ids)}")

        # Combine the filters and apply it to household_points
        eligible_ids = np.asarray(warned_ids & not_evacuated_ids & responsive_ids)
        eligible_households = self.var.households_with_postal_codes.loc[eligible_ids]

        self.logger.info(f"Eligible households for action: {len(eligible_households)}")

        # Get the list of possible actions for eligible households
        # (this is a list of all possible actions, not the recommended actions given by the warning strategies)
        possible_actions = self.var.possible_measures
        evac_idx = possible_actions.index("evacuate")

        # Initialize an empty list to log actions taken
        actions_log = []

        self.logger.info(
            "Processing eligible households for action based on warnings and lead time..."
        )

        # For every eligible household, do the recommended measures in the communicated warning
        for i, household_id in enumerate(eligible_households.index):
            self.logger.info(
                f"Processing household {household_id} ({i + 1}/{len(eligible_households)})"
            )

            # Check if elevated possessions action is being taken for the FIRST TIME
            # (before updating actions_taken array)
            elevated_possessions_idx = possible_actions.index("elevate possessions")
            newly_taking_elevated_possessions = (
                not self.var.actions_taken[household_id, elevated_possessions_idx]
                and self.var.recommended_measures[
                    household_id, elevated_possessions_idx
                ]
            )

            # For measure in recommended measures, add new measures to existing actions_taken array
            # Use OR operation to preserve previously taken actions while adding new ones
            for i, measure_recommended in enumerate(
                self.var.recommended_measures[household_id]
            ):
                if measure_recommended:
                    self.var.actions_taken[household_id, i] = True

            # Only update action_lead_time if elevated possessions action is being taken for the first time
            if newly_taking_elevated_possessions:
                self.var.action_lead_time[household_id] = lead_time

            # If evacuation is among the actions taken, mark household as evacuated
            if self.var.actions_taken[household_id, evac_idx]:
                self.var.evacuated[household_id] = 1
                print(f"  Household {household_id} evacuated")

            # Log the actions taken
            actions = []
            for j, action in enumerate(possible_actions):
                if self.var.actions_taken[household_id, j]:
                    actions.append(action)

            print(f"  Actions taken: {actions}")

            # Haal de warning_level (range_id) op voor deze household
            warning_level = int(self.var.warning_level[household_id])
            # Haal de triggering wlrange op voor deze household
            triggered_wlrange = None
            if hasattr(self.var, "triggered_wlrange"):
                triggered_wlrange = self.var.triggered_wlrange[household_id]
            # Haal alle acties per range op voor deze household
            action_per_range = None
            if hasattr(self.var, "action_per_range"):
                action_per_range = self.var.action_per_range[household_id].tolist()
            actions_log.append(
                {
                    "lead_time": lead_time,
                    "date_time": date_time.isoformat(),
                    "postal_code": self.var.households_with_postal_codes.loc[
                        household_id, "postcode"
                    ],
                    "household_id": household_id,
                    "actions": actions,
                    "warning_level": warning_level,
                    "triggered_wlrange": triggered_wlrange,
                    "action_per_range": action_per_range,
                }
            )

        print("\n=== SUMMARY ===")
        print(f"Total actions logged: {len(actions_log)}")
        total_actions = sum(len(entry["actions"]) for entry in actions_log)
        print(f"Total individual actions taken: {total_actions}")

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
        self.var.ead_usd_per_year[:] = self.flood_risk_module.calculate_ead(
            damages_do_not_adapt, damages_adapt, self.var.adapted.data
        ).astype(np.float32)

    def spinup(self) -> None:
        """This function runs the spin-up process for the household agents."""
        self.construct_income_distribution()
        self.assign_household_attributes()
        if self.model.config["agent_settings"]["households"]["warning_response"]:
            self.assign_households_and_buildings_to_postal_codes()

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

        return damages_do_not_adapt, damages_adapt

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
            "triggered_wlrange",
            "action_per_range",
            "action_lead_time",
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
            "triggered_wlrange",
            "action_lead_time",
        ]:
            if hasattr(self.var, name):
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

        # Add action_per_range columns
        if hasattr(self.var, "action_per_range"):
            range_ids = list(self.var.wlranges_and_measures.keys())
            for idx, range_id in enumerate(range_ids):
                household_points[f"action_range_{range_id}"] = (
                    self.var.action_per_range[:, idx]
                )
        # Create the action_maps directory if it doesn't exist
        action_maps_dir = self.model.output_folder / "action_maps"
        action_maps_dir.mkdir(parents=True, exist_ok=True)
        household_points.to_parquet(
            action_maps_dir
            / f"households_with_warning_parameters_{date_time.strftime('%Y%m%dT%H%M%S')}.geoparquet"
        )

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
            self.var.water_demand_per_household_m3_gridded.copy(),
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
