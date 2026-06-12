"""This module contains the Households agent class for simulating household behavior in the GEB model."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import xarray as xr

from geb.geb_types import ArrayFloat32, TwoDArrayFloat32
from geb.workflows.io import read_geom

from ..store import Bucket, DynamicArray
from ..workflows.io import read_array, read_table, read_zarr
from .decision_module import DecisionModule
from .general import AgentBaseClass
from .modules.early_warning import EarlyWarningModule
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

        self.logger = model.logger
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
        self.load_objects()
        self.flood_risk_module = FloodRiskModule(model=self.model, households=self)
        if self.config["adapt"]:
            self.flood_risk_perceptions = []  # Store the flood risk perceptions in here
            self.flood_risk_perceptions_statistics = []  # Store some statistics on flood risk perceptions here

        if self.model.config["agent_settings"]["households"]["warning_response"]:
            self.early_warning_module = EarlyWarningModule(
                model=self.model, households=self
            )
            self.load_critical_infrastructure()
            self.early_warning_module.load_wlranges_and_measures()

        if self.model.in_spinup:
            self.spinup()
        else:
            # In run mode, reload warning configuration data after store loading
            # to ensure JSON updates are reflected immediately
            if self.model.config["agent_settings"]["households"]["warning_response"]:
                self.early_warning_module.load_wlranges_and_measures()

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
        # TODO make the measures not hardcoded but create them automatically based on the loaded wl_ranges_and_measures and implementation times (currently this is hardcoded in the decision module,damage calculations as well, should be made more flexible)
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

    def load_critical_infrastructure(self) -> None:
        """Load critical infrastructure elements (vulnerable and emergency facilities, energy substations) and assign them to postal codes.

        NOTE: This function is currently disabled due to missing infrastructure data.
        """
        asset_type = self.model.config["hazards"]["floods"][
            "critical_infrastructure_warning_strategy"
        ]["asset_type"]
        # Load postal codes
        postal_codes = self.postal_codes.copy()

        # Get critical facilities (vulnerable and emergency) and update buildings with relevant attributes
        # TODO move loading of critical facilities to build phase and save it as a file, instead of loading it from OSM every time in the spinup
        self.get_critical_facilities()

        # Assign critical facilities to postal codes
        assets = read_geom(self.model.files["geom"][f"assets/{asset_type}"])
        self.assign_CI_to_postal_codes(assets, asset_type, postal_codes)

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

    def assign_CI_to_postal_codes(
        self, assets: gpd.GeoDataFrame, asset_type: str, postal_codes: gpd.GeoDataFrame
    ) -> None:
        """Assign critical infrastructure assets to postal codes based on spatial proximity. Every postal code gets assigned to the nearest asset.

        Args:
            assets: GeoDataFrame of critical infrastructure assets.
            asset_type: Type of critical infrastructure assets.
            postal_codes: GeoDataFrame of postal codes.

        Raises:
            ValueError: If the asset type is not recognized.
        """
        if asset_type == "energy_substations":
            postal_codes_with_assets = gpd.sjoin_nearest(
                postal_codes,
                assets[["fid", "geometry"]],
                how="left",
                distance_col="distance",
            )
            # Rename the fid_right column for clarity
            postal_codes_with_assets.rename(
                columns={"fid_right": "asset_id"},
                inplace=True,
            )

        elif asset_type == "critical_facilities":
            critical_facilities_centroid = assets.copy()
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
            postal_codes_with_assets = critical_facilities_with_postal_codes.rename(
                columns={"id": "asset_id"}
            )
        else:
            raise ValueError(
                f"Invalid asset type: {asset_type}. Must be 'energy_substations' or 'critical_facilities'. Will be elaborated later to include more types of critical infrastructure assets."
            )
        # TODO: need to improve this with Thiessen polygons or similar

        # Save the postal codes with associated critical infrastructure assets to a file
        path = (
            self.model.input_folder
            / "geom"
            / "assets"
            / f"postal_codes_with_critical_infrastructure_{asset_type}.geoparquet"
        )
        postal_codes_with_assets.to_parquet(path)

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
