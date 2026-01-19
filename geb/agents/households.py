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
from rasterio.features import geometry_mask, shapes
from rasterio.transform import Affine
from rasterstats import point_query, zonal_stats
from scipy import interpolate
from shapely import unary_union
from shapely.geometry import Polygon, shape

from geb.geb_types import ArrayFloat32, TwoDArrayBool, TwoDArrayInt
from geb.workflows.io import read_params

from ..hydrology.landcovers import (
    FOREST,
)
from ..store import Bucket, DynamicArray
from ..workflows.damage_scanner import VectorScanner, VectorScannerMultiCurves
from ..workflows.io import read_array, read_table, read_zarr, write_zarr
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

        if self.config["adapt"]:
            self.load_flood_maps()
            self.flood_risk_perceptions = []  # Store the flood risk perceptions in here
            self.flood_risk_perceptions_statistics = []  # Store some statistics on flood risk perceptions here

        self.load_objects()
        self.load_max_damage_values()
        self.load_damage_curves()
        if self.model.config["agent_settings"]["households"]["warning_response"]:
            # TODO: Temporarily disabled - missing infrastructure data
            # self.load_critical_infrastructure()
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
        print(f"SENSITIVITY ANALYSIS COMPLETE")
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
        print(f"SENSITIVITY ANALYSIS COMPLETE")
        print(f"Output folder: {analyzer.output_folder}")
        print(f"{'=' * 80}\n")

        return results_df

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
                "agents/households/municipal_water_demand_per_capita_m3_baseline"
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

    def assign_households_to_postal_codes(self) -> None:
        """This function associates the household points with their postal codes to get the correct geometry for the warning function."""
        households = self.var.household_points.copy()

        # Associate households with their postal codes to use it later in the warning function
        postal_codes: gpd.GeoDataFrame = gpd.read_parquet(
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

        self.var.household_points = households_with_postal_codes

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

            print(f"Created river buffers in Mercator projection for masking")
        else:
            print(f"Rivers file not found at {rivers_path} - skipping river masking")

        members = []
        flood_before = None
        flood_after = None
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

                    # Store for debug plot (first member only)
                    flood_before = flood_map_da.copy(deep=True)
                    rivers_gdf = gdf_buffered

                # Apply mask: set river pixels (where river_mask == True) to 0
                masked_flood_map_da = flood_map_da.where(~self.river_mask_da, 0)

                # Store for debug plot (first member only)
                if flood_after is None and flood_before is not None:
                    flood_after = masked_flood_map_da.copy(deep=True)
            members.append(masked_flood_map_da)

        # Concatenate all masked members
        ensemble_flood_maps = xr.concat(members, dim="member")
        if river_mask_array is not None:
            print(f"\n=== VERIFICATION: River masking across all members ===")
            for member_idx in range(ensemble_flood_maps.sizes["member"]):
                member_data = ensemble_flood_maps.isel(member=member_idx).values
                river_pixels = member_data[river_mask_array]
                n_nonzero = np.sum(river_pixels > 0)
                print(
                    f"Member {member_idx + 1}: {n_nonzero} non-zero pixels in river mask"
                )
                if n_nonzero > 0:
                    print(
                        f"  WARNING: Member {member_idx + 1} has non-zero values in river!"
                    )
            print("=== END VERIFICATION ===\n")

        # Create debug plot if masking was applied
        if (
            flood_before is not None
            and flood_after is not None
            and river_mask_array is not None
        ):
            import matplotlib.pyplot as plt

            mask = np.asarray(river_mask_array, dtype=bool)
            da = (
                masked_flood_map_da.isel(time=0)
                if "time" in masked_flood_map_da.dims
                else masked_flood_map_da
            )
            rows, cols = np.where(mask)
            vals = da.values[rows, cols]
            xs = da.x.values[cols]
            ys = da.y.values[rows]
            fig, ax = plt.subplots(figsize=(8, 8))
            sc = ax.scatter(xs, ys, c=vals, cmap="plasma", vmin=0, vmax=1, s=10)
            ax.set_title("prob_data on river mask (mask==True)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(sc, ax=ax, label="Probability")
            plt.tight_layout()
            fig.savefig(
                self.model.output_folder
                / f"debug_river_mask_values_{date_time.strftime('%Y%m%dT%H%M%S')}.png",
                dpi=200,
            )
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

            axes[0].imshow(river_mask_array, cmap="gray")
            axes[0].set_title("River mask")

            axes[1].imshow(flood_before.squeeze(), cmap="viridis", vmin=0, vmax=1)
            axes[1].set_title("Flood BEFORE masking")

            axes[2].imshow(flood_after.squeeze(), cmap="viridis", vmin=0, vmax=1)
            axes[2].set_title("Flood AFTER masking")

            if rivers_gdf is not None:
                rivers_gdf.boundary.plot(ax=axes[2], color="red", linewidth=0.4)

            for ax in axes:
                ax.set_axis_off()

            debug_path = (
                self.model.output_folder
                / f"debug_river_removal_{date_time.strftime('%Y%m%dT%H%M%S')}.png"
            )
            fig.savefig(debug_path, dpi=200)
            plt.close(fig)
            print(f"Debug plot saved to {debug_path}")

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

            # DEBUG: Print ensemble flood map statistics
            print(f"DEBUG - Range {range_id}: wl_min={wl_min}, wl_max={wl_max}")
            print(f"DEBUG - Ensemble shape: {daily_ensemble.shape}")
            print(f"DEBUG - Ensemble min: {daily_ensemble.min().values}")
            print(f"DEBUG - Ensemble max: {daily_ensemble.max().values}")
            print(f"DEBUG - Ensemble mean: {daily_ensemble.mean().values}")
            print(
                f"DEBUG - Non-zero values in ensemble: {(daily_ensemble > 0).sum().values}"
            )

            if wl_max is not None:
                condition = (daily_ensemble >= wl_min) & (daily_ensemble <= wl_max)
                print(f"DEBUG - Using range: {wl_min} <= x <= {wl_max}")
            else:
                condition = daily_ensemble >= wl_min
                print(f"DEBUG - Using exceedance: x >= {wl_min}")

            print(f"DEBUG - Condition true count: {condition.sum().values}")
            probability = condition.sum(dim="member") / condition.sizes["member"]
            print(f"DEBUG - Probability min: {probability.min().values}")
            print(f"DEBUG - Probability max: {probability.max().values}")
            print(f"DEBUG - Probability mean: {probability.mean().values}")
            print(
                f"DEBUG - Non-zero probability pixels: {(probability > 0).sum().values}"
            )
            if probability.y.values[0] < probability.y.values[-1]:
                print("flipping y axis so it can be used for zonal stats later")
                probability = probability.sortby("y", ascending=False)

                # Also flip the stored river mask to maintain alignment
                if hasattr(self, "river_mask_da"):
                    self.river_mask_da = self.river_mask_da.sortby("y", ascending=False)

            # VERIFICATION 2: Sample probability values in river mask area
            # Use the stored river mask that was applied during ensemble loading
            river_mask_array: np.ndarray | None = None

            if hasattr(self, "river_mask_da"):
                # Reuse the stored river mask (already aligned after potential y-flip)
                river_mask_array = self.river_mask_da.values.astype(bool)
                print(
                    f"\n=== RIVER MASK VERIFICATION (in create_flood_probability_maps) ==="
                )
                print(f"River mask pixels: {np.sum(river_mask_array)} pixels")
            else:
                print("No river mask available for verification")
                river_mask_array = None

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

            probability = probability.astype(np.float32)
            probability = probability.rio.write_nodata(np.nan)
            probability = write_zarr(da=probability, path=file_path, crs=crs)

            probability_maps[(date_time, range_id)] = probability
            # VERIFICATION 2: After saving and reloading
        if river_mask_array is not None:
            # Reload the saved probability map
            reloaded_prob = open_zarr(file_path)

            # Use stored river mask for verification (should be aligned)
            if hasattr(self, "river_mask_da"):
                # The stored mask should already be aligned to the reloaded data
                river_mask_reloaded = self.river_mask_da.values.astype(bool)
                prob_values_reloaded = reloaded_prob.values[river_mask_reloaded]

            else:
                print("WARNING: No stored river mask available for verification")
                prob_values_reloaded = np.array([])

            print(f"\n=== AFTER SAVING (Range {range_id}) ===")
            if len(prob_values_reloaded) > 0:
                print(f"  Non-zero pixels in river: {np.sum(prob_values_reloaded > 0)}")
                print(
                    f"  Max probability in river: {np.nanmax(prob_values_reloaded):.6f}"
                )
            else:
                print("  Could not verify river mask alignment")

            # Create debug plot if there are non-zero values after saving
            if len(prob_values_reloaded) > 0 and np.sum(prob_values_reloaded > 0) > 0:
                import matplotlib.pyplot as plt

                mask_2d = np.asarray(river_mask_reloaded, dtype=bool)
                rows, cols = np.where(mask_2d)
                vals = reloaded_prob.values[rows, cols]
                xs = reloaded_prob.x.values[cols]
                ys = reloaded_prob.y.values[rows]

                fig, ax = plt.subplots(figsize=(10, 8))
                sc = ax.scatter(xs, ys, c=vals, cmap="plasma", vmin=0, vmax=1, s=10)
                ax.set_title(
                    f"Probability in river AFTER SAVING (Range {range_id})\n"
                    f"Non-zero: {np.sum(vals > 0)}/{len(vals)} pixels"
                )
                ax.set_xlabel("x (longitude)")
                ax.set_ylabel("y (latitude)")
                plt.colorbar(sc, ax=ax, label="Probability")
                plt.tight_layout()

                debug_path = (
                    self.model.output_folder
                    / f"debug_probability_SAVED_range{range_id}_{date_time.strftime('%Y%m%dT%H%M%S')}.png"
                )
                fig.savefig(debug_path, dpi=200)
                plt.close(fig)
                print(f"  Debug plot saved to {debug_path}")

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
        exceedance: bool = False,
        strategy_id: int = 1,
    ) -> None:
        """Implements the water level warning strategy based on flood probability maps.

        Args:
            date_time: The forecast date time for which to implement the warning strategy.
            exceedance: Whether to use exceedance probability maps (True) or regular probability maps (False).
            strategy_id: Identifier of the warning strategy (1 for water level ranges with measures).
        """
        # Get warning system config settings
        warning_config = self.model.config["agent_settings"]["households"][
            "warning_system"
        ]

        # Check if water level warnings are enabled
        if not warning_config["strategies"]["water_level_warnings"]:
            print(
                f"Water level warnings disabled in config for {date_time.isoformat()}"
            )
            return

        prob_threshold = warning_config["probability_threshold"]
        area_threshold = warning_config["area_threshold"]
        building_threshold = warning_config["building_threshold"]
        warning_type = warning_config["strategies"]["warning_type"]

        # Get the range ids and initialize the warning_log
        range_ids = list(self.var.wlranges_and_measures.keys())
        warning_log = []

        # Create probability maps with exceedance=True for water level warnings
        # self.create_flood_probability_maps(
        #     strategy=strategy_id, date_time=date_time, exceedance=exceedance
        # )

        # Load households and postal codes
        households = self.var.household_points.copy()
        postal_codes = self.postal_codes.copy()

        # Clip postal codes to catchment boundary to avoid issues with areas outside the flood model domain
        catchment_boundary = gpd.read_parquet(
            self.model.files["geom"]["routing/subbasins"]
        )
        merged_watershed = catchment_boundary.dissolve()
        merged_watershed_geom = unary_union(merged_watershed.geometry)
        merged_watershed_one = (
            Polygon(merged_watershed_geom.exterior)
            if isinstance(merged_watershed_geom, Polygon)
            else merged_watershed_geom
        )
        watershed_gdf = gpd.GeoDataFrame(
            geometry=[merged_watershed_one], crs=catchment_boundary.crs
        )
        # Ensure both have the same CRS
        if postal_codes.crs != watershed_gdf.crs:
            watershed_gdf = watershed_gdf.to_crs(postal_codes.crs)

        # Clip postal codes to catchment boundary
        postal_codes = gpd.clip(postal_codes, watershed_gdf)
        print(
            f"Clipped postal codes to catchment boundary: {len(postal_codes)} postal codes remaining"
        )
        # Maybe load this as a global var (?) instead of loading it each time

        # Add education and income data to the households
        # IMPORTANT: We use the index of households to correctly map the data
        # because household_points.index corresponds to the arrays in self.var
        if "Education" not in households.columns:
            households["Education"] = households.index.map(
                lambda idx: self.var.education_level.data[idx]
                if idx < len(self.var.education_level.data)
                else np.nan
            )
        if "Income" not in households.columns:
            households["Income"] = households.index.map(
                lambda idx: self.var.income.data[idx]
                if idx < len(self.var.income.data)
                else np.nan
            )

        for range_id in range_ids:
            # Build path to probability map dynamically based on exceedance parameter
            if exceedance:
                folder_name = "flood_prob_exceedance_maps"
                file_prefix = "prob_exceedance_map"
            else:
                folder_name = "flood_prob_maps"
                file_prefix = "prob_map"

            prob_map_path = Path(
                self.model.output_folder
                / folder_name
                / f"forecast_{date_time.isoformat().replace(':', '').replace('-', '')}"
                / f"{file_prefix}_range{range_id}_strategy{strategy_id}.zarr"
            )

            # Open the probability map
            prob_map = read_zarr(prob_map_path)
            affine = prob_map.rio.transform()
            prob_map_array = np.asarray(prob_map.values)

            # DEBUG: Check what values are actually in the probability map
            print(f"DEBUG - Prob map array info:")
            print(f"       Shape: {prob_map_array.shape}")
            print(f"       Min: {np.nanmin(prob_map_array)}")
            print(f"       Max: {np.nanmax(prob_map_array)}")
            print(f"       NaN count: {np.sum(np.isnan(prob_map_array))}")
            print(f"       Zero count: {np.sum(prob_map_array == 0)}")
            print(f"       Non-zero count: {np.sum(prob_map_array > 0)}")

            # Get the pixel values for each postal code
            # Don't treat any specific value as nodata, let zonal_stats handle all values
            stats = zonal_stats(
                postal_codes,
                prob_map_array,
                affine=affine,
                raster_out=True,
                all_touched=True,
                nodata=None,  # Don't exclude any values automatically
            )

            # Iterate through each postal code and check warning criteria
            for i, postalcode in enumerate(stats):
                pixel_values = postalcode["mini_raster_array"]
                postal_code = postal_codes.iloc[i]["postcode"]

                # Handle masking more carefully - only exclude actual NaN values
                if hasattr(pixel_values, "mask"):
                    # If it's a masked array, get unmasked values
                    postalcode_pixels = pixel_values[~pixel_values.mask]
                    n_masked = np.sum(pixel_values.mask)
                else:
                    # If it's a regular array, exclude only NaN values
                    valid_mask = ~np.isnan(pixel_values)
                    postalcode_pixels = pixel_values[valid_mask]
                    n_masked = np.sum(~valid_mask)

                # Calculate the total number of valid pixels within the postal code
                n_total = len(postalcode_pixels)

                if n_total == 0:
                    # Enhanced debug info to understand why this occurs
                    print(f"DEBUG: No valid pixels found for postal code {postal_code}")
                    if hasattr(pixel_values, "mask"):
                        print(
                            f"       Unique values in masked array: {np.unique(pixel_values.data[~pixel_values.mask]) if np.any(~pixel_values.mask) else 'All masked'}"
                        )
                    else:
                        print(
                            f"       Unique values in array: {np.unique(pixel_values[~np.isnan(pixel_values)]) if np.any(~np.isnan(pixel_values)) else 'All NaN'}"
                        )
                    continue

                # Choose warning criteria based on warning type
                if warning_type == "impact_based":
                    # For impact-based warnings, check percentage of flooded buildings
                    # Get buildings in this postal code
                    postal_code_geometry = postal_codes.iloc[i]["geometry"]
                    buildings_in_postcode = self.buildings[
                        self.buildings.geometry.intersects(postal_code_geometry)
                    ]

                    if len(buildings_in_postcode) == 0:
                        print(
                            f"DEBUG: No buildings found for postal code {postal_code}"
                        )
                        continue

                    # Calculate flooded buildings using same pixel values but different threshold
                    # Get the centroid of each building and sample probability values
                    building_centroids = buildings_in_postcode.geometry.centroid

                    # Sample probability values at building locations
                    building_probs = []
                    for building_centroid in building_centroids:
                        # Extract flood value at building location using xarray selection
                        flood_value = prob_map.sel(
                            x=building_centroid.x,
                            y=building_centroid.y,
                            method="nearest",
                        ).values

                        # Add valid flood values to the list
                        if not np.isnan(flood_value):
                            building_probs.append(flood_value)

                    if len(building_probs) == 0:
                        print(
                            f"DEBUG: No valid building probabilities for postal code {postal_code}"
                        )
                        continue

                    # Calculate percentage of flooded buildings (using building_threshold from config)
                    # Use total buildings in postcode (not just those with valid flood values)
                    n_buildings_total = len(buildings_in_postcode)
                    n_buildings_flooded = np.sum(
                        np.array(building_probs) >= prob_threshold
                    )
                    building_percentage = n_buildings_flooded / n_buildings_total

                    issue_warning = building_percentage >= building_threshold

                    if issue_warning:
                        print(
                            f"Warning issued to postal code {postal_code} on {date_time.strftime('%d-%m-%Y T%H:%M:%S')} for range {range_id}: {building_percentage:.0%} of {n_buildings_total} buildings flooded (threshold: {building_threshold:.0%})"
                        )
                else:
                    # For non-impact-based warnings, use original pixel-based approach
                    # Calculate the number of pixels with flood prob above the threshold
                    n_above = np.sum(postalcode_pixels >= prob_threshold)
                    percentage = n_above / n_total

                    issue_warning = percentage >= area_threshold

                    if issue_warning:
                        print(
                            f"Warning issued to postal code {postal_code} on {date_time.strftime('%d-%m-%Y T%H:%M:%S')} for range {range_id}: {percentage:.0%} of pixels > {prob_threshold}"
                        )

                # If warning should be issued, process affected households
                if issue_warning:
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
                        triggered_wlrange=range_id,
                    )

                    # Store warning log details based on warning type
                    if warning_type == "impact_based":
                        warning_log.append(
                            {
                                "date_time": date_time.isoformat(),
                                "postcode": postal_code,
                                "range": range_id,
                                "n_affected_households": len(affected_households),
                                "n_warned_households": n_warned_households,
                                "warning_type": warning_type,
                                "n_buildings_total": n_buildings_total,
                                "n_buildings_flooded": n_buildings_flooded,
                                "building_percentage": f"{building_percentage:.2f}",
                                "building_threshold": f"{building_threshold:.2f}",
                            }
                        )
                    else:
                        warning_log.append(
                            {
                                "date_time": date_time.isoformat(),
                                "postcode": postal_code,
                                "range": range_id,
                                "n_affected_households": len(affected_households),
                                "n_warned_households": n_warned_households,
                                "warning_type": warning_type,
                                "percentage_above_threshold": f"{percentage:.2f}",
                            }
                        )

        # Save the warning log to a csv file
        warning_config = self.model.config["agent_settings"]["households"][
            "warning_system"
        ]
        subfolder_name = warning_config["strategies"]["warning_type"]
        warnings_folder = self.model.output_folder / "warning_logs" / subfolder_name
        warnings_folder.mkdir(exist_ok=True, parents=True)
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
        households = self.var.household_points.copy()

        # Add education and income data here as well for consistency
        if "Education" not in households.columns:
            households["Education"] = households.index.map(
                lambda idx: self.var.education_level.data[idx]
                if idx < len(self.var.education_level.data)
                else np.nan
            )
        if "Income" not in households.columns:
            households["Income"] = households.index.map(
                lambda idx: self.var.income.data[idx]
                if idx < len(self.var.income.data)
                else np.nan
            )

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

    def warning_communication(
        self,
        target_households: gpd.GeoDataFrame,
        measures: set[str],
        evacuate: bool,
        trigger: str,
        triggered_wlrange: int | None = None,
    ) -> int:
        """Send warnings to a subset of target households according to communication_efficiency.

        Makes sure to send a single message and allows only upward escalation:
        none (0) -> measures (1) -> evacuate (2)
        Evacuation warning only sent if evacuate is True and lead_time <= threshold.

        Args:
            target_households: GeoDataFrame of household points targeted by the warning.
            measures: Set of recommended protective measures to communicate (strings).
            evacuate: Whether evacuation should be advised for this warning.
            trigger: Identifier of the trigger that initiated the warning.
            triggered_wlrange: The water level range that triggered the warning, if applicable.

        Returns:
            int: Number of households that were successfully warned.
        """
        print("Running the warning communication for households...")

        # Get warning system config settings
        warning_config = self.model.config["agent_settings"]["households"][
            "warning_system"
        ]
        communication_efficiency = warning_config["communication_efficiency"]
        lt_threshold_evacuation = warning_config["evacuation_lead_time_threshold"]

        # Get the number of target households
        n_target_households = len(target_households)
        if n_target_households == 0:
            print("No target households to warn.")
            return 0

        # Calculate individual communication efficiency probabilities based on socio-economic factors
        warning_weights = self.calculate_communication_efficiency_probability(
            target_households
        )

        # Normalize weights to ensure they sum to exactly 1.0 (avoid floating point precision errors)
        warning_weights = warning_weights / warning_weights.sum()

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
                replace=False,  # Each household can only be selected once for warning
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

            # DEBUG: Print what measures are being passed and what's in implementation_times
            print(f"DEBUG pick_recommendations:")
            print(f"  available_measures: {available_measures}")
            print(f"  implementation_times keys: {list(implementation_times.keys())}")

            # Filter out empty strings and invalid measures to prevent KeyError
            valid_measures = {
                m for m in available_measures if m and m in implementation_times
            }
            print(f"  valid_measures after filtering: {valid_measures}")

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
        if not hasattr(self.var, "triggered_wlrange"):
            self.var.triggered_wlrange = np.full(self.n, np.nan)

        for household_id in selected_households.index:
            # Skip al geëvacueerden voor daadwerkelijke warning
            if self.var.evacuated[household_id] == 1:
                continue
            current_level = int(self.var.warning_level[household_id])
            # Log potentiele actie per household en range (altijd, ongeacht warning)
            if not hasattr(self.var, "action_per_range"):
                n_ranges = len(self.var.wlranges_and_measures)
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
                print(
                    f"DEBUG: household_id={household_id}, oude triggered_wlrange={old_val}, nieuwe triggered_wlrange={triggered_wlrange}, trigger={trigger}"
                )
                self.var.triggered_wlrange[household_id] = triggered_wlrange
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

    def household_decision_making(self, date_time: datetime) -> None:
        """Simulate household emergency response decisions based on warnings and lead time.

        Args:
            date_time: The forecast date time for which to run the decision-making.

        Raises:
            ValueError: If an unknown warning type is specified in the configuration.
        """
        print("\n=== RUNNING HOUSEHOLD DECISION-MAKING ===")
        print(f"Date/time: {date_time}")

        # Get warning system config settings
        warning_config = self.model.config["agent_settings"]["households"][
            "warning_system"
        ]
        warning_type = warning_config["strategies"]["warning_type"]
        print(f"Warning type: {warning_type}")

        # Determine response rate based on warning type
        if warning_type == "impact_based":
            responsive_ratio = warning_config["response_rates"]["impact_based_warnings"]

        elif warning_type == "flood_general":
            responsive_ratio = warning_config["response_rates"][
                "flood_general_warnings"
            ]
        else:
            raise ValueError(
                f"Unknown warning type: {warning_type} selected in config, choose 'impact_based' or 'flood_general'."
            )

        print(f"Responsive ratio: {responsive_ratio}")

        # Get lead time in hours
        lead_time = self.compute_lead_time()
        print(f"Lead time: {lead_time:.1f} hours")

        # Filter households that did not evacuate, were warned and are responsive
        not_evacuated_ids = self.var.evacuated == 0
        warned_ids = self.var.warning_reached == 1
        responsive_ids = self.var.response_probability < responsive_ratio

        print(f"\n--- HOUSEHOLD FILTERING ---")
        print(f"Total households: {len(self.var.household_points)}")
        print(f"Not evacuated: {np.sum(not_evacuated_ids)}")
        print(f"Warned: {np.sum(warned_ids)}")
        print(f"Responsive: {np.sum(responsive_ids)}")

        if np.sum(warned_ids) > 0:
            print(
                f"Households that received warnings: {np.where(warned_ids)[0][:10]}..."
            )  # Show first 10
            # Check what recommended measures they have
            warned_household_indices = np.where(warned_ids)[0][:5]  # Check first 5
            print("\nRecommended measures for first few warned households:")
            for idx in warned_household_indices:
                measures_for_hh = self.var.recommended_measures[idx]
                if np.any(measures_for_hh):
                    recommended_actions = [
                        self.var.possible_measures[i]
                        for i, has_measure in enumerate(measures_for_hh)
                        if has_measure
                    ]
                    print(f"  Household {idx}: {recommended_actions}")
                else:
                    print(f"  Household {idx}: No measures recommended")

        if np.sum(responsive_ids) == 0:
            print(
                f"WARNING: No households are responsive (all response_probability >= {responsive_ratio})"
            )
            print(
                f"Response probabilities range: {self.var.response_probability.min():.3f} - {self.var.response_probability.max():.3f}"
            )

        # Combine the filters and apply it to household_points
        eligible_ids = np.asarray(warned_ids & not_evacuated_ids & responsive_ids)
        eligible_households = self.var.household_points.loc[eligible_ids]

        print(f"Eligible households for action: {len(eligible_households)}")

        # Get the list of possible actions for eligible households
        # (this is a list of all possible actions, not the recommended actions given by the warning strategies)
        possible_actions = self.var.possible_measures
        evac_idx = possible_actions.index("evacuate")

        # Initialize an empty list to log actions taken
        actions_log = []

        print(f"\n--- PROCESSING ELIGIBLE HOUSEHOLDS ---")

        # For every eligible household, do the recommended measures in the communicated warning
        for i, household_id in enumerate(eligible_households.index):
            print(
                f"\nProcessing household {household_id} ({i + 1}/{len(eligible_households)})"
            )

            # Check what was recommended for this household
            recommended_for_hh = self.var.recommended_measures[household_id]
            recommended_actions = [
                self.var.possible_measures[i]
                for i, has_measure in enumerate(recommended_for_hh)
                if has_measure
            ]
            print(f"  Recommended actions: {recommended_actions}")

            # For measure in recommended measures, change it to true in the actions_taken array
            self.var.actions_taken[household_id] = self.var.recommended_measures[
                household_id
            ].copy()

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
                    "postal_code": self.var.household_points.loc[
                        household_id, "postcode"
                    ],
                    "household_id": household_id,
                    "actions": actions,
                    "warning_level": warning_level,
                    "triggered_wlrange": triggered_wlrange,
                    "action_per_range": action_per_range,
                }
            )

        print(f"\n=== SUMMARY ===")
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

        # keep a full copy for model-wide use
        self.buildings_all = self.buildings.copy()

        # create a filtered view based on household type/occupancy if available
        # Keep only rows where occupancy is 'UNK' or contains 'RES'
        if "occupancy" in self.buildings.columns:
            self.buildings = self.buildings_all.copy()
            mask = (self.buildings["occupancy"] == "UNK") | self.buildings[
                "occupancy"
            ].str.contains("RES", case=False, na=False)
            print(f"Filtered buildings count (RES or UNK kept): {mask.sum()}")
            self.buildings = self.buildings[mask]

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
        # DISABLED: Sandbags measure not being used
        # self.buildings_structure_curve["building_with_sandbags"] = (
        #     self.buildings_structure_curve["building_unprotected"] * 0.85
        # )

        # create another column (curve) in the buildings structure curve for
        # protected buildings with elevated possessions -- no effect on structure
        self.buildings_structure_curve["building_elevated_possessions"] = (
            self.buildings_structure_curve["building_unprotected"]
        )

        # create another column (curve) in the buildings structure curve for
        # protected buildings with both sandbags and elevated possessions -- only sandbags have an effect on structure
        # DISABLED: Sandbags measure not being used - using unprotected curve instead
        # self.buildings_structure_curve["building_all_forecast_based"] = (
        #    self.buildings_structure_curve["building_with_sandbags"]
        # )

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

        # TODO: need to adjust the vulnerability curves
        # create another column (curve) in the buildings content curve for
        # protected buildings with sandbags
        # DISABLED: Sandbags measure not being used
        # self.buildings_content_curve["building_with_sandbags"] = (
        #     self.buildings_content_curve["building_unprotected"] * 0.85
        # )

        # create another column (curve) in the buildings content curve for
        # protected buildings with elevated possessions
        self.buildings_content_curve["building_elevated_possessions"] = (
            self.buildings_content_curve["building_unprotected"] * 0.80
        )

        # create another column (curve) in the buildings content curve for
        # protected buildings with both sandbags and elevated possessions
        self.buildings_content_curve["building_all_forecast_based"] = (
            self.buildings_content_curve["building_unprotected"] * 0.80
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
                "damages": self.buildings_structure_curve["building_unprotected"],
                "damages_flood_proofed": self.buildings_structure_curve[
                    "building_flood_proofed"
                ],
            }
            damage_buildings: pd.DataFrame = VectorScannerMultiCurves(
                features=building_multicurve.rename(
                    columns={"maximum_damage_m2": "maximum_damage"}
                ),
                hazard=flood_map,
                multi_curves=multi_curves,
            )
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

    def update_households_geodataframe_w_warning_variables(
        self, date_time: datetime
    ) -> None:
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
            "triggered_wlrange",
            "action_per_range",
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
            # DISABLED: Sandbags measure not being used
            # household_points["sandbags"] = False
            household_points["elevated_possessions"] = False

            # mark households that took protective actions
            household_points.loc[
                np.asarray(self.var.actions_taken)[:, 0] == 1, "elevated_possessions"
            ] = True
            # DISABLED: Sandbags measure not being used
            # household_points.loc[
            #     np.asarray(self.var.actions_taken)[:, 1] == 1, "sandbags"
            # ] = True

            # spatial join to get household attributes to buildings
            buildings: gpd.GeoDataFrame = gpd.sjoin_nearest(
                buildings, household_points, how="left", exclusive=True
            )

            # Assign object types for buildings based on protective measures taken
            buildings["object_type"] = "building_unprotected"  # reset
            buildings.loc[buildings["elevated_possessions"], "object_type"] = (
                "building_elevated_possessions"
            )
            # DISABLED: Sandbags measure not being used
            # buildings.loc[buildings["sandbags"], "object_type"] = (
            #     "building_with_sandbags"
            # )
            # buildings.loc[
            #     buildings["elevated_possessions"] & buildings["sandbags"], "object_type"
            # ] = "building_all_forecast_based"
            # TODO: need to move the update of the actions takens by households to outside the flood function

            # Save the buildings with actions taken
            # Create the action_maps directory if it doesn't exist
            action_maps_dir = self.model.output_folder / "action_maps"
            action_maps_dir.mkdir(parents=True, exist_ok=True)

            buildings.to_parquet(
                action_maps_dir / "buildings_with_protective_measures.geoparquet"
            )

            # Assign object types for buildings centroid based on protective measures taken
            buildings_centroid = household_points.to_crs(flood_depth.rio.crs)
            # DISABLED: Sandbags measure not being used - simplified logic
            buildings_centroid["object_type"] = np.select(
                [
                    buildings_centroid["elevated_possessions"],
                ],
                [
                    "building_elevated_possessions",
                ],
                default="building_unprotected",
            )
            # Original code with sandbags (disabled):
            # buildings_centroid["object_type"] = np.select(
            #     [
            #         (
            #             buildings_centroid["elevated_possessions"]
            #             & buildings_centroid["sandbags"]
            #         ),
            #         buildings_centroid["elevated_possessions"],
            #         buildings_centroid["sandbags"],
            #     ],
            #     [
            #         "building_all_forecast_based",
            #         "building_elevated_possessions",
            #         "building_with_sandbags",
            #     ],
            #     default="building_unprotected",
            # )
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
                    "flood_proofed"
                ].apply(lambda x: "building_protected" if x else "building_unprotected")
                buildings_centroid["maximum_damage"] = (
                    self.var.max_dam_buildings_content
                )
            else:
                # Adaptation is disabled: all buildings are unprotected
                buildings["object_type"] = "building_unprotected"

                buildings_centroid = household_points.to_crs(flood_depth.rio.crs)
                buildings_centroid["object_type"] = "building_unprotected"
                buildings_centroid["maximum_damage"] = (
                    self.var.max_dam_buildings_content
                )

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
