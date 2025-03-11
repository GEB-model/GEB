import numpy as np
import geopandas as gpd
import calendar
from .general import downscale_volume, AgentBaseClass
from ..store import DynamicArray
from ..hydrology.landcover import (
    SEALED,
    FOREST,
)
import pandas as pd
from damagescanner.core import object_scanner
import json
import xarray as xr
from rasterio.features import shapes
from shapely.geometry import shape
from .decision_module_flood import DecisionModule
from shapely.geometry import Point
import pyproj
from honeybees.library.raster import sample_from_map
from scipy import interpolate


def from_landuse_raster_to_polygon(mask, transform, crs):
    """
    Convert raster data into separate GeoDataFrames for specified land use values.

    Parameters:
    - landuse: An xarray DataArray or similar with land use data and 'x' and 'y' coordinates.
    - values_to_extract: List of integer values to extract (e.g., [0, 1] for forest and agriculture).

    Returns:
    - Geodataframe
    """

    shapes_gen = shapes(mask.astype(np.uint8), mask=mask, transform=transform)

    polygons = []
    for geom, _ in shapes_gen:
        polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=crs)

    return gdf


class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
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

        if self.model.in_spinup:
            self.spinup()

    def load_flood_maps(self):
        """Load flood maps for different return periods. This might be quite ineffecient for RAM, but faster then loading them each timestep for now."""

        self.return_periods = np.array(
            self.model.config["hazards"]["floods"]["return_periods"]
        )

        flood_maps = {}
        for return_period in self.return_periods:
            file_path = (
                self.model.report_folder_root
                / "estimate_risk"
                / "flood_maps"
                / f"{return_period}.zarr.zip"
            )
            flood_maps[return_period] = xr.open_dataarray(file_path, engine="zarr")
        flood_maps["crs"] = flood_maps[return_period].rio.crs
        flood_maps["gdal_geotransform"] = (
            flood_maps[return_period].rio.transform().to_gdal()
        )
        self.flood_maps = flood_maps

    def assign_household_attributes(self):
        """Household locations are already sampled from population map in GEBModel.setup_population()
        These are loaded in the spinup() method.
        Here we assign additional attributes (dummy data) to the households that are used in the decision module."""

        # load household locations
        locations = np.load(self.model.files["binary"]["agents/households/locations"])[
            "data"
        ]
        self.max_n = int(locations.shape[0] * (1 + self.reduncancy) + 1)
        self.var.locations = DynamicArray(locations, max_n=self.max_n)

        # load household sizes
        sizes = np.load(self.model.files["binary"]["agents/households/sizes"])["data"]
        self.var.sizes = DynamicArray(sizes, max_n=self.max_n)

        # initiate array for adaptation status [0=not adapted, 1=dryfloodproofing implemented]
        self.var.adapted = DynamicArray(np.zeros(self.n, np.int32), max_n=self.max_n)

        # initiate array with household incomes [dummy data for now]
        self.var.income = DynamicArray(
            np.random.randint(16_000, 30_000, self.n), max_n=self.max_n
        )

        # initiate array with houshold wealth [dummy data for now]
        self.var.wealth = DynamicArray(
            np.int64(self.var.income.data * 2.8), max_n=self.max_n
        )

        # initiate array with property values (used as max damage) [dummy data for now]
        self.var.property_value = DynamicArray(
            np.int64(self.var.wealth.data * 0.8), max_n=self.max_n
        )
        # initiate array with RANDOM adaptation costs [dummy data for now]
        self.var.adaptation_costs = DynamicArray(
            np.int64(self.var.property_value.data * 0.1), max_n=self.max_n
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

        # initiate array with amenity value [dummy data for now]
        amenity_premiums = np.random.uniform(0, 0.2, self.n)
        self.var.amenity_value = DynamicArray(
            amenity_premiums * self.var.wealth, max_n=self.max_n
        )

        # initiate array with time adapted
        self.var.time_adapted = DynamicArray(
            np.zeros(self.n, np.int32), max_n=self.max_n
        )

        # initiate array with time since last flood
        self.var.years_since_last_flood = DynamicArray(
            np.full(self.n, 25, np.int32), max_n=self.max_n
        )

        # reproject households to flood maps and store in var bucket
        household_points = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lon, lat in self.var.locations.data],
            crs="EPSG:4326",
        )
        household_points["maximum_damage"] = self.var.property_value.data
        household_points["object_type"] = (
            "building_content"  # this must match damage curves  # this must match damage curves
        )
        self.var.household_points = household_points.to_crs(self.flood_maps["crs"])

        transformer = pyproj.Transformer.from_crs(
            self.grid.crs, self.flood_maps["crs"], always_xy=True
        )
        locations[:, 0], locations[:, 1] = transformer.transform(
            self.var.locations[:, 0], self.var.locations[:, 1]
        )
        self.var.locations_reprojected_to_flood_map = locations
        print(f"Household attributes assigned for {self.n} households.")

    def get_flood_risk_information_honeybees(self):
        # preallocate array for damages
        damages_do_not_adapt = np.zeros((self.return_periods.size, self.n), np.float32)
        damages_adapt = np.zeros((self.return_periods.size, self.n), np.float32)

        # load damage interpolators (cannot be store in bucket, therefor outside spinup)
        if not hasattr(self, "buildings_content_curve_interpolator"):
            self.create_damage_interpolators()

        # loop over return periods
        for i, return_period in enumerate(self.return_periods):
            # get flood map
            flood_map = self.flood_maps[return_period]

            water_levels = sample_from_map(
                flood_map.values,
                self.var.locations_reprojected_to_flood_map.data,
                self.flood_maps["gdal_geotransform"],
            )

            # cap water levels at damage curve max inundation
            water_levels = np.minimum(
                water_levels, self.buildings_content_curve_interpolator.x.max()
            )

            # interpolate damages
            damages_do_not_adapt[i, :] = (
                self.buildings_content_curve_interpolator(water_levels)
                * self.var.property_value.data
            )

            damages_adapt[i, :] = (
                self.buildings_content_curve_adapted_interpolator(water_levels)
                * self.var.property_value.data
            )

        return damages_do_not_adapt, damages_adapt

    def get_flood_risk_information_damage_scanner(self):
        """Initiate flood risk information for each household. This information is used in the decision module.
        For now also only dummy data is created."""

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

    def update_risk_perceptions(self):
        # update timer
        self.var.years_since_last_flood.data += 1

        # generate random flood (not based on actual modeled flood data, replace this later with events)
        if np.random.random() < 0.2:
            print("Flood event!")
            self.var.years_since_last_flood.data = 0

        self.var.risk_perception.data = (
            self.var.risk_perc_max
            * 1.6 ** (self.var.risk_decr * self.var.years_since_last_flood)
            + self.var.risk_perc_min
        )

    def decide_household_strategy(self):
        """This function calculates the utility of adapting to flood risk for each household and decides whether to adapt or not."""

        # update risk perceptions
        self.update_risk_perceptions()

        # get flood risk information
        damages_do_not_adapt, damages_adapt = (
            self.get_flood_risk_information_honeybees()
        )

        # calculate expected utilities
        EU_adapt = self.decision_module.calcEU_adapt(
            geom_id="NoID",
            n_agents=self.n,
            wealth=self.var.wealth.data,
            income=self.var.income.data,
            expendature_cap=10,  # realy high for now
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

        EU_do_not_adapt = self.decision_module.calcEU_do_nothing(
            geom_id="NoID",
            n_agents=self.n,
            wealth=self.var.wealth.data,
            income=self.var.income.data,
            expendature_cap=10,
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

        # print percentage of households that adapted
        print(
            f"Percentage of households that adapted: {len(household_adapting) / self.n * 100}%"
        )

    def load_objects(self):
        # Load buildings
        self.var.buildings = gpd.read_parquet(
            self.model.files["geoms"]["assets/buildings"]
        )
        self.var.buildings["object_type"] = "building_structure"
        self.var.buildings_centroid = gpd.GeoDataFrame(
            geometry=self.var.buildings.centroid
        )
        self.var.buildings_centroid["object_type"] = "building_content"

        # Load roads
        self.var.roads = gpd.read_parquet(
            self.model.files["geoms"]["assets/roads"]
        ).rename(columns={"highway": "object_type"})

        # Load rail
        self.var.rail = gpd.read_parquet(self.model.files["geoms"]["assets/rails"])
        self.var.rail["object_type"] = "rail"

    def load_max_damage_values(self):
        # Load maximum damages
        with open(
            self.model.files["dict"][
                "damage_parameters/flood/buildings/structure/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_buildings_structure = float(json.load(f)["maximum_damage"])
        self.var.buildings["maximum_damage"] = self.var.max_dam_buildings_structure

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
        self.var.buildings_centroid["maximum_damage"] = (
            self.var.max_dam_buildings_content
        )

        with open(
            self.model.files["dict"][
                "damage_parameters/flood/rail/main/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_rail = float(json.load(f)["maximum_damage"])
        self.var.rail["maximum_damage"] = self.var.max_dam_rail

        self.var.max_dam_road = {}
        road_types = [
            ("residential", "damage_parameters/flood/road/residential/maximum_damage"),
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
            self.var.max_dam_road[road_type] = max_damage["maximum_damage"]

        self.var.roads["maximum_damage"] = self.var.roads["object_type"].map(
            self.var.max_dam_road
        )

        with open(
            self.model.files["dict"][
                "damage_parameters/flood/land_use/forest/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_forest = float(json.load(f)["maximum_damage"])

        with open(
            self.model.files["dict"][
                "damage_parameters/flood/land_use/agriculture/maximum_damage"
            ],
            "r",
        ) as f:
            self.var.max_dam_agriculture = float(json.load(f)["maximum_damage"])

    def load_damage_curves(self):
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

        self.var.buildings_structure_curve = pd.read_parquet(
            self.model.files["table"][
                "damage_parameters/flood/buildings/structure/curve"
            ]
        )
        self.var.buildings_structure_curve.set_index("severity", inplace=True)
        self.var.buildings_structure_curve = self.var.buildings_structure_curve.rename(
            columns={"damage_ratio": "building_structure"}
        )

        self.var.buildings_content_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/buildings/content/curve"]
        )
        self.var.buildings_content_curve.set_index("severity", inplace=True)
        self.var.buildings_content_curve = self.var.buildings_content_curve.rename(
            columns={"damage_ratio": "building_content"}
        )

        # create damage curves for adaptation
        buildings_content_curve_adapted = self.var.buildings_content_curve.copy()
        buildings_content_curve_adapted.loc[0:1] = (
            0  # assuming zero damages untill 1m water depth
        )
        buildings_content_curve_adapted.loc[1:] *= (
            0.8  # assuming 80% damages above 1m water depth
        )
        self.var.buildings_content_curve_adapted = buildings_content_curve_adapted

        self.var.rail_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/rail/main/curve"]
        )
        self.var.rail_curve.set_index("severity", inplace=True)
        self.var.rail_curve = self.var.rail_curve.rename(
            columns={"damage_ratio": "rail"}
        )

    def create_damage_interpolators(self):
        # create interpolation function for damage curves [interpolation objects cannot be stored in bucket]
        self.buildings_content_curve_interpolator = interpolate.interp1d(
            x=self.var.buildings_content_curve.index,
            y=self.var.buildings_content_curve["building_content"],
            # fill_value="extrapolate",
        )
        self.buildings_content_curve_adapted_interpolator = interpolate.interp1d(
            x=self.var.buildings_content_curve_adapted.index,
            y=self.var.buildings_content_curve_adapted["building_content"],
            # fill_value="extrapolate",
        )

    def spinup(self):
        self.var = self.model.store.create_bucket("agents.households.var")
        self.load_objects()
        self.load_max_damage_values()
        self.load_damage_curves()
        if self.config["adapt"]:
            self.assign_household_attributes()

        super().__init__()

        water_demand, efficiency = self.update_water_demand()
        self.var.current_water_demand = water_demand
        self.var.current_efficiency = efficiency

    def flood(self, flood_map):
        agriculture = from_landuse_raster_to_polygon(
            self.HRU.decompress(self.HRU.var.land_owners != -1),
            self.HRU.transform,
            self.model.crs,
        )
        agriculture["object_type"] = "agriculture"
        agriculture["maximum_damage"] = self.var.max_dam_agriculture

        agriculture = agriculture.to_crs(flood_map.rio.crs)

        damages_agriculture = object_scanner(
            objects=agriculture,
            hazard=flood_map,
            curves=self.var.agriculture_curve,
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
        forest["maximum_damage"] = self.var.max_dam_forest

        forest = forest.to_crs(flood_map.rio.crs)

        damages_forest = object_scanner(
            objects=forest, hazard=flood_map, curves=self.var.forest_curve
        )
        total_damages_forest = damages_forest.sum()
        print(f"damages to forest are: {total_damages_forest}")

        buildings = self.var.buildings.to_crs(flood_map.rio.crs)
        damages_buildings_structure = object_scanner(
            objects=buildings,
            hazard=flood_map,
            curves=self.var.buildings_structure_curve,
        )
        total_damage_structure = damages_buildings_structure.sum()
        print(f"damages to building structure are: {total_damage_structure}")

        buildings_centroid = self.var.buildings_centroid.to_crs(flood_map.rio.crs)
        damages_buildings_content = object_scanner(
            objects=buildings_centroid,
            hazard=flood_map,
            curves=self.var.buildings_content_curve,
        )
        total_damages_content = damages_buildings_content.sum()
        print(f"damages to building content are: {total_damages_content}")

        roads = self.var.roads.to_crs(flood_map.rio.crs)
        damages_roads = object_scanner(
            objects=roads,
            hazard=flood_map,
            curves=self.var.road_curves,
        )
        total_damages_roads = damages_roads.sum()
        print(f"damages to roads are: {total_damages_roads} ")

        rail = self.var.rail.to_crs(flood_map.rio.crs)
        damages_rail = object_scanner(
            objects=rail,
            hazard=flood_map,
            curves=self.var.rail_curve,
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

    def update_water_demand(self):
        """
        Dynamic part of the water demand module - domestic
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """
        downscale_mask = self.model.hydrology.HRU.var.land_use_type != SEALED
        days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365
        water_demand = (
            self.model.domestic_water_demand_ds.sel(
                time=self.model.current_time, method="ffill", tolerance="366D"
            ).domestic_water_demand
            * 1_000_000
            / days_in_year
        )
        water_demand = (
            water_demand.rio.set_crs(4326).rio.reproject(
                4326,
                shape=self.model.hydrology.grid.shape,
                transform=self.model.hydrology.grid.transform,
            )
            / (water_demand.rio.transform().a / self.model.hydrology.grid.transform.a)
            ** 2
        )
        water_demand = downscale_volume(
            water_demand.rio.transform().to_gdal(),
            self.model.hydrology.grid.gt,
            water_demand.values,
            self.model.hydrology.grid.mask,
            self.model.hydrology.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.hydrology.HRU.var.land_use_ratio,
        )
        water_demand = self.model.hydrology.HRU.M3toM(water_demand)

        water_consumption = (
            self.model.domestic_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill"
            ).domestic_water_consumption
            * 1_000_000
            / days_in_year
        )
        water_consumption = (
            water_consumption.rio.set_crs(4326).rio.reproject(
                4326,
                shape=self.model.hydrology.grid.shape,
                transform=self.model.hydrology.grid.transform,
            )
            / (
                water_consumption.rio.transform().a
                / self.model.hydrology.grid.transform.a
            )
            ** 2
        )
        water_consumption = downscale_volume(
            water_consumption.rio.transform().to_gdal(),
            self.model.hydrology.grid.gt,
            water_consumption.values,
            self.model.hydrology.grid.mask,
            self.model.hydrology.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.hydrology.HRU.var.land_use_ratio,
        )
        water_consumption = self.model.hydrology.HRU.M3toM(water_consumption)

        efficiency = np.divide(
            water_consumption,
            water_demand,
            out=np.zeros_like(water_consumption, dtype=float),
            where=water_demand != 0,
        )

        efficiency = self.model.hydrology.to_grid(HRU_data=efficiency, fn="max")

        assert (efficiency <= 1).all()
        assert (efficiency >= 0).all()
        self.var.last_water_demand_update = self.model.current_time
        return water_demand, efficiency

    def water_demand(self):
        if (
            np.datetime64(self.model.current_time, "ns")
            in self.model.domestic_water_consumption_ds.time
        ):
            water_demand, efficiency = self.update_water_demand()
            self.var.current_water_demand = water_demand
            self.var.current_efficiency = efficiency

        assert (
            self.model.current_time - self.var.last_water_demand_update
        ).days < 366, (
            "Water demand has not been updated for over a year. "
            "Please check the household water demand datasets."
        )
        return self.var.current_water_demand, self.var.current_efficiency

    def step(self) -> None:
        if (
            self.config["adapt"]
            and self.model.current_time.month == 1
            and self.model.current_time.day == 1
        ):
            print("Thinking about adapting...")
            self.decide_household_strategy()

    @property
    def n(self):
        return self.var.locations.shape[0]
