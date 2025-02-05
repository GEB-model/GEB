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
from os.path import join
from damagescanner.core import object_scanner
import json
import rioxarray
from rasterio.features import shapes
from shapely.geometry import shape


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
        self.HRU = model.data.HRU
        self.grid = model.data.grid
        self.agents = agents
        self.reduncancy = reduncancy

        if self.model.spinup:
            self.spinup()

    def spinup(self):
        self.var = self.model.store.create_bucket("agents.households.var")

        # Load buildings
        self.var.buildings = gpd.read_file(
            self.model.files["geoms"]["assets/buildings"]
        )
        self.var.buildings["object_type"] = "building_structure"
        self.var.buildings_centroid = gpd.GeoDataFrame(
            geometry=self.var.buildings.centroid
        )
        self.var.buildings_centroid["object_type"] = "building_content"

        # Load roads
        self.var.roads = gpd.read_file(
            self.model.files["geoms"]["assets/roads"]
        ).rename(columns={"highway": "object_type"})

        # Load rail
        self.var.rail = gpd.read_file(self.model.files["geoms"]["assets/rails"])
        self.var.rail["object_type"] = "rail"

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

        # Load vulnerability curves
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

        self.var.rail_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/rail/main/curve"]
        )
        self.var.rail_curve.set_index("severity", inplace=True)
        self.var.rail_curve = self.var.rail_curve.rename(
            columns={"damage_ratio": "rail"}
        )

        super().__init__()

        water_demand, efficiency = self.update_water_demand()
        self.var.current_water_demand = water_demand
        self.var.current_efficiency = efficiency

        locations = np.load(self.model.files["binary"]["agents/households/locations"])[
            "data"
        ]
        self.max_n = int(locations.shape[0] * (1 + self.reduncancy) + 1)

        self.var.locations = DynamicArray(locations, max_n=self.max_n)

        sizes = np.load(self.model.files["binary"]["agents/households/sizes"])["data"]
        self.var.sizes = DynamicArray(sizes, max_n=self.max_n)

    def flood(self, flood_map, simulation_root, return_period=None):
        if return_period is not None:
            flood_path = join(simulation_root, f"hmax RP {int(return_period)}.tif")
        else:
            flood_path = join(simulation_root, "hmax.tif")

        print(f"using this flood map: {flood_path}")
        flood_map = rioxarray.open_rasterio(flood_path)

        agriculture = from_landuse_raster_to_polygon(
            self.model.data.HRU.decompress(self.model.data.HRU.var.land_owners != -1),
            self.model.data.HRU.transform,
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
            self.model.data.HRU.decompress(
                self.model.data.HRU.var.land_use_type == FOREST
            ),
            self.model.data.HRU.transform,
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
        downscale_mask = self.HRU.var.land_use_type != SEALED
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
                shape=self.model.data.grid.shape,
                transform=self.model.data.grid.transform,
            )
            / (water_demand.rio.transform().a / self.model.data.grid.transform.a) ** 2
        )
        water_demand = downscale_volume(
            water_demand.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_demand.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.var.land_use_ratio,
        )
        water_demand = self.model.data.HRU.M3toM(water_demand)

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
                shape=self.model.data.grid.shape,
                transform=self.model.data.grid.transform,
            )
            / (water_consumption.rio.transform().a / self.model.data.grid.transform.a)
            ** 2
        )
        water_consumption = downscale_volume(
            water_consumption.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_consumption.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.var.land_use_ratio,
        )
        water_consumption = self.model.data.HRU.M3toM(water_consumption)

        efficiency = np.divide(
            water_consumption,
            water_demand,
            out=np.zeros_like(water_consumption, dtype=float),
            where=water_demand != 0,
        )

        efficiency = self.model.data.to_grid(HRU_data=efficiency, fn="max")

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
        return None

    @property
    def n(self):
        return self.locations.shape[0]
