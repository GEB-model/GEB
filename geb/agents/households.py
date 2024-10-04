import numpy as np
import geopandas as gpd
import pyproj
import calendar
from .general import AgentArray, downscale_volume, AgentBaseClass
from ..hydrology.landcover import SEALED
import pandas as pd
from os.path import join
from damagescanner.core import RasterScanner
from damagescanner.core import VectorScanner
import json

# from damagescanner.core import RasterScanner
# from damagescanner.core import VectorScanner

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass


class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents
        self.reduncancy = reduncancy
        self.config = self.model.config

        if self.model.config["hazards"]["damage"]["simulate"]:
            # Load exposure data
            if "assets/buildings" in self.model.files["geoms"]:
                self.buildings = gpd.read_file(
                    self.model.files["geoms"]["assets/buildings"]
                )
            else:
                self.buildings = None

            if "assets/roads" in self.model.files["geoms"]:
                self.roads = gpd.read_file(self.model.files["geoms"]["assets/roads"])
            else:
                self.roads = None

            if "assets/rails" in self.model.files["geoms"]:
                self.rail = gpd.read_file(self.model.files["geoms"]["assets/rails"])
            else:
                self.rail = None

            self.landuse = self.model.files["region_subgrid"][
                "landsurface/full_region_cultivated_land"
            ]

            # Processing of exposure data
            if self.buildings is not None:
                all_buildings_geul_polygons = self.buildings[
                    self.buildings.geometry.type != "Point"
                ]
                all_buildings_geul_polygons.loc[:, "landuse"] = "building"
                all_buildings_geul_polygons.reset_index(drop=True, inplace=True)
                reproject_buildings = all_buildings_geul_polygons.to_crs(32631)
                reproject_buildings["area"] = reproject_buildings["geometry"].area
                self.selected_buildings = reproject_buildings[
                    reproject_buildings["area"] > 18
                ]
                self.selected_buildings.reset_index(drop=True, inplace=True)

                # Only take the center point for each building
                self.centroid_gdf = gpd.GeoDataFrame(
                    geometry=self.selected_buildings.centroid
                )
                self.centroid_gdf.loc[:, "landuse"] = "building"
                self.centroid_gdf.reset_index(drop=True, inplace=True)

                # Load maximum damages
                with open(
                    model.files["dict"][
                        "damage_parameters/flood/buildings/structure/maximum_damage"
                    ],
                    "r",
                ) as f:
                    self.max_dam_buildings_structure = json.load(f)
                self.max_dam_buildings_structure["building"] = (
                    self.max_dam_buildings_structure.pop("maximum_damage")
                )

                with open(
                    model.files["dict"][
                        "damage_parameters/flood/buildings/content/maximum_damage"
                    ],
                    "r",
                ) as f:
                    self.max_dam_buildings_content = json.load(f)
                self.max_dam_buildings_content["building"] = (
                    self.max_dam_buildings_content.pop("maximum_damage")
                )

                # Load vulnerability curves
                self.buildings_structure_curve = pd.read_parquet(
                    self.model.files["table"][
                        "damage_parameters/flood/buildings/structure/curve"
                    ]
                )
                self.buildings_structure_curve.rename(
                    columns={"damage_ratio": "building"}, inplace=True
                )

                self.buildings_content_curve = pd.read_parquet(
                    self.model.files["table"][
                        "damage_parameters/flood/buildings/content/curve"
                    ]
                )
                self.buildings_content_curve.rename(
                    columns={"damage_ratio": "building"}, inplace=True
                )
            else:
                self.selected_buildings = None
                self.centroid_gdf = None
                self.max_dam_buildings_structure = None
                self.max_dam_buildings_content = None
                self.buildings_structure_curve = None
                self.buildings_content_curve = None

            if self.roads is not None:
                self.roads = self.roads.to_crs(32631)

                # Load maximum damages for roads
                self.max_dam_road = {}
                road_types = [
                    (
                        "residential",
                        "damage_parameters/flood/road/residential/maximum_damage",
                    ),
                    (
                        "unclassified",
                        "damage_parameters/flood/road/unclassified/maximum_damage",
                    ),
                    (
                        "tertiary",
                        "damage_parameters/flood/road/tertiary/maximum_damage",
                    ),
                    ("primary", "damage_parameters/flood/road/primary/maximum_damage"),
                    (
                        "primary_link",
                        "damage_parameters/flood/road/primary_link/maximum_damage",
                    ),
                    (
                        "secondary",
                        "damage_parameters/flood/road/secondary/maximum_damage",
                    ),
                    (
                        "secondary_link",
                        "damage_parameters/flood/road/secondary_link/maximum_damage",
                    ),
                    (
                        "motorway",
                        "damage_parameters/flood/road/motorway/maximum_damage",
                    ),
                    (
                        "motorway_link",
                        "damage_parameters/flood/road/motorway_link/maximum_damage",
                    ),
                    ("trunk", "damage_parameters/flood/road/trunk/maximum_damage"),
                    (
                        "trunk_link",
                        "damage_parameters/flood/road/trunk_link/maximum_damage",
                    ),
                ]

                for road_type, path in road_types:
                    with open(model.files["dict"][path], "r") as f:
                        max_damage = json.load(f)
                    self.max_dam_road[road_type] = max_damage["maximum_damage"]

        with open(
            model.files["dict"][
                "damage_parameters/flood/land_use/forest/maximum_damage"
            ],
            "r",
        ) as f:
            self.max_dam_forest = json.load(f)
        self.max_dam_forest["0"] = self.max_dam_forest.pop("maximum_damage")

        with open(
            model.files["dict"][
                "damage_parameters/flood/land_use/agriculture/maximum_damage"
            ],
            "r",
        ) as f:
            self.max_dam_agriculture = json.load(f)
        self.max_dam_agriculture["1"] = self.max_dam_agriculture.pop("maximum_damage")

        self.max_dam_landuse = {**self.max_dam_forest, **self.max_dam_agriculture}
        self.max_dam_landuse = pd.DataFrame.from_dict(
            self.max_dam_landuse, orient="index", columns=["maximum_damage"]
        )
        self.max_dam_landuse["landuse"] = [
            "0",
            "1",
        ]
        self.max_dam_landuse = self.max_dam_landuse[["landuse", "maximum_damage"]]

        # Here we load in all vulnerability curves
        self.road_curves = []
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

            self.road_curves.append(df[[road_type]])
        self.road_curves = pd.concat([severity_column] + self.road_curves, axis=1)

        self.forest_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/land_use/forest/curve"]
        )
        self.forest_curve.rename(columns={"damage_ratio": "0"}, inplace=True)

        self.agriculture_curve = pd.read_parquet(
            self.model.files["table"][
                "damage_parameters/flood/land_use/agriculture/curve"
            ]
        )
        self.agriculture_curve.rename(columns={"damage_ratio": "1"}, inplace=True)

        self.curves_landuse = pd.merge(
            self.forest_curve, self.agriculture_curve, on="severity"
        )

        self.buildings_structure_curve = pd.read_parquet(
            self.model.files["table"][
                "damage_parameters/flood/buildings/structure/curve"
            ]
        )
        self.buildings_structure_curve.rename(
            columns={"damage_ratio": "building"}, inplace=True
        )

        self.buildings_content_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/buildings/content/curve"]
        )
        self.buildings_content_curve.rename(
            columns={"damage_ratio": "building"}, inplace=True
        )

        self.rail_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/rail/main/curve"]
        )
        self.rail_curve.rename(columns={"damage_ratio": "rail"}, inplace=True)

        super().__init__()

    def initiate(self) -> None:
        """Calls functions to initialize all agent attributes"""

        water_demand, efficiency = self.update_water_demand()
        self.current_water_demand = water_demand
        self.current_efficiency = efficiency
        self.risk_perception = np.full_like(water_demand, 1)

    def flood(self, flood_map, simulation_root, return_period=None):
        self.flood_depth.fill(0)  # Reset flood depth for all households

        # (Your existing code for handling the flood map)

        if return_period is not None:
            flood_map = join(simulation_root, f"hmax RP {int(return_period)}.tif")
        else:
            flood_map = join(simulation_root, "hmax.tif")
        # print(f"using this flood map: {flood_map}")

        # Initialize total damages
        total_flood_damages = 0

        # Calculate loss_landuse
        loss_landuse = RasterScanner(
            landuse_file=self.landuse,
            hazard_file=flood_map,
            curve_path=self.curves_landuse,
            maxdam_path=self.max_dam_landuse,
            lu_crs=4326,
            haz_crs=32631,
            dtype=np.int32,
            save=True,
            scenario_name="raster",
        )
        total_flood_damages += loss_landuse["damages"].sum()

        # Calculate damages for buildings
        if self.selected_buildings is not None and self.centroid_gdf is not None:
            damages_buildings_structure = VectorScanner(
                exposure_file=self.selected_buildings,
                hazard_file=flood_map,
                curve_path=self.buildings_structure_curve,
                maxdam_path=self.max_dam_buildings_structure,
                cell_size=20,
                exp_crs=32631,
                haz_crs=32631,
                object_col="landuse",
                hazard_col="hmax",
                lat_col="xc",
                lon_col="yc",
                centimeters=False,
                save=False,
                plot=False,
                grouped=False,
                scenario_name="building_structure2",
            )
            total_damage_structure = damages_buildings_structure["damage"].sum()
            print(f"Building structure damage is: {total_damage_structure}")

            damages_buildings_content = VectorScanner(
                exposure_file=self.centroid_gdf,
                hazard_file=flood_map,
                curve_path=self.buildings_content_curve,
                maxdam_path=self.max_dam_buildings_content,
                cell_size=20,
                exp_crs=32631,
                haz_crs=32631,
                object_col="landuse",
                hazard_col="hmax",
                lat_col="xc",
                lon_col="yc",
                centimeters=False,
                save=False,
                plot=False,
                grouped=False,
                scenario_name="building_contents2",
            )
            total_damages_content = damages_buildings_content["damage"].sum()
            print(f"Building content damage is: {total_damages_content}")

            total_damages_buildings = total_damage_structure + total_damages_content
            print(f"Total damage to buildings is: {total_damages_buildings}")
            total_flood_damages += total_damages_buildings
        else:
            print("No buildings data available. Skipping buildings damage calculation.")

        # Calculate damages for roads
        if self.roads is not None:
            damages_roads = VectorScanner(
                exposure_file=self.roads,
                hazard_file=flood_map,
                curve_path=self.road_curves,
                maxdam_path=self.max_dam_road,
                cell_size=20,
                exp_crs=32631,
                haz_crs=32631,
                object_col="highway",
                hazard_col="hmax",
                lat_col="xc",
                lon_col="yc",
                centimeters=False,
                save=False,
                plot=False,
                grouped=False,
                scenario_name="roads",
            )
            total_damages_roads = damages_roads["damage"].sum()
            print(f"Damages to roads: {total_damages_roads}")
            total_flood_damages += total_damages_roads
        else:
            print("No roads data available. Skipping roads damage calculation.")

        # Calculate damages for rail
        if self.rail is not None:
            damages_rail = VectorScanner(
                exposure_file=self.rail,
                hazard_file=flood_map,
                curve_path=self.rail_curve,
                maxdam_path=self.max_dam_rail,
                cell_size=20,
                exp_crs=32631,
                haz_crs=32631,
                object_col="railway",
                hazard_col="hmax",
                lat_col="xc",
                lon_col="yc",
                centimeters=False,
                save=False,
                plot=False,
                grouped=False,
                scenario_name="rail",
            )
            total_damages_rail = damages_rail["damage"].sum()
            print(f"Damage to rail is: {total_damages_rail}")
            total_flood_damages += total_damages_rail
        else:
            print("No rail data available. Skipping rail damage calculation.")

        print(f"The total flood damages are: {total_flood_damages}")

        return total_flood_damages

    def update_water_demand(self):
        """
        Dynamic part of the water demand module - domestic
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """
        downscale_mask = self.model.data.HRU.land_use_type != SEALED
        if self.model.use_gpu:
            downscale_mask = downscale_mask.get()
        days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365
        water_demand = (
            self.model.domestic_water_demand_ds.sel(
                time=self.model.current_time, method="ffill", tolerance="366D"
            ).domestic_water_demand
            * 1_000_000
            / days_in_year
        )
        water_demand = downscale_volume(
            self.model.domestic_water_demand_ds.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_demand.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.land_use_ratio,
        )
        if self.model.use_gpu:
            water_demand = cp.array(water_demand)
        water_demand = self.model.data.HRU.M3toM(water_demand)

        water_consumption = (
            self.model.domestic_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill"
            ).domestic_water_consumption
            * 1_000_000
            / days_in_year
        )
        water_consumption = downscale_volume(
            self.model.domestic_water_consumption_ds.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_consumption.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.land_use_ratio,
        )
        if self.model.use_gpu:
            water_consumption = cp.array(water_consumption)
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
        self.last_water_demand_update = self.model.current_time
        return water_demand, efficiency

    def water_demand(self):
        time_years = self.model.domestic_water_consumption_ds.time.dt.year
        years_list = time_years.values.tolist()
        if self.model.current_time.year in years_list:
            water_demand, efficiency = self.update_water_demand()
            self.current_water_demand = water_demand
            self.current_efficiency = efficiency

        # assert (self.model.current_time - self.last_water_demand_update).days < 366, (
        #     "Water demand has not been updated for over a year. "
        #     "Please check the household water demand datasets."
        # )
        return self.current_water_demand, self.current_efficiency

    def step(self) -> None:
        # self.risk_perception *= self.risk_perception
        return None

    @property
    def n(self):
        return self.locations.shape[0]
