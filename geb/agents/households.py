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

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass


class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents
        self.reduncancy = reduncancy

        #Load exposure data 
        self.buildings = gpd.read_file(self.model.files["geoms"]["assets/buildings"])
        self.roads = gpd.read_file(self.model.files["geoms"]["assets/roads"])
        self.rail = gpd.read_file(self.model.files["geoms"]["assets/rails"])
        self.landuse=self.model.files["region_subgrid"]["landsurface/full_region_cultivated_land"]

        #Processing of exposure data
        all_buildings_geul_polygons = self.buildings[self.buildings != 'Point'] # Only use polygons in analysis
        all_buildings_geul_polygons.loc[:, "landuse"] = 'building' # Add column
        all_buildings_geul_polygons.reset_index(drop=True, inplace=True)
        reproject_buildings = all_buildings_geul_polygons.to_crs(32631) #Change CRS
        reproject_buildings["area"] = reproject_buildings["geometry"].area #calculate the area for all buildings
        self.selected_buildings = reproject_buildings[reproject_buildings["area"] > 18] # Bouwbesluit: only allowed to live in buildings larger than 18 m2 https://rijksoverheid.bouwbesluit.com/Inhoud/docs/wet/bb2012/hfd4?tableid=docs/wet/bb2012[35]/hfd4/afd4-1/par4-1-1 
        self.selected_buildings.reset_index(drop=True, inplace=True)
        self.roads = self.roads.to_crs(32631)
        self.rail = self.rail.to_crs(32631)

        # Only take the center point for each building 
        self.centroid_gdf = gpd.GeoDataFrame(geometry=self.selected_buildings.centroid)
        self.centroid_gdf.loc[:, "landuse"] = 'building'
        self.centroid_gdf.reset_index(drop=True, inplace=True)

        # Load maximum damages 
        #data = self.model.files["dict"]["damage_parameters/flood/buildings/structure/maximum_damage"]
        #self.max_dam_buildings_structure = pd.DataFrame([data])
        #print(self.max_dam_buildings_structure)
        with open(model.files["dict"]["damage_parameters/flood/buildings/structure/maximum_damage"], "r") as f:
            self.max_dam_buildings_structure = json.load(f)
        self.max_dam_buildings_structure['building'] = self.max_dam_buildings_structure.pop('maximum_damage')
        print(self.max_dam_buildings_structure)
        
        with open(model.files["dict"]["damage_parameters/flood/buildings/content/maximum_damage"], "r") as f:
            self.max_dam_buildings_content = json.load(f)
        self.max_dam_buildings_content['building'] = self.max_dam_buildings_content.pop('maximum_damage')
        print(self.max_dam_buildings_content)
       
        # self.max_dam_road_main = pd.read_json(self.model.files["damage_parameters/flood/rail/main/maximum_damage"])
        # self.max_dam_road_residential = pd.read_json(self.model.files["damage_parameters/flood/road/residential/maximum_damage"])
        # self.max_dam_road_unclassified = pd.read_json(self.model.files["damage_parameters/flood/road/unclassified/maximum_damage"])
        # self.max_dam_road_tertiary = pd.read_json(self.model.files["damage_parameters/flood/road/tertiary/maximum_damage"])
        # self.max_dam_road_primary = pd.read_json(self.model.files["damage_parameters/flood/road/primary/maximum_damage"])
        # self.max_dam_road_secondary = pd.read_json(self.model.files["damage_parameters/flood/road/secondary/maximum_damage"])
        # self.max_dam_road_motorway = pd.read_json(self.model.files["damage_parameters/flood/road/motorway/maximum_damage"])
        # self.max_dam_road_motorway_link = pd.read_json(self.model.files["damage_parameters/flood/road/motorway_link/maximum_damage"])
        # self.max_dam_road_trunk = pd.read_json(self.model.files["damage_parameters/flood/road/trunk/maximum_damage"])
        # self.max_dam_road_trunk_link = pd.read_json(self.model.files["damage_parameters/flood/road/trunk_link/maximum_damage"])
        # self.max_dam_primary_link = pd.read_json(self.model.files["damage_parameters/flood/road/primary_link/maximum_damage"])
        # self.max_dam_secondary_link = pd.read_json(self.model.files["damage_parameters/flood/road/secondary_link/maximum_damage"])
        # self.max_dam_forest = pd.read_json(self.model.files["damage_parameters/flood/land_use/forest/maximum_damage"])
        # self.max_dam_agriculture = pd.read_json(self.model.files["damage_parameters/flood/land_use/agriculture/maximum_damage"])

        # Here we load in all vulnerability curves 
        # self.road_residential_curve = pd.read_parquet(self.model.files["table"]["damage_parameters/flood/road/residential/curve"])
        # self.road_unclassified_curve = pd.read_parquet(self.model.files["damage_parameters/flood/road/unclassified/curve"])
        # self.road_tertiary_curve = pd.read_parquet(self.model.files["damage_parameters/flood/road/tertiary/curve"])
        # self.road_primary_curve = pd.read_parquet(self.model.files["damage_parameters/flood/road/primary/curve"])
        # self.road_secondary_curve = pd.read_parquet(self.model.files["damage_parameters/flood/road/secondary/curve"])
        # self.road_motorway_curve = pd.read_parquet(self.model.files["damage_parameters/flood/road/motorway/curve"])
        # self.road_motorway_link_curve = pd.read_parquet(self.model.files["damage_parameters/flood/road/motorway_link/curve"])
        # self.road_trunk_curve = pd.read_parquet(self.model.files["damage_parameters/flood/road/trunk/curve"])
        # self.road_trunk_link_curve = pd.read_parquet(self.model.files["damage_parameters/flood/road/trunk_link/curve"])
        # self.road_primary_link_curve = pd.read_parquet(self.model.files["damage_parameters/flood/road/primary_link/curve"])
        # self.road_secondary_link_curve = pd.read_parquet(self.model.files["damage_parameters/flood/road/secondary_link/curve"])
        # self.forest_curve = pd.read_parquet(self.model.files["damage_parameters/flood/land_use/forest/curve"])
        # self.agriculture_curve = pd.read_parquet(self.model.files["damage_parameters/flood/land_use/agriculture/curve"])
        self.buildings_structure_curve = pd.read_parquet(self.model.files["table"]["damage_parameters/flood/buildings/structure/curve"])
        self.buildings_structure_curve.rename(columns={'damage_ratio': 'building'}, inplace=True)
        print(self.buildings_structure_curve)
        self.buildings_content_curve = pd.read_parquet(self.model.files["table"]["damage_parameters/flood/buildings/content/curve"])
        self.buildings_content_curve.rename(columns={'damage_ratio': 'building'}, inplace=True)
        print(self.buildings_content_curve)

        #self.rail_curve = pd.read_parquet(self.model.files["damage_parameters/flood/rail/main/curve"])

        super().__init__()

        water_demand, efficiency = self.update_water_demand()
        self.current_water_demand = water_demand
        self.current_efficiency = efficiency

    def initiate(self) -> None:
        locations = np.load(self.model.files["binary"]["agents/households/locations"])[
            "data"
        ]
        self.max_n = int(locations.shape[0] * (1 + self.reduncancy) + 1)

        self.locations = AgentArray(locations, max_n=self.max_n)

        sizes = np.load(self.model.files["binary"]["agents/households/sizes"])["data"]
        self.sizes = AgentArray(sizes, max_n=self.max_n)

        self.flood_depth = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=0, dtype=np.float32
        )
        self.risk_perception = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=1, dtype=np.float32
        )

        self.buildings = gpd.read_file(self.model.files["geoms"]["assets/buildings"])

    def flood(self, flood_map, simulation_root, return_period=None):
        self.flood_depth.fill(0)  # Reset flood depth for all households

        import matplotlib.pyplot as plt

        plt.figure()

        flood_map.plot()
        plt.savefig("flood.png")

        transformer = pyproj.Transformer.from_crs(
            4326, flood_map.raster.crs, always_xy=True
        )
        x, y = transformer.transform(self.locations[:, 0], self.locations[:, 1])

        forward_transform = flood_map.raster.transform
        backward_transform = ~forward_transform

        pixel_x, pixel_y = backward_transform * (x, y)
        pixel_x = pixel_x.astype(int)  # TODO: Should I add 0.5?
        pixel_y = pixel_y.astype(int)  # TODO: Should I add 0.5?

        # Create a mask that includes only the pixels inside the grid
        mask = (
            (pixel_x >= 0)
            & (pixel_x < flood_map.shape[1])
            & (pixel_y >= 0)
            & (pixel_y < flood_map.shape[0])
        )

        flood_depth_per_household = flood_map.values[pixel_y[mask], pixel_x[mask]]
        self.flood_depth[mask] = flood_depth_per_household > 0

        self.risk_perception[(self.flood_depth > 0)] *= 10

        print("mean risk perception", self.risk_perception.mean())

        if return_period is not None:     
            flood_map=join(simulation_root, f"hmax RP {int(return_period)}.tif")
        else:
            flood_map = join(simulation_root,"hmax.tif")
        print(f"using this flood map: {flood_map}")

        damages_buildings_structure = VectorScanner(exposure_file=self.selected_buildings,
                  hazard_file=flood_map,
                  curve_path=self.buildings_structure_curve,
                  maxdam_path = self.max_dam_buildings_structure,
                  cell_size = 25,
                  exp_crs= 32631, 
                  haz_crs= 32631,                 
                  object_col='landuse',
                  hazard_col="hmax",
                  lat_col="xc",
                  lon_col="yc",
                  centimeters=False,
                  save=False,
                  plot=False,
                  grouped=False,
                  scenario_name = 'building_structure2'
                  )

        #print(damages_buildings_structure)
        total_damage_structure = damages_buildings_structure['damage'].sum()
        print(total_damage_structure)
                  
        damages_buildings_content = VectorScanner(exposure_file=self.centroid_gdf,
                  hazard_file=flood_map,
                  curve_path=self.buildings_content_curve,
                  maxdam_path = self.max_dam_buildings_content,
                  cell_size = 25,
                  exp_crs=32631,
                  haz_crs=32631,                   
                  object_col='landuse',
                  hazard_col='hmax',
                  lat_col="xc",
                  lon_col="yc",
                  centimeters=False,
                  save=False,
                  plot=False,
                  grouped=False,
                  scenario_name = 'building_contents2'
                 )
       # print(damages_buildings_contents)
        total_damages_content = damages_buildings_content['damage'].sum()
        print(total_damages_content)

        total_damages_buildings = total_damage_structure + total_damages_content
        print(total_damages_buildings)

        return total_damages_buildings

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
        if (
            np.datetime64(self.model.current_time, "ns")
            in self.model.domestic_water_consumption_ds.time
        ):
            water_demand, efficiency = self.update_water_demand()
            self.current_water_demand = water_demand
            self.current_efficiency = efficiency

        assert (self.model.current_time - self.last_water_demand_update).days < 366, (
            "Water demand has not been updated for over a year. "
            "Please check the household water demand datasets."
        )
        return self.current_water_demand, self.current_efficiency

    def step(self) -> None:
        self.risk_perception *= self.risk_perception
        return None

    @property
    def n(self):
        return self.locations.shape[0]
