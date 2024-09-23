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
import xarray as xr

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
        self.landuse=self.model.files["region_subgrid"]["landsurface/full_region_cultivated_land"]#xr.open_dataset(,engine="zarr")
        

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
        print(self.rail)
        #self.rail.to_csv("rail.csv")

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
       
        with open(model.files["dict"]["damage_parameters/flood/rail/main/maximum_damage"], "r") as f:
            self.max_dam_rail = json.load(f)
        self.max_dam_rail['rail'] = self.max_dam_rail.pop('maximum_damage')
        print(self.max_dam_rail)

        self.max_dam_road = {}
        road_types = [
            ("residential", "damage_parameters/flood/road/residential/maximum_damage"),
            ("unclassified", "damage_parameters/flood/road/unclassified/maximum_damage"),
            ("tertiary", "damage_parameters/flood/road/tertiary/maximum_damage"),
            ("primary", "damage_parameters/flood/road/primary/maximum_damage"),
            ("primary_link", "damage_parameters/flood/road/primary_link/maximum_damage"),
            ("secondary", "damage_parameters/flood/road/secondary/maximum_damage"),
            ("secondary_link", "damage_parameters/flood/road/secondary_link/maximum_damage"),
            ("motorway", "damage_parameters/flood/road/motorway/maximum_damage"),
            ("motorway_link", "damage_parameters/flood/road/motorway_link/maximum_damage"),
            ("trunk", "damage_parameters/flood/road/trunk/maximum_damage"),
            ("trunk_link", "damage_parameters/flood/road/trunk_link/maximum_damage")
        ]

        # Loop through each road type and load the corresponding data
        for road_type, path in road_types:
            with open(model.files["dict"][path], "r") as f:
                max_damage = json.load(f)
                # Rename the key from 'maximum_damage' to the corresponding road type
            self.max_dam_road[road_type] = max_damage['maximum_damage']
        print(self.max_dam_road)

        with open(model.files["dict"]["damage_parameters/flood/land_use/forest/maximum_damage"], "r") as f:
            self.max_dam_forest= json.load(f)
        self.max_dam_forest['0'] = self.max_dam_forest.pop('maximum_damage')
        print(self.max_dam_forest)
        with open(model.files["dict"]["damage_parameters/flood/land_use/agriculture/maximum_damage"], "r") as f:
            self.max_dam_agriculture= json.load(f)
        self.max_dam_agriculture['1'] = self.max_dam_agriculture.pop('maximum_damage')
        print(self.max_dam_agriculture)        
        self.max_dam_landuse = {**self.max_dam_forest, **self.max_dam_agriculture}
        self.max_dam_landuse = pd.DataFrame.from_dict(self.max_dam_landuse, orient='index', columns=['maximum_damage'])
        self.max_dam_landuse['landuse'] = ['0', '1']  # Or assign programmatically if needed
        self.max_dam_landuse = self.max_dam_landuse[['landuse', 'maximum_damage']]
        print(self.max_dam_landuse)


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
            ("trunk_link", "damage_parameters/flood/road/trunk_link/curve")
        ]

        severity_column = None
        # Loop through each road type and load the corresponding data
        for road_type, path in road_types:
            # Read the parquet file for each road type
            df = pd.read_parquet(self.model.files["table"][path])
            
            # If this is the first DataFrame, save the 'severity' column
            if severity_column is None:
                severity_column = df['severity'] / 100
            
            # Rename the 'damage_ratio' column to the road type name
            df = df.rename(columns={"damage_ratio": road_type})
            
            # Append the modified DataFrame to the list, keeping only the new road type column
            self.road_curves.append(df[[road_type]])

        # Concatenate the severity column with all the road type DataFrames
        self.road_curves = pd.concat([severity_column] + self.road_curves, axis=1)
        print(self.road_curves)
        
        self.forest_curve = pd.read_parquet(self.model.files["table"]["damage_parameters/flood/land_use/forest/curve"])
        self.forest_curve.rename(columns={'damage_ratio': '0'}, inplace=True)
        print(self.forest_curve)
        self.agriculture_curve = pd.read_parquet(self.model.files["table"]["damage_parameters/flood/land_use/agriculture/curve"])
        self.agriculture_curve.rename(columns={'damage_ratio': '1'}, inplace=True)
        print(self.agriculture_curve)

        self.curves_landuse = pd.merge(self.forest_curve, self.agriculture_curve, on="severity")
        print(self.curves_landuse)
        
        self.buildings_structure_curve = pd.read_parquet(self.model.files["table"]["damage_parameters/flood/buildings/structure/curve"])
        self.buildings_structure_curve.rename(columns={'damage_ratio': 'building'}, inplace=True)
        print(self.buildings_structure_curve)
        self.buildings_content_curve = pd.read_parquet(self.model.files["table"]["damage_parameters/flood/buildings/content/curve"])
        self.buildings_content_curve.rename(columns={'damage_ratio': 'building'}, inplace=True)
        print(self.buildings_content_curve)

        self.rail_curve = pd.read_parquet(self.model.files["table"]["damage_parameters/flood/rail/main/curve"])
        self.rail_curve.rename(columns={'damage_ratio': "rail"}, inplace=True)
        print(self.rail_curve)

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

        loss_landuse = RasterScanner(landuse_file=self.landuse,
                                      hazard_file=flood_map,
                                      curve_path=self.curves_landuse,maxdam_path=self.max_dam_landuse, lu_crs=4326, haz_crs=32631, dtype=np.int32, save=True, scenario_name='raster')   

        print(loss_landuse)

        damages_buildings_structure = VectorScanner(exposure_file=self.selected_buildings,
                  hazard_file=flood_map,
                  curve_path=self.buildings_structure_curve,
                  maxdam_path = self.max_dam_buildings_structure,
                  cell_size = 20,
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
        print(f"building structure damage is: {total_damage_structure}")
                  
        damages_buildings_content = VectorScanner(exposure_file=self.centroid_gdf,
                  hazard_file=flood_map,
                  curve_path=self.buildings_content_curve,
                  maxdam_path = self.max_dam_buildings_content,
                  cell_size = 20,
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
        print(f"building content damage is: { total_damages_content}")

        total_damages_buildings = total_damage_structure + total_damages_content
        print(f"total damage to buildings is: { total_damages_buildings} ")

        damages_roads = VectorScanner(exposure_file=self.roads,
                  hazard_file=flood_map,
                  curve_path=self.road_curves,
                  maxdam_path = self.max_dam_road,
                  cell_size = 20,
                  exp_crs=32631,
                  haz_crs=32631,                   
                  object_col='highway',
                  hazard_col='hmax',
                  lat_col="xc",
                  lon_col="yc",
                  centimeters=False,
                  save=False,
                  plot=False,
                  grouped=False,
                  scenario_name = 'roads'
                 )
        
        total_damages_roads = damages_roads['damage'].sum()
        print(f"damages to roads: {total_damages_roads} ")

        damages_rail = VectorScanner(exposure_file=self.rail,
                  hazard_file=flood_map,
                  curve_path=self.rail_curve,
                  maxdam_path = self.max_dam_rail,
                  cell_size = 20,
                  exp_crs=32631,
                  haz_crs=32631,                   
                  object_col='railway',
                  hazard_col='hmax',
                  lat_col="xc",
                  lon_col="yc",
                  centimeters=False,
                  save=False,
                  plot=False,
                  grouped=False,
                  scenario_name = 'rail'
                 )
        total_damages_rail = damages_rail['damage'].sum()
        print(f"damage to rail is: {total_damages_rail}")

        total_flood_damages = loss_landuse["damages"].sum() + total_damage_structure + total_damages_content + total_damages_roads +total_damages_rail 
        print(f"the total flood damages are: {total_flood_damages}")

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
