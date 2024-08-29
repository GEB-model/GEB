import numpy as np
import geopandas as gpd
import pyproj
import calendar
from .general import AgentArray, downscale_volume
from . import AgentBaseClass

from damagescanner.core import RasterScanner
from damagescanner.core import VectorScanner 
import pandas as pd 
from os.path import join 


class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents
        self.reduncancy = reduncancy
        
        #Load exposure data 
        self.buildings = gpd.read_file(self.model.model_structure["geoms"]["assets/buildings"])
        self.roads = gpd.read_file(self.model.model_structure["geoms"]["assets/roads"])
        self.rail = gpd.read_file(self.model.model_structure["geoms"]["assets/rail"])

        self.landuse=self.model.model_structure["region_subgrid"]["landsurface/full_region_cultivated_land"]

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

        #Here we load the information for the flood risk function
        self.max_dam_landuse = pd.read_csv(self.model.model_structure["table"]["floodrisk/max_damage_raster"], delimiter = ";") # TODO: Match categories to GEB landuse 
        self.curves_landuse = pd.read_csv(self.model.model_structure["table"]["floodrisk/damage_curves_raster"], delimiter = ";") # TODO: Match categories to GEB Landuse 
        self.max_dam_road = pd.read_csv(self.model.model_structure["table"]["floodrisk/max_damage_road"], delimiter = ";") 
        self.curves_road = pd.read_csv(self.model.model_structure["table"]["floodrisk/damage_curves_road"], delimiter = ";")
        self.max_dam_rail = pd.read_csv(self.model.model_structure["table"]["floodrisk/max_damage_rail"], delimiter = ";") 
        self.curves_rail = pd.read_csv(self.model.model_structure["table"]["floodrisk/damage_curves_rail"], delimiter = ";")
        self.max_dam_buildings_structure = pd.read_csv(self.model.model_structure["table"]["floodrisk/max_damage_buildings_structure"], delimiter = ";") 
        self.curves_buildings_structure = pd.read_csv(self.model.model_structure["table"]["floodrisk/damage_curves_buildings_structure"], delimiter = ";")
        self.max_dam_buildings_contents = pd.read_csv(self.model.model_structure["table"]["floodrisk/max_damage_buildings_contents"], delimiter = ";") 
        self.curves_buildings_contents = pd.read_csv(self.model.model_structure["table"]["floodrisk/damage_curves_buildings_contents"], delimiter = ";")

        super().__init__()

        water_demand, efficiency = self.update_water_demand()
        self.current_water_demand = water_demand
        self.current_efficiency = efficiency

    def initiate(self) -> None:
        locations = np.load(
            self.model.model_structure["binary"]["agents/households/locations"]
        )["data"]
        self.max_n = int(locations.shape[0] * (1 + self.reduncancy) + 1)

        self.locations = AgentArray(locations, max_n=self.max_n) # Load locations of households

        sizes = np.load(
            self.model.model_structure["binary"]["agents/households/sizes"] # Load sizes of households
        )["data"]
        self.sizes = AgentArray(sizes, max_n=self.max_n)

        self.flood_depth = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=0, dtype=np.float32
        ) #Make an agent array that stores the flood depth of each household, starts with the value 0 everywhere
        self.risk_perception = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=1, dtype=np.float32
        ) # Make an agent array that stores the risk perception of each household, starts with value 1 everywhere

               

    def flood(self, flood_map, folder, return_period=None):
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
            flood_map=join(folder, f"hmax RP {int(return_period)}.tif")
        else:
            flood_map = join(folder,"hmax.tif")
        print(f"using this flood map: {flood_map}")
        
        #Call RasterScanner and VectorScanner 
        loss_landuse = RasterScanner(landuse_file=self.landuse,
                                     hazard_file=flood_map,
                                     curve_path=self.curves_landuse,maxdam_path=self.max_dam_landuse, lu_crs=4326, haz_crs=32631, dtype=np.int32, save=True, scenario_name='raster')   

        #print(loss_landuse)
        
        damages_buildings_structure = VectorScanner(exposure_file=self.selected_buildings,
                  hazard_file=flood_map,
                  curve_path=self.curves_buildings_structure,
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
       # print(total_damage_structure)
                  
        damages_buildings_contents = VectorScanner(exposure_file=self.centroid_gdf,
                  hazard_file=flood_map,
                  curve_path=self.curves_buildings_contents,
                  maxdam_path = self.max_dam_buildings_contents,
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
        total_damages_contents = damages_buildings_contents['damage'].sum()
       # print(total_damages_contents)

        total_damages_buildings = total_damage_structure + total_damages_contents
      #  print(total_damages_buildings)

        damages_roads = VectorScanner(exposure_file=self.roads,
                  hazard_file=flood_map,
                  curve_path=self.curves_road,
                  maxdam_path = self.max_dam_road,
                  cell_size = 25,
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
      #  print(damages_roads)
        total_damages_roads = damages_roads['damage'].sum()
      #  print(total_damages_roads)

        damages_rail = VectorScanner(exposure_file=self.rail,
                  hazard_file=flood_map,
                  curve_path=self.curves_rail,
                  maxdam_path = self.max_dam_rail,
                  cell_size = 25,
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
     #   print(damages_rail)
        total_damages_rail = damages_rail['damage'].sum()
      #  print(total_damages_rail)

        total_flood_damages = loss_landuse["damages"].sum() + total_damage_structure + total_damages_contents + total_damages_roads +total_damages_rail 
        print(f"the total flood damages are: {total_flood_damages}")

        return total_flood_damages
    

    def update_water_demand(self):
        """
        Dynamic part of the water demand module - domestic
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """
        downscale_mask = self.model.data.HRU.land_use_type != 4
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
