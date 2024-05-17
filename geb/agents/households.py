import numpy as np
import geopandas as gpd
import pyproj

from .general import AgentArray
from . import AgentBaseClass

#from damagescanner.core import RasterScanner
#from damagescanner.core import VectorScanner 
import pandas as pd 

class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents
        self.reduncancy = reduncancy
        self.buildings = gpd.read_file(
            self.model.model_structure["geoms"]["assets/buildings"] #here we read the gdf with all buildings 
        )
        print(self.buildings)

        super().__init__()

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

        

    def flood(self, flood_map):
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

         # Import: vulnerability curves and max damages 
    #     max_dam_landuse = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\max_damage_raster.csv", delimiter = ";") # TODO: Match categories to GEB landuse 
    #     curves_landuse = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\damage_curves_raster.csv", delimiter = ";") # TODO: Match categories to GEB Landuse 

    #     max_dam_road = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\max_damage_road.csv", delimiter = ";") 
    #     curves_road = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\damage_curves_road.csv", delimiter = ";")

    #     max_dam_rail = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\max_damage_rail.csv", delimiter = ";") 
    #     curves_rail = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\damage_curves_rail.csv", delimiter = ";")

    #     max_dam_buildings_structure = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\max_damage_buildings_structure.csv", delimiter = ";") 
    #     curves_buildings_structure = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\damage_curves_buildings_structure.csv", delimiter = ";")

    #     max_dam_buildings_contents = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\max_damage_buildings_contents.csv", delimiter = ";") 
    #     curves_buildings_contents = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\damage_curves_buildings_contents.csv", delimiter = ";")

    #     # Import: all exposure data (for now: buildings and landuse) + processing of data
    #     all_buildings_geul_polygons = self.buildings[self.buildings != 'Point']
    #     all_buildings_geul_polygons.loc[:, "landuse"] = 'building'
    #     all_buildings_geul_polygons.reset_index(drop=True, inplace=True)
    #     reproject_buildings = all_buildings_geul_polygons.to_crs(32631)
    #     reproject_buildings["area"] = reproject_buildings["geometry"].area #calculate the area for all buildings
    #     selected_buildings = reproject_buildings[reproject_buildings["area"] > 18] # Bouwbesluit: only allowed to live in buildings larger than 18 m2 https://rijksoverheid.bouwbesluit.com/Inhoud/docs/wet/bb2012/hfd4?tableid=docs/wet/bb2012[35]/hfd4/afd4-1/par4-1-1 
    #     selected_buildings.reset_index(drop=True, inplace=True)

    #     # Only take the center point for each building 
    #     centroid_gdf = gpd.GeoDataFrame(geometry=selected_buildings.centroid)
    #     centroid_gdf.loc[:, "landuse"] = 'building'
    #     centroid_gdf.reset_index(drop=True, inplace=True)

    #     # also get landuse in here 

        
    #     #from os.path import join # TODO: fix the make the flood map part more flexible 
    #     #simulation_root = snakemake.params.sim_root  # (relative) path to sfincs root
    #     flood_map2= r"C:\Users\vbl220\Documents\PhD\models\geul\base\simulation_root\SFINCS\23250850\simulations\20210720T000000\hmax.tif"
    #     #flood_map3=join(simulation_root, "hmax.tif")
        
    #     # Call RasterScanner and VectorScanner 
    #    # loss_landuse = RasterScanner(landuse_file=landuse,hazard_file=flood_map,curve_path=curves,maxdam_path=max_dam, lu_crs=28992, haz_crs=28992, dtype=np.int32, save=True, scenario_name='rastertest')   
       
    #     damages_buildings_structure = VectorScanner(exposure_file=selected_buildings,
    #               hazard_file=flood_map2,
    #               curve_path=curves_buildings_structure,
    #               maxdam_path = max_dam_buildings_structure,
    #               cell_size = 20,
    #               exp_crs= 32631, 
    #               haz_crs= 32631,                 
    #               object_col='landuse',
    #               hazard_col="hmax",
    #               lat_col="xc",
    #               lon_col="yc",
    #               centimeters=False,
    #               save=False,
    #               plot=False,
    #               grouped=False,
    #               scenario_name = 'building_structure2'
    #               )

    #     print(damages_buildings_structure)
    #     total_damage = damages_buildings_structure['damage'].sum()
    #     print(total_damage)
                  
    #     damages_buildings_contents = VectorScanner(exposure_file=centroid_gdf,
    #               hazard_file=flood_map2,
    #               curve_path=curves_buildings_contents,
    #               maxdam_path = max_dam_buildings_contents,
    #               cell_size = 20,
    #               exp_crs=32631,
    #               haz_crs=32631,                   
    #               object_col='landuse',
    #               hazard_col='hmax',
    #               lat_col="xc",
    #               lon_col="yc",
    #               centimeters=False,
    #               save=False,
    #               plot=False,
    #               grouped=False,
    #               scenario_name = 'building_contents2'
    #              )
    #     print(damages_buildings_contents)
    #     total_damages_contents = damages_buildings_contents['damage'].sum()
    #     print(total_damages_contents)

    #     total_damages_buildings = total_damage +total_damages_contents
    #     print(total_damages_buildings)

    #     return None
    

    def step(self) -> None:
        self.risk_perception *= self.risk_perception
        return None

    @property
    def n(self):
        return self.locations.shape[0]
