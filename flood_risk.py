# Part 1: Calculate damage for agriculture, nature, wetland
import numpy as np
import pandas as pd
from damagescanner.core import RasterScanner

# Give the input files
inun_map =  r'C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\Inundation map\inundation_depth_cm_new.tif' # TODO: replace with Sfincs output
landuse = r'C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\reclassified_landuse_geul2.tif' # TODO: use landuse map from GEB
max_dam = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\max_damage_raster.csv", delimiter = ";") # TODO: Match categories to GEB landuse 
curves = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\damage_curves_raster.csv", delimiter = ";") # TODO: Match categories to GEB Landuse 

# The raster calculation 
# Hazard and landuse data must have the same CRS and resolution, in this case CRS = 29882, resolution = 25 (m)
#loss_df = RasterScanner(landuse_file=landuse,hazard_file=inun_map,curve_path=curves,maxdam_path=max_dam, lu_crs=28992, haz_crs=28992, dtype=np.int32, save=True, scenario_name='rastertest')   
#print('loss is calculated')
#print(loss_df)

# Part 2: Calculate damage for roads and railroads 
import matplotlib.pyplot as plt
import osm_flex.download as dl
import osm_flex.extract as ex
import osm_flex.config
import osm_flex.clip as cp
import pandas as pd 
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
from damagescanner.core import VectorScanner
import damagescanner

# Get data from NL, Belgium and Germany 
iso3 = 'NLD'
dl.get_country_geofabrik(iso3)
#print('Nederland data is downloaded')

iso3 = 'BEL'
dl.get_country_geofabrik(iso3)
#print('Belgium data is downloaded')

iso3 = 'DEU'
dl.get_country_geofabrik(iso3)
#print('Germany data is downloaded')

# Clip NLD data to polygon of catchment
coordinates = [ (5.766666666497713, 50.91666666663955), (5.766666666497713, 50.90833333330621), (5.741666666497736, 50.90833333330621), (5.741666666497736, 50.89999999997287 ), (5.716666666497758, 50.89999999997287), (5.716666666497758, 50.90833333330621), (5.708333333164433, 50.90833333330621), (5.708333333164433, 50.89166666663953), (5.716666666497758, 50.89166666663953), (5.716666666497758, 50.88333333330619 ), (5.724999999831084, 50.88333333330619) , (5.724999999831084, 50.87499999997285), (5.73333333316441, 50.87499999997285), (5.73333333316441, 50.88333333330619 ), (5.741666666497736, 50.88333333330619 ), ( 5.741666666497736, 50.87499999997285 ), (5.766666666497713, 50.87499999997285 ), ( 5.766666666497713, 50.86666666663951 ), ( 5.774999999831039, 50.86666666663951 ), ( 5.774999999831039, 50.85833333330617 ), ( 5.816666666497667, 50.85833333330617 ), ( 5.816666666497667, 50.84999999997283 ), ( 5.824999999830993, 50.84999999997283 ), ( 5.824999999830993, 50.84166666663949 ), ( 5.841666666497645, 50.84166666663949 ), ( 5.841666666497645, 50.83333333330615 ), ( 5.833333333164319, 50.83333333330615 ), ( 5.833333333164319, 50.82499999997281 ), ( 5.841666666497645, 50.82499999997281 ), ( 5.841666666497645, 50.80833333330613 ), ( 5.833333333164319, 50.80833333330613 ), ( 5.833333333164319, 50.79999999997279 ), ( 5.824999999830993, 50.79999999997279 ), ( 5.824999999830993, 50.791666666639451 ), ( 5.833333333164319, 50.791666666639451 ), ( 5.833333333164319, 50.783333333306111 ), ( 5.841666666497645, 50.783333333306111 ), ( 5.841666666497645, 50.774999999972771 ), ( 5.84999999983097, 50.774999999972771 ), ( 5.84999999983097, 50.741666666639411 ), ( 5.858333333164296, 50.741666666639411 ), ( 5.858333333164296, 50.733333333306071 ), ( 5.866666666497622, 50.733333333306071 ), ( 5.866666666497622, 50.716666666639391 ), ( 5.883333333164273, 50.716666666639391 ), ( 5.883333333164273, 50.708333333306051 ), ( 5.899999999830925, 50.708333333306051 ), ( 5.899999999830925, 50.691666666639371 ), ( 5.908333333164251, 50.691666666639371 ), ( 5.908333333164251, 50.683333333306031 ), ( 5.924999999830902, 50.683333333306031 ), ( 5.924999999830902, 50.674999999972691 ), ( 5.966666666497531, 50.674999999972691 ), ( 5.966666666497531, 50.666666666639351 ), ( 5.983333333164182, 50.666666666639351 ), ( 5.983333333164182, 50.649999999972671 ), ( 5.999999999830834, 50.649999999972671 ), ( 5.999999999830834, 50.658333333306011 ), ( 6.024999999830811, 50.658333333306011 ), ( 6.024999999830811, 50.649999999972671 ), ( 6.033333333164137, 50.649999999972671 ), ( 6.033333333164137, 50.658333333306011 ), ( 6.06666666649744, 50.658333333306011 ), ( 6.06666666649744, 50.666666666639351 ), ( 6.074999999830766, 50.666666666639351 ), ( 6.074999999830766, 50.674999999972691 ), ( 6.091666666497417, 50.674999999972691 ), ( 6.091666666497417, 50.683333333306031 ), ( 6.108333333164069, 50.683333333306031 ), ( 6.108333333164069, 50.691666666639371 ), ( 6.116666666497395, 50.691666666639371 ), ( 6.116666666497395, 50.699999999972711 ), ( 6.12499999983072, 50.699999999972711 ), ( 6.12499999983072, 50.716666666639391 ), ( 6.116666666497395, 50.716666666639391 ), ( 6.116666666497395, 50.724999999972731 ), ( 6.091666666497417, 50.724999999972731 ), ( 6.091666666497417, 50.733333333306071 ), ( 6.06666666649744, 50.733333333306071 ), ( 6.06666666649744, 50.741666666639411 ), ( 6.058333333164114, 50.741666666639411 ), ( 6.058333333164114, 50.749999999972751 ), ( 6.041666666497463, 50.749999999972751 ), ( 6.041666666497463, 50.741666666639411 ), ( 6.024999999830811, 50.741666666639411 ), ( 6.024999999830811, 50.758333333306091 ), ( 6.033333333164137, 50.758333333306091 ), ( 6.033333333164137, 50.766666666639431 ), ( 6.041666666497463, 50.766666666639431 ), ( 6.041666666497463, 50.774999999972771 ), ( 6.024999999830811, 50.774999999972771 ), ( 6.024999999830811, 50.79999999997279 ), ( 6.033333333164137, 50.79999999997279 ), ( 6.033333333164137, 50.82499999997281 ), ( 6.024999999830811, 50.82499999997281 ), ( 6.024999999830811, 50.83333333330615 ), ( 6.016666666497485, 50.83333333330615 ), ( 6.016666666497485, 50.84166666663949 ), ( 6.00833333316416, 50.84166666663949 ), ( 6.00833333316416, 50.84999999997283 ), ( 5.983333333164182, 50.84999999997283 ), ( 5.983333333164182, 50.84166666663949 ), ( 5.966666666497531, 50.84166666663949 ), ( 5.966666666497531, 50.84999999997283 ), ( 5.949999999830879, 50.84999999997283 ), ( 5.949999999830879, 50.85833333330617 ), ( 5.941666666497554, 50.85833333330617 ), ( 5.941666666497554, 50.84999999997283 ), ( 5.924999999830902, 50.84999999997283 ), ( 5.924999999830902, 50.85833333330617 ), ( 5.908333333164251, 50.85833333330617 ), ( 5.908333333164251, 50.86666666663951 ), ( 5.866666666497622, 50.86666666663951 ), ( 5.866666666497622, 50.87499999997285 ), ( 5.874999999830948, 50.87499999997285 ), ( 5.874999999830948, 50.88333333330619 ), ( 5.84999999983097, 50.88333333330619 ), ( 5.84999999983097, 50.89166666663953 ), ( 5.833333333164319, 50.89166666663953 ), ( 5.833333333164319, 50.89999999997287 ), ( 5.816666666497667, 50.89999999997287 ), ( 5.816666666497667, 50.90833333330621 ), ( 5.79166666649769, 50.90833333330621 ), ( 5.79166666649769, 50.91666666663955 ), ( 5.766666666497713, 50.91666666663955 )]
catchment = Polygon(coordinates)
#print('polygon for catchment is made')

cp.clip_from_shapes([catchment],
                    osmpbf_output=osm_flex.config.OSM_DATA_DIR.joinpath('geul_NL.osm.pbf'),
                    osmpbf_clip_from=osm_flex.config.OSM_DATA_DIR.joinpath('netherlands-latest.osm.pbf'),
                    kernel='osmconvert', overwrite=True)
#print('NL data is clipped')

# Get the roads & rails for NL
NL_geul = osm_flex.config.OSM_DATA_DIR.joinpath('geul_NL.osm.pbf')
NL_road = ex.extract_cis(NL_geul, 'road')
NL_rail = ex.extract_cis(NL_geul, 'rail')
NL_filtered_rail = NL_rail[NL_rail['railway'] == 'rail'] # Filter out unnecessary elements (such as stops, stations, etc.) from rail dataset

# Moving on the Germany 
cp.clip_from_shapes([catchment],
                    osmpbf_output=osm_flex.config.OSM_DATA_DIR.joinpath('geul_DEU.osm.pbf'),
                    osmpbf_clip_from=osm_flex.config.OSM_DATA_DIR.joinpath('germany-latest.osm.pbf'),
                    kernel='osmconvert', overwrite=True)
#print('DEU data is clipped')

DEU_geul = osm_flex.config.OSM_DATA_DIR.joinpath('geul_DEU.osm.pbf')
DEU_road = ex.extract_cis(DEU_geul, 'road')
DEU_rail = ex.extract_cis(DEU_geul, 'rail')
DEU_filtered_rail = DEU_rail[DEU_rail['railway'] == 'rail']

# Repeating for Belgium 
cp.clip_from_shapes([catchment],
                    osmpbf_output=osm_flex.config.OSM_DATA_DIR.joinpath('geul_BEL.osm.pbf'),
                    osmpbf_clip_from=osm_flex.config.OSM_DATA_DIR.joinpath('belgium-latest.osm.pbf'),
                    kernel='osmconvert', overwrite=True)
#print('BEL data is clipped')

BEL_geul = osm_flex.config.OSM_DATA_DIR.joinpath('geul_BEL.osm.pbf')
BEL_road = ex.extract_cis(BEL_geul, 'road')
BEL_rail = ex.extract_cis(BEL_geul, 'rail')
BEL_filtered_rail = BEL_rail[BEL_rail['railway'] == 'rail']

# Compile everything
all_road = [NL_road, BEL_road, DEU_road]
all_road = pd.concat(all_road)
all_road.reset_index(drop=True, inplace=True)
all_rail = [NL_filtered_rail, BEL_filtered_rail, DEU_filtered_rail]
all_rail = pd.concat(all_rail) 
all_rail.reset_index(drop=True, inplace=True)

# Provide input files
max_dam_road = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\max_damage_road.csv", delimiter = ";") 
curves_road = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\damage_curves_road.csv", delimiter = ";")

# Calculate the damages for roads
#loss_roads_df = VectorScanner(exposure_file=all_road,
 #                 hazard_file=inun_map,
  #                curve_path=curves_road,
   #               maxdam_path = max_dam_road,
    #              cell_size = 25,
     #             exp_crs=4326,
      #            haz_crs=28992,                   
       #           object_col='highway',
        #          hazard_col='inun_val',
         #         centimeters=True,
          #        save=True,
           #       plot=True,
            #      grouped=False,
             #     scenario_name = 'roads2'
              #    )

#total_damage = loss_roads_df['damage'].sum()
#print("Total damage of roads:", total_damage)

# Moving on to rails 
max_dam_rail = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\max_damage_rail.csv", delimiter = ";") 
curves_rail = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\damage_curves_rail.csv", delimiter = ";")

# Calculate the damages for roads
#loss_rail_df = VectorScanner(exposure_file=all_rail,
#                 hazard_file=inun_map,
#                 curve_path=curves_rail,
##                 maxdam_path = max_dam_rail,
 #                cell_size = 25,
 #                exp_crs=4326,
  #               haz_crs=28992,                   
   #              object_col='railway',
    #             hazard_col='inun_val',
     #            centimeters=True,
      #           save=True,
       #          plot=True,
        #         grouped=False,
         #        scenario_name = 'rail2'
          #       )

#total_damage = loss_rail_df['damage'].sum()
#print("Total damage to rail:", total_damage)


# Part 3: Calculate damage for buildings 

# Extract building data 
NL_geul = osm_flex.config.OSM_DATA_DIR.joinpath('geul_NL.osm.pbf')
NL_buildings_geul = ex.extract_cis(NL_geul, 'buildings')
BEL_geul = osm_flex.config.OSM_DATA_DIR.joinpath('geul_BEL.osm.pbf')
BEL_buildings_geul = ex.extract_cis(BEL_geul, 'buildings')
DEU_geul = osm_flex.config.OSM_DATA_DIR.joinpath('geul_DEU.osm.pbf')
DEU_buildings_geul = ex.extract_cis(DEU_geul, 'buildings')

# Make one dataframe containing all buildings
all = [NL_buildings_geul, BEL_buildings_geul, DEU_buildings_geul]
all_buildings_geul = pd.concat(all)
all_buildings_geul.reset_index(drop=True, inplace=True)


# Remove the points from the dataframe 
all_buildings_geul_polygons = all_buildings_geul[all_buildings_geul.geom_type != 'Point']
all_buildings_geul_polygons.loc[:, "landuse"] = 'building'
all_buildings_geul_polygons.reset_index(drop=True, inplace=True)
reproject_buildings = all_buildings_geul_polygons.to_crs(28992)
reproject_buildings["area"] = reproject_buildings["geometry"].area #calculate the area for all buildings
#print(reproject_buildings)
selected_buildings = reproject_buildings[reproject_buildings["area"] > 18] # Bouwbesluit: only allowed to live in buildings larger than 18 m2 https://rijksoverheid.bouwbesluit.com/Inhoud/docs/wet/bb2012/hfd4?tableid=docs/wet/bb2012[35]/hfd4/afd4-1/par4-1-1 
selected_buildings.reset_index(drop=True, inplace=True)
#print(selected_buildings)

max_dam_buildings = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\max_damage_buildings.csv", delimiter = ";") 
curves_buildings = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\damage_curves_buildings.csv", delimiter = ";")

# Try other flood map
#inun_map_vector = r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\flood_map_vector.shp"

loss_buildings_df = VectorScanner(exposure_file=selected_buildings,
                  hazard_file=inun_map,
                  curve_path=curves_buildings,
                  maxdam_path = max_dam_buildings,
                  cell_size = 25,
                  exp_crs= 28992, #4326
                  haz_crs=28992,                   
                  object_col='landuse', # Change to correct column
                  hazard_col='inun_val',
                  centimeters=True,
                  save=True,
                  plot=False,
                  grouped=False,
                  scenario_name = 'building_structure2'
                  )

total_damage = loss_buildings_df['damage'].sum()
print("Total damage to buildings structure:", total_damage)

# Let's move on to building contents 
max_dam_buildings_contents = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\max_damage_buildings_contents.csv", delimiter = ";") 
curves_buildings_contents = pd.read_csv(r"C:\Users\vbl220\OneDrive - Vrije Universiteit Amsterdam\PhD\P1_Flood risk module\damage_curves_buildings_contents.csv", delimiter = ";")

# Only take the center point for each building 
centroid_gdf = gpd.GeoDataFrame(geometry=selected_buildings.centroid)
centroid_gdf.loc[:, "landuse"] = 'building'
centroid_gdf.reset_index(drop=True, inplace=True)

loss_contents_df = VectorScanner(exposure_file=centroid_gdf,
                  hazard_file=inun_map,
                  curve_path=curves_buildings_contents,
                  maxdam_path = max_dam_buildings_contents,
                  cell_size = 25,
                  exp_crs=28992,
                  haz_crs=28992,                   
                  object_col='landuse', # Change to correct column
                  hazard_col='inun_val',
                  centimeters=True,
                  save=True,
                  plot=False,
                  grouped=False,
                  scenario_name = 'building_contents2'
                  )

total_damage = loss_contents_df['damage'].sum()
print("Total damage to building contents:", total_damage)