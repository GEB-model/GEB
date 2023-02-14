import os
import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt

from plotconfig import ORIGINAL_DATA

color = '#FAB337'

os.makedirs('barcharts', exist_ok=True)
for i in range(10):
    heights = np.random.randint(3, 10, 4)
    colors = ['#4298FF', '#66E036', '#A43ADE', '#F05B54']
    plt.bar(range(4), heights, color=colors, width=1)
    plt.axis('off')
    plt.savefig(f'barcharts/{i}.svg', bbox_inches='tight', pad_inches=0, transparent=True)

gdf = gpd.read_file(os.path.join(ORIGINAL_DATA, 'GADM', 'gadm36_IND_0.shp'))
gdf.plot(facecolor=color, figsize=(20, 20))
plt.axis('off')
plt.tight_layout()
plt.savefig('country.svg', transparent=True)

gdf = gpd.read_file(os.path.join(ORIGINAL_DATA, 'census', 'tehsils.shp'))
gdf = gdf[gdf['objectid'].isin((1488, 1473, 1491))]

def create_gdf(shape):
    if isinstance(shape, gpd.geoseries.GeoSeries):
        geometry = shape
    elif isinstance(shape, shapely.geometry.polygon.Polygon):
        geometry = [shape]
    else:
        raise NotImplementedError(f"Shape type {type(shape)} not implemented")
    return gpd.GeoDataFrame(geometry=geometry, crs=gdf.crs)

exterior = create_gdf(gdf.exterior)
unary_exterior = create_gdf(create_gdf(gdf.geometry.unary_union).exterior)
interior_lines = unary_exterior.overlay(exterior, how='symmetric_difference')

ax = gdf.plot(edgecolor='none', facecolor=color, figsize=(20, 20))
interior_lines.plot(ax=ax, color='k', lw=2)
plt.axis('off')
plt.tight_layout()
plt.savefig('tehsils.svg', transparent=True)