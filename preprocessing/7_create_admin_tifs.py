import os
import math
import rasterio
import geopandas as gpd
import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio import Affine

from config import ORIGINAL_DATA, INPUT

def cut_bhima():
    output_file = os.path.join(INPUT, 'areamaps', 'subdistricts_bhima.shp')
    if not os.path.exists(output_file):
        subdistricts = gpd.read_file(os.path.join(ORIGINAL_DATA, 'subdistricts', 'subdistricts.shp')).to_crs(epsg=32643)
        study_area = gpd.read_file(os.path.join(ORIGINAL_DATA, 'study_region.geojson'))

        merged = gpd.overlay(subdistricts, study_area)
        merged['area_overlap'] = merged.area
        subdistricts['area_total'] = subdistricts.area

        joined = merged.merge(subdistricts[['OBJECTID', 'area_total']], on='OBJECTID')
        joined['overlap'] = joined['area_overlap'] / joined['area_total']
        joined.plot(column='overlap')
        joined = joined.drop(['area_overlap', 'area_total'], axis=1)
        joined['ID'] = joined.index
        joined.to_file(output_file)

def create_tif():
    output_file = os.path.join(INPUT, 'tehsils.tif')
    with rasterio.open(os.path.join(os.path.join(INPUT, 'areamaps', 'submask.tif')), 'r') as src:
        profile = src.profile
    gdf = gpd.read_file(os.path.join(INPUT, 'tehsils.geojson')).to_crs(profile['crs'])
    minx, miny, maxx, maxy = gdf.total_bounds
    affine = profile['transform']

    width = profile['width']
    height = profile['height']

    xmin_offset = math.ceil(((affine.c - minx) / affine.a))
    xmax_offset = math.ceil((maxx - (affine.a * profile['width'] + affine.c)) / affine.a)
    width += xmin_offset
    width += xmax_offset

    ymax_offset = math.ceil(((affine.f - maxy) / affine.e))
    ymin_offset = math.ceil((miny - (affine.e * profile['height'] + affine.f)) / affine.e)
    height += ymax_offset
    height += ymin_offset

    newaffine = Affine(
        a=affine.a,
        b=affine.b,
        c=affine.c - xmin_offset * affine.a,
        d=affine.d,
        e=affine.e,
        f=affine.f - ymax_offset * affine.e,
    )
    new_profile = dict(profile)  # copy
    new_profile['height'] = height
    new_profile['width'] = width
    new_profile['transform'] = newaffine
    new_profile['nodata'] = -1

    geometries = [
        (shapely.geometry.mapping(geom), value)
        for value, geom
        in zip(gdf['ID'].tolist(), gdf['geometry'].tolist())
    ]
    admin_areas = rasterize(
        geometries,
        out_shape=(new_profile['height'], new_profile['width']),
        fill=-1,
        transform=new_profile['transform'],
        dtype=np.int32,
        all_touched=False
    )
    with rasterio.open(output_file, 'w', **new_profile) as dst:
        dst.write(admin_areas, 1)

if __name__ == '__main__':
    cut_bhima()
    create_tif()