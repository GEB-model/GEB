import os
import math
import json
import rasterio
import geopandas as gpd
import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio import Affine

from preconfig import ORIGINAL_DATA, INPUT

def cut(study_region_fn):
    output_file = os.path.join(INPUT, 'areamaps', 'subdistricts.geojson')
    study_area = gpd.read_file(study_region_fn)
    subdistricts = gpd.read_file(os.path.join(ORIGINAL_DATA, 'census', 'tehsils.geojson')).to_crs(study_area.crs)

    merged = gpd.overlay(subdistricts, study_area).dissolve(by='objectid')
    # merged['area_overlap'] = merged.area
    # subdistricts['area_total'] = subdistricts.area

    joined = merged.merge(subdistricts['objectid'], on='objectid')
    overlapping_subdistricts = subdistricts[subdistricts['objectid'].isin(joined['objectid'])].reset_index(drop=True)
    overlapping_subdistricts['ID'] = joined.index
    overlapping_subdistricts.to_file(output_file)

def create_tif():
    output_file = os.path.join(INPUT, 'areamaps', 'tehsils.tif')
    with rasterio.open(os.path.join(os.path.join(INPUT, 'areamaps', 'submask.tif')), 'r') as src:
        profile = src.profile
    gdf = gpd.read_file(os.path.join(INPUT, 'areamaps', 'subdistricts.geojson')).to_crs(profile['crs'])
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
    # replace TELENGANA with TELANGANA in 2015 states
    gdf['15_state'] = gdf['15_state'].replace('TELENGANA', 'TELANGANA')
    # get dictionary of 2015 states for each subdistrict
    subdistrict2state = gdf.set_index('ID')['15_state'].to_dict()
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

    with open(os.path.join(INPUT, 'areamaps', 'subdistrict2state.json'), 'w') as f:
        json.dump(subdistrict2state, f)

if __name__ == '__main__':
    cut(os.path.join(INPUT, 'areamaps', 'mask.geojson'))
    create_tif()