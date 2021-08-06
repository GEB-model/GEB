# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:24:15 2019

@author: Luca G., Jens de Bruijn
"""

import numpy as np
import os
import geopandas as gpd
import math
import rasterio
import rasterio.features
from rasterio import Affine
from rasterio.warp import reproject, Resampling
from pyproj import Transformer

class ModflowPreprocess:
    def __init__(self, modflow_path, resolution, cwatm_basin_mask_fn, modflow_epsg):
        self.output_folder = os.path.join(modflow_path, f'{resolution}m')   # Folder where input maps will be saved
        self.modflow_epsg = modflow_epsg
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        with rasterio.open(cwatm_basin_mask_fn, 'r') as src:
            self.cwatm_basin_mask = src.read(1)  # Matrix containing the studied CWATM variable
            self.cwatm_profile = src.profile
            self.cwatm_epsg = self.cwatm_profile['crs'].to_epsg()

        self.modflow_affine, self.ncol_ModFlow, self.nrow_ModFlow = self.get_modflow_transform_and_size()

    def get_modflow_transform_and_size(self):
        cwatm_lon = [self.cwatm_profile['transform'].c + self.cwatm_profile['transform'].a * i for i in range(self.cwatm_profile['width'])]
        cwatm_lat = [self.cwatm_profile['transform'].f + self.cwatm_profile['transform'].e * i for i in range(self.cwatm_profile['height'])]
        
        a, b = np.meshgrid(
            np.append(cwatm_lon, 2*cwatm_lon[-1] - cwatm_lon[-2]),
            np.append(cwatm_lat, 2*cwatm_lat[-1] - cwatm_lat[-2])
        )
        
        transformer = Transformer.from_crs(f"epsg:{self.cwatm_epsg}", f"epsg:{self.modflow_epsg}")
        a_utm, b_utm = transformer.transform(b, a)

        xmin = np.min(a_utm)
        xmax = np.max(a_utm)
        ncols = math.ceil((xmax-xmin) / modflow_resolution)
        xmax = xmin + ncols * modflow_resolution
        ymin = np.min(b_utm)
        ymax = np.max(b_utm)
        nrows = math.ceil((ymax-ymin) / modflow_resolution)
        ymax = ymin + nrows * modflow_resolution

        gt_modlfow = (
            xmin, modflow_resolution, 0, ymax, 0, -modflow_resolution
        )

        modflow_affine = Affine.from_gdal(*gt_modlfow)

        return modflow_affine, ncols, nrows

    def create_raster(self, transform, xsize, ysize, epsg):
        array = np.arange(ysize * xsize, dtype=np.int32).reshape((ysize, xsize))
        shapes = list(rasterio.features.shapes(array, transform=transform))
        geoms = [{'geometry': geom, 'properties': {'cell_id': int(v)}} for geom, v in shapes]
        gdf = gpd.GeoDataFrame.from_features(geoms, crs=epsg)

        assert np.all(np.diff(gdf['cell_id']) >= 0) # check if cell_ids are sorted
        gdf = gdf.drop('cell_id', axis=1)

        gdf['x'] = np.tile(np.arange(xsize), ysize)
        gdf['y'] = np.arange(ysize).repeat(xsize)

        return gdf

    def create_rasters(self):
        cwatm_gdf = self.create_raster(self.cwatm_profile['transform'], self.cwatm_profile['width'], self.cwatm_profile['height'], self.cwatm_epsg)
        self.cwatm_gdf = cwatm_gdf.to_crs(epsg=modflow_epsg)
        print('created CWatM gdf')
        self.modflow_gdf = self.create_raster(self.modflow_affine, self.ncol_ModFlow, self.nrow_ModFlow, self.modflow_epsg)
        print('created modflow gdf')

    def create_indices(self):
        self.cwatm_gdf['cwatm_geometry'] = self.cwatm_gdf.geometry  # save geometry for after join
        intersect = gpd.sjoin(self.modflow_gdf, self.cwatm_gdf, how='inner', op='intersects', lsuffix='modflow', rsuffix='cwatm')

        intersect['area'] = intersect.apply(lambda x: x.cwatm_geometry.intersection(x.geometry).area, axis=1)

        self.modflow_y = intersect['y_modflow'].to_numpy()  # ModFlow column
        self.modflow_x = intersect['x_modflow'].to_numpy()  # ModFlow row
        self.cwatm_y = intersect['y_cwatm'].to_numpy()  # CWatM row
        self.cwatm_x = intersect['x_cwatm'].to_numpy()  # CWatM col
        self.area = intersect['area'].to_numpy()  # Area shared by each CWatM and ModFlow cell [m2]

        indices_folder = os.path.join(self.output_folder, 'indices')
        if not os.path.exists(indices_folder):
            os.makedirs(indices_folder)

        np.save(os.path.join(indices_folder, 'modflow_x.npy'), self.modflow_x)
        np.save(os.path.join(indices_folder, 'modflow_y.npy'), self.modflow_y)
        np.save(os.path.join(indices_folder, 'cwatm_x.npy'), self.cwatm_x)
        np.save(os.path.join(indices_folder, 'cwatm_y.npy'), self.cwatm_y)
        np.save(os.path.join(indices_folder, 'area.npy'), self.area)

    def create_modflow_basin(self):
        # Creating 1D arrays containing ModFlow and CWatM indices anf Interesected area [m2]
        ModFlow_index = np.array(self.modflow_y * self.ncol_ModFlow + self.modflow_x)
        CWatM_index = np.array(self.cwatm_y * self.cwatm_profile['width'] + self.cwatm_x)  # associated CWatM cell index

        ModFlowcellarea = modflow_resolution * modflow_resolution
        # Opening the file containing basin cells flag in the rectangular CWATM area (Mask)

        tempmask = np.invert(self.cwatm_basin_mask.astype(bool))
        # Looking at if the ModFlow cell is mainly out of the basin
        ratio_ModFlowcellarea = np.bincount(ModFlow_index, weights=tempmask.ravel()[CWatM_index] * self.area, minlength=self.nrow_ModFlow * self.ncol_ModFlow) / ModFlowcellarea
        basin_limits = np.zeros(self.nrow_ModFlow * self.ncol_ModFlow, dtype=np.int32)
        basin_limits[ratio_ModFlowcellarea > 0] = 1  # Cell = 0 will be INACTIVE in ModFlow simulation
        basin_limits = basin_limits.reshape(self.nrow_ModFlow, self.ncol_ModFlow)

        with rasterio.open(os.path.join(self.output_folder, 'modflow_basin.tif'), 'w', driver='GTiff', width=self.ncol_ModFlow,
                height=self.nrow_ModFlow, count=1, dtype=np.int32, nodata=-1, transform=self.modflow_affine,
                epsg=modflow_epsg) as dst:
            dst.write(basin_limits, 1)

    def project_input_map(self, input_map, output_map):
        with rasterio.open(input_map) as src:
            src_transform = src.transform
            src_crs = src.crs
            source = src.read(1)
            source[source == src.profile['nodata']] = 0

        destination = np.zeros((self.nrow_ModFlow, self.ncol_ModFlow), dtype=source.dtype)

        reproject(source, destination, src_transform=src_transform, src_crs=src_crs, dst_transform=self.modflow_affine,
                    dst_crs=modflow_epsg, resampling=Resampling.average)
        
        with rasterio.open(os.path.join(self.output_folder, output_map), 'w', driver='GTiff', width=self.ncol_ModFlow,
                            height=self.nrow_ModFlow, count=1, dtype=destination.dtype, nodata=0, transform=self.modflow_affine,
                            crs=modflow_epsg) as dst:
            dst.write(destination, indexes=1)

if __name__ == '__main__':
    main_path = 'DataDrive/GEB/input/groundwater/modflow'
    modflow_resolution = 2000  # ModFlow model's resolution [m]
    cwatm_basin_mask_fn = "DataDrive/GEB/input/areamaps/mask.tif"  # Mask of the CWATM model ##
    modflow_epsg = 32643
    m = ModflowPreprocess(main_path, modflow_resolution, cwatm_basin_mask_fn, modflow_epsg)
    # m.create_rasters()
    # m.create_indices()
    # m.create_modflow_basin()
    m.project_input_map('DataDrive/GEB/original_data/merit_hydro_03sec/elv.tif', 'elevation_modflow.tif')