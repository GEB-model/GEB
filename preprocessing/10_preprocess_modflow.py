# -*- coding: utf-8 -*-
import numpy as np
import os
import geopandas as gpd
import math
import rasterio
import rasterio.features
from rasterio import Affine
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from rasterio.merge import merge

class ModflowPreprocess:
    """
    This class is used to do all modflow preprocessing. CWatM works in a lon-lat grid, while MODFLOW works in a cartesian grid. Therefore a custom MODFLOW grid is created that encompasses the entirety of the CWatM mask. In addition, a mapping is created that maps the area of a given cell in the CWatM grid to a given MODFLOW cell. All required data for MODFLOW is also projected to the newly created MODFLOW grid.

    Args:
        modflow_path: path where processed data will be saved.
        resolution: resolution in meters of MODFLOW data.
        cwatm_basin_mask_fn: filepath of cwatm basin mask
        modflow_epsg: MODFLOW EPSG
    """
    def __init__(self, modflow_path: str, resolution: int, cwatm_basin_mask_fn:str, modflow_epsg:int) -> None:
        self.output_folder = os.path.join(modflow_path, f'{resolution}m')   # Folder where input maps will be saved
        self.modflow_epsg = modflow_epsg
        self.modflow_resolution = resolution
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        with rasterio.open(cwatm_basin_mask_fn, 'r') as src:
            self.cwatm_basin_mask = src.read(1)  # Matrix containing the studied CWATM variable
            self.cwatm_profile = src.profile
            self.cwatm_epsg = self.cwatm_profile['crs'].to_epsg()

        self.modflow_affine, self.ncol_ModFlow, self.nrow_ModFlow = self.get_modflow_transform_and_size()

    def get_modflow_transform_and_size(self) -> tuple[Affine, int, int]:
        """
        Calculate modflow geotransformation and size of grid.

        Returns:
            modflow_affine: affine transformation for new MODFLOW grid.
            ncols: number of columns in the MODFLOW grid.
            nrows: number of rows in the MODFLOW grid.
        """
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
        ncols = math.ceil((xmax-xmin) / self.modflow_resolution)
        xmax = xmin + ncols * self.modflow_resolution
        ymin = np.min(b_utm)
        ymax = np.max(b_utm)
        nrows = math.ceil((ymax-ymin) / self.modflow_resolution)
        ymax = ymin + nrows * self.modflow_resolution

        gt_modlfow = (
            xmin, self.modflow_resolution, 0, ymax, 0, -self.modflow_resolution
        )

        modflow_affine = Affine.from_gdal(*gt_modlfow)

        return modflow_affine, ncols, nrows

    def create_raster_shapefile(self, transform: Affine, xsize: int, ysize: int, epsg: int) -> gpd.GeoDataFrame:
        """
        First creates a raster for the given transform and size, where the top-left cell is 0, and counting up, rows first. Then the raster is transformed into a GeoDataFrame (shapefile), where the numbers of the cells are used as IDs.

        Args:
            transform: geotransformation of the raster
            xsize: number of columns
            ysize: number of rows
            epsg: epsg of the raster

        Returns:
            gdf: GeoDataFrame with raster cells as geometries, each having a unique ID.
        """
        array = np.arange(ysize * xsize, dtype=np.int32).reshape((ysize, xsize))
        shapes = list(rasterio.features.shapes(array, transform=transform))
        geoms = [{'geometry': geom, 'properties': {'cell_id': int(v)}} for geom, v in shapes]
        gdf = gpd.GeoDataFrame.from_features(geoms, crs=epsg)

        assert np.all(np.diff(gdf['cell_id']) >= 0) # check if cell_ids are sorted
        gdf = gdf.drop('cell_id', axis=1)

        gdf['x'] = np.tile(np.arange(xsize), ysize)
        gdf['y'] = np.arange(ysize).repeat(xsize)

        return gdf

    def create_indices(self):
        """
        Creates a mapping of cells between MODFLOW and CWatM, and saves them to npy-files. The mapping contains the x and y index of a CWatM and MODFLOW cell and the area size (m\ :sup:`2`) of their overlap. Cell combinations that are not in the indices do not overlap.
        """

        # Create CWatM shapefile
        cwatm_gdf = self.create_raster_shapefile(self.cwatm_profile['transform'], self.cwatm_profile['width'], self.cwatm_profile['height'], self.cwatm_epsg)
        cwatm_gdf = cwatm_gdf.to_crs(epsg=self.modflow_epsg)
        print('created CWatM gdf')
        # Create MODFLOW shapefile
        modflow_gdf = self.create_raster_shapefile(self.modflow_affine, self.ncol_ModFlow, self.nrow_ModFlow, self.modflow_epsg)
        print('created modflow gdf')

        cwatm_gdf['cwatm_geometry'] = cwatm_gdf.geometry  # save geometry for after join
        # intersect CWatM and MODFLOW shapefiles
        intersect = gpd.sjoin(modflow_gdf, cwatm_gdf, how='inner', op='intersects', lsuffix='modflow', rsuffix='cwatm')

        # calculate size of intersection
        intersect['area'] = intersect.apply(lambda x: x.cwatm_geometry.intersection(x.geometry).area, axis=1)

        # and finally export
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

    def create_modflow_basin(self) -> None:
        """
        Creates a mask for the MODFLOW basin. All cells that have any area overlapping with any CWatM cell are considered to be part of the MODFLOW basin.
        """
        # Creating 1D arrays containing ModFlow and CWatM indices anf Interesected area [m2]
        ModFlow_index = np.array(self.modflow_y * self.ncol_ModFlow + self.modflow_x)
        CWatM_index = np.array(self.cwatm_y * self.cwatm_profile['width'] + self.cwatm_x)  # associated CWatM cell index

        ModFlowcellarea = self.modflow_resolution * self.modflow_resolution
        # Opening the file containing basin cells flag in the rectangular CWATM area (Mask)

        tempmask = np.invert(self.cwatm_basin_mask.astype(bool))
        # Looking at if the ModFlow cell is mainly out of the basin
        ratio_ModFlowcellarea = np.bincount(ModFlow_index, weights=tempmask.ravel()[CWatM_index] * self.area, minlength=self.nrow_ModFlow * self.ncol_ModFlow) / ModFlowcellarea
        basin_mask = np.ones(self.nrow_ModFlow * self.ncol_ModFlow, dtype=np.int32)
        basin_mask[ratio_ModFlowcellarea > 0] = 0
        basin_mask = basin_mask.reshape(self.nrow_ModFlow, self.ncol_ModFlow)

        with rasterio.open(os.path.join(self.output_folder, 'modflow_mask.tif'), 'w', driver='GTiff', width=self.ncol_ModFlow,
                height=self.nrow_ModFlow, count=1, dtype=np.int32, nodata=-1, transform=self.modflow_affine,
                epsg=self.modflow_epsg) as dst:
            dst.write(basin_mask, 1)
        self.basin_mask = basin_mask
        return basin_mask

    def project_input_map(self, input_map_fp: str, output_map_fp: str) -> None:
        """
        Projects input map to output map at resolution of the MODFLOW grid. Average resampling is used.

        Args:
            input_map_fp: filepath of the input map
            output_map_fp: filepath of the output map
        """
        with rasterio.open(input_map_fp) as src:
            src_transform = src.transform
            src_crs = src.crs
            source = src.read(1)
            source_nodata = src.profile['nodata']

        destination = np.zeros((self.nrow_ModFlow, self.ncol_ModFlow), dtype=source.dtype)

        reproject(source, destination, src_transform=src_transform, src_crs=src_crs, dst_transform=self.modflow_affine,
                    dst_crs=self.modflow_epsg, src_nodata=source_nodata, resampling=Resampling.average)
        
        with rasterio.open(os.path.join(self.output_folder, output_map_fp), 'w', driver='GTiff', width=self.ncol_ModFlow,
                            height=self.nrow_ModFlow, count=1, dtype=destination.dtype, nodata=0, transform=self.modflow_affine,
                            crs=self.modflow_epsg) as dst:
            dst.write(destination, indexes=1)
        return destination

if __name__ == '__main__':
    MODFLOW_PATH = 'DataDrive/GEB/input/groundwater/modflow'
    MODFLOW_RESOLUTION = 1000  # ModFlow model's resolution [m]
    cwatm_basin_mask_fn = "DataDrive/GEB/input/areamaps/mask.tif"  # Mask of the CWATM model
    MODFLOW_EPSG = 32643
    m = ModflowPreprocess(MODFLOW_PATH, MODFLOW_RESOLUTION, cwatm_basin_mask_fn, MODFLOW_EPSG)
    m.create_indices()
    m.create_modflow_basin()

    merit_hydro_03sec_folder = os.path.join('DataDrive', 'GEB', 'original_data', 'merit_hydro_03sec')
    elv_maps = []
    for fn in os.listdir(merit_hydro_03sec_folder):
        fp = os.path.join(merit_hydro_03sec_folder, fn)
        if os.path.splitext(fp)[0].endswith('_elv'):
            src = rasterio.open(fp)
            profile = src.profile
            elv_maps.append(rasterio.open(fp))
    
    DEM, transform = merge(elv_maps)
    DEM = DEM[0, :, :]
    profile.update({
        'transform': transform,
        'width': DEM.shape[1],
        'height': DEM.shape[0],
    })
    with rasterio.open(os.path.join(m.output_folder, 'elevation.tif'), 'w', **profile) as dst:
        dst.write(DEM, 1)

    m.project_input_map(os.path.join(m.output_folder, 'elevation.tif'), 'elevation_modflow.tif')