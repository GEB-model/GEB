from osgeo import gdal
import os
import numpy as np
import re
from collections import defaultdict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from hyve.library.raster import CreateNetCDF

data = {
    'rainfed': defaultdict(dict),
    'irrigated': defaultdict(dict)
}
folder = 'DataDrive/GEB/original_data/MIRCA2000/monthly_growing_areas'
for fn in os.listdir(folder):
    if not fn.endswith('.asc'):
        continue
    fp = os.path.join(folder, fn)
    res = re.findall(r'crop_([0-9]{2})_(rainfed|irrigated)_([0-9]{3}).asc', fn)[0]
    crop, kind, month = res
    
    data[kind][crop][month] = fp

crop_names = {
    1: "Wheat",
    2: "Maize",
    3: "Rice",
    4: "Barley",
    5: "Rye",
    6: "Millet",
    7: "Sorghum",
    8: "Soybeans",
    9: "Sunflower",
    10: "Potatoes",
    11: "Cassava",
    12: "Sugar cane",
    13: "Sugar beet",
    14: "Oil palm",
    15: "Rape seed / Canola",
    16: "Groundnuts / Peanuts",
    17: "Pulses",
    18: "Citrus",
    19: "Date palm",
    20: "Grapes / Vine",
    21: "Cotton",
    22: "Cocoa",
    23: "Coffee",
    24: "Others perennial",
    25: "Fodder grasses",
    26: "Others annual",
}

for kind, crops in data.items():
    for crop, months in crops.items():
        print(kind, crop)
        timesteps = [datetime(2000, 1, 1)]
        for _ in range(11):
            timesteps.append(timesteps[-1] + relativedelta(months=1))

        ds = gdal.Open(months['001'])
        gt = ds.GetGeoTransform()
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        lons = np.arange(gt[0] + gt[1] * 0.5, gt[0] + xsize * gt[1], gt[1])
        lats = np.arange(gt[3] + gt[5] * 0.5, gt[3] + ysize * gt[5], gt[5])
        varname = 'cropland'
        units = 'ha'
        dtype = 'f4'
        chunksizes = (1, 50, 50)
        fill_value = -1
        compression_level = 1
        title = 'MIRCA 2000 cropland in ha'
        source = 'MIRCA2000'
        
        with CreateNetCDF(
            os.path.join(folder, f'{kind}_{int(crop)-1:02d}.nc'),
            crop_names[int(crop)],
            source,
            "",
            len(timesteps),
            lons,
            lats,
            4326,
            varname,
            units,
            dtype,
            chunksizes,
            fill_value,
            compression_level,
        ) as f:
            for fp, dt in zip(months.values(), timesteps):
                ds = gdal.Open(fp)
                band = ds.GetRasterBand(1)
                array = band.ReadAsArray()
                f.write(array, dt=dt)
