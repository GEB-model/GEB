import pyflwdir
import os
import matplotlib.pyplot as plt
import rasterio
import numpy as np

fp = os.path.join('DataDrive/GEB/input', 'routing/kinematic/ldd.tif')

with rasterio.open(fp, 'r') as src:
    ldd = src.read(1)
    ldd[ldd == 0] = 5

with rasterio.open('DataDrive/GEB/input/areamaps/mask.tif', 'r') as src:
    mask = ~src.read(1).astype(bool)


flw = pyflwdir.from_array(ldd, mask=mask, cache=True)
subbasins = flw.subbasins_pfafstetter()
plt.imshow(subbasins)
plt.show()