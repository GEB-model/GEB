import os
import numpy as np
from dateutil.relativedelta import relativedelta
from hyve.library.mapreader import NetCDFReader, ArrayReader
from hyve.library.raster import write_to_array
from hyve.data import BaseData

class ArrayWriter:
    def __init__(self, model, gt, xsize, ysize, dtype):
        self.model = model
        self.array = np.zeros((xsize, ysize), dtype=dtype)
        self.gt = gt
        self.xsize = xsize
        self.ysize = ysize

    def write(self, coords, values):
        write_to_array(self.array, values, coords, self.gt, self.xsize, self.ysize)

    def __repr__(self):
        return self.array.__repr__()


class Data(BaseData):
    def __init__(self, model):
        self.model = model
        self.data_folder = 'DataDrive'

        self.elevation = ArrayReader(
            fp='DataDrive/GEB/input/landsurface/topo/subelv.tif',
            bounds=self.model.bounds
        )