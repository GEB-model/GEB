from hyve.library.mapIO import ArrayReader
from hyve.data import BaseData

class Data(BaseData):
    def __init__(self, model):
        self.model = model
        self.data_folder = 'DataDrive'

        self.elevation = ArrayReader(
            fp='DataDrive/GEB/input/landsurface/topo/subelv.tif',
            bounds=self.model.bounds
        )