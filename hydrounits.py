from numba import njit
import rasterio
import os
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
from cwatm.management_modules.data_handling import loadsetclone, metaNetCDF, readnetcdfInitial, checkOption, cbinding

@njit(cache=True)
def _decompress_subvar(mixed_array, subcell_locations, scaling, mask):
    ysize, xsize = mask.shape
    subarray = np.full((ysize * scaling, xsize * scaling), np.nan, dtype=mixed_array.dtype)
    
    i = 0
    
    for y in range(ysize):
        for x in range(xsize):
            is_masked = mask[y, x]
            if not is_masked:
                for ys in range(scaling):
                    for xs in range(scaling):
                        subarray[y * scaling + ys, x * scaling + xs] = mixed_array[subcell_locations[i]]
                        i += 1

    return subarray


class BaseVariables:
    def __init__(self):
        pass

    def plot(self, data, ax=None):
        import matplotlib.pyplot as plt
        data = self.decompress(data)
        if ax:
            ax.imshow(data)
        else:
            plt.imshow(data)
            plt.show()

    def MtoM3(self, array):
        return array * self.cellArea

    def M3toM(self, array):
        return array / self.cellArea


class Variables(BaseVariables):
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.scaling = 1
        self.load_mask()
        BaseVariables.__init__(self)

    @property
    def size(self):
        return self.compressed_size

    def full(self, *args, **kwargs):
        return np.full(self.mask.shape, *args, **kwargs)

    def full_compressed(self, *args, **kwargs):
        return np.full(self.compressed_size, *args, **kwargs)

    def load_mask(self):
        mask_fn = 'DataDrive/CWatM/krishna/input/areamaps/mask.tif'
        with rasterio.open(mask_fn) as mask_src:
            self.mask = mask_src.read()[0]
            self.gt = mask_src.transform.to_gdal()
            self.cell_size = mask_src.transform.a
            # assert self.cell_size == -mask_src.transform.e
        
        self.mask_flat = self.mask.ravel()
        self.compressed_size = self.mask_flat.size - self.mask_flat.sum()
        with rasterio.open('DataDrive/CWatM/krishna/input/areamaps/cell_area.tif') as cell_area_src:
            self.cellArea_uncompressed = cell_area_src.read(1)
            self.cellArea = self.compress(self.cellArea_uncompressed)

    def compress(self, array):
        return array.ravel()[self.mask_flat == False]

    def decompress(self, array, nanvalue=None):
        if nanvalue is None:
            if array.dtype in (np.float32, np.float64):
                nanvalue = np.nan
            else:
                nanvalue = 0
        outmap = self.full(nanvalue, dtype=array.dtype).reshape(self.mask_flat.size)
        outmap[self.mask_flat == False] = array
        return outmap.reshape(self.mask.shape)

    def load_initial(self, name, default=0.0, number=None):
        """
        First it is checked if the initial value is given in the settings file

        * if it is <> None it is used directly
        * if None it is loaded from the init netcdf file

        :param name: Name of the init value
        :param default: default value -> default is 0.0
        :param number: in case of snow or runoff concentration several layers are included: number = no of the layer
        :return: spatial map or value of initial condition
        """

        if number is not None:
            name = name + str(number)

        if self.loadInit:
            return readnetcdfInitial(self.initLoadFile, name)
        else:
            return default

    def plot(self, array):
        plt.imshow(array)
        plt.show()

    def plot_compressed(self, array, nanvalue=None):
        self.plot(self.decompress(array, nanvalue=nanvalue))


class HydroUnits(BaseVariables):
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.scaling = 20

        self.cell_size = self.data.var.cell_size / self.scaling
        self.land_use_type, self.land_use_ratios, self.land_owners, self.subvar_to_var, self.var_to_subvar, self.var_to_subvar_uncompressed, self.subcell_locations = self.create_subcell_mask()
        self.land_use_type[self.land_use_type == 2] = 1
        self.land_use_type[self.land_use_type == 3] = 1
        if self.model.config['general']['use_gpu']:
            self.land_owners = cp.array(self.land_owners)
            self.land_use_type = cp.array(self.land_use_type)
        BaseVariables.__init__(self)

    @property
    def size(self):
        return self.land_use_type.size

    @staticmethod
    @njit()
    def _create_subcell_mask(farms, land_use_classes, mask, scaling):
        ysize, xsize = mask.shape

        n_nonmasked_cells = mask.size - mask.sum()
        var_to_subvar = np.full(n_nonmasked_cells, -1, dtype=np.int32)
        var_to_subvar_uncompressed = np.full(mask.size, -1, dtype=np.int32)
        subvar_to_var = np.full(farms.size, -1, dtype=np.int32)
        land_use_array = np.full(farms.size, -1, dtype=np.int32)
        land_use_size = np.full(farms.size, -1, dtype=np.int32)
        land_use_owner = np.full(farms.size, -1, dtype=np.int32)
        subcells_locations = np.full(farms.size, -1, dtype=np.uint32)

        j = 0
        var_cell_count_compressed = 0
        l = 0
        var_cell_count_uncompressed = 0

        for y in range(0, ysize):
            for x in range(0, xsize):
                is_masked = mask[y, x]
                if not is_masked:
                    cell_farms = farms[y * scaling : (y + 1) * scaling, x * scaling : (x + 1) * scaling].ravel()  # find farms in cell
                    cell_land_use_classes = land_use_classes[y * scaling : (y + 1) * scaling, x * scaling : (x + 1) * scaling].ravel()  # get land use classes for cells
                    assert ((cell_land_use_classes == 0) | (cell_land_use_classes == 1) | (cell_land_use_classes == 4) | (cell_land_use_classes == 5)).all()

                    sort_idx = np.argsort(cell_farms)
                    cell_farms_sorted = cell_farms[sort_idx]
                    cell_land_use_classes_sorted = cell_land_use_classes[sort_idx]

                    prev_farm = -1  # farm is never -1
                    for i in range(cell_farms_sorted.size):
                        farm = cell_farms_sorted[i]
                        land_use = cell_land_use_classes_sorted[i]
                        if farm == -1:  # if area is not a farm
                            continue
                        if farm != prev_farm:
                            assert land_use_array[j] == -1
                            land_use_array[j] = land_use
                            assert land_use_size[j] == -1
                            land_use_size[j] = 1
                            land_use_owner[j] = farm

                            subvar_to_var[j] = var_cell_count_compressed

                            prev_farm = farm
                            j += 1
                        else:
                            land_use_size[j-1] += 1

                        subcells_locations[sort_idx[i] + var_cell_count_compressed * (scaling ** 2)] = j - 1
                        l += 1

                    sort_idx = np.argsort(cell_land_use_classes)
                    cell_farms_sorted = cell_farms[sort_idx]
                    cell_land_use_classes_sorted = cell_land_use_classes[sort_idx]

                    prev_land_use = -1
                    assert prev_land_use != cell_land_use_classes[0]
                    for i in range(cell_farms_sorted.size):
                        land_use = cell_land_use_classes_sorted[i]
                        farm = cell_farms_sorted[i]
                        if farm != -1:
                            continue
                        if land_use != prev_land_use:
                            assert land_use_array[j] == -1
                            land_use_array[j] = land_use
                            assert land_use_size[j] == -1
                            land_use_size[j] = 1
                            prev_land_use = land_use

                            subvar_to_var[j] = var_cell_count_compressed

                            j += 1
                        else:
                            land_use_size[j-1] += 1

                        subcells_locations[sort_idx[i] + var_cell_count_compressed * (scaling ** 2)] = j - 1
                        l += 1

                    var_to_subvar[var_cell_count_compressed] = j
                    var_cell_count_compressed += 1
                var_to_subvar_uncompressed[var_cell_count_uncompressed] = j
                var_cell_count_uncompressed += 1
        
        land_use_size = land_use_size[:j]
        land_use_array = land_use_array[:j]
        land_use_owner = land_use_owner[:j]
        subvar_to_var = subvar_to_var[:j]
        subcells_locations = subcells_locations[:var_cell_count_compressed * (scaling ** 2)]
        assert int(land_use_size.sum()) == n_nonmasked_cells * (scaling ** 2)
        
        land_use_ratio = land_use_size / (scaling ** 2)
        return land_use_array, land_use_ratio, land_use_owner, subvar_to_var, var_to_subvar, var_to_subvar_uncompressed, subcells_locations

    def create_subcell_mask(self):
        with rasterio.open(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'farms.tif'), 'r') as farms_src:
            farms = farms_src.read()[0]
        with rasterio.open(os.path.join('DataDrive', 'GEB', 'input', 'landsurface', 'land_use_classes.tif'), 'r') as src:
            land_use_classes = src.read()[0]
        return self._create_subcell_mask(farms, land_use_classes, self.data.var.mask, self.scaling)

    def zeros(self, *args, **kwargs):
        if checkOption('useGPU'):
            return cp.zeros(*args, **kwargs)
        else:
            return np.zeros(*args, **kwargs)        

    def full_compressed(self, fill_value, dtype, *args, **kwargs):
        if checkOption('useGPU'):
            return cp.full(self.land_use_type.size, fill_value, dtype, *args, **kwargs)
        else:
            return np.full(self.land_use_type.size, fill_value, dtype, *args, **kwargs)

    def load_initial(self, *args, **kwargs):
        return self.model.data.var.load_initial(*args, **kwargs)

    @staticmethod
    @njit(cache=True)
    def _load_map(array, scaling, output_size, has_subcells, fn=np.mean):
        output_array = np.empty(output_size, dtype=array.dtype)
        j = 0

        ysize, xsize = has_subcells.shape
        for y in range(0, ysize):
            for x in range(0, xsize):
                cell_has_subcells = has_subcells[y, x]
                cell = array[y * scaling : (y + 1) * scaling, x * scaling : (x + 1) * scaling]
                if cell_has_subcells:
                    j_next = j + (scaling ** 2)
                    output_array[j: j_next] = cell.ravel()
                    j = j_next
                else:
                    output_array[j] = np.mean(cell)
                    j += 1

        return output_array

    def load_map(self, fp, fn):
        with rasterio.open(fp) as src:
            array = src.read()[0]
        output = self._load_map(array, self.scaling, self.mixed_size, self.has_subcells.reshape(self.model.data.var.mask.shape))
        if self.model.config['general']['use_gpu']:
            return cp.array(output)
        else:
            return output

    def decompress(self, array):
        if isinstance(array, cp.ndarray):
            array = array.get()
        return _decompress_subvar(array, self.subcell_locations, self.scaling, self.model.data.var.mask)

    def plot(self, mixed_array, ax=None, show=True):
        assert mixed_array.size == self.land_use_type.size
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.decompress(mixed_array), resample=False)
        if show:
            plt.show()



class Data:
    def __init__(self, model):
        self.model = model
        self.var = Variables(self, model)
        self.subvar = HydroUnits(self, model)
        self.subvar.cellArea = self.to_subvar(data=self.var.cellArea, fn='mean')

    @staticmethod
    @njit
    def _to_subvar(array, var_to_subvar, area_sizes, mask=None, fn=None):
        assert var_to_subvar[-1] == area_sizes.size
        assert array.shape == var_to_subvar.shape
        output_array = np.zeros(area_sizes.size, dtype=array.dtype)
        prev_index = 0

        if mask.size == 0:  # no mask
            if fn is None:
                for i in range(var_to_subvar.size):
                    cell_index = var_to_subvar[i]
                    output_array[prev_index:cell_index] = array[i]
                    prev_index = cell_index
            elif fn == 'mean':
                for i in range(var_to_subvar.size):
                    cell_index = var_to_subvar[i]
                    cell_sizes = area_sizes[prev_index:cell_index]
                    output_array[prev_index:cell_index] = array[i] / cell_sizes.sum() * cell_sizes
                    prev_index = cell_index
            else:
                raise NotImplementedError
        else:
            if fn is None:
                for i in range(var_to_subvar.size):
                    cell_index = var_to_subvar[i]
                    output_array[prev_index:cell_index][~mask[prev_index:cell_index]] = array[i]
                    prev_index = cell_index
            else:
                raise NotImplementedError
                
        return output_array

    def to_subvar(self, *, data=None, varname=None, fn=None, mask=np.zeros(0, dtype=np.bool), delete=True):
        assert bool(data is not None) != bool(varname is not None)
        if varname:
            data = getattr(self.var, varname)
        assert not isinstance(data, list)
        if isinstance(data, (float, int)):  # check if data is simple float. Otherwise should be numpy array.
            outdata = data
        else:
            outdata = self._to_subvar(data, self.subvar.var_to_subvar, self.subvar.land_use_ratios, mask=mask, fn=fn)
            if self.model.config['general']['use_gpu']:
                outdata = cp.asarray(outdata)
        
        if varname:
            if delete:
                delattr(self.var, varname)
            setattr(self.subvar, varname, outdata)
        return outdata

    @staticmethod
    @njit
    def _to_var(array, var_to_subvar, cell_sizes, fn='mean'):
        output_array = np.empty(var_to_subvar.size, dtype=array.dtype)
        assert var_to_subvar[-1] == cell_sizes.size

        prev_index = 0
        for i in range(var_to_subvar.size):
            cell_index = var_to_subvar[i]
            if fn == 'mean':
                values = array[prev_index:cell_index]
                weights = cell_sizes[prev_index:cell_index]
                output_array[i] = (values * weights).sum() / weights.sum()
            elif fn == 'sum':
                output_array[i] = np.sum(array[prev_index:cell_index])
            elif fn == 'nansum':
                output_array[i] = np.nansum(array[prev_index:cell_index])
            elif fn == 'max':
                output_array[i] = np.max(array[prev_index: cell_index])
            elif fn == 'min':
                output_array[i] = np.min(array[prev_index: cell_index])
            else:
                raise NotImplementedError
            prev_index = cell_index
        return output_array

    def to_var(self, *, subdata=None, varname=None, fn=None, delete=True):
        assert bool(subdata is not None) != bool(varname is not None)
        assert fn is not None
        if varname:
            subdata = getattr(self.subvar, varname)
        assert not isinstance(subdata, list)
        if isinstance(subdata, float):  # check if data is simple float. Otherwise should be numpy array.
            outdata = subdata
        else:
            if self.model.config['general']['use_gpu'] and isinstance(subdata, cp.ndarray):
                subdata = subdata.get()
            outdata = self._to_var(subdata, self.subvar.var_to_subvar, self.subvar.land_use_ratios, fn)

        if varname:
            if delete:
                delattr(self.subvar, varname)
            setattr(self.var, varname, outdata)
        return outdata