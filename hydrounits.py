from typing import Union
from numba import njit
import rasterio
import os
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    pass
from cwatm.management_modules.data_handling import readnetcdfInitial, checkOption

@njit(cache=True)
def _decompress_subvar(subarray: np.ndarray, outarray: np.ndarray, subcell_locations: np.ndarray, mask: np.ndarray, scaling: int, ysize: int, xsize: int) -> np.ndarray:
    """Decompress subvar array.

    Args:
        subarray: Subarray.
        subcell_locations: Array that maps the locations of subcells to cells.
        scaling: The scaling used for map between cells and hydrounits.
        mask: Mask of study area.
        nanvalue: Value to use for values outside the mask.

    Returns:
        outarray: Decompressed subarray.
    """  
    i = 0
    
    for y in range(ysize):
        for x in range(xsize):
            is_masked = mask[y, x]
            if not is_masked:
                for ys in range(scaling):
                    for xs in range(scaling):
                        outarray[y * scaling + ys, x * scaling + xs] = subarray[subcell_locations[i]]
                        i += 1

    return outarray


class BaseVariables:
    """This class has some basic functions that can be used for variables regardless of scale."""
    def __init__(self):
        pass

    def plot(self, data: np.ndarray, ax=None) -> None:
        """Create a simple plot for data.
        
        Args:
            data: Array to plot.
            ax: Optional matplotlib axis object. If given, data will be plotted on given axes.
        """
        import matplotlib.pyplot as plt
        data = self.decompress(data)
        if ax:
            ax.imshow(data)
        else:
            plt.imshow(data)
            plt.show()

    def MtoM3(self, array: np.ndarray) -> np.ndarray:
        """Convert array from meters to cubic meters.
        
        Args:
            array: Data in meters.
            
        Returns:
            array: Data in cubic meters.
        """
        return array * self.cellArea

    def M3toM(self, array: np.ndarray) -> np.ndarray:
        """Convert array from cubic meters to meters.
        
        Args:
            array: Data in cubic meters.
            
        Returns:
            array: Data in meters.
        """
        return array / self.cellArea


class Variables(BaseVariables):
    """This class is to store data in the 'normal' grid cells. This class works with compressed and uncompressed arrays. On initialization of the class, the mask of the study area is read from disk. This is the shape of any uncompressed array. Many values in this array, however, fall outside the stuy area as they are masked. Therefore, the array can be compressed by saving only the non-masked values.
    
    On initialization, as well as geotransformation and cell size are set, and the cell area is read from disk.

    Then, the mask is compressed by removing all masked cells, resulting in a compressed array.
    """
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.scaling = 1
        mask_fn = 'DataDrive/CWatM/krishna/input/areamaps/mask.tif'
        with rasterio.open(mask_fn) as mask_src:
            self.mask = mask_src.read(1).astype(np.bool)
            self.gt = mask_src.transform.to_gdal()
            self.cell_size = mask_src.transform.a
        with rasterio.open('DataDrive/CWatM/krishna/input/areamaps/cell_area.tif') as cell_area_src:
            self.cell_area_uncompressed = cell_area_src.read(1)
        
        self.mask_flat = self.mask.ravel()
        self.compressed_size = self.mask_flat.size - self.mask_flat.sum()
        self.cellArea = self.compress(self.cell_area_uncompressed)
        BaseVariables.__init__(self)

    def full(self, *args, **kwargs) -> np.ndarray:
        """Return a full array with size of mask. Takes any other argument normally used in np.full.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        return np.full(self.mask.shape, *args, **kwargs)

    def full_compressed(self, *args, **kwargs) -> np.ndarray:
        """Return a full array with size of compressed array. Takes any other argument normally used in np.full.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        return np.full(self.compressed_size, *args, **kwargs)

    def compress(self, array: np.ndarray) -> np.ndarray:
        """Compress array.
        
        Args:
            array: Uncompressed array.
            
        Returns:
            array: Compressed array.
        """
        return array.ravel()[self.mask_flat == False]

    def decompress(self, array: np.ndarray, fillvalue: Union[np.ufunc, int, float]=None) -> np.ndarray:
        """Decompress array.
        
        Args:
            array: Compressed array.
            fillvalue: Value to use for masked values.
            
        Returns:
            array: Decompressed array.
        """
        if fillvalue is None:
            if array.dtype in (np.float32, np.float64):
                fillvalue = np.nan
            else:
                fillvalue = 0
        outmap = self.full(fillvalue, dtype=array.dtype).reshape(self.mask_flat.size)
        outmap[self.mask_flat == False] = array
        return outmap.reshape(self.mask.shape)

    def load_initial(self, name, default=0.0, number=None):
        raise ValueError
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

    def plot(self, array: np.ndarray) -> None:
        """Plot array.
        
        Args:
            array: Array to plot.
        """
        import matplotlib.pyplot as plt
        plt.imshow(array)
        plt.show()

    def plot_compressed(self, array: np.ndarray, fillvalue: Union[np.ufunc, int, float]=None):
        """Plot compressed array.
        
        Args:
            array: Compressed array to plot.
            fillvalue: Value to use for masked values.
        """
        self.plot(self.decompress(array, fillvalue=fillvalue))


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

    def decompress(self, subarray):
        if isinstance(subarray, cp.ndarray):
            array = subarray.get()
        if np.issubdtype(subarray, np.integer):
            nanvalue = -1
        else:
            nanvalue = np.nan
        ysize, xsize = self.model.data.var.mask.shape
        decompresssed = np.full((ysize * self.scaling, xsize * self.scaling), nanvalue, dtype=subarray.dtype)
        return _decompress_subvar(array, outarray=decompresssed, subcell_locations=self.subcell_locations, mask=self.model.data.var.mask, scaling=self.scaling, ysize=ysize, xsize=xsize)

    def plot(self, subarray, ax=None, show=True):
        import matplotlib.pyplot as plt
        assert subarray.size == self.land_use_type.size
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.decompress(subarray), resample=False)
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