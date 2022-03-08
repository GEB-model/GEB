# -*- coding: utf-8 -*-
from typing import Any, Union
from numba import njit
import rasterio
import os
import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass

import matplotlib.pyplot as plt


class BaseVariables:
    """This class has some basic functions that can be used for variables regardless of scale."""
    def __init__(self):
        pass

    @property
    def shape(self):
        return self.mask.shape

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

    def load_initial(self, name, default=.0, gpu=False):
        if self.model.load_initial:
            fp = os.path.join(self.model.initial_conditions_folder, f"{name}.npy")
            if gpu:
                return cp.load(fp)
            else:
                return np.load(fp)
        else:
            return default
class Grid(BaseVariables):
    """This class is to store data in the 'normal' grid cells. This class works with compressed and uncompressed arrays. On initialization of the class, the mask of the study area is read from disk. This is the shape of any uncompressed array. Many values in this array, however, fall outside the stuy area as they are masked. Therefore, the array can be compressed by saving only the non-masked values.
    
    On initialization, as well as geotransformation and cell size are set, and the cell area is read from disk.

    Then, the mask is compressed by removing all masked cells, resulting in a compressed array.
    """
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.scaling = 1
        mask_fn = os.path.join(self.model.config['general']['input_folder'], 'areamaps', 'mask.tif')
        with rasterio.open(mask_fn) as mask_src:
            self.mask = mask_src.read(1).astype(bool)
            self.gt = mask_src.transform.to_gdal()
            self.bounds = mask_src.bounds
            self.cell_size = mask_src.transform.a
        cell_area_fn = os.path.join(self.model.config['general']['input_folder'], 'areamaps', 'cell_area.tif')
        with rasterio.open(cell_area_fn) as cell_area_src:
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

    def load_initial(self, name, default=.0):
        return super().load_initial('grid.' + name, default=default)


class HRUs(BaseVariables):
    """This class forms the basis for the HRUs. To create the `HRUs`, each individual field owned by a farmer becomes a `HRU` first. Then, in addition, each other land use type becomes a separate HRU. `HRUs` never cross cell boundaries. This means that farmers whose fields are dispersed across multiple cells are simulated by multiple `HRUs`. Here, we assume that each `HRU`, is relatively homogeneous as it each `HRU` is operated by 1) a single farmer, or by a single other (i.e., non-farm) land-use type and 2) never crosses the boundary a hydrological model cell.

    On initalization, the mask of the study area for the cells are loaded first, and a mask on the maximum resolution of the HRUs is created. In this case, the maximum resolution of the HRUs is 20 times higher than the mask. Then the HRUs are actually created.

    Args:
        data: Data class for model.
        model: The GEB model.
    """
    def __init__(self, data, model) -> None:
        self.data = data
        self.model = model
        self.scaling = 20
        self.gt = (self.data.grid.gt[0], self.data.grid.gt[1] / self.scaling, self.data.grid.gt[2], self.data.grid.gt[3], self.data.grid.gt[4], self.data.grid.gt[5] / self.scaling)

        self.mask = self.data.grid.mask.repeat(self.scaling, axis=0).repeat(self.scaling, axis=1)
        self.cell_size = self.data.grid.cell_size / self.scaling
        self.land_use_type, self.land_use_ratio, self.land_owners, self.HRU_to_grid, self.grid_to_HRU, self.unmerged_HRU_indices = self.create_HRUs()
        if self.model.args.use_gpu:
            self.land_use_type = cp.array(self.land_use_type)
        BaseVariables.__init__(self)

    @property
    def compressed_size(self) -> int:
        """Gets the compressed size of a full HRU array.
        
        Returns:
            compressed_size: Compressed size of HRU array.
        """
        return self.land_use_type.size

    @staticmethod
    @njit(cache=True)
    def create_HRUs_numba(farms, land_use_classes, mask, scaling) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Numba helper function to create HRUs.
        
        Args:
            farms: Map of farms. Each unique integer is a unique farm. -1 is no farm.
            land_use_classes: CWatM land use class map [0-5].
            mask: Mask of the normal grid cells.
            scaling: Scaling between mask and maximum resolution of HRUs.

        Returns:
            land_use_array: Land use of each HRU.
            land_use_ratio: Relative size of HRU to grid.
            land_use_owner: Owner of HRU.
            HRU_to_grid: Maps HRUs to index of compressed cell index.
            grid_to_HRU: Array of size of the compressed grid cells. Each value maps to the index of the first unit of the next cell.
            unmerged_HRU_indices: The index of the HRU to the grid cell.
        """
        assert farms.size == mask.size * scaling * scaling
        assert farms.size == land_use_classes.size
        ysize, xsize = mask.shape

        n_nonmasked_cells = mask.size - mask.sum()
        grid_to_HRU = np.full(n_nonmasked_cells, -1, dtype=np.int32)
        HRU_to_grid = np.full(farms.size, -1, dtype=np.int32)
        land_use_array = np.full(farms.size, -1, dtype=np.int32)
        land_use_size = np.full(farms.size, -1, dtype=np.int32)
        land_use_owner = np.full(farms.size, -1, dtype=np.int32)
        unmerged_HRU_indices = np.full(farms.shape, -1, dtype=np.int32)

        HRU = 0
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
                            assert land_use_array[HRU] == -1
                            land_use_array[HRU] = land_use
                            assert land_use_size[HRU] == -1
                            land_use_size[HRU] = 1
                            land_use_owner[HRU] = farm

                            HRU_to_grid[HRU] = var_cell_count_compressed

                            prev_farm = farm
                            HRU += 1
                        else:
                            land_use_size[HRU-1] += 1

                        unmerged_HRU_indices[y * scaling + sort_idx[i] // scaling, x * scaling + sort_idx[i] % scaling] = HRU - 1
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
                            assert land_use_array[HRU] == -1
                            land_use_array[HRU] = land_use
                            assert land_use_size[HRU] == -1
                            land_use_size[HRU] = 1
                            prev_land_use = land_use

                            HRU_to_grid[HRU] = var_cell_count_compressed

                            HRU += 1
                        else:
                            land_use_size[HRU-1] += 1

                        unmerged_HRU_indices[y * scaling + sort_idx[i] // scaling, x * scaling + sort_idx[i] % scaling] = HRU - 1
                        l += 1

                    grid_to_HRU[var_cell_count_compressed] = HRU
                    var_cell_count_compressed += 1
                var_cell_count_uncompressed += 1
        
        land_use_size = land_use_size[:HRU]
        land_use_array = land_use_array[:HRU]
        land_use_owner = land_use_owner[:HRU]
        HRU_to_grid = HRU_to_grid[:HRU]
        assert int(land_use_size.sum()) == n_nonmasked_cells * scaling * scaling
        
        land_use_ratio = land_use_size / (scaling ** 2)
        return land_use_array, land_use_ratio, land_use_owner, HRU_to_grid, grid_to_HRU, unmerged_HRU_indices

    def create_HRUs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Function to create HRUs.
        
        Returns:
            land_use_array: Land use of each HRU.
            land_use_ratio: Relative size of HRU to grid.
            land_use_owner: Owner of HRU.
            HRU_to_grid: Maps HRUs to index of compressed cell index.
            grid_to_HRU: Array of size of the compressed grid cells. Each value maps to the index of the first unit of the next cell.
            unmerged_HRU_indices: The index of the HRU to the subcell.
            """
        with rasterio.open(os.path.join(self.model.config['general']['input_folder'], 'agents', 'farms.tif'), 'r') as farms_src:
            farms = farms_src.read()[0]
        with rasterio.open(os.path.join(self.model.config['general']['input_folder'], 'landsurface', 'land_use_classes.tif'), 'r') as src:
            land_use_classes = src.read()[0]
        return self.create_HRUs_numba(farms, land_use_classes, self.data.grid.mask, self.scaling)

    def zeros(self, size, dtype, *args, **kwargs) -> np.ndarray:
        """Return an array (CuPy or Numpy) of zeros with given size. Takes any other argument normally used in np.zeros.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            array: Array with size of number of HRUs.
        """
        if self.model.args.use_gpu:
            return cp.zeros(size, dtype, *args, **kwargs)
        else:
            return np.zeros(size, dtype, *args, **kwargs)        

    def full_compressed(self, fill_value, dtype, *args, **kwargs) -> np.ndarray:
        """Return a full array with size of number of HRUs. Takes any other argument normally used in np.full.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            array: Array with size of number of HRUs.
        """
        if self.model.args.use_gpu:
            return cp.full(self.compressed_size, fill_value, dtype, *args, **kwargs)
        else:
            return np.full(self.compressed_size, fill_value, dtype, *args, **kwargs)

    def decompress(self, HRU_array: np.ndarray) -> np.ndarray:
        """Decompress HRU array.

        Args:
            HRU_array: HRU_array.

        Returns:
            outarray: Decompressed HRU_array.
        """  
        if self.model.args.use_gpu:
            HRU_array = HRU_array.get()
        if np.issubdtype(HRU_array.dtype, np.integer):
            nanvalue = -1
        else:
            nanvalue = np.nan
        outarray = HRU_array[self.unmerged_HRU_indices]
        outarray[self.mask] = nanvalue
        return outarray

    def plot(self, HRU_array: np.ndarray, ax=None, show: bool=True):
        """Function to plot HRU data.

        Args:
            HRU_array: Data to plot. Size must be equal to number of HRUs.
            ax: Optional matplotlib axis object. If given, data will be plotted on given axes.
            show: Boolean whether to show the plot or not.
        """
        import matplotlib.pyplot as plt
        assert HRU_array.size == self.compressed_size
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.decompress(HRU_array), resample=False)
        if show:
            plt.show()

    def load_initial(self, name, default=.0):
        return super().load_initial('HRU.' + name, default=default, gpu=self.model.args.use_gpu)


class Modflow(BaseVariables):
    def __init__(self, data, model):
        self.data = data
        self.model = model

        BaseVariables.__init__(self)

    def load_initial(self, name, default=.0):
        return super().load_initial('modflow.' + name, default=default)
    
class Data:
    """The base data class for the GEB model. This class contains the data for the normal grid, the HRUs, and has methods to convert between the grid and HRUs.
    
    Args:
        model: The GEB model.
    """
    def __init__(self, model):
        self.model = model
        self.grid = Grid(self, model)
        self.HRU = HRUs(self, model)
        self.HRU.cellArea = self.to_HRU(data=self.grid.cellArea, fn='mean')
        self.modflow = Modflow(self, model)

    @staticmethod
    @njit(cache=True)
    def to_HRU_numba(data, grid_to_HRU, land_use_ratio, fn=None):
        """Numba helper function to convert from grid to HRU.
        
        Args:
            data: The grid data to be converted.
            grid_to_HRU: Array of size of the compressed grid cells. Each value maps to the index of the first unit of the next cell.
            land_use_ratio: Relative size of HRU to grid.
            fn: Name of function to apply to data. None if data should be directly inserted into HRUs - generally used when units are irrespective of area. 'mean' if data should first be corrected relative to the land use ratios - generally used when units are relative to area.

        Returns:
            ouput_data: Data converted to HRUs.
        """
        assert grid_to_HRU[-1] == land_use_ratio.size
        assert data.shape == grid_to_HRU.shape
        output_data = np.zeros(land_use_ratio.size, dtype=data.dtype)
        prev_index = 0

        if fn is None:
            for i in range(grid_to_HRU.size):
                cell_index = grid_to_HRU[i]
                output_data[prev_index:cell_index] = data[i]
                prev_index = cell_index
        elif fn == 'mean':
            for i in range(grid_to_HRU.size):
                cell_index = grid_to_HRU[i]
                cell_sizes = land_use_ratio[prev_index:cell_index]
                output_data[prev_index:cell_index] = data[i] / cell_sizes.sum() * cell_sizes
                prev_index = cell_index
        else:
            raise NotImplementedError

        return output_data

    def to_HRU(self, *, data=None, varname=None, fn=None, delete=True):
        """Function to convert from grid to HRU.
        
        Args:
            data: The grid data to be converted (if set, varname cannot be set).
            varname: Name of variable to be converted. Must be present in grid class. (if set, data cannot be set).
            fn: Name of function to apply to data. None if data should be directly inserted into HRUs - generally used when units are irrespective of area. 'mean' if data should first be corrected relative to the land use ratios - generally used when units are relative to area.
            delete: Whether to delete the data from the grid class. Can only be set if varname is given.

        Returns:
            ouput_data: Data converted to HRUs.
        """
        assert bool(data is not None) != bool(varname is not None)
        if varname:
            data = getattr(self.grid, varname)
        assert not isinstance(data, list)
        if isinstance(data, (float, int)):  # check if data is simple float. Otherwise should be numpy array.
            outdata = data
        else:
            outdata = self.to_HRU_numba(data, self.HRU.grid_to_HRU, self.HRU.land_use_ratio, fn=fn)
            if self.model.args.use_gpu:
                outdata = cp.asarray(outdata)
        
        if varname:
            if delete:
                delattr(self.grid, varname)
            setattr(self.HRU, varname, outdata)
        return outdata

    @staticmethod
    @njit(cache=True)
    def to_grid_numba(data, grid_to_HRU, land_use_ratio, fn='mean'):
        """Numba helper function to convert from HRU to grid.
        
        Args:
            data: The grid data to be converted.
            grid_to_HRU: Array of size of the compressed grid cells. Each value maps to the index of the first unit of the next cell.
            land_use_ratio: Relative size of HRU to grid.
            fn: Name of function to apply to data. In most cases, several HRUs are combined into one grid unit, so a function must be applied. Choose from `mean`, `sum`, `nansum`, `max` and `min`.

        Returns:
            ouput_data: Data converted to HRUs.
        """
        output_data = np.empty(grid_to_HRU.size, dtype=data.dtype)
        assert grid_to_HRU[-1] == land_use_ratio.size

        prev_index = 0
        for i in range(grid_to_HRU.size):
            cell_index = grid_to_HRU[i]
            if fn == 'mean':
                values = data[prev_index:cell_index]
                weights = land_use_ratio[prev_index:cell_index]
                output_data[i] = (values * weights).sum() / weights.sum()
            elif fn == 'sum':
                output_data[i] = np.sum(data[prev_index:cell_index])
            elif fn == 'nansum':
                output_data[i] = np.nansum(data[prev_index:cell_index])
            elif fn == 'max':
                output_data[i] = np.max(data[prev_index: cell_index])
            elif fn == 'min':
                output_data[i] = np.min(data[prev_index: cell_index])
            else:
                raise NotImplementedError
            prev_index = cell_index
        return output_data

    def to_grid(self, *, HRU_data=None, varname=None, fn=None, delete=True):
        """Function to convert from HRUs to grid.
        
        Args:
            HRU_data: The HRU data to be converted (if set, varname cannot be set).
            varname: Name of variable to be converted. Must be present in HRU class. (if set, data cannot be set).
            fn: Name of function to apply to data. In most cases, several HRUs are combined into one grid unit, so a function must be applied. Choose from `mean`, `sum`, `nansum`, `max` and `min`.
            delete: Whether to delete the data from the grid class. Can only be set if varname is given.

        Returns:
            ouput_data: Data converted to grid units.
        """
        assert bool(HRU_data is not None) != bool(varname is not None)
        assert fn is not None
        if varname:
            HRU_data = getattr(self.HRU, varname)
        assert not isinstance(HRU_data, list)
        if isinstance(HRU_data, float):  # check if data is simple float. Otherwise should be numpy array.
            outdata = HRU_data
        else:
            if self.model.args.use_gpu and isinstance(HRU_data, cp.ndarray):
                HRU_data = HRU_data.get()
            outdata = self.to_grid_numba(HRU_data, self.HRU.grid_to_HRU, self.HRU.land_use_ratio, fn)

        if varname:
            if delete:
                delattr(self.HRU, varname)
            setattr(self.var, varname, outdata)
        return outdata

    def split_HRU_data(self, a, i, ratio=None):
        assert ratio is None or (ratio > 0 and ratio < 1)
        assert ratio is None or np.issubdtype(a.dtype, np.floating)
        if self.model.args.use_gpu and isinstance(a, cp.ndarray):
            is_cupy = True
            a = a.get()
        else:
            is_cupy = False
        if a.ndim == 1:
            a = np.insert(a, i, a[i] * (ratio or 1), axis=0)
        elif a.ndim == 2:
            a = np.insert(a, i, a[:, i] * (ratio or 1), axis=1)
        else:
            raise NotImplementedError
        if ratio is not None:
            a[i+1] = (1 - ratio) * a[i+1]
        if is_cupy:
            a = cp.array(a)
        return a

    @property
    def grid_to_HRU_uncompressed(self):
        return self.grid.decompress(self.HRU.grid_to_HRU, fillvalue=-1).ravel()

    def split(self, HRU_indices):
        HRU = self.HRU.unmerged_HRU_indices[HRU_indices]
        assert (HRU == HRU[0]).all()  # assert all indices belong to same HRU - so only works for single grid cell at this moment
        HRU = HRU[0]
        assert HRU != -1

        # outarray = np.zeros_like(self.HRU.mask, dtype=np.int32)
        # i = 0
        # for y in range(self.grid.mask.shape[0]):
        #     for x in range(self.grid.mask.shape[1]):
        #         is_masked = self.grid.mask[y, x]
        #         if not is_masked:
        #             for ys in range(self.HRU.scaling):
        #                 for xs in range(self.HRU.scaling):
        #                     # outarray[y * self.HRU.scaling + ys, x * self.HRU.scaling + xs] = HRU_array[unmerged_HRU_indices[i]]
        #                     outarray[y * self.HRU.scaling + ys, x * self.HRU.scaling + xs] = 1
        #                     if i in HRU_indices:
        #                         outarray[y * self.HRU.scaling + ys, x * self.HRU.scaling + xs] = 2
        #                     i += 1

        all_HRU_indices = np.where(self.HRU.unmerged_HRU_indices == HRU)  # this could probably be speed up
        assert all_HRU_indices[0].size > HRU_indices[0].size  # ensure that not all indices are split off
        ratio = HRU_indices[0].size / all_HRU_indices[0].size

        self.HRU.unmerged_HRU_indices[self.HRU.unmerged_HRU_indices > HRU] += 1
        self.HRU.unmerged_HRU_indices[HRU_indices] += 1

        # self.grid_to_HRU_uncompressed,

        # plt.imshow(outarray[:100, 150:400])
        # plt.savefig('test.png')
        # exit()
        # # plt.show()

        self.HRU.HRU_to_grid = self.split_HRU_data(self.HRU.HRU_to_grid, HRU)
        self.HRU.grid_to_HRU[self.HRU.HRU_to_grid[HRU]:] += 1
        
        self.HRU.land_owners = self.split_HRU_data(self.HRU.land_owners, HRU)
        self.model.agents.farmers.update_field_indices()

        self.model.agents.farmers.field_indices = self.split_HRU_data(self.model.agents.farmers.field_indices, HRU)
        
        self.HRU.land_use_type = self.split_HRU_data(self.HRU.land_use_type, HRU)
        self.HRU.land_use_ratio = self.split_HRU_data(self.HRU.land_use_ratio, HRU, ratio=ratio)
        self.HRU.cellArea = self.split_HRU_data(self.HRU.cellArea, HRU, ratio=ratio)
        self.HRU.crop_map = self.split_HRU_data(self.HRU.crop_map, HRU)
        self.HRU.crop_age_days_map = self.split_HRU_data(self.HRU.crop_age_days_map, HRU)
        self.HRU.Precipitation = self.split_HRU_data(self.HRU.Precipitation, HRU)
        self.HRU.SnowCoverS = self.split_HRU_data(self.HRU.SnowCoverS, HRU)
        self.HRU.DeltaTSnow = self.split_HRU_data(self.HRU.DeltaTSnow, HRU)
        self.HRU.FrostIndex = self.split_HRU_data(self.HRU.FrostIndex, HRU)
        self.HRU.percolationImp = self.split_HRU_data(self.HRU.percolationImp, HRU)
        self.HRU.cropGroupNumber = self.split_HRU_data(self.HRU.cropGroupNumber, HRU)
        self.HRU.capriseindex = self.split_HRU_data(self.HRU.capriseindex, HRU)
        self.HRU.actBareSoilEvap = self.split_HRU_data(self.HRU.actBareSoilEvap, HRU)
        self.HRU.actTransTotal = self.split_HRU_data(self.HRU.actTransTotal, HRU)
        self.HRU.KSat1 = self.split_HRU_data(self.HRU.KSat1, HRU)
        self.HRU.KSat2 = self.split_HRU_data(self.HRU.KSat2, HRU)
        self.HRU.KSat3 = self.split_HRU_data(self.HRU.KSat3, HRU)
        self.HRU.lambda1 = self.split_HRU_data(self.HRU.lambda1, HRU)
        self.HRU.lambda2 = self.split_HRU_data(self.HRU.lambda2, HRU)
        self.HRU.lambda3 = self.split_HRU_data(self.HRU.lambda3, HRU)
        self.HRU.wwp1 = self.split_HRU_data(self.HRU.wwp1, HRU)
        self.HRU.wwp2 = self.split_HRU_data(self.HRU.wwp2, HRU)
        self.HRU.wwp3 = self.split_HRU_data(self.HRU.wwp3, HRU)
        self.HRU.ws1 = self.split_HRU_data(self.HRU.ws1, HRU)
        self.HRU.ws2 = self.split_HRU_data(self.HRU.ws2, HRU)
        self.HRU.ws3 = self.split_HRU_data(self.HRU.ws3, HRU)
        self.HRU.wres1 = self.split_HRU_data(self.HRU.wres1, HRU)
        self.HRU.wres2 = self.split_HRU_data(self.HRU.wres2, HRU)
        self.HRU.wres3 = self.split_HRU_data(self.HRU.wres3, HRU)
        self.HRU.wfc1 = self.split_HRU_data(self.HRU.wfc1, HRU)
        self.HRU.wfc2 = self.split_HRU_data(self.HRU.wfc2, HRU)
        self.HRU.wfc3 = self.split_HRU_data(self.HRU.wfc3, HRU)
        self.HRU.kunSatFC12 = self.split_HRU_data(self.HRU.kunSatFC12, HRU)
        self.HRU.kunSatFC23 = self.split_HRU_data(self.HRU.kunSatFC23, HRU)
        self.HRU.adjRoot = self.split_HRU_data(self.HRU.adjRoot, HRU)
        self.HRU.arnoBeta = self.split_HRU_data(self.HRU.arnoBeta, HRU)
        self.HRU.w1 = self.split_HRU_data(self.HRU.w1, HRU)
        self.HRU.w2 = self.split_HRU_data(self.HRU.w2, HRU)
        self.HRU.w3 = self.split_HRU_data(self.HRU.w3, HRU)
        self.HRU.topwater = self.split_HRU_data(self.HRU.topwater, HRU)
        self.HRU.totAvlWater = self.split_HRU_data(self.HRU.totAvlWater, HRU)
        self.HRU.minInterceptCap = self.split_HRU_data(self.HRU.minInterceptCap, HRU)
        self.HRU.interceptStor = self.split_HRU_data(self.HRU.interceptStor, HRU)
        self.HRU.potential_transpiration_crop = self.split_HRU_data(self.HRU.potential_transpiration_crop, HRU)
        self.HRU.actual_transpiration_crop = self.split_HRU_data(self.HRU.actual_transpiration_crop, HRU)
        return HRU