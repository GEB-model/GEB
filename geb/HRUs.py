# -*- coding: utf-8 -*-
from typing import Union
from numba import njit
from pathlib import Path
import rasterio
import warnings
import os
import math
from affine import Affine
from operator import attrgetter
import xarray as xr
import zarr
import numpy as np
import zarr.convenience
from geb.workflows import AsyncForcingReader

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass


def load_grid(filepath, layer=1, return_transform_and_crs=False):
    if filepath.suffix == ".tif":
        warnings.warn("tif files are now deprecated. Consider rebuilding the model.")
        with rasterio.open(filepath) as src:
            data = src.read(layer)
            data = data.astype(np.float32) if data.dtype == np.float64 else data
            if return_transform_and_crs:
                return data, src.transform, src.crs
            else:
                return data
    elif filepath.suffixes == [".zarr", ".zip"]:
        ds = zarr.convenience.open_group(filepath)
        data = ds["data"][:]
        if return_transform_and_crs:
            x = ds.x[:]
            y = ds.y[:]
            x_diff = np.diff(x[:]).mean()
            y_diff = np.diff(y[:]).mean()
            transform = Affine(
                a=x_diff,
                b=0,
                c=x[:][0] - x_diff / 2,
                d=0,
                e=y_diff,
                f=ds.y[:][0] - y_diff / 2,
            )
            wkt = ds.spatial_ref.attrs["spatial_ref"]
            return data, transform, wkt
        else:
            return data
    else:
        raise ValueError("File format not supported.")


@njit(cache=True)
def to_grid(data, grid_to_HRU, land_use_ratio, fn="weightedmean"):
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

    assert (
        grid_to_HRU[0] != 0
    ), "First value of grid_to_HRU cannot be 0. This would mean that the first HRU is empty."
    assert (
        grid_to_HRU[-1] == land_use_ratio.size
    ), "The last value of grid_to_HRU must be equal to the size of land_use_ratio. Otherwise, the last HRU would not be used."

    prev_index = 0
    for i in range(grid_to_HRU.size):
        cell_index = grid_to_HRU[i]
        if fn == "weightedmean":
            values = data[prev_index:cell_index]
            weights = land_use_ratio[prev_index:cell_index]
            output_data[i] = (values * weights).sum() / weights.sum()
        elif fn == "weightednanmean":
            values = data[prev_index:cell_index]
            weights = land_use_ratio[prev_index:cell_index]
            weights = weights[~np.isnan(values)]
            values = values[~np.isnan(values)]
            if values.size == 0:
                output_data[i] = np.nan
            else:
                output_data[i] = (values * weights).sum() / weights.sum()
        elif fn == "sum":
            output_data[i] = np.sum(data[prev_index:cell_index])
        elif fn == "nansum":
            output_data[i] = np.nansum(data[prev_index:cell_index])
        elif fn == "max":
            output_data[i] = np.max(data[prev_index:cell_index])
        elif fn == "min":
            output_data[i] = np.min(data[prev_index:cell_index])
        else:
            raise NotImplementedError
        prev_index = cell_index
    return output_data


@njit(cache=True)
def to_HRU(data, grid_to_HRU, land_use_ratio, output_data, fn=None):
    """Numba helper function to convert from grid to HRU.

    Args:
        data: The grid data to be converted.
        grid_to_HRU: Array of size of the compressed grid cells. Each value maps to the index of the first unit of the next cell.
        land_use_ratio: Relative size of HRU to grid.
        fn: Name of function to apply to data. None if data should be directly inserted into HRUs - generally used when units are irrespective of area. 'mean' if data should first be corrected relative to the land use ratios - generally used when units are relative to area.

    Returns:
        ouput_data: Data converted to HRUs.
    """
    assert grid_to_HRU[0] != 0
    assert grid_to_HRU[-1] == land_use_ratio.size
    assert data.shape == grid_to_HRU.shape
    prev_index = 0

    if fn is None:
        for i in range(grid_to_HRU.size):
            cell_index = grid_to_HRU[i]
            output_data[prev_index:cell_index] = data[i]
            prev_index = cell_index
    elif fn == "weightedsplit":
        for i in range(grid_to_HRU.size):
            cell_index = grid_to_HRU[i]
            cell_sizes = land_use_ratio[prev_index:cell_index]
            output_data[prev_index:cell_index] = data[i] / cell_sizes.sum() * cell_sizes
            prev_index = cell_index
    else:
        raise NotImplementedError

    return output_data


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

    def register_initial_data(self, name: str) -> None:
        self.data.initial_conditions.append(name)

    def load_initial(self, name, default=0.0, gpu=False):
        if self.model.load_initial_data:
            fp = os.path.join(self.data.get_save_state_path(), f"{name}.npz")
            if gpu:
                return cp.load(fp)["data"]
            else:
                return np.load(fp)["data"]
        else:
            self.register_initial_data(name)
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
        mask, transform, self.crs = load_grid(
            self.model.files["grid"]["areamaps/grid_mask"],
            return_transform_and_crs=True,
        )
        self.mask = mask.astype(bool)
        self.gt = transform.to_gdal()
        self.bounds = (
            transform.c,
            transform.f + transform.e * mask.shape[0],
            transform.c + transform.a * mask.shape[1],
            transform.f,
        )
        self.lon = np.linspace(
            transform.c + transform.a / 2,
            transform.c + transform.a * mask.shape[1] - transform.a / 2,
            mask.shape[1],
        )
        self.lat = np.linspace(
            transform.f + transform.e / 2,
            transform.f + transform.e * mask.shape[0] - transform.e / 2,
            mask.shape[0],
        )

        assert math.isclose(transform.a, -transform.e)
        self.cell_size = transform.a

        self.cell_area_uncompressed = load_grid(
            self.model.files["grid"]["areamaps/cell_area"]
        )

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
        return array[..., ~self.mask]

    def decompress(
        self, array: np.ndarray, fillvalue: Union[np.ufunc, int, float] = None
    ) -> np.ndarray:
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
        output_shape = self.mask.shape
        if array.ndim == 2:
            assert array.shape[1] == self.mask_flat.size - self.mask_flat.sum()
            outmap = np.broadcast_to(outmap, (array.shape[0], outmap.size)).copy()
            output_shape = (array.shape[0], *output_shape)
        outmap[..., ~self.mask_flat] = array
        return outmap.reshape(output_shape)

    def plot(self, array: np.ndarray) -> None:
        """Plot array.

        Args:
            array: Array to plot.
        """
        import matplotlib.pyplot as plt

        plt.imshow(array)
        plt.show()

    def plot_compressed(
        self, array: np.ndarray, fillvalue: Union[np.ufunc, int, float] = None
    ):
        """Plot compressed array.

        Args:
            array: Compressed array to plot.
            fillvalue: Value to use for masked values.
        """
        self.plot(self.decompress(array, fillvalue=fillvalue))

    def load(self, filepath, compress=True, layer=1):
        """Load array from disk.

        Args:
            filepath: Filepath of map.
            compress: Whether to compress array.

        Returns:
            array: Loaded array.
        """
        data = load_grid(filepath, layer=layer)
        if compress:
            data = self.data.grid.compress(data)
        return data

    def load_initial(self, name, default=0.0):
        return super().load_initial("grid." + name, default=default)

    def load_forcing_ds(self, name):
        reader = AsyncForcingReader(
            self.model.files["forcing"][f"climate/{name}"],
            name,
        )
        assert reader.ds["y"][0] > reader.ds["y"][-1]
        return reader

    def load_forcing(self, reader, time, compress=True):
        data = reader.read_timestep(time)
        if compress:
            data = self.compress(data)
        return data

    @property
    def hurs(self):
        if not hasattr(self, "hurs_ds"):
            self.hurs_ds = self.load_forcing_ds("hurs")
        hurs = self.load_forcing(self.hurs_ds, self.model.current_time)
        assert (hurs > 1).all() and (hurs <= 100).all(), "hurs out of range"
        return hurs

    @property
    def pr(self):
        if not hasattr(self, "pr_ds"):
            self.pr_ds = self.load_forcing_ds("pr")
        pr = self.load_forcing(self.pr_ds, self.model.current_time)
        assert (pr >= 0).all(), "Precipitation must be positive or zero"
        return pr

    @property
    def ps(self):
        if not hasattr(self, "ps_ds"):
            self.ps_ds = self.load_forcing_ds("ps")
        ps = self.load_forcing(self.ps_ds, self.model.current_time)
        assert (
            (ps > 30_000).all() and (ps < 120_000).all()
        ), "ps out of range"  # top of mount everest is 33700 Pa, highest pressure ever measures is 108180 Pa
        return ps

    @property
    def rlds(self):
        if not hasattr(self, "rlds_ds"):
            self.rlds_ds = self.load_forcing_ds("rlds")
        rlds = self.load_forcing(self.rlds_ds, self.model.current_time)
        rlds = rlds.astype(np.float32)
        assert (rlds >= 0).all(), "rlds must be positive or zero"
        return rlds

    @property
    def rsds(self):
        if not hasattr(self, "rsds_ds"):
            self.rsds_ds = self.load_forcing_ds("rsds")
        rsds = self.load_forcing(self.rsds_ds, self.model.current_time)
        assert (rsds >= 0).all(), "rsds must be positive or zero"
        return rsds

    @property
    def tas(self):
        if not hasattr(self, "tas_ds"):
            self.tas_ds = self.load_forcing_ds("tas")
        tas = self.load_forcing(self.tas_ds, self.model.current_time)
        assert (tas > 170).all() and (tas < 370).all(), "tas out of range"
        return tas

    @property
    def tasmin(self):
        if not hasattr(self, "tasmin_ds"):
            self.tasmin_ds = self.load_forcing_ds("tasmin")
        tasmin = self.load_forcing(self.tasmin_ds, self.model.current_time)
        assert (tasmin > 170).all() and (tasmin < 370).all(), "tasmin out of range"
        return tasmin

    @property
    def tasmax(self):
        if not hasattr(self, "tasmax_ds"):
            self.tasmax_ds = self.load_forcing_ds("tasmax")
        tasmax = self.load_forcing(self.tasmax_ds, self.model.current_time)
        assert (tasmax > 170).all() and (tasmax < 370).all(), "tasmax out of range"
        return tasmax

    @property
    def sfcWind(self):
        if not hasattr(self, "sfcWind_ds"):
            self.sfcWind_ds = self.load_forcing_ds("sfcwind")
        sfcWind = self.load_forcing(self.sfcWind_ds, self.model.current_time)
        assert (
            (sfcWind >= 0).all() and (sfcWind < 150).all()
        ), "sfcWind must be positive or zero. Highest wind speed ever measured is 113 m/s."
        return sfcWind

    @property
    def spei_uncompressed(self):
        if not hasattr(self, "spei_ds"):
            self.spei_ds = self.load_forcing_ds("spei")

        current_time = self.model.current_time

        # Determine the nearest first day of the month
        if current_time.day <= 15:
            spei_time = current_time.replace(day=1)
        else:
            # Move to the first day of the next month
            if current_time.month == 12:
                spei_time = current_time.replace(
                    year=current_time.year + 1, month=1, day=1
                )
            else:
                spei_time = current_time.replace(month=current_time.month + 1, day=1)

        spei = self.load_forcing(self.spei_ds, spei_time, compress=False)
        assert not np.isnan(
            spei[~self.mask]
        ).any()  # Ensure no NaN values in non-masked cells
        return spei

    @property
    def spei(self):
        if not hasattr(self, "spei_ds"):
            self.spei_ds = self.load_forcing_ds("spei")
        spei = self.load_forcing(self.spei_ds, self.model.current_time)
        assert not np.isnan(spei).any()
        return spei

    @property
    def gev_c(self):
        return load_grid(self.model.files["grid"]["climate/gev_c"])

    @property
    def gev_loc(self):
        return load_grid(self.model.files["grid"]["climate/gev_loc"])

    @property
    def gev_scale(self):
        return load_grid(self.model.files["grid"]["climate/gev_scale"])


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

        subgrid_mask = load_grid(self.model.files["subgrid"]["areamaps/sub_grid_mask"])
        submask_height, submask_width = subgrid_mask.shape

        self.scaling = submask_height // self.data.grid.shape[0]
        assert submask_width // self.data.grid.shape[1] == self.scaling
        self.gt = (
            self.data.grid.gt[0],
            self.data.grid.gt[1] / self.scaling,
            self.data.grid.gt[2],
            self.data.grid.gt[3],
            self.data.grid.gt[4],
            self.data.grid.gt[5] / self.scaling,
        )

        self.mask = self.data.grid.mask.repeat(self.scaling, axis=0).repeat(
            self.scaling, axis=1
        )
        self.cell_size = self.data.grid.cell_size / self.scaling
        if self.model.load_initial_data:
            self.land_use_type = np.load(
                os.path.join(self.data.get_save_state_path(), "HRU.land_use_type.npz")
            )["data"]
            self.land_use_ratio = np.load(
                os.path.join(self.data.get_save_state_path(), "HRU.land_use_ratio.npz")
            )["data"]
            self.land_owners = np.load(
                os.path.join(self.data.get_save_state_path(), "HRU.land_owners.npz")
            )["data"]
            self.HRU_to_grid = np.load(
                os.path.join(self.data.get_save_state_path(), "HRU.HRU_to_grid.npz")
            )["data"]
            self.grid_to_HRU = np.load(
                os.path.join(self.data.get_save_state_path(), "HRU.grid_to_HRU.npz")
            )["data"]
            self.unmerged_HRU_indices = np.load(
                os.path.join(
                    self.data.get_save_state_path(), "HRU.unmerged_HRU_indices.npz"
                )
            )["data"]
        else:
            (
                self.land_use_type,
                self.land_use_ratio,
                self.land_owners,
                self.HRU_to_grid,
                self.grid_to_HRU,
                self.unmerged_HRU_indices,
            ) = self.create_HRUs()
            self.register_initial_data("HRU.land_use_type")
            self.register_initial_data("HRU.land_use_ratio")
            self.register_initial_data("HRU.land_owners")
            self.register_initial_data("HRU.HRU_to_grid")
            self.register_initial_data("HRU.grid_to_HRU")
            self.register_initial_data("HRU.unmerged_HRU_indices")
        if self.model.use_gpu:
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
    def create_HRUs_numba(
        farms, land_use_classes, mask, scaling
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
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
            var_to_HRU: Array of size of the compressed grid cells. Each value maps to the index of the first unit of the next cell.
            # var_to_HRU_uncompressed: Array of size of the grid cells. Each value maps to the index of the first unit of the next cell.
            unmerged_HRU_indices: The index of the HRU to the subcell.
        """
        assert farms.size == mask.size * scaling * scaling
        assert farms.size == land_use_classes.size
        ysize, xsize = mask.shape

        n_nonmasked_cells = mask.size - mask.sum()
        grid_to_HRU = np.full(n_nonmasked_cells, -1, dtype=np.int32)
        # var_to_HRU_uncompressed = np.full(mask.size, -1, dtype=np.int32)
        HRU_to_grid = np.full(farms.size, -1, dtype=np.int32)
        land_use_array = np.full(farms.size, -1, dtype=np.int32)
        land_use_size = np.full(farms.size, -1, dtype=np.int32)
        land_use_owner = np.full(farms.size, -1, dtype=np.int32)
        unmerged_HRU_indices = np.full(farms.shape, -1, dtype=np.int32)

        HRU = 0
        var_cell_count_compressed = 0
        var_cell_count_uncompressed = 0

        for y in range(0, ysize):
            for x in range(0, xsize):
                is_masked = mask[y, x]
                if not is_masked:
                    cell_farms = farms[
                        y * scaling : (y + 1) * scaling, x * scaling : (x + 1) * scaling
                    ].ravel()  # find farms in cell
                    cell_land_use_classes = land_use_classes[
                        y * scaling : (y + 1) * scaling, x * scaling : (x + 1) * scaling
                    ].ravel()  # get land use classes for cells
                    assert (
                        (cell_land_use_classes == 0)
                        | (cell_land_use_classes == 1)
                        | (cell_land_use_classes == 4)
                        | (cell_land_use_classes == 5)
                    ).all()

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
                            assert land_use == 1  # must be one because farm
                            land_use_array[HRU] = land_use
                            assert land_use_size[HRU] == -1
                            land_use_size[HRU] = 1
                            land_use_owner[HRU] = farm

                            HRU_to_grid[HRU] = var_cell_count_compressed

                            prev_farm = farm
                            HRU += 1
                        else:
                            land_use_size[HRU - 1] += 1

                        unmerged_HRU_indices[
                            y * scaling + sort_idx[i] // scaling,
                            x * scaling + sort_idx[i] % scaling,
                        ] = HRU - 1

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
                            land_use_size[HRU - 1] += 1

                        unmerged_HRU_indices[
                            y * scaling + sort_idx[i] // scaling,
                            x * scaling + sort_idx[i] % scaling,
                        ] = HRU - 1

                    grid_to_HRU[var_cell_count_compressed] = HRU
                    var_cell_count_compressed += 1
                var_cell_count_uncompressed += 1

        land_use_size = land_use_size[:HRU]
        land_use_array = land_use_array[:HRU]
        land_use_owner = land_use_owner[:HRU]
        HRU_to_grid = HRU_to_grid[:HRU]
        assert int(land_use_size.sum()) == n_nonmasked_cells * scaling * scaling

        land_use_ratio = land_use_size / (scaling**2)
        return (
            land_use_array,
            land_use_ratio,
            land_use_owner,
            HRU_to_grid,
            grid_to_HRU,
            unmerged_HRU_indices,
        )

    def create_HRUs(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Function to create HRUs.

        Returns:
            land_use_array: Land use of each HRU.
            land_use_ratio: Relative size of HRU to grid.
            land_use_owner: Owner of HRU.
            HRU_to_grid: Maps HRUs to index of compressed cell index.
            grid_to_HRU: Array of size of the compressed grid cells. Each value maps to the index of the first unit of the next cell.
            unmerged_HRU_indices: The index of the HRU to the subcell.
        """
        land_use_classes = load_grid(
            self.model.files["subgrid"]["landsurface/land_use_classes"]
        )
        return self.create_HRUs_numba(
            self.data.farms, land_use_classes, self.data.grid.mask, self.scaling
        )

    def zeros(self, size, dtype, *args, **kwargs) -> np.ndarray:
        """Return an array (CuPy or Numpy) of zeros with given size. Takes any other argument normally used in np.zeros.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            array: Array with size of number of HRUs.
        """
        if self.model.use_gpu:
            return cp.zeros(size, dtype, *args, **kwargs)
        else:
            return np.zeros(size, dtype, *args, **kwargs)

    def full_compressed(
        self, fill_value, dtype, gpu=None, *args, **kwargs
    ) -> np.ndarray:
        """Return a full array with size of number of HRUs. Takes any other argument normally used in np.full.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            array: Array with size of number of HRUs.
        """
        if gpu is None:
            gpu = self.model.use_gpu
        if gpu:
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
        if self.model.use_gpu:
            HRU_array = HRU_array.get()
        if np.issubdtype(HRU_array.dtype, np.integer):
            nanvalue = -1
        else:
            nanvalue = np.nan
        outarray = HRU_array[self.unmerged_HRU_indices]
        outarray[self.mask] = nanvalue
        return outarray

    @staticmethod
    @njit(cache=True)
    def compress_numba(array, unmerged_HRU_indices, outarray, nodatavalue):
        array = array.ravel()
        unmerged_HRU_indices = unmerged_HRU_indices.ravel()
        for i in range(array.size):
            value = array[i]
            if value != nodatavalue:
                HRU = unmerged_HRU_indices[i]
                outarray[HRU] = value
        return outarray

    def compress(self, array: np.ndarray, method="last") -> np.ndarray:
        assert method == "last", "Only last method is implemented"
        assert self.mask.shape == array.shape, "Array must have same shape as mask"
        if np.issubdtype(array.dtype, np.integer):
            fill_value = -1
        else:
            fill_value = np.nan
        outarray = self.full_compressed(fill_value, array.dtype)
        outarray = self.compress_numba(
            array, self.unmerged_HRU_indices, outarray, nodatavalue=fill_value
        )
        return outarray

    def plot(self, HRU_array: np.ndarray, ax=None, show: bool = True):
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

    def load_initial(self, name, default=0.0, gpu=None):
        if gpu is None:
            gpu = self.model.use_gpu
        return super().load_initial("HRU." + name, default=default, gpu=gpu)

    @property
    def hurs(self):
        hurs = self.data.grid.hurs
        return self.data.to_HRU(data=hurs, fn=None)

    @property
    def pr(self):
        pr = self.data.grid.pr
        return self.data.to_HRU(data=pr, fn=None)

    @property
    def ps(self):
        ps = self.data.grid.ps
        return self.data.to_HRU(data=ps, fn=None)

    @property
    def rlds(self):
        rlds = self.data.grid.rlds
        return self.data.to_HRU(data=rlds, fn=None)

    @property
    def rsds(self):
        rsds = self.data.grid.rsds
        return self.data.to_HRU(data=rsds, fn=None)

    @property
    def tas(self):
        tas = self.data.grid.tas
        return self.data.to_HRU(data=tas, fn=None)

    @property
    def tasmin(self):
        tasmin = self.data.grid.tasmin
        return self.data.to_HRU(data=tasmin, fn=None)

    @property
    def tasmax(self):
        tasmax = self.data.grid.tasmax
        return self.data.to_HRU(data=tasmax, fn=None)

    @property
    def sfcWind(self):
        sfcWind = self.data.grid.sfcWind
        return self.data.to_HRU(data=sfcWind, fn=None)


class Modflow(BaseVariables):
    def __init__(self, data, model):
        self.data = data
        self.model = model

        BaseVariables.__init__(self)

    def load_initial(self, name, default=0.0):
        return super().load_initial("modflow." + name, default=default)


class Data:
    """The base data class for the GEB model. This class contains the data for the normal grid, the HRUs, and has methods to convert between the grid and HRUs.

    Args:
        model: The GEB model.
    """

    def __init__(self, model):
        self.model = model

        self.farms = load_grid(self.model.files["subgrid"]["agents/farmers/farms"])

        self.initial_conditions = []

        self.grid = Grid(self, model)
        self.HRU = HRUs(self, model)
        self.HRU.cellArea = self.to_HRU(data=self.grid.cellArea, fn="weightedsplit")
        self.modflow = Modflow(self, model)

        if self.model.config["general"]["simulate_hydrology"]:
            self.load_water_demand()

    def load_water_demand(self):
        self.model.domestic_water_consumption_ds = xr.open_dataset(
            self.model.files["forcing"]["water_demand/domestic_water_consumption"],
            engine="zarr",
        )
        self.model.domestic_water_demand_ds = xr.open_dataset(
            self.model.files["forcing"]["water_demand/domestic_water_demand"],
            engine="zarr",
        )
        self.model.industry_water_consumption_ds = xr.open_dataset(
            self.model.files["forcing"]["water_demand/industry_water_consumption"],
            engine="zarr",
        )
        self.model.industry_water_demand_ds = xr.open_dataset(
            self.model.files["forcing"]["water_demand/industry_water_demand"],
            engine="zarr",
        )
        self.model.livestock_water_consumption_ds = xr.open_dataset(
            self.model.files["forcing"]["water_demand/livestock_water_consumption"],
            engine="zarr",
        )

    def to_HRU(self, *, data=None, fn=None):
        """Function to convert from grid to HRU (Hydrologic Response Units).

        This method is designed to transform spatial grid data into a format suitable for HRUs, which are used in to represent distinct areas with homogeneous land use, soil type, and management conditions.

        Args:
            data (array-like or None): The grid data to be converted. If this parameter is set, `varname` must not be provided. Data should be an array where each element corresponds to grid cell values.
            fn (str or None): The name of the function to apply to the data before assigning it to HRUs. If `None`, the data is used as is. This is usually the case for variables that are independent of area, like temperature or precipitation fluxes. If 'weightedsplit', the data will be adjusted according to the ratios of land use within each HRU. This is important when dealing with variables that are area-dependent like precipitation or runoff volumes.

        Returns:
            output_data (array-like): Data converted to HRUs format. The structure and the type of the output depend on the input and the transformation function.

        Example:
            Suppose we have an instance of a class with a grid property containing temperature data under the attribute name 'temperature'. To convert this grid-based temperature data into HRU format, we would use:

            ```python
            temperature_HRU = instance.to_HRU(data=temperature, fn=None)
            ```

            This will fetch the temperature data from `instance.grid.temperature`, assigning the temperature to HRU within a grid cell. In other words, each HRU within a grid cell has the same temperature.

            Another example, where want to plant forest in all HRUs with grassland within an area specified by a boolean mask.

            ```python
            mask_HRU = instance.to_HRU(data=mask_grid, fn=None)
            mask_HRU[land_use_type == grass_land_use_type] = False  # set all non-grassland HRUs to False
            ```
        """
        assert not isinstance(data, list)
        # make data same size as grid, but with last dimension being size of HRU
        output_data = np.zeros(
            (*data.shape[:-1], self.HRU.land_use_ratio.size), dtype=data.dtype
        )

        if data.ndim == 1:
            to_HRU(
                data,
                self.HRU.grid_to_HRU,
                self.HRU.land_use_ratio,
                output_data=output_data,
                fn=fn,
            )
        elif data.ndim == 2:
            for i in range(data.shape[0]):
                to_HRU(
                    data[i],
                    self.HRU.grid_to_HRU,
                    self.HRU.land_use_ratio,
                    output_data=output_data[i],
                    fn=fn,
                )
        else:
            raise NotImplementedError
        if self.model.use_gpu:
            output_data = cp.asarray(output_data)

        return output_data

    def to_grid(self, *, HRU_data=None, fn=None):
        """Function to convert from HRUs to grid.

        Args:
            HRU_data: The HRU data to be converted (if set, varname cannot be set).
            fn: Name of function to apply to data. In most cases, several HRUs are combined into one grid unit, so a function must be applied. Choose from `mean`, `sum`, `nansum`, `max` and `min`.

        Returns:
            ouput_data: Data converted to grid units.
        """
        assert fn is not None
        assert not isinstance(HRU_data, list)
        if isinstance(
            HRU_data, float
        ):  # check if data is simple float. Otherwise should be numpy array.
            outdata = HRU_data
        else:
            if self.model.use_gpu and isinstance(HRU_data, cp.ndarray):
                HRU_data = HRU_data.get()
            outdata = to_grid(
                HRU_data, self.HRU.grid_to_HRU, self.HRU.land_use_ratio, fn
            )

        return outdata

    def split_HRU_data(self, a, i, ratio=None):
        assert ratio is None or (ratio > 0 and ratio < 1)
        assert ratio is None or np.issubdtype(a.dtype, np.floating)
        if self.model.use_gpu and isinstance(a, cp.ndarray):
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
            a[i + 1] = (1 - ratio) * a[i + 1]
        if is_cupy:
            a = cp.array(a)
        return a

    @property
    def grid_to_HRU_uncompressed(self):
        return self.grid.decompress(self.HRU.grid_to_HRU, fillvalue=-1).ravel()

    def split(self, HRU_indices):
        HRU = self.HRU.unmerged_HRU_indices[HRU_indices]
        assert (
            HRU == HRU[0]
        ).all()  # assert all indices belong to same HRU - so only works for single grid cell at this moment
        HRU = HRU[0]
        assert HRU != -1

        all_HRU_indices = np.where(
            self.HRU.unmerged_HRU_indices == HRU
        )  # this could probably be speed up
        assert (
            all_HRU_indices[0].size > HRU_indices[0].size
        )  # ensure that not all indices are split off
        ratio = HRU_indices[0].size / all_HRU_indices[0].size

        self.HRU.unmerged_HRU_indices[self.HRU.unmerged_HRU_indices > HRU] += 1
        self.HRU.unmerged_HRU_indices[HRU_indices] += 1

        self.HRU.HRU_to_grid = self.split_HRU_data(self.HRU.HRU_to_grid, HRU)
        self.HRU.grid_to_HRU[self.HRU.HRU_to_grid[HRU] :] += 1

        self.HRU.land_owners = self.split_HRU_data(self.HRU.land_owners, HRU)
        self.model.agents.farmers.update_field_indices()

        self.model.agents.farmers.field_indices = self.split_HRU_data(
            self.model.agents.farmers.field_indices, HRU
        )

        self.HRU.land_use_type = self.split_HRU_data(self.HRU.land_use_type, HRU)
        self.HRU.land_use_ratio = self.split_HRU_data(
            self.HRU.land_use_ratio, HRU, ratio=ratio
        )
        self.HRU.cellArea = self.split_HRU_data(self.HRU.cellArea, HRU, ratio=ratio)
        self.HRU.crop_map = self.split_HRU_data(self.HRU.crop_map, HRU)
        self.HRU.crop_age_days_map = self.split_HRU_data(
            self.HRU.crop_age_days_map, HRU
        )
        self.HRU.crop_harvest_age_days = self.split_HRU_data(
            self.HRU.crop_harvest_age_days, HRU
        )
        self.HRU.SnowCoverS = self.split_HRU_data(self.HRU.SnowCoverS, HRU)
        self.HRU.DeltaTSnow = self.split_HRU_data(self.HRU.DeltaTSnow, HRU)
        self.HRU.frost_index = self.split_HRU_data(self.HRU.frost_index, HRU)
        self.HRU.percolationImp = self.split_HRU_data(self.HRU.percolationImp, HRU)
        self.HRU.cropGroupNumber = self.split_HRU_data(self.HRU.cropGroupNumber, HRU)
        self.HRU.capriseindex = self.split_HRU_data(self.HRU.capriseindex, HRU)
        self.HRU.actual_bare_soil_evaporation = self.split_HRU_data(
            self.HRU.actual_bare_soil_evaporation, HRU
        )
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
        self.HRU.arnoBeta = self.split_HRU_data(self.HRU.arnoBeta, HRU)
        self.HRU.w1 = self.split_HRU_data(self.HRU.w1, HRU)
        self.HRU.w2 = self.split_HRU_data(self.HRU.w2, HRU)
        self.HRU.w3 = self.split_HRU_data(self.HRU.w3, HRU)
        self.HRU.topwater = self.split_HRU_data(self.HRU.topwater, HRU)
        self.HRU.totAvlWater = self.split_HRU_data(self.HRU.totAvlWater, HRU)
        self.HRU.minInterceptCap = self.split_HRU_data(self.HRU.minInterceptCap, HRU)
        self.HRU.interceptStor = self.split_HRU_data(self.HRU.interceptStor, HRU)
        self.HRU.potential_evapotranspiration_crop_life = self.split_HRU_data(
            self.HRU.potential_evapotranspiration_crop_life, HRU
        )
        self.HRU.actual_evapotranspiration_crop_life = self.split_HRU_data(
            self.HRU.actual_evapotranspiration_crop_life, HRU
        )
        return HRU

    def step(self):
        pass

    def get_save_state_path(self, mkdir=False):
        folder = Path(self.model.initial_conditions_folder, "grid")
        if mkdir:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_state(self):
        with open(Path(self.get_save_state_path(mkdir=True), "state.txt"), "w") as f:
            for var in self.initial_conditions:
                f.write(f"{var}\n")
                fp = self.get_save_state_path(mkdir=True) / f"{var}.npz"
                values = attrgetter(var)(self)
                np.savez_compressed(fp, data=values)
        return True
