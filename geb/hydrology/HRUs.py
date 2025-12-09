"""This module contains classes and functions to handle Hydrological Response Units (HRUs) and grid cells."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from affine import Affine
from numba import njit
from scipy.spatial import KDTree

from geb.types import (
    AnyDArrayWithScalar,
    Array,
    ArrayFloat32,
    ArrayInt32,
    ArrayWithScalar,
    T_ArrayNumber,
    T_OneorTwoDArray,
    ThreeDArray,
    ThreeDArrayFloat32,
    ThreeDArrayWithScalar,
    TwoDArray,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayInt32,
    TwoDArrayWithScalar,
)
from geb.workflows.io import read_grid, read_zarr
from geb.workflows.raster import compress, decompress_with_mask

if TYPE_CHECKING:
    from geb.model import GEBModel


def determine_nearest_river_cell(
    upstream_area: npt.NDArray[np.float32],
    HRU_to_grid: npt.NDArray[np.int32],
    mask: npt.NDArray[np.bool_],
    threshold_m2: float | int,
) -> npt.NDArray[np.int32]:
    """This function finds the nearest river cell to each HRU.

    It does so by first selecting the rivers, by checking if the upstream area is
    above a certain threshold. then for each grid cell, it finds the nearest
    river cell. Finally, it maps the nearest river cell to each HRU.

    Args:
        upstream_area: 2D-array of upstream area in m².
        HRU_to_grid: Array mapping HRUs to grid cells.
        mask: Mask of the study area.
        threshold_m2: Threshold in m² to consider a cell as a river.

    Returns:
        For each HRU, the index of the nearest river cell in the valid grid cells.
    """
    valid_indices: npt.NDArray[np.int64] = np.argwhere(~mask)
    valid_values: npt.NDArray[np.float32] = upstream_area[~mask]

    grid_cells_above_threshold_mask: npt.NDArray[np.bool_] = valid_values > threshold_m2
    grid_cells_above_threshold_indices: npt.NDArray[np.int64] = valid_indices[
        grid_cells_above_threshold_mask
    ]
    grid_cells_above_threshold_indices_in_valid: npt.NDArray[np.int64] = np.flatnonzero(
        grid_cells_above_threshold_mask
    )

    tree: KDTree = KDTree(grid_cells_above_threshold_indices)
    distances, indices_in_above = tree.query(valid_indices)

    nearest_indices_in_valid: npt.NDArray[np.int32] = (
        grid_cells_above_threshold_indices_in_valid[indices_in_above]
    ).astype(np.int32)

    assert nearest_indices_in_valid.max() < (~mask).sum()

    return nearest_indices_in_valid[HRU_to_grid]


def load_water_demand_xr(filepath: str | Path) -> xr.Dataset:
    """Load a water demand dataset from disk.

    Args:
        filepath: Path to the water demand dataset file.

    Returns:
        An xarray Dataset containing the water demand data.
    """
    return xr.open_dataset(
        filepath,
        engine="zarr",
        consolidated=False,
    )


@njit(cache=True)
def to_grid(
    data: T_ArrayNumber,
    grid_to_HRU: ArrayInt32,
    land_use_ratio: ArrayFloat32,
    fn: str = "weightedmean",
) -> T_ArrayNumber:
    """Numba helper function to convert from HRU to grid.

    Args:
        data: The HRU data to be converted (1D array).
        grid_to_HRU: Array of size of the compressed grid cells. Each value maps to the index of the first unit of the next cell.
        land_use_ratio: Relative size of HRU to grid.
        fn: Name of function to apply to data. In most cases, several HRUs are combined into one grid unit, so a function must be applied. Choose from `mean`, `sum`, `nansum`, `max` and `min`.

    Returns:
        ouput_data: Data converted to grid.
    """
    output_data = np.empty(grid_to_HRU.size, dtype=data.dtype)

    assert grid_to_HRU[0] != 0, (
        "First value of grid_to_HRU cannot be 0. This would mean that the first HRU is empty."
    )
    assert grid_to_HRU[-1] == land_use_ratio.size, (
        "The last value of grid_to_HRU must be equal to the size of land_use_ratio. Otherwise, the last HRU would not be used."
    )

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
        elif fn == "nanmax":
            output_data[i] = np.nanmax(data[prev_index:cell_index])
        elif fn == "min":
            output_data[i] = np.min(data[prev_index:cell_index])
        elif fn == "nanmin":
            output_data[i] = np.nanmin(data[prev_index:cell_index])
        else:
            raise NotImplementedError
        prev_index = cell_index
    return output_data


@njit(cache=True)
def to_HRU(
    data: T_ArrayNumber,
    grid_to_HRU: ArrayInt32,
    land_use_ratio: ArrayFloat32,
    output_data: T_ArrayNumber,
    fn: str | None = None,
) -> T_ArrayNumber:
    """Numba helper function to convert from grid to HRU.

    Args:
        data: The grid data to be converted.
        grid_to_HRU: Array of size of the compressed grid cells. Each value maps to the index of the first unit of the next cell.
        land_use_ratio: Relative size of HRU to grid.
        output_data: Array to store the output data. Must be of size of the HRUs.
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

    def __init__(self) -> None:
        """Initialize BaseVariables class."""
        pass

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the uncompressed variable by returning the shape of the mask.

        Returns:
            shape: Shape of the uncompressed variable.
        """
        return self.mask.shape


class Grid(BaseVariables):
    """This class is to store data in the 'normal' grid cells. This class works with compressed and uncompressed arrays. On initialization of the class, the mask of the study area is read from disk. This is the shape of any uncompressed array. Many values in this array, however, fall outside the stuy area as they are masked. Therefore, the array can be compressed by saving only the non-masked values.

    On initialization, as well as geotransformation and cell size are set, and the cell area is read from disk.

    Then, the mask is compressed by removing all masked cells, resulting in a compressed array.
    """

    def __init__(self, data: Data, model: GEBModel) -> None:
        """Initialize Grid class.

        Args:
            data: Data class for model. Contains the various types of grids used in the GEB Model.
            model: The GEB model.
        """
        self.data = data
        self.model = model
        self.var = self.model.store.create_bucket("hydrology.grid.var")

        self.scaling = 1
        mask, self.transform, self.crs = read_grid(
            self.model.files["grid"]["mask"],
            return_transform_and_crs=True,
        )
        self.mask = mask.astype(bool)
        self.gt = self.transform.to_gdal()
        self.bounds = (
            self.transform.c,
            self.transform.f + self.transform.e * mask.shape[0],
            self.transform.c + self.transform.a * mask.shape[1],
            self.transform.f,
        )
        self.lon = np.linspace(
            self.transform.c + self.transform.a / 2,
            self.transform.c + self.transform.a * mask.shape[1] - self.transform.a / 2,
            mask.shape[1],
        )
        self.lat = np.linspace(
            self.transform.f + self.transform.e / 2,
            self.transform.f + self.transform.e * mask.shape[0] - self.transform.e / 2,
            mask.shape[0],
        )

        assert math.isclose(self.transform.a, -self.transform.e)
        self.cell_size = self.transform.a

        self.cell_area_uncompressed = read_grid(self.model.files["grid"]["cell_area"])

        self.mask_flat = self.mask.ravel()
        self.compressed_size = self.mask_flat.size - self.mask_flat.sum()
        self.var.cell_area = self.compress(self.cell_area_uncompressed)
        self.linear_mapping: TwoDArrayInt32 = self.create_linear_mapping(self.mask)

        BaseVariables.__init__(self)

    def create_linear_mapping(self, mask: TwoDArrayBool) -> TwoDArrayInt32:
        """Create a linear mapping from uncompressed to compressed indices.

        Returns:
            mapping: Linear mapping array.
        """
        mapping = np.full(mask.shape, -1, dtype=np.int32)
        mapping[~mask] = np.arange(self.compressed_size, dtype=np.int32)
        return mapping

    def full(self, *args: Any, **kwargs: Any) -> TwoDArrayFloat32:
        """Return a full array with size of mask. Takes any other argument normally used in np.full.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            array: Full array of mask size.
        """
        return np.full(self.mask.shape, *args, **kwargs)

    def full_compressed(self, *args: Any, **kwargs: Any) -> ArrayFloat32:
        """Return a full array with size of compressed array. Takes any other argument normally used in np.full.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            array: Full array of compressed size.
        """
        return np.full(self.compressed_size, *args, **kwargs)

    @overload
    def compress(self, array: TwoDArrayWithScalar) -> ArrayWithScalar: ...

    @overload
    def compress(self, array: ThreeDArrayWithScalar) -> TwoDArrayWithScalar: ...

    def compress(
        self,
        array: TwoDArrayWithScalar | ThreeDArrayWithScalar,
    ) -> ArrayWithScalar | TwoDArrayWithScalar:
        """Compress array.

        Args:
            array: Uncompressed array.

        Returns:
            array: Compressed array.
        """
        return compress(array, self.mask)

    def decompress(
        self, array: np.ndarray, fillvalue: int | float | None = None
    ) -> np.ndarray:
        """Decompress array.

        Args:
            array: Compressed array.
            fillvalue: Value to use for masked values. If None, uses NaN for float arrays and 0 for int arrays.

        Returns:
            array: Decompressed array.
        """
        return decompress_with_mask(array, self.mask, fillvalue=fillvalue)

    def plot(self, array: np.ndarray) -> None:
        """Plot array.

        Args:
            array: Array to plot.
        """
        import matplotlib.pyplot as plt

        plt.imshow(array)
        plt.show()

    def plot_compressed(
        self, array: np.ndarray, fillvalue: int | float | None = None
    ) -> None:
        """Plot compressed array.

        Args:
            array: Compressed array to plot.
            fillvalue: Value to use for masked values. If None, uses NaN for float arrays and 0 for int arrays.
        """
        self.plot(self.decompress(array, fillvalue=fillvalue))

    @overload
    def load(
        self, filepath: Path, compress: Literal[True] = True, layer: int = 1
    ) -> Array: ...

    @overload
    def load(
        self, filepath: Path, compress: Literal[True] = True, layer: None = None
    ) -> TwoDArray: ...

    @overload
    def load(
        self, filepath: Path, compress: Literal[False] = False, layer: int = 1
    ) -> TwoDArray: ...

    @overload
    def load(
        self, filepath: Path, compress: Literal[False] = False, layer: None = None
    ) -> ThreeDArray: ...

    def load(
        self, filepath: Path, compress: bool = True, layer: int | None = 1
    ) -> Array | TwoDArray | ThreeDArray:
        """Load array from disk.

        Args:
            filepath: Filepath of map.
            compress: Whether to compress array.
            layer: Layer to load from file. Defaults to 1. If None, all layers are loaded.

        Returns:
            array: Loaded array.
        """
        data = read_grid(filepath, layer=layer, return_transform_and_crs=False)
        assert isinstance(data, np.ndarray) and data.ndim in (2, 3)
        if compress:
            data = self.data.grid.compress(data)
        return data

    @property
    def pr_kg_per_m2_per_s(self) -> TwoDArrayFloat32:
        """Get precipitation rate for grid in kg/m²/s."""
        return self.compress(self.model.forcing.load("pr_kg_per_m2_per_s"))

    @property
    def ps_pascal(self) -> TwoDArrayFloat32:
        """Get surface pressure for grid in Pa."""
        return self.compress(self.model.forcing.load("ps_pascal"))

    @property
    def rlds_W_per_m2(self) -> TwoDArrayFloat32:
        """Get downward longwave radiation for grid in W/m²."""
        return self.compress(self.model.forcing.load("rlds_W_per_m2"))

    @property
    def rsds_W_per_m2(self) -> TwoDArrayFloat32:
        """Get downward shortwave radiation for grid in W/m²."""
        return self.compress(self.model.forcing.load("rsds_W_per_m2"))

    @property
    def tas_2m_K(self) -> TwoDArrayFloat32:
        """Get air temperature at 2m for grid in K."""
        return self.compress(self.model.forcing.load("tas_2m_K"))

    @property
    def dewpoint_tas_2m_K(self) -> TwoDArrayFloat32:
        """Get dewpoint temperature at 2m for grid in K."""
        return self.compress(self.model.forcing.load("dewpoint_tas_2m_K"))

    @property
    def wind_u10m_m_per_s(self) -> TwoDArrayFloat32:
        """Get u-component of wind at 10m for grid in m/s."""
        return self.compress(self.model.forcing.load("wind_u10m_m_per_s"))

    @property
    def wind_v10m_m_per_s(self) -> TwoDArrayFloat32:
        """Get v-component of wind at 10m for grid in m/s."""
        return self.compress(self.model.forcing.load("wind_v10m_m_per_s"))

    @property
    def spei_uncompressed(self) -> TwoDArrayFloat32:
        """Get uncompressed version of SPEI.

        We want to get the closest SPEI value, so if we are in the second
        half of the month, we want to get the first day of the next month.

        This is UNLESS we are at the end of the model run and the next
        SPEI value does not exist, in which case we want to keep using the
        last SPEI value available.
        """
        current_time: datetime = self.model.current_time

        # Determine the nearest first day of the month
        if current_time.day <= 15:
            spei_time: datetime = current_time.replace(day=1)
        else:
            # Move to the first day of the next month
            if current_time.month == 12:
                spei_time: datetime = current_time.replace(
                    year=current_time.year + 1, month=1, day=1
                )
            else:
                spei_time: datetime = current_time.replace(
                    month=current_time.month + 1, day=1
                )

            # Check if we ran out of SPEI data. If we did, revert to using the last month
            if (
                np.datetime64(spei_time, "ns")
                > self.model.forcing["SPEI"].reader.datetime_index[-1]
            ):
                spei_time: datetime = current_time.replace(day=1)

        spei: ThreeDArrayFloat32 = self.model.forcing.load(name="SPEI", dt=spei_time)
        assert spei.ndim == 3 and spei.shape[0] == 1
        spei: TwoDArrayFloat32 = spei[0]
        return spei

    @property
    def gev_c(self) -> xr.DataArray:
        """Get GEV (Generalized Extreme Value distribution) shape parameter of SPEI for grid."""
        return read_zarr(self.model.files["other"]["climate/gev_c"])

    @property
    def gev_loc(self) -> xr.DataArray:
        """Get GEV (Generalized Extreme Value distribution) location parameter of SPEI for grid."""
        return read_zarr(self.model.files["other"]["climate/gev_loc"])

    @property
    def gev_scale(self) -> xr.DataArray:
        """Get GEV (Generalized Extreme Value distribution) scale parameter of SPEI for grid."""
        return read_zarr(self.model.files["other"]["climate/gev_scale"])

    @property
    def pr_gev_c(self) -> TwoDArrayFloat32:
        """Get GEV (Generalized Extreme Value distribution) shape parameter of rainfall distribution for grid."""
        return read_grid(self.model.files["other"]["climate/pr_gev_c"])

    @property
    def pr_gev_loc(self) -> TwoDArrayFloat32:
        """Get GEV (Generalized Extreme Value distribution) location parameter of rainfall distribution for grid."""
        return read_grid(self.model.files["other"]["climate/pr_gev_loc"])

    @property
    def pr_gev_scale(self) -> TwoDArrayFloat32:
        """Get GEV (Generalized Extreme Value distribution) scale parameter of rainfall distribution for grid."""
        return read_grid(self.model.files["other"]["climate/pr_gev_scale"])


class HRUs(BaseVariables):
    """This class forms the basis for the HRUs. To create the `HRUs`, each individual field owned by a farmer becomes a `HRU` first. Then, in addition, each other land use type becomes a separate HRU. `HRUs` never cross cell boundaries. This means that farmers whose fields are dispersed across multiple cells are simulated by multiple `HRUs`. Here, we assume that each `HRU`, is relatively homogeneous as it each `HRU` is operated by 1) a single farmer, or by a single other (i.e., non-farm) land-use type and 2) never crosses the boundary a hydrological model cell.

    On initalization, the mask of the study area for the cells are loaded first, and a mask on the maximum resolution of the HRUs is created. In this case, the maximum resolution of the HRUs is 20 times higher than the mask. Then the HRUs are actually created.

    Args:
        data: Data class for model.
        model: The GEB model.
    """

    def __init__(self, data: Data, model: GEBModel) -> None:
        """Initialize HRUs class.

        Args:
            data: Data class for model. Contains the various types of grids used in the GEB Model.
            model: The GEB model.
        """
        self.data: Data = data
        self.model: GEBModel = model

        subgrid_mask = read_grid(self.model.files["subgrid"]["mask"])
        submask_height, submask_width = subgrid_mask.shape

        self.scaling = submask_height // self.data.grid.shape[0]
        assert submask_width // self.data.grid.shape[1] == self.scaling

        self.transform = self.data.grid.transform * Affine.scale(1 / self.scaling)
        self.crs = self.data.grid.crs

        self.gt = self.transform.to_gdal()

        self.mask = self.data.grid.mask.repeat(self.scaling, axis=0).repeat(
            self.scaling, axis=1
        )
        self.cell_size = self.data.grid.cell_size / self.scaling

        # get lats and lons for subgrid
        self.lon = np.linspace(
            self.gt[0] + self.gt[1] / 2,
            self.gt[0] + self.gt[1] * submask_width - self.gt[1] / 2,
            submask_width,
        )
        self.lat = np.linspace(
            self.gt[3] + self.gt[5] / 2,
            self.gt[3] + self.gt[5] * submask_height - self.gt[5] / 2,
            submask_height,
        )
        BaseVariables.__init__(self)

        if self.model.in_spinup:
            self.spinup()

    def spinup(self) -> None:
        """Create HRUs based on land cover and use.

        Creates the HRUs by reading the land use and farm maps and analysing them per grid cell.
        Each land use type becomes a separate HRU, and each farm field becomes a separate HRU.

        In addition, several mapping arrays are created to map between HRUs and grid cells. These are
        later used in functions to convert between HRU and grid scales.
        """
        self.var = self.model.store.create_bucket(
            "hydrology.HRU.var",
            validator=lambda x: isinstance(x, np.ndarray)
            and (not np.issubdtype(x.dtype, np.floating) or x.dtype == np.float32),
        )

        (
            self.var.land_use_type,
            self.var.land_use_ratio,
            self.var.land_owners,
            self.var.HRU_to_grid,
            self.var.grid_to_HRU,
            self.var.linear_mapping,
        ) = self.create_HRUs()

        upstream_area = read_grid(self.model.files["grid"]["routing/upstream_area"])

        self.var.nearest_river_grid_cell = determine_nearest_river_cell(
            upstream_area,
            self.var.HRU_to_grid,
            mask=self.data.grid.mask,
            threshold_m2=25_000_000,  # 25 km² to align with MERIT-Basins defintion of a river, https://www.reachhydro.org/home/params/merit-basins
        )

    @property
    def linear_mapping(self) -> TwoDArrayInt32:
        """Get the linear mapping from uncompressed to compressed indices.

        Returns:
            linear_mapping: Linear mapping array.
        """
        return self.var.linear_mapping

    @property
    def compressed_size(self) -> int:
        """Gets the compressed size of a full HRU array.

        Returns:
            compressed_size: Compressed size of HRU array.
        """
        return self.var.land_use_type.size

    @staticmethod
    @njit(cache=True)
    def create_HRUs_numba(
        farms: TwoDArrayInt32,
        land_use_classes: TwoDArrayInt32,
        mask: TwoDArrayBool,
        scaling: int,
    ) -> tuple[
        ArrayInt32,
        ArrayFloat32,
        ArrayInt32,
        ArrayInt32,
        ArrayInt32,
        TwoDArrayInt32,
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
            linear_mapping: The index of the HRU to the subcell.
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
        linear_mapping = np.full(farms.shape, -1, dtype=np.int32)

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

                        linear_mapping[
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

                        linear_mapping[
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

        land_use_ratio = (land_use_size / (scaling**2)).astype(np.float32)
        return (
            land_use_array,
            land_use_ratio,
            land_use_owner,
            HRU_to_grid,
            grid_to_HRU,
            linear_mapping,
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
            linear_mapping: The index of the HRU to the subcell.
        """
        land_use_classes = read_grid(
            self.model.files["subgrid"]["landsurface/land_use_classes"]
        )
        return self.create_HRUs_numba(
            self.data.farms, land_use_classes, self.data.grid.mask, self.scaling
        )

    def zeros(self, size: int, dtype: type, *args: Any, **kwargs: Any) -> Array:
        """Return an array of zeros with given size. Takes any other argument normally used in np.zeros.

        Args:
            size: Size of the array to create.
            dtype: Data type of the array.
            *args: Additional arguments for np.zeros.
            **kwargs: Additional keyword arguments for np.zeros.

        Returns:
            array: Array with size of number of HRUs.
        """
        return np.zeros(size, dtype, *args, **kwargs)

    def full_compressed(
        self,
        fill_value: int | float | np.integer | np.floating | bool,
        dtype: type,
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        """Return a full array with size of number of HRUs. Takes any other argument normally used in np.full.

        Args:
            fill_value: Value to fill the array with.
            dtype: Data type of the array.
            *args: Additional arguments for np.full.
            **kwargs: Arbitrary keyword arguments for np.full.

        Returns:
            array: Array with size of number of HRUs.
        """
        return np.full(self.compressed_size, fill_value, dtype, *args, **kwargs)

    def decompress(self, HRU_array: np.ndarray) -> np.ndarray:
        """Decompress HRU array.

        Args:
            HRU_array: HRU_array.

        Returns:
            outarray: Decompressed HRU_array.
        """
        if np.issubdtype(HRU_array.dtype, np.integer):
            nanvalue = -1
        elif np.issubdtype(HRU_array.dtype, bool):
            nanvalue = False
        else:
            nanvalue = np.nan
        outarray = HRU_array[self.var.linear_mapping]
        outarray[self.mask] = nanvalue
        return outarray

    @staticmethod
    @njit(cache=True)
    def convert_subgrid_to_HRU_numba(
        array: npt.NDArray[np.generic],
        linear_mapping: npt.NDArray[np.int32],
        outarray: npt.NDArray[np.generic],
        nodatavalue: int | float | np.generic,
        method: str,
    ) -> npt.NDArray[np.generic]:
        """Numba helper function to compress subgrid array to HRU array.

        Args:
            array: Uncompressed array.
            linear_mapping: The index of the HRU to the subcell.
            outarray: Array to store the output data. Must be of size of the HRUs.
            nodatavalue: Value to use for nodata.
            method: Method to use for compression. "last" uses the last value found in the uncompressed array. "mean" uses the mean value found in the uncompressed array.

        Returns:
            outarray: Compressed array.

        Raises:
            ValueError: If method is not implemented.
        """
        array = array.ravel()
        linear_mapping = linear_mapping.ravel()
        if method == "last":
            if np.isnan(nodatavalue):
                for i in range(array.size):
                    value = array[i]
                    if not np.isnan(value):
                        HRU = linear_mapping[i]
                        outarray[HRU] = value
            else:
                for i in range(array.size):
                    value = array[i]
                    if value != nodatavalue:
                        HRU = linear_mapping[i]
                        outarray[HRU] = value
        elif method == "mean":
            array = array[linear_mapping != -1]
            linear_mapping = linear_mapping[linear_mapping != -1]
            outarray[:] = np.bincount(linear_mapping, weights=array) / np.bincount(
                linear_mapping
            )
        else:
            raise ValueError("Method not implemented")
        return outarray

    def convert_subgrid_to_HRU(
        self, array: AnyDArrayWithScalar, method: str = "last"
    ) -> AnyDArrayWithScalar:
        """Convert subgrid array to HRU array.

        Because HRUs describe multiple subgrid cells, the data within
        those subgrid cells needs to be compressed to a single value per HRU.

        It can be done in two ways:
        - "last": use the last value found in the uncompressed array.
        - "mean": use the mean value found in the uncompressed array.

        Args:
            array: Uncompressed array.
            method: Method to use for compression.
                "last" uses the last value found in the uncompressed array.
                "mean" uses the mean value found in the uncompressed array.

        Returns:
            Compressed array.
        """
        assert method in ("last", "mean"), "Only last and mean method are implemented"
        assert self.mask.shape == array.shape[-2:], "Array must have same shape as mask"
        if np.issubdtype(array.dtype, np.integer):
            fill_value = -1
        else:
            fill_value = np.nan

        output_data = np.full(
            (*array.shape[:-2], self.var.land_use_ratio.size),
            fill_value,
            dtype=array.dtype,
        )

        if array.ndim == 2:
            self.convert_subgrid_to_HRU_numba(
                array,
                self.var.linear_mapping,
                output_data,
                nodatavalue=fill_value,
                method=method,
            )
        elif array.ndim == 3:
            for i in range(array.shape[0]):
                self.convert_subgrid_to_HRU_numba(
                    array[i],
                    self.var.linear_mapping,
                    output_data[i],
                    nodatavalue=fill_value,
                    method=method,
                )
        else:
            raise NotImplementedError
        return output_data

    def plot(
        self, HRU_array: np.ndarray, ax: plt.Axes | None = None, show: bool = True
    ) -> None:
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

    @property
    def pr_kg_per_m2_per_s(self) -> TwoDArrayFloat32:
        """Get precipitation rate for HRUs in kg/m²/s."""
        pr_kg_per_m2_per_s: TwoDArrayFloat32 = self.data.grid.pr_kg_per_m2_per_s
        return self.data.to_HRU(data=pr_kg_per_m2_per_s, fn=None)

    @property
    def ps_pascal(self) -> TwoDArrayFloat32:
        """Get surface pressure for HRUs in Pa."""
        ps_pascal: TwoDArrayFloat32 = self.data.grid.ps_pascal
        return self.data.to_HRU(data=ps_pascal, fn=None)

    @property
    def rlds_W_per_m2(self) -> TwoDArrayFloat32:
        """Get downward longwave radiation for HRUs in W/m²."""
        rlds_W_per_m2: TwoDArrayFloat32 = self.data.grid.rlds_W_per_m2
        return self.data.to_HRU(data=rlds_W_per_m2, fn=None)

    @property
    def rsds_W_per_m2(self) -> TwoDArrayFloat32:
        """Get downward shortwave radiation for HRUs in W/m²."""
        rsds_W_per_m2: TwoDArrayFloat32 = self.data.grid.rsds_W_per_m2
        return self.data.to_HRU(data=rsds_W_per_m2, fn=None)

    @property
    def tas_2m_K(self) -> TwoDArrayFloat32:
        """Get air temperature at 2m for HRUs in K."""
        tas_2m_K: TwoDArrayFloat32 = self.data.grid.tas_2m_K
        return self.data.to_HRU(data=tas_2m_K, fn=None)

    @property
    def dewpoint_tas_2m_K(self) -> TwoDArrayFloat32:
        """Get dewpoint temperature at 2m for HRUs in K."""
        dewpoint_tas_2m_K: TwoDArrayFloat32 = self.data.grid.dewpoint_tas_2m_K
        return self.data.to_HRU(data=dewpoint_tas_2m_K, fn=None)

    @property
    def wind_u10m_m_per_s(self) -> TwoDArrayFloat32:
        """Get u-component of wind at 10m for HRUs in m/s."""
        wind_u10m_m_per_s: TwoDArrayFloat32 = self.data.grid.wind_u10m_m_per_s
        return self.data.to_HRU(data=wind_u10m_m_per_s, fn=None)

    @property
    def wind_v10m_m_per_s(self) -> TwoDArrayFloat32:
        """Get v-component of wind at 10m for HRUs in m/s."""
        wind_v10m_m_per_s: TwoDArrayFloat32 = self.data.grid.wind_v10m_m_per_s
        return self.data.to_HRU(data=wind_v10m_m_per_s, fn=None)


class Data:
    """The base data class for the GEB model. This class contains the data for the normal grid, the HRUs, and has methods to convert between the grid and HRUs."""

    def __init__(self, model: GEBModel) -> None:
        """Initialize Data class.

        Contains the data for the normal grid, the HRUs, and has methods to convert between the grid and HRUs.

        Args:
            model: The GEB model.
        """
        self.model = model

        self.farms = read_grid(self.model.files["subgrid"]["agents/farmers/farms"])

        self.grid = Grid(self, model)
        self.HRU = HRUs(self, model)

        if self.model.in_spinup:
            self.spinup()

        self.load_water_demand()

    def spinup(self) -> None:
        """Spinup data class.

        Computes cell area for HRUs.
        """
        self.HRU.var.cell_area = self.to_HRU(
            data=self.grid.var.cell_area, fn="weightedsplit"
        )

    def load_water_demand(self) -> None:
        """Load water demand data."""
        self.model.industry_water_consumption_ds: xr.Dataset = load_water_demand_xr(
            self.model.files["other"]["water_demand/industry_water_consumption"]
        )
        self.model.industry_water_demand_ds: xr.Dataset = load_water_demand_xr(
            self.model.files["other"]["water_demand/industry_water_demand"]
        )
        self.model.livestock_water_consumption_ds: xr.Dataset = load_water_demand_xr(
            self.model.files["other"]["water_demand/livestock_water_consumption"]
        )

    def to_HRU(
        self, *, data: T_OneorTwoDArray, fn: str | None = None
    ) -> T_OneorTwoDArray:
        """Function to convert from grid to HRU (Hydrologic Response Units).

        This method is designed to transform spatial grid data into a format suitable for HRUs, which are used in to represent distinct areas with homogeneous land use, soil type, and management conditions.

        Args:
            data (array-like): The grid data to be converted. If this parameter is set, `varname` must not be provided. Data should be an array where each element corresponds to grid cell values.
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
            (*data.shape[:-1], self.HRU.var.land_use_ratio.size), dtype=data.dtype
        )

        if data.ndim == 1:
            to_HRU(
                data,
                self.HRU.var.grid_to_HRU,
                self.HRU.var.land_use_ratio,
                output_data=output_data,
                fn=fn,
            )
        elif data.ndim == 2:
            for i in range(data.shape[0]):
                to_HRU(
                    data[i],
                    self.HRU.var.grid_to_HRU,
                    self.HRU.var.land_use_ratio,
                    output_data=output_data[i],
                    fn=fn,
                )
        else:
            raise NotImplementedError
        return output_data

    def to_grid(self, *, HRU_data: np.ndarray, fn: str) -> np.ndarray:
        """Function to convert from HRUs to grid.

        Args:
            HRU_data: The HRU data to be converted (if set, varname cannot be set).
            fn: Name of function to apply to data. In most cases, several HRUs are combined into one grid unit, so a function must be applied. Choose from `mean`, `sum`, `nansum`, `max` and `min`.

        Returns:
            ouput_data: Data converted to grid units.

        Raises:
            NotImplementedError: If HRU_data is not 1D or 2D array
        """
        assert fn is not None
        assert not isinstance(HRU_data, list)
        assert isinstance(HRU_data, np.ndarray)
        if HRU_data.ndim == 1:
            outdata = to_grid(
                HRU_data,
                self.HRU.var.grid_to_HRU,
                self.HRU.var.land_use_ratio,
                fn,
            )
        elif HRU_data.ndim == 2:
            # Create output array with shape (first_dim, grid_size)
            output_data = np.zeros(
                (HRU_data.shape[0], self.HRU.var.grid_to_HRU.size),
                dtype=HRU_data.dtype,
            )
            for i in range(HRU_data.shape[0]):
                output_data[i] = to_grid(
                    HRU_data[i],
                    self.HRU.var.grid_to_HRU,
                    self.HRU.var.land_use_ratio,
                    fn,
                )
            outdata = output_data
        else:
            raise NotImplementedError("Only 1D and 2D arrays are supported")

        return outdata

    def split_HRU_data(
        self, array: AnyDArrayWithScalar, i: int, ratio: float | None = None
    ) -> AnyDArrayWithScalar:
        """Function to split HRU data.

        Args:
            array: HRU data to split.
            i: Index of HRU to split.
            ratio: Ratio of new HRU to old HRU. If None, the new HRU will have the same value as the old HRU.

        Example:
            To split HRU 5 into two HRUs, where the new HRU has 30% of the area of the old HRU, use:

            ```python
            new_HRU_data = data.split_HRU_data(old_HRU_data, 5, ratio=0.3)
            ```

            To split HRU 5 into two HRUs while keeping the original values, use:
            ```python
            new_HRU_data = data.split_HRU_data(old_HRU_data, 5)
            ```

        Returns:
            HRU data with new HRU added.
        """
        assert ratio is None or (ratio > 0 and ratio < 1)
        assert ratio is None or np.issubdtype(array.dtype, np.floating)
        assert np.issubdtype(array.dtype, (np.floating, np.integer))
        if array.ndim == 1:
            array = np.insert(array, i, array[i] * (ratio or 1), axis=0)
        elif array.ndim == 2:
            array = np.insert(array, i, array[:, i] * (ratio or 1), axis=1)
        else:
            raise NotImplementedError
        if ratio is not None:
            array[i + 1] = (1 - ratio) * array[i + 1]
        return array

    @property
    def mapping_grid_to_HRU_uncompressed(self) -> npt.NDArray[np.int32]:
        """Get uncompressed grid to HRU mapping.

        Returns:
            Uncompressed grid to HRU mapping.
        """
        return self.grid.decompress(self.HRU.var.grid_to_HRU, fillvalue=-1).ravel()

    def split(self, HRU_indices: npt.NDArray[np.int32]) -> int:
        """Function to split an HRU into two HRUs.

        Args:
            HRU_indices: Indices of the HRU to split. All indices must belong to the same HRU.

        Returns:
            New HRU index.

        """
        HRU = self.HRU.var.linear_mapping[HRU_indices]
        assert (
            HRU == HRU[0]
        ).all()  # assert all indices belong to same HRU - so only works for single grid cell at this moment
        HRU = HRU[0]
        assert HRU != -1

        all_HRU_indices = np.where(
            self.HRU.var.linear_mapping == HRU
        )  # this could probably be speed up
        assert (
            all_HRU_indices[0].size > HRU_indices[0].size
        )  # ensure that not all indices are split off
        ratio = HRU_indices[0].size / all_HRU_indices[0].size

        self.HRU.var.linear_mapping[self.HRU.var.linear_mapping > HRU] += 1
        self.HRU.var.linear_mapping[HRU_indices] += 1

        self.HRU.var.HRU_to_grid = self.split_HRU_data(self.HRU.var.HRU_to_grid, HRU)
        self.HRU.var.grid_to_HRU[self.HRU.var.HRU_to_grid[HRU] :] += 1

        self.HRU.var.land_owners = self.split_HRU_data(self.HRU.var.land_owners, HRU)
        self.model.agents.farmers.update_field_indices()

        self.model.agents.farmers.field_indices = self.split_HRU_data(
            self.model.agents.farmers.field_indices, HRU
        )

        self.HRU.var.land_use_type = self.split_HRU_data(
            self.HRU.var.land_use_type, HRU
        )
        self.HRU.var.land_use_ratio = self.split_HRU_data(
            self.HRU.var.land_use_ratio, HRU, ratio=ratio
        )
        self.HRU.var.cell_area = self.split_HRU_data(
            self.HRU.var.cell_area, HRU, ratio=ratio
        )
        self.HRU.var.crop_map = self.split_HRU_data(self.HRU.var.crop_map, HRU)
        self.HRU.var.crop_age_days_map = self.split_HRU_data(
            self.HRU.var.crop_age_days_map, HRU
        )
        self.HRU.var.crop_harvest_age_days = self.split_HRU_data(
            self.HRU.var.crop_harvest_age_days, HRU
        )
        self.HRU.var.SnowCoverS = self.split_HRU_data(self.HRU.var.SnowCoverS, HRU)
        self.HRU.var.DeltaTSnow = self.split_HRU_data(self.HRU.var.DeltaTSnow, HRU)
        self.HRU.var.frost_index = self.split_HRU_data(self.HRU.var.frost_index, HRU)
        self.HRU.var.percolationImp = self.split_HRU_data(
            self.HRU.var.percolationImp, HRU
        )
        self.HRU.var.cropGroupNumber = self.split_HRU_data(
            self.HRU.var.cropGroupNumber, HRU
        )
        self.HRU.var.actual_bare_soil_evaporation = self.split_HRU_data(
            self.HRU.var.actual_bare_soil_evaporation, HRU
        )
        self.HRU.var.KSat1 = self.split_HRU_data(self.HRU.var.KSat1, HRU)
        self.HRU.var.KSat2 = self.split_HRU_data(self.HRU.var.KSat2, HRU)
        self.HRU.var.KSat3 = self.split_HRU_data(self.HRU.var.KSat3, HRU)
        self.HRU.var.lambda1 = self.split_HRU_data(self.HRU.var.lambda1, HRU)
        self.HRU.var.lambda2 = self.split_HRU_data(self.HRU.var.lambda2, HRU)
        self.HRU.var.lambda3 = self.split_HRU_data(self.HRU.var.lambda3, HRU)
        self.HRU.var.wwp1 = self.split_HRU_data(self.HRU.var.wwp1, HRU)
        self.HRU.var.wwp2 = self.split_HRU_data(self.HRU.var.wwp2, HRU)
        self.HRU.var.wwp3 = self.split_HRU_data(self.HRU.var.wwp3, HRU)
        self.HRU.var.ws1 = self.split_HRU_data(self.HRU.var.ws1, HRU)
        self.HRU.var.ws2 = self.split_HRU_data(self.HRU.var.ws2, HRU)
        self.HRU.var.ws3 = self.split_HRU_data(self.HRU.var.ws3, HRU)
        self.HRU.var.wres1 = self.split_HRU_data(self.HRU.var.wres1, HRU)
        self.HRU.var.wres2 = self.split_HRU_data(self.HRU.var.wres2, HRU)
        self.HRU.var.wres3 = self.split_HRU_data(self.HRU.var.wres3, HRU)
        self.HRU.var.wfc1 = self.split_HRU_data(self.HRU.var.wfc1, HRU)
        self.HRU.var.wfc2 = self.split_HRU_data(self.HRU.var.wfc2, HRU)
        self.HRU.var.wfc3 = self.split_HRU_data(self.HRU.var.wfc3, HRU)
        self.HRU.var.kunSatFC12 = self.split_HRU_data(self.HRU.var.kunSatFC12, HRU)
        self.HRU.var.kunSatFC23 = self.split_HRU_data(self.HRU.var.kunSatFC23, HRU)
        self.HRU.var.arnoBeta = self.split_HRU_data(self.HRU.var.arnoBeta, HRU)
        self.HRU.var.w1 = self.split_HRU_data(self.HRU.var.w1, HRU)
        self.HRU.var.w2 = self.split_HRU_data(self.HRU.var.w2, HRU)
        self.HRU.var.w3 = self.split_HRU_data(self.HRU.var.w3, HRU)
        self.HRU.var.topwater = self.split_HRU_data(self.HRU.var.topwater, HRU)
        self.HRU.var.totAvlWater = self.split_HRU_data(self.HRU.var.totAvlWater, HRU)
        self.HRU.var.minInterceptCap = self.split_HRU_data(
            self.HRU.var.minInterceptCap, HRU
        )
        self.HRU.var.interception_storage = self.split_HRU_data(
            self.HRU.var.interception_storage, HRU
        )
        self.HRU.var.potential_transpiration_crop_life = self.split_HRU_data(
            self.HRU.var.potential_transpiration_crop_life, HRU
        )
        self.HRU.var.transpiration_crop_life = self.split_HRU_data(
            self.HRU.var.transpiration_crop_life, HRU
        )
        return HRU
