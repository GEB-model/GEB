"""Implementation of the abstract base class for model components."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from hydromt import DataCatalog

if TYPE_CHECKING:
    from geb.build import DelayedReader
    from geb.build.data_catalog import NewDataCatalog


class BuildModelBase(ABC):
    """Abstract base class for model components."""

    @property
    @abstractmethod
    def grid(self) -> xr.Dataset:
        """Abstract property for grid data."""
        pass

    @grid.setter
    @abstractmethod
    def grid(self, value: xr.Dataset) -> None:
        """Abstract setter for grid data."""
        pass

    @property
    @abstractmethod
    def subgrid(self) -> xr.Dataset:
        """Abstract property for subgrid data."""
        pass

    @subgrid.setter
    @abstractmethod
    def subgrid(self, value: xr.Dataset) -> None:
        """Abstract setter for subgrid data."""
        pass

    @property
    @abstractmethod
    def region_subgrid(self) -> xr.Dataset:
        """Abstract property for region_subgrid data."""
        pass

    @subgrid.setter
    @abstractmethod
    def region_subgrid(self, value: xr.Dataset) -> None:
        """Abstract setter for region_subgrid data."""
        pass

    @property
    @abstractmethod
    def geom(self) -> DelayedReader:
        """Abstract property for geometry data."""
        pass

    @geom.setter
    @abstractmethod
    def geom(self, value: DelayedReader) -> None:
        """Abstract setter for geometry data."""
        pass

    @property
    @abstractmethod
    def table(self) -> DelayedReader:
        """Abstract property for table data."""
        pass

    @table.setter
    @abstractmethod
    def table(self, value: DelayedReader) -> None:
        """Abstract setter for table data."""
        pass

    @property
    @abstractmethod
    def other(self) -> DelayedReader:
        """Abstract property for other data."""
        pass

    @other.setter
    @abstractmethod
    def other(self, value: DelayedReader) -> None:
        """Abstract setter for other data."""
        pass

    @property
    @abstractmethod
    def root(self) -> Path:
        """Abstract property for root directory."""
        pass

    @property
    @abstractmethod
    def array(self) -> DelayedReader:
        """Abstract property for array data."""
        pass

    @array.setter
    @abstractmethod
    def array(self, value: DelayedReader) -> None:
        """Abstract setter for array data."""
        pass

    @property
    @abstractmethod
    def params(self) -> DelayedReader:
        """Abstract property for params data."""
        pass

    @params.setter
    @abstractmethod
    def params(self, value: Any) -> None:
        """Abstract setter for params data."""
        pass

    @root.setter
    @abstractmethod
    def root(self, value: Path) -> None:
        """Abstract setter for root directory."""
        pass

    @property
    @abstractmethod
    def data_catalog(self) -> DataCatalog:
        """Abstract property for data catalog."""
        pass

    @data_catalog.setter
    @abstractmethod
    def data_catalog(self, value: DataCatalog) -> None:
        """Abstract setter for data catalog."""
        pass

    @property
    @abstractmethod
    def new_data_catalog(self) -> NewDataCatalog:
        """Abstract property for data catalog."""
        pass

    @new_data_catalog.setter
    @abstractmethod
    def new_data_catalog(self, value: NewDataCatalog) -> None:
        """Abstract setter for data catalog."""
        pass

    @property
    @abstractmethod
    def logger(self) -> Logger:
        """Abstract property for logger."""
        pass

    @logger.setter
    @abstractmethod
    def logger(self, value: Logger) -> None:
        """Abstract setter for logger."""
        pass

    @property
    @abstractmethod
    def report_dir(self) -> Path:
        """Abstract property for report directory."""
        pass

    @report_dir.setter
    @abstractmethod
    def report_dir(self, value: Path) -> None:
        """Abstract setter for report directory."""
        pass

    @abstractmethod
    def set_table(self, table: pd.DataFrame, name: str, write: bool = True) -> None:
        """Abstract method to set a table."""
        pass

    @abstractmethod
    def set_geom(self, geom: gpd.GeoDataFrame, name: str) -> None:
        """Abstract method to set geometry."""
        pass

    @abstractmethod
    def set_grid(self, data: xr.DataArray, name: str) -> xr.DataArray:
        """Abstract method to set grid data."""
        pass

    @abstractmethod
    def set_subgrid(self, data: xr.DataArray, name: str) -> xr.DataArray:
        """Abstract method to set subgrid data."""
        pass

    @abstractmethod
    def set_region_subgrid(self, data: xr.DataArray, name: str) -> xr.DataArray:
        """Abstract method to set region subgrid."""
        pass

    @abstractmethod
    def set_other(self, da: xr.DataArray, name: str, **kwargs: Any) -> xr.DataArray:
        """Abstract method to set other data."""
        pass

    @abstractmethod
    def set_array(self, data: np.ndarray, name: str) -> None:
        """Abstract method to set an array."""
        pass

    @abstractmethod
    def set_params(self, data: Any, name: str) -> None:
        """Abstract method to set a dictionary."""
        pass

    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """Abstract method to get bounds."""
        pass

    @property
    @abstractmethod
    def region(self) -> gpd.GeoDataFrame:
        """Abstract method to get region geometry."""
        pass

    @region.setter
    @abstractmethod
    def region(self, value: gpd.GeoDataFrame) -> None:
        """Abstract setter for region geometry."""
        pass

    @abstractmethod
    def full_like(
        self,
        data: xr.DataArray,
        fill_value: int | float | bool,
        nodata: int | float | bool | None,
        attrs: dict | None = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Abstract method to create a full_like DataArray."""
        pass

    @property
    @abstractmethod
    def subgrid_factor(self) -> int:
        """Abstract method to get subgrid factor."""
        pass

    @abstractmethod
    def ldd_scale_factor(self) -> int:
        """Abstract method to get LDD scale factor."""
        pass

    @property
    @abstractmethod
    def files(self) -> dict[str, dict[str, Path]]:
        """Abstract method to get files dictionary."""
        pass

    @files.setter
    @abstractmethod
    def files(self, value: dict[str, dict[str, Path]]) -> None:
        """Abstract setter for files dictionary."""
        pass

    @property
    @abstractmethod
    def start_date(self) -> datetime.datetime:
        """Abstract method to get start date."""
        pass

    @start_date.setter
    @abstractmethod
    def start_date(self, value: datetime.datetime) -> None:
        """Abstract setter for start date."""
        pass

    @property
    @abstractmethod
    def end_date(self) -> datetime.datetime:
        """Abstract method to get end date."""
        pass

    @end_date.setter
    @abstractmethod
    def end_date(self, value: datetime.datetime) -> None:
        """Abstract setter for end date."""
        pass

    @property
    @abstractmethod
    def ISIMIP_ssp(self) -> str:
        """Abstract method to get ISIMIP SSP scenario."""
        pass

    @ISIMIP_ssp.setter
    @abstractmethod
    def ISIMIP_ssp(self, value: str) -> None:
        """Abstract setter for ISIMIP SSP scenario."""
        pass

    @property
    @abstractmethod
    def ssp(self) -> str:
        """Abstract method to get SSP scenario."""
        pass

    @ssp.setter
    @abstractmethod
    def ssp(self, value: str) -> None:
        """Abstract setter for SSP scenario."""
        pass
