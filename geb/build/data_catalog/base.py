"""Base class for data adapters in the GEB data catalog."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import xarray as xr

from geb.workflows.geometry import read_parquet_with_geom
from geb.workflows.io import read_zarr


class Adapter:
    """Base class for data adapters in the GEB data catalog."""

    def __init__(
        self,
        folder: Path | str | None = None,
        filename: Path | str | None = None,
        local_version: int | None = None,
        cache: str | None = None,
    ) -> None:
        """Initialize the data adapter.

        Each adapter manages a specific dataset, handling its download, processing, and storage.

        Either all or none of folder, filename, local_version, and cache must be provided.

        The cache directory is determined by the GEB_DATA_ROOT environment variable,
        or defaults to ~/.geb_cache if the variable is not set.

        Args:
            folder: The subfolder within the cache directory for this dataset.
            filename: The filename for the processed dataset.
            local_version: The local version number of the dataset.
            cache: Either 'global' or 'local'. 'global' uses the GEB_DATA_ROOT or ~/.geb_cache,
                   'local' uses a local cache directory within the model directory.

        Raises:
            ValueError: If only some of folder, filename, local_version, and cache are provided
                        instead of all or none.
        """
        if (
            folder is None or filename is None or local_version is None or cache is None
        ) and (
            folder is not None
            or filename is not None
            or local_version is not None
            or cache is not None
        ):
            raise ValueError(
                "Either all of folder, filename, local_version, and cache must be provided, or none."
            )

        self.folder = folder
        self.local_version = local_version
        self.filename = filename
        self.cache = cache

    @property
    def path(self) -> Path:
        """Local path to the processed data file.

        Returns:
            The full path to the processed data file.

        Raises:
            ValueError: If the root directory is not set.
            ValueError: If the filename is not set.
        """
        if self.root is None:
            raise ValueError("Root directory is not set; cannot determine data path.")
        if self.filename is None:
            raise ValueError("Filename is not set; cannot determine data path.")
        return self.root / self.filename

    @property
    def root(self) -> Path | None:
        """Root directory for the dataset.

        If the directory does not exist, it will be created.

        Use the GEB_DATA_ROOT environment variable if set, otherwise defaults to ~/.geb_cache.

        Returns:
            The root directory path for the dataset.

        Raises:
            ValueError: If the cache attribute is not 'global' or 'local'.
        """
        if self.folder is None or self.local_version is None or self.cache is None:
            raise ValueError("Root directory is not set; cannot determine data root.")
        if self.cache == "global":
            geb_data_root: str | None = os.getenv(key="GEB_DATA_ROOT", default=None)
            if geb_data_root:
                catalog_root: Path = Path(geb_data_root) / ".." / "datacatalog"
            else:
                catalog_root: Path = Path.home() / ".geb_cache"

            root = catalog_root / self.folder / f"v{self.local_version}"
            root.mkdir(parents=True, exist_ok=True)
            return root
        elif self.cache == "local":
            return Path("cache") / self.folder / f"v{self.local_version}"
        else:
            raise ValueError("Cache must be either 'global' or 'local'")

    @property
    def is_ready(self) -> bool:
        """Check if the data is already downloaded and processed.

        Returns:
            True if the data file exists, False otherwise.
        """
        is_ready = self.path.exists()
        if not is_ready:
            print(
                f"Data not found at {self.path}, downloading and/or processing may take a while..."
            )
        return is_ready

    def fetch(self) -> Adapter:
        """Process the data after downloading.

        Returns:
            The Adapter instance.
        """
        return self

    def read(self, **kwargs: Any) -> xr.DataArray | pd.DataFrame | gpd.GeoDataFrame:
        """Read the processed data from storage.

        Detects the file format based on the file extension and uses the appropriate reader.

        Args:
            *args: Additional positional arguments to pass to the reader function.
            **kwargs: Additional keyword arguments to pass to the reader function.

        Returns:
            The processed data as an xarray DataArray or GeoDataFrame.

        Raises:
            ValueError: If the file format is unsupported.

        """
        if self.path.suffix == ".zarr":
            return read_zarr(self.path)
        elif self.path.suffix == ".nc":
            return xr.open_dataarray(self.path, **kwargs)
        elif self.path.suffix in (".tif", ".asc"):
            return xr.open_dataarray(self.path, **kwargs)
        elif self.path.suffix == ".parquet":
            if "columns" in kwargs and "geometry" not in kwargs["columns"]:
                return pd.read_parquet(path=self.path, **kwargs)
            else:
                return read_parquet_with_geom(path=self.path, **kwargs)
        elif self.path.suffix == ".gpkg":
            return gpd.read_file(self.path, **kwargs)
        elif self.path.suffix == ".vrt":
            return xr.open_dataarray(self.path, **kwargs)
        elif self.path.suffix == ".csv":
            return pd.read_csv(self.path, **kwargs)
        elif self.path.suffix == ".xlsx":
            return pd.read_excel(self.path, **kwargs)
        else:
            raise ValueError("Unsupported file format for reading data.")
