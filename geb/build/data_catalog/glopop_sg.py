"""Data adapter for GLOPOP-SG population data."""

import gzip
import io
import zipfile

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from tqdm import tqdm

from geb.workflows.io import RemoteFile

from .base import Adapter


class GLOPOP_SG(Adapter):
    """Adapter for GLOPOP-SG population data."""

    def fetch(self, url: str) -> GLOPOP_SG:
        """Fetch data for a specific region.

        Args:
            url: Optional URL to override the default.

        Returns:
            The GLOPOP_SG instance.
        """
        self.url = url
        return self

    def _extract_from_remote_to_memory(self, filename: str, url: str) -> io.BytesIO:
        """Extract a single file from the remote ZIP into memory.

        Args:
            filename: The name of the file within the ZIP archive.
            url: URL of the remote ZIP archive.

        Returns:
            A BytesIO object containing the file data.

        Raises:
            FileNotFoundError: If the file is not found in the remote zip.
        """
        buffer = io.BytesIO()
        with zipfile.ZipFile(RemoteFile(url), "r") as zf:
            try:
                file_info = zf.getinfo(filename)
                file_size = file_info.file_size
                with (
                    zf.open(filename) as source,
                    tqdm(
                        total=file_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {filename}",
                    ) as pbar,
                ):
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        buffer.write(chunk)
                        pbar.update(len(chunk))
            except KeyError:
                raise FileNotFoundError(f"{filename} not found in remote zip.")
        buffer.seek(0)
        return buffer

    def read(self, region: str) -> tuple[pd.DataFrame, xr.DataArray]:
        """Read GLOPOP-SG data for a region.

        Note:
            Downloads the necessary files directly into memory.

        Args:
            region: The GDL region code.

        Returns:
            Tuple of (DataFrame of population, DataArray of grid).
        """
        tif_name = f"{region}_grid_nr.tif"
        gz_name = f"synthpop_{region}_grid.dat.gz"

        # Download tif and gz into memory
        tif_buffer = self._extract_from_remote_to_memory(tif_name, self.url)
        gz_buffer = self._extract_from_remote_to_memory(gz_name, self.url)

        # Open raster from buffer
        GLOPOP_grid = rxr.open_rasterio(tif_buffer)
        assert isinstance(GLOPOP_grid, xr.DataArray)
        GLOPOP_grid: xr.DataArray = GLOPOP_grid.load()

        # Open gz from buffer
        with gzip.open(gz_buffer, "rb") as f:
            GLOPOP_s = np.frombuffer(f.read(), dtype=np.int32)

        GLOPOP_s_attribute_names: list[str] = [
            "HID",
            "RELATE_HEAD",
            "INCOME",
            "WEALTH",
            "RURAL",
            "AGE",
            "GENDER",
            "EDUC",
            "HHTYPE",
            "HHSIZE_CAT",
            "AGRI_OWNERSHIP",
            "FLOOR",
            "WALL",
            "ROOF",
            "SOURCE",
            "GRID_CELL",
        ]

        n_attr = len(GLOPOP_s_attribute_names)
        total = GLOPOP_s.size
        n_people = total // n_attr

        countries_with_17_columns = ["COD", "IRN", "KWT", "MKD", "THA"]
        # check if string of region contains any of the country codes with 17 columns (SHOULD BE FIXED BY MARIJN IN NEW GLOPOP VERSION)
        if any(code in region for code in countries_with_17_columns):  # 17 columns
            n_columns = 17
            n_people = GLOPOP_s.size // n_columns
            data_reshaped = np.reshape(GLOPOP_s, (n_columns, n_people)).transpose()
            data_reshaped = np.hstack((data_reshaped[:, :-2], data_reshaped[:, -1:]))
            print(region)
            print("17 columns")
        else:  # default to 16 columns
            # reshape data
            data_reshaped = np.reshape(GLOPOP_s, (n_attr, n_people)).transpose()

        df = pd.DataFrame(
            data_reshaped,
            columns=np.array(GLOPOP_s_attribute_names),
        )

        grid_vals = GLOPOP_grid.values
        if grid_vals.ndim == 3:
            grid_vals = grid_vals[0]

        mask = grid_vals > -1
        rows, cols = np.where(mask)
        grid_cells = grid_vals[mask]

        grid_coords = pd.DataFrame(
            {
                "GRID_Y": rows,
                "GRID_X": cols,
                "GRID_CELL": grid_cells,
            }
        )

        grid_coords["coord_Y"] = GLOPOP_grid.y.values[grid_coords["GRID_Y"].values]
        grid_coords["coord_X"] = GLOPOP_grid.x.values[grid_coords["GRID_X"].values]

        df = df.merge(grid_coords, on="GRID_CELL", how="left")

        return df, GLOPOP_grid
