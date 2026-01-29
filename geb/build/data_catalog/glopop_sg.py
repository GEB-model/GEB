"""Data adapter for GLOPOP-SG population data."""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from tqdm import tqdm

from geb.workflows.io import RemoteFile

from .base import Adapter


class GLOPOP_SG(Adapter):
    """Adapter for GLOPOP-SG population data."""

    def fetch(self, region: str, url: str) -> GLOPOP_SG:
        """Fetch data for a specific region.

        Args:
            region: The GDL region code (e.g., 'AFG').
            url: Optional URL to override the default.

        Returns:
            Self.
        """
        tif_name = f"{region}_grid_nr.tif"
        gz_name = f"synthpop_{region}_grid.dat.gz"

        tif_path = self.root / tif_name
        gz_path = self.root / gz_name

        if tif_path.exists() and gz_path.exists():
            return self

        self._extract_from_remote(tif_name, tif_path, url)
        self._extract_from_remote(gz_name, gz_path, url)

        return self

    def _extract_from_remote(self, filename: str, output_path: Path, url: str) -> None:
        """Extract a single file from the remote ZIP to the output path.

        Raises:
            FileNotFoundError: If the file is not found in the remote zip.
        """
        if output_path.exists():
            return

        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            with zipfile.ZipFile(RemoteFile(url), "r") as zf:
                try:
                    file_info = zf.getinfo(filename)
                    file_size = file_info.file_size
                    with (
                        zf.open(filename) as source,
                        open(temp_path, "wb") as target,
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
                            target.write(chunk)
                            pbar.update(len(chunk))
                except KeyError:
                    raise FileNotFoundError(f"{filename} not found in remote zip.")
            temp_path.rename(output_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def read(self, region: str) -> tuple[pd.DataFrame, xr.DataArray]:
        """Read GLOPOP-SG data for a region.

        Args:
            region: The GDL region code.

        Returns:
            Tuple of (DataFrame of population, DataArray of grid).

        Raises:
            ValueError: If the adapter is not initialized with a folder.
        """
        if self.root is None:
            raise ValueError("Generic adapter not initialized with a folder.")

        tif_path = self.root / f"{region}_grid_nr.tif"
        gz_path = self.root / f"synthpop_{region}_grid.dat.gz"

        GLOPOP_grid = rxr.open_rasterio(tif_path)
        if isinstance(GLOPOP_grid, list):
            GLOPOP_grid = GLOPOP_grid[0]

        if not isinstance(GLOPOP_grid, xr.DataArray):
            # If it is a Dataset, we might need to select a variable or convert.
            # Usually rioxarray returns DataArray for single band tifs.
            # If it's a Dataset, try converting or raising.
            if isinstance(GLOPOP_grid, xr.Dataset):
                # Convert to array if possible or take the first variable
                GLOPOP_grid = GLOPOP_grid.to_array().squeeze()

        assert isinstance(GLOPOP_grid, xr.DataArray)

        with gzip.open(gz_path, "rb") as f:
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

        trimmed_GLOPOP = GLOPOP_s[: n_people * n_attr]

        df = pd.DataFrame(
            np.reshape(trimmed_GLOPOP, (n_attr, n_people)).transpose(),
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

        if df["GRID_Y"].isna().any():
            df = df[~df.GRID_Y.isna()]

        return df, GLOPOP_grid
