"""Adapter for MIRCA-OS admin boundaries datasets."""

from typing import Any

import geopandas as gpd

from .base import Adapter


class MIRCAOSAdminBoundaries(Adapter):
    """Adapter for MIRCA-OS admin boundaries datasets."""

    def fetch(self, url: str) -> MIRCAOSAdminBoundaries:
        """Fetch the MIRCA-OS admin boundaries dataset.

        Args:
            url: The URL to download the dataset from.

        Returns:
            MIRCAOSAdminBoundaries: The adapter instance.
        """
        if not self.is_ready:
            while not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                print(
                    "\033[91mThis dataset requires manual download and extraction. "
                    + f"Please download MIRCA-OS Admin Boundaries (MIRCA-OS_Admin_Boundaries_v1.rar) from: {url}\n"
                    + f"Then extract it and place the shapefile (e.g., MIRCAOS_2000_Admin_v1.shp) at: {self.path}\033[0m"
                )
                input(
                    "\033[91mPress Enter after placing the file to continue...\033[0m"
                )

        return self

    def read(self, **kwargs: Any) -> gpd.GeoDataFrame:
        """Read the admin boundaries shapefile.

        Returns:
            gpd.GeoDataFrame: The admin boundaries data.
        """
        gdf = gpd.read_file(self.path, **kwargs)
        gdf["unit_code"] = gdf["unit_code"].astype(int)

        return gdf
