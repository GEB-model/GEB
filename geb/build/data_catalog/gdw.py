"""Download Global Dam Watch v1.0 barriers and reservoir outlines."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd

from geb.workflows.io import fetch_and_save, write_geom

from .base import Adapter


def prepare_gdw(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rename GDW fields, replace missing-data codes, and convert units.

    Args:
        data: One layer from the GDW v1.0 geodatabase.

    Returns:
        Data in EPSG:4326 with missing-data codes replaced by nulls. Area uses m2,
        capacity m3, and discharge m3/s. Other units stay as given by GDW.

    Raises:
        ValueError: If required fields, a CRS, or unique positive IDs are missing.
    """
    data = data.copy()
    data.columns = data.columns.str.lower()
    required_fields: set[str] = {
        "gdw_id",
        "hylak_id",
        "dam_type",
        "lake_ctrl",
        "cap_mcm",
        "area_skm",
        "dis_avg_ls",
        "geometry",
    }
    if not required_fields.issubset(data.columns) or data.crs is None:
        raise ValueError("GDW must have a CRS and the required v1.0 fields.")
    if (
        data["gdw_id"].isna().any()
        or not data["gdw_id"].is_unique
        or (data["gdw_id"] <= 0).any()
    ):
        raise ValueError("GDW IDs must be unique positive integers.")

    column: str
    for column in data.columns.drop("geometry"):
        if pd.api.types.is_numeric_dtype(data[column]):
            # Keep -99 where it is a valid coordinate or elevation.
            if column == "elev_masl":
                data[column] = data[column].mask(data[column] == -9999)
            elif column not in {"long_riv", "lat_riv", "long_dam", "lat_dam"}:
                data[column] = data[column].mask(data[column] == -99)
        else:
            data[column] = data[column].str.strip().replace("", pd.NA)

    data.rename(
        columns={
            "hylak_id": "hydrolakes_id",
            "res_name": "reservoir_name",
            "lake_ctrl": "lake_control",
            "year_dam": "construction_year",
            "cap_mcm": "capacity_m3",
            "area_skm": "area_m2",
            "dis_avg_ls": "average_discharge_m3_per_s",
            "poly_src": "polygon_source",
            "orig_src": "source",
        },
        inplace=True,
    )
    data["gdw_id"] = data["gdw_id"].astype("int64")
    data["hydrolakes_id"] = (
        data["hydrolakes_id"].mask(data["hydrolakes_id"] <= 0).astype("Int64")
    )
    data["capacity_m3"] *= 1e6
    data["area_m2"] *= 1e6
    data["average_discharge_m3_per_s"] /= 1000
    return data.to_crs(4326)


class GlobalDamWatch(Adapter):
    """Download GDW once and save its dam points and reservoir outlines."""

    def fetch(self, url: str) -> GlobalDamWatch:
        """Download GDW v1.0 and save its points and outlines as GeoParquet.

        Args:
            url: Download URL for the GDW v1.0 geodatabase ZIP archive.

        Returns:
            This adapter, ready to read its selected layer.

        Raises:
            ValueError: If the archive does not contain a single geodatabase.
        """
        if self.is_ready:
            return self

        with TemporaryDirectory(dir=self.root) as folder:
            archive: Path = Path(folder) / "gdw.zip"
            fetch_and_save(url=url, file_path=archive)
            with ZipFile(archive) as zipped:
                zipped.extractall(folder)
            databases: list[Path] = list(Path(folder).glob("*.gdb"))
            if len(databases) != 1:
                raise ValueError("Expected one GDW geodatabase in the archive.")
            layer: str
            for layer in ("barriers", "reservoirs"):
                data: gpd.GeoDataFrame = prepare_gdw(
                    gpd.read_file(databases[0], layer=f"GDW_{layer}_v1_0")
                )
                write_geom(
                    data, self.root / f"{layer}.parquet", write_covering_bbox=True
                )
        return self
