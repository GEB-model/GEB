"""Adapter for GRDC-Caravan catchment attributes."""

from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from geb.workflows.io import RemoteFile

from .base import Adapter


class GRDCCaravan(Adapter):
    """Download and combine the small GRDC-Caravan attribute tables."""

    ATTRIBUTE_PATHS: tuple[str, ...] = (
        "attributes/grdc/attributes_other_grdc.csv",
        "attributes/grdc/attributes_additional_grdc.csv",
        "attributes/grdc/attributes_caravan_grdc.csv",
        "attributes/grdc/attributes_hydroatlas_grdc.csv",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GRDC-Caravan adapter.

        Args:
            *args: Positional arguments passed to the base adapter.
            **kwargs: Keyword arguments passed to the base adapter.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> GRDCCaravan:
        """Fetch and cache the GRDC-Caravan catchment attributes.

        HTTP range requests extract only four CSV members from the remote
        archive, avoiding a full download of the multi-gigabyte time series.

        Args:
            url: URL of the GRDC-Caravan NetCDF ZIP archive on Zenodo.

        Returns:
            This adapter with the combined attribute table available at ``path``.

        Raises:
            RuntimeError: If the remote archive cannot be read.
            ValueError: If an attribute table has invalid station identifiers.
        """
        if self.is_ready:
            return self

        archive_root: str = "GRDC_Caravan_extension_nc/"
        attribute_tables: list[pd.DataFrame] = []
        try:
            with ZipFile(RemoteFile(url)) as archive:
                for relative_path in self.ATTRIBUTE_PATHS:
                    member_data: bytes = archive.read(archive_root + relative_path)
                    attribute_table: pd.DataFrame = pd.read_csv(BytesIO(member_data))
                    if "gauge_id" not in attribute_table.columns:
                        raise ValueError(
                            f"GRDC-Caravan table {relative_path} has no gauge_id column."
                        )
                    attribute_tables.append(attribute_table)
        except ValueError:
            raise
        except Exception as error:
            raise RuntimeError(
                "Could not read GRDC-Caravan attributes from Zenodo. "
                "Check internet access or pre-populate the global data cache."
            ) from error

        # The second table uses an ISO code while the first spells out the country.
        attribute_tables[1] = attribute_tables[1].rename(
            columns={"country": "country_code"}
        )
        # The basic metadata table defines the released station set; a few
        # source-table ID inconsistencies must not create attribute-only rows.
        combined_attributes: pd.DataFrame = attribute_tables[0]
        for attribute_table in attribute_tables[1:]:
            combined_attributes = combined_attributes.merge(
                attribute_table,
                on="gauge_id",
                how="left",
                validate="one_to_one",
            )

        if combined_attributes["gauge_id"].duplicated().any():
            raise ValueError("GRDC-Caravan contains duplicate gauge_id values.")

        output_path: Path = self.path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_attributes.to_parquet(output_path, index=False)
        self.logger.info(
            "Cached %d GRDC-Caravan catchment attribute records at %s.",
            len(combined_attributes),
            output_path,
        )
        return self

    def read(self, **kwargs: Any) -> pd.DataFrame:
        """Read the cached station attribute table.

        Args:
            **kwargs: Optional arguments passed to ``pandas.read_parquet``.

        Returns:
            Combined GRDC-Caravan station attributes.
        """
        return pd.read_parquet(self.path, **kwargs)
