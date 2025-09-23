"""Data catalog for predefined datasets in GEB."""

from typing import Any

from .base import Adapter
from .esa_worldcover import ESAWorldCover
from .gadm import GADM
from .hydrolakes import HydroLakes
from .merit_basins import MeritBasinsCatchments, MeritBasinsRivers
from .merit_hydro import MeritHydroDir, MeritHydroElv
from .merit_sword import MeritSword
from .sword import Sword
from .world_bank import WorldBankData

data_catalog: dict[str, dict[str, Any]] = {
    "wb_inflation_rate": {
        "adapter": WorldBankData(
            folder="world_bank_inflation_rate",
            local_version=1,
            filename="wb_inflation_rate.csv",
            cache="global",
        ),
        "url": "https://api.worldbank.org/v2/en/indicator/FP.CPI.TOTL.ZG?downloadformat=csv",
        "source": {
            "name": "World Bank Inflation Data",
            "author": "The World Bank",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        },
    },
    "world_bank_price_ratio": {
        "adapter": WorldBankData(
            folder="world_bank_price_ratio",
            local_version=1,
            filename="world_bank_price_ratio.csv",
            cache="global",
        ),
        "url": "https://api.worldbank.org/v2/en/indicator/PA.NUS.PPPC.RF?downloadformat=csv",
        "source": {
            "name": "Official exchange rate (LCU per US$, period average)",
            "author": "The World Bank",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "url": "https://data.worldbank.org/indicator/PA.NUS.PPPC.RF",
        },
    },
    "world_bank_LCU_per_USD": {
        "adapter": WorldBankData(
            folder="world_bank_LCU_per_USD",
            local_version=1,
            filename="world_bank_LCU_per_USD.csv",
            cache="global",
        ),
        "url": "https://api.worldbank.org/v2/en/indicator/PA.NUS.FCRF?downloadformat=csv",
        "source": {
            "name": "World Bank Local Currency Unit per US Dollar Data",
            "author": "The World Bank",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "url": "https://data.worldbank.org/indicator/PA.NUS.FCRF",
        },
    },
    "esa_worldcover_2020": {
        "adapter": ESAWorldCover(),
        "url": "https://services.terrascope.be/stac/collections/urn:eop:VITO:ESA_WorldCover_10m_2020_AWS_V1",
        "source": {
            "name": "ESA WorldCover",
            "author": "European Space Agency (ESA)",
            "version": "v100",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        },
    },
    "esa_worldcover_2021": {
        "adapter": ESAWorldCover(),
        "url": "https://services.terrascope.be/stac/collections/urn:eop:VITO:ESA_WorldCover_10m_2021_AWS_V2",
        "source": {
            "name": "ESA WorldCover",
            "author": "European Space Agency (ESA)",
            "version": "v200",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        },
    },
    "hydrolakes": {
        "adapter": HydroLakes(
            folder="hydrolakes",
            local_version=1,
            filename="hydrolakes.parquet",
            cache="global",
        ),
        "url": "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10.gdb.zip",
        "source": {
            "name": "HydroLAKES",
            "author": "Arjen Haag",
            "version": "v10.0",
            "license": "Creative Commons Attribution (CC-BY) 4.0 International License",
        },
    },
    "GADM_level0": {
        "adapter": GADM(
            level=0,
            folder="gadm_level0",
            local_version=1,
            filename="gadm_level0.parquet",
            cache="global",
        ),
        "url": "https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip",
        "source": {
            "name": "GADM",
            "author": "GADM",
            "version": "4.1",
            "license": "https://gadm.org/license.html",
        },
    },
    "GADM_level1": {
        "adapter": GADM(
            level=1,
            folder="gadm_level1",
            local_version=1,
            filename="gadm_level1.parquet",
            cache="global",
        ),
        "url": "https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip",
        "source": {
            "name": "GADM",
            "author": "GADM",
            "version": "4.1",
            "license": "https://gadm.org/license.html",
        },
    },
    "merit_hydro_dir": {
        "adapter": MeritHydroDir(
            folder="merit_hydro_dir",
            local_version=1,
            filename="merit_hydro_dir.zarr",
            cache="local",
        ),
        "url": "https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/distribute/v1.0",
        "source": {
            "name": "MERIT Hydro",
            "author": "Yamazaki et al.",
            "version": "2019",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0) or Open Database License (ODbL 1.0)",
        },
    },
    "merit_hydro_elv": {
        "adapter": MeritHydroElv(
            folder="merit_hydro_elv",
            local_version=1,
            filename="merit_hydro_elv.zarr",
            cache="local",
        ),
        "url": "https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/distribute/v1.0",
        "source": {
            "name": "MERIT Hydro",
            "author": "Yamazaki et al.",
            "version": "2019",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0) or Open Database License (ODbL 1.0)",
        },
    },
    "merit_basins_catchments": {
        "adapter": MeritBasinsCatchments(
            folder="merit_basins_catchments",
            local_version=1,
            filename="merit_basins_catchments.parquet",
            cache="global",
        ),
        "url": "https://drive.google.com/uc?export=download&id={FILE_ID}",
        "source": {
            "name": "MERIT Basins",
            "author": "Lin et al.",
            "version": "v0.7",
            "license": "CC BY-NC-SA 4.0",
        },
    },
    "merit_basins_rivers": {
        "adapter": MeritBasinsRivers(
            folder="merit_basins_rivers",
            local_version=1,
            filename="merit_basins_rivers.parquet",
            cache="global",
        ),
        "url": "https://drive.google.com/uc?export=download&id={FILE_ID}",
        "source": {
            "name": "MERIT Basins",
            "author": "Lin et al.",
            "version": "v0.7",
            "license": "CC BY-NC-SA 4.0",
        },
    },
    "merit_sword": {
        "adapter": MeritSword(
            folder="merit_sword",
            local_version=1,
            filename="merit_sword.zarr",
            cache="global",
        ),
        "url": "https://zenodo.org/records/14675925/files/ms_translate.zip",
        "source": {
            "name": "MERIT-SWORD",
            "version": "v0.4",
            "license": "Creative Commons Attribution Non Commercial Share Alike 4.0 International",
            "url": "doi.org/10.5281/zenodo.14675925",
        },
    },
    "sword": {
        "adapter": Sword(
            folder="sword",
            local_version=1,
            filename="sword.gpkg",
            cache="global",
        ),
        "url": "https://zenodo.org/records/10013982/files/SWORD_v16_gpkg.zip",
        "source": {
            "name": "SWORD",
            "version": "v16",
            "license": "Creative Commons Attribution 4.0 International",
            "url": "doi.org/10.5281/zenodo.14727521",
        },
    },
}


class NewDataCatalog:
    """The GEB data catalog for accessing predefined datasets."""

    def __init__(self) -> None:
        """Initialize the data catalog with predefined entries."""
        self.catalog = data_catalog

    def fetch(self, name: str, *args: Any, **kwargs: Any) -> Adapter:
        """Get a data catalog entry by name.

        Args:
            name: The name of the data entry to retrieve.
            *args: Additional positional arguments to pass to the fetcher.
            **kwargs: Additional keyword arguments to pass to the fetcher.

        Returns:
            The data catalog entry as a dictionary.
        """
        return self.catalog[name]["adapter"].fetch(
            url=self.catalog[name]["url"],
            *args,
            **kwargs,
        )

    def size_on_disk(
        self, name: str | None = None, format: str | None = None
    ) -> int | str:
        """Calculate the total size on disk for specified data entries.

        Args:
            name: Name of the data entry to check. If None, calculates for all entries.
            format: Optional unit to format the size ('KB', 'MB', 'GB', 'TB'). If None, returns size in bytes as int.

        Returns:
            Total size in bytes as int if format is None, otherwise a formatted string with the specified unit.

        Raises:
            ValueError: If format is not one of 'KB', 'MB', 'GB', 'TB'.
        """
        if name is None:
            names: list[str] = list(self.catalog.keys())
        else:
            names: list[str] = [name]

        total_size: int = 0
        for name in names:
            path = self.catalog[name]["adapter"].processor(
                url=self.catalog[name]["url"],
            )
            total_size += path.stat().st_size

        if format is None:
            return total_size

        # Validate format
        valid_formats = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
        if format not in valid_formats:
            raise ValueError(
                f"Invalid format '{format}'. Must be one of {list(valid_formats.keys())}."
            )

        # Convert and format
        divisor = valid_formats[format]
        formatted_size = total_size / divisor
        return f"{formatted_size:.2f} {format}"
