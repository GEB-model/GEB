"""Data catalog for predefined datasets in GEB."""

from typing import Any

from .base import Adapter
from .destination_earth import DestinationEarth
from .ecmwf import ECMWFForecasts
from .esa_worldcover import ESAWorldCover
from .fabdem import Fabdem as Fabdem
from .fao import GMIA
from .gadm import GADM
from .gebco import GEBCO
from .global_data_lab import GlobalDataLabShapefile
from .globgm import GlobGM, GlobGMDEM
from .grdc import GRDC
from .hydrolakes import HydroLakes
from .isimip import ISIMIPCO2
from .lowder import Lowder
from .merit_basins import MeritBasinsCatchments, MeritBasinsRivers
from .merit_hydro import MeritHydroDir, MeritHydroElv
from .merit_sword import MeritSword
from .open_building_map import OpenBuildingMap
from .open_street_map import OpenStreetMap
from .soilgrids import SoilGrids
from .sword import Sword
from .why_map import WhyMap
from .world_bank import WorldBankData

data_catalog: dict[str, dict[str, Any]] = {
    "isimip_co2": {
        "adapter": ISIMIPCO2(),
        "url": "https://files.isimip.org",
        "source": {
            "name": "ISIMIP CO2",
            "author": "Matthias Büchner, Christopher P.O. Reyer",
            "url": "https://data.isimip.org",
            "license": "CC BY-SA 4.0",
            "paper_doi": "10.1007/s10584-011-0156-z",
        },
    },
    "era5": {
        "adapter": DestinationEarth(),
        "url": "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr",
        "source": {
            "name": "ERA5",
            "author": "ECMWF",
        },
    },
    "ecmwf_forecasts": {
        "adapter": ECMWFForecasts(
            folder="ecmwf_forecasts",
            filename="ecmwf_{forecast_date}_{forecast_model}_{forecast_resolution}_{forecast_horizon}h_{forecast_timestep_hours}h.grb",
            local_version=1,
            cache="local",
        ),
        "url": None,
        "source": {
            "name": "ECMWF Forecasts",
            "author": "ECMWF",
            "url": "https://www.ecmwf.int/en/forecasts/access-forecasts/access-archive-datasets",
            "license": "https://www.ecmwf.int/en/forecasts/accessing-forecasts/licences-available",
        },
    },
    "soilgrids": {
        "adapter": SoilGrids(),
        "url": "https://files.isric.org/soilgrids/latest/data/{variable}/{variable}_{depth}_mean.vrt",
        "source": {
            "name": "SoilGrids",
            "author": "ISRIC - World Soil Information",
            "license": "CC BY 4.0",
            "url": "https://soilgrids.org",
            "version": "2020",
            "paper_doi": "10.5194/soil-2020-65",
        },
    },
    "why_map": {
        "adapter": WhyMap(
            folder="why_map",
            local_version=1,
            filename="whymap.parquet",
            cache="global",
        ),
        "url": "https://download.bgr.de/bgr/grundwasser/whymap/shp/WHYMAP_GWR_v1.zip",
        "source": {
            "name": "WHYMAP",
            "author": "BGR (Federal Institute for Geosciences and Natural Resources)",
            "license": "CC BY-SA",
            "url": "https://www.whymap.org/whymap/EN/Maps_Data/Gwr/gwr_node_en.html",
        },
    },
    "lowder_farm_size_distribution": {
        "adapter": Lowder(
            folder="lowder_farm_size_distribution",
            local_version=1,
            filename="lowder_farm_size_distribution.xlsx",
            cache="global",
        ),
        "url": "https://ars.els-cdn.com/content/image/1-s2.0-S0305750X15002703-mmc1.xlsx",
        "source": {
            "name": "The Number, Size, and Distribution of Farms, Smallholder Farms, and Family Farms Worldwide",
            "author": "Lowder et al. (2016)",
            "paper_doi": "10.1016/j.worlddev.2015.10.041",
            "license": "CC BY-NC-ND 4.0",
            "url": "https://doi.org/10.1016/j.worlddev.2015.10.041",
        },
    },
    "gebco": {
        "adapter": GEBCO(
            folder="gebco",
            local_version=1,
            filename="gebco.zarr",
            cache="global",
        ),
        "url": "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2024/geotiff/",
        "source": {
            "name": "GEBCO",
            "author": "Weatherall et al (2020)",
            "license": "https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use",
            "url": "https://www.gebco.net/",
            "paper_doi": "10.5285/a29c5465-b138-234d-e053-6c86abc040b9",
        },
    },
    "GRDC": {
        "adapter": GRDC(
            folder="grdc",
            local_version=1,
            filename="GRDC.zip",
            cache="global",
        ),
        "url": "https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/Stations",
        "source": {
            "name": "Global Runoff Data Centre",
            "author": "Global Runoff Data Centre",
            "license": "https://grdc.bafg.de/downloads/policy_guidelines.pdf",
        },
    },
    "global_irrigation_area_groundwater": {
        "adapter": GMIA(
            folder="global_irrigation_area_groundwater",
            local_version=1,
            filename="global_irrigation_area_groundwater.asc",
            cache="global",
        ),
        "url": "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/GIS%2Fgmia_v5_aeigw_pct_aei_asc.zip?alt=media",
        "source": {
            "name": "Global Map of Irrigation Areas",
            "author": "Siebert et al. (2010)",
            "paper_doi": "10.5194/hess-14-1863-2010",
            "license": "CC BY 4.0",
            "url": "https://www.fao.org/aquastat/en/geospatial-information/global-maps-irrigated-areas/latest-version/",
        },
    },
    "global_irrigation_area_surface_water": {
        "adapter": GMIA(
            folder="global_irrigation_area_surface_water",
            local_version=1,
            filename="global_irrigation_area_surface_water.asc",
            cache="global",
        ),
        "url": "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/GIS%2Fgmia_v5_aeisw_pct_aei_asc.zip?alt=media",
        "source": {
            "name": "Global Map of Irrigation Areas",
            "author": "Siebert et al. (2010)",
            "paper_doi": "10.5194/hess-14-1863-2010",
            "license": "CC BY 4.0",
            "url": "https://www.fao.org/aquastat/en/geospatial-information/global-maps-irrigated-areas/latest-version/",
        },
    },
    "hydraulic_conductivity_globgm": {
        "adapter": GlobGM(
            folder="hydraulic_conductivity_globgm",
            local_version=1,
            filename="hydraulic_conductivity_globgm.nc",
            cache="global",
        ),
        "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/k_conductivity_aquifer_filled_30sec.nc",
        "source": {
            "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/",
            "author": "Verkaik et al. (2024)",
            "paper_doi": "10.5194/gmd-17-275-2024",
            "license": "CC BY 4.0",
        },
    },
    "specific_yield_aquifer_globgm": {
        "adapter": GlobGM(
            folder="specific_yield_aquifer_globgm",
            local_version=1,
            filename="specific_yield_aquifer_globgm.nc",
            cache="global",
        ),
        "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/specific_yield_aquifer_filled_30sec.nc",
        "source": {
            "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/",
            "author": "Verkaik et al. (2024)",
            "paper_doi": "10.5194/gmd-17-275-2024",
            "license": "CC BY 4.0",
        },
    },
    "water_table_depth_globgm": {
        "adapter": GlobGM(
            folder="water_table_depth_globgm",
            local_version=1,
            filename="water_table_depth_globgm.tif",
            cache="global",
        ),
        "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/output/version_1.0/steady-state/globgm-wtd-ss.tif",
        "source": {
            "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/output/version_1.0/steady-state/",
            "author": "Verkaik et al. (2024)",
            "paper_doi": "10.5194/gmd-17-275-2024",
            "license": "CC BY 4.0",
        },
    },
    "head_lower_layer_globgm": {
        "adapter": GlobGM(
            folder="head_lower_layer_globgm",
            local_version=1,
            filename="head_lower_layer_globgm.tif",
            cache="global",
        ),
        "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/output/version_1.0/steady-state/globgm-heads-lower-layer-ss.tif",
        "source": {
            "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/output/version_1.0/steady-state/",
            "author": "Verkaik et al. (2024)",
            "paper_doi": "10.5194/gmd-17-275-2024",
            "license": "CC BY 4.0",
        },
    },
    "head_upper_layer_globgm": {
        "adapter": GlobGM(
            folder="head_upper_layer_globgm",
            local_version=1,
            filename="head_upper_layer_globgm.tif",
            cache="global",
        ),
        "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/output/version_1.0/steady-state/globgm-heads-upper-layer-ss.tif",
        "source": {
            "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/output/version_1.0/steady-state/",
            "author": "Verkaik et al. (2024)",
            "paper_doi": "10.5194/gmd-17-275-2024",
            "license": "CC BY 4.0",
        },
    },
    "recession_coefficient_globgm": {
        "adapter": GlobGM(
            folder="recession_coefficient_globgm",
            local_version=1,
            filename="recession_coefficient_globgm.nc",
            cache="global",
        ),
        "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/recession_coefficient_30sec.nc",
        "source": {
            "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/",
            "author": "Verkaik et al. (2024)",
            "paper_doi": "10.5194/gmd-17-275-2024",
            "license": "CC BY 4.0",
        },
    },
    "thickness_confining_layer_globgm": {
        "adapter": GlobGM(
            folder="thickness_confining_layer_globgm",
            local_version=1,
            filename="thickness_confining_layer_globgm.nc",
            cache="global",
        ),
        "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/confining_layer_thickness_version_2016_remapbil_to_30sec.nc",
        "source": {
            "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/",
            "author": "Verkaik et al. (2024)",
            "paper_doi": "10.5194/gmd-17-275-2024",
            "license": "CC BY 4.0",
        },
    },
    "total_groundwater_thickness_globgm": {
        "adapter": GlobGM(
            folder="total_groundwater_thickness_globgm",
            local_version=1,
            filename="total_groundwater_thickness_globgm.nc",
            cache="global",
        ),
        "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/thickness_05min_remapbil_to_30sec_filled_with_pcr_correct_lat.nc",
        "source": {
            "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/",
            "author": "Verkaik et al. (2024)",
            "paper_doi": "10.5194/gmd-17-275-2024",
            "license": "CC BY 4.0",
        },
    },
    "dem_globgm": {
        "adapter": GlobGMDEM(
            folder="dem_globgm",
            local_version=1,
            filename="dem_globgm.nc",
            cache="global",
        ),
        "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/topography_parameters_30sec_february_2021_global_covered_with_zero.nc",
        "source": {
            "url": "https://geo.public.data.uu.nl/vault-globgm/research-globgm%5B1669042611%5D/original/input/version_1.0/",
            "author": "Verkaik et al. (2024)",
            "paper_doi": "10.5194/gmd-17-275-2024",
            "license": "CC BY 4.0",
        },
    },
    "GDL_regions_v4": {
        "adapter": GlobalDataLabShapefile(
            folder="global_data_lab",
            local_version=1,
            filename="gdl_regions_v4.parquet",
            cache="global",
        ),
        "url": "https://globaldatalab.org/mygdl/downloads/shapefiles/",
        "source": {
            "name": "Global Data Lab",
            "author": "Radboud University",
            "license": "https://globaldatalab.org/termsofuse/",
            "url": "https://globaldatalab.org/mygdl/downloads/shapefiles/",
        },
    },
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
            "license": "CC BY 4.0",
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
            "license": "CC BY 4.0",
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
            "license": "CC BY 4.0",
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
            "license": "CC BY 4.0",
        },
    },
    "esa_worldcover_2021": {
        "adapter": ESAWorldCover(),
        "url": "https://services.terrascope.be/stac/collections/urn:eop:VITO:ESA_WorldCover_10m_2021_AWS_V2",
        "source": {
            "name": "ESA WorldCover",
            "author": "European Space Agency (ESA)",
            "version": "v200",
            "license": "CC BY 4.0",
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
            "license": "CC BY 4.0",
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
            "license": "CC BY 4.0 or ODbL 1.0",
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
            "license": "CC BY 4.0 or ODbL 1.0",
        },
    },
    "fabdem": {
        "adapter": Fabdem(
            folder="fabdem",
            local_version=1,
            filename="fabdem.zarr",
            cache="local",
        ),
        "url": "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn",
        "source": {
            "name": "FABDEM",
            "author": "Hawker et al. (2022)",
            "version": "1-2",
            "license": "CC BY-NC-SA 4.0",
            "url": "https://data.bris.ac.uk/data/dataset/25wfy0f9ukoge2gs7a5mqpq2j7",
            "paper_doi": "10.1088/1748-9326/ac4d4f",
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
            "license": "CC BY-NC-SA 4.0",
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
            "license": "CC BY 4.0",
            "url": "doi.org/10.5281/zenodo.14727521",
        },
    },
    "open_building_map": {
        "adapter": OpenBuildingMap(
            folder="open_building_map",
            local_version=1,
            filename="open_building_map.parquet",
            cache="local",
        ),
        "url": "https://datapub.gfz.de/download/10.5880.GFZ.LKUT.2025.002-Caweb/2025-002_Oostwegel-et-al_data/",
        "source": {
            "name": "OpenBuildingMap",
            "author": "Oostwegel et al. (2025)",
            "version": "1",
            "license": "CC BY-NC-SA 4.0",
            "url": "https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=45829b80-e892-11ef-914a-f12b0080820d",
            "paper_doi": "https://doi.org/10.5880/GFZ.LKUT.2025.002",
        },
    },
    "open_street_map": {
        "adapter": OpenStreetMap(),
        "url": "https://osm.download.movisda.io",
        "source": {
            "name": "OpenStreetMap",
            "author": "OpenStreetMap contributors",
            "license": "ODbL 1.0",
            "url": "https://www.openstreetmap.org/copyright",
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

    def size(self, name: str | None = None, format: str | None = "GB") -> int | str:
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
            adapter = self.catalog[name]["adapter"]
            if adapter.cache is not None and adapter.path.exists():
                total_size += adapter.path.stat().st_size

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

    def print_licenses(self, name: str | None = None) -> None:
        """Print the licence information for specified data entries.

        Args:
            name: Name of the data entry to check. If None, prints for all entries.
        """
        if name is None:
            names: list[str] = list(self.catalog.keys())
        else:
            names: list[str] = [name]

        # Collect license data for table formatting
        license_data = []
        for name in names:
            source = self.catalog[name].get("source", {})
            name = source.get("name", name)
            license_info = source.get("license", "N/A")
            license_data.append((name, license_info))

        # Print table header
        print(f"{'Name':<50} {'License'}")
        print("-" * 80)

        # Print each row
        for name, license_info in license_data:
            print(f"{name:<50} {license_info}")

    def fetch_global(self) -> None:
        """Fetch all data entries with global cache setting."""
        for name, entry in self.catalog.items():
            adapter = entry["adapter"]
            if adapter.cache == "global":
                if not adapter.is_ready:
                    print(f"Fetching {name}...")
                    adapter.fetch(url=entry["url"])
                    print(f"Fetched {name} to {adapter.path}")

        print("All global data fetched, catalog size:", self.size(format="GB"))
