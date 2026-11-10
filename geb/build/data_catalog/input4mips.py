"""input4MIPS CO2 data adapter."""

from __future__ import annotations

from typing import Any

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

from .base import Adapter


class INPUT4MIPS(Adapter):
    """Adapter for input4MIPS CO2 data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the input4MIPS CO2 data adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> INPUT4MIPS:
        """Set the URL for the INPUT4MIPS CO2 data source.

        Args:
            url: The URL of the INPUT4MIPS CO2 data source.

        Returns:
            The current instance of the INPUT4MIPS adapter.
        """
        self.url = url
        return self

    def read(self, scenario: str, **kwargs: Any) -> pd.DataFrame:
        """Read the INPUT4MIPS CO2 data for a specific scenario.

        Args:
            scenario: The scenario to read (e.g., "ssp126", "ssp370", "ssp585").
            **kwargs: Additional keyword arguments (not used).

        Returns:
            A pandas DataFrame containing the CO2 concentration data with years as index and a column 'co2_ppm'.

        Raises:
            ValueError: If the scenario is invalid.
        """
        model = {
            "ssp119": "IMAGE",
            "ssp126": "IMAGE",
            "ssp434": "GCAM4",
            "ssp245": "MESSAGE-GLOBIOM",
            "ssp460": "GCAM4",
            "ssp370": "AIM",
            "ssp585": "REMIND-MAGPIE",
        }

        if scenario not in model:
            raise ValueError(
                "Invalid scenario. Choose from 'ssp119', 'ssp126', 'ssp434', 'ssp245', 'ssp460', 'ssp370', or 'ssp585'."
            )

        def get_file_url(scenario: str) -> str:
            if scenario == "historical":
                base_url = f"https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/UoM/UoM-CMIP-1-2-0/atmos/yr/mole-fraction-of-carbon-dioxide-in-air/gr1-GMNHSH/v20160830/"
                file_name = f"mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_CMIP_UoM-CMIP-1-2-0_gr1-GMNHSH_0000-2014.nc"
            else:
                base_url = f"https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/ScenarioMIP/UoM/UoM-{model[ssp]}-{ssp}-1-2-1/atmos/yr/mole_fraction_of_carbon_dioxide_in_air/gr1-GMNHSH/v20181127/"
                file_name = f"mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-{model[ssp]}-{ssp}-1-2-1_gr1-GMNHSH_2015-2500.nc"
            return base_url + file_name

        co2_historical = xr.open_dataset(
            fsspec.open(get_file_url("historical"), mode="rb").open(),
            engine="h5netcdf",
            decode_times=False,
        ).sel({"sector": 0, "bound": 0})
        co2_historical = (
            co2_historical.to_dataframe()
            .drop(columns=["time_bnds", "sector_bnds", "sector"])
            .rename(columns={"mole_fraction_of_carbon_dioxide_in_air": "co2_ppm"})
        )
        co2_historical.index = pd.Index(np.arange(0, 2015), name="year")

        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        co2_scenario = xr.open_dataset(
            fsspec.open(get_file_url(scenario), mode="rb").open(),
            engine="h5netcdf",
            decode_times=time_coder,
        ).sel({"sector": 0, "bound": 0})
        co2_scenario = (
            co2_scenario.to_dataframe()
            .drop(columns=["time_bnds", "sector_bnds", "sector"])
            .rename(columns={"mole_fraction_of_carbon_dioxide_in_air": "co2_ppm"})
        )
        co2_scenario.index = pd.Index(
            co2_scenario.index.map(lambda x: x.year), name="year"
        )
        co2_ppm = pd.concat([co2_historical, co2_scenario], axis="index").loc[1850:2100]

        return co2_ppm
