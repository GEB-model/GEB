"""Tests for the forcing module functions."""

from math import isclose
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from geb.build.modules.forcing import Forcing
from geb.forcing import get_pressure_correction_factor

from ...testconfig import output_folder


def test_get_pressure_correction_factor() -> None:
    """Test the pressure correction factor function.

    Pressure decreases with elevation, following the barometric formula.
    This test checks that the function returns reasonable values.
    """
    elevation: np.ndarray = np.linspace(0, 8848.86, 1000).astype(
        np.float32
    )  # From sea level to Everest
    g: float = 9.80665
    Mo: float = 0.0289644
    lapse_rate: float = -0.0065

    pressure_at_sea_level: float = 101325  # Pa
    pressure: np.ndarray = pressure_at_sea_level * get_pressure_correction_factor(
        elevation, g, Mo, lapse_rate
    )
    assert isclose(pressure[-1], 33700, abs_tol=3000)

    fig, ax = plt.subplots()

    ax.plot(pressure, elevation)
    ax.set_xlabel("Pressure (Pa)")
    ax.set_ylabel("Elevation (m)")
    ax.set_ylim(0, 9000)
    ax.set_xlim(0, 105000)

    plt.savefig(output_folder / "pressure_correction_factor.png")


def test_set_forcing_variable_source_attribute() -> None:
    """Test that _set_forcing_variable requires source and sets or omits attrs['source']."""
    mock_forcing: MagicMock = MagicMock()
    mock_forcing.set_other.side_effect = lambda da, name, **kwargs: da
    mock_forcing._get_forcing_keep_mask.return_value = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    )

    times: pd.DatetimeIndex = pd.date_range(
        "2000-01-01 01:00:00", periods=24, freq="1h"
    )
    data: xr.DataArray = xr.DataArray(
        np.ones((24, 2, 2), dtype=np.float32),
        dims=("time", "y", "x"),
        coords={"time": times, "y": [1.0, 0.0], "x": [0.0, 1.0]},
    )

    # 1. Source provided as string
    res: xr.DataArray = Forcing._set_forcing_variable(
        mock_forcing,
        data.copy(),
        name="climate/test",
        attrs={"units": "m", "_FillValue": np.nan},
        min_value=0.0,
        max_value=10.0,
        precision=0.1,
        offset=0.0,
        source="ERA5-Land",
    )
    assert res.attrs.get("source") == "ERA5-Land"

    # 2. Source provided as None
    res_none: xr.DataArray = Forcing._set_forcing_variable(
        mock_forcing,
        data.copy(),
        name="climate/test_none",
        attrs={"units": "m", "_FillValue": np.nan},
        min_value=0.0,
        max_value=10.0,
        precision=0.1,
        offset=0.0,
        source=None,
    )
    assert "source" not in res_none.attrs

    # 3. Source is None even if input da has source in attrs
    data_with_source: xr.DataArray = data.copy()
    data_with_source.attrs["source"] = "PreExistingSource"
    res_none_with_da_attr: xr.DataArray = Forcing._set_forcing_variable(
        mock_forcing,
        data_with_source,
        name="climate/test_none_override",
        attrs={"units": "m", "_FillValue": np.nan},
        min_value=0.0,
        max_value=10.0,
        precision=0.1,
        offset=0.0,
        source=None,
    )
    assert "source" not in res_none_with_da_attr.attrs


def test_set_spei_source_none() -> None:
    """Test that set_SPEI defaults source to None and does not set source attribute."""
    mock_forcing: MagicMock = MagicMock()
    mock_forcing.set_other.side_effect = lambda da, name, **kwargs: da
    mock_forcing._get_forcing_keep_mask.return_value = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    )

    times: pd.DatetimeIndex = pd.date_range("2000-01-01", periods=12, freq="MS")
    data: xr.DataArray = xr.DataArray(
        np.zeros((12, 2, 2), dtype=np.float32),
        dims=("time", "y", "x"),
        coords={"time": times, "y": [1.0, 0.0], "x": [0.0, 1.0]},
    )

    res: xr.DataArray = Forcing.set_SPEI(mock_forcing, data)
    assert "source" not in res.attrs


def test_set_forcing_setters_require_source() -> None:
    """Test that forcing setters like set_pr_kg_per_m2_per_s pass source through."""
    mock_forcing: MagicMock = MagicMock()
    mock_forcing.set_other.side_effect = lambda da, name, **kwargs: da
    mock_forcing._set_forcing_variable.side_effect = lambda *args, **kwargs: (
        Forcing._set_forcing_variable(mock_forcing, *args, **kwargs)
    )
    mock_forcing._get_forcing_keep_mask.return_value = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
    )

    times: pd.DatetimeIndex = pd.date_range(
        "2000-01-01 01:00:00", periods=24, freq="1h"
    )
    data: xr.DataArray = xr.DataArray(
        np.ones((24, 2, 2), dtype=np.float32),
        dims=("time", "y", "x"),
        coords={"time": times, "y": [1.0, 0.0], "x": [0.0, 1.0]},
    )

    res: xr.DataArray = Forcing.set_pr_kg_per_m2_per_s(
        mock_forcing, data, source="MSWEP"
    )
    assert res.attrs.get("source") == "MSWEP"

    res_tas: xr.DataArray = Forcing.set_tas_2m_K(mock_forcing, data, source="ERA5-Land")
    assert res_tas.attrs.get("source") == "ERA5-Land"
