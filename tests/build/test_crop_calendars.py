"""Tests for the crop calendar functions."""

from datetime import date

import numpy as np
import pandas as pd

from geb.build.workflows.crop_calendars import (
    donate_and_receive_crop_prices,
    get_day_index,
    get_growing_season_length,
)


def test_growing_season_length() -> None:
    """Test growing season length calculation for various start and end days.

    Especially important are cases where the season wraps around the end of the year.
    """
    assert get_growing_season_length(0, 364) == 364
    assert get_growing_season_length(0, 0) == 365
    assert get_growing_season_length(0, 1) == 1
    assert get_growing_season_length(300, 1) == 66
    assert get_growing_season_length(2, 1) == 364


def test_day_of_year() -> None:
    """Test day index calculation for various dates."""
    assert get_day_index(date(2000, 1, 1)) == 0
    assert get_day_index(date(2000, 1, 2)) == 1
    assert get_day_index(date(2000, 2, 1)) == 31
    assert get_day_index(date(2000, 12, 31)) == 365  # Leap year
    assert get_day_index(date(2001, 12, 31)) == 364  # Non-leap year


def _crop_price_table(prices: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    """Build a donor table indexed by (ISO3, year) as in ``process_crop_data``.

    Args:
        prices: Mapping of ISO3 to a mapping of crop name to prices for 2010-2012 (USD/kg).

    Returns:
        DataFrame with one column per crop plus an 'ISO3' column.
    """
    tables: dict[str, pd.DataFrame] = {}
    for iso3, crops in prices.items():
        table = pd.DataFrame(crops, index=pd.Index([2010, 2011, 2012], name="year"))
        table["ISO3"] = iso3
        tables[iso3] = table
    return pd.concat(tables, names=["ISO3", "year"])


def test_donate_crop_prices_uses_trade_region_median() -> None:
    """Missing crops get the per-year median of the trade region; nothing comes from elsewhere."""
    nan = np.nan
    donor_data = _crop_price_table(
        {
            "GTM": {"maize": [0.2, 0.2, 0.2], "palm": [nan] * 3, "cocoa": [nan] * 3},
            "HND": {
                "maize": [0.3, 0.3, 0.3],
                "palm": [0.10, nan, 0.30],
                "cocoa": [nan] * 3,
            },
            "NIC": {
                "maize": [0.3, 0.3, 0.3],
                "palm": [0.20, 0.40, nan],
                "cocoa": [nan] * 3,
            },
            "SLV": {
                "maize": [0.3, 0.3, 0.3],
                "palm": [0.60, nan, nan],
                "cocoa": [nan] * 3,
            },
            # other trade region: its prices must not be used
            "FRA": {"maize": [0.5, 0.5, 0.5], "palm": [9.0] * 3, "cocoa": [9.0] * 3},
        }
    )
    trade_regions = {
        "GTM": "RCAM",
        "HND": "RCAM",
        "NIC": "RCAM",
        "SLV": "RCAM",
        "FRA": "EU",
    }
    recipients = pd.DataFrame({"region_id": [7], "ISO3": ["GTM"]})

    result = donate_and_receive_crop_prices(donor_data, recipients, trade_regions)

    # own data is kept; palm is the median over HND/NIC/SLV per year
    np.testing.assert_allclose(result.loc[7, "maize"].to_numpy(), [0.2, 0.2, 0.2])
    np.testing.assert_allclose(result.loc[7, "palm"].to_numpy(), [0.20, 0.40, 0.30])
    # no cocoa price anywhere in the trade region: stays missing (no fallback to FRA)
    assert "cocoa" not in result.columns or result.loc[7, "cocoa"].isna().all()
