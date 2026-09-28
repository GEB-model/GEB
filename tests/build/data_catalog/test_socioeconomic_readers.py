"""Regression tests for country identifiers in socioeconomic data readers."""

from pathlib import Path

import pandas as pd
import pytest

from geb.build.data_catalog.aquastat import AQUASTAT
from geb.build.data_catalog.superwell import GCAMElectricityRates


@pytest.mark.parametrize("padding", ["", " \t"])
def test_electricity_country_mapping(tmp_path: Path, padding: str) -> None:
    """Retain the first row and map clean and padded country names.

    Args:
        tmp_path: Temporary directory for the source CSV.
        padding: Whitespace around country names.
    """
    adapter: GCAMElectricityRates = GCAMElectricityRates(
        folder=tmp_path, filename="rates.csv", local_version=1, cache="global"
    )
    adapter.path.write_text(
        f"{padding}Afghanistan{padding},0.092\n"
        f"Albania,0.115\n{padding}United Kingdom{padding},0.145\n",
        encoding="utf-8-sig",
    )
    assert adapter.read() == {"AFG": 0.092, "ALB": 0.115, "GBR": 0.145}


def test_electricity_unknown_country(tmp_path: Path) -> None:
    """Report unknown countries instead of returning an invalid dictionary key.

    Args:
        tmp_path: Temporary directory for the source CSV.
    """
    adapter: GCAMElectricityRates = GCAMElectricityRates(
        folder=tmp_path, filename="rates.csv", local_version=1, cache="global"
    )
    adapter.path.write_text("Unknown country,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown country"):
        adapter.read()


@pytest.mark.parametrize("padding", ["", " \t"])
def test_aquastat_country_mapping(tmp_path: Path, padding: str) -> None:
    """Map padded names while preserving indicator filtering and values.

    Args:
        tmp_path: Temporary directory for the cached parquet file.
        padding: Whitespace around country names.
    """
    adapter: AQUASTAT = AQUASTAT(
        folder=tmp_path, filename="water.parquet", local_version=1, cache="global"
    )
    data: pd.DataFrame = pd.DataFrame(
        {
            "AREA": [f"{padding}Albania{padding}"] * 2,
            "aquastatElement.1": ["selected", "other"],
            "timePointYears": [2000, 2001],
            "Value": [12.0, 99.0],
        }
    )
    column: str
    for column in [
        "[flagObservationStatus] flagObservationStatus - flagObservationStatus",
        "[flagMethod] flagMethod - flagMethod",
        "aquastatElement",
        "REF_AREA",
        "timePointYears.1",
    ]:
        data[column] = "unused"
    data.to_parquet(adapter.path, index=False)
    result: pd.DataFrame = adapter.read(indicator="selected")
    assert result.index.tolist() == ["ALB"]
    assert result["Year"].tolist() == [2000]
    assert result["Value"].tolist() == [12.0]
