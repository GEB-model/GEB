"""Tests for the flood protection standards of river reaches."""

import pandas as pd
import pytest

from geb.workflows.flood_protection import (
    NATURAL_PROTECTION_STANDARD_YEARS,
    classify_flood_protection_standards,
)


def test_classify_flood_protection_standards() -> None:
    """Built-up reaches of large enough rivers get FLOPROS; large rivers can be protected throughout."""
    reaches = pd.Index([1, 2, 3, 4], name="COMID")
    flopros = pd.Series([100.0, 100.0, 200.0, 50.0], index=reaches)
    # 1: built up, large; 2: built up, small; 3: rural, very large; 4: rural, large
    density = pd.Series([800.0, 800.0, 10.0, 10.0], index=reaches)
    upstream_area = pd.Series([1000.0, 100.0, 40000.0, 1000.0], index=reaches)

    standards = classify_flood_protection_standards(
        flopros_standard_years=flopros,
        building_density_per_km2=density,
        upstream_area_km2=upstream_area,
        minimum_protected_upstream_area_km2=250.0,
        building_density_threshold_per_km2=500.0,
        continuous_protection_upstream_area_km2=None,
    )
    natural = NATURAL_PROTECTION_STANDARD_YEARS
    assert standards.tolist() == [100.0, natural, natural, natural]

    continuous = classify_flood_protection_standards(
        flopros_standard_years=flopros,
        building_density_per_km2=density,
        upstream_area_km2=upstream_area,
        minimum_protected_upstream_area_km2=250.0,
        building_density_threshold_per_km2=500.0,
        continuous_protection_upstream_area_km2=30000.0,
    )
    assert continuous.tolist() == [100.0, natural, 200.0, natural]


def test_classify_flood_protection_standards_rejects_missing_flopros() -> None:
    """A reach without a FLOPROS standard is an error, not a default."""
    reaches = pd.Index([1, 2])
    with pytest.raises(ValueError, match="FLOPROS"):
        classify_flood_protection_standards(
            flopros_standard_years=pd.Series([100.0, float("nan")], index=reaches),
            building_density_per_km2=pd.Series([800.0, 800.0], index=reaches),
            upstream_area_km2=pd.Series([1000.0, 1000.0], index=reaches),
            minimum_protected_upstream_area_km2=250.0,
            building_density_threshold_per_km2=500.0,
            continuous_protection_upstream_area_km2=None,
        )
