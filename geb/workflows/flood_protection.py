"""Flood protection standards of river reaches, shared by routing, agents and evaluation."""

from pathlib import Path
from typing import Any

import pandas as pd

# Natural reaches (rural valleys, headwaters) have no levees: they hold the bankfull
# discharge, which is about the 2-year flood.
NATURAL_PROTECTION_STANDARD_YEARS: float = 2.0


def classify_flood_protection_standards(
    flopros_standard_years: pd.Series,
    building_density_per_km2: pd.Series,
    upstream_area_km2: pd.Series,
    minimum_protected_upstream_area_km2: float,
    building_density_threshold_per_km2: float,
    continuous_protection_upstream_area_km2: float | None,
) -> pd.Series:
    """Flood protection standard of each river reach.

    A reach is protected by the FLOPROS standard of its region when people live in its
    floodplain (building density at or above the threshold) and it drains at least the
    minimum protected upstream area. Larger rivers can be protected along their whole
    length: reaches draining at least the continuous protection area are protected
    regardless of the building density (e.g. the dikes along the main Rhine). All other
    reaches are natural.

    Args:
        flopros_standard_years: FLOPROS protection standard per reach (years).
        building_density_per_km2: Buildings per km2 of floodplain per reach.
        upstream_area_km2: Upstream area per reach (km2).
        minimum_protected_upstream_area_km2: Reaches with a smaller upstream area are
            natural (km2).
        building_density_threshold_per_km2: Reaches with a lower floodplain building
            density are natural, unless continuously protected (buildings/km2).
        continuous_protection_upstream_area_km2: Reaches with at least this upstream
            area are protected regardless of the building density (km2); None for no
            continuous protection.

    Returns:
        Protection standard per reach (years), with the index of the inputs.

    Raises:
        ValueError: If the inputs do not cover the same reaches or a FLOPROS standard
            is missing.
    """
    if not (
        flopros_standard_years.index.equals(building_density_per_km2.index)
        and flopros_standard_years.index.equals(upstream_area_km2.index)
    ):
        raise ValueError("All inputs must be indexed by the same river reaches.")
    if flopros_standard_years.isna().any():
        raise ValueError("Every river reach needs a FLOPROS protection standard.")

    is_built_up_or_large_river: pd.Series = (
        building_density_per_km2 >= building_density_threshold_per_km2
    )
    if continuous_protection_upstream_area_km2 is not None:
        is_built_up_or_large_river |= (
            upstream_area_km2 >= continuous_protection_upstream_area_km2
        )
    is_protected: pd.Series = (
        upstream_area_km2 >= minimum_protected_upstream_area_km2
    ) & is_built_up_or_large_river
    return flopros_standard_years.where(
        is_protected, NATURAL_PROTECTION_STANDARD_YEARS
    ).rename("flood_protection_standard")


def read_flood_protection_standards(
    table_path: Path, flood_protection_config: dict[str, Any]
) -> pd.Series:
    """Read the reach data written by the build and classify the protection standards.

    Args:
        table_path: Flood protection standards table of the build
            (``setup_flood_protection_standards``), indexed by river reach (COMID).
        flood_protection_config: The ``hydrology.routing.flood_protection`` configuration.

    Returns:
        Protection standard per reach (years), indexed by COMID.
    """
    table: pd.DataFrame = pd.read_parquet(table_path)
    return classify_flood_protection_standards(
        flopros_standard_years=table["flopros_standard"],
        building_density_per_km2=table["building_density_per_km2"],
        upstream_area_km2=table["upstream_area_km2"],
        minimum_protected_upstream_area_km2=flood_protection_config[
            "minimum_protected_upstream_area_km2"
        ],
        building_density_threshold_per_km2=flood_protection_config[
            "building_density_threshold_per_km2"
        ],
        continuous_protection_upstream_area_km2=flood_protection_config[
            "continuous_protection_upstream_area_km2"
        ],
    )
