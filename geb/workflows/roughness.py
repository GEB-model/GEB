"""Manning roughness of land cover, shared by the river routing and the flood model.

The flood model (SFINCS) reads the roughness of every land cover pixel from this table;
the river routing of the hydrological model uses the same values for the flow over the
floodplains, per land use type of the hydrological response units.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from geb.geb_types import ArrayFloat32
from geb.hydrology.landcovers import (
    FOREST,
    GRASSLAND_LIKE,
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)

# Manning roughness coefficient n (s/m^(1/3)) per ESA WorldCover class: (description, n)
ESA_WORLDCOVER_MANNINGS_N: dict[int, tuple[str, float]] = {
    10: ("Tree cover", 0.12),
    20: ("Shrubland", 0.05),
    30: ("Grasland", 0.034),
    40: ("Cropland", 0.037),
    50: ("Built-up", 0.1),
    60: ("Bare / sparse vegetation", 0.023),
    70: ("Snow and Ice", 0.01),
    80: ("Permanent water bodies", 0.02),
    90: ("Herbaceous wetland", 0.035),
    95: ("Mangroves", 0.07),
    100: ("Moss and lichen", 0.025),
    0: ("No data", 0.1),
}

# ESA WorldCover class that stands for each land use type of the hydrological model
LAND_USE_TYPE_TO_ESA_WORLDCOVER: dict[int, int] = {
    FOREST: 10,
    GRASSLAND_LIKE: 30,
    PADDY_IRRIGATED: 40,
    NON_PADDY_IRRIGATED: 40,
    SEALED: 50,
    OPEN_WATER: 80,
}
ESA_WORLDCOVER_CROPLAND: int = 40

# Roughness added for obstructions on a floodplain that a one-dimensional river model
# does not resolve: hedges, fences, road and rail embankments, bridges, buildings.
# Cowan (1956) and Arcement and Schneider (1989) add 0.020-0.030 to the roughness of
# the surface for "appreciable" obstructions (15-50 % of the cross-section). The flood
# model resolves such obstacles in its elevation grid and uses the surface roughness only.
FLOODPLAIN_OBSTRUCTION_MANNINGS_N: float = 0.025


def get_land_cover_mannings_roughness_table() -> pd.DataFrame:
    """Get the Manning roughness per ESA WorldCover class as a table.

    Returns:
        Table with the columns esa_worldcover (class), description, landuse (class)
        and N (Manning roughness coefficient, s/m^(1/3)).
    """
    return pd.DataFrame(
        data=[
            [land_cover, description, land_cover, mannings_n]
            for land_cover, (
                description,
                mannings_n,
            ) in ESA_WORLDCOVER_MANNINGS_N.items()
        ],
        columns=np.array(["esa_worldcover", "description", "landuse", "N"], dtype=str),
    )


def get_land_use_mannings_n(
    land_use_type: npt.NDArray[np.integer],
    is_cultivated: npt.NDArray[np.bool_],
) -> ArrayFloat32:
    """Get the Manning roughness of hydrological response units from their land use.

    Args:
        land_use_type: Land use type of each unit (see geb.hydrology.landcovers).
        is_cultivated: Whether the unit is a farm field; fields are cropland, also when
            they are rainfed (grassland-like land use type) or fallow.

    Returns:
        Manning roughness coefficient per unit (s/m^(1/3)).
    """
    mannings_n = np.empty(land_use_type.size, dtype=np.float32)
    for land_use, land_cover in LAND_USE_TYPE_TO_ESA_WORLDCOVER.items():
        mannings_n[land_use_type == land_use] = ESA_WORLDCOVER_MANNINGS_N[land_cover][1]
    mannings_n[is_cultivated] = ESA_WORLDCOVER_MANNINGS_N[ESA_WORLDCOVER_CROPLAND][1]
    return mannings_n


def get_floodplain_mannings_n(
    land_use_type: npt.NDArray[np.integer],
    is_cultivated: npt.NDArray[np.bool_],
) -> ArrayFloat32:
    """Get the Manning roughness of floodplain land in a one-dimensional river model.

    The roughness of the land cover plus the roughness of the obstructions that the
    river model does not resolve (see FLOODPLAIN_OBSTRUCTION_MANNINGS_N).

    Args:
        land_use_type: Land use type of each unit (see geb.hydrology.landcovers).
        is_cultivated: Whether the unit is a farm field.

    Returns:
        Manning roughness coefficient per unit (s/m^(1/3)).
    """
    return (
        get_land_use_mannings_n(land_use_type, is_cultivated)
        + np.float32(FLOODPLAIN_OBSTRUCTION_MANNINGS_N)
    ).astype(np.float32)
