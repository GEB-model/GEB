"""Land cover types used in the hydrology module."""

from typing import Literal

# All natural areas MUST be before the sealed and water areas
FOREST: Literal[0] = 0
GRASSLAND_LIKE: Literal[1] = 1
PADDY_IRRIGATED: Literal[2] = 2
NON_PADDY_IRRIGATED: Literal[3] = 3
SEALED: Literal[4] = 4
OPEN_WATER: Literal[5] = 5


ALL_LAND_COVER_TYPES: list[int] = [
    FOREST,
    GRASSLAND_LIKE,
    PADDY_IRRIGATED,
    NON_PADDY_IRRIGATED,
    SEALED,
    OPEN_WATER,
]
