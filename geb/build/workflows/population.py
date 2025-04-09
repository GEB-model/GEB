import gzip

import numpy as np
import pandas as pd
import rioxarray


def load_GLOPOP_S(data_catalog, GDL_region):
    # Load GLOPOP-S data. This is a binary file and has no proper loading in hydromt. So we use the data catalog to get the path and format the path with the regions and load it with NumPy
    GLOPOP_S_attribute_names = [
        "HID",
        "RELATE_HEAD",
        "INCOME",
        "WEALTH",
        "RURAL",
        "AGE",
        "GENDER",
        "EDUC",
        "HHTYPE",
        "HHSIZE_CAT",
        "AGRI_OWNERSHIP",
        "FLOOR",
        "WALL",
        "ROOF",
        "SOURCE",
        "GRID_CELL",
    ]

    GLOPOP_S = data_catalog.get_source("GLOPOP-S")
    # get path to GLOPOP grid
    GLOPOP_S_GRID = data_catalog.get_source("GLOPOP-S_grid")

    with gzip.open(GLOPOP_S.path.format(region=GDL_region), "rb") as f:
        GLOPOP_S_region = np.frombuffer(f.read(), dtype=np.int32)

    n_people = GLOPOP_S_region.size // len(GLOPOP_S_attribute_names)
    GLOPOP_S_region = pd.DataFrame(
        np.reshape(
            GLOPOP_S_region, (len(GLOPOP_S_attribute_names), n_people)
        ).transpose(),
        columns=GLOPOP_S_attribute_names,
    )

    # load grid
    GLOPOP_GRID_region = rioxarray.open_rasterio(
        GLOPOP_S_GRID.path.format(region=GDL_region)
    )

    # create lookup table for grid cells
    populated_cells = GLOPOP_S_region["GRID_CELL"].unique()
    GLOPOP_S_region['GRID_X'] = np.nan
    GLOPOP_S_region['GRID_Y'] = np.nan

    for GRID_CELL in populated_cells:
        x_y_lookup = np.concat(np.where(GLOPOP_GRID_region.values[0] == GRID_CELL))
        GLOPOP_S_region.loc[GLOPOP_S_region["GRID_CELL"] == GRID_CELL, 'GRID_X'] = GLOPOP_GRID_region.x.values[x_y_lookup[1]]
        GLOPOP_S_region.loc[GLOPOP_S_region["GRID_CELL"] == GRID_CELL, 'GRID_Y'] = GLOPOP_GRID_region.y.values[x_y_lookup[0]]

    return GLOPOP_S_region, GLOPOP_GRID_region
