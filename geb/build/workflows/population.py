import gzip
import zipfile

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

    # get path to GLOPOP tables
    GLOPOP_S = data_catalog.get_source("GLOPOP-S")

    # get path to GLOPOP grid
    GLOPOP_S_GRID = data_catalog.get_source("GLOPOP-SG")

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
    fn_grid = f"/vsizip/{GLOPOP_S_GRID.path}/{GDL_region}_grid_nr.tif"
    GLOPOP_GRID_region = rioxarray.open_rasterio(fn_grid)

    # Get coordinates of each GRID_CELL in GLOPOP_GRID_region
    grid_coords = pd.DataFrame(
        np.stack(np.where(GLOPOP_GRID_region.values[0] > -1), axis=1),
        columns=["GRID_Y", "GRID_X"],
    )
    grid_coords["GRID_CELL"] = GLOPOP_GRID_region.values[0][
        GLOPOP_GRID_region.values[0] > -1
    ]

    # also get the coordinates of the grid cells
    grid_coords["coord_Y"] = GLOPOP_GRID_region.y.values[grid_coords["GRID_Y"]]
    grid_coords["coord_X"] = GLOPOP_GRID_region.x.values[grid_coords["GRID_X"]]

    # Merge the coordinates onto GLOPOP_S_region
    GLOPOP_S_region = GLOPOP_S_region.merge(grid_coords, on="GRID_CELL", how="left")

    return GLOPOP_S_region, GLOPOP_GRID_region
