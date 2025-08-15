import gzip
import zipfile

import numpy as np
import pandas as pd
import rioxarray


def load_GHS_OBAT(data_catalog, iso3):
    GHS_OBAT = data_catalog.get_source("GHS_OBAT")
    data = pd.read_csv(GHS_OBAT.path.format(iso3=iso3))[
        ["id", "lon", "lat", "height", "use", "area"]
    ]
    # only use residential (for now)
    data = data[data["use"] == 1]

    return data


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
        "GRID_CELL",  # CHECK WHAT THE NEW COLUMN IS (ASK MARIJN)
    ]

    GLOPOP_SG = data_catalog.get_source("GLOPOP-SG")

    # load the GLOPOP files for the specified GDL region
    file_name_tif = f"{GDL_region}_grid_nr.tif"
    file_name_gz = f"synthpop_{GDL_region}_grid.dat.gz"
    # Open the zip file
    with zipfile.ZipFile(GLOPOP_SG.path, "r") as zip_ref:
        # Open the GLOPOP_SG grid file
        with zip_ref.open(file_name_tif) as file:
            GLOPOP_GRID_region = rioxarray.open_rasterio(file)
        # Open the GLOPOP_SG synthpop file
        with zip_ref.open(file_name_gz) as file:
            with gzip.open(file, "rb") as f:
                GLOPOP_S_region = np.frombuffer(f.read(), dtype=np.int32)

    n_people = GLOPOP_S_region.size // len(GLOPOP_S_attribute_names)
    GLOPOP_S_region = pd.DataFrame(
        np.reshape(
            GLOPOP_S_region, (len(GLOPOP_S_attribute_names), n_people)
        ).transpose(),
        columns=GLOPOP_S_attribute_names,
    )

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

    # no nans in GRID_Y

    # Merge the coordinates onto GLOPOP_S_region
    GLOPOP_S_region = GLOPOP_S_region.merge(grid_coords, on="GRID_CELL", how="left")
    # assert not GLOPOP_S_region["GRID_Y"].isna().any(), (
    #     "GRID_Y contains NaN values, CHECK GLOPOP DATA"
    # )
    if GLOPOP_S_region["GRID_Y"].isna().any():
        GLOPOP_S_region = GLOPOP_S_region[~GLOPOP_S_region.GRID_Y.isna()]
        print(
            f"WARNING: GRID_Y contains NaN values, CHECK GLOPOP DATA. REGION: {GDL_region}"
        )

    return GLOPOP_S_region, GLOPOP_GRID_region
