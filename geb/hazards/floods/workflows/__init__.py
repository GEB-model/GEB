import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pyflwdir
import xarray as xr


def get_river_depth(
    river_segments: gpd.GeoDataFrame,
    method: str,
    parameters: dict[str, float | int],
    bankfull_column: str,
) -> npt.NDArray[np.float32]:
    if method == "manning":
        # Set a minimum value for 'rivslp'
        min_rivslp = 1e-5
        # Replace NaN values in 'rivslp' with the minimum value
        slope = river_segments["slope"].fillna(min_rivslp)
        # Replace 'rivslp' values with the minimum value where they are less than the minimum
        slope = np.where(
            slope < min_rivslp,
            min_rivslp,
            slope,
        )
        # Calculate 'river depth' using the Manning equation
        width = river_segments["width"]
        bankfull_discharge = river_segments[bankfull_column]
        assert (bankfull_discharge[width == 0] == 0).all()
        width = np.where(
            width == 0, 1, width
        )  # Avoid division by zero. Since Q is 0, depth will also be 0.
        depth = ((0.030 * bankfull_discharge) / (np.sqrt(slope) * width)) ** (3 / 5)

    elif method == "power_law":
        # Calculate 'river depth' using the power law equation
        # Powerlaw equation from Andreadis et al (2013)
        c = parameters["c"]
        d = parameters["d"]
        depth = c * (river_segments[bankfull_column].astype(float) ** d)

    else:
        raise ValueError(f"Unknown depth calculation method: {method}")

    # Set a minimum value for 'river depth'
    # Note: Making this value higher or lower can affect results
    min_rivdph = 0
    # Replace 'rivslp' values with the minimum value where they are less than the minimum
    depth = np.where(depth < min_rivdph, min_rivdph, depth)
    # Convert 'rivdph' to float
    return depth.astype(np.float32)


def get_river_manning(river_segments):
    return np.full(len(river_segments), 0.02)


def do_mask_flood_plains(sf) -> None:
    elevation, d8 = pyflwdir.dem.fill_depressions(sf.grid.dep.values)

    flw = pyflwdir.from_array(
        d8,
        transform=sf.grid.raster.transform,
        latlon=False,
    )
    floodplains = flw.floodplains(elevation, upa_min=10)

    mask = xr.full_like(sf.grid.dep, 0, dtype=np.uint8).rename("mask")
    mask.raster.set_nodata(0)
    mask.values = floodplains.astype(mask.dtype)
    sf.set_grid(mask, name="msk")
    sf.config.update({"mskfile": "sfincs.msk"})
    sf.config.update({"indexfile": "sfincs.ind"})
