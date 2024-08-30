import os
import shutil
import tempfile
import requests
from tqdm import tqdm
import time
import pandas as pd
from datetime import date
from typing import Any, Union
import xarray
import numpy as np
from collections.abc import Mapping


def repeat_grid(data, factor):
    return data.repeat(factor, axis=-2).repeat(factor, axis=-1)


def calculate_cell_area(affine_transform, shape):
    RADIUS_EARTH_EQUATOR = 40075017  # m
    distance_1_degree_latitude = RADIUS_EARTH_EQUATOR / 360

    height, width = shape

    lat_idx = np.arange(0, height).repeat(width).reshape((height, width))
    lat = (lat_idx + 0.5) * affine_transform.e + affine_transform.f
    width_m = (
        distance_1_degree_latitude * np.cos(np.radians(lat)) * abs(affine_transform.a)
    )
    height_m = distance_1_degree_latitude * abs(affine_transform.e)
    return width_m * height_m


def clip_with_grid(ds, mask):
    assert ds.shape == mask.shape
    cells_along_y = mask.sum(dim="x").values.ravel()
    miny = (cells_along_y > 0).argmax().item()
    maxy = cells_along_y.size - (cells_along_y[::-1] > 0).argmax().item()

    cells_along_x = mask.sum(dim="y").values.ravel()
    minx = (cells_along_x > 0).argmax().item()
    maxx = cells_along_x.size - (cells_along_x[::-1] > 0).argmax().item()

    bounds = {"y": slice(miny, maxy), "x": slice(minx, maxx)}

    return ds.isel(bounds), bounds


def bounds_are_within(small_bounds, large_bounds):
    assert small_bounds[0] >= large_bounds[0], "Region bounds do not match"
    assert small_bounds[1] >= large_bounds[1], "Region bounds do not match"
    assert small_bounds[2] <= large_bounds[2], "Region bounds do not match"
    assert small_bounds[3] <= large_bounds[3], "Region bounds do not match"
    return True


def pad_xy(
    self,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    constant_values: Union[
        float, tuple[int, int], Mapping[Any, tuple[int, int]], None
    ] = None,
    return_slice: bool = False,
) -> xarray.DataArray:
    """Pad the array to x,y bounds.

    Parameters
    ----------
    minx: float
        Minimum bound for x coordinate.
    miny: float
        Minimum bound for y coordinate.
    maxx: float
        Maximum bound for x coordinate.
    maxy: float
        Maximum bound for y coordinate.
    constant_values: scalar, tuple or mapping of hashable to tuple
        The value used for padding. If None, nodata will be used if it is
        set, and np.nan otherwise.

    Returns
    -------
    :obj:`xarray.DataArray`:
        The padded object.
    """
    # pylint: disable=too-many-locals
    left, bottom, right, top = self._internal_bounds()
    resolution_x, resolution_y = self.resolution()
    y_before = y_after = 0
    x_before = x_after = 0
    y_coord: Union[xarray.DataArray, np.ndarray] = self._obj[self.y_dim]
    x_coord: Union[xarray.DataArray, np.ndarray] = self._obj[self.x_dim]

    if top - resolution_y < maxy:
        new_y_coord: np.ndarray = np.arange(bottom, maxy, -resolution_y)[::-1]
        y_before = len(new_y_coord) - len(y_coord)
        y_coord = new_y_coord
        top = y_coord[0]
    if bottom + resolution_y > miny:
        new_y_coord = np.arange(top, miny, resolution_y)
        y_after = len(new_y_coord) - len(y_coord)
        y_coord = new_y_coord
        bottom = y_coord[-1]

    if left - resolution_x > minx:
        new_x_coord: np.ndarray = np.arange(right, minx, -resolution_x)[::-1]
        x_before = len(new_x_coord) - len(x_coord)
        x_coord = new_x_coord
        left = x_coord[0]
    if right + resolution_x < maxx:
        new_x_coord = np.arange(left, maxx, resolution_x)
        x_after = len(new_x_coord) - len(x_coord)
        x_coord = new_x_coord
        right = x_coord[-1]

    if constant_values is None:
        constant_values = np.nan if self.nodata is None else self.nodata

    superset = self._obj.pad(
        pad_width={
            self.x_dim: (x_before, x_after),
            self.y_dim: (y_before, y_after),
        },
        constant_values=constant_values,  # type: ignore
    ).rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
    superset[self.x_dim] = x_coord
    superset[self.y_dim] = y_coord
    superset.rio.write_transform(inplace=True)
    if return_slice:
        return superset, {
            "x": slice(x_before, superset["x"].size - x_after),
            "y": slice(y_before, superset["y"].size - y_after),
        }
    else:
        return superset


def fetch_and_save(
    url, file_path, overwrite=False, max_retries=3, delay=5, chunk_size=16384
):
    """
    Fetches data from a URL and saves it to a temporary file, with a retry mechanism.
    Moves the file to the destination if the download is complete.
    Removes the temporary file if the download is interrupted.
    """
    if not overwrite and file_path.exists():
        return True

    attempts = 0
    temp_file = None

    while attempts < max_retries:
        try:
            print(f"Downloading {url} to {file_path}")
            # Attempt to make the request
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raises HTTPError for bad status codes

            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)

            # Write to the temporary file
            total_size = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
            for data in response.iter_content(chunk_size=chunk_size):
                temp_file.write(data)
                progress_bar.update(len(data))
            progress_bar.close()

            # Close the temporary file
            temp_file.close()

            # Move the temporary file to the destination
            shutil.move(temp_file.name, file_path)

            return True  # Exit the function after successful write

        except requests.RequestException as e:
            # Log the error
            print(f"Request failed: {e}. Attempt {attempts + 1} of {max_retries}")

            # Remove the temporary file if it exists
            if temp_file is not None and os.path.exists(temp_file.name):
                os.remove(temp_file.name)

            # Increment the attempt counter and wait before retrying
            attempts += 1
            time.sleep(delay)

    # If all attempts fail, raise an exception
    raise Exception("All attempts to download the file have failed.")


def project_to_future(df, project_future_until_year, inflation_rates):
    # expand table until year
    assert isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex)
    future_index = pd.date_range(
        df.index[-1],
        date(project_future_until_year, 12, 31),
        freq=pd.infer_freq(df.index),
        inclusive="right",
    )
    df = df.reindex(df.index.union(future_index))
    for future_date in tqdm(future_index):
        source_date = future_date - pd.DateOffset(years=1)  # source is year ago
        inflation_index = inflation_rates["time"].index(str(future_date.year))
        for region_id, _ in df.columns:
            region_inflation_rate = inflation_rates["data"][str(region_id)][
                inflation_index
            ]
            df.loc[future_date, region_id] = (
                df.loc[source_date, region_id] * region_inflation_rate
            ).values
    return df
