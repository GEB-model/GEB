import math

import numpy as np
import pandas as pd
import xarray as xr


class Observations:
    def __init__(self):
        pass

    def setup_discharge_observations(self, files):
        transform = self.grid.rio.transform(recalc=True)

        discharge_data = []
        for i, file in enumerate(files):
            filename = file["filename"]
            longitude, latitude = file["longitude"], file["latitude"]
            data = pd.read_csv(filename, index_col=0, parse_dates=True)

            # assert data has one column
            assert data.shape[1] == 1

            px, py = ~transform * (longitude, latitude)
            px = math.floor(px)
            py = math.floor(py)

            discharge_data.append(
                xr.DataArray(
                    np.expand_dims(data.iloc[:, 0].values, 0),
                    dims=["pixel", "time"],
                    coords={
                        "time": data.index.values,
                        "pixel": [i],
                        "px": ("pixel", [px]),
                        "py": ("pixel", [py]),
                    },
                )
            )
        discharge_data = xr.concat(discharge_data, dim="pixel")
        self.set_other(
            discharge_data,
            name="observations/discharge",
            time_chunksize=1e99,  # no chunking
        )
