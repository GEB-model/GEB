import os
import pandas as pd
import xarray as xr
import time
import numpy as np
import warnings
import traceback
import geopandas as gpd

warnings.filterwarnings("ignore")
import itertools
import netCDF4
import scipy.signal as ss
from datetime import timedelta, datetime
import matplotlib.pyplot as plt


def generate_tide_signals(model, make_plot=True):
    gtsm_folder = model.input_folder / "other" / "gtsm"
    # read geojson file to get station ids
    station_ids = gpd.read_file(gtsm_folder / "stations.geojson")
    for station in station_ids["station_id"]:
        station_data_path = gtsm_folder / f"gtsm_water_levels_{int(station):04d}.pkl"
        tidepd = pd.read_pickle(station_data_path)
        # tidepd = tidepd.set_index("time")
        tide_array = tidepd.waterlevel.values  # open dataframe as numpy array

        if abs(np.quantile(tide_array, 0.99)) < abs(
            np.quantile(tide_array, 0.01)
        ):  # check whether minima or maxima have higher absolute values
            index = [
                tide_array[0:150].argmin()
            ]  # if minima are higher in absolute terms, find first minima index
            for c in itertools.count():
                search_around_index = (
                    index[-1] + 6 * 24 + 5
                )  # search for minima of the next tidal cycle (24 hours 50 min. later)
                try:
                    minima_index = tidepd[
                        search_around_index - 24 : search_around_index + 25
                    ].waterlevel.values.argmin()  # search for minima between +20 hours & 50 min. and +28 hours & 50 min.
                    index.append(
                        search_around_index + minima_index - 24
                    )  # save index of minima
                except:
                    break

            steps = []
            tidal_cycles = []
            for i in range(len(index) - 1):  # loop over index minima
                tidal_cycle = tide_array[
                    index[i] - 75 : index[i + 1] - 35
                ]  # select part of time series (-75 is around previous low tide)
                if (
                    len(tidal_cycle) > 150
                ):  # length of the selected tidal cycle should be at least 150 steps, otherwise the cycle is not a complete one and cannot be used to extract the average tidal cycle
                    tidal_cycles.append(tidal_cycle)  # save tidal cycle
                    steps.append(
                        len(tidal_cycle)
                    )  # save number of steps (minimum of all saved steps becomes the length)

        else:
            index = [
                tide_array[0:150].argmax()
            ]  # if maxima are higher in absolute terms, find first maxima index
            for c in (
                itertools.count()
            ):  # search for minima of the next tidal cycle (24 hours 50 min. later)
                search_around_index = index[-1] + 6 * 24 + 5
                try:
                    maxima_index = tidepd[
                        search_around_index - 24 : search_around_index + 25
                    ].waterlevel.values.argmax()  # search for maxima between +20 hours & 50 min. and +28 hours & 50 min.
                    index.append(
                        search_around_index + maxima_index - 24
                    )  # save index of maxima
                except:
                    break

            steps = []
            tidal_cycles = []
            for i in range(len(index) - 1):  # loop over index maxima
                tidal_cycle = tide_array[
                    index[i] - 37 : index[i + 1]
                ]  # select part of time series (-37 is around previous low tide)
                if (
                    len(tidal_cycle) > 150
                ):  # length of the selected tidal cycle should be at least 150 steps, otherwise the cycle is not a complete one and cannot be used to extract the average tidal cycle
                    tidal_cycles.append(tidal_cycle)  # save tidal cycle
                    steps.append(
                        len(tidal_cycle)
                    )  # save number of steps (minimum of all saved steps becomes the length)

        length = np.min(steps)
        tides_equal_length = tidal_cycles[0][:length]
        for i in range(1, len(tidal_cycles)):
            tides_equal_length = np.vstack(
                (tides_equal_length, tidal_cycles[i][:length])
            )

        tides_mean = np.mean(
            tides_equal_length, axis=0
        )  # compute the mean of all tidal cycles to get the average tidal cycle
        idx_max = tides_mean[:149].argmax() + 447
        average_tide_signal = np.tile(tides_mean[:149], 7)[
            idx_max - 447 : idx_max + 447
        ]

        # plot average tide signal
        os.makedirs("figures", exist_ok=True)
        if make_plot:
            tides_min = np.min(tides_equal_length, axis=0)
            tides_max = np.max(tides_equal_length, axis=0)
            plt.fill_between(
                np.arange(298) / 6,
                np.concatenate((tides_min[:149], tides_min[:149])),
                np.concatenate((tides_max[:149], tides_max[:149])),
                fc="grey",
                alpha=0.3,
                label="tidal cycles",
            )
            plt.plot(
                np.arange(298) / 6,
                np.concatenate((tides_mean[:149], tides_mean[:149])),
                linewidth=2,
                color="black",
                linestyle="solid",
                label="average tide signal",
            )  # plot average tidal cycle as a dashed black line

        # Compute spring tide signal
        tide_array = tidepd.waterlevel.values
        tide_maxima_index, tide_maxima_values = ss.find_peaks(
            tide_array, distance=1728, height=-15
        )
        tide_maxima_values = tide_maxima_values["peak_heights"]
        tide_time_array = tidepd.index.values
        maxima_tide_time = tide_time_array[tide_maxima_index.tolist()]

        spring_tides = []
        for p in range(1, len(maxima_tide_time) - 1):
            spring_tide = tide_array[
                tide_maxima_index[p] - 447 : tide_maxima_index[p] + 447
            ]
            spring_tides.append(spring_tide)

        spring_tide_signal = np.mean(spring_tides, axis=0)

        # plot spring tide signal
        if make_plot:
            spring_tides_min = np.min(spring_tides, axis=0)
            spring_tides_max = np.max(spring_tides, axis=0)
            plt.fill_between(
                np.arange(225) / 6,
                spring_tides_min[335:560],
                spring_tides_max[335:560],
                fc="red",
                alpha=0.3,
                label="spring tidal cycles",
            )
            plt.plot(
                np.arange(225) / 6,
                spring_tide_signal[335:560],
                linewidth=2,
                color="red",
                linestyle="solid",
                label="spring tide signal",
            )
            plt.title("tidal cycles")
            plt.ylabel("water level (m)")
            plt.xlabel("time (hours)")
            plt.xticks(
                [0, 6, 12, 18, 24, 30, 36], ["0", "6", "12", "18", "24", "30", "36"]
            )
            plt.legend(loc="upper left", fontsize=9)
            ymin, ymax = plt.ylim()
            plt.ylim(top=((ymax - ymin) * 0.25 + ymax))
            plt.xlim(0, 36)
            plt.grid()
            plt.savefig(
                "figures/tide_signal_station_%05d.png" % (int(station)),
                format="png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close("all")

    # Placeholder for actual tide signal generation logic
    # This function should return a DataFrame with tide signals
    # For now, we return an empty DataFrame
    # return pd.DataFrame()  # Replace with actual implementation
