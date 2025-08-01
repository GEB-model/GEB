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


def generate_storm_surge_hydrographs(model, make_plot=True):
    gtsm_folder = model.input_folder / "other" / "gtsm"
    # read geojson file to get station ids
    station_ids = gpd.read_file(gtsm_folder / "stations.geojson")
    os.makedirs("plot/gtsm", exist_ok=True)

    for station in station_ids["station_id"]:
        average_tide_signal, spring_tide_signal = generate_tide_signals(
            station, gtsm_folder, make_plot=make_plot
        )
        surge_hydrograph_height, surge_hydrograph_duration_mean = (
            generate_surge_hydrograph(station, gtsm_folder, 0.99, make_plot)
        )


def generate_tide_signals(station, gtsm_folder, make_plot=True):
    station_data_path = gtsm_folder / f"gtsm_water_levels_{int(station):04d}.pkl"
    tidepd = pd.read_pickle(station_data_path)
    tidepd = tidepd[datetime(1980, 1, 1) : datetime(2017, 12, 31, 23, 50)]
    tidepd = tidepd.dropna()  # drop NaN values
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
        tides_equal_length = np.vstack((tides_equal_length, tidal_cycles[i][:length]))

    tides_mean = np.mean(
        tides_equal_length, axis=0
    )  # compute the mean of all tidal cycles to get the average tidal cycle
    idx_max = tides_mean[:149].argmax() + 447
    average_tide_signal = np.tile(tides_mean[:149], 7)[idx_max - 447 : idx_max + 447]

    # plot average tide signal
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
        plt.xticks([0, 6, 12, 18, 24, 30, 36], ["0", "6", "12", "18", "24", "30", "36"])
        plt.legend(loc="upper left", fontsize=9)
        ymin, ymax = plt.ylim()
        plt.ylim(top=((ymax - ymin) * 0.25 + ymax))
        plt.xlim(0, 36)
        plt.grid()
        plt.savefig(
            "plot/gtsm/tide_signal_station_%05d.png" % (int(station)),
            format="png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")
    return average_tide_signal, spring_tide_signal


def generate_surge_hydrograph(station, gtsm_folder, percentile, make_plot):
    surgepd = pd.read_pickle(
        gtsm_folder / f"gtsm_surge_{int(station):04d}.pkl"
    )  # read surge data
    surgepd = surgepd[datetime(1980, 1, 1) : datetime(2017, 12, 31, 23, 50)]
    surgepd = surgepd.dropna()  # drop NaN values

    # select surge maxima's
    distance = 432
    surge_maxima_index, surge_maxima_values = ss.find_peaks(
        surgepd.surge.values, distance=distance, height=-10
    )
    surge_peaks = pd.DataFrame(
        data={"waterlevel": surge_maxima_values["peak_heights"]},
        index=surgepd.index.values[surge_maxima_index.tolist()],
    ).sort_values(by="waterlevel")
    surge_peaks_POT = surge_peaks[
        surge_peaks.waterlevel >= surge_peaks.quantile(percentile).waterlevel
    ]  # 0.6 #1.5 #0.8
    surge_array = surgepd.surge.values

    # generate storm surge hydrograph
    hours = 36
    df_before_peak = pd.DataFrame(index=np.around(np.arange(0, 1.0001, 0.005), 3))
    df_after_peak = pd.DataFrame(index=np.around(np.arange(0, 1.0001, 0.005), 3))
    for k in range(len(surge_peaks_POT)):
        timeseries_before_peak = surgepd.loc[
            surge_peaks_POT.iloc[k].name
            - timedelta(hours=hours) : surge_peaks_POT.iloc[k].name
        ]
        timeseries_after_peak = surgepd.loc[
            surge_peaks_POT.iloc[k].name : surge_peaks_POT.iloc[k].name
            + timedelta(hours=hours)
        ]

        normalized_before_peak = (
            timeseries_before_peak / timeseries_before_peak.surge.max()
        )
        normalized_after_peak = (
            timeseries_after_peak / timeseries_after_peak.surge.max()
        )

        # maybe select that part from where the values never exceed zero anymore? Or just use the whole timeslice, independent from value?
        if normalized_after_peak.surge.min() < 0:
            select_stop = np.argwhere(normalized_after_peak.surge.values < 0)[0][0]
        else:
            select_stop = hours * 6

        if normalized_before_peak.surge.min() < 0:
            select_start = np.argwhere(normalized_before_peak.surge.values < 0)[-1][0]
        else:
            select_start = 1

        # this part is only for plotting
        normalized_before_25 = normalized_before_peak[select_start:]
        normalized_after_25 = normalized_after_peak[:select_stop]
        normalized_after_25_plot = normalized_after_peak[: select_stop + 1]
        yy = np.concatenate(
            (
                normalized_before_25.surge.values[:-1],
                normalized_after_25_plot.surge.values,
            )
        )
        xx = (
            np.arange(-len(normalized_before_25) + 1, len(normalized_after_25_plot)) / 6
        )
        if k == 0:
            plt.plot(
                xx, yy, linewidth=0.5, color="grey", alpha=0.5, label="storm surges"
            )
        else:
            plt.plot(xx, yy, linewidth=0.5, color="grey", alpha=0.5)

        for l in df_before_peak.index.values:
            df_before_peak.loc[l, "event" + str(k)] = np.nansum(
                normalized_before_25.surge.values > l
            )

        for l in df_after_peak.index.values:
            df_after_peak.loc[l, "event" + str(k)] = np.nansum(
                normalized_after_25.surge.values > l
            )

    df_before_peak["mean"] = df_before_peak.mean(axis=1)
    df_before_peak["75th"] = df_before_peak.drop("mean", axis=1).quantile(
        q=0.75, axis=1
    )
    df_before_peak["25th"] = df_before_peak.drop("mean", axis=1).quantile(
        q=0.25, axis=1
    )
    df_after_peak["mean"] = df_after_peak.mean(axis=1)
    df_after_peak["75th"] = df_after_peak.drop("mean", axis=1).quantile(q=0.75, axis=1)
    df_after_peak["25th"] = df_after_peak.drop("mean", axis=1).quantile(q=0.25, axis=1)

    surge_hydrograph_height = np.concatenate(
        (
            np.zeros(247),
            np.hstack(
                (df_before_peak.index.values, np.flipud(df_after_peak.index.values)[1:])
            ),
            np.zeros(246),
        )
    )
    surge_hydrograph_duration_mean = np.concatenate(
        (
            np.full(247, np.nan),
            np.hstack(
                (
                    df_before_peak["mean"].values,
                    np.flipud(df_after_peak["mean"].values)[1:],
                )
            ),
            np.full(246, np.nan),
        )
    )

    if make_plot:
        plt.plot(
            -df_before_peak["mean"].values * (1 / 6),
            df_before_peak.index.values,
            label="surge hydrograph",
            color="green",
            linewidth=3,
            linestyle="--",
        )
        plt.plot(
            df_after_peak["mean"].values * (1 / 6),
            df_after_peak.index.values,
            color="green",
            linewidth=3,
            linestyle="--",
        )
        plt.fill_betweenx(
            df_before_peak.index.values,
            -df_before_peak["25th"].values * (1 / 6),
            -df_before_peak["75th"].values * (1 / 6),
            fc="green",
            alpha=0.3,
            label="P 25th-75th",
        )
        plt.fill_betweenx(
            df_after_peak.index.values,
            df_after_peak["25th"].values * (1 / 6),
            df_after_peak["75th"].values * (1 / 6),
            fc="green",
            alpha=0.3,
        )

        plt.legend(loc="upper left", fontsize=9)
        plt.xlabel("time relative to peak (hours)")
        plt.ylabel("normalized surge level")
        plt.title("surge hydrograph")
        plt.ylim(0, 1.25)
        plt.xlim(-35, 35)
        plt.xticks(
            [-36, -24, -12, 0, 12, 24, 36], ["-36", "-24", "-12", "0", "12", "24", "36"]
        )
        plt.yticks(
            [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
            ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2"],
        )
        plt.grid()
        plt.savefig(
            "plot/gtsm/surge_hydrograph_station_%05d_percentile_%02d.png"
            % (int(station), percentile * 100),
            format="png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")

    return surge_hydrograph_height, surge_hydrograph_duration_mean
