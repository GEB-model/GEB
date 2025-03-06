#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calibration tool for Hydrological models
using a distributed evolutionary algorithms in python
DEAP library
https://github.com/DEAP/deap/blob/master/README.md

Félix-Antoine Fortin, François-Michel De Rainville, Marc-André Gardner, Marc Parizeau and Christian Gagné, "DEAP: Evolutionary Algorithms Made Easy", Journal of Machine Learning Research, vol. 13, pp. 2171-2175, jul 2012

The calibration tool was created by Hylke Beck 2014 (JRC, Princeton) hylkeb@princeton.edu
Thanks Hylke for making it available for use and modification
Modified by Peter Burek and Jens de Bruijn
"""

import os
from pathlib import Path
import xarray as xr
import time
import array
import random
import numpy as np
from copy import deepcopy
import signal
import pandas as pd
import yaml
import geopandas as gpd
from io import StringIO
import collections

from deap import creator, base, tools, algorithms
from functools import wraps, partial

import json
import sys
import datetime
import traceback

import multiprocessing
from subprocess import Popen

import pickle


def KGE_calculation(s, o):
    """
    Kling Gupta Efficiency (Kling et al., 2012, http://dx.doi.org/10.1016/j.jhydrol.2012.01.011)
    input:
            s: simulated
            o: observed
    output:
            KGE: Kling Gupta Efficiency
    """
    B = np.mean(s) / np.mean(o)
    y = (np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o))
    r = np.corrcoef(o, s)[0, 1]
    kge = 1 - np.sqrt((r - 1) ** 2 + (B - 1) ** 2 + (y - 1) ** 2)
    return kge


def compute_score(sim, obs):
    diff = sum(abs(s - o) for s, o in zip(sim, obs))
    score = 1 - diff
    return max(0, min(1, score))


def load_ds(output_folder, name, start_time, end_time, time=True):
    var = xr.open_dataset(
        output_folder / f"{name}.zarr.zip",
        mode="r",
        engine="zarr",
    )[name]
    if time and start_time is not None and end_time is not None:
        var = var.sel(time=slice(start_time, end_time))
    return np.squeeze(var)


def load_zarr(
    config,
    directory,
    parameters,
    start_time,
    end_time,
):
    """
    Loads the specified parameters from Zarr and returns a single DataFrame
    with a MultiIndex (time, agents) for the rows, and one column per parameter.
    """
    dfs_simulated = []
    for param in parameters:
        da = load_ds(directory, param, start_time=start_time, end_time=end_time)
        df_param = da.to_dataframe(name=param)
        dfs_simulated.append(df_param)

    df_simulated = pd.concat(dfs_simulated, axis=1)
    return df_simulated


def parse_water_year(wy_str):
    return pd.to_datetime(f"{int(wy_str.split('/')[0]) + 1}-06-30")


def get_observed_well_ratio(config):
    observed_irrigation_sources = gpd.read_file(
        os.path.join(
            config["general"]["original_data"],
            "census",
            "output",
            "irrigation_source_2010-2011.geojson",
        )
    ).to_crs(3857)
    simulated_subdistricts = gpd.read_file(
        os.path.join(config["general"]["input_folder"], "areamaps", "regions.geojson")
    )
    # set index to unique ID combination of state, district and subdistrict
    observed_irrigation_sources.set_index(
        ["state_code", "district_c", "sub_distri"], inplace=True
    )
    simulated_subdistricts.set_index(
        ["state_code", "district_c", "sub_distri"], inplace=True
    )
    # select rows from observed_irrigation_sources where the index is in simulated_subdistricts
    observed_irrigation_sources = observed_irrigation_sources.loc[
        simulated_subdistricts.index
    ]

    region_mask = gpd.read_file(
        os.path.join(config["general"]["input_folder"], "areamaps", "region.geojson")
    ).to_crs(3857)
    assert len(region_mask) == 1
    # get overlapping areas of observed_irrigation_sources and region_mask
    observed_irrigation_sources["area_in_region_mask"] = (
        gpd.overlay(observed_irrigation_sources, region_mask, how="intersection").area
        / observed_irrigation_sources.area.values
    ).values

    # ANALYSIS_THRESHOLD = 0.5

    # observed_irrigation_sources = observed_irrigation_sources[observed_irrigation_sources['area_in_region_mask'] > ANALYSIS_THRESHOLD]
    observed_irrigation_sources = observed_irrigation_sources.join(
        simulated_subdistricts["region_id"]
    )
    observed_irrigation_sources.set_index("region_id", inplace=True)

    total_holdings_observed = observed_irrigation_sources[
        [c for c in observed_irrigation_sources.columns if c.endswith("total_holdings")]
    ].sum(axis=1)
    total_holdings_with_well_observed = observed_irrigation_sources[
        [c for c in observed_irrigation_sources.columns if c.endswith("well_holdings")]
    ].sum(axis=1) + observed_irrigation_sources[
        [
            c
            for c in observed_irrigation_sources.columns
            if c.endswith("tubewell_holdings")
        ]
    ].sum(axis=1)
    ratio_holdings_with_well_observed = (
        total_holdings_with_well_observed / total_holdings_observed
    )
    return ratio_holdings_with_well_observed


def handle_ctrl_c(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global ctrl_c_entered
        if not ctrl_c_entered:
            signal.signal(signal.SIGINT, default_sigint_handler)  # the default
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                ctrl_c_entered = True
                return KeyboardInterrupt
            finally:
                signal.signal(signal.SIGINT, pool_ctrl_c_handler)
        else:
            return KeyboardInterrupt

    return wrapper


def pool_ctrl_c_handler(*args, **kwargs):
    global ctrl_c_entered
    ctrl_c_entered = True


def multi_set(dict_obj, value, *attrs):
    d = dict_obj
    for attr in attrs[:-1]:
        d = d[attr]
    if attrs[-1] not in d:
        raise KeyError(f"Key {attrs} does not exist in config file.")
    d[attrs[-1]] = value


def get_irrigation_wells_score(run_directory, individual, config):
    regions = np.load(
        os.path.join(
            run_directory,
            config["calibration"]["scenario"],
            "region_id",
            "20110101T000000.npz",
        )
    )["data"]
    # field_size = np.load(os.path.join(run_directory, config['calibration']['scenario'], 'field_size', '20110101T000000.npz'))['data']
    irrigation_source = np.load(
        os.path.join(
            run_directory,
            config["calibration"]["scenario"],
            "irrigation_source",
            "20110101T000000.npz",
        )
    )["data"]

    with open(
        os.path.join(
            config["general"]["input_folder"],
            "agents",
            "farmers",
            "irrigation_sources.json",
        )
    ) as f:
        irrigation_source_key = json.load(f)

    well_irrigated = np.isin(
        irrigation_source,
        [irrigation_source_key["well"], irrigation_source_key["tubewell"]],
    )
    # Calculate the ratio of farmers with a well per tehsil
    farmers_per_region = np.bincount(regions)
    well_irrigated_per_tehsil = np.bincount(regions, weights=well_irrigated)
    minimum_farmer_mask = np.where(farmers_per_region > 100)
    ratio_well_irrigated = (
        well_irrigated_per_tehsil[minimum_farmer_mask]
        / farmers_per_region[minimum_farmer_mask]
    )

    ratio_holdings_with_well_observed = get_observed_well_ratio(config)

    ratio_holdings_with_well_observed = ratio_holdings_with_well_observed[
        minimum_farmer_mask[0]
    ].values
    ratio_holdings_with_well_simulated = ratio_well_irrigated

    minimum_well_mask = np.where(ratio_holdings_with_well_observed > 0.01)

    irrigation_well_score = 1 - abs(
        (
            (ratio_holdings_with_well_simulated - ratio_holdings_with_well_observed)
            / ratio_holdings_with_well_observed
        )
    )

    total_farmers = farmers_per_region.sum()
    farmers_fraction = farmers_per_region[minimum_farmer_mask] / total_farmers

    irrigation_well_score = float(
        np.sum(
            irrigation_well_score[minimum_well_mask]
            * farmers_fraction[minimum_well_mask]
        )
    )
    print(
        "run_id: "
        + str(individual.label)
        + ", IWS: "
        + "{0:.3f}".format(irrigation_well_score)
    )
    with open(
        os.path.join(config["calibration"]["path"], "IWS_log.csv"), "a"
    ) as myfile:
        myfile.write(str(individual.label) + "," + str(irrigation_well_score) + "\n")

    return irrigation_well_score


def KGE_water_price(run_directory, individual, config):
    regions_of_interest = [
        "VIC Goulburn-Broken",
        "VIC Murray Above",
        "VIC Loddon-Campaspe",
        "NSW Murray Below",
        "NSW Murray Above",
    ]

    ################ Water price #######################
    # Observed data path
    water_price_observed_fp = Path(
        "calibration_data/water_use/WaterMarketOutlook_2023-04_data_tables_v1.0.0.xlsx"
    )
    water_price_observed_df = pd.read_excel(water_price_observed_fp, sheet_name=3)

    # Filter relevant regions, rename columns
    filtered_prices_df = water_price_observed_df[
        water_price_observed_df["Region"].isin(regions_of_interest)
    ].copy()
    filtered_prices_df = filtered_prices_df[
        ["Date", "Region", "Monthly average price ($/ML)"]
    ]
    filtered_prices_df.columns = ["time", "Region", "water_price_observed"]
    filtered_prices_df["time"] = pd.to_datetime(
        filtered_prices_df["time"]
    ).dt.normalize()

    pivot_df = filtered_prices_df.pivot(
        index="time", columns="Region", values="water_price_observed"
    ).sort_index()
    pivot_df = pivot_df.apply(pd.to_numeric, errors="coerce").interpolate(method="time")
    pivot_df["water_price_observed"] = pivot_df[regions_of_interest].mean(axis=1)
    combined_regions_df = pivot_df[["water_price_observed"]].copy()

    # Convert from AUD to USD if needed
    fp_conversion = os.path.join(
        config["general"]["input_folder"],
        "economics",
        "lcu_per_usd_conversion_rates.json",
    )
    with open(fp_conversion, "r", encoding="utf-8") as file:
        conversion_rates = json.load(file)
    yearly_rates = dict(
        zip(
            map(int, conversion_rates["time"]),
            map(float, conversion_rates["data"]["0"]),
        )
    )
    combined_regions_df["year"] = combined_regions_df.index.year
    combined_regions_df["conversion_rate"] = combined_regions_df["year"].map(
        yearly_rates
    )
    combined_regions_df["water_price_observed"] = (
        combined_regions_df["water_price_observed"]
        / combined_regions_df["conversion_rate"]
    )
    df_observed_price = combined_regions_df.drop(columns=["year", "conversion_rate"])
    df_observed_price = df_observed_price.rename(
        columns={"water_price_observed": "observed"}
    )
    df_observed_price.index.name = "time"

    # Add in the simulated data
    fp_simulated = Path(
        os.path.join(
            run_directory,
            config["calibration"]["scenario"],
        )
    )
    water_price_simulated = load_ds(
        fp_simulated,
        "water_price",
        start_time=config["calibration"]["start_time"],
        end_time=config["calibration"]["end_time"],
    )
    df_simulated = water_price_simulated.to_dataframe(name="simulated")

    # Combine
    df_combined = pd.concat(
        [
            df_simulated.resample("YE-JUN").mean(),
            df_observed_price.resample("YE-JUN").mean(),
        ],
        axis=1,
        join="inner",
    )

    kge = KGE_calculation(s=df_combined["simulated"], o=df_combined["observed"])
    print(
        "run_id: "
        + str(individual.label)
        + ", KGE_water_price: "
        + "{0:.3f}".format(kge)
    )
    with open(
        os.path.join(config["calibration"]["path"], "KGE_water_price_log.csv"), "a"
    ) as myfile:
        myfile.write(str(individual.label) + "," + str(kge) + "\n")
    return kge


def KGE_region_diversion(run_directory, individual, config):
    fp_simulated = Path(os.path.join(run_directory, config["calibration"]["scenario"]))

    # Load simulated data and resample yearly (July–June)
    parameters = ["allocation_vic", "allocation_nsw"]
    dfs_simulated = []
    for parameter in parameters:
        da = load_ds(
            fp_simulated,
            parameter,
            start_time=config["calibration"]["start_time"],
            end_time=config["calibration"]["end_time"],
        )
        df_param = da.to_dataframe(name=parameter)
        dfs_simulated.append(df_param)

    df_simulated = pd.concat(dfs_simulated, axis=1)
    df_simulated_yearly = df_simulated.resample("A-JUN").sum() / 1_000

    # ---------------------------------------------------------
    # Load observed data; treat the 'Date' as YYYY-07-01
    # so it aligns with the July–June year used above.
    # ---------------------------------------------------------
    # Load the new dataset
    regions_of_interest = [
        "VIC Goulburn-Broken",
        "VIC Murray Above",
        "VIC Loddon-Campaspe",
        "NSW Murray Below",
        "NSW Murray Above",
    ]
    diversions_observed_fp = Path(
        "calibration_data/water_use/MDBWaterMarketCatchmentDataset_Supply_v1.0.0.xlsx"
    )
    diversions_observed_df = pd.read_excel(diversions_observed_fp, sheet_name=1)

    # Filter to only the regions of interest
    filtered_diversions_df = diversions_observed_df[
        diversions_observed_df["Region"].isin(regions_of_interest)
    ].copy()

    # Keep only relevant columns
    filtered_diversions_df = filtered_diversions_df[["Year", "Region", "U"]].rename(
        columns={"Year": "time", "U": "diversion_observed"}
    )

    # Convert 'time' to datetime
    filtered_diversions_df["time"] = pd.to_datetime(
        filtered_diversions_df["time"].astype(str) + "-06-30"
    )

    pivot_diversions_df = filtered_diversions_df.pivot(
        index="time", columns="Region", values="diversion_observed"
    ).sort_index()

    # pivot_price_df = pivot_price_df.shift(1)
    pivot_diversions_df = pivot_diversions_df.apply(
        pd.to_numeric, errors="coerce"
    ).interpolate("time")

    pivot_diversions_df["diversions_vic"] = pivot_diversions_df[
        ["VIC Goulburn-Broken", "VIC Murray Above", "VIC Loddon-Campaspe"]
    ].sum(axis=1)
    pivot_diversions_df["diversions_nsw"] = pivot_diversions_df[
        ["NSW Murray Below", "NSW Murray Above"]
    ].sum(axis=1)

    df_vic = pd.concat(
        [df_simulated_yearly["allocation_vic"], pivot_diversions_df["diversions_vic"]],
        axis=1,
        join="inner",
    )
    df_vic.columns = ["simulated", "observed"]
    df_vic["simulated"] += 0.0001
    kge_vic = KGE_calculation(df_vic["simulated"], df_vic["observed"])

    # NSW
    df_nsw = pd.concat(
        [df_simulated_yearly["allocation_nsw"], pivot_diversions_df["diversions_nsw"]],
        axis=1,
        join="inner",
    )
    df_nsw.columns = ["simulated", "observed"]
    df_nsw["simulated"] += 0.0001
    kge_nsw = KGE_calculation(df_nsw["simulated"], df_nsw["observed"])

    kge = (kge_vic + kge_nsw) / 2.0

    print(
        f"run_id: {individual.label}, KGE_diversions_VIC: {kge_vic:.3f}, KGE_diversions_NSW: {kge_nsw:.3f}, KGE_mean: {kge:.3f}"
    )
    with open(
        os.path.join(config["calibration"]["path"], "KGE_diversions_log.csv"), "a"
    ) as f:
        f.write(f"{individual.label},{kge}\n")

    return kge


def get_KGE_discharge(run_directory, individual, config, gauges, observed_streamflow):
    def get_streamflows(gauge, observed_streamflow):
        # Get the path of the simulated streamflow file
        Qsim_tss = os.path.join(
            run_directory,
            config["calibration"]["scenario"],
            f"{gauge[0]} {gauge[1]}.csv",
        )
        # os.path.join(run_directory, 'base/discharge.csv')

        # Check if the simulated streamflow file exists
        if not os.path.isfile(Qsim_tss):
            print("run_id: " + str(individual.label) + " File: " + Qsim_tss)
            raise Exception(
                "No simulated streamflow found. Is the data exported in the ini-file (e.g., 'OUT_TSS_Daily = var.discharge'). Probably the model failed to start? Check the log files of the run!"
            )

        # Read the simulated streamflow data from the file
        simulated_streamflow = pd.read_csv(
            Qsim_tss, sep=",", parse_dates=True, index_col=0
        )

        # parse the dates in the index
        # simulated_streamflow.index = pd.date_range(config['calibration']['start_time'] + timedelta(days=1), config['calibration']['end_time'])

        simulated_streamflow_gauge = simulated_streamflow[" ".join(map(str, gauge))]
        simulated_streamflow_gauge.name = "simulated"
        observed_streamflow_gauge = observed_streamflow[gauge]
        observed_streamflow_gauge.name = "observed"

        # Combine the simulated and observed streamflow data
        streamflows = pd.concat(
            [simulated_streamflow_gauge, observed_streamflow_gauge],
            join="inner",
            axis=1,
        )

        # Add a small value to the simulated streamflow to avoid division by zero
        streamflows["simulated"] += 0.0001
        return streamflows

    streamflows = [get_streamflows(gauge, observed_streamflow) for gauge in gauges]
    streamflows = [streamflow for streamflow in streamflows if not streamflow.empty]
    if config["calibration"]["monthly"] is True:
        # Calculate the monthly mean of the streamflow data
        streamflows = [streamflows.resample("M").mean() for streamflows in streamflows]

    KGEs = []
    for streamflow in streamflows:
        # print(f"Processing: {streamflow}")
        KGEs.append(
            KGE_calculation(s=streamflow["simulated"], o=streamflow["observed"])
        )

    assert KGEs  # Check if KGEs is not empty
    kge = np.mean(KGEs)

    print(
        "run_id: " + str(individual.label) + ", KGE_discharge: " + "{0:.3f}".format(kge)
    )
    with open(
        os.path.join(config["calibration"]["path"], "KGE_discharge_log.csv"), "a"
    ) as myfile:
        myfile.write(str(individual.label) + "," + str(kge) + "\n")

    return kge


def get_KGE_yield_ratio(run_directory, individual, config):
    observed_yield_ratios = get_observed_yield_ratios(run_directory, config)
    yield_ratios_simulated_path = os.path.join(
        run_directory, config["calibration"]["scenario"], "yield_ratio.csv"
    )
    # Check if the simulated streamflow file exists
    if not os.path.isfile(yield_ratios_simulated_path):
        print(
            "run_id: " + str(individual.label) + " File: " + yield_ratios_simulated_path
        )
        raise Exception(
            "No simulated streamflow found. Is the data exported in the ini-file (e.g., 'OUT_TSS_Daily = var.discharge'). Probably the model failed to start? Check the log files of the run!"
        )

    # Read the simulated yield ratios from the file
    simulated_yield_ratio = pd.read_csv(
        yield_ratios_simulated_path, sep=",", parse_dates=True, index_col=0
    )
    simulated_yield_ratio = simulated_yield_ratio["yield_ratio"]

    # Name and resample to yearly data
    simulated_yield_ratio.name = "simulated"
    simulated_yield_ratio = simulated_yield_ratio.resample("Y").mean()

    # Take the first instead of last day of the year
    simulated_yield_ratio.index = simulated_yield_ratio.index.to_period("Y").start_time

    yield_ratios_combined = pd.concat(
        [simulated_yield_ratio, observed_yield_ratios], join="inner", axis=1
    )
    # Add a small value to the simulated streamflow to avoid division by zero
    yield_ratios_combined["simulated"] += 0.0001

    kge = KGE_calculation(
        s=yield_ratios_combined["simulated"], o=yield_ratios_combined["observed"]
    )

    print(
        "run_id: "
        + str(individual.label)
        + ", KGE yield ratio: "
        + "{0:.3f}".format(kge)
    )
    with open(
        os.path.join(config["calibration"]["path"], "KGE_yield_ratio_log.csv"), "a"
    ) as myfile:
        myfile.write(str(individual.label) + "," + str(kge) + "\n")

    return kge


def get_observed_yield_ratios(run_directory, config):
    regions = np.load(
        os.path.join(
            run_directory,
            config["calibration"]["scenario"],
            "region_id",
            "20030101T000000.npz",
        )
    )["data"]
    simulated_subdistricts = gpd.read_file(
        os.path.join(config["general"]["input_folder"], "areamaps", "regions.geojson")
    )
    unique_subdistricts = np.unique(simulated_subdistricts["district_c"])

    observed_yield_ratios = {}
    for subdistrict in unique_subdistricts:
        district_path = os.path.join(
            config["general"]["original_data"],
            "calibration",
            "yield_ratio",
            f"{subdistrict}.csv",
        )
        yield_ratio_data = pd.read_csv(
            district_path, sep=";", parse_dates=True, index_col=0
        )

        observed_yield_ratios[subdistrict] = yield_ratio_data["yield_ratio"]
        assert (observed_yield_ratios[subdistrict] >= 0).all()

    # Determine the proportion of farmers per district
    district_c_series = simulated_subdistricts["district_c"].astype(int)
    farmers_per_subregion = np.bincount(regions)

    # combine the
    combined_dataframe = pd.DataFrame(
        {"district": district_c_series, "total_farmers": farmers_per_subregion}
    )

    # Determine the fractions of farmers per district
    farmers_per_district = combined_dataframe.groupby("district")["total_farmers"].sum()
    total_farmers = farmers_per_district.sum()
    farmers_fraction = farmers_per_district / total_farmers

    summed_series = pd.Series(dtype=float)
    # Use the fractions to get the average yield ratios for this region
    for subdistrict in unique_subdistricts:
        yield_ratio_fraction = (
            observed_yield_ratios[subdistrict] * farmers_fraction[int(subdistrict)]
        )
        summed_series = summed_series.add(yield_ratio_fraction, fill_value=0)

    summed_series.name = "observed"

    return summed_series


def get_observed_irrigation_method(config):
    calibration_config = config["calibration"]

    # Read data
    fp = os.path.join(
        calibration_config["observed_data"], "water_use", "murray_water_use.csv"
    )
    data_df = pd.read_csv(fp)
    data_df["Value"] = pd.to_numeric(data_df["Value"], errors="coerce")

    # Filter and rename columns
    irrigation_types = [
        "surface_irrigation",
        "drip_or_trickle_irrigation",
        "sprinkler_irrigation",
    ]
    irrigation_df = data_df[data_df["Description"].isin(irrigation_types)]
    irrigation_df = irrigation_df.rename(columns={"Year": "time"})

    # Parse e.g. "2002/3" -> datetime(2003-06-30)
    def parse_water_year(wy_str):
        return pd.to_datetime(f"{int(wy_str.split('/')[0]) + 1}-06-30")

    irrigation_df["time"] = irrigation_df["time"].apply(parse_water_year)

    # Pivot and compute total + fraction
    irrigation_pivot = irrigation_df.pivot_table(
        index=["time", "Region"], columns="Description", values="Value", aggfunc="first"
    )
    irrigation_pivot["total_irrigation"] = irrigation_pivot[irrigation_types].sum(
        axis=1
    )
    for irr_type in irrigation_types:
        irrigation_pivot["fraction_" + irr_type] = (
            irrigation_pivot[irr_type] / irrigation_pivot["total_irrigation"]
        )

    # Melt fractions into long format, combine with original
    fraction_cols = ["fraction_" + x for x in irrigation_types]
    fraction_df = (
        irrigation_pivot[fraction_cols]
        .reset_index()
        .melt(
            id_vars=["time", "Region"],
            value_vars=fraction_cols,
            var_name="Description",
            value_name="Value",
        )
    )
    combined_data_df = pd.concat(
        [data_df, fraction_df[["time", "Region", "Description", "Value"]]],
        ignore_index=True,
    )

    # Pivot for ratio-based estimation
    pivot_df = combined_data_df.pivot_table(
        index=["time", "Description"], columns="Region", values="Value", aggfunc="first"
    )

    # Estimate missing data via ratio
    ref_regions = {
        "murray": "NSW",
        "goulburn_broken": "VICT",
        "north_central": "VICT",
        "north_east": "VICT",
    }
    for target, ref in ref_regions.items():
        if target in pivot_df.columns and ref in pivot_df.columns:
            pivot_df["Ratio"] = pivot_df[target] / pivot_df[ref]
            desc_ratio = pivot_df.groupby(level="Description")["Ratio"].mean()
            pivot_df["Avg_Ratio"] = pivot_df.index.get_level_values("Description").map(
                desc_ratio
            )
            pivot_df["Estimated"] = pivot_df[ref] * pivot_df["Avg_Ratio"]
            pivot_df[target] = pivot_df[target].fillna(pivot_df["Estimated"])
            pivot_df = pivot_df.drop(columns=["Ratio", "Avg_Ratio", "Estimated"])

    # Normalize fraction rows to sum to 1
    frac_desc = ["fraction_" + x for x in irrigation_types]
    frac_df = pivot_df.loc[
        pivot_df.index.get_level_values("Description").isin(frac_desc)
    ].copy()
    frac_df = frac_df.fillna(0)
    frac_sum = frac_df.groupby(level=["time"]).sum()
    frac_norm = frac_df.div(frac_sum)
    pivot_df.update(frac_norm)

    # Flatten for interpolation
    df_flat = pivot_df.reset_index()
    df_flat = df_flat.sort_values("time")
    min_date = df_flat["time"].min()
    max_date = df_flat["time"].max()

    # For each Description, reindex on full date range, interpolate by time
    all_desc = df_flat["Description"].unique()
    groups = []
    for d in all_desc:
        g = df_flat[df_flat["Description"] == d].copy()

        # set index to a proper datetime
        g = g.set_index("time")
        g.index = pd.to_datetime(g.index, errors="coerce")

        g = g.sort_index()

        # reindex to fill with np.nan
        dates = pd.date_range(min_date, max_date, freq="YE-JUN")
        g = g.reindex(dates, fill_value=np.nan)

        # separate out non-numeric columns so they won't raise warnings
        non_numeric_cols = g.select_dtypes(exclude=["number"]).columns
        temp_non_numeric = g[non_numeric_cols]
        g = g.drop(columns=non_numeric_cols)

        g = g.infer_objects(copy=False)
        g = g.interpolate(method="time", limit_direction="both")

        # bring back non-numeric columns
        g[non_numeric_cols] = temp_non_numeric

        # ensure "Description" column is still set
        g["Description"] = d

        groups.append(g)

    # Recombine, pivot back
    df_flat_int = pd.concat(groups).reset_index().rename(columns={"index": "time"})
    pivot_df = df_flat_int.pivot_table(
        index=["time", "Description"],
        aggfunc="first",  # or pivot columns if needed
    ).sort_index(level="time")

    return pivot_df


def get_irrigation_method_score(run_directory, individual, config):
    def get_simulated_fractions_by_region(
        array_simulated, region_agents, region_value, field_sizes
    ):
        # Use field area rather than count
        mask = region_agents == region_value
        subset = array_simulated[mask]
        subset_field_sizes = field_sizes[mask]
        total_area = np.sum(subset_field_sizes)
        if total_area == 0:
            return 0.0, 0.0, 0.0
        sprinkler_area = np.sum(subset_field_sizes[subset == 0.7])
        drip_area = np.sum(subset_field_sizes[subset == 0.9])
        surface_area = np.sum(subset_field_sizes[subset == 0.5])
        return (
            sprinkler_area / total_area,
            drip_area / total_area,
            surface_area / total_area,
        )

    observed_pivot = get_observed_irrigation_method(config)
    start_time = observed_pivot.index.get_level_values("time").min()
    end_time = observed_pivot.index.get_level_values("time").max()
    fp_simulated = Path(os.path.join(run_directory, config["calibration"]["scenario"]))

    irrigation_efficiency_simulated = load_ds(
        fp_simulated,
        "irrigation_efficiency",
        start_time=start_time,
        end_time=end_time,
    )

    region_id = load_ds(
        fp_simulated,
        "region_id",
        start_time=config["calibration"]["start_time"],
        end_time=config["calibration"]["end_time"],
    ).values

    aus_states = gpd.read_file(
        os.path.join(config["general"]["input_folder"], "areamaps", "regions.gpkg")
    )
    region_agents = np.where(
        aus_states["NAME_1"].values[np.int32(region_id)] == "New South Wales", 1, 0
    )

    channel_irrigation = load_ds(
        fp_simulated,
        "channel_irrigation",
        start_time=config["calibration"]["start_time"],
        end_time=config["calibration"]["end_time"],
    ).values

    groundwater_irrigation = load_ds(
        fp_simulated,
        "groundwater_irrigation",
        start_time=config["calibration"]["start_time"],
        end_time=config["calibration"]["end_time"],
    ).values

    irrigation_mask = channel_irrigation.any(axis=0) | groundwater_irrigation.any(
        axis=0
    )
    # Load field size per farmer
    field_size_per_farmer = load_ds(
        Path("report/base"),
        "field_size_per_farmer",
        start_time=config["calibration"]["start_time"],
        end_time=config["calibration"]["end_time"],
    ).values

    time_index = irrigation_efficiency_simulated.indexes["time"]
    df_simulated_nsw = pd.DataFrame(
        index=time_index, columns=["sprinkler", "drip", "surface"], dtype=float
    )
    df_simulated_vict = pd.DataFrame(
        index=time_index, columns=["sprinkler", "drip", "surface"], dtype=float
    )

    for i, t in enumerate(time_index):
        subset = irrigation_efficiency_simulated.values[i, irrigation_mask]
        # Subset field sizes for irrigating agents
        field_size_subset = field_size_per_farmer[irrigation_mask]
        s_nsw, d_nsw, w_nsw = get_simulated_fractions_by_region(
            subset, region_agents[irrigation_mask], 1, field_size_subset
        )
        s_vict, d_vict, w_vict = get_simulated_fractions_by_region(
            subset, region_agents[irrigation_mask], 0, field_size_subset
        )
        df_simulated_nsw.loc[t] = [s_nsw, d_nsw, w_nsw]
        df_simulated_vict.loc[t] = [s_vict, d_vict, w_vict]

    def extract_observed_fractions(obs_pivot, region_name):
        idx = obs_pivot.index.get_level_values("time").unique()
        df = pd.DataFrame(
            index=idx, columns=["sprinkler", "drip", "surface"], dtype=float
        )
        for col, key in zip(
            ["sprinkler", "drip", "surface"],
            [
                "fraction_sprinkler_irrigation",
                "fraction_drip_or_trickle_irrigation",
                "fraction_surface_irrigation",
            ],
        ):
            df[col] = obs_pivot.xs(key, level="Description")[region_name]
        return df

    df_observed_nsw = extract_observed_fractions(observed_pivot, "NSW")
    df_observed_vict = extract_observed_fractions(observed_pivot, "VICT")

    def compute_kge_for_fraction(df_sim, df_obs, fraction_name):
        df_concat = pd.concat(
            [df_sim[fraction_name], df_obs[fraction_name]], axis=1, join="inner"
        )
        df_concat.columns = ["sim", "obs"]
        df_concat["sim"] += 1e-4
        return KGE_calculation(df_concat["sim"], df_concat["obs"])

    kge_results_nsw, kge_results_vict = {}, {}
    for fraction_name in ["sprinkler", "drip", "surface"]:
        kge_results_nsw[fraction_name] = compute_kge_for_fraction(
            df_simulated_nsw, df_observed_nsw, fraction_name
        )
        kge_results_vict[fraction_name] = compute_kge_for_fraction(
            df_simulated_vict, df_observed_vict, fraction_name
        )

    all_results = np.nan_to_num(
        list(kge_results_nsw.values()) + list(kge_results_vict.values()), nan=-2.5
    )
    kge = np.nanmean(all_results)

    print(
        "run_id: "
        + str(individual.label)
        + ", KGE_irr_method: "
        + "{0:.3f}".format(kge)
    )
    with open(
        os.path.join(config["calibration"]["path"], "KGE_irr_method_log.csv"), "a"
    ) as myfile:
        myfile.write(str(individual.label) + "," + str(kge) + "\n")

    return kge


def get_observed_water_use(config):
    calibration_config = config["calibration"]

    # Read data
    fp = os.path.join(
        calibration_config["observed_data"], "water_use", "murray_water_use.csv"
    )
    data_df = pd.read_csv(fp)
    data_df["Value"] = pd.to_numeric(data_df["Value"], errors="coerce")

    # Subset by desired parameters
    parameters = ["volume_applied_cultivated", "application_rate_cultivated"]
    water_use_df = data_df[data_df["Description"].isin(parameters)]
    water_use_df = water_use_df.rename(columns={"Year": "time"})

    # Parse e.g. "2002/3" -> datetime(2003-06-30)
    water_use_df["time"] = water_use_df["time"].apply(parse_water_year)

    # PIVOT so that "Region" => columns, and "Description" => part of the index
    # This keeps "Description" in the index (like your first code snippet).
    water_use_pivot = water_use_df.pivot_table(
        index=["time", "Description"],  # pivot on (time, Description)
        columns="Region",  # region becomes columns
        values="Value",
        aggfunc="first",
    ).sort_index(level="time")

    # Estimate missing data via ratio of target region to reference region
    ref_regions = {
        "murray": "NSW",
        "goulburn_broken": "VICT",
        "north_central": "VICT",
        "north_east": "VICT",
    }
    for target, ref in ref_regions.items():
        # Only proceed if both target & ref columns exist
        if target in water_use_pivot.columns and ref in water_use_pivot.columns:
            water_use_pivot["Ratio"] = water_use_pivot[target] / water_use_pivot[ref]
            # Compute the average ratio per Description
            desc_ratio = water_use_pivot.groupby(level="Description")["Ratio"].mean()

            # Map that ratio back to each row
            water_use_pivot["Avg_Ratio"] = water_use_pivot.index.get_level_values(
                "Description"
            ).map(desc_ratio)
            # Estimate missing values
            water_use_pivot["Estimated"] = (
                water_use_pivot[ref] * water_use_pivot["Avg_Ratio"]
            )
            water_use_pivot[target] = water_use_pivot[target].fillna(
                water_use_pivot["Estimated"]
            )

            # Drop temporary columns
            water_use_pivot = water_use_pivot.drop(
                columns=["Ratio", "Avg_Ratio", "Estimated"]
            )

    # Flatten for time interpolation
    df_flat = (
        water_use_pivot.reset_index()
    )  # brings 'time' & 'Description' back as columns
    df_flat = df_flat.sort_values("time")
    min_date = df_flat["time"].min()
    max_date = df_flat["time"].max()

    # For each Description, reindex by full date range, interpolate by time
    all_desc = df_flat["Description"].unique()
    groups = []
    for d in all_desc:
        g = df_flat[df_flat["Description"] == d].copy()

        # set index to datetime
        g = g.set_index("time")
        g.index = pd.to_datetime(g.index, errors="coerce")
        g = g.sort_index()

        # reindex to fill with NaN
        dates = pd.date_range(min_date, max_date, freq="YE-JUN")
        g = g.reindex(dates, fill_value=np.nan)

        # separate out non-numeric columns to avoid warnings
        non_numeric_cols = g.select_dtypes(exclude=["number"]).columns
        temp_non_numeric = g[non_numeric_cols]
        g = g.drop(columns=non_numeric_cols)

        # interpolate in time
        g = g.infer_objects(copy=False)
        g = g.interpolate(method="time", limit_direction="both")

        # bring back non-numeric columns
        g[non_numeric_cols] = temp_non_numeric
        g["Description"] = d

        groups.append(g)

    # Recombine, pivot again if desired
    df_flat_int = pd.concat(groups).reset_index().rename(columns={"index": "time"})
    pivot_df = df_flat_int.pivot_table(
        index=["time", "Description"],
        aggfunc="first",  # keep numeric columns or pivot more specifically
    ).sort_index(level="time")

    return pivot_df


def get_water_use_score(run_directory, individual, config):
    observed_pivot = get_observed_water_use(config)

    start_time = observed_pivot.index.get_level_values("time").min()
    end_time = observed_pivot.index.get_level_values("time").max()

    fp_simulated = Path(os.path.join(run_directory, config["calibration"]["scenario"]))

    # Load region_id for each agent
    region_id = load_ds(
        Path("report/base"),
        "region_id",
        start_time=config["calibration"]["start_time"],
        end_time=config["calibration"]["end_time"],
    ).values

    # Load shapefile (or GeoPackage) with state names
    aus_states = gpd.read_file(
        os.path.join(config["general"]["input_folder"], "areamaps", "regions.gpkg")
    )

    # Create masks for each state
    agent_states = aus_states["NAME_1"].values[np.int32(region_id)]
    mask_nsw = agent_states == "New South Wales"
    mask_vic = agent_states == "Victoria"

    # Parameters for irrigation water use
    parameters_water_use = [
        "channel_irrigation",
    ]

    # Load the relevant Zarr data
    df_simulated = load_zarr(
        config, fp_simulated, parameters_water_use, start_time, end_time
    )

    # Sum the two irrigation parameters
    df_total_use = df_simulated[parameters_water_use[0]]

    # Convert negative/zero to NaN or filter them out
    df_total_use = df_total_use[df_total_use > 0]

    # --- Split by region ---
    df_total_use_nsw = df_total_use[
        df_total_use.index.get_level_values("agents").isin(np.where(mask_nsw)[0])
    ].dropna()

    df_total_use_vic = df_total_use[
        df_total_use.index.get_level_values("agents").isin(np.where(mask_vic)[0])
    ].dropna()

    # Aggregate
    df_simulated_nsw = (
        df_total_use_nsw.groupby(level="time").sum().resample("YE-JUN").sum()
    ) / 1000  # aggregate and m3 -> ML
    df_simulated_nsw.name = "simulated"

    df_simulated_vic = (
        df_total_use_vic.groupby(level="time").sum().resample("YE-JUN").sum()
    ) / 1000  # aggregate and m3 -> ML
    df_simulated_vic.name = "simulated"

    df_observed_nsw = observed_pivot.xs(
        "volume_applied_cultivated", level="Description"
    )["murray"]
    df_observed_nsw.name = "observed"

    df_observed_vic = observed_pivot.xs(
        "volume_applied_cultivated", level="Description"
    )[["goulburn_broken", "north_central", "north_east"]].sum(axis=1)
    df_observed_vic.name = "observed"

    df_vic = pd.concat(
        [df_simulated_vic, df_observed_vic],
        axis=1,
        join="inner",
    )

    df_nsw = pd.concat(
        [df_simulated_nsw, df_observed_nsw],
        axis=1,
        join="inner",
    )

    kge_vic = KGE_calculation(
        df_vic["simulated"].iloc[1:-1], df_vic["observed"].iloc[1:-1]
    )

    kge_nsw = KGE_calculation(
        df_nsw["simulated"].iloc[1:-1], df_nsw["observed"].iloc[1:-1]
    )

    kge = (kge_vic + kge_nsw) / 2.0
    print(
        "run_id: " + str(individual.label) + ", KGE_water_use: " + "{0:.3f}".format(kge)
    )
    with open(
        os.path.join(config["calibration"]["path"], "KGE_water_use_log.csv"), "a"
    ) as myfile:
        myfile.write(str(individual.label) + "," + str(kge) + "\n")
    return kge


def get_observed_crops(run_directory, individual, config):
    def parse_water_year(year_int):
        return pd.to_datetime(f"{year_int}-06-30")

    def clean_csv_by_majority_length_in_memory(fp_in):
        with open(fp_in, "r") as f_in:
            lines = [line.strip() for line in f_in]
        lengths = [len(line.split(",")) for line in lines]
        counts = collections.Counter(lengths)
        majority_len = counts.most_common(1)[0][0]

        cleaned_lines = []
        for line in lines:
            parts = line.split(",")
            if len(parts) == majority_len:
                cleaned_lines.append(parts)
            elif len(parts) > majority_len:
                nums = list(map(int, parts))
                diff = len(parts) - majority_len
                for _ in range(diff):
                    nums.remove(min(nums))
                cleaned_lines.append(list(map(str, nums)))
            else:
                continue

        csv_str = "\n".join([",".join(line) for line in cleaned_lines])
        return csv_str

    years = [2000, 2005, 2010, 2015]

    calibration_config = config["calibration"]
    folder_path = Path(os.path.join(calibration_config["observed_data"], "crops"))

    # Prepare a container for results
    all_data = []

    for year in years:
        csv_pattern = Path(f"crop_calendar_{year}.csv")
        fp = folder_path / csv_pattern

        csv_str = clean_csv_by_majority_length_in_memory(fp)
        df = pd.read_csv(StringIO(csv_str))

        # Take the mean across rows => mean of all runs
        mean_series = df.mean()

        # Put that mean into a single-row DataFrame
        # (columns become 0,1,2,... if no header in CSV)
        mean_df = pd.DataFrame([mean_series.values], columns=mean_series.index)

        # Add a time column (or rename if you prefer 'Year')
        mean_df["time"] = parse_water_year(year)

        # Append to the list of yearly results
        all_data.append(mean_df)

    # Combine all years into one DataFrame
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    final_df = pd.DataFrame(final_df)

    final_df.set_index("time", inplace=True)

    final_pivot = final_df.pivot_table(
        index=["time"],
        aggfunc="first",  # or pivot columns if needed
    ).sort_index(level="time")

    final_pivot.columns = final_pivot.columns.astype(int)

    return final_pivot


def get_simulated_crops(run_directory, individual, config, observed_pivot):
    start_time = observed_pivot.index.get_level_values("time").min()
    end_time = observed_pivot.index.get_level_values("time").max()
    fp_simulated = Path(os.path.join(run_directory, config["calibration"]["scenario"]))

    crop_calendar_simulated = load_ds(
        fp_simulated,
        "crop_calendar",
        start_time=start_time,
        end_time=end_time,
    )

    field_size_per_farmer = load_ds(
        fp_simulated,
        "field_size_per_farmer",
        start_time=config["calibration"]["start_time"],
        end_time=config["calibration"]["end_time"],
    ).values

    times = (
        crop_calendar_simulated.time.values
    )  # or crop_calendar_simulated.coords['time'].values
    results = []

    for i, current_time in enumerate(times):
        # crop_calendar at this time step, shape: (agents,)
        crops_for_this_timestep = crop_calendar_simulated.values[i, :]

        # Compute unique crop IDs and the bincount index
        unique_crops, inverse = np.unique(crops_for_this_timestep, return_inverse=True)

        # Sum field sizes for each unique crop
        # field_size_per_farmer should have shape: (agents,) or broadcastable
        area_per_crop = np.bincount(inverse, weights=field_size_per_farmer)

        # Convert to a dict: {crop_id: total_area, ..., 'time': current_time}
        row_dict = dict(zip(unique_crops, area_per_crop))
        row_dict["time"] = current_time

        # Append this dict to the results list
        results.append(row_dict)

    df_area_by_crop = pd.DataFrame(results)
    # Make 'time' the index
    df_area_by_crop.set_index("time", inplace=True)

    final_pivot = df_area_by_crop.pivot_table(
        index=["time"],
        aggfunc="first",  # or pivot columns if needed
    ).sort_index(level="time")

    final_pivot.columns = final_pivot.columns.astype(int)

    return final_pivot


def get_crops_KGE(run_directory, individual, config):
    # Get observed data
    observed_df = get_observed_crops(run_directory, individual, config)
    simulated_df = get_simulated_crops(run_directory, individual, config, observed_df)

    # Filter both DataFrames to keep ccommon dates
    common_index = simulated_df.index.intersection(observed_df.index)
    simulated_filtered = simulated_df.loc[common_index]
    observed_filtered = observed_df.loc[common_index]

    common_columns = simulated_filtered.columns.intersection(observed_filtered.columns)
    simulated_filtered = simulated_filtered[common_columns]
    observed_filtered = observed_filtered[common_columns]

    kge_results = {}
    for crop_col in common_columns:
        s = simulated_filtered[crop_col].values
        o = observed_filtered[crop_col].values
        kge_results[crop_col] = KGE_calculation(s, o)

    kge_values = [-2.5 if np.isnan(val) else val for val in kge_results.values()]
    if len(kge_values) > 0:
        kge = np.mean(kge_values)
    else:
        kge = np.nan

    print("run_id: " + str(individual.label) + ", KGE_crops: " + "{0:.3f}".format(kge))
    with open(
        os.path.join(config["calibration"]["path"], "KGE_crops_log.csv"), "a"
    ) as myfile:
        myfile.write(str(individual.label) + "," + str(kge) + "\n")

    return kge


def export_front_history(config, ngen, effmax, effmin, effstd, effavg):
    print(">> Saving optimization history (front_history.csv)")
    front_history = {}
    for i, calibration_value in enumerate(config["calibration"]["calibration_targets"]):
        front_history.update(
            {
                (
                    calibration_value,
                    "effmax_R",
                ): effmax[:, i],
                (calibration_value, "effmin_R"): effmin[:, i],
                (calibration_value, "effstd_R"): effstd[:, i],
                (calibration_value, "effavg_R"): effavg[:, i],
            }
        )
    front_history = pd.DataFrame(front_history, index=list(range(ngen)))
    front_history.to_excel(
        os.path.join(config["calibration"]["path"], "front_history.xlsx")
    )


@handle_ctrl_c
def run_model(individual, config, gauges, observed_streamflow):
    """
    This function takes an individual from the population and runs the model
    with the corresponding parameters in a subfolder. Then it returns the
    fitness scores (KGE, irrigation wells score, etc.) for that run.
    """
    os.makedirs(config["calibration"]["path"], exist_ok=True)
    runs_path = os.path.join(config["calibration"]["path"], "runs")
    os.makedirs(runs_path, exist_ok=True)
    logs_path = os.path.join(config["calibration"]["path"], "logs")
    os.makedirs(logs_path, exist_ok=True)

    run_directory = os.path.join(runs_path, individual.label)
    spinup_done_path = os.path.join(run_directory, "spinup_done.txt")
    run_done_path = os.path.join(run_directory, "done.txt")

    if os.path.isdir(run_directory):
        # If "done.txt" is present, we do not re-run
        runmodel = not os.path.exists(run_done_path)
    else:
        runmodel = True

    spinup_completed = os.path.exists(spinup_done_path)

    if runmodel:
        individual_parameter_ratio = individual.tolist()
        assert (np.array(individual_parameter_ratio) >= 0).all() and (
            np.array(individual_parameter_ratio) <= 1
        ).all()

        # Create a dictionary of the individual's parameters
        individual_parameters = {}
        for i, parameter_data in enumerate(
            config["calibration"]["parameters"].values()
        ):
            individual_parameters[parameter_data["variable"]] = parameter_data[
                "min"
            ] + individual_parameter_ratio[i] * (
                parameter_data["max"] - parameter_data["min"]
            )

        while True:
            os.makedirs(run_directory, exist_ok=True)
            template = deepcopy(config)

            template["general"]["report_folder"] = run_directory
            template["general"]["name"] = config["calibration"]["scenario"]
            template["general"]["initial_conditions_folder"] = os.path.join(
                run_directory, "initial"
            )
            template["general"]["spinup_time"] = config["calibration"]["spinup_time"]
            template["general"]["start_time"] = config["calibration"]["start_time"]
            template["general"]["end_time"] = config["calibration"]["end_time"]

            template["report"] = {}
            template["report_cwatm"] = {}
            template.update(config["calibration"]["target_variables"])

            # Fill in the individual's parameters
            for parameter, value in individual_parameters.items():
                multi_set(template, value, *parameter.split("."))

            config_path = os.path.join(run_directory, "config.yml")
            with open(config_path, "w") as f:
                yaml.dump(template, f)

            lock.acquire()
            if current_gpu_use_count.value < n_gpu_spots:
                use_gpu = int(
                    current_gpu_use_count.value
                    / config["calibration"]["DEAP"]["models_per_gpu"]
                )
                current_gpu_use_count.value += 1
                print(
                    f"Using 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}"
                )
            else:
                use_gpu = False
                print(
                    f"Not using GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}"
                )
            lock.release()

            def run_model_scenario(run_command):
                """
                Run the shell command for spinup or run scenario,
                redirecting stdout/stderr to their own log files.
                """
                env = os.environ.copy()
                # Already set globally, but we can re-ensure here:
                # env["GFORTRAN_UNBUFFERED_ALL"] = "1"
                # env["OMP_NUM_THREADS"] = "1"

                conda_env_name = "geb_p2"
                cli_py_path = os.path.join(os.environ.get("GEB_PACKAGE_DIR"), "cli.py")
                conda_activate = os.path.join(
                    "/scistor/ivm/mka483/miniconda3", "bin", "activate"
                )

                # Construct the command
                # Example: GFORTRAN_UNBUFFERED_ALL=1 OMP_NUM_THREADS=1 source activate ...
                command = (
                    f"source {conda_activate} {conda_env_name} && "
                    f"{sys.executable} {cli_py_path} {run_command} --config {config_path}"
                )

                if use_gpu is not False:
                    command += f" --GPU --gpu_device {use_gpu}"

                print("Executing command:", command, flush=True)

                max_retries = 10000
                retries = 0

                while retries <= max_retries:
                    # Redirect to dedicated files
                    out_file_path = os.path.join(
                        logs_path, f"model_out_{run_command}_{individual.label}.txt"
                    )
                    err_file_path = os.path.join(
                        logs_path, f"model_err_{run_command}_{individual.label}.txt"
                    )
                    with (
                        open(out_file_path, "w") as out_file,
                        open(err_file_path, "w") as err_file,
                    ):
                        p = Popen(
                            command,
                            stdout=out_file,
                            stderr=err_file,
                            shell=True,
                            executable="/bin/bash",
                            env=env,
                        )
                        p.wait()

                    if p.returncode == 0:
                        return p.returncode  # Success
                    elif p.returncode == 1:
                        return p.returncode  # Failure
                    elif p.returncode == 2:
                        retries += 1
                        if retries > max_retries:
                            break
                        print(
                            f"Return code 2 received. Retrying {retries}/{max_retries}..."
                        )
                        time.sleep(1)
                        continue
                    elif p.returncode == 66:
                        retries += 1
                        if retries > max_retries:
                            break
                        print(
                            f"Return code 66 received. Retrying {retries}/{max_retries}..."
                        )
                        time.sleep(1)
                        continue
                    else:
                        timestamp = datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        log_filename = os.path.join(
                            logs_path, f"log_error_{run_command}_{individual.label}.txt"
                        )
                        with open(log_filename, "w") as f:
                            content = (
                                f"Timestamp: {timestamp}\n"
                                f"Process ID: {os.getpid()}\n"
                                f"Command: {command}\n\n"
                                f"Return code: {p.returncode}\n"
                                f"Traceback:\n{traceback.format_exc()}"
                            )
                            f.write(content)
                        raise ValueError(
                            f"Return code was {p.returncode}. See log file {log_filename} for details."
                        )
                raise ValueError(
                    f"Return code 2/66 received {max_retries} times. See log file for details."
                )

            if not spinup_completed:
                template["general"]["export_inital_on_spinup"] = True
                with open(config_path, "w") as f:
                    yaml.dump(template, f)

                return_code = run_model_scenario("spinup")
                if return_code == 0:
                    with open(spinup_done_path, "w") as f:
                        f.write("spinup done")
                    spinup_completed = True
                    with open(config_path, "r") as f:
                        template = yaml.safe_load(f)
                    template["general"]["import_inital"] = True
                    if "export_inital_on_spinup" in template["general"]:
                        del template["general"]["export_inital_on_spinup"]
                    with open(config_path, "w") as f:
                        yaml.dump(template, f)
                else:
                    if use_gpu is not False:
                        lock.acquire()
                        current_gpu_use_count.value -= 1
                        lock.release()
                        print(
                            f"Released 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}"
                        )
                    break
            else:
                # If spinup is already done, ensure config has import_inital
                if not os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        template = yaml.safe_load(f)
                    template["general"]["import_inital"] = True
                    if "export_inital_on_spinup" in template["general"]:
                        del template["general"]["export_inital_on_spinup"]
                    with open(config_path, "w") as f:
                        yaml.dump(template, f)

            return_code = run_model_scenario("run")
            if return_code == 0:
                if use_gpu is not False:
                    lock.acquire()
                    current_gpu_use_count.value -= 1
                    lock.release()
                    print(
                        f"Released 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}"
                    )
                with open(run_done_path, "w") as f:
                    f.write("done")
                break
            else:
                if use_gpu is not False:
                    lock.acquire()
                    current_gpu_use_count.value -= 1
                    lock.release()
                    print(
                        f"Released 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}"
                    )
                break

    # Now gather the scores:
    scores = []
    for score in config["calibration"]["calibration_targets"]:
        if score == "KGE_discharge":
            scores.append(
                get_KGE_discharge(
                    run_directory, individual, config, gauges, observed_streamflow
                )
            )
        if score == "KGE_water_use":
            scores.append(get_water_use_score(run_directory, individual, config))
        if score == "KGE_crops":
            scores.append(get_crops_KGE(run_directory, individual, config))
        if score == "KGE_irrigation_method":
            scores.append(
                get_irrigation_method_score(run_directory, individual, config)
            )
        if score == "KGE_yield_ratio":
            scores.append(get_KGE_yield_ratio(run_directory, individual, config))
        if score == "KGE_water_price":
            scores.append(KGE_water_price(run_directory, individual, config))
        if score == "KGE_region_diversion":
            scores.append(KGE_region_diversion(run_directory, individual, config))

    return tuple(scores)


def init_pool(manager_current_gpu_use_count, manager_lock, gpus, models_per_gpu):
    """
    Initialize the global variables for the process pool.
    """
    global ctrl_c_entered
    global default_sigint_handler
    ctrl_c_entered = False
    default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)

    global lock
    global current_gpu_use_count
    global n_gpu_spots
    n_gpu_spots = gpus * models_per_gpu
    lock = manager_lock
    current_gpu_use_count = manager_current_gpu_use_count


def calibrate(config, working_directory):
    calibration_config = config["calibration"]

    use_multiprocessing = calibration_config["DEAP"]["use_multiprocessing"]
    select_best_n_individuals = calibration_config["DEAP"]["select_best"]

    ngen = calibration_config["DEAP"]["ngen"]
    mu = calibration_config["DEAP"]["mu"]
    lambda_ = calibration_config["DEAP"]["lambda_"]
    config["calibration"]["scenario"] = calibration_config["scenario"]

    gauges = [tuple(g) for g in config["general"]["gauges"]]
    observed_streamflow = {}
    for gauge in gauges:
        streamflow_path = os.path.join(
            calibration_config["observed_data"],
            "streamflow",
            f"{gauge[0]} {gauge[1]}.csv",
        )
        streamflow_data = pd.read_csv(
            streamflow_path, sep=",", parse_dates=False, index_col=None
        )
        df = streamflow_data.reset_index(drop=True)
        df.columns = df.iloc[1]
        df = df.drop([0, 1]).reset_index(drop=True)
        df["Date"] = pd.to_datetime(
            df["Date"], errors="coerce", infer_datetime_format=True
        )
        df = df.dropna(subset=["Date"])
        df["Discharge (ML/Day)"] = pd.to_numeric(
            df["Discharge (ML/Day)"], errors="coerce"
        )
        df = df.dropna(subset=["Discharge (ML/Day)"])
        df = df[["Date", "Discharge (ML/Day)"]].rename(
            columns={"Date": "date", "Discharge (ML/Day)": "flow"}
        )
        df["flow"] = df["flow"] * (1000.0 / 86400.0)  # ML/day to m³/s
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df = df.set_index(pd.to_datetime(df["date"]))
        df.index.name = "time"
        observed_streamflow[gauge] = df["flow"]
        observed_streamflow[gauge].name = "observed"

    # Create DEAP classes
    creator.create(
        "FitnessMulti",
        base.Fitness,
        weights=tuple(config["calibration"]["calibration_targets"].values()),
    )
    creator.create(
        "Individual", array.array, typecode="d", fitness=creator.FitnessMulti
    )

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("select", tools.selBest)
    toolbox.register(
        "Individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        len(calibration_config["parameters"]),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

    def checkBounds(min_val, max_val):
        def decorator(func):
            def wrappper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] > max_val:
                            child[i] = max_val
                        elif child[i] < min_val:
                            child[i] = min_val
                return offspring

            return wrappper

        return decorator

    partial_run_model = partial(
        run_model, config=config, gauges=gauges, observed_streamflow=observed_streamflow
    )
    toolbox.register("evaluate", partial_run_model)
    toolbox.register("mate", tools.cxBlend, alpha=0.15)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
    toolbox.register("select", tools.selNSGA2)
    toolbox.decorate("mate", checkBounds(0, 1))
    toolbox.decorate("mutate", checkBounds(0, 1))

    history = tools.History()

    if use_multiprocessing:
        manager = multiprocessing.Manager()
        current_gpu_use_count = manager.Value("i", 0)
        manager_lock = manager.Lock()
        pool_size = int(os.getenv("SLURM_CPUS_PER_TASK") or 3)
        print(f"Pool size: {pool_size}")

        # Ignore Ctrl+C in parent, let worker procs handle
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = multiprocessing.Pool(
            processes=pool_size,
            initializer=init_pool,
            initargs=(
                current_gpu_use_count,
                manager_lock,
                calibration_config["gpus"],
                calibration_config["models_per_gpu"]
                if "models_per_gpu" in calibration_config
                else 1,
            ),
        )
        toolbox.register("map", pool.map)
    else:
        # Single-threaded fallback
        manager = multiprocessing.Manager()
        current_gpu_use_count = manager.Value("i", 0)
        manager_lock = manager.Lock()
        init_pool(
            current_gpu_use_count,
            manager_lock,
            calibration_config["gpus"],
            calibration_config.get("models_per_gpu", 1),
        )

    cxpb = 0.7
    mutpb = 0.3
    assert cxpb + mutpb == 1, "cxpb + mutpb must sum to 1."

    effmax = np.full((ngen, len(config["calibration"]["calibration_targets"])), np.nan)
    effmin = np.full((ngen, len(config["calibration"]["calibration_targets"])), np.nan)
    effavg = np.full((ngen, len(config["calibration"]["calibration_targets"])), np.nan)
    effstd = np.full((ngen, len(config["calibration"]["calibration_targets"])), np.nan)

    checkpoint = os.path.join(config["calibration"]["path"], "checkpoint.pkl")
    if os.path.exists(checkpoint):
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
            population = cp["population"]
            start_gen = cp["generation"]
            random.setstate(cp["rndstate"])
            if start_gen > 0:
                offspring = cp.get("offspring", [])
            pareto_front = cp["pareto_front"]
    else:
        os.makedirs(config["calibration"]["path"], exist_ok=True)
        start_gen = 0
        population = toolbox.population(n=mu)
        for i, ind in enumerate(population):
            ind.label = str(start_gen % 1000).zfill(2) + "_" + str(i % 1000).zfill(3)
        pareto_front = tools.ParetoFront()
        history.update(population)

    for generation in range(start_gen, ngen):
        if generation == 0:
            cp = dict(
                population=population,
                generation=generation,
                rndstate=random.getstate(),
                pareto_front=pareto_front,
            )
        else:
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
            for i, child in enumerate(offspring):
                child.label = (
                    str(generation % 1000).zfill(2) + "_" + str(i % 1000).zfill(3)
                )
            cp = dict(
                population=population,
                generation=generation,
                rndstate=random.getstate(),
                offspring=offspring,
                pareto_front=pareto_front,
            )

        with open(checkpoint, "wb") as cp_file:
            pickle.dump(cp, cp_file)

        if generation == 0:
            individuals_to_evaluate = [
                ind for ind in population if not ind.fitness.valid
            ]
        else:
            individuals_to_evaluate = [
                ind for ind in offspring if not ind.fitness.valid
            ]

        fitnesses = list(toolbox.map(toolbox.evaluate, individuals_to_evaluate))
        if any(map(lambda x: isinstance(x, KeyboardInterrupt), fitnesses)):
            raise KeyboardInterrupt

        for ind, fit in zip(individuals_to_evaluate, fitnesses):
            ind.fitness.values = fit

        if generation == 0:
            pareto_front.update(population)
            population[:] = toolbox.select(population, lambda_)
        else:
            pareto_front.update(offspring)
            population[:] = toolbox.select(
                population + offspring, select_best_n_individuals
            )

        history.update(population)

        # Gather objective stats from Pareto front
        for ii in range(len(config["calibration"]["calibration_targets"])):
            effmax[generation, ii] = np.amax(
                [pf.fitness.values[ii] for pf in pareto_front]
            )
            effmin[generation, ii] = np.amin(
                [pf.fitness.values[ii] for pf in pareto_front]
            )
            effavg[generation, ii] = np.average(
                [pf.fitness.values[ii] for pf in pareto_front]
            )
            effstd[generation, ii] = np.std(
                [pf.fitness.values[ii] for pf in pareto_front]
            )

    if use_multiprocessing:
        pool.close()

    global ctrl_c_entered
    global default_sigint_handler
    ctrl_c_entered = False
    default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)

    export_front_history(config, ngen, effmax, effmin, effstd, effavg)
