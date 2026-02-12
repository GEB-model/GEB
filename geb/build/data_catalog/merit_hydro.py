"""Utilities to download MERIT Hydro tiles for a given bounding box.

This module provides a streaming downloader that extracts only the needed
GeoTIFF tiles from the remote 30x30-degree tar packages hosted by MERIT Hydro.
It avoids saving the full tar archives to disk by iterating over the HTTP
response stream and writing only the requested members.

Notes:
    - MERIT Hydro distributes 5x5-degree tiles grouped in 30x30-degree tar files.
      Tile filenames follow the pattern "n30w120_elv.tif" where the lat/lon refer
      to the lower-left corner of the tile in degrees. Package tar files follow
      the pattern "elv_n30w120.tar" where the coordinates indicate the lower-left
      corner of the 30x30 group. See the MERIT Hydro documentation for details.
    - Coverage spans from S60 to N90. Some packages or tiles are fully ocean and
      therefore do not exist. This module distinguishes between "missing because
      ocean/not provided" and actual download errors.

"""

from __future__ import annotations

import os
import tarfile
import time
from pathlib import Path
from typing import IO, Any, Iterable

import numpy as np
import requests
import rioxarray as rxr
import xarray as xr
from requests.auth import HTTPBasicAuth
from rioxarray import merge
from tqdm import tqdm

from geb.workflows.raster import convert_nodata

from .base import Adapter

# MERIT Hydro is only available over land, so not all tiles exist. This is a set of
# all available 5x5-degree tile names which we use to check if the tile
# should exist before attempting to read and if that fails download it.
available_tiles: set[str] = {
    "n00e005",
    "n00e010",
    "n00e015",
    "n00e020",
    "n00e025",
    "n00e030",
    "n00e035",
    "n00e040",
    "n00e045",
    "n00e070",
    "n00e095",
    "n00e100",
    "n00e105",
    "n00e110",
    "n00e115",
    "n00e120",
    "n00e125",
    "n00e130",
    "n00e150",
    "n00e165",
    "n00e170",
    "n00w005",
    "n00w010",
    "n00w050",
    "n00w055",
    "n00w060",
    "n00w065",
    "n00w070",
    "n00w075",
    "n00w080",
    "n00w085",
    "n00w090",
    "n00w095",
    "n00w160",
    "n00w165",
    "n00w180",
    "n05e000",
    "n05e005",
    "n05e010",
    "n05e015",
    "n05e020",
    "n05e025",
    "n05e030",
    "n05e035",
    "n05e040",
    "n05e045",
    "n05e050",
    "n05e070",
    "n05e075",
    "n05e080",
    "n05e090",
    "n05e095",
    "n05e100",
    "n05e105",
    "n05e110",
    "n05e115",
    "n05e120",
    "n05e125",
    "n05e130",
    "n05e135",
    "n05e140",
    "n05e145",
    "n05e150",
    "n05e155",
    "n05e160",
    "n05e165",
    "n05e170",
    "n05w005",
    "n05w010",
    "n05w015",
    "n05w055",
    "n05w060",
    "n05w065",
    "n05w070",
    "n05w075",
    "n05w080",
    "n05w085",
    "n05w090",
    "n05w165",
    "n10e000",
    "n10e005",
    "n10e010",
    "n10e015",
    "n10e020",
    "n10e025",
    "n10e030",
    "n10e035",
    "n10e040",
    "n10e045",
    "n10e050",
    "n10e070",
    "n10e075",
    "n10e080",
    "n10e090",
    "n10e095",
    "n10e100",
    "n10e105",
    "n10e110",
    "n10e115",
    "n10e120",
    "n10e125",
    "n10e135",
    "n10e140",
    "n10e145",
    "n10e160",
    "n10e165",
    "n10e170",
    "n10w005",
    "n10w010",
    "n10w015",
    "n10w020",
    "n10w025",
    "n10w060",
    "n10w065",
    "n10w070",
    "n10w075",
    "n10w080",
    "n10w085",
    "n10w090",
    "n10w095",
    "n10w110",
    "n15e000",
    "n15e005",
    "n15e010",
    "n15e015",
    "n15e020",
    "n15e025",
    "n15e030",
    "n15e035",
    "n15e040",
    "n15e045",
    "n15e050",
    "n15e055",
    "n15e070",
    "n15e075",
    "n15e080",
    "n15e085",
    "n15e090",
    "n15e095",
    "n15e100",
    "n15e105",
    "n15e110",
    "n15e115",
    "n15e120",
    "n15e145",
    "n15e165",
    "n15w005",
    "n15w010",
    "n15w015",
    "n15w020",
    "n15w025",
    "n15w030",
    "n15w065",
    "n15w070",
    "n15w075",
    "n15w080",
    "n15w085",
    "n15w090",
    "n15w095",
    "n15w100",
    "n15w105",
    "n15w110",
    "n15w115",
    "n15w155",
    "n15w160",
    "n15w170",
    "n20e000",
    "n20e005",
    "n20e010",
    "n20e015",
    "n20e020",
    "n20e025",
    "n20e030",
    "n20e035",
    "n20e040",
    "n20e045",
    "n20e050",
    "n20e055",
    "n20e065",
    "n20e070",
    "n20e075",
    "n20e080",
    "n20e085",
    "n20e090",
    "n20e095",
    "n20e100",
    "n20e105",
    "n20e110",
    "n20e115",
    "n20e120",
    "n20e125",
    "n20e130",
    "n20e135",
    "n20e140",
    "n20e145",
    "n20e150",
    "n20w005",
    "n20w010",
    "n20w015",
    "n20w020",
    "n20w075",
    "n20w080",
    "n20w085",
    "n20w090",
    "n20w095",
    "n20w100",
    "n20w105",
    "n20w110",
    "n20w115",
    "n20w120",
    "n20w160",
    "n20w165",
    "n20w170",
    "n25e000",
    "n25e005",
    "n25e010",
    "n25e015",
    "n25e020",
    "n25e025",
    "n25e030",
    "n25e035",
    "n25e040",
    "n25e045",
    "n25e050",
    "n25e055",
    "n25e060",
    "n25e065",
    "n25e070",
    "n25e075",
    "n25e080",
    "n25e085",
    "n25e090",
    "n25e095",
    "n25e100",
    "n25e105",
    "n25e110",
    "n25e115",
    "n25e120",
    "n25e125",
    "n25e130",
    "n25e140",
    "n25w005",
    "n25w010",
    "n25w015",
    "n25w020",
    "n25w080",
    "n25w085",
    "n25w090",
    "n25w095",
    "n25w100",
    "n25w105",
    "n25w110",
    "n25w115",
    "n25w120",
    "n25w175",
    "n25w180",
    "n30e000",
    "n30e005",
    "n30e010",
    "n30e015",
    "n30e020",
    "n30e025",
    "n30e030",
    "n30e035",
    "n30e040",
    "n30e045",
    "n30e050",
    "n30e055",
    "n30e060",
    "n30e065",
    "n30e070",
    "n30e075",
    "n30e080",
    "n30e085",
    "n30e090",
    "n30e095",
    "n30e100",
    "n30e105",
    "n30e110",
    "n30e115",
    "n30e120",
    "n30e125",
    "n30e130",
    "n30e135",
    "n30e140",
    "n30w005",
    "n30w010",
    "n30w020",
    "n30w065",
    "n30w080",
    "n30w085",
    "n30w090",
    "n30w095",
    "n30w100",
    "n30w105",
    "n30w110",
    "n30w115",
    "n30w120",
    "n30w125",
    "n35e000",
    "n35e005",
    "n35e010",
    "n35e015",
    "n35e020",
    "n35e025",
    "n35e030",
    "n35e035",
    "n35e040",
    "n35e045",
    "n35e050",
    "n35e055",
    "n35e060",
    "n35e065",
    "n35e070",
    "n35e075",
    "n35e080",
    "n35e085",
    "n35e090",
    "n35e095",
    "n35e100",
    "n35e105",
    "n35e110",
    "n35e115",
    "n35e120",
    "n35e125",
    "n35e130",
    "n35e135",
    "n35e140",
    "n35w005",
    "n35w010",
    "n35w025",
    "n35w030",
    "n35w035",
    "n35w075",
    "n35w080",
    "n35w085",
    "n35w090",
    "n35w095",
    "n35w100",
    "n35w105",
    "n35w110",
    "n35w115",
    "n35w120",
    "n35w125",
    "n40e000",
    "n40e005",
    "n40e010",
    "n40e015",
    "n40e020",
    "n40e025",
    "n40e030",
    "n40e035",
    "n40e040",
    "n40e045",
    "n40e050",
    "n40e055",
    "n40e060",
    "n40e065",
    "n40e070",
    "n40e075",
    "n40e080",
    "n40e085",
    "n40e090",
    "n40e095",
    "n40e100",
    "n40e105",
    "n40e110",
    "n40e115",
    "n40e120",
    "n40e125",
    "n40e130",
    "n40e135",
    "n40e140",
    "n40e145",
    "n40w005",
    "n40w010",
    "n40w060",
    "n40w065",
    "n40w070",
    "n40w075",
    "n40w080",
    "n40w085",
    "n40w090",
    "n40w095",
    "n40w100",
    "n40w105",
    "n40w110",
    "n40w115",
    "n40w120",
    "n40w125",
    "n45e000",
    "n45e005",
    "n45e010",
    "n45e015",
    "n45e020",
    "n45e025",
    "n45e030",
    "n45e035",
    "n45e040",
    "n45e045",
    "n45e050",
    "n45e055",
    "n45e060",
    "n45e065",
    "n45e070",
    "n45e075",
    "n45e080",
    "n45e085",
    "n45e090",
    "n45e095",
    "n45e100",
    "n45e105",
    "n45e110",
    "n45e115",
    "n45e120",
    "n45e125",
    "n45e130",
    "n45e135",
    "n45e140",
    "n45e145",
    "n45e150",
    "n45e155",
    "n45w005",
    "n45w010",
    "n45w055",
    "n45w060",
    "n45w065",
    "n45w070",
    "n45w075",
    "n45w080",
    "n45w085",
    "n45w090",
    "n45w095",
    "n45w100",
    "n45w105",
    "n45w110",
    "n45w115",
    "n45w120",
    "n45w125",
    "n45w130",
    "n50e000",
    "n50e005",
    "n50e010",
    "n50e015",
    "n50e020",
    "n50e025",
    "n50e030",
    "n50e035",
    "n50e040",
    "n50e045",
    "n50e050",
    "n50e055",
    "n50e060",
    "n50e065",
    "n50e070",
    "n50e075",
    "n50e080",
    "n50e085",
    "n50e090",
    "n50e095",
    "n50e100",
    "n50e105",
    "n50e110",
    "n50e115",
    "n50e120",
    "n50e125",
    "n50e130",
    "n50e135",
    "n50e140",
    "n50e150",
    "n50e155",
    "n50e160",
    "n50e165",
    "n50e170",
    "n50e175",
    "n50w005",
    "n50w010",
    "n50w015",
    "n50w060",
    "n50w065",
    "n50w070",
    "n50w075",
    "n50w080",
    "n50w085",
    "n50w090",
    "n50w095",
    "n50w100",
    "n50w105",
    "n50w110",
    "n50w115",
    "n50w120",
    "n50w125",
    "n50w130",
    "n50w135",
    "n50w160",
    "n50w165",
    "n50w170",
    "n50w175",
    "n50w180",
    "n55e000",
    "n55e005",
    "n55e010",
    "n55e015",
    "n55e020",
    "n55e025",
    "n55e030",
    "n55e035",
    "n55e040",
    "n55e045",
    "n55e050",
    "n55e055",
    "n55e060",
    "n55e065",
    "n55e070",
    "n55e075",
    "n55e080",
    "n55e085",
    "n55e090",
    "n55e095",
    "n55e100",
    "n55e105",
    "n55e110",
    "n55e115",
    "n55e120",
    "n55e125",
    "n55e130",
    "n55e135",
    "n55e140",
    "n55e145",
    "n55e150",
    "n55e155",
    "n55e160",
    "n55e165",
    "n55e170",
    "n55w005",
    "n55w010",
    "n55w015",
    "n55w045",
    "n55w050",
    "n55w060",
    "n55w065",
    "n55w070",
    "n55w075",
    "n55w080",
    "n55w085",
    "n55w090",
    "n55w095",
    "n55w100",
    "n55w105",
    "n55w110",
    "n55w115",
    "n55w120",
    "n55w125",
    "n55w130",
    "n55w135",
    "n55w140",
    "n55w145",
    "n55w150",
    "n55w155",
    "n55w160",
    "n55w165",
    "n55w170",
    "n55w175",
    "n60e000",
    "n60e005",
    "n60e010",
    "n60e015",
    "n60e020",
    "n60e025",
    "n60e030",
    "n60e035",
    "n60e040",
    "n60e045",
    "n60e050",
    "n60e055",
    "n60e060",
    "n60e065",
    "n60e070",
    "n60e075",
    "n60e080",
    "n60e085",
    "n60e090",
    "n60e095",
    "n60e100",
    "n60e105",
    "n60e110",
    "n60e115",
    "n60e120",
    "n60e125",
    "n60e130",
    "n60e135",
    "n60e140",
    "n60e145",
    "n60e150",
    "n60e155",
    "n60e160",
    "n60e165",
    "n60e170",
    "n60e175",
    "n60w005",
    "n60w010",
    "n60w015",
    "n60w020",
    "n60w025",
    "n60w040",
    "n60w045",
    "n60w050",
    "n60w055",
    "n60w065",
    "n60w070",
    "n60w075",
    "n60w080",
    "n60w085",
    "n60w090",
    "n60w095",
    "n60w100",
    "n60w105",
    "n60w110",
    "n60w115",
    "n60w120",
    "n60w125",
    "n60w130",
    "n60w135",
    "n60w140",
    "n60w145",
    "n60w150",
    "n60w155",
    "n60w160",
    "n60w165",
    "n60w170",
    "n60w175",
    "n60w180",
    "n65e010",
    "n65e015",
    "n65e020",
    "n65e025",
    "n65e030",
    "n65e035",
    "n65e040",
    "n65e045",
    "n65e050",
    "n65e055",
    "n65e060",
    "n65e065",
    "n65e070",
    "n65e075",
    "n65e080",
    "n65e085",
    "n65e090",
    "n65e095",
    "n65e100",
    "n65e105",
    "n65e110",
    "n65e115",
    "n65e120",
    "n65e125",
    "n65e130",
    "n65e135",
    "n65e140",
    "n65e145",
    "n65e150",
    "n65e155",
    "n65e160",
    "n65e165",
    "n65e170",
    "n65e175",
    "n65w015",
    "n65w020",
    "n65w025",
    "n65w030",
    "n65w035",
    "n65w040",
    "n65w045",
    "n65w050",
    "n65w055",
    "n65w065",
    "n65w070",
    "n65w075",
    "n65w080",
    "n65w085",
    "n65w090",
    "n65w095",
    "n65w100",
    "n65w105",
    "n65w110",
    "n65w115",
    "n65w120",
    "n65w125",
    "n65w130",
    "n65w135",
    "n65w140",
    "n65w145",
    "n65w150",
    "n65w155",
    "n65w160",
    "n65w165",
    "n65w170",
    "n65w175",
    "n65w180",
    "n70e015",
    "n70e020",
    "n70e025",
    "n70e030",
    "n70e050",
    "n70e055",
    "n70e060",
    "n70e065",
    "n70e070",
    "n70e075",
    "n70e080",
    "n70e085",
    "n70e090",
    "n70e095",
    "n70e100",
    "n70e105",
    "n70e110",
    "n70e115",
    "n70e120",
    "n70e125",
    "n70e130",
    "n70e135",
    "n70e140",
    "n70e145",
    "n70e150",
    "n70e155",
    "n70e160",
    "n70e165",
    "n70e170",
    "n70e175",
    "n70w010",
    "n70w020",
    "n70w025",
    "n70w030",
    "n70w035",
    "n70w040",
    "n70w045",
    "n70w050",
    "n70w055",
    "n70w060",
    "n70w070",
    "n70w075",
    "n70w080",
    "n70w085",
    "n70w090",
    "n70w095",
    "n70w100",
    "n70w105",
    "n70w110",
    "n70w115",
    "n70w120",
    "n70w125",
    "n70w130",
    "n70w135",
    "n70w145",
    "n70w150",
    "n70w155",
    "n70w160",
    "n70w165",
    "n70w180",
    "n75e010",
    "n75e015",
    "n75e020",
    "n75e025",
    "n75e030",
    "n75e045",
    "n75e050",
    "n75e055",
    "n75e060",
    "n75e065",
    "n75e075",
    "n75e080",
    "n75e085",
    "n75e090",
    "n75e095",
    "n75e100",
    "n75e105",
    "n75e110",
    "n75e135",
    "n75e140",
    "n75e145",
    "n75e150",
    "n75e155",
    "n75w020",
    "n75w025",
    "n75w030",
    "n75w035",
    "n75w040",
    "n75w045",
    "n75w050",
    "n75w055",
    "n75w060",
    "n75w065",
    "n75w070",
    "n75w075",
    "n75w080",
    "n75w085",
    "n75w090",
    "n75w095",
    "n75w100",
    "n75w105",
    "n75w110",
    "n75w115",
    "n75w120",
    "n75w125",
    "n80e010",
    "n80e015",
    "n80e020",
    "n80e025",
    "n80e030",
    "n80e035",
    "n80e040",
    "n80e045",
    "n80e050",
    "n80e055",
    "n80e060",
    "n80e065",
    "n80e075",
    "n80e080",
    "n80e090",
    "n80e095",
    "n80w015",
    "n80w020",
    "n80w025",
    "n80w030",
    "n80w035",
    "n80w040",
    "n80w045",
    "n80w050",
    "n80w055",
    "n80w060",
    "n80w065",
    "n80w070",
    "n80w075",
    "n80w080",
    "n80w085",
    "n80w090",
    "n80w095",
    "n80w100",
    "n80w105",
    "s05e005",
    "s05e010",
    "s05e015",
    "s05e020",
    "s05e025",
    "s05e030",
    "s05e035",
    "s05e040",
    "s05e050",
    "s05e055",
    "s05e070",
    "s05e095",
    "s05e100",
    "s05e105",
    "s05e110",
    "s05e115",
    "s05e120",
    "s05e125",
    "s05e130",
    "s05e135",
    "s05e140",
    "s05e145",
    "s05e150",
    "s05e155",
    "s05e165",
    "s05e170",
    "s05e175",
    "s05w035",
    "s05w040",
    "s05w045",
    "s05w050",
    "s05w055",
    "s05w060",
    "s05w065",
    "s05w070",
    "s05w075",
    "s05w080",
    "s05w085",
    "s05w090",
    "s05w095",
    "s05w155",
    "s05w165",
    "s05w175",
    "s10e010",
    "s10e015",
    "s10e020",
    "s10e025",
    "s10e030",
    "s10e035",
    "s10e045",
    "s10e050",
    "s10e055",
    "s10e070",
    "s10e100",
    "s10e105",
    "s10e110",
    "s10e115",
    "s10e120",
    "s10e125",
    "s10e130",
    "s10e135",
    "s10e140",
    "s10e145",
    "s10e150",
    "s10e155",
    "s10e160",
    "s10e165",
    "s10e175",
    "s10w015",
    "s10w035",
    "s10w040",
    "s10w045",
    "s10w050",
    "s10w055",
    "s10w060",
    "s10w065",
    "s10w070",
    "s10w075",
    "s10w080",
    "s10w085",
    "s10w140",
    "s10w145",
    "s10w155",
    "s10w160",
    "s10w165",
    "s10w175",
    "s15e010",
    "s15e015",
    "s15e020",
    "s15e025",
    "s15e030",
    "s15e035",
    "s15e040",
    "s15e045",
    "s15e050",
    "s15e055",
    "s15e095",
    "s15e105",
    "s15e115",
    "s15e120",
    "s15e125",
    "s15e130",
    "s15e135",
    "s15e140",
    "s15e145",
    "s15e150",
    "s15e155",
    "s15e160",
    "s15e165",
    "s15e170",
    "s15e175",
    "s15w040",
    "s15w045",
    "s15w050",
    "s15w055",
    "s15w060",
    "s15w065",
    "s15w070",
    "s15w075",
    "s15w080",
    "s15w140",
    "s15w145",
    "s15w150",
    "s15w155",
    "s15w165",
    "s15w170",
    "s15w175",
    "s15w180",
    "s20e010",
    "s20e015",
    "s20e020",
    "s20e025",
    "s20e030",
    "s20e035",
    "s20e040",
    "s20e045",
    "s20e050",
    "s20e055",
    "s20e060",
    "s20e115",
    "s20e120",
    "s20e125",
    "s20e130",
    "s20e135",
    "s20e140",
    "s20e145",
    "s20e150",
    "s20e155",
    "s20e160",
    "s20e165",
    "s20e170",
    "s20e175",
    "s20w010",
    "s20w040",
    "s20w045",
    "s20w050",
    "s20w055",
    "s20w060",
    "s20w065",
    "s20w070",
    "s20w075",
    "s20w080",
    "s20w140",
    "s20w145",
    "s20w150",
    "s20w155",
    "s20w160",
    "s20w165",
    "s20w175",
    "s20w180",
    "s25e010",
    "s25e015",
    "s25e020",
    "s25e025",
    "s25e030",
    "s25e035",
    "s25e040",
    "s25e045",
    "s25e055",
    "s25e110",
    "s25e115",
    "s25e120",
    "s25e125",
    "s25e130",
    "s25e135",
    "s25e140",
    "s25e145",
    "s25e150",
    "s25e155",
    "s25e160",
    "s25e165",
    "s25e170",
    "s25w030",
    "s25w045",
    "s25w050",
    "s25w055",
    "s25w060",
    "s25w065",
    "s25w070",
    "s25w075",
    "s25w125",
    "s25w130",
    "s25w135",
    "s25w140",
    "s25w145",
    "s25w150",
    "s25w155",
    "s25w160",
    "s25w175",
    "s25w180",
    "s30e010",
    "s30e015",
    "s30e020",
    "s30e025",
    "s30e030",
    "s30e040",
    "s30e045",
    "s30e110",
    "s30e115",
    "s30e120",
    "s30e125",
    "s30e130",
    "s30e135",
    "s30e140",
    "s30e145",
    "s30e150",
    "s30e165",
    "s30w050",
    "s30w055",
    "s30w060",
    "s30w065",
    "s30w070",
    "s30w075",
    "s30w080",
    "s30w085",
    "s30w110",
    "s30w135",
    "s30w145",
    "s30w180",
    "s35e015",
    "s35e020",
    "s35e025",
    "s35e030",
    "s35e110",
    "s35e115",
    "s35e120",
    "s35e125",
    "s35e130",
    "s35e135",
    "s35e140",
    "s35e145",
    "s35e150",
    "s35e155",
    "s35e170",
    "s35w055",
    "s35w060",
    "s35w065",
    "s35w070",
    "s35w075",
    "s35w080",
    "s35w085",
    "s35w180",
    "s40e075",
    "s40e115",
    "s40e135",
    "s40e140",
    "s40e145",
    "s40e150",
    "s40e170",
    "s40e175",
    "s40w015",
    "s40w060",
    "s40w065",
    "s40w070",
    "s40w075",
    "s45e140",
    "s45e145",
    "s45e165",
    "s45e170",
    "s45e175",
    "s45w010",
    "s45w015",
    "s45w065",
    "s45w070",
    "s45w075",
    "s45w080",
    "s45w180",
    "s50e035",
    "s50e050",
    "s50e065",
    "s50e070",
    "s50e165",
    "s50e170",
    "s50e175",
    "s50w070",
    "s50w075",
    "s50w080",
    "s55e000",
    "s55e065",
    "s55e070",
    "s55e155",
    "s55e165",
    "s55w040",
    "s55w060",
    "s55w065",
    "s55w070",
    "s55w075",
    "s55w080",
    "s60w030",
    "s60w070",
    "s60w075",
}


class _ProgressReader:
    """Wrapper for file-like object to track read progress."""

    def __init__(self, fileobj: IO[bytes], pbar: tqdm) -> None:
        """Initialize the progress reader.

        Args:
            fileobj: File-like object to read from.
            pbar: tqdm progress bar to update.
        """
        self.fileobj: IO[bytes] = fileobj
        self.pbar: tqdm = pbar

    def read(self, size: int = -1) -> bytes:
        """Read data and update progress bar.

        Args:
            size: Read size in bytes. If -1, read all available. Defaults to -1.

        Returns:
            Data read from the file-like object.
        """
        data = self.fileobj.read(size)
        if data:
            self.pbar.update(len(data))
        return data

    def close(self) -> None:
        self.fileobj.close()


class MeritHydro(Adapter):
    """Dataset adapter for MERIT Hydro variables."""

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for a specific MERIT Hydro variable.

        Args:
            variable: MERIT Hydro variable to download ("elv" or "dir").

            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        self.variable = variable
        self._xmin: float | None = None
        self._xmax: float | None = None
        self._ymin: float | None = None
        self._ymax: float | None = None
        self._source_nodata: int | float | bool | None = None
        self._target_nodata: int | float | bool | None = None
        super().__init__(*args, **kwargs)

    @property
    def is_ready(self) -> bool:
        """Check if the data is already downloaded and processed.

        For MERIT Hydro, we always return False because readiness depends
        on the required tiles for a specific bounding box, which are checked
        during fetch().

        Returns:
            Always False to ensure fetch() is called.
        """
        return False

    def _get_latitude_hemisphere_and_degrees(self, lat_deg: int) -> tuple[str, int]:
        """Return hemisphere letter and absolute degrees for latitude.

        Args:
            lat_deg: Integer degrees latitude (lower-left corner of tile) (degrees).

        Returns:
            Tuple of hemisphere letter ("n" or "s") and two-digit absolute degrees.

        Raises:
            ValueError: If latitude is outside valid range [-90, 90].
        """
        if lat_deg < -90 or lat_deg > 90:
            raise ValueError("Latitude must be within [-90, 90] degrees.")
        if lat_deg >= 0:
            return "n", lat_deg
        return "s", -lat_deg

    def _get_longitude_hemisphere_and_degrees(self, lon_deg: int) -> tuple[str, int]:
        """Return hemisphere letter and absolute degrees for longitude.

        Args:
            lon_deg: Integer degrees longitude (lower-left corner of tile) (degrees).

        Returns:
            Tuple of hemisphere letter ("e" or "w") and three-digit absolute degrees.

        Raises:
            ValueError: If longitude is outside valid range [-180, 180].
        """
        if lon_deg < -180 or lon_deg > 180:
            raise ValueError("Longitude must be within [-180, 180] degrees.")
        if lon_deg >= 0:
            return "e", lon_deg
        return "w", -lon_deg

    def _compose_tile_filename(self, lat_ll: int, lon_ll: int) -> str:
        """Compose a 5x5-degree tile filename for a MERIT variable.

        Args:
            lat_ll: Lower-left latitude of tile (integer multiple of 5) (degrees).
            lon_ll: Lower-left longitude of tile (integer multiple of 5) (degrees).

        Returns:
            Tile filename like "n30w120_elv.tif".

        Raises:
            ValueError: If lat_ll or lon_ll are not multiples of 5.
        """
        if lat_ll % 5 != 0 or lon_ll % 5 != 0:
            raise ValueError(
                "lat_ll and lon_ll must be 5-degree aligned (multiples of 5)."
            )
        ns, alat = self._get_latitude_hemisphere_and_degrees(lat_ll)
        ew, alon = self._get_longitude_hemisphere_and_degrees(lon_ll)
        return f"{ns}{alat:02d}{ew}{alon:03d}_{self.variable}.tif"

    def _package_name(self, lat_ll: int, lon_ll: int) -> str:
        """Compose a 30x30-degree package tar filename for a MERIT variable.

        The package is defined by the lower-left corner of the 30-degree grid cell
        that contains the tile lower-left corner.

        Args:
            lat_ll: Lower-left latitude of tile (integer multiple of 5) (degrees).
            lon_ll: Lower-left longitude of tile (integer multiple of 5) (degrees).

        Returns:
            Package tar filename like "elv_n30w120.tar".
        """
        # Floor to 30-degree grid
        lat30 = (lat_ll // 30) * 30 if lat_ll >= 0 else -(((-lat_ll + 29) // 30) * 30)
        lon30 = (lon_ll // 30) * 30 if lon_ll >= 0 else -(((-lon_ll + 29) // 30) * 30)
        ns, alat = self._get_latitude_hemisphere_and_degrees(lat30)
        ew, alon = self._get_longitude_hemisphere_and_degrees(lon30)
        return f"{self.variable}_{ns}{alat:02d}{ew}{alon:03d}.tar"

    def _tiles_for_bbox(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> list[tuple[int, int]]:
        """Compute all 5x5-degree lower-left tile coordinates intersecting a bbox.

        Args:
            xmin: Minimum longitude (degrees).
            xmax: Maximum longitude (degrees). May be less than xmin if crossing dateline is needed (not supported here).
            ymin: Minimum latitude (degrees).
            ymax: Maximum latitude (degrees).

        Returns:
            Sorted list of (lat_ll, lon_ll) integer tuples on 5-degree grid.

        Raises:
            ValueError: If bbox is invalid or crosses the antimeridian.
        """
        if xmax <= xmin:
            raise ValueError("Expected xmax > xmin and bbox not crossing antimeridian.")
        if ymax <= ymin:
            raise ValueError("Expected ymax > ymin.")

        # Clamp to plausible world bounds to avoid generating excessive tiles
        xmin_c = max(-180.0, xmin)
        xmax_c = min(180.0, xmax)
        ymin_c = max(-90.0, ymin)
        ymax_c = min(90.0, ymax)

        # Align to 5-degree grid (lower-left corners)
        def floor5(v: float) -> int:
            vi = int(v // 5 * 5)
            # Adjust for negatives that are exact multiples to ensure lower-left
            if v < 0 and v % 5 == 0:
                return vi
            return vi

        lat_start = floor5(ymin_c)
        lon_start = floor5(xmin_c)

        tiles: list[tuple[int, int]] = []
        lat = lat_start
        while lat < ymax_c:
            lon = lon_start
            while lon < xmax_c:
                # Tile bounds
                t_xmin, t_xmax = lon, lon + 5
                t_ymin, t_ymax = lat, lat + 5
                if not (
                    t_xmin >= xmax_c
                    or t_xmax <= xmin_c
                    or t_ymin >= ymax_c
                    or t_ymax <= ymin_c
                ):
                    tiles.append((lat, lon))
                lon += 5
            lat += 5
        # Unique and sorted for reproducibility
        tiles = sorted(set(tiles))
        return tiles

    def _group_tiles_by_package(
        self, tiles: Iterable[tuple[int, int]]
    ) -> dict[str, list[tuple[int, int]]]:
        """Group tile ll coords by the 30x30 package tar they belong to.

        Args:
            tiles: Iterable of (lat_ll, lon_ll) pairs (degrees).

        Returns:
            Mapping of package tar filename -> list of tile coords in that package.
        """
        groups: dict[str, list[tuple[int, int]]] = {}
        for lat_ll, lon_ll in tiles:
            pkg = self._package_name(lat_ll, lon_ll)
            groups.setdefault(pkg, []).append((lat_ll, lon_ll))
        return groups

    def _package_url(self, package_name: str, base_url: str) -> str:
        """Construct the full URL to a MERIT tar package for the given variable.

        Args:
            package_name: Tar filename produced by _package_name.
            base_url: Base URL for MERIT Hydro downloads.

        Returns:
            Full URL string to the tar file.
        """
        return f"{base_url}/{package_name}"

    def _merge_merit_tiles(self, tile_paths: list[Path]) -> xr.DataArray:
        """Load MERIT Hydro tiles into a single xarray DataArray.

        This function opens the provided GeoTIFF tile files using rioxarray,
        merges them into a single DataArray, and returns it in memory.

        Args:
            tile_paths: List of Paths to GeoTIFF files from download_merit.

        Returns:
            xarray DataArray with merged tiles, preserving CRS and coordinates.
        """
        das: list[xr.DataArray] = []
        for path in tile_paths:
            src = rxr.open_rasterio(path)
            assert isinstance(src, xr.DataArray)
            das.append(src.sel(band=1))

        da: xr.DataArray = merge.merge_arrays(das)
        return da

    def _missing_marker_path(self, tile_name: str) -> Path:
        """Construct path for a missing tile marker file.

        Args:
            tile_name: Name of the tile.

        Returns:
            Path to the marker file.
        """
        root = self.root
        assert root is not None
        return root / f"{tile_name}.missing.txt"

    def fetch(
        self,
        *,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        url: str,
        source_nodata: int | float | bool,
        target_nodata: int | float | bool,
        session: requests.Session | None = None,
        request_timeout_s: float = 60.0,
        attempts: int = 3,
    ) -> MeritHydro:
        """Ensure MERIT Hydro tiles intersecting a bbox are available locally.

        The function first checks for pre-staged 5x5-degree tiles in
        ``{cache_root}/{variable}/`` or ``{cache_root}/``. If tiles are
        missing, it downloads only the required GeoTIFFs for a single MERIT variable
        by streaming the remote 30x30-degree tar packages without saving the tars.
        If a package does not exist (HTTP 404), it is silently skipped. If a needed
        tile is not present inside an existing package (commonly ocean), it is also
        silently skipped. If a tile should be on disk but is missing, it tries to download it.
        Any other error is retried up to ``attempts`` times and
        then raised.

        Authentication:
            Basic auth is required and read from environment variables:
            MERIT_USERNAME and MERIT_PASSWORD.

        Args:
            xmin: Minimum longitude of area of interest (degrees).
            xmax: Maximum longitude of area of interest (degrees).
            ymin: Minimum latitude of area of interest (degrees).
            ymax: Maximum latitude of area of interest (degrees).
            url: Base URL of the MERIT Hydro server.
            source_nodata: Nodata value in the source GeoTIFF files.
            target_nodata: Nodata value to use in the output DataArray.
            session: Optional requests.Session (used in tests or advanced usage).
            request_timeout_s: Timeout per HTTP request (seconds).
            attempts: Number of attempts for transient failures (errors are raised after this many tries).

        Returns:
            The MeritHydro instance.

        Raises:
            ValueError: If inputs are invalid or auth variables are missing.
            RuntimeError: If the HTTP client dependency is not available.
            requests.RequestException: If repeated HTTP errors occur.
            tarfile.ReadError: If tar parsing repeatedly fails.
        """
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._source_nodata = source_nodata
        self._target_nodata = target_nodata

        username = os.getenv("MERIT_USERNAME")
        password = os.getenv("MERIT_PASSWORD")
        if not username or not password:
            raise ValueError(
                "Authentication required: set MERIT_USERNAME and MERIT_PASSWORD in environment."
            )

        tiles: list[tuple[int, int]] = self._tiles_for_bbox(xmin, xmax, ymin, ymax)

        local_tile_dir: Path = self.root / self.variable
        missing_names: set[str] = set()
        for lat_ll, lon_ll in tiles:
            tile_name: str = self._compose_tile_filename(lat_ll, lon_ll)
            if (
                not (self.root / tile_name).exists()
                and not (local_tile_dir / tile_name).exists()
                and tile_name[:7] in available_tiles
            ):
                if not self._missing_marker_path(tile_name).exists():
                    missing_names.add(tile_name)

        if not missing_names:
            return self

        # Prepare HTTP session
        if session is None:
            if requests is None:
                raise RuntimeError(
                    "requests is required for downloading but is not available."
                )
            session = requests.Session()
        auth = HTTPBasicAuth(username, password)

        missing_coords = []
        for lat_ll, lon_ll in tiles:
            if self._compose_tile_filename(lat_ll, lon_ll) in missing_names:
                missing_coords.append((lat_ll, lon_ll))

        groups = self._group_tiles_by_package(missing_coords)

        for package_name, coords in groups.items():
            package_url = self._package_url(package_name, base_url=url)
            needed_names = {
                self._compose_tile_filename(lat, lon) for lat, lon in coords
            }

            # HEAD to detect ocean/no-data packages (404) → skip silently
            try:
                head_resp = session.head(
                    package_url,
                    auth=auth,
                    allow_redirects=True,
                    timeout=request_timeout_s,
                )
            except Exception:
                if attempts <= 1:
                    raise
                retried = 1
                while retried < attempts:
                    time.sleep(min(2.0 * retried, 5.0))
                    try:
                        head_resp = session.head(
                            package_url,
                            auth=auth,
                            allow_redirects=True,
                            timeout=request_timeout_s,
                        )
                        break
                    except Exception:
                        retried += 1
                else:
                    raise

            if head_resp.status_code == 404:
                for tname in needed_names:
                    mm = self._missing_marker_path(tname)
                    mm.write_text(
                        "MERIT Hydro: tile not provided (ocean/no-data).\n",
                        encoding="utf-8",
                    )
                continue
            if head_resp.status_code in (401, 403):
                raise requests.RequestException(
                    f"Unauthorized: HTTP {head_resp.status_code} for {package_url}"
                )
            if head_resp.status_code >= 400:
                raise requests.RequestException(
                    f"HEAD error: HTTP {head_resp.status_code} for {package_url}"
                )

            content_length = int(head_resp.headers.get("Content-Length", 0))
            found_names: set[str] = set()
            for attempt in range(1, attempts + 1):
                try:
                    resp = session.get(
                        package_url, auth=auth, stream=True, timeout=request_timeout_s
                    )
                except Exception:
                    if attempt >= attempts:
                        raise
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue

                if resp.status_code >= 400:
                    resp.close()
                    if attempt >= attempts:
                        raise requests.RequestException(
                            f"GET error: HTTP {resp.status_code} for {package_url}"
                        )
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue

                try:
                    resp.raw.decode_content = True
                    print(f"Downloading and extracting from: {package_url}")
                    with tqdm(
                        total=content_length if content_length > 0 else None,
                        desc=f"Downloading {package_name}",
                        unit="B",
                        unit_scale=True,
                    ) as pbar:
                        progress_reader = _ProgressReader(resp.raw, pbar)
                        with tarfile.open(fileobj=progress_reader, mode="r|*") as tf:  # type: ignore
                            for member in tf:
                                if not member.isreg():
                                    continue
                                mname = Path(member.name).name
                                if mname in needed_names:
                                    ex = tf.extractfile(member)
                                    if ex is None:
                                        continue
                                    out_file = self.root / mname
                                    with out_file.open("wb") as fout:
                                        while True:
                                            chunk = ex.read(1024 * 1024)
                                            if not chunk:
                                                break
                                            fout.write(chunk)
                                    found_names.add(mname)
                                    if found_names == needed_names:
                                        break
                except tarfile.ReadError:
                    if attempt >= attempts:
                        raise
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                finally:
                    resp.close()
                break

            for tname in needed_names - found_names:
                mm = self._missing_marker_path(tname)
                mm.write_text(
                    "MERIT Hydro: tile not present in package (likely ocean).\n",
                    encoding="utf-8",
                )

        return self

    def read(self, **kwargs: Any) -> xr.DataArray:
        """Read and merge the MERIT Hydro tiles into a single DataArray.

        This method should be called after fetch(). It finds all available tiles
        for the stored bounding box, merges them, crops to the exact bbox,
        and applies nodata conversion.

        Args:
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Merged and cropped xarray DataArray.

        Raises:
            ValueError: If fetch() has not been called.
        """
        if self._xmin is None:
            raise ValueError(
                "fetch() must be called before read() to set the bounding box."
            )

        tiles: list[tuple[int, int]] = self._tiles_for_bbox(
            self._xmin, self._xmax, self._ymin, self._ymax
        )
        local_tile_dir: Path = self.root / self.variable
        results: list[Path] = []
        for lat_ll, lon_ll in tiles:
            tname = self._compose_tile_filename(lat_ll, lon_ll)
            tif_path = self.root / tname
            if tif_path.exists():
                results.append(tif_path)
                continue
            local_path = local_tile_dir / tname
            if local_path.exists():
                results.append(local_path)
                continue

        if not results:
            raise ValueError(
                f"No MERIT Hydro tiles found for bbox ({self._xmin}, {self._ymin}, {self._xmax}, {self._ymax})."
            )

        da: xr.DataArray = self._merge_merit_tiles(results)

        if "_FillValue" in da.attrs:
            assert da.attrs["_FillValue"] == self._source_nodata, (
                f"Expected source _FillValue {self._source_nodata}, got {da.attrs['_FillValue']}"
            )
        else:
            da.attrs["_FillValue"] = self._source_nodata

        da = da.sel(x=slice(self._xmin, self._xmax), y=slice(self._ymax, self._ymin))
        da = convert_nodata(da, self._target_nodata)
        return da


class MeritHydroDir(MeritHydro):
    """Dataset adapter for MERIT Hydro flow direction.

    Args:
        MeritHydro: Base class for MERIT Hydro datasets.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the adapter for flow direction data.

        Args:
            *args: Positional arguments passed to the base class.
            **kwargs: Keyword arguments passed to the base class.

        """
        super().__init__(variable="dir", **kwargs)

    def fetch(self, **kwargs: Any) -> MeritHydro:
        """Process and download flow direction data with specific fill value.

        Args:
            **kwargs: Keyword arguments passed to the base class fetcher.

        Returns:
            The MeritHydro instance.

        """
        return super().fetch(
            source_nodata=247,
            target_nodata=247,
            **kwargs,
        )


class MeritHydroElv(MeritHydro):
    """Dataset adapter for MERIT Hydro elevation.

    Args:
        MeritHydro: Base class for MERIT Hydro datasets.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the adapter for elevation data.

        Args:
            *args: Positional arguments passed to the base class.
            **kwargs: Keyword arguments passed to the base class.

        """
        super().__init__(variable="elv", **kwargs)

    def fetch(self, **kwargs: Any) -> MeritHydro:
        """Process and download elevation data with specific fill value.

        Args:
            **kwargs: Keyword arguments passed to the base class fetcher.

        Returns:
            The MeritHydro instance.

        """
        return super().fetch(
            source_nodata=-9999.0,
            target_nodata=np.nan,
            **kwargs,
        )
