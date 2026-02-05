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
    "n00e005_dir.tif",
    "n00e010_dir.tif",
    "n00e015_dir.tif",
    "n00e020_dir.tif",
    "n00e025_dir.tif",
    "n00e030_dir.tif",
    "n00e035_dir.tif",
    "n00e040_dir.tif",
    "n00e045_dir.tif",
    "n00e070_dir.tif",
    "n00e095_dir.tif",
    "n00e100_dir.tif",
    "n00e105_dir.tif",
    "n00e110_dir.tif",
    "n00e115_dir.tif",
    "n00e120_dir.tif",
    "n00e125_dir.tif",
    "n00e130_dir.tif",
    "n00e150_dir.tif",
    "n00e165_dir.tif",
    "n00e170_dir.tif",
    "n00w005_dir.tif",
    "n00w010_dir.tif",
    "n00w050_dir.tif",
    "n00w055_dir.tif",
    "n00w060_dir.tif",
    "n00w065_dir.tif",
    "n00w070_dir.tif",
    "n00w075_dir.tif",
    "n00w080_dir.tif",
    "n00w085_dir.tif",
    "n00w090_dir.tif",
    "n00w095_dir.tif",
    "n00w160_dir.tif",
    "n00w165_dir.tif",
    "n00w180_dir.tif",
    "n05e000_dir.tif",
    "n05e005_dir.tif",
    "n05e010_dir.tif",
    "n05e015_dir.tif",
    "n05e020_dir.tif",
    "n05e025_dir.tif",
    "n05e030_dir.tif",
    "n05e035_dir.tif",
    "n05e040_dir.tif",
    "n05e045_dir.tif",
    "n05e050_dir.tif",
    "n05e070_dir.tif",
    "n05e075_dir.tif",
    "n05e080_dir.tif",
    "n05e090_dir.tif",
    "n05e095_dir.tif",
    "n05e100_dir.tif",
    "n05e105_dir.tif",
    "n05e110_dir.tif",
    "n05e115_dir.tif",
    "n05e120_dir.tif",
    "n05e125_dir.tif",
    "n05e130_dir.tif",
    "n05e135_dir.tif",
    "n05e140_dir.tif",
    "n05e145_dir.tif",
    "n05e150_dir.tif",
    "n05e155_dir.tif",
    "n05e160_dir.tif",
    "n05e165_dir.tif",
    "n05e170_dir.tif",
    "n05w005_dir.tif",
    "n05w010_dir.tif",
    "n05w015_dir.tif",
    "n05w055_dir.tif",
    "n05w060_dir.tif",
    "n05w065_dir.tif",
    "n05w070_dir.tif",
    "n05w075_dir.tif",
    "n05w080_dir.tif",
    "n05w085_dir.tif",
    "n05w090_dir.tif",
    "n05w165_dir.tif",
    "n10e000_dir.tif",
    "n10e005_dir.tif",
    "n10e010_dir.tif",
    "n10e015_dir.tif",
    "n10e020_dir.tif",
    "n10e025_dir.tif",
    "n10e030_dir.tif",
    "n10e035_dir.tif",
    "n10e040_dir.tif",
    "n10e045_dir.tif",
    "n10e050_dir.tif",
    "n10e070_dir.tif",
    "n10e075_dir.tif",
    "n10e080_dir.tif",
    "n10e090_dir.tif",
    "n10e095_dir.tif",
    "n10e100_dir.tif",
    "n10e105_dir.tif",
    "n10e110_dir.tif",
    "n10e115_dir.tif",
    "n10e120_dir.tif",
    "n10e125_dir.tif",
    "n10e135_dir.tif",
    "n10e140_dir.tif",
    "n10e145_dir.tif",
    "n10e160_dir.tif",
    "n10e165_dir.tif",
    "n10e170_dir.tif",
    "n10w005_dir.tif",
    "n10w010_dir.tif",
    "n10w015_dir.tif",
    "n10w020_dir.tif",
    "n10w025_dir.tif",
    "n10w060_dir.tif",
    "n10w065_dir.tif",
    "n10w070_dir.tif",
    "n10w075_dir.tif",
    "n10w080_dir.tif",
    "n10w085_dir.tif",
    "n10w090_dir.tif",
    "n10w095_dir.tif",
    "n10w110_dir.tif",
    "n15e000_dir.tif",
    "n15e005_dir.tif",
    "n15e010_dir.tif",
    "n15e015_dir.tif",
    "n15e020_dir.tif",
    "n15e025_dir.tif",
    "n15e030_dir.tif",
    "n15e035_dir.tif",
    "n15e040_dir.tif",
    "n15e045_dir.tif",
    "n15e050_dir.tif",
    "n15e055_dir.tif",
    "n15e070_dir.tif",
    "n15e075_dir.tif",
    "n15e080_dir.tif",
    "n15e085_dir.tif",
    "n15e090_dir.tif",
    "n15e095_dir.tif",
    "n15e100_dir.tif",
    "n15e105_dir.tif",
    "n15e110_dir.tif",
    "n15e115_dir.tif",
    "n15e120_dir.tif",
    "n15e145_dir.tif",
    "n15e165_dir.tif",
    "n15w005_dir.tif",
    "n15w010_dir.tif",
    "n15w015_dir.tif",
    "n15w020_dir.tif",
    "n15w025_dir.tif",
    "n15w030_dir.tif",
    "n15w065_dir.tif",
    "n15w070_dir.tif",
    "n15w075_dir.tif",
    "n15w080_dir.tif",
    "n15w085_dir.tif",
    "n15w090_dir.tif",
    "n15w095_dir.tif",
    "n15w100_dir.tif",
    "n15w105_dir.tif",
    "n15w110_dir.tif",
    "n15w115_dir.tif",
    "n15w155_dir.tif",
    "n15w160_dir.tif",
    "n15w170_dir.tif",
    "n20e000_dir.tif",
    "n20e005_dir.tif",
    "n20e010_dir.tif",
    "n20e015_dir.tif",
    "n20e020_dir.tif",
    "n20e025_dir.tif",
    "n20e030_dir.tif",
    "n20e035_dir.tif",
    "n20e040_dir.tif",
    "n20e045_dir.tif",
    "n20e050_dir.tif",
    "n20e055_dir.tif",
    "n20e065_dir.tif",
    "n20e070_dir.tif",
    "n20e075_dir.tif",
    "n20e080_dir.tif",
    "n20e085_dir.tif",
    "n20e090_dir.tif",
    "n20e095_dir.tif",
    "n20e100_dir.tif",
    "n20e105_dir.tif",
    "n20e110_dir.tif",
    "n20e115_dir.tif",
    "n20e120_dir.tif",
    "n20e125_dir.tif",
    "n20e130_dir.tif",
    "n20e135_dir.tif",
    "n20e140_dir.tif",
    "n20e145_dir.tif",
    "n20e150_dir.tif",
    "n20w005_dir.tif",
    "n20w010_dir.tif",
    "n20w015_dir.tif",
    "n20w020_dir.tif",
    "n20w075_dir.tif",
    "n20w080_dir.tif",
    "n20w085_dir.tif",
    "n20w090_dir.tif",
    "n20w095_dir.tif",
    "n20w100_dir.tif",
    "n20w105_dir.tif",
    "n20w110_dir.tif",
    "n20w115_dir.tif",
    "n20w120_dir.tif",
    "n20w160_dir.tif",
    "n20w165_dir.tif",
    "n20w170_dir.tif",
    "n25e000_dir.tif",
    "n25e005_dir.tif",
    "n25e010_dir.tif",
    "n25e015_dir.tif",
    "n25e020_dir.tif",
    "n25e025_dir.tif",
    "n25e030_dir.tif",
    "n25e035_dir.tif",
    "n25e040_dir.tif",
    "n25e045_dir.tif",
    "n25e050_dir.tif",
    "n25e055_dir.tif",
    "n25e060_dir.tif",
    "n25e065_dir.tif",
    "n25e070_dir.tif",
    "n25e075_dir.tif",
    "n25e080_dir.tif",
    "n25e085_dir.tif",
    "n25e090_dir.tif",
    "n25e095_dir.tif",
    "n25e100_dir.tif",
    "n25e105_dir.tif",
    "n25e110_dir.tif",
    "n25e115_dir.tif",
    "n25e120_dir.tif",
    "n25e125_dir.tif",
    "n25e130_dir.tif",
    "n25e140_dir.tif",
    "n25w005_dir.tif",
    "n25w010_dir.tif",
    "n25w015_dir.tif",
    "n25w020_dir.tif",
    "n25w080_dir.tif",
    "n25w085_dir.tif",
    "n25w090_dir.tif",
    "n25w095_dir.tif",
    "n25w100_dir.tif",
    "n25w105_dir.tif",
    "n25w110_dir.tif",
    "n25w115_dir.tif",
    "n25w120_dir.tif",
    "n25w175_dir.tif",
    "n25w180_dir.tif",
    "n30e000_dir.tif",
    "n30e005_dir.tif",
    "n30e010_dir.tif",
    "n30e015_dir.tif",
    "n30e020_dir.tif",
    "n30e025_dir.tif",
    "n30e030_dir.tif",
    "n30e035_dir.tif",
    "n30e040_dir.tif",
    "n30e045_dir.tif",
    "n30e050_dir.tif",
    "n30e055_dir.tif",
    "n30e060_dir.tif",
    "n30e065_dir.tif",
    "n30e070_dir.tif",
    "n30e075_dir.tif",
    "n30e080_dir.tif",
    "n30e085_dir.tif",
    "n30e090_dir.tif",
    "n30e095_dir.tif",
    "n30e100_dir.tif",
    "n30e105_dir.tif",
    "n30e110_dir.tif",
    "n30e115_dir.tif",
    "n30e120_dir.tif",
    "n30e125_dir.tif",
    "n30e130_dir.tif",
    "n30e135_dir.tif",
    "n30e140_dir.tif",
    "n30w005_dir.tif",
    "n30w010_dir.tif",
    "n30w020_dir.tif",
    "n30w065_dir.tif",
    "n30w080_dir.tif",
    "n30w085_dir.tif",
    "n30w090_dir.tif",
    "n30w095_dir.tif",
    "n30w100_dir.tif",
    "n30w105_dir.tif",
    "n30w110_dir.tif",
    "n30w115_dir.tif",
    "n30w120_dir.tif",
    "n30w125_dir.tif",
    "n35e000_dir.tif",
    "n35e005_dir.tif",
    "n35e010_dir.tif",
    "n35e015_dir.tif",
    "n35e020_dir.tif",
    "n35e025_dir.tif",
    "n35e030_dir.tif",
    "n35e035_dir.tif",
    "n35e040_dir.tif",
    "n35e045_dir.tif",
    "n35e050_dir.tif",
    "n35e055_dir.tif",
    "n35e060_dir.tif",
    "n35e065_dir.tif",
    "n35e070_dir.tif",
    "n35e075_dir.tif",
    "n35e080_dir.tif",
    "n35e085_dir.tif",
    "n35e090_dir.tif",
    "n35e095_dir.tif",
    "n35e100_dir.tif",
    "n35e105_dir.tif",
    "n35e110_dir.tif",
    "n35e115_dir.tif",
    "n35e120_dir.tif",
    "n35e125_dir.tif",
    "n35e130_dir.tif",
    "n35e135_dir.tif",
    "n35e140_dir.tif",
    "n35w005_dir.tif",
    "n35w010_dir.tif",
    "n35w025_dir.tif",
    "n35w030_dir.tif",
    "n35w035_dir.tif",
    "n35w075_dir.tif",
    "n35w080_dir.tif",
    "n35w085_dir.tif",
    "n35w090_dir.tif",
    "n35w095_dir.tif",
    "n35w100_dir.tif",
    "n35w105_dir.tif",
    "n35w110_dir.tif",
    "n35w115_dir.tif",
    "n35w120_dir.tif",
    "n35w125_dir.tif",
    "n40e000_dir.tif",
    "n40e005_dir.tif",
    "n40e010_dir.tif",
    "n40e015_dir.tif",
    "n40e020_dir.tif",
    "n40e025_dir.tif",
    "n40e030_dir.tif",
    "n40e035_dir.tif",
    "n40e040_dir.tif",
    "n40e045_dir.tif",
    "n40e050_dir.tif",
    "n40e055_dir.tif",
    "n40e060_dir.tif",
    "n40e065_dir.tif",
    "n40e070_dir.tif",
    "n40e075_dir.tif",
    "n40e080_dir.tif",
    "n40e085_dir.tif",
    "n40e090_dir.tif",
    "n40e095_dir.tif",
    "n40e100_dir.tif",
    "n40e105_dir.tif",
    "n40e110_dir.tif",
    "n40e115_dir.tif",
    "n40e120_dir.tif",
    "n40e125_dir.tif",
    "n40e130_dir.tif",
    "n40e135_dir.tif",
    "n40e140_dir.tif",
    "n40e145_dir.tif",
    "n40w005_dir.tif",
    "n40w010_dir.tif",
    "n40w060_dir.tif",
    "n40w065_dir.tif",
    "n40w070_dir.tif",
    "n40w075_dir.tif",
    "n40w080_dir.tif",
    "n40w085_dir.tif",
    "n40w090_dir.tif",
    "n40w095_dir.tif",
    "n40w100_dir.tif",
    "n40w105_dir.tif",
    "n40w110_dir.tif",
    "n40w115_dir.tif",
    "n40w120_dir.tif",
    "n40w125_dir.tif",
    "n45e000_dir.tif",
    "n45e005_dir.tif",
    "n45e010_dir.tif",
    "n45e015_dir.tif",
    "n45e020_dir.tif",
    "n45e025_dir.tif",
    "n45e030_dir.tif",
    "n45e035_dir.tif",
    "n45e040_dir.tif",
    "n45e045_dir.tif",
    "n45e050_dir.tif",
    "n45e055_dir.tif",
    "n45e060_dir.tif",
    "n45e065_dir.tif",
    "n45e070_dir.tif",
    "n45e075_dir.tif",
    "n45e080_dir.tif",
    "n45e085_dir.tif",
    "n45e090_dir.tif",
    "n45e095_dir.tif",
    "n45e100_dir.tif",
    "n45e105_dir.tif",
    "n45e110_dir.tif",
    "n45e115_dir.tif",
    "n45e120_dir.tif",
    "n45e125_dir.tif",
    "n45e130_dir.tif",
    "n45e135_dir.tif",
    "n45e140_dir.tif",
    "n45e145_dir.tif",
    "n45e150_dir.tif",
    "n45e155_dir.tif",
    "n45w005_dir.tif",
    "n45w010_dir.tif",
    "n45w055_dir.tif",
    "n45w060_dir.tif",
    "n45w065_dir.tif",
    "n45w070_dir.tif",
    "n45w075_dir.tif",
    "n45w080_dir.tif",
    "n45w085_dir.tif",
    "n45w090_dir.tif",
    "n45w095_dir.tif",
    "n45w100_dir.tif",
    "n45w105_dir.tif",
    "n45w110_dir.tif",
    "n45w115_dir.tif",
    "n45w120_dir.tif",
    "n45w125_dir.tif",
    "n45w130_dir.tif",
    "n50e000_dir.tif",
    "n50e005_dir.tif",
    "n50e010_dir.tif",
    "n50e015_dir.tif",
    "n50e020_dir.tif",
    "n50e025_dir.tif",
    "n50e030_dir.tif",
    "n50e035_dir.tif",
    "n50e040_dir.tif",
    "n50e045_dir.tif",
    "n50e050_dir.tif",
    "n50e055_dir.tif",
    "n50e060_dir.tif",
    "n50e065_dir.tif",
    "n50e070_dir.tif",
    "n50e075_dir.tif",
    "n50e080_dir.tif",
    "n50e085_dir.tif",
    "n50e090_dir.tif",
    "n50e095_dir.tif",
    "n50e100_dir.tif",
    "n50e105_dir.tif",
    "n50e110_dir.tif",
    "n50e115_dir.tif",
    "n50e120_dir.tif",
    "n50e125_dir.tif",
    "n50e130_dir.tif",
    "n50e135_dir.tif",
    "n50e140_dir.tif",
    "n50e150_dir.tif",
    "n50e155_dir.tif",
    "n50e160_dir.tif",
    "n50e165_dir.tif",
    "n50e170_dir.tif",
    "n50e175_dir.tif",
    "n50w005_dir.tif",
    "n50w010_dir.tif",
    "n50w015_dir.tif",
    "n50w060_dir.tif",
    "n50w065_dir.tif",
    "n50w070_dir.tif",
    "n50w075_dir.tif",
    "n50w080_dir.tif",
    "n50w085_dir.tif",
    "n50w090_dir.tif",
    "n50w095_dir.tif",
    "n50w100_dir.tif",
    "n50w105_dir.tif",
    "n50w110_dir.tif",
    "n50w115_dir.tif",
    "n50w120_dir.tif",
    "n50w125_dir.tif",
    "n50w130_dir.tif",
    "n50w135_dir.tif",
    "n50w160_dir.tif",
    "n50w165_dir.tif",
    "n50w170_dir.tif",
    "n50w175_dir.tif",
    "n50w180_dir.tif",
    "n55e000_dir.tif",
    "n55e005_dir.tif",
    "n55e010_dir.tif",
    "n55e015_dir.tif",
    "n55e020_dir.tif",
    "n55e025_dir.tif",
    "n55e030_dir.tif",
    "n55e035_dir.tif",
    "n55e040_dir.tif",
    "n55e045_dir.tif",
    "n55e050_dir.tif",
    "n55e055_dir.tif",
    "n55e060_dir.tif",
    "n55e065_dir.tif",
    "n55e070_dir.tif",
    "n55e075_dir.tif",
    "n55e080_dir.tif",
    "n55e085_dir.tif",
    "n55e090_dir.tif",
    "n55e095_dir.tif",
    "n55e100_dir.tif",
    "n55e105_dir.tif",
    "n55e110_dir.tif",
    "n55e115_dir.tif",
    "n55e120_dir.tif",
    "n55e125_dir.tif",
    "n55e130_dir.tif",
    "n55e135_dir.tif",
    "n55e140_dir.tif",
    "n55e145_dir.tif",
    "n55e150_dir.tif",
    "n55e155_dir.tif",
    "n55e160_dir.tif",
    "n55e165_dir.tif",
    "n55e170_dir.tif",
    "n55w005_dir.tif",
    "n55w010_dir.tif",
    "n55w015_dir.tif",
    "n55w045_dir.tif",
    "n55w050_dir.tif",
    "n55w060_dir.tif",
    "n55w065_dir.tif",
    "n55w070_dir.tif",
    "n55w075_dir.tif",
    "n55w080_dir.tif",
    "n55w085_dir.tif",
    "n55w090_dir.tif",
    "n55w095_dir.tif",
    "n55w100_dir.tif",
    "n55w105_dir.tif",
    "n55w110_dir.tif",
    "n55w115_dir.tif",
    "n55w120_dir.tif",
    "n55w125_dir.tif",
    "n55w130_dir.tif",
    "n55w135_dir.tif",
    "n55w140_dir.tif",
    "n55w145_dir.tif",
    "n55w150_dir.tif",
    "n55w155_dir.tif",
    "n55w160_dir.tif",
    "n55w165_dir.tif",
    "n55w170_dir.tif",
    "n55w175_dir.tif",
    "n60e000_dir.tif",
    "n60e005_dir.tif",
    "n60e010_dir.tif",
    "n60e015_dir.tif",
    "n60e020_dir.tif",
    "n60e025_dir.tif",
    "n60e030_dir.tif",
    "n60e035_dir.tif",
    "n60e040_dir.tif",
    "n60e045_dir.tif",
    "n60e050_dir.tif",
    "n60e055_dir.tif",
    "n60e060_dir.tif",
    "n60e065_dir.tif",
    "n60e070_dir.tif",
    "n60e075_dir.tif",
    "n60e080_dir.tif",
    "n60e085_dir.tif",
    "n60e090_dir.tif",
    "n60e095_dir.tif",
    "n60e100_dir.tif",
    "n60e105_dir.tif",
    "n60e110_dir.tif",
    "n60e115_dir.tif",
    "n60e120_dir.tif",
    "n60e125_dir.tif",
    "n60e130_dir.tif",
    "n60e135_dir.tif",
    "n60e140_dir.tif",
    "n60e145_dir.tif",
    "n60e150_dir.tif",
    "n60e155_dir.tif",
    "n60e160_dir.tif",
    "n60e165_dir.tif",
    "n60e170_dir.tif",
    "n60e175_dir.tif",
    "n60w005_dir.tif",
    "n60w010_dir.tif",
    "n60w015_dir.tif",
    "n60w020_dir.tif",
    "n60w025_dir.tif",
    "n60w040_dir.tif",
    "n60w045_dir.tif",
    "n60w050_dir.tif",
    "n60w055_dir.tif",
    "n60w065_dir.tif",
    "n60w070_dir.tif",
    "n60w075_dir.tif",
    "n60w080_dir.tif",
    "n60w085_dir.tif",
    "n60w090_dir.tif",
    "n60w095_dir.tif",
    "n60w100_dir.tif",
    "n60w105_dir.tif",
    "n60w110_dir.tif",
    "n60w115_dir.tif",
    "n60w120_dir.tif",
    "n60w125_dir.tif",
    "n60w130_dir.tif",
    "n60w135_dir.tif",
    "n60w140_dir.tif",
    "n60w145_dir.tif",
    "n60w150_dir.tif",
    "n60w155_dir.tif",
    "n60w160_dir.tif",
    "n60w165_dir.tif",
    "n60w170_dir.tif",
    "n60w175_dir.tif",
    "n60w180_dir.tif",
    "n65e010_dir.tif",
    "n65e015_dir.tif",
    "n65e020_dir.tif",
    "n65e025_dir.tif",
    "n65e030_dir.tif",
    "n65e035_dir.tif",
    "n65e040_dir.tif",
    "n65e045_dir.tif",
    "n65e050_dir.tif",
    "n65e055_dir.tif",
    "n65e060_dir.tif",
    "n65e065_dir.tif",
    "n65e070_dir.tif",
    "n65e075_dir.tif",
    "n65e080_dir.tif",
    "n65e085_dir.tif",
    "n65e090_dir.tif",
    "n65e095_dir.tif",
    "n65e100_dir.tif",
    "n65e105_dir.tif",
    "n65e110_dir.tif",
    "n65e115_dir.tif",
    "n65e120_dir.tif",
    "n65e125_dir.tif",
    "n65e130_dir.tif",
    "n65e135_dir.tif",
    "n65e140_dir.tif",
    "n65e145_dir.tif",
    "n65e150_dir.tif",
    "n65e155_dir.tif",
    "n65e160_dir.tif",
    "n65e165_dir.tif",
    "n65e170_dir.tif",
    "n65e175_dir.tif",
    "n65w015_dir.tif",
    "n65w020_dir.tif",
    "n65w025_dir.tif",
    "n65w030_dir.tif",
    "n65w035_dir.tif",
    "n65w040_dir.tif",
    "n65w045_dir.tif",
    "n65w050_dir.tif",
    "n65w055_dir.tif",
    "n65w065_dir.tif",
    "n65w070_dir.tif",
    "n65w075_dir.tif",
    "n65w080_dir.tif",
    "n65w085_dir.tif",
    "n65w090_dir.tif",
    "n65w095_dir.tif",
    "n65w100_dir.tif",
    "n65w105_dir.tif",
    "n65w110_dir.tif",
    "n65w115_dir.tif",
    "n65w120_dir.tif",
    "n65w125_dir.tif",
    "n65w130_dir.tif",
    "n65w135_dir.tif",
    "n65w140_dir.tif",
    "n65w145_dir.tif",
    "n65w150_dir.tif",
    "n65w155_dir.tif",
    "n65w160_dir.tif",
    "n65w165_dir.tif",
    "n65w170_dir.tif",
    "n65w175_dir.tif",
    "n65w180_dir.tif",
    "n70e015_dir.tif",
    "n70e020_dir.tif",
    "n70e025_dir.tif",
    "n70e030_dir.tif",
    "n70e050_dir.tif",
    "n70e055_dir.tif",
    "n70e060_dir.tif",
    "n70e065_dir.tif",
    "n70e070_dir.tif",
    "n70e075_dir.tif",
    "n70e080_dir.tif",
    "n70e085_dir.tif",
    "n70e090_dir.tif",
    "n70e095_dir.tif",
    "n70e100_dir.tif",
    "n70e105_dir.tif",
    "n70e110_dir.tif",
    "n70e115_dir.tif",
    "n70e120_dir.tif",
    "n70e125_dir.tif",
    "n70e130_dir.tif",
    "n70e135_dir.tif",
    "n70e140_dir.tif",
    "n70e145_dir.tif",
    "n70e150_dir.tif",
    "n70e155_dir.tif",
    "n70e160_dir.tif",
    "n70e165_dir.tif",
    "n70e170_dir.tif",
    "n70e175_dir.tif",
    "n70w010_dir.tif",
    "n70w020_dir.tif",
    "n70w025_dir.tif",
    "n70w030_dir.tif",
    "n70w035_dir.tif",
    "n70w040_dir.tif",
    "n70w045_dir.tif",
    "n70w050_dir.tif",
    "n70w055_dir.tif",
    "n70w060_dir.tif",
    "n70w070_dir.tif",
    "n70w075_dir.tif",
    "n70w080_dir.tif",
    "n70w085_dir.tif",
    "n70w090_dir.tif",
    "n70w095_dir.tif",
    "n70w100_dir.tif",
    "n70w105_dir.tif",
    "n70w110_dir.tif",
    "n70w115_dir.tif",
    "n70w120_dir.tif",
    "n70w125_dir.tif",
    "n70w130_dir.tif",
    "n70w135_dir.tif",
    "n70w145_dir.tif",
    "n70w150_dir.tif",
    "n70w155_dir.tif",
    "n70w160_dir.tif",
    "n70w165_dir.tif",
    "n70w180_dir.tif",
    "n75e010_dir.tif",
    "n75e015_dir.tif",
    "n75e020_dir.tif",
    "n75e025_dir.tif",
    "n75e030_dir.tif",
    "n75e045_dir.tif",
    "n75e050_dir.tif",
    "n75e055_dir.tif",
    "n75e060_dir.tif",
    "n75e065_dir.tif",
    "n75e075_dir.tif",
    "n75e080_dir.tif",
    "n75e085_dir.tif",
    "n75e090_dir.tif",
    "n75e095_dir.tif",
    "n75e100_dir.tif",
    "n75e105_dir.tif",
    "n75e110_dir.tif",
    "n75e135_dir.tif",
    "n75e140_dir.tif",
    "n75e145_dir.tif",
    "n75e150_dir.tif",
    "n75e155_dir.tif",
    "n75w020_dir.tif",
    "n75w025_dir.tif",
    "n75w030_dir.tif",
    "n75w035_dir.tif",
    "n75w040_dir.tif",
    "n75w045_dir.tif",
    "n75w050_dir.tif",
    "n75w055_dir.tif",
    "n75w060_dir.tif",
    "n75w065_dir.tif",
    "n75w070_dir.tif",
    "n75w075_dir.tif",
    "n75w080_dir.tif",
    "n75w085_dir.tif",
    "n75w090_dir.tif",
    "n75w095_dir.tif",
    "n75w100_dir.tif",
    "n75w105_dir.tif",
    "n75w110_dir.tif",
    "n75w115_dir.tif",
    "n75w120_dir.tif",
    "n75w125_dir.tif",
    "n80e010_dir.tif",
    "n80e015_dir.tif",
    "n80e020_dir.tif",
    "n80e025_dir.tif",
    "n80e030_dir.tif",
    "n80e035_dir.tif",
    "n80e040_dir.tif",
    "n80e045_dir.tif",
    "n80e050_dir.tif",
    "n80e055_dir.tif",
    "n80e060_dir.tif",
    "n80e065_dir.tif",
    "n80e075_dir.tif",
    "n80e080_dir.tif",
    "n80e090_dir.tif",
    "n80e095_dir.tif",
    "n80w015_dir.tif",
    "n80w020_dir.tif",
    "n80w025_dir.tif",
    "n80w030_dir.tif",
    "n80w035_dir.tif",
    "n80w040_dir.tif",
    "n80w045_dir.tif",
    "n80w050_dir.tif",
    "n80w055_dir.tif",
    "n80w060_dir.tif",
    "n80w065_dir.tif",
    "n80w070_dir.tif",
    "n80w075_dir.tif",
    "n80w080_dir.tif",
    "n80w085_dir.tif",
    "n80w090_dir.tif",
    "n80w095_dir.tif",
    "n80w100_dir.tif",
    "n80w105_dir.tif",
    "s05e005_dir.tif",
    "s05e010_dir.tif",
    "s05e015_dir.tif",
    "s05e020_dir.tif",
    "s05e025_dir.tif",
    "s05e030_dir.tif",
    "s05e035_dir.tif",
    "s05e040_dir.tif",
    "s05e050_dir.tif",
    "s05e055_dir.tif",
    "s05e070_dir.tif",
    "s05e095_dir.tif",
    "s05e100_dir.tif",
    "s05e105_dir.tif",
    "s05e110_dir.tif",
    "s05e115_dir.tif",
    "s05e120_dir.tif",
    "s05e125_dir.tif",
    "s05e130_dir.tif",
    "s05e135_dir.tif",
    "s05e140_dir.tif",
    "s05e145_dir.tif",
    "s05e150_dir.tif",
    "s05e155_dir.tif",
    "s05e165_dir.tif",
    "s05e170_dir.tif",
    "s05e175_dir.tif",
    "s05w035_dir.tif",
    "s05w040_dir.tif",
    "s05w045_dir.tif",
    "s05w050_dir.tif",
    "s05w055_dir.tif",
    "s05w060_dir.tif",
    "s05w065_dir.tif",
    "s05w070_dir.tif",
    "s05w075_dir.tif",
    "s05w080_dir.tif",
    "s05w085_dir.tif",
    "s05w090_dir.tif",
    "s05w095_dir.tif",
    "s05w155_dir.tif",
    "s05w165_dir.tif",
    "s05w175_dir.tif",
    "s10e010_dir.tif",
    "s10e015_dir.tif",
    "s10e020_dir.tif",
    "s10e025_dir.tif",
    "s10e030_dir.tif",
    "s10e035_dir.tif",
    "s10e045_dir.tif",
    "s10e050_dir.tif",
    "s10e055_dir.tif",
    "s10e070_dir.tif",
    "s10e100_dir.tif",
    "s10e105_dir.tif",
    "s10e110_dir.tif",
    "s10e115_dir.tif",
    "s10e120_dir.tif",
    "s10e125_dir.tif",
    "s10e130_dir.tif",
    "s10e135_dir.tif",
    "s10e140_dir.tif",
    "s10e145_dir.tif",
    "s10e150_dir.tif",
    "s10e155_dir.tif",
    "s10e160_dir.tif",
    "s10e165_dir.tif",
    "s10e175_dir.tif",
    "s10w015_dir.tif",
    "s10w035_dir.tif",
    "s10w040_dir.tif",
    "s10w045_dir.tif",
    "s10w050_dir.tif",
    "s10w055_dir.tif",
    "s10w060_dir.tif",
    "s10w065_dir.tif",
    "s10w070_dir.tif",
    "s10w075_dir.tif",
    "s10w080_dir.tif",
    "s10w085_dir.tif",
    "s10w140_dir.tif",
    "s10w145_dir.tif",
    "s10w155_dir.tif",
    "s10w160_dir.tif",
    "s10w165_dir.tif",
    "s10w175_dir.tif",
    "s15e010_dir.tif",
    "s15e015_dir.tif",
    "s15e020_dir.tif",
    "s15e025_dir.tif",
    "s15e030_dir.tif",
    "s15e035_dir.tif",
    "s15e040_dir.tif",
    "s15e045_dir.tif",
    "s15e050_dir.tif",
    "s15e055_dir.tif",
    "s15e095_dir.tif",
    "s15e105_dir.tif",
    "s15e115_dir.tif",
    "s15e120_dir.tif",
    "s15e125_dir.tif",
    "s15e130_dir.tif",
    "s15e135_dir.tif",
    "s15e140_dir.tif",
    "s15e145_dir.tif",
    "s15e150_dir.tif",
    "s15e155_dir.tif",
    "s15e160_dir.tif",
    "s15e165_dir.tif",
    "s15e170_dir.tif",
    "s15e175_dir.tif",
    "s15w040_dir.tif",
    "s15w045_dir.tif",
    "s15w050_dir.tif",
    "s15w055_dir.tif",
    "s15w060_dir.tif",
    "s15w065_dir.tif",
    "s15w070_dir.tif",
    "s15w075_dir.tif",
    "s15w080_dir.tif",
    "s15w140_dir.tif",
    "s15w145_dir.tif",
    "s15w150_dir.tif",
    "s15w155_dir.tif",
    "s15w165_dir.tif",
    "s15w170_dir.tif",
    "s15w175_dir.tif",
    "s15w180_dir.tif",
    "s20e010_dir.tif",
    "s20e015_dir.tif",
    "s20e020_dir.tif",
    "s20e025_dir.tif",
    "s20e030_dir.tif",
    "s20e035_dir.tif",
    "s20e040_dir.tif",
    "s20e045_dir.tif",
    "s20e050_dir.tif",
    "s20e055_dir.tif",
    "s20e060_dir.tif",
    "s20e115_dir.tif",
    "s20e120_dir.tif",
    "s20e125_dir.tif",
    "s20e130_dir.tif",
    "s20e135_dir.tif",
    "s20e140_dir.tif",
    "s20e145_dir.tif",
    "s20e150_dir.tif",
    "s20e155_dir.tif",
    "s20e160_dir.tif",
    "s20e165_dir.tif",
    "s20e170_dir.tif",
    "s20e175_dir.tif",
    "s20w010_dir.tif",
    "s20w040_dir.tif",
    "s20w045_dir.tif",
    "s20w050_dir.tif",
    "s20w055_dir.tif",
    "s20w060_dir.tif",
    "s20w065_dir.tif",
    "s20w070_dir.tif",
    "s20w075_dir.tif",
    "s20w080_dir.tif",
    "s20w140_dir.tif",
    "s20w145_dir.tif",
    "s20w150_dir.tif",
    "s20w155_dir.tif",
    "s20w160_dir.tif",
    "s20w165_dir.tif",
    "s20w175_dir.tif",
    "s20w180_dir.tif",
    "s25e010_dir.tif",
    "s25e015_dir.tif",
    "s25e020_dir.tif",
    "s25e025_dir.tif",
    "s25e030_dir.tif",
    "s25e035_dir.tif",
    "s25e040_dir.tif",
    "s25e045_dir.tif",
    "s25e055_dir.tif",
    "s25e110_dir.tif",
    "s25e115_dir.tif",
    "s25e120_dir.tif",
    "s25e125_dir.tif",
    "s25e130_dir.tif",
    "s25e135_dir.tif",
    "s25e140_dir.tif",
    "s25e145_dir.tif",
    "s25e150_dir.tif",
    "s25e155_dir.tif",
    "s25e160_dir.tif",
    "s25e165_dir.tif",
    "s25e170_dir.tif",
    "s25w030_dir.tif",
    "s25w045_dir.tif",
    "s25w050_dir.tif",
    "s25w055_dir.tif",
    "s25w060_dir.tif",
    "s25w065_dir.tif",
    "s25w070_dir.tif",
    "s25w075_dir.tif",
    "s25w125_dir.tif",
    "s25w130_dir.tif",
    "s25w135_dir.tif",
    "s25w140_dir.tif",
    "s25w145_dir.tif",
    "s25w150_dir.tif",
    "s25w155_dir.tif",
    "s25w160_dir.tif",
    "s25w175_dir.tif",
    "s25w180_dir.tif",
    "s30e010_dir.tif",
    "s30e015_dir.tif",
    "s30e020_dir.tif",
    "s30e025_dir.tif",
    "s30e030_dir.tif",
    "s30e040_dir.tif",
    "s30e045_dir.tif",
    "s30e110_dir.tif",
    "s30e115_dir.tif",
    "s30e120_dir.tif",
    "s30e125_dir.tif",
    "s30e130_dir.tif",
    "s30e135_dir.tif",
    "s30e140_dir.tif",
    "s30e145_dir.tif",
    "s30e150_dir.tif",
    "s30e165_dir.tif",
    "s30w050_dir.tif",
    "s30w055_dir.tif",
    "s30w060_dir.tif",
    "s30w065_dir.tif",
    "s30w070_dir.tif",
    "s30w075_dir.tif",
    "s30w080_dir.tif",
    "s30w085_dir.tif",
    "s30w110_dir.tif",
    "s30w135_dir.tif",
    "s30w145_dir.tif",
    "s30w180_dir.tif",
    "s35e015_dir.tif",
    "s35e020_dir.tif",
    "s35e025_dir.tif",
    "s35e030_dir.tif",
    "s35e110_dir.tif",
    "s35e115_dir.tif",
    "s35e120_dir.tif",
    "s35e125_dir.tif",
    "s35e130_dir.tif",
    "s35e135_dir.tif",
    "s35e140_dir.tif",
    "s35e145_dir.tif",
    "s35e150_dir.tif",
    "s35e155_dir.tif",
    "s35e170_dir.tif",
    "s35w055_dir.tif",
    "s35w060_dir.tif",
    "s35w065_dir.tif",
    "s35w070_dir.tif",
    "s35w075_dir.tif",
    "s35w080_dir.tif",
    "s35w085_dir.tif",
    "s35w180_dir.tif",
    "s40e075_dir.tif",
    "s40e115_dir.tif",
    "s40e135_dir.tif",
    "s40e140_dir.tif",
    "s40e145_dir.tif",
    "s40e150_dir.tif",
    "s40e170_dir.tif",
    "s40e175_dir.tif",
    "s40w015_dir.tif",
    "s40w060_dir.tif",
    "s40w065_dir.tif",
    "s40w070_dir.tif",
    "s40w075_dir.tif",
    "s45e140_dir.tif",
    "s45e145_dir.tif",
    "s45e165_dir.tif",
    "s45e170_dir.tif",
    "s45e175_dir.tif",
    "s45w010_dir.tif",
    "s45w015_dir.tif",
    "s45w065_dir.tif",
    "s45w070_dir.tif",
    "s45w075_dir.tif",
    "s45w080_dir.tif",
    "s45w180_dir.tif",
    "s50e035_dir.tif",
    "s50e050_dir.tif",
    "s50e065_dir.tif",
    "s50e070_dir.tif",
    "s50e165_dir.tif",
    "s50e170_dir.tif",
    "s50e175_dir.tif",
    "s50w070_dir.tif",
    "s50w075_dir.tif",
    "s50w080_dir.tif",
    "s55e000_dir.tif",
    "s55e065_dir.tif",
    "s55e070_dir.tif",
    "s55e155_dir.tif",
    "s55e165_dir.tif",
    "s55w040_dir.tif",
    "s55w060_dir.tif",
    "s55w065_dir.tif",
    "s55w070_dir.tif",
    "s55w075_dir.tif",
    "s55w080_dir.tif",
    "s60w030_dir.tif",
    "s60w070_dir.tif",
    "s60w075_dir.tif",
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
                and tile_name in available_tiles
            ):
                if not self._missing_marker_path(tile_name).exists():
                    missing_names.add(tile_name)
                continue

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
