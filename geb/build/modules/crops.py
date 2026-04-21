"""Crops data processing and setup methods for GEB."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from geb.agents.crop_farmers import (
    FIELD_EXPANSION_ADAPTATION,
    INDEX_INSURANCE_ADAPTATION,
    IRRIGATION_EFFICIENCY_ADAPTATION_DRIP,
    IRRIGATION_EFFICIENCY_ADAPTATION_SPRINKLER,
    PR_INSURANCE_ADAPTATION,
    SURFACE_IRRIGATION_EQUIPMENT,
    TRADITIONAL_INSURANCE_ADAPTATION,
    WELL_ADAPTATION,
)
from geb.build.methods import build_method
from geb.geb_types import ArrayBool, ArrayInt32, ArrayInt64, ArrayUint8, TwoDArrayInt32
from geb.workflows.io import get_window
from geb.workflows.raster import (
    get_linear_indices,
    get_neighbor_cell_ids_for_linear_indices,
    interpolate_na_2d,
    interpolate_na_along_dim,
    pad_xy,
    sample_from_map,
    snap_to_grid,
)

from ..workflows.conversions import TRADE_REGIONS
from ..workflows.crop_calendars import (
    donate_and_receive_crop_prices,
    parse_MIRCA2000_crop_calendar,
)
from ..workflows.farmers import get_farm_locations
from .base import BuildModelBase

# CROP_DATA from Siebert et al. (2010)
# source doi: 10.1016/j.jhydrol.2009.07.031
# license: Creative Commons Attribution 4.0 International
CROP_DATA = [
    {
        "name": "wheat",
        "id": 0,
        "is_paddy": False,
        "a": 0.9885,
        "b": 0.1103,
        "P0": 0.1,
        "P1": 0.25,
        "l_ini": 15.0,
        "l_dev": 25.0,
        "l_mid": 40,
        "l_late": 20,
        "kc_initial": 0.4,
        "kc_mid": 1.15,
        "kc_end": 0.3,
        "rd_irr": 1.25,
        "rd_rain": 1.6,
        "reference_yield_kg_m2": 1.0,
        "crop_group_number": 4.5,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/whtfra.dat",
    },
    {
        "name": "maize",
        "id": 1,
        "is_paddy": False,
        "a": 1.2929,
        "b": -0.0798,
        "P0": 0.1,
        "P1": 0.4,
        "l_ini": 17.0,
        "l_dev": 28.000000000000004,
        "l_mid": 33,
        "l_late": 22,
        "kc_initial": 0.3,
        "kc_mid": 1.2,
        "kc_end": 0.4,
        "rd_irr": 1.0,
        "rd_rain": 1.6,
        "reference_yield_kg_m2": 2.4,
        "crop_group_number": 4.5,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/maiz.w41",
    },
    {
        "name": "rice",
        "id": 2,
        "is_paddy": True,
        "a": 1.0,
        "b": -0.1,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 17.0,
        "l_dev": 18.0,
        "l_mid": 44,
        "l_late": 21,
        "kc_initial": 1.05,
        "kc_mid": 1.2,
        "kc_end": 0.75,
        "rd_irr": 0.5,
        "rd_rain": 1.0,
        "reference_yield_kg_m2": 1.2,
        "crop_group_number": 3.5,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/ric501.cab",
    },
    {
        "name": "barley",
        "id": 3,
        "is_paddy": False,
        "a": 1.478,
        "b": -0.4288,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 15.0,
        "l_dev": 25.0,
        "l_mid": 40,
        "l_late": 20,
        "kc_initial": 0.3,
        "kc_mid": 1.15,
        "kc_end": 0.25,
        "rd_irr": 1.0,
        "rd_rain": 1.5,
        "reference_yield_kg_m2": 0.8,
        "crop_group_number": 4.5,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/bar301.cab",
    },
    {
        "name": "rye",
        "id": 4,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.1,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 10.0,
        "l_dev": 60.0,
        "l_mid": 20,
        "l_late": 10,
        "kc_initial": 0.4,
        "kc_mid": 1.15,
        "kc_end": 0.3,
        "rd_irr": 1.25,
        "rd_rain": 1.6,
        "reference_yield_kg_m2": 0.75,
        "crop_group_number": 3.0,
        "crop_group_number_reference": "grass: https://edepot.wur.nl/336784",
    },
    {
        "name": "millet",
        "id": 5,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.1,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 14.000000000000002,
        "l_dev": 22.0,
        "l_mid": 40,
        "l_late": 24,
        "kc_initial": 0.3,
        "kc_mid": 1.0,
        "kc_end": 0.3,
        "rd_irr": 1.0,
        "rd_rain": 1.8,
        "reference_yield_kg_m2": 0.8,
        "crop_group_number": 4.5,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/millet.w41",
    },
    {
        "name": "sorghum",
        "id": 6,
        "is_paddy": False,
        "a": 0.8681,
        "b": 0.2753,
        "P0": 0.1,
        "P1": 0.3,
        "l_ini": 15.0,
        "l_dev": 28.000000000000004,
        "l_mid": 33,
        "l_late": 24,
        "kc_initial": 0.3,
        "kc_mid": 1.1,
        "kc_end": 0.55,
        "rd_irr": 1.0,
        "rd_rain": 1.8,
        "reference_yield_kg_m2": 1.5,
        "crop_group_number": 5.0,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/sorghum.w41",
    },
    {
        "name": "soybeans",
        "id": 7,
        "is_paddy": False,
        "a": 0.8373,
        "b": 0.208,
        "P0": 0.1,
        "P1": 0.4,
        "l_ini": 15.0,
        "l_dev": 20.0,
        "l_mid": 45,
        "l_late": 20,
        "kc_initial": 0.4,
        "kc_mid": 1.15,
        "kc_end": 0.5,
        "rd_irr": 0.6,
        "rd_rain": 1.3,
        "reference_yield_kg_m2": 0.4,
        "crop_group_number": 5.0,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/soy0901.cab",
    },
    {
        "name": "sunflower",
        "id": 8,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.0,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 19.0,
        "l_dev": 27.0,
        "l_mid": 35,
        "l_late": 19,
        "kc_initial": 0.35,
        "kc_mid": 1.1,
        "kc_end": 0.25,
        "rd_irr": 0.8,
        "rd_rain": 1.5,
        "reference_yield_kg_m2": 0.4,
        "crop_group_number": 3.5,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/sun1101.cab",
    },
    {
        "name": "potatoes",
        "id": 9,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.1,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 20.0,
        "l_dev": 25.0,
        "l_mid": 35,
        "l_late": 20,
        "kc_initial": 0.35,
        "kc_mid": 1.15,
        "kc_end": 0.5,
        "rd_irr": 0.4,
        "rd_rain": 0.6,
        "reference_yield_kg_m2": 7.0,
        "crop_group_number": 3.0,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/pot701.cab",
    },
    {
        "name": "cassava",
        "id": 10,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.1,
        "P0": 0.15,
        "P1": 0.5,
        "l_ini": 10.0,
        "l_dev": 20.0,
        "l_mid": 43,
        "l_late": 27,
        "kc_initial": 0.3,
        "kc_mid": 0.95,
        "kc_end": 0.4,
        "rd_irr": 0.6,
        "rd_rain": 0.9,
        "reference_yield_kg_m2": 4.0,
        "crop_group_number": 4.5,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/cassava.w41",
    },
    {
        "name": "sugar cane",
        "id": 11,
        "is_paddy": False,
        "a": 1.0,
        "b": -0.1,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 0.0,
        "l_dev": 0.0,
        "l_mid": 100,
        "l_late": 0,
        "kc_initial": 0.0,
        "kc_mid": 0.9,
        "kc_end": 0.0,
        "rd_irr": 1.2,
        "rd_rain": 1.8,
        "reference_yield_kg_m2": 15.0,
        "crop_group_number": 5.0,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/sugrcane.w41",
    },
    {
        "name": "sugar beets",
        "id": 12,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.1,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 20.0,
        "l_dev": 25.0,
        "l_mid": 35,
        "l_late": 20,
        "kc_initial": 0.35,
        "kc_mid": 1.2,
        "kc_end": 0.8,
        "rd_irr": 0.7,
        "rd_rain": 1.2,
        "reference_yield_kg_m2": 9.0,
        "crop_group_number": 2.0,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/sug0601.cab",
    },
    {
        "name": "oil palm",
        "id": 13,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.0,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 0.0,
        "l_dev": 0.0,
        "l_mid": 100,
        "l_late": 0,
        "kc_initial": 0.0,
        "kc_mid": 1.0,
        "kc_end": 0.0,
        "rd_irr": 0.7,
        "rd_rain": 1.1,
        "reference_yield_kg_m2": 3.2,
        "crop_group_number": 5.0,
        "crop_group_number_reference": "similar to olive: https://wofost.readthedocs.io/en/7.2/_downloads/cf58a94b422342c8378f99ed36d6eb76/WOFOST_system_description.pdf",
    },
    {
        "name": "rapeseed",
        "id": 14,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.1,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 30.0,
        "l_dev": 25.0,
        "l_mid": 30,
        "l_late": 15,
        "kc_initial": 0.35,
        "kc_mid": 1.1,
        "kc_end": 0.35,
        "rd_irr": 1.0,
        "rd_rain": 1.5,
        "reference_yield_kg_m2": 0.5,
        "crop_group_number": 4.5,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/rap1001.cab",
    },
    {
        "name": "groundnuts",
        "id": 15,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.0,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 22.0,
        "l_dev": 28.000000000000004,
        "l_mid": 30,
        "l_late": 20,
        "kc_initial": 0.4,
        "kc_mid": 1.15,
        "kc_end": 0.6,
        "rd_irr": 0.5,
        "rd_rain": 1.0,
        "reference_yield_kg_m2": 0.85,
        "crop_group_number": 4.0,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/gr_nut.w41",
    },
    {
        "name": "pulses",
        "id": 16,
        "is_paddy": False,
        "a": 1.3,
        "b": -0.2,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 18.0,
        "l_dev": 27.0,
        "l_mid": 35,
        "l_late": 20,
        "kc_initial": 0.45,
        "kc_mid": 1.1,
        "kc_end": 0.6,
        "rd_irr": 0.55,
        "rd_rain": 0.85,
        "reference_yield_kg_m2": 0.6,
        "crop_group_number": 4.5,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/pigeopea.w41; https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/chickpea.w41",
    },
    {
        "name": "citrus",
        "id": 17,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.0,
        "P0": 0.15,
        "P1": 0.5,
        "l_ini": 16.0,
        "l_dev": 25.0,
        "l_mid": 33,
        "l_late": 26,
        "kc_initial": 0.8,
        "kc_mid": 0.8,
        "kc_end": 0.8,
        "rd_irr": 1.0,
        "rd_rain": 1.3,
        "reference_yield_kg_m2": 4.0,
        "crop_group_number": 4.0,
        "crop_group_number_reference": "https://wofost.readthedocs.io/en/7.2/_downloads/cf58a94b422342c8378f99ed36d6eb76/WOFOST_system_description.pdf",
    },
    {
        "name": "date palm",
        "id": 18,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.1,
        "P0": 0.05,
        "P1": 0.3,
        "l_ini": 0.0,
        "l_dev": 0.0,
        "l_mid": 100,
        "l_late": 0,
        "kc_initial": 0.95,
        "kc_mid": 0.95,
        "kc_end": 0.95,
        "rd_irr": 1.5,
        "rd_rain": 2.2,
        "reference_yield_kg_m2": 4.0,
        "crop_group_number": 5.0,
        "crop_group_number_reference": "similar to olive: https://wofost.readthedocs.io/en/7.2/_downloads/cf58a94b422342c8378f99ed36d6eb76/WOFOST_system_description.pdf",
    },
    {
        "name": "grapes",
        "id": 19,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.15,
        "P0": 0.05,
        "P1": 0.3,
        "l_ini": 30.0,
        "l_dev": 14.000000000000002,
        "l_mid": 20,
        "l_late": 36,
        "kc_initial": 0.3,
        "kc_mid": 0.8,
        "kc_end": 0.3,
        "rd_irr": 1.0,
        "rd_rain": 1.8,
        "reference_yield_kg_m2": 4.0,
        "crop_group_number": 3.0,
        "crop_group_number_reference": "https://wofost.readthedocs.io/en/7.2/_downloads/cf58a94b422342c8378f99ed36d6eb76/WOFOST_system_description.pdf",
    },
    {
        "name": "cotton",
        "id": 20,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.0,
        "P0": 0.1,
        "P1": 0.2,
        "l_ini": 17.0,
        "l_dev": 33.0,
        "l_mid": 25,
        "l_late": 25,
        "kc_initial": 0.35,
        "kc_mid": 1.18,
        "kc_end": 0.6,
        "rd_irr": 1.0,
        "rd_rain": 1.5,
        "reference_yield_kg_m2": 0.55,
        "crop_group_number": 4.5,
        "crop_group_number_reference": "https://github.com/ajwdewit/WOFOST/blob/deac197d3c74741832b815581699a6c825894758/cropd/cotton.w41",
    },
    {
        "name": "cocoa",
        "id": 21,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.1,
        "P0": 0.15,
        "P1": 0.6,
        "l_ini": 0.0,
        "l_dev": 0.0,
        "l_mid": 100,
        "l_late": 0,
        "kc_initial": 1.05,
        "kc_mid": 1.05,
        "kc_end": 1.05,
        "rd_irr": 0.7,
        "rd_rain": 1.0,
        "reference_yield_kg_m2": 0.15,
        "crop_group_number": 2.5,
        "crop_group_number_reference": "similar to banana, pepper: https://wofost.readthedocs.io/en/7.2/_downloads/cf58a94b422342c8378f99ed36d6eb76/WOFOST_system_description.pdf",
    },
    {
        "name": "coffee",
        "id": 22,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.1,
        "P0": 0.15,
        "P1": 0.6,
        "l_ini": 0.0,
        "l_dev": 0.0,
        "l_mid": 100,
        "l_late": 0,
        "kc_initial": 1.0,
        "kc_mid": 1.0,
        "kc_end": 1.0,
        "rd_irr": 0.9,
        "rd_rain": 1.5,
        "reference_yield_kg_m2": 0.6,
        "crop_group_number": 2.5,
        "crop_group_number_reference": "similar to banana, pepper: https://wofost.readthedocs.io/en/7.2/_downloads/cf58a94b422342c8378f99ed36d6eb76/WOFOST_system_description.pdf",
    },
    {
        "name": "others perennial",
        "id": 23,
        "is_paddy": False,
        "a": 1.2,
        "b": -0.1,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 0.0,
        "l_dev": 0.0,
        "l_mid": 100,
        "l_late": 0,
        "kc_initial": 0.0,
        "kc_mid": 0.8,
        "kc_end": 0.0,
        "rd_irr": 0.8,
        "rd_rain": 1.2,
        "reference_yield_kg_m2": 4.0,
        "crop_group_number": 4.0,
        "crop_group_number_reference": "mixed group, thus impossible to say. However most perennials are quite adapted to droughts, thus 4",
    },
    {
        "name": "fodder grasses",
        "id": 24,
        "is_paddy": False,
        "a": 1.0,
        "b": 0.0,
        "P0": 0.05,
        "P1": 0.2,
        "l_ini": 0.0,
        "l_dev": 0.0,
        "l_mid": 100,
        "l_late": 0,
        "kc_initial": 1.0,
        "kc_mid": 1.0,
        "kc_end": 1.0,
        "rd_irr": 1.0,
        "rd_rain": 1.5,
        "reference_yield_kg_m2": 10.0,
        "crop_group_number": 3.0,
        "crop_group_number_reference": "grass: https://edepot.wur.nl/336784",
    },
    {
        "name": "others annual",
        "id": 25,
        "is_paddy": False,
        "a": 1.2,
        "b": -0.1,
        "P0": 0.1,
        "P1": 0.5,
        "l_ini": 15.0,
        "l_dev": 25.0,
        "l_mid": 40,
        "l_late": 20,
        "kc_initial": 0.4,
        "kc_mid": 1.05,
        "kc_end": 0.5,
        "rd_irr": 1.0,
        "rd_rain": 1.5,
        "reference_yield_kg_m2": 5.0,
        "crop_group_number": 2.5,
        "crop_group_number_reference": "Many similar, taking the average",
    },
]


class Crops(BuildModelBase):
    """Contains all build methods for setting up crops for GEB."""

    def __init__(self) -> None:
        """Initialize the Crops module."""
        pass

    @build_method(depends_on=[], required=True)
    def setup_crops(
        self,
        crop_data: dict | None = None,
        source_type: str = "MIRCA2000",
    ) -> None:
        """Validate and set crop data used by the model.

        Requires crop data to contain specific fields depending on the type.

        For both types, the following fields are required:
        - name
        - reference_yield_kg_m2
        - is_paddy (whether the crop is a paddy irrigated)
        - rd_rain (maximum root depth for rainfed crops)
        - rd_irr (maximum root depth for irrigated crops)
        - crop_group_number (adaptation level to drought, 0-5). See WOFOST documentation.

        For 'GAEZ' type, the following additional fields are required:
        - d1, d2a, d2b, d3a, d3b, d4 (lengths of growth stages)
        - KyT, Ky1, Ky2a, Ky2b, Ky3a, Ky3b, Ky4 (crop coefficients for growth stages)

        For 'MIRCA2000' type, the following additional fields are required:
        - a, b, P0, P1 (parameters for yield response to water)
        - l_ini, l_dev, l_mid, l_late (lengths of growth stages)
        - kc_initial, kc_mid, kc_end (crop coefficients for growth stages)

        Args:
            crop_data: Dictionary keyed by crop id with metadata for each crop.
            source_type: Source/type of crop parameters ('MIRCA2000' or 'GAEZ').

        Raises:
            ValueError: If source_type is not recognized or required fields are missing.
        """
        if crop_data is None:
            if source_type != "MIRCA2000":
                raise ValueError(
                    f"crop_variables_source {source_type} not understood, must be 'MIRCA2000'"
                )

            crop_data = {
                "data": (
                    pd.DataFrame(CROP_DATA).set_index("id").to_dict(orient="index")
                ),
                "type": "MIRCA2000",
            }

            crop_data["data"] = dict(sorted(crop_data["data"].items()))
            self.set_params(crop_data, name="crops/crop_data")

        else:
            if source_type not in ["MIRCA2000", "GAEZ"]:
                raise ValueError(
                    f"crop_variables_source {source_type} not understood, must be 'MIRCA2000' or 'GAEZ'"
                )
            for crop_id, crop_values in crop_data.items():
                assert "name" in crop_values
                assert "reference_yield_kg_m2" in crop_values
                assert "is_paddy" in crop_values
                assert "rd_rain" in crop_values  # root depth rainfed crops
                assert "rd_irr" in crop_values  # root depth irrigated crops
                assert (
                    "crop_group_number" in crop_values
                )  # adaptation level to drought (see WOFOST: https://wofost.readthedocs.io/en/7.2/)
                assert 5 >= crop_values["crop_group_number"] >= 0
                assert (
                    crop_values["rd_rain"] >= crop_values["rd_irr"]
                )  # root depth rainfed crops should be larger than irrigated crops

                if source_type == "GAEZ":
                    crop_values["l_ini"] = crop_values["d1"]
                    crop_values["l_dev"] = crop_values["d2a"] + crop_values["d2b"]
                    crop_values["l_mid"] = crop_values["d3a"] + crop_values["d3b"]
                    crop_values["l_late"] = crop_values["d4"]

                    assert "KyT" in crop_values
                    assert "Ky1" in crop_values
                    assert "Ky2a" in crop_values
                    assert "Ky2b" in crop_values
                    assert "Ky3a" in crop_values
                    assert "Ky3b" in crop_values
                    assert "Ky4" in crop_values

                elif source_type == "MIRCA2000":
                    assert "a" in crop_values
                    assert "b" in crop_values
                    assert "P0" in crop_values
                    assert "P1" in crop_values
                    assert "l_ini" in crop_values
                    assert "l_dev" in crop_values
                    assert "l_mid" in crop_values
                    assert "l_late" in crop_values
                    assert "kc_initial" in crop_values
                    assert "kc_mid" in crop_values
                    assert "kc_end" in crop_values

                assert (
                    crop_values["l_ini"]
                    + crop_values["l_dev"]
                    + crop_values["l_mid"]
                    + crop_values["l_late"]
                    == 100
                ), "Sum of l_ini, l_dev, l_mid, and l_late must be 100[%]"

            crop_data = {
                "data": crop_data,
                "type": source_type,
            }

            self.set_params(crop_data, name="crops/crop_data")

    @build_method(depends_on=[], required=False)
    def setup_crops_from_source(
        self,
        source: str | None = "MIRCA2000",
    ) -> None:
        """Sets up the crops data for the model."""
        self.logger.info("Preparing crops data")

        raise NotImplementedError(
            "setup_crops_from_source is removed, use setup_crops instead."
        )

    def process_crop_data(
        self,
        crop_prices: str | int | float,
        translate_crop_names: dict[str, str] | None = None,
        adjust_currency: bool = False,
    ) -> dict[str, Any]:
        """Process crop price inputs into model-ready time series or constants.

        Args:
            crop_prices: Either 'FAO_stat' to fetch FAO data, a path to JSON prices,
                or a constant numeric price (USD/kg, nominal for the years in question).
            translate_crop_names: Optional mapping from model crop name to list/alias used in source.
            adjust_currency: Whether to convert to USD using currency conversion when available.

        Returns:
            A dictionary with either type='time_series' and per-region series or type='constant'.

        Raises:
            ValueError: If crop_prices is not a valid path, number, or 'FAO_stat'.

        Notes:
            The function performs the following steps:
            1. Fetches and processes crop data from FAO statistics if crop_prices is 'FAO_stat'.
            2. Adjusts the data for countries with missing values using PPP conversion rates.
            3. Determines price variability and performs interpolation/extrapolation of crop prices.
            4. Formats the processed data into a nested dictionary structure.
        """
        if crop_prices == "FAO_stat":
            faostat = self.data_catalog.fetch("faostat_prices").read()
            assert isinstance(faostat, pd.DataFrame)

            all_years_faostat: list[int] = [
                c for c in faostat.columns if isinstance(c, int)
            ]
            all_years_faostat.sort()
            all_crops_faostat = faostat["crop"].unique()

            ISO3_codes_region: set[str] = set(self.geom["regions"]["ISO3"].unique())
            relevant_trade_regions: dict[str, str] = {
                ISO3: TRADE_REGIONS[ISO3]
                for ISO3 in ISO3_codes_region
                if ISO3 in TRADE_REGIONS
            }

            all_ISO3_across_relevant_regions: set[str] = set(
                relevant_trade_regions.keys()
            )

            # Setup dataFrame for further data corrections
            donor_data: dict[str, pd.DataFrame] = {}
            for ISO3 in all_ISO3_across_relevant_regions:
                region_faostat: pd.DataFrame = (
                    faostat[faostat["ISO3"] == ISO3]
                    .set_index("crop")
                    .transpose()
                    .reindex(index=all_years_faostat, columns=all_crops_faostat)
                ).astype(np.float64)

                region_faostat["ISO3"] = ISO3
                donor_data[ISO3] = region_faostat

            # Concatenate all regional data into a single DataFrame with MultiIndex
            donor_data: pd.DataFrame = pd.concat(donor_data, names=["ISO3", "year"])

            # Drop crops with no data at all for these regions
            donor_data = donor_data.dropna(axis=1, how="all")

            # Filter out columns that contain the word 'meat'
            donor_data = donor_data[
                [
                    column
                    for column in donor_data.columns
                    if "meat" not in column.lower()
                ]
            ]

            national_data = False
            # Check whether there is national or subnational data
            duplicates = donor_data.index.duplicated(keep=False)
            if duplicates.any():
                # Data is subnational
                unique_regions = self.geom["regions"]
            else:
                # Data is national
                unique_regions = (
                    self.geom["regions"].groupby("ISO3").first().reset_index()
                )
                national_data = True

            # filter for model start and end year (important to do this before donation)
            donor_data = donor_data.loc[
                (slice(None), slice(self.start_date.year, self.end_date.year)), :
            ]

            # here, also countries that are not in the trade regions (e.g. Kosovo) are included (in self.geom["regions"]) and found a donor for (in the setup_donor_countries function)
            data = donate_and_receive_crop_prices(
                donor_data,
                unique_regions,
                TRADE_REGIONS,
                self.data_catalog,
                self.geom["global_countries"],
                self.geom["regions"],
            )

            # exand data to include all data empty rows from start to end year
            data = data.reindex(
                pd.MultiIndex.from_product(
                    [
                        unique_regions["region_id"],
                        range(self.start_date.year, self.end_date.year + 1),
                    ],
                    names=["region_id", "year"],
                )
            )

            data = self.assign_crop_price_inflation(data, unique_regions)

            # combine and rename crops
            all_crop_names_model = [
                d["name"] for d in self.params["crops/crop_data"]["data"].values()
            ]
            for crop_name in all_crop_names_model:
                if (
                    translate_crop_names is not None
                    and crop_name in translate_crop_names
                ):
                    sub_crops = [
                        crop
                        for crop in translate_crop_names[crop_name]
                        if crop in data.columns
                    ]
                    if sub_crops:
                        data[crop_name] = data[sub_crops].mean(axis=1, skipna=True)
                    else:
                        data[crop_name] = np.nan
                else:
                    if crop_name not in data.columns:
                        data[crop_name] = np.nan

            # Extract the crop names from the dictionary and convert them to lowercase
            crop_names = [
                crop["name"].lower()
                for crop in self.params["crops/crop_data"]["data"].values()
            ]

            # Filter the columns of the data DataFrame
            data = data[
                [
                    col
                    for col in data.columns
                    if col.lower() in crop_names
                    or col in ("_crop_price_inflation", "_crop_price_LCU_USD")
                ]
            ]

            data: pd.DataFrame = self.inter_and_extrapolate_prices(data, unique_regions)

            # Create a dictionary structure with regions as keys and crops as nested dictionaries
            # This is the required format for crop_farmers.py
            crop_data = self.params["crops/crop_data"]["data"]
            time_index = data.index.get_level_values("year").unique().tolist()
            data_per_region: dict[str, dict] = {}

            # If national_data is True, create a mapping from ISO3 code to representative region_id
            if national_data:
                unique_regions = data.index.get_level_values("region_id").unique()
                iso3_codes = (
                    self.geom["regions"]
                    .set_index("region_id")
                    .loc[unique_regions]["ISO3"]
                )
                iso3_to_representative_region_id = dict(zip(iso3_codes, unique_regions))

            for _, region in self.geom["regions"].iterrows():
                region_dict = {}
                region_id = region["region_id"]
                region_iso3 = region["ISO3"]

                # Determine the region_id to use based on national_data
                if national_data:
                    # Use the representative region_id for this ISO3 code
                    selected_region_id = iso3_to_representative_region_id.get(
                        region_iso3
                    )
                else:
                    # Use the actual region_id
                    selected_region_id = region_id

                # Fetch the data for the selected region_id
                if selected_region_id in data.index.get_level_values("region_id"):
                    region_data = data.loc[selected_region_id]
                else:
                    # If data is not available for the region, fill with NaNs
                    region_data = pd.DataFrame(
                        np.nan, index=time_index, columns=data.columns
                    )

                region_data.index.name = "year"  # Ensure index name is 'year'

                crop_calendars_in_region = self.array["agents/farmers/crop_calendar"][
                    self.array["agents/farmers/region_id"] == region_id
                ]
                crops_in_region = crop_calendars_in_region[..., 0].ravel()
                crops_in_region = np.unique(crops_in_region[crops_in_region != -1])

                # Ensuring all crops are present according to the crop_data keys
                for crop_id, crop_info in crop_data.items():
                    crop_name = crop_info["name"]

                    if crop_name.endswith("_flood") or crop_name.endswith("_drought"):
                        crop_name = crop_name.rsplit("_", 1)[0]

                    if crop_name in region_data.columns:
                        # raise an error if the crop is in the crop calendar and has NaN values
                        if (
                            float(crop_id) in crops_in_region
                            and np.isnan(region_data[crop_name]).any()
                        ):
                            raise ValueError(
                                f"Crop {crop_name} has NaN values in region {region_id} data."
                            )
                        region_dict[str(crop_id)] = region_data[crop_name].tolist()
                    # check if crop is in the crop calendar, if is raise an error because it must be
                    elif crop_id in crops_in_region:
                        raise ValueError(
                            f"Crop {crop_name} not found in region {region_id} data, but is in crop calendar."
                        )
                    else:
                        # If data is not available for the crop, but is not in the crop calendar, it
                        # is no issue, so we can fill with NaNs
                        region_dict[str(crop_id)] = [np.nan] * len(time_index)
                data_per_region[str(region_id)] = region_dict

            parsed_crop_prices: dict[str, str | dict | list[int]] = {
                "type": "time_series",
                "data": data_per_region,
                "time": time_index,  # Extract unique years for the time key
            }

        # data is a file path
        elif isinstance(crop_prices, str):
            crop_prices_path = Path(crop_prices)
            if not crop_prices_path.exists():
                raise ValueError(f"file {crop_prices_path.resolve()} does not exist")
            with open(crop_prices_path) as f:
                data = json.load(f)
            data = pd.DataFrame(
                {
                    crop_id: data["crops"][crop_data["name"]]
                    for crop_id, crop_data in self.params["crops/crop_data"][
                        "data"
                    ].items()
                },
                index=pd.to_datetime(data["time"]),
            )
            # compute mean price per year, using start day as index
            data = data.resample("AS").mean()
            # extend dataframe to include start and end years
            data = data.reindex(
                index=pd.date_range(
                    start=self.start_date,
                    end=self.end_date,
                    freq="YS",
                )
            )
            # only use year identifier as index
            data.index = data.index.year

            data = data.reindex(
                index=pd.MultiIndex.from_product(
                    [
                        self.geom["regions"]["region_id"],
                        data.index,
                    ],
                    names=["region_id", "date"],
                ),
                level=1,
            )

            data = self.assign_crop_price_inflation(data, self.geom["regions"])
            data = self.inter_and_extrapolate_prices(
                data, self.geom["regions"], adjust_currency
            )

            parsed_crop_prices = {
                "type": "time_series",
                "time": data.xs(
                    data.index.get_level_values(0)[0], level=0
                ).index.tolist(),
                "data": {
                    str(region_id): data.loc[region_id].to_dict(orient="list")
                    for region_id in self.geom["regions"]["region_id"]
                },
            }

        elif isinstance(crop_prices, (int, float)):
            parsed_crop_prices: dict[str, str | int | float | dict] = {
                "type": "constant",
                "data": crop_prices,
            }
        else:
            raise ValueError(
                f"must be a file path or an integer, got {type(crop_prices)}"
            )

        return parsed_crop_prices

    def assign_crop_price_inflation(
        self, costs: pd.DataFrame, unique_regions: pd.DataFrame
    ) -> pd.DataFrame:
        """Determines the price inflation of all crops in the region and adds a column that describes this inflation.

        If there is no data for a certain year, the inflation rate is taken from the socioeconomics data.

        Args:
            costs: A DataFrame containing crop prices for different regions. The DataFrame should be indexed by region IDs.
            unique_regions: A DataFrame containing unique regions with their IDs and other attributes.

        Returns:
            The updated DataFrame with a new column 'changes' that contains the average price changes for each region.

        To Do:
            Is it possible to use the regions from the costs DataFrame instead of the unique_regions DataFrame?

        """
        costs["_crop_price_inflation"] = np.nan
        costs["_crop_price_LCU_USD"] = np.nan

        # Determine the average changes of price of all crops in the region and add it to the data
        for _, region in unique_regions.iterrows():
            region_id = region["region_id"]
            region_data = costs.loc[region_id]

            # only consider years with at least 1 crops with data
            region_data = region_data.interpolate(
                method="linear", limit_direction="both"
            )
            region_data = region_data[(~region_data.isnull()).sum(axis=1) > 0]

            changes = np.nanmean(
                region_data[1:].to_numpy() / region_data[:-1].to_numpy(), axis=1
            )
            changes = np.insert(changes, 0, np.nan)

            for year, change in zip(region_data.index, changes, strict=True):
                costs.at[(region_id, year), "_crop_price_inflation"] = change

            region_inflation_rates = self.params["socioeconomics/inflation_rates"][
                "data"
            ][str(region["region_id"])]
            region_currency_conversion_rates = self.params[
                "socioeconomics/LCU_per_USD"
            ]["data"][str(region["region_id"])]

            for year, row in costs.loc[region_id].iterrows():
                crop_inflation_rate = row["_crop_price_inflation"]
                year_currency_conversion = region_currency_conversion_rates[
                    self.params["socioeconomics/LCU_per_USD"]["time"].index(str(year))
                ]
                costs.at[(region_id, year), "_crop_price_LCU_USD"] = (
                    year_currency_conversion
                )
                if np.isnan(crop_inflation_rate):
                    year_inflation_rate = region_inflation_rates[
                        self.params["socioeconomics/inflation_rates"]["time"].index(
                            str(year)
                        )
                    ]
                    costs.at[(region_id, year), "_crop_price_inflation"] = (
                        year_inflation_rate
                    )

        return costs

    def inter_and_extrapolate_prices(
        self,
        data: pd.DataFrame,
        unique_regions: pd.DataFrame,
        adjust_currency: bool = False,
    ) -> pd.DataFrame:
        """Interpolates and extrapolates crop prices for different regions based on the given data and predefined crop categories.

        Args:
            data: A DataFrame containing crop price data for different regions. The DataFrame should be indexed by region IDs
            and have columns corresponding to different crops.
            unique_regions: A DataFrame containing unique regions with their IDs and other attributes.
            adjust_currency: If True, adjusts the crop prices based on currency conversion rates.

        Returns:
            Updated DataFrame with interpolated and extrapolated crop prices. Columns for 'others perennial' and 'others annual'
                crops are also added.

        Notes:
            The function performs the following steps:
                1. Extracts crop names from the internal crop data dictionary.
                2. Defines additional crops that fall under 'others perennial' and 'others annual' categories.
                3. Processes the data to compute average prices for these additional crops.
                4. Filters and updates the original data with the computed averages.
                5. Interpolates and extrapolates missing prices for each crop in each region based on the 'changes' column.

        To Do:
            Ensure adjust_currency is better explained and used correctly.
        """
        # Interpolate and extrapolate missing prices for each crop in each region based on the 'changes' column
        for _, region in unique_regions.iterrows():
            region_id = region["region_id"]
            region_data = data.loc[region_id]

            n = len(region_data)

            for crop in region_data.columns:
                if crop == "_crop_price_inflation":
                    continue
                crop_data = region_data[crop].to_numpy()
                if np.isnan(crop_data).all():
                    continue
                changes_data = region_data["_crop_price_inflation"].to_numpy()
                k = -1
                while np.isnan(crop_data[k]):
                    k -= 1
                for i in range(k + 1, 0, 1):
                    crop_data[i] = crop_data[i - 1] * changes_data[i]
                k = 0
                while np.isnan(crop_data[k]):
                    k += 1
                for i in range(k - 1, -1, -1):
                    crop_data[i] = crop_data[i + 1] / changes_data[i + 1]
                for j in range(0, n):
                    if np.isnan(crop_data[j]):
                        k = j
                        while np.isnan(crop_data[k]):
                            k += 1
                        empty_size = k - j
                        step_crop_price_inflation = changes_data[j : k + 1]
                        total_crop_price_inflation = np.prod(step_crop_price_inflation)
                        real_crop_price_inflation = crop_data[k] / crop_data[j - 1]
                        scaled_crop_price_inflation = (
                            step_crop_price_inflation
                            * (real_crop_price_inflation ** (1 / empty_size))
                            / (total_crop_price_inflation ** (1 / empty_size))
                        )
                        for i, change in zip(range(j, k), scaled_crop_price_inflation):
                            crop_data[i] = crop_data[i - 1] * change
                if adjust_currency and not crop == "_crop_price_LCU_USD":
                    conversion_data = region_data["_crop_price_LCU_USD"].to_numpy()
                    data.loc[region_id, crop] = crop_data / conversion_data
                else:
                    data.loc[region_id, crop] = crop_data

        # remove columns that are not needed anymore
        data = data.drop(columns=["_crop_price_inflation"])
        data = data.drop(columns=["_crop_price_LCU_USD"])

        return data

    @build_method(depends_on=["set_time_range"], required=False)
    def setup_cultivation_costs(
        self,
        cultivation_costs: str | int | float = 0,
        translate_crop_names: dict[str, str] | None = None,
        adjust_currency: bool = False,
    ) -> None:
        """Set cultivation costs per crop and region for the model run.

        Args:
            cultivation_costs: 'FAO_stat', file path, or constant (USD/kg, nominal).
            translate_crop_names: Optional mapping to aggregate/rename source crop columns.
            adjust_currency: Whether to convert to USD using currency conversion when available.
        """
        parsed_cultivation_costs = self.process_crop_data(
            crop_prices=cultivation_costs,
            translate_crop_names=translate_crop_names,
            adjust_currency=adjust_currency,
        )
        self.set_params(parsed_cultivation_costs, name="crops/cultivation_costs")

    @build_method(
        depends_on=[
            "set_time_range",
            "setup_regions_and_land_use",
            "setup_economic_data",
            "setup_crops",
            "setup_farmer_crop_calendar",
        ],
        required=True,
    )
    def setup_crop_prices(
        self,
        crop_prices: str | int | float = "FAO_stat",
        translate_crop_names: dict[str, str] | None = None,
        adjust_currency: bool = False,
    ) -> None:
        """Set crop prices per crop and region for the model run.

        Args:
            crop_prices: 'FAO_stat', file path, or constant (USD/kg, nominal).
            translate_crop_names: Optional mapping to aggregate/rename source crop columns.
            adjust_currency: Whether to convert to USD using currency conversion when available.
        """
        parsed_crop_prices = self.process_crop_data(
            crop_prices=crop_prices,
            translate_crop_names=translate_crop_names,
            adjust_currency=adjust_currency,
        )
        self.set_params(parsed_crop_prices, name="crops/crop_prices")
        self.set_params(parsed_crop_prices, name="crops/cultivation_costs")

    @build_method(depends_on=[], required=False)
    def determine_crop_area_fractions(self, resolution: str = "5-arcminute") -> None:
        """This method is removed. You can remove it entirely.

        Args:
            resolution: Resolution tag for plotting/output naming.

        Raises:
            ValueError: This method is removed.
        """
        raise ValueError("This method is removed. You can remove it entirely.")

    def get_crop_area_fractions(
        self, year: int, resolution: str = "5-arcminute"
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute MIRCA crop area fractions and summarize per region.

        Args:
            year: The year for which to compute crop area fractions.
            resolution: Resolution tag for plotting/output naming.

        Returns:
            A tuple containing two xarray DataArrays:
            - rainfed_crop_fraction: DataArray with dimensions (crop, y, x) representing the fraction of rainfed crop area for each crop.
            - irrigated_crop_fraction: DataArray with dimensions (crop, y, x) representing the fraction of irrigated crop area for each crop.
        """
        crops: dict[str, int] = {
            "Wheat": 0,
            "Maize": 1,
            "Rice": 2,
            "Barley": 3,
            "Rye": 4,
            "Millet": 5,
            "Sorghum": 6,
            "Soybeans": 7,
            "Sunflower": 8,
            "Potatoes": 9,
            "Cassava": 10,
            "Sugar_cane": 11,
            "Sugar_beet": 12,
            "Oil_palm": 13,
            "Rapeseed": 14,
            "Groundnuts": 15,
            "Pulses": 16,
            "Citrus": 17,
            "Date palms": 18,
            "Grapes": 19,
            "Cotton": 20,
            "Cocoa": 21,
            "Coffee": 22,
            "Others_perennial": 23,
            "Fodder": 24,
            "Others_annual": 25,
        }
        missing_crops = ["Citrus", "Date palms", "Grapes", "Cotton", "Cocoa", "Coffee"]

        irrigation_types: list[str] = ["ir", "rf"]

        # the datasets have slightly different grids, so we need to snap them to the same grid. We take the first one as reference and snap the others to it
        reference: None | xr.DataArray = None

        crop_maps_per_irrigation_type: dict[str, xr.DataArray] = {}
        for irrigation_type in irrigation_types:
            crop_maps: list[xr.DataArray] = []
            for crop_name, crop_id in crops.items():
                if crop_name in missing_crops:
                    assert reference is not None
                    # when crop is not available in MIRCA-OS, set to nan
                    crop_map = self.full_like(
                        reference, fill_value=np.nan, nodata=np.nan
                    )
                else:
                    crop_map = self.data_catalog.fetch(
                        f"mirca_os_cropping_area_{year}_{resolution}_{crop_name}_{irrigation_type}"
                    ).read()

                    crop_map = crop_map.isel(
                        get_window(
                            crop_map.x,
                            crop_map.y,
                            self.bounds,
                            buffer=100,
                            raise_on_buffer_out_of_bounds=False,
                        )  # use a very large buffer so that we use don't get edge effects in the interpolation
                    ).compute()

                    # do the snapping unless it is the first dataset, then we take this one as reference for the others
                    # see above comment for more info
                    if reference is None:
                        reference = crop_map.copy()
                    else:
                        # some maps are smaller than the reference, so we need to pad them
                        # with np.nan values, so that they can be combined in one dataset.
                        crop_map = pad_xy(
                            crop_map,
                            reference.x[0].item(),
                            reference.y[0].item(),
                            reference.x[-1].item(),
                            reference.y[-1].item(),
                            constant_values=np.nan,
                        )
                        crop_map = snap_to_grid(crop_map, reference)

                crop_map = crop_map.assign_coords(crop=crop_id)

                crop_maps.append(crop_map)

            # Concatenate the list of fractions into a single DataArray along the 'crop' dimension
            crop_map: xr.DataArray = xr.concat(crop_maps, dim="crop")
            crop_maps_per_irrigation_type[irrigation_type] = crop_map

        # MIRCA does not make a distinction between no cropping and missing data.
        # So we first need to find where specific crops are missing but others are defined,
        # and set those to 0. Then we can interpolate the remaining missing values so
        # that we have a complete map of crop fractions.

        # When there is crop data for some crops in a cell, but not for others, we assume that the
        # nan value should actually be 0.
        has_crops_data = ~(
            np.isnan(crop_maps_per_irrigation_type["ir"]).all(dim="crop")  # ty:ignore[no-matching-overload]
            & np.isnan(crop_maps_per_irrigation_type["rf"]).all(dim="crop")  # ty:ignore[no-matching-overload]
        )
        crop_maps_per_irrigation_type["ir"] = xr.where(
            has_crops_data, crop_maps_per_irrigation_type["ir"].fillna(0), np.nan
        )
        crop_maps_per_irrigation_type["rf"] = xr.where(
            has_crops_data, crop_maps_per_irrigation_type["rf"].fillna(0), np.nan
        )

        # Then, we look at all cells again, and fill cells with no crops with nan, so that we can interpolate those later on.
        # There are two cases where we want to do this:
        # - ocean cells that have all nan data (and thus no crops)
        # - cells that have crop data but sum to 0. We need some data here, and we want to fill it eventually with data from the closest cell
        no_crops_data_in_total = (
            crop_maps_per_irrigation_type["ir"].sum(dim="crop", skipna=True)
            + crop_maps_per_irrigation_type["rf"].sum(dim="crop", skipna=True)
            == 0
        )
        crop_maps_per_irrigation_type["ir"] = (
            crop_maps_per_irrigation_type["ir"]
            .where(~no_crops_data_in_total, np.nan)
            .transpose("crop", "y", "x")
        )
        crop_maps_per_irrigation_type["rf"] = (
            crop_maps_per_irrigation_type["rf"]
            .where(~no_crops_data_in_total, np.nan)
            .transpose("crop", "y", "x")
        )

        # then finally, all cells without crop data are interpolated using the nearest neighbor cell with crop data.
        # Now we should have data everywhere.
        crop_maps_per_irrigation_type["ir"] = interpolate_na_along_dim(
            crop_maps_per_irrigation_type["ir"],
            dim="crop",
        )
        crop_maps_per_irrigation_type["rf"] = interpolate_na_along_dim(
            crop_maps_per_irrigation_type["rf"],
            dim="crop",
        )

        # Ensure that we indeed have data everywhere (no nans, no cells with all zeros)
        assert not np.isnan(crop_maps_per_irrigation_type["rf"].values).any()
        assert not np.isnan(crop_maps_per_irrigation_type["ir"].values).any()

        crop_area_irrigated = crop_maps_per_irrigation_type["ir"].sum(dim="crop")
        crop_area_rainfed = crop_maps_per_irrigation_type["rf"].sum(dim="crop")
        total_crop_area = crop_area_irrigated + crop_area_rainfed

        assert (total_crop_area > 0).all(), (
            "Total crop area must be greater than zero to compute fractions."
        )

        # Normalize to get fractions
        rainfed_crop_fraction = (crop_maps_per_irrigation_type["rf"]) / total_crop_area
        irrigated_crop_fraction = crop_maps_per_irrigation_type["ir"] / total_crop_area

        rainfed_crop_fraction = rainfed_crop_fraction.rio.write_crs(4326)
        irrigated_crop_fraction = irrigated_crop_fraction.rio.write_crs(4326)

        # reduce map to the area of interest, with a buffer of 2 cells to avoid edge effects
        rainfed_crop_fraction = rainfed_crop_fraction.isel(
            get_window(
                rainfed_crop_fraction.x, rainfed_crop_fraction.y, self.bounds, buffer=2
            )
        )
        irrigated_crop_fraction = irrigated_crop_fraction.isel(
            get_window(
                irrigated_crop_fraction.x,
                irrigated_crop_fraction.y,
                self.bounds,
                buffer=2,
            )
        )

        assert np.allclose(
            (rainfed_crop_fraction + irrigated_crop_fraction).sum(dim="crop"),
            1.0,
            atol=1e-5,
        )
        return rainfed_crop_fraction, irrigated_crop_fraction

    @build_method(depends_on=[], required=False)
    def setup_farmer_crop_calendar_multirun(
        self,
        reduce_crops: bool = False,
        replace_base: bool = False,
        export: bool = False,
    ) -> None:
        """Generate crop calendars for multiple years for multirun scenarios."""
        years = [2000, 2005, 2010, 2015]
        nr_runs = 20

        for year_nr in years:
            for run in range(nr_runs):
                self.setup_farmer_crop_calendar(
                    year_nr, reduce_crops, replace_base, export
                )

    def setup_farmer_irrigation_source(
        self, irrigating_farmers: ArrayBool, year: int
    ) -> None:
        """Sets up the irrigation source for farmers based on global irrigation area data.

        Args:
            irrigating_farmers: A boolean array indicating which farmers are irrigating.
            year: The year for which to set up the irrigation source.
        """
        fraction_sw_irrigation_data = self.data_catalog.fetch(
            "global_irrigation_area_surface_water"
        ).read()
        fraction_sw_irrigation_data.attrs["_FillValue"] = np.nan

        fraction_sw_irrigation_data = fraction_sw_irrigation_data.isel(
            get_window(
                fraction_sw_irrigation_data.x,
                fraction_sw_irrigation_data.y,
                self.bounds,
                buffer=5,
            ),
        )
        fraction_sw_irrigation_data: xr.DataArray = interpolate_na_2d(
            fraction_sw_irrigation_data
        )

        fraction_gw_irrigation_data = self.data_catalog.fetch(
            "global_irrigation_area_groundwater"
        ).read()
        fraction_gw_irrigation_data.attrs["_FillValue"] = np.nan

        fraction_gw_irrigation_data = fraction_gw_irrigation_data.isel(
            get_window(
                fraction_gw_irrigation_data.x,
                fraction_gw_irrigation_data.y,
                self.bounds,
                buffer=5,
            ),
        )
        fraction_gw_irrigation_data: xr.DataArray = interpolate_na_2d(
            fraction_gw_irrigation_data
        )

        farmer_locations = get_farm_locations(
            self.subgrid["agents/farmers/farms"], method="centroid"
        )

        # Determine which farmers are irrigating
        grid_id_da = get_linear_indices(fraction_sw_irrigation_data)
        ny, nx = (
            fraction_sw_irrigation_data.sizes["y"],
            fraction_sw_irrigation_data.sizes["x"],
        )

        n_cells = grid_id_da.max().item()
        n_farmers = self.array["agents/farmers/region_id"].size

        farmer_cells = sample_from_map(
            grid_id_da.values,
            farmer_locations,
            grid_id_da.rio.transform(recalc=True).to_gdal(),
        )
        fraction_sw_irrigation_farmers = sample_from_map(
            fraction_sw_irrigation_data.values,
            farmer_locations,
            fraction_sw_irrigation_data.rio.transform(recalc=True).to_gdal(),
        )
        fraction_gw_irrigation_farmers = sample_from_map(
            fraction_gw_irrigation_data.values,
            farmer_locations,
            fraction_gw_irrigation_data.rio.transform(recalc=True).to_gdal(),
        )

        adaptations = np.full(
            (
                n_farmers,
                max(
                    [
                        FIELD_EXPANSION_ADAPTATION,
                        INDEX_INSURANCE_ADAPTATION,
                        IRRIGATION_EFFICIENCY_ADAPTATION_DRIP,
                        IRRIGATION_EFFICIENCY_ADAPTATION_SPRINKLER,
                        TRADITIONAL_INSURANCE_ADAPTATION,
                        PR_INSURANCE_ADAPTATION,
                        SURFACE_IRRIGATION_EQUIPMENT,
                        WELL_ADAPTATION,
                    ]
                )
                + 1,
            ),
            0,
            dtype=np.bool_,
        )

        for i in range(n_cells):
            farmers_cell_mask = farmer_cells == i  # Boolean mask for farmers in cell i
            farmers_cell_indices = np.where(farmers_cell_mask)[0]  # Absolute indices

            irrigating_farmers_mask = irrigating_farmers[farmers_cell_mask]
            num_irrigating_farmers = np.sum(irrigating_farmers_mask)

            if num_irrigating_farmers > 0:
                fraction_sw = fraction_sw_irrigation_farmers[farmers_cell_mask][0]
                fraction_gw = fraction_gw_irrigation_farmers[farmers_cell_mask][0]

                # Normalize fractions
                total_fraction = fraction_sw + fraction_gw

                # Handle edge cases if there are irrigating farmers but no data on sw/gw
                if total_fraction == 0:
                    # Find neighboring cells with valid data
                    neighbor_ids = get_neighbor_cell_ids_for_linear_indices(i, nx, ny)
                    found_valid_neighbor = False

                    for neighbor_id in neighbor_ids:
                        if neighbor_id not in np.unique(farmer_cells):
                            continue

                        neighbor_mask = farmer_cells == neighbor_id
                        fraction_sw_neighbor = fraction_sw_irrigation_farmers[
                            neighbor_mask
                        ][0]
                        fraction_gw_neighbor = fraction_gw_irrigation_farmers[
                            neighbor_mask
                        ][0]
                        neighbor_total_fraction = (
                            fraction_sw_neighbor + fraction_gw_neighbor
                        )

                        if neighbor_total_fraction > 0:
                            # Found valid neighbor
                            fraction_sw = fraction_sw_neighbor
                            fraction_gw = fraction_gw_neighbor
                            total_fraction = neighbor_total_fraction

                            found_valid_neighbor = True
                            break
                    if not found_valid_neighbor:
                        # No valid neighboring cells found, handle accordingly
                        self.logger.warning(
                            f"No valid data found for cell {i} and its neighbors."
                        )
                        continue  # Skip this cell

                # Normalize fractions
                probabilities = np.array([fraction_sw, fraction_gw], dtype=np.float64)
                probabilities_sum = probabilities.sum()
                probabilities /= probabilities_sum

                # Indices of irrigating farmers in the region (absolute indices)
                farmer_indices_in_region = farmers_cell_indices[irrigating_farmers_mask]

                # Assign irrigation sources using np.random.choice
                irrigation_equipment_per_farmer = np.random.choice(
                    [SURFACE_IRRIGATION_EQUIPMENT, WELL_ADAPTATION],
                    size=len(farmer_indices_in_region),
                    p=probabilities,
                )

                adaptations[
                    farmer_indices_in_region, irrigation_equipment_per_farmer
                ] = True

        self.set_array(adaptations, name="agents/farmers/adaptations")

    @build_method(depends_on=["setup_create_farms"], required=True)
    def setup_farmer_crop_calendar(
        self,
        year: int = 2000,
        reduce_crops: bool = False,
        unify_variants: bool = False,
        replace_base: bool = False,
        minimum_area_ratio: float = 0.01,
        replace_crop_calendar_unit_code: dict = {},
    ) -> None:
        """Build per-farmer crop calendars for a single reference year.

        Args:
            year: Reference year (calendar year).
            reduce_crops: If True, reduce the number of crops per calendar based on area.
            unify_variants: If True, make different cropping patterns of the same crop into one.
            replace_base: If True, replace base crop definitions with alternatives.
            minimum_area_ratio: Threshold for considering a crop present in a unit.
            replace_crop_calendar_unit_code: Optional mapping to replace MIRCA unit codes.

        Raises:
            ValueError: If no rotations are found for a crop in a unit or no valid neighbor data is found.
        """
        n_farmers = self.array["agents/farmers/region_id"].size

        MIRCA_unit_grid = (
            self.data_catalog.fetch("mirca2000_unit_grid").read().astype(np.int32)
        ).isel(band=0)
        assert isinstance(MIRCA_unit_grid, xr.DataArray)

        MIRCA_unit_grid = MIRCA_unit_grid.isel(
            get_window(MIRCA_unit_grid.x, MIRCA_unit_grid.y, self.bounds, buffer=2)
        )

        crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]] = (
            parse_MIRCA2000_crop_calendar(
                self.data_catalog,
                MIRCA_units=np.unique(MIRCA_unit_grid.values).tolist(),
            )
        )

        def fix_365_in_crop_calendar(
            crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
        ) -> dict[int, list[tuple[float, TwoDArrayInt32]]]:
            """Replace any 365 day-of-year values with 364 in the 4th column.

            Scans each (area, arr) pair in every dictionary entry. If a value 365 is
            found, it asserts that it appears only in column index 3 and then rewrites
            it to 364. Increments a running count of replacements and raises a
            ValueError if a 365 is found outside column 3.

            Raises:
                ValueError: If any 365 is found outside column index 3 (the 4th column).

            Returns:
                A dictionary of crop calendars where the 365 length crops are now 364 days.
            """
            total_replacements = 0

            crop_calendar_adjusted = crop_calendar.copy()

            for key, entries in crop_calendar_adjusted.items():
                for i, (area, arr) in enumerate(entries):
                    rows, cols = np.where(arr == 365)

                    if rows.size == 0:
                        continue  # nothing to change in this array

                    # Safety: all 365s must be in column index 3 (4th column)
                    if not np.all(cols == 3):
                        raise ValueError(
                            f"Found 365 outside column 3 for key={key}, index={i}: "
                            f"indices={list(zip(rows, cols))}"
                        )

                    # Do the replacement
                    arr[rows, 3] = 364
                    entries[i] = (area, arr)
                    total_replacements += rows.size

            return crop_calendar_adjusted

        # Replace crop growth time of 365 with 364 as 365 leads to many issues
        crop_calendar = fix_365_in_crop_calendar(crop_calendar)

        if any(value in [None, "", [], {}] for value in crop_calendar.values()):
            missing_mirca_unit = [
                unit for unit, calendars in crop_calendar.items() if not calendars
            ]
            self.logger.warning(
                f"Missing crop calendar for MIRCA unit(s): {missing_mirca_unit}"
            )

            for mirca_unit in missing_mirca_unit:
                # Filter out the current mirca_unit from crop_calendar.keys()
                valid_keys = [key for key in crop_calendar.keys() if key != mirca_unit]

                # Find the closest MIRCA unit with a crop calendar
                if valid_keys:  # Ensure there are valid keys to process
                    closest_mirca_unit = min(
                        valid_keys, key=lambda x: abs(x - mirca_unit)
                    )
                else:
                    raise ValueError(
                        f"No valid MIRCA units found to replace missing crop calendar for {mirca_unit}."
                    )

                # use this closest_mirca_unit to fill the missing crop calendar
                crop_calendar[mirca_unit] = crop_calendar[closest_mirca_unit]
                self.logger.info(
                    f"Filling missing crop calendar for MIRCA unit {mirca_unit} with data from {closest_mirca_unit}."
                )

        else:
            self.logger.debug("All keys have valid values.")

        farmer_locations = get_farm_locations(
            self.subgrid["agents/farmers/farms"], method="centroid"
        )

        farmer_mirca_units = sample_from_map(
            MIRCA_unit_grid.values,
            farmer_locations,
            MIRCA_unit_grid.rio.transform(recalc=True).to_gdal(),
        )

        farmer_crops, is_irrigated = self.assign_crops(
            crop_calendar,
            farmer_locations,
            farmer_mirca_units,
            year,
            MIRCA_unit_grid,
            minimum_area_ratio=minimum_area_ratio,
            replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
        )

        self.setup_farmer_irrigation_source(is_irrigated, year)

        all_farmers_assigned = []

        crop_calendar_per_farmer = np.full((n_farmers, 3, 4), -1, dtype=np.int32)
        for mirca_unit in np.unique(farmer_mirca_units):
            farmers_in_unit = np.where(farmer_mirca_units == mirca_unit)[0]

            area_per_crop_rotation = []
            cropping_calenders_crop_rotation = []
            for crop_rotation in crop_calendar[
                replace_crop_calendar_unit_code.get(mirca_unit, mirca_unit)
            ]:
                area_per_crop_rotation.append(crop_rotation[0])
                crop_rotation_matrix = crop_rotation[1]
                starting_days = crop_rotation_matrix[:, 2]
                starting_days = starting_days[starting_days != -1]
                assert np.unique(starting_days).size == starting_days.size, (
                    "ensure all starting days are unique"
                )
                # TODO: Add check to ensure crop calendars are not overlapping.
                cropping_calenders_crop_rotation.append(crop_rotation_matrix)
            area_per_crop_rotation = np.array(area_per_crop_rotation)
            cropping_calenders_crop_rotation = np.stack(
                cropping_calenders_crop_rotation
            )

            crops_in_unit = np.unique(farmer_crops[farmers_in_unit])
            for crop_id in crops_in_unit:
                # Find rotations that include this crop
                rotations_with_crop_idx = []
                for idx, rotation in enumerate(cropping_calenders_crop_rotation):
                    # Get crop IDs in the rotation, excluding -1 entries
                    crop_ids_in_rotation = rotation[:, 0]
                    crop_ids_in_rotation = crop_ids_in_rotation[
                        crop_ids_in_rotation != -1
                    ]
                    if crop_id in crop_ids_in_rotation:
                        rotations_with_crop_idx.append(idx)

                if not rotations_with_crop_idx:
                    raise ValueError(
                        f"No rotations found for crop ID {crop_id} in mirca unit {mirca_unit}"
                    )

                # Get the area fractions and rotations for these indices
                areas_with_crop = area_per_crop_rotation[rotations_with_crop_idx]
                rotations_with_crop = cropping_calenders_crop_rotation[
                    rotations_with_crop_idx
                ]

                # Normalize the area fractions
                total_area_for_crop = areas_with_crop.sum()
                fractions = areas_with_crop / total_area_for_crop

                # Get farmers with this crop in the mirca_unit
                farmers_with_crop_in_unit = farmers_in_unit[
                    farmer_crops[farmers_in_unit] == crop_id
                ]

                # Assign crop rotations to these farmers
                assigned_rotation_indices = np.random.choice(
                    np.arange(len(rotations_with_crop)),
                    size=len(farmers_with_crop_in_unit),
                    replace=True,
                    p=fractions,
                )

                # Assign the crop calendars to the farmers
                for farmer_idx, rotation_idx in zip(
                    farmers_with_crop_in_unit, assigned_rotation_indices
                ):
                    assigned_rotation = rotations_with_crop[rotation_idx]
                    # Assign to farmer's crop calendar, taking columns [0, 2, 3, 4]
                    # Columns: [crop_id, planting_date, harvest_date, additional_attribute]
                    crop_calendar_per_farmer[farmer_idx] = assigned_rotation[
                        :, [0, 2, 3, 4]
                    ]
                    all_farmers_assigned.append(farmer_idx)

        def check_crop_calendar(crop_calendar_per_farmer: np.ndarray) -> None:
            """Validate that no overlapping crops exist per farmer calendar."""
            # this part asserts that the crop calendar is correctly set up
            # particularly that no two crops are planted at the same time
            for farmer_crop_calender in crop_calendar_per_farmer:
                farmer_crop_calender = farmer_crop_calender[
                    farmer_crop_calender[:, -1] != -1
                ]
                if farmer_crop_calender.shape[0] > 1:
                    assert (
                        np.unique(farmer_crop_calender[:, [1, 3]], axis=0).shape[0]
                        == farmer_crop_calender.shape[0]
                    )

        check_crop_calendar(crop_calendar_per_farmer)

        # Define constants for crop IDs
        WHEAT = 0
        MAIZE = 1
        RICE = 2
        BARLEY = 3
        RYE = 4
        MILLET = 5
        SORGHUM = 6
        SOYBEANS = 7
        SUNFLOWER = 8
        POTATOES = 9
        CASSAVA = 10
        SUGAR_CANE = 11
        SUGAR_BEETS = 12
        OIL_PALM = 13
        RAPESEED = 14
        GROUNDNUTS = 15
        # PULSES = 16
        # CITRUS = 17
        # # DATE_PALM = 18
        # # GRAPES = 19
        # COTTON = 20
        COCOA = 21
        COFFEE = 22
        OTHERS_PERENNIAL = 23
        FODDER_GRASSES = 24
        OTHERS_ANNUAL = 25
        WHEAT_DROUGHT = 26
        WHEAT_FLOOD = 27
        MAIZE_DROUGHT = 28
        MAIZE_FLOOD = 29
        RICE_DROUGHT = 30
        RICE_FLOOD = 31
        SOYBEANS_DROUGHT = 32
        SOYBEANS_FLOOD = 33
        POTATOES_DROUGHT = 34
        POTATOES_FLOOD = 35

        # Manual replacement of certain crops
        def replace_crop(
            crop_calendar_per_farmer: np.ndarray,
            crop_values: np.ndarray | list[int],
            replaced_crop_values: np.ndarray | list[int],
        ) -> np.ndarray:
            # Find the most common crop value among the given crop_values
            crop_instances = crop_calendar_per_farmer[:, :, 0][
                np.isin(crop_calendar_per_farmer[:, :, 0], crop_values)
            ]

            # if none of the crops are present, no need to replace anything
            if crop_instances.size == 0:
                return crop_calendar_per_farmer

            crops, crop_counts = np.unique(crop_instances, return_counts=True)
            most_common_crop = crops[np.argmax(crop_counts)]

            # Determine if there are multiple cropping versions of this crop and assign it to the most common
            new_crop_types = crop_calendar_per_farmer[
                (crop_calendar_per_farmer[:, :, 0] == most_common_crop).any(axis=1),
                :,
                :,
            ]
            unique_rows, counts = np.unique(new_crop_types, axis=0, return_counts=True)
            max_index = np.argmax(counts)
            crop_replacement = unique_rows[max_index]

            crop_replacement_only_crops = crop_replacement[
                crop_replacement[:, -1] != -1
            ]
            if crop_replacement_only_crops.shape[0] > 1:
                assert (
                    np.unique(crop_replacement_only_crops[:, [1, 3]], axis=0).shape[0]
                    == crop_replacement_only_crops.shape[0]
                )

            for replaced_crop in replaced_crop_values:
                # Check where to be replaced crop is
                crop_mask = (crop_calendar_per_farmer[:, :, 0] == replaced_crop).any(
                    axis=1
                )
                # Replace the crop
                crop_calendar_per_farmer[crop_mask] = crop_replacement

            return crop_calendar_per_farmer

        def unify_crop_variants(
            crop_calendar_per_farmer: np.ndarray,
            target_crop: int,
        ) -> np.ndarray:
            """Replace all full rotation blocks for one crop by the most common block.

            Assumes crop_calendar_per_farmer has shape:
                (n_farmers, n_rotation_slots, 4)

            and that the dominant crop of a block is stored in [0, 0].

            Returns:
                The updated crop calendar with only one crop rotation type per crop.
            """
            # Select full farmer blocks belonging to this dominant crop
            block_mask = crop_calendar_per_farmer[:, 0, 0] == target_crop

            if not np.any(block_mask):
                return crop_calendar_per_farmer

            # Full 3D blocks, not individual rows
            crop_blocks = crop_calendar_per_farmer[block_mask]

            # Count unique full-block variants
            unique_variants, variant_counts = np.unique(
                crop_blocks,
                axis=0,
                return_counts=True,
            )

            # Pick most frequent full rotation block
            most_common_variant = unique_variants[np.argmax(variant_counts)]

            # Replace all matching blocks with that dominant full block
            crop_calendar_per_farmer[block_mask] = most_common_variant

            return crop_calendar_per_farmer

        def insert_other_variant_crop(
            crop_calendar_per_farmer: np.ndarray,
            base_crops: int | list[int],
            resistant_crops: tuple[int, int] | list[int] | np.ndarray,
        ) -> np.ndarray:
            # find crop rotation mask
            base_crop_rotation_mask = (
                crop_calendar_per_farmer[:, :, 0] == base_crops
            ).any(axis=1)

            # Find the indices of the crops to be replaced
            indices = np.where(base_crop_rotation_mask)[0]

            # Shuffle the indices to randomize the selection
            np.random.shuffle(indices)

            # Determine the number of crops for each category (stay same, first resistant, last resistant)
            n = len(indices)
            n_same = n // 3
            n_first_resistant = (n // 3) + (
                n % 3 > 0
            )  # Ensuring we account for rounding issues

            # Assign the new values
            crop_calendar_per_farmer[indices[:n_same], 0, 0] = base_crops
            crop_calendar_per_farmer[
                indices[n_same : n_same + n_first_resistant], 0, 0
            ] = resistant_crops[0]
            crop_calendar_per_farmer[indices[n_same + n_first_resistant :], 0, 0] = (
                resistant_crops[1]
            )

            return crop_calendar_per_farmer

        check_crop_calendar(crop_calendar_per_farmer)

        # Reduces certain crops of the same GCAM category to the one that is most common in that region
        # First line checks which crop is most common, second denotes which crops will be replaced by the most common one
        if reduce_crops:
            # Conversion based on the classification in table S1 by Yoon, J., Voisin, N., Klassert, C., Thurber, T., & Xu, W. (2024).
            # Representing farmer irrigated crop area adaptation in a large-scale hydrological model. Hydrology and Earth
            # System Sciences, 28(4), 899–916. https://doi.org/10.5194/hess-28-899-2024

            # Replace fodder with the most common grain crop
            most_common_check = [BARLEY, RYE, MILLET, SORGHUM]
            replaced_value = [FODDER_GRASSES]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change the grain crops to one
            most_common_check = [BARLEY, RYE, MILLET, SORGHUM]
            replaced_value = [BARLEY, RYE, MILLET, SORGHUM]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change other annual / misc to one
            most_common_check = [GROUNDNUTS, COCOA, COFFEE, OTHERS_ANNUAL]
            replaced_value = [GROUNDNUTS, COCOA, COFFEE, OTHERS_ANNUAL]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change oils to one
            most_common_check = [SOYBEANS, SUNFLOWER, RAPESEED]
            replaced_value = [SOYBEANS, SUNFLOWER, RAPESEED]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change tubers to one
            most_common_check = [POTATOES, CASSAVA]
            replaced_value = [POTATOES, CASSAVA]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Reduce sugar crops to one
            most_common_check = [SUGAR_CANE, SUGAR_BEETS]
            replaced_value = [SUGAR_CANE, SUGAR_BEETS]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change perennial to annual, otherwise counted double in esa dataset
            most_common_check = [OIL_PALM, OTHERS_PERENNIAL]
            replaced_value = [OIL_PALM, OTHERS_PERENNIAL]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Replace others_annual by potatoes as it has the most similar hydrological parameters
            # Is replaced because others annual is difficult to parametrize (in terms of costs)
            most_common_check = [POTATOES]
            replaced_value = [OTHERS_ANNUAL]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            unique_rows = np.unique(crop_calendar_per_farmer, axis=0)
            values = unique_rows[:, 0, 0]
            unique_values, counts = np.unique(values, return_counts=True)

            # this part asserts that the crop calendar is correctly set up
            # particularly that no two crops are planted at the same time
            for farmer_crop_calender in crop_calendar_per_farmer:
                farmer_crop_calender = farmer_crop_calender[
                    farmer_crop_calender[:, -1] != -1
                ]
                if farmer_crop_calender.shape[0] > 1:
                    assert (
                        np.unique(farmer_crop_calender[:, [1, 3]], axis=0).shape[0]
                        == farmer_crop_calender.shape[0]
                    )

            if unify_variants:
                duplicates = unique_values[counts > 1]
                if len(duplicates) > 0:
                    for duplicate in duplicates:
                        crop_calendar_per_farmer = unify_crop_variants(
                            crop_calendar_per_farmer, duplicate
                        )

        check_crop_calendar(crop_calendar_per_farmer)

        if replace_base:
            base_crops = [WHEAT]
            resistant_crops = [WHEAT_DROUGHT, WHEAT_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [MAIZE]
            resistant_crops = [MAIZE_DROUGHT, MAIZE_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [RICE]
            resistant_crops = [RICE_DROUGHT, RICE_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [SOYBEANS]
            resistant_crops = [SOYBEANS_DROUGHT, SOYBEANS_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [POTATOES]
            resistant_crops = [POTATOES_DROUGHT, POTATOES_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

        assert crop_calendar_per_farmer[:, :, 3].max() == 0

        check_crop_calendar(crop_calendar_per_farmer)

        self.set_array(crop_calendar_per_farmer, name="agents/farmers/crop_calendar")
        self.set_array(
            np.full_like(is_irrigated, 1, dtype=np.int32),
            name="agents/farmers/crop_calendar_rotation_years",
        )

    def assign_crops(
        self,
        crop_calendar: dict,
        farmer_locations: np.ndarray,
        farmer_mirca_units: np.ndarray,
        year: int,
        MIRCA_unit_grid: xr.DataArray,
        minimum_area_ratio: float,
        replace_crop_calendar_unit_code: dict = {},
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign crops and irrigation status to farmers for a given year.

        Args:
            crop_calendar: Mapping from MIRCA unit to list of rotations (fraction, matrix).
            farmer_locations: Array of farmer pixel coordinates (x, y) order.
            farmer_mirca_units: Array mapping farmer index to MIRCA unit id.
            year: Year to select fractions from raster inputs.
            MIRCA_unit_grid: Grid of MIRCA unit ids aligned with fraction rasters.
            minimum_area_ratio: Minimum fraction for a crop to be considered when sampling.
            replace_crop_calendar_unit_code: Optional remapping for MIRCA unit ids.

        Returns:
            A tuple of (farmer_crops, farmer_irrigated) arrays.
        """
        rainfed_fraction, irrigated_fraction = self.get_crop_area_fractions(year)

        n_farmers: int = farmer_mirca_units.size

        # Prepare empty arrays for all farmers, to be filled in the loop below. Initialized with -1 for crops and False for irrigation.
        farmer_crops = np.full(n_farmers, -1, dtype=np.int8)
        farmer_irrigated = np.full(n_farmers, False, dtype=bool)

        # We use a linear index to make the loops below more efficient. This also allows us to process
        # the farmers in the same cell together, which is more efficient than processing them one by one.
        linear_indices = get_linear_indices(rainfed_fraction)

        # Reshape arrays to 2D (n_crops, n_cells) for easier indexing in the loop below
        rainfed_fraction = rainfed_fraction.values.reshape(
            rainfed_fraction.shape[0], -1
        )
        irrigated_fraction = irrigated_fraction.values.reshape(
            irrigated_fraction.shape[0], -1
        )
        MIRCA_unit_grid: ArrayInt32 = MIRCA_unit_grid.values.ravel()

        # Here, we extract the linear index for each farmer based on their location,
        # which allows us to efficiently loop through each cell and assign crops to all farmers in that cell at once.
        farmer_linear_indices: ArrayInt64 = sample_from_map(
            linear_indices.values,
            farmer_locations,
            linear_indices.rio.transform(recalc=True).to_gdal(),
        )

        for linear_index in linear_indices.values.ravel():
            farmers_cell_mask: ArrayBool = farmer_linear_indices == linear_index
            n_farmer_in_cell: int = farmers_cell_mask.sum()
            if n_farmer_in_cell == 0:
                continue  # No farmers in this cell, skip

            # Set type to np.float64 so that we can safely do normalization and don't run into
            # issues with the fractions not summing to 1 due to rounding errors when they are in lower precision types.
            farmer_crop_rainfed_fractions = rainfed_fraction[:, linear_index].astype(
                np.float64
            )
            farmer_crop_irrigated_fractions = irrigated_fraction[
                :, linear_index
            ].astype(np.float64)

            n_irrigating_farmers = round(
                n_farmer_in_cell * farmer_crop_irrigated_fractions.sum()
            )
            n_rainfed_farmers = n_farmer_in_cell - n_irrigating_farmers
            assert n_irrigating_farmers >= 0 and n_rainfed_farmers >= 0

            MIRCA_unit_cell = MIRCA_unit_grid[linear_index]

            # If given, map the MIRCA unit code to a different one (e.g. to fill in missing crop calendars with those from similar units)
            MIRCA_unit_cell = replace_crop_calendar_unit_code.get(
                MIRCA_unit_cell, MIRCA_unit_cell
            )

            assert len(crop_calendar[MIRCA_unit_cell]) > 0, (
                f"Error: No crop calendar found for cell {linear_index} with MIRCA unit {MIRCA_unit_cell}."
            )

            available_crops: ArrayInt64 = np.concat(
                [crop for _, crop in crop_calendar[MIRCA_unit_cell]]
            )[:, 0, ...]
            is_irrigated: ArrayInt64 = np.concat(
                [crop for _, crop in crop_calendar[MIRCA_unit_cell]]
            )[:, 1, ...]
            crop_mask = available_crops != -1
            available_crops = available_crops[crop_mask]
            is_irrigated = is_irrigated[crop_mask].astype(
                bool
            )  # 1 is irrigated, 0 is rainfed

            available_crops_irrigated = np.unique(available_crops[is_irrigated])
            available_crops_rainfed = np.unique(available_crops[~is_irrigated])

            # Remove crops that are not available for either rainfed or irrigated
            available_crops_mask_rainfed = np.zeros_like(
                farmer_crop_rainfed_fractions, dtype=bool
            )
            available_crops_mask_rainfed[available_crops_rainfed] = True
            farmer_crop_rainfed_fractions[~available_crops_mask_rainfed] = 0

            available_crops_mask_irrigated = np.zeros_like(
                farmer_crop_irrigated_fractions, dtype=bool
            )
            available_crops_mask_irrigated[available_crops_irrigated] = True
            farmer_crop_irrigated_fractions[~available_crops_mask_irrigated] = 0

            if n_rainfed_farmers > 0:
                # Normalize the area fractions
                farmer_crop_rainfed_fractions = (
                    farmer_crop_rainfed_fractions / farmer_crop_rainfed_fractions.sum()
                )
                # Discard crops with area smaller than minimum_area_ratio
                farmer_crop_rainfed_fractions[
                    farmer_crop_rainfed_fractions < minimum_area_ratio
                ] = 0

                assert not farmer_crop_rainfed_fractions.sum() == 0, (
                    f"Error: All rainfed crop fractions are zero for cell {linear_index} with MIRCA unit {MIRCA_unit_cell}."
                )
                # Normalize the area fractions again after removing small crops
                farmer_crop_rainfed_fractions = (
                    farmer_crop_rainfed_fractions / farmer_crop_rainfed_fractions.sum()
                )

                # Choose crops for rainfed and irrigated farmers based on the fractions, then combine and shuffle them to assign to farmers in the cell
                # TODO: Consider farm area here
                rainfed_crop_choices: ArrayUint8 = np.random.choice(
                    farmer_crop_rainfed_fractions.size,
                    size=n_rainfed_farmers,
                    replace=True,
                    p=farmer_crop_rainfed_fractions,
                ).astype(np.uint8)
            else:
                rainfed_crop_choices: ArrayUint8 = np.array([], dtype=np.uint8)

            if n_irrigating_farmers > 0:
                # Normalize the area fractions
                farmer_crop_irrigated_fractions = (
                    farmer_crop_irrigated_fractions
                    / farmer_crop_irrigated_fractions.sum()
                )

                # Discard crops with area smaller than minimum_area_ratio
                farmer_crop_irrigated_fractions[
                    farmer_crop_irrigated_fractions < minimum_area_ratio
                ] = 0

                assert not farmer_crop_irrigated_fractions.sum() == 0, (
                    f"Error: All irrigated crop fractions are zero for cell {linear_index} with MIRCA unit {MIRCA_unit_cell}."
                )

                farmer_crop_irrigated_fractions = (
                    farmer_crop_irrigated_fractions
                    / farmer_crop_irrigated_fractions.sum()
                )

                irrigated_crop_choices: ArrayUint8 = np.random.choice(
                    farmer_crop_irrigated_fractions.size,
                    size=n_irrigating_farmers,
                    replace=True,
                    p=farmer_crop_irrigated_fractions,
                ).astype(np.uint8)

            else:
                irrigated_crop_choices: ArrayUint8 = np.array([], dtype=np.uint8)

            crop_choices: ArrayUint8 = np.concatenate(
                [rainfed_crop_choices, irrigated_crop_choices]
            )
            is_irrigated_choices: ArrayBool = np.concatenate(
                [
                    np.zeros(n_rainfed_farmers, dtype=bool),
                    np.ones(n_irrigating_farmers, dtype=bool),
                ]
            )

            # Shuffle the choices to avoid any ordering effects
            # Crops and irrigation are shuffled together to maintain the correct pairing
            shuffle_indices = np.random.permutation(len(crop_choices))
            crop_choices = crop_choices[shuffle_indices]
            is_irrigated_choices = is_irrigated_choices[shuffle_indices]

            # Finally assign to farmers
            farmer_crops[farmers_cell_mask] = crop_choices
            farmer_irrigated[farmers_cell_mask] = is_irrigated_choices

        assert not (farmer_crops == -1).any(), (
            "Error: some farmers have no crops assigned"
        )

        return farmer_crops, farmer_irrigated
