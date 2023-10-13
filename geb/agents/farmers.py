# -*- coding: utf-8 -*-
import os
import math
from datetime import date
from datetime import datetime
import pandas as pd
import cftime
import json
import random
import calendar
from scipy.stats import genextreme
from scipy.stats import linregress
from scipy.stats import norm
from sklearn.feature_selection import mutual_info_regression

from pathlib import Path

import numpy as np
from numba import njit
from pyproj import Transformer
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass

from scipy import interpolate
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

from honeybees.library.mapIO import NetCDFReader
from honeybees.library.mapIO import ArrayReader
from honeybees.library.mapIO import MapReader
from honeybees.agents import AgentBaseClass
from honeybees.library.raster import pixels_to_coords, sample_from_map
from honeybees.library.neighbors import find_neighbors
import xarray as xr

from ..data import load_regional_crop_data_from_dict, load_crop_variables, load_crop_ids, load_economic_data
from .decision_module import DecisionModule
from .general import AgentArray

@njit(cache=True)
def get_farmer_HRUs(field_indices: np.ndarray, field_indices_by_farmer: np.ndarray, farmer_index: int) -> np.ndarray:
    """Gets indices of field for given farmer.
    
    Args:
        field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
        field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  

    Returns:
        field_indices_for_farmer: the indices of the fields for the given farmer.
    """
    return field_indices[field_indices_by_farmer[farmer_index, 0]: field_indices_by_farmer[farmer_index, 1]]

class Farmers(AgentBaseClass):
    """The agent class for the farmers. Contains all data and behaviourial methods. The __init__ function only gets the model as arguments, the agent parent class and the redundancy. All other variables are loaded at later stages.
    
    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
        redundancy: a lot of data is saved in pre-allocated NumPy arrays. While this allows much faster operation, it does mean that the number of agents cannot grow beyond the size of the pre-allocated arrays. This parameter allows you to specify how much redundancy should be used. A lower redundancy means less memory is used, but the model crashes if the redundancy is insufficient.
    """
    __slots__ = [
        "model",
        "agents",
        "var",
        "redundancy",
        "crop_stage_lengths",
        "crop_factors",
        "crop_yield_factors",
        "reference_yield",
        "growth_length",
        "elevation_subgrid",
        "plant_day",
        "field_indices",
        "_field_indices_by_farmer",
        "n",
        "max_n",
        "activation_order_by_elevation_fixed",
        "agent_attributes_meta",
        "sample",
        "subdistrict_map",
        "crop_names",
        "cultivation_costs",
        "crop_prices",
        "inflation", 
        ## risk perception & SEUT 
        "previous_month",
        "moving_average_loss",
        "absolute_threshold_loss",
        "SPEI_map",
        "total_spinup_time",
        "p_droughts",
    ]
    agent_attributes = [
        "_locations",
        "_region_id",
        "_elevation",
        "_crops",
        "_irrigation_source",
        "_household_size",
        "_disposable_income",
        "_daily_non_farm_income",
        "_daily_expenses_per_capita",
        "_irrigation_efficiency",
        "_n_water_accessible_days",
        "_n_water_accessible_years",
        "_channel_abstraction_m3_by_farmer",
        "_groundwater_abstraction_m3_by_farmer",
        "_reservoir_abstraction_m3_by_farmer",
        "_yearly_abstraction_m3_by_farmer",
        "_latest_profits",
        "_yearly_profits",
        "_yearly_yield_ratio",
        "_total_crop_age",
        "_per_harvest_yield_ratio",
        "_per_harvest_SPEI",
        "_yearly_SPEI_probability",
        "_monthly_SPEI",
        "_latest_potential_profits",
        "_farmer_yield_probability_relation",
        "_groundwater_depth",
        "_profit",
        "_farmer_is_in_command_area",
        "_farmer_class",
        "_water_use",
        "_flooded",
        ## expected utility 
        "_adapted",
        "_time_adapted",
        "_wealth",
        "_risk_perception",
        "_drought_timer",
        "_drought_loss",
        "_risk_perc_min",
        "_risk_perc_max",
        "_risk_decr",
        "_decision_horizon",
        "_annual_costs_all_adaptations",
        "_GEV_parameters",
        "_yield_ratios_drought_event",
    ]
    __slots__.extend(agent_attributes)
    agent_attributes_new = [
        "risk_aversion"
    ]
    __slots__.extend(agent_attributes_new)

    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents
        self.sample = [2000, 5500, 10000]
        self.var = model.data.HRU
        self.redundancy = reduncancy

        self.crop_ids = load_crop_ids(self.model.model_structure)
        # reverse dictionary
        self.crop_names = {crop_name: crop_id for crop_id, crop_name in self.crop_ids.items()}
        self.crop_variables = load_crop_variables(self.model.model_structure)
        
        ## Set parameters required for drought event perception, risk perception and SEUT 
        self.previous_month = 0
        self.moving_average_threshold = self.model.config['agent_settings']['expected_utility']['drought_risk_calculations']['event_perception']['moving_average_threshold']
        self.absolute_threshold = self.model.config['agent_settings']['expected_utility']['drought_risk_calculations']['event_perception']['absolute_threshold']
       
        # Assign risk aversion sigma, time discounting preferences, expendature cap 
        self.r_time = self.model.config['agent_settings']['expected_utility']['decisions']['time_discounting']
        self.expenditure_cap = self.model.config['agent_settings']['expected_utility']['decisions']['expenditure_cap']

        self.inflation_rate = load_economic_data(self.model.model_structure['dict']['economics/inflation_rates'])
        self.lending_rate = load_economic_data(self.model.model_structure['dict']['economics/lending_rates'])
        self.well_price = load_economic_data(self.model.model_structure['dict']['economics/well_prices'])
        self.well_upkeep_price_per_m2 = load_economic_data(self.model.model_structure['dict']['economics/upkeep_prices_well_per_m2'])
        self.drip_irrigation_price = load_economic_data(self.model.model_structure['dict']['economics/drip_irrigation_prices'])
        self.drip_irrigation_upkeep_per_m2 = load_economic_data(self.model.model_structure['dict']['economics/upkeep_prices_drip_irrigation_per_m2'])
        self.well_investment_time_years = 10
        self.p_droughts = np.array([1000, 500, 250, 100, 50, 25, 10, 5, 2, 1])

        self.elevation_subgrid = MapReader(
            fp=self.model.model_structure['MERIT_grid']["landsurface/topo/subgrid_elevation"],
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax,
        )
        self.elevation_grid = self.model.data.grid.compress(MapReader(
            fp=self.model.model_structure['grid']["landsurface/topo/elevation"],
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax,
        ).get_data_array())
        
        self.SPEI_map = NetCDFReader(
            fp=self.model.model_structure['forcing']["climate/spei"],
            varname= 'spei',
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax,
            latname='y',
            lonname='x',
            timename= 'time'
        )

        with open(self.model.model_structure['dict']["agents/farmers/irrigation_sources"], 'r') as f:
            self.irrigation_source_key = json.load(f)

        # load map of all subdistricts
        self.subdistrict_map = MapReader(
            fp=self.model.model_structure['region_subgrid']["areamaps/region_subgrid"],
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax,
        )
   
        self.crop_prices = load_regional_crop_data_from_dict(self.model.model_structure, "crops/crop_prices")
        self.cultivation_costs = load_regional_crop_data_from_dict(self.model.model_structure, "crops/cultivation_costs")
        self.total_spinup_time = self.model.config['general']['start_time'].year - self.model.config['general']['spinup_time'].year

        self.agent_attributes_meta = {
            "_locations": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan]
            },
            "_region_id": {
                "dtype": np.int32,
                "nodata": -1
            },
            "_elevation": {
                "dtype": np.float32,
                "nodata": np.nan
            },
            "_crops": {
                "dtype": np.int32,
                "nodata": [-1, -1, -1],
                "nodatacheck": False
            },
            "_groundwater_depth": {
                "dtype": np.float32,
                "nodata": np.nan,
            },
            "_household_size": {
                "dtype": np.int32,
                "nodata": -1,
            },
            "_disposable_income": {
                "dtype": np.float32,
                "nodata": np.nan
            },
            "_daily_non_farm_income": {
                "dtype": np.float32,
                "nodata": np.nan
            },
            "_daily_expenses_per_capita": {
                "dtype": np.float32,
                "nodata": np.nan
            },
            "_irrigation_efficiency": {
                "dtype": np.float32,
                "nodata": np.nan
            },
            "_n_water_accessible_days": {
                "dtype": np.int32,
                "nodata": -1
            },
            "_n_water_accessible_years": {
                "dtype": np.int32,
                "nodata": -1
            },
            "_irrigation_source": {
                "dtype": np.int32,
                "nodata": -1,
            },
            "_water_availability_by_farmer": {
                "dtype": np.float32,
                "nodata": np.nan
            },
            "_channel_abstraction_m3_by_farmer": {
                "dtype": np.float32,
                "nodata": np.nan
            },
            "_groundwater_abstraction_m3_by_farmer": {
                "dtype": np.float32,
                "nodata": np.nan
            },
            "_reservoir_abstraction_m3_by_farmer": {
                "dtype": np.float32,
                "nodata": np.nan
            },
            "_yearly_abstraction_m3_by_farmer": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan, np.nan]
            },
            "_profit": {
                "dtype": np.float32,
                "nodata": np.nan,
            },
            "_latest_profits": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            },
            "_yearly_profits": {
                "dtype": np.float32,
                "nodata": [np.nan] * (self.total_spinup_time + 1),
            },
            "_yearly_yield_ratio": {
                "dtype": np.float32,
                "nodata": [np.nan] * (self.total_spinup_time + 1),
            },
            "_total_crop_age": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan],
            },
            "_per_harvest_yield_ratio": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan],
            },
            "_per_harvest_SPEI": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan],
            },
            "_yearly_SPEI_probability": {
                "dtype": np.float32,
                "nodata": [np.nan] * (self.total_spinup_time + 1),
            },
            "_monthly_SPEI": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            },
            "_latest_potential_profits": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            },
            "_farmer_yield_probability_relation": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan],
            },
            "_farmer_class": {
                "dtype": np.int32,
                "nodata": -1,
            },
            "_water_use": {
                "dtype": np.int32,
                "nodata": [np.nan, np.nan, np.nan, np.nan],
            },
            "_farmer_is_in_command_area": {
                "dtype": bool,
                "nodata": False,
                "nodatacheck": False
            },
            "_flooded": {
                "dtype": bool,
                "nodata": False,
                "nodatacheck": False
            },
            "_adapted": {
                "dtype": np.int32,
                "nodata": [np.nan, np.nan],
            },
            "_time_adapted": {
                "dtype": np.int32,
                "nodata": [np.nan, np.nan],
            },
            "_wealth": {
                "dtype": np.float32,
                "nodata": np.nan,
            },
            "_decision_horizon": {
                "dtype": np.float32,
                "nodata": [np.nan],
            },
            "_risk_perception": {
                "dtype": np.float32,
                "nodata": -1,
            },
            "_drought_timer": {
                "dtype": np.float32,
                "nodata": False,
                "nodatacheck": False
            },
            "_drought_loss": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            },
            "_risk_perc_min": {
                "dtype": np.float32,
                "nodata": -1,
            },
            "_risk_perc_max": {
                "dtype": np.float32,
                "nodata": False,
            },
            "_risk_decr": {
                "dtype": np.float32,
                "nodata": False,
            },
            "_annual_costs_all_adaptations": {
                "dtype": np.float32,
                "nodata": np.nan,
            },
            "_GEV_parameters": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan],
            },
            "_yield_ratios_drought_event": {
                "dtype": np.float32,
                "nodata": [-1, -1, -1, -1, -1,-1, -1, -1, -1, -1],
            },
        }
        self.initiate_agents()

    @property
    def locations(self):
        return self._locations[:self.n]

    @locations.setter
    def locations(self, value):
        self._locations[:self.n] = value

    @property
    def region_id(self):
        return self._region_id[:self.n]

    @region_id.setter
    def region_id(self, value):
        self._region_id[:self.n] = value

    @property
    def crops(self):
        return self._crops[:self.n]

    @crops.setter
    def crops(self, value):
        self._crops[:self.n] = value

    @property
    def latest_profits(self):
        return self._latest_profits[:self.n]

    @latest_profits.setter
    def latest_profits(self, value):
        self._latest_profits[:self.n] = value

    @property
    def latest_potential_profits(self):
        return self._latest_potential_profits[:self.n]

    @latest_potential_profits.setter
    def latest_potential_profits(self, value):
        self._latest_potential_profits[:self.n] = value

    @property
    def irrigation_efficiency(self):
        return self._irrigation_efficiency[:self.n]

    @irrigation_efficiency.setter
    def irrigation_efficiency(self, value):
        self._irrigation_efficiency[:self.n] = value

    @property
    def elevation(self):
        return self._elevation[:self.n]

    @elevation.setter
    def elevation(self, value):
        self._elevation[:self.n] = value

    @property
    def reservoir_abstraction_m3_by_farmer(self):
        return self._reservoir_abstraction_m3_by_farmer[:self.n]

    @reservoir_abstraction_m3_by_farmer.setter
    def reservoir_abstraction_m3_by_farmer(self, value):
        self._reservoir_abstraction_m3_by_farmer[:self.n] = value

    @property
    def groundwater_abstraction_m3_by_farmer(self):
        return self._groundwater_abstraction_m3_by_farmer[:self.n]

    @groundwater_abstraction_m3_by_farmer.setter
    def groundwater_abstraction_m3_by_farmer(self, value):
        self._groundwater_abstraction_m3_by_farmer[:self.n] = value

    @property
    def channel_abstraction_m3_by_farmer(self):
        return self._channel_abstraction_m3_by_farmer[:self.n]

    @channel_abstraction_m3_by_farmer.setter
    def channel_abstraction_m3_by_farmer(self, value):
        self._channel_abstraction_m3_by_farmer[:self.n] = value

    @property
    def yearly_abstraction_m3_by_farmer(self):
        return self._yearly_abstraction_m3_by_farmer[:self.n]

    @yearly_abstraction_m3_by_farmer.setter
    def yearly_abstraction_m3_by_farmer(self, value):
        self._yearly_abstraction_m3_by_farmer[:self.n] = value

    @property
    def n_water_accessible_days(self):
        return self._n_water_accessible_days[:self.n]

    @n_water_accessible_days.setter
    def n_water_accessible_days(self, value):
        self._n_water_accessible_days[:self.n] = value

    @property
    def n_water_accessible_years(self):
        return self._n_water_accessible_years[:self.n]

    @n_water_accessible_years.setter
    def n_water_accessible_years(self, value):
        self._n_water_accessible_years[:self.n] = value

    @property
    def disposable_income(self):
        return self._disposable_income[:self.n]

    @disposable_income.setter
    def disposable_income(self, value):
        self._disposable_income[:self.n] = value

    @property
    def household_size(self):
        return self._household_size[:self.n]

    @household_size.setter
    def household_size(self, value):
        self._household_size[:self.n] = value

    @property
    def groundwater_depth(self):
        return self._groundwater_depth[:self.n]

    @groundwater_depth.setter
    def groundwater_depth(self, value):
        self._groundwater_depth[:self.n] = value

    @property
    def daily_expenses_per_capita(self):
        return self._daily_expenses_per_capita[:self.n]

    @daily_expenses_per_capita.setter
    def daily_expenses_per_capita(self, value):
        self._daily_expenses_per_capita[:self.n] = value

    @property
    def daily_non_farm_income(self):
        return self._daily_non_farm_income[:self.n]

    @daily_non_farm_income.setter
    def daily_non_farm_income(self, value):
        self._daily_non_farm_income[:self.n] = value

    @property
    def irrigation_source(self):
        return self._irrigation_source[:self.n]

    @irrigation_source.setter
    def irrigation_source(self, value):
        self._irrigation_source[:self.n] = value

    @property
    def profit(self):
        return self._profit[:self.n]

    @profit.setter
    def profit(self, value):
        self._profit[:self.n] = value

    @property
    def yearly_profits(self):
        return self._yearly_profits[:self.n]

    @yearly_profits.setter
    def yearly_profits(self, value):
        self._yearly_profits[:self.n] = value

    @property
    def yearly_yield_ratio(self):
        return self._yearly_yield_ratio[:self.n]

    @yearly_yield_ratio.setter
    def yearly_yield_ratio(self, value):
        self._yearly_yield_ratio[:self.n] = value

    @property
    def total_crop_age(self):
        return self._total_crop_age[:self.n]

    @total_crop_age.setter
    def total_crop_age(self, value):
        self._total_crop_age[:self.n] = value

    @property
    def per_harvest_yield_ratio(self):
        return self._per_harvest_yield_ratio[:self.n]

    @per_harvest_yield_ratio.setter
    def per_harvest_yield_ratio(self, value):
        self._per_harvest_yield_ratio[:self.n] = value

    @property
    def per_harvest_SPEI(self):
        return self._per_harvest_SPEI[:self.n]

    @per_harvest_SPEI.setter
    def per_harvest_SPEI(self, value):
        self._per_harvest_SPEI[:self.n] = value
    
    @property
    def yearly_SPEI_probability(self):
        return self._yearly_SPEI_probability[:self.n]

    @yearly_SPEI_probability.setter
    def yearly_SPEI_probability(self, value):
        self._yearly_SPEI_probability[:self.n] = value

    @property
    def monthly_SPEI(self):
        return self._monthly_SPEI[:self.n]

    @monthly_SPEI.setter
    def monthly_SPEI(self, value):
        self._monthly_SPEI[:self.n] = value

    @property
    def farmer_is_in_command_area(self):
        return self._farmer_is_in_command_area[:self.n]

    @farmer_is_in_command_area.setter
    def farmer_is_in_command_area(self, value):
        self._farmer_is_in_command_area[:self.n] = value

    @property
    def farmer_yield_probability_relation(self):
        return self._farmer_yield_probability_relation[:self.n]

    @farmer_yield_probability_relation.setter
    def farmer_yield_probability_relation(self, value):
        self._farmer_yield_probability_relation[:self.n] = value

    @property
    def farmer_class(self):
        return self._farmer_class[:self.n]

    @farmer_class.setter
    def farmer_class(self, value):
        self._farmer_class[:self.n] = value

    @property
    def water_use(self):
        return self._water_use[:self.n]

    @water_use.setter
    def water_use(self, value):
        self._water_use[:self.n] = value

    @property
    def flooded(self):
        return self._flooded[:self.n]

    @flooded.setter
    def flooded(self, value):
        self._flooded[:self.n] = value

    @property
    def field_indices_by_farmer(self):
        return self._field_indices_by_farmer[:self.n]

    @field_indices_by_farmer.setter
    def field_indices_by_farmer(self, value):
        self._field_indices_by_farmer[:self.n] = value

    @property
    def adapted(self):
        return self._adapted[:self.n]

    @adapted.setter
    def adapted(self, value):
        self._adapted[:self.n] = value
    
    @property
    def time_adapted(self):
        return self._time_adapted[:self.n]

    @time_adapted.setter
    def time_adapted(self, value):
        self._time_adapted[:self.n] = value

    @property
    def wealth(self):
        return self._wealth[:self.n]

    @wealth.setter
    def wealth(self, value):
        self._wealth[:self.n] = value

    @property
    def drought_loss(self):
        return self._drought_loss[:self.n]

    @drought_loss.setter
    def drought_loss(self, value):
        self._drought_loss[:self.n] = value

    @property
    def drought_timer(self):
        return self._drought_timer[:self.n]

    @drought_timer.setter
    def drought_timer(self, value):
        self._drought_timer[:self.n] = value

    @property
    def risk_perception(self):
        return self._risk_perception[:self.n]

    @risk_perception.setter
    def risk_perception(self, value):
        self._risk_perception[:self.n] = value

    @property
    def annual_costs_all_adaptations(self):
        return self._annual_costs_all_adaptations[:self.n]

    @annual_costs_all_adaptations.setter
    def annual_costs_all_adaptations(self, value):
        self._annual_costs_all_adaptations[:self.n] = value

    @property
    def GEV_parameters(self):
        return self._GEV_parameters[:self.n]

    @GEV_parameters.setter
    def GEV_parameters(self, value):
        self._GEV_parameters[:self.n] = value

    @property
    def yield_ratios_drought_event(self):
        return self._yield_ratios_drought_event[:self.n]

    @yield_ratios_drought_event.setter
    def yield_ratios_drought_event(self, value):
        self._yield_ratios_drought_event[:self.n] = value
    
    @staticmethod
    def is_in_command_area(n, command_areas, field_indices, field_indices_by_farmer):
        farmer_is_in_command_area = np.zeros(n, dtype=bool)
        for farmer_i in range(n):
            farmer_fields = get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer_i)
            for field in farmer_fields:
                command_area = command_areas[field]
                if command_area != -1:
                    farmer_is_in_command_area[farmer_i] = True
                    break
        return farmer_is_in_command_area

    def get_max_n(self, n):
        max_n = math.ceil(n * (1 + self.redundancy))
        assert max_n < 4294967295 # max value of uint32, consider replacing with uint64
        return max_n

    def initiate_agents(self) -> None:
        """Calls functions to initialize all agent attributes, including their locations. Then, crops are initially planted. 
        """
        # If initial conditions based on spinup period need to be loaded, load them. Otherwise, generate them.
        if self.model.load_initial_data:
            for attribute in self.agent_attributes:
                fp = os.path.join(self.model.initial_conditions_folder, f"farmers.{attribute}.npz")
                values = np.load(fp)['data']
                setattr(self, attribute, values)
            for attribute in self.agent_attributes_new:
                fp = os.path.join(self.model.initial_conditions_folder, f"farmers.{attribute}.npz")
                values = np.load(fp)['data']
                if not hasattr(self, 'max_n'):
                    self.max_n = self.get_max_n(values.shape[0])
                values = AgentArray(values, max_size=self.max_n)
                setattr(self, attribute, values)
            self.n = np.where(np.isnan(self._locations[:,0]))[0][0]  # first value where location is not defined (np.nan)
            self.max_n = self._locations.shape[0]
        else:
            farms = self.model.data.farms

            # Get number of farmers and maximum number of farmers that could be in the entire model run based on the redundancy.
            self.n = np.unique(farms[farms != -1]).size
            self.max_n = self.get_max_n(self.n)

            # The code below obtains the coordinates of the farmers' locations.
            # First the horizontal and vertical indices of the pixels that are not -1 are obtained. Then, for each farmer the
            # average of the horizontal and vertical indices is calculated. This is done by using the bincount function.
            # Finally, the coordinates are obtained by adding .5 to the pixels and converting them to coordinates using pixel_to_coord.
            vertical_index = np.arange(farms.shape[0]).repeat(farms.shape[1]).reshape(farms.shape)[farms != -1]
            horizontal_index = np.tile(np.arange(farms.shape[1]), farms.shape[0]).reshape(farms.shape)[farms != -1]
            pixels = np.zeros((self.n, 2), dtype=np.int32)
            pixels[:,0] = np.round(np.bincount(farms[farms != -1], horizontal_index) / np.bincount(farms[farms != -1])).astype(int)
            pixels[:,1] = np.round(np.bincount(farms[farms != -1], vertical_index) / np.bincount(farms[farms != -1])).astype(int)

            for attribute in self.agent_attributes:
                if isinstance(self.agent_attributes_meta[attribute]["nodata"], list):
                    shape = (self.max_n, len(self.agent_attributes_meta[attribute]["nodata"]))
                else:
                    shape = self.max_n
                setattr(self, attribute, np.full(shape, self.agent_attributes_meta[attribute]["nodata"], dtype=self.agent_attributes_meta[attribute]["dtype"]))

            self.locations = pixels_to_coords(pixels + .5, self.var.gt)

            self.risk_aversion = AgentArray(n=self.n, max_size=self.max_n, dtype=np.float32, fill_value=np.nan)
            self.risk_aversion[:] = np.load(self.model.model_structure['binary']["agents/farmers/risk_aversion"])['data']

            # Load the region_code of each farmer.
            self.region_id = np.load(self.model.model_structure['binary']["agents/farmers/region_id"])['data']

            # Find the elevation of each farmer on the map based on the coordinates of the farmer as calculated before.
            self.elevation = self.elevation_subgrid.sample_coords(self.locations)

            # Initiate adaptation status. 0 = not adapted, 1 adapted. Column 0 = sprinkler, 1 = well
            self.adapted = np.zeros((self.n, 2), dtype=np.int32) 
            # the time each agent has been paying off their dry flood proofing investment loan. Column 0 = sprinkler, 1 = well. -1 if they do not have adaptations
            self.time_adapted = np.full((self.n, 2), -1, dtype = np.int32)


            # Load the crops planted for each farmer in the season #1, season #2 and season #3.
            self.crops[:, 0] = np.load(self.model.model_structure['binary']["agents/farmers/season_#1_crop"])['data']
            self.crops[:, 1] = np.load(self.model.model_structure['binary']["agents/farmers/season_#2_crop"])['data']
            self.crops[:, 2] = np.load(self.model.model_structure['binary']["agents/farmers/season_#3_crop"])['data']
            assert self.crops.max() < len(self.crop_ids)

            # Set irrigation source 
            self.irrigation_source = np.load(self.model.model_structure['binary']["agents/farmers/irrigation_source"])['data']
            # set the adaptation of wells to 1 if farmers have well 
            self.adapted[:,1][np.isin(self.irrigation_source, [self.irrigation_source_key['well'], self.irrigation_source_key['tubewell']])] = 1
            # Set how long the agents have adapted somewhere across the lifespan of farmers, would need to be a bit more realistic likely 
            self.time_adapted[self.adapted[:,1] == 1, 1] = np.random.uniform(1, self.model.config['agent_settings']['expected_utility']['adaptation_well']['lifespan'], np.sum(self.adapted[:,1] == 1))
            
            #self.irrigation_source = np.zeros(15657)

            # Initiate a number of arrays with Nan, zero or -1 values for variables that will be used during the model run.

            self.channel_abstraction_m3_by_farmer[:] = 0
            self.reservoir_abstraction_m3_by_farmer[:] = 0
            self.groundwater_abstraction_m3_by_farmer[:] = 0
            # 2D-array for storing yearly abstraction by farmer. 0: channel abstraction, 1: reservoir abstraction, 2: groundwater abstraction, 3: total abstraction
            self.yearly_abstraction_m3_by_farmer = np.zeros((self.n, 4), dtype=np.float32)
            self.n_water_accessible_days[:] = 0
            self.n_water_accessible_years[:] = 0

            self.profit[:] = 0

            self.yearly_profits = np.zeros((self.n, self.total_spinup_time + 1), dtype=np.float32)
            self.yearly_yield_ratio = np.zeros((self.n, self.total_spinup_time + 1), dtype=np.float32)

            # 0 = kharif age, 1 = rabi age, 2 = summer age, 3 = total growth time 
            self.total_crop_age = np.zeros((self.n, 3), dtype=np.float32)
            # 0 = kharif yield_ratio, 1 = rabi yield_ratio, 2 = summer yield_ratio
            self.per_harvest_yield_ratio = np.zeros((self.n, 3), dtype=np.float32)
            self.per_harvest_SPEI = np.zeros((self.n, 3), dtype=np.float32)
            self.yearly_SPEI_probability = np.zeros((self.n, self.total_spinup_time + 1), dtype=np.float32)
            self.monthly_SPEI = np.zeros((self.n, 10), dtype=np.float32)
            self.disposable_income[:] = 0
            self.household_size = np.load(self.model.model_structure['binary']["agents/farmers/household_size"])['data']
            self.daily_non_farm_income = np.load(self.model.model_structure['binary']["agents/farmers/daily_non_farm_income_family"])['data']
            self.daily_expenses_per_capita = np.load(self.model.model_structure['binary']["agents/farmers/daily_consumption_per_capita"])['data']
            self.flooded[:] = False

            self.farmer_yield_probability_relation = np.zeros((self.n, 2), dtype=np.float32)
            self.yield_ratios_drought_event = np.full((self.n,  self.p_droughts.size), 0, dtype=np.float32)
            
            ## Base initial wealth on x days of daily expenses, sort of placeholder 
            self.wealth = self.daily_expenses_per_capita * self.household_size * ((365/12)*18)
        
            ## Risk perception variables 
            self.risk_perception = np.full(self.n, self.model.config['agent_settings']['expected_utility']['drought_risk_calculations']['risk_perception']['min'], dtype = np.float32)
            self.drought_timer = np.full(self.n, 99, dtype = np.float32)
            self.drought_loss = np.zeros((self.n, 6), dtype=np.float32)

            # Create a random set of irrigating farmers --> chance that it does not line up with farmers that are expected to have this 
            self.irrigation_efficiency = np.random.uniform(0.50, 0.95, self.n)
            # Set the people who already have more van 90% irrigation efficiency to already adapted for the sprinkler irrgation adaptation  
            self.adapted[:,0][self.irrigation_efficiency >= .85] = 1
            self.time_adapted[self.adapted[:,0] == 1, 0] = np.random.uniform(1, self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['lifespan'], np.sum(self.adapted[:,0] == 1))
            
            # Initiate array that tracks the overall yearly costs for all adaptations 
            self.annual_costs_all_adaptations = np.zeros(self.n, dtype=np.float32) 

            # 0 is surface water / channel-dependent, 1 is reservoir-dependent, 2 is groundwater-dependent, 3 is rainwater-dependent
            self.farmer_class[:] = 0  
            self.water_use = np.zeros((self.n, 4), dtype=np.int32)

            self.farmer_is_in_command_area[:] = False

            ## Load in the GEV_parameters, calculated from the extreme value distribution of the SPEI timeseries, and load in the original SPEI data 
            parameter_names = ['c', 'loc', 'scale']
            self.GEV_parameters = np.zeros((len(self.locations), len(parameter_names)))

            for i, varname in enumerate(parameter_names):
                GEV_map = MapReader(
                    fp=self.model.model_structure['grid'][f"climate/gev_{varname}"],
                    xmin=self.model.xmin,
                    ymin=self.model.ymin,
                    xmax=self.model.xmax,
                    ymax=self.model.ymax,
                )
                self.GEV_parameters[:, i] = GEV_map.sample_coords(self.locations)
            
        self.var.actual_transpiration_crop = self.var.load_initial('actual_transpiration_crop', default=self.var.full_compressed(0, dtype=np.float32, gpu=False), gpu=False)
        self.var.potential_transpiration_crop = self.var.load_initial('potential_transpiration_crop', default=self.var.full_compressed(0, dtype=np.float32, gpu=False), gpu=False)
        self.var.crop_map = self.var.load_initial('crop_map', default=np.full_like(self.var.land_owners, -1), gpu=False)
        self.var.crop_age_days_map = self.var.load_initial('crop_age_days_map', default=np.full_like(self.var.land_owners, -1), gpu=False)
        self.var.crop_harvest_age_days = self.var.load_initial('crop_harvest_age_days', default=np.full_like(self.var.land_owners, -1), gpu=False)

        self.risk_perc_min = np.full(self.n, self.model.config['agent_settings']['expected_utility']['drought_risk_calculations']['risk_perception']['min'], dtype = np.float32)
        self.risk_perc_max = np.full(self.n, self.model.config['agent_settings']['expected_utility']['drought_risk_calculations']['risk_perception']['max'], dtype = np.float32)
        self.risk_decr = np.full(self.n, self.model.config['agent_settings']['expected_utility']['drought_risk_calculations']['risk_perception']['coef'], dtype = np.float32)
        self.decision_horizon = np.full(self.n, self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'])

        self._field_indices_by_farmer = np.full((self.max_n, 2), -1, dtype=np.int32)
        self.update_field_indices()

        print(f'Loaded {self.n} farmer agents')

        # check whether none of the attributes have nodata values
        for attr in self.agent_attributes:
            assert getattr(self, attr[1:]).shape[0] == self.n
            if "nodatacheck" not in self.agent_attributes_meta[attr] or self.agent_attributes_meta[attr]['nodatacheck'] is True:
                assert (getattr(self, attr[1:]) != self.agent_attributes_meta[attr]['nodata']).all(axis=-1).all()

    @staticmethod
    @njit(cache=True)
    def update_field_indices_numba(land_owners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Creates `field_indices_by_farmer` and `field_indices`. These indices are used to quickly find the fields for a specific farmer.

        Args:
            land_owners: Array of the land owners. Each unique ID is a different land owner. -1 means the land is not owned by anyone.

        Returns:
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  
        """
        agents = np.unique(land_owners)
        if agents[0] == -1:
            n_agents = agents.size -1
        else:
            n_agents = agents.size
        field_indices_by_farmer = np.full((n_agents, 2), -1, dtype=np.int32)
        field_indices = np.full(land_owners.size, -1, dtype=np.int32)

        land_owners_sort_idx = np.argsort(land_owners)
        land_owners_sorted = land_owners[land_owners_sort_idx]

        last_not_owned = np.searchsorted(land_owners_sorted, -1, side='right')

        prev_land_owner = -1
        for i in range(last_not_owned, land_owners.size):
            land_owner = land_owners[land_owners_sort_idx[i]]
            if land_owner != -1:
                if land_owner != prev_land_owner:
                    field_indices_by_farmer[land_owner, 0] = i - last_not_owned
                field_indices_by_farmer[land_owner, 1] = i + 1 - last_not_owned
                field_indices[i - last_not_owned] = land_owners_sort_idx[i]
                prev_land_owner = land_owner
        field_indices = field_indices[:-last_not_owned]
        return field_indices_by_farmer, field_indices

    def update_field_indices(self) -> None:
        """Creates `field_indices_by_farmer` and `field_indices`. These indices are used to quickly find the fields for a specific farmer."""
        self.field_indices_by_farmer, self.field_indices = self.update_field_indices_numba(self.var.land_owners)
    
    @property
    def activation_order_by_elevation(self):
        """
        Activation order is determined by the agent elevation, starting from the highest.
        Agents with the same elevation are randomly shuffled.
        """
        # if activation order is fixed. Get the random state, and set a fixed seet.
        if self.model.config['agent_settings']['fix_activation_order']:
            if hasattr(self, 'activation_order_by_elevation_fixed') and self.activation_order_by_elevation_fixed[0] == self.n:
                return self.activation_order_by_elevation_fixed[1]
            random_state = np.random.get_state()
            np.random.seed(42)
        elevation = self.elevation
        # Shuffle agent elevation and agent_ids in unision.
        p = np.random.permutation(elevation.size)
        # if activation order is fixed, set random state to previous state
        if self.model.config['agent_settings']['fix_activation_order']:
            np.random.set_state(random_state)
        elevation_shuffled = elevation[p]
        agent_ids_shuffled = np.arange(0, elevation.size, 1, dtype=np.int32)[p]
        # Use argsort to find the order or the shuffled elevation. Using a stable sorting
        # algorithm such that the random shuffling in the previous step is conserved
        # in groups with identical elevation.
        activation_order_shuffled = np.argsort(elevation_shuffled, kind="stable")[::-1]
        argsort_agend_ids = agent_ids_shuffled[activation_order_shuffled]
        # Return the agent ids ranks in the order of activation.
        ranks = np.empty_like(argsort_agend_ids)
        ranks[argsort_agend_ids] = np.arange(argsort_agend_ids.size)
        if self.model.config['agent_settings']['fix_activation_order']:
            self.activation_order_by_elevation_fixed = (self.n, ranks)
        return ranks

    @staticmethod
    @njit(cache=True)
    def abstract_water_numba(
        n: int,
        activation_order: np.ndarray,
        field_indices_by_farmer: np.ndarray,
        field_indices: np.ndarray,
        elevation_grid: np.ndarray,
        irrigation_efficiency: np.ndarray,
        surface_irrigated: np.ndarray,
        well_irrigated: np.ndarray,
        cell_area: np.ndarray,
        HRU_to_grid: np.ndarray,
        crop_map: np.ndarray,
        totalPotIrrConsumption: np.ndarray,
        available_channel_storage_m3: np.ndarray,
        available_groundwater_m3: np.ndarray,
        groundwater_head: np.ndarray,
        available_reservoir_storage_m3: np.ndarray,
        command_areas: np.ndarray,
        return_fraction: float,
        well_depth: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function is used to regulate the irrigation behavior of farmers. The farmers are "activated" by the given `activation_order` and each farmer can irrigate from the various water sources, given water is available and the farmers has the means to abstract water. The abstraction order is channel irrigation, reservoir irrigation, groundwater irrigation. 

        Args:
            activation_order: Order in which the agents are activated. Agents that are activated first get a first go at extracting water, leaving less water for other farmers.
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  
            irrigation_efficiency: Boolean array that specifies whether the specific farmer is efficient with water use.
            irrigated: Array that specifies whether a farm is irrigated.
            well_irrigated: Array that specifies whether a farm is groundwater irrigated.
            cell_area: The area of each subcell in m2.
            HRU_to_grid: Array to map the index of each subcell to the corresponding cell.
            crop_map: Map of the currently growing crops.
            totalPotIrrConsumption: Potential irrigation consumption.
            available_channel_storage_m3: Water available for irrigation from channels.
            groundwater_head: Groundwater head.
            available_groundwater_m3: Water available for irrigation from groundwater.
            available_reservoir_storage_m3: Water available for irrigation from reservoirs.
            command_areas: Command areas associated with reservoirs (i.e., which areas can access water from which reservoir.)

        Returns:
            channel_abstraction_m3_by_farmer: Channel abstraction by farmer in m3.
            reservoir_abstraction_m3_by_farmer: Revervoir abstraction by farmer in m3.
            groundwater_abstraction_m3_by_farmer: Groundwater abstraction by farmer in m3.
            water_withdrawal_m: Water withdrawal in meters.
            water_consumption_m: Water consumption in meters.
            returnFlowIrr_m: Return flow in meters.
            addtoevapotrans_m: Evaporated irrigation water in meters.
        """
        assert n == activation_order.size
          
        land_unit_array_size = cell_area.size
        water_withdrawal_m = np.zeros(land_unit_array_size, dtype=np.float32)
        water_consumption_m = np.zeros(land_unit_array_size, dtype=np.float32)
  
        returnFlowIrr_m = np.zeros(land_unit_array_size, dtype=np.float32)
        addtoevapotrans_m = np.zeros(land_unit_array_size, dtype=np.float32)
  
        groundwater_abstraction_m3 = np.zeros(available_groundwater_m3.size, dtype=np.float32)
        channel_abstraction_m3 = np.zeros(available_channel_storage_m3.size, dtype=np.float32)
  
        reservoir_abstraction_m_per_basin_m3 = np.zeros(available_reservoir_storage_m3.size, dtype=np.float32)
        reservoir_abstraction_m = np.zeros(land_unit_array_size, dtype=np.float32)

        channel_abstraction_m3_by_farmer = np.zeros(activation_order.size, dtype=np.float32)
        reservoir_abstraction_m3_by_farmer = np.zeros(activation_order.size, dtype=np.float32)
        groundwater_abstraction_m3_by_farmer = np.zeros(activation_order.size, dtype=np.float32)
        hydraulic_head_per_farmer = np.zeros(activation_order.size, dtype=np.float32)
  
        has_access_to_irrigation_water = np.zeros(activation_order.size, dtype=np.bool_)
        for activated_farmer_index in range(activation_order.size):
            farmer = activation_order[activated_farmer_index]
            farmer_fields = get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer)
            irrigation_efficiency_farmer = irrigation_efficiency[farmer]

            # Determine whether farmer would have access to irrigation water this timestep. Regardless of whether the water is actually used. This is used for making investment decisions.
            farmer_has_access_to_irrigation_water = False
            for field in farmer_fields:
                f_var = HRU_to_grid[field]
                if well_irrigated[farmer] == 1:
                    if groundwater_head[f_var] - elevation_grid[f_var] < well_depth:
                        farmer_has_access_to_irrigation_water = True
                        break
                elif surface_irrigated[farmer] == 1:
                    if available_channel_storage_m3[f_var] > 100:
                        farmer_has_access_to_irrigation_water = True
                        break
                    command_area = command_areas[field]
                    # -1 means no command area
                    if command_area != -1 and available_reservoir_storage_m3[command_area] > 100:
                        farmer_has_access_to_irrigation_water = True
                        break
            has_access_to_irrigation_water[activated_farmer_index] = farmer_has_access_to_irrigation_water
      
            # Actual irrigation from surface, reservoir and groundwater
            if surface_irrigated[farmer] == 1 or well_irrigated[farmer] == 1:
                for field in farmer_fields:
                    f_var = HRU_to_grid[field]
                    hydraulic_head_per_farmer[farmer] = groundwater_head[f_var]
                    if crop_map[field] != -1:
                        irrigation_water_demand_field = totalPotIrrConsumption[field] / irrigation_efficiency_farmer

                        if surface_irrigated[farmer]:
                            # channel abstraction
                            available_channel_storage_cell_m = available_channel_storage_m3[f_var] / cell_area[field]
                            channel_abstraction_cell_m = min(available_channel_storage_cell_m, irrigation_water_demand_field)
                            channel_abstraction_cell_m3 = channel_abstraction_cell_m * cell_area[field]
                            available_channel_storage_m3[f_var] -= channel_abstraction_cell_m3
                            water_withdrawal_m[field] += channel_abstraction_cell_m
                            channel_abstraction_m3[f_var] = channel_abstraction_cell_m3

                            channel_abstraction_m3_by_farmer[farmer] += channel_abstraction_cell_m3
                      
                            irrigation_water_demand_field -= channel_abstraction_cell_m
                      
                            # command areas
                            command_area = command_areas[field]
                            if command_area >= 0:  # -1 means no command area
                                water_demand_cell_M3 = irrigation_water_demand_field * cell_area[field]
                                reservoir_abstraction_m_cell_m3 = min(available_reservoir_storage_m3[command_area], water_demand_cell_M3)
                                available_reservoir_storage_m3[command_area] -= reservoir_abstraction_m_cell_m3
                                reservoir_abstraction_m_per_basin_m3[command_area] += reservoir_abstraction_m_cell_m3
                                reservoir_abstraction_m_cell = reservoir_abstraction_m_cell_m3 / cell_area[field]
                                reservoir_abstraction_m[field] += reservoir_abstraction_m_cell
                                water_withdrawal_m[field] += reservoir_abstraction_m_cell

                                reservoir_abstraction_m3_by_farmer[farmer] += reservoir_abstraction_m_cell_m3
                          
                                irrigation_water_demand_field -= reservoir_abstraction_m_cell

                        if well_irrigated[farmer]:
                            # groundwater irrigation
                            groundwater_depth = groundwater_head[f_var] - elevation_grid[f_var]
                            if groundwater_depth < well_depth:
                                available_groundwater_cell_m = available_groundwater_m3[f_var] / cell_area[field]
                                groundwater_abstraction_cell_m = min(available_groundwater_cell_m, irrigation_water_demand_field)
                                groundwater_abstraction_cell_m3 = groundwater_abstraction_cell_m * cell_area[field]
                                groundwater_abstraction_m3[f_var] = groundwater_abstraction_cell_m3
                                available_groundwater_m3[f_var] -= groundwater_abstraction_cell_m3
                                water_withdrawal_m[field] += groundwater_abstraction_cell_m

                                groundwater_abstraction_m3_by_farmer[farmer] += groundwater_abstraction_cell_m3
                
                                irrigation_water_demand_field -= groundwater_abstraction_cell_m
              
                        assert irrigation_water_demand_field >= -1e15  # Make sure irrigation water demand is zero, or positive. Allow very small error.

                    water_consumption_m[field] = water_withdrawal_m[field] * irrigation_efficiency_farmer
                    irrigation_loss_m = water_withdrawal_m[field] - water_consumption_m[field]
                    returnFlowIrr_m[field] = irrigation_loss_m * return_fraction
                    addtoevapotrans_m[field] = irrigation_loss_m * (1 - return_fraction)
  
        return (
            channel_abstraction_m3_by_farmer,
            reservoir_abstraction_m3_by_farmer,
            groundwater_abstraction_m3_by_farmer,
            water_withdrawal_m,
            water_consumption_m,
            returnFlowIrr_m,
            addtoevapotrans_m,
            has_access_to_irrigation_water,
            hydraulic_head_per_farmer
        )

    def abstract_water(
        self,
        cell_area: np.ndarray,
        HRU_to_grid: np.ndarray,
        totalPotIrrConsumption: np.ndarray,
        available_channel_storage_m3: np.ndarray,
        available_groundwater_m3: np.ndarray,
        groundwater_head: np.ndarray,
        available_reservoir_storage_m3: np.ndarray,
        command_areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function allows the abstraction of water by farmers for irrigation purposes. It's main purpose is to call the relevant numba function to do the actual abstraction. In addition, the function saves the abstraction from the various sources by farmer.

        Args:
            cell_area: the area of each subcell in m2.
            HRU_to_grid: array to map the index of each subcell to the corresponding cell.
            totalPotIrrConsumption: potential irrigation consumption.
            available_channel_storage_m3: water available for irrigation from channels.
            groundwater_head: groundwater head.
            available_groundwater_m3: water available for irrigation from groundwater.
            available_reservoir_storage_m3: water available for irrigation from reservoirs.
            command_areas: command areas associated with reservoirs (i.e., which areas can access water from which reservoir.)

        Returns:
            water_withdrawal_m: water withdrawal in meters
            water_consumption_m: water consumption in meters
            returnFlowIrr_m: return flow in meters
            addtoevapotrans_m: evaporated irrigation water in meters
        """
        (
            self.channel_abstraction_m3_by_farmer,
            self.reservoir_abstraction_m3_by_farmer,
            self.groundwater_abstraction_m3_by_farmer,
            water_withdrawal_m,
            water_consumption_m,
            returnFlowIrr_m,
            addtoevapotrans_m,
            has_access_to_irrigation_water,
            hydraulic_head_per_farmer
        ) = self.abstract_water_numba(
            self.n,
            self.activation_order_by_elevation,
            self.field_indices_by_farmer,
            self.field_indices,
            self.elevation_grid,
            self.irrigation_efficiency,
            surface_irrigated=np.isin(self.irrigation_source, [self.irrigation_source_key['canals'], self.irrigation_source_key['other']]),
            well_irrigated=np.isin(self.irrigation_source, [self.irrigation_source_key['well'], self.irrigation_source_key['tubewell']]),
            cell_area=cell_area,
            HRU_to_grid=HRU_to_grid,
            crop_map=self.var.crop_map,
            totalPotIrrConsumption=totalPotIrrConsumption,
            available_channel_storage_m3=available_channel_storage_m3,
            available_groundwater_m3=available_groundwater_m3,
            groundwater_head=groundwater_head,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
            command_areas=command_areas,
            return_fraction=self.model.config['agent_settings']['farmers']['return_fraction'],
            well_depth=30
        )
        self.n_water_accessible_days += has_access_to_irrigation_water
        self.groundwater_depth = self.elevation - hydraulic_head_per_farmer
        return (
            water_withdrawal_m,
            water_consumption_m,
            returnFlowIrr_m,
            addtoevapotrans_m
        )

    @staticmethod
    @njit
    def get_yield_ratio_numba(crop_map: np.array, evap_ratios: np.array, KyT) -> float:
        """Calculate yield ratio based on https://doi.org/10.1016/j.jhydrol.2009.07.031
  
        Args:
            crop_map: array of currently harvested crops.
            evap_ratios: ratio of actual to potential evapotranspiration of harvested crops.
            alpha: alpha value per crop used in MIRCA2000.
            beta: beta value per crop used in MIRCA2000.
            P0: P0 value per crop used in MIRCA2000.
            P1: P1 value per crop used in MIRCA2000.
  
        Returns:
            yield_ratios: yield ratio (as ratio of maximum obtainable yield) per harvested crop.
        """
        yield_ratios = np.full(evap_ratios.size, -1, dtype=np.float32)

        assert crop_map.size == evap_ratios.size
  
        for i in range(evap_ratios.size):
            evap_ratio = evap_ratios[i]
            crop = crop_map[i]
            yield_ratios[i] = max(1 - KyT[crop] * (1 - evap_ratio), 0)  # Yield ratio is never lower than 0.
  
        return yield_ratios

    def get_yield_ratio(self, harvest: np.ndarray, actual_transpiration: np.ndarray, potential_transpiration: np.ndarray, crop_map: np.ndarray) -> np.ndarray:
        """Gets yield ratio for each crop given the ratio between actual and potential evapostranspiration during growth.
  
        Args:
            harvest: Map of crops that are harvested.
            actual_transpiration: Actual evapotranspiration during crop growth period.
            potential_transpiration: Potential evapotranspiration during crop growth period.
            crop_map: Subarray of type of crop grown.

        Returns:
            yield_ratio: Map of yield ratio.

        TODO: Implement GAEZ crop stage function
        """
        yield_ratio = self.get_yield_ratio_numba(
            crop_map[harvest],
            actual_transpiration[harvest] / potential_transpiration[harvest],
            self.crop_variables['KyT'].values,
        )
        assert not np.isnan(yield_ratio).any()
        return yield_ratio

    def precipitation_sum(self) -> None:
        # To create unique groups based on water abstraction, sum the abstractions. Later used to make groups.  To do: chang e
        self.yearly_abstraction_m3_by_farmer[:,0] += self.channel_abstraction_m3_by_farmer
        self.yearly_abstraction_m3_by_farmer[:,1] += self.reservoir_abstraction_m3_by_farmer
        self.yearly_abstraction_m3_by_farmer[:,2] += self.groundwater_abstraction_m3_by_farmer
        self.yearly_abstraction_m3_by_farmer[:,3] += self.channel_abstraction_m3_by_farmer + self.reservoir_abstraction_m3_by_farmer + self.groundwater_abstraction_m3_by_farmer
    
    def SPEI_sum(self) -> None:
        #  ## SPEI is recorded on the 16th, except in february, then it is on the 15th 
        # if self.model.current_time.month == 2:
        #     day_of_the_month = 15
        # else:
        #     day_of_the_month = 16
        
        # Store SPEI per month in a 2d array of past days. After new day is stored, shift the daily precipitation a day back. 
        self.monthly_SPEI[:,1:] = self.monthly_SPEI[:,0:-1]
        self.monthly_SPEI[:,0] = self.SPEI_map.sample_coords(self.locations, datetime(self.model.current_time.year, self.model.current_time.month, 1))

    def save_profit_water_rain(self, harvesting_farmers: np.ndarray, profit: np.ndarray, potential_profit: np.ndarray) -> None:
        """Saves the current harvest for harvesting farmers in a 2-dimensional array. The first dimension is the different farmers, while the second dimension are the previous harvests. 
        First the previous harvests are moved by 1 column (dropping old harvests) to make room for the new harvest. Then, the new harvest is placed in the array.
  
        Args:
            harvesting_farmers: farmers that harvest in this timestep.
        """
        
        assert (profit >= 0).all()
        assert (potential_profit >= 0).all()

        # shift all columns one column further, the last falls off.
        self.latest_profits[harvesting_farmers, 1:] = self.latest_profits[harvesting_farmers, 0:-1]
        self.latest_profits[harvesting_farmers, 0] = profit[harvesting_farmers]
        # shift all columns one column further, the last falls off.
        self.latest_potential_profits[harvesting_farmers, 1:] = self.latest_potential_profits[harvesting_farmers, 0:-1]
        self.latest_potential_profits[harvesting_farmers, 0] = potential_profit[harvesting_farmers]

        # The cumulative inflation at the current year compared to the first year of the model run 
        inflation_arrays = [self.get_value_per_farmer_from_region_id(self.inflation_rate, datetime(year, 1, 1)) for year in range(self.model.config['general']['spinup_time'].year, self.model.current_time.year + 1)]
        cum_inflation = np.ones_like(inflation_arrays[0])
        for inflation in inflation_arrays:
            cum_inflation *= inflation
        ## Variable that sums harvests within a year. After a year has passed the total is moved to the second column (next function). Correct for field size and inflation
        self.yearly_profits[harvesting_farmers, 0] += self.latest_profits[harvesting_farmers, 0] / self.field_size_per_farmer[harvesting_farmers] / cum_inflation[harvesting_farmers]

    def store_long_term_damages_profits_rain(self) -> None:
        """Saves the yearly profit, rainfall and yield ratios and stores them. Then calculates for each unique farmer type what their yearly mean is. 
        """
        # calculate the average yield ratio for that year 
        total_planted_time = self.total_crop_age[:,0] + self.total_crop_age[:,1] + self.total_crop_age[:,2]
        
        ## Mask where total_planted time is 0 (no planting was done)
        total_planted_time = np.ma.masked_where(total_planted_time == 0, total_planted_time)

        # add the yield ratio proportional to the total planting time 
        self.yearly_yield_ratio[:, 0] = (
            self.total_crop_age[:, 0] / total_planted_time * self.per_harvest_yield_ratio[:, 0] + #season 1 yield ratio 
            self.total_crop_age[:, 1] / total_planted_time * self.per_harvest_yield_ratio[:, 1] + #season 2 yield ratio 
            self.total_crop_age[:, 2] / total_planted_time * self.per_harvest_yield_ratio[:, 2]   #season 3 yield ratio 
            ) 
        
        # Convert the seasonal SPEI to yearly SPEI probability 
        seasonal_SPEI_probability = np.zeros((self.n, 3), dtype=np.float32)

        seasonal_SPEI_probability[self.per_harvest_SPEI[:,0] != 0,0] = genextreme.sf((self.per_harvest_SPEI[self.per_harvest_SPEI[:,0] != 0 ,0]), 
                                                           self.GEV_parameters[self.per_harvest_SPEI[:,0] != 0 ,0], 
                                                           self.GEV_parameters[self.per_harvest_SPEI[:,0] != 0 ,1], 
                                                           self.GEV_parameters[self.per_harvest_SPEI[:,0] != 0 ,2])
        seasonal_SPEI_probability[self.per_harvest_SPEI[:,1] != 0,1] = genextreme.sf((self.per_harvest_SPEI[self.per_harvest_SPEI[:,1] != 0 ,1]), 
                                                           self.GEV_parameters[self.per_harvest_SPEI[:,1] != 0 ,0], 
                                                           self.GEV_parameters[self.per_harvest_SPEI[:,1] != 0 ,1], 
                                                           self.GEV_parameters[self.per_harvest_SPEI[:,1] != 0 ,2])
        seasonal_SPEI_probability[self.per_harvest_SPEI[:,2] != 0,2] = genextreme.sf((self.per_harvest_SPEI[self.per_harvest_SPEI[:,2] != 0 ,2]), 
                                                           self.GEV_parameters[self.per_harvest_SPEI[:,2] != 0 ,0], 
                                                           self.GEV_parameters[self.per_harvest_SPEI[:,2] != 0 ,1], 
                                                           self.GEV_parameters[self.per_harvest_SPEI[:,2] != 0 ,2])
        
        # seasonal_SPEI_probability[self.per_harvest_SPEI[:,0] != 0,0] = norm.cdf(self.per_harvest_SPEI[self.per_harvest_SPEI[:,0] != 0 ,0])
        # seasonal_SPEI_probability[self.per_harvest_SPEI[:,1] != 0,1] = norm.cdf(self.per_harvest_SPEI[self.per_harvest_SPEI[:,1] != 0 ,1])
        # seasonal_SPEI_probability[self.per_harvest_SPEI[:,2] != 0,2] = norm.cdf(self.per_harvest_SPEI[self.per_harvest_SPEI[:,2] != 0 ,2])
        

        # Save the average yearly probability in the yearly precipitation probability by summing and dividing through the planting seasons
        nonzero_count = np.count_nonzero(seasonal_SPEI_probability, axis=1)
        nr_planting_seasons = np.where(nonzero_count == 0, 1, nonzero_count)
        self.yearly_SPEI_probability[:, 0] = np.sum(seasonal_SPEI_probability, axis=1) / nr_planting_seasons

        self.per_harvest_SPEI  = 0

        # shift all columns one column further, the last falls off. 
        self.yearly_yield_ratio[:, 1:] = self.yearly_yield_ratio[:, 0:-1]
        # Set the first column to 0
        self.yearly_yield_ratio[:, 0] = 0
        # set the total crop age and per crop yield ratios to 0 
        self.total_crop_age[:, :] = 0
        self.per_harvest_yield_ratio[:, :] = 0

        #shift all columns one column further, the last falls off. 
        self.yearly_SPEI_probability[:, 1:] = self.yearly_SPEI_probability[:, 0:-1]
        #Reset the first column of the HRU and yearly precipitation columns to 0 
        self.yearly_SPEI_probability[:, 0] = 0
        
        # Now convert the yearly values per individual farmer to unique farmer types
        # First check if these variables already exist, otherwise make them. The total years they use is equal to the spinup time. 
        unique_yearly_yield_ratio = np.empty((0, self.total_spinup_time))
        unique_SPEI_probability = np.empty((0, self.total_spinup_time))

        # Make a new variable that has crop combination and the farmer class (what type of water they use), as to make unique groups based on this
        crop_irrigation_groups = self.crops[:]

        # Save the profits, rainfall and water shortage for each farmer type. To do: change this to a vectorized operation, determine and add more homogene farmer groups 
        for crop_combination in np.unique(crop_irrigation_groups, axis=0):

            unique_farmer_groups = np.where((crop_irrigation_groups==crop_combination[None, ...]).all(axis=1))[0]
            
            # Calculate averages of profits, yield ratio, precipitation, and probability for the same farmer groups
            average_yield_ratio = np.mean(self.yearly_yield_ratio[unique_farmer_groups, 1:], axis=0)
            average_probability = np.mean(self.yearly_SPEI_probability[unique_farmer_groups, 1:], axis=0)
            
            # Prepend the averages to respective arrays
            unique_yearly_yield_ratio = np.vstack((unique_yearly_yield_ratio, average_yield_ratio))
            unique_SPEI_probability = np.vstack((unique_SPEI_probability, average_probability))

        # Mask rows that consist only of 0s --> no relation possible 
        mask_rows = np.any((unique_yearly_yield_ratio != 0), axis=1) & np.any((unique_SPEI_probability != 0), axis=1)
        unique_yearly_yield_ratio_mask = unique_yearly_yield_ratio[mask_rows]
        unique_SPEI_probability_mask = unique_SPEI_probability[mask_rows]
        
        # Mask columns with only zeros
        mask_columns = np.any((unique_yearly_yield_ratio_mask != 0), axis=0) & np.any((unique_SPEI_probability_mask != 0), axis=0)
        unique_yearly_yield_ratio_mask = unique_yearly_yield_ratio_mask[:,mask_columns]
        unique_SPEI_probability_mask = unique_SPEI_probability_mask[:,mask_columns]

        ## Mask the minimum and the maximum value 
        arrays = [ unique_yearly_yield_ratio_mask, unique_SPEI_probability_mask]
        masked_arrays = []
        
        ## To do: perhaps mask all zeros 
        for array in arrays:
            mask = np.ones_like(array, dtype=bool)
            # Find the indices of the maximum and minimum values along axis 1 (columns)
            max_indices = np.argmax(array, axis=1)
            min_indices = np.argmin(array, axis=1)

            # Update the mask for the corresponding column indices
            mask[np.arange(array.shape[0]), max_indices] = False
            mask[np.arange(array.shape[0]), min_indices] = False

            # Create a masked array using the mask
            masked_array = np.ma.array(array, mask=~mask)
            masked_arrays.append(masked_array)

        masked_unique_yearly_yield_ratio, masked_unique_SPEI_probability= masked_arrays
        
        ## Create empty lists to append the relations to 
        group_yield_probability_relation = []

        yield_probability_R2_scipy = []
        yield_probability_p_scipy = []
        yield_probability_std_err_scipy = []
        yield_probability_adj_r_squared_scipy = []
        
        # Determine the relation between the remaining rows. TO DO: vectorize further
        for i, (row1, row3) in enumerate(zip(masked_unique_yearly_yield_ratio, masked_unique_SPEI_probability), 1):
            ## Determine the relation between yield ratio and profit for all farmer types
            # Calculate the coefficients and save them  
            coefficients_yield_probability = np.polyfit(row1, row3, 1)
            poly_yield_probability= np.poly1d(coefficients_yield_probability)
            group_yield_probability_relation.append(poly_yield_probability)

            if len(masked_unique_yearly_yield_ratio[0,:]) > 20:
                slope_yield, intercept_yield, r_value_yield, p_value_yield, std_err_yield = linregress(row1, row3)
                r_squared = r_value_yield**2
                yield_probability_R2_scipy.append(r_squared)
                # Calculate adj_r_squared for a simple linear 
                n = len(row1) # number of observations
                adj_r_squared = 1 - ((1 - r_squared) * (n - 1)) / (n - 2)
                yield_probability_adj_r_squared_scipy.append(adj_r_squared)
                yield_probability_p_scipy.append(p_value_yield)
                yield_probability_std_err_scipy.append(std_err_yield)

        # Sample the individual agent relation from the agent groups 
        # Where does each agent sit in the list of unique groups
        positions_agent = np.where(np.all(crop_irrigation_groups[:, np.newaxis, :] == np.unique(crop_irrigation_groups, axis=0), axis=-1))
        exact_position = positions_agent[1]

        if len(group_yield_probability_relation) <= max(exact_position):
            pass
        else:
            self.farmer_yield_probability_relation = np.array(group_yield_probability_relation)[exact_position]
            assert isinstance(self.farmer_yield_probability_relation, np.ndarray), "self.farmer_yield_probability_relation must be a np.ndarray"
        
        print('r2:', np.median(yield_probability_R2_scipy), 'adj_r2:', np.median(yield_probability_adj_r_squared_scipy), 'p: ', np.median(yield_probability_p_scipy), 'std_err: ', np.median(yield_probability_std_err_scipy))

    def by_field(self, var, nofieldvalue=-1):
        if self.n:
            by_field = np.take(var, self.var.land_owners)
            by_field[self.var.land_owners == -1] = nofieldvalue
            return by_field
        else:
            return np.full_like(self.var.land_owners, nofieldvalue)

    def decompress(self, array):
        if np.issubsctype(array, np.floating):
            nofieldvalue = np.nan
        else:
            nofieldvalue = -1
        return self.model.data.HRU.decompress(self.by_field(array, nofieldvalue=nofieldvalue))

    @property
    def mask(self):
        mask = self.model.data.HRU.mask.copy()
        mask[self.decompress(self.var.land_owners) == -1] = True
        return mask
          
    @staticmethod
    @njit(cache=True)
    def harvest_numba(
        n: np.ndarray,
        field_indices_by_farmer: np.ndarray,
        field_indices: np.ndarray,
        crop_map: np.ndarray,
        crop_age_days: np.ndarray,
        crop_harvest_age_days: np.ndarray,
    ) -> np.ndarray:
        """This function determines whether crops are ready to be harvested by comparing the crop harvest age to the current age of the crop. If the crop is harvested, the crops next multicrop index and next plant day are determined.

        Args:
            n: Number of farmers.
            start_day_per_month: Array containing the starting day of each month.
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  
            crop_map: Subarray map of crops.
            crop_age_days: Subarray map of current crop age in days.
            n_water_accessible_days: Number of days that crop was water limited.
            crop: Crops grown by each farmer.
            switch_crops: Whether to switch crops or not.

        Returns:
            harvest: Boolean subarray map of fields to be harvested.
        """
        harvest = np.zeros(crop_map.shape, dtype=np.bool_)
        for farmer_i in range(n):
            farmer_fields = get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer_i)
            for field in farmer_fields:
                crop_age = crop_age_days[field]
                if crop_age >= 0:
                    crop = crop_map[field]
                    assert crop != -1
                    if crop_age == crop_harvest_age_days[field]:
                        harvest[field] = True
                        crop_harvest_age_days[field] = -1
                else:
                    assert crop_map[field] == -1
        return harvest      

    def harvest(self):
        """This function determines which crops needs to be harvested, based on the current age of the crops and the harvest age of the crop. First a helper function is used to obtain the harvest map. Then, if at least 1 field is harvested, the yield ratio is obtained for all fields using the ratio of actual to potential evapotranspiration, saves the harvest per farmer and potentially invests in water saving techniques.
        """
        harvest = self.harvest_numba(
            n=self.n,
            field_indices_by_farmer=self.field_indices_by_farmer,
            field_indices=self.field_indices,
            crop_map=self.var.crop_map,
            crop_age_days=self.var.crop_age_days_map,
            crop_harvest_age_days=self.var.crop_harvest_age_days,
        )
        if np.count_nonzero(harvest):  # Check if any harvested fields. Otherwise we don't need to run this.
            yield_ratio = self.get_yield_ratio(harvest, self.var.actual_transpiration_crop, self.var.potential_transpiration_crop, self.var.crop_map)
            assert (yield_ratio >= 0).all()

            harvesting_farmer_fields = self.var.land_owners[harvest]
            harvested_area = self.var.cellArea[harvest]
            if self.model.use_gpu:
                harvested_area = harvested_area.get()
            harvested_crops = self.var.crop_map[harvest]
            max_yield_per_crop = np.take(self.crop_variables['reference_yield_kg_m2'].values, harvested_crops)
      
            crop_prices = self.crop_prices[1][self.crop_prices[0].get(self.model.current_time)]
            assert not np.isnan(crop_prices).any()
            
            harvesting_farmers = np.unique(harvesting_farmer_fields)

            # get potential crop profit per farmer
            crop_yield_kg = harvested_area * yield_ratio * max_yield_per_crop
            assert (crop_yield_kg >= 0).all()
            crop_prices_per_field = crop_prices[harvested_crops]
            profit = crop_yield_kg * crop_prices_per_field
            assert (profit >= 0).all()
            
            self.profit = np.bincount(harvesting_farmer_fields, weights=profit, minlength=self.n)

            ## Set the current crop age
            crop_age = self.var.crop_age_days_map[harvest]
            total_crop_age = np.bincount(harvesting_farmer_fields, weights = crop_age, minlength=self.n) / np.bincount(harvesting_farmer_fields, minlength=self.n)

            ## Convert the yield_ratio per field to the average yield ratio per farmer 
            yield_ratio_agent = np.bincount(harvesting_farmer_fields, weights = yield_ratio, minlength=self.n) / np.bincount(harvesting_farmer_fields, minlength=self.n)

            # Take the mean of the growing months and change the sign to fit the GEV distribution 
            cum_SPEI_latest_harvest = np.mean(self.monthly_SPEI[harvesting_farmers, :int((crop_age[0] / 30))], axis=1) * -1
            # cum_SPEI_latest_harvest = self.monthly_SPEI[harvesting_farmers, 0] * -1
            

            ## Add the yield ratio, precipitation and the crop age to the array corresponding to the current season. Precipitation is already converted to daily rainfall
            if self.current_season_idx == 0:
                self.total_crop_age[harvesting_farmers, 0] = total_crop_age[harvesting_farmers]
                self.per_harvest_yield_ratio[harvesting_farmers, 0] = yield_ratio_agent[harvesting_farmers]
                self.per_harvest_SPEI[harvesting_farmers,0] = cum_SPEI_latest_harvest
            elif self.current_season_idx == 1:
                self.total_crop_age[harvesting_farmers, 1] = total_crop_age[harvesting_farmers]
                self.per_harvest_yield_ratio[harvesting_farmers, 1] = yield_ratio_agent[harvesting_farmers]
                self.per_harvest_SPEI[harvesting_farmers,1] = cum_SPEI_latest_harvest
            elif self.current_season_idx == 2:
                self.total_crop_age[harvesting_farmers, 2] = total_crop_age[harvesting_farmers]
                self.per_harvest_yield_ratio[harvesting_farmers, 2] = yield_ratio_agent[harvesting_farmers]
                self.per_harvest_SPEI[harvesting_farmers,2] = cum_SPEI_latest_harvest
           
            # get potential crop profit per farmer
            potential_crop_yield = harvested_area * max_yield_per_crop
            potential_profit = potential_crop_yield * crop_prices_per_field
            potential_profit = np.bincount(harvesting_farmer_fields, weights=potential_profit, minlength=self.n)
      
            self.save_profit_water_rain(harvesting_farmers, self.profit, potential_profit)
            
            self.drought_risk_perception(harvesting_farmers)

           
            
            ## After updating the drought risk perception, set the previous month for the next timestep as the current for this timestep.
            self.previous_month = self.model.current_time.month

            self.disposable_income += self.profit

        
        else:
            self.profit = np.zeros(self.n, dtype=np.float32)
  
        self.var.actual_transpiration_crop[harvest] = 0
        self.var.potential_transpiration_crop[harvest] = 0

        # remove crops from crop_map where they are harvested
        self.var.crop_map[harvest] = -1
        self.var.crop_age_days_map[harvest] = -1

        # when a crop is harvested set to non irrigated land
        self.var.land_use_type[harvest] = 1

        # increase crop age by 1 where crops are not harvested and growing
        self.var.crop_age_days_map[(harvest == False) & (self.var.crop_map >= 0)] += 1

        assert (self.var.crop_age_days_map <= self.var.crop_harvest_age_days).all()

    @staticmethod
    @njit(cache=True)
    def plant_numba(
        n: int,
        season_idx: int,
        is_first_day_of_season: bool,
        growth_length: np.ndarray,
        crop_map: np.ndarray,
        crop_harvest_age_days: np.ndarray,
        crops: np.ndarray,
        cultivation_cost_per_crop: np.ndarray,
        field_indices_by_farmer: np.ndarray,
        field_indices: np.ndarray,
        field_size_per_farmer: np.ndarray,
        disposable_income: np.ndarray,
        farmers_going_out_of_business: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Determines when and what crop should be planted, by comparing the current day to the next plant day. Also sets the haverst age of the plant.
  
        Args:
            n: Number of farmers.
            start_day_per_month: Starting day of each month of year.
            current_day: Current day.
            crop: Crops grown by each farmer. 
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices. 
            field_size_per_farmer: Field size per farmer in m2

        Returns:
            plant: Subarray map of what crops are planted this day.
        """
        plant = np.full_like(crop_map, -1, dtype=np.int32)
        sell_land = np.zeros(disposable_income.size, dtype=np.bool_)
        for farmer_idx in range(n):
            farmer_fields = get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer_idx)
            if is_first_day_of_season:
                farmer_crop = crops[farmer_idx, season_idx]
                if farmer_crop == -1:
                    continue
            else:
                continue
            assert farmer_crop != -1
            cultivation_cost = cultivation_cost_per_crop[farmer_crop] * field_size_per_farmer[farmer_idx]
            assert not np.isnan(cultivation_cost)
            if not farmers_going_out_of_business or disposable_income[farmer_idx] > cultivation_cost:
                disposable_income[farmer_idx] -= cultivation_cost
                field_harvest_age = growth_length[farmer_crop, season_idx]
                for field in farmer_fields:
                    # a crop is still growing here.
                    if crop_harvest_age_days[field] != -1:
                        continue
                    plant[field] = farmer_crop
                    crop_harvest_age_days[field] = field_harvest_age
            else:
                sell_land[farmer_idx] = True
        farmers_selling_land = np.where(sell_land)[0]
        return plant, farmers_selling_land

    def plant(self) -> None:
        """Determines when and what crop should be planted, mainly through calling the :meth:`agents.farmers.Farmers.plant_numba`. Then converts the array to cupy array if model is running with GPU.
        """
        index = self.cultivation_costs[0].get(self.model.current_time)
        cultivation_cost_per_crop = self.cultivation_costs[1][index]

        # create numpy stack of growth length per crop and season
        growth_length = np.stack([
            self.crop_variables['season_#1_duration'],
            self.crop_variables['season_#2_duration'],
            self.crop_variables['season_#3_duration']
        ], axis=1)

        plant_map, farmers_selling_land = self.plant_numba(
            n=self.n,
            season_idx=self.current_season_idx,
            is_first_day_of_season=self.is_first_day_of_season,
            growth_length=growth_length,
            crop_map=self.var.crop_map,
            crop_harvest_age_days=self.var.crop_harvest_age_days,
            crops=self.crops,
            cultivation_cost_per_crop=cultivation_cost_per_crop,
            field_indices_by_farmer=self.field_indices_by_farmer,
            field_indices=self.field_indices,
            field_size_per_farmer=self.field_size_per_farmer,
            disposable_income=self.disposable_income,
            farmers_going_out_of_business=(
                self.model.config['agent_settings']['farmers']['farmers_going_out_of_business']
                and not self.model.scenario == 'spinup'  # farmers can only go out of business when not in spinup scenario
            )
        )
        if farmers_selling_land.size > 0:
            self.remove_agents(farmers_selling_land)

        self.var.crop_map = np.where(plant_map >= 0, plant_map, self.var.crop_map)
        self.var.crop_age_days_map[plant_map >= 0] = 0

        assert (self.var.crop_age_days_map[self.var.crop_map > 0] >= 0).all()

        field_is_paddy_irrigated = (self.var.crop_map == self.crop_names['Paddy'])
        self.var.land_use_type[(self.var.crop_map >= 0) & (field_is_paddy_irrigated == True)] = 2
        self.var.land_use_type[(self.var.crop_map >= 0) & (field_is_paddy_irrigated == False)] = 3
    
    def drought_risk_perception(self, harvesting_farmers: np.ndarray):
        ## convert the harvesting farmers array to a full length array with true values  TO DO: check whether the drought timer and such is logical here. 
        harvesting_farmers_long = np.zeros(self.n, dtype=bool)
        harvesting_farmers_long[harvesting_farmers] = True

        ## Update the drought timer for every timestep (harvest) that it is activated, update is the difference between last and current timestep 
        months_passed = (self.model.current_time.month - self.previous_month) % 12
        self.drought_timer += (months_passed / 12) 
        
        ## Determine the loss between potential and actual profits in % for all the harvesting farmers 
        self.drought_loss[harvesting_farmers_long] = np.abs(((self.latest_profits[harvesting_farmers_long] - self.latest_potential_profits[harvesting_farmers_long]) / self.latest_potential_profits[harvesting_farmers_long]) * 100)
        
        ## Determine latest and past average loss percentages 
        drought_loss_latest = self.drought_loss[:,0]
        drought_loss_past = np.mean(self.drought_loss[:, 1:], axis=1)

        ## Farmers experience drought events if the loss is larger than that of the last few years, or very large (thresholds t.b.d.)
        experienced_drought_event = np.logical_or(drought_loss_past - drought_loss_latest >= self.moving_average_threshold, drought_loss_latest >= self.absolute_threshold)
        
        # Reset drought timer on locations that have harvesting farmers and that have experienced a drought event 
        self.drought_timer[np.logical_and(harvesting_farmers_long, experienced_drought_event)] = 0

        # Calculate the updated risk perception of all farmers 
        self.risk_perception = self.risk_perc_max * (1.6 ** (self.risk_decr * self.drought_timer)) + self.risk_perc_min

        print('Risk perception mean = ',np.mean(self.risk_perception))
    
    def profits_SEUT(self, adaptation_type):
        yield_ratios = self.yield_ratios_drought_event[:]

        # Calculate the alternate yield-ratio with an adaptation 
        gains_adaptation = self.adaptation_yield_ratio_difference(adaptation_type)
        yield_ratios_adaptation = yield_ratios * gains_adaptation[:, None]

        # Set the maximum yield ratio to 1 
        yield_ratios_adaptation[yield_ratios_adaptation > 1] = 1

        total_profits = np.zeros((self.n, len(self.p_droughts)))
        total_profits_adaptation = np.zeros((self.n, len(self.p_droughts)))

        # Create a mask for valid crops (all non -1 and none above the max length of the reference yield)
        crops_mask = (self.crops >= 0) & (self.crops < len(self.crop_variables['reference_yield_kg_m2']))
        # Create an output array with NaNs
        array_with_reference = np.full_like(self.crops, fill_value=np.nan, dtype=float)

        # Now iterate over each individual yield ratio to get the profit
        for col in range(yield_ratios.shape[1]):
            YR_one_probability = yield_ratios[:, col]
            total_profits[:,col] = self.yield_ratio_to_profit(YR_one_probability, crops_mask, array_with_reference)
            YR_one_probability_adaptation = yield_ratios_adaptation[:, col]
            total_profits_adaptation[:,col] = self.yield_ratio_to_profit(YR_one_probability_adaptation, crops_mask, array_with_reference)

        # Transpose columns because to fit the format of the decision module 
        total_profits = total_profits.T[:-1, :]
        total_profits_adaptation = total_profits_adaptation.T[:-1, :]
        # The last event is with probability 1/1, so the "no" event 
        profits_no_event = total_profits[-1,:]
        profits_no_event_adaptation = total_profits_adaptation[-1,:]
        
        return total_profits, total_profits_adaptation, profits_no_event, profits_no_event_adaptation
    
    def adaptation_yield_ratio_difference(self, adaptation_type):
        # add 0s to the last column, to create a loopable group of those farmers which have not adapted 
        crop_groups_onlyzeros = np.hstack((self.crops[:,:], np.zeros(self.n).reshape(-1,1)))
        # Also create the full, original group 
        crop_groups = np.hstack((self.crops[:,:], self.adapted[:,adaptation_type].reshape(-1,1)))
        # Create empty array to put the groups' values in 
        unique_yield_ratio_gain_relative = np.zeros(len(np.unique(crop_groups_onlyzeros, axis=0)), dtype= np.float32)

        # Determine for each farmer group the average yield ratio of the portion that has/has not adapted 
        for unique_combination in np.unique(crop_groups_onlyzeros, axis=0):
            # Determine the group that has not adapted
            unique_farmer_groups = (crop_groups == unique_combination[None, ...]).all(axis=1)
            
            # Determine their counterpart group that has adapted by changing the 0 (not adapted) to a 1 (adapted)
            unique_combination_adapted = unique_combination[:]
            unique_combination_adapted[-1] = 1
            unique_farmer_groups_adapted = (crop_groups == unique_combination_adapted[None, ...]).all(axis=1)

            # calculate the mean of the groups over the past x recorded years.
            unadapted_yield_ratio = np.mean(self.yearly_yield_ratio[unique_farmer_groups, :10], axis=1)
            adapted_yield_ratio = np.mean(self.yearly_yield_ratio[unique_farmer_groups_adapted, :10], axis=1)

            # If there are no people in either the adapted or not adapted groups, the yield ratio difference becomes 0. TO DO: dynamic groupmaking
            if len(unadapted_yield_ratio) == 0 or len(adapted_yield_ratio) == 0:
                unadapted_yield_ratio = np.array(1)
                adapted_yield_ratio = np.array(1)
            
            # Calculate the relative yield ratio gain 
            yield_ratio_gain_relative = np.median(adapted_yield_ratio) / np.median(unadapted_yield_ratio)

            # Calculate how big the adapted group is as opposed to the unadapted group. if the groups are equal size or adapted is bigger, prob = 100%
            adapted_unadapted_ratio = min(adapted_yield_ratio.size / unadapted_yield_ratio.size, 1.0)

            # If the adapted group size is large, there is a higher chance of changing the yield ratio
            # This is to prevent few adapted farmers changing the ratios of the whole group 
            if np.random.rand() < adapted_unadapted_ratio:
                # If the adapted group is equal or larger, probability is 100%. 
                unique_yield_ratio_gain_relative = np.hstack((yield_ratio_gain_relative, unique_yield_ratio_gain_relative))
            else:
                # Else, the difference becomes 1 (no difference)
                unique_yield_ratio_gain_relative = np.hstack((1, unique_yield_ratio_gain_relative))
            

        # Where does each agent sit in the list of unique groups
        positions_agent = np.where(np.all(crop_groups_onlyzeros[:, np.newaxis, :] == np.unique(crop_groups_onlyzeros, axis=0), axis=-1))
        exact_position = positions_agent[1]

        # Convert the group ratio gain to the agent ratio gain 
        return unique_yield_ratio_gain_relative[exact_position]

    def yield_ratio_to_profit(self, yield_ratios, crops_mask, array_with_reference):
        # Determine the crop prices at this moment (to do: maybe average to that of past x years)
        crop_prices = self.crop_prices[1][self.crop_prices[0].get(self.model.current_time)]
        assert not np.isnan(crop_prices).any()

        array_with_reference_yield = array_with_reference[:]
        array_with_price = array_with_reference[:]

        # Apply np.take only on the valid values for both the reference yield and the crop prices 
        array_with_reference_yield[crops_mask] = np.take(self.crop_variables['reference_yield_kg_m2'].values, self.crops[crops_mask].astype(int))
        array_with_price[crops_mask] = np.take(crop_prices, self.crops[crops_mask].astype(int))
        
        # Calculate the mean along axis 1 (rows), ignoring NaN values
        average_reference_yield = np.nanmean(array_with_reference_yield, axis=1)
        average_crop_price = np.nanmean(array_with_price, axis=1)

        # Now calculate the max yield in grams, then calculate the profit. 
        # Change harvested area in fields to harvested area per farmer 
        farmer_fields_ID = self.var.land_owners
        harvested_area = np.bincount(farmer_fields_ID[farmer_fields_ID != -1], weights=self.var.cellArea[farmer_fields_ID != -1], minlength=self.n)

        crop_yield_kg = harvested_area * yield_ratios * average_reference_yield
        assert (crop_yield_kg >= 0).all()

        profit = crop_yield_kg * average_crop_price

        return profit 
    
    def convert_probability_to_yield_ratio(self) -> None:

        # Initialize a 2D array to store the yield ratios connected to each drought probability. will be a p_droughts, 
        yield_ratios = np.zeros((self.farmer_yield_probability_relation.shape[0], self.p_droughts.size))

        for i, coeffs in enumerate(self.farmer_yield_probability_relation):
            # Invert the relationship to obtain the probability-yield ratio
            a = coeffs[0]
            b = coeffs[1]
            if a != 0:
                inverse_coefficients = [1/a, -b/a]
                inverse_polynomial = np.poly1d(inverse_coefficients)
            else:
                raise AssertionError("The relationship is not invertible, as the slope is zero.")
            # Calculate the yield ratio per farmer, placeholder name 
            yield_ratios[i, :] = inverse_polynomial(1 / self.p_droughts) 
        
        # Change all negative yield ratios to 0 
        yield_ratios[yield_ratios < 0] = 0
        # Set the maximum yield ratio to 1. Technically this would not be needed, but both the SPEI return periods and SPEI - yield ratio relations are not perfect so some outliers can be expected. 
        yield_ratios[yield_ratios > 1] = 1

        self.yield_ratios_drought_event = yield_ratios[:] 

    def adapt_SEUT(self, adaptation_type, annual_cost, loan_duration):
        decision_module = DecisionModule(self)
        total_profits, total_profits_adaptation, profits_no_event, profits_no_event_adaptation = self.profits_SEUT(adaptation_type)

        # Determine the total yearly costs of farmers if they would adapt this cycle 
        total_annual_costs = annual_cost + self.annual_costs_all_adaptations
        
        decision_params = {'loan_duration': loan_duration,  
                        'expenditure_cap': self.expenditure_cap, 
                        'n_agents':  self.n, 
                        'sigma': self.risk_aversion,
                        'p_droughts': 1 / self.p_droughts[:-1], 
                        'total_profits': total_profits,
                        'total_profits_adaptation': total_profits_adaptation,
                        'profits_no_event': profits_no_event,
                        'profits_no_event_adaptation': profits_no_event_adaptation,
                        'risk_perception': self.risk_perception, 
                        'total_annual_costs': total_annual_costs,
                        'adaptation_costs': annual_cost, 
                        'adapted': self.adapted[:,adaptation_type], 
                        'time_adapted' : self.time_adapted[:,adaptation_type], 
                        'T': self.decision_horizon, 
                        'r': self.r_time, 
                        }

        # Determine EU of adaptation or doing nothing            
        EU_do_nothing = decision_module.calcEU_do_nothing(**decision_params)
        EU_adapt = decision_module.calcEU_adapt(**decision_params) 
        
        # Check output for missing data (if something went wrong in calculating EU)
        assert(EU_do_nothing != -1).any or (EU_adapt != -1).any()

        # Compare EU of adapting vs. not adapting for those who have not adapted yet 
        EU_adaptation_decision = (EU_adapt[self.adapted[:,adaptation_type] == 0] > EU_do_nothing[self.adapted[:,adaptation_type] == 0])

        # Initialize a boolean array of the same size as the original with all False values
        EU_adapt_mask = np.zeros_like(self.adapted[:,adaptation_type], dtype=bool)

        # Place EU_stay_adapt_bool values into the correct positions of expanded_mask
        EU_adapt_mask[self.adapted[:,adaptation_type] == 0] = EU_adaptation_decision

        # Compare EU of adapting vs neighbors that have adapted.
        adapt_due_to_neighbor = self.compare_neighbor_EUT(
            EU_do_nothing = EU_do_nothing, 
            EU_adapt = EU_adapt, 
            adaptation_type = adaptation_type,
            expenditure_cap = self.expenditure_cap,
            total_annual_costs = total_annual_costs,
            profits_no_event = profits_no_event)

        adaptation_mask = np.logical_or(adapt_due_to_neighbor, EU_adapt_mask)

        # For wells, also check whether the well can reach the groundwater 
        if adaptation_type == 1:
            well_depth = 30
            well_reaches_groundwater = self.groundwater_depth < well_depth
            # If the well doesnt reach the groundwater, let the farmer not adapt 
            adaptation_mask = well_reaches_groundwater * adaptation_mask

        # Change the adaptation status of wells (1) to 1 if mask is true 
        self.adapted[adaptation_mask, adaptation_type] = 1

        # Set the timer for people that are adapting to 0 
        self.time_adapted[adaptation_mask, adaptation_type] = 0
        # Update timer for next year
        self.time_adapted[self.time_adapted[:,adaptation_type] != -1,adaptation_type] += 1

        # Either the neighbor or the general algorithm states the farmer will or wont adapt. 
        return adaptation_mask
        
    def adapt_drip_irrigation(self) -> None:
        adaptation_type = 0
         
        # Convert adaptation cost to annual cost based on loan duration and interest rate
        # Adaptation price is sprinkler/drip irrigation price per acre, use field size in acres times the price per acre 
        total_cost = self.field_size_per_farmer * 0.000247105 * np.full(self.n, self.get_value_per_farmer_from_region_id(self.drip_irrigation_price, self.model.current_time), dtype = np.float32)
        loan_duration = self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['loan_duration'] ## loan duration
        r_loan =  self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['interest_rate']  ## loan interest
        # Calculate annnual costs of one adaptation loan based on interest rate and loan duration
        annual_cost = total_cost * (r_loan *( 1+r_loan) ** loan_duration/ ((1+r_loan)**loan_duration -1))

        # Reset timer and adaptation status when lifespan of adaptation is exceeded 
        self.adapted[:,adaptation_type][self.time_adapted[:,adaptation_type] == self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['lifespan']] = 0
        self.time_adapted[:,adaptation_type][self.time_adapted[:,adaptation_type] == self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['lifespan']] = -1 
        
        # Calculate which farmers will adapt
        adaptation_mask = self.adapt_SEUT(adaptation_type, annual_cost, loan_duration)
        
        ## Print the percentage of adapted households 
        percentage_adapted = round(np.sum(self.adapted[:,0])/ len(self.adapted[:,0]) * 100, 2)
        print('Sprinkler irrigation farms',percentage_adapted,'(%)')

        ## Change the irrigation efficiency if the farmer has made the investment 
        self.irrigation_efficiency[adaptation_mask] = .85

        ## Add the added annual costs to the overal annual costs of all adaptations, do this only to first time adapters (costs are repeating)
        self.annual_costs_all_adaptations[adaptation_mask] += annual_cost[adaptation_mask]
        ## Reduce the wealth of the farmer by the annual cost of the adaptation if it has made an adaptation in the last 30 years 
        self.disposable_income[self.time_adapted[:,adaptation_type] != -1] -= annual_cost[self.time_adapted[:,adaptation_type] != -1]

    def adapt_irrigation_well(self) -> None:
        adaptation_type = 1

        # Convert adaptation + upkeep cost to annual cost based on loan duration and interest rate
        total_cost = np.full(self.n, self.get_value_per_farmer_from_region_id(self.well_price, self.model.current_time), dtype = np.float32)
        loan_duration = self.model.config['agent_settings']['expected_utility']['adaptation_well']['loan_duration'] 
        r_loan =  self.model.config['agent_settings']['expected_utility']['adaptation_well']['interest_rate'] 

        # Calculate annnual costs of adaptation loan based on interest rate and loan duration
        annual_cost = total_cost * (r_loan *( 1+r_loan) ** loan_duration/ ((1+r_loan)**loan_duration -1))

        # Reset timer and adaptation status when lifespan of adaptation is exceeded 
        self.adapted[:,adaptation_type][self.time_adapted[:,adaptation_type] == self.model.config['agent_settings']['expected_utility']['adaptation_well']['lifespan']] = 0
        self.time_adapted[:,adaptation_type][self.time_adapted[:,adaptation_type] == self.model.config['agent_settings']['expected_utility']['adaptation_well']['lifespan']] = -1 
        
        # Calculate which farmers will adapt
        adaptation_mask = self.adapt_SEUT(adaptation_type, annual_cost, loan_duration)
        
        ## Print the percentage of adapted households 
        percentage_adapted = round(np.sum(self.adapted[:,adaptation_type])/ len(self.adapted[:,adaptation_type]) * 100, 2)
        print('Irrigation well farms',percentage_adapted,'(%)')

        # ## Change the well status if the farmer has made the investment 
        self.irrigation_source[adaptation_mask] = self.irrigation_source_key['tubewell']
        
        ## Add the added annual costs to the overal annual costs of all adaptations, do this only to first time adapters (costs are repeating)
        self.annual_costs_all_adaptations[adaptation_mask] += annual_cost[adaptation_mask]
        ## Reduce the wealth of the farmer by the annual cost of the adaptation
        self.disposable_income[self.time_adapted[:,adaptation_type] != -1] -= annual_cost[self.time_adapted[:,adaptation_type] != -1]

    def compare_neighbor_EUT(self, EU_do_nothing, EU_adapt, adaptation_type, expenditure_cap, total_annual_costs, profits_no_event):
        # Now check whether neighbors have adapted 
        nbits = 19
        # Check whether farmers have a adaptation 
        has_adaptation = (self.adapted[:,adaptation_type] == 1)
        invest_in_adaptation = np.zeros(self.n, dtype=np.bool_)
        for crop_option in np.unique(self.crops, axis=0):
            farmers_with_crop_option = np.where((self.crops==crop_option[None, ...]).all(axis=1))[0]

            # Local enumeration for each unique crop_option
            local_indices = np.arange(len(farmers_with_crop_option))

            # Create boolean masks for adapted and not adapted farmers
            farmers_adapted = self.adapted[farmers_with_crop_option, adaptation_type] == 1
            farmers_not_adapted = ~farmers_adapted

            # Using the conditions directly on the farmers_with_crop_option to get global indices
            filtered_indices_adapted = local_indices[farmers_adapted]
            filtered_indices_not_adapted = local_indices[farmers_not_adapted]
            # Cast the indices to the right data type 
            filtered_indices_adapted = filtered_indices_adapted.astype(np.int64)
            filtered_indices_not_adapted = filtered_indices_not_adapted.astype(np.int64)


            if filtered_indices_not_adapted.size > 0 and filtered_indices_adapted.size > 0:
                neighbors_with_adaptation = find_neighbors(
                    self.locations[farmers_with_crop_option],
                    radius=5_000,
                    n_neighbor=10,
                    bits=nbits,
                    minx=self.model.bounds[0],
                    maxx=self.model.bounds[1],
                    miny=self.model.bounds[2],
                    maxy=self.model.bounds[3],
                    search_ids=filtered_indices_not_adapted,
                    search_target_ids=filtered_indices_adapted
                )

                invest_in_adaptation_numba = self.invest_numba(
                    neighbors_with_adaptation = neighbors_with_adaptation,
                    farmers_without_adaptation = filtered_indices_not_adapted,
                    EU_do_nothing = EU_do_nothing,
                    EU_adapt = EU_adapt,
                    adapted = self.adapted[:,adaptation_type],
                    n = self.n,
                    expenditure_cap = expenditure_cap, 
                    total_annual_costs = total_annual_costs, 
                    profits_no_event = profits_no_event)
                
                invest_in_adaptation[invest_in_adaptation_numba] = True

        return invest_in_adaptation
    
    @staticmethod
    @njit(cache=True)
    def invest_numba(neighbors_with_adaptation: np.ndarray, 
                     farmers_without_adaptation: np.ndarray, 
                     EU_do_nothing: np.ndarray, 
                     EU_adapt: np.ndarray,
                     adapted: np.ndarray,
                     n: int,
                     expenditure_cap: float, 
                     total_annual_costs: np.ndarray, 
                     profits_no_event: np.ndarray,
                     ):
        
        invest_in_adaptation = np.zeros(n, dtype=np.bool_)
        neighbor_nan_value = np.iinfo(neighbors_with_adaptation.dtype).max
        for i, farmer_idx in enumerate(farmers_without_adaptation):
            # See whether the farmer can afford it
            if (profits_no_event[farmer_idx] * expenditure_cap > total_annual_costs[farmer_idx]):
                # EU of farmer if they would adapt 
                EU_farmer = EU_adapt[farmer_idx]
                farmer_neighbors_with_adaptation = neighbors_with_adaptation[i]
                farmer_neighbors_with_adaptation = farmer_neighbors_with_adaptation[farmer_neighbors_with_adaptation != neighbor_nan_value]
                if farmer_neighbors_with_adaptation.size > 0:
                    # EU of neighbor with adaptation 
                    EU_neighbor = EU_do_nothing[farmer_neighbors_with_adaptation]
                    # Make sure that it is only neighbors that have adapted 
                    EU_neighbor = EU_neighbor[adapted[farmer_neighbors_with_adaptation] == 1]
                    if EU_neighbor.size > 0:  # Check if EU_neighbor is not empty
                        mean_neighbors = np.mean(EU_neighbor)
                        if mean_neighbors > EU_farmer:
                            invest_in_adaptation[farmer_idx] = True

        return invest_in_adaptation

    def get_value_per_farmer_from_region_id(self, data, time) -> np.ndarray:
        index = data[0].get(time)
        unique_region_ids, inv = np.unique(self.region_id, return_inverse=True)
        values = np.full_like(unique_region_ids, np.nan, dtype=np.float32)
        for i, region_id in enumerate(unique_region_ids):
            values[i] = data[1][region_id][index]
        return values[inv]

    @staticmethod
    @njit(cache=True)
    def field_size_per_farmer_numba(field_indices_by_farmer: np.ndarray, field_indices: np.ndarray, cell_area: np.ndarray) -> np.ndarray:
        """Gets the field size for each farmer.

        Args:
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  
            cell_area: Subarray of cell_area.

        Returns:
            field_size_per_farmer: Field size for each farmer in m2.
        """
        field_size_per_farmer = np.zeros(field_indices_by_farmer.shape[0], dtype=np.float32)
        for farmer in range(field_indices_by_farmer.shape[0]):
            for field in get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer):
                field_size_per_farmer[farmer] += cell_area[field]
        return field_size_per_farmer

    @property
    def field_size_per_farmer(self) -> np.ndarray:
        """Gets the field size for each farmer.
  
        Returns:
            field_size_per_farmer: Field size for each farmer in m2.
        """
        return self.field_size_per_farmer_numba(
            self.field_indices_by_farmer,
            self.field_indices,
            self.var.cellArea.get() if self.model.use_gpu else self.var.cellArea
        )

    def expenses_and_income(self):
        self.disposable_income += self.daily_non_farm_income
        self.disposable_income -= self.daily_expenses_per_capita * self.household_size
        self.disposable_income[self.disposable_income < 0] = 0  # for now, weassume that farmers cannot go into debt

    def upkeep_assets(self):
        has_well = np.isin(self.irrigation_source, [self.irrigation_source_key['well'], self.irrigation_source_key['tubewell']])
        well_upkeep_price_per_m2 = self.get_value_per_farmer_from_region_id(self.well_upkeep_price_per_m2, self.model.current_time)

        self.disposable_income -= well_upkeep_price_per_m2 * self.field_size_per_farmer * has_well

    @staticmethod
    @njit
    def switch_crops_numba(ids, crops, neighbours, profits_last_harvest) -> None:
        """Switches crops for each farmer."""
        nodata_value_neighbors = np.iinfo(neighbours.dtype).max
        for i, farmer_idx in enumerate(ids):
            profit_last_harvest = profits_last_harvest[farmer_idx]
            neighbor_farmers = neighbours[i]
            neighbor_farmers = neighbor_farmers[neighbor_farmers != nodata_value_neighbors]  # delete farmers without neighbors
            if neighbor_farmers.size == 0:  # no neighbors
                continue

            neighbor_profits = profits_last_harvest[neighbor_farmers]
            neighbor_with_max_profit = np.argmax(neighbor_profits)
            if neighbor_profits[neighbor_with_max_profit] > profit_last_harvest:
                crops[farmer_idx] = crops[neighbor_farmers[neighbor_with_max_profit]]

    def switch_crops(self):
        """Switches crops for each farmer."""
        for farmer_class in np.unique(self.farmer_class):
            ids = np.where(self.farmer_class == farmer_class)[0]
            neighbors = find_neighbors(
                self.locations,
                radius=1_000,
                n_neighbor=3,
                bits=19,
                minx=self.model.bounds[0],
                maxx=self.model.bounds[1],
                miny=self.model.bounds[2],
                maxy=self.model.bounds[3],
                search_ids=ids,
                search_target_ids=ids
            )
            self.switch_crops_numba(ids, self.crops, neighbors, self.latest_profits[:, 0])

    def step(self) -> None:
        """
        This function is called at the beginning of each timestep.

        Then, farmers harvest and plant crops.
        """
        month = self.model.current_time.month
        if month in (6, 7, 8, 9, 10):
            self.current_season_idx = 0  # season #1
            if month == 6 and self.model.current_time.day == 1:
                self.is_first_day_of_season = True
            else:
                self.is_first_day_of_season = False
        elif month in (11, 12, 1, 2):
            self.current_season_idx = 1  # season #2
            if month == 11 and self.model.current_time.day == 1:
                self.is_first_day_of_season = True
            else:
                self.is_first_day_of_season = False
        elif month in (3, 4, 5):
            self.current_season_idx = 2  # season #3
            if month == 3 and self.model.current_time.day == 1:
                self.is_first_day_of_season = True
            else:
                self.is_first_day_of_season = False
        else:
            raise ValueError(f"Invalid month: {month}")
        
        self.harvest()
        self.plant()
        self.expenses_and_income()
        self.precipitation_sum()

        # monthly actions 
        if self.model.current_time.day == 1: 
            self.SPEI_sum()
            
        ## yearly actions 
        if self.model.current_time.month == 1 and self.model.current_time.day == 1:
            # relation test SPEI 
            # self.check_market_prices()

            self.farmer_is_in_command_area = self.is_in_command_area(self.n, self.var.reservoir_command_areas, self.field_indices, self.field_indices_by_farmer)
            # for now class is only dependent on being in a command area or not
            self.farmer_class = self.farmer_is_in_command_area.copy()
            
            # Set to 0 if channel abstraction is bigger than reservoir and groundwater, 1 for reservoir, 2 for groundwater 
            self.farmer_class[(self.yearly_abstraction_m3_by_farmer[:,0] > self.yearly_abstraction_m3_by_farmer[:,1]) & 
                              (self.yearly_abstraction_m3_by_farmer[:,0] > self.yearly_abstraction_m3_by_farmer[:,2])] = 0 
            self.farmer_class[(self.yearly_abstraction_m3_by_farmer[:,1] > self.yearly_abstraction_m3_by_farmer[:,0]) & 
                              (self.yearly_abstraction_m3_by_farmer[:,1] > self.yearly_abstraction_m3_by_farmer[:,2])] = 1 
            self.farmer_class[(self.yearly_abstraction_m3_by_farmer[:,2] > self.yearly_abstraction_m3_by_farmer[:,0]) & 
                              (self.yearly_abstraction_m3_by_farmer[:,2] > self.yearly_abstraction_m3_by_farmer[:,1])] = 2 
            
            # Set to 3 for precipitation if there is no abstraction 
            self.farmer_class[self.yearly_abstraction_m3_by_farmer[:,3] == 0] = 3 
            
            # Categorize water use based on the abstraction of the farmer. These limits could be better updated. Currently the above, relative, system works better
            # 0 is surface water / channel-dependent, 1 is reservoir-dependent, 2 is groundwater-dependent, 3 is rainwater-dependent
            for i in range(3):
                self.water_use[:,i] = np.where(self.yearly_abstraction_m3_by_farmer[:, i] == 0, 0, 
                                            np.where(self.yearly_abstraction_m3_by_farmer[:, i] < 0.25, 1,
                                                     np.where(self.yearly_abstraction_m3_by_farmer[:, i] < 0.5, 2, 3)))

            self.water_use[:,3] = np.where(self.yearly_abstraction_m3_by_farmer[:, 3] == 0, 0, 1)

            
            # check if current year is a leap year
            days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365
            has_access_to_water_all_year = self.n_water_accessible_days >= 365
            self.n_water_accessible_years[has_access_to_water_all_year] += 1
            self.n_water_accessible_days[~has_access_to_water_all_year] = 0
            self.n_water_accessible_days[:] = 0 # reset water accessible days
            
            self.upkeep_assets()
            # Save all profits, damages and rainfall for farmer estimations 
            # Can only be done if there has been a harvest of any sort 
            if not np.all(self.total_crop_age == 0):
                self.store_long_term_damages_profits_rain()
            else: 
                print("No harvests occurred yet, no yield - probability relation saved this year ")

            # Alternative scenarios: 'sprinkler'
            if self.model.scenario not in ['spinup', 'noadaptation', 'base']:
                # Convert the probability to yield ratio regardless of adaptation
                self.switch_crops()
                # These adaptations can only be done if there is a yield-probability relation 
                if not np.all(self.farmer_yield_probability_relation == 0): 
                    self.convert_probability_to_yield_ratio()
                    self.adapt_irrigation_well()   
                    self.adapt_drip_irrigation() 
                else:
                    raise AssertionError("Cannot adapt without yield - probability relation")


            self.wealth += self.disposable_income * 0.05

            print(np.mean(self.wealth))

            # reset disposable income and profits
            self.disposable_income[:] = 0
        
        
        # if self.model.current_timestep == 100:
        #     self.add_agent(indices=(np.array([310, 309]), np.array([69, 69])))
        # if self.model.current_timestep == 105:
        #     self.remove_agent(farmer_idx=1000)

    def remove_agents(self, farmer_indices: list[int]):
        if farmer_indices.size > 0:
            farmer_indices = np.sort(farmer_indices)[::-1]
            for idx in farmer_indices:
                self.remove_agent(idx)

    def remove_agent(self, farmer_idx: int) -> np.ndarray:
        last_farmer_HRUs = get_farmer_HRUs(self.field_indices, self.field_indices_by_farmer, self.n-1)
        last_farmer_field_size = self.field_size_per_farmer[self.n-1]
        for name, values in self.agent_attributes_meta.items():
            # get agent attribute
            attribute = getattr(self, name[1:])
            # move data of last agent to the agent that is to be removed, effectively removing that agent.
            attribute[farmer_idx] = attribute[self.n-1]
            # set value for last agent (which was now moved) to nodata
            attribute[self.n - 1] = values['nodata']

        # disown the farmer.
        HRUs_farmer_to_be_removed = get_farmer_HRUs(self.field_indices, self.field_indices_by_farmer, farmer_idx)
        self.var.land_owners[HRUs_farmer_to_be_removed] = -1

        # reduce number of agents
        self.n -= 1

        if self.n != farmer_idx:  # only move agent when agent was not the last agent
            HRUs_farmer_moved = get_farmer_HRUs(self.field_indices, self.field_indices_by_farmer, self.n)
            self.var.land_owners[HRUs_farmer_moved] = farmer_idx
  
        # TODO: Speed up field index updating.
        self.update_field_indices()
        if self.n == farmer_idx:
            assert get_farmer_HRUs(self.field_indices, self.field_indices_by_farmer, farmer_idx).size == 0
        else:
            assert np.array_equal(
                np.sort(last_farmer_HRUs),
                np.sort(get_farmer_HRUs(self.field_indices, self.field_indices_by_farmer, farmer_idx))
            )
            assert math.isclose(last_farmer_field_size, self.field_size_per_farmer[farmer_idx], abs_tol=1)

        for attr in self.agent_attributes:
            assert attr.startswith('_')
            assert getattr(self, attr[1:]).shape[0] == self.n
            assert np.array_equal(getattr(self, attr)[self.n], self.agent_attributes_meta[attr]['nodata'], equal_nan=True)

        assert (self.var.land_owners[HRUs_farmer_to_be_removed] == -1).all()
        return HRUs_farmer_to_be_removed

    def add_agent(self, indices):
        """This function can be used to add new farmers."""
        for attr in self.agent_attributes:
            assert attr.startswith('_')
            assert getattr(self, attr[1:]).shape[0] == self.n
            assert np.array_equal(getattr(self, attr)[self.n], self.agent_attributes_meta[attr]['nodata'], equal_nan=True)
  
        HRU = self.model.data.split(indices)
        self.var.land_owners[HRU] = self.n

        self.n += 1  # increment number of agents

        pixels = np.column_stack(indices)[:,[1, 0]]
        agent_location = np.mean(pixels_to_coords(pixels + .5, self.var.gt), axis=0)  # +.5 to use center of pixels

        # TODO: Speed up field index updating.
        self.update_field_indices()
  
        self.locations[self.n-1] = agent_location
        self.elevation[self.n-1] = self.elevation_map.sample_coords(np.expand_dims(agent_location, axis=0))
        self.region_id[self.n-1] = self.subdistrict_map.sample_coords(np.expand_dims(agent_location, axis=0))
        self.crops[self.n-1] = 1
        self.irrigated[self.n-1] = False
        self.wealth[self.n-1] = 0
        self.irrigation_efficiency[self.n-1] = False
        self.n_water_accessible_days[self.n-1] = 0
        self.n_water_accessible_years[self.n-1] = 0
        self.channel_abstraction_m3_by_farmer[self.n-1] = 0
        self.groundwater_abstraction_m3_by_farmer[self.n-1] = 0
        self.reservoir_abstraction_m3_by_farmer[self.n-1] = 0
        self.latest_profits[self.n-1] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        self.latest_potential_profits[self.n-1] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        for attr in self.agent_attributes:
            assert getattr(self, attr[1:]).shape[0] == self.n
            if "nodatacheck" not in self.agent_attributes_meta[attr] or self.agent_attributes_meta[attr]['nodatacheck'] is True:
                assert not np.array_equal(getattr(self, attr)[self.n-1], self.agent_attributes_meta[attr]['nodata'], equal_nan=True)

    def flood(self, flood_depth, crs, gt):
        transformer = Transformer.from_crs("epsg:4326", crs)
        x, y = transformer.transform(self.locations[:, 1], self.locations[:, 0])

        coordinates = np.column_stack((x, y))

        map_extent = (gt[0], gt[0] + gt[1] * flood_depth.shape[1], gt[3] + gt[5] * flood_depth.shape[0], gt[3])
        agents_in_map_extent = (coordinates[:, 0] >= map_extent[0]) & (coordinates[:, 0] <= map_extent[1]) & (coordinates[:, 1] >= map_extent[2]) & (coordinates[:, 1] <= map_extent[3])
        
        coordinates_in_extent = coordinates[agents_in_map_extent]

        flood_depth_per_agent = sample_from_map(flood_depth, coordinates_in_extent, gt)

        has_flooded = flood_depth_per_agent > 0

        self.flooded[agents_in_map_extent] = has_flooded

        # import matplotlib.pyplot as plt
        # plt.scatter(coordinates_in_extent[has_flooded][:, 0], coordinates_in_extent[has_flooded][:, 1], c='red')
        # plt.scatter(coordinates_in_extent[~has_flooded][:, 0], coordinates_in_extent[~has_flooded][:, 1], c='blue')
        # plt.show()