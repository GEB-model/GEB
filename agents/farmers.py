# -*- coding: utf-8 -*-
import os
import math
from datetime import date
import cftime
import json
import random
import calendar
from scipy.stats import genextreme

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
from honeybees.agents import AgentBaseClass
from honeybees.library.raster import pixels_to_coords, sample_from_map
from honeybees.library.neighbors import find_neighbors

from data import load_crop_prices, load_cultivation_costs, load_crop_factors, load_crop_names, load_inflation_rates, load_lending_rates, load_well_prices, load_sprinkler_prices

## Import the DecisionModule class from the other file 
from agents.decision_module import DecisionModule


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
        "elevation_map",
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
        "yearly_precipitation_HRU",
        "SPEI_map",
    ]
    agent_attributes = [
        "_locations",
        "_tehsil",
        "_elevation",
        "_crops",
        "_irrigation_source",
        "_household_size",
        "_disposable_income",
        "_loan_amount",
        "_loan_interest",
        "_loan_duration",
        "_loan_end_year",
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
        "_per_harvest_precipitation",
        "_per_harvest_SPEI",
        "_yearly_precipitation",
        "_yearly_SPEI_probability",
        "_yearly_SPEI",
        "_monthly_SPEI",
        "_latest_potential_profits",
        "_groundwater_depth",
        "_profit",
        "_farmer_is_in_command_area",
        "_farmer_class",
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
    ]
    __slots__.extend(agent_attributes)

    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents
        self.sample = [2000, 5500, 10000]
        self.var = model.data.HRU
        self.redundancy = reduncancy

        self.crop_names = load_crop_names()
        self.growth_length, self.crop_stage_lengths, self.crop_factors, self.crop_yield_factors, self.reference_yield = load_crop_factors()
        self.cultivation_costs = load_cultivation_costs()
        
        ## Set parameters required for drought event perception, risk perception and SEUT 
        self.previous_month = 0
        self.moving_average_threshold = self.model.config['agent_settings']['expected_utility']['drought_risk_calculations']['event_perception']['moving_average_threshold']
        self.absolute_threshold = self.model.config['agent_settings']['expected_utility']['drought_risk_calculations']['event_perception']['absolute_threshold']
       
        # Assign risk aversion sigma, time discounting preferences, expendature cap 
        self.sigma = self.model.config['agent_settings']['expected_utility']['decisions']['risk_aversion']
        self.r_time = self.model.config['agent_settings']['expected_utility']['decisions']['time_discounting']
        self.expenditure_cap = self.model.config['agent_settings']['expected_utility']['decisions']['expenditure_cap']

        # Set costs for adaptations 
        self.inflation_rate = load_inflation_rates('India')
        self.lending_rate = load_lending_rates('India')
        self.well_price, self.well_upkeep_price_per_m2 = load_well_prices(self, self.inflation_rate)
        self.sprinkler_price = load_sprinkler_prices(self, self.inflation_rate)
        self.well_investment_time_years = 10

        self.elevation_map = ArrayReader(
            fp=os.path.join(self.model.config['general']['input_folder'], 'landsurface', 'topo', 'subelv.tif'),
            bounds=self.model.bounds
        )
        self.elevation_grid = self.model.data.grid.compress(ArrayReader(
            fp=os.path.join(self.model.config['general']['input_folder'], 'landsurface', 'topo', 'elv.tif'),
            bounds=self.model.bounds
        ).get_data_array())

        self.SPEI_map = NetCDFReader(
            fp=os.path.join(self.model.config['general']['input_folder'], 'drought', 'SPEI', 'spei12.nc'),
            varname= 'spei',
            bounds=self.model.bounds,
            latname='lat',
            lonname='lon',
            timename= 'time'
        )
        with open(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'irrigation_sources.json')) as f:
            self.irrigation_source_key = json.load(f)

        # load map of all subdistricts
        self.subdistrict_map = ArrayReader(
            fp=os.path.join(self.model.config['general']['input_folder'], 'areamaps', 'tehsils.tif'),
            bounds=self.model.bounds
        )
        # load dictionary that maps subdistricts to state names
        with open(os.path.join(self.model.config['general']['input_folder'], 'areamaps', 'subdistrict2state.json'), 'r') as f:
            subdistrict2state = json.load(f)
            subdistrict2state = {int(subdistrict): state for subdistrict, state in subdistrict2state.items()}
            # assert that all subdistricts keys are integers
            assert all([isinstance(subdistrict, int) for subdistrict in subdistrict2state.keys()])
            # make sure all keys are consecutive integers starting at 0
            assert min(subdistrict2state.keys()) == 0
            assert max(subdistrict2state.keys()) == len(subdistrict2state) - 1
            # load unique states
            self.states = list(set(subdistrict2state.values()))
            # create numpy array mapping subdistricts to states
            state2int = {state: i for i, state in enumerate(self.states)}
            self.subdistrict2state = np.zeros(len(subdistrict2state), dtype=np.int32)
            for subdistrict, state in subdistrict2state.items():
                self.subdistrict2state[subdistrict] = state2int[state]
        
        self.crop_prices = load_crop_prices(state2int, self.inflation_rate)

        self.agent_attributes_meta = {
            "_locations": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan]
            },
            "_tehsil": {
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
            "_loan_amount": {
                "dtype": np.float32,
                "nodata": np.nan,
                "nodatacheck": False
            },
            "_profit": {
                "dtype": np.float32,
                "nodata": np.nan,
                "nodatacheck": False
            },
            "_loan_interest": {
                "dtype": np.float32,
                "nodata": np.nan,
                "nodatacheck": False
            },
            "_loan_end_year": {
                "dtype": np.int32,
                "nodata": -1,
                "nodatacheck": False
            },
            "_loan_duration": {
                "dtype": np.int32,
                "nodata": -1,
                "nodatacheck": False
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
            "_latest_profits": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan],
            },
            "_yearly_profits": {
                "dtype": np.float32,
                "nodata": [np.nan] * (self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'] + 1),
            },
            "_yearly_yield_ratio": {
                "dtype": np.float32,
                "nodata": [np.nan] * (self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'] + 1),
            },
            "_total_crop_age": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan],
            },
            "_per_harvest_yield_ratio": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan],
            },
            "_per_harvest_precipitation": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan],
            },
            "_per_harvest_SPEI": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan],
            },
            "_yearly_precipitation": {
                "dtype": np.float32,
                "nodata": [np.nan] * (self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'] + 1),
            },
            "_yearly_SPEI_probability": {
                "dtype": np.float32,
                "nodata": [np.nan] * (self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'] + 1),
            },
            "_yearly_SPEI": {
                "dtype": np.float32,
                "nodata": [np.nan] * (self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'] + 1),
            },
            "_monthly_SPEI": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan,np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan]
            },
            "_latest_potential_profits": {
                "dtype": np.float32,
                "nodata": [np.nan, np.nan, np.nan],
            },
            "_farmer_class": {
                "dtype": np.int32,
                "nodata": -1,
            },
            "_farmer_is_in_command_area": {
                "dtype": np.bool,
                "nodata": False,
                "nodatacheck": False
            },
            "_flooded": {
                "dtype": np.bool,
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
                "nodata": [np.nan, np.nan, np.nan],
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
        }
        self.initiate_agents()

    @property
    def locations(self):
        return self._locations[:self.n]

    @locations.setter
    def locations(self, value):
        self._locations[:self.n] = value

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
    def loan_amount(self):
        return self._loan_amount[:self.n]

    @loan_amount.setter
    def loan_amount(self, value):
        self._loan_amount[:self.n] = value

    @property
    def loan_interest(self):
        return self._loan_interest[:self.n]

    @loan_interest.setter
    def loan_interest(self, value):
        self._loan_interest[:self.n] = value

    @property
    def loan_duration(self):
        return self._loan_duration[:self.n]

    @loan_duration.setter
    def loan_duration(self, value):
        self._loan_duration[:self.n] = value

    @property
    def loan_interest(self):
        return self._loan_interest[:self.n]

    @loan_interest.setter
    def loan_interest(self, value):
        self._loan_interest[:self.n] = value

    @property
    def loan_interest(self):
        return self._loan_interest[:self.n]

    @loan_interest.setter
    def loan_interest(self, value):
        self._loan_interest[:self.n] = value

    @property
    def loan_end_year(self):
        return self._loan_end_year[:self.n]

    @loan_end_year.setter
    def loan_end_year(self, value):
        self._loan_end_year[:self.n] = value

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
    def per_harvest_precipitation(self):
        return self._per_harvest_precipitation[:self.n]

    @per_harvest_precipitation.setter
    def per_harvest_precipitation(self, value):
        self._per_harvest_precipitation[:self.n] = value

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
    def yearly_SPEI(self):
        return self._yearly_SPEI[:self.n]

    @yearly_SPEI.setter
    def yearly_SPEI(self, value):
        self._yearly_SPEI[:self.n] = value

    @property
    def monthly_SPEI(self):
        return self._monthly_SPEI[:self.n]

    @monthly_SPEI.setter
    def monthly_SPEI(self, value):
        self._monthly_SPEI[:self.n] = value

    @property
    def yearly_precipitation(self):
        return self._yearly_precipitation[:self.n]

    @yearly_precipitation.setter
    def yearly_precipitation(self, value):
        self._yearly_precipitation[:self.n] = value

    @property
    def farmer_is_in_command_area(self):
        return self._farmer_is_in_command_area[:self.n]

    @farmer_is_in_command_area.setter
    def farmer_is_in_command_area(self, value):
        self._farmer_is_in_command_area[:self.n] = value

    @property
    def farmer_class(self):
        return self._farmer_class[:self.n]

    @farmer_class.setter
    def farmer_class(self, value):
        self._farmer_class[:self.n] = value

    @property
    def tehsil(self):
        return self._tehsil[:self.n]

    @tehsil.setter
    def tehsil(self, value):
        self._tehsil[:self.n] = value

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
    
    @staticmethod
    def is_in_command_area(n, command_areas, field_indices, field_indices_by_farmer):
        farmer_is_in_command_area = np.zeros(n, dtype=np.bool)
        for farmer_i in range(n):
            farmer_fields = get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer_i)
            for field in farmer_fields:
                command_area = command_areas[field]
                if command_area != -1:
                    farmer_is_in_command_area[farmer_i] = True
                    break
        return farmer_is_in_command_area

    def initiate_agents(self) -> None:
        """Calls functions to initialize all agent attributes, including their locations. Then, crops are initially planted. 
        """
        # If initial conditions based on spinup period need to be loaded, load them. Otherwise, generate them.
        if self.model.load_initial_data:
            for attribute in self.agent_attributes:
                fp = os.path.join(self.model.initial_conditions_folder, f"farmers.{attribute}.npz")
                values = np.load(fp)['data']
                setattr(self, attribute, values)
            self.n = np.where(np.isnan(self._locations[:,0]))[0][0]  # first value where location is not defined (np.nan)
            self.max_n = self._locations.shape[0]
        else:
            farms = self.model.data.farms

            # Get number of farmers and maximum number of farmers that could be in the entire model run based on the redundancy.
            self.n = np.unique(farms[farms != -1]).size
            self.max_n = math.ceil(self.n * (1 + self.redundancy))
            assert self.max_n < 4294967295 # max value of uint32, consider replacing with uint64

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

            # Load the tehsil code of each farmer.
            self.tehsil = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'tehsil_code.npy'))

            # Find the elevation of each farmer on the map based on the coordinates of the farmer as calculated before.
            self.elevation = self.elevation_map.sample_coords(self.locations)

            # Initiate adaptation status
            self.adapted = np.zeros((self.n, 2), dtype=np.int32) # 2d array which indicates if agents have adapted. 0 = not adapted, 1 adapted. Column 0 = sprinkler, 1 = well
            self.time_adapted = np.zeros((self.n, 2), dtype=np.int32) # array containing the time each agent has been paying off their dry flood proofing investment loan. Column 0 = sprinkler, 1 = well


            # Load the crops planted for each farmer in the Kharif, Rabi and Summer seasons.
            self.crops[:, 0] = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'kharif crop.npy'))
            self.crops[:, 1] = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'rabi crop.npy'))
            self.crops[:, 2] = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'summer crop.npy'))
            assert self.crops.max() < len(self.crop_names)

            # Set irrigation source 
            self.irrigation_source = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'irrigation_source.npy'))
            # set the adaptation of wells to 1 if farmers have well 
            self.adapted[:,1][np.isin(self.irrigation_source, [self.irrigation_source_key['well'], self.irrigation_source_key['tubewell']])] = 1
            
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

            self.yearly_profits = np.zeros((self.n, self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'] + 1), dtype=np.float32)
            self.yearly_yield_ratio = np.zeros((self.n, self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'] + 1), dtype=np.float32)

            # 0 = kharif age, 1 = rabi age, 2 = summer age, 3 = total growth time 
            self.total_crop_age = np.zeros((self.n, 3), dtype=np.float32)
            # 0 = kharif yield_ratio, 1 = rabi yield_ratio, 2 = summer yield_ratio
            self.per_harvest_yield_ratio = np.zeros((self.n, 3), dtype=np.float32)
            self.per_harvest_precipitation = np.zeros((self.n, 3), dtype=np.float32)
            self.per_harvest_SPEI = np.zeros((self.n, 3), dtype=np.float32)

            self.yearly_precipitation = np.zeros((self.n, self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'] + 1), dtype=np.float32)
            self.yearly_SPEI = np.zeros((self.n, self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'] + 1), dtype=np.float32)
            self.yearly_SPEI_probability = np.zeros((self.n, self.model.config['agent_settings']['expected_utility']['decisions']['decision_horizon'] + 1), dtype=np.float32)
            # Create 2d array in which the past years rainfall is stored 
            self.yearly_precipitation_HRU = np.zeros((250, self.var.land_owners[self.var.land_owners != -1].size), dtype=np.float32)
            self.monthly_SPEI = np.zeros((self.n, 10), dtype=np.float32)
            self.disposable_income[:] = 0
            self.household_size = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'household size.npy'))
            self.daily_non_farm_income = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'daily non farm income family.npy'))
            self.daily_expenses_per_capita = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'daily consumption per capita.npy'))
            self.flooded[:] = False

            ## Base initial wealth on x days of daily expenses
            self.wealth = self.daily_expenses_per_capita * self.household_size * ((365/12)*18)
        
            ## Risk perception variables 
            self.risk_perception = np.full(self.n, self.model.config['agent_settings']['expected_utility']['drought_risk_calculations']['risk_perception']['min'], dtype = np.float32)
            self.drought_timer = np.full(self.n, 99, dtype = np.float32)
            self.drought_loss = np.zeros((self.n, 3), dtype=np.float32)

            # Set irrigation efficiency to 70% for all farmers.
            self.irrigation_efficiency[:] = .70
            # Set the people who already have more van 90% irrigation efficiency to already adapted for the sprinkler irrgation adaptation  
            self.adapted[:,0][self.irrigation_efficiency >= .90] = 1
            
            # Initiate array that tracks the overall yearly costs for all adaptations 
            self.annual_costs_all_adaptations = np.zeros(self.n, dtype=np.float32) 

            self.farmer_class[:] = 0  # 0 is precipitation-dependent, 1 is surface water-dependent, 2 is reservoir-dependent, 3 is groundwater-dependent

            self.farmer_is_in_command_area[:] = False

            ## Load in the GEV_parameters, calculated from the extreme value distribution of the SPEI timeseries, and load in the original SPEI data 
            parameter_names = ['shape', 'loc', 'scale']
            self.GEV_parameters = np.zeros((len(self.locations), len(parameter_names)))

            for i, varname in enumerate(parameter_names):
                GEV_map = NetCDFReader(
                    fp=os.path.join(self.model.config['general']['input_folder'], 'drought', 'SPEI', 'GEV_SPEI_12_1901_2021_Bhima.nc'),
                    varname=varname,
                    bounds=self.model.bounds,
                    latname='lat',
                    lonname='lon',
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

        print("use farmer_is_in_command_area")
  
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
            self.crop_yield_factors['KyT'],
        )
        assert not np.isnan(yield_ratio).any()
        return yield_ratio

    def precipitation_sum(self) -> None:
        # Store precipitation per day in a 2d array of past days. After new day is stored, shift the daily precipitation a day back. 
        self.yearly_precipitation_HRU[1:,: ] = self.yearly_precipitation_HRU[0:-1, :]
        self.yearly_precipitation_HRU[0,:] = self.model.data.HRU.Precipitation[self.var.land_owners != -1]

        # To create unique groups based on water abstraction, sum the abstractions. Later used to make groups. 
        self.yearly_abstraction_m3_by_farmer[:,0] += self.channel_abstraction_m3_by_farmer
        self.yearly_abstraction_m3_by_farmer[:,1] += self.reservoir_abstraction_m3_by_farmer
        self.yearly_abstraction_m3_by_farmer[:,2] += self.groundwater_abstraction_m3_by_farmer
        self.yearly_abstraction_m3_by_farmer[:,3] += self.channel_abstraction_m3_by_farmer + self.reservoir_abstraction_m3_by_farmer + self.groundwater_abstraction_m3_by_farmer
    
    def SPEI_sum(self) -> None:
         ## SPEI is recorded on the 16th, except in february, then it is on the 15th 
        if self.model.current_time.month == 2:
            day_of_the_month = 15
        else:
            day_of_the_month = 16
        
        # Store SPEI per month in a 2d array of past days. After new day is stored, shift the daily precipitation a day back. 
        self.monthly_SPEI[:,1:] = self.monthly_SPEI[:,0:-1]
        self.monthly_SPEI[:,0] = self.SPEI_map.sample_coords(self.locations, cftime.DatetimeGregorian(self.model.current_time.year, self.model.current_time.month, day_of_the_month, 0, 0, 0, 0, has_year_zero=False))

    def save_profit_water_rain(self, harvesting_farmers: np.ndarray, profit: np.ndarray, potential_profit: np.ndarray) -> None:
        """Saves the current harvest for harvesting farmers in a 2-dimensional array. The first dimension is the different farmers, while the second dimension are the previous harvests. 
        First the previous harvests are moved by 1 column (dropping old harvests) to make room for the new harvest. Then, the new harvest is placed in the array.
        
        Also saves the profits, yield ratio and precipitation. It does this per harvest and resets & saves the total at the start of each year.
  
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

        ## Variable that sums harvests within a year. After a year has passed the total is moved to the second column (next function). Correct for field size 
        self.yearly_profits[harvesting_farmers, 0] += self.latest_profits[harvesting_farmers, 0] / self.field_size_per_farmer[harvesting_farmers]
        
        
    def store_long_term_damages_profits_rain(self) -> None:
        """Saves the yearly profit, rainfall and yield ratios and stores them. Then calculates for each unique farmer type what their yearly mean is. 
        """
        # shift all columns one column further, the last falls off. 
        self.yearly_profits[:, 1:] = self.yearly_profits[:, 0:-1]
        # Set the first column to 0
        self.yearly_profits[:, 0] = 0

        # calculate the average yield ratio for that year 
        total_planted_time = self.total_crop_age[:,0] + self.total_crop_age[:,1] + self.total_crop_age[:,2]
        
        ## Mask where total_planted time is 0 (no planting was done)
        total_planted_time = np.ma.masked_where(total_planted_time == 0, total_planted_time)

        # add the yield ratio proportional to the total planting time 
        self.yearly_yield_ratio[:, 0] = (
            self.total_crop_age[:, 0] / total_planted_time * self.per_harvest_yield_ratio[:, 0] + #kharif yield ratio 
            self.total_crop_age[:, 1] / total_planted_time * self.per_harvest_yield_ratio[:, 1] + #rabi yield ratio 
            self.total_crop_age[:, 2] / total_planted_time * self.per_harvest_yield_ratio[:, 2]   #summer yield ratio 
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
        
        # Save the average yearly probability in the yearly precipitation probability by summing and dividing through the planting seasons
        nonzero_count = np.count_nonzero(seasonal_SPEI_probability, axis=1)
        nr_planting_seasons = np.where(nonzero_count == 0, 1, nonzero_count)
        self.yearly_SPEI_probability[:, 0] = np.sum(seasonal_SPEI_probability, axis=1) / nr_planting_seasons
        self.yearly_SPEI[:,0] = np.sum(self.per_harvest_SPEI, axis=1) / nr_planting_seasons

        self.per_harvest_precipitation, self.per_harvest_SPEI  = 0, 0

        # shift all columns one column further, the last falls off. 
        self.yearly_yield_ratio[:, 1:] = self.yearly_yield_ratio[:, 0:-1]
        # Set the first column to 0
        self.yearly_yield_ratio[:, 0] = 0
        # set the total crop age and per crop yield ratios to 0 
        self.total_crop_age[:, :] = 0
        self.per_harvest_yield_ratio[:, :] = 0

        #shift all columns one column further, the last falls off. 
        self.yearly_precipitation[:, 1:] = self.yearly_precipitation[:, 0:-1]
        #Reset the first column of the HRU and yearly precipitation columns to 0 
        self.yearly_precipitation[:, 0] = 0

        #shift all columns one column further, the last falls off. 
        self.yearly_SPEI_probability[:, 1:] = self.yearly_SPEI_probability[:, 0:-1]
        #Reset the first column of the HRU and yearly precipitation columns to 0 
        self.yearly_SPEI_probability[:, 0] = 0

        #shift all columns one column further, the last falls off. 
        self.yearly_SPEI[:, 1:] = self.yearly_SPEI[:, 0:-1]
        #Reset the first column of the HRU and yearly precipitation columns to 0 
        self.yearly_SPEI[:, 0] = 0
        
        # Now convert the yearly values per individual farmer to unique farmer types
        # First check if these variables already exist, otherwise make them. The total years they use is equal to the decision horizon. 
        unique_yearly_profits = locals().get('unique_yearly_profits', 
                                             np.empty((0, self.decision_horizon[0])))
        unique_yearly_yield_ratio = locals().get('unique_yearly_yield_ratio', 
                                                 np.empty((0, self.decision_horizon[0])))
        unique_yearly_precipitation = locals().get('unique_yearly_precipitation', 
                                                   np.empty((0, self.decision_horizon[0])))
        unique_SPEI_probability = locals().get('unique_SPEI_probability', 
                                                   np.empty((0, self.decision_horizon[0])))
        unique_yearly_SPEI = locals().get('unique_yearly_SPEI', 
                                                   np.empty((0, self.decision_horizon[0])))

        # Make a new variable that has crop combination and the farmer class (what type of water they use), as to make unique groups based on this
        # Should perhaps make this into a model wide variable 
        crop_irrigation_groups = np.hstack((self.crops, self.farmer_class.reshape(-1, 1)))

        # Save the profits, rainfall and water shortage for each farmer type. To do: change this to a vectorized operation, determine and add more homogene farmer groups 
        for crop_combination in np.unique(crop_irrigation_groups, axis=0):

            unique_farmer_groups = np.where((crop_irrigation_groups==crop_combination[None, ...]).all(axis=1))[0]
            
            # Calculate averages of profits, yield ratio, precipitation, and probability for the same farmer groups
            average_profits = np.mean(self.yearly_profits[unique_farmer_groups, 1:], axis=0)
            average_yield_ratio = np.mean(self.yearly_yield_ratio[unique_farmer_groups, 1:], axis=0)
            average_precipitation = np.mean(self.yearly_precipitation[unique_farmer_groups, 1:], axis=0)
            average_probability = np.mean(self.yearly_SPEI_probability[unique_farmer_groups, 1:], axis=0)
            average_SPEI = np.mean(self.yearly_SPEI[unique_farmer_groups, 1:], axis=0)
            
            # Prepend the averages to respective arrays
            unique_yearly_profits = np.vstack((average_profits, unique_yearly_profits))
            unique_yearly_yield_ratio = np.vstack((average_yield_ratio, unique_yearly_yield_ratio))
            unique_yearly_precipitation = np.vstack((average_precipitation, unique_yearly_precipitation))
            unique_SPEI_probability = np.vstack((average_probability, unique_SPEI_probability))
            unique_yearly_SPEI = np.vstack((average_SPEI, unique_yearly_SPEI))


        # Make sure that it is max X values that are saved
        unique_yearly_profits = unique_yearly_profits[:,:self.decision_horizon[0]]
        unique_yearly_yield_ratio = unique_yearly_yield_ratio[:,:self.decision_horizon[0]]
        unique_yearly_precipitation = unique_yearly_precipitation[:,:self.decision_horizon[0]]
        unique_SPEI_probability = unique_SPEI_probability[:,:self.decision_horizon[0]]
        unique_yearly_SPEI = unique_yearly_SPEI[:,:self.decision_horizon[0]]

        ## Mask the minimum and the maximum value 
        arrays = [unique_yearly_profits, unique_yearly_yield_ratio, unique_yearly_precipitation, unique_SPEI_probability, unique_yearly_SPEI]
        masked_arrays = []

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

        unique_yearly_profits_mask, unique_yearly_yield_ratio_mask, unique_yearly_precipitation_mask, unique_SPEI_probability_mask, unique_yearly_SPEI_mask = masked_arrays
        

        # Mask rows that consist only of 0s --> no relation possible 
        mask = np.any((unique_yearly_profits_mask != 0), axis=1) & np.any((unique_yearly_yield_ratio_mask != 0), axis=1) & np.any((unique_yearly_SPEI_mask != 0), axis=1)
        masked_unique_yearly_profits = unique_yearly_profits_mask[mask]
        masked_unique_yearly_yield_ratio = unique_yearly_yield_ratio_mask[mask]
        masked_unique_yearly_precipitation = unique_yearly_precipitation_mask[mask]
        masked_unique_SPEI_probability= unique_SPEI_probability_mask[mask]
        masked_unique_yearly_SPEI= unique_yearly_SPEI_mask[mask]


        ## Clear previous plots from the cache
        plt.clf()

        # For testing, only show the first five unique farmer groups 
        trendline_count = 0

        ## Create empty lists to append the relations to 
        farmer_profit_yield_relation = []
        farmer_profit_rainfall_relation = []

        fig, (ax1, ax2) = plt.subplots(1, 2)
        # Determine the relation between the remaining rows. TO DO: vectorize further and change names to reflect current variables being used 
        for i, (row1, row2, row3, row4) in enumerate(zip(masked_unique_yearly_yield_ratio, masked_unique_yearly_profits, masked_unique_SPEI_probability), 1):
            ## Determine the relation between yield ratio and profit for all farmer types
            # Calculate the coefficients and save them  
            coefficients_profit_yield = np.polyfit(row1, row3, 1)
            poly_profit_yield = np.poly1d(coefficients_profit_yield)
            farmer_profit_yield_relation.append(poly_profit_yield)
            
            ## Visualize the scatterplot and trendline 
            ax1.scatter(row1, row3)
            ax1.plot(row1, poly_profit_yield(row1), label= f'Farmer group {i}')
            
            ## Determine the relation between profit and precipitation for all farmer types 
            # Calculate the coefficients and save them 
            coefficients_profit_rainfall = np.polyfit(row3, row2, 1)
            poly_profit_rainfall = np.poly1d(coefficients_profit_rainfall)
            farmer_profit_rainfall_relation.append(poly_profit_rainfall)

            ## Visualize the scatterplot and trendline 
            ax2.scatter(row2, row4)
            ax2.plot(row2, poly_profit_rainfall(row2), label= f'Farmer group {i}')

            # For testing, only calculate the first 5 farmer types 
            trendline_count += 1
            if trendline_count == 15:
                break

        ax1.set_xlabel('Yield ratio')
        ax1.set_ylabel('SPEI probability')

        ax2.set_xlabel('SPEI probability')
        ax2.set_ylabel('Profit')

        plt.tight_layout()

        plt.legend()
        plt.show()

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
            if self.model.args.use_gpu:
                harvested_area = harvested_area.get()
            harvested_crops = self.var.crop_map[harvest]
            max_yield_per_crop = np.take(self.reference_yield, harvested_crops)
      
            year = self.model.current_time.year
            month = self.model.current_time.month
            crop_price_index = self.crop_prices[0][date(year, month, 1)]
            crop_prices_per_state = self.crop_prices[1][crop_price_index]
            assert not np.isnan(crop_prices_per_state).any()
            
            tehsil_per_field = self.tehsil[harvesting_farmer_fields]
            state_per_field = np.take(self.subdistrict2state, tehsil_per_field)

            harvesting_farmers = np.unique(harvesting_farmer_fields)

            # get potential crop profit per farmer
            crop_yield_gr = harvested_area * yield_ratio * max_yield_per_crop
            assert (crop_yield_gr >= 0).all()
            crop_prices_per_field = crop_prices_per_state[state_per_field, harvested_crops]
            profit = crop_yield_gr * crop_prices_per_field
            assert (profit >= 0).all()
            
            self.profit = np.bincount(harvesting_farmer_fields, weights=profit, minlength=self.n)

            ## Set the current crop age
            crop_age = self.var.crop_age_days_map[harvest]
            total_crop_age = np.bincount(harvesting_farmer_fields, weights = crop_age, minlength=self.n) / np.bincount(harvesting_farmer_fields, minlength=self.n)

            ## Convert the yield_ratio per field to the average yield ratio per farmer 
            yield_ratio_agent = np.bincount(harvesting_farmer_fields, weights = yield_ratio, minlength=self.n) / np.bincount(harvesting_farmer_fields, minlength=self.n)
            
            # Convert the precipitation per HRU to precipitation per farmer -- first sum it over the days the crop has grown
            cum_prec_HRU_latest_harvest = np.sum(self.yearly_precipitation_HRU[:crop_age[0], :], axis=0)
            precipitation_agent = np.bincount(self.var.land_owners[self.var.land_owners != -1], weights=cum_prec_HRU_latest_harvest, minlength=self.n) / np.bincount(self.var.land_owners[self.var.land_owners != -1], minlength=self.n)
            # Add  precipitation to the yearly variable, this is reset after each year: 
            self.yearly_precipitation[harvesting_farmers, 0] += precipitation_agent[harvesting_farmers]

            # Take the mean of the growing months and change the sign to fit the GEV distribution 
            cum_SPEI_latest_harvest = np.mean(self.monthly_SPEI[harvesting_farmers, :int((crop_age[0] / 30))], axis=1) * -1

            ## Add the yield ratio, precipitation and the crop age to the array corresponding to the current season. Precipitation is already converted to daily rainfall
            if self.current_season_idx == 0:
                self.total_crop_age[harvesting_farmers, 0] = total_crop_age[harvesting_farmers]
                self.per_harvest_yield_ratio[harvesting_farmers, 0] = yield_ratio_agent[harvesting_farmers]
                self.per_harvest_precipitation[harvesting_farmers, 0] = precipitation_agent[harvesting_farmers] / total_crop_age[harvesting_farmers] # average daily precipitation
                self.per_harvest_SPEI[harvesting_farmers,0] = cum_SPEI_latest_harvest
            elif self.current_season_idx == 1:
                self.total_crop_age[harvesting_farmers, 1] = total_crop_age[harvesting_farmers]
                self.per_harvest_yield_ratio[harvesting_farmers, 1] = yield_ratio_agent[harvesting_farmers]
                self.per_harvest_precipitation[harvesting_farmers, 1] = precipitation_agent[harvesting_farmers] / total_crop_age[harvesting_farmers] # average daily precipitation
                self.per_harvest_SPEI[harvesting_farmers,1] = cum_SPEI_latest_harvest
            elif self.current_season_idx == 2:
                self.total_crop_age[harvesting_farmers, 2] = total_crop_age[harvesting_farmers]
                self.per_harvest_yield_ratio[harvesting_farmers, 2] = yield_ratio_agent[harvesting_farmers]
                self.per_harvest_precipitation[harvesting_farmers, 2] = precipitation_agent[harvesting_farmers] / total_crop_age[harvesting_farmers] # average daily precipitation
                self.per_harvest_SPEI[harvesting_farmers,2] = cum_SPEI_latest_harvest
           
            # get potential crop profit per farmer
            potential_crop_yield = harvested_area * max_yield_per_crop
            potential_profit = potential_crop_yield * crop_prices_per_field
            potential_profit = np.bincount(harvesting_farmer_fields, weights=potential_profit, minlength=self.n)
      
            self.save_profit_water_rain(harvesting_farmers, self.profit, potential_profit)
            
            self.drought_risk_perception(harvesting_farmers)
            
            ## After updating the drought risk perception, set the previous month for the next timestep as the current for this timestep
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
        year = self.model.current_time.year
        month = self.model.current_time.month
        if month < 7:  # Agricultural year in India runs from June to July.
            agricultural_year = f"{year-1}-{year}"
        else:
            agricultural_year = f"{year}-{year+1}"
        year_index = self.cultivation_costs[0][agricultural_year]
        cultivation_cost_per_crop = self.cultivation_costs[1][year_index]
        plant_map, farmers_selling_land = self.plant_numba(
            n=self.n,
            season_idx=self.current_season_idx,
            is_first_day_of_season=self.is_first_day_of_season,
            growth_length=self.growth_length,
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
                and not self.model.args.scenario == 'spinup'  # farmers can only go out of business when not in spinup scenario
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
        ## convert the harvesting farmers array to a full length array with true values  
        harvesting_farmers_long = np.zeros(self.n, dtype=bool)
        harvesting_farmers_long[harvesting_farmers] = True

        ## Update the drought timer for every timestep (harvest) that it is activated, update is the difference between last and current timestep 
        months_passed = (self.model.current_time.month - self.previous_month) % 12
        self.drought_timer += (months_passed / 12) 
        
        ## Determine the loss between potential and actual profits of all recorded events in % for all the harvesting farmers 
        self.drought_loss[harvesting_farmers_long] = np.abs(((self.latest_profits[harvesting_farmers_long] - self.latest_potential_profits[harvesting_farmers_long]) / self.latest_potential_profits[harvesting_farmers_long]) * 100)
        
        ## Determine latest and past average loss percentages 
        drought_loss_latest = self.drought_loss[:,0]
        drought_loss_past = np.mean(self.drought_loss[:, 1:], axis=1)

        ## Farmers experience drought events if the loss is larger than that of the last few years, or very large (thresholds t.b.d.)
        experienced_drought_event = np.logical_or(drought_loss_past - drought_loss_latest >= self.moving_average_threshold, drought_loss_latest >= self.absolute_threshold)
        
        # Reset drought timer on locations that have harvesting farmers and that have experienced a drought event 
        self.drought_timer[np.logical_and(harvesting_farmers_long, experienced_drought_event)] = 0

        # Calculate the updated risk perception of all farmers 
        self.risk_perception = (self.risk_perc_max * 1.6 ** (self.risk_decr * self.drought_timer)) + self.risk_perc_min

        print('Risk perception mean = ',np.mean(self.risk_perception))


    def SEUT_irrigation_well(self) -> None:
        decision_module = DecisionModule(self)
        
        ## Set the probabilities for future droughts -- will be changed later 
        p_droughts = np.array([1000, 500, 250, 100, 50, 25, 10, 5, 2])
        
        ## How much of the yield is lost during drought
        p_droughts_loss = np.array([300, 200, 150, 100, 60, 50, 40, 30, 10]) / 100
        # Create random damages in 9d arrays to provide input for the decision module 
        # Firstcreate an empty array with the specified dimensions
        expected_damages = np.zeros((len(p_droughts), self.n))
        expected_damages_adapt = np.zeros((len(p_droughts), self.n))

        # loop over the first dimension of the array
        for i, p in enumerate(p_droughts_loss):
            reduced_damage_factor = self.model.config['agent_settings']['expected_utility']['adaptation_well']['reduced_damage']
            ## Yearly yield (rupees) per hectare, based on proportional cultivation of major crops and crop prices in maharastra
            average_price_per_m2 = 130000 / 10000
            # generate a random 1D array using the given parameters
            expected_damages[i] = self.field_size_per_farmer * average_price_per_m2 * p 

            expected_damages_adapt[i] = self.field_size_per_farmer * average_price_per_m2 * p * reduced_damage_factor 


        # Convert adaptation + upkeep cost to annual cost based on loan duration and interest rate
        total_cost = np.full(self.n, self.well_price[self.model.current_time.year], dtype = np.float32)
        loan_duration = self.model.config['agent_settings']['expected_utility']['adaptation_well']['loan_duration'] 
        r_loan =  self.model.config['agent_settings']['expected_utility']['adaptation_well']['interest_rate'] 

        # Calculate annnual costs of adaptation loan based on interest rate and loan duration
        annual_cost = total_cost * (r_loan *( 1+r_loan) ** loan_duration/ ((1+r_loan)**loan_duration -1))
        
        # Determine the total yearly costs of farmers if they would adapt this cycle 
        total_annual_costs = annual_cost + self.annual_costs_all_adaptations

        # Reset timer and adaptation status when lifespan of adaptation is exceeded 
        self.adapted[:,1][self.time_adapted[:,1] == self.model.config['agent_settings']['expected_utility']['adaptation_well']['lifespan']] = 0
        self.time_adapted[:,1][self.time_adapted[:,1] == self.model.config['agent_settings']['expected_utility']['adaptation_well']['lifespan']] = -1 

        decision_params = {'loan_duration': self.model.config['agent_settings']['expected_utility']['adaptation_well']['loan_duration'],  
                        'expenditure_cap': self.expenditure_cap, 
                        #'lifespan' : self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['lifespan'], ## Not in model yet, can be added 
                        'n_agents':  self.n, 
                        'sigma': self.sigma,
                        'wealth': self.wealth, 
                        'income': self.disposable_income, 
                        'p_droughts': 1 / p_droughts, 
                        'risk_perception': self.risk_perception, 
                        'expected_damages': expected_damages, 
                        'expected_damages_adapt': expected_damages_adapt,
                        'total_annual_costs': total_annual_costs,
                        'adaptation_costs': annual_cost, 
                        'adapted': self.adapted[:,1], 
                        'time_adapted' : self.time_adapted[:,1], 
                        'T': self.decision_horizon, 
                        'r': self.r_time, 
                        }
        
        nbits = 19

        # Check whether farmers have a well 
        #has_well = self.adapted[:,1] == 1
        
        # Determine EU of adaptation or doing nothing            
        EU_do_nothing = decision_module.calcEU_do_nothing(**decision_params)
        EU_adapt = decision_module.calcEU_adapt(**decision_params) 

        # Check output for missing data (if something went wrong in calculating EU)
        assert(EU_do_nothing != -1).any or (EU_adapt != -1).any()
                
        # Compare EU of adapting vs. not adapting
        EU_stay_adapt_bool = (EU_adapt >= EU_do_nothing)

        self.adapted[:,1] = EU_stay_adapt_bool * 1

        # Check which people will adapt and whether they made this decision for the first time
        pos_first_time_adapted = (self.adapted[:,1] == 1) * (self.time_adapted[:,1] == -1)
        # Set the timer for these people to 0
        self.time_adapted[:,1][pos_first_time_adapted] = 0
        # Update timer for next year
        self.time_adapted[:,1][self.time_adapted[:,1] != -1] += 1

        # Check for missing data
        assert (self.adapted[:,1] != -1).any()

        ## Print the percentage of adapted households 
        percentage_adapted = round(np.sum(self.adapted[:,1])/ len(self.adapted[:,1]) * 100, 2)
        print('Irrigation well farms',percentage_adapted,'(%)')

        # ## Change the well status if the farmer has made the investment 
        invest_in_well = EU_stay_adapt_bool * 1
        self.irrigation_source[invest_in_well] = self.irrigation_source_key['tubewell']
        
        ## Add the added annual costs to the overal annual costs of all adaptations, do this only to first time adapters (costs are repeating)
        self.annual_costs_all_adaptations[pos_first_time_adapted] += annual_cost[pos_first_time_adapted]

        ## Reduce the wealth of the farmer by the annual cost of the adaptation if it has made an adaptation in the last 30 years 
        self.wealth[self.time_adapted[:,1] != -1] -= annual_cost
    
        # for crop_option in np.unique(self.crops, axis=0):
            
        #     farmers_with_crop_option = np.where((self.crops==crop_option[None, ...]).all(axis=1))[0]
            
        #     farmers_with_well_crop_option = np.where(has_well[farmers_with_crop_option] == True)[0]
        #     farmers_without_well = np.where(has_well[farmers_with_crop_option] == False)[0]

        #     if farmers_without_well.size > 0 and farmers_with_well_crop_option.size > 0:

    @staticmethod
    @njit(cache=True)
    def invest_numba(
        n: int,
        year: int,
        farmers_without_well: np.ndarray,
        neighbors_with_well: np.ndarray,
        latest_profits: np.ndarray,
        latest_potential_profits: np.ndarray,
        farm_size_m2: np.ndarray,
        loan_interest: np.ndarray,
        loan_amount: np.ndarray,
        loan_duration: np.ndarray,
        loan_end_year: np.ndarray,
        well_price: float,
        well_upkeep_price_per_m2: float,
        well_investment_time_years: int,
        interest_rate: float,
        disposable_income: np.ndarray,
        disposable_income_threshold: int=0,
        intention_behaviour_gap: int=1
    ):  
        """Determines whether a farmer without a well invests in an irrigation well and takes a loan. Each farmer has
        a probability of investing in a well based on the following factors:
            - The farmer's latest profit ratio (latest profit / latest potential profit)
            - The profit ratio of the farmer's neighbors with wells (average of neighbors' latest profit / latest potential profit)
            - Whether farmer disposable income is sufficient to pay for a loan for a well as well as the yearly upkeep cost
            - The farmer's disposable income relative to the minimum disposable income required to invest in a well
  
        Args:
            n: number of farmers
            year: the current year
            farmers_without_well: farmers currently without a well
            has_well: farmers currently with a will (collary of farmers_without_well)
            neighbors_with_well: the neighbors of each farmer without a well that does have a well
            latest_profits: the latest profits of each farmer
            latest_potential_profits: the latest potential profits of each farmer
            farm_size_m2: the size of each farmer's farm in m2
            loan_interest: current interest rate of each farmer's loan
            loan_amount: current loan amount for each farmer
            loan_duration: total loan duration for each farmer
            loan_end_year: the year the loan ends for each farmer
            well_price: the price of a well
            well_upkeep_price_per_m2: the yearly upkeep price of a well per m2
            well_investment_time_years: the time farmers consider when investing in a well / time that well is expected to be in operation
            interest_rate: current interest rate
            disposable_income: disposable income of each farmer
            disposable_income_threshold: the minimum disposable income required to invest in a well after considering total well cost
            intention_behaviour_gap: the ratio of action to intention. For example, if the intention_behaviour_gap is 1, then the farmer will always implement the measure when intended, if the intention_behaviour_gap is 0 the measure will never be implemented.
        """
        
        invest_in_well = np.zeros(n, dtype=np.bool_)
        neighbor_nan_value = np.iinfo(neighbors_with_well.dtype).max
        for i, farmer_idx in enumerate(farmers_without_well):
            latest_profit = latest_profits[farmer_idx, 0]
            latest_potential_profit = latest_potential_profits[farmer_idx, 0]

            farmer_neighbors_with_well = neighbors_with_well[i]
            farmer_neighbors_with_well = farmer_neighbors_with_well[farmer_neighbors_with_well != neighbor_nan_value]
            if farmer_neighbors_with_well.size > 0:
                latest_profits_neighbors = latest_profits[farmer_neighbors_with_well, 0]
                latest_potential_profits_neighbors = latest_potential_profits[farmer_neighbors_with_well, 0]
                profit_ratio_neighbors = latest_profits_neighbors / latest_potential_profits_neighbors
                profit_ratio_neighbors = profit_ratio_neighbors[~np.isnan(profit_ratio_neighbors)]
                
                if profit_ratio_neighbors.size == 0:
                    continue
                
                profit_ratio_neighbors = np.mean(profit_ratio_neighbors)
                profit_with_neighbor_efficiency = latest_potential_profit * profit_ratio_neighbors

                potential_benefit = profit_with_neighbor_efficiency - latest_profit
                potential_benefit_over_investment_time = potential_benefit * well_investment_time_years
                total_cost_over_investment_time = well_price + well_upkeep_price_per_m2 * farm_size_m2[farmer_idx] * well_investment_time_years

                if potential_benefit_over_investment_time <= total_cost_over_investment_time:
                    continue
                    
                # assume linear loan
                money_left_for_investment = disposable_income[farmer_idx] - well_upkeep_price_per_m2 * farm_size_m2[farmer_idx]
                well_loan_duration = 30
                loan_size = well_price
                yearly_payment = loan_size / well_loan_duration + loan_size * interest_rate

                if money_left_for_investment < yearly_payment + disposable_income_threshold:
                    continue

                if random.random() < intention_behaviour_gap:
                    invest_in_well[farmer_idx] = True
                    
                    loan_interest[farmer_idx] = interest_rate
                    loan_amount[farmer_idx] = loan_size
                    loan_duration[farmer_idx] = well_loan_duration
                    loan_end_year[farmer_idx] = year + well_loan_duration

                    # set profits to nan, just to make sure that only profits with new adaptation measure are used.
                    latest_profits[farmer_idx] = np.nan
                    latest_potential_profits[farmer_idx] = np.nan

            # farmer_fields = get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer_idx)
            # channel_storage_farmer_m3 = 0
            # in_reservoir_command_area = False
            # for field in farmer_fields:
            #     grid_cell = HRU_to_grid[field]
            #     channel_storage_farmer_m3 += channel_storage_m3[grid_cell]
            #     if reservoir_command_areas[field] >= 0:
            #         in_reservoir_command_area = True

            # if channel_storage_farmer_m3 > 100 and not surface_irrigated[farmer_idx]:
            #     surface_irrigated[farmer_idx] = True

            # if in_reservoir_command_area and not surface_irrigated[farmer_idx]:
            #     surface_irrigated[farmer_idx] = True

        return invest_in_well

    def invest_in_irrigation_well(self) -> None:
        nbits = 19

        has_well = np.isin(self.irrigation_source, [self.irrigation_source_key['well'], self.irrigation_source_key['tubewell']])

        for crop_option in np.unique(self.crops, axis=0):
            
            farmers_with_crop_option = np.where((self.crops==crop_option[None, ...]).all(axis=1))[0]
            
            farmers_with_well_crop_option = np.where(has_well[farmers_with_crop_option] == True)[0]
            farmers_without_well = np.where(has_well[farmers_with_crop_option] == False)[0]
            farmers_without_well_indices = farmers_with_crop_option[farmers_without_well]

            if farmers_without_well.size > 0 and farmers_with_well_crop_option.size > 0:
                neighbors_with_well = find_neighbors(
                    self.locations[farmers_with_crop_option],
                    radius=5_000,
                    n_neighbor=10,
                    bits=nbits,
                    minx=self.model.bounds[0],
                    maxx=self.model.bounds[1],
                    miny=self.model.bounds[2],
                    maxy=self.model.bounds[3],
                    search_ids=farmers_without_well,
                    search_target_ids=farmers_with_well_crop_option
                )

                interest_rate = self.lending_rate[self.model.current_time.year]
                assert not np.isnan(interest_rate)
                invest_in_well = self.invest_numba(
                    n=self.n,
                    year=self.model.current_time.year,
                    farmers_without_well=farmers_without_well_indices,
                    neighbors_with_well=neighbors_with_well,
                    latest_profits=self.latest_profits,
                    latest_potential_profits=self.latest_potential_profits,
                    farm_size_m2=self.field_size_per_farmer,
                    loan_interest=self.loan_interest,
                    loan_amount=self.loan_amount,
                    loan_duration=self.loan_duration,
                    loan_end_year=self.loan_end_year,
                    well_price=self.well_price[self.model.current_time.year],
                    well_upkeep_price_per_m2=self.well_upkeep_price_per_m2[self.model.current_time.year],
                    well_investment_time_years=self.well_investment_time_years,
                    interest_rate=interest_rate,
                    disposable_income=self.disposable_income,
                    intention_behaviour_gap=self.model.config['agent_settings']['farmers']['well_implementation_intention_behaviour_gap']
                )
                self.irrigation_source[invest_in_well] = self.irrigation_source_key['tubewell']
    
    
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
            self.var.cellArea.get() if self.model.args.use_gpu else self.var.cellArea
        )

    def expenses_and_income(self):
        self.disposable_income += self.daily_non_farm_income
        self.disposable_income -= self.daily_expenses_per_capita * self.household_size
        self.disposable_income[self.disposable_income < 0] = 0  # for now, weassume that farmers cannot go into debt

    def upkeep_assets(self):
        has_well = np.isin(self.irrigation_source, [self.irrigation_source_key['well'], self.irrigation_source_key['tubewell']])
        self.disposable_income -= self.well_upkeep_price_per_m2[self.model.current_time.year] * self.field_size_per_farmer * has_well

    def make_loan_payment(self):
        has_loan = self.loan_amount > 0
        years_left = self.loan_end_year[has_loan] - self.model.current_time.year
        loan_payoff = self.loan_amount[has_loan] / years_left
        loan_interest = self.loan_amount[has_loan] * self.loan_interest[has_loan]
        loan_payment = loan_payoff + loan_interest

        self.loan_amount[has_loan] -= loan_payoff
        self.disposable_income[has_loan] -= loan_payment
        self.disposable_income[self.disposable_income < 0] = 0  # for now, we assume that farmers don't have negative disposable income
        

    # @staticmethod
    # @njit(cache=True)
    # def switch_crops(sugarcane_idx, n_water_accessible_years, crops, days_in_year) -> None:
    #     """Switches crops for each farmer.
    #     """
    #     assert (n_water_accessible_years <= days_in_year + 1).all() # make sure never higher than full year
    #     for farmer_idx in range(n_water_accessible_years.size):
    #         # each farmer with all-year access to water has a 20% probability of switching to sugarcane
    #         if n_water_accessible_years[farmer_idx] >= 3 and np.random.random() < .20:
    #             crops[farmer_idx] = sugarcane_idx

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
    

    def invest_in_sprinkler_irrigation(self) -> None:
        decision_module = DecisionModule(self)

        ## Set the probabilities for future droughts -- will be changed later 
        p_droughts = np.array([1000, 500, 250, 100, 50, 25, 10, 5, 2])
        
        ## How much of the yield is lost during drought
        p_droughts_loss = np.array([300, 200, 150, 100, 60, 50, 40, 30, 10]) / 100
        # Create random damages in 9d arrays to provide input for the decision module 
        # Firstcreate an empty array with the specified dimensions
        expected_damages = np.zeros((len(p_droughts), self.n))
        expected_damages_adapt = np.zeros((len(p_droughts), self.n))

        # loop over the first dimension of the array
        for i, p in enumerate(p_droughts_loss):
            reduced_damage_factor = 1 - self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['reduced_damage']
            ## Yearly yield (rupees) per hectare, based on proportional cultivation of major crops and crop prices in maharastra
            average_price_per_m2 = 130000 / 10000
            # generate a random 1D array using the given parameters
            expected_damages[i] = self.field_size_per_farmer * average_price_per_m2 * p 

            expected_damages_adapt[i] = self.field_size_per_farmer * average_price_per_m2 * p * reduced_damage_factor 

        # Convert adaptation cost to annual cost based on loan duration and interest rate
        ## Adaptation price is sprinkler/drip irrigation price per acre, use field size in acres times the price per acre 
        total_cost = self.field_size_per_farmer * 0.000247105 * self.sprinkler_price[self.model.current_time.year]
        loan_duration = self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['loan_duration'] ## loan duration
        r_loan =  self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['interest_rate']  ## loan interest

        # Calculate annnual costs of one adaptation loan based on interest rate and loan duration
        annual_cost = total_cost * (r_loan *( 1+r_loan) ** loan_duration/ ((1+r_loan)**loan_duration -1))
        # Determine the total yearly costs of farmers if they would adapt this cycle 
        total_annual_costs = annual_cost + self.annual_costs_all_adaptations

        # Reset timer and adaptation status when lifespan of adaptation is exceeded 
        self.adapted[:,0][self.time_adapted[:,0] == self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['lifespan']] = 0
        self.time_adapted[:,0][self.time_adapted[:,0] == self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['lifespan']] = -1 

        decision_params = {'loan_duration': self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['loan_duration'],  
                        'expenditure_cap': self.expenditure_cap, 
                        #'lifespan' : self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['lifespan'], ## Not in model yet, can be added 
                        'n_agents':  self.n, 
                        'sigma': self.sigma,
                        'wealth': self.wealth, 
                        'income': self.disposable_income, 
                        'p_droughts': 1 / p_droughts, 
                        'risk_perception': self.risk_perception, 
                        'expected_damages': expected_damages, 
                        'expected_damages_adapt': expected_damages_adapt,
                        'total_annual_costs': total_annual_costs,
                        'adaptation_costs': annual_cost, 
                        'adapted': self.adapted[:,0], 
                        'time_adapted' : self.time_adapted[:,0], 
                        'T': self.decision_horizon, 
                        'r': self.r_time, 
                        } 

        # Determine EU of adaptation or doing nothing            
        EU_do_nothing = decision_module.calcEU_do_nothing(**decision_params)
        EU_adapt = decision_module.calcEU_adapt(**decision_params) 
        
        # Check output for missing data (if something went wrong in calculating EU)
        assert(EU_do_nothing != -1).any or (EU_adapt != -1).any()

        # Only compare EU of adapting vs. not adapting
        EU_stay_adapt_bool = (EU_adapt >= EU_do_nothing)

        self.adapted[:,0] = EU_stay_adapt_bool * 1

        # Check which people will adapt and whether they made this decision for the first time
        pos_first_time_adapted = (self.adapted[:,0] == 1) * (self.time_adapted[:,0] == -1)
        # Set the timer for these people to 0
        self.time_adapted[:,0][pos_first_time_adapted] = 0
        # Update timer for next year
        self.time_adapted[:,0][self.time_adapted[:,0] != -1] += 1

        # Update the percentage of households implementing drip irrigation 
        # Check for missing data
        assert (self.adapted[:,0] != -1).any()

        ## Print the percentage of adapted households 
        percentage_adapted = round(np.sum(self.adapted[:,0])/ len(self.adapted[:,0]) * 100, 2)
        print('Sprinkler irrigation farms',percentage_adapted,'(%)')

        ## Change the irrigation efficiency if the farmer has made the investment 
        invest_in_sprinkler_irrigation = EU_stay_adapt_bool * 1
        self.irrigation_efficiency[invest_in_sprinkler_irrigation] = .90

        ## Add the added annual costs to the overal annual costs of all adaptations, do this only to first time adapters (costs are repeating)
        self.annual_costs_all_adaptations[pos_first_time_adapted] += annual_cost[pos_first_time_adapted]

        ## Reduce the wealth of the farmer by the annual cost of the adaptation if it has made an adaptation in the last 30 years 
        self.wealth[self.time_adapted[:,0] != -1] -= annual_cost

    def step(self) -> None:
        """
        This function is called at the beginning of each timestep.

        Then, farmers harvest and plant crops.
        """
        month = self.model.current_time.month
        if month in (6, 7, 8, 9, 10):
            self.current_season_idx = 0  # kharif
            if month == 6 and self.model.current_time.day == 1:
                self.is_first_day_of_season = True
            else:
                self.is_first_day_of_season = False
        elif month in (11, 12, 1, 2):
            self.current_season_idx = 1  # rabi
            if month == 11 and self.model.current_time.day == 1:
                self.is_first_day_of_season = True
            else:
                self.is_first_day_of_season = False
        elif month in (3, 4, 5):
            self.current_season_idx = 2  # summer
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

            self.farmer_is_in_command_area = self.is_in_command_area(self.n, self.var.reservoir_command_areas, self.field_indices, self.field_indices_by_farmer)
            # for now class is only dependent on being in a command area or not
            self.farmer_class = self.farmer_is_in_command_area.copy()
            # Set to 0 for precipitation if there is no abstraction 
            self.farmer_class[self.yearly_abstraction_m3_by_farmer[:,3] == 0] = 0 
            # Set to 1 if channel abstraction is bigger than reservoir and groundwater, 2 for reservoir, 3 for groundwater 
            self.farmer_class[(self.yearly_abstraction_m3_by_farmer[:,0] > self.yearly_abstraction_m3_by_farmer[:,1]) & 
                              (self.yearly_abstraction_m3_by_farmer[:,0] > self.yearly_abstraction_m3_by_farmer[:,2])] = 1 
            self.farmer_class[(self.yearly_abstraction_m3_by_farmer[:,1] > self.yearly_abstraction_m3_by_farmer[:,0]) & 
                              (self.yearly_abstraction_m3_by_farmer[:,1] > self.yearly_abstraction_m3_by_farmer[:,2])] = 2 
            self.farmer_class[(self.yearly_abstraction_m3_by_farmer[:,2] > self.yearly_abstraction_m3_by_farmer[:,0]) & 
                              (self.yearly_abstraction_m3_by_farmer[:,2] > self.yearly_abstraction_m3_by_farmer[:,1])] = 3 
            
            # check if current year is a leap year
            days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365
            has_access_to_water_all_year = self.n_water_accessible_days >= 365
            self.n_water_accessible_years[has_access_to_water_all_year] += 1
            self.n_water_accessible_days[~has_access_to_water_all_year] = 0
            self.n_water_accessible_days[:] = 0 # reset water accessible days
            
            if self.model.args.scenario not in ['spinup', 'noadaptation', 'noHI', 'base', 'noCC_base', 'noCC_noHI','sprinkler']:
                # self.switch_crops(self.crop_names["Sugarcane"], self.n_water_accessible_years, self.crops, days_in_year)
                self.switch_crops()
            
            self.upkeep_assets()
            self.make_loan_payment()
            ## Save all profits, damages and rainfall for farmer estimations 
            self.store_long_term_damages_profits_rain()
            
            if self.model.args.scenario not in ['spinup', 'noadaptation', 'noHI', 'base', 'noCC_base', 'noCC_noHI']:
                self.invest_in_irrigation_well()
            if self.model.args.scenario not in ['spinup', 'noadaptation', 'noHI', 'base', 'noCC_base', 'noCC_noHI']:
                self.invest_in_sprinkler_irrigation()
                # self.SEUT_irrigation_well()     

            self.wealth += self.disposable_income * 0.2

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
        self.tehsil[self.n-1] = self.subdistrict_map.sample_coords(np.expand_dims(agent_location, axis=0))
        self.crops[self.n-1] = 1
        self.irrigated[self.n-1] = False
        self.wealth[self.n-1] = 0
        self.irrigation_efficiency[self.n-1] = False
        self.n_water_accessible_days[self.n-1] = 0
        self.n_water_accessible_years[self.n-1] = 0
        self.channel_abstraction_m3_by_farmer[self.n-1] = 0
        self.groundwater_abstraction_m3_by_farmer[self.n-1] = 0
        self.reservoir_abstraction_m3_by_farmer[self.n-1] = 0
        self.latest_profits[self.n-1] = [np.nan, np.nan, np.nan]
        self.latest_potential_profits[self.n-1] = [np.nan, np.nan, np.nan]

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