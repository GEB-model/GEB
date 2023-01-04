# -*- coding: utf-8 -*-
import os
import math
from datetime import date
import json

import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass

from numba import njit

from honeybees.library.mapIO import ArrayReader
from honeybees.agents import AgentBaseClass
from honeybees.library.raster import pixels_to_coords
from honeybees.library.neighbors import find_neighbors

from data import load_crop_prices, load_cultivation_costs, load_crop_factors, load_crop_names, load_inflation_rates, load_lending_rates, load_well_prices

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
        "inflation"
    ]
    agent_attributes = [
        "_locations",
        "_tehsil",
        "_elevation",
        "_crops",
        "_irrigated",
        "_has_well",
        "_household_size",
        "_disposable_income",
        "_loan_amount",
        "_loan_interest",
        "_loan_duration",
        "_loan_end_year",
        "_daily_non_farm_income",
        "_daily_expenses_per_capita",
        "_irrigation_efficiency",
        "_n_water_limited_days",
        "_water_availability_by_farmer",
        "_channel_abstraction_m3_by_farmer",
        "_groundwater_abstraction_m3_by_farmer",
        "_reservoir_abstraction_m3_by_farmer",
        "_latest_profits",
        "_latest_potential_profits",
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
        
        self.inflation_rate = load_inflation_rates('India')
        self.lending_rate = load_lending_rates('India')
        self.well_price, self.well_upkeep_price_per_m2 = load_well_prices(self.inflation_rate)
        self.well_investment_time_years = 10

        self.elevation_map = ArrayReader(
            fp=os.path.join(self.model.config['general']['input_folder'], 'landsurface', 'topo', 'subelv.tif'),
            bounds=self.model.bounds
        )

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
            states = list(set(subdistrict2state.values()))
            # create numpy array mapping subdistricts to states
            state2int = {state: i for i, state in enumerate(states)}
            self.subdistrict2state = np.zeros(len(subdistrict2state), dtype=np.int32)
            for subdistrict, state in subdistrict2state.items():
                self.subdistrict2state[subdistrict] = state2int[state]
        
        self.crop_prices = load_crop_prices(state2int)

        self.agent_attributes_meta = {
            "_locations": {
                "nodata": [np.nan, np.nan]
            },
            "_tehsil": {
                "nodata": -1
            },
            "_elevation": {
                "nodata": np.nan
            },
            "_crops": {
                "nodata": [-1, -1, -1],
                "nodatacheck": False
            },
            "_irrigated": {
                "nodata": -1,
            },
            "_household_size": {
                "nodata": -1,
            },
            "_disposable_income": {
                "nodata": -1
            },
            "_loan_amount": {
                "nodata": -1,
                "nodatacheck": False
            },
            "_loan_interest": {
                "nodata": -1,
                "nodatacheck": False
            },
            "_loan_end_year": {
                "nodata": -1,
                "nodatacheck": False
            },
            "_loan_duration": {
                "nodata": -1,
                "nodatacheck": False
            },
            "_daily_non_farm_income": {
                "nodata": -1
            },
            "_daily_expenses_per_capita": {
                "nodata": -1
            },
            "_irrigation_efficiency": {
                "nodata": -1
            },
            "_n_water_limited_days": {
                "nodata": -1
            },
            "_has_well": {
                "nodata": False,
                "nodatacheck": False
            },
            "_water_availability_by_farmer": {
                "nodata": np.nan
            },
            "_channel_abstraction_m3_by_farmer": {
                "nodata": np.nan
            },
            "_groundwater_abstraction_m3_by_farmer": {
                "nodata": np.nan
            },
            "_reservoir_abstraction_m3_by_farmer": {
                "nodata": np.nan
            },
            "_latest_profits": {
                "nodata": [np.nan, np.nan, np.nan],
                # "nodatacheck": False
            },
            "_latest_potential_profits": {
                "nodata": [np.nan, np.nan, np.nan],
                # "nodatacheck": False
            },
        }
        self.initiate_agents()
        self.plant_initial()

    def initiate_agents(self) -> None:
        """Calls functions to initialize all agent attributes, including their locations. Then, crops are initially planted. 
        """
        # If initial conditions based on spinup period need to be loaded, load them. Otherwise, generate them.
        if self.model.load_initial_data:
            for attribute in self.agent_attributes:
                fp = os.path.join(self.model.initial_conditions_folder, f"farmers.{attribute}.npy")
                values = np.load(fp)
                setattr(self, attribute, values)
            self.n = np.where(np.isnan(self._locations[:,0]))[0][0]  # first value where location is not defined (np.nan)
            self.max_n = self._locations.shape[0]
            self.latest_profits.fill(np.nan)
            self.latest_potential_profits.fill(np.nan)
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
            pixels = np.zeros((self.n, 2))
            pixels[:,0] = np.round(np.bincount(farms[farms != -1], horizontal_index) / np.bincount(farms[farms != -1])).astype(int)
            pixels[:,1] = np.round(np.bincount(farms[farms != -1], vertical_index) / np.bincount(farms[farms != -1])).astype(int)
            self._locations = np.full((self.max_n, 2), np.nan, dtype=np.float32)
            self.locations = pixels_to_coords(pixels + .5, self.var.gt)

            # Load the tehsil code of each farmer.
            self._tehsil = np.full(self.max_n, -1, dtype=np.int32)
            self.tehsil = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'tehsil_code.npy'))

            # Find the elevation of each farmer on the map based on the coordinates of the farmer as calculated before.
            self._elevation = np.full(self.max_n, np.nan, dtype=np.float32)
            self.elevation = self.elevation_map.sample_coords(self.locations)
      
            # Load the crops planted for each farmer in the Kharif, Rabi and Summer seasons.
            self._crops = np.full((self.max_n, 3), -1, dtype=np.int32)  # kharif, rabi, summer
            self.crops[:, 0] = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'kharif crop.npy'))
            self.crops[:, 1] = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'rabi crop.npy'))
            self.crops[:, 2] = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'summer crop.npy'))
            assert self.crops.max() < len(self.crop_names)
      
            # Load the irrigation status of each farmer. This is a boolean array with three columns, one for each season.      
            irrigated = np.full((self.n, 3), -1, dtype=np.int8)  # kharif, rabi, summer
            irrigated[:, 0] = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'kharif irrigation.npy'))
            irrigated[:, 1] = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'rabi irrigation.npy'))
            irrigated[:, 2] = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'summer irrigation.npy'))
      
            # Assume that a farmer that is irrigated in one season, is also irrigated in the other seasons. Thus using any to get the irrigation status
            # for all seasons.
            self._irrigated = np.full(self.max_n, -1, dtype=np.int8)
            self.irrigated = irrigated.any(axis=1)

            # Load the irrigation type from file and assign all farmers with irrigation type 1 (well) a well.
            irrigation_type = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'irrigation type.npy'))
            self._has_well= np.full(self.max_n, 0, dtype=bool)
            self.has_well[:] = (irrigation_type == 1).any(axis=1)

            # Set irrigation efficiency to 70% for all farmers.
            self._irrigation_efficiency = np.full(self.max_n, -1, dtype=np.float32)
            self.irrigation_efficiency[:] = .70

            # Initiate a number of arrays with Nan, zero or -1 values for variables that will be used during the model run.
            self._latest_profits = np.full((self.max_n, 3), np.nan, dtype=np.float32)
            self._latest_potential_profits = np.full((self.max_n, 3), np.nan, dtype=np.float32)

            self._channel_abstraction_m3_by_farmer = np.full(self.max_n, np.nan, dtype=np.float32)
            self.channel_abstraction_m3_by_farmer[:] = 0
            self._reservoir_abstraction_m3_by_farmer = np.full(self.max_n, np.nan, dtype=np.float32)
            self.reservoir_abstraction_m3_by_farmer[:] = 0
            self._groundwater_abstraction_m3_by_farmer = np.full(self.max_n, np.nan, dtype=np.float32)
            self.groundwater_abstraction_m3_by_farmer[:] = 0
            self._water_availability_by_farmer = np.full(self.max_n, np.nan, dtype=np.float32)
            self.water_availability_by_farmer[:] = 0
            self._n_water_limited_days = np.full(self.max_n, -1, dtype=np.int32)
            self.n_water_limited_days[:] = 0

            self._disposable_income = np.full(self.max_n, -1, dtype=np.float32)
            self.disposable_income[:] = 0

            self._loan_amount = np.full(self.max_n, 0, dtype=np.float32)
            self._loan_interest = np.full(self.max_n, 0, dtype=np.float32)
            self._loan_duration = np.full(self.max_n, -1, dtype=np.int32)
            self._loan_end_year = np.full(self.max_n, -1, dtype=np.int32)

            self._household_size = np.full(self.max_n, -1, dtype=np.int32)
            self.household_size = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'household size.npy'))

            self._daily_non_farm_income = np.full(self.max_n, -1, dtype=np.float32)
            self.daily_non_farm_income = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'daily non farm income family.npy'))

            self._daily_expenses_per_capita = np.full(self.max_n, -1, dtype=np.float32)
            self.daily_expenses_per_capita = np.load(os.path.join(self.model.config['general']['input_folder'], 'agents', 'attributes', 'daily consumption per capita.npy'))

        self._field_indices_by_farmer = np.full((self.max_n, 2), -1, dtype=np.int32)
        self.update_field_indices()

        print(f'Loaded {self.n} farmer agents')

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
        water_limited_days: np.ndarray,
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
        return_fraction: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function is used to regulate the irrigation behavior of farmers. The farmers are "activated" by the given `activation_order` and each farmer can irrigate from the various water sources, given water is available and the farmers has the means to abstract water. The abstraction order is channel irrigation, reservoir irrigation, groundwater irrigation. 

        Args:
            activation_order: Order in which the agents are activated. Agents that are activated first get a first go at extracting water, leaving less water for other farmers.
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  
            water_limited_days: Current number of days where farmer has been water limited.
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
  
        for activated_farmer_index in range(activation_order.size):
            farmer = activation_order[activated_farmer_index]
            farmer_fields = get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer)
            if irrigation_efficiency[farmer]:
                efficiency = 0.8
            else:
                efficiency = 0.6
      
            if surface_irrigated[farmer] == 1 or well_irrigated[farmer] == 1:
                farmer_is_water_limited = False
                for field in farmer_fields:
                    if crop_map[field] != -1:
                        f_var = HRU_to_grid[field]
                        irrigation_water_demand_field = totalPotIrrConsumption[field] / efficiency

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
                            available_groundwater_cell_m = available_groundwater_m3[f_var] / cell_area[field]
                            groundwater_abstraction_cell_m = min(available_groundwater_cell_m, irrigation_water_demand_field)
                            groundwater_abstraction_cell_m3 = groundwater_abstraction_cell_m * cell_area[field]
                            groundwater_abstraction_m3[f_var] = groundwater_abstraction_cell_m3
                            available_groundwater_m3[f_var] -= groundwater_abstraction_cell_m3
                            water_withdrawal_m[field] += groundwater_abstraction_cell_m

                            groundwater_abstraction_m3_by_farmer[farmer] += groundwater_abstraction_cell_m3
              
                            irrigation_water_demand_field -= groundwater_abstraction_cell_m
              
                        assert irrigation_water_demand_field >= -1e15  # Make sure irrigation water demand is zero, or positive. Allow very small error.

                        if irrigation_water_demand_field > 1e-15:
                            farmer_is_water_limited = True

                    water_consumption_m[field] = water_withdrawal_m[field] * efficiency
                    irrigation_loss_m = water_withdrawal_m[field] - water_consumption_m[field]
                    returnFlowIrr_m[field] = irrigation_loss_m * return_fraction
                    addtoevapotrans_m[field] = irrigation_loss_m * (1 - return_fraction)
            else:
                farmer_is_water_limited = True

            if farmer_is_water_limited:
                water_limited_days[farmer] += 1
  
        return (
            channel_abstraction_m3_by_farmer,
            reservoir_abstraction_m3_by_farmer,
            groundwater_abstraction_m3_by_farmer,
            water_withdrawal_m,
            water_consumption_m,
            returnFlowIrr_m,
            addtoevapotrans_m,
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
            addtoevapotrans_m
        ) = self.abstract_water_numba(
            self.n,
            self.activation_order_by_elevation,
            self.field_indices_by_farmer,
            self.field_indices,
            self.n_water_limited_days,
            self.irrigation_efficiency,
            self.irrigated,
            self.irrigated,
            cell_area,
            HRU_to_grid,
            self.var.crop_map.get() if self.model.args.use_gpu else self.var.crop_map,
            totalPotIrrConsumption,
            available_channel_storage_m3,
            available_groundwater_m3,
            groundwater_head,
            available_reservoir_storage_m3,
            command_areas,
            return_fraction=self.model.config['agent_settings']['farmers']['return_fraction']
        )
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

    def save_profit(self, harvesting_farmers: np.ndarray, profit_per_farmer: np.ndarray, potential_profit_per_farmer: np.ndarray) -> None:
        """Saves the current harvest for harvesting farmers in a 2-dimensional array. The first dimension is the different farmers, while the second dimension are the previous harvests. First the previous harvests are moved by 1 column (dropping old harvests when they don't visit anymore) to make room for the new harvest. Then, the new harvest is placed in the array.
  
        Args:
            harvesting_farmers: farmers that harvest in this timestep.
        """
        assert (profit_per_farmer >= 0).all()
        assert (potential_profit_per_farmer >= 0).all()

        self.latest_profits[harvesting_farmers, 1:] = self.latest_profits[harvesting_farmers, 0:-1]
        self.latest_profits[harvesting_farmers, 0] = profit_per_farmer[harvesting_farmers]
  
        self.latest_potential_profits[harvesting_farmers, 1:] = self.latest_potential_profits[harvesting_farmers, 0:-1]
        self.latest_potential_profits[harvesting_farmers, 0] = potential_profit_per_farmer[harvesting_farmers]

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
  
    def plant_initial(self) -> None:
        self.var.actual_transpiration_crop = self.var.full_compressed(0, dtype=np.float32)
        self.var.potential_transpiration_crop = self.var.full_compressed(0, dtype=np.float32)

        self.var.land_use_type[self.var.land_use_type == 2] = 1
        self.var.land_use_type[self.var.land_use_type == 3] = 1
  
        self.var.crop_map = np.full_like(self.var.land_owners, -1)
        self.var.crop_age_days_map = np.full_like(self.var.land_owners, -1)
        self.var.crop_harvest_age_days = np.full_like(self.var.crop_age_days_map, -1)
          
    @staticmethod
    # @njit(cache=True)
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
            n_water_limited_days: Number of days that crop was water limited.
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
        actual_transpiration = self.var.actual_transpiration_crop.get() if self.model.args.use_gpu else self.var.actual_transpiration_crop
        potential_transpiration = self.var.potential_transpiration_crop.get() if self.model.args.use_gpu else self.var.potential_transpiration_crop
        crop_map = self.var.crop_map.get() if self.model.args.use_gpu else self.var.crop_map
        crop_age_days = self.var.crop_age_days_map.get() if self.model.args.use_gpu else self.var.crop_age_days_map
        harvest = self.harvest_numba(
            n=self.n,
            field_indices_by_farmer=self.field_indices_by_farmer,
            field_indices=self.field_indices,
            crop_map=crop_map,
            crop_age_days=crop_age_days,
            crop_harvest_age_days=self.var.crop_harvest_age_days,
        )
        if np.count_nonzero(harvest):  # Check if any harvested fields. Otherwise we don't need to run this.
            yield_ratio = self.get_yield_ratio(harvest, actual_transpiration, potential_transpiration, crop_map)
            assert (yield_ratio >= 0).all()

            harvesting_farmer_fields = self.var.land_owners[harvest]
            harvested_area = self.var.cellArea[harvest]
            if self.model.args.use_gpu:
                harvested_area = harvested_area.get()
            harvested_crops = crop_map[harvest]
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
            
            profit_per_farmer = np.bincount(harvesting_farmer_fields, weights=profit, minlength=self.n)

            # get potential crop profit per farmer
            potential_crop_yield = harvested_area * max_yield_per_crop
            potential_profit = potential_crop_yield * crop_prices_per_field
            potential_profit_per_farmer = np.bincount(harvesting_farmer_fields, weights=potential_profit, minlength=self.n)
      
            self.save_profit(harvesting_farmers, profit_per_farmer, potential_profit_per_farmer)
      
            self.disposable_income += profit_per_farmer

        if self.model.args.use_gpu:
            harvest = cp.array(harvest)
  
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
    # @njit(cache=True)
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
            plant: Subarray map of what crops are plantn this day.
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
            crop_map=self.var.crop_map if not self.model.args.use_gpu else self.var.crop_map.get(),
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
        if self.model.args.use_gpu:
            plant_map = cp.array(plant_map)
        if farmers_selling_land.size > 0:
            self.remove_agents(farmers_selling_land)
        if self.model.args.use_gpu:
            plant_map = cp.array(plant_map)

        self.var.crop_map = np.where(plant_map >= 0, plant_map, self.var.crop_map)
        self.var.crop_age_days_map[plant_map >= 0] = 0

        assert (self.var.crop_age_days_map[self.var.crop_map > 0] >= 0).all()

        field_is_paddy_irrigated = (self.var.crop_map == self.crop_names['Paddy'])
        self.var.land_use_type[(self.var.crop_map >= 0) & (field_is_paddy_irrigated == True)] = 2
        self.var.land_use_type[(self.var.crop_map >= 0) & (field_is_paddy_irrigated == False)] = 3

    @staticmethod
    @njit(cache=True)
    def invest_numba(
        year: int,
        farmers_without_well: np.ndarray,
        has_well: np.ndarray,
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
        disposable_income_threshold: int=0
    ):  
        """Determines whether a farmer without a well invests in an irrigation well and takes a loan. Each farmer has
        a probability of investing in a well based on the following factors:
            - The farmer's latest profit ratio (latest profit / latest potential profit)
            - The profit ratio of the farmer's neighbors with wells (average of neighbors' latest profit / latest potential profit)
            - Whether farmer disposable income is sufficient to pay for a loan for a well as well as the yearly upkeep cost
            - The farmer's disposable income relative to the minimum disposable income required to invest in a well
  
        Args:
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
        """
        neighbor_nan_value = np.iinfo(neighbors_with_well.dtype).max
        for i, farmer_idx in enumerate(farmers_without_well):
            assert not has_well[farmer_idx]

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

                has_well[farmer_idx] = True
                
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

    def invest(self) -> None:
        nbits = 19

        # from honeybees.library.geohash import window
        # width_lon, height_lat = window(nbits, *self.model.bounds)
        # import matplotlib.pyplot as plt
        # colors = ['red', 'blue', 'green', 'yellow', 'orange']
        # for i in range(5):
        #     own_location = self.locations[farmers_with_well[i]]
        #     plt.scatter(own_location[0], own_location[1], c=colors[i])
        #     agent_neighbours = neighbors_with_well[i]
        #     plt.scatter(self.locations[agent_neighbours][:,0], self.locations[agent_neighbours][:,1], c=colors[i], alpha=.5)
        # plt.show()

        # do not invest during spinup
        if self.model.args == 'spinup':
            return

        for crop_option in np.unique(self.crops, axis=0):
            
            farmers_with_crop_option = np.where((self.crops==crop_option[None, ...]).all(axis=1))[0]
            
            farmers_with_well = np.where(self.has_well[farmers_with_crop_option] == True)[0]
            farmers_without_well = np.where(self.has_well[farmers_with_crop_option] == False)[0]
            farmers_without_well_indices = farmers_with_crop_option[farmers_without_well]

            if farmers_without_well.size > 0 and farmers_with_well.size > 0:
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
                    search_target_ids=farmers_with_well
                )

                interest_rate = self.lending_rate[self.model.current_time.year]
                assert not np.isnan(interest_rate)
                self.invest_numba(
                    self.model.current_time.year,
                    farmers_without_well_indices,
                    self.has_well,
                    neighbors_with_well,
                    self.latest_profits,
                    self.latest_potential_profits,
                    self.field_size_per_farmer,
                    self.loan_interest,
                    self.loan_amount,
                    self.loan_duration,
                    self.loan_end_year,
                    self.well_price[self.model.current_time.year],
                    self.well_upkeep_price_per_m2[self.model.current_time.year],
                    self.well_investment_time_years,
                    interest_rate,
                    self.disposable_income,
                )
    
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
    def irrigated(self):
        return self._irrigated[:self.n]

    @irrigated.setter
    def irrigated(self, value):
        self._irrigated[:self.n] = value

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
    def water_availability_by_farmer(self):
        return self._water_availability_by_farmer[:self.n]

    @water_availability_by_farmer.setter
    def water_availability_by_farmer(self, value):
        self._water_availability_by_farmer[:self.n] = value

    @property
    def n_water_limited_days(self):
        return self._n_water_limited_days[:self.n]

    @n_water_limited_days.setter
    def n_water_limited_days(self, value):
        self._n_water_limited_days[:self.n] = value

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
    def has_well(self):
        return self._has_well[:self.n]

    @has_well.setter
    def has_well(self, value):
        self._has_well[:self.n] = value

    @property
    def tehsil(self):
        return self._tehsil[:self.n]

    @tehsil.setter
    def tehsil(self, value):
        self._tehsil[:self.n] = value

    @property
    def field_indices_by_farmer(self):
        return self._field_indices_by_farmer[:self.n]

    @field_indices_by_farmer.setter
    def field_indices_by_farmer(self, value):
        self._field_indices_by_farmer[:self.n] = value

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

    def adjust_prices_inflation(self):
        pass

    def upkeep_assets(self):
        self.disposable_income -= self.well_upkeep_price_per_m2[self.model.current_time.year] * self.field_size_per_farmer * self.has_well

    def make_loan_payment(self):
        has_loan = self.loan_amount > 0
        years_left = self.loan_end_year[has_loan] - self.model.current_time.year
        loan_payoff = self.loan_amount[has_loan] / years_left
        loan_interest = self.loan_amount[has_loan] * self.loan_interest[has_loan]
        loan_payment = loan_payoff + loan_interest

        self.loan_amount[has_loan] -= loan_payoff
        self.disposable_income[has_loan] -= loan_payment
        self.disposable_income[self.disposable_income < 0] = 0  # for now, we assume that farmers don't have negative disposable income

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
        if self.model.current_time.month == 1 and self.model.current_time.day == 1 and self.model.args.scenario != 'spinup':
            self.adjust_prices_inflation()
            self.upkeep_assets()
            self.make_loan_payment()
            self.invest()
            # reset disposable income
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
        self.n_water_limited_days[self.n-1] = 0
        self.water_availability_by_farmer[self.n-1] = 0
        self.channel_abstraction_m3_by_farmer[self.n-1] = 0
        self.groundwater_abstraction_m3_by_farmer[self.n-1] = 0
        self.reservoir_abstraction_m3_by_farmer[self.n-1] = 0
        self.latest_profits[self.n-1] = [np.nan, np.nan, np.nan]
        self.latest_potential_profits[self.n-1] = [np.nan, np.nan, np.nan]

        for attr in self.agent_attributes:
            assert getattr(self, attr[1:]).shape[0] == self.n
            if "nodatacheck" not in self.agent_attributes_meta[attr] or self.agent_attributes_meta[attr]['nodatacheck'] is True:
                assert not np.array_equal(getattr(self, attr)[self.n-1], self.agent_attributes_meta[attr]['nodata'], equal_nan=True)