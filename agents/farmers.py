# -*- coding: utf-8 -*-
from hyve.library.mapIO import ArrayReader
import os
import numpy as np
import pandas as pd
try:
    import cupy as cp
except ModuleNotFoundError:
    pass
import math

from numba import njit

from hyve.library.neighbors import find_neighbors
from hyve.agents import AgentBaseClass


@njit
def is_currently_growing(plant_month: int, harvest_month: int, current_month: int, current_day: int, start_day_per_month: np.ndarray) -> int:
    """Checks whether a crop is currently growing based on the plant month, harvest month, and current model date. Used for initalization of the model.

    Args:
        plant_month: month that the crop is planted.
        harvest_month: month that the crop is harvested.
        current_month: Current month.
        current_day: Current day year.
        start_day_per_month: 1d NumPy array with starting day per month [31, ..., 31]

    Returns:
        crop_age_days: Crop age in days at current day. Returns -1 when crop is not currently growing.
    """
    if harvest_month < plant_month:
        harvest_month += 12
    if current_month >= plant_month and current_month < harvest_month:
        crop_age_days = (current_day - start_day_per_month[plant_month - 1]) % 365
    else:
        crop_age_days = -1
    return crop_age_days

@njit
def get_farmer_fields(field_indices: np.ndarray, field_indices_per_farmer: np.ndarray, farmer_index: int) -> np.ndarray:
    """Gets indices of field for given farmer.
    
    Args:
        field_indices_per_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
        field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  

    Returns:
        field_indices_for_farmer: the indices of the fields for the given farmer.
    """
    return field_indices[field_indices_per_farmer[farmer_index, 0]: field_indices_per_farmer[farmer_index, 1]]

def take_with_ignore(a, indices, ignore_index, ignore_value=np.nan):
    array = np.take(a, indices)
    array[indices == ignore_index] = ignore_value
    return array


class Farmers(AgentBaseClass):
    """The agent class for the farmers. Contains all data and behaviourial methods. The __init__ function only gets the model as arguments, the agent parent class and the redundancy. All other variables are loaded at later stages.
    
    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
        redundancy: a lot of data is saved in pre-allocated NumPy arrays. While this allows much faster operation, it does mean that the number of agents cannot grow beyond the size of the pre-allocated arrays. This parameter allows you to specify how much redundancy should be used. A lower redundancy means less memory is used, but the model crashes if the redundancy is insufficient.
    """
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents
        self.var = model.data.landunit
        self.redundancy = reduncancy
        self.input_folder = 'DataDrive/GEB/input'
        self.crop_yield_factors = self.get_crop_yield_factors()
        self.initiate_agents()

    def initiate_agents(self) -> None:
        """Calls functions to initialize all agent attributes, including their locations. Then, crops are initially planted. 
        """
        self.fields = self.var.land_owners
        if self.model.args.use_gpu:
            self.fields = self.fields.get()
        self.create_field_indices(self.fields)
        
        self.initiate_locations()
        self.initiate_attributes()
        self.plant_initial()
        print(f'initialized {self.n} agents')

    def initiate_locations(self) -> None:
        """
        Loads locations of the farmers from .npy-file and saves to self.locations. Sets self.n to the current numbers of farmers, and sets self.max_n to the maximum number of farmers that can be expected in the model (model will fail if more than max_n farmers.) considering reducancy parameter.
        """
        agent_locations = np.load(os.path.join(self.input_folder, "agents/farmer_locations.npy"))

        self.n = agent_locations.shape[0]
        assert self.n > 0
        self.max_n = math.ceil(self.n * (1 + self.redundancy))
        assert self.max_n < 4294967295 # max value of uint32, consider replacing with uint64
        
        self._locations = np.zeros((self.max_n, 2))
        self.locations = agent_locations

    def initiate_attributes(self) -> None:
        """
        This function is used to initiate all agent attributes, such as crop type, irrigiation type and elevation.
        """
        elevation_map = ArrayReader(
            fp='DataDrive/GEB/input/landsurface/topo/subelv.tif',
            bounds=self.model.bounds
        )
        self._elevation = np.zeros(self.max_n, dtype=np.float32)
        self.elevation = elevation_map.sample_coords(self.locations)
        crop_file = os.path.join('DataDrive', 'GEB', 'input', 'agents', 'crop.npy')
        if self.model.args.use_gpu:
            self._crop = cp.load(crop_file)
        else:
            self._crop = np.load(crop_file)
        self._irrigating = np.zeros(self.max_n, dtype=np.int8)
        self.irrigating = np.load(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'irrigating.npy'))
        self._planting_scheme = np.zeros(self.max_n, dtype=np.int8)
        self.planting_scheme = np.load(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'planting_scheme.npy'))
        self._unit_code = np.zeros(self.max_n, dtype=np.int32)
        self.unit_code = np.load(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'farmer_unit_codes.npy'))
        if self.model.args.use_gpu:
            self._is_paddy_irrigated = cp.zeros(self.max_n, dtype=np.bool_)
        else:
            self._is_paddy_irrigated = np.zeros(self.max_n, dtype=np.bool_)
        self.is_paddy_irrigated[self.crop == 2] = True  # set rice to paddy-irrigated

        self._yield_ratio_per_farmer = np.zeros(self.max_n, dtype=np.float32)

        self._is_water_efficient = np.zeros(self.max_n, dtype=np.bool)
        self._latest_harvests = np.zeros((self.max_n, 3), dtype=np.float32)

        self._channel_abstraction_m3_by_farmer = np.zeros(self.max_n, dtype=np.float32)
        self._reservoir_abstraction_m3_by_farmer = np.zeros(self.max_n, dtype=np.float32)
        self._groundwater_abstraction_m3_by_farmer = np.zeros(self.max_n, dtype=np.float32)
        self._water_availability_by_farmer = np.zeros(self.max_n, dtype=np.float32)
        self._n_water_limited_days = np.zeros(self.max_n, dtype=np.int32)

        self.planting_schemes = np.load(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'planting_schemes.npy'))

        self.actual_transpiration_crop = self.var.full_compressed(0, dtype=np.float32)
        self.potential_transpiration_crop = self.var.full_compressed(0, dtype=np.float32)

    @staticmethod
    @njit(cache=True)
    def create_field_indices_numba(land_owners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Creates `field_indices_per_farmer` and `field_indices`. These indices are used to quickly find the fields for a specific farmer.

        Args:
            land_owners: Array of the land owners. Each unique ID is a different land owner. -1 means the land is not owned by anyone.

        Returns:
            field_indices_per_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  
        """
        agents = np.unique(land_owners)
        if agents[0] == -1:
            n_agents = agents.size -1
        else:
            n_agents = agents.size
        field_indices_per_farmer = np.full((n_agents, 2), -1, dtype=np.int32)
        field_indices = np.full(land_owners.size, -1, dtype=np.int32)

        land_owners_sort_idx = np.argsort(land_owners)
        land_owners_sorted = land_owners[land_owners_sort_idx]

        last_not_owned = np.searchsorted(land_owners_sorted, -1, side='right')

        prev_land_owner = -1
        for i in range(last_not_owned, land_owners.size):
            land_owner = land_owners[land_owners_sort_idx[i]]
            if land_owner != -1:
                if land_owner != prev_land_owner:
                    field_indices_per_farmer[land_owner, 0] = i - last_not_owned
                field_indices_per_farmer[land_owner, 1] = i + 1 - last_not_owned
                field_indices[i - last_not_owned] = land_owners_sort_idx[i]
                prev_land_owner = land_owner
        field_indices = field_indices[:-last_not_owned]
        return field_indices_per_farmer, field_indices

    def create_field_indices(self, *args, **kwargs) -> None:
        """Creates `field_indices_per_farmer` and `field_indices`. These indices are used to quickly find the fields for a specific farmer."""
        self.field_indices_per_farmer, self.field_indices = self.create_field_indices_numba(*args, **kwargs)
    
    @property
    def activation_order_by_elevation(self):
        """
        Activation order is determined by the agent elevation, starting from the highest.
        Agents with the same elevation are randomly shuffled.
        """
        # if activation order is fixed. Get the random state, and set a fixed seet.
        if self.model.config['agent_settings']['fix_activation_order']:
            random_state = np.random.get_state()
            np.random.seed(42)
        # Shuffle agent elevation and agent_ids in unision.
        p = np.random.permutation(self.elevation.size)
        # if activation order is fixed, set random state to previous state
        if self.model.config['agent_settings']['fix_activation_order']:
            np.random.set_state(random_state)
        elevation_shuffled = self.elevation[p]
        agent_ids_shuffled = np.arange(0, self.elevation.size, 1, dtype=np.int32)[p]
        # Use argsort to find the order or the shuffled elevation. Using a stable sorting
        # algorithm such that the random shuffling in the previous step is conserved
        # in groups with identical elevation.
        activation_order_shuffled = np.argsort(elevation_shuffled, kind="stable")[::-1]
        argsort_agend_ids = agent_ids_shuffled[activation_order_shuffled]
        # Return the agent ids ranks in the order of activation.
        ranks = np.empty_like(argsort_agend_ids)
        ranks[argsort_agend_ids] = np.arange(argsort_agend_ids.size)
        return ranks

    @staticmethod
    @njit
    def abstract_water_numba(
        activation_order: np.ndarray,
        field_indices_per_farmer: np.ndarray,
        field_indices: np.ndarray,
        is_water_efficient: np.ndarray,
        irrigated: np.ndarray,
        cell_area: np.ndarray,
        landunit_to_grid: np.ndarray,
        crop_map: np.ndarray,
        totalPotIrrConsumption: np.ndarray,
        available_channel_storage_m3: np.ndarray,
        available_groundwater_m3: np.ndarray,
        groundwater_head: np.ndarray,
        available_reservoir_storage_m3: np.ndarray,
        command_areas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function is used to regulate the irrigation behavior of farmers. The farmers are "activated" by the given `activation_order` and each farmer can irrigate from the various water sources, given water is available and the farmers has the means to abstract water. The abstraction order is channel irrigation, reservoir irrigation, groundwater irrigation. 

        Args:
            activation_order: Order in which the agents are activated. Agents that are activated first get a first go at extracting water, leaving less water for other farmers.
            field_indices_per_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  
            is_water_efficient: Boolean array that specifies whether the specific farmer is efficient with water use.
            cell_area: The area of each subcell in m2.
            landunit_to_grid: Array to map the index of each subcell to the corresponding cell.
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
        
        mixed_array_size = cell_area.size
        water_withdrawal_m = np.zeros(mixed_array_size, dtype=np.float32)
        water_consumption_m = np.zeros(mixed_array_size, dtype=np.float32)
        
        returnFlowIrr_m = np.zeros(mixed_array_size, dtype=np.float32)
        addtoevapotrans_m = np.zeros(mixed_array_size, dtype=np.float32)
        
        groundwater_abstraction_m3 = np.zeros(available_groundwater_m3.size, dtype=np.float32)
        channel_abstraction_m3 = np.zeros(available_channel_storage_m3.size, dtype=np.float32)
        
        reservoir_abstraction_m_per_basin_m3 = np.zeros(available_reservoir_storage_m3.size, dtype=np.float32)
        reservoir_abstraction_m = np.zeros(mixed_array_size, dtype=np.float32)

        channel_abstraction_m3_by_farmer = np.zeros(activation_order.size, dtype=np.float32)
        reservoir_abstraction_m3_by_farmer = np.zeros(activation_order.size, dtype=np.float32)
        groundwater_abstraction_m3_by_farmer = np.zeros(activation_order.size, dtype=np.float32)
        
        for farmer in range(activation_order.size):
            farmer_fields = get_farmer_fields(field_indices, field_indices_per_farmer, farmer)
            if is_water_efficient[farmer]:
                efficiency = 0.8
            else:
                efficiency = 0.6
            
            return_fraction = 0.5
            farmer_irrigation = irrigated[farmer]
            
            for field in farmer_fields:
                f_var = landunit_to_grid[field]
                crop = crop_map[field]
                irrigation_water_demand_cell = totalPotIrrConsumption[field] / efficiency
                
                if crop != -1:
                    if farmer_irrigation:
                        # channel abstraction
                        available_channel_storage_cell_m = available_channel_storage_m3[f_var] / cell_area[field]
                        channel_abstraction_cell_m = min(available_channel_storage_cell_m, irrigation_water_demand_cell)
                        channel_abstraction_cell_m3 = channel_abstraction_cell_m * cell_area[field]
                        available_channel_storage_m3[f_var] -= channel_abstraction_cell_m3
                        water_withdrawal_m[field] += channel_abstraction_cell_m
                        channel_abstraction_m3[f_var] = channel_abstraction_cell_m3

                        channel_abstraction_m3_by_farmer[farmer] += channel_abstraction_cell_m3
                        
                        irrigation_water_demand_cell -= channel_abstraction_cell_m
                        
                        # command areas
                        command_area = command_areas[field]
                        if command_area >= 0:  # -1 means no command area
                            water_demand_cell_M3 = irrigation_water_demand_cell * cell_area[field]
                            reservoir_abstraction_m_cell_m3 = min(available_reservoir_storage_m3[command_area], water_demand_cell_M3)
                            available_reservoir_storage_m3[command_area] -= reservoir_abstraction_m_cell_m3
                            reservoir_abstraction_m_per_basin_m3[command_area] += reservoir_abstraction_m_cell_m3
                            reservoir_abstraction_m_cell = reservoir_abstraction_m_cell_m3 / cell_area[field]
                            reservoir_abstraction_m[field] += reservoir_abstraction_m_cell
                            water_withdrawal_m[field] += reservoir_abstraction_m_cell

                            reservoir_abstraction_m3_by_farmer[farmer] += reservoir_abstraction_m_cell_m3
                            
                            irrigation_water_demand_cell -= reservoir_abstraction_m_cell

                        # groundwater irrigation
                        available_groundwater_cell_m = available_groundwater_m3[f_var] / cell_area[field]
                        groundwater_abstraction_cell_m = min(available_groundwater_cell_m, irrigation_water_demand_cell)
                        groundwater_abstraction_cell_m3 = groundwater_abstraction_cell_m * cell_area[field]
                        groundwater_abstraction_m3[f_var] = groundwater_abstraction_cell_m3
                        available_groundwater_m3[f_var] -= groundwater_abstraction_cell_m3
                        water_withdrawal_m[field] += groundwater_abstraction_cell_m

                        groundwater_abstraction_m3_by_farmer[farmer] += groundwater_abstraction_cell_m3
                
                        irrigation_water_demand_cell -= groundwater_abstraction_cell_m
                
                assert irrigation_water_demand_cell >= -1e15  # Make sure irrigation water demand is zero, or positive. Allow very small error.

                water_consumption_m[field] = water_withdrawal_m[field] * efficiency
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
        )

    def abstract_water(
        self,
        cell_area: np.ndarray,
        landunit_to_grid: np.ndarray,
        totalPotIrrConsumption: np.ndarray,
        available_channel_storage_m3: np.ndarray,
        available_groundwater_m3: np.ndarray,
        groundwater_head: np.ndarray,
        available_reservoir_storage_m3: np.ndarray,
        command_areas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function allows the abstraction of water by farmers for irrigation purposes. It's main purpose is to call the relevant numba function to do the actual abstraction. In addition, the function saves the abstraction from the various sources by farmer.

        Args:
            cell_area: the area of each subcell in m2.
            landunit_to_grid: array to map the index of each subcell to the corresponding cell.
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
        activation_order = self.activation_order_by_elevation
        (
            self.channel_abstraction_m3_by_farmer,
            self.reservoir_abstraction_m3_by_farmer,
            self.groundwater_abstraction_m3_by_farmer,
            water_withdrawal_m,
            water_consumption_m,
            returnFlowIrr_m,
            addtoevapotrans_m
        ) = self.abstract_water_numba(
            activation_order,
            self.field_indices_per_farmer,
            self.field_indices,
            self.is_water_efficient,
            self.irrigating,
            cell_area,
            landunit_to_grid,
            self.var.crop_map.get() if self.model.args.use_gpu else self.var.crop_map,
            totalPotIrrConsumption,
            available_channel_storage_m3,
            available_groundwater_m3,
            groundwater_head,
            available_reservoir_storage_m3,
            command_areas
        )
        return (
            water_withdrawal_m,
            water_consumption_m,
            returnFlowIrr_m,
            addtoevapotrans_m
        )

    def get_crop_factors(self) -> pd.DataFrame:
        """Loads crop factors from disk (DataDrive/GEB/input/crop_data/crop_factors.csv). The length of the various crop stages is relative and is cumulated to one."""
        df = pd.read_csv(os.path.join('DataDrive', 'GEB', 'input', 'crop_data', 'crop_factors.csv'))
        df['L_dev'] = df['L_ini'] + df['L_dev']
        df['L_mid'] = df['L_dev'] + df['L_mid']
        df['L_late'] = df['L_mid'] + df['L_late']
        assert np.allclose(df['L_late'], 1.0)
        assert len(df) == 26
        return df

    def get_crop_yield_factors(self) -> dict[np.ndarray]:
        """Read csv-file of values for crop water depletion. Obtained from Table 2 of this paper: https://doi.org/10.1016/j.jhydrol.2009.07.031
        
        Returns:
            yield_factors: dictonary with np.ndarray of values per crop for each variable.
        """
        df = pd.read_csv(os.path.join('DataDrive', 'GEB', 'input', 'crop_data', 'yield_ratios.csv'))
        yield_factors = df[['alpha', 'beta', 'P0', 'P1']].to_dict(orient='list')
        yield_factors = {
            key: np.array(value) for key, value in yield_factors.items()
        }
        return yield_factors

    @property
    def start_day_per_month(self) -> np.ndarray:
        """Get starting day for each month of year
        
        Returns:
            starting_day_per_month: Starting day of each month of year.
        """
        return np.cumsum(np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]))

    @property
    def current_day_of_year(self) -> int:
        """Gets the current day of the year.
        
        Returns:
            day: current day of the year.
        """
        return self.start_day_per_month[self.model.current_time.month - 1] + self.model.current_time.day

    @staticmethod
    @njit
    def get_yield_ratio_numba(crop_map: np.array, evap_ratios: np.array, alpha: np.array, beta: np.array, P0: np.array, P1: np.array) -> float:
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
        yield_ratios = np.empty(evap_ratios.size, dtype=np.float32)

        assert crop_map.size == evap_ratios.size
        
        for i in range(evap_ratios.size):
            evap_ratio = evap_ratios[i]
            crop = crop_map[i]
            if alpha[crop] * evap_ratio + beta[crop] > 1:
                yield_ratio = 1
            elif P0[crop] < evap_ratio < P1[crop]:
                yield_ratio = alpha[crop] * P1[crop] + beta[crop] - (P1[crop] - evap_ratio) * (alpha[crop] * P1[crop] + beta[crop]) / (P1[crop] - P0[crop])
            elif evap_ratio < P0[crop]:
                yield_ratio = 0
            else:
                yield_ratio = alpha[crop] * evap_ratio + beta[crop]
            yield_ratios[i] = yield_ratio
        
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
        """
        return self.get_yield_ratio_numba(
            crop_map[harvest],
            actual_transpiration[harvest] / potential_transpiration[harvest],
            self.crop_yield_factors['alpha'],
            self.crop_yield_factors['beta'],
            self.crop_yield_factors['P0'],
            self.crop_yield_factors['P1'],
        )

    def save_harvest(self, harvesting_farmers: np.ndarray) -> None:
        """Saves the current harvest for harvesting farmers in a 2-dimensional array. The first dimension is the different farmers, while the second dimension are the previous harvests. First the previous harvests are moved by 1 column (dropping old harvests when they don't visit anymore) to make room for the new harvest. Then, the new harvest is placed in the array.
        
        Args:
            harvesting_farmers: farmers that harvest in this timestep.
        """
        self.latest_harvests[harvesting_farmers, 1:] = self.latest_harvests[harvesting_farmers, 0:-1]
        self.latest_harvests[harvesting_farmers, 0] = self.yield_ratio_per_farmer[harvesting_farmers]

    def invest_in_water_efficiency(self, harvesting_farmers: np.ndarray) -> None:
        """In the scenario `self_investments` farmers invest in water saving techniques when their harvest starts dropping. Concretely if their harvest is lower than the previous harvests, they do so.
        
        Args:
            harvesting_farmers: farmers that harvest in this timestep.
        """
        if self.model.args.scenario == 'self_investment':
            invest = (
                self.latest_harvests[harvesting_farmers, 0] < np.mean(self.latest_harvests[harvesting_farmers, 1:], axis=1)
            )
            self.is_water_efficient[harvesting_farmers] |= invest

    @staticmethod
    @njit
    def get_crop_age_and_harvest_day_initial(
        n: int,
        current_month: int,
        current_day: int,
        planting_schemes: np.ndarray,
        crop: np.ndarray,
        irrigating: np.ndarray,
        unit_code: np.ndarray,
        planting_scheme: np.ndarray,
        start_day_per_month: np.ndarray,
        days_in_year: int=365
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        When the model is initalized, crops are already growing. This function finds out which crops are growing at that stage, and how far in their growth stage they are.

        Args:
            n: Number of farmers.
            current_month: Current month.
            current_day: Current day.
            planting_schemes: A 6-dimensional array which contains the planing schemes.

               1. Whether the crop is irrigated or not.
               2. The unit code for the crop.
               3. The crop.
               4. The planting scheme (e.g., single or double cropping).
               5. If multicropping, the various crop stages.
               6. The planting month and the harvest month.
            crop: Crops grown by each farmer.
            irrigating: Whether each farmers irrigates or not (True is irrigating, False is not).
            unit_code: The unit code of each farmer.
            planting_scheme: The planting scheme for each farmer.
            start_day_per_month: The starting day of the year for each month.
            days_in_year: Number of days in year.
        """
        crop_age_days = np.full(n, -1, dtype=np.int32)
        crop_harvest_age_days = np.full(n, -1, dtype=np.int32)
        next_plant_day = np.full(n, -1, dtype=np.int32)
        next_multicrop_index = np.full(n, -1, dtype=np.int32)
        n_multicrop_periods = np.full(n, -1, dtype=np.int32)

        assert days_in_year == 365

        for i in range(n):
            farmer_planting_scheme = planting_schemes[irrigating[i], unit_code[i], crop[i], planting_scheme[i]]
            n_cropping_periods = (farmer_planting_scheme[:, 0] != -1).sum()
            n_multicrop_periods[i] =n_cropping_periods
            for j in range(n_cropping_periods):
                currently_planted = is_currently_growing(farmer_planting_scheme[j, 0], farmer_planting_scheme[j, 1], current_month, current_day, start_day_per_month)
                if currently_planted != -1:
                    crop_age_days[i] = currently_planted
                    crop_data = farmer_planting_scheme[j]
                    crop_harvest_age_days[i] = (start_day_per_month[crop_data[1] - 1] - start_day_per_month[crop_data[0] - 1]) % 365
                    next_multicrop_index[i] = j
                    break
                elif n_cropping_periods == 1:
                    next_plant_day[i] = start_day_per_month[farmer_planting_scheme[j, 0] - 1]
                    next_multicrop_index[i] = 0
                    break
            else:  # no crops are currently planted, so let's find out the first crop with a plant day
                assert n_cropping_periods > 1
                for j in range(n_cropping_periods):
                    if j == 0:  # if first option, just set it as the next plant day
                        next_plant_day[i] = start_day_per_month[farmer_planting_scheme[j, 0] - 1]
                    else:  # if not, find the first option starting from the current day
                        potential_next_plant_day = start_day_per_month[farmer_planting_scheme[j, 0] - 1]
                        next_plant_day[i] = ((min((potential_next_plant_day - current_day) % days_in_year, (next_plant_day[i] - current_day) % days_in_year)) + current_day) % days_in_year

                for j in range(n_cropping_periods):
                    if start_day_per_month[farmer_planting_scheme[j, 0] - 1] == next_plant_day[i]:
                        next_multicrop_index[i] = j
                        break
                else:  # make sure one of the options is picked
                    assert False

            assert crop_harvest_age_days[i] != -1 or next_plant_day[i] != -1

        assert (next_multicrop_index != -1).all()
        assert (n_multicrop_periods != -1).all()

        return crop_age_days, crop_harvest_age_days, next_plant_day, n_multicrop_periods, next_multicrop_index

    def by_field(self, var, nofieldvalue=-1):
        by_field = np.take(var, self.fields)
        by_field[self.fields == -1] = nofieldvalue
        return by_field
        
    def plant_initial(self) -> None:
        """When the model is initalized, crops are already growing. This function first finds out which farmers are already growing crops, how old these are, as well as the multicrop indices. Then, these arrays per farmer are converted to the field array.
        """

        self.var.land_use_type[self.var.land_use_type == 2] = 1
        self.var.land_use_type[self.var.land_use_type == 3] = 1

        if self.model.args.use_gpu:
            crop = self.crop.get()
        else:
            crop = self.crop.copy()

        crop_age_days, crop_harvest_age_days, next_plant_day, n_multicrop_periods, next_multicrop_index = self.get_crop_age_and_harvest_day_initial(self.n, self.model.current_time.month, self.current_day_of_year, self.planting_schemes, crop, self.irrigating, self.unit_code, self.planting_scheme, self.start_day_per_month)
        assert np.logical_xor((next_plant_day == -1), (crop_harvest_age_days == -1)).all()
        if self.model.args.use_gpu:
            crop_age_days = cp.array(crop_age_days)
            crop_harvest_age_days = cp.array(crop_harvest_age_days)

        plant = self.crop.copy()
        plant[crop_age_days == -1] = -1
        
        self.var.crop_map = self.by_field(plant)
        self.var.crop_age_days_map = self.by_field(crop_age_days)
        self.var.crop_harvest_age_days_map = self.by_field(crop_harvest_age_days)
        self.var.next_plant_day = self.by_field(next_plant_day)
        self.var.n_multicrop_periods = self.by_field(n_multicrop_periods)
        self.var.next_multicrop_index = self.by_field(next_multicrop_index)
                
        assert (self.var.crop_harvest_age_days_map[self.var.crop_age_days_map >= 0] != -1).all()
        assert self.var.crop_map.shape == self.var.crop_age_days_map.shape == self.var.crop_harvest_age_days_map.shape

        assert (self.var.crop_harvest_age_days_map[self.var.crop_map > 0] > 0).all()

        field_is_paddy_irrigated = self.by_field(self.is_paddy_irrigated, nofieldvalue=0)
        self.var.land_use_type[(self.var.crop_map >= 0) & (field_is_paddy_irrigated == True)] = 2
        self.var.land_use_type[(self.var.crop_map >= 0) & (field_is_paddy_irrigated == False)] = 3

    @staticmethod
    @njit
    def harvest_numba(
        n: np.ndarray,
        start_day_per_month: np.ndarray,
        field_indices_per_farmer: np.ndarray,
        field_indices: np.ndarray,
        crop_map: np.ndarray,
        crop_age_days: np.ndarray,
        crop_harvest_age_days: np.ndarray,
        next_plant_day: np.ndarray,
        next_multicrop_index: np.ndarray,
        n_multicrop_periods: np.ndarray,
        planting_schemes: np.ndarray,
        irrigating: np.ndarray,
        unit_code: np.ndarray,
        crop: np.ndarray,
        planting_scheme: np.ndarray
    ) -> np.ndarray:
        """This function determines whether crops are ready to be harvested by comparing the crop harvest age to the current age of the crop. If the crop is harvested, the crops next multicrop index and next plant day are determined.

        Args:
            n: Number of farmers.
            start_day_per_month: Array containing the starting day of each month.
            field_indices_per_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  
            crop_map: Subarray map of crops.
            crop_age_days: Subarray map of current crop age in days.
            crop_harvest_age_days: Subarray map of crop harvest age in days.
            next_plant_day: Subarray map of next planting day.
            next_multicrop_index: Subarray map of the next multicropping index.
            n_multicrop_periods: Subarray map of the number of multicropping periods.
            planting_schemes: A 6-dimensional array which contains the planing schemes.

               1. Whether the crop is irrigated or not.
               2. The unit code for the crop.
               3. The crop.
               4. The planting scheme (e.g., single or double cropping).
               5. If multicropping, the various crop stages.
               6. The planting month and the harvest month.
            irrigating: Whether each farmers irrigates or not (True is irrigating, False is not).
            unit_code: The unit code of each farmer.
            crop: Crops grown by each farmer.
            planting_scheme: The planting scheme for each farmer.

        Returns:
            harvest: Boolean subarray map of fields to be harvested.
        """
        harvest = np.zeros(crop_map.shape, dtype=np.bool_)
        for i in range(n):
            farmer_fields = get_farmer_fields(field_indices, field_indices_per_farmer, i)
            for field in farmer_fields:
                crop_age = crop_age_days[field]
                if crop_age >= 0:
                    assert crop_map[field] != -1
                    assert crop_harvest_age_days[field] != -1
                    if crop_age == crop_harvest_age_days[field]:
                        harvest[field] = True
                        next_multicrop_index[field] = (next_multicrop_index[field] + 1) % n_multicrop_periods[field]
                        assert next_multicrop_index[field] < n_multicrop_periods[field] 
                        next_plant_month = planting_schemes[irrigating[i], unit_code[i], crop[i], planting_scheme[i], next_multicrop_index[field]]
                        assert next_plant_month[0] != -1
                        next_plant_day[field] = start_day_per_month[next_plant_month - 1][0]
                else:
                    assert crop_map[field] == -1
                    assert crop_harvest_age_days[field] == -1
        return harvest
        
    def harvest(self):
        """This function determines which crops needs to be harvested, based on the current age of the crops and the harvest age of the crop. First a helper function is used to obtain the harvest map. Then, if at least 1 field is harvested, the yield ratio is obtained for all fields using the ratio of actual to potential evapotranspiration, saves the harvest per farmer and potentially invests in water saving techniques.
        """
        actual_transpiration = self.actual_transpiration_crop.get() if self.model.args.use_gpu else self.actual_transpiration_crop
        potential_transpiration = self.potential_transpiration_crop.get() if self.model.args.use_gpu else self.potential_transpiration_crop
        crop_map = self.var.crop_map.get() if self.model.args.use_gpu else self.var.crop_map
        crop_age_days = self.var.crop_age_days_map.get() if self.model.args.use_gpu else self.var.crop_age_days_map
        crop_harvest_age_days = self.var.crop_harvest_age_days_map.get() if self.model.args.use_gpu else self.var.crop_harvest_age_days_map
        harvest = self.harvest_numba(
            self.n,
            self.start_day_per_month,
            self.field_indices_per_farmer,
            self.field_indices,
            crop_map,
            crop_age_days,
            crop_harvest_age_days,
            self.var.next_plant_day,
            self.var.next_multicrop_index,
            self.var.n_multicrop_periods,
            self.planting_schemes,
            self.irrigating,
            self.unit_code,
            self.crop.get() if self.model.args.use_gpu else self.crop,
            self.planting_scheme
        )
        if np.count_nonzero(harvest):
            yield_ratio = self.get_yield_ratio(harvest, actual_transpiration, potential_transpiration, crop_map)
            harvesting_farmer_fields = self.fields[harvest]
            self.yield_ratio_per_farmer = np.bincount(harvesting_farmer_fields, weights=yield_ratio, minlength=self.n)
            harvesting_farmers = np.unique(harvesting_farmer_fields)
            if self.model.current_timestep > 365:
                self.save_harvest(harvesting_farmers)
            self.invest_in_water_efficiency(harvesting_farmers)
        if self.model.args.use_gpu:
            harvest = cp.array(harvest)
        
        self.actual_transpiration_crop[harvest] = 0
        self.potential_transpiration_crop[harvest] = 0

        # remove crops from crop_map where they are harvested
        self.var.crop_map[harvest] = -1
        self.var.crop_age_days_map[harvest] = -1
        self.var.crop_harvest_age_days_map[harvest] = -1

        # when a crop is harvested set to non irrigated land
        self.var.land_use_type[harvest] = 1

        # increase crop age by 1 where crops are not harvested and growing
        self.var.crop_age_days_map[(harvest == False) & (self.var.crop_map >= 0)] += 1

    @staticmethod
    @njit
    def plant_numba(
        n: int,
        start_day_per_month: np.ndarray,
        current_day: int,
        next_plant_day: np.ndarray,
        crop: np.ndarray,
        field_indices_per_farmer: np.ndarray,
        field_indices: np.ndarray,
        planting_schemes: np.ndarray,
        irrigating: np.ndarray,
        unit_code: np.ndarray,
        planting_scheme: np.ndarray,
        next_multicrop_index: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Determines when and what crop should be planted, by comparing the current day to the next plant day. Also sets the haverst age of the plant.
        
        Args:
            n: Number of farmers.
            start_day_per_month: Starting day of each month of year.
            current_day: Current day.
            next_plant_day: Subarray map of next planting day.
            crop: Crops grown by each farmer. 
            field_indices_per_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices. 
            planting_schemes: A 6-dimensional array which contains the planing schemes.

               1. Whether the crop is irrigated or not.
               2. The unit code for the crop.
               3. The crop.
               4. The planting scheme (e.g., single or double cropping).
               5. If multicropping, the various crop stages.
               6. The planting month and the harvest month.
            irrigating: Whether each farmers irrigates or not (True is irrigating, False is not).
            unit_code: The unit code of each farmer.
            planting_scheme: The planting scheme for each farmer.
            next_multicrop_index: Subarray map of the next multicropping index.

        Returns:
            plant: Subarray map of what crops are plantn this day.
            crop_harvest_age_days: Subarray map of the havest age in days.
        """
        plant = np.full_like(next_plant_day, -1, dtype=np.int32)
        crop_harvest_age_days = np.full_like(next_plant_day, -1, dtype=np.int32)
        for i in range(n):
            farmer_fields = get_farmer_fields(field_indices, field_indices_per_farmer, i)
            for field in farmer_fields:
                if next_plant_day[field] != -1 and next_plant_day[field] == current_day:
                    plant[field] = crop[i]
                    crop_data = planting_schemes[irrigating[i], unit_code[i], crop[i], planting_scheme[i], next_multicrop_index[field]]
                    crop_harvest_age_days[field] = (start_day_per_month[crop_data[1] - 1] - start_day_per_month[crop_data[0] - 1]) % 365
        return plant, crop_harvest_age_days

    def plant(self) -> None:
        """Determines when and what crop should be planted, mainly through calling the :meth:`agents.farmers.Farmers.plant_numba`. Then converts the array to cupy array if model is running with GPU.
        """
        plant_map, crop_harvest_age_days = self.plant_numba(
            self.n,
            self.start_day_per_month,
            self.current_day_of_year,
            self.var.next_plant_day,
            self.crop.get() if self.model.args.use_gpu else self.crop,
            self.field_indices_per_farmer,
            self.field_indices,
            self.planting_schemes,
            self.irrigating,
            self.unit_code,
            self.planting_scheme,
            self.var.next_multicrop_index
        )
        if self.model.args.use_gpu:
            plant_map = cp.array(plant_map)
            crop_harvest_age_days = cp.array(crop_harvest_age_days)

        self.var.crop_map = np.where(plant_map >= 0, plant_map, self.var.crop_map)
        self.var.crop_age_days_map[plant_map >= 0] = 0
        self.var.crop_harvest_age_days_map[plant_map >= 0] = crop_harvest_age_days[plant_map >= 0]

        assert (self.var.crop_harvest_age_days_map[self.var.crop_map > 0] > 0).all()
        assert (self.var.crop_age_days_map[self.var.crop_map > 0] >= 0).all()

        field_is_paddy_irrigated = self.by_field(self.is_paddy_irrigated, nofieldvalue=0)
        self.var.land_use_type[(self.var.crop_map >= 0) & (field_is_paddy_irrigated == True)] = 2
        self.var.land_use_type[(self.var.crop_map >= 0) & (field_is_paddy_irrigated == False)] = 3
    
    @property
    def locations(self):
        return self._locations[:self.n]

    @locations.setter
    def locations(self, value):
        self._locations[:self.n] = value

    @property
    def is_paddy_irrigated(self):
        return self._is_paddy_irrigated[:self.n]

    @is_paddy_irrigated.setter
    def is_paddy_irrigated(self, value):
        self._is_paddy_irrigated[:self.n] = value

    @property
    def yield_ratio_per_farmer(self):
        return self._yield_ratio_per_farmer[:self.n]

    @yield_ratio_per_farmer.setter
    def yield_ratio_per_farmer(self, value):
        self._yield_ratio_per_farmer[:self.n] = value

    @property
    def crop(self):
        return self._crop[:self.n]

    @crop.setter
    def crop(self, value):      
        self._crop[:self.n] = value

    @property
    def latest_harvests(self):
        return self._latest_harvests[:self.n]

    @latest_harvests.setter
    def latest_harvests(self, value):      
        self._latest_harvests[:self.n] = value

    @property
    def irrigating(self):
        return self._irrigating[:self.n]

    @irrigating.setter
    def irrigating(self, value):      
        self._irrigating[:self.n] = value

    @property
    def unit_code(self):
        return self._unit_code[:self.n]

    @unit_code.setter
    def unit_code(self, value):      
        self._unit_code[:self.n] = value

    @property
    def planting_scheme(self):
        return self._planting_scheme[:self.n]

    @planting_scheme.setter
    def planting_scheme(self, value):      
        self._planting_scheme[:self.n] = value

    @property
    def is_water_efficient(self):
        return self._is_water_efficient[:self.n]

    @is_water_efficient.setter
    def is_water_efficient(self, value):      
        self._is_water_efficient[:self.n] = value

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

    @staticmethod
    @njit
    def field_size_per_farmer_numba(field_indices_per_farmer: np.ndarray, field_indices: np.ndarray, cell_area: np.ndarray) -> np.ndarray:
        """Gets the field size for each farmer.

        Args:
            field_indices_per_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.  
            cell_area: Subarray of cell_area.

        Returns:
            field_size_per_farmer: Field size for each farmer in m2.
        """
        field_size_per_farmer = np.zeros(field_indices_per_farmer.shape[0], dtype=np.float32)
        for farmer in range(field_indices_per_farmer.shape[0]):
            for field in get_farmer_fields(field_indices, field_indices_per_farmer, farmer):
                field_size_per_farmer[farmer] += cell_area[field]
        return field_size_per_farmer

    @property
    def field_size_per_farmer(self) -> np.ndarray:
        """Gets the field size for each farmer.
        
        Returns:
            field_size_per_farmer: Field size for each farmer in m2.
        """
        return self.field_size_per_farmer_numba(
            self.field_indices_per_farmer,
            self.field_indices,
            self.var.cellArea.get() if self.model.args.use_gpu else self.var.cellArea
        )

    def diffuse_water_efficiency_knowledge(self) -> None:
        """
        When this method is called, all farmers that have a high water efficiency, spread knowledge to a maximum of three other farmers within 5000 meter.
        """
        neighbors = find_neighbors(
            self.locations,
            5000,
            3,
            29,
            grid='longlat',
            search_ids=np.where(self.is_water_efficient)[0],
        )
        neighbors = neighbors[neighbors != -1]
        self.is_water_efficient[neighbors] = True

    def step(self) -> None:
        """
        This function is called at the beginning of each timestep. First, in the `ngo_training` scenario water efficiency knowledge diffuses through the farmer population. Only occurs each year on January 1st.

        Then, farmers harvest and plant crops.
        """
        if self.model.args.scenario == 'ngo_training' and self.model.current_time.month == 1 and self.model.current_time.day == 1:
            self.diffuse_water_efficiency_knowledge()

        self.harvest()
        self.plant()
        
    def add_agents(self):
        """This function can be used to add new farmers, but is not yet implemented."""
        raise NotImplementedError