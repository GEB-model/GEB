import numpy as np
import os
import pandas as pd
try:
    import cupy as cp
except ModuleNotFoundError:
    pass
import math

from numba import njit

from hyve.agents import AgentBaseClass
from hyve.library.neighbors import find_neighbors

@njit
def is_currently_growing(plant_month: int, harvest_month: int, current_month: int, current_day: int, start_day_per_month: np.ndarray) -> int:
    """Checks whether a crop is currently growing based on the sow month, harvest month, and current model date. Used for initalization of the model.

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

# @njit
def _get_farmer_fields(fields_per_farmer: np.ndarray, field_indices: np.ndarray, farmer_index: int):
    """Gets indices of field for the farmer
    
    Args:
        field_indices_per_farmer:
        
    """
    return fields_per_farmer[field_indices[farmer_index, 0]: field_indices[farmer_index, 1]]

@njit(cache=True)
def _set_fields(land_owners):
    agents = np.unique(land_owners)
    if agents[0] == -1:
        n_agents = agents.size -1
    else:
        n_agents = agents.size
    field_indices = np.full((n_agents, 2), -1, dtype=np.int32)
    fields_per_farmer = np.full(land_owners.size, -1, dtype=np.int32)

    land_owners_sort_idx = np.argsort(land_owners)
    land_owners_sorted = land_owners[land_owners_sort_idx]

    last_not_owned = np.searchsorted(land_owners_sorted, -1, side='right')

    prev_land_owner = -1
    for i in range(last_not_owned, land_owners.size):
        land_owner = land_owners[land_owners_sort_idx[i]]
        if land_owner != -1:
            if land_owner != prev_land_owner:
                field_indices[land_owner, 0] = i - last_not_owned
            field_indices[land_owner, 1] = i + 1 - last_not_owned
            fields_per_farmer[i - last_not_owned] = land_owners_sort_idx[i]
            prev_land_owner = land_owner
    fields_per_farmer = fields_per_farmer[:-last_not_owned]
    return field_indices, fields_per_farmer

@njit
def _abstract_water(
    activation_order,
    field_indices,
    fields_per_farmer,
    is_water_efficient,
    irrigated,
    cell_area,
    subvar_to_var,
    crop_map,
    totalPotIrrConsumption,
    available_channel_storage_m3,
    available_groundwater_m3,
    groundwater_head,
    available_reservoir_storage_m3,
    command_areas
):
    """
    Abstraction in order:
    1. Channel abstraction
    2. Reservoir abstraction (in m3)
    3. Groundwater abstraction

    In general the groundwater pumps are continously turned on due to the free availability of electricity. However
    there are many power outages, limiting the groundwater pumping. Irrigation decisions made by farmers are likely
    influenced when farmers are required to pay for the electricity. Here, we assume that farmers irrigate to ...

    Costs of irrigation
    - Labour costs
    - Material costs
    - Energy charges
    - Maintenance and repair charges
    - Land revenue charges
    - Water taxes (for lift and canal irrigation only: see https://sg.inflibnet.ac.in/bitstream/10603/36082/12/12_chapter_04.pdf)
    - Education tax
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
        farmer_fields = _get_farmer_fields(fields_per_farmer, field_indices, farmer)
        if is_water_efficient[farmer]:
            efficiency = 0.8
        else:
            efficiency = 0.6
        
        return_fraction = 0.5
        farmer_irrigation = irrigated[farmer]
        
        for field in farmer_fields:
            f_var = subvar_to_var[field]
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

def take_with_ignore(a, indices, ignore_index, ignore_value=np.nan):
    array = np.take(a, indices)
    array[indices == ignore_index] = ignore_value
    return array


class Farmers(AgentBaseClass):
    """Test"""
    def __init__(self, model, agents, reduncancy):
        self.redundancy = reduncancy
        self.input_folder = 'DataDrive/GEB/input'
        AgentBaseClass.__init__(self, model, agents)
        self.crop_yield_factors = self.get_crop_yield_factors()

    def _initiate_agents(self):
        self.initiate_locations()
        self.initiate_attributes()
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
        self._elevation = np.zeros(self.max_n, dtype=np.float32)
        self.elevation = self.model.data.elevation.sample_coords(self.locations)
        crop_file = os.path.join('DataDrive', 'GEB', 'input', 'agents', 'crop.npy')
        if self.model.config['general']['use_gpu']:
            self._crop = cp.load(crop_file)
        else:
            self._crop = np.load(crop_file)
        self._irrigating = np.zeros(self.max_n, dtype=np.int8)
        self.irrigating = np.load(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'irrigating.npy'))
        self._planting_scheme = np.zeros(self.max_n, dtype=np.int8)
        self.planting_scheme = np.load(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'planting_scheme.npy'))
        self._unit_code = np.zeros(self.max_n, dtype=np.int32)
        self.unit_code = np.load(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'farmer_unit_codes.npy'))
        if self.model.config['general']['use_gpu']:
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

        self.planting_schemes = np.load(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'planting_schemes.npy'))

    def set_fields(self, *args, **kwargs):
        self.field_indices, self.fields_per_farmer = _set_fields(*args, **kwargs)
    
    @staticmethod
    @njit
    def test_farmers(array, field_indices, fields_per_farmer):
        for i in range(field_indices.shape[0]):
            fields = fields_per_farmer[field_indices[i, 0]: field_indices[i, 1]]
            for field in fields:
                array[field] = i
        return array

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

    def abstract_water(self, *args, **kwargs):
        activation_order = self.activation_order_by_elevation
        (
            self.channel_abstraction_m3_by_farmer,
            self.reservoir_abstraction_m3_by_farmer,
            self.groundwater_abstraction_m3_by_farmer,
            *values
        ) = _abstract_water(
            activation_order,
            self.field_indices,
            self.fields_per_farmer,
            self.is_water_efficient,
            self.irrigating,
            *args,
            **kwargs
        )
        return values

    def get_crop_factors(self):
        # https://doi.org/10.1016/j.jhydrol.2009.07.031
        df = pd.read_csv(os.path.join('DataDrive', 'GEB', 'input', 'agents', 'crop_factors.csv'))
        df['L_dev'] = df['L_ini'] + df['L_dev']
        df['L_mid'] = df['L_dev'] + df['L_mid']
        df['L_late'] = df['L_mid'] + df['L_late']
        assert np.allclose(df['L_late'], 1.0)
        return df

    def get_crop_yield_factors(self) -> dict[np.ndarray]:
        """Read csv-file of values for crop water depletion. Obtained from Table 2 of this paper: https://doi.org/10.1029/2008GB003435
        
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
    def _get_yield_ratio(crop_map: np.array, evap_ratios: np.array, alpha: np.array, beta: np.array, P0: np.array, P1: np.array) -> float:
        """Calculate yield ratio based on https://doi.org/10.1016/j.jhydrol.2009.07.031
        
        Args:
            crop_map: array of currently harvested crops.
            evap_ratios: ratio of actual to potential evapotranspiration.
            alpha: alpha value per crop used in MIRCA2000.
            beta: beta value per crop used in MIRCA2000.
            P0: P0 value per crop used in MIRCA2000.
            P1: P1 value per crop used in MIRCA2000.
        
        Returns:
            yield_ratios: yield ratio (as ratio of maximum obtainable yield) per harvested crop.
        """
        yield_ratios = np.empty(evap_ratios.size, dtype=np.float32)
        
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

    def get_yield_ratio(self, harvest, actual_transpiration, potential_transpiration, crop_map):
        print("check crop map, might be an error here")
        return self._get_yield_ratio(
            crop_map,
            actual_transpiration[harvest] / potential_transpiration[harvest],
            self.crop_yield_factors['alpha'],
            self.crop_yield_factors['beta'],
            self.crop_yield_factors['P0'],
            self.crop_yield_factors['P1'],
        )

    def save_harvest(self, harvesting_farmers):
        self.latest_harvests[harvesting_farmers, 1:] = self.latest_harvests[harvesting_farmers, 0:-1]
        self.latest_harvests[harvesting_farmers, 0] = self.yield_ratio_per_farmer[harvesting_farmers]

    def invest_in_water_efficiency(self, harvesting_farmers):
        if self.model.args.scenario == 'self_investment':
            invest = (
                self.latest_harvests[harvesting_farmers, 0] < np.mean(self.latest_harvests[harvesting_farmers, 1:], axis=1)
            )
            self.is_water_efficient[harvesting_farmers] |= invest

    @staticmethod
    @njit
    def get_crop_age_and_harvest_day_initial(n, current_month, current_day, planting_schemes, crop, irrigating, unit_code, planting_scheme, start_day_per_month, days_in_year=365):
        crop_age_days = np.full(n, -1, dtype=np.int32)
        crop_harvest_age_days = np.full(n, -1, dtype=np.int32)
        next_sow_day = np.full(n, -1, dtype=np.int32)
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
                    next_sow_day[i] = start_day_per_month[farmer_planting_scheme[j, 0] - 1]
                    next_multicrop_index[i] = 0
                    break
            else:  # no crops are currently planted, so let's find out the first crop with a sow day
                assert n_cropping_periods > 1
                for j in range(n_cropping_periods):
                    if j == 0:  # if first option, just set it as the next sow day
                        next_sow_day[i] = start_day_per_month[farmer_planting_scheme[j, 0] - 1]
                    else:  # if not, find the first option starting from the current day
                        potential_next_sow_day = start_day_per_month[farmer_planting_scheme[j, 0] - 1]
                        next_sow_day[i] = ((min((potential_next_sow_day - current_day) % days_in_year, (next_sow_day[i] - current_day) % days_in_year)) + current_day) % days_in_year

                for j in range(n_cropping_periods):
                    if start_day_per_month[farmer_planting_scheme[j, 0] - 1] == next_sow_day[i]:
                        next_multicrop_index[i] = j
                        break
                else:  # make sure one of the options is picked
                    assert False

            assert crop_harvest_age_days[i] != -1 or next_sow_day[i] != -1

        assert (next_multicrop_index != -1).all()
        assert (n_multicrop_periods != -1).all()

        return crop_age_days, crop_harvest_age_days, next_sow_day, n_multicrop_periods, next_multicrop_index
        
    def sow_initial(self):
        if self.model.config['general']['use_gpu']:
            crop = self.crop.get()
        else:
            crop = self.crop.copy()

        crop_age_days, crop_harvest_age_days, next_sow_day, n_multicrop_periods, next_multicrop_index = self.get_crop_age_and_harvest_day_initial(self.n, self.model.current_time.month, self.current_day_of_year, self.planting_schemes, crop, self.irrigating, self.unit_code, self.planting_scheme, self.start_day_per_month)
        assert np.logical_xor((next_sow_day == -1), (crop_harvest_age_days == -1)).all()
        if self.model.config['general']['use_gpu']:
            crop_age_days = cp.array(crop_age_days)
            crop_harvest_age_days = cp.array(crop_harvest_age_days)
            fields = self.fields.get()
        else:
            fields = self.fields

        sow = self.crop.copy()
        sow[crop_age_days == -1] = -1
        sow = np.take(sow, self.fields)
        sow[self.fields == -1] = -1
        
        crop_age_days = np.take(crop_age_days, self.fields)
        crop_age_days[self.fields == -1] = -1
        
        crop_harvest_age_days = np.take(crop_harvest_age_days, self.fields)
        crop_harvest_age_days[self.fields == -1] = -1

        self.next_sow_day = np.take(next_sow_day, fields)
        self.next_sow_day[fields == -1] = -1

        self.n_multicrop_periods = np.take(n_multicrop_periods, fields)
        self.n_multicrop_periods[fields == -1] = -1

        self.next_multicrop_index = np.take(next_multicrop_index, fields)
        self.next_multicrop_index[fields == -1] = -1

        assert (crop_harvest_age_days[crop_age_days >= 0] != -1).all()
        assert sow.shape == crop_age_days.shape == crop_harvest_age_days.shape

        assert (crop_harvest_age_days[sow > 0] > 0).all()
        
        return sow, crop_age_days, crop_harvest_age_days

    @staticmethod
    @njit
    def _harvest(
        start_day_per_month,
        activation_order,
        field_indices,
        fields_per_farmer,
        crop_map,
        crop_age_days,
        crop_harvest_age_days,
        next_sow_day,
        next_multicrop_index,
        n_multicrop_periods,
        planting_schemes,
        irrigating,
        unit_code,
        crop,
        planting_scheme
    ):
        harvest = np.zeros(crop_map.shape, dtype=np.bool_)
        for i in range(activation_order.size):
            farmer_fields = _get_farmer_fields(fields_per_farmer, field_indices, i)
            for field in farmer_fields:
                crop_age = crop_age_days[field]
                if crop_age >= 0:
                    assert crop_map[field] != -1
                    assert crop_harvest_age_days[field] != -1
                    if crop_age == crop_harvest_age_days[field]:
                        harvest[field] = True
                        next_multicrop_index[field] = (next_multicrop_index[field] + 1) % n_multicrop_periods[field]
                        assert next_multicrop_index[field] < n_multicrop_periods[field] 
                        next_sow_month = planting_schemes[irrigating[i], unit_code[i], crop[i], planting_scheme[i], next_multicrop_index[field]]
                        assert next_sow_month[0] != -1
                        next_sow_day[field] = start_day_per_month[next_sow_month - 1][0]
                else:
                    assert crop_map[field] == -1
                    assert crop_harvest_age_days[field] == -1
        return harvest
        
    def harvest(self, actual_transpiration, potential_transpiration, crop_map, crop_age_days, crop_harvest_age_days):
        harvest = self._harvest(
            self.start_day_per_month,
            self.activation_order_random,
            self.field_indices,
            self.fields_per_farmer,
            crop_map,
            crop_age_days,
            crop_harvest_age_days,
            self.next_sow_day,
            self.next_multicrop_index,
            self.n_multicrop_periods,
            self.planting_schemes,
            self.irrigating,
            self.unit_code,
            self.crop.get() if self.model.config['general']['use_gpu'] else self.crop,
            self.planting_scheme
        )
        if np.count_nonzero(harvest):
            yield_ratio = self.get_yield_ratio(harvest, actual_transpiration, potential_transpiration, crop_map)
            fields = self.fields.get() if self.model.config['general']['use_gpu'] else self.fields
            harvesting_farmer_fields = fields[harvest]
            self.yield_ratio_per_farmer = np.bincount(harvesting_farmer_fields, weights=yield_ratio, minlength=self.n)
            harvesting_farmers = np.unique(harvesting_farmer_fields)
            if self.model.current_timestep > 365:
                self.save_harvest(harvesting_farmers)
            self.invest_in_water_efficiency(harvesting_farmers)
        if self.model.config['general']['use_gpu']:
            harvest = cp.array(harvest)
        return harvest

    @staticmethod
    @njit
    def _sow(
        n,
        start_day_per_month,
        current_day,
        next_sow_day,
        crop,
        fields_per_farmer,
        field_indices,
        planting_schemes,
        irrigating,
        unit_code,
        planting_scheme,
        next_multicrop_index
    ):
        sow = np.full_like(next_sow_day, -1, dtype=np.int32)
        harvest_age = np.full_like(next_sow_day, -1, dtype=np.int32)
        for i in range(n):
            farmer_fields = _get_farmer_fields(fields_per_farmer, field_indices, i)
            for field in farmer_fields:
                if next_sow_day[field] != -1 and next_sow_day[field] == current_day:
                    sow[field] = crop[i]
                    crop_data = planting_schemes[irrigating[i], unit_code[i], crop[i], planting_scheme[i], next_multicrop_index[field]]
                    harvest_age[field] = (start_day_per_month[crop_data[1] - 1] - start_day_per_month[crop_data[0] - 1]) % 365
        return sow, harvest_age

    def sow(self):
        sow, harvest_day = self._sow(
            self.n,
            self.start_day_per_month,
            self.current_day_of_year,
            self.next_sow_day,
            self.crop.get() if self.model.config['general']['use_gpu'] else self.crop,
            self.fields_per_farmer,
            self.field_indices,
            self.planting_schemes,
            self.irrigating,
            self.unit_code,
            self.planting_scheme,
            self.next_multicrop_index
        )
        if self.model.config['general']['use_gpu']:
            sow = cp.array(sow)
            harvest_day = cp.array(harvest_day)
        return sow, harvest_day
    
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
    def field_is_paddy_irrigated(self):
        return np.take(self.is_paddy_irrigated, self.fields)

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

    @staticmethod
    @njit
    def _field_size_per_farmer(field_indices, fields_per_farmer, cell_area):
        field_size_per_farmer = np.zeros(field_indices.shape[0], dtype=np.float32)
        for farmer in range(field_indices.shape[0]):
            for field in _get_farmer_fields(fields_per_farmer, field_indices, farmer):
                field_size_per_farmer[farmer] += cell_area[field]
        return field_size_per_farmer

    @property
    def field_size_per_farmer(self):
        return self._field_size_per_farmer(
            self.field_indices,
            self.fields_per_farmer,
            self.model.subvar.cellArea.get() if self.model.config['general']['use_gpu'] else self.model.subvar.cellArea
        )

    def diffuse_water_efficiency_knowledge(self):
        neighbors = find_neighbors(
            self.locations,
            np.where(self.is_water_efficient)[0],
            5000,
            3,
            29
        )
        neighbors = neighbors[neighbors != -1]
        self.is_water_efficient[neighbors] = True

    def process(self):
        if self.model.args.scenario == 'ngo_training' and self.model.current_time.month == 1 and self.model.current_time.day == 1:
            self.diffuse_water_efficiency_knowledge()
        
    def step(self):
        self.process()

    def add_agents(self):
        raise NotImplementedError

class NGO(AgentBaseClass):
    def __init__(self, model, agents):
        AgentBaseClass.__init__(self, model, agents)

    def _initiate_agents(self): pass

    def provide_training(self):
        rng = np.random.default_rng()
        trained_farmers = rng.choice(self.agents.farmers.is_water_efficient.size, size=100_000, replace=False)
        self.agents.farmers.is_water_efficient[trained_farmers] = True

    def step(self):
        if self.model.current_timestep == 0 and self.model.args.scenario == 'ngo_training':
            self.provide_training()

class Government(AgentBaseClass):
    def __init__(self, model, agents):
        AgentBaseClass.__init__(self, model, agents)

    def _initiate_agents(self): pass

    def provide_subsidies(self):
        total_water_use = self.agents.farmers.reservoir_abstraction_m3_by_farmer + self.agents.farmers.groundwater_abstraction_m3_by_farmer + self.agents.farmers.channel_abstraction_m3_by_farmer
        total_water_use_m = total_water_use / self.agents.farmers.field_size_per_farmer
        n_farmers_to_upgrade = self.agents.farmers.n // 20

        farmer_indices = np.arange(0, self.agents.farmers.n)
        indices_not_yet_water_efficient = farmer_indices[~self.agents.farmers.is_water_efficient]

        if indices_not_yet_water_efficient.size <= n_farmers_to_upgrade:
            self.agents.farmers.is_water_efficient[:] = True
        else:
            total_water_use_m_not_yet_water_efficient = total_water_use_m[~self.agents.farmers.is_water_efficient]
            self.agents.farmers.is_water_efficient[indices_not_yet_water_efficient[np.argpartition(-total_water_use_m_not_yet_water_efficient, n_farmers_to_upgrade)[:n_farmers_to_upgrade]]] = True

    def step(self):
        if self.model.current_time.day == 1 and self.model.current_time.month == 1 and self.model.current_timestep != 0 and self.model.args.scenario == 'government_subsidies':
            self.provide_subsidies()

class Agents:
    def __init__(self, model):
        self.model = model
        self.farmers = Farmers(model, self, 0.1)
        self.ngo = NGO(model, self)
        self.government = Government(model, self)

    def step(self):
        self.ngo.step()
        self.government.step()
        self.farmers.step()
