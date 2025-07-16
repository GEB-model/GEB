# -*- coding: utf-8 -*-
import calendar
import copy
import math
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from honeybees.library.neighbors import find_neighbors
from honeybees.library.raster import pixels_to_coords, sample_from_map
from numba import njit
from scipy.optimize import curve_fit
from scipy.stats import genextreme

from geb.workflows import TimingModule

from ..data import (
    load_crop_data,
    load_economic_data,
    load_regional_crop_data_from_dict,
)
from ..hydrology.HRUs import load_grid
from ..hydrology.landcover import GRASSLAND_LIKE, NON_PADDY_IRRIGATED, PADDY_IRRIGATED
from ..store import DynamicArray
from ..workflows import balance_check
from ..workflows.io import load_array
from .decision_module import DecisionModule
from .general import AgentBaseClass
from .workflows.crop_farmers import (
    abstract_water,
    compute_premiums_and_best_contracts_numba,
    crop_profit_difference_njit_parallel,
    farmer_command_area,
    find_most_similar_index,
    get_farmer_groundwater_depth,
    get_farmer_HRUs,
    get_gross_irrigation_demand_m3,
    plant,
)

NO_IRRIGATION: int = -1
CHANNEL_IRRIGATION: int = 0
RESERVOIR_IRRIGATION: int = 1
GROUNDWATER_IRRIGATION: int = 2
TOTAL_IRRIGATION: int = 3

SURFACE_IRRIGATION_EQUIPMENT: int = 0
WELL_ADAPTATION: int = 1
IRRIGATION_EFFICIENCY_ADAPTATION: int = 2
FIELD_EXPANSION_ADAPTATION: int = 3
PERSONAL_INSURANCE_ADAPTATION: int = 4
INDEX_INSURANCE_ADAPTATION: int = 5


def cumulative_mean(mean, counter, update, mask=None):
    """Calculates the cumulative mean of a series of numbers. This function operates in place.

    Args:
        mean: The cumulative mean.
        counter: The number of elements that have been added to the mean.
        update: The new elements that needs to be added to the mean.
        mask: A mask that indicates which elements should be updated. If None, all elements are updated.

    """
    if mask is not None:
        mean[mask] = (mean[mask] * counter[mask] + update[mask]) / (counter[mask] + 1)
        counter[mask] += 1
    else:
        mean[:] = (mean * counter + update) / (counter + 1)
        counter += 1


def shift_and_update(array, update):
    """Shifts the array and updates the first element with the update value.

    Args:
        array: The array that needs to be shifted.
        update: The value that needs to be added to the first element of the array.
    """
    array[:, 1:] = array[:, :-1]
    array[:, 0] = update


def shift_and_reset_matrix(matrix: np.ndarray) -> None:
    """Shifts columns to the right in the matrix and sets the first column to zero."""
    matrix[:, 1:] = matrix[:, 0:-1]  # Shift columns to the right
    matrix[:, 0] = 0  # Reset the first column to 0


def advance_crop_rotation_year(
    current_crop_calendar_rotation_year_index: np.ndarray,
    crop_calendar_rotation_years: np.ndarray,
):
    """Update the crop rotation year for each farmer. This function is used to update the crop rotation year for each farmer at the end of the year.

    Args:
        current_crop_calendar_rotation_year_index: The current crop rotation year for each farmer.
        crop_calendar_rotation_years: The number of years in the crop rotation cycle for each farmer.
    """
    current_crop_calendar_rotation_year_index[:] = (
        current_crop_calendar_rotation_year_index + 1
    ) % crop_calendar_rotation_years


class CropFarmers(AgentBaseClass):
    """The agent class for the farmers. Contains all data and behaviourial methods. The __init__ function only gets the model as arguments, the agent parent class and the redundancy. All other variables are loaded at later stages.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
        redundancy: a lot of data is saved in pre-allocated NumPy arrays. While this allows much faster operation, it does mean that the number of agents cannot grow beyond the size of the pre-allocated arrays. This parameter allows you to specify how much redundancy should be used. A lower redundancy means less memory is used, but the model crashes if the redundancy is insufficient.
    """

    def __init__(self, model, agents, reduncancy: float) -> None:
        super().__init__(model)
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["farmers"]
            if "farmers" in self.model.config["agent_settings"]
            else {}
        )
        if self.model.simulate_hydrology:
            self.HRU = model.hydrology.HRU
            self.grid = model.hydrology.grid

        self.redundancy = reduncancy
        self.decision_module = DecisionModule(self)

        self.inflation_rate = load_economic_data(
            self.model.files["dict"]["socioeconomics/inflation_rates"]
        )
        # self.lending_rate = load_economic_data(
        #     self.model.files["dict"]["socioeconomics/lending_rates"]
        # )
        self.electricity_cost = load_economic_data(
            self.model.files["dict"]["socioeconomics/electricity_cost"]
        )

        self.why_10 = load_economic_data(
            self.model.files["dict"]["socioeconomics/why_10"]
        )
        self.why_20 = load_economic_data(
            self.model.files["dict"]["socioeconomics/why_20"]
        )
        self.why_30 = load_economic_data(
            self.model.files["dict"]["socioeconomics/why_30"]
        )

        self.crop_prices = load_regional_crop_data_from_dict(
            self.model, "crops/crop_prices"
        )

        self.cultivation_costs = load_regional_crop_data_from_dict(
            self.model, "crops/cultivation_costs"
        )

        self.adjust_cultivation_costs()

        if self.model.in_spinup:
            self.spinup()

        # ruleset variables
        self.wells_adaptation_active = (
            not self.config["expected_utility"]["adaptation_well"]["ruleset"]
            == "no-adaptation"
        )
        self.sprinkler_adaptation_active = (
            not self.config["expected_utility"]["adaptation_sprinkler"]["ruleset"]
            == "no-adaptation"
        )
        self.crop_switching_adaptation_active = (
            not self.config["expected_utility"]["crop_switching"]["ruleset"]
            == "no-adaptation"
        )
        self.personal_insurance_adaptation_active = (
            not self.config["expected_utility"]["insurance"]["personal_insurance"][
                "ruleset"
            ]
            == "no-adaptation"
        )
        self.index_insurance_adaptation_active = (
            not self.config["expected_utility"]["insurance"]["index_insurance"][
                "ruleset"
            ]
            == "no-adaptation"
        )
        self.microcredit_adaptation_active = (
            not self.config["microcredit"]["ruleset"] == "no-adaptation"
        )

    @property
    def name(self):
        return "agents.crop_farmers"

    def spinup(self):
        self.var.crop_data_type, self.var.crop_data = load_crop_data(self.model.files)
        self.var.crop_ids = self.var.crop_data["name"].to_dict()
        # reverse dictionary
        self.var.crop_names = {
            crop_name: crop_id for crop_id, crop_name in self.var.crop_ids.items()
        }

        ## Set parameters required for drought event perception, risk perception and SEUT
        self.var.moving_average_threshold = self.model.config["agent_settings"][
            "farmers"
        ]["expected_utility"]["drought_risk_calculations"]["event_perception"][
            "drought_threshold"
        ]
        self.var.previous_month = 0

        # Assign risk aversion sigma, time discounting preferences, expenditure_cap
        self.var.expenditure_cap = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["decisions"]["expenditure_cap"]

        # New global well variables
        self.var.pump_hours = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["pump_hours"]
        self.var.specific_weight_water = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["specific_weight_water"]
        self.var.max_initial_sat_thickness = self.model.config["agent_settings"][
            "farmers"
        ]["expected_utility"]["adaptation_well"]["max_initial_sat_thickness"]
        self.var.lifespan_well = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["lifespan"]
        self.var.pump_efficiency = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["pump_efficiency"]
        self.var.maintenance_factor = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["maintenance_factor"]

        self.var.insurance_duration = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["insurance"]["duration"]
        self.var.p_droughts = np.array([100, 50, 25, 10, 5, 2, 1])

        # Set water costs
        self.var.water_costs_m3_channel = 0.20
        self.var.water_costs_m3_reservoir = 0.20
        self.var.water_costs_m3_groundwater = 0.20

        # Irr efficiency variables
        self.var.lifespan_irrigation = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["lifespan"]

        # load map of all subdistricts
        self.var.subdistrict_map = load_grid(
            self.model.files["region_subgrid"]["region_ids"]
        )
        region_mask = load_grid(self.model.files["region_subgrid"]["mask"])
        self.HRU_regions_map = np.zeros_like(self.HRU.mask, dtype=np.int8)
        self.HRU_regions_map[~self.HRU.mask] = self.var.subdistrict_map[
            region_mask == 0
        ]
        self.HRU_regions_map = self.HRU.compress(self.HRU_regions_map)

        self.crop_prices = load_regional_crop_data_from_dict(
            self.model, "crops/crop_prices"
        )

        self.adjust_cultivation_costs()

        # Test with a high variable for now
        self.var.total_spinup_time = max(
            self.model.config["general"]["start_time"].year
            - self.model.config["general"]["spinup_time"].year,
            30,
        )

        self.HRU.var.actual_evapotranspiration_crop_life = self.HRU.full_compressed(
            0, dtype=np.float32
        )
        self.HRU.var.potential_evapotranspiration_crop_life = self.HRU.full_compressed(
            0, dtype=np.float32
        )
        self.HRU.var.crop_map = np.full_like(self.HRU.var.land_owners, -1)
        self.HRU.var.crop_age_days_map = np.full_like(self.HRU.var.land_owners, -1)
        self.HRU.var.crop_harvest_age_days = np.full_like(self.HRU.var.land_owners, -1)

        """Calls functions to initialize all agent attributes, including their locations. Then, crops are initially planted."""
        # If initial conditions based on spinup period need to be loaded, load them. Otherwise, generate them.

        farms = self.model.hydrology.farms

        # Get number of farmers and maximum number of farmers that could be in the entire model run based on the redundancy.
        self.var.n = np.unique(farms[farms != -1]).size
        self.var.max_n = self.get_max_n(self.var.n)

        # The code below obtains the coordinates of the farmers' locations.
        # First the horizontal and vertical indices of the pixels that are not -1 are obtained. Then, for each farmer the
        # average of the horizontal and vertical indices is calculated. This is done by using the bincount function.
        # Finally, the coordinates are obtained by adding .5 to the pixels and converting them to coordinates using pixel_to_coord.
        vertical_index = (
            np.arange(farms.shape[0])
            .repeat(farms.shape[1])
            .reshape(farms.shape)[farms != -1]
        )
        horizontal_index = np.tile(np.arange(farms.shape[1]), farms.shape[0]).reshape(
            farms.shape
        )[farms != -1]
        pixels = np.zeros((self.var.n, 2), dtype=np.int32)
        pixels[:, 0] = np.round(
            np.bincount(farms[farms != -1], horizontal_index)
            / np.bincount(farms[farms != -1])
        ).astype(int)
        pixels[:, 1] = np.round(
            np.bincount(farms[farms != -1], vertical_index)
            / np.bincount(farms[farms != -1])
        ).astype(int)

        self.var.locations = DynamicArray(
            pixels_to_coords(pixels + 0.5, self.HRU.gt), max_n=self.var.max_n
        )

        self.set_social_network()

        self.var.risk_aversion = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=np.nan
        )
        self.var.risk_aversion[:] = load_array(
            self.model.files["array"]["agents/farmers/risk_aversion"]
        )

        self.var.discount_rate = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=np.nan
        )
        self.var.discount_rate[:] = load_array(
            self.model.files["array"]["agents/farmers/discount_rate"]
        )

        self.var.intention_factor = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=np.nan
        )

        self.var.intention_factor[:] = load_array(
            self.model.files["array"]["agents/farmers/intention_factor"]
        )

        self.var.interest_rate = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=0.05
        )
        self.var.interest_rate[:] = load_array(
            self.model.files["array"]["agents/farmers/interest_rate"]
        )

        # Load the region_code of each farmer.
        self.var.region_id = DynamicArray(
            input_array=load_array(
                self.model.files["array"]["agents/farmers/region_id"]
            ),
            max_n=self.var.max_n,
        )

        self.var.elevation = self.get_farmer_elevation()

        self.var.crop_calendar = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(3, 4),
            extra_dims_names=("rotation", "calendar"),
            dtype=np.int32,
            fill_value=-1,
        )  # first dimension is the farmers, second is the rotation, third is the crop, planting and growing length
        self.var.crop_calendar[:] = load_array(
            self.model.files["array"]["agents/farmers/crop_calendar"]
        )
        # assert self.var.crop_calendar[:, :, 0].max() < len(self.var.crop_ids)

        self.var.crop_calendar_rotation_years = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.int32,
            fill_value=0,
        )
        self.var.crop_calendar_rotation_years[:] = load_array(
            self.model.files["array"]["agents/farmers/crop_calendar_rotation_years"]
        )

        self.var.current_crop_calendar_rotation_year_index = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.int32,
            fill_value=0,
        )
        # For each farmer set a random crop rotation year. The farmer starts in that year. First set a seed for reproducibility.
        np.random.seed(42)
        self.var.current_crop_calendar_rotation_year_index[:] = np.random.randint(
            0, self.var.crop_calendar_rotation_years
        )

        self.var.adaptations = DynamicArray(
            load_array(self.model.files["array"]["agents/farmers/adaptations"]),
            max_n=self.var.max_n,
            extra_dims_names=("adaptation_type",),
        )

        # the time each agent has been paying off their loan
        # 0 = no cost adaptation, 1 = well, 2 = irr efficiency, 3 = irr. field expansion  -1 if they do not have adaptations
        self.var.time_adapted = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=self.var.adaptations.shape[1:],
            extra_dims_names=self.var.adaptations.extra_dims_names,
            dtype=np.int32,
            fill_value=-1,
        )

        # Set the initial well depth
        self.var.well_depth = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["adaptation_well"]["max_initial_sat_thickness"],
            dtype=np.float32,
        )

        # Set how long the agents have adapted somewhere across the lifespan of farmers, would need to be a bit more realistic likely
        rng_wells = np.random.default_rng(17)
        self.var.time_adapted[
            self.var.adaptations[:, WELL_ADAPTATION] == 1, WELL_ADAPTATION
        ] = rng_wells.uniform(
            1,
            self.var.lifespan_well,
            np.sum(self.var.adaptations[:, WELL_ADAPTATION] == 1),
        )

        # Initiate a number of arrays with Nan, zero or -1 values for variables that will be used during the model run.
        self.var.channel_abstraction_m3_by_farmer = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=0
        )
        self.var.reservoir_abstraction_m3_by_farmer = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=0
        )
        self.var.groundwater_abstraction_m3_by_farmer = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=0
        )

        # 2D-array for storing yearly abstraction by farmer. 0: channel abstraction, 1: reservoir abstraction, 2: groundwater abstraction, 3: total abstraction
        self.var.yearly_abstraction_m3_by_farmer = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(4, self.var.total_spinup_time),
            extra_dims_names=("abstraction_type", "year"),
            dtype=np.float32,
            fill_value=0,
        )

        self.var.max_paddy_water_level = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=0.05,
        )

        self.var.cumulative_SPEI_during_growing_season = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=0,
        )
        self.var.cumulative_SPEI_count_during_growing_season = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.int32,
            fill_value=0,
        )

        # set no irrigation limit for farmers by default
        self.var.irrigation_limit_m3 = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=np.nan,  # m3
        )
        # set the remaining irrigation limit to the irrigation limit
        self.var.remaining_irrigation_limit_m3 = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, fill_value=np.nan, dtype=np.float32
        )

        self.var.yield_ratios_drought_event = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.p_droughts.size,),
            extra_dims_names=("drought_event",),
            dtype=np.float32,
            fill_value=0,
        )

        self.var.actual_yield_per_farmer = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=np.nan,
        )

        self.var.harvested_crop = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.int32,
            fill_value=-1,
        )

        ## Risk perception variables
        self.var.risk_perception = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["drought_risk_calculations"]["risk_perception"]["min"],
        )
        self.var.drought_timer = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=99
        )

        self.var.yearly_SPEI_probability = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        self.var.yearly_SPEI = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        self.var.yearly_SPEI = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        self.var.yearly_yield_ratio = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        # note that this is NOT inflation corrected
        self.var.yearly_income = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        self.var.insured_yearly_income = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        # note that this is NOT inflation corrected
        self.var.yearly_potential_income = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        self.var.farmer_yield_probability_relation = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(2,),
            extra_dims_names=("log function parameters",),
            dtype=np.float32,
            fill_value=0,
        )

        self.var.household_size = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.int32, fill_value=-1
        )
        self.var.household_size[:] = load_array(
            self.model.files["array"]["agents/farmers/household_size"]
        )

        self.var.yield_ratios_drought_event = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.p_droughts.size,),
            extra_dims_names=("drought_event",),
            dtype=np.float32,
            fill_value=0,
        )

        # Set irrigation efficiency data
        irrigation_mask = self.is_irrigated
        self.var.irrigation_efficiency = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=0.50
        )

        rng = np.random.default_rng(42)
        self.var.irrigation_efficiency[irrigation_mask] = rng.choice(
            [0.50, 0.90], size=irrigation_mask.sum(), p=[0.8, 0.2]
        )
        self.var.adaptations[:, IRRIGATION_EFFICIENCY_ADAPTATION][
            self.var.irrigation_efficiency >= 0.90
        ] = 1
        rng_drip = np.random.default_rng(70)
        self.var.time_adapted[
            self.var.adaptations[:, IRRIGATION_EFFICIENCY_ADAPTATION] == 1,
            IRRIGATION_EFFICIENCY_ADAPTATION,
        ] = rng_drip.uniform(
            1,
            self.var.lifespan_irrigation,
            np.sum(self.var.adaptations[:, IRRIGATION_EFFICIENCY_ADAPTATION] == 1),
        )

        # Set irrigation expansion data
        self.var.fraction_irrigated_field = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=1
        )
        self.var.adaptations[:, FIELD_EXPANSION_ADAPTATION][
            self.var.fraction_irrigated_field >= 1
        ] = 1

        self.var.base_management_yield_ratio = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "base_management_yield_ratio"
            ],
        )

        # Set insurance adaptation data (placeholder)
        rng_personal_insurance = np.random.default_rng(15)
        mask_personal_insurance = rng_personal_insurance.random(self.var.n) < 0.25
        self.var.adaptations[:, PERSONAL_INSURANCE_ADAPTATION][
            mask_personal_insurance
        ] = 1

        free_idx = np.flatnonzero(
            self.var.adaptations[:, PERSONAL_INSURANCE_ADAPTATION] == -1
        )
        num_index: int = int(self.var.n * 0.25)
        rng_index_insurance = np.random.default_rng(60)
        mask_index_insurance = rng_index_insurance.choice(
            free_idx, size=num_index, replace=False
        )
        self.var.adaptations[:, INDEX_INSURANCE_ADAPTATION][mask_index_insurance] = 1

        self.var.time_adapted[
            self.var.adaptations[:, PERSONAL_INSURANCE_ADAPTATION] == 1,
            PERSONAL_INSURANCE_ADAPTATION,
        ] = 0

        self.var.time_adapted[
            self.var.adaptations[:, INDEX_INSURANCE_ADAPTATION] == 1,
            INDEX_INSURANCE_ADAPTATION,
        ] = 0

        # Initiate array that tracks the overall yearly costs for all adaptations
        # 0 is input, 1 is microcredit, 2 is adaptation 1 (well), 3 is adaptation 2 (drip irrigation), 4 irr. field expansion, 5 is water costs, last is total
        # Columns are the individual loans, i.e. if there are 2 loans for 2 wells, the first and second slot is used

        self.var.n_loans = self.var.adaptations.shape[1] + 2

        self.var.all_loans_annual_cost = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.n_loans + 1, 5),
            extra_dims_names=("loan_type", "loans"),
            dtype=np.float32,
            fill_value=0,
        )

        self.var.adjusted_annual_loan_cost = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.n_loans + 1, 5),
            extra_dims_names=("loan_type", "loans"),
            dtype=np.float32,
            fill_value=np.nan,
        )

        # 0 is input, 1 is microcredit, 2 is adaptation 1 (well), 3 is adaptation 2 (drip irrigation)
        self.var.loan_tracker = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(self.var.n_loans, 5),
            extra_dims_names=("loan_type", "loans"),
            dtype=np.int32,
            fill_value=0,
        )

        self.var.farmer_base_class = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.int32, fill_value=-1
        )
        self.var.water_use = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(4,),
            extra_dims_names=("water_source",),
            dtype=np.int32,
            fill_value=0,
        )

        # Load the why class of agent's aquifer
        self.var.why_class = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.int32,
            fill_value=0,
        )

        why_map: np.ndarray = load_grid(self.model.files["grid"]["groundwater/why_map"])

        self.var.why_class[:] = sample_from_map(
            why_map, self.var.locations.data, self.grid.gt
        )

        ## Load in the GEV_parameters, calculated from the extreme value distribution of the SPEI timeseries, and load in the original SPEI data
        self.var.GEV_parameters = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(3,),
            extra_dims_names=("gev_parameters",),
            dtype=np.float32,
            fill_value=np.nan,
        )

        for i, varname in enumerate(["gev_c", "gev_loc", "gev_scale"]):
            GEV_grid = getattr(self.grid, varname)
            self.var.GEV_parameters[:, i] = sample_from_map(
                GEV_grid, self.var.locations.data, self.grid.gt
            )

        self.var.risk_perc_min = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["drought_risk_calculations"]["risk_perception"]["min"],
        )
        self.var.risk_perc_max = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["drought_risk_calculations"]["risk_perception"]["max"],
        )
        self.var.risk_decr = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["drought_risk_calculations"]["risk_perception"]["coef"],
        )
        self.var.decision_horizon = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.int32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["decisions"]["decision_horizon"],
        )

        self.var.cumulative_water_deficit_m3 = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(366,),
            extra_dims_names=("day",),
            dtype=np.float32,
            fill_value=0,
        )
        self.var.cumulative_water_deficit_current_day = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=0,
        )

        self.var.field_indices_by_farmer = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(2,),
            dtype=np.int32,
            fill_value=-1,
            extra_dims_names=("index",),
        )

        self.update_field_indices()

    @staticmethod
    @njit(cache=True)
    def update_field_indices_numba(
        land_owners: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Creates `field_indices_by_farmer` and `field_indices`. These indices are used to quickly find the fields for a specific farmer.

        Args:
            land_owners: Array of the land owners. Each unique ID is a different land owner. -1 means the land is not owned by anyone.

        Returns:
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.
        """
        agents = np.unique(land_owners)
        if agents[0] == -1:
            n_agents = agents.size - 1
        else:
            n_agents = agents.size
        field_indices_by_farmer = np.full((n_agents, 2), -1, dtype=np.int32)
        field_indices = np.full(land_owners.size, -1, dtype=np.int32)

        land_owners_sort_idx = np.argsort(land_owners)
        land_owners_sorted = land_owners[land_owners_sort_idx]

        last_not_owned = np.searchsorted(land_owners_sorted, -1, side="right")

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
        (
            self.var.field_indices_by_farmer,
            self.var.field_indices,
        ) = self.update_field_indices_numba(self.HRU.var.land_owners)

    def set_social_network(self) -> None:
        """Determines for each farmer a group of neighbors which constitutes their social network."""
        nbits = 19
        radius = self.model.config["agent_settings"]["farmers"]["social_network"][
            "radius"
        ]
        n_neighbor = self.model.config["agent_settings"]["farmers"]["social_network"][
            "size"
        ]

        self.var.social_network = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(n_neighbor,),
            extra_dims_names=("neighbors",),
            dtype=np.int32,
            fill_value=np.nan,
        )

        bounds = self.grid.bounds
        self.var.social_network[:] = find_neighbors(
            self.var.locations.data,
            radius=radius,
            n_neighbor=n_neighbor,
            bits=nbits,
            minx=bounds[0],
            miny=bounds[1],
            maxx=bounds[2],
            maxy=bounds[3],
        )

    def adjust_cultivation_costs(self):
        # Set the cultivation costs
        self.cultivation_costs = load_regional_crop_data_from_dict(
            self.model, "crops/cultivation_costs"
        )
        cultivation_cost_fraction = self.model.config["agent_settings"]["farmers"][
            "cultivation_cost_fraction"
        ]  # Cultivation costs are set as a fraction of crop prices
        date_index, cultivation_costs_array = self.cultivation_costs

        if (
            "calibration" in self.model.config
            and "KGE_crops" in self.model.config["calibration"]["calibration_targets"]
        ):
            # Load price change factors 0 to 25 into a NumPy array
            factors = np.array(
                [
                    self.model.config["agent_settings"]["calibration_crops"][
                        f"price_{i}"
                    ]
                    for i in range(len(self.var.crop_ids))
                ]
            )

            # Multiply the cultivation_costs_array by the factors along the last axis
            cultivation_costs_array *= factors
        else:
            cultivation_costs_array = (
                cultivation_costs_array * cultivation_cost_fraction
            )
        self.cultivation_costs = (date_index, cultivation_costs_array)

    @property
    def activation_order_by_elevation(self):
        """Activation order is determined by the agent elevation, starting from the highest.

        Agents with the same elevation are randomly shuffled.
        """
        # if activation order is fixed. Get the random state, and set a fixed seet.
        if self.model.config["agent_settings"]["fix_activation_order"]:
            if hasattr(self, "activation_order_by_elevation_fixed"):
                return self.var.activation_order_by_elevation_fixed
            random_state = np.random.get_state()
            np.random.seed(42)
        elevation = self.var.elevation
        # Shuffle agent elevation and agent_ids in unision.
        p = np.random.permutation(elevation.size)
        # if activation order is fixed, set random state to previous state
        if self.model.config["agent_settings"]["fix_activation_order"]:
            np.random.set_state(random_state)
        elevation_shuffled = elevation[p]
        agent_ids_shuffled = np.arange(0, elevation.size, 1, dtype=np.int32)[p]
        # Use argsort to find the order or the shuffled elevation. Using a stable sorting
        # algorithm such that the random shuffling in the previous step is conserved
        # in groups with identical elevation.
        activation_order_shuffled = np.argsort(elevation_shuffled, kind="stable")[::-1]
        # unshuffle the agent_ids to get the activation order
        activation_order = agent_ids_shuffled[activation_order_shuffled]
        activation_order = DynamicArray(activation_order, max_n=self.var.max_n)
        if self.model.config["agent_settings"]["fix_activation_order"]:
            self.var.activation_order_by_elevation_fixed = activation_order
        # Check if the activation order is correct, by checking if elevation is decreasing
        assert np.diff(elevation[activation_order]).max() <= 0
        return activation_order

    @property
    def farmer_command_area(self):
        return farmer_command_area(
            self.var.n,
            self.var.field_indices,
            self.var.field_indices_by_farmer.data,
            self.HRU.var.reservoir_command_areas,
        )

    @property
    def is_in_command_area(self):
        return self.farmer_command_area != -1

    def save_water_deficit(self, discount_factor=0.2):
        water_deficit_day_m3 = (
            self.HRU.var.ETRef - self.HRU.pr
        ) * self.HRU.var.cell_area
        water_deficit_day_m3[water_deficit_day_m3 < 0] = 0

        water_deficit_day_m3_per_farmer = np.bincount(
            self.HRU.var.land_owners[self.HRU.var.land_owners != -1],
            weights=water_deficit_day_m3[self.HRU.var.land_owners != -1],
        )

        day_index: int = self.model.current_day_of_year - 1

        (
            self.var.cumulative_water_deficit_current_day,
            self.var.cumulative_water_deficit_previous_day,
        ) = (
            (self.var.cumulative_water_deficit_m3[:, day_index]).copy(),
            self.var.cumulative_water_deficit_current_day,
        )

        if day_index == 0:
            self.var.cumulative_water_deficit_m3[:, day_index] = (
                self.var.cumulative_water_deficit_m3[:, day_index]
                * (1 - discount_factor)
                + water_deficit_day_m3_per_farmer * discount_factor
            )
        else:
            self.var.cumulative_water_deficit_m3[:, day_index] = (
                self.var.cumulative_water_deficit_m3[:, day_index - 1]
                + water_deficit_day_m3_per_farmer * discount_factor
                + (1 - discount_factor)
                * (
                    self.var.cumulative_water_deficit_m3[:, day_index]
                    - self.var.cumulative_water_deficit_previous_day
                )
            )
            assert (
                self.var.cumulative_water_deficit_m3[:, day_index]
                >= self.var.cumulative_water_deficit_m3[:, day_index - 1]
            ).all()
            # if this is the last day of the year, but not a leap year, the virtual
            # 366th day of the year is the same as the 365th day of the year
            # this avoids complications with the leap year
            if day_index == 364 and not calendar.isleap(self.model.current_time.year):
                self.var.cumulative_water_deficit_m3[:, 365] = (
                    self.var.cumulative_water_deficit_m3[:, 364]
                )

    def get_gross_irrigation_demand_m3(
        self, potential_evapotranspiration, available_infiltration
    ) -> npt.NDArray[np.float32]:
        gross_irrigation_demand_m3: npt.NDArray[np.float32] = (
            get_gross_irrigation_demand_m3(
                day_index=self.model.current_day_of_year - 1,
                n=self.var.n,
                currently_irrigated_fields=self.currently_irrigated_fields,
                field_indices_by_farmer=self.var.field_indices_by_farmer.data,
                field_indices=self.var.field_indices,
                irrigation_efficiency=self.var.irrigation_efficiency.data,
                fraction_irrigated_field=self.var.fraction_irrigated_field.data,
                cell_area=self.model.hydrology.HRU.var.cell_area,
                crop_map=self.HRU.var.crop_map,
                topwater=self.HRU.var.topwater,
                available_infiltration=available_infiltration,
                potential_evapotranspiration=potential_evapotranspiration,
                root_depth=self.HRU.var.root_depth,
                soil_layer_height=self.HRU.var.soil_layer_height,
                field_capacity=self.HRU.var.wfc,
                wilting_point=self.HRU.var.wwp,
                w=self.HRU.var.w,
                ws=self.HRU.var.ws,
                arno_beta=self.HRU.var.arnoBeta,
                remaining_irrigation_limit_m3=self.var.remaining_irrigation_limit_m3.data,
                cumulative_water_deficit_m3=self.var.cumulative_water_deficit_m3.data,
                crop_calendar=self.var.crop_calendar.data,
                crop_group_numbers=self.var.crop_data[
                    "crop_group_number"
                ].values.astype(np.float32),
                paddy_irrigated_crops=self.var.crop_data["is_paddy"].values,
                current_crop_calendar_rotation_year_index=self.var.current_crop_calendar_rotation_year_index.data,
                max_paddy_water_level=self.var.max_paddy_water_level.data,
                minimum_effective_root_depth=np.float32(
                    self.model.hydrology.soil.var.minimum_effective_root_depth
                ),
            )
        )

        assert (
            gross_irrigation_demand_m3 < self.model.hydrology.HRU.var.cell_area
        ).all()
        return gross_irrigation_demand_m3

    @property
    def surface_irrigated(self):
        return self.var.adaptations[:, SURFACE_IRRIGATION_EQUIPMENT] > 0

    @property
    def well_irrigated(self):
        return self.var.adaptations[:, WELL_ADAPTATION] > 0

    @property
    def irrigated(self):
        return self.surface_irrigated | self.well_irrigated  # | is the OR operator

    @property
    def currently_irrigated_fields(self):
        return self.farmer_to_field(self.is_irrigated, False) & (
            self.HRU.var.crop_map != -1
        )

    def abstract_water(
        self,
        gross_irrigation_demand_m3_per_field: npt.NDArray[np.float32],
        available_channel_storage_m3: npt.NDArray[np.float32],
        available_groundwater_m3: npt.NDArray[np.float64],
        groundwater_depth: npt.NDArray[np.float64],
        available_reservoir_storage_m3: npt.NDArray[np.float32],
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float64],
    ]:
        """This function allows the abstraction of water by farmers for irrigation purposes.

        Its main purpose is to call the relevant numba function to do the actual abstraction.
        In addition, the function saves the abstraction from the various sources by farmer.

        Args:
            gross_irrigation_demand_m3_per_field: gross irrigation demand in m3 per field
            available_channel_storage_m3: available channel storage in m3 per grid cell
            available_groundwater_m3: available groundwater storage in m3 per grid cell
            groundwater_depth: groundwater depth in meters per grid cell
            available_reservoir_storage_m3: available reservoir storage in m3 per reservoir

        Returns:
            water_withdrawal_m: water withdrawal in meters
            water_consumption_m: water consumption in meters
            returnFlowIrr_m: return flow in meters
            addtoevapotrans_m: evaporated irrigation water in meters
        """
        assert (available_channel_storage_m3 >= 0).all()
        assert (available_groundwater_m3 >= 0).all()
        assert (available_reservoir_storage_m3 >= 0).all()

        if __debug__:
            irrigation_limit_pre = self.var.remaining_irrigation_limit_m3.copy()
            available_channel_storage_m3_pre = available_channel_storage_m3.copy()
        (
            self.var.channel_abstraction_m3_by_farmer[:],
            self.var.reservoir_abstraction_m3_by_farmer[:],
            self.var.groundwater_abstraction_m3_by_farmer[:],
            water_withdrawal_m,
            water_consumption_m,
            returnFlowIrr_m,
            addtoevapotrans_m,
            reservoir_abstraction_m3,
            groundwater_abstraction_m3,
        ) = abstract_water(
            activation_order=self.activation_order_by_elevation.data,
            field_indices_by_farmer=self.var.field_indices_by_farmer,
            field_indices=self.var.field_indices,
            irrigation_efficiency=self.var.irrigation_efficiency.data,
            surface_irrigated=self.surface_irrigated,
            well_irrigated=self.well_irrigated,
            cell_area=self.model.hydrology.HRU.var.cell_area,
            HRU_to_grid=self.HRU.var.HRU_to_grid,
            nearest_river_grid_cell=self.HRU.var.nearest_river_grid_cell,
            crop_map=self.HRU.var.crop_map,
            available_channel_storage_m3=available_channel_storage_m3,
            available_groundwater_m3=available_groundwater_m3,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
            groundwater_depth=groundwater_depth,
            farmer_command_area=self.farmer_command_area,
            return_fraction=self.model.config["agent_settings"]["farmers"][
                "return_fraction"
            ],
            well_depth=self.var.well_depth.data,
            remaining_irrigation_limit_m3=self.var.remaining_irrigation_limit_m3.data,
            gross_irrigation_demand_m3_per_field=gross_irrigation_demand_m3_per_field,
        )

        assert (water_withdrawal_m < 1).all()
        assert (water_consumption_m < 1).all()
        assert (returnFlowIrr_m < 1).all()
        assert (addtoevapotrans_m < 1).all()

        if __debug__:
            # make sure the withdrawal per source is identical to the total withdrawal in m (corrected for cell area)
            balance_check(
                name="water withdrawal_1",
                how="sum",
                influxes=(
                    self.var.channel_abstraction_m3_by_farmer,
                    self.var.reservoir_abstraction_m3_by_farmer,
                    self.var.groundwater_abstraction_m3_by_farmer,
                ),
                outfluxes=[
                    (water_withdrawal_m * self.model.hydrology.HRU.var.cell_area)
                ],
                tollerance=50,
            )

            # assert that the total amount of water withdrawn is equal to the total storage before and after abstraction
            balance_check(
                name="water withdrawal channel",
                how="sum",
                outfluxes=self.var.channel_abstraction_m3_by_farmer,
                prestorages=available_channel_storage_m3_pre,
                poststorages=available_channel_storage_m3,
                tollerance=50,
            )

            balance_check(
                name="water withdrawal reservoir",
                how="sum",
                outfluxes=self.var.reservoir_abstraction_m3_by_farmer,
                influxes=reservoir_abstraction_m3,
                tollerance=50,
            )

            balance_check(
                name="water withdrawal groundwater",
                how="sum",
                outfluxes=self.var.groundwater_abstraction_m3_by_farmer,
                influxes=groundwater_abstraction_m3,
                tollerance=10,
            )

            # assert that the total amount of water withdrawn is equal to the total storage before and after abstraction
            balance_check(
                name="water withdrawal_2",
                how="sum",
                outfluxes=(
                    self.var.channel_abstraction_m3_by_farmer[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3)
                    ].astype(np.float64),
                    self.var.reservoir_abstraction_m3_by_farmer[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3)
                    ].astype(np.float64),
                    self.var.groundwater_abstraction_m3_by_farmer[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3)
                    ].astype(np.float64),
                ),
                prestorages=irrigation_limit_pre[
                    ~np.isnan(self.var.remaining_irrigation_limit_m3)
                ].astype(np.float64),
                poststorages=self.var.remaining_irrigation_limit_m3[
                    ~np.isnan(self.var.remaining_irrigation_limit_m3)
                ].astype(np.float64),
                tollerance=50,
            )

            # make sure the total water consumption plus 'wasted' irrigation water (evaporation + return flow) is equal to the total water withdrawal
            balance_check(
                name="water consumption",
                how="sum",
                influxes=(
                    water_consumption_m,
                    returnFlowIrr_m,
                    addtoevapotrans_m,
                ),
                outfluxes=water_withdrawal_m,
                tollerance=50,
            )

            assert water_withdrawal_m.dtype == np.float32
            assert water_consumption_m.dtype == np.float32
            assert returnFlowIrr_m.dtype == np.float32
            assert addtoevapotrans_m.dtype == np.float32

        return (
            water_withdrawal_m,
            water_consumption_m,
            returnFlowIrr_m,
            addtoevapotrans_m,
            reservoir_abstraction_m3,
            groundwater_abstraction_m3,
        )

    @staticmethod
    @njit(cache=True)
    def get_yield_ratio_numba_GAEZ(
        crop_map: np.ndarray, evap_ratios: np.ndarray, KyT
    ) -> float:
        """Calculate yield ratio based on https://doi.org/10.1016/j.jhydrol.2009.07.031.

        Args:
            crop_map: array of currently harvested crops.
            evap_ratios: ratio of actual to potential evapotranspiration of harvested crops.
            KyT: Water stress reduction factor from GAEZ.

        Returns:
            yield_ratios: yield ratio (as ratio of maximum obtainable yield) per harvested crop.
        """
        yield_ratios = np.full(evap_ratios.size, -1, dtype=np.float32)

        assert crop_map.size == evap_ratios.size

        for i in range(evap_ratios.size):
            evap_ratio = evap_ratios[i]
            crop = crop_map[i]
            yield_ratios[i] = max(
                1 - KyT[crop] * (1 - evap_ratio), 0
            )  # Yield ratio is never lower than 0.

        return yield_ratios

    @staticmethod
    @njit(cache=True)
    def get_yield_ratio_numba_MIRCA2000(
        crop_map: np.ndarray,
        evap_ratios: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        P0: np.ndarray,
        P1: np.ndarray,
    ) -> float:
        """Calculate yield ratio based on https://doi.org/10.1016/j.jhydrol.2009.07.031.

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
            if alpha[crop] * evap_ratio + beta[crop] > 1:
                yield_ratio = 1
            elif P0[crop] < evap_ratio < P1[crop]:
                yield_ratio = (
                    alpha[crop] * P1[crop]
                    + beta[crop]
                    - (P1[crop] - evap_ratio)
                    * (alpha[crop] * P1[crop] + beta[crop])
                    / (P1[crop] - P0[crop])
                )
            elif evap_ratio < P0[crop]:
                yield_ratio = 0
            else:
                yield_ratio = alpha[crop] * evap_ratio + beta[crop]
            yield_ratios[i] = yield_ratio

        return yield_ratios

    def get_yield_ratio(
        self,
        harvest: np.ndarray,
        actual_transpiration: np.ndarray,
        potential_transpiration: np.ndarray,
        crop_map: np.ndarray,
    ) -> np.ndarray:
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
        if self.var.crop_data_type == "GAEZ":
            yield_ratio = self.get_yield_ratio_numba_GAEZ(
                crop_map[harvest],
                actual_transpiration[harvest] / potential_transpiration[harvest],
                self.var.crop_data["KyT"].values,
            )
        elif self.var.crop_data_type == "MIRCA2000":
            yield_ratio = self.get_yield_ratio_numba_MIRCA2000(
                crop_map[harvest],
                actual_transpiration[harvest] / potential_transpiration[harvest],
                self.var.crop_data["a"].values,
                self.var.crop_data["b"].values,
                self.var.crop_data["P0"].values,
                self.var.crop_data["P1"].values,
            )
            if np.any(yield_ratio == 0):
                pass
        else:
            raise ValueError(
                f"Unknown crop data type: {self.var.crop_data_type}, must be 'GAEZ' or 'MIRCA2000'"
            )
        assert not np.isnan(yield_ratio).any()

        return yield_ratio

    def field_to_farmer(
        self,
        array: npt.NDArray[np.floating],
        method: str = "sum",
    ) -> npt.NDArray[np.floating]:
        assert method == "sum", "Only sum is implemented"
        farmer_fields: npt.NDArray[np.int32] = self.HRU.var.land_owners[
            self.HRU.var.land_owners != -1
        ]
        masked_array: npt.NDArray[np.floating] = array[self.HRU.var.land_owners != -1]
        return np.bincount(farmer_fields, masked_array, minlength=self.var.n).astype(
            masked_array.dtype
        )

    def farmer_to_field(self, array, nodata):
        by_field = np.take(array, self.HRU.var.land_owners)
        by_field[self.HRU.var.land_owners == -1] = nodata
        return by_field

    def decompress(self, array):
        if np.issubdtype(array.dtype, np.floating):
            nofieldvalue = np.nan
        else:
            nofieldvalue = -1
        by_field = self.farmer_to_field(array, nodata=nofieldvalue)
        return self.HRU.decompress(by_field)

    @property
    def mask(self):
        mask = self.HRU.mask.copy()
        mask[self.decompress(self.HRU.var.land_owners) == -1] = True
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
            crop_harvest_age_days: Subarray map of crop harvest age in days. I.e., the age at which the crop is ready to be harvested.

        Returns:
            harvest: Boolean subarray map of fields to be harvested.
        """
        harvest = np.zeros(crop_map.shape, dtype=np.bool_)
        for farmer_i in range(n):
            farmer_fields = get_farmer_HRUs(
                field_indices, field_indices_by_farmer, farmer_i
            )
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
        """Determine which crops need to be harvested based on their current age and their harvest age.

        Once harvested, compute various metrics related to the harvest including potential profit,
        actual profit, crop age, drought perception, and update corresponding attributes of the model.
        Save the corresponding SPEI over the last harvest.

        Attributes:
            harvest_numba: A helper function to obtain the harvest map.
            get_yield_ratio: A function to calculate yield ratio based on the ratio of actual to potential evapotranspiration.

        Note:
            The function also updates the drought risk perception and tracks disposable income.
        """
        # Using the helper function to determine which crops are ready to be harvested
        harvest = self.harvest_numba(
            n=self.var.n,
            field_indices_by_farmer=self.var.field_indices_by_farmer.data,
            field_indices=self.var.field_indices,
            crop_map=self.HRU.var.crop_map,
            crop_age_days=self.HRU.var.crop_age_days_map,
            crop_harvest_age_days=self.HRU.var.crop_harvest_age_days,
        )

        self.var.actual_yield_per_farmer.fill(np.nan)
        self.var.harvested_crop.fill(-1)
        # If there are fields to be harvested, compute yield ratio and various related metrics
        if np.count_nonzero(harvest):
            print(f"Harvesting {np.count_nonzero(harvest)} fields.")
            # Get yield ratio for the harvested crops
            yield_ratio_per_field = self.get_yield_ratio(
                harvest,
                self.HRU.var.actual_evapotranspiration_crop_life,
                self.HRU.var.potential_evapotranspiration_crop_life,
                self.HRU.var.crop_map,
            )
            assert (yield_ratio_per_field >= 0).all()

            harvesting_farmer_fields = self.HRU.var.land_owners[harvest]
            harvested_area = self.HRU.var.cell_area[harvest]

            harvested_crops = self.HRU.var.crop_map[harvest]
            max_yield_per_crop = np.take(
                self.var.crop_data["reference_yield_kg_m2"].values, harvested_crops
            )
            harvesting_farmers = np.unique(harvesting_farmer_fields)

            # it's okay for some crop prices to be nan, as they will be filtered out in the next step
            crop_prices = self.agents.market.crop_prices
            region_id_per_field = self.var.region_id

            # Determine the region ids of harvesting farmers, as crop prices differ per region

            region_id_per_field = self.var.region_id[self.HRU.var.land_owners]
            region_id_per_field[self.HRU.var.land_owners == -1] = -1
            region_id_per_harvested_field = region_id_per_field[harvest]

            # Calculate the crop price per field
            crop_price_per_field = crop_prices[
                region_id_per_harvested_field, harvested_crops
            ]

            # but it's not okay for the crop price to be nan now
            assert not np.isnan(crop_price_per_field).any()

            # Correct yield ratio
            yield_ratio_per_field = (
                self.var.base_management_yield_ratio[harvesting_farmer_fields]
                * yield_ratio_per_field
            )

            # Calculate the potential yield per field
            potential_yield_per_field = max_yield_per_crop * harvested_area

            # Calculate the total yield per field
            actual_yield_per_field = yield_ratio_per_field * potential_yield_per_field

            # And sum the total yield per field to get the total yield per farmer
            self.var.actual_yield_per_farmer[:] = np.bincount(
                harvesting_farmer_fields,
                weights=actual_yield_per_field,
                minlength=self.var.n,
            )

            # get the harvested crop per farmer. This assumes each farmer only harvests one crop
            # on the same day
            self.var.harvested_crop[harvesting_farmers] = harvested_crops[
                np.unique(self.HRU.var.land_owners[harvest], return_index=True)[1]
            ]

            # Determine the actual and potential profits
            potential_profit_per_field = (
                potential_yield_per_field * crop_price_per_field
            )
            actual_profit_per_field = actual_yield_per_field * crop_price_per_field
            assert (potential_profit_per_field >= 0).all()
            assert (actual_profit_per_field >= 0).all()

            # Convert from the profit and potential profit per field to the profit per farmer
            potential_income_farmer = np.bincount(
                harvesting_farmer_fields,
                weights=potential_profit_per_field,
                minlength=self.var.n,
            )
            self.income_farmer = np.bincount(
                harvesting_farmer_fields,
                weights=actual_profit_per_field,
                minlength=self.var.n,
            )

            # Convert the yield_ratio per field to the average yield ratio per farmer
            # yield_ratio_per_farmer = income_farmer / potential_income_farmer

            # Get the crop age
            crop_age = self.HRU.var.crop_age_days_map[harvest]
            current_crop_age = np.bincount(
                harvesting_farmer_fields, weights=crop_age, minlength=self.var.n
            ) / np.bincount(harvesting_farmer_fields, minlength=self.var.n)

            harvesting_farmers_mask = np.zeros(self.var.n, dtype=bool)
            harvesting_farmers_mask[harvesting_farmers] = True

            self.save_yearly_income(self.income_farmer, potential_income_farmer)
            self.save_harvest_spei(harvesting_farmers)
            self.drought_risk_perception(harvesting_farmers, current_crop_age)

            ## After updating the drought risk perception, set the previous month for the next timestep as the current for this timestep.
            # TODO: This seems a bit like a quirky solution, perhaps there is a better way to do this.
            self.var.previous_month = self.model.current_time.month

        else:
            self.income_farmer = np.zeros(self.var.n, dtype=np.float32)

        # Reset transpiration values for harvested fields
        self.HRU.var.actual_evapotranspiration_crop_life[harvest] = 0
        self.HRU.var.potential_evapotranspiration_crop_life[harvest] = 0

        # Update crop and land use maps after harvest
        self.HRU.var.crop_map[harvest] = -1
        self.HRU.var.crop_age_days_map[harvest] = -1
        self.HRU.var.land_use_type[harvest] = GRASSLAND_LIKE

        # For unharvested growing crops, increase their age by 1
        self.HRU.var.crop_age_days_map[(~harvest) & (self.HRU.var.crop_map >= 0)] += 1

        assert (
            self.HRU.var.crop_age_days_map <= self.HRU.var.crop_harvest_age_days
        ).all()

    def drought_risk_perception(
        self, harvesting_farmers: np.ndarray, current_crop_age: np.ndarray
    ) -> None:
        """Calculate and update the drought risk perception for harvesting farmers.

        This function computes the risk perception of farmers based on the difference
        between their latest profits and potential profits. The perception is influenced
        by the historical losses and time since the last drought event. Farmers who have
        experienced a drought event will have their drought timer reset.

        Args:
            harvesting_farmers: Index array of farmers that are currently harvesting.
            current_crop_age: Array of current crop age for each farmer.

        TODO: Perhaps move the constant to the model.yml
        """
        # constants
        HISTORICAL_PERIOD = min(5, self.var.yearly_potential_income.shape[1])  # years

        # Convert the harvesting farmers index array to a boolean array of full length
        harvesting_farmers_long = np.zeros(self.var.n, dtype=bool)
        harvesting_farmers_long[harvesting_farmers] = True

        # Update the drought timer based on the months passed since the previous check
        months_passed = (self.model.current_time.month - self.var.previous_month) % 12
        self.var.drought_timer += months_passed / 12

        # Create an empty drought loss np.ndarray
        drought_loss_historical = np.zeros(
            (self.var.n, HISTORICAL_PERIOD), dtype=np.float32
        )

        # Calculate the cumulative inflation from the start year to the current year for each farmer
        # the base year is not important here as we are only interested in the relative change
        cumulative_inflation_since_base_year = np.cumprod(
            np.stack(
                [
                    self.get_value_per_farmer_from_region_id(
                        self.inflation_rate,
                        datetime(year, 1, 1),
                        subset=harvesting_farmers_long,
                    )
                    for year in range(
                        self.model.current_time.year + 1 - HISTORICAL_PERIOD,
                        self.model.current_time.year + 1,
                    )
                ],
                axis=1,
            ),
            axis=1,
        )

        # Compute the percentage loss between potential and actual profits for harvesting farmers
        potential_profits_inflation_corrected = (
            self.var.yearly_potential_income[
                harvesting_farmers_long, :HISTORICAL_PERIOD
            ]
            / cumulative_inflation_since_base_year
        )
        actual_profits_inflation_corrected = (
            self.var.yearly_income[harvesting_farmers_long, :HISTORICAL_PERIOD]
            / cumulative_inflation_since_base_year
        )

        drought_loss_historical[harvesting_farmers_long] = (
            (potential_profits_inflation_corrected - actual_profits_inflation_corrected)
            / potential_profits_inflation_corrected
        ) * 100

        # Calculate the current and past average loss percentages
        drought_loss_latest = drought_loss_historical[:, 0]
        drought_loss_past = np.mean(drought_loss_historical[:, 1:], axis=1)

        # Identify farmers who experienced a drought event based on loss comparison with historical losses
        drought_loss_current = drought_loss_latest - drought_loss_past

        experienced_drought_event = (
            drought_loss_current >= self.var.moving_average_threshold
        )

        # Reset the drought timer for farmers who have harvested and experienced a drought event
        self.var.drought_timer[
            np.logical_and(harvesting_farmers_long, experienced_drought_event)
        ] = 0

        # Update the risk perception of all farmers
        self.var.risk_perception = (
            self.var.risk_perc_max
            * (1.6 ** (self.var.risk_decr * self.var.drought_timer))
            + self.var.risk_perc_min
        )

        print(
            "Risk perception mean = ",
            np.mean(self.var.risk_perception),
            "STD",
            np.std(self.var.risk_perception),
        )

        # Determine which farmers need emergency microcredit to keep farming
        loaning_farmers = drought_loss_current >= self.var.moving_average_threshold

        # Determine their microcredit
        if self.microcredit_adaptation_active:
            # print(np.count_nonzero(loaning_farmers), "farmers are getting microcredit")
            self.microcredit(loaning_farmers, drought_loss_current, current_crop_age)

    def microcredit(
        self,
        loaning_farmers: np.ndarray,
        drought_loss_current: np.ndarray,
        current_crop_age: np.ndarray,
    ) -> None:
        """Compute the microcredit for farmers based on their average profits, drought losses, and the age of their crops with respect to their total cropping time.

        Args:
            loaning_farmers: Boolean mask of farmers looking to obtain a loan, based on drought loss of harvesting farmers.
            drought_loss_current: Array of drought losses of the most recent harvest for each farmer.
            current_crop_age: Array of current crop age for each farmer.
        """
        # Compute the maximum loan amount based on the average profits of the last 10 years
        max_loan = np.median(self.var.yearly_income[loaning_farmers, :5], axis=1)

        # Compute the crop age as a percentage of the average total time a farmer has had crops planted
        crop_growth_duration = self.var.crop_calendar[:, :, 2].data
        total_crop_age = np.where(
            crop_growth_duration == -1, 0, crop_growth_duration
        ).sum(axis=1)
        crop_age_fraction = (
            current_crop_age[loaning_farmers] / total_crop_age[loaning_farmers]
        )

        # Calculate the total loan amount based on drought loss, crop age percentage, and the maximum loan
        total_loan = (
            (drought_loss_current[loaning_farmers] / 100) * crop_age_fraction * max_loan
        )

        # Fetch loan configurations from the model settings
        loan_duration = self.model.config["agent_settings"]["farmers"]["microcredit"][
            "loan_duration"
        ]

        # interest_rate = self.get_value_per_farmer_from_region_id(
        #     self.var.lending_rate, self.model.current_time
        # )
        interest_rate = self.var.interest_rate.data

        # Compute the annual cost of the loan using the interest rate and loan duration
        annual_cost_microcredit = total_loan * (
            interest_rate[loaning_farmers]
            * (1 + interest_rate[loaning_farmers]) ** loan_duration
            / ((1 + interest_rate[loaning_farmers]) ** loan_duration - 1)
        )

        # Add the amounts to the individual loan slots
        self.set_loans_numba(
            all_loans_annual_cost=self.var.all_loans_annual_cost.data,
            loan_tracker=self.var.loan_tracker.data,
            loaning_farmers=loaning_farmers,
            annual_cost_loan=annual_cost_microcredit,
            loan_duration=loan_duration,
            loan_type=1,
        )

        # Add it to the loan total
        self.var.all_loans_annual_cost[loaning_farmers, -1, 0] += (
            annual_cost_microcredit
        )

    def potential_insured_loss(self):
        # Calculating personal pure premiums and Bühlmann-Straub parameters to get the credibility premium
        # Mask out unfilled years
        mask_columns = np.all(self.var.yearly_income == 0, axis=0)

        # Apply the mask to data
        income_masked = self.var.yearly_income.data[:, ~mask_columns]
        n_agents, n_years = income_masked.shape

        # Calculate personal loss
        self.var.avg_income_per_agent = np.nanmean(income_masked, axis=1)

        potential_insured_loss = np.zeros_like(self.var.yearly_income, dtype=np.float32)

        potential_insured_loss[:, ~mask_columns] = np.maximum(
            self.var.avg_income_per_agent[..., None] - income_masked, 0
        )

        # Add the insured loss to the income of this year's insured farmers
        insured_farmers_mask = (
            self.var.adaptations[:, PERSONAL_INSURANCE_ADAPTATION] > 0
        )

        self.var.insured_yearly_income[insured_farmers_mask, 0] += (
            potential_insured_loss[insured_farmers_mask, 0]
        )

        return potential_insured_loss

    def premium_personal_insurance(self):
        # Calculating personal pure premiums and Bühlmann-Straub parameters to get the credibility premium
        # Mask out unfilled years
        mask_columns = np.all(self.var.yearly_income == 0, axis=0)

        group_indices, n_groups = self.create_unique_groups(
            self.main_irrigation_source,
        )
        # assert (np.any(self.var.yearly_SPEI_probability != 0, axis=1) > 0).all()

        # Apply the mask to data
        income_masked = self.var.yearly_income.data[:, ~mask_columns]
        n_agents, n_years = income_masked.shape

        # Calculate personal loss
        avg_income_per_agent = np.nanmean(income_masked, axis=1)
        losses = np.maximum(avg_income_per_agent[:, None] - income_masked, 0)
        years_observed = np.sum(~np.isnan(income_masked), axis=1)
        self.var.agent_pure_premiums = np.mean(losses, axis=1)

        # Initialize arrays for coefficients and R²
        group_mean_premiums = np.zeros(n_groups, dtype=float)
        for group_idx in range(n_groups):
            agent_indices = np.where(group_indices == group_idx)[0]
            group_mean_premiums[group_idx] = np.mean(
                self.var.agent_pure_premiums[agent_indices]
            )

        sample_var_per_agent = np.var(losses, axis=1, ddof=1)
        valid_for_within = years_observed > 1

        within_variance = np.sum(
            (years_observed[valid_for_within] - 1)
            * sample_var_per_agent[valid_for_within]
        ) / np.sum(years_observed[valid_for_within] - 1)
        between_variance = np.var(self.var.agent_pure_premiums, ddof=1)
        credibility_param_K = (
            within_variance / between_variance if between_variance > 0 else np.inf
        )

        # Classical Bühlmann–Straub: Z = n / (n + K)
        credibility_weights = years_observed / (years_observed + credibility_param_K)
        credibility_premiums = (
            credibility_weights * self.var.agent_pure_premiums
            + (1 - credibility_weights) * group_mean_premiums[group_indices]
        )

        return credibility_premiums

    def premium_index_insurance(self, potential_insured_loss):
        mask_columns = np.all(self.var.yearly_income == 0, axis=0)
        gev_params = self.var.GEV_parameters.data
        strike_vals = np.round(np.arange(0.0, -2.0, -0.1), 2)
        exit_vals = np.round(np.arange(-2, -3.6, -0.1), 2)
        rate_vals = np.linspace(500, 60000, 20)

        potential_insured_loss_masked = potential_insured_loss[:, ~mask_columns]
        spei_hist = self.var.yearly_SPEI.data[:, ~mask_columns]

        (
            best_strike_idx,
            best_exit_idx,
            best_rate_idx,
            best_rmse,
            best_prem,
        ) = compute_premiums_and_best_contracts_numba(
            gev_params,
            spei_hist,
            potential_insured_loss_masked,
            strike_vals,
            exit_vals,
            rate_vals,
            n_sims=100,
            seed=42,
        )

        n_agents = gev_params.shape[0]
        best_strike = np.empty(n_agents, dtype=np.float64)
        best_exit = np.empty(n_agents, dtype=np.float64)
        best_rate = np.empty(n_agents, dtype=np.float64)
        best_prem = np.empty(n_agents, dtype=np.float64)

        for i in range(n_agents):
            best_strike[i] = strike_vals[best_strike_idx[i]]
            best_exit[i] = exit_vals[best_exit_idx[i]]
            best_rate[i] = rate_vals[best_rate_idx[i]]
        best_premiums = best_prem

        return (best_strike, best_exit, best_rate, best_premiums)

    def insured_payouts_index(self, strike, exit, rate):
        # Determine what the index insurance would have paid out in the past
        mask_columns = np.all(self.var.yearly_income == 0, axis=0)
        spei_hist = self.var.yearly_SPEI.data[:, ~mask_columns]

        denom = strike - exit
        shortfall = strike[:, None] - spei_hist
        # (no payout if rainfall ≥ strike)
        shortfall = np.clip(shortfall, 0.0, None)
        # (full payout once exit is breached)
        shortfall = np.minimum(shortfall, denom[:, None])
        # convert to fraction of maximum shortfall
        ratio = shortfall / denom[:, None]
        # scale by each agent’s rate
        payouts = ratio * rate[:, None]

        potential_insured_loss = np.zeros_like(self.var.yearly_income, dtype=np.float32)
        potential_insured_loss[:, ~mask_columns] = payouts

        # Add the insured loss to the income of this year's insured farmers
        insured_farmers_mask = self.var.adaptations[:, INDEX_INSURANCE_ADAPTATION] > 0

        self.var.insured_yearly_income[insured_farmers_mask, 0] += (
            potential_insured_loss[insured_farmers_mask, 0]
        )
        return potential_insured_loss

    def insured_yields(self, potential_insured_loss):
        insured_yearly_income = self.var.yearly_income + potential_insured_loss

        insured_yearly_yield_ratio = (
            insured_yearly_income / self.var.yearly_potential_income
        )

        insured_yearly_yield_ratio = np.clip(insured_yearly_yield_ratio.data, 0, 1)

        insured_yield_probability_relation = self.calculate_yield_spei_relation_group(
            insured_yearly_yield_ratio, self.var.yearly_SPEI_probability
        )
        return insured_yield_probability_relation

    @staticmethod
    @njit(cache=True)
    def set_loans_numba(
        all_loans_annual_cost: np.ndarray,
        loan_tracker: np.ndarray,
        loaning_farmers: np.ndarray,
        annual_cost_loan: np.ndarray,
        loan_duration: int,
        loan_type: int,
    ) -> None:
        farmers_getting_loan = np.where(loaning_farmers)[0]

        # Update the agent's loans and total annual costs with the computed annual cost
        # Make sure it is in an empty loan slot
        for farmer in farmers_getting_loan:
            for i in range(4):
                if all_loans_annual_cost[farmer, loan_type, i] == 0:
                    local_index = np.where(farmers_getting_loan == farmer)[0][0]
                    all_loans_annual_cost[farmer, loan_type, i] += annual_cost_loan[
                        local_index
                    ]
                    loan_tracker[farmer, loan_type, i] = loan_duration
                    break  # Exit the loop after adding to the first zero value

    def plant(self) -> None:
        """Determines when and what crop should be planted, mainly through calling the :meth:`agents.farmers.Farmers.plant_numba`. Then converts the array to cupy array if model is running with GPU."""
        if self.cultivation_costs[0] is None:
            cultivation_cost = self.cultivation_costs[1]
        else:
            index = self.cultivation_costs[0].get(self.model.current_time)
            cultivation_cost = self.cultivation_costs[1][index]
            assert cultivation_cost.shape[0] == len(self.model.regions)
            assert cultivation_cost.shape[1] == len(self.var.crop_ids)

        plant_map, farmers_selling_land = plant(
            n=self.var.n,
            day_index=self.model.current_time.timetuple().tm_yday - 1,  # 0-indexed
            crop_calendar=self.var.crop_calendar.data,
            current_crop_calendar_rotation_year_index=self.var.current_crop_calendar_rotation_year_index.data,
            crop_map=self.HRU.var.crop_map,
            crop_harvest_age_days=self.HRU.var.crop_harvest_age_days,
            cultivation_cost=cultivation_cost,
            region_ids_per_farmer=self.var.region_id.data,
            field_indices_by_farmer=self.var.field_indices_by_farmer.data,
            field_indices=self.var.field_indices,
            field_size_per_farmer=self.field_size_per_farmer.data,
            all_loans_annual_cost=self.var.all_loans_annual_cost.data,
            loan_tracker=self.var.loan_tracker.data,
            interest_rate=self.var.interest_rate.data,
            farmers_going_out_of_business=False,
        )
        if farmers_selling_land.size > 0:
            self.remove_agents(farmers_selling_land)

        number_of_planted_fields = np.count_nonzero(plant_map >= 0)
        if number_of_planted_fields > 0:
            print(
                f"Planting {number_of_planted_fields} fields with crops: "
                f"{np.unique(plant_map[plant_map >= 0])}"
            )

        self.HRU.var.crop_map = np.where(
            plant_map >= 0, plant_map, self.HRU.var.crop_map
        )
        self.HRU.var.crop_age_days_map[plant_map >= 0] = 1

        assert (self.HRU.var.crop_age_days_map[self.HRU.var.crop_map > 0] >= 0).all()

        is_paddy_crop = np.isin(
            self.HRU.var.crop_map,
            self.var.crop_data[self.var.crop_data["is_paddy"]].index,
        )

        self.HRU.var.land_use_type[(self.HRU.var.crop_map >= 0) & is_paddy_crop] = (
            PADDY_IRRIGATED
        )
        self.HRU.var.land_use_type[(self.HRU.var.crop_map >= 0) & (~is_paddy_crop)] = (
            NON_PADDY_IRRIGATED
        )

    def water_abstraction_sum(self) -> None:
        """Aggregates yearly water abstraction from different sources (channel, reservoir, groundwater) for each farmer.

        Also computes the total abstraction per farmer.

        Note:
            This function performs the following steps:
                1. Updates the yearly channel water abstraction for each farmer.
                2. Updates the yearly reservoir water abstraction for each farmer.
                3. Updates the yearly groundwater water abstraction for each farmer.
                4. Computes and updates the total water abstraction for each farmer.

        """
        # Update yearly channel water abstraction for each farmer
        self.var.yearly_abstraction_m3_by_farmer[:, CHANNEL_IRRIGATION, 0] += (
            self.var.channel_abstraction_m3_by_farmer
        )

        # Update yearly reservoir water abstraction for each farmer
        self.var.yearly_abstraction_m3_by_farmer[:, RESERVOIR_IRRIGATION, 0] += (
            self.var.reservoir_abstraction_m3_by_farmer
        )

        # Update yearly groundwater water abstraction for each farmer
        self.var.yearly_abstraction_m3_by_farmer[:, GROUNDWATER_IRRIGATION, 0] += (
            self.var.groundwater_abstraction_m3_by_farmer
        )

        # Compute and update the total water abstraction for each farmer
        self.var.yearly_abstraction_m3_by_farmer[:, TOTAL_IRRIGATION, 0] += (
            self.var.channel_abstraction_m3_by_farmer
            + self.var.reservoir_abstraction_m3_by_farmer
            + self.var.groundwater_abstraction_m3_by_farmer
        )

    def save_harvest_spei(self, harvesting_farmers) -> None:
        """Update the monthly Standardized Precipitation Evapotranspiration Index (SPEI) array by shifting past records and adding the SPEI for the current month.

        Note:
            This method updates the `monthly_SPEI` attribute in place.
        """
        current_SPEI_per_farmer = sample_from_map(
            array=self.model.hydrology.grid.spei_uncompressed,
            coords=self.var.locations[harvesting_farmers],
            gt=self.grid.gt,
        )

        full_size_SPEI_per_farmer = np.zeros_like(
            self.var.cumulative_SPEI_during_growing_season
        )
        full_size_SPEI_per_farmer[harvesting_farmers] = current_SPEI_per_farmer

        cumulative_mean(
            mean=self.var.cumulative_SPEI_during_growing_season,
            counter=self.var.cumulative_SPEI_count_during_growing_season,
            update=full_size_SPEI_per_farmer,
            mask=harvesting_farmers,
        )
        print(
            "season SPEI",
            np.mean(full_size_SPEI_per_farmer[harvesting_farmers]),
        )

    def save_yearly_spei(self):
        assert self.model.current_time.month == 1

        # calculate the SPEI probability using GEV parameters
        SPEI_probability = genextreme.sf(
            self.var.cumulative_SPEI_during_growing_season,
            self.var.GEV_parameters[:, 0],
            self.var.GEV_parameters[:, 1],
            self.var.GEV_parameters[:, 2],
        )

        spei = self.var.cumulative_SPEI_during_growing_season.copy()

        # SPEI_probability_norm = norm.cdf(self.var.cumulative_SPEI_during_growing_season)

        print("Yearly probability", np.nanmean(1 - SPEI_probability))

        shift_and_update(self.var.yearly_SPEI_probability, (1 - SPEI_probability))
        shift_and_update(self.var.yearly_SPEI, (spei))

        # Reset the cumulative SPEI array at the beginning of the year
        self.var.cumulative_SPEI_during_growing_season.fill(0)
        self.var.cumulative_SPEI_count_during_growing_season.fill(0)

    def save_yearly_income(
        self,
        income: np.ndarray,
        potential_income: np.ndarray,
    ) -> None:
        """Saves the latest profit and potential profit values for harvesting farmers to determine yearly profits, considering inflation and field size.

        Note:
            This function performs the following operations:
                1. Asserts that all profit and potential profit values are non-negative.
                2. Updates the latest profits and potential profits matrices by shifting all columns one column further.
                The last column value is dropped.
                3. Adjusts the yearly profits by accounting for the latest profit, field size, and inflation.
        """
        # Ensure that all profit and potential profit values are non-negative
        assert (income >= 0).all()
        assert (potential_income >= 0).all()
        # Adjust yearly profits by the cumulative inflation for each harvesting farmer
        self.var.yearly_income[:, 0] += income
        self.var.yearly_potential_income[:, 0] += potential_income

    def calculate_yield_spei_relation_test_solo(self):
        import matplotlib

        matplotlib.use("Agg")  # Use the 'Agg' backend for non-interactive plotting
        import matplotlib.pyplot as plt

        # Number of agents
        n_agents = self.var.yearly_yield_ratio.shape[0]

        # Define regression models
        def linear_model(X, a, b):
            return a * X + b

        def exponential_model(X, a, b):
            return a * np.exp(b * X)

        def logarithmic_model(X, a, b):
            return a * np.log(X) + b

        def quadratic_model(X, a, b, c):
            return a * X**2 + b * X + c

        def power_model(X, a, b):
            return a * X**b

        # Initialize dictionaries for coefficients and R² values
        model_names = ["linear", "exponential", "logarithmic", "quadratic", "power"]
        r_squared_dict = {model: np.zeros(n_agents) for model in model_names}
        coefficients_dict = {model: [] for model in model_names}

        # Create a folder to save the plots
        output_folder = "plot/relation_test"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # For each agent, perform regression with different models
        for agent_idx in range(n_agents):
            # Get data for the agent
            y_data = self.var.yearly_yield_ratio[agent_idx, :]  # shape (n_years,)
            X_data = self.var.yearly_SPEI_probability[agent_idx, :]  # shape (n_years,)

            # Filter out invalid values
            valid_mask = (
                (~np.isnan(X_data)) & (~np.isnan(y_data)) & (X_data > 0) & (y_data != 0)
            )
            X_valid = X_data[valid_mask]
            y_valid = y_data[valid_mask]

            if len(X_valid) >= 2:
                # Prepare data
                X_log = np.log10(X_valid)

                # Model 1: Linear in log-transformed X
                try:
                    popt, _ = curve_fit(linear_model, X_log, y_valid, maxfev=10000)
                    a, b = popt
                    y_pred = linear_model(X_log, a, b)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["linear"][agent_idx] = r_squared
                    coefficients_dict["linear"].append((a, b))
                except RuntimeError:
                    r_squared_dict["linear"][agent_idx] = np.nan
                    coefficients_dict["linear"].append((np.nan, np.nan))

                # Model 2: Exponential
                try:
                    popt, _ = curve_fit(
                        exponential_model, X_valid, y_valid, maxfev=10000
                    )
                    y_pred = exponential_model(X_valid, *popt)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["exponential"][agent_idx] = r_squared
                    coefficients_dict["exponential"].append(popt)
                except RuntimeError:
                    r_squared_dict["exponential"][agent_idx] = np.nan
                    coefficients_dict["exponential"].append((np.nan, np.nan))

                # Model 3: Logarithmic (ensure X > 0)
                try:
                    popt, _ = curve_fit(
                        logarithmic_model, X_valid, y_valid, maxfev=10000
                    )
                    y_pred = logarithmic_model(X_valid, *popt)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["logarithmic"][agent_idx] = r_squared
                    coefficients_dict["logarithmic"].append(popt)
                except RuntimeError:
                    r_squared_dict["logarithmic"][agent_idx] = np.nan
                    coefficients_dict["logarithmic"].append((np.nan, np.nan))

                # Model 4: Quadratic
                try:
                    popt, _ = curve_fit(quadratic_model, X_valid, y_valid, maxfev=10000)
                    y_pred = quadratic_model(X_valid, *popt)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["quadratic"][agent_idx] = r_squared
                    coefficients_dict["quadratic"].append(popt)
                except RuntimeError:
                    r_squared_dict["quadratic"][agent_idx] = np.nan
                    coefficients_dict["quadratic"].append((np.nan, np.nan))

                # Model 5: Power
                try:
                    popt, _ = curve_fit(power_model, X_valid, y_valid, maxfev=10000)
                    y_pred = power_model(X_valid, *popt)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["power"][agent_idx] = r_squared
                    coefficients_dict["power"].append(popt)
                except RuntimeError:
                    r_squared_dict["power"][agent_idx] = np.nan
                    coefficients_dict["power"].append((np.nan, np.nan))
            else:
                # Not enough data points
                for model in model_names:
                    r_squared_dict[model][agent_idx] = np.nan
                    coefficients_dict[model].append(None)

            # Plotting code for this agent

            # Create a new figure
            plt.figure(figsize=(10, 6))

            # Plot the data points
            plt.scatter(X_valid, y_valid, label="Data", color="black")

            # Generate x values for plotting fitted curves
            x_min = np.min(X_valid)
            x_max = np.max(X_valid)
            x_plot = np.linspace(x_min, x_max, 100)

            # Plot each fitted model with R² in the label
            for model in model_names:
                coeffs = coefficients_dict[model][agent_idx]
                r_squared = r_squared_dict[model][agent_idx]

                if (
                    coeffs is not None
                    and not any([np.isnan(c) for c in np.atleast_1d(coeffs)])
                    and not np.isnan(r_squared)
                ):
                    # Depending on the model, compute y values for plotting
                    if model == "linear":
                        a, b = coeffs
                        x_plot_log = np.log10(x_plot[x_plot > 0])
                        if len(x_plot_log) > 0:
                            y_plot = linear_model(x_plot_log, a, b)
                            plt.plot(
                                x_plot[x_plot > 0],
                                y_plot,
                                label=f"{model} (R²={r_squared:.3f})",
                                linewidth=2,
                            )
                    elif model == "exponential":
                        y_plot = exponential_model(x_plot, *coeffs)
                        plt.plot(
                            x_plot,
                            y_plot,
                            label=f"{model} (R²={r_squared:.3f})",
                            linewidth=2,
                        )
                    elif model == "logarithmic":
                        x_plot_positive = x_plot[x_plot > 0]
                        if len(x_plot_positive) > 0:
                            y_plot = logarithmic_model(x_plot_positive, *coeffs)
                            plt.plot(
                                x_plot_positive,
                                y_plot,
                                label=f"{model} (R²={r_squared:.3f})",
                                linewidth=2,
                            )
                    elif model == "quadratic":
                        y_plot = quadratic_model(x_plot, *coeffs)
                        plt.plot(
                            x_plot,
                            y_plot,
                            label=f"{model} (R²={r_squared:.3f})",
                            linewidth=2,
                        )
                    elif model == "power":
                        x_plot_positive = x_plot[x_plot > 0]
                        if len(x_plot_positive) > 0:
                            y_plot = power_model(x_plot_positive, *coeffs)
                            plt.plot(
                                x_plot_positive,
                                y_plot,
                                label=f"{model} (R²={r_squared:.3f})",
                                linewidth=2,
                            )
                else:
                    continue  # Skip models with invalid coefficients or R²

            # Add labels and legend
            plt.xlabel("SPEI Probability")
            plt.ylabel("Yield Ratio")
            plt.title(
                f"Agent {agent_idx}, irr class {self.var.farmer_class[agent_idx]}, crop {self.var.crop_calendar[agent_idx, 0, 0]} "
            )
            plt.legend()
            plt.grid(True)

            # Save the plot to a file
            filename = os.path.join(output_folder, f"agent_{agent_idx}.png")
            plt.savefig(filename)
            plt.close()

        # Compute median R² for each model
        for model in model_names:
            valid_r2 = r_squared_dict[model][~np.isnan(r_squared_dict[model])]
            median_r2 = np.median(valid_r2) if len(valid_r2) > 0 else np.nan
            print(f"Median R² for {model}: {median_r2}")

    def calculate_yield_spei_relation_test_group(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.optimize import curve_fit

        # Create unique groups based on agent properties
        crop_elevation_group = self.create_unique_groups(10)
        unique_crop_combinations, group_indices = np.unique(
            crop_elevation_group, axis=0, return_inverse=True
        )

        # Mask out empty rows (agents) where data is zero or NaN
        mask_agents = np.any(self.var.yearly_yield_ratio != 0, axis=1) & np.any(
            self.var.yearly_SPEI_probability != 0, axis=1
        )

        # Apply the mask to data and group indices
        masked_yearly_yield_ratio = self.var.yearly_yield_ratio[mask_agents, :]
        masked_SPEI_probability = self.var.yearly_SPEI_probability[mask_agents, :]
        group_indices = group_indices[mask_agents]

        # Number of groups
        n_groups = unique_crop_combinations.shape[0]

        # Define regression models
        def linear_model(X, a, b):
            return a * X + b

        def exponential_model(X, a, b):
            return a * np.exp(b * X)

        def logarithmic_model(X, a, b):
            return a * np.log(X) + b

        def quadratic_model(X, a, b, c):
            return a * X**2 + b * X + c

        def power_model(X, a, b):
            return a * X**b

        # Initialize dictionaries for coefficients and R² values
        model_names = ["linear", "exponential", "logarithmic", "quadratic", "power"]
        r_squared_dict = {model: np.full(n_groups, np.nan) for model in model_names}
        coefficients_dict = {model: [None] * n_groups for model in model_names}

        # Create a folder to save the plots
        output_folder = "plots/relation_test"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # For each group, perform regression with different models
        for group_idx in range(n_groups):
            # Get indices of agents in this group
            agent_indices = np.where(group_indices == group_idx)[0]

            if len(agent_indices) == 0:
                # No data for this group
                continue

            # Get data for the group
            y_data = masked_yearly_yield_ratio[
                agent_indices, :
            ]  # shape (num_agents_in_group, num_years)
            X_data = masked_SPEI_probability[agent_indices, :]  # same shape

            # Remove values where SPEI probability is greater than 1
            invalid_mask = X_data >= 1
            y_data[invalid_mask] = np.nan
            X_data[invalid_mask] = np.nan

            # Compute mean over agents in the group (axis=0 corresponds to years)
            y_group = np.nanmean(y_data, axis=0)  # shape (num_years,)
            X_group = np.nanmean(X_data, axis=0)  # same shape

            # Remove any years with NaN values
            valid_indices = (~np.isnan(y_group)) & (~np.isnan(X_group)) & (X_group > 0)
            y_group_valid = y_group[valid_indices]
            X_group_valid = X_group[valid_indices]

            if len(X_group_valid) >= 2:
                # Prepare data
                X_group_log = np.log10(X_group_valid)

                # Model 1: Linear in log-transformed X
                try:
                    popt, _ = curve_fit(
                        linear_model, X_group_log, y_group_valid, maxfev=10000
                    )
                    a, b = popt
                    y_pred = linear_model(X_group_log, a, b)
                    ss_res = np.sum((y_group_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["linear"][group_idx] = r_squared
                    coefficients_dict["linear"][group_idx] = (a, b)
                except (RuntimeError, ValueError):
                    pass  # Keep NaN in R² and None in coefficients

                # Model 2: Exponential
                try:
                    popt, _ = curve_fit(
                        exponential_model, X_group_valid, y_group_valid, maxfev=10000
                    )
                    y_pred = exponential_model(X_group_valid, *popt)
                    ss_res = np.sum((y_group_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["exponential"][group_idx] = r_squared
                    coefficients_dict["exponential"][group_idx] = popt
                except (RuntimeError, ValueError):
                    pass

                # Model 3: Logarithmic
                try:
                    popt, _ = curve_fit(
                        logarithmic_model, X_group_valid, y_group_valid, maxfev=10000
                    )
                    y_pred = logarithmic_model(X_group_valid, *popt)
                    ss_res = np.sum((y_group_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["logarithmic"][group_idx] = r_squared
                    coefficients_dict["logarithmic"][group_idx] = popt
                except (RuntimeError, ValueError):
                    pass

                # Model 4: Quadratic
                try:
                    popt, _ = curve_fit(
                        quadratic_model, X_group_valid, y_group_valid, maxfev=10000
                    )
                    y_pred = quadratic_model(X_group_valid, *popt)
                    ss_res = np.sum((y_group_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["quadratic"][group_idx] = r_squared
                    coefficients_dict["quadratic"][group_idx] = popt
                except (RuntimeError, ValueError):
                    pass

                # Model 5: Power
                try:
                    popt, _ = curve_fit(
                        power_model, X_group_valid, y_group_valid, maxfev=10000
                    )
                    y_pred = power_model(X_group_valid, *popt)
                    ss_res = np.sum((y_group_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["power"][group_idx] = r_squared
                    coefficients_dict["power"][group_idx] = popt
                except (RuntimeError, ValueError):
                    pass

                # Plotting code for this group
                plt.figure(figsize=(10, 6))
                plt.scatter(X_group_valid, y_group_valid, label="Data", color="black")

                # Generate x values for plotting fitted curves
                x_min = np.min(X_group_valid)
                x_max = np.max(X_group_valid)
                x_plot = np.linspace(x_min, x_max, 100)

                for model in model_names:
                    coeffs = coefficients_dict[model][group_idx]
                    r_squared = r_squared_dict[model][group_idx]

                    if (
                        coeffs is not None
                        and not any([np.isnan(c) for c in np.atleast_1d(coeffs)])
                        and not np.isnan(r_squared)
                    ):
                        if model == "linear":
                            a, b = coeffs
                            x_plot_positive = x_plot[x_plot > 0]
                            x_plot_log = np.log10(x_plot_positive)
                            if len(x_plot_log) > 0:
                                y_plot = linear_model(x_plot_log, a, b)
                                plt.plot(
                                    x_plot_positive,
                                    y_plot,
                                    label=f"{model} (R²={r_squared:.3f})",
                                    linewidth=2,
                                )
                        elif model == "exponential":
                            y_plot = exponential_model(x_plot, *coeffs)
                            plt.plot(
                                x_plot,
                                y_plot,
                                label=f"{model} (R²={r_squared:.3f})",
                                linewidth=2,
                            )
                        elif model == "logarithmic":
                            x_plot_positive = x_plot[x_plot > 0]
                            if len(x_plot_positive) > 0:
                                y_plot = logarithmic_model(x_plot_positive, *coeffs)
                                plt.plot(
                                    x_plot_positive,
                                    y_plot,
                                    label=f"{model} (R²={r_squared:.3f})",
                                    linewidth=2,
                                )
                        elif model == "quadratic":
                            y_plot = quadratic_model(x_plot, *coeffs)
                            plt.plot(
                                x_plot,
                                y_plot,
                                label=f"{model} (R²={r_squared:.3f})",
                                linewidth=2,
                            )
                        elif model == "power":
                            x_plot_positive = x_plot[x_plot > 0]
                            if len(x_plot_positive) > 0:
                                y_plot = power_model(x_plot_positive, *coeffs)
                                plt.plot(
                                    x_plot_positive,
                                    y_plot,
                                    label=f"{model} (R²={r_squared:.3f})",
                                    linewidth=2,
                                )
                # Add labels and legend
                plt.xlabel("SPEI Probability")
                plt.ylabel("Yield Ratio")
                plt.title(f"Group {group_idx}")
                plt.legend()
                plt.grid(True)

                # Save the plot to a file
                filename = os.path.join(output_folder, f"group_{group_idx}.png")
                plt.savefig(filename)
                plt.close()
            else:
                # Not enough data points for this group
                continue

        # Compute median R² for each model across all groups
        for model in model_names:
            valid_r2 = r_squared_dict[model][~np.isnan(r_squared_dict[model])]
            median_r2 = np.median(valid_r2) if len(valid_r2) > 0 else np.nan
            print(f"Median R² for {model}: {median_r2}")

        # Assign relations to agents based on their group
        # Here, we'll choose the model with the highest median R²
        # Alternatively, you can select the best model per group
        # For simplicity, we'll assign the linear model coefficients to agents

        # Example: Assign linear model coefficients to agents
        a_array = np.full(len(group_indices), np.nan)
        b_array = np.full(len(group_indices), np.nan)

        for group_idx in range(n_groups):
            if coefficients_dict["linear"][group_idx] is not None:
                a, b = coefficients_dict["linear"][group_idx]
                agent_mask = group_indices == group_idx
                a_array[agent_mask] = a
                b_array[agent_mask] = b

        # Assign to agents
        self.var.farmer_yield_probability_relation = np.column_stack((a_array, b_array))

        # Print overall best-fitting model based on median R²
        median_r2_values = {
            model: np.nanmedian(r_squared_dict[model]) for model in model_names
        }
        best_model_overall = max(median_r2_values, key=median_r2_values.get)
        print(f"Best-fitting model overall: {best_model_overall}")

    def calculate_yield_spei_relation(self):
        # Number of agents
        n_agents = self.var.yearly_yield_ratio.shape[0]

        # Initialize arrays for coefficients and R²
        a_array = np.zeros(n_agents)
        b_array = np.zeros(n_agents)
        r_squared_array = np.zeros(n_agents)

        # Loop over each agent
        for agent_idx in range(n_agents):
            # Get data for the agent
            y_data = self.var.yearly_yield_ratio[agent_idx, :]
            X_data = self.var.yearly_SPEI_probability[agent_idx, :]

            # Log-transform X_data, handling zeros
            with np.errstate(divide="ignore"):
                X_data_log = np.log10(X_data)

            # Mask out zeros and NaNs
            valid_mask = (
                (~np.isnan(y_data))
                & (~np.isnan(X_data_log))
                & (y_data != 0)
                & (X_data != 0)
            )
            y_valid = y_data[valid_mask]
            X_valid = X_data_log[valid_mask]

            if len(X_valid) >= 2:
                # Prepare matrices
                X_matrix = np.vstack([X_valid, np.ones(len(X_valid))]).T
                # Perform linear regression
                coefficients = np.linalg.lstsq(X_matrix, y_valid, rcond=None)[0]
                a, b = coefficients

                # Calculate R²
                y_pred = a * X_valid + b
                ss_res = np.sum((y_valid - y_pred) ** 2)
                ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            else:
                # Not enough data points
                a, b, r_squared = np.nan, np.nan, np.nan

            a_array[agent_idx] = a
            b_array[agent_idx] = b
            r_squared_array[agent_idx] = r_squared

        # Assign relations to agents
        self.var.farmer_yield_probability_relation = np.column_stack((a_array, b_array))

        # Print median R²
        valid_r2 = r_squared_array[~np.isnan(r_squared_array)]
        print("Median R²:", np.median(valid_r2) if len(valid_r2) > 0 else "N/A")

    def calculate_yield_spei_relation_group(
        self, yearly_yield_ratio, yearly_SPEI_probability
    ):
        # Create unique groups
        group_indices, n_groups = self.create_unique_groups(
            self.main_irrigation_source,
        )
        assert (np.any(self.var.yearly_SPEI_probability != 0, axis=1) > 0).all()

        # Apply the mask to data
        masked_yearly_yield_ratio = yearly_yield_ratio
        masked_SPEI_probability = yearly_SPEI_probability

        # Initialize arrays for coefficients and R²
        a_array = np.zeros(n_groups)
        b_array = np.zeros(n_groups)
        r_squared_array = np.zeros(n_groups)

        for group_idx in range(n_groups):
            agent_indices = np.where(group_indices == group_idx)[0]

            # Get data for the group
            y_data = masked_yearly_yield_ratio[agent_indices, :]
            X_data = masked_SPEI_probability[agent_indices, :]

            # Remove invalid values where SPEI probability >= 1 or yield <= 0
            mask = (X_data >= 1) | (y_data <= 0)
            y_data[mask] = np.nan
            X_data[mask] = np.nan

            # Compute mean over agents (axis=0 corresponds to years)
            y_group = np.nanmean(y_data, axis=0)  # shape (num_years,)
            X_group = np.nanmean(X_data, axis=0)  # shape (num_years,)

            # Remove any entries with NaN values
            valid_indices = (~np.isnan(y_group)) & (~np.isnan(X_group)) & (y_group > 0)
            y_group_valid = y_group[valid_indices]
            X_group_valid = X_group[valid_indices]

            if len(X_group_valid) >= 2:
                # Take the natural logarithm of y_group_valid
                ln_y_group = np.log(y_group_valid)

                # Prepare matrices for linear regression
                X_matrix = np.vstack([X_group_valid, np.ones(len(X_group_valid))]).T

                # Perform linear regression on ln(y) = b * X + ln(a)
                coefficients = np.linalg.lstsq(X_matrix, ln_y_group, rcond=None)[0]
                b, ln_a = coefficients
                a = np.exp(ln_a)

                # Calculate predicted y values
                y_pred = a * np.exp(b * X_group_valid)

                # Calculate R²
                ss_res = np.sum((y_group_valid - y_pred) ** 2)
                ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            else:
                # Not enough data points
                a, b, r_squared = np.nan, np.nan, np.nan

            a_array[group_idx] = a
            b_array[group_idx] = b
            r_squared_array[group_idx] = r_squared

        # Assign relations to agents
        farmer_yield_probability_relation = np.column_stack(
            (a_array[group_indices], b_array[group_indices])
        )

        # Print median R²
        weighted_r2 = r_squared_array[group_indices]
        valid_r2 = weighted_r2[~np.isnan(weighted_r2)]
        print(
            "Median R² for exponential model:",
            np.median(valid_r2) if len(valid_r2) > 0 else "N/A",
        )
        return farmer_yield_probability_relation

    def adapt_crops(self, farmer_yield_probability_relation) -> None:
        # Fetch loan configuration
        loan_duration = 2

        index = self.cultivation_costs[0].get(self.model.current_time)
        cultivation_cost = self.cultivation_costs[1][index]

        # Determine the cultivation costs of the current rotation
        current_crop_calendar = self.var.crop_calendar[:, :, 0].copy()
        mask_valid_crops = current_crop_calendar != -1
        rows, cols = np.nonzero(mask_valid_crops)
        costs = cultivation_cost[
            self.var.region_id[rows], current_crop_calendar[rows, cols]
        ]
        cultivation_costs_current_rotation = np.bincount(
            rows, weights=costs, minlength=current_crop_calendar.shape[0]
        ).astype(np.float32)

        annual_cost_empty = np.zeros(self.var.n, dtype=np.float32)

        # No constraint
        extra_constraint = np.ones_like(annual_cost_empty, dtype=bool)

        # Set variable which indicates all possible crop options
        unique_crop_calendars = np.unique(self.var.crop_calendar[:, :, 0], axis=0)

        timer_crops = TimingModule("crops_adaptation")

        (
            total_profits,
            profits_no_event,
            total_profits_adaptation,
            profits_no_event_adaptation,
            new_farmer_id,
        ) = self.profits_SEUT_crops(
            unique_crop_calendars, farmer_yield_probability_relation
        )
        timer_crops.new_split("profit_difference")
        total_annual_costs_m2 = (
            self.var.all_loans_annual_cost[:, -1, 0] / self.field_size_per_farmer
        )

        # Construct a dictionary of parameters to pass to the decision module functions
        decision_params = {
            "loan_duration": loan_duration,
            "expenditure_cap": self.var.expenditure_cap,
            "n_agents": self.var.n,
            "sigma": self.var.risk_aversion.data,
            "p_droughts": 1 / self.var.p_droughts[:-1],
            "profits_no_event": profits_no_event,
            "profits_no_event_adaptation": profits_no_event_adaptation,
            "total_profits": total_profits,
            "total_profits_adaptation": total_profits_adaptation,
            "risk_perception": self.var.risk_perception.data,
            "total_annual_costs": total_annual_costs_m2,
            "adaptation_costs": annual_cost_empty,
            "adapted": np.zeros(self.var.n, dtype=np.bool),
            "time_adapted": np.full(self.var.n, 2),
            "T": np.full(
                self.var.n,
                2,
            ),
            "discount_rate": self.var.discount_rate.data,
            "extra_constraint": extra_constraint,
        }

        # Determine the SEUT of the current crop
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing(**decision_params)

        # Determine the SEUT of the other crop options
        SEUT_crop_options = np.full(
            (self.var.n, len(unique_crop_calendars)), 0, dtype=np.float32
        )
        for idx, crop_option in enumerate(unique_crop_calendars[:, 0]):
            # Determine the cost difference between the old and potential new crop rotation
            new_id_option = new_farmer_id[:, idx]
            new_calendar_option = self.var.crop_calendar[new_id_option, :, 0]
            mask_valid_crops = new_calendar_option != -1

            rows, cols = np.nonzero(mask_valid_crops)
            costs = cultivation_cost[
                self.var.region_id[rows], new_calendar_option[rows, cols]
            ]
            cultivation_costs_new_rotation = np.bincount(
                rows, weights=costs, minlength=new_calendar_option.shape[0]
            ).astype(np.float32)

            cost_difference_adaptation = (
                cultivation_costs_new_rotation - cultivation_costs_current_rotation
            )

            # Update the decision parameters with the values of this potential crop rotation
            decision_params_option = copy.deepcopy(decision_params)
            decision_params_option.update(
                {
                    "total_profits_adaptation": total_profits_adaptation[idx, :, :],
                    "profits_no_event_adaptation": profits_no_event_adaptation[idx, :],
                    "adaptation_costs": cost_difference_adaptation,
                }
            )
            SEUT_crop_options[:, idx] = self.decision_module.calcEU_adapt(
                **decision_params_option
            )

        assert np.any(SEUT_do_nothing != -1) or np.any(SEUT_crop_options != -1)
        timer_crops.new_split("SEUT")
        # Determine the best adaptation option
        best_option_SEUT = np.max(SEUT_crop_options, axis=1)
        chosen_option = np.argmax(SEUT_crop_options, axis=1)

        # Determine the crop of the best option
        row_indices = np.arange(new_farmer_id.shape[0])
        new_id_temp = new_farmer_id[row_indices, chosen_option]

        # Determine for which agents it is beneficial to switch crops
        SEUT_adaptation_decision = (
            (best_option_SEUT > (SEUT_do_nothing)) & (new_id_temp != -1)
        )  # Filter out crops chosen due to small diff in do_nothing and adapt SEUT calculation

        # Adjust the intention threshold based on whether neighbors already have similar crop
        # Check for each farmer which crops their neighbors are cultivating
        social_network_crops = self.var.crop_calendar[self.var.social_network, :, 0]

        potential_new_rotation = self.var.crop_calendar[new_id_temp, :, 0]

        matches_per_neighbor = np.all(
            social_network_crops == potential_new_rotation[:, None, :], axis=2
        )
        # Check whether adapting agents have adaptation type in their network and create mask
        network_has_rotation = np.any(matches_per_neighbor, axis=1)

        # Increase intention factor if someone in network has crop
        intention_factor_adjusted = self.var.intention_factor.copy()
        intention_factor_adjusted[network_has_rotation] += 0.333

        # Determine whether it passed the intention threshold
        random_values = np.random.rand(*intention_factor_adjusted.shape)
        intention_mask = random_values < intention_factor_adjusted

        # # Set the adaptation mask
        SEUT_adaptation_decision = SEUT_adaptation_decision & intention_mask
        new_id_final = new_id_temp[SEUT_adaptation_decision]

        print("Crop switching farmers", np.count_nonzero(SEUT_adaptation_decision))

        assert not np.any(new_id_final == -1)

        # Switch their crops and update their yield-SPEI relation
        self.var.crop_calendar[SEUT_adaptation_decision, :, :] = self.var.crop_calendar[
            new_id_final, :, :
        ]

        # Update yield-SPEI relation
        self.var.yearly_income[SEUT_adaptation_decision, :] = self.var.yearly_income[
            new_id_final, :
        ]
        self.var.yearly_potential_income[SEUT_adaptation_decision, :] = (
            self.var.yearly_potential_income[new_id_final, :]
        )
        self.var.yearly_SPEI_probability[SEUT_adaptation_decision, :] = (
            self.var.yearly_SPEI_probability[new_id_final, :]
        )
        timer_crops.new_split("final steps")
        print(timer_crops)

    def adapt_irrigation_well(
        self,
        farmer_yield_probability_relation,
        average_extraction_speed,
        energy_cost,
        water_cost,
    ) -> None:
        """Handle the adaptation of farmers to irrigation wells.

        This function checks which farmers will adopt irrigation wells based on their expected utility
        and the provided constraints. It calculates the costs and the benefits for each farmer and updates
        their statuses (e.g., irrigation source, adaptation costs) accordingly.

        Todo:
            - Possibly externalize hard-coded values.
        """
        groundwater_depth = self.groundwater_depth.copy()
        groundwater_depth[groundwater_depth <= 0] = 0.001

        annual_cost, well_depth = self.calculate_well_costs_global(
            groundwater_depth, average_extraction_speed
        )

        # Compute the total annual per square meter costs if farmers adapt during this cycle
        # This cost is the cost if the farmer would adapt, plus its current costs of previous
        # adaptations
        total_annual_costs_m2 = (
            annual_cost + self.var.all_loans_annual_cost[:, -1, 0]
        ) / self.field_size_per_farmer

        # Solely the annual cost of the adaptation
        annual_cost_m2 = annual_cost / self.field_size_per_farmer

        loan_duration = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["loan_duration"]

        adapted = self.var.adaptations[:, WELL_ADAPTATION] > 0
        additional_diffentiator = (
            self.var.adaptations[:, PERSONAL_INSURANCE_ADAPTATION] > 0
        )
        # Reset farmers' status and irrigation type who exceeded the lifespan of their adaptation
        # and who's wells are much shallower than the groundwater depth
        self.reset_well_status(
            farmer_yield_probability_relation,
            adapted,
            groundwater_depth,
            additional_diffentiator,
        )

        # Define extra constraints (farmers' wells must reach groundwater)
        well_reaches_groundwater = self.var.well_depth > groundwater_depth
        extra_constraint = well_reaches_groundwater

        energy_cost_m2 = energy_cost / self.field_size_per_farmer
        water_cost_m2 = water_cost / self.field_size_per_farmer

        (
            energy_diff_m2,
            water_diff_m2,
        ) = self.adaptation_water_cost_difference(
            additional_diffentiator, adapted, energy_cost_m2, water_cost_m2
        )

        (
            total_profits,
            profits_no_event,
            total_profits_adaptation,
            profits_no_event_adaptation,
            ids_to_switch_to,
        ) = self.profits_SEUT(
            additional_diffentiator, adapted, farmer_yield_probability_relation
        )

        total_profits_adaptation = (
            total_profits_adaptation + energy_diff_m2 + water_diff_m2
        )
        profits_no_event_adaptation = (
            profits_no_event_adaptation + energy_diff_m2 + water_diff_m2
        )

        # Construct a dictionary of parameters to pass to the decision module functions
        decision_params = {
            "loan_duration": loan_duration,
            "expenditure_cap": self.var.expenditure_cap,
            "n_agents": self.var.n,
            "sigma": self.var.risk_aversion.data,
            "p_droughts": 1 / self.var.p_droughts[:-1],
            "total_profits_adaptation": total_profits_adaptation,
            "profits_no_event": profits_no_event,
            "profits_no_event_adaptation": profits_no_event_adaptation,
            "total_profits": total_profits,
            "risk_perception": self.var.risk_perception.data,
            "total_annual_costs": total_annual_costs_m2,
            "adaptation_costs": annual_cost_m2,
            "adapted": adapted,
            "time_adapted": self.var.time_adapted[:, WELL_ADAPTATION].data,
            "T": np.full(
                self.var.n,
                self.model.config["agent_settings"]["farmers"]["expected_utility"][
                    "adaptation_well"
                ]["decision_horizon"],
            ),
            "discount_rate": self.var.discount_rate.data,
            "extra_constraint": extra_constraint,
        }

        # Calculate the EU of not adapting and adapting respectively
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing(**decision_params)
        SEUT_adapt = self.decision_module.calcEU_adapt(**decision_params)

        assert (SEUT_do_nothing != -1).any() or (SEUT_adapt != -1).any()

        SEUT_adaptation_decision = self.update_adaptation_decision(
            adaptation_type=WELL_ADAPTATION,
            adapted=adapted,
            loan_duration=loan_duration,
            annual_cost=annual_cost,
            SEUT_do_nothing=SEUT_do_nothing,
            SEUT_adapt=SEUT_adapt,
            ids_to_switch_to=ids_to_switch_to,
        )

        # Set their well depth
        self.var.well_depth[SEUT_adaptation_decision] = well_depth[
            SEUT_adaptation_decision
        ]

        # Print the percentage of adapted households
        percentage_adapted = round(
            np.sum(self.var.adaptations[:, WELL_ADAPTATION] > 0) / self.var.n * 100,
            2,
        )
        print(
            "Irrigation well farms:",
            percentage_adapted,
            "(%)",
            "New/Renewed wells:",
            np.sum(SEUT_adaptation_decision),
            "(-)",
        )

    def adapt_irrigation_efficiency(
        self, farmer_yield_probability_relation, energy_cost, water_cost
    ) -> None:
        """Handle the adaptation of farmers to irrigation wells.

        This function checks which farmers will adopt irrigation wells based on their expected utility
        and the provided constraints. It calculates the costs and the benefits for each farmer and updates
        their statuses (e.g., irrigation source, adaptation costs) accordingly.

        Todo:
            - Possibly externalize hard-coded values.
        """
        # placeholder
        m2_adaptation_costs = np.full(
            self.var.n,
            self.model.config["agent_settings"]["farmers"]["expected_utility"][
                "adaptation_sprinkler"
            ]["m2_cost"],
        )

        loan_duration = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_sprinkler"]["loan_duration"]

        # Placeholder
        costs_irrigation_system = m2_adaptation_costs * self.field_size_per_farmer

        interest_rate = self.var.interest_rate.data

        annual_cost = costs_irrigation_system * (
            interest_rate
            * (1 + interest_rate) ** loan_duration
            / ((1 + interest_rate) ** loan_duration - 1)
        )

        total_annual_costs_m2 = (
            annual_cost + self.var.all_loans_annual_cost[:, -1, 0]
        ) / self.field_size_per_farmer

        # Solely the annual cost of the adaptation
        annual_cost_m2 = annual_cost / self.field_size_per_farmer

        # Create mask for those who have access to irrigation water
        has_irrigation_access = ~np.all(
            self.var.yearly_abstraction_m3_by_farmer[:, TOTAL_IRRIGATION, :] == 0,
            axis=1,
        )

        # Reset farmers' status and irrigation type who exceeded the lifespan of their adaptation
        # or who's never had access to irrigation water
        expired_adaptations = (
            self.var.time_adapted[:, IRRIGATION_EFFICIENCY_ADAPTATION]
            == self.var.lifespan_irrigation
        ) | has_irrigation_access
        self.var.adaptations[expired_adaptations, IRRIGATION_EFFICIENCY_ADAPTATION] = -1
        self.var.time_adapted[
            expired_adaptations, IRRIGATION_EFFICIENCY_ADAPTATION
        ] = -1

        # To determine the benefit of irrigation, those who have above 90% irrigation efficiency have adapted
        adapted = self.var.adapted[:, IRRIGATION_EFFICIENCY_ADAPTATION] > 0

        energy_cost_m2 = energy_cost / self.field_size_per_farmer
        water_cost_m2 = water_cost / self.field_size_per_farmer

        (
            energy_diff_m2,
            water_diff_m2,
        ) = self.adaptation_water_cost_difference(
            self.main_irrigation_source, adapted, energy_cost_m2, water_cost_m2
        )

        (
            total_profits,
            profits_no_event,
            total_profits_adaptation,
            profits_no_event_adaptation,
            ids_to_switch_to,
        ) = self.profits_SEUT(
            self.main_irrigation_source,
            adapted,
            farmer_yield_probability_relation,
        )

        total_profits_adaptation = (
            total_profits_adaptation + energy_diff_m2 + water_diff_m2
        )
        profits_no_event_adaptation = (
            profits_no_event_adaptation + energy_diff_m2 + water_diff_m2
        )

        # Construct a dictionary of parameters to pass to the decision module functions
        decision_params = {
            "loan_duration": loan_duration,
            "expenditure_cap": self.var.expenditure_cap,
            "n_agents": self.var.n,
            "sigma": self.var.risk_aversion.data,
            "p_droughts": 1 / self.var.p_droughts[:-1],
            "total_profits_adaptation": total_profits_adaptation,
            "profits_no_event": profits_no_event,
            "profits_no_event_adaptation": profits_no_event_adaptation,
            "total_profits": total_profits,
            "risk_perception": self.var.risk_perception.data,
            "total_annual_costs": total_annual_costs_m2,
            "adaptation_costs": annual_cost_m2,
            "adapted": adapted,
            "time_adapted": self.var.time_adapted[:, IRRIGATION_EFFICIENCY_ADAPTATION],
            "T": np.full(
                self.var.n,
                self.model.config["agent_settings"]["farmers"]["expected_utility"][
                    "adaptation_sprinkler"
                ]["decision_horizon"],
            ),
            "discount_rate": self.var.discount_rate.data,
            "extra_constraint": has_irrigation_access,
        }

        # Calculate the EU of not adapting and adapting respectively
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing(**decision_params)
        SEUT_adapt = self.decision_module.calcEU_adapt(**decision_params)

        assert (SEUT_do_nothing != -1).any() or (SEUT_adapt != -1).any()

        SEUT_adaptation_decision = self.update_adaptation_decision(
            adaptation_type=IRRIGATION_EFFICIENCY_ADAPTATION,
            adapted=adapted,
            loan_duration=loan_duration,
            annual_cost=annual_cost,
            SEUT_do_nothing=SEUT_do_nothing,
            SEUT_adapt=SEUT_adapt,
            ids_to_switch_to=ids_to_switch_to,
        )

        # Update irrigation efficiency for farmers who adapted
        self.var.irrigation_efficiency[SEUT_adaptation_decision] = 0.9

        # Print the percentage of adapted households
        percentage_adapted = round(
            np.sum(self.var.adapted[:, IRRIGATION_EFFICIENCY_ADAPTATION])
            / len(self.var.adapted[:, IRRIGATION_EFFICIENCY_ADAPTATION])
            * 100,
            2,
        )
        print("Irrigation efficient farms:", percentage_adapted, "(%)")

    def adapt_irrigation_expansion(
        self, farmer_yield_probability_relation, energy_cost, water_cost
    ) -> None:
        # Constants
        adaptation_type = 3

        loan_duration = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_sprinkler"]["loan_duration"]

        # If the farmers have drip/sprinkler irrigation, they would also have additional costs of expanding that
        # Costs are less than the initial expansion
        adapted_irr_eff = np.where((self.var.adapted[:, 2] == 1), 1, 0)
        total_costs = np.zeros(self.var.n, dtype=np.float32)
        total_costs[adapted_irr_eff] = 2 * self.field_size_per_farmer * 0.5

        interest_rate = self.var.interest_rate.data

        annual_cost = total_costs * (
            interest_rate
            * (1 + interest_rate) ** loan_duration
            / ((1 + interest_rate) ** loan_duration - 1)
        )

        # Farmers will have the same yearly water costs added if they expand
        annual_cost += energy_cost
        annual_cost += water_cost

        # Will also have the input/labor costs doubled
        annual_cost += np.sum(self.var.all_loans_annual_cost[:, 0, :], axis=1)

        total_annual_costs_m2 = (
            annual_cost + self.var.all_loans_annual_cost[:, -1, 0]
        ) / self.field_size_per_farmer

        annual_cost_m2 = annual_cost / self.field_size_per_farmer

        # Create mask for those who have access to irrigation water
        has_irrigation_access = ~np.all(
            self.var.yearly_abstraction_m3_by_farmer[:, TOTAL_IRRIGATION, :] == 0,
            axis=1,
        )

        # Reset farmers' status and irrigation type who exceeded the lifespan of their adaptation
        # or who's never had access to irrigation water
        expired_adaptations = (
            self.var.time_adapted[:, adaptation_type] == self.var.lifespan_irrigation
        ) | np.all(
            self.var.yearly_abstraction_m3_by_farmer[:, TOTAL_IRRIGATION, :] == 0,
            axis=1,
        )
        self.var.adapted[expired_adaptations, adaptation_type] = 0
        self.var.time_adapted[expired_adaptations, adaptation_type] = -1

        adapted = np.where((self.var.adapted[:, adaptation_type] == 1), 1, 0)

        (
            total_profits,
            profits_no_event,
            total_profits_adaptation,
            profits_no_event_adaptation,
            ids_to_switch_to,
        ) = self.profits_SEUT(
            self.main_irrigation_source,
            adapted,
            farmer_yield_probability_relation,
        )

        # Construct a dictionary of parameters to pass to the decision module functions
        decision_params = {
            "loan_duration": loan_duration,
            "expenditure_cap": self.var.expenditure_cap,
            "n_agents": self.var.n,
            "sigma": self.var.risk_aversion.data,
            "p_droughts": 1 / self.var.p_droughts[:-1],
            "total_profits_adaptation": total_profits_adaptation,
            "profits_no_event": profits_no_event,
            "profits_no_event_adaptation": profits_no_event_adaptation,
            "total_profits": total_profits,
            "risk_perception": self.var.risk_perception.data,
            "total_annual_costs": total_annual_costs_m2,
            "adaptation_costs": annual_cost_m2,
            "adapted": adapted,
            "time_adapted": self.var.time_adapted[:, adaptation_type],
            "T": np.full(
                self.var.n,
                self.model.config["agent_settings"]["farmers"]["expected_utility"][
                    "adaptation_sprinkler"
                ]["decision_horizon"],
            ),
            "discount_rate": self.var.discount_rate.data,
            "extra_constraint": has_irrigation_access,
        }

        # Calculate the EU of not adapting and adapting respectively
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing(**decision_params)
        SEUT_adapt = self.decision_module.calcEU_adapt(**decision_params)

        assert (SEUT_do_nothing != -1).any or (SEUT_adapt != -1).any()

        SEUT_adaptation_decision = self.update_adaptation_decision(
            adaptation_type=adaptation_type,
            adapted=adapted,
            loan_duration=loan_duration,
            annual_cost=annual_cost,
            SEUT_do_nothing=SEUT_do_nothing,
            SEUT_adapt=SEUT_adapt,
            ids_to_switch_to=ids_to_switch_to,
        )

        # Update irrigation efficiency for farmers who adapted
        self.var.fraction_irrigated_field[SEUT_adaptation_decision] = 1

        # Print the percentage of adapted households
        percentage_adapted = round(
            np.sum(self.var.adapted[:, adaptation_type])
            / len(self.var.adapted[:, adaptation_type])
            * 100,
            2,
        )
        print("Irrigation expanded farms:", percentage_adapted, "(%)")

    def adapt_insurance(
        self,
        adaptation_types,
        adaptation_names,
        farmer_yield_probability_relation_base,
        farmer_yield_probability_relations_insured,
        premiums,
    ):
        """Handle the adaptation of farmers to irrigation wells.

        This function checks which farmers will adopt irrigation wells based on their expected utility
        and the provided constraints. It calculates the costs and the benefits for each farmer and updates
        their statuses (e.g., irrigation source, adaptation costs) accordingly.

        Todo:
            - Possibly externalize hard-coded values.
        """
        loan_duration = self.var.insurance_duration
        interest_rate = self.var.interest_rate.data

        # Determine the income of each farmer with or without insurance
        # Profits without insurance
        regular_yield_ratios = self.convert_probability_to_yield_ratio(
            farmer_yield_probability_relation_base
        )
        total_profits = self.compute_total_profits(regular_yield_ratios)
        total_profits, profits_no_event = self.format_results(total_profits)

        SEUT_insurance_options = np.full(
            (self.var.n, len(premiums)), 0, dtype=np.float32
        )
        annual_cost_array = np.full((self.var.n, len(premiums)), 0, dtype=np.float32)

        for idx, adaptation_type in enumerate(adaptation_types):
            # Parameters
            farmer_yield_probability_relation_insured = (
                farmer_yield_probability_relations_insured[idx]
            )
            annual_cost = premiums[idx]

            # Reset farmers' adatations that exceeded their lifespan
            expired_adaptations = (
                self.var.time_adapted[:, adaptation_type] == loan_duration
            )
            self.var.adaptations[expired_adaptations, adaptation_type] = -1
            self.var.time_adapted[expired_adaptations, adaptation_type] = -1

            adapted = self.var.adaptations[:, adaptation_type] > 0

            if len(adaptation_types) > 1:
                # Define extra constraints -- cant adapt another insurance type while having one before
                other_masks = [
                    (self.var.adaptations[:, t] < 0)
                    for t in adaptation_types
                    if t != adaptation_type
                ]
                extra_constraint = np.logical_or.reduce(other_masks)
            else:
                extra_constraint = np.ones_like(adapted, dtype=bool)

            # Compute profits with index insurance
            annual_cost = annual_cost * (
                interest_rate
                * (1 + interest_rate) ** loan_duration
                / ((1 + interest_rate) ** loan_duration - 1)
            )
            annual_cost_array[:, idx] = annual_cost

            # Total cost = adaptation costs + existing loans
            total_annual_costs_m2 = (
                annual_cost + self.var.all_loans_annual_cost[:, -1, 0]
            ) / self.field_size_per_farmer

            # Solely the annual cost of the adaptation
            annual_cost_m2 = annual_cost / self.field_size_per_farmer

            # Determine the would be income with insurance
            insured_yield_ratios = self.convert_probability_to_yield_ratio(
                farmer_yield_probability_relation_insured
            )
            total_profits_insured = self.compute_total_profits(insured_yield_ratios)
            total_profits_index_insured, profits_no_event_index_insured = (
                self.format_results(total_profits_insured)
            )

            # Construct a dictionary of parameters to pass to the decision module functions
            decision_params = {
                "loan_duration": loan_duration,
                "expenditure_cap": self.var.expenditure_cap,
                "n_agents": self.var.n,
                "sigma": self.var.risk_aversion.data,
                "p_droughts": 1 / self.var.p_droughts[:-1],
                "total_profits_adaptation": total_profits_index_insured,
                "profits_no_event": profits_no_event,
                "profits_no_event_adaptation": profits_no_event_index_insured,
                "total_profits": total_profits,
                "risk_perception": self.var.risk_perception.data,
                "total_annual_costs": total_annual_costs_m2,
                "adaptation_costs": annual_cost_m2,
                "adapted": adapted,
                "time_adapted": self.var.time_adapted[:, adaptation_type].data,
                "T": np.full(
                    self.var.n,
                    self.model.config["agent_settings"]["farmers"]["expected_utility"][
                        "adaptation_well"
                    ]["decision_horizon"],
                ),
                "discount_rate": self.var.discount_rate.data,
                "extra_constraint": extra_constraint,
            }

            SEUT_insurance_options[:, idx] = self.decision_module.calcEU_adapt(
                **decision_params
            )

        # Calculate the EU of not adapting and adapting respectively
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing(**decision_params)

        assert (SEUT_do_nothing != -1).any() or (SEUT_insurance_options != -1).any()

        best_option_SEUT = np.max(SEUT_insurance_options, axis=1)
        chosen_option = np.argmax(SEUT_insurance_options, axis=1)

        for idx, adaptation_type in enumerate(adaptation_types):
            adapted = self.var.adaptations[:, adaptation_type] > 0
            annual_cost = annual_cost_array[:, idx]
            adaptation_name = adaptation_names[idx]

            mask_highest_SEUT = chosen_option == idx

            SEUT_decision_array = np.full(self.var.n, -np.inf, dtype=np.float32)
            SEUT_decision_array[mask_highest_SEUT] = best_option_SEUT[mask_highest_SEUT]

            SEUT_adaptation_decision = self.update_adaptation_decision(
                adaptation_type=adaptation_type,
                adapted=adapted,
                loan_duration=loan_duration,
                annual_cost=annual_cost,
                SEUT_do_nothing=SEUT_do_nothing,
                SEUT_adapt=SEUT_decision_array,
                ids_to_switch_to=np.arange(self.var.n),
            )

            if np.count_nonzero(SEUT_adaptation_decision) > 0:
                assert np.min(SEUT_decision_array[SEUT_adaptation_decision]) != -np.inf

            # Print the percentage of adapted households
            percentage_adapted = round(
                np.sum(self.var.adaptations[:, adaptation_type] > 0) / self.var.n * 100,
                2,
            )

            print(
                f"{adaptation_name} Insured farms:",
                percentage_adapted,
                "(%)",
                "New/Renewed insurance contracts:",
                np.sum(SEUT_adaptation_decision),
                "(-)",
            )

    def update_adaptation_decision(
        self,
        adaptation_type,
        adapted,
        loan_duration,
        annual_cost,
        SEUT_do_nothing,
        SEUT_adapt,
        ids_to_switch_to,
    ):
        # Compare EU values for those who haven't adapted yet and get boolean results
        SEUT_adaptation_decision = SEUT_adapt > SEUT_do_nothing

        social_network_adaptation = adapted[self.var.social_network]

        # Check whether adapting agents have adaptation type in their network and create mask
        network_has_adaptation = np.any(social_network_adaptation == 1, axis=1)

        # Increase intention factor if someone in network has crop
        intention_factor_adjusted = self.var.intention_factor.copy()
        intention_factor_adjusted[network_has_adaptation] += 0.33

        # Determine whether it passed the intention threshold
        random_values = np.random.rand(*intention_factor_adjusted.shape)
        intention_mask = random_values < intention_factor_adjusted

        SEUT_adaptation_decision = SEUT_adaptation_decision & intention_mask

        # Update the adaptation status
        self.var.adaptations[SEUT_adaptation_decision, adaptation_type] = 1

        # Reset the timer for newly adapting farmers and update timers for others
        self.var.time_adapted[SEUT_adaptation_decision, adaptation_type] = 0
        self.var.time_adapted[
            self.var.time_adapted[:, adaptation_type] != -1, adaptation_type
        ] += 1

        # Update annual costs and disposable income for adapted farmers
        self.var.all_loans_annual_cost[
            SEUT_adaptation_decision, adaptation_type + 1, 0
        ] += annual_cost[SEUT_adaptation_decision]
        self.var.all_loans_annual_cost[SEUT_adaptation_decision, -1, 0] += annual_cost[
            SEUT_adaptation_decision
        ]  # Total loan amount

        # set loan tracker
        self.var.loan_tracker[SEUT_adaptation_decision, adaptation_type + 1, 0] += (
            loan_duration
        )

        # Update yield-SPEI relation
        new_id_final = ids_to_switch_to[SEUT_adaptation_decision]

        self.var.yearly_income[SEUT_adaptation_decision, :] = self.var.yearly_income[
            new_id_final, :
        ]
        self.var.yearly_potential_income[SEUT_adaptation_decision, :] = (
            self.var.yearly_potential_income[new_id_final, :]
        )
        self.var.yearly_SPEI_probability[SEUT_adaptation_decision, :] = (
            self.var.yearly_SPEI_probability[new_id_final, :]
        )
        return SEUT_adaptation_decision

    def calculate_water_costs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the water and energy costs per agent and the average extraction speed.

        This method computes the energy costs for agents using groundwater, the water costs for all agents
        depending on their water source, and the average extraction speed per agent. It also updates the
        loans and annual costs associated with water and energy use.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - energy_costs (np.ndarray): Energy costs per agent (LCU/year).
                - water_costs (np.ndarray): Water costs per agent (LCU/year).
                - average_extraction_speed (np.ndarray): Average water extraction speed per agent (m³/s).
        """
        # Get electricity costs per agent based on their region and current time
        electricity_costs = np.full(
            self.var.n,
            self.get_value_per_farmer_from_region_id(
                self.electricity_cost, self.model.current_time
            ),
            dtype=np.float32,
        )

        # Initialize energy and water costs arrays
        energy_costs = np.zeros(self.var.n, dtype=np.float32)
        water_costs = np.zeros(self.var.n, dtype=np.float32)

        # Compute total pump duration per agent (average over crops)
        crop_growth_duration = self.var.crop_calendar[:, :, 2].data
        total_pump_duration = np.where(
            crop_growth_duration == -1, 0, crop_growth_duration
        ).sum(axis=1)

        # Get groundwater depth per agent and ensure non-negative values
        groundwater_depth = self.groundwater_depth.copy()
        groundwater_depth[groundwater_depth < 0] = 0

        # Create unique groups based on crop combinations and elevation
        has_well = self.var.adaptations[:, WELL_ADAPTATION] > 0
        group_indices, n_groups = self.create_unique_groups()

        # Compute yearly water abstraction per m² per agent
        yearly_abstraction_m3_per_m2 = (
            self.var.yearly_abstraction_m3_by_farmer
            / self.field_size_per_farmer[..., None, None]
        )

        # Initialize array to store average extraction per agent
        average_extraction_m2 = np.full(self.var.n, np.nan, dtype=np.float32)
        # Loop over each unique crop group to compute average extraction
        for group_idx in range(n_groups):
            farmers_in_group = group_indices == group_idx
            # Only select the agents with a well
            farmers_in_group_with_well = np.where(farmers_in_group & has_well)

            # Extract the abstraction values for the group, excluding zeros
            extraction_values = yearly_abstraction_m3_per_m2[
                farmers_in_group_with_well, :3, :
            ]
            non_zero_extractions = extraction_values[extraction_values != 0]

            # Compute average extraction for the group if there are non-zero values
            if non_zero_extractions.size > 0:
                average_extraction = np.mean(non_zero_extractions)  # m³ per m² per year
            else:
                average_extraction = 0.0

            # Store the average extraction for each group
            average_extraction_m2[farmers_in_group] = average_extraction

        # Compute average extraction per agent (m³/year)
        average_extraction = average_extraction_m2 * self.field_size_per_farmer

        # Compute average extraction speed per agent (m³/s)
        average_extraction_speed = (
            average_extraction / 365 / self.var.pump_hours / 3600
        )  # Convert from m³/year to m³/s

        # Create boolean masks for different types of water sources
        main_irrigation_sources = self.main_irrigation_source

        mask_channel = main_irrigation_sources == CHANNEL_IRRIGATION
        mask_reservoir = main_irrigation_sources == RESERVOIR_IRRIGATION
        mask_groundwater = main_irrigation_sources == GROUNDWATER_IRRIGATION

        # Compute power required for groundwater extraction per agent (kW)
        power = (
            self.var.specific_weight_water
            * groundwater_depth[mask_groundwater]
            * average_extraction_speed[mask_groundwater]
            / self.var.pump_efficiency
        ) / 1000  # Convert from W to kW

        # Compute energy consumption per agent (kWh/year)
        energy = power * (total_pump_duration[mask_groundwater] * self.var.pump_hours)

        # Get energy cost rate per agent (LCU per kWh)
        energy_cost_rate = electricity_costs[mask_groundwater]

        # Compute energy costs per agent (LCU/year) for groundwater irrigating farmers
        energy_costs[mask_groundwater] = energy * energy_cost_rate

        # Compute water costs for agents using channel water (LCU/year)
        water_costs[mask_channel] = (
            average_extraction[mask_channel] * self.var.water_costs_m3_channel
        )

        # Compute water costs for agents using reservoir water (LCU/year)
        water_costs[mask_reservoir] = (
            average_extraction[mask_reservoir] * self.var.water_costs_m3_reservoir
        )

        # Compute water costs for agents using groundwater (LCU/year)
        water_costs[mask_groundwater] = (
            average_extraction[mask_groundwater] * self.var.water_costs_m3_groundwater
        )

        # Assume minimal interest rate as farmers pay directly
        interest_rate_farmer = 0.0001  # Annual interest rate
        loan_duration = 2  # Loan duration in years

        # Compute annual cost of water and energy using annuity formula
        # A = P * [r(1+r)^n] / [(1+r)^n -1], where P is principal, r is interest rate, n is loan duration
        annuity_factor = (
            interest_rate_farmer
            * (1 + interest_rate_farmer) ** loan_duration
            / ((1 + interest_rate_farmer) ** loan_duration - 1)
        )
        annual_cost_water_energy = (water_costs + energy_costs) * annuity_factor

        # Update loan records with the annual cost of water and energy
        for i in range(4):
            # Find the first available loan slot
            if np.all(self.var.all_loans_annual_cost.data[:, 5, i] == 0):
                self.var.all_loans_annual_cost.data[:, 5, i] = annual_cost_water_energy
                self.var.loan_tracker[annual_cost_water_energy > 0, 5, i] = (
                    loan_duration
                )
                break

        # Add the annual cost to the total loan annual costs
        self.var.all_loans_annual_cost.data[:, -1, 0] += annual_cost_water_energy

        return energy_costs, water_costs, average_extraction_speed

    def calculate_well_costs_global(
        self, groundwater_depth: np.ndarray, average_extraction_speed: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the annual costs associated with well installation and operation globally.

        This function computes the annual costs for installing wells, maintaining them, and the energy costs
        associated with pumping groundwater for each agent (farmer). It takes into account regional variations
        in costs and agent-specific parameters such as groundwater depth and extraction speed.

        Args:
            groundwater_depth (np.ndarray): Array of groundwater depths per agent (in meters).
            average_extraction_speed (np.ndarray): Array of average water extraction speeds per agent (m³/s).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            annual_cost (np.ndarray): Annual cost per agent (local currency units per year).
            potential_well_length (np.ndarray): Potential well length per agent (in meters).
        """
        # Retrieve aquifer-specific unit costs for well drilling per meter
        well_cost_class_1 = self.get_value_per_farmer_from_region_id(
            self.why_10, self.model.current_time
        )
        well_cost_class_2 = self.get_value_per_farmer_from_region_id(
            self.why_20, self.model.current_time
        )
        well_cost_class_3 = self.get_value_per_farmer_from_region_id(
            self.why_30, self.model.current_time
        )

        # Initialize the well unit cost array with zeros
        well_unit_cost = np.zeros_like(self.var.why_class, dtype=np.float32)

        # Assign unit costs to each agent based on their well class using boolean indexing
        well_unit_cost[self.var.why_class == 1] = well_cost_class_1[
            self.var.why_class == 1
        ]
        well_unit_cost[self.var.why_class == 2] = well_cost_class_2[
            self.var.why_class == 2
        ]
        well_unit_cost[self.var.why_class == 3] = well_cost_class_3[
            self.var.why_class == 3
        ]

        # Get electricity costs per agent based on their region and current time
        electricity_costs = self.get_value_per_farmer_from_region_id(
            self.electricity_cost, self.model.current_time
        )

        # Calculate potential well length per agent
        # Potential well length is the sum of the maximum initial saturated thickness and the groundwater depth
        potential_well_length = self.var.max_initial_sat_thickness + groundwater_depth

        # Calculate the installation cost per agent (cost per meter * potential well length)
        install_cost = well_unit_cost * potential_well_length

        # Calculate maintenance cost per agent (as a fraction of the installation cost)
        maintenance_cost = self.var.maintenance_factor * install_cost

        # Calculate the total pump duration per agent (average over crops)
        crop_growth_duration = self.var.crop_calendar[:, :, 2].data
        total_pump_duration = np.where(
            crop_growth_duration == -1, 0, crop_growth_duration
        ).sum(axis=1)  # days

        # Calculate the power required per agent for pumping groundwater (in kilowatts)
        # specific_weight_water (N/m³), groundwater_depth (m), average_extraction_speed (m³/s), pump_efficiency (%)
        power = (
            self.var.specific_weight_water
            * groundwater_depth
            * average_extraction_speed
            / self.var.pump_efficiency
        ) / 1000  # Convert from watts to kilowatts

        # Calculate the energy consumption per agent (in kilowatt-hours)
        # power (kW), total_pump_duration (days), pump_hours (hours per day)
        energy = power * (total_pump_duration * self.var.pump_hours)

        # Calculate the energy cost per agent (USD per year)
        energy_cost = energy * electricity_costs

        # Fetch loan configuration for well installation
        loan_config = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]
        loan_duration = loan_config["loan_duration"]  # Loan duration in years

        # Calculate annuity factor for loan repayment using the annuity formula
        # A = P * [r(1+r)^n] / [(1+r)^n -1], where:
        # A = annual payment, P = principal amount (install_cost), r = interest rate, n = loan duration
        interest_rate = self.var.interest_rate.data

        n = loan_duration
        annuity_factor = (interest_rate * (1 + interest_rate) ** n) / (
            (1 + interest_rate) ** n - 1
        )

        # Calculate the annual cost per agent (local currency units per year)
        annual_cost = (install_cost * annuity_factor) + energy_cost + maintenance_cost

        return annual_cost, potential_well_length

    def profits_SEUT(
        self, additional_diffentiators, adapted, farmer_yield_probability_relation
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Calculate total profits under different drought probability scenarios, with and without adaptation measures for adaptation types 0 and 1.

        Args:
            additional_diffentiators: Additional differentiators for grouping agents.
            adapted (np.ndarray): An array indicating which agents have adapted (relevant for adaptation_type == 1).
            farmer_yield_probability_relation (np.ndarray): Yield probability relation for farmers.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            total_profits (np.ndarray): Total profits under different drought scenarios without adaptation.
            profits_no_event (np.ndarray): Profits under the 'no drought' scenario without adaptation.
            total_profits_adaptation (np.ndarray): Total profits under different drought scenarios with adaptation.
            profits_no_event_adaptation (np.ndarray): Profits under the 'no drought' scenario with adaptation.
        """
        # Main function logic
        yield_ratios = self.convert_probability_to_yield_ratio(
            farmer_yield_probability_relation
        )

        # Compute profits without adaptation
        total_profits = self.compute_total_profits(yield_ratios)
        total_profits, profits_no_event = self.format_results(total_profits)

        gains_adaptation, ids_to_switch_to = self.adaptation_yield_ratio_difference(
            additional_diffentiators, adapted, yield_ratios
        )
        # Clip yield ratios to physical boundaries of 0 and 1
        yield_ratios_adaptation = np.clip(yield_ratios + gains_adaptation, 0, 1)
        # Compute profits with adaptation
        total_profits_adaptation = self.compute_total_profits(yield_ratios_adaptation)
        total_profits_adaptation, profits_no_event_adaptation = self.format_results(
            total_profits_adaptation
        )
        return (
            total_profits,
            profits_no_event,
            total_profits_adaptation,
            profits_no_event_adaptation,
            ids_to_switch_to,
        )

    def profits_SEUT_crops(
        self, unique_crop_calendars, farmer_yield_probability_relation
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Calculate total profits under different drought probability scenarios with crop adaptation measures (Adaptation Type 2).

        Returns:
            Tuple containing profits without adaptation, profits with adaptation options, and additional data.
        """
        # Main function logic
        yield_ratios = self.convert_probability_to_yield_ratio(
            farmer_yield_probability_relation
        )

        # Compute profits without adaptation
        total_profits = self.compute_total_profits(yield_ratios)
        profit_events, profits_no_event = self.format_results(total_profits)
        insurance_differentiator = (
            self.var.adaptations[:, PERSONAL_INSURANCE_ADAPTATION] > 0
        )

        crop_elevation_group = np.hstack(
            (
                self.var.crop_calendar[:, :, 0].data,
                self.farmer_command_area.reshape(-1, 1),
                self.up_or_downstream.reshape(-1, 1),
                np.where(
                    self.var.yearly_abstraction_m3_by_farmer[:, CHANNEL_IRRIGATION, 0]
                    > 0,
                    1,
                    0,
                ).reshape(-1, 1),
                insurance_differentiator.reshape(-1, 1),
            )
        )

        unique_crop_groups, group_indices = np.unique(
            crop_elevation_group, axis=0, return_inverse=True
        )

        # Calculate the yield gains for crop switching for different farmers
        (
            profit_gains,
            new_farmer_id,
        ) = crop_profit_difference_njit_parallel(
            yearly_profits=self.var.yearly_income.data
            / self.field_size_per_farmer[..., None],
            crop_elevation_group=crop_elevation_group,
            unique_crop_groups=unique_crop_groups,
            group_indices=group_indices,
            crop_calendar=self.var.crop_calendar.data,
            unique_crop_calendars=unique_crop_calendars,
            p_droughts=self.var.p_droughts,
            past_window=5,
        )

        total_profits_adaptation = np.full(
            (len(unique_crop_calendars), len(self.var.p_droughts[:-1]), self.var.n),
            0.0,
            dtype=np.float32,
        )
        profits_no_event_adaptation = np.full(
            (len(unique_crop_calendars), self.var.n),
            0.0,
            dtype=np.float32,
        )

        for crop_option in range(len(unique_crop_calendars)):
            profit_gains_option = profit_gains[:, crop_option]
            profits_adaptation_option = total_profits + profit_gains_option[..., None]

            (
                total_profits_adaptation_option,
                profits_no_event_adaptation_option,
            ) = self.format_results(profits_adaptation_option)

            total_profits_adaptation[crop_option, :, :] = (
                total_profits_adaptation_option
            )
            profits_no_event_adaptation[crop_option, :] = (
                profits_no_event_adaptation_option
            )

        return (
            profit_events,
            profits_no_event,
            total_profits_adaptation,
            profits_no_event_adaptation,
            new_farmer_id,
        )

    def compute_total_profits(self, yield_ratios: np.ndarray) -> np.ndarray:
        """Compute total profits for all agents across different drought scenarios.

        Args:
            yield_ratios (np.ndarray): Yield ratios for agents under different drought scenarios.
            crops_mask (np.ndarray): Mask indicating valid crop entries.
            nan_array (np.ndarray): Array filled with NaNs for reference.

        Returns:
            np.ndarray: Total profits for agents under each drought scenario.
        """
        crops_mask = (self.var.crop_calendar[:, :, 0] >= 0) & (
            self.var.crop_calendar[:, :, 0]
            < len(self.var.crop_data["reference_yield_kg_m2"])
        )
        nan_array = np.full_like(
            self.var.crop_calendar[:, :, 0], fill_value=np.nan, dtype=float
        )
        total_profits = np.zeros((self.var.n, yield_ratios.shape[1]))
        for col in range(yield_ratios.shape[1]):
            total_profits[:, col] = self.yield_ratio_to_profit(
                yield_ratios[:, col], crops_mask, nan_array
            )
        return total_profits

    def format_results(
        self, total_profits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transpose and slice the total profits matrix, and extract the 'no drought' scenario profits.

        Args:
            total_profits (np.ndarray): Total profits matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transposed profits and 'no drought' scenario profits.
        """
        total_profits = total_profits.T
        profits_no_event = total_profits[-1, :]
        total_profits = total_profits[:-1, :]
        return total_profits, profits_no_event

    def convert_probability_to_yield_ratio(
        self, farmer_yield_probability_relation
    ) -> np.ndarray:
        """Convert drought probabilities to yield ratios based on the given polynomial relationship.

        For each farmer's yield-probability relationship (represented as a polynomial),
        this function calculates the inverse of the relationship and then applies the
        inverted polynomial to a set of given probabilities to obtain yield ratios.
        The resulting yield ratios are then adjusted to lie between 0 and 1. The final
        results are stored in `self.var.yield_ratios_drought_event`.

        Note:
            - It assumes that the polynomial relationship is invertible.
            - Adjusts yield ratios to be non-negative and capped at 1.0.
        """

        def logarithmic_function(probability, params):
            a = params[:, 0]
            b = params[:, 1]
            x = probability[:, np.newaxis]
            return a * np.log10(x) + b

        def exponential_function(probability, params):
            # Extract parameters
            a = params[:, 0]  # Shape: (num_agents,)
            b = params[:, 1]  # Shape: (num_agents,)
            x = probability  # Shape: (num_events,)

            # Reshape arrays for broadcasting
            a = a[:, np.newaxis]  # Shape: (num_agents, 1)
            b = b[:, np.newaxis]  # Shape: (num_agents, 1)
            x = x[np.newaxis, :]  # Shape: (1, num_events)

            # Compute the exponential function
            return a * np.exp(b * x)  # Shape: (num_agents, num_events)

        yield_ratios = exponential_function(
            1 / self.var.p_droughts, farmer_yield_probability_relation
        )

        # Adjust the yield ratios to lie between 0 and 1
        yield_ratios = np.clip(yield_ratios, 0, 1)

        # Store the results in a variable
        yield_ratios_drought_event = yield_ratios[:]

        return yield_ratios_drought_event

    def create_unique_groups(self, *additional_diffentiators):
        """Create unique groups based on elevation data and merge with crop calendar.

        Returns:
            numpy.ndarray: Merged array with crop calendar and elevation distribution groups.
        """
        if additional_diffentiators:
            agent_classes = np.stack(
                [self.var.farmer_base_class.data, *additional_diffentiators], axis=1
            )
        else:
            agent_classes = self.var.farmer_base_class
        groups, group_indices = np.unique(agent_classes, axis=0, return_inverse=True)
        return group_indices, groups.shape[0]

    def adaptation_yield_ratio_difference(
        self, additional_diffentiators, adapted: np.ndarray, yield_ratios
    ) -> np.ndarray:
        """Calculate the relative yield ratio improvement for farmers adopting a certain adaptation.

        This function determines how much better farmers that have adopted a particular adaptation
        are doing in terms of their yield ratio as compared to those who haven't.

        The additional differentiator must be different from the adapted class

        Returns:
            An array representing the relative yield ratio improvement for each agent.

        TO DO: vectorize
        """
        group_indices, n_groups = self.create_unique_groups(additional_diffentiators)

        # Initialize array to store relative yield ratio improvement for unique groups
        gains_adaptation = np.zeros(
            (self.var.n, self.var.p_droughts.size),
            dtype=np.float32,
        )
        ids_to_switch_to = np.full(
            self.var.n,
            -1,
            dtype=np.int32,
        )
        # Loop over each unique group of farmers to determine their average yield ratio
        for group_idx in range(n_groups):
            # Agents in the current group
            group_members = group_indices == group_idx

            # Split agents into adapted and unadapted within the group
            unadapted_agents = group_members & (~adapted)
            adapted_agents = group_members & adapted

            if unadapted_agents.any() and adapted_agents.any():
                # Calculate mean yield ratio over past years for both adapted and unadapted groups
                unadapted_yield_ratio = np.mean(
                    yield_ratios[unadapted_agents, :], axis=0
                )
                adapted_yield_ratio = np.mean(yield_ratios[adapted_agents, :], axis=0)

                yield_ratio_gain = adapted_yield_ratio - unadapted_yield_ratio

                id_to_switch_to = find_most_similar_index(
                    yield_ratio_gain,
                    yield_ratios,
                    adapted_agents,
                )

                ids_to_switch_to[group_members] = id_to_switch_to
                gains_adaptation[group_members, :] = yield_ratio_gain

        assert np.max(gains_adaptation) != np.inf, "gains adaptation value is inf"

        return gains_adaptation, ids_to_switch_to

    def reset_well_status(
        self,
        farmer_yield_probability_relation,
        adapted,
        groundwater_depth,
        additional_diffentiator,
    ):
        expired_adaptations = (
            self.var.time_adapted[:, WELL_ADAPTATION] == self.var.lifespan_well
        ) | (groundwater_depth > self.var.well_depth)
        self.var.adaptations[expired_adaptations, WELL_ADAPTATION] = -1
        self.var.time_adapted[expired_adaptations, WELL_ADAPTATION] = -1

        # Determine the IDS of the most similar group of yield
        (
            total_profits,
            profits_no_event,
            total_profits_adaptation,
            profits_no_event_adaptation,
            ids_to_switch_to,
        ) = self.profits_SEUT(
            additional_diffentiator, ~adapted, farmer_yield_probability_relation
        )
        # Update yield-SPEI relation
        new_id_final = ids_to_switch_to[expired_adaptations]
        own_nr = np.arange(new_id_final.shape[0])
        new_id_final = np.where(new_id_final == -1, own_nr, new_id_final)

        self.var.yearly_income[expired_adaptations, :] = self.var.yearly_income[
            new_id_final, :
        ]
        self.var.yearly_potential_income[expired_adaptations, :] = (
            self.var.yearly_potential_income[new_id_final, :]
        )
        self.var.yearly_SPEI_probability[expired_adaptations, :] = (
            self.var.yearly_SPEI_probability[new_id_final, :]
        )

    def adaptation_water_cost_difference(
        self, additional_diffentiators, adapted: np.ndarray, energy_cost, water_cost
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the relative yield ratio improvement for farmers adopting a certain adaptation.

        This function determines how much better farmers that have adopted a particular adaptation
        are doing in terms of their yield ratio as compared to those who haven't.

        Args:
            additional_diffentiators (np.ndarray): Additional differentiators for grouping agents.
            adapted (np.ndarray): Array indicating adaptation status (0 or 1) for each agent.
            energy_cost (np.ndarray): Array of energy costs for each agent.
            water_cost (np.ndarray): Array of water costs for each agent.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays representing the relative energy cost and water cost improvements for each agent.
        """
        assert adapted.dtype == bool, "adapted should be a boolean array"

        # Create unique groups based on elevation data
        group_indices, n_groups = self.create_unique_groups(additional_diffentiators)

        # Initialize arrays to store gains per group
        unique_water_cost_gain = np.zeros(n_groups, dtype=np.float32)
        unique_energy_cost_gain = np.zeros(n_groups, dtype=np.float32)

        # For each group, compute gains
        for group_idx in range(n_groups):
            # Agents in the current group
            group_members = group_indices == group_idx

            # Split agents into adapted and unadapted within the group
            unadapted_agents = group_members & (~adapted)
            adapted_agents = group_members & adapted

            # Check if both adapted and unadapted agents are present
            if np.any(unadapted_agents) and np.any(adapted_agents):
                # Calculate mean water and energy costs for unadapted agents
                unadapted_water_cost = np.mean(water_cost[unadapted_agents], axis=0)
                unadapted_energy_cost = np.mean(energy_cost[unadapted_agents], axis=0)

                # Calculate mean water and energy costs for adapted agents
                adapted_water_cost = np.mean(water_cost[adapted_agents], axis=0)
                adapted_energy_cost = np.mean(energy_cost[adapted_agents], axis=0)

                # Calculate gains
                water_cost_gain = adapted_water_cost - unadapted_water_cost
                energy_cost_gain = adapted_energy_cost - unadapted_energy_cost

                # Store gains for the group
                unique_water_cost_gain[group_idx] = water_cost_gain
                unique_energy_cost_gain[group_idx] = energy_cost_gain
            else:
                # If not enough data, set gains to zero or np.nan
                unique_water_cost_gain[group_idx] = 0  # or np.nan
                unique_energy_cost_gain[group_idx] = 0  # or np.nan

        # Map gains back to agents using group indices
        water_cost_adaptation_gain = unique_water_cost_gain[group_indices]
        energy_cost_adaptation_gain = unique_energy_cost_gain[group_indices]

        return energy_cost_adaptation_gain, water_cost_adaptation_gain

    def yield_ratio_to_profit(
        self, yield_ratios: np.ndarray, crops_mask: np.ndarray, nan_array: np.ndarray
    ) -> np.ndarray:
        """Convert yield ratios to monetary profit values.

        This function computes the profit values for each crop based on given yield ratios.
        The profit is calculated by multiplying the crop yield in kilograms per sqr. meter with
        the average crop price. The function leverages various data inputs, such as current crop
        prices and reference yields.

        Args:
            yield_ratios: The array of yield ratios for the crops.
            crops_mask: A mask that denotes valid crops, based on certain conditions.
            array_with_reference: An array initialized with NaNs, later used to store reference yields and crop prices.
            nan_array: An array filled with NaNs, used for initializing arrays.

        Returns:
            An array representing the profit values for each crop based on the given yield ratios.

        Note:
            - It assumes that the crop prices are non-NaN for the current model time.
            - The function asserts that crop yields in kg are always non-negative.

        TODO: Take the average crop prices over the last x years.
        """
        # Create blank arrays with only nans
        array_with_reference_yield = nan_array.copy()
        array_with_price = nan_array.copy()

        # Check if prices are monthly or yearly
        price_frequency = self.model.config["agent_settings"]["market"][
            "price_frequency"
        ]

        if price_frequency == "monthly":
            total_price = 0
            month_count = 0

            # Ending date and start date set to one year prior
            end_date = self.model.current_time
            start_date = datetime(end_date.year - 1, 1, 1)

            # Loop through each month from start_date to end_date to get the sum of crop costs over the past year
            current_date = start_date
            while current_date <= end_date:
                assert self.crop_prices[0] is not None, (
                    "behavior needs crop prices to work"
                )
                monthly_price = self.crop_prices[1][
                    self.crop_prices[0].get(current_date)
                ]
                total_price += monthly_price
                # Move to the next month
                if current_date.month == 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(
                        current_date.year, current_date.month + 1, 1
                    )
                month_count += 1

            # Calculate the average price over the last year
            crop_prices = total_price / month_count

        else:
            crop_prices = self.agents.market.crop_prices[self.var.region_id]

        # Assign the reference yield and current crop price to the array based on valid crop mask
        array_with_price[crops_mask] = np.take(
            crop_prices, self.var.crop_calendar[:, :, 0][crops_mask].astype(int)
        )
        assert not np.isnan(
            array_with_price[crops_mask]
        ).any()  # Ensure there are no NaN values in crop prices

        array_with_reference_yield[crops_mask] = np.take(
            self.var.crop_data["reference_yield_kg_m2"].values,
            self.var.crop_calendar[:, :, 0][crops_mask].astype(int),
        )

        # Calculate the product of the average reference yield and average crop price ignoring NaN values
        reference_profit_m2 = np.nansum(
            array_with_reference_yield * array_with_price, axis=1
        )
        assert (
            reference_profit_m2 >= 0
        ).all()  # Ensure all crop yields are non-negative

        # Calculate profit by multiplying yield with price
        profit_m2 = yield_ratios * reference_profit_m2

        return profit_m2

    def update_loans(self) -> None:
        # Subtract 1 off each loan duration, except if that loan is at 0
        self.var.loan_tracker -= self.var.loan_tracker != 0
        # If the loan tracker is at 0, cancel the loan amount and subtract it of the total
        expired_loan_mask = self.var.loan_tracker == 0

        # Add a column to make it the same shape as the loan amount array
        new_column = np.full((self.var.n, 1, 5), False)
        expired_loan_mask = np.column_stack((expired_loan_mask, new_column))

        # Sum the expired loan amounts
        ending_loans = expired_loan_mask * self.var.all_loans_annual_cost
        total_loan_reduction = np.sum(ending_loans, axis=(1, 2))

        # Subtract it from the total loans and set expired loans to 0
        self.var.all_loans_annual_cost[:, -1, 0] -= total_loan_reduction
        self.var.all_loans_annual_cost[expired_loan_mask] = 0

        # Adjust for inflation in separate array for export
        # Calculate the cumulative inflation from the start year to the current year for each farmer
        cumulative_inflation = np.prod(
            [
                self.get_value_per_farmer_from_region_id(
                    self.inflation_rate, datetime(year, 1, 1)
                )
                for year in range(
                    self.model.config["general"]["spinup_time"].year,
                    self.model.current_time.year + 1,
                )
            ],
            axis=0,
        )

        self.var.adjusted_annual_loan_cost = (
            self.var.all_loans_annual_cost / cumulative_inflation[..., None, None]
        )

        self.var.adjusted_yearly_income = (
            self.var.insured_yearly_income / cumulative_inflation[..., None]
        )

    def get_value_per_farmer_from_region_id(
        self, data, time, subset=None
    ) -> np.ndarray:
        index = data[0].get(time)
        if subset is not None:
            region_id = self.var.region_id[subset]
        else:
            region_id = self.var.region_id
        unique_region_ids, inv = np.unique(region_id, return_inverse=True)
        values = np.full_like(unique_region_ids, np.nan, dtype=np.float32)
        for i, region_id in enumerate(unique_region_ids):
            values[i] = data[1][region_id][index]
        return values[inv]

    @staticmethod
    @njit(cache=True)
    def field_size_per_farmer_numba(
        field_indices_by_farmer: np.ndarray,
        field_indices: np.ndarray,
        cell_area: np.ndarray,
    ) -> np.ndarray:
        """Gets the field size for each farmer.

        Args:
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.
            cell_area: Subarray of cell_area.

        Returns:
            field_size_per_farmer: Field size for each farmer in m2.
        """
        field_size_per_farmer = np.zeros(
            field_indices_by_farmer.shape[0], dtype=np.float32
        )
        for farmer in range(field_indices_by_farmer.shape[0]):
            for field in get_farmer_HRUs(
                field_indices, field_indices_by_farmer, farmer
            ):
                field_size_per_farmer[farmer] += cell_area[field]
        return field_size_per_farmer

    @property
    def field_size_per_farmer(self) -> np.ndarray:
        """Gets the field size for each farmer.

        Returns:
            field_size_per_farmer: Field size for each farmer in m2.
        """
        return self.field_size_per_farmer_numba(
            self.var.field_indices_by_farmer.data,
            self.var.field_indices,
            self.HRU.var.cell_area,
        )

    @property
    def is_irrigated(self):
        return (
            self.var.adaptations[:, [SURFACE_IRRIGATION_EQUIPMENT, WELL_ADAPTATION]] > 0
        ).any(axis=1)

    @property
    def irrigated_fields(self) -> np.ndarray:
        """Gets the indices of fields that are irrigated.

        Returns:
            irrigated_fields: Indices of fields that are irrigated.
        """
        irrigated_fields = np.take(
            self.is_irrigated,
            self.HRU.var.land_owners,
        )
        irrigated_fields[self.HRU.var.land_owners == -1] = False
        return irrigated_fields

    @property
    def groundwater_depth(self):
        groundwater_depth = get_farmer_groundwater_depth(
            self.var.n,
            self.model.hydrology.groundwater.groundwater_depth,
            self.HRU.var.HRU_to_grid,
            self.var.field_indices,
            self.var.field_indices_by_farmer.data,
            self.HRU.var.cell_area,
        )
        assert not np.isnan(groundwater_depth).any(), "groundwater depth is nan"
        return groundwater_depth

    def create_farmer_classes(self, *characteristics):
        agent_classes = np.unique(
            np.stack(characteristics), axis=1, return_inverse=True
        )[1]
        return agent_classes

    @property
    def main_irrigation_source(self):
        # Set to 0 if channel abstraction is bigger than reservoir and groundwater, 1 for reservoir, 2 for groundwater and -1 no abstraction
        main_irrigation_source = np.argmax(
            self.var.yearly_abstraction_m3_by_farmer[:, :TOTAL_IRRIGATION, 0],
            axis=1,
        )
        # Set to -1 for precipitation if there is no abstraction
        main_irrigation_source[
            self.var.yearly_abstraction_m3_by_farmer[:, :TOTAL_IRRIGATION, 0].sum(
                axis=1
            )
            == 0
        ] = NO_IRRIGATION
        return main_irrigation_source

    def step(self) -> None:
        """This function is called at the beginning of each timestep.

        Then, farmers harvest and plant crops.
        """
        if not self.model.simulate_hydrology:
            return

        timer = TimingModule("crop_farmers")

        self.harvest()
        timer.new_split("harvest")
        self.plant()
        timer.new_split("planting")

        self.water_abstraction_sum()
        timer.new_split("water abstraction calculation")

        ## yearly actions
        if self.model.current_time.month == 1 and self.model.current_time.day == 1:
            if self.model.current_time.year - 1 > self.model.spinup_start.year:
                # reset the irrigation limit, but only if a full year has passed already. Otherwise
                # the cumulative water deficit is not year completed.
                self.var.remaining_irrigation_limit_m3[:] = 0
                self.var.remaining_irrigation_limit_m3[:] = (
                    self.var.irrigation_limit_m3[:]
                )

                # Save SPEI after 1 year, otherwise doesnt line up with harvests
                self.save_yearly_spei()

            # Set yearly yield ratio based on the difference between saved actual and potential profit
            self.var.yearly_yield_ratio = (
                self.var.yearly_income / self.var.yearly_potential_income
            )

            # Create a DataFrame with command area and elevation
            df = pd.DataFrame(
                {
                    "command_area": self.farmer_command_area,
                    "elevation": self.var.elevation,
                }
            )

            # Compute group-specific median elevation
            df["group_median"] = df.groupby("command_area")["elevation"].transform(
                "median"
            )

            # Determine lower or higher part and assign distinct ids
            self.up_or_downstream = np.where(
                df["elevation"] <= df["group_median"], 0, 1
            )

            # create a unique index for each type of crop calendar that a farmer follows
            crop_calendar_group = np.unique(
                self.var.crop_calendar[:, :, 0], axis=0, return_inverse=True
            )[1]

            channel_irrigator = np.where(
                self.var.yearly_abstraction_m3_by_farmer[:, CHANNEL_IRRIGATION, 0] > 0,
                1,
                0,
            )

            self.var.farmer_base_class[:] = self.create_farmer_classes(
                crop_calendar_group,
                self.farmer_command_area,
                self.up_or_downstream,
                channel_irrigator,
            )

            print(
                "well",
                np.mean(
                    self.var.yearly_yield_ratio[
                        self.main_irrigation_source == GROUNDWATER_IRRIGATION, 1
                    ]
                ),
                "no irrigation",
                np.mean(
                    self.var.yearly_yield_ratio[
                        self.main_irrigation_source == NO_IRRIGATION, 1
                    ]
                ),
                "total_mean",
                np.mean(self.var.yearly_yield_ratio[:, 1]),
            )

            energy_cost, water_cost, average_extraction_speed = (
                self.calculate_water_costs()
            )

            timer.new_split("water & energy costs")

            if (
                not self.model.in_spinup
                and "ruleset" in self.config
                and not self.config["ruleset"] == "no-adaptation"
            ):
                # Determine the relation between drought probability and yield
                # self.calculate_yield_spei_relation()
                farmer_yield_probability_relation = (
                    self.calculate_yield_spei_relation_group(
                        self.var.yearly_yield_ratio, self.var.yearly_SPEI_probability
                    )
                )

                self.var.insured_yearly_income[:, 0] = self.var.yearly_income[
                    :, 0
                ].copy()

                timer.new_split("yield-spei relation")

                if (
                    self.personal_insurance_adaptation_active
                    or self.index_insurance_adaptation_active
                ):
                    # save the base relations for determining the difference with and without insurance
                    farmer_yield_probability_relation_base = (
                        farmer_yield_probability_relation.copy()
                    )
                    potential_insured_loss = self.potential_insured_loss()
                if self.personal_insurance_adaptation_active:
                    # Now determine the potential (past & current) indemnity payments and recalculate
                    # probability and yield relation
                    personal_premium = self.premium_personal_insurance()
                    farmer_yield_probability_relation_insured_personal = (
                        self.insured_yields(potential_insured_loss)
                    )

                    # Give only the insured agents the relation with covered losses
                    insured_agents_mask = (
                        self.var.adaptations[:, PERSONAL_INSURANCE_ADAPTATION] > 0
                    )

                    farmer_yield_probability_relation[insured_agents_mask, :] = (
                        farmer_yield_probability_relation_insured_personal[
                            insured_agents_mask, :
                        ]
                    )
                    timer.new_split("personal insurance")
                if self.index_insurance_adaptation_active:
                    # Calculate best strike, exit, rate for chosen contract
                    strike, exit, rate, index_premium = self.premium_index_insurance(
                        potential_insured_loss
                    )
                    potential_insured_loss_index = self.insured_payouts_index(
                        strike, exit, rate
                    )
                    farmer_yield_probability_relation_insured_index = (
                        self.insured_yields(potential_insured_loss_index)
                    )
                    timer.new_split("index insurance")

                # These adaptations can only be done if there is a yield-probability relation
                if not np.all(farmer_yield_probability_relation == 0):
                    if self.wells_adaptation_active:
                        self.adapt_irrigation_well(
                            farmer_yield_probability_relation,
                            average_extraction_speed,
                            energy_cost,
                            water_cost,
                        )
                        timer.new_split("irr well")
                    if self.sprinkler_adaptation_active:
                        self.adapt_irrigation_efficiency(
                            farmer_yield_probability_relation, energy_cost, water_cost
                        )

                        timer.new_split("irr efficiency")
                    if self.crop_switching_adaptation_active:
                        self.adapt_crops(farmer_yield_probability_relation)
                        timer.new_split("adapt crops")

                    if (
                        self.personal_insurance_adaptation_active
                        and self.index_insurance_adaptation_active
                    ):
                        # In scenario with both insurance, compare simultaneously
                        self.adapt_insurance(
                            np.array(
                                [
                                    PERSONAL_INSURANCE_ADAPTATION,
                                    INDEX_INSURANCE_ADAPTATION,
                                ]
                            ),
                            ["Personal", "Index"],
                            farmer_yield_probability_relation_base,
                            [
                                farmer_yield_probability_relation_insured_personal,
                                farmer_yield_probability_relation_insured_index,
                            ],
                            [personal_premium, index_premium],
                        )
                    elif self.personal_insurance_adaptation_active:
                        self.adapt_insurance(
                            [PERSONAL_INSURANCE_ADAPTATION],
                            ["Personal"],
                            farmer_yield_probability_relation_base,
                            [farmer_yield_probability_relation_insured_personal],
                            [personal_premium],
                        )
                        timer.new_split("pers. insurance")
                    elif self.index_insurance_adaptation_active:
                        self.adapt_insurance(
                            [INDEX_INSURANCE_ADAPTATION],
                            ["Index"],
                            farmer_yield_probability_relation_base,
                            [farmer_yield_probability_relation_insured_index],
                            [index_premium],
                        )

                        timer.new_split("index insurance")
                else:
                    raise AssertionError(
                        "Cannot adapt without yield - probability relation"
                    )

            advance_crop_rotation_year(
                current_crop_calendar_rotation_year_index=self.var.current_crop_calendar_rotation_year_index,
                crop_calendar_rotation_years=self.var.crop_calendar_rotation_years,
            )

            # Update loans
            self.update_loans()

            for i in range(len(self.var.yearly_abstraction_m3_by_farmer[0, :, 0])):
                shift_and_reset_matrix(
                    self.var.yearly_abstraction_m3_by_farmer[:, i, :]
                )

            # Shift the potential and yearly profits forward
            shift_and_reset_matrix(self.var.yearly_income)
            shift_and_reset_matrix(self.var.yearly_potential_income)
            shift_and_reset_matrix(self.var.insured_yearly_income)

            print(timer)
        # if self.model.current_timestep == 100:
        #     self.add_agent(indices=(np.array([310, 309]), np.array([69, 69])))
        # if self.model.current_timestep == 105:
        #     self.remove_agent(farmer_idx=1000)

        if self.model.timing:
            print(timer)

        self.report(self, locals())

    def remove_agents(
        self, farmer_indices: list[int], new_land_use_type: int
    ) -> np.ndarray:
        farmer_indices = np.array(farmer_indices)
        if farmer_indices.size > 0:
            farmer_indices = np.sort(farmer_indices)[::-1]
            HRUs_with_removed_farmers = []
            for idx in farmer_indices:
                HRUs_with_removed_farmers.append(
                    self.remove_agent(idx, new_land_use_type)
                )

        # TODO: remove the social network of the removed farmers only.
        # because farmers are removed and the current farmers may still
        # be looking for their friends that are gone, we need to reset
        # the social network.

        self.set_social_network()

        return np.concatenate(HRUs_with_removed_farmers)

    def remove_agent(self, farmer_idx: int, new_land_use_type: int) -> np.ndarray:
        assert farmer_idx >= 0, "Farmer index must be positive."
        assert farmer_idx < self.var.n, (
            "Farmer index must be less than the number of agents."
        )

        del self.var.activation_order_by_elevation_fixed

        last_farmer_HRUs = get_farmer_HRUs(
            self.var.field_indices, self.var.field_indices_by_farmer.data, -1
        )
        last_farmer_field_size = self.field_size_per_farmer[-1]  # for testing only

        # disown the farmer.
        HRUs_farmer_to_be_removed = get_farmer_HRUs(
            self.var.field_indices,
            self.var.field_indices_by_farmer.data,
            farmer_idx,
        )
        self.HRU.var.land_owners[HRUs_farmer_to_be_removed] = -1
        self.HRU.var.crop_map[HRUs_farmer_to_be_removed] = -1
        self.HRU.var.crop_age_days_map[HRUs_farmer_to_be_removed] = -1
        self.HRU.var.crop_harvest_age_days[HRUs_farmer_to_be_removed] = -1
        self.HRU.var.land_use_type[HRUs_farmer_to_be_removed] = new_land_use_type

        # reduce number of agents
        self.var.n -= 1

        if not self.var.n == farmer_idx:
            # move data of last agent to the index of the agent that is to be removed, effectively removing that agent.
            for name, agent_array in self.agent_arrays.items():
                agent_array[farmer_idx] = agent_array[-1]
                # reduce the number of agents by 1
                assert agent_array.n == self.var.n + 1
                agent_array.n = self.var.n

            # update the field indices of the last agent
            self.HRU.var.land_owners[last_farmer_HRUs] = farmer_idx
        else:
            for agent_array in self.agent_arrays.values():
                agent_array.n = self.var.n

        # TODO: Speed up field index updating.
        self.update_field_indices()
        self.activation_order_by_elevation  # recreate the activation order

        if self.var.n == farmer_idx:
            assert (
                get_farmer_HRUs(
                    self.var.field_indices,
                    self.var.field_indices_by_farmer.data,
                    farmer_idx,
                ).size
                == 0
            )
        else:
            assert np.array_equal(
                np.sort(last_farmer_HRUs),
                np.sort(
                    get_farmer_HRUs(
                        self.var.field_indices,
                        self.var.field_indices_by_farmer.data,
                        farmer_idx,
                    )
                ),
            )
            assert math.isclose(
                last_farmer_field_size,
                self.field_size_per_farmer[farmer_idx],
                abs_tol=1,
            )

        assert (self.HRU.var.land_owners[HRUs_farmer_to_be_removed] == -1).all()
        return HRUs_farmer_to_be_removed

    def add_agent(
        self,
        indices,
        values={
            "risk_aversion": 1,
            "interest_rate": 1,
            "discount_rate": 1,
            "adapted": False,
            "time_adapted": False,
            "SEUT_no_adapt": 1,
            "EUT_no_adapt": 1,
            "crops": -1,
            "irrigation_source": -1,
            "well_depth": -1,
            "channel_abstraction_m3_by_farmer": 0,
            "reservoir_abstraction_m3_by_farmer": 0,
            "groundwater_abstraction_m3_by_farmer": 0,
            "yearly_abstraction_m3_by_farmer": 0,
            "total_crop_age": 0,
            "per_harvest_yield_ratio": 0,
            "per_harvest_SPEI": 0,
            "monthly_SPEI": 0,
            "disposable_income": 0,
            "household_size": 2,
            "yield_ratios_drought_event": 1,
            "risk_perception": 1,
            "drought_timer": 1,
            "yearly_SPEI_probability": 1,
            "yearly_yield_ratio": 1,
            "yearly_income": 1,
            "yearly_potential_income": 1,
            "farmer_yield_probability_relation": 1,
            "irrigation_efficiency": 0.9,
            "base_management_yield_ratio": 1,
            "yield_ratio_management": 1,
            "annual_costs_all_adaptations": 1,
            "farmer_class": 1,
            "water_use": 1,
            "GEV_parameters": 1,
            "risk_perc_min": 1,
            "risk_perc_max": 1,
            "risk_decr": 1,
            "decision_horizon": 1,
        },
    ):
        """This function can be used to add new farmers."""
        HRU = self.model.data.split(indices)
        assert self.HRU.var.land_owners[HRU] == -1, "There is already a farmer here."
        self.HRU.var.land_owners[HRU] = self.var.n

        pixels = np.column_stack(indices)[:, [1, 0]]
        agent_location = np.mean(
            pixels_to_coords(pixels + 0.5, self.HRU.var.gt), axis=0
        )  # +.5 to use center of pixels

        self.var.n += 1  # increment number of agents
        for name, agent_array in self.agent_arrays.items():
            agent_array.n += 1
            if name == "locations":
                agent_array[self.var.n - 1] = agent_location
            elif name == "elevation":
                agent_array[self.var.n - 1] = self.elevation_subgrid.sample_coords(
                    np.expand_dims(agent_location, axis=0)
                )
            elif name == "region_id":
                agent_array[self.var.n - 1] = self.var.subdistrict_map.sample_coords(
                    np.expand_dims(agent_location, axis=0)
                )
            elif name == "field_indices_by_farmer":
                # TODO: Speed up field index updating.
                self.update_field_indices()
            else:
                agent_array[self.var.n - 1] = values[name]

    @property
    def n(self):
        return self.var._n

    @n.setter
    def n(self, value):
        self.var._n = value

    def get_farmer_elevation(self):
        # get elevation per farmer
        elevation_subgrid = load_grid(
            self.model.files["subgrid"]["landsurface/elevation"],
        )
        elevation_subgrid = np.nan_to_num(elevation_subgrid, copy=False, nan=0.0)
        decompressed_land_owners = self.HRU.decompress(self.HRU.var.land_owners)
        mask = decompressed_land_owners != -1
        return DynamicArray(
            np.bincount(
                decompressed_land_owners[mask],
                weights=elevation_subgrid[mask],
            )
            / np.bincount(decompressed_land_owners[mask]),
            max_n=self.var.max_n,
        )
