"""This module contains the CropFarmers agent class for the GEB model."""

from __future__ import annotations

import calendar
import copy
import math
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.stats import genextreme

from geb.types import ArrayInt64
from geb.workflows import TimingModule
from geb.workflows.io import read_grid
from geb.workflows.neighbors import find_neighbors
from geb.workflows.raster import pixels_to_coords, sample_from_map

from ..data import (
    DateIndex,
    load_crop_data,
    load_economic_data,
    load_regional_crop_data_from_dict,
)
from ..hydrology.landcovers import GRASSLAND_LIKE, NON_PADDY_IRRIGATED, PADDY_IRRIGATED
from ..store import Bucket, DynamicArray
from ..workflows import balance_check
from ..workflows.io import read_array
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

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel

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
PR_INSURANCE_ADAPTATION: int = 6


def _fit_linear(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute a least-squares linear fit for y = m * X + c.

    Fits a simple linear model using ordinary least squares and returns the
    slope and intercept. Used to fit the yield - spei relations for linear
    relations.

    Args:
        X: One-dimensional predictor values.
        y: One-dimensional response values.

    Returns:
        tuple[float, float]: The fitted `(m, c)` where `m` is the slope and
        `c` is the intercept.
    """
    Xmat = np.vstack([X, np.ones(len(X))]).T
    m, c = np.linalg.lstsq(Xmat, y, rcond=None)[0]
    return m, c


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Compute the coefficient of determination (R²).

    Used to determine the fit for the yield - spei relation.

    Args:
        y: True target values.
        yhat: Predicted target values.

    Returns:
        float: R² score. Returns ``nan`` if the variance of ``y`` is zero.
    """
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan


def cumulative_mean(
    mean: np.ndarray,
    counter: np.ndarray,
    update: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    """Update a cumulative mean in place.

    Computes the running mean for each element and writes results back into
    ``mean`` (and increments ``counter``). If ``mask`` is provided, only the
    masked elements are updated.

    Args:
        mean: Current cumulative mean array; updated in place.
        counter: Count of observations per element; updated in place.
        update: New observation(s) to incorporate.
        mask: Boolean mask selecting elements to
            update. If ``None``, all elements are updated. Defaults to ``None``.

    Notes:
        - ``mean``, ``counter``, and ``update`` should be broadcastable to the
          same shape.
        - ``mask`` (if given) should be boolean and broadcastable to the same
          shape as ``mean``.
    """
    if mask is not None:
        mean[mask] = (mean[mask] * counter[mask] + update[mask]) / (counter[mask] + 1)
        counter[mask] += 1
    else:
        mean[:] = (mean * counter + update) / (counter + 1)
        counter += 1


def shift_and_update(array: np.ndarray, update: np.ndarray | float | int) -> None:
    """Shift each row right by one and set the first column to ``update``.

    Args:
        array: 2D array modified in place; shape ``(n, m)`` with ``m >= 1``.
        update: Values assigned to the first column.
            May be a scalar or a 1D array of length ``n``.

    Notes:
        - The operation is in place: ``array`` is mutated.
        - If ``update`` is a 1D array, it is broadcast to ``array[:, 0]``.
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
) -> None:
    """Update the crop rotation year for each farmer. This function is used to update the crop rotation year for each farmer at the end of the year.

    Args:
        current_crop_calendar_rotation_year_index: The current crop rotation year for each farmer.
        crop_calendar_rotation_years: The number of years in the crop rotation cycle for each farmer.
    """
    current_crop_calendar_rotation_year_index[:] = (
        current_crop_calendar_rotation_year_index + 1
    ) % crop_calendar_rotation_years


class CropFarmersVariables(Bucket):
    """Variables for the CropFarmers agent."""

    n: int
    max_n: int
    channel_abstraction_m3_by_farmer: DynamicArray
    reservoir_abstraction_m3_by_farmer: DynamicArray
    groundwater_abstraction_m3_by_farmer: DynamicArray
    remaining_irrigation_limit_m3_channel: DynamicArray
    remaining_irrigation_limit_m3_reservoir: DynamicArray
    remaining_irrigation_limit_m3_groundwater: DynamicArray


class CropFarmers(AgentBaseClass):
    """The agent class for the farmers. Contains all data and behaviourial methods. The __init__ function only gets the model as arguments, the agent parent class and the redundancy. All other variables are loaded at later stages.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
        redundancy: a lot of data is saved in pre-allocated NumPy arrays. While this allows much faster operation, it does mean that the number of agents cannot grow beyond the size of the pre-allocated arrays. This parameter allows you to specify how much redundancy should be used. A lower redundancy means less memory is used, but the model crashes if the redundancy is insufficient.
    """

    var: CropFarmersVariables
    well_status: npt.NDArray[np.int32]
    insurance_diffentiator: npt.NDArray[np.int32]
    in_command_area: npt.NDArray[np.int32]
    elev_class: npt.NDArray[np.int32]

    def __init__(self, model: GEBModel, agents: Agents, reduncancy: float) -> None:
        """Initialize the CropFarmers agent module.

        Args:
            model: The GEB model.
            agents: The class that includes all agent types (allowing easier communication between agents).
            reduncancy: a lot of data is saved in pre-allocated NumPy arrays.
                While this allows much faster operation, it does mean that the number of agents cannot
                grow beyond the size of the pre-allocated arrays. This parameter allows you to specify
                how much redundancy should be used. A lower redundancy means less memory is used, but the
                model crashes if the redundancy is insufficient. The redundancy is specified as a fraction of
                the number of agents, e.g. 0.2 means 20% more space is allocated than the number of agents.

        """
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
        self.decision_module = DecisionModule()

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
        self.pr_insurance_adaptation_active = (
            not self.config["expected_utility"]["insurance"]["pr_insurance"]["ruleset"]
            == "no-adaptation"
        )
        self.microcredit_adaptation_active = (
            not self.config["microcredit"]["ruleset"] == "no-adaptation"
        )

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Return the name of the module.

        Returns:
            The name of the module.
        """
        return "agents.crop_farmers"

    def spinup(self) -> None:
        """Perform any necessary spinup for the crop farmers module.

        This method initializes all agent attributes, such as behavioral factors
        location, crop rotation, loans etc. Furthermore, it creates empty
        AgentArrays to store information about agents.

        """
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
        self.var.water_costs_m3_channel = self.model.config["agent_settings"][
            "farmers"
        ]["expected_utility"]["water_price"]["water_costs_m3_channel"]
        self.var.water_costs_m3_reservoir = self.model.config["agent_settings"][
            "farmers"
        ]["expected_utility"]["water_price"]["water_costs_m3_groundwater"]
        self.var.water_costs_m3_groundwater = self.model.config["agent_settings"][
            "farmers"
        ]["expected_utility"]["water_price"]["water_costs_m3_channel"]

        # Irr efficiency variables
        self.var.lifespan_irrigation = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["lifespan"]

        # load map of all subdistricts
        self.var.subdistrict_map = read_grid(
            self.model.files["region_subgrid"]["region_ids"]
        )
        region_mask = read_grid(self.model.files["region_subgrid"]["mask"])
        self.HRU_regions_map = np.zeros_like(self.HRU.mask, dtype=np.int8)
        self.HRU_regions_map[~self.HRU.mask] = self.var.subdistrict_map[
            region_mask == 0
        ]
        self.HRU_regions_map = self.HRU.convert_subgrid_to_HRU(self.HRU_regions_map)

        self.crop_prices = load_regional_crop_data_from_dict(
            self.model, "crops/crop_prices"
        )

        # Test with a high variable for now
        self.var.total_spinup_time = 20

        self.HRU.var.transpiration_crop_life = self.HRU.full_compressed(
            0, dtype=np.float32
        )
        self.HRU.var.potential_transpiration_crop_life = self.HRU.full_compressed(
            0, dtype=np.float32
        )
        self.HRU.var.transpiration_crop_life_per_crop_stage = np.zeros(
            (6, self.HRU.var.transpiration_crop_life.size), dtype=np.float32
        )
        self.HRU.var.potential_transpiration_crop_life_per_crop_stage = np.zeros(
            (6, self.HRU.var.potential_transpiration_crop_life.size),
            dtype=np.float32,
        )
        self.HRU.var.crop_map = np.full_like(self.HRU.var.land_owners, -1)
        self.HRU.var.crop_age_days_map = np.full_like(self.HRU.var.land_owners, -1)
        self.HRU.var.crop_harvest_age_days = np.full_like(self.HRU.var.land_owners, -1)

        """Calls functions to initialize all agent attributes, including their locations. Then, crops are initially planted."""
        # If initial conditions based on spinup period need to be loaded, load them. Otherwise, generate them.

        farms = self.model.hydrology.farms

        # Get number of farmers and maximum number of farmers that could be in the entire model run based on the redundancy.
        self.var.n = np.unique(farms[farms != -1]).size
        self.var.max_n = math.ceil(self.var.n * (1 + self.redundancy))

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
        self.var.risk_aversion[:] = read_array(
            self.model.files["array"]["agents/farmers/risk_aversion"]
        )

        self.var.discount_rate = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=np.nan
        )
        self.var.discount_rate[:] = read_array(
            self.model.files["array"]["agents/farmers/discount_rate"]
        )

        self.var.intention_factor = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=np.nan
        )

        self.var.intention_factor[:] = read_array(
            self.model.files["array"]["agents/farmers/intention_factor"]
        )

        self.var.interest_rate = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, dtype=np.float32, fill_value=0.05
        )
        self.var.interest_rate[:] = read_array(
            self.model.files["array"]["agents/farmers/interest_rate"]
        )

        # Load the region_code of each farmer.
        self.var.region_id = DynamicArray(
            input_array=read_array(
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
        self.var.crop_calendar[:] = read_array(
            self.model.files["array"]["agents/farmers/crop_calendar"]
        )
        # assert self.var.crop_calendar[:, :, 0].max() < len(self.var.crop_ids)

        self.var.crop_calendar_rotation_years = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.int32,
            fill_value=0,
        )
        self.var.crop_calendar_rotation_years[:] = read_array(
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
            read_array(self.model.files["array"]["agents/farmers/adaptations"]),
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

        self.var.cumulative_pr_during_growing_season = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
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
        self.var.remaining_irrigation_limit_m3_reservoir = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, fill_value=np.nan, dtype=np.float32
        )
        self.var.remaining_irrigation_limit_m3_channel = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, fill_value=np.nan, dtype=np.float32
        )
        self.var.remaining_irrigation_limit_m3_groundwater = DynamicArray(
            n=self.var.n, max_n=self.var.max_n, fill_value=np.nan, dtype=np.float32
        )
        self.var.irrigation_limit_reset_day_index = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.int32,
            fill_value=0,  # reset on day 0
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
        self.var.yearly_pr = DynamicArray(
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
        self.var.household_size[:] = read_array(
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
        self.var.irr_eff_surface = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_sprinkler"]["irr_eff_surface"]
        self.var.return_fraction_surface = self.model.config["agent_settings"][
            "farmers"
        ]["expected_utility"]["adaptation_sprinkler"]["return_fraction_surface"]

        self.var.irr_eff_drip = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_sprinkler"]["irr_eff_drip"]
        self.var.return_fraction_drip = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_sprinkler"]["return_fraction_drip"]

        self.var.return_fraction = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=self.var.return_fraction_surface,
        )
        self.var.irrigation_efficiency = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=self.var.irr_eff_surface,
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
        # 0 is input, 1 is microcredit, 2 is adaptation 1 (well), 3 is adaptation 2 (drip irrigation), 4 irr. field expansion,
        # 5 is water costs, 6 is personal insurance, 7 is index insurance last is total
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

        self.var.pr_premium = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=0,
        )

        self.var.index_premium = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=0,
        )

        self.var.personal_premium = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            dtype=np.float32,
            fill_value=0,
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

        why_map: np.ndarray = read_grid(self.model.files["grid"]["groundwater/why_map"])

        self.var.why_class[:] = sample_from_map(
            why_map, self.var.locations.data, self.grid.gt
        )
        # TODO: check why some agents sample outside map boundaries
        classes, counts = np.unique(
            self.var.why_class[self.var.why_class >= 0], return_counts=True
        )
        self.var.why_class[self.var.why_class < 0] = classes[counts.argmax()]

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
                GEV_grid.values,
                self.var.locations.data,
                GEV_grid.rio.transform().to_gdal(),
            )

        assert not np.all(np.isnan(self.var.GEV_parameters))

        self.var.GEV_pr_parameters = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(3,),
            extra_dims_names=("gev_parameters",),
            dtype=np.float32,
            fill_value=np.nan,
        )

        if (
            self.personal_insurance_adaptation_active
            or self.index_insurance_adaptation_active
            or self.pr_insurance_adaptation_active
        ):
            for i, varname in enumerate(["pr_gev_c", "pr_gev_loc", "pr_gev_scale"]):
                GEV_pr_grid = getattr(self.grid, varname)
                self.var.GEV_pr_parameters[:, i] = sample_from_map(
                    GEV_pr_grid, self.var.locations.data, self.grid.gt
                )

            assert not np.all(np.isnan(self.var.GEV_pr_parameters))

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

        self.var.cumulative_pr_mm = DynamicArray(
            n=self.var.n,
            max_n=self.var.max_n,
            extra_dims=(366,),
            extra_dims_names=("day",),
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
            field_indices_by_farmer,
            self.var.field_indices,
        ) = self.update_field_indices_numba(self.HRU.var.land_owners)

        self.var.field_indices_by_farmer[:] = field_indices_by_farmer

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

    def adjust_cultivation_costs(self) -> None:
        """Adjust cultivation costs based on configuration and calibration settings.

        Loads regional cultivation costs for crops, then either:
        (1) applies per-crop calibration factors when the model is configured to
        calibrate against ``"KGE_crops"``, or (2) scales costs by the configured
        ``cultivation_cost_fraction``. The updated values overwrite
        ``self.cultivation_costs`` in place.
        """
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
    def activation_order_by_elevation(self) -> DynamicArray:
        """Determine activation order by elevation, highest first.

        Agents with identical elevation are randomly shuffled among themselves. If
        ``agent_settings.fix_activation_order`` is enabled, a fixed permutation is
        used and cached for repeatability.
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

    def save_pr(self, pr_kg_per_m2_per_s: npt.NDArray[np.floating]) -> None:
        """Aggregate and store daily precipitation per farmer.

        Converts HRU precipitation (kg·m⁻²·s⁻¹) to mm/day, aggregates by
        land ownership, and writes the per-farmer values into
        ``self.var.cumulative_pr_mm[:, day_index]`` for the current day of year.
        On non-leap years, the value for day 365 is copied from day 364.

        Args:
            pr_kg_per_m2_per_s: Precipitation time-step
                series per HRU in kg·m⁻²·s⁻¹ (1 kg·m⁻² = 1 mm).

        """
        # take mean pr for day and convert to mm/day
        pr_mm_per_day = pr_kg_per_m2_per_s.sum(axis=0) * np.float32(3600)  # mm / day

        pr_mm_per_day_per_farmer = np.bincount(
            self.HRU.var.land_owners[self.HRU.var.land_owners != -1],
            weights=pr_mm_per_day[self.HRU.var.land_owners != -1],
        ) / np.bincount(self.HRU.var.land_owners[self.HRU.var.land_owners != -1])

        day_index = self.model.current_day_of_year - 1

        self.var.cumulative_pr_mm[:, day_index] = pr_mm_per_day_per_farmer

        if day_index == 364 and not calendar.isleap(self.model.current_time.year):
            self.var.cumulative_pr_mm[:, 365] = self.var.cumulative_pr_mm[:, 364]

    def save_water_deficit(
        self,
        reference_evapotranspiration_grass_m_per_day: npt.NDArray[np.floating],
        pr_kg_per_m2_per_s: npt.NDArray[np.floating],
        discount_factor: float = 0.2,
    ) -> None:
        """Accumulate daily water deficit per farmer with exponential smoothing.

        Computes daily water deficit in m³ from reference evapotranspiration and
        precipitation, aggregates by farmer, and updates
        ``self.var.cumulative_water_deficit_m3`` for the current day of year. Uses
        ``discount_factor`` for exponential smoothing of the daily series. On
        non-leap years, day 366 mirrors day 365.

        Args:
            reference_evapotranspiration_grass_m_per_day: Reference ET (m/day) per HRU.
            pr_kg_per_m2_per_s: Precipitation (kg·m⁻²·s⁻¹) per HRU per time-step.
            discount_factor: Smoothing factor in [0, 1] applied to
                the new day's deficit (higher values weight the current day more).
                Defaults to 0.2.
        """
        pr: npt.NDArray[np.float32] = pr_kg_per_m2_per_s.sum(axis=0) * np.float32(
            3600 / 1000
        )  # m / day
        water_deficit_day_m3 = (
            reference_evapotranspiration_grass_m_per_day - pr
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
        # leap years can cause issues when determining the future water deficit
        # to prevent this day index 365 is always set the same as day index 364
        # when we are in a leap year, the day is passed, otherwise it will only have 1 in 4 year data
        elif day_index == 365:
            pass
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
            if day_index == 364:
                self.var.cumulative_water_deficit_m3[:, 365] = (
                    self.var.cumulative_water_deficit_m3[:, 364]
                )

    def get_gross_irrigation_demand_m3(
        self,
        root_depth_m: npt.NDArray[np.float32],
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        """Calculates the gross irrigation demand in m3 for each farmer.

        Args:
            root_depth_m: root depth in meters for each HRU

        Returns:
            gross_irrigation_demand_m3: gross irrigation demand in m3 for each farmer
            gross_potential_irrigation_m3_limit_adjusted: adjusted gross potential irrigation in m3 limit for each farmer
        """
        (
            gross_potential_irrigation_m3,
            gross_potential_irrigation_m3_limit_adjusted_reservoir,
            gross_potential_irrigation_m3_limit_adjusted_channel,
            gross_potential_irrigation_m3_limit_adjusted_groundwater,
        ) = get_gross_irrigation_demand_m3(
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
            root_depth_m=root_depth_m,
            soil_layer_height=self.HRU.var.soil_layer_height,
            field_capacity=self.HRU.var.wfc,
            wilting_point=self.HRU.var.wwp,
            w=self.HRU.var.w,
            ws=self.HRU.var.ws,
            saturated_hydraulic_conductivity_m_per_day=self.HRU.var.saturated_hydraulic_conductivity_m_per_s
            * np.float32(86400),
            remaining_irrigation_limit_m3_reservoir=self.var.remaining_irrigation_limit_m3_reservoir.data,
            remaining_irrigation_limit_m3_channel=self.var.remaining_irrigation_limit_m3_channel.data,
            remaining_irrigation_limit_m3_groundwater=self.var.remaining_irrigation_limit_m3_groundwater.data,
            irrigation_limit_reset_day_index=self.var.irrigation_limit_reset_day_index.data,
            cumulative_water_deficit_m3=self.var.cumulative_water_deficit_m3.data,
            crop_calendar=self.var.crop_calendar.data,
            crop_group_numbers=self.var.crop_data["crop_group_number"].values.astype(
                np.float32
            ),
            paddy_irrigated_crops=self.var.crop_data["is_paddy"].values,
            current_crop_calendar_rotation_year_index=self.var.current_crop_calendar_rotation_year_index.data,
            max_paddy_water_level=self.var.max_paddy_water_level.data,
            minimum_effective_root_depth_m=self.model.hydrology.landsurface.var.minimum_effective_root_depth_m,
        )

        assert (
            gross_potential_irrigation_m3 < self.model.hydrology.HRU.var.cell_area
        ).all()
        return (
            gross_potential_irrigation_m3,
            gross_potential_irrigation_m3_limit_adjusted_reservoir,
            gross_potential_irrigation_m3_limit_adjusted_channel,
            gross_potential_irrigation_m3_limit_adjusted_groundwater,
        )

    @property
    def irrigation_limit_groundwater(self) -> np.ndarray:
        """Yearly groundwater irrigation limit per farmer (m³/year).

        Computed as an hourly maximum derived from groundwater depth, multiplied by
        the total hours (``365 * 5``). Follows the maximum flow rate in an (Indian) tubewell
        by Robert et al. (2018) https://doi.org/10.1016/j.ejor.2017.08.029
        """
        hourly_irrigation_maximum = 79.93 * (self.groundwater_depth + 0.01) ** -0.728
        # crop_growth_lengths = self.var.crop_calendar[:, :, 2].data
        # crop_growth_lengths = np.where(
        #     crop_growth_lengths == -1, 0, crop_growth_lengths
        # )
        total_hours = 365 * 5
        yearly_irrigation_total = hourly_irrigation_maximum * total_hours
        return yearly_irrigation_total

    @property
    def command_area(self) -> np.ndarray:
        """Which command area a farmer is in, derived from field indices and reservoir areas."""
        return farmer_command_area(
            self.var.n,
            self.var.field_indices,
            self.var.field_indices_by_farmer.data,
            self.HRU.var.reservoir_command_areas,
        )

    @property
    def is_in_command_area(self) -> np.ndarray:
        """Whether a farmer is in anu command area."""
        return self.command_area != -1

    @property
    def surface_irrigated(self) -> np.ndarray:
        """Boolean mask of farmers that have surface-irrigation equipment."""
        return self.var.adaptations[:, SURFACE_IRRIGATION_EQUIPMENT] > 0

    @property
    def reservoir_channel_irrigated(self) -> np.ndarray:
        """Per-farmer indicator (int8) of reservoir/channel irrigation.

        Combines reservoir access and any yearly channel abstraction.
        """
        return (self.command_area >= 0).astype(np.int8) | np.int8(
            self.var.yearly_abstraction_m3_by_farmer[:, CHANNEL_IRRIGATION, 0] > 0
        )

    @property
    def well_irrigated(self) -> np.ndarray:
        """Boolean mask of farmers that have a well adaptation."""
        return self.var.adaptations[:, WELL_ADAPTATION] > 0

    @property
    def irrigated(self) -> np.ndarray:
        """Boolean mask of farmers that are irrigated (surface or well)."""
        return self.surface_irrigated | self.well_irrigated  # | is the OR operator

    @property
    def currently_irrigated_fields(self) -> np.ndarray:
        """Boolean mask of fields currently irrigated (and with a valid crop)."""
        return self.farmer_to_field(self.is_irrigated, False) & (
            self.HRU.var.crop_map != -1
        )

    def abstract_water(
        self,
        gross_irrigation_demand_m3_per_field: npt.NDArray[np.float32],
        gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir: npt.NDArray[
            np.float32
        ],
        gross_irrigation_demand_m3_per_field_limit_adjusted_channel: npt.NDArray[
            np.float32
        ],
        gross_irrigation_demand_m3_per_field_limit_adjusted_groundwater: npt.NDArray[
            np.float32
        ],
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
        """Abstract water for per-source irrigation withdrawals.

        Delegates the core allocation to a Numba-accelerated routine and updates
        per-farmer withdrawals from channel, reservoir, and groundwater. Also
        returns field-scale fluxes in meters for the current step. Checks whether
        abstractions fit with change in hydrological storages.

        Args:
            gross_irrigation_demand_m3_per_field: Gross irrigation demand per field (m³).
            gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir: Demand per field (m³) limited by reservoir rules/capacity.
            gross_irrigation_demand_m3_per_field_limit_adjusted_channel: Demand per field (m³) limited by channel rules/capacity.
            gross_irrigation_demand_m3_per_field_limit_adjusted_groundwater: Demand per field (m³) limited by groundwater rules/capacity.
            available_channel_storage_m3: Available canal/channel storage per grid cell (m³).
            available_groundwater_m3: Available groundwater storage per grid cell (m³).
            groundwater_depth: Groundwater depth per groundwater grid cell (m).
            available_reservoir_storage_m3: Available reservoir storage per reservoir (m³).

        Returns:
            tuple containing:
                - water_withdrawal_m: Water withdrawn at field scale (m).
                - water_consumption_m: Consumed (non-returned) water at field scale (m).
                - returnFlowIrr_m: Return flow to the system at field scale (m).
                - addtoevapotrans_m: Irrigation water evaporated/transpired (m).
                - reservoir_abstraction_m3: Per-reservoir abstraction volumes (m³).
                - groundwater_abstraction_m3: Per-well/area groundwater abstraction volumes (m³).
        """
        assert (available_channel_storage_m3 >= 0).all()
        assert (available_groundwater_m3 >= 0).all()
        assert (available_reservoir_storage_m3 >= 0).all()

        gross_irrigation_demand_m3_per_farmer_limit_adjusted_reservoir = (
            self.field_to_farmer(
                gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir
            )
        )

        maximum_abstraction_reservoir_m3_by_farmer = (
            self.agents.reservoir_operators.get_maximum_abstraction_m3_by_farmer(
                self.command_area,
                gross_irrigation_demand_m3_per_farmer_limit_adjusted_reservoir,
            )
        )
        maximum_abstraction_channel_m3_by_farmer = self.field_to_farmer(
            gross_irrigation_demand_m3_per_field_limit_adjusted_channel
        )
        maximum_abstraction_groundwater_m3_by_farmer = self.field_to_farmer(
            gross_irrigation_demand_m3_per_field_limit_adjusted_groundwater
        )

        if __debug__:
            irrigation_limit_pre_reservoir = (
                self.var.remaining_irrigation_limit_m3_reservoir.copy()
            )
            irrigation_limit_pre_channel = (
                self.var.remaining_irrigation_limit_m3_channel.copy()
            )
            irrigation_limit_pre_groundwater = (
                self.var.remaining_irrigation_limit_m3_groundwater.copy()
            )
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
            field_indices_by_farmer=self.var.field_indices_by_farmer.data,
            field_indices=self.var.field_indices,
            irrigation_efficiency=self.var.irrigation_efficiency.data,
            surface_irrigated=self.surface_irrigated.data,
            well_irrigated=self.well_irrigated.data,
            cell_area=self.model.hydrology.HRU.var.cell_area,
            HRU_to_grid=self.HRU.var.HRU_to_grid,
            nearest_river_grid_cell=self.HRU.var.nearest_river_grid_cell,
            crop_map=self.HRU.var.crop_map,
            available_channel_storage_m3=available_channel_storage_m3,
            available_groundwater_m3=available_groundwater_m3,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
            groundwater_depth=groundwater_depth,
            command_area_by_farmer=self.command_area,
            return_fraction=self.var.return_fraction.data,
            well_depth=self.var.well_depth.data,
            remaining_irrigation_limit_m3_reservoir=self.var.remaining_irrigation_limit_m3_reservoir.data,
            remaining_irrigation_limit_m3_channel=self.var.remaining_irrigation_limit_m3_channel.data,
            remaining_irrigation_limit_m3_groundwater=self.var.remaining_irrigation_limit_m3_groundwater.data,
            maximum_abstraction_reservoir_m3_by_farmer=maximum_abstraction_reservoir_m3_by_farmer,
            maximum_abstraction_channel_m3_by_farmer=maximum_abstraction_channel_m3_by_farmer,
            maximum_abstraction_groundwater_m3_by_farmer=maximum_abstraction_groundwater_m3_by_farmer,
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
                influxes=[
                    self.var.channel_abstraction_m3_by_farmer,
                    self.var.reservoir_abstraction_m3_by_farmer,
                    self.var.groundwater_abstraction_m3_by_farmer,
                ],
                outfluxes=[
                    (water_withdrawal_m * self.model.hydrology.HRU.var.cell_area)
                ],
                tolerance=50,
            )

            # assert that the total amount of water withdrawn is equal to the total storage before and after abstraction
            balance_check(
                name="water withdrawal channel",
                how="sum",
                outfluxes=[self.var.channel_abstraction_m3_by_farmer],
                prestorages=[available_channel_storage_m3_pre],
                poststorages=[available_channel_storage_m3],
                tolerance=50,
            )

            balance_check(
                name="water withdrawal reservoir",
                how="sum",
                outfluxes=[self.var.reservoir_abstraction_m3_by_farmer],
                influxes=[reservoir_abstraction_m3],
                tolerance=50,
            )

            balance_check(
                name="water withdrawal groundwater",
                how="sum",
                outfluxes=[self.var.groundwater_abstraction_m3_by_farmer],
                influxes=[groundwater_abstraction_m3],
                tolerance=10,
            )

            # assert that the total amount of water withdrawn is equal to the total storage before and after abstraction
            balance_check(
                name="water withdrawal_2",
                how="sum",
                outfluxes=[
                    self.var.channel_abstraction_m3_by_farmer[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3_channel)
                    ].astype(np.float64),
                    self.var.reservoir_abstraction_m3_by_farmer[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3_reservoir)
                    ].astype(np.float64),
                    self.var.groundwater_abstraction_m3_by_farmer[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3_groundwater)
                    ].astype(np.float64),
                ],
                prestorages=[
                    irrigation_limit_pre_reservoir[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3_reservoir)
                    ].astype(np.float64),
                    irrigation_limit_pre_channel[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3_channel)
                    ].astype(np.float64),
                    irrigation_limit_pre_groundwater[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3_groundwater)
                    ].astype(np.float64),
                ],
                poststorages=[
                    self.var.remaining_irrigation_limit_m3_channel[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3_channel)
                    ].astype(np.float64),
                    self.var.remaining_irrigation_limit_m3_reservoir[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3_reservoir)
                    ].astype(np.float64),
                    self.var.remaining_irrigation_limit_m3_groundwater[
                        ~np.isnan(self.var.remaining_irrigation_limit_m3_groundwater)
                    ].astype(np.float64),
                ],
                tolerance=50,
            )

            # make sure the total water consumption plus 'wasted' irrigation water (evaporation + return flow) is equal to the total water withdrawal
            balance_check(
                name="water consumption",
                how="sum",
                influxes=[
                    water_consumption_m,
                    returnFlowIrr_m,
                    addtoevapotrans_m,
                ],
                outfluxes=[water_withdrawal_m],
                tolerance=50,
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
        crop_map: np.ndarray,
        evaporation_ratio: np.ndarray,
        evaporation_ratio_per_crop_stage: npt.NDArray[np.float32],
        KyT: npt.NDArray[np.float32],
        Ky1: npt.NDArray[np.float32],
        Ky2a: npt.NDArray[np.float32],
        Ky2b: npt.NDArray[np.float32],
        Ky3a: npt.NDArray[np.float32],
        Ky3b: npt.NDArray[np.float32],
        Ky4: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Calculate yield ratio based on https://doi.org/10.1016/j.jhydrol.2009.07.031.

        Args:
            crop_map: array of currently harvested crops.
            evaporation_ratio: ratio of actual to potential evapotranspiration of harvested crops.
            evaporation_ratio_per_crop_stage: ratio of actual to potential evapotranspiration per crop stage.
            KyT: Water stress reduction factor from GAEZ.
            Ky1: Water stress reduction factor for crop stage 1 from GAEZ.
            Ky2a: Water stress reduction factor for crop stage 2a from GAEZ.
            Ky2b: Water stress reduction factor for crop stage 2b from GAEZ.
            Ky3a: Water stress reduction factor for crop stage 3a from GAEZ.
            Ky3b: Water stress reduction factor for crop stage 3b from GAEZ.
            Ky4: Water stress reduction factor for crop stage 4 from GAEZ.

        Returns:
            yield_ratios: yield ratio (as ratio of maximum obtainable yield) per harvested crop.
        """
        yield_ratios = np.full(evaporation_ratio.size, -1, dtype=np.float32)

        assert crop_map.size == evaporation_ratio.size

        for i in range(evaporation_ratio.size):
            evap_ratio = evaporation_ratio[i]
            crop = crop_map[i]
            yield_ratio_crop = 1 - KyT[crop] * (1 - evap_ratio)

            if not np.isnan(evaporation_ratio_per_crop_stage[0, i]):
                yield_ratio_crop = np.minimum(
                    yield_ratio_crop,
                    1 - Ky1[crop] * (1 - evaporation_ratio_per_crop_stage[0, i]),
                )
            if not np.isnan(evaporation_ratio_per_crop_stage[1, i]):
                yield_ratio_crop = np.minimum(
                    yield_ratio_crop,
                    1 - Ky2a[crop] * (1 - evaporation_ratio_per_crop_stage[1, i]),
                )
            if not np.isnan(evaporation_ratio_per_crop_stage[2, i]):
                yield_ratio_crop = np.minimum(
                    yield_ratio_crop,
                    1 - Ky2b[crop] * (1 - evaporation_ratio_per_crop_stage[2, i]),
                )
            if not np.isnan(evaporation_ratio_per_crop_stage[3, i]):
                yield_ratio_crop = np.minimum(
                    yield_ratio_crop,
                    1 - Ky3a[crop] * (1 - evaporation_ratio_per_crop_stage[3, i]),
                )
            if not np.isnan(evaporation_ratio_per_crop_stage[4, i]):
                yield_ratio_crop = np.minimum(
                    yield_ratio_crop,
                    1 - Ky3b[crop] * (1 - evaporation_ratio_per_crop_stage[4, i]),
                )
            if not np.isnan(evaporation_ratio_per_crop_stage[5, i]):
                yield_ratio_crop = np.minimum(
                    yield_ratio_crop,
                    1 - Ky4[crop] * (1 - evaporation_ratio_per_crop_stage[5, i]),
                )

            yield_ratios[i] = np.maximum(yield_ratio_crop, 0)

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
    ) -> npt.NDArray[np.float32]:
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
        actual_transpiration: npt.NDArray[np.float32],
        potential_transpiration: npt.NDArray[np.float32],
        actual_transpiration_per_crop_stage: npt.NDArray[np.float32],
        potential_transpiration_per_crop_stage: npt.NDArray[np.float32],
        crop_map: np.ndarray,
    ) -> np.ndarray:
        """Gets yield ratio for each crop given the ratio between actual and potential evapostranspiration during growth.

        Args:
            harvest: Map of crops that are harvested.
            actual_transpiration: Actual evapotranspiration during crop growth period.
            potential_transpiration: Potential evapotranspiration during crop growth period.
            actual_transpiration_per_crop_stage: Actual evapotranspiration per crop stage.
            potential_transpiration_per_crop_stage: Potential evapotranspiration per crop stage.
            crop_map: Subarray of type of crop grown.

        Returns:
            yield_ratio: Map of yield ratio.

        Raises:
            ValueError: If crop data type is not GAEZ or MIRCA2000.
        """
        if self.var.crop_data_type == "GAEZ":
            yield_ratio: npt.NDArray[np.float32] = self.get_yield_ratio_numba_GAEZ(
                crop_map[harvest],
                evaporation_ratio=actual_transpiration[harvest]
                / potential_transpiration[harvest],
                evaporation_ratio_per_crop_stage=actual_transpiration_per_crop_stage[
                    :, harvest
                ]
                / potential_transpiration_per_crop_stage[:, harvest],
                KyT=self.var.crop_data["KyT"].values,
                Ky1=self.var.crop_data["Ky1"].values,
                Ky2a=self.var.crop_data["Ky2a"].values,
                Ky2b=self.var.crop_data["Ky2b"].values,
                Ky3a=self.var.crop_data["Ky3a"].values,
                Ky3b=self.var.crop_data["Ky3b"].values,
                Ky4=self.var.crop_data["Ky4"].values,
            )
        elif self.var.crop_data_type == "MIRCA2000":
            yield_ratio: npt.NDArray[np.float32] = self.get_yield_ratio_numba_MIRCA2000(
                crop_map[harvest],
                actual_transpiration[harvest] / potential_transpiration[harvest],
                self.var.crop_data["a"].values,
                self.var.crop_data["b"].values,
                self.var.crop_data["P0"].values,
                self.var.crop_data["P1"].values,
            )
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
        """Aggregate a field HRU-level array to farmer-level totals.

        Args:
            array: Values per field (compressed over owned fields).
            method: Aggregation method; only ``"sum"`` is supported.
                Defaults to ``"sum"``.

        Returns:
            npt.NDArray[np.floating]: Per-farmer aggregated values.
        """
        assert method == "sum", "Only sum is implemented"
        farmer_fields: npt.NDArray[np.int32] = self.HRU.var.land_owners[
            self.HRU.var.land_owners != -1
        ]
        masked_array: npt.NDArray[np.floating] = array[self.HRU.var.land_owners != -1]
        return np.bincount(farmer_fields, masked_array, minlength=self.var.n).astype(
            masked_array.dtype
        )

    def farmer_to_field(
        self,
        array: npt.NDArray,
        nodata: float | int | bool,
    ) -> npt.NDArray:
        """Expand a per-farmer array to per-field values.

        Args:
            array: Values per farmer.
            nodata: Fill value for fields without an owner.

        Returns:
            npt.NDArray: Values mapped to each field (same shape as ``land_owners``).
        """
        by_field = np.take(array, self.HRU.var.land_owners)
        by_field[self.HRU.var.land_owners == -1] = nodata
        return by_field

    def decompress(self, array: npt.NDArray) -> npt.NDArray:
        """Decompress a per-farmer array to the full HRU raster.

        Uses ``NaN`` for unowned fields when the dtype is floating, otherwise ``-1``.

        Args:
            array: Values per farmer.

        Returns:
            npt.NDArray: Decompressed array aligned with the HRU raster.
        """
        if np.issubdtype(array.dtype, np.floating):
            nofieldvalue = np.nan
        else:
            nofieldvalue = -1
        by_field = self.farmer_to_field(array, nodata=nofieldvalue)
        return self.HRU.decompress(by_field)

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Mask of invalid or unowned HRU cells."""
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

    def harvest(self) -> None:
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
            # Get yield ratio for the harvested crops
            yield_ratio_per_field = self.get_yield_ratio(
                harvest,
                self.HRU.var.transpiration_crop_life,
                self.HRU.var.potential_transpiration_crop_life,
                self.HRU.var.transpiration_crop_life_per_crop_stage,
                self.HRU.var.potential_transpiration_crop_life_per_crop_stage,
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

            number_of_harvesting_fields = np.count_nonzero(harvested_crops)
            print(
                f"Harvesting {number_of_harvesting_fields} fields with crops: "
                f"{np.unique(harvested_crops[harvested_crops >= 0])}"
            )
            if 23 in np.unique(harvested_crops[harvested_crops >= 0]):
                pass
            # it's okay for some crop prices to be nan, as they will be filtered out in the next step
            crop_prices = self.agents.market.crop_prices

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
            self.save_harvest_precipitation(
                harvesting_farmers, current_crop_age[harvesting_farmers]
            )
            self.drought_risk_perception(harvesting_farmers, current_crop_age)

            ## After updating the drought risk perception, set the previous month for the next timestep as the current for this timestep.
            # TODO: This seems a bit like a quirky solution, perhaps there is a better way to do this.
            self.var.previous_month = self.model.current_time.month

        else:
            self.income_farmer = np.zeros(self.var.n, dtype=np.float32)

        # Reset transpiration values for harvested fields
        self.HRU.var.transpiration_crop_life[harvest] = 0
        self.HRU.var.potential_transpiration_crop_life[harvest] = 0

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
        self,
        harvesting_farmers: np.ndarray,
        current_crop_age: np.ndarray,
    ) -> None:
        """Update drought risk perception for harvesting farmers.

        Computes farmers' risk perception from the difference between their latest
        profits and potential profits, adjusted for inflation and recent history.
        Farmers that experience a drought event have their drought timer reset.

        Args:
            harvesting_farmers: Indices of farmers currently harvesting.
            current_crop_age: Current crop age for each farmer.

        Todo:
            Perhaps move the constant to the model.yml.
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
        loaning_farmers: npt.NDArray[np.bool_],
        drought_loss_current: npt.NDArray[np.floating],
        current_crop_age: npt.NDArray[np.floating],
    ) -> None:
        """Compute and assign microcredit based on profits, drought loss, and crop age.

        Uses recent profits, the latest drought loss, and the fraction of the cropping
        period completed to size loans. Updates per-farmer loan costs and trackers.

        Args:
            loaning_farmers: Boolean mask of farmers applying for a loan.
            drought_loss_current: Latest drought loss (%)
                per farmer.
            current_crop_age: Current crop age per farmer
                (days or time units consistent with the crop calendar).
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

    def government_premium_cap(self) -> np.ndarray:
        """Compute per-farmer government premium cap based on income and crop mix.

        Farmers are grouped by well status. If all farmers in a group have
        sugarcane (``crop_calendar[..., -1, 0] == 4``), the cap is 5% of mean
        income per m²; otherwise 2%. Caps are then scaled by each farmer's field
        size.

        Returns:
            numpy.ndarray: Premium cap per farmer.
        """
        year_income_m2 = self.var.yearly_income[:, 0] / self.field_size_per_farmer

        group_indices, n_groups = self.create_unique_groups(
            self.well_status,
        )
        group_mean_cap = np.zeros(n_groups, dtype=float)
        for group_idx in range(n_groups):
            agent_indices = np.where(group_indices == group_idx)[0]
            sugarcane_check = np.all(self.var.crop_calendar[agent_indices, -1, 0] == 4)
            if sugarcane_check:
                group_mean_cap[group_idx] = (
                    np.mean(year_income_m2[agent_indices]) * 0.05
                )
            else:
                group_mean_cap[group_idx] = (
                    np.mean(year_income_m2[agent_indices]) * 0.02
                )

        agent_caps = group_mean_cap[group_indices] * self.field_size_per_farmer

        return agent_caps

    def potential_insured_loss(self) -> np.ndarray:
        """Compute potential insured loss per farmer-year.

        Masks unfilled years (all-zero income), computes each farmer's average
        income over filled years, and sets the potential insured loss as the
        positive difference between that average and the realized income.

        Returns:
            np.ndarray: Array shaped like ``yearly_income`` with per farmer-year
                potential insured losses (``float32``). Masked years remain zero.
        """
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

        return potential_insured_loss

    def premium_personal_insurance(
        self,
        potential_insured_loss: npt.NDArray[np.floating],
        government_premium_cap: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute capped personal insurance premiums via Bühlmann–Straub credibility.

        Uses each farmer's pure premium (mean potential loss per m²), blends it with
        their group's mean using credibility weights ``Z = n / (n + K)``, and then
        caps the resulting premium by the government cap.

        Args:
            potential_insured_loss: Potential insured loss per
                farmer-year; shape matches ``yearly_income``.
            government_premium_cap: Maximum allowed premium per
                farmer (currency units).

        Returns:
            npt.NDArray[np.floating]: Capped personal premium per farmer.
        """
        # Calculating personal pure premiums and Bühlmann-Straub parameters to get the credibility premium
        # Mask out unfilled years
        mask_columns = np.all(self.var.yearly_income == 0, axis=0)

        # Apply the mask to data
        income_masked = self.var.yearly_income.data[:, ~mask_columns]
        # Calculate personal loss
        agent_pure_premiums_m2 = (
            np.mean(potential_insured_loss, axis=1) / self.field_size_per_farmer
        )

        group_indices, n_groups = self.create_unique_groups(
            self.well_status,
        )

        years_observed = np.sum(~np.isnan(income_masked), axis=1)
        # Initialize arrays for coefficients and R²
        group_mean_premiums = np.zeros(n_groups, dtype=float)
        for group_idx in range(n_groups):
            agent_indices = np.where(group_indices == group_idx)[0]
            group_mean_premiums[group_idx] = np.mean(
                agent_pure_premiums_m2[agent_indices]
            )

        sample_var_per_agent = np.var(potential_insured_loss, axis=1, ddof=1)
        valid_for_within = years_observed > 1

        within_variance = np.sum(
            (years_observed[valid_for_within] - 1)
            * sample_var_per_agent[valid_for_within]
        ) / np.sum(years_observed[valid_for_within] - 1)
        between_variance = np.var(agent_pure_premiums_m2, ddof=1)
        credibility_param_K = (
            within_variance / between_variance if between_variance > 0 else np.inf
        )

        # Classical Bühlmann–Straub: Z = n / (n + K)
        credibility_weights = years_observed / (years_observed + credibility_param_K)
        credibility_premiums_m2 = (
            credibility_weights * agent_pure_premiums_m2
            + (1 - credibility_weights) * group_mean_premiums[group_indices]
        )
        # Return to personal prices and add loading factor
        personal_premium = credibility_premiums_m2 * self.field_size_per_farmer * 1.3

        return np.minimum(government_premium_cap, personal_premium)

    def insured_payouts_personal(
        self,
        insured_farmers_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating]:
        """Compute insured payouts for personal insurance and update state.

        Uses a trailing mean income threshold per farmer-year (first 7 years use the
        cumulative mean; afterwards a 7-year moving average), compares it to current
        income, and pays the positive shortfall for insured farmers. Updates
        ``insured_yearly_income`` for the current year and records payout events in
        ``payout_mask``.

        Args:
            insured_farmers_mask: Boolean mask indicating
                which farmers are covered by personal insurance.

        Returns:
            npt.NDArray[np.floating]: Per farmer-year insured losses (same shape as
            ``yearly_income``).
        """
        data_full = self.var.yearly_income.data

        mask_cols = (data_full == 0).all(axis=0)
        data_masked = data_full[:, ~mask_cols]

        cumsum = np.cumsum(data_masked, axis=1, dtype=float)

        n_agents, T = data_masked.shape
        years = np.arange(T)
        thr_m = np.empty_like(data_masked, dtype=float)

        first7 = years < 7
        thr_m[:, first7] = cumsum[:, first7] / (years[first7] + 1)

        if T > 7:
            window_sum = cumsum[:, 7:] - cumsum[:, :-7]
            thr_m[:, 7:] = window_sum / 7

        thr_m = thr_m[:, ::-1]

        threshold_full = np.zeros_like(data_full, dtype=float)
        threshold_full[:, ~mask_cols] = thr_m

        insured_losses = np.maximum(threshold_full - self.var.yearly_income, 0)

        self.var.insured_yearly_income[insured_farmers_mask, 0] += insured_losses[
            insured_farmers_mask, 0
        ]

        # Improve intention factor of farmers who have had a payout
        new_payouts = (insured_losses[:, 0] > 0) & insured_farmers_mask
        self.var.payout_mask[:, PERSONAL_INSURANCE_ADAPTATION] |= new_payouts

        return insured_losses

    def premium_index_insurance(
        self,
        potential_insured_loss: npt.NDArray[np.floating],
        history: npt.NDArray[np.floating],
        gev_params: npt.NDArray[np.floating],
        strike_vals: npt.NDArray[np.floating],
        exit_vals: npt.NDArray[np.floating],
        rate_vals: npt.NDArray[np.floating],
        government_premium_cap: npt.NDArray[np.floating],
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        """Select an index-insurance contract and compute capped premiums.

        Builds candidate contracts (strike/exit/rate), evaluates basis risk using
        past losses, and selects the best contract per farmer. Premiums include a
        loading factor and are capped by the government premium cap.

        Args:
            potential_insured_loss: Potential insured loss per
                farmer-year; shape matches ``yearly_income``.
            history: Historical index (e.g., rainfall) per
                farmer-year aligned with ``potential_insured_loss``.
            gev_params: Fitted GEV parameters per farmer
                (shape as required by the pricing routine).
            strike_vals: Candidate strike levels.
            exit_vals: Candidate exit levels.
            rate_vals: Candidate rate-on-line values.
            government_premium_cap: Max premium per farmer.

        Returns:
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating],
            npt.NDArray[np.floating], npt.NDArray[np.floating]]: Best strike, exit,
                rate, and capped premium per farmer.
        """
        # Make a series of candidate insurance contracts and find the optimal contract
        # with the least basis risk considering past losses
        mask_columns = np.all(self.var.yearly_income == 0, axis=0)

        potential_insured_loss_masked = potential_insured_loss[:, ~mask_columns]
        history_masked = history[:, ~mask_columns]

        (
            best_strike_idx,
            best_exit_idx,
            best_rate_idx,
            best_rmse,
            best_prem,
        ) = compute_premiums_and_best_contracts_numba(
            gev_params,
            history_masked,
            potential_insured_loss_masked,
            strike_vals,
            exit_vals,
            rate_vals,
            n_sims=100,
            seed=42,
        )

        best_strike = strike_vals[best_strike_idx]
        best_exit = exit_vals[best_exit_idx]
        best_rate = rate_vals[best_rate_idx]
        best_premiums = best_prem * 1.3  # add loading factor

        return (
            best_strike,
            best_exit,
            best_rate,
            np.minimum(best_premiums, government_premium_cap),
        )

    def insured_payouts_index(
        self,
        strike: npt.NDArray[np.floating],
        exit: npt.NDArray[np.floating],
        rate: npt.NDArray[np.floating],
        insured_farmers_mask: npt.NDArray[np.bool_],
        index_nr: int,
    ) -> npt.NDArray[np.floating]:
        """Compute index-insurance payouts historically and update state.

        Uses strike/exit thresholds and per-farmer rates to derive payouts from the
        historical SPEI index, updates ``insured_yearly_income`` for insured farmers,
        and records payout events in ``payout_mask`` at ``index_nr``.

        Args:
            strike: Strike level per farmer.
            exit: Exit level per farmer (≤ strike).
            rate: Rate-on-line per farmer.
            insured_farmers_mask: Boolean mask of insured farmers.
            index_nr: Column in ``payout_mask`` corresponding to this product.

        Returns:
            npt.NDArray[np.floating]: Per farmer-year payouts shaped like
            ``yearly_income`` (masked years are zero).
        """
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

        self.var.insured_yearly_income[insured_farmers_mask, 0] += (
            potential_insured_loss[insured_farmers_mask, 0]
        )

        # Improve intention factor of farmers who have had a payout
        self.var.payout_mask[:, index_nr] |= (
            potential_insured_loss[:, 0] > 0
        ) & insured_farmers_mask

        return potential_insured_loss

    def insured_yields(
        self,
        potential_insured_loss: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute insured yield-SPEI relation given the agent had insurance.

        Adds the potential insured loss to yearly income, converts to a yield ratio
        relative to potential income (clipped to ``[0, 1]``), and derives the
        yield-SPEI relationship using the groupwise linear relation.

        Args:
            potential_insured_loss: Potential insured loss
                per farmer-year; shape compatible with ``yearly_income``.

        Returns:
            npt.NDArray[np.floating]: Insured yield-SPEI relationship per farmer-year.
        """
        insured_yearly_income = self.var.yearly_income + potential_insured_loss

        insured_yearly_yield_ratio = (
            insured_yearly_income / self.var.yearly_potential_income
        )

        insured_yearly_yield_ratio = np.clip(insured_yearly_yield_ratio.data, 0, 1)

        insured_yield_probability_relation = (
            self.calculate_yield_spei_relation_group_lin(
                insured_yearly_yield_ratio, self.var.yearly_SPEI_probability
            )
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
        """Assign microcredit annual costs into available loan slots (Numba).

        Args:
            all_loans_annual_cost: Array of per-farmer annual loan costs
                with shape ``(n_farmers, n_types, n_slots)``; updated in place.
            loan_tracker: Remaining duration per loan slot; updated in place.
            loaning_farmers: Boolean mask of farmers receiving a loan.
            annual_cost_loan: Annual cost for new loan, to be added to
                all_loans_annual_cost
            loan_duration: Duration (years) to set for new loans.
            loan_type: Loan type index (e.g., 0=..., 1=microcredit).
        """
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
            self.remove_agents(farmers_selling_land, GRASSLAND_LIKE)

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

    def save_harvest_spei(self, harvesting_farmers: npt.NDArray[np.bool_]) -> None:
        """Update monthly SPEI by shifting history and adding the current month.

        Updates ``monthly_SPEI``-related state in place using the current SPEI
        sampled at harvesting farmers' locations.

        Args:
            harvesting_farmers: Boolean mask of farmers
                who are harvesting this step.
        """
        spei = self.model.hydrology.grid.spei_uncompressed
        current_SPEI_per_farmer = sample_from_map(
            array=spei,
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

    def save_harvest_precipitation(
        self,
        harvesting_farmers: npt.NDArray[np.bool_],
        crop_age: npt.NDArray[np.floating],
    ) -> None:
        """Accumulate seasonal precipitation for harvesting farmers.

        Uses the average crop age to define a seasonal window ending today and sums
        precipitation over that window per farmer, adding it to the cumulative total.

        Args:
            harvesting_farmers: Boolean mask of farmers who are harvesting.
            crop_age: Current crop age per farmer (days).
        """
        avg_age = np.mean(crop_age, dtype=np.int32)
        end_day = self.model.current_day_of_year - 1
        start_day = end_day - avg_age

        n_days = self.var.cumulative_pr_mm.shape[1]
        day_idx = np.arange(start_day, end_day) % n_days

        season_pr_per_farmer = np.sum(
            self.var.cumulative_pr_mm[np.ix_(harvesting_farmers, day_idx)], axis=1
        )

        self.var.cumulative_pr_during_growing_season[harvesting_farmers] += (
            season_pr_per_farmer
        )

    def save_yearly_spei(self) -> None:
        """Finalize and save yearly SPEI and its exceedance probability.

        Computes the annual SPEI probability via the generalized extreme value (GEV)
        distribution, shifts historical arrays, and resets cumulative seasonal SPEI.
        """
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

    def save_yearly_pr(self) -> None:
        """Save and reset yearly precipitation totals per farmer."""
        assert self.model.current_time.month == 1

        shift_and_update(
            self.var.yearly_pr, self.var.cumulative_pr_during_growing_season
        )
        self.var.cumulative_pr_during_growing_season.fill(0)

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

    def calculate_yield_spei_relation_group_exp(
        self,
        yearly_yield_ratio: npt.NDArray[np.floating],
        yearly_SPEI_probability: npt.NDArray[np.floating],
        drop_k: int = 2,
    ) -> npt.NDArray[np.floating]:
        """Fit grouped exponential yield-SPEI model and return per-farmer parameters.

        Model form per group: ``y = a * exp(b * X)`` (fit in log-space). For each
        group, up to ``drop_k`` points with the largest absolute residuals (in
        log-space) are removed and the model is refit; parameters ``a`` and ``b``
        are then mapped back to each farmer in that group.

        Args:
            yearly_yield_ratio: Yearly yield ratio per
                farmer-year (0-1).
            yearly_SPEI_probability: Yearly SPEI exceedance
                probabilities per farmer-year.
            drop_k: Number of worst absolute residuals to drop per
                group before refitting. Defaults to ``2``.

        Returns:
            npt.NDArray[np.floating]: Per-farmer parameters with shape ``(n_farmers, 2)``,
                columns ``[a, b]`` for ``y = a * exp(b * X)``.
        """
        # Create groups (unchanged)
        group_indices, n_groups = self.create_unique_groups(self.well_status)
        assert (np.any(self.var.yearly_SPEI_probability != 0, axis=1) > 0).all()

        masked_yearly_yield_ratio = yearly_yield_ratio
        masked_SPEI_probability = yearly_SPEI_probability

        a_array = np.zeros(n_groups)
        b_array = np.zeros(n_groups)
        r_squared_array = np.zeros(n_groups)

        for g in range(n_groups):
            agent_idx = np.where(group_indices == g)[0]

            y_data = masked_yearly_yield_ratio[agent_idx, :].copy()
            X_data = masked_SPEI_probability[agent_idx, :].copy()

            # Same validity rules as your original (ln requires y>0 anyway)
            mask_bad = (X_data >= 1) | (y_data <= 0)
            y_data[mask_bad] = np.nan
            X_data[mask_bad] = np.nan

            y_group = np.nanmean(y_data, axis=0)
            X_group = np.nanmean(X_data, axis=0)

            valid = (~np.isnan(y_group)) & (~np.isnan(X_group)) & (y_group > 0)
            Xv = X_group[valid]
            yv = y_group[valid]

            if len(Xv) >= 2:
                # First fit in log-space
                ln_y = np.log(yv)
                b, ln_a = _fit_linear(Xv, ln_y)  # returns (slope, intercept)
                a = np.exp(ln_a)

                # Optionally drop worst k residuals (log-space) and refit
                if drop_k and len(Xv) > drop_k + 1:
                    ln_y_hat = b * Xv + ln_a
                    resid = ln_y - ln_y_hat
                    worst_idx = np.argsort(np.abs(resid))[-drop_k:]
                    keep = np.ones(len(Xv), dtype=bool)
                    keep[worst_idx] = False

                    Xv2 = Xv[keep]
                    ln_y2 = ln_y[keep]
                    # Refit
                    b, ln_a = _fit_linear(Xv2, ln_y2)
                    a = np.exp(ln_a)

                    # R² in y-space on the kept points
                    y_pred2 = a * np.exp(b * Xv2)
                    r2 = _r2(np.exp(ln_y2), y_pred2)
                else:
                    # R² in y-space on all valid points
                    y_pred = a * np.exp(b * Xv)
                    r2 = _r2(yv, y_pred)
            else:
                a, b, r2 = np.nan, np.nan, np.nan

            a_array[g] = a
            b_array[g] = b
            r_squared_array[g] = r2

        # Assign per farmer (cols: intercept=a, slope=b)
        farmer_params = np.column_stack(
            (a_array[group_indices], b_array[group_indices])
        )

        # Print median R²
        r2_mapped = r_squared_array[group_indices]
        r2_valid = r2_mapped[~np.isnan(r2_mapped)]
        print("Median R² (exp):", np.median(r2_valid) if len(r2_valid) else "N/A")
        if np.median(r2_valid) < 0.2:
            pass

        return farmer_params

    def calculate_yield_spei_relation_group_lin(
        self,
        yearly_yield_ratio: npt.NDArray[np.floating],
        yearly_SPEI_probability: npt.NDArray[np.floating],
        drop_k: int = 2,
    ) -> npt.NDArray[np.floating]:
        """Fit grouped linear yield-SPEI model and return per-farmer parameters.

        Linear model per group: ``y = m * X + c``. Uses the same invalid-data mask
        as the exponential version for comparability. For each group, drops up to
        ``drop_k`` points with the largest absolute residuals (in y-space), refits,
        and maps parameters back to farmers.

        Args:
            yearly_yield_ratio: Yearly yield ratio per
                farmer-year (0-1).
            yearly_SPEI_probability: Yearly SPEI
                exceedance probabilities per farmer-year.
            drop_k: Number of worst absolute residuals to drop per
                group before refitting. Defaults to ``2``.

        Returns:
            npt.NDArray[np.floating]: Per-farmer parameters with shape
                ``(n_farmers, 2)``, columns ``[c, m]`` for ``y = m * X + c``.
        """
        group_indices, n_groups = self.create_unique_groups(self.well_status)
        assert (np.any(self.var.yearly_SPEI_probability != 0, axis=1) > 0).all()

        y_all = yearly_yield_ratio
        X_all = yearly_SPEI_probability

        c_array = np.zeros(n_groups)  # intercept
        m_array = np.zeros(n_groups)  # slope
        r_squared_array = np.zeros(n_groups)

        for g in range(n_groups):
            agent_idx = np.where(group_indices == g)[0]

            y_data = y_all[agent_idx, :].copy()
            X_data = X_all[agent_idx, :].copy()

            # Keep the same mask as before (you can relax this for linear if desired)
            mask_bad = (X_data >= 1) | (y_data <= 0)
            y_data[mask_bad] = np.nan
            X_data[mask_bad] = np.nan

            y_group = np.nanmean(y_data, axis=0)
            X_group = np.nanmean(X_data, axis=0)

            valid = (~np.isnan(y_group)) & (~np.isnan(X_group))
            Xv = X_group[valid]
            yv = y_group[valid]

            if len(Xv) >= 2:
                m, c = _fit_linear(Xv, yv)

                if drop_k and len(Xv) > drop_k + 1:
                    yhat = m * Xv + c
                    resid = yv - yhat
                    worst_idx = np.argsort(np.abs(resid))[-drop_k:]
                    keep = np.ones(len(Xv), dtype=bool)
                    keep[worst_idx] = False

                    Xv2 = Xv[keep]
                    yv2 = yv[keep]
                    m, c = _fit_linear(Xv2, yv2)

                    yhat2 = m * Xv2 + c
                    r2 = _r2(yv2, yhat2)
                else:
                    yhat = m * Xv + c
                    r2 = _r2(yv, yhat)
            else:
                m, c, r2 = np.nan, np.nan, np.nan

            c_array[g] = c
            m_array[g] = m
            r_squared_array[g] = r2

        # Assign per farmer (cols: intercept=c, slope=m)
        farmer_params = np.column_stack(
            (c_array[group_indices], m_array[group_indices])
        )

        r2_mapped = r_squared_array[group_indices]
        r2_valid = r2_mapped[~np.isnan(r2_mapped)]
        print("Median R² (lin):", np.median(r2_valid) if len(r2_valid) else "N/A")

        return farmer_params

    def adapt_crops(
        self,
        farmer_yield_probability_relation: npt.NDArray[np.floating],
    ) -> None:
        """Switches crop rotation based on SEUT and yield-SPEI relations.

        Computes expected utilities (SEUT) for the current crop plan versus all
        alternative crop-calendar options, accounts for cultivation-cost differences
        and loan costs, and switches a capped subset of farmers to the best
        alternative where beneficial.

        Args:
            farmer_yield_probability_relation: Per-farmer
                yield-SPEI relationship used to evaluate profits under drought risk.
        """
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
        ]  # Costs per m2
        cultivation_costs_current_rotation = np.bincount(
            rows, weights=costs, minlength=current_crop_calendar.shape[0]
        ).astype(np.float32)

        annual_cost_empty = np.zeros(self.var.n, dtype=np.float32)

        # No constraint
        extra_constraint = np.ones_like(annual_cost_empty, dtype=bool)
        within_budget = np.ones_like(annual_cost_empty, dtype=bool)

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
        timer_crops.finish_split("profit_difference")
        total_annual_costs_m2 = (
            self.var.all_loans_annual_cost[:, -1, 0] / self.field_size_per_farmer
        )

        # Construct a dictionary of parameters to pass to the decision module functions
        decision_params: dict[str, Any] = {
            "loan_duration": loan_duration,
            "expenditure_cap": within_budget,
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
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing_drought(
            **decision_params
        )

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
            SEUT_crop_options[:, idx] = self.decision_module.calcEU_adapt_drought(
                **decision_params_option
            )

        assert np.any(SEUT_do_nothing != -1) or np.any(SEUT_crop_options != -1)
        timer_crops.finish_split("SEUT")
        # Determine the best adaptation option
        best_option_SEUT = np.max(SEUT_crop_options, axis=1)
        chosen_option = np.argmax(SEUT_crop_options, axis=1)

        # Determine the crop of the best option
        row_indices = np.arange(new_farmer_id.shape[0])
        new_id_temp = new_farmer_id[row_indices, chosen_option]

        # adjusted_do_nothing = SEUT_do_nothing * (factor ** np.sign(SEUT_do_nothing))
        # Determine for which agents it is beneficial to switch crops
        initial_mask = (
            (best_option_SEUT > (SEUT_do_nothing)) & (new_id_temp != -1)
        )  # Filter out crops chosen due to small diff in do_nothing and adapt SEUT calculation

        cap = int(np.floor(0.07 * best_option_SEUT.size))
        n_true = int(initial_mask.sum())
        if n_true <= cap:
            SEUT_adaptation_decision = initial_mask
        else:
            # compute percent difference among eligible items
            # percent diff = (best - adjusted) / |adjusted|
            # protect against division by zero
            denom = np.maximum(np.abs(SEUT_do_nothing), 1e-12)
            pct_diff = (best_option_SEUT - SEUT_do_nothing) / denom

            # consider only indices where initial_mask is True
            idx_true = np.flatnonzero(initial_mask)
            scores = pct_diff[idx_true]  # positive by construction

            # pick top 'cap' by score (no full sort needed)
            top_k_rel = np.argpartition(scores, -cap)[-cap:]
            keep_idx = idx_true[top_k_rel]

            # build the final capped mask
            SEUT_adaptation_decision = np.zeros_like(initial_mask, dtype=bool)
            SEUT_adaptation_decision[keep_idx] = True

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
        timer_crops.finish_split("final steps")
        print(timer_crops)

    def adapt_irrigation_well(
        self,
        farmer_yield_probability_relation: npt.NDArray[np.floating],
        average_extraction_speed: npt.NDArray[np.floating] | float,
        energy_cost: npt.NDArray[np.floating],
        water_cost: npt.NDArray[np.floating],
    ) -> None:
        """Checks farmers will take irrigation wells based on expected utility and constraints.

        Checks which farmers adopt/renew wells by comparing SEUT of adapting vs.
        doing nothing, accounting for well costs, energy/water cost differences,
        loan terms, and feasibility (well must reach groundwater). Updates state
        (adaptations, well depth, etc.) in place.

        Args:
            farmer_yield_probability_relation: Per-farmer
                yield-SPEI relationship used to evaluate profits under drought risk.
            average_extraction_speed: Average pump
                extraction speed (can be scalar or per-farmer array).
            energy_cost: Energy cost per farmer (currency/yr).
            water_cost: Water cost per farmer (currency/yr).
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

        # Define extra constraints (farmers' wells must reach groundwater)
        well_reaches_groundwater = self.var.well_depth > groundwater_depth
        extra_constraint = well_reaches_groundwater

        # Reset farmers' status and irrigation type who exceeded the lifespan of their adaptation
        # and who's wells are much shallower than the groundwater depth
        self.reset_adaptation_status(
            farmer_yield_probability_relation=farmer_yield_probability_relation,
            adapted=adapted,
            additional_diffentiator_expiration=extra_constraint,
            additional_diffentiator_grouping=self.blank_additional_differentiator,
            adaptation_type=WELL_ADAPTATION,
        )

        energy_cost_m2 = energy_cost / self.field_size_per_farmer
        water_cost_m2 = water_cost / self.field_size_per_farmer

        (
            energy_diff_m2,
            water_diff_m2,
        ) = self.adaptation_water_cost_difference(
            self.blank_additional_differentiator, adapted, energy_cost_m2, water_cost_m2
        )

        (
            total_profits,
            profits_no_event,
            total_profits_adaptation,
            profits_no_event_adaptation,
            ids_to_switch_to,
        ) = self.profits_SEUT(
            self.blank_additional_differentiator,
            adapted,
            farmer_yield_probability_relation,
        )

        within_budget = self.budget_check(total_annual_costs_m2)

        total_profits_adaptation = (
            total_profits_adaptation + energy_diff_m2 + water_diff_m2
        )
        profits_no_event_adaptation = (
            profits_no_event_adaptation + energy_diff_m2 + water_diff_m2
        )

        # Construct a dictionary of parameters to pass to the decision module functions
        decision_params: dict[str, Any] = {
            "loan_duration": loan_duration,
            "expenditure_cap": within_budget,
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
            "adapted": adapted.data,
            "time_adapted": self.var.time_adapted[:, WELL_ADAPTATION].data,
            "T": np.full(
                self.var.n,
                self.model.config["agent_settings"]["farmers"]["expected_utility"][
                    "adaptation_well"
                ]["decision_horizon"],
            ),
            "discount_rate": self.var.discount_rate.data,
            "extra_constraint": extra_constraint.data,
        }

        # Calculate the EU of not adapting and adapting respectively
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing_drought(
            **decision_params
        )
        SEUT_adapt = self.decision_module.calcEU_adapt_drought(**decision_params)

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
        self,
        farmer_yield_probability_relation: npt.NDArray[np.floating[Any]],
        energy_cost: npt.NDArray[np.floating[Any]],
        water_cost: npt.NDArray[np.floating[Any]],
        adaptation_costs_m2: npt.NDArray[np.floating[Any]],
        adaptation_type: int,
        efficiency: float,
        return_fraction: float,
    ) -> None:
        """Handle the adaptation of farmers to more efficient irrigation.

        Evaluates whether farmers adopt higher-efficiency irrigation systems by
        comparing expected utilities (SEUT) of adapting versus doing nothing, while
        accounting for loan costs, energy/water cost differences, and access
        constraints. Updates adaptation state, irrigation efficiency, and return
        fraction in place.

        Args:
            farmer_yield_probability_relation: Per-farmer parameters for the
                yield-SPEI relation used to evaluate profits under drought risk.
                Shape (n_agents, 2).
            energy_cost: Annual energy cost per farmer (LCU/year). Shape (n_agents,).
            water_cost: Annual water cost per farmer (LCU/year). Shape (n_agents,).
            adaptation_costs_m2: Capital cost per m² for the efficiency upgrade
                (LCU/m²). Shape (n_agents,).
            adaptation_type: Index of the adaptation slot/column to update.
            efficiency: New irrigation efficiency to assign to adopters (0-1).
            return_fraction: Fraction of non-consumed irrigation water returning
                as runoff (0-1).

        """
        loan_duration = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_sprinkler"]["loan_duration"]

        costs_irrigation_system = adaptation_costs_m2 * self.field_size_per_farmer

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

        # To determine the benefit of irrigation, those who have above 90% irrigation efficiency have adapted
        adapted = self.var.adaptations[:, adaptation_type] > 0

        # Reset farmers' status and irrigation type who exceeded the lifespan of their adaptation
        # or who's never had access to irrigation water
        self.reset_adaptation_status(
            farmer_yield_probability_relation=farmer_yield_probability_relation,
            adapted=adapted,
            additional_diffentiator_expiration=has_irrigation_access,
            additional_diffentiator_grouping=self.blank_additional_differentiator,
            adaptation_type=adaptation_type,
        )

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

        within_budget = self.budget_check(total_annual_costs_m2)

        total_profits_adaptation = (
            total_profits_adaptation + energy_diff_m2 + water_diff_m2
        )
        profits_no_event_adaptation = (
            profits_no_event_adaptation + energy_diff_m2 + water_diff_m2
        )

        # Construct a dictionary of parameters to pass to the decision module functions
        decision_params: dict[str, Any] = {
            "loan_duration": loan_duration,
            "expenditure_cap": within_budget,
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
            "adapted": adapted.data,
            "time_adapted": self.var.time_adapted[:, adaptation_type].data,
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
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing_drought(
            **decision_params
        )
        SEUT_adapt = self.decision_module.calcEU_adapt_drought(**decision_params)

        assert (SEUT_do_nothing != -1).any() or (SEUT_adapt != -1).any()

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
        self.var.irrigation_efficiency[SEUT_adaptation_decision] = efficiency
        self.var.return_fraction[SEUT_adaptation_decision] = return_fraction

        # Print the percentage of adapted households
        percentage_adapted = round(
            np.sum(self.var.adaptations[:, adaptation_type] > 0)
            / len(self.var.adaptations[:, adaptation_type])
            * 100,
            2,
        )
        print("Irrigation efficient farms:", percentage_adapted, "(%)")

    def adapt_irrigation_expansion(
        self,
        farmer_yield_probability_relation: npt.NDArray[np.floating],
        energy_cost: npt.NDArray[np.floating],
        water_cost: npt.NDArray[np.floating],
    ) -> None:
        """Evaluate and execute expansion of irrigated area based on SEUT.

        Compares expected utilities of expanding irrigation versus doing nothing,
        including loan amortization and added energy/water costs, subject to access
        constraints. Updates adaptation state and fraction of irrigated field.

        Args:
            farmer_yield_probability_relation: Per-farmer
                yield-SPEI relationship used for profit evaluation.
            energy_cost: Annual energy cost per farmer.
            water_cost: Annual water cost per farmer.
        """
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
        decision_params: dict[str, Any] = {
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
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing_drought(
            **decision_params
        )
        SEUT_adapt = self.decision_module.calcEU_adapt_drought(**decision_params)

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
        adaptation_types: Sequence[int] | npt.NDArray[np.integer],
        adaptation_names: Sequence[str],
        farmer_yield_probability_relation_base: npt.NDArray[np.floating],
        farmer_yield_probability_relations_insured: Sequence[npt.NDArray[np.floating]],
        premiums: list[DynamicArray],
    ) -> None:
        """Evaluate and adopt insurance options using expected utility (SEUT).

        Computes expected utilities for one or more insurance adaptations, using
        base and insured yield-SPEI relations, and adopts the highest-SEUT option
        subject to constraints. Updates adaptation state in place.

        Args:
            adaptation_types: Numeric codes of insurance adaptations
                to evaluate (e.g., personal, index, etc.).
            adaptation_names: Human-readable names aligned with
                ``adaptation_types`` for logging.
            farmer_yield_probability_relation_base:
                Base (uninsured) yield-SPEI relation per farmer.
            farmer_yield_probability_relations_insured:
                Per-option insured yield-SPEI relations; same order as
                ``adaptation_types``.
            premiums: Annual premium per farmer for each
                option; shape should broadcast with the per-option loop.
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

        # Determine the additional spending room insurance brings
        # Is done separately as index insurance does affect income,
        # but does not affect adaptation decisions
        yield_ratios_exp_cap = self.convert_probability_to_yield_ratio(
            self.farmer_yield_probability_relation_exp_cap
        )
        total_profits_exp_cap = self.compute_total_profits(yield_ratios_exp_cap)
        _, profits_no_event_exp_cap = self.format_results(total_profits_exp_cap)

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

        if len(adaptation_types) > 1:
            test_array = np.zeros_like(interest_rate, dtype=np.int8)

            # Define extra constraints -- cant adapt another insurance type while having one before
            for t in adaptation_types:
                test_array += (self.var.adaptations[:, t] >= 0).astype(np.int8)

            extra_constraint = test_array < 1
            adapted = np.zeros_like(interest_rate, dtype=bool)
        else:
            extra_constraint = np.ones_like(interest_rate, dtype=bool)
            adapted = (self.var.adaptations[:, adaptation_type] > 0).data

        for idx, adaptation_type in enumerate(adaptation_types):
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
            within_budget = (
                profits_no_event_exp_cap * self.var.expenditure_cap > annual_cost_m2
            )

            # Determine the would be income with insurance
            insured_yield_ratios = self.convert_probability_to_yield_ratio(
                farmer_yield_probability_relation_insured, model="linear"
            )
            total_profits_insured = self.compute_total_profits(insured_yield_ratios)
            total_profits_index_insured, profits_no_event_index_insured = (
                self.format_results(total_profits_insured)
            )

            # Construct a dictionary of parameters to pass to the decision module functions
            decision_params: dict[str, Any] = {
                "loan_duration": loan_duration,
                "expenditure_cap": within_budget,
                "n_agents": self.var.n,
                "sigma": self.var.risk_aversion.data,
                "p_droughts": 1 / self.var.p_droughts[:-1],
                "total_profits_adaptation": total_profits_index_insured,
                "profits_no_event": profits_no_event,
                "profits_no_event_adaptation": profits_no_event_index_insured,
                "total_profits": total_profits,
                "risk_perception": self.var.risk_perception.data,
                "total_annual_costs": total_annual_costs_m2.data,
                "adaptation_costs": annual_cost_m2.data,
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

            SEUT_insurance_options[:, idx] = self.decision_module.calcEU_adapt_drought(
                **decision_params
            )

        # Calculate the EU of not adapting and adapting respectively
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing_drought(
            **decision_params
        )

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

        # Check whether a single agent does not have multiple insurances
        assert not np.any(
            (self.var.adaptations.data[:, adaptation_types] == 1).sum(1) > 1
        )

    def update_adaptation_decision(
        self,
        adaptation_type: int,
        adapted: npt.NDArray[np.bool_],
        loan_duration: int,
        annual_cost: npt.NDArray[np.floating],
        SEUT_do_nothing: npt.NDArray[np.floating],
        SEUT_adapt: npt.NDArray[np.floating],
        ids_to_switch_to: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.bool_]:
        """Update adaptation status based on SEUT and return the decision mask.

        Compares expected utility (SEUT) of adapting versus doing nothing, adjusted
        by a configuration-dependent factor, and updates internal state for agents
        who adapt (timers, loan costs, yield-SPEI relationships).

        Args:
            adaptation_type: Adaptation code (e.g., well, insurance variant).
            adapted: Current boolean mask of adapted agents.
            loan_duration: Loan duration in years for the adaptation.
            annual_cost: Annualized adaptation cost per agent.
            SEUT_do_nothing: SEUT values for not adapting.
            SEUT_adapt: SEUT values for adapting.
            ids_to_switch_to: Mapping of agents to donor
                indices for updating historical series (yield/income/SPEI).

        Returns:
            Boolean mask indicating which agents adapt this step.
        """
        if adaptation_type in (
            PERSONAL_INSURANCE_ADAPTATION,
            INDEX_INSURANCE_ADAPTATION,
            PR_INSURANCE_ADAPTATION,
        ):
            factor = self.model.config["agent_settings"]["farmers"]["expected_utility"][
                "insurance"
            ]["seut_factor"]
        else:
            factor = self.model.config["agent_settings"]["farmers"]["expected_utility"][
                "adaptation_well"
            ]["seut_factor"]
        # Compare EU values for those who haven't adapted yet and get boolean results

        adjusted_do_nothing = SEUT_do_nothing * (factor ** np.sign(SEUT_do_nothing))
        SEUT_adaptation_decision = SEUT_adapt > adjusted_do_nothing

        # social_network_adaptation = adapted[self.var.social_network]

        # # Check whether adapting agents have adaptation type in their network and create mask
        # network_has_adaptation = np.any(social_network_adaptation == 1, axis=1)

        # # Increase intention factor if someone in network has adaptation
        # intention_factor_adjusted = self.var.intention_factor.copy()
        # intention_factor_adjusted = np.clip(
        #     self.var.intention_factor.data - 0.3, 0.05, 0.2
        # )

        # if adaptation_type in (
        #     PERSONAL_INSURANCE_ADAPTATION,
        #     INDEX_INSURANCE_ADAPTATION,
        #     PR_INSURANCE_ADAPTATION,
        # ):
        #     social_network_payout = self.var.payout_mask[
        #         self.var.social_network, adaptation_type
        #     ]
        #     network_has_payout = np.any(social_network_payout == 1, axis=1)
        #     intention_factor_adjusted[network_has_payout] += 0.4

        #     agent_has_payout = self.var.payout_mask[:, adaptation_type]
        #     intention_factor_adjusted[agent_has_payout] += 0.4
        # else:
        #     intention_factor_adjusted[network_has_adaptation] += 0.4

        # # Determine whether it passed the intention threshold
        # random_values = np.random.rand(*intention_factor_adjusted.shape)
        # intention_mask = random_values < intention_factor_adjusted

        # SEUT_adaptation_decision = SEUT_adaptation_decision  # & intention_mask

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

    def calculate_water_costs(
        self,
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        """Calculate water/energy costs and average extraction speed per agent.

        Computes:
        - energy costs for groundwater users (USD/year),
        - water costs for all agents by source (USD/year),
        - average extraction speed per agent (m³/s),

        and updates loan-related arrays in place.

        Returns:
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
                Energy costs, water costs, and average extraction speed per agent.
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

        # Get energy cost rate per agent (USD per kWh)
        energy_cost_rate = electricity_costs[mask_groundwater]

        # Compute energy costs per agent (USD/year) for groundwater irrigating farmers
        energy_costs[mask_groundwater] = energy * energy_cost_rate

        # Compute water costs for agents using channel water (USD/year)
        water_costs[mask_channel] = (
            average_extraction[mask_channel] * self.var.water_costs_m3_channel
        )

        # Compute water costs for agents using reservoir water (USD/year)
        water_costs[mask_reservoir] = (
            average_extraction[mask_reservoir] * self.var.water_costs_m3_reservoir
        )

        # Compute water costs for agents using groundwater (USD/year)
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
        self,
        groundwater_depth: npt.NDArray[np.floating],
        average_extraction_speed: npt.NDArray[np.floating] | float,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Compute annual well-installation/operation costs and potential well length.

        Args:
            groundwater_depth: Groundwater depth per agent (m).
            average_extraction_speed: Average water extraction
                speed per agent (m³/s).

        Returns:
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
                annual_cost: Annual cost per agent (LCU/year).
                potential_well_length: Potential well length per agent (m).
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
        well_unit_cost = np.full_like(self.var.why_class.data, np.nan, dtype=np.float32)

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

        assert not np.isnan(well_unit_cost).any(), (
            "Some agents have undefined well unit costs."
        )

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
        self,
        additional_diffentiators: npt.NDArray,
        adapted: npt.NDArray[np.bool_],
        farmer_yield_probability_relation: npt.NDArray[np.floating],
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.integer],
    ]:
        """Calculate profits with/without adaptation under drought scenarios.

        First converts probabilities to yield per probability using the yield/probability relation
        Then turns yield into profit, and adds the adaptation yield difference to those probabilities

        Args:
            additional_diffentiators: Extra differentiators for grouping agents.
            adapted: Mask indicating which agents are adapted.
            farmer_yield_probability_relation: Per-farmer yield–SPEI relation.

        Returns:
            tuple[
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
                npt.NDArray[np.integer],
            ]:
                - total_profits: Total profits for each drought scenario (no adaptation).
                - profits_no_event: Profits for the no-drought scenario (no adaptation).
                - total_profits_adaptation: Total profits for each scenario (with adaptation).
                - profits_no_event_adaptation: Profits for the no-drought scenario (with adaptation).
                - ids_to_switch_to: Mapping indices used to update agent histories.
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
        self,
        unique_crop_calendars: npt.NDArray[np.integer],
        farmer_yield_probability_relation: npt.NDArray[np.floating],
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.integer],
    ]:
        """Calculate profits for crop-adaptation options under drought scenarios.

        Computes baseline profits and profits for each candidate crop-calendar
        option using yield-SPEI relations, then aggregates per scenario and for
        the no-drought case.

        Args:
            unique_crop_calendars: Distinct crop-calendar
                options to evaluate (calendar IDs per phase).
            farmer_yield_probability_relation: Per-farmer
                yield-SPEI relationship used to compute yield ratios.

        Returns:
            tuple[
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
                npt.NDArray[np.integer],
            ]:
                - Baseline total profits across drought scenarios (events x farmers).
                - Baseline profits for the no-drought scenario (farmers,).
                - Total profits for each crop option across scenarios
                (options x events x farmers).
                - No-drought profits for each crop option (options x farmers).
                - Mapping of farmers to the donor farmer index for each option
                (farmers x options).
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
                self.in_command_area.reshape(-1, 1),
                self.elev_class.reshape(-1, 1),
                insurance_differentiator.reshape(-1, 1),
                self.well_status.reshape(-1, 1),
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
            / self.field_size_per_farmer[..., None],  # income per m2
            crop_elevation_group=crop_elevation_group,
            unique_crop_groups=unique_crop_groups,
            group_indices=group_indices,
            crop_calendar=self.var.crop_calendar.data,
            unique_crop_calendars=unique_crop_calendars,
            p_droughts=self.var.p_droughts,
            past_window=7,
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
            yield_ratios: Yield ratios for agents under different drought scenarios.
            crops_mask: Mask indicating valid crop entries.
            nan_array: Array filled with NaNs for reference.

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

    def budget_check(
        self,
        total_annual_costs_m2: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.bool_]:
        """Check if farmers' budgets can cover annual per-m² costs for insurance.

        Uses the insured spending-cap relationship (from
        ``farmer_yield_probability_relation_exp_cap``) to compute each farmer’s
        no-event profits, multiplies by the expenditure cap, and compares against
        the required annual cost per m².

        Args:
            total_annual_costs_m2: Annual adaptation cost
                per m² for each farmer.

        Returns:
            npt.NDArray[np.bool_]: Boolean mask indicating which farmers are within budget.
        """
        # Determine the additional spending room insurance brings
        # Is done separately as index insurance does affect income,
        # but does not affect adaptation decisions
        yield_ratios_exp_cap = self.convert_probability_to_yield_ratio(
            self.farmer_yield_probability_relation_exp_cap
        )
        total_profits_exp_cap = self.compute_total_profits(yield_ratios_exp_cap)
        _, profits_no_event_exp_cap = self.format_results(total_profits_exp_cap)

        return (
            profits_no_event_exp_cap * self.var.expenditure_cap
        ) > total_annual_costs_m2

    def format_results(
        self, total_profits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transpose and slice the total profits matrix, and extract the 'no drought' scenario profits.

        Args:
            total_profits: Total profits matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transposed profits and 'no drought' scenario profits.
        """
        total_profits = total_profits.T
        profits_no_event = total_profits[-1, :]
        total_profits = total_profits[:-1, :]
        return total_profits, profits_no_event

    def convert_probability_to_yield_ratio(
        self,
        farmer_yield_probability_relation: npt.NDArray[np.floating],
        model: Literal["exponential", "linear"] = "exponential",
    ) -> npt.NDArray[np.floating]:
        """Convert drought probabilities to yield ratios using a the yield-spei relation.

        Models:
        - exponential: y = a * exp(b * x)
        - linear:      y = c + m * x

        The parameter columns in ``farmer_yield_probability_relation`` must be
        ``[intercept, slope]`` as returned by the fitters. Output is clipped to
        ``[0, 1]``.

        Args:
            farmer_yield_probability_relation: Per-farmer
                parameters of shape ``(n_farmers, 2)`` where column 0 is the intercept
                and column 1 is the slope.
            model: Relation type. Defaults
                to ``"exponential"``.

        Returns:
            npt.NDArray[np.floating]: Yield ratios per farmer and event with shape
            ``(n_farmers, n_events)``.

        Raises:
            ValueError: If ``model`` is invalid, parameters do not have two columns,
                or drought probabilities are non-positive.
        """
        # x: same driver you used before (note: you’re feeding 1/p_droughts)
        x = 1.0 / self.var.p_droughts  # shape: (num_events,)
        x = x[np.newaxis, :]  # -> (1, num_events)

        # params
        P = farmer_yield_probability_relation  # -> (num_agents, 2)
        p0 = P[:, 0:1]  # intercept column, shape: (num_agents, 1)
        p1 = P[:, 1:2]  # slope column,     shape: (num_agents, 1)

        if model == "exponential":
            # y = a * exp(b * x)
            y = p0 * np.exp(p1 * x)  # broadcast -> (num_agents, num_events)
        elif model == "linear":
            # y = c + m * x
            y = p0 + p1 * x
        else:
            raise ValueError("model must be 'exponential' or 'linear'")

        # Clamp to [0, 1]
        y = np.clip(y, 0.0, 1.0)

        return y

    def create_unique_groups(
        self,
        *additional_diffentiators: npt.NDArray[np.integer],
    ) -> tuple[npt.NDArray[np.int_], int]:
        """Create per-agent group indices from base classes and optional differentiators.

        If extra differentiator arrays are provided, they are stacked with
        ``self.var.farmer_base_class`` (column-wise) to form the grouping key; otherwise
        only the base class is used. Groups are defined by unique rows.

        Args:
            *additional_diffentiators: Optional per-agent arrays
                (all length ``n_agents``) that further distinguish groups.

        Returns:
            tuple[npt.NDArray[np.int_], int]:
                - group_indices: Array of length ``n_agents`` mapping each agent to a group id.
                - n_groups: Number of unique groups.
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
        self,
        additional_diffentiators: npt.NDArray[np.integer],
        adapted: npt.NDArray[np.bool_],
        yield_ratios: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """Compute yield-ratio gains from an adaptation within peer groups.

        Forms peer groups using ``additional_diffentiators`` (distinct from the
        adaptation status), compares mean yield ratios of adapted vs. unadapted
        farmers within each group across drought events, and assigns each farmer
        the group's gain vector. Also finds, per group, the most similar adapted
        farmer to use as a donor index.

        Args:
            additional_diffentiators: Per-farmer grouping
                keys (e.g., classes/categorical columns) used to form peer groups.
            adapted: Boolean mask indicating which farmers
                have adopted the adaptation.
            yield_ratios: Yield ratios per farmer and
                drought event, shaped ``(n_farmers, n_events)``.

        Returns:
            tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
                - gains_adaptation: Per-farmer yield-ratio gain vectors,
                shape ``(n_farmers, n_events)``.
                - ids_to_switch_to: Donor farmer indices per farmer (``-1`` if none).
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

    def reset_adaptation_status(
        self,
        farmer_yield_probability_relation: npt.NDArray[np.floating],
        adapted: npt.NDArray[np.bool_],
        additional_diffentiator_expiration: npt.NDArray[np.bool_],
        additional_diffentiator_grouping: npt.NDArray[np.integer],
        adaptation_type: npt.NDArray[np.integer] | int,
    ) -> None:
        """Expire shallow/aged wells and refresh affected farmers' histories.

        Marks wells as expired if their time-adapted reaches the configured lifespan
        or if groundwater depth exceeds the current well depth. For those farmers,
        finds the most similar unadapted peer group (via SEUT) and updates their
        income/potential-income/SPEI histories accordingly.
        """
        expired_adaptations = (
            self.var.time_adapted[:, adaptation_type] == self.var.lifespan_well
        ) | additional_diffentiator_expiration
        self.var.adaptations[expired_adaptations, adaptation_type] = -1
        self.var.time_adapted[expired_adaptations, adaptation_type] = -1

        # Determine the IDS of the most similar group of yield
        (
            total_profits,
            profits_no_event,
            total_profits_adaptation,
            profits_no_event_adaptation,
            ids_to_switch_to,
        ) = self.profits_SEUT(
            additional_diffentiator_grouping,
            ~adapted,
            farmer_yield_probability_relation,
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
        self,
        additional_diffentiators: npt.NDArray[np.integer],
        adapted: npt.NDArray[np.bool_],
        energy_cost: npt.NDArray[np.floating],
        water_cost: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Compute per-farmer energy and water cost differences for an adaptation.

        Forms peer groups using ``additional_diffentiators`` and, within each group
        that contains both adapted and unadapted farmers, computes the mean cost
        difference (adapted - unadapted) for energy and water. These group-level
        gains are then mapped back to each farmer.

        Args:
            additional_diffentiators: Per-farmer grouping
                keys used to define comparable peer groups.
            adapted: Boolean mask indicating which farmers
                have adopted the adaptation.
            energy_cost: Annual energy cost per farmer.
            water_cost: Annual water cost per farmer.

        Returns:
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
                - energy_cost_adaptation_gain: Per-farmer energy-cost gain
                (adapted - unadapted) mapped from group means.
                - water_cost_adaptation_gain: Per-farmer water-cost gain
                (adapted - unadapted) mapped from group means.
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
        """Update loan trackers, clear expired loans, and inflation-adjust exports.

        Decrements active loan durations, zeros out expired loan amounts (and
        deducts them from totals), and computes inflation-adjusted views of loan
        costs, yearly income, and premiums for export.
        """
        # Subtract 1 off each loan duration, except if that loan is at 0
        self.var.loan_tracker -= self.var.loan_tracker != 0
        # If the loan tracker is at 0, cancel the loan amount and subtract it of the total
        expired_loan_mask: npt.NDArray[np.bool_] = self.var.loan_tracker.data == 0

        # Add a column to make it the same shape as the loan amount array
        new_column = np.full((self.var.n, 1, 5), False)
        expired_loan_mask = np.column_stack((expired_loan_mask, new_column))

        # Sum the expired loan amounts
        ending_loans = expired_loan_mask * self.var.all_loans_annual_cost
        total_loan_reduction = np.sum(ending_loans, axis=(1, 2))

        # Subtract it from the total loans and set expired loans to 0
        self.var.all_loans_annual_cost[:, -1, 0] -= total_loan_reduction
        self.var.all_loans_annual_cost[expired_loan_mask] = 0
        self.var.all_loans_annual_cost[self.var.all_loans_annual_cost < 0] = 0

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

        self.var.adjusted_pr_premium = self.var.pr_premium / cumulative_inflation
        self.var.adjusted_index_premium = self.var.index_premium / cumulative_inflation
        self.var.adjusted_personal_premium = (
            self.var.personal_premium / cumulative_inflation
        )

    def get_value_per_farmer_from_region_id(
        self,
        data: tuple[DateIndex, dict[int, npt.NDArray[Any]]],
        time: datetime,
        subset: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Map region-level values to farmers for a given time.

        Looks up the time index in ``data[0]``, then selects per-region values from
        ``data[1]`` and broadcasts them to each farmer according to
        ``self.var.region_id`` (or a subset of farmers if provided).

        Args:
            data: A pair
                ``(date_index, values_by_region)`` where ``date_index[time] -> int``
                and ``values_by_region[region_id][index] -> float``.
            time: Timestamp used to obtain the column/index into the values array.
            subset: Boolean mask of farmers to map.
                If ``None``, all farmers are used.

        Returns:
            npt.NDArray[np.float32]: Per-farmer values aligned with the (sub)set of farmers.
        """
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
    def is_irrigated(self) -> npt.NDArray[np.bool_]:
        """Return a boolean mask of farmers who have any irrigation adaptation."""
        return (
            self.var.adaptations[:, [SURFACE_IRRIGATION_EQUIPMENT, WELL_ADAPTATION]] > 0
        ).any(axis=1)

    @property
    def irrigated_fields(self) -> npt.NDArray[np.bool_]:
        """Return a boolean mask of fields that are irrigated."""
        irrigated_fields = np.take(
            self.is_irrigated,
            self.HRU.var.land_owners,
        )
        irrigated_fields[self.HRU.var.land_owners == -1] = False
        return irrigated_fields

    @property
    def groundwater_depth(self) -> npt.NDArray[np.floating]:
        """Return per-farmer groundwater depth and assert no NaNs."""
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

    def create_farmer_classes(
        self, *characteristics: npt.NDArray[Any]
    ) -> npt.NDArray[np.int_]:
        """Return per-farmer class ids from one or more categorical characteristic arrays."""
        agent_classes = np.unique(
            np.stack(characteristics), axis=1, return_inverse=True
        )[1]
        return agent_classes

    @property
    def main_irrigation_source(self) -> npt.NDArray[np.int_]:
        """Return the dominant irrigation source per farmer (or NO_IRRIGATION if none)."""
        # Set to 0 if channel abstraction is bigger than reservoir and groundwater,
        # 1 for reservoir, 2 for groundwater and -1 no abstraction
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

        Raises:
            ValueError: When farmers don't have a yield probability relation yet.
        """
        if not self.model.simulate_hydrology:
            return

        timer = TimingModule("crop_farmers")

        self.harvest()
        timer.finish_split("harvest")
        self.plant()
        timer.finish_split("planting")

        self.water_abstraction_sum()
        timer.finish_split("water abstraction calculation")

        ## yearly actions
        if self.model.current_time.month == 1 and self.model.current_time.day == 1:
            if self.model.current_time.year > self.model.spinup_start.year:
                # reset the irrigation limit, but only if a full year has passed already. Otherwise
                # the cumulative water deficit is not year completed.
                self.var.remaining_irrigation_limit_m3_reservoir[:] = (
                    self.var.irrigation_limit_m3[:]
                )
                self.var.remaining_irrigation_limit_m3_channel[:] = (
                    self.var.irrigation_limit_m3[:]
                )
                self.var.remaining_irrigation_limit_m3_groundwater[:] = (
                    self.var.irrigation_limit_m3[:]
                )

                self.save_yearly_spei()
                self.save_yearly_pr()

            # Set yearly yield ratio based on the difference between saved actual and potential profit
            self.var.yearly_yield_ratio = (
                self.var.yearly_income / self.var.yearly_potential_income
            )

            k = 8

            self.in_command_area = (self.command_area >= 0).astype(np.int8)

            edges = np.nanpercentile(
                self.var.elevation, np.linspace(100 / k, 100 - 100 / k, k - 1)
            )
            self.elev_class = np.digitize(self.var.elevation, edges, right=True).astype(
                np.int8
            )

            # create a unique index for each type of crop calendar that a farmer follows
            crop_calendar_group = np.unique(
                self.var.crop_calendar[:, :, 0], axis=0, return_inverse=True
            )[1]

            self.insurance_diffentiator = (
                self.var.adaptations[:, PERSONAL_INSURANCE_ADAPTATION] > 0
            ).astype(np.int32)  # only personal insurance affects adaptation

            self.blank_additional_differentiator = np.zeros(
                self.var.n, dtype=np.float32
            )
            self.well_status = (self.var.adaptations[:, WELL_ADAPTATION] > 0).astype(
                np.int32
            )

            self.var.farmer_base_class[:] = self.create_farmer_classes(
                crop_calendar_group,
                self.in_command_area,
                self.elev_class,
                self.insurance_diffentiator,
            )
            print("Nr of base groups", len(np.unique(self.var.farmer_base_class[:])))

            energy_cost, water_cost, average_extraction_speed = (
                self.calculate_water_costs()
            )

            timer.finish_split("water & energy costs")

            if (
                not self.model.in_spinup
                and "ruleset" in self.config
                and not self.config["ruleset"] == "no-adaptation"
            ):
                # Determine the relation between drought probability and yield
                farmer_yield_probability_relation = (
                    self.calculate_yield_spei_relation_group_exp(
                        self.var.yearly_yield_ratio, self.var.yearly_SPEI_probability
                    )
                )
                self.farmer_yield_probability_relation_exp_cap = (
                    farmer_yield_probability_relation.copy()
                )
                self.calculate_yield_spei_relation_group_lin(
                    self.var.yearly_yield_ratio, self.var.yearly_SPEI_probability
                )
                # Set the base insured income of this year as the yearly income
                # Later, insured losses will be added to this
                self.var.insured_yearly_income[:, 0] = self.var.yearly_income[
                    :, 0
                ].copy()

                timer.finish_split("yield-spei relation")

                if (
                    self.personal_insurance_adaptation_active
                    or self.index_insurance_adaptation_active
                    or self.pr_insurance_adaptation_active
                ):
                    # save the base relations for determining the difference with and without insurance
                    farmer_yield_probability_relation_base = (
                        farmer_yield_probability_relation.copy()
                    )
                    potential_insured_loss = self.potential_insured_loss()

                    self.var.payout_mask = np.zeros((self.var.n, 7), dtype=np.bool)

                    government_premium_cap = self.government_premium_cap()
                if self.personal_insurance_adaptation_active:
                    # Now determine the potential (past & current) indemnity payments and recalculate
                    # probability and yield relation
                    self.var.personal_premium[:] = self.premium_personal_insurance(
                        potential_insured_loss, government_premium_cap
                    )
                    # Give only the insured agents the relation with covered losses
                    personal_insured_farmers_mask = (
                        self.var.adaptations[:, PERSONAL_INSURANCE_ADAPTATION] > 0
                    )

                    # Add the insured loss to the income of this year's insured farmers
                    potential_insured_loss_personal = self.insured_payouts_personal(
                        personal_insured_farmers_mask
                    )

                    farmer_yield_probability_relation_insured_personal = (
                        self.insured_yields(potential_insured_loss_personal)
                    )

                    farmer_yield_probability_relation[
                        personal_insured_farmers_mask, :
                    ] = farmer_yield_probability_relation_insured_personal[
                        personal_insured_farmers_mask, :
                    ]
                    self.farmer_yield_probability_relation_exp_cap[
                        personal_insured_farmers_mask, :
                    ] = farmer_yield_probability_relation_insured_personal[
                        personal_insured_farmers_mask, :
                    ]
                    timer.finish_split("personal insurance")
                if self.index_insurance_adaptation_active:
                    gev_params = self.var.GEV_parameters.data
                    strike_vals = np.round(np.arange(0.0, -2.6, -0.2), 2)
                    exit_vals = np.round(np.arange(-2, -3.6, -0.2), 2)
                    rate_vals = np.geomspace(10, 5000, 10)
                    # Calculate best strike, exit, rate for chosen contract
                    strike, exit, rate, self.var.index_premium[:] = (
                        self.premium_index_insurance(
                            potential_insured_loss=potential_insured_loss,
                            history=self.var.yearly_SPEI.data,
                            gev_params=gev_params,
                            strike_vals=strike_vals,
                            exit_vals=exit_vals,
                            rate_vals=rate_vals,
                            government_premium_cap=government_premium_cap,
                        )
                    )
                    index_insured_farmers_mask = (
                        self.var.adaptations[:, INDEX_INSURANCE_ADAPTATION] > 0
                    )
                    potential_insured_loss_index = self.insured_payouts_index(
                        strike,
                        exit,
                        rate,
                        index_insured_farmers_mask,
                        INDEX_INSURANCE_ADAPTATION,
                    )
                    farmer_yield_probability_relation_insured_index = (
                        self.insured_yields(potential_insured_loss_index)
                    )
                    index_insured_farmers_mask = (
                        self.var.adaptations[:, INDEX_INSURANCE_ADAPTATION] > 0
                    )
                    self.farmer_yield_probability_relation_exp_cap[
                        index_insured_farmers_mask, :
                    ] = farmer_yield_probability_relation_insured_index[
                        index_insured_farmers_mask, :
                    ]
                    timer.finish_split("index insurance")
                if self.pr_insurance_adaptation_active:
                    gev_params = self.var.GEV_pr_parameters.data
                    strike_vals = np.round(np.arange(1500, 300, -100), 2)
                    low, high, N = 0, 800, 10
                    u = np.linspace(0, 1, N)  # linear grid on [0,1]
                    s = 0.5 * (1 - np.cos(np.pi * u))
                    exit_vals = low + s * (high - low)
                    # exit_vals = np.round(np.arange(600, 50, -50), 2)
                    rate_vals = np.geomspace(10, 5000, 10)
                    # Calculate best strike, exit, rate for chosen contract
                    strike, exit, rate, self.var.pr_premium[:] = (
                        self.premium_index_insurance(
                            potential_insured_loss=potential_insured_loss,
                            history=self.var.yearly_pr.data,
                            gev_params=gev_params,
                            strike_vals=strike_vals,
                            exit_vals=exit_vals,
                            rate_vals=rate_vals,
                            government_premium_cap=government_premium_cap,
                        )
                    )
                    pr_insured_farmers_mask = (
                        self.var.adaptations[:, PR_INSURANCE_ADAPTATION] > 0
                    )
                    potential_insured_loss_pr = self.insured_payouts_index(
                        strike,
                        exit,
                        rate,
                        pr_insured_farmers_mask,
                        PR_INSURANCE_ADAPTATION,
                    )
                    farmer_yield_probability_relation_insured_pr = self.insured_yields(
                        potential_insured_loss_pr
                    )
                    pr_insured_farmers_mask = (
                        self.var.adaptations[:, PR_INSURANCE_ADAPTATION] > 0
                    )
                    self.farmer_yield_probability_relation_exp_cap[
                        pr_insured_farmers_mask, :
                    ] = farmer_yield_probability_relation_insured_pr[
                        pr_insured_farmers_mask, :
                    ]
                    timer.finish_split("precipitation insurance")
                # These adaptations can only be done if there is a yield-probability relation
                if not np.all(farmer_yield_probability_relation == 0):
                    if self.wells_adaptation_active:
                        self.adapt_irrigation_well(
                            farmer_yield_probability_relation,
                            average_extraction_speed,
                            energy_cost,
                            water_cost,
                        )
                        timer.finish_split("irr well")
                    if self.sprinkler_adaptation_active:
                        adaptation_costs_m2 = np.full(
                            self.var.n,
                            self.model.config["agent_settings"]["farmers"][
                                "expected_utility"
                            ]["adaptation_sprinkler"]["m2_cost"],
                        )
                        self.adapt_irrigation_efficiency(
                            farmer_yield_probability_relation,
                            energy_cost,
                            water_cost,
                            adaptation_costs_m2,
                            IRRIGATION_EFFICIENCY_ADAPTATION,
                            self.var.irr_eff_drip,
                            self.var.return_fraction_drip,
                        )

                        timer.finish_split("irr efficiency")
                    if self.crop_switching_adaptation_active:
                        self.adapt_crops(farmer_yield_probability_relation)
                        timer.finish_split("adapt crops")

                    if (
                        self.personal_insurance_adaptation_active
                        and self.index_insurance_adaptation_active
                        and self.pr_insurance_adaptation_active
                    ):
                        # In scenario with both insurance, compare simultaneously
                        self.adapt_insurance(
                            np.array(
                                [
                                    PERSONAL_INSURANCE_ADAPTATION,
                                    INDEX_INSURANCE_ADAPTATION,
                                    PR_INSURANCE_ADAPTATION,
                                ]
                            ),
                            ["Personal", "Precipitation"],
                            farmer_yield_probability_relation_base,
                            [
                                farmer_yield_probability_relation_insured_personal,
                                farmer_yield_probability_relation_insured_index,
                                farmer_yield_probability_relation_insured_pr,
                            ],
                            [
                                self.var.personal_premium,
                                self.var.index_premium,
                                self.var.pr_premium,
                            ],
                        )
                    elif self.personal_insurance_adaptation_active:
                        self.adapt_insurance(
                            [PERSONAL_INSURANCE_ADAPTATION],
                            ["Personal"],
                            farmer_yield_probability_relation_base,
                            [farmer_yield_probability_relation_insured_personal],
                            [self.var.personal_premium],
                        )
                        timer.finish_split("adapt pers. insurance")
                    elif self.index_insurance_adaptation_active:
                        self.adapt_insurance(
                            [INDEX_INSURANCE_ADAPTATION],
                            ["Index"],
                            farmer_yield_probability_relation_base,
                            [farmer_yield_probability_relation_insured_index],
                            [self.var.index_premium],
                        )
                        timer.finish_split("adapt index insurance")
                    elif self.pr_insurance_adaptation_active:
                        self.adapt_insurance(
                            [PR_INSURANCE_ADAPTATION],
                            ["Precipitation"],
                            farmer_yield_probability_relation_base,
                            [farmer_yield_probability_relation_insured_pr],
                            [self.var.pr_premium],
                        )
                        timer.finish_split("adapt prec. insurance")
                else:
                    raise ValueError(
                        "Cannot adapt without yield - probability relation"
                    )

            advance_crop_rotation_year(
                current_crop_calendar_rotation_year_index=self.var.current_crop_calendar_rotation_year_index,
                crop_calendar_rotation_years=self.var.crop_calendar_rotation_years,
            )

            # Update loans
            self.update_loans()

            matrix_abstraction = (
                self.var.yearly_abstraction_m3_by_farmer
            )  # shape (n_farmers, 4, 20)
            shift_and_reset_matrix(
                matrix_abstraction.reshape(-1, matrix_abstraction.shape[-1])
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

        self.report(locals())

    def remove_agents(
        self,
        farmer_indices: ArrayInt64,
        new_land_use_type: int,
    ) -> np.ndarray:
        """Remove multiple farmers and reassign their HRUs.

        Removes the specified farmers (highest index first), updates model state,
        resets the social network, and returns the concatenated HRU indices that
        were disowned.

        Args:
            farmer_indices: Farmer indices to remove.
            new_land_use_type: Land-use code to assign to vacated HRUs.

        Returns:
            np.ndarray: Concatenated array of HRU indices that were disowned.
        """
        farmer_indices = np.array(farmer_indices)
        if farmer_indices.size > 0:
            farmer_indices = np.sort(farmer_indices)[::-1]
            HRUs_with_removed_farmers: list[np.ndarray] = []
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
        """Remove a single farmer and transfer the last farmer's data into its slot.

        Disowns the farmer's HRUs, updates arrays (moving the last farmer into
        the removed slot when needed), updates field indices, and returns the HRU
        indices that were disowned.

        Args:
            farmer_idx: Index of the farmer to remove.
            new_land_use_type: Land-use code to assign to vacated HRUs.

        Returns:
            np.ndarray: HRU indices that were disowned for this farmer.
        """
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
        indices: tuple[np.ndarray, np.ndarray],
        values: dict[str, object] = {
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
    ) -> None:
        """Add a new farmer at given HRU indices and initialize arrays.

        Args:
            indices: Row/column index arrays that
                define the HRUs to assign to the new farmer.
            values: Per-array initialization values
                keyed by agent array name.
        """
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
    def n(self) -> int:
        """Number of farmer agents."""
        return self.var._n

    @n.setter
    def n(self, value: int) -> None:
        """Set the number of farmer agents."""
        self.var._n = value

    def get_farmer_elevation(self) -> DynamicArray:
        """Compute mean elevation per farmer.

        Returns:
            DynamicArray: Mean elevation per farmer (meters), sized to ``max_n``.
        """
        # get elevation per farmer
        elevation_subgrid = read_grid(
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
