"""Water demand module for the hydrological model."""

# --------------------------------------------------------------------------------
# This file contains code that has been adapted from an original source available
# in a public repository under the GNU General Public License. The original code
# has been modified to fit the specific needs of this project.
#
# Original source repository: https://github.com/iiasa/CWatM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from scipy.ndimage import value_indices

from geb.geb_types import (
    ArrayFloat32,
    ArrayInt32,
    ArrayInt64,
    TwoDArrayFloat32,
    TwoDArrayInt32,
)
from geb.module import Module
from geb.store import Bucket
from geb.workflows import TimingModule, balance_check
from geb.workflows.io import read_grid
from geb.workflows.raster import write_to_array

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


class WaterDemandVariables(Bucket):
    """Variables for the WaterDemand module."""

    return_flow_m3_agents: ArrayFloat32
    abstraction_area_indices: TwoDArrayInt32
    abstraction_river_indices: TwoDArrayInt32


def create_abstraction_areas(
    basin_ids: ArrayInt32,
    river_ids: ArrayInt32,
    rivers: gpd.GeoDataFrame,
    minimum_shreve_stream_order: int = 20,
) -> tuple[TwoDArrayInt32, TwoDArrayInt32]:
    """Create abstraction areas based on the river network.

    Water demand abstraction (such as domestic and industrial) is assumed to occur
    from larger rivers. If users abstract from each grid cell that has demand, water
    is abstracted from very small rivers, which also leads to very high groundwater
    abstraction in those cells because the demand is not satisfiable from the river.
    This is highly unrealistic.

    Therefore, we define abstraction areas based on the river network. Each abstraction
    area is associated with a river of shreve stream order above a set threshold.

    All water demands in an abstraction area are transferred downstream to the river
    of the abstraction area, and abstraction is assumed to occur from that river.

    Note:
        When the model is run in an area without a river of the minimum shreve stream order,
        empty index arrays are returned.

    Args:
        basin_ids: The basin IDs for each grid cell.
        river_ids: The river IDs for each grid cell.
        rivers: The active rivers in the model.
        minimum_shreve_stream_order: The minimum shreve stream order of rivers that can be abstraction rivers.

    Returns:
        A tuple containing:
        - A 2D array mapping abstraction area IDs to linear indices of grid cells in those areas.
        - A 2D array mapping abstraction area IDs to linear indices of river cells in the
            associated abstraction river.
    """
    linear_idx_per_basin_id: dict[int, tuple[ArrayInt64]] = value_indices(
        basin_ids, ignore_value=-1
    )
    abstraction_area_ids_per_river_id: dict[int, int] = {}
    abstraction_river_ids_by_area_id: list[int] = []
    linear_idx_per_abstraction_area_id: list[list[tuple[ArrayInt64]]] = []

    for river_idx, river in rivers.iterrows():
        if not river["represented_in_grid"]:
            continue
        assert isinstance(river_idx, int)
        abstraction_river = river
        while (
            abstraction_river["shreve_stream_order"] < minimum_shreve_stream_order
            or not abstraction_river["represented_in_grid"]
        ):
            downstream_idx = abstraction_river["downstream_ID"]
            try:
                abstraction_river = rivers.loc[downstream_idx]
            except KeyError:
                abstraction_river = None
                break

        if abstraction_river is None or river_idx not in linear_idx_per_basin_id:
            continue

        abstraction_river_id: int = abstraction_river.name  # ty:ignore[invalid-assignment]
        if abstraction_river_id not in abstraction_area_ids_per_river_id:
            abstraction_area_ids_per_river_id[abstraction_river_id] = len(
                abstraction_river_ids_by_area_id
            )
            abstraction_river_ids_by_area_id.append(abstraction_river_id)
            linear_idx_per_abstraction_area_id.append([])

        abstraction_area_id = abstraction_area_ids_per_river_id[abstraction_river_id]
        linear_idx_per_abstraction_area_id[abstraction_area_id].append(
            linear_idx_per_basin_id[river_idx]
        )

    # Convert to 2D arrays with -1 padding
    if not abstraction_river_ids_by_area_id:
        return (
            np.zeros((0, 0), dtype=np.int32),
            np.zeros((0, 0), dtype=np.int32),
        )

    # Abstraction areas (source cells)
    area_indices_list = [
        np.concatenate([xy[0] for xy in xy_list]).astype(np.int32)
        for xy_list in linear_idx_per_abstraction_area_id
    ]
    max_area_size = max(len(indices) for indices in area_indices_list)
    abstraction_area_indices = np.full(
        (len(area_indices_list), max_area_size), -1, dtype=np.int32
    )
    for i, indices in enumerate(area_indices_list):
        abstraction_area_indices[i, : len(indices)] = indices

    # Abstraction river cells (target cells)
    linear_idx_per_river_id: dict[int, tuple[ArrayInt64]] = value_indices(
        river_ids, ignore_value=-1
    )
    river_indices_list = [
        linear_idx_per_river_id[river_id][0].astype(np.int32)
        for river_id in abstraction_river_ids_by_area_id
    ]
    max_river_size = max(len(indices) for indices in river_indices_list)
    abstraction_river_indices = np.full(
        (len(river_indices_list), max_river_size), -1, dtype=np.int32
    )
    for i, indices in enumerate(river_indices_list):
        abstraction_river_indices[i, : len(indices)] = indices

    return abstraction_area_indices, abstraction_river_indices


def assign_demand_to_abstraction_rivers(
    water_demand: ArrayFloat32,
    abstraction_area_indices: TwoDArrayInt32,
    abstraction_river_indices: TwoDArrayInt32,
) -> ArrayFloat32:
    """Assign gridded water demand to river cells in abstraction areas.

    All water demands within each abstraction area are aggregated and distributed
    evenly across the river cells of the associated abstraction river reach.

    Args:
        water_demand: Compressed 1D array of water demand per grid cell (m3/day).
        abstraction_area_indices: 2D array mapping abstraction areas to source grid cell indices.
        abstraction_river_indices: 2D array mapping abstraction areas to target river cell indices.

    Returns:
        Water demand assigned to river cells in m3/day.
    """
    water_demand_assigned_to_rivers: ArrayFloat32 = np.zeros(
        water_demand.shape[0], dtype=np.float32
    )

    for i in range(abstraction_area_indices.shape[0]):
        area_indices = abstraction_area_indices[i]
        area_indices = area_indices[area_indices != -1]
        river_indices = abstraction_river_indices[i]
        river_indices = river_indices[river_indices != -1]

        if river_indices.size == 0 or area_indices.size == 0:
            continue

        water_demand_in_abstraction_area: float = float(
            water_demand[area_indices].sum()
        )
        demand_per_cell: float = water_demand_in_abstraction_area / river_indices.size
        water_demand_assigned_to_rivers[river_indices] = demand_per_cell

    return water_demand_assigned_to_rivers


def assign_demand_and_return_flow_to_abstraction_rivers(
    water_demand: ArrayFloat32,
    water_consumption: ArrayFloat32,
    abstraction_area_indices: TwoDArrayInt32,
    abstraction_river_indices: TwoDArrayInt32,
) -> tuple[ArrayFloat32, ArrayFloat32]:
    """Assign gridded water demand and return flow to river cells in abstraction areas.

    All water demands and consumption within each abstraction area are aggregated.
    Return flow is calculated as demand minus consumption (clamped to at least 0).
    Both are distributed evenly across the river cells of the associated abstraction river reach.

    Args:
        water_demand: Compressed 1D array of water demand per grid cell (m3/day).
        water_consumption: Compressed 1D array of water consumption per grid cell (m3/day).
        abstraction_area_indices: 2D array mapping abstraction areas to source grid cell indices.
        abstraction_river_indices: 2D array mapping abstraction areas to target river cell indices.

    Returns:
        A tuple containing:
        - Water demand assigned to river cells in m3/day.
        - Return flow assigned to river cells in m3/day.
    """
    water_demand_assigned_to_rivers: ArrayFloat32 = np.zeros(
        water_demand.shape[0], dtype=np.float32
    )
    return_flow_assigned_to_rivers: ArrayFloat32 = np.zeros(
        water_demand.shape[0], dtype=np.float32
    )

    for i in range(abstraction_area_indices.shape[0]):
        area_indices = abstraction_area_indices[i]
        area_indices = area_indices[area_indices != -1]
        river_indices = abstraction_river_indices[i]
        river_indices = river_indices[river_indices != -1]

        if river_indices.size == 0 or area_indices.size == 0:
            continue

        water_demand_in_abstraction_area: float = float(
            water_demand[area_indices].sum()
        )
        water_consumption_in_abstraction_area: float = float(
            water_consumption[area_indices].sum()
        )
        return_flow_in_abstraction_area: float = (
            water_demand_in_abstraction_area - water_consumption_in_abstraction_area
        )

        demand_per_cell: float = water_demand_in_abstraction_area / river_indices.size
        return_flow_per_cell: float = (
            return_flow_in_abstraction_area / river_indices.size
        )
        if return_flow_per_cell < 0.0:
            return_flow_per_cell = 0.0

        water_demand_assigned_to_rivers[river_indices] = demand_per_cell
        return_flow_assigned_to_rivers[river_indices] = return_flow_per_cell

    return water_demand_assigned_to_rivers, return_flow_assigned_to_rivers


def weighted_sum_per_reservoir(
    farmer_command_area: npt.NDArray[np.int32],
    weights: npt.NDArray[np.float32],
    min_length: int,
) -> npt.NDArray[np.float32]:
    """Calculate weighted sum of values per reservoir.

    Args:
        farmer_command_area: Array mapping each farmer to a reservoir command area.
        weights: Values to be summed, weighted by the command area.
        min_length: Minimum length of the output array, typically the number of reservoirs.

    Returns:
        Weighted sum of values per reservoir.
    """
    mask: npt.NDArray[np.bool] = farmer_command_area != -1
    farmer_command_area = farmer_command_area[mask]
    weights = weights[mask]
    return np.bincount(
        farmer_command_area, weights=weights, minlength=min_length
    ).astype(weights.dtype)


class WaterDemand(Module):
    """Water demand module for the hydrological model.

    Args:
        model: The GEB model instance.
        hydrology: The hydrology submodel instance.
    """

    var: WaterDemandVariables

    def __init__(self, model: GEBModel, hydrology: Hydrology) -> None:
        """Initialize the water demand module.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology submodel instance.
        """
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Name of the module.

        Should be identical to the path of the module in the model.

        Returns:
            The name of the module.
        """
        return "hydrology.water_demand"

    def spinup(self) -> None:
        """Perform any necessary spinup for the water demand module.

        This method initializes the reservoir command areas for the HRU grid and
        river abstraction areas for water demand abstractions.
        """
        subgrid_command_areas = read_grid(
            self.model.files["subgrid"]["waterbodies/subcommand_areas"], ndim=2
        )
        reservoir_command_areas = self.HRU.convert_subgrid_to_HRU(
            subgrid_command_areas,
            method="last",
        )

        waterbody_mapping = self.hydrology.waterbodies.var.waterbody_mapping
        self.HRU.var.reservoir_command_areas = np.take(
            waterbody_mapping, reservoir_command_areas, mode="clip"
        )

        if self.model.simulate_hydrology:
            basin_ids: ArrayInt32 = self.grid.load2d(
                self.model.files["grid"]["routing/basin_ids"]
            )
            river_ids: ArrayInt32 = self.hydrology.routing.var.river_ids
            rivers: gpd.GeoDataFrame = self.hydrology.routing.get_active_rivers().copy()
            (
                self.var.abstraction_area_indices,
                self.var.abstraction_river_indices,
            ) = create_abstraction_areas(basin_ids, river_ids, rivers)

    def get_available_water(
        self, gross_irrigation_demand_m3_per_command_area: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get available water from reservoirs, channels, and groundwater.

        Args:
            gross_irrigation_demand_m3_per_command_area: Gross irrigation demand in m3 per command area.

        Returns:
            Available water in m3 from channels
            Available water in m3 reservoirs
            Available water in groundwater.
        """
        available_reservoir_storage_m3: np.ndarray = np.zeros(
            self.hydrology.waterbodies.n, dtype=np.float32
        )

        available_reservoir_storage_m3[self.hydrology.waterbodies.is_reservoir] = (
            self.model.agents.reservoir_operators.get_command_area_release(
                gross_irrigation_demand_m3_per_command_area
            )
        )

        available_channel_storage_m3: np.ndarray = (
            self.hydrology.routing.router.get_available_storage(
                discharge=self.hydrology.routing.var.discharge_in_rivers_m3_s_substep,
                river_storage_alpha=self.hydrology.routing.var.river_storage_alpha,
                river_storage_beta=self.hydrology.routing.var.river_storage_beta,
                maximum_abstraction_ratio=0.1,
            )
        )

        available_channel_storage_m3 = np.maximum(available_channel_storage_m3 - 100, 0)

        assert (
            available_channel_storage_m3[self.grid.var.waterbody_ids != -1] == 0.0
        ).all()

        available_groundwater_m3: np.ndarray = (
            self.hydrology.groundwater.modflow.available_groundwater_m3.copy()
        )

        return (
            available_channel_storage_m3,
            available_reservoir_storage_m3,
            available_groundwater_m3,
        )

    def withdraw(
        self, source: npt.NDArray[np.floating], demand: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Withdraw water from a source to meet a demand.

        When the source is less than the demand, all available water is withdrawn.
        When the source is more than the demand, only the demanded amount is withdrawn.

        Source and demand are expected to be in the same units (e.g., m3).

        Source and demand are updated in place.

        Args:
            source: Available water from a source (e.g., channel, reservoir, groundwater).
            demand: Water demand.

        Returns:
            Water withdrawn from the source to meet the demand.
        """
        withdrawal = np.minimum(source, demand)
        source -= withdrawal  # update in place
        demand -= withdrawal  # update in place
        return withdrawal

    def step(
        self, root_depth_m: ArrayFloat32
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        float,
        ArrayFloat32,
    ]:
        """Perform a single time step of the water demand module.

        Water is abstracted in the following order:
            1. Domestic water demand (surface water first, then groundwater)
            2. Industry water demand (surface water first, then groundwater)
            3. Livestock water demand (surface water only)
            4. Irrigation water demand (surface water first, then reservoir water, then groundwater)

        For the domestic and irrigation water demand, the agent-based model is used,
        while for the industry and livestock water demand, a gridded approach is used.

        Args:
            root_depth_m: Root depth in meters for each HRU.

        Returns:
            Groundwater abstraction per grid cell [m3].
            Channel abstraction per grid cell [m3].
            Return flow from all sources per grid cell [m].
                This is added to the channel flow in the routing module.
            Irrigation loss to evaporation per HRU [m].
            Total water demand loss [m3].
            The actual irrigation consumption [m].

        Raises:
            ValueError: If reservoir abstraction doesn't fully deplete calculated volume.
        """
        timer: TimingModule = TimingModule("Water demand")

        total_water_demand_loss_m3 = 0.0

        def household_demand_to_grid(
            water_demand_per_household_m3: ArrayFloat32,
            household_locations: TwoDArrayFloat32,
        ) -> ArrayFloat32:
            """Convert household water demand to grid-level water demand.

            This function is passed to the household agent's water demand function, which calculates
            the water demand per household and then uses this function to convert it to a grid-level
            water demand that can be used in the hydrological model.

            Args:
                water_demand_per_household_m3: Water demand per household in m3 per day.
                household_locations: Locations of households in the same order as the water demand array.

            Returns:
                Grid-level water demand in m3 per day.
            """
            assert (water_demand_per_household_m3 >= 0).all()

            domestic_water_demand_m3_map = np.zeros(
                self.model.hydrology.grid.shape, np.float32
            )

            domestic_water_demand_m3 = write_to_array(
                domestic_water_demand_m3_map,
                water_demand_per_household_m3,
                household_locations,
                self.model.hydrology.grid.gt,
            )
            domestic_water_demand_m3 = self.model.hydrology.grid.compress(
                domestic_water_demand_m3
            )
            return domestic_water_demand_m3

        (
            domestic_water_demand_m3,
            domestic_water_efficiency_per_household,
        ) = self.model.agents.households.water_demand(household_demand_to_grid)

        # Assign domestic water demand from households to downstream abstraction rivers
        domestic_water_demand_m3 = assign_demand_to_abstraction_rivers(
            water_demand=domestic_water_demand_m3,
            abstraction_area_indices=self.var.abstraction_area_indices,
            abstraction_river_indices=self.var.abstraction_river_indices,
        )

        timer.finish_split("Domestic")
        industry_water_demand_m3, industry_return_flow_m3 = (
            self.model.agents.industry.water_demand()
        )
        timer.finish_split("Industry")
        livestock_water_demand_m3, livestock_return_flow_m3 = (
            self.model.agents.livestock_farmers.water_demand()
        )
        timer.finish_split("Livestock")

        (
            gross_irrigation_demand_m3_per_field,
            gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir,
            gross_irrigation_demand_m3_per_field_limit_adjusted_channel,
            gross_irrigation_demand_m3_per_field_limit_adjusted_groundwater,
        ) = self.model.agents.crop_farmers.get_gross_irrigation_demand_m3(root_depth_m)

        gross_irrigation_demand_m3_per_farmer_reservoir: npt.NDArray[np.float32] = (
            self.model.agents.crop_farmers.field_to_farmer(
                gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir
            )
        )

        gross_irrigation_demand_m3_per_waterbody: npt.NDArray[np.float32] = (
            weighted_sum_per_reservoir(
                self.model.agents.crop_farmers.command_area,
                gross_irrigation_demand_m3_per_farmer_reservoir,
                min_length=self.hydrology.waterbodies.n,
            )
        )

        gross_irrigation_demand_m3_per_reservoir: npt.NDArray[np.float32] = (
            gross_irrigation_demand_m3_per_waterbody[
                self.hydrology.waterbodies.is_reservoir
            ]
        )

        assert (industry_water_demand_m3 >= 0).all()
        assert (livestock_water_demand_m3 >= 0).all()

        (
            available_channel_storage_m3,
            available_reservoir_storage_m3,
            available_groundwater_m3,
        ) = self.get_available_water(gross_irrigation_demand_m3_per_reservoir)

        available_channel_storage_m3_pre = available_channel_storage_m3.copy()
        available_reservoir_storage_m3_pre = available_reservoir_storage_m3.copy()
        available_groundwater_m3_pre = available_groundwater_m3.copy()

        assert (domestic_water_efficiency_per_household == 1).all()
        domestic_water_efficiency = 1

        # 1. domestic (surface + ground)
        domestic_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, domestic_water_demand_m3
        )  # withdraw from surface water
        domestic_withdrawal_m3 += self.withdraw(
            available_groundwater_m3, domestic_water_demand_m3
        )  # withdraw from groundwater
        domestic_return_flow_m3 = domestic_withdrawal_m3 * (
            1 - domestic_water_efficiency
        )
        domestic_return_flow_m = domestic_return_flow_m3 / self.grid.var.cell_area

        domestic_water_loss_m3: np.float64 = np.float64(
            (domestic_withdrawal_m3 - domestic_return_flow_m3).sum()
        )
        total_water_demand_loss_m3 += domestic_water_loss_m3

        # 2. industry (surface + ground)
        # Pre-compute the fraction of demand that is returned at full supply.
        # If demand cannot be fully met, return flow is scaled by the same ratio
        # so that consumption (= withdrawal - return flow) stays proportional.
        industry_return_flow_fraction: npt.NDArray[np.float32] = np.zeros_like(
            industry_water_demand_m3
        )
        np.divide(
            industry_return_flow_m3,
            industry_water_demand_m3,
            out=industry_return_flow_fraction,
            where=industry_water_demand_m3 > 0,
        )
        del industry_return_flow_m3

        industry_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, industry_water_demand_m3
        )  # withdraw from surface water
        industry_withdrawal_m3 += self.withdraw(
            available_groundwater_m3, industry_water_demand_m3
        )  # withdraw from groundwater
        industry_return_flow_m3 = industry_withdrawal_m3 * industry_return_flow_fraction
        industry_return_flow_m = industry_return_flow_m3 / self.grid.var.cell_area

        industry_water_loss_m3: np.float64 = np.float64(
            (industry_withdrawal_m3 - industry_return_flow_m3).sum()
        )
        total_water_demand_loss_m3 += industry_water_loss_m3

        # 3. livestock (surface)
        # Same proportional scaling of return flow as for industry.
        livestock_return_flow_fraction: npt.NDArray[np.float32] = np.zeros_like(
            livestock_water_demand_m3
        )
        np.divide(
            livestock_return_flow_m3,
            livestock_water_demand_m3,
            out=livestock_return_flow_fraction,
            where=livestock_water_demand_m3 > 0,
        )
        del livestock_return_flow_m3

        livestock_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, livestock_water_demand_m3
        )  # withdraw from surface water
        livestock_return_flow_m3 = (
            livestock_withdrawal_m3 * livestock_return_flow_fraction
        )
        livestock_return_flow_m = livestock_return_flow_m3 / self.grid.var.cell_area

        livestock_water_loss_m3: np.float64 = np.float64(
            (livestock_withdrawal_m3 - livestock_return_flow_m3).sum()
        )
        total_water_demand_loss_m3 += livestock_water_loss_m3

        timer.finish_split("Water withdrawal")

        # 4. irrigation (surface + reservoir + ground)
        (
            irrigation_water_withdrawal_m,
            irrigation_water_consumption_m,
            return_flow_irrigation_m,
            irrigation_loss_to_evaporation_m,
            reservoir_abstraction_m3_farmers,
            groundwater_abstraction_m3_farmers,
        ) = self.model.agents.crop_farmers.abstract_water(
            gross_irrigation_demand_m3_per_field=gross_irrigation_demand_m3_per_field,
            gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir=gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir,
            gross_irrigation_demand_m3_per_field_limit_adjusted_channel=gross_irrigation_demand_m3_per_field_limit_adjusted_channel,
            gross_irrigation_demand_m3_per_field_limit_adjusted_groundwater=gross_irrigation_demand_m3_per_field_limit_adjusted_groundwater,
            available_channel_storage_m3=available_channel_storage_m3,
            available_groundwater_m3=available_groundwater_m3,
            groundwater_depth=self.hydrology.groundwater.modflow.groundwater_depth,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
        )

        self.withdraw(available_reservoir_storage_m3, reservoir_abstraction_m3_farmers)
        self.withdraw(available_groundwater_m3, groundwater_abstraction_m3_farmers)

        reservoir_storage_tolerance_m3 = 10000
        offending_mask = (
            available_reservoir_storage_m3 >= reservoir_storage_tolerance_m3
        )
        if offending_mask.any():
            raise ValueError(
                "Reservoir storage should be empty after abstraction. "
                f"Found remaining storage >= {reservoir_storage_tolerance_m3} m3: "
                f"{available_reservoir_storage_m3[offending_mask]}"
            )

        timer.finish_split("Irrigation")

        if __debug__:
            assert balance_check(
                name="water_demand_1",
                how="cellwise",
                influxes=[irrigation_water_withdrawal_m],
                outfluxes=[
                    irrigation_water_consumption_m,
                    irrigation_loss_to_evaporation_m,
                    return_flow_irrigation_m,
                ],
                tolerance=1e-5,
            )

        actual_irrigation_consumption = irrigation_water_consumption_m

        assert (actual_irrigation_consumption + 1e-5 >= 0).all()

        groundwater_abstraction_m3 = (
            available_groundwater_m3_pre - available_groundwater_m3
        )
        available_groundwater_modflow = (
            self.hydrology.groundwater.modflow.available_groundwater_m3
        )
        assert (groundwater_abstraction_m3 <= available_groundwater_modflow + 1e9).all()
        groundwater_abstraction_m3 = np.minimum(
            available_groundwater_modflow, groundwater_abstraction_m3
        )
        channel_abstraction_m3 = (
            available_channel_storage_m3_pre - available_channel_storage_m3
        )

        return_flow = (
            self.hydrology.to_grid(HRU_data=return_flow_irrigation_m)
            + domestic_return_flow_m
            + industry_return_flow_m
            + livestock_return_flow_m
        )

        if __debug__:
            assert balance_check(
                name="water_demand_1",
                how="cellwise",
                influxes=[irrigation_water_withdrawal_m],
                outfluxes=[
                    irrigation_water_consumption_m,
                    irrigation_loss_to_evaporation_m,
                    return_flow_irrigation_m,
                ],
                tolerance=1e-6,
            )
            balance_check(
                name="water_demand_2",
                how="sum",
                influxes=[],
                outfluxes=[
                    domestic_withdrawal_m3,
                    industry_withdrawal_m3,
                    livestock_withdrawal_m3,
                    (irrigation_water_withdrawal_m * self.HRU.var.cell_area).sum(),
                ],
                prestorages=[
                    available_channel_storage_m3_pre,
                    available_reservoir_storage_m3_pre,
                    available_groundwater_m3_pre,
                ],
                poststorages=[
                    available_channel_storage_m3,
                    available_reservoir_storage_m3,
                    available_groundwater_m3,
                ],
                tolerance=10000,
            )
        if self.model.timing:
            self.model.logger.debug(timer)

        self.var.return_flow_m3_agents = np.bincount(
            self.HRU.var.land_owners[self.HRU.var.land_owners != -1],
            weights=return_flow_irrigation_m[self.HRU.var.land_owners != -1]
            * self.HRU.var.cell_area[self.HRU.var.land_owners != -1],
        )

        self.report(locals())

        return (
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow,  # from all sources, re-added in routing
            irrigation_loss_to_evaporation_m,
            total_water_demand_loss_m3,
            actual_irrigation_consumption,
        )
