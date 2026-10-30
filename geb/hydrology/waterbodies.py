"""Lakes and Reservoirs Module."""

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
import pandas as pd

from geb.geb_types import Array, ArrayBool, ArrayFloat32, ArrayFloat64, ArrayInt32
from geb.module import Module
from geb.store import Bucket
from geb.workflows import balance_check
from geb.workflows.io import read_geom

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology

OFF: int = 0
LAKE: int = 1
RESERVOIR: int = 2


def get_lake_height_from_bottom(
    lake_storage: ArrayFloat64, lake_area: ArrayFloat32
) -> ArrayFloat32:
    """Calculate the height of a lake above the bottom given its storage and area.

    Assumes a box-shaped lake. Could be extended in the future to account for different lake shapes.

    Args:
        lake_storage: Storage of the lake in m3
        lake_area: Area of the lake in m2

    Returns:
        Height of the lake above the bottom in m
    """
    height_from_bottom = (lake_storage / lake_area).astype(np.float32)
    return height_from_bottom


def get_lake_storage_from_height_above_bottom(
    lake_height: ArrayFloat32, lake_area: ArrayFloat32
) -> ArrayFloat64:
    """Calculate the storage of a lake given its height above the bottom and area.

    Args:
        lake_height: Height of the lake above the bottom in m
        lake_area: Area of the lake in m2

    Returns:
        Storage of the lake in m3
    """
    return (lake_height * lake_area).astype(np.float64)


def get_lake_height_above_outflow(
    lake_storage: ArrayFloat64,
    lake_area: ArrayFloat32,
    outflow_height: ArrayFloat32,
) -> ArrayFloat32:
    """Calculate the height of a lake above the outflow given its storage, area, and outflow height.

    Assumes a box-shaped lake. Could be extended in the future to account for different lake shapes.
    First calculates the height above the bottom, then subtracts the outflow height.

    Args:
        lake_storage: Storage of the lake in m3
        lake_area: Area of the lake in m2
        outflow_height: Height of the outflow in m above the bottom of the lake in m (assuming a rectangular lake)

    Returns:
        Height of the lake above the outflow in m
    """
    height_above_outflow = (
        get_lake_height_from_bottom(lake_storage, lake_area) - outflow_height
    )
    height_above_outflow[height_above_outflow < 0] = 0
    return height_above_outflow


class WaterBodyVariables(Bucket):
    """Variables for the Lakes and Reservoirs module."""

    waterbodies: gpd.GeoDataFrame
    storage: ArrayFloat64
    lake_area: ArrayFloat32
    waterbody_mapping: ArrayInt32
    average_discharge_m3_per_s: ArrayFloat32
    capacity: ArrayFloat64
    waterbody_type: ArrayInt32
    outflow_height: ArrayFloat32
    waterbody_outflow_linear_mapping: ArrayInt32
    construction_year: ArrayInt32
    all_waterbody_ids: ArrayInt32
    active: ArrayBool


class WaterBodies(Module):
    """Implements all lakes and reservoir operations in the hydrological model.

    For reservoir it gets the outflow from the reservoir operator agents.

    Args:
        model: The GEB model instance.
        hydrology: The hydrology submodel instance.
    """

    var: WaterBodyVariables

    def __init__(self, model: GEBModel, hydrology: Hydrology) -> None:
        """Initializes the Lakes and Reservoirs module.

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

        Returns:
            The name of the module.
        """
        return "hydrology.waterbodies"

    def spinup(self) -> None:
        """Load waterbody parameters and initialize storage for spinup.

        Map source IDs to consecutive model IDs, locate outlets, and estimate
        lake outflow parameters. Start built reservoirs at half capacity and
        lakes at their outlet level. Future reservoirs start empty, with their
        cells treated as rivers until their construction year.

        Raises:
            ValueError: If saved waterbody types are invalid or still use the
                HydroLAKES controlled-lake code; rebuild setup_waterbodies.
        """
        # load lakes/reservoirs map with a single ID for each lake/reservoir
        waterbody_id_unmapped: np.ndarray = self.grid.load2d(
            self.model.files["grid"]["waterbodies/waterbody_id"]
        )

        waterbody_data_raw: gpd.GeoDataFrame = read_geom(
            self.model.files["geom"]["waterbodies/waterbody_data"],
        ).set_index("waterbody_id")  # ty:ignore[invalid-assignment]

        if not waterbody_data_raw["waterbody_type"].isin((OFF, LAKE, RESERVOIR)).all():
            raise ValueError(
                "Waterbody types must be off, lake, or reservoir. Rebuild setup_waterbodies."
            )

        # Identify active waterbodies (filter out OFF waterbodies during loading)
        active_wb_mask = waterbody_data_raw["waterbody_type"] != OFF
        active_waterbody_ids = waterbody_data_raw.index[active_wb_mask].values

        # Remove OFF waterbody cells from the unmapped raster
        is_active_cell = np.isin(waterbody_id_unmapped, active_waterbody_ids)
        waterbody_id_unmapped[~is_active_cell] = -1

        waterbody_outflow_points_original_ids = self.grid.load2d(
            self.model.files["grid"]["waterbodies/waterbody_outflow_points"]
        )
        is_active_outflow = np.isin(
            waterbody_outflow_points_original_ids, active_waterbody_ids
        )
        waterbody_outflow_points_original_ids[~is_active_outflow] = -1

        order_of_waterbodies_in_grid = waterbody_outflow_points_original_ids[
            waterbody_outflow_points_original_ids != -1
        ]

        self.grid.var.waterbody_ids, self.var.waterbody_mapping = (
            self.map_waterbody_ids(waterbody_id_unmapped, order_of_waterbodies_in_grid)
        )

        self.grid.var.waterbody_outflow_points = self.var.waterbody_mapping[
            waterbody_outflow_points_original_ids
        ]

        # set discharge to NaN for all cells that are part of an active water body
        self.hydrology.routing.var.discharge_in_rivers_m3_s_substep[
            self.grid.var.waterbody_ids != -1
        ] = np.nan

        self.var.waterbody_outflow_linear_mapping = np.where(
            self.grid.var.waterbody_outflow_points != -1
        )[0].astype(np.int32)

        waterbody_data: gpd.GeoDataFrame = waterbody_data_raw.loc[
            order_of_waterbodies_in_grid
        ]
        bankfull_river_elev: ArrayFloat32 = self.grid.load2d(
            self.model.files["grid"]["routing/bankfull_river_elevation_m"]
        )
        waterbody_data["elevation"] = bankfull_river_elev[
            self.var.waterbody_outflow_linear_mapping
        ].astype(np.float32)
        self.var.waterbodies = waterbody_data
        # Reservoirs with unknown years stay active.
        self.var.construction_year = (
            waterbody_data.get(
                "gdw_construction_year", pd.Series(0, index=waterbody_data.index)
            )
            .fillna(0)
            .to_numpy(dtype=np.int32)
        )
        self.var.all_waterbody_ids = self.grid.var.waterbody_ids.copy()

        self.var.waterbody_type = waterbody_data["waterbody_type"].to_numpy(
            dtype=np.int32, copy=True
        )

        # Natural lakes stay active regardless of the dam year.
        self.var.construction_year[self.var.waterbody_type != RESERVOIR] = 0
        self.var.active = np.ones(self.n, dtype=bool)
        dated_reservoirs: ArrayBool = self.var.construction_year > 0
        if dated_reservoirs.any():
            self.var.active[dated_reservoirs] = (
                self.var.construction_year[dated_reservoirs]
                <= self.model.current_time.year
            )

        assert (np.isin(self.var.waterbody_type, [LAKE, RESERVOIR])).all()

        self.var.lake_area = waterbody_data["average_area"].values.copy()
        self.var.capacity = waterbody_data["volume_total"].values.copy()
        self.var.average_discharge_m3_per_s = waterbody_data[
            "average_discharge"
        ].values.copy()

        self.grid.var.capacity = self.map_to_grid_outflow(
            self.var.capacity, fill_value=0
        )

        self.var.storage = np.zeros_like(self.var.capacity, dtype=np.float64)

        # initialize reservoir storage to 50% of the capacity. This is arbitrary, but
        # ok since we use a spinup period
        self.reservoir_storage = self.reservoir_capacity * 0.5
        self.var.outflow_height = np.zeros_like(self.var.capacity, dtype=np.float32)

        # Before construction, keep storage empty and route water as a river.
        self.var.storage[~self.is_active] = 0
        self.set_active_waterbody_maps()
        self.hydrology.routing.var.discharge_in_rivers_m3_s_substep[
            (self.var.all_waterbody_ids != -1) & (self.grid.var.waterbody_ids == -1)
        ] = 0

    @property
    def is_active(self) -> ArrayBool:
        """Return which waterbodies are active this year.

        Returns:
            True for lakes and built reservoirs, including those with unknown years.
        """
        return self.var.active

    def set_active_waterbody_maps(self) -> None:
        """Update waterbody maps for the reservoirs operating this year.

        Future reservoirs use river routing; waterbody IDs stay unchanged.
        """
        # River cells have ID -1 and use the last (False) entry.
        active_cells: ArrayBool = np.append(self.is_active, False)[
            self.var.all_waterbody_ids
        ]
        self.grid.var.waterbody_ids = np.where(
            active_cells, self.var.all_waterbody_ids, -1
        )
        self.grid.var.waterbody_outflow_points = self.map_to_grid_outflow(
            np.arange(self.n, dtype=np.int32), fill_value=-1
        )
        self.grid.var.capacity = self.map_to_grid_outflow(
            self.var.capacity, fill_value=0
        )

    def map_waterbody_ids(
        self, waterbody_id_unmapped: ArrayInt32, order_of_waterbodies: ArrayInt32
    ) -> tuple[ArrayInt32, ArrayInt32]:
        """Maps the water body IDs to a continuous range of IDs starting from 0.

        If there are no water bodies, it returns an array of -1.

        Args:
            waterbody_id_unmapped: The original water body IDs from the grid.
            order_of_waterbodies: The order in which the water bodies should be processed.

        Returns:
            A tuple containing:
                - waterbody_id_mapped: The mapped water body IDs.
                - waterbody_mapping: The mapping from original IDs to mapped IDs.
        """
        if order_of_waterbodies.size == 0:
            return np.full_like(waterbody_id_unmapped, -1), np.full(
                1, -1, dtype=np.int32
            )
        else:
            waterbody_mapping = np.full(
                order_of_waterbodies.max() + 2, -1, dtype=np.int32
            )  # make sure that the last entry is also -1, so that -1 maps to -1
            waterbody_mapping[order_of_waterbodies] = np.arange(
                0, order_of_waterbodies.size, dtype=np.int32
            )
            return waterbody_mapping[waterbody_id_unmapped], waterbody_mapping

    def load_waterbody_data(
        self,
        order_of_waterbodies_in_grid: ArrayInt32,
    ) -> gpd.GeoDataFrame:
        """Loads water body data from disk and sorts it based on the order of water bodies in the grid.

        Args:
            order_of_waterbodies_in_grid: The order of water bodies in the grid. Original ids.

        Returns:
            A GeoDataFrame containing the water body data with the index set to the mapped water body IDs.
        """
        waterbody_data: gpd.GeoDataFrame = read_geom(
            self.model.files["geom"]["waterbodies/waterbody_data"],
        ).set_index("waterbody_id")  # ty:ignore[invalid-assignment]

        return waterbody_data.loc[order_of_waterbodies_in_grid]

    def routing_lakes(self, routing_step_length_seconds: int | float) -> ArrayFloat32:
        """Lake routine to calculate lake outflow.

        Lake outflow is calculated dynamically within the local inertial routing substeps.
        Returns NaN values as a flag telling the routing module to calculate dynamic outflow
        at each substep using the weir rating curve.

        Args:
            routing_step_length_seconds: Length of the routing step in seconds.

        Returns:
            lake_outflow_m3: Array of NaN values with length equal to the number of lakes.
        """
        return np.full(self.is_lake.sum(), np.nan, dtype=np.float32)

    def routing_reservoirs(
        self, n_routing_substeps: int, current_substep: int
    ) -> tuple[ArrayFloat32, ArrayFloat32]:
        """Routine to update reservoir volumes and calculate reservoir outflow.

        Args:
            n_routing_substeps: Number of routing substeps per time step
            current_substep: Current substep in the routing process

        Returns:
            main_channel_release_m3: Outflow from the main channel in m3 per routing step.
            command_area_release_m3: Outflow from the command area in m3 per routing step
        """
        main_channel_release_m3, command_area_release_m3 = (
            self.model.agents.reservoir_operators.release(
                daily_substeps=n_routing_substeps,
                current_substep=current_substep,
            )
        )

        assert (main_channel_release_m3 <= self.reservoir_storage).all()
        assert (self.reservoir_storage >= 0).all()

        return main_channel_release_m3, command_area_release_m3

    def substep(
        self,
        current_substep: int,
        n_routing_substeps: int,
        routing_step_length_seconds: int,
    ) -> tuple[ArrayFloat32, ArrayFloat32]:
        """Routes lakes and reservoirs for a single routing substep.

        Importantly, the outflow is not removed from the storage here. This is
        done in the hydrology routing module.

        For lakes, it calculates the outflow based on the current storage and lake parameters.
        This is returned as outflow to the drainage network.

        For reservoirs, it gets the outflow and the command area release from the reservoir operators.
        The outflow is returned as outflow to the drainage network, while the command area release
        is returned separately.

        The command area release is not routed through the drainage network, but is instead
        added directly to the irrigation demand in the irrigation module.

        Args:
            current_substep: Current substep in the routing process
            n_routing_substeps: Number of routing substeps per time step
            routing_step_length_seconds: length of the routing step in seconds.

        Returns:
            tuple containing:
                - outflow_to_drainage_network_m3: Outflow to the drainage network in m3 per routing step.
                - command_area_release_m3: Outflow from the command area in m3 per routing step.
        """
        if __debug__:
            prestorage = self.var.storage.copy()

        outflow_to_drainage_network_m3 = np.zeros_like(
            self.var.storage, dtype=np.float32
        )
        command_area_release_m3 = np.zeros_like(self.var.storage, dtype=np.float32)

        outflow_to_drainage_network_m3[self.is_lake] = self.routing_lakes(
            routing_step_length_seconds
        )

        (
            outflow_to_drainage_network_m3[self.is_reservoir],
            command_area_release_m3[self.is_reservoir],
        ) = self.routing_reservoirs(n_routing_substeps, current_substep)

        valid_outflows = ~np.isnan(outflow_to_drainage_network_m3)
        if valid_outflows.any():
            assert (
                outflow_to_drainage_network_m3[valid_outflows]
                <= self.var.storage[valid_outflows].astype(np.float32)
            ).all(), (
                f"Outflow exceeds storage: {outflow_to_drainage_network_m3[valid_outflows].max()} > {self.var.storage[valid_outflows].max()}"
            )

        if __debug__:
            balance_check(
                name="command_area_release",
                how="cellwise",
                influxes=[],
                outfluxes=[],
                prestorages=[prestorage[self.is_reservoir]],
                poststorages=[self.var.storage[self.is_reservoir]],
                tolerance=1,  # 1 m3
            )

        return outflow_to_drainage_network_m3, command_area_release_m3

    @property
    def is_reservoir(self) -> ArrayBool:
        """Returns a boolean array indicating which water bodies are reservoirs.

        Returns:
            A boolean array where True indicates that the corresponding water body is a reservoir.
        """
        return self.var.waterbody_type == RESERVOIR

    @property
    def is_lake(self) -> ArrayBool:
        """Returns a boolean array indicating which water bodies are lakes.

        Returns:
            A boolean array where True indicates that the corresponding water body is a lake.
        """
        return self.var.waterbody_type == LAKE

    @property
    def reservoir_storage(self) -> ArrayFloat64:
        """Gets the storage of each reservoir in the model.

        Returns:
            The storage of each reservoir in the model. [m3]
        """
        return self.var.storage[self.is_reservoir]

    @reservoir_storage.setter
    def reservoir_storage(self, value: ArrayFloat64) -> None:
        """Sets the storage of each reservoir in the model."""
        self.var.storage[self.is_reservoir] = value

    @property
    def reservoir_capacity(self) -> ArrayFloat64:
        """Gets the capacity of each reservoir in the model.

        Returns:
            The capacity of each reservoir in the model. [m3]
        """
        return self.var.capacity[self.is_reservoir]

    @reservoir_capacity.setter
    def reservoir_capacity(self, value: ArrayFloat64) -> None:
        """Sets the capacity of each reservoir in the model.

        Args:
            value: The capacity of each reservoir in the model. [m3]
        """
        self.var.capacity[self.is_reservoir] = value

    @property
    def lake_storage(self) -> ArrayFloat64:
        """Gets the storage of each lake in the model.

        Returns:
            The storage of each lake in the model. [m3]
        """
        return self.var.storage[self.is_lake]

    @lake_storage.setter
    def lake_storage(self, value: ArrayFloat64) -> None:
        """Sets the storage of each lake in the model."""
        self.var.storage[self.is_lake] = value

    @property
    def lake_capacity(self) -> ArrayFloat64:
        """Gets the capacity of each lake in the model.

        Returns:
            The capacity of each lake in the model. [m3]
        """
        return self.var.capacity[self.is_lake]

    @lake_capacity.setter
    def lake_capacity(self, value: ArrayFloat64) -> None:
        """Sets the capacity of each lake in the model."""
        self.var.capacity[self.is_lake] = value

    @property
    def reservoir_fill_percentage(self) -> ArrayFloat32:
        """Returns the fill percentage of each reservoir in the model.

        Returns:
            The fill percentage of each reservoir in the model.
        """
        return (self.reservoir_storage / self.reservoir_capacity * 100).astype(
            np.float32
        )

    @property
    def water_depth_from_bottom(self) -> ArrayFloat32:
        """Returns the water depth of each waterbody above its bottom.

        Returns:
            Water depth in meters [m].
        """
        depth: ArrayFloat32 = (
            self.var.storage / np.maximum(self.var.lake_area, np.float32(1.0))
        ).astype(np.float32)
        return np.maximum(depth, np.float32(0.0))

    def flatten_waterbody_elevations(
        self, bankfull_river_elevation: ArrayFloat32
    ) -> ArrayFloat32:
        """Flattens channel bed elevations within each waterbody to its outflow elevation.

        Ensures that all cells within a lake/reservoir share a uniform horizontal bed datum
        matching the waterbody outflow outlet elevation, guaranteeing that all upstream inlet
        reaches have non-negative bed drops into the waterbody and flow remains strictly downward.

        Args:
            bankfull_river_elevation: Array of channel bed elevations across the compressed grid [m].

        Returns:
            Bed elevation array with waterbody cells flattened to outflow bed elevations [m].
        """
        flattened_elevation: ArrayFloat32 = bankfull_river_elevation.copy()
        if self.n > 0:
            for mapped_wb_id in range(self.n):
                outflow_idx: int = int(
                    self.var.waterbody_outflow_linear_mapping[mapped_wb_id]
                )
                outflow_elevation: np.float32 = bankfull_river_elevation[outflow_idx]
                wb_cells: ArrayInt32 = np.where(
                    self.grid.var.waterbody_ids == mapped_wb_id
                )[0].astype(np.int32)
                if wb_cells.size > 0:
                    flattened_elevation[wb_cells] = outflow_elevation
        return flattened_elevation

    def get_water_stage_m(self, outflow_bed_elevation: ArrayFloat32) -> ArrayFloat32:
        """Calculate the water surface elevation of each waterbody in meters above datum.

        Args:
            outflow_bed_elevation: Channel bed elevation at the outflow point of each waterbody [m].

        Returns:
            Water surface elevation of each waterbody in meters above datum [m].
        """
        depth_above_sill: ArrayFloat32 = (
            self.water_depth_from_bottom - self.var.outflow_height
        )
        water_stage: ArrayFloat32 = outflow_bed_elevation + depth_above_sill
        bottom_elevation: ArrayFloat32 = outflow_bed_elevation - self.var.outflow_height
        return np.maximum(water_stage, bottom_elevation).astype(np.float32)

    @property
    def n(self) -> int:
        """Returns the number of lakes and reservoirs in the model.

        Returns:
            The number of lakes and reservoirs in the model.
        """
        return self.var.waterbody_outflow_linear_mapping.size

    def map_to_grid_outflow(
        self,
        values_at_outflow_points: Array,
        fill_value: float | int = np.nan,
        out: Array | None = None,
    ) -> Array:
        """Map values at active waterbody outlets to the full grid.

        Args:
            values_at_outflow_points: One value per waterbody, including inactive ones.
            fill_value: Value for cells without an active outlet when out is None.
            out: Grid array to update, or None to create one.

        Returns:
            The grid array with active outlets filled. Other cells keep their
            previous values when out is supplied, or use fill_value otherwise.
        """
        if out is None:
            out: Array = np.full(
                self.grid.compressed_size,
                fill_value,
                dtype=values_at_outflow_points.dtype,
            )
        out[self.var.waterbody_outflow_linear_mapping[self.is_active]] = (
            values_at_outflow_points[self.is_active]
        )
        return out

    def step(self) -> None:
        """Activate reservoirs from January 1 of their construction year.

        Keep existing water storage when reservoirs start operating.
        """
        active: ArrayBool = self.var.construction_year <= self.model.current_time.year
        if not np.array_equal(active, self.is_active):
            river_storage_m3: ArrayFloat64 = self.hydrology.routing.var.river_storage_m3
            self.var.active = active
            previous_waterbody_ids: ArrayInt32 = self.grid.var.waterbody_ids
            self.set_active_waterbody_maps()
            new_reservoir_cells: ArrayBool = (previous_waterbody_ids == -1) & (
                self.grid.var.waterbody_ids != -1
            )
            # Keep the river water when the reservoir starts operating.
            self.var.storage += np.bincount(
                self.grid.var.waterbody_ids[new_reservoir_cells],
                weights=river_storage_m3[new_reservoir_cells],
                minlength=self.n,
            )
            self.hydrology.routing.var.discharge_in_rivers_m3_s_substep[
                new_reservoir_cells
            ] = np.nan
            river_storage_m3[new_reservoir_cells] = 0
            # The solver caches river connections and waterbody boundaries.
            self.hydrology.routing.update_river_cells()
            self.hydrology.routing.set_router(initialize_storage=False)

        self.report(locals())
