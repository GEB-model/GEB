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
from geb.workflows.io import read_geom, read_grid

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology

OFF: int = 0
LAKE: int = 1
RESERVOIR: int = 2
LAKE_CONTROL: int = 3  # currently modelled as normal lake


GRAVITY: np.float32 = np.float32(9.81)
SHAPE: str = "parabola"
# http://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf

if SHAPE == "rectangular":
    overflow_coefficient_mu: np.float32 = np.float32(0.577)

    def estimate_lake_outflow(
        lake_factor: ArrayFloat32,
        height_above_outflow: ArrayFloat32,
    ) -> ArrayFloat32:
        """Estimates the outflow from a lake given its height above the outflow for a rectangular shape.

        References:
            http://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf

        Args:
            lake_factor: A lake-specific constant factor used in the lake outflow equations.
            height_above_outflow: Height of the lake above elevation of the outflow [m]

        Returns:
            Outflow in m3/s
        """
        return lake_factor * height_above_outflow**1.5

    def outflow_to_height_above_outflow(
        lake_factor: ArrayFloat32, outflow: ArrayFloat32
    ) -> ArrayFloat32:
        """Inverse function of estimate_lake_outflow for a rectangular shape.

        Args:
            lake_factor: A lake-specific constant factor used in the lake outflow equations.
            outflow: Outflow in m3/s

        Returns:
            Height of the lake above elevation of the outflow [m]
        """
        return (outflow / lake_factor) ** np.float32((2 / 3))

elif SHAPE == "parabola":
    overflow_coefficient_mu: np.float32 = np.float32(0.612)

    def estimate_lake_outflow(
        lake_factor: ArrayFloat32,
        height_above_outflow: ArrayFloat32,
    ) -> ArrayFloat32:
        """Estimates the outflow from a lake given its height above the outflow for a parabolic shape.

        References:
            http://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf

        Args:
            lake_factor: A lake-specific constant factor used in the lake outflow equations.
            height_above_outflow: Height of the lake above the outflow [m]

        Returns:
            Outflow in m3/s
        """
        return lake_factor * height_above_outflow**2

    def outflow_to_height_above_outflow(
        lake_factor: ArrayFloat32, outflow: ArrayFloat32
    ) -> ArrayFloat32:
        """Inverse function of estimate_lake_outflow for a parabolic shape.

        References:
            http://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf

        Args:
            lake_factor: A lake-specific constant factor used in the lake outflow equations.
            outflow: Outflow in m3/s

        Returns:
            Height of the lake above the outflow [m]
        """
        return np.sqrt(outflow / lake_factor)

else:
    raise ValueError("Invalid shape")


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


def get_river_width(
    average_discharge: ArrayFloat32,
) -> ArrayFloat32:
    """Estimate river width at lake outflow from average discharge using an empirical relationship.

    TODO: Check if this can be improved using river width data from the hydrological model.

    Args:
        average_discharge: Average discharge in m3/s

    Returns:
        Estimated river width in m
    """
    return np.float32(7.1) * np.power(average_discharge, np.float32(0.539))


def get_lake_factor(
    river_width: ArrayFloat32,
    overflow_coefficient_mu: np.float32,
    lake_a_factor: np.float32,
) -> ArrayFloat32:
    """A lake-constant factor that is used in the equations for lake outflow.

    Pre-calculated to save computation time.

    References:
        http://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf

    Args:
        river_width: Width of the river in m
        overflow_coefficient_mu: Overflow coefficient, depends on the shape of the lake outflow
        lake_a_factor: Calibration factor for the lake outflow

    Returns:
        A lake-specific constant factor used in the lake outflow equations.
    """
    return (
        lake_a_factor
        * overflow_coefficient_mu
        * np.float32((2 / 3))
        * river_width
        * np.float32((2 * GRAVITY) ** 0.5)
    )


def estimate_outflow_height(
    lake_capacity: ArrayFloat64,
    lake_factor: ArrayFloat32,
    lake_area: ArrayFloat32,
    avg_outflow: ArrayFloat32,
) -> ArrayFloat32:
    """Estimate the outflow height of a lake given its capacity, lake factor, area, and average outflow.

    Args:
        lake_capacity: Capacity of the lake in m3
        lake_factor: Factor for the Modified Puls approach to calculate retention of the lake
        lake_area: Area of the lake in m2
        avg_outflow: Average outflow in m3/s

    Returns:
        outflow_height: Height of the outflow in m above the bottom of the lake in m (assuming a rectangular lake)

    """
    height_above_outflow = outflow_to_height_above_outflow(lake_factor, avg_outflow)
    lake_height_when_full = get_lake_height_from_bottom(lake_capacity, lake_area)
    outflow_height = lake_height_when_full - height_above_outflow
    outflow_height[outflow_height < 0] = 0
    return outflow_height


def get_lake_outflow(
    dt: float,
    storage: ArrayFloat64,
    lake_factor: ArrayFloat32,
    lake_area: ArrayFloat32,
    outflow_height: ArrayFloat32,
) -> tuple[ArrayFloat32, ArrayFloat32]:
    """Calculate outflow and storage for a lake using the Modified Puls method.

    Args:
        dt: Time step in seconds
        storage: Current storage in m3
        lake_factor: Factor for the Modified Puls approach to calculate retention of the lake
        lake_area: Area of the lake in m2
        outflow_height: Height of the outflow in m above the bottom of the lake in m (assuming a rectangular lake)

    Returns:
        outflow_m3: Outflow in m3.
        height_above_outflow: Height of the lake above the outflow in m.
    """
    height_above_outflow = get_lake_height_above_outflow(
        lake_storage=storage, lake_area=lake_area, outflow_height=outflow_height
    )
    storage_above_outflow = height_above_outflow * lake_area
    storage_above_outflow = np.minimum(storage_above_outflow, storage)

    outflow_m3_s = estimate_lake_outflow(lake_factor, height_above_outflow)
    outflow_m3 = outflow_m3_s * dt

    outflow_m3 = np.minimum(outflow_m3, storage_above_outflow)

    assert (outflow_m3 <= storage).all()
    return outflow_m3, height_above_outflow


class WaterBodyVariables(Bucket):
    """Variables for the Lakes and Reservoirs module."""

    storage: ArrayFloat64
    lake_area: ArrayFloat32
    lake_factor: ArrayFloat32
    waterbody_mapping: ArrayInt32
    waterbody_data: pd.DataFrame
    capacity: ArrayFloat64
    waterbody_type: ArrayInt32
    outflow_height: ArrayFloat32
    waterbody_outflow_linear_mapping: ArrayInt32


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
        self.hydrological_year_start = self.model.config["general"][
            "hydrological_year_start"
        ]
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
        """Spinup part to initialize lakes and reservoirs.

        This function is called during the spinup phase of the model to set up lakes and reservoirs.

        Steps:
        1. Load the water bodies map and create a mapping of water body IDs.
        2. Set discharge to NaN for all cells that are not part of a water body.
        3. Identify outflow points for each water body.
        4. Load water body data from a Parquet file and filter it based on the water bodies present in the grid.
        5. Extract relevant attributes such as water body type, area, capacity, and average discharge.
        6. Calculate river width and lake factor for each water body.
        7. Initialize storage for lakes and reservoirs, setting initial values based on capacity.
        8. Estimate outflow height for each water body.

        """
        # load lakes/reservoirs map with a single ID for each lake/reservoir
        waterbody_id_unmapped: np.ndarray = self.grid.load(
            self.model.files["grid"]["waterbodies/waterbody_id"]
        )
        waterbody_outflow_points_original_ids = self.get_outflows(waterbody_id_unmapped)
        order_of_waterbodies_in_grid = waterbody_outflow_points_original_ids[
            waterbody_outflow_points_original_ids != -1
        ]

        self.grid.var.waterbody_ids, self.var.waterbody_mapping = (
            self.map_waterbody_ids(waterbody_id_unmapped, order_of_waterbodies_in_grid)
        )

        self.grid.var.waterbody_outflow_points = self.var.waterbody_mapping[
            waterbody_outflow_points_original_ids
        ]

        # set discharge to NaN for all cells that are part of a water body
        self.grid.var.discharge_in_rivers_m3_s_substep[
            self.grid.var.waterbody_ids != -1
        ] = np.nan

        self.var.waterbody_outflow_linear_mapping = np.where(
            self.grid.var.waterbody_outflow_points != -1
        )[0].astype(np.int32)

        assert (
            self.map_to_grid_outflow(np.arange(self.n), fill_value=-1)
            == self.grid.var.waterbody_outflow_points
        ).all()

        self.var.waterbody_data = self.load_waterbody_data(order_of_waterbodies_in_grid)

        self.var.waterbody_type = self.var.waterbody_data["waterbody_type"].values
        # change water body type to LAKE if it is a control lake, thus currently modelled as normal lake
        self.var.waterbody_type[self.var.waterbody_type == LAKE_CONTROL] = LAKE

        assert (np.isin(self.var.waterbody_type, [LAKE, RESERVOIR])).all()

        self.var.lake_area = self.var.waterbody_data["average_area"].values
        self.var.capacity = self.var.waterbody_data["volume_total"].values

        self.grid.var.capacity = self.map_to_grid_outflow(
            self.var.capacity, fill_value=0
        )

        # lake discharge at outlet to calculate alpha: parameter of channel width, gravity and weir coefficient
        # Lake parameter A (suggested  value equal to outflow width in [m])
        average_discharge = np.maximum(
            self.var.waterbody_data["average_discharge"].values,
            0.1,
        )

        # channel width in [m]
        river_width = get_river_width(average_discharge)

        self.var.lake_factor = get_lake_factor(
            river_width,
            overflow_coefficient_mu,
            self.model.config["parameters"]["lake_outflow_multiplier"],
        )

        self.var.storage = np.full_like(self.var.capacity, np.nan, dtype=np.float64)

        # initialize storage to 50% of the capacity. This is arbitrary, but
        # ok since we use a spinup period
        self.reservoir_storage = self.reservoir_capacity * 0.5
        self.var.outflow_height = estimate_outflow_height(
            self.var.capacity,
            self.var.lake_factor,
            self.var.lake_area,
            average_discharge,
        )

        # initialize lake storage to exactly the level of the outflow height
        self.lake_storage = get_lake_storage_from_height_above_bottom(
            self.var.outflow_height[self.is_lake], self.var.lake_area[self.is_lake]
        )
        assert (
            get_lake_height_above_outflow(
                self.lake_storage,
                self.var.lake_area[self.is_lake],
                self.var.outflow_height[self.is_lake],
            )
            < 1e-5
        ).all()

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

    def get_outflows(self, waterbody_id: ArrayInt32) -> ArrayInt32:
        """Identifies the outflow points for each water body.

        Finds the cell with the highest upstream area in each water body as the outflow point.
        If there are multiple cells with the same upstream area, the one with the lowest elevation is chosen.

        Args:
            waterbody_id: The mapped water body IDs from the grid.

        Returns:
            An array containing the outflow point for each water body.
        """
        # calculate biggest outlet = biggest accumulation of ldd network
        upstream_area_n_cells = self.hydrology.routing.grid.var.upstream_area_n_cells
        upstream_area_within_waterbodies = np.zeros_like(
            upstream_area_n_cells,
            shape=waterbody_id.max() + 2,
        )
        upstream_area_within_waterbodies[-1] = -1
        np.maximum.at(
            upstream_area_within_waterbodies,
            waterbody_id[waterbody_id != -1],
            upstream_area_n_cells[waterbody_id != -1],
        )
        upstream_area_within_waterbodies = np.take(
            upstream_area_within_waterbodies, waterbody_id
        )

        # in some cases the cell with the highest number of upstream cells
        # has mulitple occurences in the same lake, this seems to happen
        # especially for very small lakes with a small drainage area.
        # In such cases, we take the outflow cell with the lowest elevation.
        outflow_elevation = read_grid(
            self.model.files["grid"]["landsurface/elevation_min_m"]
        )
        outflow_elevation = self.grid.compress(outflow_elevation)

        waterbody_outflow_points = np.where(
            upstream_area_n_cells == upstream_area_within_waterbodies,
            waterbody_id,
            -1,
        )

        number_of_outflow_points_per_waterbody = np.unique(
            waterbody_outflow_points, return_counts=True
        )
        duplicate_outflow_points = number_of_outflow_points_per_waterbody[0][
            number_of_outflow_points_per_waterbody[1] > 1
        ]
        duplicate_outflow_points = duplicate_outflow_points[
            duplicate_outflow_points != -1
        ]

        if duplicate_outflow_points.size > 0:
            # in some cases the cell with the highest number of upstream cells
            # has mulitple occurences in the same lake, this seems to happen
            # especially for very small lakes with a small drainage area.
            # In such cases, we take the outflow cell with the lowest elevation.
            outflow_elevation = self.grid.compress(
                read_grid(self.model.files["grid"]["landsurface/elevation_min_m"])
            )

            for duplicate_outflow_point in duplicate_outflow_points:
                duplicate_outflow_points_indices = np.where(
                    waterbody_outflow_points == duplicate_outflow_point
                )[0]
                minimum_elevation_outflows_idx = np.argmin(
                    outflow_elevation[duplicate_outflow_points_indices]
                )
                non_minimum_elevation_outflows_indices = (
                    duplicate_outflow_points_indices[
                        (
                            np.arange(duplicate_outflow_points_indices.size)
                            != minimum_elevation_outflows_idx
                        )
                    ]
                )
                waterbody_outflow_points[non_minimum_elevation_outflows_indices] = -1

        if __debug__:
            # make sure that each water body has an outflow
            assert np.array_equal(
                np.unique(waterbody_outflow_points), np.unique(waterbody_id)
            )
            # make sure that each outflow point is only used once
            unique_outflow_points = np.unique(
                waterbody_outflow_points[waterbody_outflow_points != -1],
                return_counts=True,
            )[1]
            if unique_outflow_points.size > 0:
                assert unique_outflow_points.max() == 1

        return waterbody_outflow_points

    def routing_lakes(self, routing_step_length_seconds: int | float) -> ArrayFloat32:
        """Lake routine to calculate lake outflow.

        Args:
            routing_step_length_seconds: length of the routing step in seconds.

        Returns:
            lake_outflow_m3: Outflow from the lakes in m3 per routing step.

        """
        is_lake = self.is_lake
        # check if there are any lakes in the model
        if is_lake.any():
            (
                lake_outflow_m3,
                _,
            ) = get_lake_outflow(
                routing_step_length_seconds,
                self.var.storage[is_lake],
                self.var.lake_factor[is_lake],
                self.var.lake_area[is_lake],
                self.var.outflow_height[is_lake],
            )
        else:
            lake_outflow_m3 = np.zeros(0, dtype=np.float32)

        return lake_outflow_m3

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

        assert (
            outflow_to_drainage_network_m3 <= self.var.storage.astype(np.float32)
        ).all(), (
            f"Outflow exceeds storage: {outflow_to_drainage_network_m3.max()} > {self.var.storage.max()}"
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
        """Maps values at the outflow points to the full grid.

        Args:
            values_at_outflow_points: An array containing values at the outflow points of the water bodies.
            fill_value: The value to fill in for grid cells that are not outflow points. Only used if out is None.
            out: An optional array to write the output to. If None, a new array will be created.

        Returns:
            An array of the same shape as the grid, where the values at the outflow points are filled in and the rest is 0.
        """
        if out is None:
            out: Array = np.full(
                self.grid.compressed_size,
                fill_value,
                dtype=values_at_outflow_points.dtype,
            )
        out[self.var.waterbody_outflow_linear_mapping] = values_at_outflow_points
        return out

    def step(self) -> None:
        """Dynamic part set lakes and reservoirs for each year."""
        # if first timestep, or beginning of new year
        if self.model.current_timestep == 1 or (
            self.model.current_time.month == self.hydrological_year_start
            and self.model.current_time.day == 1
        ):
            if self.hydrology.dynamic_waterbodies:
                raise NotImplementedError("dynamic_waterbodies not implemented yet")

        self.report(locals())
