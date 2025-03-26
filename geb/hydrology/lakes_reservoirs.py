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


import math

import numpy as np
import pandas as pd

from geb.HRUs import load_grid
from geb.workflows import balance_check

from .routing.subroutines import (
    PIT,
    define_river_network,
    subcatchment1,
    upstream1,
)

OFF = 0
LAKE = 1
RESERVOIR = 2
LAKE_CONTROL = 3  # currently modelled as normal lake


def laketotal(values, areaclass, nan_class):
    mask = areaclass != nan_class
    class_totals = np.bincount(areaclass[mask], weights=values[mask])
    return class_totals.astype(values.dtype)


GRAVITY = 9.81
SHAPE = "parabola"
# http://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf

if SHAPE == "rectangular":
    overflow_coefficient_mu = 0.577

    def estimate_lake_outflow(lake_factor, height_above_outflow):
        return lake_factor * height_above_outflow**1.5

    def outflow_to_height_above_outflow(lake_factor, outflow):
        """inverse function of estimate_lake_outflow"""
        return (outflow / lake_factor) ** (2 / 3)

elif SHAPE == "parabola":
    overflow_coefficient_mu = 0.612

    def estimate_lake_outflow(lake_factor, height_above_outflow):
        return lake_factor * height_above_outflow**2

    def outflow_to_height_above_outflow(lake_factor, outflow):
        """inverse function of estimate_lake_outflow"""
        return np.sqrt(outflow / lake_factor)

else:
    raise ValueError("Invalid shape")


def get_lake_height_above_outflow(storage, lake_area, outflow_height):
    height_above_outflow = storage / lake_area - outflow_height
    height_above_outflow[height_above_outflow < 0] = 0
    return height_above_outflow


def get_river_width(average_discharge):
    return 7.1 * np.power(average_discharge, 0.539)


def get_lake_factor(river_width, overflow_coefficient_mu, lake_a_factor):
    return (
        lake_a_factor
        * overflow_coefficient_mu
        * (2 / 3)
        * river_width
        * (2 * GRAVITY) ** 0.5
    )


def estimate_outflow_height(lake_storage, lake_factor, lake_area, avg_outflow):
    height_above_outflow = outflow_to_height_above_outflow(lake_factor, avg_outflow)
    outflow_height = (lake_storage / lake_area) - height_above_outflow
    outflow_height[outflow_height < 0] = 0
    return outflow_height


def get_lake_outflow_and_storage(
    dt,
    storage,
    inflow_m3,
    lake_factor,
    lake_area,
    outflow_height,
):
    """
    Calculate outflow and storage for a lake using the Modified Puls method

    Parameters
    ----------
    dt : float
        Time step in seconds
    storage : float
        Current storage in m3
    inflow : float
        Inflow to the lake in m3/s
    inflow_prev : float
        Inflow to the lake in the previous time step in m3/s
    outflow_prev : float
        Outflow from the lake in the previous time step in m3/s
    lake_factor : float
        Factor for the Modified Puls approach to calculate retention of the lake
    lake_area : float
        Area of the lake in m2
    outflow_height : float
        Height of the outflow in m above the bottom of the lake in m (assuming a rectangular lake)

    Returns
    -------
    outflow : float
        New outflow from the lake in m3/s
    storage : float
        New storage in m3

    """
    storage += inflow_m3

    height_above_outflow = get_lake_height_above_outflow(
        storage=storage, lake_area=lake_area, outflow_height=outflow_height
    )
    storage_above_outflow = height_above_outflow * lake_area

    outflow_m3_s = estimate_lake_outflow(lake_factor, height_above_outflow)
    outflow_m3 = outflow_m3_s * dt

    outflow_m3 = np.minimum(outflow_m3, storage_above_outflow)

    new_storage = storage - outflow_m3
    # this is required to avoid negative storage due to numerical errors
    new_storage[new_storage < 0] = 0

    return outflow_m3, new_storage, height_above_outflow


class LakesReservoirs(object):
    def __init__(self, model, hydrology):
        self.model = model
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    def spinup(self):
        self.var = self.model.store.create_bucket("hydrology.lakes_reservoirs.var")

        # load lakes/reservoirs map with a single ID for each lake/reservoir
        waterBodyID_unmapped = self.grid.load(
            self.model.files["grid"]["waterbodies/water_body_id"]
        )
        waterBodyID_unmapped[waterBodyID_unmapped == OFF] = -1

        waterbody_outflow_points = self.get_outflows(waterBodyID_unmapped)

        # dismiss water bodies that are not a subcatchment of an outlet
        # after this, this is the final set of water bodies
        sub = subcatchment1(
            self.grid.var.dirUp,
            waterbody_outflow_points,
            self.grid.var.upstream_area_n_cells,
        )
        waterBodyID_unmapped[waterBodyID_unmapped != sub] = -1

        # we need to re-calculate the outflows, because the ID might have changed due
        # to the earlier dismissal of water bodies
        waterbody_outflow_points = self.get_outflows(waterBodyID_unmapped)

        self.grid.var.waterBodyID, self.var.waterbody_mapping = (
            self.map_water_bodies_IDs(waterBodyID_unmapped)
        )

        # we need to re-calculate the outflows, because the ID might have changed due
        # to the earlier operations. This is the final one as IDs have now been mapped
        self.grid.var.waterbody_outflow_points = self.get_outflows(
            self.grid.var.waterBodyID
        )

        # we compress the waterbody_outflow_points, which we can later use to decompress
        self.var.waterBodyIDC = np.unique(
            self.grid.var.waterbody_outflow_points[
                self.grid.var.waterbody_outflow_points != -1
            ]
        )

        self.var.water_body_data = self.load_water_body_data(
            self.var.waterbody_mapping, waterBodyID_unmapped
        )
        # sort the water bodies in the same order as the compressed water body IDs (waterBodyIDC)
        self.var.water_body_data = self.var.water_body_data.sort_index()

        assert np.array_equal(self.var.water_body_data.index, self.var.waterBodyIDC)

        # change ldd: put pits in where lakes are:
        ldd_LR = self.hydrology.grid.decompress(
            np.where(self.grid.var.waterBodyID != -1, 5, self.grid.var.lddCompress),
            fillvalue=0,
        )

        # set new ldds without lakes reservoirs
        (
            self.grid.var.lddCompress_LR,
            _,
            self.grid.var.dirUp,
            self.grid.var.dirupLen,
            self.grid.var.dirupID,
            self.grid.var.downstruct,
            _,
            self.grid.var.dirDown,
            self.grid.var.lendirDown,
        ) = define_river_network(
            ldd_LR,
            self.hydrology.grid,
        )

        self.var.water_body_type = self.var.water_body_data["waterbody_type"].values
        self.var.waterBodyOrigID = self.var.water_body_data[
            "original_waterbody_id"
        ].values
        # change water body type to LAKE if it is a control lake, thus currently modelled as normal lake
        self.var.water_body_type[self.var.water_body_type == LAKE_CONTROL] = LAKE

        assert (np.isin(self.var.water_body_type, [OFF, LAKE, RESERVOIR])).all()

        self.var.lake_area = self.var.water_body_data["average_area"].values
        # a factor which increases evaporation from lake because of wind TODO: use wind to set this factor
        self.var.lakeEvaFactor = self.model.config["parameters"]["lakeEvaFactor"]

        self.var.capacity = self.var.water_body_data["volume_total"].values

        self.var.total_inflow_from_other_water_bodies = np.zeros_like(
            self.var.capacity, dtype=np.float32
        )

        # lake discharge at outlet to calculate alpha: parameter of channel width, gravity and weir coefficient
        # Lake parameter A (suggested  value equal to outflow width in [m])
        average_discharge = np.maximum(
            self.var.water_body_data["average_discharge"].values,
            0.1,
        )

        # channel width in [m]
        river_width = get_river_width(average_discharge)

        self.var.lake_factor = get_lake_factor(
            river_width,
            overflow_coefficient_mu,
            self.model.config["parameters"]["lakeAFactor"],
        )

        # initialize storage with (full) capacity
        self.var.storage = self.var.capacity.copy()
        self.var.outflow_height = estimate_outflow_height(
            self.var.storage,
            self.var.lake_factor,
            self.var.lake_area,
            average_discharge,
        )

    def map_water_bodies_IDs(self, waterBodyID_unmapped):
        unique_water_bodies = np.unique(waterBodyID_unmapped)
        unique_water_bodies = unique_water_bodies[unique_water_bodies != -1]
        if unique_water_bodies.size == 0:
            return np.full_like(waterBodyID_unmapped, -1), np.full(
                1, -1, dtype=np.int32
            )
        else:
            water_body_mapping = np.full(
                unique_water_bodies.max() + 2, -1, dtype=np.int32
            )  # make sure that the last entry is also -1, so that -1 maps to -1
            water_body_mapping[unique_water_bodies] = np.arange(
                0, unique_water_bodies.size, dtype=np.int32
            )
            return water_body_mapping[waterBodyID_unmapped], water_body_mapping

    def load_water_body_data(self, waterbody_mapping, waterbody_original_ids):
        water_body_data = pd.read_parquet(
            self.model.files["table"]["waterbodies/waterbody_data"],
        )
        # drop all data that is not in the original ids
        waterbody_original_ids_compressed = np.unique(waterbody_original_ids)
        waterbody_original_ids_compressed = waterbody_original_ids_compressed[
            waterbody_original_ids_compressed != -1
        ]
        water_body_data = water_body_data[
            water_body_data["waterbody_id"].isin(waterbody_original_ids_compressed)
        ]
        # map the waterbody ids to the new ids, save old ids
        water_body_data["original_waterbody_id"] = water_body_data["waterbody_id"]
        water_body_data["waterbody_id"] = waterbody_mapping[
            water_body_data["waterbody_id"]
        ]

        water_body_data = water_body_data.set_index("waterbody_id")
        return water_body_data

    def get_outflows(self, waterBodyID):
        # calculate biggest outlet = biggest accumulation of ldd network
        upstream_area_within_waterbodies = np.zeros_like(
            self.grid.var.upstream_area_n_cells, shape=waterBodyID.max() + 2
        )
        upstream_area_within_waterbodies[-1] = -1
        np.maximum.at(
            upstream_area_within_waterbodies,
            waterBodyID[waterBodyID != -1],
            self.grid.var.upstream_area_n_cells[waterBodyID != -1],
        )
        upstream_area_within_waterbodies = np.take(
            upstream_area_within_waterbodies, waterBodyID
        )

        # in some cases the cell with the highest number of upstream cells
        # has mulitple occurences in the same lake, this seems to happen
        # especially for very small lakes with a small drainage area.
        # In such cases, we take the outflow cell with the lowest elevation.
        outflow_elevation = load_grid(
            self.model.files["grid"]["routing/outflow_elevation"]
        )
        outflow_elevation = self.grid.compress(outflow_elevation)

        waterbody_outflow_points = np.where(
            self.grid.var.upstream_area_n_cells == upstream_area_within_waterbodies,
            waterBodyID,
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
                load_grid(self.model.files["grid"]["routing/outflow_elevation"])
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
                np.unique(waterbody_outflow_points), np.unique(waterBodyID)
            )
            # make sure that each outflow point is only used once
            unique_outflow_points = np.unique(
                waterbody_outflow_points[waterbody_outflow_points != -1],
                return_counts=True,
            )[1]
            if unique_outflow_points.size > 0:
                assert unique_outflow_points.max() == 1

        return waterbody_outflow_points

    def routing_lakes(self, inflow_m3, routing_step_length_seconds):
        """
        Lake routine to calculate lake outflow
        :param inflowC: inflow to lakes and reservoirs [m3]
        :param NoRoutingExecuted: actual number of routing substep
        :return: QLakeOutM3DtC - lake outflow in [m3] per subtime step
        """
        if __debug__:
            prestorage = self.var.storage.copy()

        lakes = self.var.water_body_type == LAKE

        lake_outflow_m3 = np.zeros_like(inflow_m3)

        # check if there are any lakes in the model
        if lakes.any():
            (
                lake_outflow_m3[lakes],
                self.var.storage[lakes],
                height_above_outflow,
            ) = get_lake_outflow_and_storage(
                routing_step_length_seconds,
                self.var.storage[lakes],
                inflow_m3[lakes],
                self.var.lake_factor[lakes],
                self.var.lake_area[lakes],
                self.var.outflow_height[lakes],
            )

        assert (self.var.storage >= 0).all()

        if __debug__:
            balance_check(
                influxes=[inflow_m3[lakes]],
                outfluxes=[lake_outflow_m3[lakes]],
                prestorages=[prestorage[lakes]],
                poststorages=[self.var.storage[lakes]],
                name="lake",
                tollerance=0.1,
            )

        return lake_outflow_m3

    def routing_reservoirs(self, inflow_m3, n_routing_steps):
        """
        Reservoir outflow
        :param inflowC: inflow to reservoirs
        :return: qResOutM3DtC - reservoir outflow in [m3] per subtime step
        """
        if __debug__:
            prestorage = self.reservoir_storage.copy()

        reservoirs = self.var.water_body_type == RESERVOIR

        # Reservoir inflow in [m3] per timestep
        # New reservoir storage [m3] = plus inflow for this sub step

        is_command_area = self.HRU.var.reservoir_command_areas != -1

        # print("WARNING: Assuming irrigation demand equal to ETRef")
        irrigation_demand_m3 = (
            np.bincount(
                self.HRU.var.reservoir_command_areas[is_command_area],
                weights=self.HRU.var.ETRef[is_command_area]
                * self.HRU.var.cell_area[is_command_area],
            )
            / n_routing_steps
        )

        reservoir_infow_m3 = inflow_m3[reservoirs]
        reservoir_release_m3 = (
            self.model.agents.reservoir_operators.regulate_reservoir_outflow_hanasaki(
                inflow_m3=reservoir_infow_m3,
                irrigation_demand_m3=irrigation_demand_m3,
                n_routing_steps=n_routing_steps,
            )
        )

        self.reservoir_storage = (
            self.reservoir_storage - reservoir_release_m3 + reservoir_infow_m3
        )
        assert (self.reservoir_storage >= 0).all()

        if __debug__:
            balance_check(
                influxes=[reservoir_infow_m3],  # In [m3/s]
                outfluxes=[reservoir_release_m3],
                prestorages=[prestorage],
                poststorages=[self.reservoir_storage],
                name="reservoirs",
                tollerance=1e-5,
            )

        return reservoir_release_m3

    def substep(
        self,
        substep,
        n_routing_steps,
        routing_step_length_seconds,
        discharge,
        total_runoff,
    ):
        """
        Dynamic part to calculate outflow from lakes and reservoirs
        * lakes with modified Puls approach
        * reservoirs with special filling levels
        :param NoRoutingExecuted: actual number of routing substep
        :return: outLdd: outflow in m3 to the network
        Note:
            outflow to adjected lakes and reservoirs is calculated separately
        """

        if __debug__:
            prestorage = self.var.storage.copy()

        if substep == 0:
            # average evaporation overeach lake
            average_evaporation_per_water_body = np.bincount(
                self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
                weights=self.grid.var.EWRef[self.grid.var.waterBodyID != -1],
            ) / np.bincount(self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1])
            self.evaporation_from_water_bodies_per_routing_step_m3 = (
                average_evaporation_per_water_body
                * self.var.lake_area
                / n_routing_steps
            )
            assert np.all(
                self.evaporation_from_water_bodies_per_routing_step_m3 >= 0.0
            ), "evaporation_from_water_bodies_per_routing_step_m3 < 0.0"

        total_runoff_m3 = total_runoff * self.grid.var.cell_area / n_routing_steps
        total_runoff_m3 = laketotal(
            total_runoff_m3, self.grid.var.waterBodyID, nan_class=-1
        )

        discharge_m3 = (
            upstream1(self.grid.var.downstruct, discharge) * routing_step_length_seconds
        )
        discharge_m3 = laketotal(discharge_m3, self.grid.var.waterBodyID, nan_class=-1)

        assert (total_runoff_m3 >= 0).all()
        assert (discharge_m3 >= 0).all()
        assert (self.var.total_inflow_from_other_water_bodies >= 0).all()

        inflow_m3 = (
            total_runoff_m3
            + discharge_m3
            + self.var.total_inflow_from_other_water_bodies
        )

        actual_evaporation_from_water_bodies_per_routing_step_m3 = np.minimum(
            self.evaporation_from_water_bodies_per_routing_step_m3, self.var.storage
        )  # evaporation is already in m3 per routing substep
        actual_evaporation_from_water_bodies_per_routing_step_m3[
            self.var.water_body_type == OFF
        ] = 0
        self.var.storage -= actual_evaporation_from_water_bodies_per_routing_step_m3

        outflow = self.routing_lakes(inflow_m3, routing_step_length_seconds)
        outflow[self.var.water_body_type == RESERVOIR] = self.routing_reservoirs(
            inflow_m3, n_routing_steps
        )

        if outflow.size > 0:
            outflow_grid = np.take(outflow, self.grid.var.waterbody_outflow_points)
            outflow_grid[self.grid.var.waterbody_outflow_points == -1] = 0
        else:
            outflow_grid = np.zeros_like(
                self.grid.var.waterbody_outflow_points, dtype=outflow.dtype
            )

        # shift outflow 1 cell downstream
        outflow_shifted_downstream = upstream1(
            self.grid.var.downstruct_no_water_bodies, outflow_grid
        )

        # in this assert we exclude any outflow that is already in a pit
        # because this should disappear when the water is shifted downstream
        assert math.isclose(
            outflow_shifted_downstream.sum(),
            outflow_grid[self.grid.var.lddCompress != PIT].sum(),
            rel_tol=0.00001,
        )

        # everything with is not going to another lake is output to river network
        # this variable is named inflow_to_river_network in the routing
        # module, because it is considered inflow there
        outflow_to_river_network = np.where(
            self.grid.var.waterBodyID != -1, 0, outflow_shifted_downstream
        )
        # everything what is not going to the network is going to another lake
        # this will be added to the inflow of the other lake in the next
        # timestep
        outflow_to_another_lake = np.where(
            self.grid.var.waterBodyID != -1, outflow_shifted_downstream, 0
        )

        # sum up all inflow from other lakes
        self.var.total_inflow_from_other_water_bodies = laketotal(
            outflow_to_another_lake, self.grid.var.waterBodyID, nan_class=-1
        )

        if __debug__:
            balance_check(
                name="lakes and reservoirs",
                how="cellwise",
                influxes=[inflow_m3],
                outfluxes=[
                    outflow,
                    actual_evaporation_from_water_bodies_per_routing_step_m3,
                ],
                prestorages=[prestorage],
                poststorages=[self.var.storage],
                tollerance=1,  # 1 m3
            )

        return (
            outflow_to_river_network,
            actual_evaporation_from_water_bodies_per_routing_step_m3,
        )

    @property
    def reservoir_storage(self):
        return self.var.storage[self.var.water_body_type == RESERVOIR]

    @reservoir_storage.setter
    def reservoir_storage(self, value):
        self.var.storage[self.var.water_body_type == RESERVOIR] = value

    @property
    def reservoir_capacity(self):
        return self.var.capacity[self.var.water_body_type == RESERVOIR]

    @reservoir_capacity.setter
    def reservoir_capacity(self, value):
        self.var.capacity[self.var.water_body_type == RESERVOIR] = value

    @property
    def lake_storage(self):
        return self.var.storage[self.var.water_body_type == LAKE]

    @lake_storage.setter
    def lake_storage(self, value):
        self.var.storage[self.var.water_body_type == LAKE] = value

    @property
    def lake_capacity(self):
        return self.var.capacity[self.var.water_body_type == LAKE]

    @lake_capacity.setter
    def lake_capacity(self, value):
        self.var.capacity[self.var.water_body_type == LAKE] = value

    @property
    def reservoir_fill_percentage(self):
        return self.reservoir_storage / self.reservoir_capacity * 100

    def decompress(self, array):
        return array

    def step(self):
        """
        Dynamic part set lakes and reservoirs for each year
        """
        # if first timestep, or beginning of new year
        if self.model.current_timestep == 1 or (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            if self.hydrology.dynamic_water_bodies:
                raise NotImplementedError("dynamic_water_bodies not implemented yet")

        print(self.reservoir_fill_percentage.astype(int))
