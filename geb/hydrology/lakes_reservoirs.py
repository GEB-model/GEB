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

import pandas as pd
import numpy as np
from geb.workflows import balance_check
from geb.HRUs import load_grid

from .routing.subroutines import (
    subcatchment1,
    define_river_network,
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


def get_channel_width(average_discharge):
    return 7.1 * np.power(average_discharge, 0.539)


def get_lake_factor(channel_width, overflow_coefficient_mu, lake_a_factor):
    return (
        lake_a_factor
        * overflow_coefficient_mu
        * (2 / 3)
        * channel_width
        * (2 * GRAVITY) ** 0.5
    )


def estimate_outflow_height(lake_volume, lake_factor, lake_area, avg_outflow):
    height_above_outflow = outflow_to_height_above_outflow(lake_factor, avg_outflow)
    outflow_height = (lake_volume / lake_area) - height_above_outflow
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
        Current storage volume in m3
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
        New storage volume in m3

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
    def __init__(self, model):
        """
        Initialize water bodies
        """

        self.grid = model.data.grid
        self.model = model

        if self.model.spinup:
            self.spinup()

    def spinup(self):
        self.bucket = self.model.store.create_bucket("lakes_reservoirs")

        # load lakes/reservoirs map with a single ID for each lake/reservoir
        waterBodyID_unmapped = self.grid.load(
            self.model.files["grid"]["routing/lakesreservoirs/lakesResID"]
        )
        waterBodyID_unmapped[waterBodyID_unmapped == OFF] = -1

        waterbody_outflow_points = self.get_outflows(waterBodyID_unmapped)

        # dismiss water bodies that are not a subcatchment of an outlet
        # after this, this is the final set of water bodies
        sub = subcatchment1(
            self.grid.bucket.dirUp,
            waterbody_outflow_points,
            self.grid.bucket.upstream_area_n_cells,
        )
        waterBodyID_unmapped[waterBodyID_unmapped != sub] = -1

        # we need to re-calculate the outflows, because the ID might have changed due
        # to the earlier dismissal of water bodies
        waterbody_outflow_points = self.get_outflows(waterBodyID_unmapped)

        self.grid.bucket.waterBodyID, self.bucket.waterbody_mapping = (
            self.map_water_bodies_IDs(waterBodyID_unmapped)
        )

        # we need to re-calculate the outflows, because the ID might have changed due
        # to the earlier operations. This is the final one as IDs have now been mapped
        self.grid.bucket.waterbody_outflow_points = self.get_outflows(
            self.grid.bucket.waterBodyID
        )

        # we compress the waterbody_outflow_points, which we can later use to decompress
        self.bucket.waterBodyIDC = np.unique(
            self.grid.bucket.waterbody_outflow_points[
                self.grid.bucket.waterbody_outflow_points != -1
            ]
        )

        self.bucket.water_body_data = self.load_water_body_data(
            self.bucket.waterbody_mapping, waterBodyID_unmapped
        )
        # sort the water bodies in the same order as the compressed water body IDs (waterBodyIDC)
        self.bucket.water_body_data = self.bucket.water_body_data.sort_index()

        assert np.array_equal(
            self.bucket.water_body_data.index, self.bucket.waterBodyIDC
        )

        # change ldd: put pits in where lakes are:
        ldd_LR = self.model.data.grid.decompress(
            np.where(
                self.grid.bucket.waterBodyID != -1, 5, self.grid.bucket.lddCompress
            ),
            fillvalue=0,
        )

        # set new ldds without lakes reservoirs
        (
            self.grid.bucket.lddCompress_LR,
            _,
            self.grid.bucket.dirUp,
            self.grid.bucket.dirupLen,
            self.grid.bucket.dirupID,
            self.grid.bucket.downstruct,
            _,
            self.grid.bucket.dirDown,
            self.grid.bucket.lendirDown,
        ) = define_river_network(
            ldd_LR,
            self.model.data.grid,
        )

        self.bucket.waterBodyTypC = self.bucket.water_body_data["waterbody_type"].values
        # change water body type to LAKE if it is a control lake, thus currently modelled as normal lake
        self.bucket.waterBodyTypC[self.bucket.waterBodyTypC == LAKE_CONTROL] = LAKE

        assert (np.isin(self.bucket.waterBodyTypC, [OFF, LAKE, RESERVOIR])).all()

        self.reservoir_operators = self.model.agents.reservoir_operators
        self.reservoir_operators.set_reservoir_data(self.bucket.water_body_data)

        self.bucket.lake_area = self.bucket.water_body_data["average_area"].values
        # a factor which increases evaporation from lake because of wind TODO: use wind to set this factor
        self.bucket.lakeEvaFactor = self.model.config["parameters"]["lakeEvaFactor"]

        self.bucket.volume = self.bucket.water_body_data["volume_total"].values

        self.bucket.total_inflow_from_other_water_bodies = np.zeros_like(
            self.bucket.volume, dtype=np.float32
        )

        # lake discharge at outlet to calculate alpha: parameter of channel width, gravity and weir coefficient
        # Lake parameter A (suggested  value equal to outflow width in [m])
        average_discharge = np.maximum(
            self.bucket.water_body_data["average_discharge"].values,
            0.1,
        )

        # channel width in [m]
        channel_width = get_channel_width(average_discharge)

        self.bucket.lake_factor = get_lake_factor(
            channel_width,
            overflow_coefficient_mu,
            self.model.config["parameters"]["lakeAFactor"],
        )

        self.bucket.storage = self.bucket.volume.copy()
        self.bucket.outflow_height = estimate_outflow_height(
            self.bucket.volume,
            self.bucket.lake_factor,
            self.bucket.lake_area,
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
            self.model.files["table"]["routing/lakesreservoirs/basin_lakes_data"],
        )
        # drop all data that is not in the original ids
        waterbody_original_ids_compressed = np.unique(waterbody_original_ids)
        waterbody_original_ids_compressed = waterbody_original_ids_compressed[
            waterbody_original_ids_compressed != -1
        ]
        water_body_data = water_body_data[
            water_body_data["waterbody_id"].isin(waterbody_original_ids_compressed)
        ]
        # map the waterbody ids to the new ids
        water_body_data["waterbody_id"] = waterbody_mapping[
            water_body_data["waterbody_id"]
        ]
        water_body_data = water_body_data.set_index("waterbody_id")
        return water_body_data

    def get_outflows(self, waterBodyID):
        # calculate biggest outlet = biggest accumulation of ldd network
        upstream_area_within_waterbodies = np.zeros_like(
            self.grid.bucket.upstream_area_n_cells, shape=waterBodyID.max() + 2
        )
        upstream_area_within_waterbodies[-1] = -1
        np.maximum.at(
            upstream_area_within_waterbodies,
            waterBodyID[waterBodyID != -1],
            self.grid.bucket.upstream_area_n_cells[waterBodyID != -1],
        )
        upstream_area_within_waterbodies = np.take(
            upstream_area_within_waterbodies, waterBodyID
        )

        # in some cases the cell with the highest number of upstream cells
        # has mulitple occurences in the same lake, this seems to happen
        # especially for very small lakes with a small drainage area.
        # In such cases, we take the outflow cell with the lowest elevation.
        outflow_elevation = load_grid(
            self.model.files["grid"]["routing/kinematic/outflow_elevation"]
        )
        outflow_elevation = self.grid.compress(outflow_elevation)

        waterbody_outflow_points = np.where(
            self.grid.bucket.upstream_area_n_cells == upstream_area_within_waterbodies,
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
                load_grid(
                    self.model.files["grid"]["routing/kinematic/outflow_elevation"]
                )
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

    def step(self):
        """
        Dynamic part set lakes and reservoirs for each year
        """
        # if first timestep, or beginning of new year
        if self.model.current_timestep == 1 or (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            # - 3 = reservoirs and lakes (used as reservoirs but before the year of construction as lakes
            # - 2 = reservoirs (regulated discharge)
            # - 1 = lakes (weirFormula)
            # - 0 = non lakes or reservoirs (e.g. wetland)
            if self.model.DynamicResAndLakes:
                raise NotImplementedError("DynamicResAndLakes not implemented yet")

    def routing_lakes(self, inflow_m3):
        """
        Lake routine to calculate lake outflow
        :param inflowC: inflow to lakes and reservoirs [m3]
        :param NoRoutingExecuted: actual number of routing substep
        :return: QLakeOutM3DtC - lake outflow in [m3] per subtime step
        """
        if __debug__:
            prestorage = self.bucket.storage.copy()

        lakes = self.bucket.waterBodyTypC == LAKE

        lake_outflow_m3 = np.zeros_like(inflow_m3)

        # check if there are any lakes in the model
        if lakes.any():
            (
                lake_outflow_m3[lakes],
                self.bucket.storage[lakes],
                height_above_outflow,
            ) = get_lake_outflow_and_storage(
                self.model.routing.bucket.dtRouting,
                self.bucket.storage[lakes],
                inflow_m3[lakes],
                self.lake_factor[lakes],
                self.bucket.lake_area[lakes],
                self.bucket.outflow_height[lakes],
            )

        assert (self.bucket.storage >= 0).all()

        if __debug__:
            balance_check(
                influxes=[inflow_m3[lakes]],
                outfluxes=[lake_outflow_m3[lakes]],
                prestorages=[prestorage[lakes]],
                poststorages=[self.bucket.storage[lakes]],
                name="lake",
                tollerance=0.1,
            )

        return lake_outflow_m3

    def routing_reservoirs(self, inflowC):
        """
        Reservoir outflow
        :param inflowC: inflow to reservoirs
        :return: qResOutM3DtC - reservoir outflow in [m3] per subtime step
        """
        if __debug__:
            prestorage = self.bucket.storage.copy()

        reservoirs = self.bucket.waterBodyTypC == RESERVOIR

        # Reservoir inflow in [m3] per timestep
        self.bucket.storage[reservoirs] += inflowC[reservoirs]
        # New reservoir storage [m3] = plus inflow for this sub step

        outflow_m3_s = np.zeros(self.bucket.waterBodyIDC.size, dtype=np.float64)
        outflow_m3_s[reservoirs] = (
            self.model.agents.reservoir_operators.regulate_reservoir_outflow(
                self.bucket.storage[reservoirs],
                inflowC[reservoirs]
                / self.model.routing.bucket.dtRouting,  # convert per timestep to per second
                self.bucket.waterBodyIDC[reservoirs],
            )
        )

        outflow_m3 = outflow_m3_s * self.model.routing.bucket.dtRouting
        assert (outflow_m3 <= self.bucket.storage).all()

        self.bucket.storage -= outflow_m3

        inflow_reservoirs = np.zeros_like(inflowC)
        inflow_reservoirs[reservoirs] = inflowC[reservoirs]
        if __debug__:
            balance_check(
                influxes=[inflow_reservoirs],  # In [m3/s]
                outfluxes=[outflow_m3],
                prestorages=[prestorage],
                poststorages=[self.bucket.storage],
                name="reservoirs",
                tollerance=1e-5,
            )

        return outflow_m3

    def routing(
        self, step, evaporation_from_water_bodies_per_routing_step, discharge, runoff
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
            prestorage = self.model.lakes_reservoirs.bucket.storage.copy()

        runoff_m3 = (
            runoff
            * self.grid.bucket.cellArea
            / self.model.routing.bucket.noRoutingSteps
        )
        runoff_m3 = laketotal(runoff_m3, self.grid.bucket.waterBodyID, nan_class=-1)

        discharge_m3 = (
            upstream1(self.grid.bucket.downstruct, discharge)
            * self.model.routing.bucket.dtRouting
        )
        discharge_m3 = laketotal(
            discharge_m3, self.grid.bucket.waterBodyID, nan_class=-1
        )

        assert (runoff_m3 >= 0).all()
        assert (discharge_m3 >= 0).all()
        assert (self.bucket.total_inflow_from_other_water_bodies >= 0).all()

        inflow_m3 = (
            runoff_m3 + discharge_m3 + self.bucket.total_inflow_from_other_water_bodies
        )

        # lakeEvaFactorC
        evaporation = np.minimum(
            evaporation_from_water_bodies_per_routing_step, self.bucket.storage
        )  # evaporation is already in m3 per routing substep
        evaporation[self.bucket.waterBodyTypC == OFF] = 0
        self.bucket.storage -= evaporation

        outflow_lakes = self.routing_lakes(inflow_m3)
        outflow_reservoirs = self.routing_reservoirs(inflow_m3)

        assert (outflow_reservoirs[outflow_lakes > 0] == 0).all()

        outflow = outflow_lakes + outflow_reservoirs

        if outflow.size > 0:
            outflow_grid = np.take(outflow, self.grid.bucket.waterbody_outflow_points)
            outflow_grid[self.grid.bucket.waterbody_outflow_points == -1] = 0
        else:
            outflow_grid = np.zeros_like(
                self.grid.bucket.waterbody_outflow_points, dtype=outflow.dtype
            )

        # shift outflow 1 cell downstream
        outflow_shifted_downstream = upstream1(
            self.grid.bucket.downstruct_no_water_bodies, outflow_grid
        )
        assert math.isclose(
            outflow_shifted_downstream.sum(), outflow_grid.sum(), rel_tol=0.00001
        )

        # everything with is not going to another lake is output to river network
        outflow_to_river_network = np.where(
            self.grid.bucket.waterBodyID != -1, 0, outflow_shifted_downstream
        )
        # everything what is not going to the network is going to another lake
        # this will be added to the inflow of the other lake in the next
        # timestep
        outflow_to_another_lake = np.where(
            self.grid.bucket.waterBodyID != -1, outflow_shifted_downstream, 0
        )

        # sum up all inflow from other lakes
        self.bucket.total_inflow_from_other_water_bodies = laketotal(
            outflow_to_another_lake, self.grid.bucket.waterBodyID, nan_class=-1
        )

        if __debug__:
            balance_check(
                name="lakes and reservoirs",
                how="cellwise",
                influxes=[inflow_m3],
                outfluxes=[outflow, evaporation],
                prestorages=[prestorage],
                poststorages=[self.bucket.storage],
                tollerance=1,  # 1 m3
            )

        return outflow_to_river_network, evaporation

    @property
    def reservoir_storage(self):
        return self.bucket.storage[self.bucket.waterBodyTypC == RESERVOIR]

    @property
    def lake_storage(self):
        return self.bucket.storage[self.bucket.waterBodyTypC == LAKE]
