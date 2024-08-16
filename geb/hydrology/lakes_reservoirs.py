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
from scipy.optimize import fsolve
from geb.workflows import balance_check

from .routing.subroutines import (
    subcatchment1,
    define_river_network,
    upstream1,
)

OFF = 0
LAKE = 1
RESERVOIR = 2


def laketotal(values, areaclass, nan_class):
    mask = areaclass != nan_class
    class_totals = np.bincount(areaclass[mask], weights=values[mask])
    return class_totals


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

    def puls_equation(new_storage, SI, lake_factor, lake_area, dt):
        height = new_storage / lake_area
        return new_storage / dt + estimate_lake_outflow(lake_factor, height) / 2 - SI

    def solve_puls_equation(storage_above_outflow, SI, lake_factor, lake_area, dt):
        res = fsolve(
            puls_equation,
            storage_above_outflow,
            args=(
                SI,
                lake_factor,
                lake_area,
                dt,
            ),
            full_output=True,
        )

        assert res[-2] == 1, "The solution did not converge"
        new_storage = res[0]
        return new_storage

elif SHAPE == "parabola":
    overflow_coefficient_mu = 0.612

    def estimate_lake_outflow(lake_factor, height_above_outflow):
        return lake_factor * height_above_outflow**2

    def outflow_to_height_above_outflow(lake_factor, outflow):
        """inverse function of estimate_lake_outflow"""
        return np.sqrt(outflow / lake_factor)

    def puls_equation(SI, lake_factor, lake_area, dt):
        """Solving quadratic equation for the new storage (S2)

        S2/dt + α/2 (S2/A)^2 - SI = 0

        A = lake area
        S2 = new storage
        dt = time step
        α = lake factor (function of channel width, gravity and weir coefficient)

        multiply by 2dtA^2 to get rid of the denominator:

        2 A^2 S2 + α S2^2 dt - 2 SI A^2 dt = 0

        rearrange to a quadratic equation:

        α dt S2^2 + 2 A^2 S2 - 2 SI A^2 dt = 0

        solve quadratic formula:

        S2 = (-b ± sqrt(b^2 - 4ac)) / 2a

        where a = α dt, b = 2 A^2, c = -2 SI A^2 dt

        S2 = (-2 A^2 ± sqrt(4 A^4 + 8 α SI A^2 dt^2)) / 2 α dt

        simplify:

        S2 = -A^2 ± A sqrt(A^2 + 2 α SI dt^2) / α dt

        use positive solution

        S2 = -A^2 + A sqrt(A^2 + 2 α SI dt^2) / α dt

        """
        nominator = -(lake_area**2) + lake_area * np.sqrt(
            lake_area**2 + 2 * lake_factor * dt**2 * SI
        )
        denominator = lake_factor * dt
        new_storage = nominator / denominator
        return new_storage

    def solve_puls_equation(storage_above_outflow, SI, lake_factor, lake_area, dt):
        return puls_equation(SI, lake_factor, lake_area, dt)

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
    assert (outflow_height >= 0).all()
    return outflow_height


def get_lake_outflow_and_storage(
    dt,
    storage,
    inflow,
    inflow_prev,
    outflow_prev,
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
    height_above_outflow = get_lake_height_above_outflow(
        storage=storage, lake_area=lake_area, outflow_height=outflow_height
    )
    storage_above_outflow = height_above_outflow * lake_area
    storage_below_outflow = storage - storage_above_outflow

    SI = (
        (storage_above_outflow / dt + outflow_prev / 2)
        - outflow_prev
        + (inflow_prev + inflow) / 2
    )

    negative_SI = SI <= 0
    positive_SI = SI > 0

    outflow = np.zeros_like(SI)
    new_storage_above_outflow = np.zeros_like(SI)

    if positive_SI.sum() > 0:

        new_storage_above_outflow_positive_SI = solve_puls_equation(
            storage_above_outflow[positive_SI],
            SI[positive_SI],
            lake_factor[positive_SI],
            lake_area[positive_SI],
            dt,
        )
        new_height_above_outflow_positive_SI = (
            new_storage_above_outflow_positive_SI / lake_area[positive_SI]
        )

        outflow_positive_SI = estimate_lake_outflow(
            lake_factor[positive_SI], new_height_above_outflow_positive_SI
        )
        outflow_positive_SI = np.minimum(
            outflow_positive_SI, new_storage_above_outflow_positive_SI / dt
        )

        outflow[positive_SI] = outflow_positive_SI
        new_storage_above_outflow[positive_SI] = new_storage_above_outflow_positive_SI

    if negative_SI.sum() > 0:
        outflow[negative_SI] = storage_above_outflow[negative_SI] / dt
        new_storage_above_outflow[negative_SI] = 0

    new_storage = storage_below_outflow + new_storage_above_outflow

    return outflow, new_storage, height_above_outflow


class LakesReservoirs(object):
    def __init__(self, model):
        """
        Initialize water bodies

        water body types:
        2 = reservoirs (regulated discharge)
        1 = lakes (weirFormula)
        0 = non lakes or reservoirs (e.g. wetland)
        """

        self.var = model.data.grid
        self.model = model

        # load lakes/reservoirs map with a single ID for each lake/reservoir
        waterBodyID = self.var.load(
            self.model.model_structure["grid"]["routing/lakesreservoirs/lakesResID"]
        )
        waterBodyID[waterBodyID == OFF] = -1

        waterbody_outflow_points = self.get_outflows(waterBodyID)

        # dismiss water bodies that are not a subcatchment of an outlet
        # after this, this is the final set of water bodies
        sub = subcatchment1(
            self.var.dirUp,
            waterbody_outflow_points,
            self.var.upstream_area_n_cells,
        )
        self.var.waterBodyID_original = np.where(waterBodyID == sub, sub, -1)

        # we need to re-calculate the outflows, because the ID might have changed due
        # to the earlier operations
        waterbody_outflow_points = self.get_outflows(waterBodyID)

        compressed_waterbody_ids = np.compress(
            waterbody_outflow_points != -1, waterbody_outflow_points
        )

        self.var.waterBodyID, self.waterbody_mapping = self.map_water_bodies_IDs(
            compressed_waterbody_ids, self.var.waterBodyID_original
        )
        self.water_body_data = self.load_water_body_data(
            self.waterbody_mapping, self.var.waterBodyID_original
        )

        # we need to re-calculate the outflows, because the ID might have changed due
        # to the earlier operations. This is the final one as IDs have now been mapped
        self.var.waterbody_outflow_points = self.get_outflows(self.var.waterBodyID)

        # change ldd: put pits in where lakes are:
        ldd_LR = self.model.data.grid.decompress(
            np.where(self.var.waterBodyID != -1, 5, self.var.lddCompress), fillvalue=0
        )

        # set new ldds without lakes reservoirs
        (
            self.var.lddCompress_LR,
            _,
            self.var.dirUp,
            self.var.dirupLen,
            self.var.dirupID,
            self.var.downstruct,
            _,
            self.var.dirDown,
            self.var.lendirDown,
        ) = define_river_network(
            ldd_LR,
            self.model.data.grid,
        )

        # boolean map as mask map for compressing and decompressing
        self.var.compress_LR = self.var.waterbody_outflow_points != -1
        self.var.waterBodyIDC = np.compress(
            self.var.compress_LR, self.var.waterbody_outflow_points
        )

        self.var.waterBodyTypC = self.water_body_data["waterbody_type"].values

        self.reservoir_operators = self.model.agents.reservoir_operators
        self.reservoir_operators.set_reservoir_data(self.water_body_data)

        self.var.lake_area = self.water_body_data["average_area"].values
        # a factor which increases evaporation from lake because of wind TODO: use wind to set this factor
        self.var.lakeEvaFactor = self.model.config["parameters"]["lakeEvaFactor"]

        self.var.volume = self.water_body_data["volume_total"].values

        self.var.total_inflow_from_other_water_bodies = self.var.load_initial(
            "total_inflow_from_other_water_bodies",
            default=np.zeros_like(self.var.volume, dtype=np.float32),
        )

        # lake discharge at outlet to calculate alpha: parameter of channel width, gravity and weir coefficient
        # Lake parameter A (suggested  value equal to outflow width in [m])
        average_discharge = np.maximum(
            self.water_body_data["average_discharge"].values,
            0.1,
        )

        # channel width in [m]
        channel_width = get_channel_width(average_discharge)

        self.lake_factor = get_lake_factor(
            channel_width,
            overflow_coefficient_mu,
            self.model.config["parameters"]["lakeAFactor"],
        )

        self.var.storage = self.var.load_initial(
            "storage", default=self.var.volume.copy()
        )
        self.var.outflow_height = estimate_outflow_height(
            self.var.volume, self.lake_factor, self.var.lake_area, average_discharge
        )

        self.var.prev_lake_inflow = self.var.load_initial(
            "prev_lake_inflow", default=average_discharge.copy()
        )
        self.var.prev_lake_outflow = self.var.load_initial(
            "prev_lake_outflow", default=self.var.prev_lake_inflow.copy()
        )

    def map_water_bodies_IDs(self, compressed_waterbody_ids, waterBodyID_original):
        water_body_mapping = np.full(
            compressed_waterbody_ids.max() + 2, -1, dtype=np.int32
        )  # make sure that the last entry is also -1, so that -1 maps to -1
        water_body_mapping[compressed_waterbody_ids] = np.arange(
            0, compressed_waterbody_ids.size, dtype=np.int32
        )
        return water_body_mapping[waterBodyID_original], water_body_mapping

    def load_water_body_data(self, waterbody_mapping, waterbody_original_ids):
        water_body_data = pd.read_csv(
            self.model.model_structure["table"][
                "routing/lakesreservoirs/basin_lakes_data"
            ],
            dtype={
                "waterbody_type": np.int32,
                "volume_total": np.float64,
                "average_discharge": np.float64,
                "average_area": np.float64,
                "volume_flood": np.float64,
                "relative_area_in_region": np.float64,
            },
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
        # sort index to align with waterBodyID
        water_body_data = water_body_data.sort_index()
        return water_body_data

    def get_outflows(self, waterBodyID):
        # calculate biggest outlet = biggest accumulation of ldd network
        upstream_area_within_waterbodies = np.zeros_like(
            self.var.upstream_area_n_cells, shape=waterBodyID.max() + 2
        )
        upstream_area_within_waterbodies[-1] = -1
        np.maximum.at(
            upstream_area_within_waterbodies,
            waterBodyID[waterBodyID != -1],
            self.var.upstream_area_n_cells[waterBodyID != -1],
        )
        upstream_area_within_waterbodies = np.take(
            upstream_area_within_waterbodies, waterBodyID
        )

        waterbody_outflow_points = np.where(
            self.var.upstream_area_n_cells == upstream_area_within_waterbodies,
            waterBodyID,
            -1,
        )
        # make sure that each water body has an outflow
        assert np.array_equal(
            np.unique(waterbody_outflow_points), np.unique(waterBodyID)
        )
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

    def routing_lakes(self, inflowC):
        """
        Lake routine to calculate lake outflow
        :param inflowC: inflow to lakes and reservoirs [m3]
        :param NoRoutingExecuted: actual number of routing substep
        :return: QLakeOutM3DtC - lake outflow in [m3] per subtime step
        """
        if __debug__:
            prestorage = self.var.storage.copy()

        lakes = self.var.waterBodyTypC == LAKE

        # Lake inflow in [m3/s]
        lake_inflow_m3_s = np.zeros_like(inflowC)
        lake_inflow_m3_s[lakes] = inflowC[lakes] / self.var.dtRouting

        self.var.lake_outflow = np.zeros_like(inflowC)

        # check if there are any lakes in the model
        if (lakes == True).any():
            (
                self.var.lake_outflow[lakes],
                self.var.storage[lakes],
                height_above_outflow,
            ) = get_lake_outflow_and_storage(
                self.var.dtRouting,
                self.var.storage[lakes],
                lake_inflow_m3_s[lakes],
                self.var.prev_lake_inflow[lakes],
                self.var.prev_lake_outflow[lakes],
                self.lake_factor[lakes],
                self.var.lake_area[lakes],
                self.var.outflow_height[lakes],
            )

        # Difference between current and previous inflow
        lakedaycorrect_m3 = np.zeros_like(inflowC)
        lakedaycorrect_m3[lakes] = (
            (lake_inflow_m3_s[lakes] - self.var.prev_lake_inflow[lakes])
            * 0.5
            * self.var.dtRouting
        ) - (
            (self.var.lake_outflow[lakes] - self.var.prev_lake_outflow[lakes])
            * 0.5
            * self.var.dtRouting
        )  # [m3]

        outflow_m3 = self.var.lake_outflow * self.var.dtRouting

        if __debug__:
            balance_check(
                influxes=[
                    (lake_inflow_m3_s + self.var.prev_lake_inflow) / 2
                ],  # In [m3/s]
                outfluxes=[
                    (self.var.lake_outflow + self.var.prev_lake_outflow) / 2,
                ],  # Out
                prestorages=[prestorage / self.var.dtRouting],
                poststorages=[self.var.storage / self.var.dtRouting],
                name="lake",
                tollerance=1e-5,
            )

            inflow_lakes = np.zeros_like(inflowC)
            inflow_lakes[lakes] = inflowC[lakes]
            balance_check(
                influxes=[inflow_lakes / self.var.dtRouting],  # In [m3/s]
                outfluxes=[
                    self.var.lake_outflow,
                    lakedaycorrect_m3 / self.var.dtRouting,
                ],
                prestorages=[prestorage / self.var.dtRouting],
                poststorages=[self.var.storage / self.var.dtRouting],
                name="lake2",
                tollerance=1e-5,
            )

            balance_check(
                influxes=[inflow_lakes],  # In [m3]
                outfluxes=[
                    outflow_m3,
                    lakedaycorrect_m3,
                ],
                prestorages=[prestorage],
                poststorages=[self.var.storage],
                name="lake3",
                tollerance=0.1,
            )

        self.var.prev_lake_inflow = lake_inflow_m3_s
        self.var.prev_lake_outflow = self.var.lake_outflow

        return outflow_m3, lakedaycorrect_m3

    def routing_reservoirs(self, inflowC):
        """
        Reservoir outflow
        :param inflowC: inflow to reservoirs
        :return: qResOutM3DtC - reservoir outflow in [m3] per subtime step
        """
        if __debug__:
            prestorage = self.var.storage.copy()

        reservoirs = self.var.waterBodyTypC == RESERVOIR

        # Reservoir inflow in [m3] per timestep
        self.var.storage[reservoirs] += inflowC[reservoirs]
        # New reservoir storage [m3] = plus inflow for this sub step

        outflow_m3_s = np.zeros(self.var.waterBodyIDC.size, dtype=np.float64)
        outflow_m3_s[reservoirs] = (
            self.model.agents.reservoir_operators.regulate_reservoir_outflow(
                self.var.storage[reservoirs],
                inflowC[reservoirs]
                / self.var.dtRouting,  # convert per timestep to per second
                self.var.waterBodyIDC[reservoirs],
            )
        )

        outflow_m3 = outflow_m3_s * self.var.dtRouting
        assert (outflow_m3 <= self.var.storage).all()

        self.var.storage -= outflow_m3

        inflow_reservoirs = np.zeros_like(inflowC)
        inflow_reservoirs[reservoirs] = inflowC[reservoirs]
        if __debug__:
            balance_check(
                influxes=[inflow_reservoirs],  # In [m3/s]
                outfluxes=[outflow_m3],
                prestorages=[prestorage],
                poststorages=[self.var.storage],
                name="reservoirs",
                tollerance=1e-6,
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
            prestorage = self.var.storage.copy()

        # collect discharge from above waterbodies
        dis_LR = upstream1(self.var.downstruct, discharge)

        # only where lakes are and unit convered to [m]
        dis_LR = (
            np.where(self.var.waterBodyID != -1, dis_LR, 0.0)
            * self.model.seconds_per_timestep
        )

        runoff_m3 = runoff * self.var.cellArea / self.var.noRoutingSteps
        runoff_m3 = laketotal(runoff_m3, self.var.waterBodyID, nan_class=-1)

        discharge_m3 = upstream1(self.var.downstruct, discharge) * self.var.dtRouting
        discharge_m3 = laketotal(discharge_m3, self.var.waterBodyID, nan_class=-1)

        inflow_m3 = (
            runoff_m3 + discharge_m3 + self.var.total_inflow_from_other_water_bodies
        )

        # lakeEvaFactorC
        evaporation = np.minimum(
            evaporation_from_water_bodies_per_routing_step, self.var.storage
        )  # evaporation is already in m3 per routing substep
        evaporation[self.var.waterBodyTypC == OFF] = 0
        self.var.storage -= evaporation

        outflow_lakes, lakedaycorrect_m3 = self.routing_lakes(inflow_m3)
        outflow_reservoirs = self.routing_reservoirs(inflow_m3)

        outflow = outflow_lakes + outflow_reservoirs

        outflow_grid = self.var.full_compressed(0, dtype=np.float32)
        outflow_grid[self.var.compress_LR] = outflow

        # shift outflow 1 cell downstream
        outflow_shifted_downstream = upstream1(
            self.var.downstruct_no_water_bodies, outflow_grid
        )
        assert math.isclose(
            outflow_shifted_downstream.sum(), outflow_grid.sum(), rel_tol=0.00001
        )

        # everything with is not going to another lake is output to river network
        outflow_to_river_network = np.where(
            self.var.waterBodyID != -1, 0, outflow_shifted_downstream
        )
        # everything what is not going to the network is going to another lake
        # this will be added to the inflow of the other lake in the next
        # timestep
        outflow_to_another_lake = np.where(
            self.var.waterBodyID != -1, outflow_shifted_downstream, 0
        )

        # sum up all inflow from other lakes
        self.var.total_inflow_from_other_water_bodies = laketotal(
            outflow_to_another_lake, self.var.waterBodyID, nan_class=-1
        )

        if __debug__:
            balance_check(
                name="lakes and reservoirs",
                how="cellwise",
                influxes=[inflow_m3],
                outfluxes=[outflow, evaporation],
                prestorages=[prestorage],
                poststorages=[self.var.storage, lakedaycorrect_m3],
                tollerance=1,  # 1 m3
            )

        return outflow_to_river_network, evaporation

    @property
    def reservoir_storage(self):
        return self.var.storage[self.var.waterBodyTypC == RESERVOIR]

    @property
    def lake_storage(self):
        return self.var.storage[self.var.waterBodyTypC == LAKE]
