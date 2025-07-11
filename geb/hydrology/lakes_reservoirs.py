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


import geopandas as gpd
import numpy as np

from geb.hydrology.HRUs import load_grid
from geb.module import Module
from geb.workflows import balance_check

OFF: int = 0
LAKE: int = 1
RESERVOIR: int = 2
LAKE_CONTROL: int = 3  # currently modelled as normal lake


def laketotal(values, areaclass, nan_class):
    mask = areaclass != nan_class
    class_totals = np.bincount(areaclass[mask], weights=values[mask])
    return class_totals.astype(values.dtype)


GRAVITY: float = 9.81
SHAPE: str = "parabola"
# http://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf

if SHAPE == "rectangular":
    overflow_coefficient_mu: float = 0.577

    def estimate_lake_outflow(lake_factor, height_above_outflow):
        return lake_factor * height_above_outflow**1.5

    def outflow_to_height_above_outflow(lake_factor, outflow):
        """Inverse function of estimate_lake_outflow."""
        return (outflow / lake_factor) ** (2 / 3)

elif SHAPE == "parabola":
    overflow_coefficient_mu: float = 0.612

    def estimate_lake_outflow(lake_factor, height_above_outflow):
        return lake_factor * height_above_outflow**2

    def outflow_to_height_above_outflow(lake_factor, outflow):
        """Inverse function of estimate_lake_outflow."""
        return np.sqrt(outflow / lake_factor)

else:
    raise ValueError("Invalid shape")


def get_lake_height_from_bottom(lake_storage, lake_area):
    height_from_bottom = lake_storage / lake_area
    return height_from_bottom


def get_lake_storage_from_height_above_bottom(lake_height, lake_area):
    """Calculate the storage of a lake given its height above the bottom and area.

    Parameters
    ----------
    lake_height : float
        Height of the lake above the bottom in m
    lake_area : float
        Area of the lake in m2

    Returns:
    -------
    float
        Storage of the lake in m3
    """
    return lake_height * lake_area


def get_lake_height_above_outflow(lake_storage, lake_area, outflow_height):
    height_above_outflow = (
        get_lake_height_from_bottom(lake_storage, lake_area) - outflow_height
    )
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


def estimate_outflow_height(lake_capacity, lake_factor, lake_area, avg_outflow):
    height_above_outflow = outflow_to_height_above_outflow(lake_factor, avg_outflow)
    lake_height_when_full = get_lake_height_from_bottom(lake_capacity, lake_area)
    outflow_height = lake_height_when_full - height_above_outflow
    outflow_height[outflow_height < 0] = 0
    return outflow_height


def get_lake_outflow(
    dt,
    storage,
    lake_factor,
    lake_area,
    outflow_height,
):
    """Calculate outflow and storage for a lake using the Modified Puls method.

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

    Returns:
    -------
    outflow : float
        New outflow from the lake in m3/s
    storage : float
        New storage in m3

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


class LakesReservoirs(Module):
    """Implements all lakes and reservoir operations in the hydrological model.

    For reservoir it gets the outflow from the reservoir operator agents.

    Args:
        model: The GEB model instance.
        hydrology: The hydrology submodel instance.
    """

    def __init__(self, model, hydrology):
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self):
        return "hydrology.lakes_reservoirs"

    def spinup(self):
        # load lakes/reservoirs map with a single ID for each lake/reservoir
        waterBodyID_unmapped: np.ndarray = self.grid.load(
            self.model.files["grid"]["waterbodies/water_body_id"]
        )
        self.grid.var.waterBodyID, self.var.waterbody_mapping = (
            self.map_water_bodies_IDs(waterBodyID_unmapped)
        )

        # set discharge to NaN for all cells that are not part of a water body
        self.grid.var.discharge_m3_s[self.grid.var.waterBodyID != -1] = np.nan

        self.grid.var.waterbody_outflow_points = self.get_outflows(
            self.grid.var.waterBodyID
        )

        # we compress the waterbody_outflow_points, which we can later use to decompress
        waterbody_ids = np.unique(
            self.grid.var.waterbody_outflow_points[
                self.grid.var.waterbody_outflow_points != -1
            ]
        )

        self.var.water_body_data = self.load_water_body_data(
            self.var.waterbody_mapping, waterBodyID_unmapped
        )
        # sort the water bodies in the same order as the compressed water body IDs (waterbody_ids)
        self.var.water_body_data = self.var.water_body_data.sort_index()

        assert np.array_equal(self.var.water_body_data.index, waterbody_ids)

        self.var.water_body_type = self.var.water_body_data["waterbody_type"].values
        self.var.waterBodyOrigID = self.var.water_body_data[
            "original_waterbody_id"
        ].values
        # change water body type to LAKE if it is a control lake, thus currently modelled as normal lake
        self.var.water_body_type[self.var.water_body_type == LAKE_CONTROL] = LAKE

        # print("setting all water body types to LAKE")
        # self.var.water_body_type.fill(LAKE)

        assert (np.isin(self.var.water_body_type, [LAKE, RESERVOIR])).all()

        self.var.lake_area = self.var.water_body_data["average_area"].values
        self.var.capacity = self.var.water_body_data["volume_total"].values

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
            < 1e-10
        ).all()

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
        water_body_data = gpd.read_parquet(
            self.model.files["geoms"]["waterbodies/waterbody_data"],
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
        upstream_area_n_cells = self.hydrology.routing.river_network.upstream_area(
            unit="cell"
        )[~self.grid.mask]
        upstream_area_within_waterbodies = np.zeros_like(
            upstream_area_n_cells,
            shape=waterBodyID.max() + 2,
        )
        upstream_area_within_waterbodies[-1] = -1
        np.maximum.at(
            upstream_area_within_waterbodies,
            waterBodyID[waterBodyID != -1],
            upstream_area_n_cells[waterBodyID != -1],
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
            upstream_area_n_cells == upstream_area_within_waterbodies,
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

    def routing_lakes(self, routing_step_length_seconds: int | float):
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

    def routing_reservoirs(self, n_routing_substeps, current_substep):
        """Routine to update reservoir volumes and calculate reservoir outflow.

        Parameters
        ----------
        inflow_m3 : np.ndarray
            Inflow to the reservoirs in m3 per routing substep
        n_routing_substeps : int
            Number of routing substeps per time step

        Returns:
        -------
        reservoir_release_m3 : np.ndarray
            Outflow from the reservoirs in m3 per routing substep
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
        current_substep,
        n_routing_substeps,
        routing_step_length_seconds,
    ):
        if __debug__:
            prestorage = self.var.storage.copy()

        outflow_to_drainage_network_m3 = np.zeros_like(self.var.storage)
        command_area_release_m3 = np.zeros_like(self.var.storage)

        outflow_to_drainage_network_m3[self.is_lake] = self.routing_lakes(
            routing_step_length_seconds
        )
        (
            outflow_to_drainage_network_m3[self.is_reservoir],
            command_area_release_m3[self.is_reservoir],
        ) = self.routing_reservoirs(n_routing_substeps, current_substep)

        assert (outflow_to_drainage_network_m3 <= self.var.storage).all()

        if __debug__:
            balance_check(
                name="command_area_release",
                how="cellwise",
                influxes=[],
                outfluxes=[],
                prestorages=[prestorage[self.is_reservoir]],
                poststorages=[self.var.storage[self.is_reservoir]],
                tollerance=1,  # 1 m3
            )

        return outflow_to_drainage_network_m3, command_area_release_m3

    @property
    def is_reservoir(self):
        return self.var.water_body_type == RESERVOIR

    @property
    def is_lake(self):
        return self.var.water_body_type == LAKE

    @property
    def reservoir_storage(self):
        return self.var.storage[self.is_reservoir]

    @reservoir_storage.setter
    def reservoir_storage(self, value):
        self.var.storage[self.is_reservoir] = value

    @property
    def reservoir_capacity(self):
        return self.var.capacity[self.is_reservoir]

    @reservoir_capacity.setter
    def reservoir_capacity(self, value):
        self.var.capacity[self.is_reservoir] = value

    @property
    def lake_storage(self):
        return self.var.storage[self.is_lake]

    @lake_storage.setter
    def lake_storage(self, value):
        self.var.storage[self.is_lake] = value

    @property
    def lake_capacity(self):
        return self.var.capacity[self.is_lake]

    @lake_capacity.setter
    def lake_capacity(self, value):
        self.var.capacity[self.is_lake] = value

    @property
    def reservoir_fill_percentage(self):
        return self.reservoir_storage / self.reservoir_capacity * 100

    @property
    def n(self):
        return self.var.capacity.size

    def decompress(self, array):
        return array

    def step(self):
        """Dynamic part set lakes and reservoirs for each year."""
        # if first timestep, or beginning of new year
        if self.model.current_timestep == 1 or (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            if self.hydrology.dynamic_water_bodies:
                raise NotImplementedError("dynamic_water_bodies not implemented yet")

        # print(self.reservoir_fill_percentage.astype(int))
        self.report(self, locals())
