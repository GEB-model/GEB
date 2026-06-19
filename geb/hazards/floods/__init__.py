"""Class to setup, run, and post-process the SFINCS hydrodynamic model."""

from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry.point import Point

from geb.geb_types import (
    ArrayFloat32,
    TwoDArrayFloat32,
)
from geb.hazards.event import Event
from geb.hazards.floods.workflows.utils import get_start_point
from geb.hydrology.routing import (
    get_discharge_per_river,
    get_upstream_represented_xys as get_upstream_represented_xys,
)
from geb.module import Module
from geb.store import Bucket
from geb.workflows.io import read_geom, read_table

from ...hydrology.landcovers import OPEN_WATER as OPEN_WATER, SEALED as SEALED
from ...workflows.io import (
    read_params,
    read_zarr,
    write_zarr,
)
from ...workflows.raster import reclassify
from .sfincs import (
    MultipleSFINCSSimulations,
    SFINCSRootModel,
    SFINCSSimulation,
)

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


def create_river_graph(rivers: gpd.GeoDataFrame) -> nx.DiGraph:
    """Creates a directed graph representation of the river network.

    Args:
        rivers: A GeoDataFrame containing the river network with downstream IDs.

    Returns:
        A directed graph (networkx DiGraph) representing the river network.
    """
    rivers_without_outflow_basin = rivers[~rivers["is_downstream_outflow"]]

    river_graph: nx.DiGraph = nx.DiGraph()
    rivers_in_network = set(rivers_without_outflow_basin.index)
    for river_id, row in rivers_without_outflow_basin.iterrows():
        river_graph.add_node(river_id, uparea_m2=row["uparea_m2"])
        downstream_id = row["downstream_ID"]

        # only add edge if downstream river is in the network and not -1 (ocean)
        if downstream_id != -1 and downstream_id in rivers_in_network:
            river_graph.add_edge(river_id, downstream_id)

    return river_graph


def group_subbasins(
    river_graph: nx.DiGraph, max_area_m2: float | int
) -> dict[int, list[int]]:
    """Groups subbasins in the river graph aiming while keeping the area of each group below max_area_m2.

    Args:
        river_graph: The river network as a directed graph (networkx DiGraph).
        max_area_m2: The maximum upstream area (in m²) allowed for each group.

    Returns:
        A dictionary mapping group IDs to lists of subbasin node IDs.
    """
    river_graph = river_graph.copy()
    # add attribute for merged nodes
    nx.set_node_attributes(
        river_graph, {node: [node] for node in river_graph.nodes}, "merged_nodes"
    )

    # for each node, derive the area of the node by excluding the area of the upstream nodes
    for node in river_graph.nodes:
        upstream_nodes = list(river_graph.predecessors(node))
        upstream_area = sum(
            river_graph.nodes[up_node]["uparea_m2"] for up_node in upstream_nodes
        )
        assert upstream_area >= 0
        river_graph.nodes[node]["area_m2"] = (
            river_graph.nodes[node]["uparea_m2"] - upstream_area
        )

    groups: dict[int, list[int]] = {}
    group_id: int = 0

    while len(river_graph.nodes) > 1:
        # get all nodes without predecessors (i.e., headwater nodes)
        headwater_nodes_to_merge = [
            (n, river_graph.nodes[n]["area_m2"])
            for n, d in river_graph.in_degree()
            if d == 0
        ]
        if not headwater_nodes_to_merge:
            break

        # sort headwater nodes by area descending
        headwater_nodes_to_merge.sort(key=lambda x: x[1], reverse=True)

        for potential_node_to_merge, _ in headwater_nodes_to_merge:
            downstream_node = list(river_graph.successors(potential_node_to_merge))
            assert len(downstream_node) == 1
            downstream_node = downstream_node[0]

            new_area_after_merge = (
                river_graph.nodes[downstream_node]["area_m2"]
                + river_graph.nodes[potential_node_to_merge]["area_m2"]
            )

            if new_area_after_merge > max_area_m2:
                groups[group_id] = river_graph.nodes[potential_node_to_merge][
                    "merged_nodes"
                ]
                river_graph.remove_node(potential_node_to_merge)
                group_id += 1
                continue

            river_graph.nodes[downstream_node]["merged_nodes"].extend(
                river_graph.nodes[potential_node_to_merge]["merged_nodes"]
            )
            river_graph.nodes[downstream_node]["area_m2"] = new_area_after_merge
            # remove node
            river_graph.remove_node(potential_node_to_merge)
            break

    # add remaining nodes as groups
    for node in river_graph.nodes:
        groups[group_id] = river_graph.nodes[node]["merged_nodes"]

    return groups


class FloodVariables(Bucket):
    """Class to hold variables for the Floods module."""

    discharge_per_timestep: deque[TwoDArrayFloat32]
    runoff_m_per_timestep: deque[TwoDArrayFloat32]


class Floods(Module):
    """The class that implements all methods to setup, run, and post-process hydrodynamic flood models.

    Args:
        model: The GEB model instance.
        n_timesteps: The number of timesteps to keep in memory for discharge calculations (default is 10).
    """

    var: FloodVariables

    def __init__(self, model: GEBModel, longest_flood_event_in_days: int = 10) -> None:
        """Initializes the Floods class.

        Args:
            model: The GEB model instance.
            longest_flood_event_in_days: The number of timesteps to keep in memory for discharge calculations (default is 10).
        """
        super().__init__(model)

        self.model: GEBModel = model
        self.config: dict[str, Any] = (
            self.model.config["hazards"]["floods"]
            if "floods" in self.model.config["hazards"]
            else {}
        )

        self.DEM_config: list[dict[str, Any]] = read_params(
            self.model.files["dict"]["hydrodynamics/DEM_config"]
        )
        for entry in self.DEM_config:
            entry["elevation"] = read_zarr(
                self.model.files["other"][entry["path"]]
            ).to_dataset(name="elevation")

        self.HRU = model.hydrology.HRU

        if self.model.simulate_hydrology:
            self.hydrology: Hydrology = model.hydrology
            self.longest_flood_event_in_days: int = longest_flood_event_in_days

            self.var.discharge_per_timestep: deque[ArrayFloat32] = deque(
                maxlen=self.longest_flood_event_in_days
            )
            self.var.runoff_m_per_timestep: deque[ArrayFloat32] = deque(
                maxlen=self.longest_flood_event_in_days
            )

    @property
    def name(self) -> str:
        """The name of the module."""
        return "hazard_driver.floods"

    def spinup(self) -> None:
        """Spinup method for the Floods module.

        Currently, this method does nothing as flood simulations do not require spinup.
        """
        pass

    def step(self) -> None:
        """Steps the Floods module.

        Currently, this method does nothing as flood simulations are handled in the HazardDriver.
        """
        pass

    def get_utm_zone(self, region_file: Path | str) -> str:
        """Determine the UTM zone based on the centroid of the region geometry.

        Args:
            region_file: Path to the region geometry file.

        Returns:
            The EPSG code for the UTM zone of the centroid of the region.
        """
        region: gpd.GeoDataFrame = read_geom(region_file)

        # Calculate the central longitude of the dataset
        centroid: Point = region.union_all().centroid

        # Determine the UTM zone based on the longitude
        utm_zone: int = int((centroid.x + 180) // 6) + 1

        # Determine if the data is in the Northern or Southern Hemisphere
        # The EPSG code for UTM in the northern hemisphere is EPSG:326xx (xx = zone)
        # The EPSG code for UTM in the southern hemisphere is EPSG:327xx (xx = zone)
        if centroid.y > 0:
            utm_crs: str = f"EPSG:326{utm_zone}"  # Northern hemisphere
        else:
            utm_crs: str = f"EPSG:327{utm_zone}"  # Southern hemisphere
        return utm_crs

    def build(
        self,
        name: str,
        all_rivers: gpd.GeoDataFrame,
        discharge_by_river: pd.DataFrame,
        subbasins: gpd.GeoDataFrame,
        coastal: bool = False,
        low_elevation_coastal_zone_mask: gpd.GeoDataFrame | None = None,
        coastal_boundary_exclude_mask: gpd.GeoDataFrame | None = None,
        initial_water_level: float | None = 0.0,
    ) -> SFINCSRootModel:
        """Builds or reads a SFINCS model without any forcing.

        Before using this model, forcing must be set.

        When the model already exists and force_overwrite is False, the existing model is read.

        Args:
            name: Name of the SFINCS model (used for the model root directory).
            subbasins: The subbasins to build the SFINCS model for. The model domain is defined based on the subbasins.
            all_rivers: All rivers in the area. In contrast to the subbasins, these rivers are not used to define the model domain,
                but are only used to get the river geometries and attributes for the rivers that are included in the model domain based on the subbasins variable.
            discharge_by_river: DataFrame containing discharge data for each river.
            coastal: Whether to only include coastal areas in the model.
            low_elevation_coastal_zone_mask: A GeoDataFrame defining the low elevation coastal zone to set as active cells.
            coastal_boundary_exclude_mask: GeoDataFrame defining the areas to exclude from the coastal model boundary cells.
            initial_water_level: The initial water level to initiate the model. SFINCS fills all cells below this level with water.

        Returns:
            The built or read SFINCSRootModel instance.
        """
        sfincs_model = SFINCSRootModel(
            self.simulation_root, name, logger=self.model.logger
        )

        sfincs_model.build(
            subbasins=subbasins,
            DEMs=self.DEM_config,
            rivers=all_rivers,
            discharge_by_river=discharge_by_river,
            river_width_alpha=self.model.hydrology.grid.decompress(
                self.model.hydrology.grid.var.river_width_alpha
            ),
            river_width_beta=self.model.hydrology.grid.decompress(
                self.model.hydrology.grid.var.river_width_beta
            ),
            mannings=self.mannings,
            grid_size_multiplier=self.config["grid_size_multiplier"],
            subgrid=self.config["subgrid"],
            depth_calculation_method=self.model.config["hydrology"]["routing"][
                "river_depth"
            ]["method"],
            depth_calculation_parameters=self.model.config["hydrology"]["routing"][
                "river_depth"
            ]["parameters"]
            if "parameters" in self.model.config["hydrology"]["routing"]["river_depth"]
            else {},
            low_elevation_coastal_zone_mask=low_elevation_coastal_zone_mask,
            coastal_boundary_exclude_mask=coastal_boundary_exclude_mask,
            coastal=coastal,
            setup_river_outflow_boundary=not coastal,
            initial_water_level=initial_water_level,
            custom_rivers_to_burn=read_geom(
                self.model.files["geom"]["routing/custom_rivers"]
            )
            if "routing/custom_rivers" in self.model.files["geom"]
            else None,
            overwrite=self.config["overwrite"],
            p_value_threshold=self.config["p_value_threshold"],
            selection_strategy=self.config["selection_strategy"],
            fixed_shape=self.config["fixed_shape"],
            write_figures=self.config["write_figures"],
        )

        return sfincs_model

    def set_forcing(
        self,
        sfincs_model: SFINCSRootModel,
        event: Event,
        active_basins: Iterable[int],
        discharge_by_river: pd.DataFrame,
    ) -> SFINCSSimulation:
        """Sets the forcing for a SFINCS simulation.

        Depending on the forcing method in the config, either headwater point discharge
        or precipitation is used as forcing.

        Args:
            sfincs_model: The SFINCSRootModel instance to create the simulation from.
            event: An Event object containing the flood event details, including 'start_time' and 'end_time'.
            active_basins: An iterator of the active basin IDs to include in the forcing.
            discharge_by_river: A DataFrame containing the discharge data for each river.
        Returns:
            The created SFINCSSimulation instance with the forcing set.

        Raises:
            ValueError: If the forcing method is unknown.
        """
        # Save the flood depth to a zarr file
        simulation: SFINCSSimulation = sfincs_model.create_simulation(
            event=event,
            write_figures=self.config["write_figures"],
            flood_map_output_interval_seconds=self.config[
                "flood_map_output_interval_seconds"
            ],
        )

        routing_substeps: int = self.var.discharge_per_timestep[0].shape[0]
        if self.config["forcing_method"] == "headwater_points":
            forcing_grid = self.hydrology.grid.decompress(
                np.vstack(
                    list(self.var.discharge_per_timestep)
                    + [
                        self.var.discharge_per_timestep[-1][-1]
                    ]  # add last timestep again (ensuring stable forcing during last hour)
                )
            )
        elif self.config["forcing_method"] == "accumulated_runoff":
            forcing_grid = self.hydrology.grid.decompress(
                np.vstack(
                    list(self.var.runoff_m_per_timestep)
                    + [self.var.runoff_m_per_timestep[-1][-1]]
                )  # add last timestep again (ensuring stable forcing during last hour)
            )
        else:
            raise ValueError(
                f"Unknown forcing method {self.config['forcing_method']}. Supported are 'headwater_points' and 'accumulated_runoff'."
            )

        substep_size: timedelta = self.model.timestep_length / routing_substeps

        # convert the forcing grid to an xarray DataArray
        forcing_grid: xr.DataArray = xr.DataArray(
            data=forcing_grid,
            coords={
                "time": pd.date_range(
                    end=self.model.current_time
                    + self.model.timestep_length,  # end of the current timestep
                    periods=len(self.var.discharge_per_timestep) * routing_substeps + 1,
                    freq=substep_size,
                ),
                "y": self.hydrology.grid.lat,
                "x": self.hydrology.grid.lon,
            },
            dims=["time", "y", "x"],
            name="forcing",
        )

        # ensure that we have forcing data for the entire event period
        assert (
            pd.to_datetime(forcing_grid.time.values[-1]).to_pydatetime()
            >= event.end_time
        )
        assert (
            pd.to_datetime(forcing_grid.time.values[0]).to_pydatetime()
            <= event.start_time
        )

        forcing_grid: xr.DataArray = forcing_grid.rio.write_crs(self.model.crs)

        if self.config["forcing_method"] == "headwater_points":
            simulation.set_headwater_forcing_from_grid(  # ty:ignore[unresolved-attribute]
                discharge_grid=forcing_grid,
            )

        elif self.config["forcing_method"] == "accumulated_runoff":
            basin_ids = self.model.hydrology.routing.basin_ids.copy()
            basin_ids[
                ~np.isin(basin_ids, list(active_basins))
            ] = -1  # set non-active basins to -1, so that they are not included in the forcing
            simulation.set_accumulated_runoff_forcing(
                runoff_m=forcing_grid,
                river_network=self.model.hydrology.routing.river_network,
                river_ids=self.model.hydrology.grid.decompress(
                    self.model.hydrology.routing.river_ids, fillvalue=-1
                ),
                river_ids_no_waterbodies_removed=self.model.hydrology.grid.decompress(
                    self.model.hydrology.routing.river_ids_no_waterbodies_removed,
                    fillvalue=-1,
                ),
                basin_ids=basin_ids,
                upstream_area=self.model.hydrology.grid.decompress(
                    self.model.hydrology.grid.var.upstream_area
                ),
                cell_area=self.model.hydrology.grid.decompress(
                    self.model.hydrology.grid.var.cell_area
                ),
            )
        else:
            raise ValueError(
                f"Unknown forcing method {self.config['forcing_method']}. Supported are 'headwater_points' and 'accumulated_runoff'."
            )

        simulation.set_river_inflow(discharge_by_river=discharge_by_river)
        return simulation

    def run_single_event(
        self,
        event: Event,
    ) -> None:
        """Runs a single flood event using the SFINCS model.

        Also updates the flood status of households in the model based on the flood depth results.

        Args:
            event: An Event object containing the flood event details, including 'start_time' and 'end_time'.

        Raises:
            ValueError: If neither 'export_max_intensity' nor 'export_final_intensity' is True in the event.
            ValueError: If both 'export_max_intensity' and 'export_final_intensity' are True in the event.
        """
        assert event.kind == "flood", (
            f"Expected event type 'flood', but got '{event.kind}'"
        )
        subbasins: gpd.GeoDataFrame = read_geom(
            self.model.files["geom"]["routing/subbasins"]
        )

        # first select only active rivers
        simulation_rivers: gpd.GeoDataFrame = self.model.hydrology.routing.active_rivers

        # but also include downstream outflows
        rivers: gpd.GeoDataFrame = self.model.hydrology.routing.rivers.copy()
        simulation_rivers: gpd.GeoDataFrame = pd.concat(
            [simulation_rivers, rivers[rivers["is_downstream_outflow"]]]
        )

        if self.config["subbasins"] != "all":
            if not isinstance(self.config["subbasins"], list):
                raise ValueError(
                    f"Invalid config for floods subbasins: {self.config['subbasins']}. Expected 'all' or a list of subbasin ids."
                )
            if not all(
                subbasin_id in subbasins.index
                for subbasin_id in self.config["subbasins"]
            ):
                raise ValueError(
                    f"Invalid subbasin IDs in config: {self.config['subbasins']}. Not all IDs available in config."
                )

            included_subbasins: gpd.GeoDataFrame = subbasins[
                subbasins.index.isin(self.config["subbasins"])
            ]

            # get downstream subbasins of the included subbasins
            downstream_subbasins: gpd.GeoDataFrame = subbasins[
                subbasins.index.isin(
                    rivers.loc[included_subbasins.index]["downstream_ID"]
                )
            ]
            assert len(downstream_subbasins) >= 1, (
                "At least one downstream subbasin must be included in the simulation."
            )

            # reset rivers downstream outflow column
            # simulation_rivers["is_downstream_outflow"] = False
            rivers["is_downstream_outflow"] = False

            # set downstream outflow for the downstream subbasins
            for downstream_subbasin_id in downstream_subbasins.index:
                # simulation_rivers.loc[
                #     simulation_rivers.index == downstream_subbasin_id,
                #     "is_downstream_outflow",
                # ] = True
                rivers.loc[
                    rivers.index == downstream_subbasin_id, "is_downstream_outflow"
                ] = True

            grouped_subbasins = {0: list(included_subbasins.index)}
        else:
            river_graph = create_river_graph(simulation_rivers)
            grouped_subbasins = group_subbasins(
                river_graph=river_graph,
                max_area_m2=1e20,  # very large to force single group only
            )

        assert len(grouped_subbasins) == 1, "currently only single group supported"
        for group_id, group in grouped_subbasins.items():
            group = set(group) | set(
                simulation_rivers.loc[simulation_rivers.index.isin(group)][
                    "downstream_ID"
                ]
            )
            subbasins_group = subbasins[subbasins.index.isin(group)]

            sfincs_root_model = self.build(
                f"group_{group_id}",
                all_rivers=rivers,
                discharge_by_river=self.discharge_by_river(
                    self.model.config["general"]["spinup_name"]
                ),
                subbasins=subbasins_group,
            )  # build or read the model
            sfincs_simulation = self.set_forcing(  # set the forcing
                sfincs_root_model,
                event,
                active_basins=group,
                discharge_by_river=self.discharge_by_river(),
            )
            self.model.logger.info(
                f"Running SFINCS for {self.model.current_time}..."
            )  # log the start of the simulation

            sfincs_simulation.run(
                gpu=self.config.get("SFINCS", {}).get("gpu", "auto"),
            )  # run the simulation

        if event.export_max_intensity and event.export_final_intensity:
            raise ValueError(
                "Only one of 'export_max_intensity' or 'export_final_intensity' can be True."
            )
        if event.export_max_intensity:
            flood_depth: xr.DataArray = sfincs_simulation.read_max_flood_depth(
                self.config["minimum_flood_depth"]
            )  # read the flood depth results
            postfix = "_max"
        elif event.export_final_intensity:
            flood_depth: xr.DataArray = sfincs_simulation.read_final_flood_depth(
                self.config["minimum_flood_depth"]
            )  # read the flood depth results
            postfix = "_final"
        else:
            raise ValueError(
                "Either 'export_max_intensity' or 'export_final_intensity' must be True."
            )

        filename: Path = (
            self.model.output_folder
            / "flood_maps"
            / (sfincs_simulation.name + postfix + ".zarr")
        )

        flood_depth: xr.DataArray = write_zarr(
            da=flood_depth,
            path=filename,
            crs=flood_depth.rio.crs,
        )  # save the flood depth to a zarr file

        # This check is done to compute damages (using ERA5) only after multiverse is finished
        if self.model.multiverse_name is None:
            if self.model.config["general"]["forecasts"]["use"]:
                print("Multiverse no longer active, now compute flood damages...")
            # Check if damage simulation is enabled before calculating damages
            if self.model.config["hazards"]["damage"]["simulate"]:
                self.model.agents.households.flood_risk_module.flood(
                    flood_depth=flood_depth
                )

    def get_return_period_maps(self, run_name: str) -> None:
        """Generates flood maps for specified return periods using the SFINCS model.

        Args:
            run_name: The name of the run to use for estimating return periods (e.g., "spinup").

        Raises:
            ValueError: If no hydrograph is found for a node and return period.
        """
        # close the zarr store
        if hasattr(self.model, "reporter"):
            self.model.reporter.variables["discharge_daily"].close()

        # load model settings
        coastal_only = self.config["coastal_only"]

        # load the subbasin geometry for the model domain
        subbasins = read_geom(self.model.files["geom"]["routing/subbasins"])
        coastal = subbasins["is_coastal"].any()

        rivers = self.model.hydrology.routing.rivers
        discharge_by_river = self.discharge_by_river(
            self.model.config["general"]["spinup_name"]
        )
        # if coastal load files
        if coastal:
            # Load mask of lower elevation coastal zones to activate cells for the different sfincs model regions
            low_elevation_coastal_zone_mask = read_geom(
                self.model.files["geom"]["coastal/low_elevation_coastal_zone_mask"]
            )
            low_elevation_coastal_zone_mask = low_elevation_coastal_zone_mask[
                ["initial_water_level", "geometry"]
            ]

            # use COMID as index and set unique index name for coastal region
            low_elevation_coastal_zone_mask.index = pd.Index([-1], name="COMID")

            # get initial_water_level for model domain
            initial_water_level = low_elevation_coastal_zone_mask[
                "initial_water_level"
            ].min()

            # # buffer lower elevation coastal zone mask to ensure proper inclusion of coastline
            low_elevation_coastal_zone_mask["geometry"] = (
                low_elevation_coastal_zone_mask.buffer(0.00833333)
            )

            # load osm land polygons to exclude from coastal boundary cells
            coastal_boundary_exclude_mask = read_geom(
                self.model.files["geom"]["coastal/land_polygons"],
            )

            coastal_subbasins = subbasins[subbasins["is_coastal"]].copy()

            # remove coastal subbasin from low elevation coastal zone mask
            low_elevation_coastal_zone_mask = gpd.overlay(
                low_elevation_coastal_zone_mask,
                coastal_subbasins,
                how="difference",
            )
            low_elevation_coastal_zone_mask["is_downstream_outflow"] = False
            low_elevation_coastal_zone_mask["COMID"] = 0  # 0 is not used. -1 is nan
            coastal_subbasins = pd.concat(
                [coastal_subbasins, low_elevation_coastal_zone_mask],
                ignore_index=False,
            )
            coastal_subbasins.to_file(
                "temp_coastal_subbasins.geojson", driver="GeoJSON"
            )

            model_name = "coastal_region"

            # load location and offset for coastal water level forcing
            coastal_forcing_locations: gpd.GeoDataFrame = read_geom(
                self.model.files["geom"]["gtsm/stations_coast_rp"]
            )

            coastal_offset = xr.open_dataarray(
                self.model.files["other"][
                    "coastal/global_ocean_mean_dynamic_topography"
                ]
            ).rio.write_crs("EPSG:4326")

            # load sea level rise data
            sea_level_rise_rcp8p5: pd.DataFrame = read_table(
                self.model.files["table"]["gtsm/sea_level_rise_rcp8p5"]
            )

            sfincs_coastal_root_model: SFINCSRootModel = self.build(
                name=model_name,
                subbasins=coastal_subbasins,
                coastal=True,
                all_rivers=rivers[rivers.intersects(coastal_subbasins.union_all())],
                discharge_by_river=discharge_by_river,
                coastal_boundary_exclude_mask=coastal_boundary_exclude_mask,
                low_elevation_coastal_zone_mask=low_elevation_coastal_zone_mask,
                initial_water_level=initial_water_level,
            )

            if coastal_only:
                # subset subbasins to those touching the low elevation coastal zone mask
                subbasins = subbasins[
                    subbasins.intersects(low_elevation_coastal_zone_mask.union_all())
                ].copy()

        sfincs_inland_root_models: list[SFINCSRootModel] = []

        # for each subbasin, build a separate sfincs model with its downstream basin
        # when there is no downstream basin (ocean), only build for the subbasin itself

        # Subset subbasins to only subbasins that are either represented in the grid,
        # or have upstream rivers that are.
        riverine_active_subbasin: gpd.GeoDataFrame = subbasins[
            subbasins.index.isin(self.model.hydrology.routing.active_rivers.index)
        ]

        for subbasin_id, subbasin in riverine_active_subbasin.iterrows():
            downstream_basin = rivers.loc[subbasin_id]["downstream_ID"]

            region_subbasins = subbasins[
                subbasins.index.isin([subbasin_id, downstream_basin])
            ].copy()
            region_rivers = rivers.copy()

            # if there is a downstream basin, mark it as downstream outflow subbasin
            if downstream_basin != -1:
                region_subbasins.at[downstream_basin, "is_downstream_outflow"] = True
                region_rivers.at[downstream_basin, "is_downstream_outflow"] = True

                sfincs_inland_root_model = self.build(
                    name=f"inland_subbasin_{subbasin_id}",
                    subbasins=region_subbasins,
                    all_rivers=region_rivers,
                    discharge_by_river=discharge_by_river,
                    coastal=False,
                )
                sfincs_inland_root_model.estimate_discharge_for_return_periods(
                    discharge_by_river=self.discharge_by_river(run_name),
                    return_periods=self.config["return_periods"],
                    p_value_threshold=self.config["p_value_threshold"],
                    selection_strategy=self.config["selection_strategy"],
                    fixed_shape=self.config["fixed_shape"],
                    write_figures=self.config["write_figures"],
                )
                sfincs_inland_root_models.append(sfincs_inland_root_model)

        for return_period in self.config["return_periods"]:
            simulations: list[SFINCSSimulation] = []

            if coastal:
                sfincs_coastal_simulation: SFINCSSimulation = (
                    sfincs_coastal_root_model.create_coastal_return_period_simulation(
                        return_period,
                        coastal_forcing_locations,
                        offset=coastal_offset,
                        sea_level_rise=sea_level_rise_rcp8p5,
                        year=self.model.current_time.year,
                    )
                )
                simulations.append(sfincs_coastal_simulation)

            for sfincs_inland_root_model in sfincs_inland_root_models:
                inflow_nodes = sfincs_inland_root_model.active_rivers[
                    ~sfincs_inland_root_model.active_rivers["is_downstream_outflow"]
                ]
                inflow_nodes["geometry"] = inflow_nodes["geometry"].apply(
                    get_start_point
                )

                # Build list of hydrograph DataFrames using the original node indices as column names
                Q: list[pd.DataFrame] = []
                for node_idx in inflow_nodes.index:
                    hydro = inflow_nodes.at[node_idx, f"hydrograph_{return_period}"]
                    if hydro is None:
                        raise ValueError(
                            f"No hydrograph found for node {node_idx} and return period {return_period}."
                        )
                    # hydro is expected to be dict-like {iso_timestamp: Q} — convert to DataFrame with column named node_idx
                    df = pd.DataFrame.from_dict(
                        hydro, orient="index", columns=np.array([node_idx])
                    )
                    Q.append(df)

                # Concatenate the per-node series into a single DataFrame; index -> timestamps
                Q: pd.DataFrame = pd.concat(Q, axis=1)
                Q.index = pd.to_datetime(Q.index)

                event = Event(
                    kind="flood",
                    name=f"rp_{return_period}",
                    start_time=Q.index[0].to_pydatetime(),
                    end_time=Q.index[-1].to_pydatetime(),
                    create_max_intensity_map=True,
                )

                sfincs_inland_simulation: SFINCSSimulation = (
                    sfincs_inland_root_model.create_simulation(
                        event=event,
                    )
                )

                sfincs_inland_simulation.set_discharge_forcing_from_nodes(
                    nodes=inflow_nodes.to_crs(sfincs_inland_root_model.crs),
                    timeseries=Q,
                )

                simulations.append(sfincs_inland_simulation)

            simulation = MultipleSFINCSSimulations(simulations=simulations)
            if simulations:
                simulation.run(
                    gpu=self.config.get("SFINCS", {}).get("gpu", "auto"),
                )
                flood_depth_return_period: xr.DataArray = (
                    simulation.read_max_flood_depth(self.config["minimum_flood_depth"])
                )
            else:
                self.model.logger.warning(
                    "No rivers found that are represented in grid and/or are not fully inside waterbodies. Creating dummy empty flood map."
                )
                dummy_sfincs_model = SFINCSRootModel(
                    self.simulation_root, "dummy", logger=self.model.logger
                )
                dummy_mask = dummy_sfincs_model.create_mask(
                    self.DEM_config,
                    subbasins[~subbasins["is_downstream_outflow"]],
                    self.config["grid_size_multiplier"],
                )

                flood_depth_return_period = dummy_mask.astype(np.float32)
                flood_depth_return_period[:] = 0
                flood_depth_return_period.attrs["_FillValue"] = np.nan

            # mask floodplain with land polygons to remove inundation in the sea
            if coastal and coastal_boundary_exclude_mask is not None:
                flood_depth_return_period = flood_depth_return_period.rio.clip(
                    coastal_boundary_exclude_mask.geometry,
                    coastal_boundary_exclude_mask.crs,
                    invert=False,
                )

            write_zarr(
                flood_depth_return_period,
                self.model.output_folder / "flood_maps" / f"{return_period}.zarr",
                crs=flood_depth_return_period.rio.crs,
            )

            # simulation.cleanup()

    def save_discharge(self, discharge_m3_s_per_substep: TwoDArrayFloat32) -> None:
        """Saves the current discharge for the current timestep.

        SFINCS is run at the end of an event rather than at the beginning. Therefore,
        we need to trace back the conditions at the beginning of the event. This function
        saves the current discharge at the beginning of each timestep,
        so it can be used later when setting up the SFINCS model.
        """
        self.var.discharge_per_timestep.append(
            discharge_m3_s_per_substep
        )  # this is a deque, so it will automatically remove the oldest discharge

    def save_runoff_m(self, overland_runoff_m: TwoDArrayFloat32) -> None:
        """Saves the current runoff for the current timestep."""
        self.var.runoff_m_per_timestep.append(
            overland_runoff_m
        )  # this is a deque, so it will automatically remove the oldest runoff

    def discharge_by_river(self, run_name: str | None = None) -> pd.DataFrame:
        """Open the discharge datasets from the model output folder.

        Args:
            run_name: The name of the run to use for estimating discharge (e.g., "spinup").
                If None (default), the current run is used.

        Returns:
            A pandas DataFrame containing the discharge time series for each river, indexed by timestamp.
        """
        rivers: gpd.GeoDataFrame = (
            self.model.hydrology.routing.get_active_and_downstream_outflow_rivers()
        )
        all_rivers = self.model.hydrology.routing.rivers

        # if spinup is requested, at least discard the first 10 years of data.
        if run_name == self.model.config["general"]["spinup_name"]:
            assert run_name is not None
            discharge = get_discharge_per_river(
                folder=self.model.report_folder.parent.parent
                / run_name
                / "report"
                / "hydrology.routing",
                rivers=rivers,
                all_rivers=all_rivers,
            )
            start_time = discharge.index[0] + pd.DateOffset(years=10)
        else:
            discharge = get_discharge_per_river(
                rivers=rivers,
                all_rivers=all_rivers,
                source="memory",
                variables_to_report=self.hydrology.routing.variables_to_report,
            )
            start_time = discharge.index[0]

        discharge = discharge.loc[start_time:]

        # set the frequency of the index
        discharge.index.freq = pd.infer_freq(discharge.index)

        return discharge

    @property
    def mannings(self) -> xr.DataArray:
        """Get the Manning's n values for the land cover types."""
        mannings = reclassify(
            self.land_cover,
            self.land_cover_mannings_rougness_classification.set_index(
                "esa_worldcover"
            )["N"].to_dict(),
            method="lookup",
        )
        return mannings

    @property
    def land_cover(self) -> xr.DataArray:
        """Get the land cover classification for the model.

        Returns:
            An xarray DataArray containing the land cover classification.
        """
        return read_zarr(self.model.files["other"]["landcover/classification"])

    @property
    def land_cover_mannings_rougness_classification(self) -> pd.DataFrame:
        """Get the land cover classification table for Manning's roughness.

        Returns:
            A DataFrame containing the land cover classification for Manning's roughness.
        """
        return pd.DataFrame(
            data=[
                [10, "Tree cover", 10, 0.12],
                [20, "Shrubland", 20, 0.05],
                [30, "Grasland", 30, 0.034],
                [40, "Cropland", 40, 0.037],
                [50, "Built-up", 50, 0.1],
                [60, "Bare / sparse vegetation", 60, 0.023],
                [70, "Snow and Ice", 70, 0.01],
                [80, "Permanent water bodies", 80, 0.02],
                [90, "Herbaceous wetland", 90, 0.035],
                [95, "Mangroves", 95, 0.07],
                [100, "Moss and lichen", 100, 0.025],
                [0, "No data", 0, 0.1],
            ],
            columns=np.array(
                ["esa_worldcover", "description", "landuse", "N"], dtype=str
            ),
        )

    @property
    def crs(self) -> str:
        """Get the coordinate reference system (CRS) for the model.

        When the CRS is set in the configuration, it will return that value.
        If the CRS is set to "auto", it will determine the UTM zone based on the routing subbasins geometry.

        Returns:
             The CRS string, either "auto" or the determined UTM zone.
        """
        crs: str = self.config["crs"]
        if crs == "auto":
            crs: str = self.get_utm_zone(self.model.files["geom"]["routing/subbasins"])
        return crs

    @property
    def simulation_root(self) -> Path:
        """Get the root directory for the SFINCS simulations."""
        folder = self.model.simulation_root / "SFINCS"
        folder.mkdir(exist_ok=True)
        return folder
