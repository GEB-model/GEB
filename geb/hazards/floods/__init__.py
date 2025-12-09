"""Class to setup, run, and post-process the SFINCS hydrodynamic model."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry.point import Point

from geb.module import Module
from geb.types import ArrayFloat32, TwoDArrayInt32
from geb.workflows.io import read_geom

from ...hydrology.landcovers import OPEN_WATER as OPEN_WATER, SEALED as SEALED
from ...workflows.io import (
    read_dict,
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


def create_river_graph(
    rivers: gpd.GeoDataFrame, subbasins: gpd.GeoDataFrame
) -> nx.DiGraph:
    """Creates a directed graph representation of the river network.

    Args:
        rivers: A GeoDataFrame containing the river network with downstream IDs.
        subbasins: A GeoDataFrame containing the subbasins.

    Returns:
        A directed graph (networkx DiGraph) representing the river network.
    """
    rivers_without_outflow_basin = rivers[~rivers["is_downstream_outflow_subbasin"]]

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


class Floods(Module):
    """The class that implements all methods to setup, run, and post-process hydrodynamic flood models.

    Args:
        model: The GEB model instance.
        n_timesteps: The number of timesteps to keep in memory for discharge calculations (default is 10).
    """

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

        self.DEM_config: list[dict[str, Any]] = read_dict(
            self.model.files["dict"]["hydrodynamics/DEM_config"]
        )

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
        return "floods"

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
        region: gpd.GeoDataFrame | None = None,
        coastal: bool = False,
        coastal_only: bool = False,
        low_elevation_coastal_zone_mask: gpd.GeoDataFrame | None = None,
        coastal_boundary_exclude_mask: gpd.GeoDataFrame | None = None,
        initial_water_level: float | None = 0.0,
    ) -> SFINCSRootModel:
        """Builds or reads a SFINCS model without any forcing.

        Before using this model, forcing must be set.

        When the model already exists and force_overwrite is False, the existing model is read.

        Args:
            name: Name of the SFINCS model (used for the model root directory).
            region: The region to build the SFINCS model for. If None, the entire model region is used.
            coastal: Whether to only include coastal areas in the model.
            coastal_only: Whether to only include the low elevation coastal zone in the model.
            low_elevation_coastal_zone_mask: A GeoDataFrame defining the low elevation coastal zone to set as active cells.
            coastal_boundary_exclude_mask: GeoDataFrame defining the areas to exclude from the coastal model boundary cells.
            initial_water_level: The initial water level to initiate the model. SFINCS fills all cells below this level with water.

        Returns:
            The built or read SFINCSRootModel instance.
        """
        sfincs_model = SFINCSRootModel(self.model.simulation_root, name)
        if self.config["force_overwrite"] or not sfincs_model.exists():
            for entry in self.DEM_config:
                entry["elevtn"] = read_zarr(
                    self.model.files["other"][entry["path"]]
                ).to_dataset(name="elevtn")

            if region is None:
                region = read_geom(self.model.files["geom"]["routing/subbasins"])

            rivers = self.model.hydrology.routing.rivers
            if coastal_only:
                rivers = rivers[rivers.intersects(region.union_all())]

            sfincs_model.build(
                region=region,
                DEMs=self.DEM_config,
                rivers=rivers,
                discharge=self.discharge_spinup_ds,
                river_width_alpha=self.model.hydrology.grid.decompress(
                    self.model.var.river_width_alpha
                ),
                river_width_beta=self.model.hydrology.grid.decompress(
                    self.model.var.river_width_beta
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
                if "parameters"
                in self.model.config["hydrology"]["routing"]["river_depth"]
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
            )
        else:
            sfincs_model.read()

        return sfincs_model

    def set_forcing(
        self,
        sfincs_model: SFINCSRootModel,
        start_time: datetime,
        end_time: datetime,
    ) -> SFINCSSimulation:
        """Sets the forcing for a SFINCS simulation.

        Depending on the forcing method in the config, either headwater point discharge
        or precipitation is used as forcing.

        Args:
            sfincs_model: The SFINCSRootModel instance to create the simulation from.
            start_time: The start time of the flood event.
            end_time: The end time of the flood event.

        Returns:
            The created SFINCSSimulation instance with the forcing set.

        Raises:
            ValueError: If the forcing method is unknown.
        """
        # Save the flood depth to a zarr file
        sfincs_simulation_name: str = f"{start_time.strftime(format='%Y%m%dT%H%M%S')} - {end_time.strftime(format='%Y%m%dT%H%M%S')}"
        if self.model.multiverse_name:
            sfincs_simulation_name: str = (
                self.model.multiverse_name + "/" + sfincs_simulation_name
            )

        simulation: SFINCSSimulation = sfincs_model.create_simulation(
            simulation_name=sfincs_simulation_name,
            start_time=start_time,
            end_time=end_time,
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
        assert pd.to_datetime(forcing_grid.time.values[-1]).to_pydatetime() >= end_time
        assert pd.to_datetime(forcing_grid.time.values[0]).to_pydatetime() <= start_time

        forcing_grid: xr.DataArray = forcing_grid.rio.write_crs(self.model.crs)

        if self.config["forcing_method"] == "headwater_points":
            simulation.set_headwater_forcing_from_grid(
                discharge_grid=forcing_grid,
            )

        elif self.config["forcing_method"] == "accumulated_runoff":
            river_ids: TwoDArrayInt32 = self.hydrology.grid.load(
                self.model.files["grid"]["routing/river_ids"], compress=False
            )
            basin_ids: TwoDArrayInt32 = self.hydrology.grid.load(
                self.model.files["grid"]["routing/basin_ids"], compress=False
            )
            simulation.set_accumulated_runoff_forcing(
                runoff_m=forcing_grid,
                river_network=self.model.hydrology.routing.river_network,
                river_ids=river_ids,
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

        return simulation

    def run_single_event(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Runs a single flood event using the SFINCS model.

        Also updates the flood status of households in the model based on the flood depth results.

        Args:
            start_time: The start time of the flood event.
            end_time: The end time of the flood event.
        """
        subbasins = read_geom(self.model.files["geom"]["routing/subbasins"])
        rivers = self.model.hydrology.routing.rivers

        river_graph = create_river_graph(rivers, subbasins)
        grouped_subbasins = group_subbasins(
            river_graph=river_graph,
            max_area_m2=1e20,  # very large to force single group only
        )

        assert len(grouped_subbasins) == 1, "currently only single group supported"
        for group_id, group in grouped_subbasins.items():
            group = set(group) | set(
                rivers.loc[rivers.index.isin(group)]["downstream_ID"]
            )
            subbasins_group = subbasins[subbasins.index.isin(group)]

            sfincs_root_model = self.build(
                f"group_{group_id}", subbasins_group
            )  # build or read the model
            sfincs_simulation = self.set_forcing(  # set the forcing
                sfincs_root_model, start_time, end_time
            )
            self.model.logger.info(
                f"Running SFINCS for {self.model.current_time}..."
            )  # log the start of the simulation

            sfincs_simulation.run(
                ncpus=self.config.get("SFINCS", {}).get("ncpus", "auto"),
                gpu=self.config.get("SFINCS", {}).get("gpu", "auto"),
            )  # run the simulation

        flood_depth: xr.DataArray = sfincs_simulation.read_max_flood_depth(
            self.config["minimum_flood_depth"]
        )  # read the flood depth results

        filename = (
            self.model.output_folder / "flood_maps" / (sfincs_simulation.name + ".zarr")
        )

        flood_depth: xr.DataArray = write_zarr(
            da=flood_depth,
            path=filename,
            crs=flood_depth.rio.crs,
        )  # save the flood depth to a zarr file

        # This check is done to compute damages (using ERA5) only after multiverse is finished
        if self.model.multiverse_name is None:
            print("Multiverse no longer active, now compute flood damages...")
            self.model.agents.households.flood(flood_depth=flood_depth)

    def get_return_period_maps(self) -> None:
        """Generates flood maps for specified return periods using the SFINCS model."""
        # close the zarr store
        if hasattr(self.model, "reporter"):
            self.model.reporter.variables["discharge_daily"].close()

        # load model settings
        coastal_only = self.config["coastal_only"]

        # load the subbasin geometry for the model domain
        subbasins = read_geom(self.model.files["geom"]["routing/subbasins"])
        coastal = subbasins["is_coastal_basin"].any()

        # if coastal load files
        if coastal:
            # Load mask of lower elevation coastal zones to activate cells for the different sfincs model regions
            low_elevation_coastal_zone_mask = read_geom(
                self.model.files["geom"]["coastal/low_elevation_coastal_zone_mask"]
            )

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

            # filter on coastal subbasins only
            if coastal_only:
                subbasins = subbasins[subbasins["is_coastal_basin"]]

            # merge region and lower elevation coastal zone mask in a single shapefile
            model_domain = subbasins.union_all().union(
                low_elevation_coastal_zone_mask.union_all()
            )

            # domain to gpd.GeoDataFrame
            model_domain = gpd.GeoDataFrame(
                geometry=[model_domain], crs=low_elevation_coastal_zone_mask.crs
            )
            model_name = "coastal_region"

            # load location and offset for coastal water level forcing
            locations = (
                read_geom(self.model.files["geom"]["gtsm/stations_coast_rp"])
                .rename(columns={"station_id": "stations"})
                .set_index("stations")
            )

            offset = xr.open_dataarray(
                self.model.files["other"][
                    "coastal/global_ocean_mean_dynamic_topography"
                ]
            ).rio.write_crs("EPSG:4326")

        else:
            model_domain = subbasins
            coastal_boundary_exclude_mask = None
            low_elevation_coastal_zone_mask = None
            initial_water_level = None
            model_name = "inland_region"
            locations = gpd.GeoDataFrame()
            offset = xr.DataArray()

        sfincs_root_model: SFINCSRootModel = self.build(
            name=model_name,
            region=model_domain,
            coastal=coastal,
            coastal_only=coastal_only,
            coastal_boundary_exclude_mask=coastal_boundary_exclude_mask,
            low_elevation_coastal_zone_mask=low_elevation_coastal_zone_mask,
            initial_water_level=initial_water_level,
        )

        if not coastal_only:
            sfincs_root_model.estimate_discharge_for_return_periods(
                discharge=self.discharge_spinup_ds,
                rivers=self.model.hydrology.routing.rivers,
                return_periods=self.config["return_periods"],
            )

        for return_period in self.config["return_periods"]:
            print(
                f"Estimated discharge for return period {return_period} years for all rivers."
            )

            simulation: MultipleSFINCSSimulations = (
                sfincs_root_model.create_simulation_for_return_period(
                    return_period,
                    coastal=coastal,
                    coastal_only=coastal_only,
                    locations=locations,
                    offset=offset,
                )
            )
            simulation.run(
                ncpus=self.config.get("SFINCS", {}).get("ncpus", "auto"),
                gpu=self.config.get("SFINCS", {}).get("gpu", "auto"),
            )
            flood_depth_return_period: xr.DataArray = simulation.read_max_flood_depth(
                self.config["minimum_flood_depth"]
            )

            write_zarr(
                flood_depth_return_period,
                self.model.output_folder / "flood_maps" / f"{return_period}.zarr",
                crs=flood_depth_return_period.rio.crs,
            )

    def run(self, event: dict[str, Any]) -> None:
        """Runs the SFINCS model for a given flood event.

        Args:
            event: A dictionary containing the flood event details, including 'start_time' and 'end_time'.
        """
        start_time = event["start_time"]
        end_time = event["end_time"]

        if self.model.config["hazards"]["floods"]["flood_risk"]:
            raise NotImplementedError(
                "Flood risk calculations are not yet implemented. Need to adapt old calculations to new flood model."
            )

        else:
            self.run_single_event(start_time, end_time)

    def save_discharge(self) -> None:
        """Saves the current discharge for the current timestep.

        SFINCS is run at the end of an event rather than at the beginning. Therefore,
        we need to trace back the conditions at the beginning of the event. This function
        saves the current discharge at the beginning of each timestep,
        so it can be used later when setting up the SFINCS model.
        """
        self.var.discharge_per_timestep.append(
            self.hydrology.grid.var.discharge_m3_s_per_substep
        )  # this is a deque, so it will automatically remove the oldest discharge

    def save_runoff_m(self) -> None:
        """Saves the current runoff for the current timestep."""
        self.var.runoff_m_per_timestep.append(
            self.model.hydrology.grid.var.total_runoff_m
        )  # this is a deque, so it will automatically remove the oldest runoff

    @property
    def discharge_spinup_ds(self) -> xr.DataArray:
        """Open the discharge datasets from the model output folder.

        Returns:
            The discharge data array after spinup period.

        Raises:
            ValueError: If there is not enough data available for reliable spinup.
        """
        da: xr.DataArray = read_zarr(
            self.model.output_folder
            / "report"
            / "spinup"
            / "hydrology.routing"
            / "discharge_daily.zarr"
        )

        start_time = pd.to_datetime(da.time[0].item()) + pd.DateOffset(years=10)
        da: xr.DataArray = da.sel(time=slice(start_time, da.time[-1]))

        # make sure there is at least 20 years of data
        if len(da.time) == 0 or len(da.time.groupby(da.time.dt.year).groups) < 20:
            raise ValueError(
                """Not enough data available for reliable spinup, should be at least 20 years of data left.
                Please run the model for at least 30 years (10 years of data is discarded)."""
            )

        return da

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
