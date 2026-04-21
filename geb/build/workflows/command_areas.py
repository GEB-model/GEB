"""Workflow helpers for waterbody and command-area setup."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

from geb.geb_types import TwoDArrayInt32
from geb.hydrology.waterbodies import RESERVOIR

COMMAND_AREA_NODATA = -1
COMMAND_AREA_DOMINANCE_RATIO = 3
MAX_UPSTREAM_HOPS = 30


def build_downstream_chain(
    start_id: int,
    next_seg: dict[int, float],
) -> list[int]:
    """Build the downstream segment chain from a starting river segment.

    Follow ``downstream_ID`` links from a starting segment along the main stem
    of the routing network.

    Args:
        start_id: Identifier of the starting river segment.
        next_seg: Mapping from segment identifier to its direct downstream
            segment.

    Returns:
        Ordered list of downstream segment identifiers, starting from the first
        downstream segment of ``start_id``.
    """
    chain: list[int] = []
    current = start_id
    seen: set[int] = set()

    while True:
        nxt = next_seg.get(current, np.nan)
        if pd.isna(nxt) or nxt not in next_seg or nxt in seen:
            break

        nxt_int = int(nxt)
        chain.append(nxt_int)
        seen.add(nxt_int)
        current = nxt_int

    return chain


def derive_command_areas_from_routing(
    waterbodies: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    basin_ids: xr.DataArray,
    grid_mask: xr.DataArray,
    subgrid_mask: xr.DataArray,
    subgrid_factor: int,
    river_graph: nx.DiGraph,
) -> tuple[TwoDArrayInt32, TwoDArrayInt32]:
    """Derive reservoir command-area rasters from the routing network.

    Reservoirs are used as command-area sources. River segments intersecting a
    reservoir inherit that reservoir's ``waterbody_id``, after which reservoir
    influence is propagated through the routing network. When multiple
    reservoirs compete for the same segment, an existing assignment is only
    retained if the previously assigned reservoir is substantially larger than
    the new candidate.

    Args:
        waterbodies: Waterbody geometries and attributes. Must include
            ``waterbody_id``, ``waterbody_type``, ``volume_total``, and
            ``geometry``.
        rivers: River routing geometries indexed by COMID and containing a
            ``downstream_ID`` column.
        basin_ids: Grid of routing basin identifiers matching river COMIDs.
        grid_mask: Grid mask used to construct the output command-area raster.
        subgrid_mask: Subgrid mask used to construct the output subcommand-area
            raster.
        subgrid_factor: Integer refinement factor between grid and subgrid.
        river_graph: Directed river graph for upstream traversal.

    Returns:
        Tuple containing:
            - command-area raster on the model grid
            - command-area raster on the model subgrid
    """
    rivers = rivers.copy()

    reservoir_mask = waterbodies["waterbody_type"] == RESERVOIR
    reservoirs = waterbodies.loc[reservoir_mask].copy()

    if reservoirs.crs != rivers.crs:
        reservoirs = reservoirs.to_crs(rivers.crs)

    next_seg = rivers["downstream_ID"].to_dict()
    rivers["all_downstream_ids"] = [
        build_downstream_chain(comid, next_seg) for comid in rivers.index
    ]

    rivers = gpd.sjoin(
        rivers,
        reservoirs[["waterbody_id", "geometry"]],
        how="left",
        predicate="intersects",
    )
    rivers = rivers.reset_index().rename(columns={"index": "COMID"})

    rivers = rivers.merge(
        reservoirs[["waterbody_id", "volume_total"]],
        on="waterbody_id",
        how="left",
    )

    rivers = rivers.sort_values("volume_total", ascending=False).drop_duplicates(
        subset="COMID",
        keep="first",
    )
    rivers = rivers.set_index("COMID").drop(columns=["index_right"])

    rivers["waterbody_id_propagated"] = np.nan
    has_reservoir = rivers["waterbody_id"].notna()

    rivers["n_downstream"] = rivers["all_downstream_ids"].str.len()
    reservoir_segments = rivers.loc[has_reservoir].sort_values(
        "n_downstream",
        ascending=False,
    )

    wb_volume = reservoirs.set_index("waterbody_id")["volume_total"]

    for comid, row in reservoir_segments.iterrows():
        wb_new = row["waterbody_id"]
        vol_new = wb_volume.get(wb_new, np.nan)
        segs = [comid] + row["all_downstream_ids"]

        for seg in segs:
            current_id = rivers.at[seg, "waterbody_id_propagated"]

            if pd.isna(current_id):
                rivers.at[seg, "waterbody_id_propagated"] = wb_new
                continue

            vol_old = wb_volume.get(current_id, np.nan)

            if pd.isna(vol_old) and not pd.isna(vol_new):
                rivers.at[seg, "waterbody_id_propagated"] = wb_new
                continue

            if pd.isna(vol_new):
                continue

            if vol_old >= COMMAND_AREA_DOMINANCE_RATIO * vol_new:
                continue

            rivers.at[seg, "waterbody_id_propagated"] = wb_new

    river_graph_rev = river_graph.reverse(copy=False)

    only_reservoir_rivers = rivers["waterbody_id_propagated"].dropna()
    unique_wb_ids = only_reservoir_rivers.unique()
    volumes_for_used = wb_volume.reindex(unique_wb_ids)
    wb_ids_sorted = volumes_for_used.sort_values().index.to_list()

    for wb_id in wb_ids_sorted:
        seed_comids = only_reservoir_rivers[
            only_reservoir_rivers == wb_id
        ].index.to_list()

        upstream_comids: set[int] = set()

        for sid in seed_comids:
            if sid not in river_graph_rev:
                continue

            lengths = nx.single_source_shortest_path_length(
                river_graph_rev,
                sid,
                cutoff=MAX_UPSTREAM_HOPS,
            )
            upstream_comids |= {node for node, dist in lengths.items() if dist > 0}

        for cid in upstream_comids:
            if cid in rivers.index and pd.isna(
                rivers.at[cid, "waterbody_id_propagated"]
            ):
                rivers.at[cid, "waterbody_id_propagated"] = wb_id

    smap_extended = rivers["waterbody_id_propagated"]
    flat_ids = basin_ids.values.ravel()
    mapped_flat = smap_extended.reindex(flat_ids).to_numpy()
    mapped_flat[np.isnan(mapped_flat)] = COMMAND_AREA_NODATA

    mapped_reservoirs_grid = mapped_flat.reshape(basin_ids.shape).astype(np.int32)
    mapped_reservoirs_subgrid = mapped_reservoirs_grid.repeat(
        subgrid_factor,
        axis=0,
    ).repeat(
        subgrid_factor,
        axis=1,
    )

    return mapped_reservoirs_grid, mapped_reservoirs_subgrid
