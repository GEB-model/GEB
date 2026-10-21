"""Build lake and reservoir inputs from HydroLAKES and GDW."""

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from geb.build.data_catalog import DataCatalog
from geb.build.workflows.river_snapping import (
    SnappingResults,
    snap_point_to_river_network,
)
from geb.hydrology.waterbodies import RESERVOIR

GDW_ID_OFFSET: int = 2_000_000  # Avoid overlap with existing HydroLAKES IDs.
GDW_FIELDS: list[str] = [
    "gdw_id",
    "dam_name",
    "reservoir_name",
    "dam_type",
    "lake_control",
    "construction_year",
    "main_use",
    "capacity_m3",
    "area_m2",
    "average_discharge_m3_per_s",
    "quality",
    "source",
    "polygon_source",
]


def load_and_enrich_waterbodies(
    data_catalog: DataCatalog,
    region: gpd.GeoDataFrame,
    mode: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load HydroLAKES waterbodies and add GDW reservoir data.

    Args:
        data_catalog: Catalog for reading HydroLAKES and GDW.
        region: Model boundary with a coordinate reference system.
        mode: Waterbody mode: 'on', 'off', 'lakes_only', or 'reservoirs_only'.
            Only 'off' is applied here; setup_waterbodies filters the other modes.

    Returns:
        Waterbodies and dam matching results. Capacity, area, and discharge
        use m3, m2, and m3/s.
    """
    region = region.to_crs(4326)
    region_shape: BaseGeometry = region.union_all()
    waterbodies: gpd.GeoDataFrame = data_catalog.fetch("hydrolakes").read(
        bbox=tuple(region.total_bounds),
        columns=[
            "waterbody_id",
            "waterbody_type",
            "volume_total",
            "average_discharge",
            "average_area",
            "geometry",
        ],
    )
    waterbodies = waterbodies[waterbodies.intersects(region_shape)].copy()
    if mode == "off":
        # Keep the column types even when waterbodies are disabled.
        return (
            waterbodies.iloc[:0],
            gpd.GeoDataFrame(geometry=[], crs=4326),
        )

    # Read all dams: a lake in the region may have its dam outside it.
    barriers: gpd.GeoDataFrame = data_catalog.fetch("gdw_barriers").read()
    reservoir_shapes: gpd.GeoDataFrame = data_catalog.fetch("gdw_reservoirs").read(
        bbox=tuple(region.total_bounds)
    )
    reservoir_shapes = reservoir_shapes[reservoir_shapes.intersects(region_shape)]
    return enrich_waterbodies(
        waterbodies,
        barriers,
        reservoir_shapes=reservoir_shapes,
        region_shape=region_shape,
    )


def enrich_waterbodies(
    waterbodies: gpd.GeoDataFrame,
    barriers: gpd.GeoDataFrame,
    reservoir_shapes: gpd.GeoDataFrame | None = None,
    region_shape: BaseGeometry | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Classify waterbodies, add GDW metadata, and add reservoirs missing from HydroLAKES.

    Controlled lakes become reservoirs. Natural lakes become reservoirs when
    an intersecting GDW Dam or Lake Control Dam has a matching or missing
    lake ID. Other attributes do not determine type.

    Args:
        waterbodies: HydroLAKES polygons with original types and unique IDs.
        barriers: GDW points, including dams outside the model boundary.
        reservoir_shapes: GDW outlines used to add missing reservoirs, if given.
        region_shape: Model boundary in the waterbody CRS. When adding reservoirs,
            retain outside dams only if linked to a waterbody or GDW outline.

    Returns:
        Waterbodies and dam matching results. Volumes use m3, areas m2, and
        discharge m3/s. Multiple-dam attributes stay in the matching table.
    """
    waterbodies = waterbodies.copy().reset_index(drop=True)
    # Keep the original type and volume so changes can be checked later.
    waterbodies["hydrolakes_type"] = waterbodies["waterbody_type"].astype("Int64")
    waterbodies["hydrolakes_volume_total"] = waterbodies["volume_total"]
    waterbodies["waterbody_source"] = "hydrolakes"
    dam_checks: gpd.GeoDataFrame = (
        barriers.to_crs(waterbodies.crs).copy().reset_index(drop=True)
    )
    dam_checks["waterbody_id"] = pd.Series(pd.NA, index=dam_checks.index, dtype="Int64")
    dam_checks["match_method"] = "unmatched"
    dam_checks["point_outside_lake"] = False

    matched_by_id: pd.Series = dam_checks["hydrolakes_id"].isin(
        waterbodies["waterbody_id"]
    )
    dam_checks.loc[matched_by_id, "waterbody_id"] = dam_checks.loc[
        matched_by_id, "hydrolakes_id"
    ]
    dam_checks.loc[matched_by_id, "match_method"] = "id"

    point_matches: gpd.GeoDataFrame = gpd.sjoin(
        dam_checks.loc[~matched_by_id, ["geometry"]],
        waterbodies[["waterbody_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    # Repeated dam rows mean the point touches several lakes; do not choose one.
    dams_in_multiple_lakes: pd.Index = point_matches.index[
        point_matches.index.duplicated(keep=False)
    ].unique()
    dam_checks.loc[dams_in_multiple_lakes, "match_method"] = "multiple_lakes"
    point_matches = point_matches.loc[~point_matches.index.isin(dams_in_multiple_lakes)]
    dam_checks.loc[point_matches.index, "waterbody_id"] = point_matches[
        "waterbody_id"
    ].astype("Int64")
    dam_checks.loc[point_matches.index, "match_method"] = "point_in_polygon"

    # IDs may differ between HydroLAKES versions.
    dam_checks["id_differs"] = (
        dam_checks["hydrolakes_id"].notna()
        & dam_checks["waterbody_id"].notna()
        & dam_checks["hydrolakes_id"].ne(dam_checks["waterbody_id"])
    ).fillna(False)

    matched_dams: gpd.GeoDataFrame = dam_checks.loc[dam_checks["waterbody_id"].notna()]
    lake_geometry: gpd.GeoSeries = waterbodies.set_index("waterbody_id").geometry
    matched_lakes: gpd.GeoSeries = gpd.GeoSeries(
        matched_dams["waterbody_id"].map(lake_geometry), crs=waterbodies.crs
    )
    dam_checks.loc[
        matched_dams.index, "point_outside_lake"
    ] = ~matched_dams.geometry.intersects(matched_lakes, align=False)

    dam_counts: pd.Series = dam_checks["waterbody_id"].value_counts()
    # Enrich lake metadata only when one dam provides the attributes.
    single_dam_matches: gpd.GeoDataFrame = dam_checks.loc[
        dam_checks["waterbody_id"].map(dam_counts).eq(1)
    ].set_index("waterbody_id")
    field: str
    for field in GDW_FIELDS:
        if field in single_dam_matches:
            column_name: str = field if field == "gdw_id" else f"gdw_{field}"
            waterbodies[column_name] = waterbodies["waterbody_id"].map(
                single_dam_matches[field]
            )

    qualifying_dams: pd.Series = (
        dam_checks["waterbody_id"].notna()
        & dam_checks["dam_type"].isin(("Dam", "Lake Control Dam"))
        & ~dam_checks["point_outside_lake"]
        & ~dam_checks["id_differs"]
    )
    reservoir_ids: pd.Series = dam_checks.loc[qualifying_dams, "waterbody_id"]
    convert_to_reservoir: pd.Series = waterbodies["hydrolakes_type"].eq(3) | (
        waterbodies["hydrolakes_type"].eq(1)
        & waterbodies["waterbody_id"].isin(reservoir_ids)
    )
    waterbodies.loc[convert_to_reservoir, "waterbody_type"] = RESERVOIR

    # Use GDW capacity when available; otherwise keep the HydroLAKES volume.
    if "gdw_capacity_m3" in waterbodies:
        use_gdw_capacity: pd.Series = (
            convert_to_reservoir
            & waterbodies["waterbody_id"].isin(reservoir_ids)
            & np.isfinite(waterbodies["gdw_capacity_m3"])
            & waterbodies["gdw_capacity_m3"].gt(0)
        )
        if use_gdw_capacity.any():
            # Preserve GDW capacity precision when HydroLAKES uses float32.
            waterbodies["volume_total"] = waterbodies["volume_total"].astype(np.float64)
            waterbodies.loc[use_gdw_capacity, "volume_total"] = waterbodies.loc[
                use_gdw_capacity, "gdw_capacity_m3"
            ]
    waterbodies["waterbody_type"] = waterbodies["waterbody_type"].astype(np.int32)
    changed_ids: pd.Series = waterbodies.loc[convert_to_reservoir, "waterbody_id"]
    dam_checks["changed_to_reservoir"] = dam_checks["waterbody_id"].isin(changed_ids)
    if reservoir_shapes is None:
        return waterbodies, dam_checks

    if region_shape is not None:
        # Keep outside dams if they belong to a lake or reservoir in this region.
        keep_dams: pd.Series = (
            dam_checks["waterbody_id"].notna()
            | dam_checks.intersects(region_shape)
            | dam_checks["gdw_id"].isin(reservoir_shapes["gdw_id"])
        )
        dam_checks = dam_checks.loc[keep_dams].copy()
    return _add_missing_gdw_reservoirs(waterbodies, dam_checks, reservoir_shapes)


def _add_missing_gdw_reservoirs(
    waterbodies: gpd.GeoDataFrame,
    dam_checks: gpd.GeoDataFrame,
    reservoir_shapes: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Add GDW reservoirs missing from HydroLAKES.

    Use the GDW outline when available, otherwise use the dam point. Require
    valid area, capacity, and discharge, and no overlap with existing waterbodies.
    Record why other points are excluded.

    Args:
        waterbodies: HydroLAKES waterbodies with GDW attributes.
        dam_checks: Dam locations, matched lake IDs, and matching results.
        reservoir_shapes: GDW reservoir outlines in the model region.

    Returns:
        Waterbodies and dam matching results. Area, capacity, and discharge
        use m2, m3, and m3/s. New IDs are 2,000,000 plus GDW ID.
    """
    reservoir_shapes = reservoir_shapes.to_crs(waterbodies.crs)
    reservoir_shapes.set_index("gdw_id", inplace=True)
    dam_checks["addition_reason"] = "already_matched"
    row_index: int
    for row_index in dam_checks.index[dam_checks["waterbody_id"].isna()]:
        gdw_id: int = int(dam_checks.at[row_index, "gdw_id"])
        reservoir_shape: BaseGeometry = (
            reservoir_shapes.geometry.loc[gdw_id]
            if gdw_id in reservoir_shapes.index
            else dam_checks.geometry.loc[row_index]
        )
        if not reservoir_shape.is_valid:
            dam_checks.at[row_index, "addition_reason"] = "invalid_polygon"
            continue
        # Avoid giving the same lake two IDs.
        if waterbodies.sindex.query(reservoir_shape, predicate="intersects").size:
            dam_checks.at[row_index, "addition_reason"] = "overlaps_waterbody"
            continue
        dam_type: str = str(dam_checks.at[row_index, "dam_type"])
        if dam_type not in ("Dam", "Lake Control Dam"):
            dam_checks.at[row_index, "addition_reason"] = "review_barrier_type"
            continue
        reservoir_values: pd.Series = dam_checks.loc[
            row_index, ["capacity_m3", "area_m2", "average_discharge_m3_per_s"]
        ].astype(float)
        if not np.isfinite(reservoir_values).all() or not (reservoir_values > 0).all():
            dam_checks.at[row_index, "addition_reason"] = "missing_model_values"
            continue
        # Keep new IDs unique and consistent across model regions.
        new_id: int = GDW_ID_OFFSET + gdw_id
        new_reservoir: gpd.GeoDataFrame = gpd.GeoDataFrame(
            {
                "waterbody_id": [new_id],
                "waterbody_type": [RESERVOIR],
                "volume_total": [reservoir_values["capacity_m3"]],
                "average_area": [reservoir_values["area_m2"]],
                "average_discharge": [reservoir_values["average_discharge_m3_per_s"]],
                "waterbody_source": ["gdw"],
            },
            geometry=[reservoir_shape],
            crs=waterbodies.crs,
        )
        field: str
        for field in GDW_FIELDS:
            if field in dam_checks:
                column_name: str = field if field == "gdw_id" else f"gdw_{field}"
                new_reservoir[column_name] = dam_checks.at[row_index, field]
        waterbodies = gpd.GeoDataFrame(
            pd.concat([waterbodies, new_reservoir], ignore_index=True),
            crs=waterbodies.crs,
        )
        dam_checks.at[row_index, "waterbody_id"] = new_id
        dam_checks.at[row_index, "match_method"] = (
            "gdw_point" if reservoir_shape.geom_type == "Point" else "gdw_polygon"
        )
        dam_checks.at[row_index, "addition_reason"] = "added"
    waterbodies["waterbody_id"] = waterbodies["waterbody_id"].astype("int32")
    waterbodies["waterbody_type"] = waterbodies["waterbody_type"].astype("int32")
    return waterbodies, dam_checks


def snap_waterbody_points(
    waterbodies: gpd.GeoDataFrame,
    waterbody_id: xr.DataArray,
    rivers: gpd.GeoDataFrame,
    upstream_area_grid: xr.DataArray,
    upstream_area_subgrid: xr.DataArray,
) -> dict[int, str]:
    """Place reservoirs without outlines in river cells, skipping unsuitable locations.

    Args:
        waterbodies: Reservoir data and geometry, updated in place.
        waterbody_id: Waterbody IDs on the model grid, updated in place.
        rivers: River segments with model-cell coordinates and upstream areas.
        upstream_area_grid: Model grid drainage area (m2).
        upstream_area_subgrid: Original drainage area (m2).

    Returns:
        Skipped GDW IDs and their reasons. Skipped reservoirs are removed in place.
    """
    skipped: dict[int, str] = {}
    row_index: int
    reservoir: pd.Series
    for row_index, reservoir in waterbodies.loc[
        waterbodies.geometry.geom_type == "Point"
    ].iterrows():
        snapped: SnappingResults | None = snap_point_to_river_network(
            point=reservoir.geometry,
            rivers=rivers,
            upstream_area_grid=upstream_area_grid,
            upstream_area_subgrid=upstream_area_subgrid,
        )
        if snapped is None:
            skipped[int(reservoir.gdw_id)] = "could_not_snap"
            waterbodies.drop(index=row_index, inplace=True)
            continue
        column: int
        row: int
        column, row = snapped.snapped_grid_pixel_xy
        if waterbody_id.values[row, column] != -1:
            skipped[int(reservoir.gdw_id)] = "occupied_waterbody_cell"
            waterbodies.drop(index=row_index, inplace=True)
            continue
        # The cell locates storage; GDW supplies the actual reservoir area.
        waterbody_id.values[row, column] = reservoir.waterbody_id
        waterbodies.at[row_index, "geometry"] = Point(
            *snapped.snapped_grid_pixel_lonlat
        )
    return skipped
