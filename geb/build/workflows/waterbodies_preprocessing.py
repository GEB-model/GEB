"""Link GDW dams to HydroLAKES lakes and compare their classifications."""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from geb.build.data_catalog import DataCatalog
from geb.hydrology.waterbodies import RESERVOIR

# HydroLAKES v1.0 IDs are below two million. Keep new IDs nearby because routing
# creates arrays indexed by the original waterbody ID.
GDW_ID_OFFSET: int = 2_000_000
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
    """Load lakes and add GDW dam data.

    Args:
        data_catalog: Catalog for reading HydroLAKES and GDW.
        region: Model boundary with a coordinate reference system.
        mode: 'off' returns no waterbodies. Other modes load all waterbodies;
            setup_waterbodies applies the lake/reservoir filter later.

    Returns:
        Waterbody outlines with GDW data, plus each dam's linked lake and any
        differences in their IDs, locations, or lake/reservoir classifications.
        Area, volume, and discharge use m2, m3, and m3/s.
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
    """Add GDW dam data and missing reservoirs to HydroLAKES.

    Match GDW's hydrolakes_id to waterbody_id first. If that ID is not found,
    the dam must lie inside or on the edge of exactly one lake. Keep ID matches
    outside the lake, but flag them for review.

    Type-3 lakes (controlled lakes) become reservoirs only with one matching GDW
    'Dam', no lake-control flag, positive capacity, and no location or ID conflict.
    New reservoirs need non-overlapping outlines and positive, finite capacity,
    area, and discharge.

    Args:
        waterbodies: HydroLAKES polygons with original types and unique IDs.
        barriers: GDW points, including dams outside the model boundary.
        reservoir_shapes: GDW outlines used to add missing reservoirs, if given.
        region_shape: Boundary in the same CRS as waterbodies. With reservoir_shapes,
            keep only dams inside it or linked to a lake or reservoir in the region.

    Returns:
        Updated waterbodies and a table linking each dam to a lake, with any
        differences in IDs, locations, or lake/reservoir classifications.
        If several dams link to one lake, their attributes stay in this table only.

    Raises:
        ValueError: If lake IDs are duplicated, a lake type is unknown, or a new ID
            is already used or too large for int32 rasters.
    """
    if not waterbodies["waterbody_id"].is_unique:
        raise ValueError("Waterbody IDs must be unique.")

    waterbodies = waterbodies.copy().reset_index(drop=True)
    # Keep the original type and volume so changes can be checked later.
    waterbodies["hydrolakes_type"] = waterbodies["waterbody_type"].astype("Int64")
    waterbodies["hydrolakes_volume_total"] = waterbodies["volume_total"]
    waterbodies["waterbody_source"] = "hydrolakes"
    dam_checks: gpd.GeoDataFrame = (
        barriers.to_crs(waterbodies.crs).copy().reset_index(drop=True)
    )
    # A missing ID means this barrier has not been linked to a lake.
    dam_checks["waterbody_id"] = pd.Series(pd.NA, index=dam_checks.index, dtype="Int64")
    dam_checks["match_method"] = "unmatched"
    dam_checks["point_outside_lake"] = False
    dam_checks["hydrolakes_type"] = pd.Series(
        pd.NA, index=dam_checks.index, dtype="Int64"
    )
    dam_checks["type_check"] = "unmatched"

    # ID matches also find dams outside the model region.
    matched_by_id: pd.Series = dam_checks["hydrolakes_id"].isin(
        waterbodies["waterbody_id"]
    )
    dam_checks.loc[matched_by_id, "waterbody_id"] = dam_checks.loc[
        matched_by_id, "hydrolakes_id"
    ]
    dam_checks.loc[matched_by_id, "match_method"] = "id"

    # Without an ID match, accept only points inside or on the edge of exactly one lake.
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

    lakes_by_id: gpd.GeoDataFrame = waterbodies.set_index("waterbody_id")
    matched_dams: gpd.GeoDataFrame = dam_checks.loc[
        dam_checks["waterbody_id"].notna()
    ].fillna({"dam_type": "", "lake_control": ""})
    row_index: int
    for row_index in matched_dams.index:
        dam: pd.Series = matched_dams.loc[row_index]
        lake_id: int = int(dam["waterbody_id"])
        lake_type: int = int(lakes_by_id.at[lake_id, "hydrolakes_type"])
        dam_checks.at[row_index, "hydrolakes_type"] = lake_type
        dam_checks.at[row_index, "point_outside_lake"] = not dam.geometry.intersects(
            lakes_by_id.geometry.loc[lake_id]
        )
        dam_type: str = dam["dam_type"]
        lake_control: str = dam["lake_control"]
        if lake_type not in (1, 2, 3):
            raise ValueError(f"Unknown HydroLAKES type: {lake_type}.")
        # Check lake control first: it is more specific than the barrier label.
        if lake_control == "Maybe":
            type_check: str = "uncertain_lake_control"
        elif dam_type == "Lake Control Dam" or lake_control in ("Yes", "Enlarged"):
            type_check = "agree" if lake_type == 3 else "lake_control_differs"
        elif lake_type == 1:
            # A barrier alone does not make a natural lake a reservoir.
            type_check = "type_1_with_barrier"
        elif dam_type == "Dam" and lake_type == 2:
            type_check = "agree"
        elif dam_type == "Dam" and lake_type == 3 and lake_control == "":
            type_check = "controlled_lake_with_dam"
        else:
            # 'Dam' is GDW's default label, so lake control is still possible.
            type_check = "review_barrier_type"
        dam_checks.at[row_index, "type_check"] = type_check

    # Keep one row per lake. Do not choose or sum attributes from several dams.
    dam_counts: pd.Series = dam_checks["waterbody_id"].value_counts()
    waterbodies["gdw_count"] = (
        waterbodies["waterbody_id"].map(dam_counts).fillna(0).astype("int32")
    )
    # One conflicting barrier is enough to flag the whole lake for review.
    type_conflicts: pd.Series = (
        dam_checks["type_check"]
        .eq("lake_control_differs")
        .groupby(dam_checks["waterbody_id"])
        .any()
    )
    waterbodies["gdw_type_conflict"] = (
        waterbodies["waterbody_id"].map(type_conflicts).fillna(False).astype(bool)
    )
    # Copy attributes only when one dam is linked, so its values are unambiguous.
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

    # Use dam capacity: the operator may not control the full lake volume.
    controlled_lakes_with_dams: pd.Series = (
        single_dam_matches["hydrolakes_type"].eq(3)
        & single_dam_matches["dam_type"].eq("Dam")
        & single_dam_matches["lake_control"].fillna("").eq("")
        & np.isfinite(single_dam_matches["capacity_m3"])
        & single_dam_matches["capacity_m3"].gt(0)
        & ~single_dam_matches["point_outside_lake"]
        & ~single_dam_matches["id_differs"]
    ).fillna(False)
    reservoir_ids: pd.Index = single_dam_matches.index[controlled_lakes_with_dams]
    use_as_reservoir: pd.Series = waterbodies["waterbody_id"].isin(reservoir_ids)
    waterbodies.loc[use_as_reservoir, "waterbody_type"] = RESERVOIR
    waterbodies.loc[use_as_reservoir, "volume_total"] = waterbodies.loc[
        use_as_reservoir, "gdw_capacity_m3"
    ]
    dam_checks["changed_to_reservoir"] = dam_checks["waterbody_id"].isin(reservoir_ids)
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

    Require an unmatched dam whose outline does not touch another waterbody,
    with positive, finite area, capacity, and discharge.
    Skip locks and lake-control dams. Record why each skipped dam was left out.

    Args:
        waterbodies: HydroLAKES waterbodies with GDW attributes.
        dam_checks: Each dam's location, linked lake ID, and any differences
            between the GDW and HydroLAKES classifications.
        reservoir_shapes: GDW reservoir outlines in the model region.

    Returns:
        Waterbodies with new reservoirs, plus the dam table showing which
        reservoirs were added or why they were skipped.
        New IDs are 2,000,000 plus GDW ID. Area, volume, and discharge use m2, m3, and m3/s.

    Raises:
        ValueError: If a new ID is already used or is too large for int32 rasters.
    """
    reservoir_shapes = reservoir_shapes.to_crs(waterbodies.crs)
    reservoir_shapes.set_index("gdw_id", inplace=True)
    dam_checks["addition_reason"] = "already_matched"
    used_ids: set[int] = set(waterbodies["waterbody_id"])
    row_index: int
    # isna() selects barriers with no linked lake. Adding linked ones would duplicate it.
    for row_index in dam_checks.index[dam_checks["waterbody_id"].isna()]:
        gdw_id: int = int(dam_checks.at[row_index, "gdw_id"])
        # A point alone cannot tell us which grid cells form the reservoir.
        if gdw_id not in reservoir_shapes.index:
            dam_checks.at[row_index, "addition_reason"] = "no_polygon_in_region"
            continue
        reservoir_shape: BaseGeometry = reservoir_shapes.geometry.loc[gdw_id]
        if (
            reservoir_shape is None
            or reservoir_shape.is_empty
            or not reservoir_shape.is_valid
            or reservoir_shape.geom_type not in ("Polygon", "MultiPolygon")
        ):
            dam_checks.at[row_index, "addition_reason"] = "invalid_polygon"
            continue
        # Avoid giving the same lake two IDs.
        if waterbodies.sindex.query(reservoir_shape, predicate="intersects").size:
            dam_checks.at[row_index, "addition_reason"] = "overlaps_waterbody"
            continue
        dam_type: str = str(dam_checks.at[row_index, "dam_type"])
        lake_control: str = str(dam_checks.at[row_index, "lake_control"])
        if dam_type != "Dam" or lake_control in ("Yes", "Maybe", "Enlarged"):
            dam_checks.at[row_index, "addition_reason"] = "review_barrier_type"
            continue
        # Routing needs a known, positive capacity, area, and average discharge.
        reservoir_values: pd.Series = dam_checks.loc[
            row_index, ["capacity_m3", "area_m2", "average_discharge_m3_per_s"]
        ].astype(float)
        if not np.isfinite(reservoir_values).all() or not (reservoir_values > 0).all():
            dam_checks.at[row_index, "addition_reason"] = "missing_model_values"
            continue
        # Use a separate ID range while keeping IDs the same across model regions.
        new_id: int = GDW_ID_OFFSET + gdw_id
        if new_id > np.iinfo(np.int32).max or new_id in used_ids:
            raise ValueError(
                f"GDW waterbody ID {new_id} is out of range or already used."
            )
        new_reservoir: gpd.GeoDataFrame = gpd.GeoDataFrame(
            {
                "waterbody_id": [new_id],
                "waterbody_type": [RESERVOIR],
                "volume_total": [reservoir_values["capacity_m3"]],
                "average_area": [reservoir_values["area_m2"]],
                "average_discharge": [reservoir_values["average_discharge_m3_per_s"]],
                "waterbody_source": ["gdw"],
                "gdw_count": [1],
                "gdw_type_conflict": [False],
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
        used_ids.add(new_id)
        dam_checks.at[row_index, "waterbody_id"] = new_id
        dam_checks.at[row_index, "match_method"] = "gdw_polygon"
        dam_checks.at[row_index, "type_check"] = "no_hydrolakes_record"
        dam_checks.at[row_index, "addition_reason"] = "added"
    waterbodies["waterbody_id"] = waterbodies["waterbody_id"].astype("int32")
    waterbodies["waterbody_type"] = waterbodies["waterbody_type"].astype("int32")
    return waterbodies, dam_checks
