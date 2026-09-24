# Waterbodies

Use `setup_waterbodies` to add lakes and reservoirs to your model. In most cases, you can use the defaults. Customize only if you need to (a) provide reservoir command areas, or (b) override reservoir capacity.

## What `setup_waterbodies` produces

After running, the model stores:

- `waterbodies/waterbody_id`: waterbody ID per coarse grid cell (`-1` means no waterbody).
- `waterbodies/sub_waterbody_id`: waterbody ID per subgrid cell (`-1` means no waterbody).
- `waterbodies/command_area`: waterbody ID per coarse grid cell where command areas exist (`-1` means no command area).
- `waterbodies/subcommand_areas`: waterbody ID per subgrid cell where command areas exist (`-1` means no command area).
- `waterbodies/waterbody_data`: a table (GeoDataFrame) with waterbody attributes used by the hydrology model.
- `waterbodies/gdw_checks`: Each GDW dam's location, linked waterbody ID, and any disagreement with HydroLAKES about lake type or location.
- `reports/waterbodies/gdw_checks.csv`: the same checks as a CSV, without geometry.

## Default setup

By default, `setup_waterbodies`:

- Reads waterbodies from the `hydrolakes` dataset.
- Adds attributes from Global Dam Watch (GDW v1.0) and adds missing GDW reservoir outlines where possible.
- Keeps only waterbodies that intersect your model region.
- Converts HydroLAKES waterbody types into GEB types:

  - `1` = `LAKE`
  - `2` = `RESERVOIR`
  - `3` = `LAKE_CONTROL`

- Initializes `volume_flood` to match `volume_total`.

If you do not provide command areas, the command area rasters are still created but filled with `-1` everywhere.

## How GDW is used

GEB keeps HydroLAKES outlines and IDs. It first matches GDW's `HYLAK_ID` to the
HydroLAKES ID. If no lake has that ID, a dam point must lie inside or on the edge of
a lake. Without an ID match, points outside all lakes are not included. GEB does not use the nearest lake.

GEB currently uses HydroLAKES v1.0, while GDW refers to v1.1. The checks mark ID
matches where the dam lies outside the lake (`point_outside_lake`). They also
mark dams linked by location whose GDW lake ID differs from the linked lake ID
(`id_differs`). These links need review.

The waterbody table includes GDW names, dam type, lake-control flag, purpose,
capacity (m3), area (m2), discharge (m3/s), and source information. GDW `YEAR_DAM`
is stored as `gdw_construction_year`. `gdw_count` gives the number of matched dam points.
When several dams match one lake, their individual attributes remain in the check table. GEB does not choose
one dam or sum their capacities.

### Construction year

Reservoirs start operating on **January 1 of their GDW construction year**.
Before then, their cells route water as rivers: the reservoir has no storage,
evaporation, or irrigation supply. At construction, river water is
transferred into the (then empty) reservoir, which then fills from inflow.

Natural lakes stay active in every year. Reservoirs with an unknown construction
year also stay active. Controlled lakes converted to reservoirs follow the
reservoir construction year.

This controls river routing and reservoir storage; it does not reconstruct
land cover for past years. To use construction years, rebuild waterbodies and rerun
spinup and the simulation. Saved states without construction years keep all reservoirs active
until spinup is rerun.

### Reading the dam table

`match_method` shows how each dam was linked:

| Value | Meaning |
| --- | --- |
| `id` | GDW's lake ID matches the HydroLAKES ID. |
| `point_in_polygon` | The dam lies inside or on the edge of exactly one lake. |
| `multiple_lakes` | The dam touches several lakes, so none was chosen. |
| `unmatched` | No lake was found by ID or location. |
| `gdw_polygon` | GEB added a reservoir using its GDW outline. |

`waterbody_id` is the linked lake or added reservoir ID; it is empty for unlinked
dams. `addition_reason` says whether a reservoir was added or why it was skipped.
`gdw_type_conflict` is true only for `lake_control_differs`: GDW identifies
lake control, but HydroLAKES does not classify the lake as controlled.
A barrier alone is a reason to review the lake, not a confirmed type conflict.

### Checking and changing types

`type_check` compares each linked GDW barrier with the **original** HydroLAKES
classification: **1 = natural lake, 2 = reservoir, 3 = controlled lake**.

The GDW fields are `Dam_type` and `Lake_ctrl`, renamed to `dam_type` and
`lake_control` in the dam table. There is no check for the words "controlled lake".
GDW indicates lake control when `Dam_type` is `Lake Control Dam`, or `Lake_ctrl`
is `Yes` or `Enlarged`.

Checks run in the order below. **The first matching row wins.**

| Order | GDW values | HydroLAKES type | `type_check` |
| --- | --- | --- | --- |
| 1 | `Lake_ctrl = Maybe`, whatever `Dam_type` says | 1, 2, or 3 | `uncertain_lake_control` |
| 2 | `Dam_type = Lake Control Dam` **or** `Lake_ctrl = Yes` or `Enlarged` | 1 or 2 | `lake_control_differs` |
| 3 | Same lake-control values as row 2 | 3 | `agree` |
| 4 | None of the rows above apply; any barrier type | 1 | `type_1_with_barrier` |
| 5 | None of the rows above apply; `Dam_type = Dam` | 2 | `agree` |
| 6 | `Dam_type = Dam` and `Lake_ctrl` is empty | 3 | `controlled_lake_with_dam` |
| 7 | Any remaining combination, such as `Lock` with type 2 or 3 | 2 or 3 | `review_barrier_type` |

A HydroLAKES type-3 lake (controlled lake) is modeled as a reservoir when it has exactly one GDW
match labeled `Dam`, an empty lake-control flag, positive finite GDW capacity,
with the dam inside or on the lake edge and no conflicting lake ID. GEB then uses GDW capacity as `volume_total`.
The checks mark the change with `changed_to_reservoir`. Other type-3 lakes retain
`LAKE_CONTROL`, which GEB currently simulates as an ordinary lake.

### Reservoirs missing from HydroLAKES

GEB adds a GDW reservoir when it has no HydroLAKES match, a valid outline
that does not overlap or touch another waterbody, a `Dam` label without a Yes, Maybe, or Enlarged lake-control flag, and positive, finite capacity, area, and discharge. Dam points alone
cannot show which grid cells belong to the reservoir.

Sources: [GDW v1.0 dataset and technical documentation](https://figshare.com/articles/dataset/25988293)
and [GDW publication](https://doi.org/10.1038/s41597-024-03752-9). GDW is licensed
under CC BY 4.0; retain the dataset attribution when sharing derived data.

## Custom setup

You can override parts of the default setup with the following options.

### Command areas

`command_areas` should be a path to a vector file (e.g., GeoPackage) containing polygons with a `waterbody_id` column.

If you provide `command_areas`, GEB will:

- Dissolve command areas by `waterbody_id`.
- Mark any waterbody that has a command area as a reservoir.
- Rasterize command areas to `waterbodies/command_area` and `waterbodies/subcommand_areas`.

Command areas that do not match any reservoir in the current region are removed.

### Custom reservoir capacity

`custom_reservoir_capacity` should be an excel-file ('.xlsx') or csv-file ('.csv').

If you provide `custom_reservoir_capacity`, GEB will override reservoir capacity by matching on `waterbody_id`.

Expected columns in the file:

- `waterbody_id`
- `volume_total` (m3)

## Examples

### Use the defaults (no changes)

```yaml
setup_waterbodies: {}
```

### Add command areas

```yaml
setup_waterbodies:
  command_areas: data/command_areas.gpkg
```

### Override reservoir capacity

```yaml
setup_waterbodies:
  custom_reservoir_capacity: data/custom_reservoir_capacity.csv
```
