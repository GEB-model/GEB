# Waterbody setup

Use `setup_waterbodies` to add lakes and reservoirs. The default settings work
for most models.

## Default setup

GEB reads lakes from HydroLAKES and reservoir data from Global Dam Watch (GDW).

It:

- keeps waterbodies that intersect the model region;
- links GDW dams to HydroLAKES waterbodies;
- adds GDW reservoirs missing from HydroLAKES;
- creates waterbody and command-area grids; and
- stores waterbody data and GDW checks.

The main outputs are:

- `waterbodies/waterbody_id` and `waterbodies/sub_waterbody_id`;
- `waterbodies/command_area` and `waterbodies/subcommand_areas`;
- `waterbodies/waterbody_data`;
- `waterbodies/gdw_checks`; and
- `waterbodies/gdw_checks.csv` in the build report directory
  (`output/build` by default).

Grid cells without a waterbody or command area have the value `-1`.

GDW matching applies these classification rules:

| HydroLAKES type | Result |
| --- | --- |
| Natural lake (`1`) | Becomes a reservoir if the GDW checks below pass; otherwise stays a lake. |
| Reservoir (`2`) | Stays a reservoir. |
| Controlled lake (`3`) | Always becomes a reservoir, even without a GDW match. |

GDW reservoirs missing from HydroLAKES are added separately.

## GDW matching

GEB first matches the GDW lake ID to the HydroLAKES ID. If that fails, the dam
must be inside or on the edge of one lake. GEB does not use the nearest lake.

The `match_method` field shows the result:

| Value | Meaning |
| --- | --- |
| `id` | The GDW and HydroLAKES IDs match. |
| `point_in_polygon` | The dam is inside one lake. |
| `multiple_lakes` | The dam touches several lakes. |
| `unmatched` | No lake was found. |
| `gdw_polygon` | A GDW outline was added as a reservoir. |
| `gdw_point` | A dam was added on one river cell. |

Review rows marked `point_outside_lake` or `id_differs`. The table keeps the GDW
name, type, purpose, construction year, capacity, area, and discharge.

A natural lake becomes a reservoir if at least one GDW point meets all three
conditions:

- `dam_type` is `Dam` or `Lake Control Dam`;
- the point is inside or on the lake boundary; and
- `hydrolakes_id` matches the lake ID or is missing.

Capacity, `lake_control`, construction year, quality, and the number of dams do
not determine the type. Conflicting IDs do not qualify. An unknown-ID point
intersecting several lakes is left unmatched.

Converted lakes keep their HydroLAKES volume unless one qualifying GDW match
provides a positive, finite capacity. Several matched dams do not block type
conversion, but no individual dam's attributes are copied to the waterbody.
The check table records conversions as `changed_to_reservoir`.

GEB can also add an unmatched GDW dam as a reservoir. The dam must intersect a
GDW outline, have type `Dam` or `Lake Control Dam`, and provide positive,
finite capacity, area, and average discharge. These values are required to add
a reservoir because no HydroLAKES values are available.
The outline must not intersect an existing waterbody. If only another GDW
reservoir's outline covers the dam, GEB places it on one river cell rather than
using that other reservoir's outline. Other unmatched dams remain in the check
table; this method does not add standalone river barriers.

Sources: [GDW v1.0](https://figshare.com/articles/dataset/25988293) and the
[GDW publication](https://doi.org/10.1038/s41597-024-03752-9). GDW uses the
CC BY 4.0 licence.

## Construction year

A reservoir starts operating on January 1 of its GDW construction year. Before
that date, its cells behave as river cells. Natural lakes and reservoirs without
a known year are always active.

Rebuild waterbodies and rerun spinup and the simulation to use construction
years. This does not rebuild historic land cover.

## Custom setup

### Command areas

Set `command_areas` to a vector file with polygons and a `waterbody_id` column.
GEB creates the command-area grids and classifies linked waterbodies as
reservoirs, regardless of their original HydroLAKES type. This existing rule is
separate from GDW classification.

```yaml
setup_waterbodies:
  command_areas: data/command_areas.gpkg
```

### Reservoir capacity

Set `custom_reservoir_capacity` to a CSV or Excel file with `waterbody_id` and
`volume_total` (m³) columns.

```yaml
setup_waterbodies:
  custom_reservoir_capacity: data/custom_reservoir_capacity.csv
```

To use all defaults:

```yaml
setup_waterbodies: {}
```

## Saved types

Controlled lakes always use reservoir behavior, including those without GDW
data. Saved `waterbody_type` values are 1 (lake) or 2 (reservoir);
`hydrolakes_type` preserves the original classification. Rebuild
`setup_waterbodies` for existing inputs that still contain type 3.

## GDW attributes

Matching uses `hydrolakes_id` and point geometry. Type conversion uses only
`dam_type`, overlap, and ID agreement. `lake_control` is metadata only.

For a single matched dam, GEB copies `gdw_id`, dam and reservoir names, dam type,
lake-control flag, construction year, main use, capacity, area, average discharge,
quality, source, and polygon source. Fields other than `gdw_id` have a `gdw_`
prefix. The full dam records remain in `waterbodies/gdw_checks`.

Construction year controls activation. Capacity can update converted lakes and
sets the capacity of added reservoirs. Area and average discharge set parameters
for added reservoirs; existing waterbodies retain their HydroLAKES values.
Names, lake-control flag, main use, quality, and source fields do not affect
simulation behavior. GDW's `main_use` does not select reservoir operating rules.
