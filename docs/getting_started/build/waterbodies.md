# Waterbody setup

Use `setup_waterbodies` to add lakes and reservoirs. The default settings work
for most models.

## Default setup

GEB reads lakes from HydroLAKES and dam data from Global Dam Watch (GDW). It:

- keeps waterbodies inside the model region;
- links GDW dams to HydroLAKES waterbodies;
- adds missing GDW reservoirs when possible;
- creates waterbody and command-area grids; and
- stores waterbody data and GDW checks.

The main outputs are:

- `waterbodies/waterbody_id` and `waterbodies/sub_waterbody_id`;
- `waterbodies/command_area` and `waterbodies/subcommand_areas`;
- `waterbodies/waterbody_data`;
- `waterbodies/gdw_checks`; and
- `reports/waterbodies/gdw_checks.csv`.

Grid cells without a waterbody or command area have the value `-1`.

HydroLAKES uses these types:

- `1`: natural lake;
- `2`: reservoir; and
- `3`: controlled lake.

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

GEB changes a controlled lake to a reservoir when one suitable GDW dam is linked
to it and GDW provides a valid capacity. The dam must be inside the lake and its
lake ID must not conflict. The check table marks this as
`changed_to_reservoir`.

GEB can also add an unmatched GDW dam as a reservoir. It needs a valid capacity,
area, discharge, and either an outline or a usable river cell. Other unmatched
dam points are handled by `setup_weirs`.

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
GEB joins these areas to reservoirs and creates the command-area grids.

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

## Weirs and gates

River weirs and gates from both GDW and AMBER are enabled by default. To run
without them, set the following in `model.yml`:

```yaml
hydrology:
  routing:
    weirs: false
```

This runtime switch does not require rebuilding inputs. Rerun spinup and the
simulation with the same setting. Set `weirs: true` to enable them again.
Lakes and reservoirs, including AMBER's lake classification, are unaffected.

`setup_weirs` places unmatched GDW barriers on rivers. It skips points that are
already linked, use an occupied cell, or cannot be placed on a valid river link.

| GDW type | Gate | Height used when GDW height is missing |
| --- | --- | --- |
| Dam or Lake Control Dam | Yes | Bankfull depth + 1 m |
| Sluice | Yes | Half the bankfull depth |
| Other types | No | Half the bankfull depth |

`routing/weir_height_m` stores positive known heights (m), zero for no
structure, -1 for bankfull depth + 1 m, and -2 for half bankfull depth.
The two negative markers are resolved into physical heights at runtime,
when the river's bankfull depth is available. Rebuild `setup_weirs` when
updating older inputs.

Fixed barriers allow flow above their crest. Gates start closed and open or
close based on the upstream water depth. The default limits are 90% and 70% of
the crest height:

```yaml
hydrology:
  routing:
    gate_opening_level_fraction: 0.9
    gate_closing_level_fraction: 0.7
```

Gate thresholds can only be configured as fractions of each structure's crest
height. Fractions are dimensionless and must satisfy
`0 <= gate_closing_level_fraction < gate_opening_level_fraction < 1`.
The resulting depths are metres above the upstream river bed. Between the
thresholds, gates retain their previous state.

Fixed-depth settings `open_depth_m` and `close_depth_m` in `setup_weirs` are no
longer supported. Remove them from existing build files and configure the
fractions in `model.yml` instead. Existing fixed-depth input grids are ignored.

Each run writes `output/<run>/weir_heights.csv`. It lists the chosen heights,
gate settings, and `gate_report_index` values. Use a gate's index
to select its hourly routing diagnostics.

After changing build settings, rebuild `setup_weirs` and rerun spinup and the
simulation. Changes to runtime limits only need a new spinup and simulation.

## Controlled lakes without GDW data

The build classifies controlled lakes as lakes unless a suitable linked GDW
dam qualifies them as reservoirs. A missing GDW match does not justify
reservoir classification. Saved `waterbody_type` values are 1 (lake) or 2 (reservoir);
`hydrolakes_type` preserves the original source classification. Rebuild
`setup_waterbodies` for existing inputs that still contain type 3.
