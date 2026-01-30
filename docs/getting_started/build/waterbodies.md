# Waterbodies

Use `setup_waterbodies` to add lakes and reservoirs to your model. In most cases, you can use the defaults. Customize only if you need to (a) provide reservoir command areas, or (b) override reservoir capacity.

## What `setup_waterbodies` produces

After running, the model stores:

- `waterbodies/waterbody_id`: waterbody ID per coarse grid cell (`-1` means no waterbody).
- `waterbodies/sub_waterbody_id`: waterbody ID per subgrid cell (`-1` means no waterbody).
- `waterbodies/command_area`: waterbody ID per coarse grid cell where command areas exist (`-1` means no command area).
- `waterbodies/subcommand_areas`: waterbody ID per subgrid cell where command areas exist (`-1` means no command area).
- `waterbodies/waterbody_data`: a table (GeoDataFrame) with waterbody attributes used by the hydrology model.

## Default setup

By default, `setup_waterbodies`:

- Reads waterbodies from the `hydrolakes` dataset.
- Keeps only waterbodies that intersect your model region.
- Converts HydroLAKES waterbody types into GEB types:

  - `1` = `LAKE`
  - `2` = `RESERVOIR`
  - `3` = `LAKE_CONTROL`

- Initializes `volume_flood` to match `volume_total`.

If you do not provide command areas, the command area rasters are still created but filled with `-1` everywhere.

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

`custom_reservoir_capacity` should be a path to a file readable by GeoPandas.

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
  custom_reservoir_capacity: data/custom_reservoir_capacity.gpkg
```

