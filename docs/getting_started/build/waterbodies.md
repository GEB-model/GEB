# Waterbody setup

Use `setup_waterbodies` to combine HydroLAKES with Global Dam Watch (GDW)
reservoir data and metadata, including purpose and construction year.
Controlled lakes use reservoir behavior; natural lakes become reservoirs when
an intersecting GDW Dam or Lake Control Dam has a matching or missing
HydroLAKES ID. Missing GDW reservoirs are added when the required model data
are available, using one river cell if no outline is available.

Reservoirs operate from their construction year; unknown years mean always
active. Rebuild `setup_waterbodies`, then rerun spinup and simulation.

```yaml
setup_waterbodies: {}
```

## Optional settings

| Setting | Use |
| --- | --- |
| `mode` | `on` (default), `off`, `lakes_only`, or `reservoirs_only` |
| `command_areas` | Irrigation polygons with `waterbody_id`; linked waterbodies become reservoirs |
| `calculate_command_areas` | Derive irrigation areas from river routing |
| `custom_reservoir_capacity` | CSV or Excel file with `waterbody_id` and `volume_total` (m³) |

Review `waterbodies/gdw_checks` or `waterbodies/gdw_checks.csv` in the build
report directory (`output/build`) for matching results. The original
classification is kept in `hydrolakes_type`.

See [Lakes and reservoirs](../../hydrology/waterbodies.md) for storage and release rules.
