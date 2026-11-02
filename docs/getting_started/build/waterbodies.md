# Waterbody setup

Use `setup_waterbodies` to add lakes and reservoirs from HydroLAKES and
Global Dam Watch (GDW). GDW adds dam information, including purpose and
construction year, and reservoirs missing from HydroLAKES.

GEB treats controlled lakes as reservoirs. A natural lake also becomes a
reservoir if GDW places a dam in it and the lake ID matches or is missing.
The GDW dam type must be `Dam` or `Lake Control Dam`.

Reservoirs start operating in their construction year. Those without a known
year operate throughout the simulation. Rebuild `setup_waterbodies`, then
rerun spinup and simulation.

```yaml
setup_waterbodies: {}
```

## Optional settings

| Setting | Use |
| --- | --- |
| `mode` | `on` (default), `off`, `lakes_only`, or `reservoirs_only` |
| `command_areas` | Irrigation areas with a `waterbody_id` column; these waterbodies become reservoirs |
| `calculate_command_areas` | Calculate irrigation areas from the river network |
| `custom_reservoir_capacity` | CSV or Excel file with `waterbody_id` and `volume_total` (m³) |

Check `waterbodies/gdw_checks.csv` in the build report folder for the GDW
matches and any reservoirs that were left out. Reservoirs without outlines
are skipped if no suitable river cell is available.

See [Lakes and reservoirs](../../hydrology/waterbodies.md) for storage and release rules.
