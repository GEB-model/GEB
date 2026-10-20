# Waterbody setup

Use `setup_waterbodies` to add lakes and reservoirs. For river barriers, see
[Weirs and gates](../../hydrology/weirs.md).

```yaml
setup_waterbodies: {}
```

## Data sources

- **HydroLAKES:** lake outlines, size, and volume.
- **Global Dam Watch (GDW):** dams, reservoir capacity, purpose, and construction year.
- **AMBER:** European barriers, including dams and weirs.

## Lakes or reservoirs?

GEB treats lakes and reservoirs differently: lake outflow follows the water
level, while reservoir releases follow operating rules. The build therefore
needs to decide which type to use for each waterbody.

HydroLAKES provides the starting classification, with three source types:

| Type | Meaning |
| --- | --- |
| 1 | Natural lake |
| 2 | Reservoir |
| 3 | Controlled lake |

A controlled lake is a lake whose water level is regulated by a structure.
This label alone does not tell GEB whether to use reservoir operating rules.
GEB checks GDW and AMBER dam data to make that decision.

GEB stores only two model types: **1 = lake** and **2 = reservoir**. It keeps
the original HydroLAKES type separately.
The saved fields are `waterbody_type` and `hydrolakes_type`, respectively.

GEB starts with the HydroLAKES waterbodies, and links data from the Global Dam Watch (GDW) and AMBER datasets based on joint IDs and/or proximity. The main purpose of this data integration is the change of type from lake to controlled lake, or reservoir: 

1. **Keep known reservoirs.** Hydrolakes waterbodies already labelled as reservoirs remain reservoirs.
2. **Checks with GDW** A suitable linked GDW dam with a valid
   capacity converts a Hydrolakes controlled lake to a reservoir.
3. **Check lakes with AMBER.** An AMBER dam inside or on the edge of a natural or controlled lake can make it a reservoir. 
4. **Add missing reservoirs.** GDW can supply reservoirs absent from
   HydroLAKES when valid capacity, area, discharge, and a usable location are
   available.

GDW dams are matched by lake ID first, then by location inside a lake. 

## Optional settings

Under `setup_waterbodies`, you can set:

| Setting | Use |
| --- | --- |
| `mode` | `on` (default), `off`, `lakes_only`, or `reservoirs_only` |
| `command_areas` | Irrigation-area polygons with a `waterbody_id` column |
| `calculate_command_areas` | Set to `true` to derive irrigation areas from the river network |
| `custom_reservoir_capacity` | CSV or Excel file with `waterbody_id` and `volume_total` (m³) |

## Check the result

Review `reports/waterbodies/gdw_checks.csv` for GDW dam checks (compared to HydroLakes), such as `point_outside_lake` and `id_differs`.

Waterbody IDs are saved in `waterbodies/waterbody_id`; cells without a
waterbody have the value `-1`.

See [Lakes and reservoirs](../../hydrology/waterbodies.md) for storage, release
rules, and construction years.
