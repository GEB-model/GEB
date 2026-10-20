# Lakes and reservoirs
GEB simulates storage in lakes and reservoirs.  

## Lakes

Lake outflow depends on the water level and the outlet size. A higher water
level allows more water to flow out. Water leaving a lake enters the downstream river or waterbody.

The local inertial solver calculates lake outflow at each routing substep.
The lake outlet is represented by a weir rating curve. For the current
parabolic outlet shape, discharge follows `Q = k × h²`, where `h` is the
water depth above the outlet (m) and `k` is a lake-specific outlet factor
(m/s). This lets lake releases change as the water level changes.

## Reservoirs

Reservoir operators decide how much water to release. The current rules use:

- the stored water and reservoir capacity;
- average inflow, including seasonal changes; and
- irrigation demand.

Water is released to the river or supplied directly to an irrigation area.
The rules limit releases when storage is low and release extra water when
storage is high.

The release scheme follows Shin et al. (2019). It combines seasonal inflow
with irrigation demand, using the ratio of reservoir capacity to average
annual inflow. Larger storage makes demand more important. Releases also
depend on storage at the start of the hydrological year.

The current rules use 85% of total capacity as the normal storage limit and
10% as the minimum storage reserve. Water needed for planned irrigation
releases is also reserved. A command area is the irrigation area supplied
directly by a reservoir. The water used for irrigation is withdrawn from the routing. GEB currently uses irrigation release rules for all reservoirs. A separate hydropower release scheme is not yet implemented.

## When a reservoir starts

A reservoir starts operating on January 1 of its construction year. Before
then, water flows through its cells as a river. Water already in the river is
kept when the reservoir starts filling.

Lakes and reservoirs without a known construction year are always active.

## Data and setup

GEB uses HydroLAKES for lake outlines, Global Dam Watch (GDW) for dam data, and
AMBER for European barriers. These data help decide which waterbodies are lakes
and which are reservoirs.

See [Waterbody setup](../getting_started/build/waterbodies.md) for info on how the waterbodies are build. 

See [Weirs and gates](weirs.md) for river barriers.
