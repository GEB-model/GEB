# Weirs

GEB represents AMBER weirs as controls on flow between river reaches. Water
collects upstream until it reaches the crest, then spills downstream. When the
downstream water level rises, submerged overflow is reduced. The existing river
and floodplain storage hold the backed-up water; a weir does not create a new
reservoir or add pond area.

This implementation includes **AMBER `Weir` records only**. Other barrier types
and gate operations are not included. Lakes and reservoirs retain their existing
routing rules.

## Setup and run

Place `setup_weirs` after `setup_waterbodies` in your build configuration:

```yaml
setup_waterbodies: {}
setup_weirs: {}
```

From your model directory, use your usual configuration options and run:

```sh
geb update -b build.yml::setup_weirs
geb spinup
geb run
```

Rerun spinup after adding weirs so the run starts with consistent river storage.
The default runtime setting is `hydrology.routing.weirs: true`. Set it to
`false` in a separate spinup/run to compare results without weir controls.
Use separate output directories or run names and matching spinup states.

## Placement and dimensions

A weir must be within **250 m of a represented river line**. Its connection
must join two river cells outside lakes and reservoirs. Each pixel has one
hydraulic control on its downstream connection. When its closest pixel is
occupied, a new weir tries the second-closest distinct pixel on the same river
segment. If that pixel is occupied or unsuitable, the new weir is excluded and
the earlier placement is retained. No farther pixels or other rivers are tried.
Assignments follow AMBER ID order for reproducibility. Outlets and waterbody
connections remain excluded.

The crest is **0.5 times the current bankfull depth above the upstream river
bed**. Bankfull depth is the water depth when the river reaches its banks.
Crest width equals modeled bankfull river width. Recorded AMBER heights are
ignored, including missing values. These dimensions are modeling assumptions
and follow channel geometry updates; they are not surveyed structures.

For example, a modeled bankfull depth of 2 m gives a 1 m crest above the bed.
See [weir hydraulics](routing.md#amber-weirs) for the rectangular formula,
submergence correction and numerical safeguards. The model currently uses the
routing-wide policy of no reverse flow, even at weirs.

## Inspect water levels

`input/geom/routing/weirs.geoparquet` retains original AMBER locations, snapped
coordinates, inclusion flags and exclusion reasons. `snap_rank` shows whether
an included weir used its closest pixel (1) or second-closest pixel (2).
`distance_to_cell_m` records the source-to-cell-center distance: the 250 m
acceptance limit applies to the river line, so coarse pixels can be farther away.
In QGIS, filter
`included = true` to inspect the placements.

On the first day of each month and the final model day, routing writes:

- `<general.output_folder>/<run_name>/weirs/water_levels_YYYY-MM-DD.png`
- A matching CSV with all supported weirs, their river/grid IDs, crest elevations,
  upstream/downstream water levels, head above the crest and water-level drop.

The PNG shows up to three local views, starting with the tallest crests and
skipping nearby weirs already covered by a selected view. Each row has two
panels:

- **River elevation:** bed, bankfull level and water surface in the same vertical
  datum. Bed and water use the same cell boundaries, with water filled in blue.
- **Depth above local bed:** the slope is removed by subtracting each cell's own
  bed. This makes shallow water and crest heights visible. This panel shows
  depths, not the longitudinal water-surface gradient.

Each profile follows up to two reaches upstream and downstream, taking the
widest branch at confluences. The focal weir is red; nearby weirs are gray.
The annotation gives upstream depth, crest height and head above the crest.
A matching `water_levels_YYYY-MM-DD_profiles.csv` contains the selected profiles'
cell IDs, distances, lengths, bed elevations, depths and levels for replotting.
The original snapshot CSV still contains every supported weir.

The plot shows cell-scale model results. It cannot resolve an overflow jet or
water-level variation inside a cell. Disabled-weir runs show the proposed
crests as dashed lines. Compare matching dates between runs to assess the effect.
Monthly snapshots can miss short flood peaks; use the normal routing reports
if you need water-level or discharge time series.

For Geul and later European applications, compare simulated water levels and
discharge with observations. The half-bankfull height, channel-width crest and
idealized discharge coefficient are a simple regional parameterization.
