# Weirs

GEB represents all AMBER barrier types as weir controls between river reaches. Water
collects upstream until it reaches the crest, then spills downstream. When the
downstream water level rises, submerged overflow is reduced. The existing river
and floodplain storage hold the backed-up water; a weir does not create a new
reservoir or add pond area.

Every AMBER type, including dams, sluices, culverts and fords, uses the same
rectangular weir equation. The original type is retained in placement diagnostics.
This is a simplified barrier experiment: it does not reproduce gate operations,
culvert flow or the individual structures' surveyed geometry. Connections inside
lakes and reservoirs remain excluded to avoid adding another control there.

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

Spinup always runs **without weir effects** and determines the channel dimensions.
The run restores these dimensions from the spinup checkpoint and keeps them fixed,
so weir-altered discharge cannot feed back into channel width or depth. Weirs
become active during the run when `hydrology.routing.weirs: true` (the default).
Set it to `false` for a run without weirs; both runs can use the same spinup.
Use separate output directories or run names for comparisons.

To apply the 150 m distance limit or include additional barrier types in existing
models, rerun `setup_weirs` first. Existing snapped inputs keep their placements
until rebuilt.
Rerun spinup once after upgrading so the checkpoint contains channel dimensions
estimated without weir effects. Older checkpoints without saved grid depths fall
back to dimensions derived from their saved spinup return-period discharge.
The default karst release time is **10 days**, during both spinup and run.

## Placement and dimensions

A weir must be within **150 m of a represented river line**. Its connection
must join two river cells outside lakes and reservoirs. Each pixel has one
hydraulic control on its downstream connection. When its closest pixel is
occupied, a new weir tries the second-closest distinct pixel on the same river
segment. If that pixel is occupied or unsuitable, the new weir is excluded and
the earlier placement is retained. No farther pixels or other rivers are tried.
Assignments follow AMBER ID order for reproducibility. Outlets and waterbody
connections remain excluded.

The crest is **0.75 times the current bankfull depth above the upstream river
bed**. Bankfull depth is the water depth when the river reaches its banks.
Crest width is the channel width at crest height: `b = B * (height / D)**r`,
where B is bankfull width, D bankfull depth, and r the channel shape exponent.
For parabolic channels, `r = 0.5`, giving `b = sqrt(0.75) * B` (about 0.866B).
Recorded AMBER heights are ignored, including missing values. These dimensions are modeling assumptions
and use the fixed spinup channel dimensions during the run; they are not surveyed structures.

For example, a modeled bankfull depth of 2 m gives a 1.5 m crest above the bed.
After upgrading to 1.0.0b38, rerun `setup_weirs`, then spinup and simulation.
Existing inputs keep their stored 0.5-depth crests until rebuilt.

See [weir hydraulics](routing.md#amber-weirs) for the rectangular formula,
submergence correction and numerical safeguards. The model currently uses the
routing-wide policy of no reverse flow, even at weirs.

## Inspect water levels

`input/geom/routing/weirs.geoparquet` retains original AMBER locations, snapped
coordinates, inclusion flags and exclusion reasons. `snap_rank` shows whether
an included weir used its closest pixel (1) or second-closest pixel (2).
`distance_to_cell_m` records the source-to-cell-center distance: the 150 m
acceptance limit applies to the river line, so coarse pixels can be farther away.
In QGIS, filter
`included = true` to inspect the placements.

Routing records **daily end-of-day values** for up to three selected weirs in
`<general.output_folder>/<run_name>/weirs/water_levels_daily.csv`.
The first day selects tall weirs while skipping immediate neighbors. The same
cells are tracked throughout the simulation, even when channel geometry changes.
`selected_weirs.csv` lists those grid cells. Other weirs still affect routing;
limiting these diagnostics to three keeps continental runs manageable.

Each selected weir has **one current snapshot**, `water_levels_weir_CELL.png`.
The same file is refreshed on the first model day, monthly and on the final
day; no growing series of dated images is produced. Starting a fresh simulation
resets the daily history. Existing older snapshot files are not removed.

- **Side view:** up to five routing cells before the weir and five after it,
  with water, bed, bankfull level and nearby weirs. The focal upstream cell
  counts toward the five cells before the weir. Follow the widest upstream
  branch at junctions and stop at routing boundaries. Distances use actual
  cell lengths, with zero at the focal weir's downstream edge. Cell boundaries
  and identifiers make the model resolution visible.
- **Across-river views:** channel cross-sections for the two cells immediately
  beside the focal weir. They show water depth, bankfull width/depth and the
  width at the top of the barrier. Each uses its own bed reference.
- Four headline values give water depth on either side, barrier height and
  upstream water above its top.

A matching `water_levels_weir_CELL.csv` holds the full current spatial profile.
The separate `water_levels_daily.csv` retains the complete daily history of
levels, depths and geometry immediately beside the selected weirs, even on days
when figures are not redrawn.

Bankfull means water just reaches the river banks. A weir is a low barrier that
water can flow over. Disabled-weir snapshots show dashed proposed barriers.
The upstream/downstream water-level difference includes bed slope; compare
matching dates with an otherwise identical no-weir run to isolate the effect.
These are daily model-cell values: short peaks within a day, internal cell
water-surface variations and overflow jets are not resolved by the figures.
