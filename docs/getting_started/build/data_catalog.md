# Data catalog

## AMBER barrier data

The `amber_barriers` entry downloads the AMBER V1 European barrier atlas and
caches its processed point data as GeoParquet. Read it through the data catalog:

```python
import logging

from geb.build.data_catalog import DataCatalog

catalog = DataCatalog(logger=logging.getLogger(__name__))
barriers = catalog.fetch("amber_barriers").read(bbox=(3.0, 50.0, 8.0, 54.0))
```

The optional bounding box is `(west, south, east, north)` in WGS84 degrees.
Points use EPSG:4326 and contain `amber_id`, `dam_type` and `dam_hgt_m` (meters).
Processing removes duplicate IDs and invalid coordinates, normalizes barrier
type labels, and stores missing, nonpositive or nonfinite heights as null.
Missing required fields or IDs raise an error. The cached dataset is reused
on subsequent fetches.

Source: [AMBER Atlas V1](https://doi.org/10.6084/m9.figshare.12629051),
licensed under CC BY 4.0. This entry provides source data for future build
workflows; fetching it does not place structures on the river network.


### Snap AMBER weirs to the model grid

Run `setup_weirs` after `setup_waterbodies` to place AMBER weirs in river routing. The default build includes this step. For custom build files:

```yaml
setup_waterbodies: {}
setup_weirs: {}
```

For existing inputs, add the step to your build configuration and run
`geb update -b build.yml::setup_weirs` or the input-version updater.

The step represents **all AMBER barrier types as weirs**, retaining their original
types as metadata. All types use the same rectangular overflow equation; gate
operations and culvert hydraulics are not represented. Each weir's height is **0.75 times
its modeled bankfull depth**, above the local river bed. There is no fixed-meter
height option. Recorded AMBER heights are ignored, including missing values.
This fraction is a modeling assumption rather than a measured crest height.
Remove any previous `height_m` setting and rerun `setup_weirs` to replace older
fixed-height inputs. A weir
must lie within **250 m of the nearest represented river line**, measured in a
local projection in meters. This is a line-distance limit, not a distance to the
coarse model-cell center. The nearest cell on that river must have an active
river cell downstream, with neither cell inside a lake or reservoir.

After upgrading to 1.0.0b38, rerun `setup_weirs`, then spinup and simulation.
Existing inputs keep their stored 0.5-depth crests until rebuilt.

Outputs are `grid/routing/weir_bankfull_depth_fraction.zarr` (0.75 at weirs,
zero elsewhere; dimensionless) and `geom/routing/weirs.geoparquet`. The latter
retains original point locations, recorded heights as metadata, the assigned
`bankfull_depth_fraction`, source IDs, source-to-line distances, snapped
coordinates, grid row and column, and inclusion/exclusion reasons. Each cell
represents one weir on its downstream interface. If the closest cell is occupied,
keep the existing weir and try the second-closest distinct cell on the same river
segment. Exclude the new weir if that cell is occupied or unsuitable; do not
search farther along the segment or switch rivers. Process weirs by AMBER ID
for reproducible assignments.

`snap_rank` is 1 for the closest cell, 2 for the fallback and 0 for exclusions.
`distance_to_cell_m` records the source-to-cell-center distance. The 250 m limit
still applies to the river line; a coarse cell center can be farther away.
Distant points and unsuitable connections are also recorded.
Rerun `setup_weirs` to include other barrier types in previously built inputs,
then rerun spinup and simulation. Distance, collision and connection exclusions
still apply, so not every source barrier necessarily receives its own model cell.

Weirs are inactive during spinup, which determines channel geometry. The run
restores the saved spinup dimensions and keeps them fixed, with weirs active
unless `hydrology.routing.weirs` is false.
Routing converts these fractions to heights in meters using the saved channel
bankfull depths (`router.weir_height_m`). For example, a 4 m bankfull depth gives
a 3 m weir. The old fixed-height grid
is ignored. River flow at each included interface uses rectangular overflow with
coefficient 0.577 and a downstream submergence reduction; see
[weir hydraulics](../../hydrology/routing.md#amber-weirs).

After preparing inputs, **rerun spinup before running the model**:

```sh
geb update -b build.yml::setup_weirs
geb spinup
geb run
```

Run these commands from the model directory, using your usual configuration and
working-directory options. The setting `hydrology.routing.weirs` applies only
during the run and defaults to `true`. Set it to `false` for a comparison run,
using separate run names/output directories and the same weir-free spinup.

Routing retains a daily history for selected weirs and refreshes their spatial
snapshots on the first day, monthly and on the final day. Each snapshot shows
upstream and downstream water levels across several routing cells, plus the two
adjacent channel cross-sections. See [weir diagnostics](../../hydrology/weirs.md#inspect-water-levels)
for file names, plotted dimensions and comparison guidance.
