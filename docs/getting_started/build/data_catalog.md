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

Run `setup_weirs` after `setup_waterbodies` to prepare weir locations for future
routing work. The default build includes this step. For custom build files:

```yaml
setup_waterbodies: {}
setup_weirs: {}
```

For existing inputs, add the step to your build configuration and run
`geb update -b build.yml::setup_weirs` or the input-version updater.

The step selects only AMBER `Weir` records with positive, finite heights. A weir
must lie within **250 m of the nearest represented river line**, measured in a
local projection in meters. This is a line-distance limit, not a distance to the
coarse model-cell center. The nearest cell on that river must have an active
river cell downstream, with neither cell inside a lake or reservoir.

Outputs are `grid/routing/weir_height_m.zarr` (meters; zero without a weir) and
`geom/routing/weirs.geoparquet`. The latter retains original point locations,
heights, source IDs, source-to-line distances, snapped coordinates, grid row and
column, and inclusion/exclusion reasons. Multiple weirs mapping to the same
connection are all excluded for review; they are never shifted to other cells.
Missing heights, distant points and unsuitable connections are also recorded.
Non-weir structure types are ignored.

These are build inputs only. River flow and storage calculations are unchanged
until a weir-flow law is implemented in the routing solver.
