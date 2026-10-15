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
