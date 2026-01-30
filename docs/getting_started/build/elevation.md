# Elevation

Use `setup_elevation` to build the elevation layers your model needs. In most cases, the defaults are all you need. Customize only if you want to swap in a local DEM or change thresholds like `zmin`/`zmax`.

After running `setup_elevation`, the model stores:

- A subgrid elevation layer at `landsurface/elevation` (from FABDEM).
- One raster per DEM under `DEM/<name>`.
- The resolved DEM configuration under `hydrodynamics/DEM_config`.

## Default setup

The default configuration uses three DEMs in this order:

1. `delta_dtm` (coastal-only, low-elevation coast)
2. `gebco` (coastal-only, bathymetry)
3. `fabdem` (global land elevation)

## Custom setup

You can override the defaults in your build configuration. Each DEM entry accepts these keys:

| Key | Required | Description |
| --- | --- | --- |
| `name` | yes | DEM identifier. Built-in names are `fabdem`, `delta_dtm`, and `gebco`. |
| `path` | for custom DEMs | Filesystem path to your DEM (required if `name` is not built-in). |
| `zmin` | no | Minimum elevation to keep (values below become `NaN`). |
| `zmax` | no | Maximum elevation to keep (values above become `NaN`). |
| `zmin_coastal` | no | Minimum elevation to keep for coastal cells (values below become `NaN`). |
| `zmax_coastal` | no | Maximum elevation to keep for coastal cells (values above become `NaN`). |
| `fill_depressions` | no | Whether to fill depressions before storing the DEM. |
| `nodata` | no | Nodata value if the file does not define one. |
| `crs` | no | CRS to set for custom DEMs when the file does not define one (EPSG code or CRS string). |
| `coastal_only` | no | Skip this DEM when there are no coastal subbasins. |

Use `zmin_coastal` and `zmax_coastal` when you want different cutoffs for coastal cells than for inland cells. If you do not set them, `zmin`/`zmax` are used everywhere.

### Custom DEM requirements

If you provide a custom DEM:

- You **must** set `path`.
- The dataset **must** have a valid CRS (for `.zarr`, CRS is parsed and attached automatically), or you must provide `crs`.
- The dataset **must** define nodata, or you must provide `nodata` in the config.

## Examples

### Use the defaults (no changes)

```yaml
build:
	setup_elevation:
```

### Add a custom DEM

```yaml
build:
	setup_elevation:
		DEMs:
			- name: my_local_dem
				path: /path/to/dem.tif
				crs: 28992
				nodata: -9999
				fill_depressions: false
			- name: fabdem
				zmin: 0.001
				fill_depressions: true
```
