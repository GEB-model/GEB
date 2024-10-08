## v1.0.0

### what's new

- GEB and can now be installed directly from Pypi, using `pip install geb`.
- CWatM is now integrated with GEB, and thus (AB)CWatM does not need to be separately installed anymore. We are immensely grateful to the CWatM team for their support and collaboration.
- Hydromt-geb is now integrated with GEB, and thus does not need to be separately installed anymore.
- MODFLOW was migrated to an irrigular grid, which makes it compatible with the GEB grid, avoiding complicated regridding.
- The model now features a two-layer groundwater model, with much better initalization of hydrological parameters (thanks to GLOBGM!). Groundwater outflow is now also directly read from MODFLOW.
- We now estimate soil parameters at 30'' resolution from the SoilGrids database using pedotransfer functions.
- The soil module now has six soil layers.
- The soil module was adapted for multi-threading, and now runs much faster.
- The lakes and reservoir module was overhauled and optimized. For example, the outflow height is now calculated based on the lake area and the outflow coefficient. This should result in a much more realistic lake outflow.
- The outflow of reservoirs was aligned with the original CWatM paper.
- The model erroneously use the elevation of the outflow as the normal elevation of the terrain. Terrain elevation is now separately calculated, while outflow elevation is still used for the calculation of Manning's roughness.
- An example model.yml, build.yml, update.yml and data_catalog.yml are now included in the model repository.
- Migrated from black to ruff for code formatting.
- All input data is now stored in zarr format rather than netCDF.
- The calculation of SPEI is optimized to be nicer for memory.
- setup_GEV in hydromt-geb was integrated with setup_SPEI and can be removed.
- Many other minor fixes and improvements.
- Included a simple market for future scenarios.
- Data exporting in model.yml is now to zarr instead of netcdf.
- Included new source for setting up assets (movisda). This can be used when geofabrik is down.
- Allow reducing the number of crops by using the most frequently grown crop in a specific grop group
- Set up crop prices from FAO stat
- Include rails and roads in preprocessing
- Migrate all tif files to zarr for consistency of geotransformation and sometimes funny errors in tif files

### Migration guide

- Unfortunately, all models need to be rebuild from scratch
- `setup_elevation_STD` in hydromt-geb was renamed to `setup_elevation`
- `setup_modflow` was renamed to `setup_groundwater`
- We migrated to the new climate data store  (CDS) API, which requires a new API key. Please register at [https://cds-beta.climate.copernicus.eu/](https://cds-beta.climate.copernicus.eu/) and update your `~/.cdsapirc` file with the new key.
- The ingested data is also slightly different, which means it needs to be re-downloaded.
- Install ruff for code formatting: `pip install ruff`
- Thus set "format: netcdf" to "format: zarr" in model.yml (if applicable)
- model_structure has been renamed to files, thus change `model_structure` to `files` in all code
- The format of basin_lakes_data has changed to parquet. If the model is rebuilt this should automatically be updated.
- `setup_cell_area_map` has been remaned to `setup_cell_area` for consistency with other function names. The old function is still available but will be removed in the next release.