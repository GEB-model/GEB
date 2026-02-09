# dev
- Add build method to set up reforestation potential (and data catalog entry)

# v1.0.0b10
- `setup_soil_parameters` is removed in favour of `setup_soil` for consistency.
- Add download and processing for soil thickness data.
- DeltaDTM is now also setup for the model region in setup_elevation. 
- Align SFINCS mask padding to the coarse grid so left and bottom edges snap to grid-size multiples.
- Improve inflow, outflow, flood plains and some other things to improve flood risk maps.
- Remove DeltaDTM and GEBCO for non-coastal regions.
- Re-indexing of OBM buildings and creating one household agent per building (per default).
- Support multiple inflow locations.
- (Deep) copy model config on initializing model avoiding reference issues.
- Filter clusters in geb init-multiple based on intersection with coastline if parsed as argument.
- Updated the GLOPOP version (from GLOPOP_SG_V2 to GLOPOP_SG_V3) to resolve missing data in some GDL regions
- Add option for variable runoff in infiltration  
- Simplify coastal model setup. No longer create multiple shapes of connected low elevation coastal zones.
- Moves to new data catalog
 - FAOSTAT
 - GLOPOP-SG
 - UNDP Human Development Index
 - OSM open_street_map_land_polygons
- Support custom DEMs
- Read custom reservoirs and waterbodies from files instead of old data catalog.
- Add LISFLOOD vegetation properties adapter with crop group number and leaf area index support in setup_vegetation.
- Add required = True/False to all build_methods allowing checking of build methods at build start rather than erroring when finally running the model.
- Combine setup_crops and setup_crops_from_source.
- Use LAI to set interception and compute crop factors for forest and grassland.
- Use GTSM station data to get sea level rise for creating (future) coastal flood maps.
- Add MIRCA2000 unit grid and crop calendar entries to the new data catalog and use them in crop calendar setup.
- Move superwell data to new data catalog.
- Switch MERIT Hydro dir/elv datasets to the global cache with a local fallback copy for offline access.
- Change MERIT Hydro to use local GeoTIFF tiles directly instead of intermediate Zarr files.
- Make trade regions inspired by globiom regions and load from file rather than data catalog.
- Move osm land polygons to new data catalog
- Add Global Exposure Model and GADM v2.8 to the datacatalog to assign building damages
- Assign damages categories of the Global Exposure Model to the building geodataframe.
- Calculate building damages both for structure and content using separate vulnerability curves for structure and content. 
- Check which MeritHydro files are present on the shared IVM datadrive. Ignore tiles that are not present in build as these are in the ocean.
- Adjust wind speed computation to use FAO56 specifications.
- Added a gadm_converter dictionary mapping incorrect GADM names to corrected versions in the global exposure model data adapter.
- Moved global exposure model to global cache to deal with request limits (only 60 per hour when unauthenticated, just to prevent this becoming an issue)
- Moved setup_buildings to its own function for quicker updating building attributes after changes. 

To support this version:

- Rename `setup_soil_parameters` to `setup_soil` in `build.yml`
- Re-run `setup_soil`: `geb update -b build.yml::setup_soil` and `setup_household_characteristics`: `geb update -b build.yml::setup_household_characteristics` 
- Re-run `setup_coastal_sfincs_model_regions`: `geb update -b build.yml::setup_coastal_sfincs_model_regions`
- Remove setup_low_elevation_coastal_zone_mask from you build.yml
- Add setup_buildings to your build.yml
- Models for inland regions need to be rebuild if floods need to be run
- Re-run `setup_gtsm_station_data`: `geb update -b build.yml::setup_gtsm_station_data` to regenerate `gtsm/sea_level_rise_rcp8p5` using the new GTSM station data.
- Re-run `setup_gtsm_water_levels`: `geb update -b build.yml::setup_gtsm_water_levels`
- Re-run `setup_buildings`: `geb update -b build.yml::setup_buildings`
- Setup cdsapi for gtsm download, see instruction here: https://cds.climate.copernicus.eu/how-to-api
- Rename `setup_crops_from_source` to `setup_crops` and use `source_type` rather than `type` (which is a reserved keyword in Python).
- Add and run `setup_vegetation` to `build.yml`. A good place is for example after `setup_soil`.

# v1.0.0b10
- Coastal inundation maps are now masked with OSM land polygons before writing to disk. 
- Add documentation for modules, variables and routing.
- Return period maps are now calculated per subbasin rather than using the whole map and making complicated calculation groups.
- Flood maps of varying spatial domains can now be merged into one return period map.
- Add llms.txt, llms-full.txt
- Add MCP server for interacting with natural language (very much in beta)
- Add a prompt to installation docs for setting up geb using llm agents
- Update and simplify installation docs
- Write documentation for spinning up and running models
- Fix rare out of bounds values in ERA5 data that led to undefined behaviour due to compression and decompression roundtrip
- Require extra_dims_names to be set in DynamicArray and update model in places where it was not set
- Fill holes in subbasin maps by deriving subbasin maps directly from rivers ourselves. This also makes the original subbasins dataset not needed anymore.
- Extend rivers to end up exactly in the ocean rather than the cell just before
- Enable return period maps for subbasins that discharge into the ocean, including several bugfixes for this.
- Allow exporting of hourly values from reporter
- Add initial soil temperature. Now still simplified but better than having no soil temperature.
- Includes soil suction into the model using an approximation of the Green-Ampt equation.
- Use Green-Ampt rather than VIC for infiltration.
- Implement interflow
- Limit drainage to groundwater to conductivity of groundwater top layer
- Fix for flood risk maps which could not be run if river was not included in grid but had upstream areas
- Fix that downstream outflow area was not included with new subbasins
- Renamed new_data_catalog to data_catalog and data_catalog to old_data_catalog

To support this version:

- The model must be rebuild from scratch

# v1.0.0b9
- Updated numba to 0.63. This version fixes an error where changes in sub-functions were not always correctly detected when using caching behaviour.
- Add a new option for flood models. We now auto-detect whether a change in the code or model input is made. If no change in the model or model input, we do not rebuild the SFINCS model. This removes the option `force_overwrite` for sfincs models.
- Fix JSON serialization error in hash file generation by properly converting NumPy scalar types (bool, int, float) to Python native types.
- Update to new SFINCS version.
- Migrated documentation to mkdocs
- Added a new option to detect floods based on actual discharge values from the hydrological model
- Added a new option so that households can adapt to actual floods in the model
- Implemented a simple version on runoff concentration so runoff is slowed down on its way to become discharge
- Updated the performance_hydrodynamics function so it uses a list of observation files and matches these to the right flood map per event from sfincs. The name of the observation file has to be the same of the flood event (i.e. startdate - enddate.zarr)
- For evaluation, the evaluation module now needs to be prefixed. So `geb evaluate --methods plot_discharge` becomes `geb evaluate --methods hydrology.plot_discharge`. Because we now have multiple evaluation files this keeps the logic clear.
- dict in the input files in now updated to params. It is recommended to change the dict entry to params manually. Otherwise, it is also possible to re-build the model. This is because dict was giving issues with the type checker, because dict is a reserved name.

To support this version:
- It is required to change the dict entry to params manually in `input/files.yml`. Otherwise, it is also possible to re-build the model.

# v1.0.0b8
- Improve model startup time
- Improve detection of outflow boundaries. Now uses intersection between river lines and geometry boundary.
- Add an option in the config to run only coastal models.
- Add tests for building a coastal model.
- Many type fixes
- Refactor reporter
- By default export discharge data for outflow points
- Use ZSTD compressor by default in write_zarr. This fixes a continuing error where forcing data was sometimes NaN
- Use ZSTD compressor in reporter. This makes exporting data much faster.
- Use a dynamically sized buffer to make writing in reporter more efficient, and reduce number of output files.
- Remove annotations from docstrings in farmers.py
- Do not use self in setup_donor_countries
- Export discharge at outflow points by default (new setting in report: _outflow_points: true/false)
- Add some tests for reporting
- Remove support for Blosc encoding because of random async issues. Requires re-run of `setup_forcing` and `setup_spei`
- Move examples to geb dir, so that they are included in the wheel

To support this version:

- Re-run `setup_forcing` and `setup_spei`
