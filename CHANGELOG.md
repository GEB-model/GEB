# dev

# v1.0.0b20
- Completely removed the region_subgrid. This subgrid was very large and led to several issues, including using lots of memory during the build. By refactoring the farms setup, this could be removed completely. This doesn't affect the model run as it never used it. Only internally in the build.
- Refactored setup farms from lowder and created a test. Non-lowder datasets are not supported anymore. This will be added back later in the simplified setup when it is required for a specific purpose.
- Write a custom zarr writer that is able to write chunked data and adapt all build methods to work with this.
- Made numerous changes throughout the build to reduce memory usage. No content changes.
- Set fill depressions to False by default in build. This option uses too much memory for large areas. If needed this can be build in again at the hydrodynamics stage.
- Remove old data catalog entirely, and all references to it. Rename NewDataCatalog to DataCatalog.
- Optimize GTSM data catalog, now pre-processing to zarr files.
- Improve CLI help so `geb evaluate --help` list the available evaluation methods.
- Fix issue in the enthalpy calculations. Previously there would be 0 infiltration even when only part of the soil was frozen. In addition, rainfall that didn't infiltrate never warmed the soil (if soil is colder..) which led to situations with too much frozen soil, too much runoff and too much discharge in winters.
- Made quite a few plots and exporters for the water balance plotting. Note that not all plots show a correct balance yet. This is highly likely not due to actual balance errors (they are checked in the running model) but because we don't yet plot the right variables. To be continued..
- Remove support for include_spinup in the evalution. This option was supported sometimes and sometimes not, which led to silent ignores and general over complications. It is still possible to run the evaluate for the spinup (only) by using the run_name: `geb evaluate --run_name spinup`.
- fix reforestation water balance flux, option 1 route excess to topwater, option 2 source from topwater. This depends on how soil behaves at first time step when forests are planted.
- For large scale (multiple basins) only: build stats are now written to CSV files under `build_memory_stats/<cluster>.csv`.  Individual files are made for each basin cluster. 
- New command "geb clean" to reset and delete the data for a model, except the .yml files. Also works for multiple basin clusters/models. 
- The buffer size check fuction (check_buffer_size) is removed as this caused errors but is redundant. 
- Fixes in water circle displaying.
- Update format for custom river discharge time series. See geb/examples/geul/data/discharge_observations
- Make full integration test, now checking discharge with observed discharge in the test_run.
- Fix recent regression where water demand for households was set to 0 except on January 1st.
- Include evaluation tests in the test_run for simplicity. 
- Include global Huizinga curves as alternative to local Endendijk in build and reasonable default config.
- Included setup_subnational_income_distribution to also setup initial subnational income distribution parameters for simulating migration decisions.
- Move evaluation of hydrodynamics to seperate file.
- Also calculate discharge metrics at daily frequency if only hourly data is available.
- Added auto-update for build methods.
- Improve memory usage of setup_soil (hopefully)
- Update to Python 3.14.4.

To support this version:
- Run `setup_flood_damage_model`: `geb update -b build.yml::setup_flood_damage_model`.
- Run `setup_subnational_income_distribution`: `geb update -b build.yml::setup_subnational_income_distribution`.
- Update to Python 3.14.4. Ensure that you update your uv first (`uv self update`).

# v1.0.0b19
- Add option for filling and raise out of bounds error for sample_from_map.
- Activate dynamic river widths during spinup. During the first years of spinup there may be some small balance errors, but they will resolve over time and in the run (when river width alpha and beta are stable).
- Only re-calculate household water demand every year (performance).
- Set SPEI calibration period to 1960-1990.
- Reduce memory usage during build with custom clip that works with dask.

To support this version:
- Add a new file called 'build_complete.txt' in your input folder. In future versions this file will be made automatically.
- Re-run `setup_hydrography`: `geb update -b build.yml::setup_hydrography`.

# v1.0.0b18
- Add loggers to groundwater model and SFINCS models.
- Close all open figures in SFINCS to reduce memory usage.
- Several fixes in sfincs.py to avoid futurewarnings for pandas 3.0.
- Compress forcing data to 1D. This makes the input folder significantly smaller (~50% depending on the area).
- Reduce area that elevation and land use maps are written for reducing size on disk.
- Remove self.buildings_centroid as attribute (appears not to be used).
- Load in buildings as pandas df, only load geometry data for flood damage calculations.
- Make filling of discharge gaps a lot more efficient (quite some reduction in run speed).
- Make it possible to specify the number of cores using `--cores`. Default is all cores (no change).
- Make an option to auto-fix the build order if it is incorrect.
- Simulate return period based flood events for updating risk perceptions (instead of fixed threshold).

To support this version:
- Re-run `setup_forcing`: `geb update -b build.yml::setup_forcing`.
- Re-run `setup_SPEI`: `geb update -b build.yml::setup_SPEI`.
- Re-run `setup_pr_GEV`: `geb update -b build.yml::setup_pr_GEV`.
- Re-run `setup_buildings`: `geb update -b build.yml::setup_buildings`.

# v1.0.0b17
- Synchronize start and end dates in reasonable default config and example.
- Add .zenodo.json

# v1.0.0b16
- Cleanup logging situation in model. Now each method (except init-multiple) should created their own log file in the logs directory and no additional logs *should* be created.
- Fix several warnings throughout model. And do not ignore some warnings globally.
- Add custom and improved logging in calibration snakemake workflow.
- In calibration, only run init and build if model.yml and build was not completed respectively.

# v1.0.0b15
- Switch back to Python 3.13 due to netcdf reading errors.
- Switch liquid water in snow and snow water equivalent to float64 to avoid floating point imprecision in thick snow layers.
- In case of water balance or enthalpy error export data for single cell that can be used to fix and test water balance seperately.

# v1.0.0b14
- Only create plots during forcing setup if specifically requested with new `create_plots` argument.
- Combine code in forcing.py so that it is more easy to maintain.
- Remove unused setup_land_use_parameters.
- Pre-process GRDC data to zarr with chunks for faster future reads.
- Add object size profile when speed-profile is used.
- Add version of when build was made.
- Switch to Python 3.14
- Make land surface build process more efficient and cleanup. As part of this update, only the original land cover within the SFINCS regions is saved. Therefore, this now depends on setup_coastal_sfincs_model_regions.
- Yield is now computed from actual evapotranspiration and potential evapotranspiration rather than actual transpiration and potential transpiration. This is in line with GAEZ documentation, and also fixes a divide by 0 error.
- Refactor runoff concentration, and solve very small WB bug due to order of operations.
- There is now a new check to check the data version against the model version. If there is a mismatch, an error is given and the user is suggested how to update to the new model version. This only works for fresh builds. If you want to force this behaviour on already existing builds, run `geb update-version`.

To support this version:
- Update to Python 3.14. If using uv, first ensure uv is updated `uv self update`, then run `uv sync` to update Python and packages.
- Move `setup_coastlines` and `setup_coastal_sfincs_model_regions` to above `setup_regions_and_land_use` in your build.yml.

# v1.0.0b13
- combine fabdem loading of elevation and forcing for saving some data on disk
- add memory profiler memray. Use option e.g., geb spinup --profile-memory
- renamed speed profiler to --profile-speed
- remove return statement from setup_forcing that was left behind from a debugging session

# v1.0.0b12
- Reforestation: add government forest planting policy and soil modification workflow.
- Convert suitable cropland/grassland to forest; update soils and remove farmers.
- Reorganized `geb/hydrology/` by moving land surface-related modules (`landsurface.py`, `evapotranspiration.py`, `interception.py`, `snow_glaciers.py`, `potential_evapotranspiration.py`) into a new `geb/hydrology/landsurface/` package.
- Split `soil.py` into `geb/hydrology/landsurface/water.py` (soil hydraulic processes) and `geb/hydrology/landsurface/energy.py` (soil thermal processes).
- Add Leaf Area Index (LAI) integration in soil net radiation calculation to account for canopy shielding and emission.
- Refactor discharge observations to support dual-frequency (hourly and daily) data tables.
- Rename generic `Q_obs` to `discharge_observations` across the codebase for clarity.
- Add frequency labels (hourly/daily) to extreme value analysis and validation plot titles.
- Allow model to run from 1960 onwards (raise clear error if earlier than 1960 is requested).
- Update `parse_demand` in `agents.py` to backward and forward fill water demand data if it doesn't cover the entire model time range.
- Update discharge observation processing to support hourly data and separate observations into hourly and daily tables.
- Update hydrology evaluation to support both hourly and daily observation datasets.
- Add build method to set up reforestation potential (and data catalog entry)
- Add units to all data from `setup_hydrography`
- Compute hillslope length based on drainage density
- Fix: Also set nodata type in _FillValue when using reporter. This is now also correctly loaded with zarr.
- Add geb tool rechunk to allow rechunking of dataset to space-optimized, time-optimized or balanced. Currently using some reasonable defaults, but if needed we can expand this with custom values.
- Add CWatM water demand to new data catalog (and remove from the old one).
- Add `--profiling` option to `geb build/update/alter`.
- Fix: fix for farm sizes that are all on the high end of the distribution.
- Fix: fix for regions with very large coastal areas beyond the riverine grid
- Use figures path for sfincs model to save all figures
- Switch to hourly values for extreme value statistics
- Use maximum of one flood peak per week
- In evaluate make a dataframe without missing timesteps and ensure that return periods are esimated on the same data for observed and simulated for comparison.
- Fix: fix for regions with very large coastal areas beyond the riverine grid.
- Fix: waterbody outflow is larger than waterbody storage (due to floating point imprecision).
- Fix: Added Liechtenstein to trade regions list which allows the model to be built in the Rhine basin
- Move MIRCA-OS to new data catalog.
- Move aquastat to new data catalog.
- Add OECD Income Distribution Database (IDD) to the new data catalog.
- Move Coast-RP to new data catalog.
- Add heat conductivity to deeper soil layers (still missing influence of water).
- Consider soil heat flux in pennmann-monteith.
- Turn of sensible and turbulent heat fluxes in case there is snow.
- Include evaporative cooling and advective heat transport from rainfall. 
- Add a daily soil enthalpy balance check.
- Generalize river snapping.
- Setup example preprocessing for retention basins.
- Fix: Add iso codes for GDL regions where those are missing 
- Fix: Fix error in GLOPOP due to regions with 17 columns, instead of 16
- Fix: Fix missing age (65) in age distribution for households 
- use GDL regions (instead of GADM) for the income distribution parameters 
- Fix: fix bug of farms that are smaller than the subgrid size 
- Fix: fix bug of countries that are not in trade regions (GLOBIOM) 
- Raise error when progress.txt contains duplicates 
- Speedup pr_gev calculation in build.
- Simplify report function arguments.
- Report water balance evaluation plot to evaluate folder.
- Save climate data in weekly chunks, also read in weekly chunks -> significant speedup (~15% is some tests).
- Use full penman-monteith for setup_SPEI.

To support this version:
- Re-run `setup_hydrography`: `geb update -b build.yml::setup_hydrography`
- Re-name `setup_mannings` to `setup_geomorphology` and run `setup_geomorphology`: `geb update -b build.yml::setup_geomorphology`
- Re-run `setup_discharge_observations`: `geb update -b build.yml::setup_discharge_observations`
- Only in case of build errors (or later in spinup/run):
     - re-run `setup_household_characteristics`: `geb update -b build yml::setup_household_characteristics`
     - re-run `setup_crops`: `geb update -b build.yml::setup_crops`
     - re-run `setup_income_distribution_parameters`: `geb update -b build.yml::setup_income_distribution_parameters`
     - re-run `setup_create_farms`: `geb update -b build.yml::setup_create_farms`

Recommended:
- Re-run `setup_forcing` and `setup_SPEI` for a significant speedup and better SPEI estimation: `geb update -b build.yml::setup_forcing` and `geb update -b build.yml::setup_SPEI`

# v1.0.0b11
- Fix numerical precision issues in waterbodies by clamping outflow to not exceed storage when handling float32 outflow with float64 storage.
- Fix GPU instability in SFINCS by disabling h73table parameter that was causing crashes during GPU-accelerated flood simulations.
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
- Removed waterbodies from gadv28 for better matching with the global exposure model.
- Added a water level boundary for coastal rivers (for now set to zero).
- Included detrending of tide data in estimation of hydrograph shape. 
- Moved Global dynamic ocean topography to the new data catalog.
- Implemented a padding of cells with values in Global dynamic ocean topography to extent the data to the coastline based on extrapolation.
- Maintain origin index of the feature dataset in VectorScanner and VectorScannerMulticurve
- Update damagescanner to v1.0.0b1
- Switch MERIT Hydro dir/elv datasets to the global cache with a local fallback copy for offline access.
- Change MERIT Hydro to use local GeoTIFF tiles directly instead of intermediate Zarr files.
- Check which MeritHydro files are present on the shared IVM datadrive. Ignore tiles that are not present in build as these are in the ocean.

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
