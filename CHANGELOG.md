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
