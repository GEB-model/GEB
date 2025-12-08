## v1.0.0b8
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

To support this version:

- Re-run `setup_forcing` and `setup_spei`