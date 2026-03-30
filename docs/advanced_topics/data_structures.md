# Data structures

In the main GEB model, we use the following input data. Adding new data should always conform to one of these data types. Essentially, within the [model build process](../getting_started/build/index.md) data from a wide range of sources are standardized. This standardization allows us to more easily understand the model. All important data that is often missing (e.g., CRS, nodata values) should be dealt with in the building, so that the model itself doesn't need to deal with it and thus no further reprojections (with some exceptions), setting of metadata etc are needed.

The input data for a model can be found in the `input/` directory, organized into folders as shown in the table below.

The I/O functions for these data structures are implemented in [geb.workflows.io].

| Kind | Folder name | Read function | Write function |
| :--- | :--- | :--- | :--- |
| Geometries | `geom` | [read_geom][geb.workflows.io.read_geom] | [write_geom][geb.workflows.io.write_geom] |
| Tables | `table` | [read_table][geb.workflows.io.read_table] | [write_table][geb.workflows.io.write_table] |
| Arrays | `array` | [read_array][geb.workflows.io.read_array] | [write_array][geb.workflows.io.write_array] |
| Parameters | `dict` | [read_params][geb.workflows.io.read_params] | [write_params][geb.workflows.io.write_params] |
| Grids | `grid` | [read_zarr][geb.workflows.io.read_zarr] | [write_zarr][geb.workflows.io.write_zarr] |
| Subgrids | `subgrid` | [read_zarr][geb.workflows.io.read_zarr] | [write_zarr][geb.workflows.io.write_zarr] |
| Other grids | `other` | [read_zarr][geb.workflows.io.read_zarr] | [write_zarr][geb.workflows.io.write_zarr] |

## Loading grid data

In the model, almost all gridded data is used as a 1D-vector, speeding up the computations in the model. Using the mask of the study area, only the active cells are selected from the 2D-arrays. Therefore, most modules in the model use the `self.grid.load()` or `self.HRU.load()` methods. These higher-level methods wrap [read_grid][geb.workflows.io.read_grid] and handle the compression of the spatial data to the model's active computational domain, ensuring that the data is ready for efficient hydrological calculations.

## Function references

::: geb.workflows.io.read_geom
::: geb.workflows.io.write_geom
::: geb.workflows.io.read_table
::: geb.workflows.io.write_table
::: geb.workflows.io.read_array
::: geb.workflows.io.write_array
::: geb.workflows.io.read_params
::: geb.workflows.io.write_params
::: geb.workflows.io.read_zarr
::: geb.workflows.io.write_zarr
::: geb.workflows.io.read_grid