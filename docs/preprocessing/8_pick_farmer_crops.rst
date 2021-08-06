Pick farmer crops
#############################

## Preprocessing
1. First we need to create the mask, submask and hydrological base maps. The base maps are created 

## MIRCA2000
1. Download monthly growing areas from the [Goethe-universität](https://www.uni-frankfurt.de/45218031/data_download) and place in `DataDrive/GEB/original_data/MIRCA2000/monthly_growing_areas`.
2. Decompress the data.
3. Also download `flt_to_asc.exe` from the same page.
4. Convert the flt-files to asc-files using the exe-file. A batch script for convenience is provided in `prepare_input_data/MIRCA2000`.
5. Create netcdf-files (also created in `DataDrive/GEB/original_data/MIRCA2000/monthly_growing_areas`) by running `prepare_input_data/MIRCA2000/asc_to_netcdf.py`.

.. automodule:: preprocessing.8_pick_farmer_crops
    :members: