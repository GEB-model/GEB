Pick farmer crops
#############################

`preprocessing.4_create_lakes_and_reservoirs` assigns a crop type, irrigation type and planting scheme to each farmer using to the MIRCA2000 dataset. We condider specified crops, growing area, and various planting schemes (e.g., single, double-cropping). But before running the file, we need to obtain MIRCA2000 input data and irrigated areas as described below.

MIRCA2000
****************
1. Download the `Monthly Growing Area Grids` from the `Goethe-universität <https://www.uni-frankfurt.de/45218031/data_download>`_ and place in `DataDrive/GEB/original_data/MIRCA2000/MGAG`.
2. Decompress the data.
3. Also download `flt_to_asc.exe` from the same page.
4. Convert the flt-files to asc-files using the exe-file. A batch script for convenience is provided in `prepare_input_data/MIRCA2000`.
5. Create netcdf-files (also created in `DataDrive/GEB/original_data/MIRCA2000/monthly_growing_areas`) by running `prepare_input_data/MIRCA2000/asc_to_netcdf.py`.
6. Download the `Condensed Crop Calendars` for both irrigated and rainfed crops, decompress, and place in `DataDrive/GEB/original_data/MIRCA2000`.
7. Download the `Calendar units`, decompress, and place in `DataDrive/GEB/original_data/MIRCA2000`.

Irrigated land
****************
You can download the irrigated areas from `Scientific Data <https://www.nature.com/articles/sdata2016118>`_, and place it in `DataDrive/GEB/original_data/india_irrigated_land`.

.. automodule:: preprocessing.9_pick_farmer_crops
    :members: