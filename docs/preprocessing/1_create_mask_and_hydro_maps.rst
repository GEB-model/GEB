#############################
Create mask and hydro maps
#############################

GEB uses a mask for the hydrological model (just as in CWatM), and a submask to represent the subcells. After creating the mask, the submask is created at a higher resolution as specified in `UPSCALE_FACTOR`. The hydrological units are then created by merging these very small subcells.

However, first we need some data. Start by downloading the original `MERIT Hydro elevation map <http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/>`_. After registering you can download the "Adjusted Elevation" for your area. This data is required to calculate determine the standard deviation for your file. Decompress and save all output files in `ROOT/original_data/merit_hydro_03sec/`. It is recommended to download the `Upstream Drainage Pixel` too and place in the same the same folder as you will need this for a later step.

Next download, `30sec_basids.tif`, `30sec_elevtn.tif`, `30sec_uparea.tif`, `30sec_rivlen_ds.tif`, `30sec_rivslp.tif` and `30sec_flwdir.tif` from `Zenodo <https://zenodo.org/record/5166932>`_ and place in `ROOT/original_data/merit_hydro_30sec`. Now run `preprocessing.1_create_mask_and_hydro_maps.py` to create the mask, submask and several hydrological maps.

.. automodule:: preprocessing.1_create_mask_and_hydro_maps
    :members: