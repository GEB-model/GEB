#############################
Scale CWatM input data
#############################

Several datasets can be obtained directly from the CWatM model, which can be upscaled to the resolution of the mask using the functions in this file. To do so place all NetCDF files in the `DataDrive/GEB/input_5min`. Then run `preprocessing.2_scale_cwatm.py` to scale the files. The file and folder structure is automatically recreated in `DataDrive/GEB/input`.

.. automodule:: preprocessing.2_scale_cwatm
    :members: