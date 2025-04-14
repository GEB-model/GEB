##############
Preprocessing
##############

GEB has a build module to preprocess all data for a specific region. This module creates a new folder "input" with all input files for the model.

Obtaining the data
------------------
Most of the data that the build module uses to create the input data for GEB is downloaded by the tool itself when it is run. However, some data needs to be aquired seperately. To obtain this data, please send an email to Jens de Bruijn (jens.de.bruijn@vu.nl).

Configuration
-------------
Some of the data that is obtained from online sources and APIs requires keys. You should take the following steps:

1. Request access to MERIT Hydro dataset `MERIT Hydro <https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/>`_, and create a ".env"-file in the GEB repository with the following content:

    MERIT_USERNAME=<your_key>
    MERIT_PASSWORD=<your_password>

2. To set up the model with ERA5-Land forcing data using the build-method `setup_forcing_era5`, create an account on `Destination Earth <https://earthdatahub.destine.eu/>`_ and following the instructions `here <https://earthdatahub.destine.eu/collections/era5/datasets/reanalysis-era5-land>`_.

Study region
-------------------

The `model.yml`-file specifies the configuration of the model, including the location of the model. An example of the `model.yml`-file is given in the examples folder in the GEB repository. Please refer to the yaml-section `general:region`. Examples are given below.

^^^^^^^^^^
Subbasin
^^^^^^^^^^

The subbasin option allows you to define your study region based on a hydrological basin. When using this option, the following parameter is required:

- `subbasin`: The subbasin ID of the model. This is the ID of the subbasin in the `MERIT-BASINS dataset <https://www.reachhydro.org/home/params/merit-basins>`_ (version MERIT-Hydro v0.7/v1.0). This can be either a single subbasin or list of subbasins. All upstream basins are automatically included in the model, so only the most downstream subbasin of a specific catchments needs to be specified.

.. code-block:: yaml

    general:
      region:
        subbasin: 23011134

or

.. code-block:: yaml

    general:
      region:
        subbasin:
        - 23011134
        - 23011135
        - 23011136

^^^^^^^^
geom
^^^^^^^^

The name of a dataset specified in the `data_catalog.yml` (e.g., GADM_level0) or any other region or path that can be loaded in geopandas. Using the column and key parameters, a subset of data can be specified, for example:

.. code-block:: yaml

    general:
      region:
        geom: GADM_level0
        column: GID_0
        key: FRA

^^^^^^^^^^^
outflow
^^^^^^^^^^^

The outflow option allows you to define your study region based on a specific outflow point using lat, lon coordinates:

.. code-block:: yaml

    general:
      region:
        outflow:
          lat: 48.8566
          lon: 2.3522

Building to model
-------------------

The `build.yml`-file contains the name of functions that should be run to preprocess the data. You can build the model using the following command, assuming you are in the working directory of the model which contains the `model.yml`-file and `build.yml`-file:

.. code-block:: python

    geb build

This will preprocess all the data required for the model. The data will be stored in the "input" folder in the working directory. The data is stored in a format that is compatible with the GEB model. Optionally, you can specify the path to the `build.yml`-file using the `-b/--build-config` flag, and the path to the `model.yml`-file using the `-c/--config` flag. You can find more information about the flags by running:

.. code-block:: python

    geb build --help

Updating the model
-------------------

It is also possible to update an already existing model by running the following command.

.. code-block:: python

    geb update

This assumes you have a "update.yml"-file in the working directory. The `update.yml`-file contains the name of functions that should be run to update the data. The functions are defined in the "geb" plugin of HydroMT. The data will be updated in the "input" folder in the working directory. The data is stored in a format that is compatible with the GEB model.

For example to update the forcing data of the model, your "update.yml"-file could look like this:

.. code-block:: yaml

    setup_forcing_era5:

Optionally, you can specify the path to the "update.yml"-file using the `-b/--build-config` flag, and the path to the `model.yml`-file using the `-c/--config` flag. You can find more information about the flags by running:

.. code-block:: python

    geb update --help