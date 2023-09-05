##############
Preprocessing
##############

GEB uses HydroMT to preprocess all data required for the model. HydroMT (Hydro Model Tools) is an open-source Python package that facilitates the process of building and analyzing spatial geoscientific models with a focus on water system models. It does so by automating the workflow to go from raw data to a complete model instance which is ready to run. This plugin provides an implementation for the GEB model, which can be found `here <https://github.com/jensdebruijn/hydromt_geb>`_.

Installing hydromt-geb
----------------------
The hydromt-geb plugin can be installed using pip:

.. code-block:: bash

    pip install git+https://github.com/jensdebruijn/hydromt_geb.git

Obtaining the data
------------------
Most of the data that the HydroMT plugin uses to create the input data for GEB is downloaded by the tool itself when it is run. However, some data needs to be aquired seperately. To obtain this data, please send an email to Jens de Bruijn (jens.de.bruijn@vu.nl).

Preprocessing
-------------
The preprocessing contains two steps:
1. Creating all non-agent data
2. Creating all data for agent attributes

The file build_model.py contains the code to build the model, including step 1 and 2. In this script the agent attributes are generated randomly. This allows you to quickly try to model. As input, build_model.py requires a configuration file, which specifies the poor point (poor_point) of the (sub-)basin in the "general" section of the yml-file. This file also contains all other parameters that are required to run the model. The configuration file is a YAML file. An example configuration file is provided in the repository (sandbox.yml).

.. code-block:: python

    python build_model.py --config sandbox.yml

For more advanced uses of the model, you can also run build_model.py first, followed by build_model_farmers.py. This file constructs model input files from a csv-file that contains the agent attributes. This allows you to use your own data to run the model. An example of how to build such a csv file is given in preprocessing/krishna. The csv-file should contain the following columns: season_#1_crop, season_#2_crop, season_#3_crop, irrigation_source, household_size, daily_non_farm_income_family and daily_consumption_per_capita. The csv file "agents/farmers/farmers.csv" should be located in the preprocessing folder, which was created by build_model.py. 

.. code-block:: python

    python build_model_farmers.py --config sandbox.yml