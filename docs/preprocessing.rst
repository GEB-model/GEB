##############
Preprocessing
##############

GEB uses HydroMT to preprocess all data required for the model. HydroMT (Hydro Model Tools) is an open-source Python package that facilitates the process of building and analyzing spatial geoscientific models with a focus on water system models. It does so by automating the workflow to go from raw data to a complete model instance which is ready to run. This plugin provides an implementation of the model API for the GEB model.

Obtaining the data
------------------
Most of the data that the HydroMT plugin uses to create the input data for GEB is downloaded by the tool itself when it is run. However, some data needs to be aquired seperately. To obtain this data, please send an email to Jens de Bruijn (jens.de.bruijn@vu.nl).

Preprocessing
-------------
The preprocessing contains two steps:
1. Creating all non-agent data
2. Creating all data for agent attributes

The file build_model.py contains the code to build the model.