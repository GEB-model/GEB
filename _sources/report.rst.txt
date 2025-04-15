Report
########

The report module provides functionality for to report model data to disk. This can be configured in `model.yml` under the `report` key. The structure is as follows:


.. code-block:: yaml

    report:
        module_name:
            filename:
                varname: varname
                type: type
                function: function
                frequency: frequency  # optional
            other_filename:
                varname: varname
                type: type
                function: function
                frequency: frequency  # optional
        other_module_name:
            filename:
                varname: varname
                type: type
                function: function
                frequency: frequency  # optional


For example, to export the total bare soil evaporation, gridded daily discharge, and the mean channel abstraction by crop_farmers, you would use the following configuration:

.. code-block:: yaml

    report:
        hydrology.soil:
            actual_bare_soil_evaporation_weighted_sum:
                varname: .actual_bare_soil_evaporation
                type: HRU
                function: weightednansum
        hydrology.routing:
            discharge_daily:
                varname: grid.var.discharge
                type: grid
                function: null
        agents.crop_farmers:
            channel_abstraction_m3_by_farmer_mean:
                varname: var.channel_abstraction_m3_by_farmer
                type: agents
                function: mean

The following options are supported.

* **module_name**: The name of the module in the GEB model. Both agents and hydrological modules are supported. Please refer to the name property of the module for the correct names.
* **filename**: The name of the file to be created. This can be a .csv or .zarr file. If the file already exists, it will be overwritten. The filename will be created in the output directory specified in the model configuration file, and placed inside the subdirectory with the name of the module.
* **varname**: The name of the variable to be reported. There are two options:

  + **module attribute**: The variable name in the module. Any variable that can be reached from "self" in the module can be used. For example, `[self.]varname` or `[self.]grid.var.discharge`. Note that `self.` is ommited.
  + **local variable**: Any variable that exists within the `step` function of the module. This is useful for reporting variables that are not stored between timesteps. Local variables are prefixed with ".", for exampe `.actual_bare_soil_evaporation` in the example above.
* **type**: The type of the variable. This can be one of the following:

  * **grid**: A variable that is stored in the grid structure of GEB.
  * **HRU**: A variable that is stored in the hydrological response unit (HRU) structure of GEB.
  * **agents**: A variable that is stored in the agents. This is a 1D array with the same length as the number of agents. The variable will be reported as a 1D array.
* **function**: The function to be used to process the variable before reporting. The supported options vary per type. Each type supports the **null** function, which means that no function is applied and the variable is reported as is in a zarr file. In all other cases the variable is reported at the end of the model run in a csv file. The following functions are supported:

  * **agents**:

    * **null**: No function is applied. The variable is reported as is.
    * **mean**: The mean of the variable is calculated and reported.
    * **nanmean**: The mean of the variable is calculated and reported, ignoring NaN values.
    * **sum**: The sum of the variable is calculated and reported.
    * **nansum**: The sum of the variable is calculated and reported, ignoring NaN values.

  * **grid** and **HRU**:

    * **null**: No function is applied. The variable is reported as is.
    * **mean**: The mean of the variable is calculated and reported.
    * **nanmean**: The mean of the variable is calculated and reported, ignoring NaN values.
    * **sum**: The sum of the variable is calculated and reported.
    * **nansum**: The sum of the variable is calculated and reported, ignoring NaN values.
    * **weightedmean**: The mean of the variable is calculated and reported, weighted by the grid cell area.
    * **weightednanmean**: The mean of the variable is calculated and reported, weighted by the grid cell area, ignoring NaN values.
    * **sample,[y],[x]**: Sample a specific variable at specific y,x pixel. 0,0 is the top left corner of the grid. Example is **sample,1,2** to sample the variable at pixel 1,2.
    * **sample_coord,[lat],[lon]**: Sample a specific variable at specific coordinates using the lat,lon coordinates of the grid. The coordinates are in the same coordinate system as the grid. Example is **sample_coord,52.38,4.89** to sample the variable at coordinates latitude 52.38 and longitude 4.89 (Amsterdam). Note that when reporting discharge, it is important to make sure that the location is in the actual river you want to sample from. You can refer to the upstream area in the input files to find the actual river.

* **frequency** (optional): The frequency at which the variable is reported. The default is **every: day** This can be one of the following:

  * **initial**: The variable is reported at the initial time step only.`
  * **final**: The variable is reported at the final time step only.
  * **every**: The variable is reported at specified timesteps, examples are given below.

Report the data every third day of the month:

.. code-block:: yaml

    report:
        module_name:
            filename: filename
            varname: varname
            type: type
            function: function
            frequency:
                every: month
                day: 3

Report every 15th of March every year:

.. code-block:: yaml

    report:
        module_name:
            filename: filename
            varname: varname
            type: type
            function: function
            frequency:
                every: year
                month: 3
                day: 15

Package contents
******************

.. automodule:: geb.reporter
    :members: