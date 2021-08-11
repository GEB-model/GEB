Configuration
#####################

The model can be configured through using `GEB.yml` (YALM-format). In the model code, this configution file is parsed and can be accessed through `self.model.config`. Below the various sections of the configuration file are discussed. CWatM can be configured using its own configuration file `CWatM_GEB.ini`. To configure this file we refer to the `CWatM documentation <https://cwatm.iiasa.ac.at/>`_.

general
*********
The model start, end and spinup time (format: YYYY-MM-DD) can be configured in the general section, as well as whether the model should use a GPU (cupy is required for this option to work).

agent_settings
****************
This section can contain settings to change the behavior or the agents.

- fix_activation_order: If this setting is set to true, the farmers in the model are activated in a set order, allowing for better reproducability of the model.

logging
********
This sections contains settings for logging of the model. 

- logfile: The file which the logs are written to.
- loglevel: The loglevel

draw
*****
This section is only used for the visual interface. This section closely works together with the :doc:`artists`.

- draw_every_nth_agent: To avoid drawing all farmers, which might become very slow, this option allows to only draw every nth agent. For example, setting this to 1000, means that every 1000th agent is drawn.
- draw_agents: Which agent types should be drawn.

report
*******
Here, you can configure which data should be saved from the model in the `report` folder. The configuration is structured as follows::

    name: "name of the folder to which the data is saved".
        type: "agent type e.g., farmer. Should be identical to attribute name in Agents class."
        function: "whether to use a function to parse the data. 'null' means the data is saved literally, 'mean' takes the mean etc. Options are given in Hyve's documentation`.
        varname: "attribute name of variable in agent class".
        format: "format to save to".
        initial_only: "if true only save the data for the first timestep".

report_cwatm
**************
In this section you can configure what CWatM data is saved, such as the groundwater head in each timestep, as follows::

    name: "name of the folder to which the data is saved".
        varname: "attribute name of CWatM data. Should be precededed by 'var.' for data from CWatM cells and 'subvar.' for data from hydrologial units.".
        function: "whether to use a function to parse the data. For example, 'mean' takes the mean etc. Options are 'mean', 'sum', 'nanmean' and 'nansum'.
        format: "format to save to".