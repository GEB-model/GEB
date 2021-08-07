#############################
Scrape command areas
#############################

GEB uses three types of irrigation: surface, groundwater and reservoir irrigation. For example, in India, many reservoirs are constructed for irrigation purposes. By digging irrigation channels, farmers within the reservoir `command area` have access to this water.

Here, the command areas are obtained from the `India Water Resources Information System <https://gis.indiawris.gov.in/server/rest/services/SubInfoSysLCC/WRP_Old/MapServer/7/>`_. The API can be queried in various ways. Here, we manually collected a list of all IDs of the command areas through the API, and then download all files using an automated script.

To do so, first use :func:`preprocessing.3_scrape_command_areas.download` to download all files to the specified directory `COMMAND_AREAS_DIR`, then merge and export using :func:`preprocessing.3_scrape_command_areas.merge_and_export`. The resulting file will be saved in `DataDrive/GEB/input/routing/lakesreservoirs/command_areas.shp`.

At a later stage, the command areas must be manually linked to the hydrolakes, see :doc:`4_create_lakes_and_reservoirs`.

.. automodule:: preprocessing.3_scrape_command_areas
    :members: