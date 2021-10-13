################################
Create land and reservoir map
################################

We need to collect some input data first. Download the `HydroLAKES dataset <https://www.hydrosheds.org/page/hydrolakes>`_ and unzip in `GEB/original_data`. The command areas obtained in :doc:`3_scrape_command_areas` now must be linked to the hydrolakes dataset. To do so, add an additional column "Hylak_id" (int) to ROOT/input/routing/lakesreservoirs/command_areas.shp, and enter the associated reservoir id from the HydroLAKES dataset.

Then run the `4_create_lakes_and_reservoirs.py` to create the input for CWatM. All lakes that are not set as reservoirs in the HydroLAKES dataset, but are associated with a command area are assumed to be reservoirs.

.. automodule:: preprocessing.4_create_lakes_and_reservoirs
    :members: