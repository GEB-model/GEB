#############################
Create land use map
#############################

This file is used to create a land use map for use in GEB. The land use map is made according to the CWatM specification. However, all cropland is set to 1 (i.e., grasland/non irrigated land) and later,   

.. list-table:: CWatM land use types
   :header-rows: 1

   * - index
     - land use type
     - specified in
   * - 0
     - Forest
     - Land use map
   * - 1
     - Grasland/non irrigated land
     - Land use map
   * - 2
     - Paddy irrigation
     - :meth:`agents.Farmers.initiate_attributes`
   * - 3
     - Non-paddy irrigation
     - :meth:`agents.Farmers.initiate_attributes`
   * - 4
     - Sealed area
     - Land use map
   * - 5
     - Water covered area
     - Land use map


.. automodule:: preprocessing.5_create_land_use_map
    :members: