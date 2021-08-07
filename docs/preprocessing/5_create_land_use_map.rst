Create land use map
#############################

This file is used to create a land use map for use in GEB. The land use map is made according to the CWatM specification. However, all cropland is set to 1 (i.e., grasland/non irrigated land) and later, in :meth:`agents.Farmers.initiate_attributes`, these land areas are set to 2 (paddy irrigation) or 3 (non-paddy irrigation) when crops are currently growing on them.

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

The file :doc:`4_create_lakes_and_reservoirs` is used to create this map. First we derive a river map. Since no very high resolution river map of the area exists which works nicely with the used ldd, we derive a river map directly from the ldd. All cells with more than 100 upstream cells at 3 arcseconds resolution are considered as river. This threshold can be set with the `threshold` variable. If you have not previously obtained the number of upstream cells, download `Upstream Drainage Pixel` from the original `MERIT Hydro elevation map <http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/>`_. Decompress and save all output files in `DataDrive/GEB/original_data/merit_hydro_03sec/`.

In addition, download the GLC30 land use map from <TODO>. Place all files for your area in `DataDrive/GEB/original_data/GLC30`. The function :func:`4_create_lakes_and_reservoirs.merge_GLC30` automatically reprojects them using the properties of the submask (using nearest sampling) and merges all files in that folder, and cuts clips them to the study area.

Finally, the GLC30 land use classes are converted to CWatM land use classes (see table above). The mapping is as follows:

.. list-table:: CWatM land use types
   :header-rows: 1

   * - GLC30 index
     - GLC30 description
     - CWatM index
     - CWatM description
   * - 0
     - <TODO>
     - 1
     - Grasland/non irrigated land
   * - 10
     - <TODO>
     - 1
     - Grasland/non irrigated land
   * - 20
     - <TODO>
     - 0
     - Forest
   * - 30
     - <TODO>
     - 1
     - Grasland/non irrigated land
   * - 40
     - <TODO>
     - 1
     - Grasland/non irrigated land
   * - 50
     - <TODO>
     - 1
     - Grasland/non irrigated land
   * - 60
     - <TODO>
     - 5
     - Water covered area
   * - 70
     - <TODO>
     - 1
     - Grasland/non irrigated land
   * - 80
     - <TODO>
     - 4
     - Sealed area
   * - 90
     - <TODO>
     - 1
     - Grasland/non irrigated land
   * - 100
     - <TODO>
     - 1
     - Grasland/non irrigated land
   * - 255
     - <TODO>
     - 1
     - Grasland/non irrigated land

All rivers are also set to 5 (i.e., water covered area). Cultivated land (i.e., land use class 30 in GLC30 & land use class 1 in CWatM) is also exported separately, which is then used in :doc:`7_create_farmers` to ensure only cultivated land is used for farming purposes.

.. automodule:: preprocessing.5_create_land_use_map
    :members: