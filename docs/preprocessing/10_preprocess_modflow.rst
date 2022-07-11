Preprocess MODFLOW data
#############################

CWatM works in a lon-lat grid, while MODFLOW works in a cartesian grid. Therefore a custom MODFLOW grid is created that encompasses the entirety of the CWatM mask. In addition, a mapping is created that maps the area of a given cell in the CWatM grid to a given MODFLOW cell. This mapping is then used in CWatM to convert flows between MODFLOW and CWatM. All required data for MODFLOW can also projected to the newly created MODFLOW grid using :meth:`preprocessing.9_preprocess_modflow.ModflowPreprocess.project_input_map`.

.. automodule:: preprocessing.10_preprocess_modflow
    :members: