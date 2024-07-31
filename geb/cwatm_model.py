# -*- coding: utf-8 -*-
import numpy as np
import os

from cwatm.model import CWatM


class CWatM_Model(CWatM):
    """
    This class is used to initalize the CWatM model from GEB. Several static configuration files are read first, then several dynamic parameters are set based on the configuration of GEB. Then, the model frame is created that can then later be used to iteratate.

    Args:
        start_time: Start date of the model.
        n_steps: Number of steps that the model will run for.
        use_gpu: Whether the model can use a GPU.
    """

    def __init__(self) -> None:
        self.init_water_table_file = os.path.join(
            self.config["general"]["init_water_table"]
        )
        self.DynamicResAndLakes = False
        self.useSmallLakes = False
        self.CHECK_WATER_BALANCE = True
        self.crop_factor_calibration_factor = 1
        self.soilLayers = 3

        CWatM.__init__(self)

    def finalize(self) -> None:
        """
        Finalize the model
        """
        # finalize modflow model
        self.groundwater_modflow_module.modflow.finalize()

        if self.config["general"]["simulate_forest"]:
            for plantFATE_model in self.model.plantFATE:
                if plantFATE_model is not None:
                    plantFATE_model.finalize()

    def export_water_table(self) -> None:
        """Function to save required water table output to file."""
        dirname = os.path.dirname(self.init_water_table_file)
        os.makedirs(dirname, exist_ok=True)
        np.save(
            self.init_water_table_file,
            self.groundwater_modflow_module.modflow.decompress(
                self.groundwater_modflow_module.modflow.head
            ),
        )
