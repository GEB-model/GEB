# -*- coding: utf-8 -*-
import numpy as np
import os
import datetime

from cwatm.cwatm_model import CWATModel
from cwatm.management_modules.dynamicModel import ModelFrame
from cwatm.management_modules.globals import binding, outDir, Flags, option
from cwatm.run_cwatm import headerinfo


class CWatM_Model(CWATModel):
    """
    This class is used to initalize the CWatM model from GEB. Several static configuration files are read first, then several dynamic parameters are set based on the configuration of GEB. Then, the model frame is created that can then later be used to iteratate.

    Args:
        start_time: Start date of the model.
        n_steps: Number of steps that the model will run for.
        use_gpu: Whether the model can use a GPU.
    """

    def __init__(self, start_time: datetime.datetime, n_steps: int) -> None:
        self.init_water_table_file = os.path.join(
            self.config["general"]["init_water_table"]
        )

        outDir["OUTPUT"] = self.report_folder
        Flags["quiet"] = True
        Flags["veryquiet"] = True

        # calibration
        for parameter, value in self.config["parameters"].items():
            binding[parameter] = value

        binding["MaskMap"] = self.model_structure["grid"]["areamaps/grid_mask"]
        if "gauges" in self.config["general"]:
            gauges = self.config["general"]["gauges"]
            binding["Gauges"] = " ".join(
                [str(item) for sublist in gauges for item in sublist]
            )
        else:
            binding["Gauges"] = (
                f"{self.config['general']['pour_point'][0]} {self.config['general']['pour_point'][1]}"
            )
        binding["StepStart"] = start_time.strftime("%d/%m/%Y")
        binding["SpinUp"] = "0"
        binding["StepEnd"] = str(n_steps)

        # setting file paths for CWatM
        binding["Ldd"] = self.model_structure["grid"]["routing/kinematic/ldd"]
        binding["ElevationStD"] = self.model_structure["grid"][
            "landsurface/topo/elevation_STD"
        ]
        binding["CellArea"] = self.model_structure["grid"]["areamaps/cell_area"]
        binding["cropgroupnumber"] = self.model_structure["grid"]["soil/cropgrp"]
        binding["KSat1"] = self.model_structure["grid"]["soil/ksat1"]
        binding["KSat2"] = self.model_structure["grid"]["soil/ksat2"]
        binding["KSat3"] = self.model_structure["grid"]["soil/ksat3"]
        binding["alpha1"] = self.model_structure["grid"]["soil/alpha1"]
        binding["alpha2"] = self.model_structure["grid"]["soil/alpha2"]
        binding["alpha3"] = self.model_structure["grid"]["soil/alpha3"]
        binding["lambda1"] = self.model_structure["grid"]["soil/lambda1"]
        binding["lambda2"] = self.model_structure["grid"]["soil/lambda2"]
        binding["lambda3"] = self.model_structure["grid"]["soil/lambda3"]
        binding["thetas1"] = self.model_structure["grid"]["soil/thetas1"]
        binding["thetas2"] = self.model_structure["grid"]["soil/thetas2"]
        binding["thetas3"] = self.model_structure["grid"]["soil/thetas3"]
        binding["thetar1"] = self.model_structure["grid"]["soil/thetar1"]
        binding["thetar2"] = self.model_structure["grid"]["soil/thetar2"]
        binding["thetar3"] = self.model_structure["grid"]["soil/thetar3"]
        binding["percolationImp"] = self.model_structure["grid"][
            "soil/percolation_impeded"
        ]

        binding["forest_KSat1"] = self.model_structure["grid"]["soil/ksat1"]
        binding["forest_KSat2"] = self.model_structure["grid"]["soil/ksat2"]
        binding["forest_KSat3"] = self.model_structure["grid"]["soil/ksat3"]
        binding["forest_alpha1"] = self.model_structure["grid"]["soil/alpha1"]
        binding["forest_alpha2"] = self.model_structure["grid"]["soil/alpha2"]
        binding["forest_alpha3"] = self.model_structure["grid"]["soil/alpha3"]
        binding["forest_lambda1"] = self.model_structure["grid"]["soil/lambda1"]
        binding["forest_lambda2"] = self.model_structure["grid"]["soil/lambda2"]
        binding["forest_lambda3"] = self.model_structure["grid"]["soil/lambda3"]
        binding["forest_thetas1"] = self.model_structure["grid"]["soil/thetas1"]
        binding["forest_thetas2"] = self.model_structure["grid"]["soil/thetas2"]
        binding["forest_thetas3"] = self.model_structure["grid"]["soil/thetas3"]
        binding["forest_thetar1"] = self.model_structure["grid"]["soil/thetar1"]
        binding["forest_thetar2"] = self.model_structure["grid"]["soil/thetar2"]
        binding["forest_thetar3"] = self.model_structure["grid"]["soil/thetar3"]

        binding["chanMan"] = self.model_structure["grid"]["routing/kinematic/mannings"]
        binding["chanLength"] = self.model_structure["grid"][
            "routing/kinematic/channel_length"
        ]
        binding["chanWidth"] = self.model_structure["grid"][
            "routing/kinematic/channel_width"
        ]
        binding["chanDepth"] = self.model_structure["grid"][
            "routing/kinematic/channel_depth"
        ]
        binding["chanRatio"] = self.model_structure["grid"][
            "routing/kinematic/channel_ratio"
        ]
        binding["chanGrad"] = self.model_structure["grid"][
            "routing/kinematic/channel_slope"
        ]

        binding["waterBodyID"] = self.model_structure["grid"][
            "routing/lakesreservoirs/lakesResID"
        ]
        binding["reservoir_command_areas"] = self.model_structure["subgrid"][
            "routing/lakesreservoirs/subcommand_areas"
        ]

        binding["crop_factor_calibration_factor"] = 1

        option["inflow"] = False
        option["calcWaterBalance"] = True
        binding["DynamicResAndLakes"] = False
        binding["useSmallLakes"] = False
        binding["chanBeta"] = 0.6
        binding["chanGradMin"] = 0.0001
        binding["SnowWaterEquivalent"] = 0.45
        binding["Afrost"] = 0.97
        binding["Kfrost"] = 0.57
        binding["FrostIndexThreshold"] = 56.0
        binding["AlbedoSoil"] = 0.15
        binding["AlbedoWater"] = 0.05
        binding["AlbedoCanopy"] = 0.23
        binding["NumberSnowLayers"] = 1.0
        binding["GlacierTransportZone"] = 1.0
        binding["TemperatureLapseRate"] = 0.0065
        binding["SnowFactor"] = 1.0
        binding["SnowSeasonAdj"] = 0.001
        binding["TempMelt"] = 1.0
        binding["TempSnow"] = 1.0
        binding["IceMeltCoef"] = 0.007
        binding["initial_water_table_depth"] = 2
        binding["depth_underlakes"] = 1.5
        binding["thickness"] = 100.0
        binding["use_soildepth_as_GWtop"] = True
        binding["correct_soildepth_underlakes"] = True
        binding["leakageriver_permea"] = 0.001
        binding["leakagelake_permea"] = 0.001
        binding["forest_arnoBeta"] = 0.2
        binding["grassland_arnoBeta"] = 0.0
        binding["irrPaddy_arnoBeta"] = 0.2
        binding["irrNonPaddy_arnoBeta"] = 0.2
        binding["grassland_minInterceptCap"] = 0.001
        binding["forest_minInterceptCap"] = 0.001
        binding["irrPaddy_minInterceptCap"] = 0.001
        binding["irrNonPaddy_minInterceptCap"] = 0.001
        binding["sealed_minInterceptCap"] = 0.001
        binding["water_minInterceptCap"] = 0.0

        headerinfo()

        CWATModel.__init__(self)
        self.stCWATM = ModelFrame(self)

    def step(self, n: int) -> None:
        """Performs n number of (daily) steps in CWatM.

        Args:
            n: Number of timesteps to perform.
        """
        for _ in range(n):
            self.stCWATM.step()

    def finalize(self) -> None:
        """Finalizes CWatM."""
        self.stCWATM.finalize()

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
