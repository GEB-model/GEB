# -*- coding: utf-8 -*-
import numpy as np  
import os
from cwatm.cwatm_model import CWATModel
from cwatm.management_modules.dynamicModel import ModelFrame
from cwatm.management_modules.configuration import parse_configuration, read_metanetcdf
from cwatm.management_modules.globals import dateVar, settingsfile, binding, option, outDir
from cwatm.management_modules.data_handling import cbinding
from cwatm.management_modules.timestep import checkifDate
from cwatm.run_cwatm import headerinfo
import datetime

from config import INPUT, DATA_FOLDER

class CWatM_Model(CWATModel):
    """
    This class is used to initalize the CWatM model from GEB. Several static configuration files are read first, then several dynamic parameters are set based on the configuration of GEB. Then, the model frame is created that can then later be used to iteratate.

    Args:
        start_time: Start date of the model.
        n_steps: Number of steps that the model will run for.
        settings: Filepath of the CWatM settingsfile. For full configuration options please refer to the `CWatM documentation <https://cwatm.iiasa.ac.at/>`.
        use_gpu: Whether the model can use a GPU.
    """
    def __init__(self, start_time: datetime.datetime, n_steps: int, settings: str) -> None:
        self.init_water_table_file = os.path.join(self.config['general']['init_water_table'])

        settingsfile.append(settings)
        parse_configuration(settings)

        subfolder = self.args.scenario
        if self.args.switch_crops:
            subfolder += '_switch_crops'

        outDir['OUTPUT'] = os.path.join(self.config['general']['report_folder'], subfolder)

        # calibration
        for parameter, value in self.config['parameters'].items():
            binding[parameter] = value
        
        # read_metanetcdf(cbinding('metaNetcdfFile'), 'metaNetcdfFile')
        binding['MaskMap'] = os.path.join(INPUT, 'areamaps', 'mask.tif')
        if 'gauges' in self.config['general']:
            gauges = self.config['general']['gauges']
            binding['Gauges'] = ' '.join([str(item) for sublist in gauges for item in sublist])
        else:
            binding['Gauges'] = f"{self.config['general']['poor_point'][0]} {self.config['general']['poor_point'][1]}"
        binding['StepStart'] = start_time.strftime('%d/%m/%Y')
        binding['SpinUp'] = '0'
        binding['StepEnd'] = str(n_steps)
        binding['Modflow_resolution'] = 1000
        
        # setting file paths for CWatM
        binding["Ldd"] = os.path.join(INPUT, 'maps', 'grid', 'ldd.tif')
        binding["ElevationStD"] = os.path.join(INPUT, 'maps', 'grid', 'elevation_STD.tif')
        binding["CellArea"] = os.path.join(INPUT, 'maps', 'grid', 'cell_area.tif')
        binding["albedoLand"] = os.path.join(INPUT, 'landsurface', 'albedo', 'albedo_land.nc:albedoLand')
        binding["albedoWater"] = os.path.join(INPUT, 'landsurface', 'albedo', 'albedo_water.nc:albedoWater')
        binding["cropgroupnumber"] = os.path.join(INPUT, 'soil', 'cropgrp.tif')
        binding["metaNetcdfFile"] = os.path.join(INPUT, 'metaNetcdf.xml')

        climate_path = os.path.join(DATA_FOLDER, 'GEB', 'input', 'climate', 'Isi-Mip2', 'wfdei')
        topo_path = os.path.join(INPUT, 'landsurface', 'topo')
        grassland_path = os.path.join(INPUT, 'landcover', 'grassland')
        groundwater_path = os.path.join(INPUT, 'groundwater')
        modflow_path = os.path.join(INPUT, 'groundwater', 'modflow')
        binding['PathGroundwaterModflow'] = modflow_path
        water_demand_path = os.path.join(DATA_FOLDER, 'GEB', 'input', 'demand')
        res_lakes_path = os.path.join(INPUT, 'routing', 'lakesreservoirs')
        
        binding["downscale_wordclim_tavg"] = os.path.join(climate_path, 'worldclim_tavg.nc:wc_tavg')
        binding["downscale_wordclim_tmin"] = os.path.join(climate_path, 'worldclim_tmin.nc:wc_tmin')
        binding["downscale_wordclim_tmax"] = os.path.join(climate_path, 'worldclim_tmax.nc:wc_tmax ')
        binding["downscale_wordclim_prec"] = os.path.join(climate_path, 'worldclim_prec.nc:wc_prec')

        path_soil = os.path.join(INPUT, 'soil')
        
        binding["KSat1"] = os.path.join(path_soil, "ksat1.tif")
        binding["KSat2"] = os.path.join(path_soil, "ksat2.tif")
        binding["KSat3"] = os.path.join(path_soil, "ksat3.tif")
        binding["alpha1"] = os.path.join(path_soil, "alpha1.tif")
        binding["alpha2"] = os.path.join(path_soil, "alpha2.tif")
        binding["alpha3"] = os.path.join(path_soil, "alpha3.tif")
        binding["lambda1"] = os.path.join(path_soil, "lambda1.tif")
        binding["lambda2"] = os.path.join(path_soil, "lambda2.tif")
        binding["lambda3"] = os.path.join(path_soil, "lambda3.tif")
        binding["thetas1"] = os.path.join(path_soil, "thetas1.tif")
        binding["thetas2"] = os.path.join(path_soil, "thetas2.tif")
        binding["thetas3"] = os.path.join(path_soil, "thetas3.tif")
        binding["thetar1"] = os.path.join(path_soil, "thetar1.tif")
        binding["thetar2"] = os.path.join(path_soil, "thetar2.tif")
        binding["thetar3"] = os.path.join(path_soil, "thetar3.tif")
        binding["percolationImp"] = os.path.join(path_soil, "percolation_impeded.tif")
        binding["StorDepth1"] = os.path.join(path_soil, "storage_depth1.tif")
        binding["StorDepth2"] = os.path.join(path_soil, "storage_depth2.tif")
        
        binding["forest_KSat1"] = os.path.join(path_soil, "ksat1.tif")
        binding["forest_KSat2"] = os.path.join(path_soil, "ksat2.tif")
        binding["forest_KSat3"] = os.path.join(path_soil, "ksat3.tif")
        binding["forest_alpha1"] = os.path.join(path_soil, "alpha1.tif")
        binding["forest_alpha2"] = os.path.join(path_soil, "alpha2.tif")
        binding["forest_alpha3"] = os.path.join(path_soil, "alpha3.tif")
        binding["forest_lambda1"] = os.path.join(path_soil, "lambda1.tif")
        binding["forest_lambda2"] = os.path.join(path_soil, "lambda2.tif")
        binding["forest_lambda3"] = os.path.join(path_soil, "lambda3.tif")
        binding["forest_thetas1"] = os.path.join(path_soil, "thetas1.tif")
        binding["forest_thetas2"] = os.path.join(path_soil, "thetas2.tif")
        binding["forest_thetas3"] = os.path.join(path_soil, "thetas3.tif")
        binding["forest_thetar1"] = os.path.join(path_soil, "thetar1.tif")
        binding["forest_thetar2"] = os.path.join(path_soil, "thetar2.tif")
        binding["forest_thetar3"] = os.path.join(path_soil, "thetar3.tif")
        
        binding["forest_rootFraction1"] = os.path.join(INPUT, 'landcover', 'forest', "rootFraction1_forest.tif")
        binding["forest_maxRootDepth"] = os.path.join(INPUT, 'landcover', 'forest', "maxRootDepth_forest.tif")
        binding["forest_cropCoefficientNC"] = os.path.join(INPUT, 'landcover', 'forest', "cropCoefficientForest_10days.nc:cropCoefficientForest_10days")
        binding["forest_interceptCapNC"] = os.path.join(INPUT, 'landcover', 'forest', "interceptCapForest_10days.nc:interceptCapForest_10days")
        
        binding["grassland_rootFraction1"] = os.path.join(INPUT, 'landcover', 'grassland', "rootFraction1_grassland.tif")
        binding["grassland_maxRootDepth"] = os.path.join(INPUT, 'landcover', 'grassland', "maxRootDepth_grassland.tif")
        binding["grassland_cropCoefficientNC"] = os.path.join(INPUT, 'landcover', 'grassland', "cropCoefficientGrassland_10days.nc:cropCoefficientGrassland_10days")
        binding["grassland_interceptCapNC"] = os.path.join(INPUT, 'landcover', 'grassland', "interceptCapGrassland_10days.nc:interceptCapGrassland_10days")
        
        binding["irrPaddy_rootFraction1"] = os.path.join(INPUT, 'landcover', 'irrPaddy', "rootFraction1_irrPaddy.tif")
        binding["irrPaddy_maxRootDepth"] = os.path.join(INPUT, 'landcover', 'irrPaddy', "maxRootDepth_irrPaddy.tif")
        binding["irrPaddy_cropCoefficientNC"] = os.path.join(INPUT, 'landcover', 'irrPaddy', "cropCoefficientirrPaddy_10days.nc:cropCoefficientirrPaddy_10days")
        
        binding["irrNonPaddy_rootFraction1"] = os.path.join(INPUT, 'landcover', 'irrNonPaddy', "rootFraction1_irrNonPaddy.tif")
        binding["irrNonPaddy_maxRootDepth"] = os.path.join(INPUT, 'landcover', 'irrNonPaddy', "maxRootDepth_irrNonPaddy.tif")
        binding["irrNonPaddy_cropCoefficientNC"] = os.path.join(INPUT, 'landcover', 'irrNonPaddy', "cropCoefficientirrNonPaddy_10days.nc:cropCoefficientirrNonPaddy_10days")
        
        binding["recessionCoeff"] = os.path.join(groundwater_path, "recessionCoeff.map")
        binding["specificYield"] = os.path.join(groundwater_path, "specificYield.map")
        binding["kSatAquifer"] = os.path.join(groundwater_path, "kSatAquifer.map")
        binding["topo_modflow"] = os.path.join(modflow_path, f"{binding['Modflow_resolution']}m", "elevation_modflow.tif")
        binding["riverPercentage"] = os.path.join(modflow_path, f"{binding['Modflow_resolution']}m", "RiverPercentage.npy")
        binding["cwatm_modflow_indices"] = os.path.join(modflow_path, f"{binding['Modflow_resolution']}m", "indices")
        binding["modflow_mask"] = os.path.join(modflow_path, f"{binding['Modflow_resolution']}m", "modflow_mask.tif")
        
        binding["domesticWaterDemandFile"]  = os.path.join(water_demand_path, "historical_dom_month_millionm3_5min_1961_2010.nc")
        binding["domesticWaterDemandFile_SSP2"]  = os.path.join(water_demand_path, "ssp2_dom_month_millionm3_5min_2005_2060.nc")
        binding["industryWaterDemandFile"] = os.path.join(water_demand_path, "historical_ind_year_millionm3_5min_1961_2010.nc")
        binding["industryWaterDemandFile_SSP2"] = os.path.join(water_demand_path, "ssp2_ind_year_millionm3_5min_2005_2060.nc")
        binding["livestockWaterDemandFile"] = os.path.join(water_demand_path, "historical_liv_month_millionm3_5min_1961_2010.nc")
        binding["livestockWaterDemandFile_SSP2"] = os.path.join(water_demand_path, "ssp2_liv_month_millionm3_5min_2005_2060.nc")
        
        binding["chanMan"] = os.path.join(INPUT, "routing", "kinematic", "mannings.tif")
        binding["chanLength"] = os.path.join(INPUT, "routing", "kinematic", "channel_length.tif")
        binding["chanWidth"] = os.path.join(INPUT, "routing", "kinematic", "channel_width.tif")
        binding["chanDepth"] = os.path.join(INPUT, "routing", "kinematic", "channel_depth.tif")
        binding["chanRatio"] = os.path.join(INPUT, "routing", "kinematic", "channel_ratio.tif")
        binding["chanGrad"] = os.path.join(INPUT, "routing", "kinematic", "channel_slope.tif")
        
        binding["waterBodyID"] = os.path.join(res_lakes_path, "lakesResID.tif")
        binding["hydroLakes"] = os.path.join(res_lakes_path, "hydrolakes.csv")
        binding["reservoir_command_areas"] =  os.path.join(res_lakes_path, "command_areas.tif")
        binding["waterBodyTyp"] = os.path.join(res_lakes_path, "lakesResType.tif")
        binding["waterBodyDis"] = os.path.join(res_lakes_path, "lakesResDis.tif")
        binding["waterBodyArea"] = os.path.join(res_lakes_path, "lakesResArea.tif")
        binding["smallLakesRes"] = os.path.join(res_lakes_path, "smallLakesRes.nc")
        binding["smallwaterBodyDis"] = os.path.join(res_lakes_path, "smallLakesResDis.nc")
        binding["waterBodyYear"] = os.path.join(res_lakes_path, "reservoir_year_construction.tif")
        binding["area_command_area_in_study_area"] = os.path.join(res_lakes_path, "area_command_area_in_study_area.tif")
        binding["TminMaps"] = os.path.join(climate_path, "tmin*")
        binding["TmaxMaps"] = os.path.join(climate_path, "tmax*")
        binding["PSurfMaps"] = os.path.join(climate_path, "ps*")
        binding["RhsMaps"] = os.path.join(climate_path, "hurs*")
        binding["QAirMaps"] = os.path.join(climate_path, "huss*")
        binding["WindMaps"] = os.path.join(climate_path, "wind*")
        binding["RSDSMaps"] = os.path.join(climate_path, "rsds*")
        binding["RSDLMaps"] = os.path.join(climate_path, "rlds*")
        binding["PrecipitationMaps"] = os.path.join(climate_path, "pr*.nc")
        binding["TavgMaps"] = os.path.join(climate_path, "tavg*")
        binding["slopeLength"] = os.path.join(topo_path, "slopeLength.map")
        binding["relativeElevation"] = os.path.join(topo_path, "dzRel.nc")

        checkifDate('StepStart', 'StepEnd', 'SpinUp', cbinding('PrecipitationMaps'))
        headerinfo()

        CWATModel.__init__(self)
        self.stCWATM = ModelFrame(self, firstTimestep=dateVar["intStart"], lastTimeStep=dateVar["intEnd"])
        self.stCWATM.initialize_run()
        CWATModel.dateVar = dateVar

    def step(self, n: int) -> None:
        """Performs n number of (daily) steps in CWatM.
        
        Args:
            n: Number of timesteps to perform.
        """
        for _ in range(n):
            self.stCWATM.step()

    def export_water_table(self) -> None:
        """Function to save required water table output to file."""
        dirname = os.path.dirname(self.init_water_table_file)
        os.makedirs(dirname, exist_ok=True)
        np.save(self.init_water_table_file, self.groundwater_modflow_module.modflow.decompress(self.groundwater_modflow_module.modflow.head))