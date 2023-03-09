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
        binding["Ldd"] = os.path.join(INPUT, 'routing', 'kinematic', 'ldd.tif')
        binding["ElevationStD"] = os.path.join(INPUT, 'landsurface', 'topo', 'elvstd.tif')
        binding["CellArea"] = os.path.join(INPUT, 'areamaps', 'cell_area.tif')
        binding["albedoLand"] = os.path.join(INPUT, 'landsurface', 'albedo.nc:albedoLand')
        binding["albedoWater"] = os.path.join(INPUT, 'landsurface', 'albedo.nc:albedoWater')
        binding["cropgroupnumber"] = os.path.join(INPUT, 'soil', 'cropgrp.nc:cropgrp')
        binding["metaNetcdfFile"] = os.path.join(INPUT, 'metaNetcdf.xml')
        
        # meteo_path = os.path.join(DATA_FOLDER, 'GEB', 'input', 'meteo')
        climate_path = os.path.join(DATA_FOLDER, 'GEB', 'input', 'climate', 'Isi-Mip2', 'wfdei')
        topo_path = os.path.join(INPUT, 'landsurface', 'topo')
        forest_path = os.path.join(INPUT, 'landcover', 'forest')
        soil_path = os.path.join(INPUT, 'soil')
        grassland_path = os.path.join(INPUT, 'landcover', 'grassland')
        paddy_irr_path = os.path.join(INPUT, 'landcover', 'irrPaddy')
        nonpaddy_irr_path = os.path.join(INPUT, 'landcover', 'irrNonPaddy')
        groundwater_path = os.path.join(INPUT, 'groundwater')
        modflow_path = os.path.join(INPUT, 'groundwater', 'modflow')
        binding['PathGroundwaterModflow'] = modflow_path
        water_demand_path = os.path.join(DATA_FOLDER, 'GEB', 'input', 'demand')
        routing_path = os.path.join(INPUT, 'routing')
        res_lakes_path = os.path.join(INPUT, 'routing', 'lakesreservoirs')
        
        binding["downscale_wordclim_tavg"] = os.path.join(climate_path, 'worldclim_tavg.nc:wc_tavg')
        binding["downscale_wordclim_tmin"] = os.path.join(climate_path, 'worldclim_tmin.nc:wc_tmin')
        binding["downscale_wordclim_tmax"] = os.path.join(climate_path, 'worldclim_tmax.nc:wc_tmax ')
        binding["downscale_wordclim_prec"] = os.path.join(climate_path, 'worldclim_prec.nc:wc_prec')
        binding["KSat1"] = os.path.join(soil_path, "ksat1.nc:ksat1")
        binding["KSat2"] = os.path.join(soil_path, "ksat2.nc:ksat2")
        binding["KSat3"] = os.path.join(soil_path, "ksat3.nc:ksat3")
        binding["alpha1"] = os.path.join(soil_path, "alpha1.nc:alpha1")
        binding["alpha2"] = os.path.join(soil_path, "alpha2.nc:alpha2")
        binding["alpha3"] = os.path.join(soil_path, "alpha3.nc:alpha3")
        binding["lambda1"] = os.path.join(soil_path, "lambda1.nc:lambda1")
        binding["lambda2"] = os.path.join(soil_path, "lambda2.nc:lambda2")
        binding["lambda3"] = os.path.join(soil_path, "lambda3.nc:lambda3")
        binding["thetas1"] = os.path.join(soil_path, "thetas1.nc:thetas1")
        binding["thetas2"] = os.path.join(soil_path, "thetas2.nc:thetas2")
        binding["thetas3"] = os.path.join(soil_path, "thetas3.nc:thetas3")
        binding["thetar1"] = os.path.join(soil_path, "thetar1.nc:thetar1")
        binding["thetar2"] = os.path.join(soil_path, "thetar2.nc:thetar2")
        binding["thetar3"] = os.path.join(soil_path, "thetar3.nc:thetar3")
        binding["percolationImp"] = os.path.join(soil_path, "percolationImp.nc:percolationImp")
        binding["StorDepth1"] = os.path.join(soil_path, "storageDepth1.nc:storageDepth1")
        binding["StorDepth2"] = os.path.join(soil_path, "storageDepth2.nc:storageDepth2")
        binding["forest_KSat1"] = os.path.join(soil_path, "forest_ksat1.nc:forest_ksat1")
        binding["forest_KSat2"] = os.path.join(soil_path, "forest_ksat2.nc:forest_ksat2")
        binding["forest_KSat3"] = os.path.join(soil_path, "ksat3.nc:ksat3")
        binding["forest_alpha1"] = os.path.join(soil_path, "forest_alpha1.nc:forest_alpha1")
        binding["forest_alpha2"] = os.path.join(soil_path, "forest_alpha2.nc:forest_alpha2")
        binding["forest_alpha3"] = os.path.join(soil_path, "alpha3.nc:alpha3")
        binding["forest_lambda1"] = os.path.join(soil_path, "forest_lambda1.nc:forest_lambda1")
        binding["forest_lambda2"] = os.path.join(soil_path, "forest_lambda2.nc:forest_lambda2")
        binding["forest_lambda3"] = os.path.join(soil_path, "lambda3.nc:lambda3")
        binding["forest_thetas1"] = os.path.join(soil_path, "forest_thetas1.nc:forest_thetas1")
        binding["forest_thetas2"] = os.path.join(soil_path, "forest_thetas2.nc:forest_thetas2")
        binding["forest_thetas3"] = os.path.join(soil_path, "thetas3.nc:thetas3")
        binding["forest_thetar1"] = os.path.join(soil_path, "forest_thetar1.nc:forest_thetar1")
        binding["forest_thetar2"] = os.path.join(soil_path, "forest_thetar2.nc:forest_thetar2")
        binding["forest_thetar3"] = os.path.join(soil_path, "thetar3.nc:thetar3")
        binding["forest_rootFraction1"] = os.path.join(forest_path, "rootFraction1.nc:rootfraction1")
        binding["forest_maxRootDepth"] = os.path.join(forest_path, "maxRootDepth.nc:maxrootdepth")
        binding["forest_minSoilDepthFrac"] = os.path.join(forest_path, "minSoilDepthFrac.map")
        binding["forest_cropCoefficientNC"] = os.path.join(forest_path, "cropCoefficientForest_10days.nc:kc")
        binding["forest_interceptCapNC"] = os.path.join(forest_path, "interceptCapForest_10days.nc:interceptcap")
        binding["grassland_rootFraction1"] = os.path.join(grassland_path, "rootFraction1.nc:rootfraction1")
        binding["grassland_maxRootDepth"] = os.path.join(grassland_path, "maxRootDepth.nc:maxrootdepth")
        binding["grassland_minSoilDepthFrac"] = os.path.join(grassland_path, "minSoilDepthFrac.map")
        binding["grassland_cropCoefficientNC"] = os.path.join(grassland_path, "cropCoefficientGrassland_10days.nc")
        binding["grassland_interceptCapNC"] = os.path.join(grassland_path, "interceptCapGrassland_10days.nc:interceptcap")
        binding["irrPaddy_rootFraction1"] = os.path.join(paddy_irr_path, "rootFraction1.nc:rootfraction1")
        binding["irrPaddy_maxRootDepth"] = os.path.join(paddy_irr_path, "maxRootDepth.nc:maxrootdepth")
        binding["irrPaddy_minSoilDepthFrac"] = os.path.join(paddy_irr_path, "minSoilDepthFrac.map")
        binding["irrPaddy_cropCoefficientNC"] = os.path.join(paddy_irr_path, "cropCoefficientirrPaddy_10days.nc")
        binding["irrNonPaddy_rootFraction1"] = os.path.join(nonpaddy_irr_path, "rootFraction1.nc:rootfraction1")
        binding["irrNonPaddy_maxRootDepth"] = os.path.join(nonpaddy_irr_path, "maxRootDepth.nc:maxrootdepth")
        binding["irrNonPaddy_minSoilDepthFrac"] = os.path.join(nonpaddy_irr_path, "minSoilDepthFrac.map")
        binding["irrNonPaddy_cropCoefficientNC"] = os.path.join(nonpaddy_irr_path, "cropCoefficientirrNonPaddy_10days.nc")
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
        binding["chanMan"] = os.path.join(routing_path, "kinematic", "manning.tif")
        binding["chanLength"] = os.path.join(routing_path, "kinematic", "chanleng.tif")
        binding["chanWidth"] = os.path.join(routing_path, "kinematic", "chanwidth.tif")
        binding["chanDepth"] = os.path.join(routing_path, "kinematic", "chandepth.tif")
        binding["chanRatio"] = os.path.join(routing_path, "kinematic", "chanratio.tif")
        binding["chanGrad"] = os.path.join(routing_path, "kinematic", "changrad.tif")
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
        binding["#RhsMaps"] = os.path.join(climate_path, "hurs*")
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