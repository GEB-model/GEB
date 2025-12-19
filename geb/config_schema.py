"""Configuration schema for the GEB model."""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ForecastsConfig(BaseModel):
    """Configuration for forecasts."""

    use: bool = Field(False, description="Whether to use forecasts.")
    provider: str = Field("ECMWF", description="The forecast provider.")


class RegionConfig(BaseModel):
    """Configuration for region."""

    subbasin: int | None = Field(None, description="Subbasin ID.")


class GeneralConfig(BaseModel):
    """General model configuration."""

    start_time: date = Field(..., description="Model start time (YYYY-MM-DD).")
    end_time: date = Field(..., description="Model end time (YYYY-MM-DD).")
    spinup_time: date = Field(..., description="Start of the spinup time (YYYY-MM-DD).")
    input_folder: str = Field("input/", description="Path to the input folder.")
    preprocessing_folder: str = Field(
        "preprocessing/", description="Path to the preprocessing folder."
    )
    output_folder: str = Field("output/", description="Path to the output folder.")
    spinup_name: str = Field("spinup", description="Name of the spinup simulation.")
    simulation_root: str = Field(
        "simulation_root", description="Root directory for the simulation."
    )
    name: str = Field("default", description="Name of the simulation.")
    export_inital_on_spinup: bool = Field(
        True, description="Whether to export initial state on spinup."
    )
    simulate_forest: bool = Field(False, description="Whether to simulate forest.")
    forecasts: ForecastsConfig = Field(
        default_factory=ForecastsConfig, description="Forecast configuration."
    )
    region: RegionConfig = Field(
        default_factory=RegionConfig, description="Region configuration."
    )


class SFINCSConfig(BaseModel):
    """Configuration for SFINCS."""

    gpu: bool | Literal["auto"] = Field(
        "auto",
        description="True, False, 'auto'. If 'auto', it will check if a GPU is available using nvidia-smi.",
    )


class FloodEventConfig(BaseModel):
    """Configuration for a flood event."""

    start_time: datetime = Field(..., description="Start time of the event.")
    end_time: datetime = Field(..., description="End time of the event.")


class FloodsConfig(BaseModel):
    """Configuration for flood simulation."""

    simulate: bool = Field(False, description="Whether to simulate floods.")
    forcing_method: Literal["headwater_points", "accumulated_runoff"] = Field(
        "accumulated_runoff",
        description="Forcing method: 'headwater_points' or 'accumulated_runoff'.",
    )
    minimum_flood_depth: float = Field(0.05, description="Minimum flood depth (m).")
    flood_map_output_interval_seconds: int | None = Field(
        None,
        description="Interval in seconds to output flood maps. null means no output.",
    )
    grid_size_multiplier: int = Field(
        1,
        description="Must be integer. The first DEM is used for the grid. This multiplier increases the grid cell size (i.e., reduces resolution) by this factor.",
    )
    subgrid: bool = Field(False, description="Whether to use subgrid.")
    overwrite: bool | Literal["auto"] = Field(
        "auto",
        description="Whether to overwrite existing flood model. True, False, 'auto'. If 'auto', it will check if a GPU is available using nvidia-smi.",
    )
    force_overwrite: bool = Field(False, description="Whether to force overwrite.")
    coastal_only: bool = Field(
        False, description="Whether to simulate coastal floods only."
    )
    write_figures: bool = Field(
        True, description="Whether to generate and save diagnostic figures."
    )
    return_periods: list[int] = Field(
        [2, 5, 10, 25, 50, 100, 250, 500, 1000],
        description="Return periods for flood maps.",
    )
    flood_risk: bool = Field(False, description="Whether to calculate flood risk.")
    ncpus: int | Literal["auto"] = Field("auto", description="Number of CPUs to use.")
    events: list[FloodEventConfig] = Field(
        default_factory=list, description="List of flood events."
    )
    SFINCS: SFINCSConfig = Field(
        default_factory=SFINCSConfig, description="SFINCS configuration."
    )


class DamageConfig(BaseModel):
    """Configuration for damage calculation."""

    simulate: bool = Field(False, description="Whether to simulate damage.")


class ErosionConfig(BaseModel):
    """Configuration for erosion simulation."""

    simulate: bool = Field(False, description="Whether to simulate erosion.")


class HazardsConfig(BaseModel):
    """Configuration for hazards."""

    floods: FloodsConfig = Field(
        default_factory=FloodsConfig, description="Flood configuration."
    )
    damage: DamageConfig = Field(
        default_factory=DamageConfig, description="Damage configuration."
    )
    erosion: ErosionConfig = Field(
        default_factory=ErosionConfig, description="Erosion configuration."
    )


class RiverWidthParameters(BaseModel):
    """Parameters for river width calculation."""

    default_alpha: float = Field(7.2, description="Default alpha parameter.")
    beta: float = Field(0.5, description="Beta parameter.")


class RiverWidthConfig(BaseModel):
    """Configuration for river width."""

    parameters: RiverWidthParameters = Field(
        default_factory=RiverWidthParameters, description="River width parameters."
    )


class RiverDepthConfig(BaseModel):
    """Configuration for river depth."""

    method: Literal["manning", "power_law"] = Field(
        "manning",
        description="Method for river depth calculation: 'manning' or 'power_law'. If power law, alpha and beta must be set additionally in the parameters.",
    )


class RoutingConfig(BaseModel):
    """Configuration for routing."""

    algorithm: Literal["accuflux", "kinematic_wave"] = Field(
        "kinematic_wave",
        description="Routing algorithm: 'accuflux' or 'kinematic_wave'.",
    )
    river_width: RiverWidthConfig = Field(
        default_factory=RiverWidthConfig, description="River width configuration."
    )
    river_depth: RiverDepthConfig = Field(
        default_factory=RiverDepthConfig, description="River depth configuration."
    )


class HydrologyConfig(BaseModel):
    """Configuration for hydrology."""

    routing: RoutingConfig = Field(
        default_factory=RoutingConfig, description="Routing configuration."
    )


class MarketConfig(BaseModel):
    """Configuration for market agent."""

    dynamic_market: bool = Field(False, description="Whether to use dynamic market.")
    price_frequency: Literal["yearly"] = Field(
        "yearly", description="Frequency of price updates."
    )


class RiskPerceptionConfig(BaseModel):
    """Configuration for risk perception."""

    base: float = Field(1.6, description="Base risk perception.")
    coef: float = Field(-2.5, description="Coefficient for risk perception.")
    max: float = Field(10.0, description="Maximum risk perception.")
    min: float = Field(0.01, description="Minimum risk perception.")


class FloodRiskCalculationsConfig(BaseModel):
    """Configuration for flood risk calculations."""

    risk_perception: RiskPerceptionConfig = Field(
        default_factory=RiskPerceptionConfig,
        description="Risk perception configuration.",
    )


class ExpectedUtilityConfig(BaseModel):
    """Configuration for expected utility."""

    flood_risk_calculations: FloodRiskCalculationsConfig = Field(
        default_factory=FloodRiskCalculationsConfig,
        description="Flood risk calculations configuration.",
    )


class WaterDemandConfig(BaseModel):
    """Configuration for water demand."""

    method: str = Field("default", description="Method for water demand calculation.")


class HouseholdsConfig(BaseModel):
    """Configuration for households agent."""

    adapt: bool = Field(False, description="Whether households adapt.")
    warning_response: bool = Field(
        False, description="Whether households respond to warnings."
    )
    water_demand: WaterDemandConfig = Field(
        default_factory=WaterDemandConfig, description="Water demand configuration."
    )
    adjust_demand_factor: float = Field(
        1.0, description="Factor to adjust water demand."
    )
    expected_utility: ExpectedUtilityConfig = Field(
        default_factory=ExpectedUtilityConfig,
        description="Expected utility configuration.",
    )


class MicrocreditConfig(BaseModel):
    """Configuration for microcredit."""

    ruleset: str = Field("no-adaptation", description="Ruleset for microcredit.")
    interest_rate: float = Field(0.2, description="Interest rate.")
    loan_duration: int = Field(2, description="Loan duration (years).")
    loss_threshold: float = Field(25.0, description="Loss threshold.")


class SocialNetworkConfig(BaseModel):
    """Configuration for social network."""

    radius: float = Field(5000.0, description="Radius of social network (m).")
    size: int = Field(10, description="Number of neighbors in social network.")


class InsuranceConfig(BaseModel):
    """Configuration for insurance."""

    duration: int = Field(2, description="Duration of insurance (years).")
    seut_factor: float = Field(1.0, description="SEUT factor.")
    personal_insurance: dict[str, Any] = Field(
        default_factory=lambda: {"ruleset": "no-adaptation"},
        description="Personal insurance configuration.",
    )
    index_insurance: dict[str, Any] = Field(
        default_factory=lambda: {"ruleset": "no-adaptation"},
        description="Index insurance configuration.",
    )
    pr_insurance: dict[str, Any] = Field(
        default_factory=lambda: {"ruleset": "no-adaptation"},
        description="PR insurance configuration.",
    )


class CropSwitchingConfig(BaseModel):
    """Configuration for crop switching."""

    ruleset: str = Field("no-adaptation", description="Ruleset for crop switching.")


class AdaptationSprinklerConfig(BaseModel):
    """Configuration for sprinkler adaptation."""

    ruleset: str = Field(
        "no-adaptation", description="Ruleset for sprinkler adaptation."
    )
    lifespan_base: int = Field(25, description="Base lifespan (years).")
    lifespan_sprink_drip: int = Field(
        15, description="Lifespan for sprinkler/drip (years)."
    )
    loan_duration: int = Field(21, description="Loan duration (years).")
    decision_horizon: int = Field(20, description="Decision horizon (years).")
    m2_cost: float = Field(1.0, description="Cost per m2.")
    return_fraction_surface: float = Field(
        0.75, description="Return fraction for surface irrigation."
    )
    return_fraction_sprinkler: float = Field(
        0.3, description="Return fraction for sprinkler irrigation."
    )
    return_fraction_drip: float = Field(
        0.2, description="Return fraction for drip irrigation."
    )
    irr_eff_surface: float = Field(
        0.60, description="Irrigation efficiency for surface irrigation."
    )
    irr_eff_sprinkler: float = Field(
        0.75, description="Irrigation efficiency for sprinkler irrigation."
    )
    irr_eff_drip: float = Field(
        0.95, description="Irrigation efficiency for drip irrigation."
    )


class AdaptationIrrigationExpansionConfig(BaseModel):
    """Configuration for irrigation expansion adaptation."""

    ruleset: str = Field(
        "no-adaptation", description="Ruleset for irrigation expansion."
    )


class AdaptationWellConfig(BaseModel):
    """Configuration for well adaptation."""

    seut_factor: float = Field(1.0, description="SEUT factor.")
    loan_duration: int = Field(21, description="Loan duration (years).")
    lifespan: int = Field(20, description="Lifespan (years).")
    decision_horizon: int = Field(20, description="Decision horizon (years).")
    ruleset: str = Field("no-adaptation", description="Ruleset for well adaptation.")
    pump_hours: float = Field(3.5, description="Pump hours.")
    specific_weight_water: float = Field(
        9800.0, description="Specific weight of water (kg/m3*m/s2)."
    )
    max_initial_sat_thickness: float = Field(
        50.0, description="Maximum initial saturated thickness (m)."
    )
    well_yield: float = Field(0.00005, description="Well yield (m3/s).")
    pump_efficiency: float = Field(0.7, description="Pump efficiency.")
    energy_cost_rate: float = Field(0.074, description="Energy cost rate ($/KWh).")
    maintenance_factor: float = Field(0.07, description="Maintenance factor.")
    WHY_10: float = Field(82.0209974, description="Cost parameter WHY_10 ($/m).")
    WHY_20: float = Field(164.0, description="Cost parameter WHY_20 ($/m).")
    WHY_30: float = Field(50.0, description="Cost parameter WHY_30 ($/m).")


class DecisionsConfig(BaseModel):
    """Configuration for decisions."""

    decision_horizon: int = Field(10, description="Decision horizon (years).")
    expenditure_cap: float = Field(0.5, description="Expenditure cap.")


class WaterPriceConfig(BaseModel):
    """Configuration for water price."""

    water_costs_m3_reservoir: float = Field(
        0.0, description="Water costs per m3 from reservoir."
    )
    water_costs_m3_groundwater: float = Field(
        0.0, description="Water costs per m3 from groundwater."
    )
    water_costs_m3_channel: float = Field(
        0.0, description="Water costs per m3 from channel."
    )


class DroughtEventPerceptionConfig(BaseModel):
    """Configuration for drought event perception."""

    drought_threshold: float = Field(5.0, description="Drought threshold.")


class DroughtRiskPerceptionConfig(BaseModel):
    """Configuration for drought risk perception."""

    base: float = Field(1.6, description="Base risk perception.")
    coef: float = Field(-2.5, description="Coefficient for risk perception.")
    max: float = Field(10.0, description="Maximum risk perception.")
    min: float = Field(0.5, description="Minimum risk perception.")


class DroughtRiskCalculationsConfig(BaseModel):
    """Configuration for drought risk calculations."""

    event_perception: DroughtEventPerceptionConfig = Field(
        default_factory=DroughtEventPerceptionConfig,
        description="Event perception configuration.",
    )
    risk_perception: DroughtRiskPerceptionConfig = Field(
        default_factory=DroughtRiskPerceptionConfig,
        description="Risk perception configuration.",
    )


class FarmersExpectedUtilityConfig(BaseModel):
    """Configuration for farmers expected utility."""

    insurance: InsuranceConfig = Field(
        default_factory=InsuranceConfig, description="Insurance configuration."
    )
    crop_switching: CropSwitchingConfig = Field(
        default_factory=CropSwitchingConfig, description="Crop switching configuration."
    )


class FarmersConfig(BaseModel):
    """Configuration for farmers agent."""

    ruleset: str | None = Field(None, description="Ruleset for farmers.")
    base_management_yield_ratio: float = Field(
        1.0, description="Base management yield ratio."
    )
    farmers_going_out_of_business: bool = Field(
        False, description="Whether farmers go out of business."
    )
    cultivation_cost_fraction: float = Field(
        0.1, description="Cultivation cost fraction."
    )
    microcredit: MicrocreditConfig = Field(
        default_factory=MicrocreditConfig, description="Microcredit configuration."
    )
    return_fraction: float = Field(0.5, description="Return fraction.")
    social_network: SocialNetworkConfig = Field(
        default_factory=SocialNetworkConfig, description="Social network configuration."
    )
    expected_utility: FarmersExpectedUtilityConfig = Field(
        default_factory=FarmersExpectedUtilityConfig,
        description="Expected utility configuration.",
    )
    adaptation_sprinkler: AdaptationSprinklerConfig = Field(
        default_factory=AdaptationSprinklerConfig,
        description="Sprinkler adaptation configuration.",
    )
    adaptation_irrigation_expansion: AdaptationIrrigationExpansionConfig = Field(
        default_factory=AdaptationIrrigationExpansionConfig,
        description="Irrigation expansion adaptation configuration.",
    )
    adaptation_well: AdaptationWellConfig = Field(
        default_factory=AdaptationWellConfig,
        description="Well adaptation configuration.",
    )
    decisions: DecisionsConfig = Field(
        default_factory=DecisionsConfig, description="Decisions configuration."
    )
    water_price: WaterPriceConfig = Field(
        default_factory=WaterPriceConfig, description="Water price configuration."
    )
    drought_risk_calculations: DroughtRiskCalculationsConfig = Field(
        default_factory=DroughtRiskCalculationsConfig,
        description="Drought risk calculations configuration.",
    )


class ReservoirOperatorsConfig(BaseModel):
    """Configuration for reservoir operators."""

    ruleset: str = Field(
        "no-adaptation", description="Ruleset for reservoir operators."
    )
    equal_abstraction: bool = Field(
        False, description="Whether to use equal abstraction."
    )
    reservoir_M_factor: float = Field(0.1, description="Reservoir M factor.")


class SensitivityAnalysisConfig(BaseModel):
    """Configuration for sensitivity analysis."""

    risk_aversion_factor: float = Field(1.0, description="Risk aversion factor.")
    discount_rate_factor: float = Field(1.0, description="Discount rate factor.")
    interest_rate_factor: float = Field(1.0, description="Interest rate factor.")
    well_cost_factor: float = Field(1.0, description="Well cost factor.")
    drought_threshold_factor: float = Field(
        1.0, description="Drought threshold factor."
    )


class AgentSettingsConfig(BaseModel):
    """Configuration for agents."""

    market: MarketConfig = Field(
        default_factory=MarketConfig, description="Market agent configuration."
    )
    households: HouseholdsConfig = Field(
        default_factory=HouseholdsConfig, description="Households agent configuration."
    )
    farmers: FarmersConfig = Field(
        default_factory=FarmersConfig, description="Farmers agent configuration."
    )
    fix_activation_order: bool = Field(
        True, description="Whether to fix activation order."
    )
    reservoir_operators: ReservoirOperatorsConfig = Field(
        default_factory=ReservoirOperatorsConfig,
        description="Reservoir operators configuration.",
    )
    sensitivity_analysis: SensitivityAnalysisConfig = Field(
        default_factory=SensitivityAnalysisConfig,
        description="Sensitivity analysis configuration.",
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    logfile: str = Field("GEB.log", description="Log file path.")
    loglevel: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "DEBUG", description="Log level."
    )


class CropFarmersDrawConfig(BaseModel):
    """Configuration for drawing crop farmers."""

    draw_every_nth: int = Field(1, description="Draw every nth farmer.")


class DrawAgentsConfig(BaseModel):
    """Configuration for drawing agents."""

    crop_farmers: CropFarmersDrawConfig = Field(
        default_factory=CropFarmersDrawConfig,
        description="Crop farmers drawing configuration.",
    )


class DrawConfig(BaseModel):
    """Configuration for drawing."""

    draw_agents: DrawAgentsConfig = Field(
        default_factory=DrawAgentsConfig, description="Agents drawing configuration."
    )


class ReportConfigInternal(BaseModel):
    """Internal report configuration."""

    compression_level: int = Field(9, description="Compression level (1-22).")
    chunk_target_size_bytes: int = Field(
        100000000, description="Target chunk size in bytes."
    )


class ReportConfig(BaseModel):
    """Configuration for reporting."""

    model_config = ConfigDict(extra="allow")

    config: ReportConfigInternal = Field(
        default_factory=ReportConfigInternal,
        alias="_config",
        description="Internal report configuration.",
    )
    water_circle: bool = Field(
        False, alias="_water_circle", description="Whether to report water circle."
    )
    discharge_stations: bool = Field(
        True,
        alias="_discharge_stations",
        description="Whether to report discharge stations.",
    )
    outflow_points: bool = Field(
        True, alias="_outflow_points", description="Whether to report outflow points."
    )


class ParametersConfig(BaseModel):
    """Configuration for parameters."""

    mannings_n_multiplier: float = Field(1.0, description="Manning's n multiplier.")
    crop_factor_multiplier: float = Field(1.0, description="Crop factor multiplier.")
    saturated_hydraulic_conductivity_multiplier: float = Field(
        1.0, description="Saturated hydraulic conductivity multiplier."
    )
    reservoir_release_factor: float = Field(
        0.1, description="Reservoir release factor."
    )
    water_demand_multiplier_industry: float = Field(
        1.0, description="Water demand multiplier for industry."
    )
    lake_outflow_multiplier: float = Field(1.0, description="Lake outflow multiplier.")


class PlantFATEConfig(BaseModel):
    """Configuration for PlantFATE."""

    spinup_ini_file: str = Field(
        "data/plantFATE/p_spinup.ini", description="Path to spinup INI file."
    )
    run_ini_file: str = Field(
        "data/plantFATE/p_run.ini", description="Path to run INI file."
    )
    new_forest_ini_file: str = Field(
        "data/plantFATE/p_new_forest.ini", description="Path to new forest INI file."
    )
    new_forest: bool = Field(False, description="Whether to simulate new forest.")
    new_forest_filename: str | None = Field(
        None, description="Filename for new forest."
    )
    n_cells: str | int = Field("all", description="Number of cells to simulate.")


class CalibrationTargetsConfig(BaseModel):
    """Configuration for calibration targets."""

    KGE_discharge: float = Field(1.0, description="KGE discharge target.")


class DEAPConfig(BaseModel):
    """Configuration for DEAP."""

    use_multiprocessing: bool = Field(
        True, description="Whether to use multiprocessing."
    )
    ngen: int = Field(10, description="Number of generations.")
    mu: int = Field(30, description="Population size.")
    lambda_: int = Field(12, alias="lambda_", description="Number of offspring.")
    select_best: int = Field(10, description="Number of best individuals to select.")
    crossover_prob: float = Field(0.7, description="Crossover probability.")
    mutation_prob: float = Field(0.3, description="Mutation probability.")
    blend_alpha: float = Field(0.15, description="Blend alpha.")
    gaussian_sigma: float = Field(0.3, description="Gaussian sigma.")
    gaussian_indpb: float = Field(0.3, description="Gaussian independent probability.")


class CalibrationParameterConfig(BaseModel):
    """Configuration for a calibration parameter."""

    variable: str = Field(..., description="Variable path.")
    min: float = Field(..., description="Minimum value.")
    max: float = Field(..., description="Maximum value.")


class CalibrationConfig(BaseModel):
    """Configuration for calibration."""

    spinup_time: date = Field(..., description="Spinup start time (YYYY-MM-DD).")
    start_time: date = Field(..., description="Calibration start time (YYYY-MM-DD).")
    end_time: date = Field(..., description="Calibration end time (YYYY-MM-DD).")
    path: str = Field("calibration", description="Path to calibration output.")
    gpus: int = Field(0, description="Number of GPUs to use.")
    scenario: str = Field("no-adaptation", description="Scenario name.")
    calibration_targets: CalibrationTargetsConfig = Field(
        default_factory=CalibrationTargetsConfig, description="Calibration targets."
    )
    DEAP: DEAPConfig = Field(
        default_factory=DEAPConfig, description="DEAP configuration."
    )
    parameters: dict[str, CalibrationParameterConfig] = Field(
        default_factory=dict, description="Calibration parameters."
    )


class Config(BaseModel):
    """Main model configuration."""

    general: GeneralConfig = Field(..., description="General configuration.")
    hazards: HazardsConfig = Field(
        default_factory=HazardsConfig, description="Hazards configuration."
    )
    hydrology: HydrologyConfig = Field(
        default_factory=HydrologyConfig, description="Hydrology configuration."
    )
    agent_settings: AgentSettingsConfig = Field(
        default_factory=AgentSettingsConfig, description="Agent settings."
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration."
    )
    draw: DrawConfig = Field(
        default_factory=DrawConfig, description="Drawing configuration."
    )
    report: ReportConfig = Field(
        default_factory=ReportConfig, description="Reporting configuration."
    )
    parameters: ParametersConfig = Field(
        default_factory=ParametersConfig, description="Parameters configuration."
    )
    plantFATE: PlantFATEConfig = Field(
        default_factory=PlantFATEConfig, description="PlantFATE configuration."
    )
    calibration: CalibrationConfig = Field(
        default_factory=CalibrationConfig, description="Calibration configuration."
    )
