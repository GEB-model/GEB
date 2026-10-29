"""Configuration schema for the GEB model."""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ForecastsConfig(BaseModel):
    """Configuration for forecasts."""

    use: bool = Field(..., description="Whether to use forecasts.")
    provider: str = Field(..., description="The forecast provider.")
    processing: Literal[
        "merged_control_ensemble", "control_forecast", "probabilistic_forecast"
    ] = Field(..., description="Forecast processing option.")


class RegionConfig(BaseModel):
    """Configuration for region."""

    subbasin: int | list[int] | None = Field(
        None, description="Subbasin ID or list of subbasin IDs."
    )


class SFINCSConfig(BaseModel):
    """Configuration for SFINCS."""

    gpu: bool | Literal["auto"] = Field(
        ...,
        description="True, False, 'auto'. If 'auto', it will check if a GPU is available using nvidia-smi.",
    )


class FloodEventConfig(BaseModel):
    """Configuration for a flood event."""

    start_time: datetime = Field(..., description="Start time of the event.")
    end_time: datetime = Field(..., description="End time of the event.")


class ShapeConfig(BaseModel):
    """Configuration for the hydrograph shape method used in return-period maps."""

    method: Literal["triangular", "direct", "anchor"] = Field(
        ...,
        description="Hydrograph shape method: 'triangular' (symmetric), 'direct' (historical shape at target discharge), or 'anchor' (historical shape at Q_2 rescaled to target).",
    )
    window_days: float = Field(
        ...,
        gt=0.0,
        description="Days before and after peak to extract when deriving shape from historical events (total window = 2 × window_days).",
    )
    tolerance: float = Field(
        ...,
        ge=0.0,
        description="Fractional tolerance around anchor discharge for event selection (e.g. 0.1 = ±10%).",
    )


class FloodProtectionStandardConfig(BaseModel):
    """Configuration for flood protection standard settings."""

    mode: Literal["auto", "manual"] = Field(
        ...,
        description="Flood protection standard mode: 'auto' (derive from data) or 'manual' (use configured value)",
    )
    manual_value: int = Field(
        ...,
        ge=1,
        description="Flood protection standard return period used when mode is 'manual' (years).",
    )


class FloodsConfig(BaseModel):
    """Configuration for flood simulation."""

    simulate: bool = Field(..., description="Whether to simulate floods.")
    subbasins: list[int] | Literal["all"] | Literal["auto"] = Field(
        ...,
        description="Subbasin ID, list of subbasin IDs, or 'all' to simulate all subbasins. Only works for flood events currently.",
    )
    forcing_method: Literal["headwater_points", "accumulated_runoff"] = Field(
        ...,
        description="Forcing method: 'headwater_points' or 'accumulated_runoff'.",
    )
    minimum_flood_depth: float = Field(
        ..., ge=0.0, description="Minimum flood depth (m)."
    )
    flood_map_output_interval_seconds: int | None = Field(
        ...,
        description="Interval in seconds to output flood maps. null means no output.",
    )
    grid_size_multiplier: int = Field(
        ...,
        ge=1,
        description="Must be integer. The first DEM is used for the grid. This multiplier increases the grid cell size (i.e., reduces resolution) by this factor.",
    )
    subgrid: bool = Field(..., description="Whether to use subgrid.")
    overwrite: bool | Literal["auto"] = Field(
        ...,
        description="Whether to overwrite existing flood model. True, False, 'auto'. If 'auto', it will check if a GPU is available using nvidia-smi.",
    )
    coastal_only: bool = Field(
        ..., description="Whether to simulate coastal floods only."
    )
    write_figures: bool = Field(
        ..., description="Whether to generate and save diagnostic figures."
    )
    return_periods: list[int] = Field(
        ...,
        description="Return periods for flood maps.",
    )
    flood_protection_standard: FloodProtectionStandardConfig = Field(
        ...,
        description="Flood protection standard settings.",
    )
    p_value_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Anderson-Darling p-value threshold for threshold selection.",
    )
    selection_strategy: Literal["first_significant", "best_fit"] = Field(
        ...,
        description="Strategy for selecting the best threshold: 'first_significant' (early stopping) or 'best_fit' (maximize p-value).",
    )
    fixed_shape: float | None = Field(
        ...,
        description="Value to fix the shape parameter (xi) of the GPD. Set to 0.0 to force an Exponential (Gumbel) tail, or null to allow it to be fitted.",
    )
    events: list[FloodEventConfig] = Field(..., description="List of flood events.")
    hydrograph_shape: ShapeConfig = Field(
        ...,
        description="Hydrograph shape configuration for return-period maps.",
    )
    run_for_validation_events: bool = Field(
        ...,
        description="Whether to run flood model for validation events (i.e., events with observed flood maps).",
    )
    SFINCS: SFINCSConfig = Field(..., description="SFINCS configuration.")
    detect_floods_from_discharge: bool = Field(
        ...,
        description="If True, floods are detected based on discharge thresholds at a specific location.",
    )
    discharge_threshold: float = Field(
        ...,
        description="Discharge threshold in m3/s for flood detection.",
    )
    threshold_location: list[str | float] = Field(
        ...,
        description="[lon, lat] of the location where the discharge threshold is applied.",
    )
    observation_files: list[str] = Field(
        ...,
        description="List of observation files for all floods.",
    )

    @field_validator("return_periods")
    @classmethod
    def validate_return_periods(cls, v: list[int]) -> list[int]:
        """Validate that return periods are positive integers.

        Args:
            v: List of return periods to validate (years).

        Returns:
            The validated list of positive return periods.

        Raises:
            ValueError: If return_periods is empty or contains non-positive integers.
        """
        if not v:
            raise ValueError("return_periods list cannot be empty.")
        for rp in v:
            if not isinstance(rp, int) or isinstance(rp, bool) or rp <= 0:
                raise ValueError(f"Return period must be a positive integer, got: {rp}")
        return v


class DamageConfig(BaseModel):
    """Configuration for damage calculation."""

    simulate: bool = Field(..., description="Whether to simulate damage.")


class ErosionConfig(BaseModel):
    """Configuration for erosion simulation."""

    simulate: bool = Field(..., description="Whether to simulate erosion.")


class HazardsConfig(BaseModel):
    """Configuration for hazards."""

    floods: FloodsConfig = Field(..., description="Flood configuration.")
    damage: DamageConfig = Field(..., description="Damage configuration.")


class RiverWidthConfig(BaseModel):
    """Configuration for river width."""

    default_alpha: float = Field(
        ...,
        description="Default alpha parameter for bankfull width = alpha * Q^beta.",
    )
    beta: float = Field(
        ...,
        description="Beta parameter for bankfull width = alpha * Q^beta.",
    )


class RiverDepthConfig(BaseModel):
    """Configuration for river depth."""

    min_depth_m: float = Field(
        ...,
        ge=0.0,
        description="Minimum allowable bankfull depth (m).",
    )
    shape_exponent: float = Field(
        ...,
        gt=0.0,
        description="Power-law cross-sectional shape exponent r (0.5 for parabolic channel).",
    )
    use_observed_width_as_bankfull: bool = Field(
        ...,
        description=(
            "Whether to use observed channel width directly as bankfull width. If False, "
            "reverts to using reach-specific alpha for expected width."
        ),
    )


class RetentionBasinsConfig(BaseModel):
    """Configuration for retention basins in routing."""

    release_threshold_factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Factor to multiply the activation threshold by to get the release threshold.",
    )
    activation_threshold_return_period_years: float = Field(
        ...,
        gt=0.0,
        description="Return period in years used to calculate the activation threshold for retention basins.",
    )


class RoutingConfig(BaseModel):
    """Configuration for routing."""

    max_threads: int | None = Field(
        ...,
        ge=1,
        description=(
            "Maximum number of CPU threads to allocate for the local inertial routing "
            "substep solver. If None, automatically determined based on domain size, "
            "available physical cores, and Slurm allocation."
        ),
    )
    fix_river_geometry_after_spinup: bool = Field(
        ...,
        description=(
            "Whether to keep river geometry (channel width and depth) fixed after spinup. "
            "If True, river geometry is only updated during the spinup period and remains fixed during the run."
        ),
    )
    mannings_n_multiplier: float = Field(
        ..., ge=0.0, description="Manning's n multiplier."
    )
    bankfull_discharge_multiplier: float = Field(
        ...,
        gt=0.0,
        description="Calibration multiplier for bankfull discharge (Q_bf) used to determine channel geometry and overbank spill capacity.",
    )
    lake_outflow_multiplier: float = Field(
        ..., ge=0.0, description="Lake outflow multiplier."
    )
    retention_basins: RetentionBasinsConfig = Field(
        ..., description="Retention basins configuration."
    )
    riverbed_hydraulic_conductivity_multiplier: float = Field(
        ...,
        ge=0.0,
        description=(
            "Multiplier applied to underlying aquifer hydraulic conductivity "
            "to calculate streambed infiltration conductivity."
        ),
    )
    river_width: RiverWidthConfig = Field(..., description="River width configuration.")
    river_depth: RiverDepthConfig = Field(..., description="River depth configuration.")


class LandSurfaceConfig(BaseModel):
    """Configuration for land surface hydrology."""

    crop_factor_multiplier: float = Field(
        ..., ge=0.0, description="Crop factor multiplier."
    )
    interflow_multiplier: float = Field(
        ..., ge=0.0, description="Interflow multiplier."
    )
    saturated_hydraulic_conductivity_multiplier: float = Field(
        ..., ge=0.0, description="Saturated hydraulic conductivity multiplier."
    )
    vertical_saturated_hydraulic_conductivity_multiplier: float = Field(
        ..., ge=0.0, description="Vertical saturated hydraulic conductivity multiplier."
    )
    variable_runoff_shape_beta_multiplier: float = Field(
        ..., ge=0.0, description="Variable runoff shape beta multiplier."
    )


class GroundWaterConfig(BaseModel):
    """Configuration for groundwater hydrology."""

    hydraulic_conductivity_multiplier: float = Field(
        ..., ge=0.0, description="Groundwater hydraulic conductivity multiplier."
    )


class RunoffConcentratorConfig(BaseModel):
    """Configuration for runoff concentration."""

    runoff_concentration_factor: float = Field(
        ...,
        gt=0.0,
        description="Multiplier for slope-dependent overland runoff concentration time using the Kerby (1959) overland flow method.",
    )


class DischargeEvaluationConfig(BaseModel):
    """Configuration for discharge evaluation."""

    minimum_upstream_area_km2: float = Field(
        ...,
        ge=0.0,
        description="Minimum modeled upstream area for stations included in discharge evaluation (km2).",
    )
    minimum_timeseries_length_years: float = Field(
        ...,
        ge=0.0,
        description="Minimum total paired observation-simulation data for stations included in discharge evaluation (years; continuity is not required).",
    )


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    discharge: DischargeEvaluationConfig = Field(
        ...,
        description="Discharge evaluation configuration.",
    )


HydrologyEvaluationConfig = EvaluationConfig


class KarstConfig(BaseModel):
    """Karst capture and release settings."""

    model_config = ConfigDict(extra="forbid")

    capture_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of runoff and interflow captured on karst land below the swallowing ceiling (0–1).",
    )
    release_time_days: float = Field(
        ...,
        gt=0.0,
        allow_inf_nan=False,
        description="Exponential release time (days): after this many dry days, exp(-1) of the initial storage remains (about 36.8%).",
    )
    vertical_conductivity_multiplier: float = Field(
        ...,
        ge=1.0,
        description="Multiplier for vertical saturated hydraulic conductivity in karst areas (>= 1.0).",
    )


class HydrologyConfig(BaseModel):
    """Configuration for hydrology."""

    simulate_forest: bool = Field(..., description="Whether to simulate forest.")
    hydrological_year_start_month: int = Field(
        ..., ge=1, le=12, description="Hydrological year start month (1-12)."
    )
    erosion: ErosionConfig = Field(..., description="Erosion configuration.")
    landsurface: LandSurfaceConfig = Field(
        ..., description="Land surface configuration."
    )
    groundwater: GroundWaterConfig = Field(
        ..., description="Groundwater configuration."
    )
    routing: RoutingConfig = Field(..., description="Routing configuration.")
    runoff_concentrator: RunoffConcentratorConfig = Field(
        ...,
        description="Runoff concentrator configuration.",
    )
    karst: KarstConfig = Field(..., description="Karst recharge settings.")


class MarketConfig(BaseModel):
    """Configuration for market agent."""

    dynamic_market: bool = Field(..., description="Whether to use dynamic market.")
    price_frequency: Literal["yearly"] = Field(
        ..., description="Frequency of price updates."
    )


class GovernmentAdaptationConfig(BaseModel):
    """Configuration for government-led adaptation policy."""

    enabled: bool = Field(
        ..., description="Whether to enable government adaptation policy."
    )
    mode: Literal["cba", "threshold"] = Field(
        ...,
        description="Adaptation policy mode: 'cba' (cost-benefit analysis) or 'threshold'.",
    )
    EAD_threshold: float = Field(
        ...,
        ge=0.0,
        description="Expected annual damage threshold that can trigger adaptation action.",
    )
    equity_indicator_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Equity indicator threshold between 0 and 1.",
    )
    ecosystem_indicator_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ecosystem indicator threshold between 0 and 1.",
    )
    adaptation_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of households selected for adaptation action.",
    )
    dike_elevation_cost_per_meter_usd: float = Field(
        ...,
        ge=0.0,
        description="Unit cost for raising a dike by 1 meter over 1 meter length (USD/m).",
    )
    dike_maintenance_cost_per_year_usd: float = Field(
        ...,
        ge=0.0,
        description="Annual maintenance cost for a dike segment of 1 meter length (USD/year).",
    )


class GovernmentConfig(BaseModel):
    """Configuration for government agent."""

    plant_forest: bool = Field(
        ...,
        description="Whether to enable forest-planting policy for reforestation scenarios.",
    )
    adaptation: GovernmentAdaptationConfig = Field(
        ...,
        description="Government adaptation policy configuration.",
    )


class RiskPerceptionConfig(BaseModel):
    """Configuration for risk perception."""

    base: float = Field(..., description="Base risk perception.")
    coef: float = Field(..., description="Coefficient for risk perception.")
    max: float = Field(..., description="Maximum risk perception.")
    min: float = Field(..., description="Minimum risk perception.")


class FloodRiskCalculationsConfig(BaseModel):
    """Configuration for flood risk calculations."""

    risk_perception: RiskPerceptionConfig = Field(
        ...,
        description="Risk perception configuration.",
    )


class ExpectedUtilityConfig(BaseModel):
    """Configuration for expected utility."""

    flood_risk_calculations: FloodRiskCalculationsConfig = Field(
        ...,
        description="Flood risk calculations configuration.",
    )


class WaterDemandConfig(BaseModel):
    """Configuration for water demand."""

    method: str = Field(..., description="Method for water demand calculation.")


class WarningResponseRatesConfig(BaseModel):
    """Configuration for warning response rates."""

    area_based_warnings: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Response rate for area-based warnings (0.0-1.0).",
    )
    building_based_warnings: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Response rate for building-based warnings with specific measures (0.0-1.0).",
    )


class ResidentialBuildingsWarningTargetConfig(BaseModel):
    """Configuration for residential buildings warning target."""

    enabled: bool = Field(
        ...,
        description="Enable water level based warnings with measures for residential buildings.",
    )
    warning_type: Literal["area_based", "building_based"] = Field(
        ...,
        description="Type of warnings: 'area_based' for basic flood warnings or 'building_based' for impact-based warnings.",
    )


class CriticalInfrastructureWarningTargetConfig(BaseModel):
    """Configuration for critical infrastructure warning target."""

    enabled: bool = Field(
        ...,
        description="Enable water level based warnings for critical infrastructure.",
    )
    get_critical_facilities: bool = Field(
        ...,
        description="If True, critical facilities (vulnerable and emergency) are extracted from OSM using the catchment polygon as boundary.",
    )
    asset_type: list[str] = Field(
        ...,
        description="Asset types for critical infrastructure warnings.",
    )


class WarningTargetConfig(BaseModel):
    """Configuration for warning targets."""

    residential_buildings: ResidentialBuildingsWarningTargetConfig = Field(
        ...,
        description="Residential buildings warning target configuration.",
    )
    critical_infrastructure: CriticalInfrastructureWarningTargetConfig = Field(
        ...,
        description="Critical infrastructure warning target configuration.",
    )


class WarningSystemConfig(BaseModel):
    """Configuration for early warning system."""

    probability_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Flood probability threshold for issuing warnings (0.0-1.0).",
    )
    area_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Area threshold (percentage of postal code area) for warnings (0.0-1.0).",
    )
    building_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Building threshold (percentage of buildings in postal code) for warnings (0.0-1.0).",
    )
    communication_efficiency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Communication efficiency: fraction of households that can be reached (0.0-1.0).",
    )
    weight_by_socioeconomic_factors: bool = Field(
        ...,
        description="Whether to consider socioeconomic factors in the distribution of warnings.",
    )
    damage_reduction_over_leadtime: bool = Field(
        ...,
        description="Whether to consider the reduction of damage over lead time due to more effective warnings.",
    )
    evacuation_lead_time_threshold: float = Field(
        ...,
        ge=0.0,
        description="Lead time threshold for evacuation warnings (hours).",
    )
    response_rates: WarningResponseRatesConfig = Field(
        ...,
        description="Response rates for different warning types.",
    )
    warning_target: WarningTargetConfig = Field(
        ...,
        description="Warning targets configuration.",
    )


class HouseholdsConfig(BaseModel):
    """Configuration for households agent."""

    adapt: bool = Field(..., description="Whether households adapt.")
    adapt_to_actual_floods: bool = Field(
        ...,
        description="Whether households adapt their houses based on actual floods experienced.",
    )
    water_demand: WaterDemandConfig = Field(
        ..., description="Water demand configuration."
    )
    adjust_demand_factor: float = Field(
        ..., ge=0.0, description="Factor to adjust water demand."
    )
    warning_response: bool = Field(
        ..., description="Whether households respond to warnings."
    )
    warning_system: WarningSystemConfig = Field(
        ..., description="Warning system configuration."
    )
    expected_utility: ExpectedUtilityConfig = Field(
        ...,
        description="Expected utility configuration.",
    )


class MicrocreditConfig(BaseModel):
    """Configuration for microcredit."""

    ruleset: str = Field(..., description="Ruleset for microcredit.")
    interest_rate: float = Field(..., ge=0.0, description="Interest rate.")
    loan_duration: int = Field(..., ge=0, description="Loan duration (years).")


class SocialNetworkConfig(BaseModel):
    """Configuration for social network."""

    radius: float = Field(..., gt=0.0, description="Radius of social network (m).")
    size: int = Field(..., ge=0, description="Number of neighbors in social network.")


class InsurancePolicyConfig(BaseModel):
    """Configuration for an insurance policy type."""

    ruleset: str = Field(..., description="Ruleset for insurance.")
    loading_rate: float = Field(..., ge=0.0, description="Loading rate.")


class InsurersConfig(BaseModel):
    """Configuration for insurers agent."""

    duration: int = Field(..., ge=1, description="Insurance duration (years).")
    seut_factor: float = Field(..., description="SEUT factor.")
    government_premium_cap: bool = Field(
        ..., description="Whether government premium cap is applied."
    )
    traditional_insurance: InsurancePolicyConfig = Field(
        ...,
        description="Traditional insurance configuration.",
    )
    index_insurance: InsurancePolicyConfig = Field(
        ...,
        description="Index insurance configuration.",
    )
    pr_insurance: InsurancePolicyConfig = Field(
        ...,
        description="PR insurance configuration.",
    )


class CropSwitchingConfig(BaseModel):
    """Configuration for crop switching."""

    ruleset: str = Field(..., description="Ruleset for crop switching.")
    loan_duration: int = Field(..., ge=0, description="Loan duration (years).")
    switching_cap: float = Field(..., ge=0.0, description="Switching cap fraction.")


class AdaptationEfficiencyConfig(BaseModel):
    """Configuration for irrigation efficiency adaptation."""

    ruleset: str = Field(..., description="Ruleset for sprinkler adaptation.")
    dynamic_adaptation: bool = Field(..., description="Whether adaptation is dynamic.")
    static_price: bool = Field(..., description="Whether price is static.")
    loan_duration: int = Field(..., ge=0, description="Loan duration (years).")
    decision_horizon: int = Field(..., ge=0, description="Decision horizon (years).")
    return_fraction_surface: float = Field(
        ..., ge=0.0, le=1.0, description="Return fraction for surface irrigation."
    )
    return_fraction_sprinkler: float = Field(
        ..., ge=0.0, le=1.0, description="Return fraction for sprinkler irrigation."
    )
    return_fraction_drip: float = Field(
        ..., ge=0.0, le=1.0, description="Return fraction for drip irrigation."
    )
    irr_eff_surface: float = Field(
        ..., ge=0.0, le=1.0, description="Irrigation efficiency for surface irrigation."
    )
    irr_eff_sprinkler: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Irrigation efficiency for sprinkler irrigation.",
    )
    irr_eff_drip: float = Field(
        ..., ge=0.0, le=1.0, description="Irrigation efficiency for drip irrigation."
    )
    efficiency_division: list[float] = Field(
        ...,
        description="Ratio of different irrigation methods (surface, sprinkler).",
    )


class AdaptationWellConfig(BaseModel):
    """Configuration for well adaptation."""

    ruleset: str = Field(..., description="Ruleset for well adaptation.")
    seut_factor: float = Field(..., description="SEUT factor.")
    loan_duration: int = Field(..., ge=0, description="Loan duration (years).")
    lifespan: int = Field(..., ge=0, description="Lifespan (years).")
    decision_horizon: int = Field(..., ge=0, description="Decision horizon (years).")
    pump_hours: float = Field(..., ge=0.0, description="Pump hours.")
    specific_weight_water: float = Field(
        ..., gt=0.0, description="Specific weight of water (kg/m3*m/s2)."
    )
    max_initial_sat_thickness: float = Field(
        ..., gt=0.0, description="Maximum initial saturated thickness (m)."
    )
    pump_efficiency: float = Field(..., gt=0.0, description="Pump efficiency.")
    maintenance_factor: float = Field(..., ge=0.0, description="Maintenance factor.")


class DecisionsConfig(BaseModel):
    """Configuration for decisions."""

    decision_horizon: int = Field(..., ge=0, description="Decision horizon (years).")
    expenditure_cap: float = Field(..., ge=0.0, description="Expenditure cap.")


class WaterPriceConfig(BaseModel):
    """Configuration for water price."""

    static: bool = Field(..., description="Whether water price is static.")
    water_costs_m3_reservoir: float = Field(
        ..., ge=0.0, description="Water costs per m3 from reservoir."
    )
    water_costs_m3_groundwater: float = Field(
        ..., ge=0.0, description="Water costs per m3 from groundwater."
    )
    water_costs_m3_channel: float = Field(
        ..., ge=0.0, description="Water costs per m3 from channel."
    )


class DroughtEventPerceptionConfig(BaseModel):
    """Configuration for drought event perception."""

    drought_threshold: float = Field(..., description="Drought threshold.")


class DroughtRiskPerceptionConfig(BaseModel):
    """Configuration for drought risk perception."""

    base: float = Field(..., description="Base risk perception.")
    coef: float = Field(..., description="Coefficient for risk perception.")
    max: float = Field(..., description="Maximum risk perception.")
    min: float = Field(..., description="Minimum risk perception.")


class DroughtRiskCalculationsConfig(BaseModel):
    """Configuration for drought risk calculations."""

    event_perception: DroughtEventPerceptionConfig = Field(
        ...,
        description="Event perception configuration.",
    )
    risk_perception: DroughtRiskPerceptionConfig = Field(
        ...,
        description="Risk perception configuration.",
    )


class FarmersExpectedUtilityConfig(BaseModel):
    """Configuration for farmers expected utility."""

    crop_switching: CropSwitchingConfig = Field(
        ..., description="Crop switching configuration."
    )
    adaptation_efficiency: AdaptationEfficiencyConfig = Field(
        ...,
        description="Adaptation efficiency configuration.",
    )
    adaptation_well: AdaptationWellConfig = Field(
        ...,
        description="Well adaptation configuration.",
    )
    decisions: DecisionsConfig = Field(..., description="Decisions configuration.")
    water_price: WaterPriceConfig = Field(..., description="Water price configuration.")
    drought_risk_calculations: DroughtRiskCalculationsConfig = Field(
        ...,
        description="Drought risk calculations configuration.",
    )


class CropFarmersConfig(BaseModel):
    """Configuration for crop farmers agent."""

    ruleset: str = Field(..., description="Ruleset for crop farmers.")
    base_management_yield_ratio: float = Field(
        ..., gt=0.0, description="Base management yield ratio."
    )
    cultivation_cost_fraction: float = Field(
        ..., ge=0.0, description="Cultivation cost fraction."
    )
    microcredit: MicrocreditConfig = Field(
        ..., description="Microcredit configuration."
    )
    social_network: SocialNetworkConfig = Field(
        ..., description="Social network configuration."
    )
    expected_utility: FarmersExpectedUtilityConfig = Field(
        ...,
        description="Expected utility configuration.",
    )


FarmersConfig = CropFarmersConfig


class ReservoirOperatorsConfig(BaseModel):
    """Configuration for reservoir operators."""

    ruleset: str = Field(..., description="Ruleset for reservoir operators.")
    equal_abstraction: bool = Field(
        ..., description="Whether to use equal abstraction."
    )
    reservoir_M_factor: float = Field(..., ge=0.0, description="Reservoir M factor.")
    reservoir_release_factor: float = Field(
        ..., ge=0.0, description="Reservoir release factor."
    )


class AgentsConfig(BaseModel):
    """Configuration for agents."""

    government: GovernmentConfig = Field(
        ..., description="Government agent configuration."
    )
    market: MarketConfig = Field(..., description="Market agent configuration.")
    households: HouseholdsConfig = Field(
        ..., description="Households agent configuration."
    )
    insurers: InsurersConfig = Field(..., description="Insurers agent configuration.")
    crop_farmers: CropFarmersConfig = Field(
        ...,
        description="Crop farmers agent configuration.",
    )
    fix_activation_order: bool = Field(
        ..., description="Whether to fix activation order."
    )
    reservoir_operators: ReservoirOperatorsConfig = Field(
        ...,
        description="Reservoir operators configuration.",
    )


class ReportConfigInternal(BaseModel):
    """Internal report configuration."""

    compression_level: int = Field(
        ..., ge=1, le=22, description="Compression level (1-22)."
    )
    chunk_target_size_bytes: int = Field(
        ..., gt=0, description="Target chunk size in bytes."
    )


class ReportConfig(BaseModel):
    """Configuration for reporting."""

    model_config = ConfigDict(extra="allow")

    config: ReportConfigInternal = Field(
        ...,
        alias="_config",
        description="Internal report configuration.",
    )
    water_circle: bool = Field(
        ..., alias="_water_circle", description="Whether to report water circle."
    )
    water_balance: bool = Field(
        ..., alias="_water_balance", description="Whether to report water balance."
    )
    water_storage: bool = Field(
        ..., alias="_water_storage", description="Whether to report water storage."
    )
    energy_balance: bool = Field(
        ..., alias="_energy_balance", description="Whether to report energy balance."
    )
    discharge_stations: bool = Field(
        ...,
        alias="_discharge_stations",
        description="Whether to report discharge stations.",
    )
    bankfull_depths: bool = Field(
        ...,
        alias="_bankfull_depths",
        description="Whether to report bankfull depths each year.",
    )
    retention_basins: bool = Field(
        ...,
        alias="_retention_basins",
        description="Whether to report retention basins.",
    )
    meteorological_stations: bool = Field(
        ...,
        alias="_meteorological_stations",
        description="Whether to report meteorological stations.",
    )


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

    id: str | None = Field(
        ...,
        description="Run ID. If provided, sets spinup_name to 'spinup_<id>' and name to 'run_<id>'.",
    )
    start_time: date = Field(..., description="Model start time (YYYY-MM-DD).")
    end_time: date = Field(..., description="Model end time (YYYY-MM-DD).")
    spinup_time: date = Field(..., description="Start of the spinup time (YYYY-MM-DD).")
    input_folder: str = Field(..., description="Path to the input folder.")
    output_folder: str = Field(..., description="Path to the output folder.")
    spinup_name: str = Field(..., description="Name of the spinup simulation.")
    simulation_root: str = Field(..., description="Root directory for the simulation.")
    name: str = Field(..., description="Name of the simulation.")
    loglevel: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        ..., description="Logging level."
    )
    region: RegionConfig | None = Field(None, description="Region configuration.")

    forecasts: ForecastsConfig = Field(..., description="Forecasts configuration.")
    hazards: HazardsConfig = Field(..., description="Hazards configuration.")
    hydrology: HydrologyConfig = Field(..., description="Hydrology configuration.")
    evaluation: EvaluationConfig = Field(..., description="Evaluation configuration.")
    agents: AgentsConfig = Field(..., description="Agents configuration.")
    report: ReportConfig = Field(..., description="Reporting configuration.")
    calibration: CalibrationConfig | dict[str, Any] | None = Field(
        None, description="Calibration configuration."
    )

    @model_validator(mode="before")
    @classmethod
    def handle_run_id(cls, data: Any) -> Any:
        """Handle auto-deriving run names from id.

        Args:
            data: Raw configuration input data.

        Returns:
            Updated configuration dictionary with auto-derived names.
        """
        if not isinstance(data, dict):
            return data
        data = data.copy()
        run_id = data.get("id")
        if run_id is not None:
            run_id_str: str = str(run_id)
            if data.get("spinup_name") in (None, "spinup"):
                data["spinup_name"] = f"spinup_{run_id_str}"
            if data.get("name") in (None, "default"):
                data["name"] = f"run_{run_id_str}"

        return data
