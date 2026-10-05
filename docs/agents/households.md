# Households
## Flood Early Warnings
### Summary
In this module of the GEB model framework, it is possible to generate action-based flood early warnings to households. This FEWS require an ensemble of flood maps generated in the multiverse function of the model.py file. These maps are here processed into flood probability maps. Then, they are evaluated based on specific conditions to determine whether to issue a warning and which measures to recommend. Once a warning is generated, it can be communicated to households. Households will "decide" on whether to take the recommended actions based their initially assigned responsive_ratio. These measures are taken into account in the damage calculation through specific vulnerability curves. 

The workflow can be summarized as:
Ensemble flood maps → Flood probabilities → Warning triggers → Warning communication → Household actions

The main functions composing this system are:

- create_flood_probability_maps
- water_level_warning_strategy
- critical_infrastructure_warning_strategy
- warning_communication
- household_decision_making

The following sections provide a more detailed description of each step.

### Ensemble flood maps

The starting point of the warning system is a set of flood maps produced for the individual members of an ensemble forecast. These flood maps should have been created by the multiverse function in the main model.py file. Please note that, for this, the correct flags need to be set in the model.yml (i.e., forecasts = True). For a given forecast initialization, the module loads the flood map from each ensemble member and combines them into a single dataset. River channels are removed from the flood maps so that permanent water in the river itself is not interpreted as flooding. The ensemble is then used to calculate flood probability maps for different water-depth thresholds.

### Flood probability maps

In the create_flood_probability_maps function, the ensemble flood maps are translated into flood probability maps for different water-depth ranges. For example, if 6 out of 10 ensemble members predict that a location will exceed a given flood depth, the exceedance probability at that location is 60%. The warning system was designed to link forecasts directly to actionable household-level measures. Therefore, the water level ranges were defined based on measures to which they are applicable. Ranges for two different impact types are considered: (i) damages due to floods at a building-scale and (ii) damages due to flooding of critical infrastructure. 

The table below summarizes the water-level ranges used for each of the strategies, and the protective measures associated to each range. The ranges were derived from the official risk data and guidelines from The Dutch Government: Overstromingsrisicozonering (Flood Risk Zoning) and Risicokaart (https://www.risicokaart.nl/). The recommended measures were also derived from these sources and complemented by literature. All these ranges and measures can be customized.

For example, these are the water level ranges for specific strategies currently used in the model:

**Water level warning strategy (action-oriented)**

| Water level range (m) | Sandbags | Elevate possessions | Evacuation |
| --- | --- | --- | --- |
| 0.05 - 0.8 | X | - | - |
| 0.05 - 2 | - | X | - |
| >0.5 | - | - | X |

**Critical infrastructure warning strategy**

| Water level range (m) | Impact level | Sandbags | Elevate possessions | Evacuation | Exposed element |
| --- | --- | --- | --- | --- | --- |
| >0.3 | Power outages | - | - | X | Energy substations |
| >0.05 | Disruption to basic services | - | - | X | Vulnerable and emergency facilities |

### Water-level warning strategy

The water-level warning strategy determines whether households should receive a warning based on the forecasted flood probabilities. Warnings are generated at the postal code level. Two approaches are available:

- building-based warnings
- area-based warnings

For the **building-based warnings**, two impact-based thresholds are applied: one for the probability of occurrence and another for the % of buildings hit within a postal code area.
First, the flood probability map is intersected with residential buildings. A building is considered potentially flooded when the forecast probability at its location exceeds the specified probability threshold. The model then calculates the fraction of buildings affected within each postal code. A warning is issued to a postal code when this fraction exceeds the buildings-hit threshold. 

The default probability threshold in the model is set at 60%, following the KNMI protocol to move from a yellow (“be alert”) to an orange (“be prepared”) warning [egusphere-2025-828]. And a threshold of 10% of buildings hit was implemented. In other words, a postal code is warned when at least 10% of its buildings are located in areas where the probability of exceeding the relevant flood-depth threshold is at least 60%.

The **area-based warnings**, on the other hand, instead of counting buildings affected, evaluates only the fraction of pixels within each postal code that exceed the probability threshold. A warning is issued when the affected fraction exceeds the specified area-hit threshold.

This process is repeated for each flood probability map, corresponding to the water level ranges defined in the configuration. If the warning thresholds are exceeded for one or more water-level ranges, the system issues a warning with the protective measures associated with those ranges. The warning is then communicated to households within the affected postal code.

### Warning communication

Not every household in a warned postal code necessarily receives the warning. The communication_efficiency parameter controls the fraction of targeted households that are reached by the warning. For example, a communication efficiency of 0.92 means that approximately 92% of the households targeted by the warning are selected to receive it. Households can be selected either randomly or using weights based on socioeconomic characteristics such as income and education. A fixed random seed is used so that the selection can be reproduced between simulations. 

The system also considers the available lead time and the time required to implement each protective measure, ensuring that only measures that can still be implemented are recommended. Warnings can become more severe over time, but they cannot be revoked. For example, a household that previously received a warning recommending 'in-place' protective measures can later receive an evacuation warning if a newer forecast indicates more severe flooding. Conversely, if a newer forecast indicates a lower flood probability, the warning level is not reduced and previously issued recommendations are maintained. The warning is then disseminated and the module stores whether each household has received a warning, the warning level, the warning trigger, and the measures that were recommended. 

### Critical infrastructure warnings

Warnings can also be triggered by forecasted impacts on critical infrastructure. For each selected infrastructure type, the model evaluates the flood probability at the location of the asset. Examples of assets are:
- energy infrastructure
- hospitals or emergency facilities

If the probability of flooding exceeds the specified threshold set for the asset, the model identifies the postal codes that depend on the affected infrastructure. Households in these postal codes can then receive an evacuation warning. 

### Household decision making

After warnings have been issued, the household decision-making component determines whether households follow the recommendations. Only a fraction of households are assumed to respond to a warning. This is controlled by their responsive_ratio, which represents a responsive or non-responsive behavior. For households that respond, the recommended measures are translated into actions taken by the household. The resulting actions can subsequently influence flood impacts and damages in the model.

### Main parameters

The behaviour of the warning system can therefore be controlled through several important parameters:

| Parameter | Meaning |
| --- | --- |
| prob_threshold | Minimum forecast probability required for an impact to be considered |
| buildings_hit_threshold | Minimum fraction of affected buildings required to warn a postal code |
| area_hit_threshold | Minimum fraction of affected area required to warn a postal code |
| communication_efficiency | Fraction of targeted households that receive the warning
| responsive_ratio | Fraction of warned households that act on the warning |
| evacuation_lead_time_threshold | Maximum lead time at which evacuation is recommended |

### Supporting functions

In the assign_household_attributes function, a few attributes relative to households are initialized for the FEWS to work:

- warning_state
- warning_level
- warning_trigger
- response_probability
- evacuated 
- recommended_measures
- actions_taken

In the load_objects function, the objects needed for the system are:

- buildings
- postal_codes

The load_wlranges_and_measures function loads the dictionaries with the measures, their associated water level ranges and implementation times. 

## Flood damage model 

Flood damage calculations are performed in `geb/agents/modules/flood_risk.py` for return period-based flood events. For flood events occurring during the simulation period, damage is calculated when both `hazards.floods.simulate: true` and `hazards.damage.simulate: true`. We use exposure data from several sources. Buildings are taken from the OpenBuildingMap dataset[@oostwegel2025footprints]. Roads and railways are taken from OpenStreetMap. Agriculture and forest areas are taken from the ESA WorldCover database[@zanaga2022esa]. The exposure data is combined with vulnerability curves to calculate the damage per object. The vulnerability curve shows the relationship between inundation depth and the damage factor (i.e. the fraction of maximum damage for a certain water depth).

The flood damage model is set up in the build either for world regions, or using a context specific "geul" model, which was specifically created for the Geul catchment (shared by the Netherlands, Belgium and Germany). The damage model is selected with the `region` option of `setup_flood_damage_model` in the build configuration (default: `global`). All damage calculations are performed in `geb/agents/modules/flood_risk.py`.

| `region` option | Damage model | Description |
| --- | --- | --- |
| `geul` | [Geul flood damage model](#geul-flood-damage-model) | Context-specific curves and maximum damages for the Netherlands, Belgium and Germany. |
| `global` | [Global flood damage model](#global-flood-damage-model) | Averaged global vulnerability curves (Huizinga et al., 2017). |
| `europe` | [Global flood damage model](#global-flood-damage-model) | Continental vulnerability curves for Europe. |
| `north america` | [Global flood damage model](#global-flood-damage-model) | Continental vulnerability curves for North America. |
| `central&south america` | [Global flood damage model](#global-flood-damage-model) | Continental vulnerability curves for Central & South America. |
| `asia` | [Global flood damage model](#global-flood-damage-model) | Continental vulnerability curves for Asia. |
| `africa` | [Global flood damage model](#global-flood-damage-model) | Continental vulnerability curves for Africa. |
| `oceania` | [Global flood damage model](#global-flood-damage-model) | Continental vulnerability curves for Oceania. |

### Global flood damage model 
The global damage model makes use of continental vulnerability curves obtained from Huizinga et al. (2017)[@huizinga2017global]. This dataset contains vulnerability curves describing the relationship between inundation levels and damages for residential, commercial, industrial, transport, infrastructure and agricultural land uses. The curves are constructed for Europe, North America, Central & South America, Asia, Africa, Oceania, and 'global' (averaged global data). Currently, only the vulnerability curves for residential land use are supported in the flood risk calculations when selecting the global damage model. 

The maximum damages for buildings are taken from the Global Exposure Model[@yepes2023global]. The Global Exposure Model is a mosaic of local and regional models with information regarding the residential, commercial, and industrial building stock at the smallest available administrative division of each country and includes details about the number of buildings, number of occupants, vulnerability characteristics, average built-up area, and average replacement cost. Damages are reported for every exposure category individually and as a combined total damage value.

### Geul flood damage model 
The Geul damage model is based on data specific to the Dutch, Belgian and German context. For buildings, we use the curves derived by Endendijk et al.[@endendijk2023flood]. For roads, we use the curves derived by Van Ginkel et al.[@van2021flood]. For railways, we use the curves developed by Kellermann et al.[@kellermann2015estimating]. For nature and agriculture, we use the curves developed by De Moel et al.[@de2014evaluating]. The figure below shows all curves and maximum damages used in the Geul damage model. Damages are reported for every exposure category individually and as a combined total damage value.
<img width="1280" height="720" alt="Vulnerability Curves and their corresponding maximum damages" src="https://github.com/user-attachments/assets/3a4dbc03-3a45-49c6-b1f6-028eac385d7d" />

## Windstorm Adaptation
The Windstorm adaptation module is focused on designing a decision-making framework for households to choose between adapting to or not adapting to windstorm risk. As a first step, the risk maps are generated externally based on ERA5 data. After generating the different return-period maps, each household's associated damage is calculated. The data are then used as inputs to the Expected Utility, which is the basis of the households' decision-making. The module is set in three scripts: wind_risk.py, households.py, and decision_module.py

### Windstorm damage model
The windstorm damage model is performed in the wind_risk script. In the script, a return period is randomly selected based on its return-period probability. Using the pre-generated wind risk map, the damage is automatically calculated. At the moment, the windstorm module only works for households obtained from the OpenBuildingMap dataset. The vulnerability curve shows the relationship between wind speed and the damage factor (i.e., the fraction of maximum damage for a certain wind speed). The curves used in this module are derived by Riedel et al. [@riedel2024calibrating]. The maximum damages for buildings are consistent through the flood and windstorm modules.

In the wind_risk.py script, the main functions composing this system are:
- load_wind_maps: loads the pre-existing windstorm return-period maps
- load_windstorm_damage_curves: loads the windstorm vulnerability curves
- calculate_building_wind_damages: calculates the damage by using the vulnerability curves and the associated wind speed at every household location
- calculate_ead: converts return-period damage to annualized risk

### Decision-making process
The windstorm decision-making framework evaluates the household-level utility of undertaking windstorm adaptation compared with remaining unadapted. This is done by calculating the expected utility of implementing window shutters by accounting for expected damages after adaptation, perceived windstorm probabilities, household wealth and income, adaptation costs, remaining loan payments, the decision horizon, discounting, and risk preferences. This is then compared to the utility of doing nothing.

In the decision_module.py script, the main functions for decision-making associated with windstorm adaptation are:
- IterateThroughEvents: Links the Expected Utility functions to the windstorm events
- calcEU_do_nothing_w: calculates the utility of not adapting
- calcEU_shutters_windstorm: calculates the utility of adapting

### Integration in the class Household
Lastly, the household script calls for the WindRiskModule to access wind risk perception data. As part of the household decisions between strategies, the wind module calculates potential building damages with and without shutters. With the resulting data, the households call the decision-making module to calculate the necessary expected utilities. Lastly, households may then adopt shutters based on the expected utilities and the set budget constraints. The resulting adaptation status is used to update the expected annual damage.

In the household.py script, the main functions associated with windstorm adaptation are:
- assign_household_attributes: Initializes household adaptation and risk-related state
- decide_household_strategy: Integrates the decision module with the household class
- update_windstorm_risk_perceptions: Updates households' years since the last simulated windstorm and adjusts risk perceptions


## Code

::: geb.agents.households
