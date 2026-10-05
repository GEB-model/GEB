# Households
## Flood Early Warnings
### Summary
In this module of the GEB model framework, it is possible to generate action-based flood early warnings to households. This FEWS require an ensemble of flood maps generated in the multiverse function of the model.py file. These maps are here processed into flood probability maps. Then, they are evaluated based on specific conditions to determine whether to issue a warning and which measures to recommend. Once a warning is generated, it can be communicated to households. Households will "decide" on whether to take the recommended actions based their initially assigned responsive_ratio. These measures are taken into account in the damage calculation through specific vulnerability curves. 

In the households.py file, the main functions composing this system are:
- create_flood_probability_maps
- water_level_warning_strategy
- critical_infrastructure_warning_strategy
- warning_communication
- household_decision_making

The following sections provide a more detailed description of each step.

### Flood probability maps

The flood probability maps are created in the create_flood_probability_maps function. The warning system was designed to link forecasts directly to actionable household-level measures. Therefore, to generate the flood probability maps, water level ranges were defined based on measures to which they are applicable. Ranges for two different impact types are considered: (i) damages due to floods at a building-scale and (ii) damages due to flooding of critical infrastructure. These are represented in the model as different strategies that can be used in combination or isolated. Table 1 summarizes the damaging water-level ranges, their associated impacts, and the ranges for which each type of measure is suitable. The ranges and associated impacts were derived from the official risk data and guidelines from The Dutch Government: Overstromingsrisicozonering (Flood Risk Zoning) and Risicokaart (https://www.risicokaart.nl/). The recommended measures were also derived from these sources and complemented by literature. All these ranges and measures can be customized.

For example, these are the water level ranges for specific strategies currently used in the model:

**Water level warning strategy**
| Water level range (m) | Impact level | Sandbags | Elevate possessions | Evacuation | Exposed element |
| --- | --- | --- | --- | --- | --- |
| 0.05 - 0.2 | People can get away on foot, minor damage | X | X | - | Buildings |
| 0.2 - 0.5 | Cars can still drive, increasing damage | X | X | - |
| 0.5 - 0.8 | Military vehicles can still drive, increasing damage| X | X | X |
| 0.8 - 2 | People can say on the 1st floor, maximum damage | - | X | X |
| >2 | Not safe for humans, maximum damage | - | - | X |

**Critical infrastructure strategy**
| Water level range (m) | Impact level | Sandbags | Elevate possessions | Evacuation | Exposed element |
| --- | --- | --- | --- | --- | --- |
| >0.3 | Power outages | - | - | X | Energy substations |
| >0.05 | Disruption to basic services | - | - | X | Vulnerable and emergency facilities |

### Warning generation and household decision-making

Warnings are generated at the postal code level. To determine when a warning should be issued, two impact-based thresholds are applied: one for the probability of occurrence and another for the % of buildings hit within a postal code area. This is applied in the water_level_warning_strategy function. The default probability threshold in the model is set at 60%, following the KNMI protocol to move from a yellow (“be alert”) to an orange (“be prepared”) warning [egusphere-2025-828]. To avoid false alarms caused by isolated pixels, a 10% critical hit threshold was implemented, meaning that a warning is generated if at least 10% of the buildings in the postal code are intersects an area with a flood probability higher than 60%. Once the thresholds are exceeded, the system issues a warning specifying appropriate measures, based on the available lead time, and the time needed for their implementation. The warning is then disseminated, accounting for the efficiency of warning communication. 

### Rule-based decision making

Finally in the decision module, once a household receives a warning, it decides whether to implement the recommended forecast-based measures depending on its responsive or non-responsive state. 

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

Flood damage calculations are performed in `geb/agents/modules/flood_risk.py` for return period-based flood events, and for historical flood events occuring during the simulation period if `hazards.floods.simulate:True`. We use exposure data from several sources. Buildings are taken from the OpenBuildingMap dataset[@oostwegel2025footprints]. Roads and railways are taken from OpenStreetMap [not yet included in bib?]. Agriculture and forest areas are taken from the ESA WorldCover database[@zanaga2022esa]. The exposure data is combined with vulnerability curves to calculate the damage per object. The vulnerability curve shows the relationship between inundation depth and the damage factor (i.e. the fraction of maximum damage for a certain water depth).

The flood damage model is set up in the build either for world regions, or using a context specific "geul" model, which was specifically created for the Geul catchment (shared by the Netherlands, Belgium and Germany). The damage model is selected with the `region` option of `setup_flood_damage_model` in the build configuration (default: `global`). All damage calculations are performed in `geb/agents/modules/flood_risk.py`.

| `region` option | Damage model | Description |
| --- | --- | --- |
| `geul` (default) | [Geul flood damage model](#geul-flood-damage-model) | Context-specific curves and maximum damages for the Netherlands, Belgium and Germany. |
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
## Code

::: geb.agents.households
