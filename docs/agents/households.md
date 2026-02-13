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

## Code

::: geb.agents.households
