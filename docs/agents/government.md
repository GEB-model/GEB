# Government

The government agent represents public institutions. Rather than making decisions at the individual level, the government acts at the system level: it sets the rules, constraints, and landscape configurations that others must work within.

In GEB, currently the government module is optional. It is activated by including a `government` section under `agent_settings` in the model configuration. When absent, no government-level interventions are applied and the model runs without top-down policy or land-use changes.

The government currently acts in two broad areas: **adaptation** — modifying the physical landscape to improve resilience or ecosystem function — and **policy** — regulating how other agents access and use resources.

---

## Adaptation

Adaptation measures are interventions the government applies to the physical environment. These are typically applied once at the start of the simulation and shape the conditions for the entire model run.

### Land-Use: Reforestation

The government can trigger a reforestation scenario in which agriculturally used land is converted to forest where ecologically suitable. This is intended to support studies on nature-based adaptation, ecosystem restoration, and land-use change impacts on hydrology.

When enabled, reforestation is applied at **timestep 0**, before the first simulation step. The process unfolds in three stages:

**1. Identifying suitable areas**

A pre-computed `forest_restoration_potential_ratio` grid (values 0–1) is loaded. This dataset represents the ecological suitability of each grid cell for forest restoration, derived from Bastin et al. (2019)[@bastin2019global] and made available as a supplementary dataset[@bastin2019dataset]. Grid cells with a ratio at or above a configurable threshold are marked as suitable for conversion. These grid-level suitability values are then mapped down to the HRU (Hydrological Response Unit) scale.

The threshold defaults to `0.5` and can be adjusted in the configuration. Both of the following are valid:

```yaml
# Simple: enable reforestation with the default threshold (0.5)
agent_settings:
  government:
    plant_forest: true
```

```yaml
# Advanced: enable reforestation with a custom threshold
agent_settings:
  government:
    plant_forest:
      forest_restoration_potential_threshold: 0.8  # range 0–1
```

```yaml
# Incremental: plant 10% of suitable area per year, ranked by potential (highest first)
agent_settings:
  government:
    plant_forest:
      forest_restoration_potential_threshold: 0.5
      increment_fraction: 0.1   # fraction of suitable area to plant per call (0–1)
```

In incremental mode all suitable HRUs (those meeting the threshold) are sorted by their `forest_restoration_potential_ratio` value in descending order. On each call the model skips HRUs already classified as FOREST (planted in previous years or originally forest) and plants the next `increment_fraction × remaining` HRUs. No manual step counter is needed — the function advances automatically each time it is called. When all suitable HRUs are already forest, a warning is logged and no action is taken.

When used together with the adaptation pathway (see [Adaptation Pathway](#adaptation-pathway) below), `prepare_modified_soil_maps_for_forest` is called every January 1st when the ecosystem indicator threshold is crossed, so the forest grows by one increment per year automatically.

**2. Updating soil properties**

For all HRUs marked as suitable, the model performs an in-memory update, replacing their soil properties with the mean values of existing forest HRUs in the domain. This ensures the converted areas behave hydrologically like forest from the start of the simulation. The following soil properties are updated:

- Saturated water content
- Field capacity
- Wilting point
- Residual water content
- Saturated hydraulic conductivity
- Bubbling pressure
- Pore size distribution index (lambda)
- Solid heat capacity

**3. Removing displaced farmers**

Crop farmers whose fields fall within the reforested HRUs are removed from the simulation. Their land use type is set to `FOREST`. The number of removed farmers is logged to the console.

**Output**

A diagnostic figure (`reforestation_scenario.png`) is saved to `output/forest_planting/`. It shows four panels: current land cover, future land cover after reforestation, the suitability map used, and the areas that were converted.

**Configuration reference**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plant_forest` | `bool` or `dict` | `false` | Enable reforestation. Set to `true` or a config dict. |
| `plant_forest.forest_restoration_potential_threshold` | `float` | `0.5` | Minimum suitability ratio for a cell to be converted (0–1). |
| `plant_forest.increment_fraction` | `float` | `null` (disabled) | Fraction of suitable area to plant per call (e.g. `0.1` = 10%). When `null`, all suitable area is planted at once. Auto-advances on each call by skipping already-forested HRUs. |

---

## Adaptation Pathway

When adaptation is enabled and set to the method pathway in the configuration, the government agent can develop an adaptation pathway, a sequence of adaptation measures over time, through an annual decision making loop. This decision making is based on feedbacks between decisions made by the government and other actors as well as the physical environment within the model, while targeting multiple objectives beyond flood risk reduction simultaneously. It runs every January 1st during the simulation.

To develop the adaptation pathway, the government agent needs to be able to decide between the different adaptation measures that are available. Currently, the government agent can decide to implement floodproofing (less damage to houses), reforestation (see above) or risk communication (increase risk perception of households to nudge them to self-adapting). The decision making for this is based on which adaptation measure would generate the most improvement in terms of a multi-objective score. The objectives of the government go beyond flood risk and also include ecosystem health and equity. This is conceptually visible in the figure below. 

<img width="550" height="300" alt="image" src="https://github.com/user-attachments/assets/d8bcbe33-45ca-464a-9435-ffba8ea1274d" />

### How a decision is made

At every decision point, the government agent enters a hypothetical world, which is a copy of the simulated system at that time step. This hypothetical world allows the agent to determine which adaptation measure is best suited at that time to meet its multiple objectives. Each year, the following steps are taken:

1. ***Implement each measure separately:*** In this hypothetical world, each adaptation measure is implemented in separate instances.
2. ***Apply the budget:*** To constrain the adaptation, the government agent has a yearly budget for adaptation. The amount of each measure that is hypothetically implemented is the maximum amount affordable with this available annual budget. If the budget does not allow for catchment wide implementation, adaptation measures are implemented in increments.The full budget is spend on one adaptation measure yearly, in the following the full budget is available again. 
3. ***Run three years ahead:*** After the implementation, the simulated hypothetical system is propagated forward for three years. To prevent the agent from having perfect foresight, the system is propagated forward using the climatic forcing from the last three years. This allows the government to determine the effect of the affordable adaptation measure over the next three years, including the feedbacks of this measure on other agents and, for example, the ecosystem within the simulated system, without knowing future conditions.
4. ***Evaluate performance:*** The effectiveness of the adaptation measures is determined by calculating their performance on the objectives. These indicator values are combined in an Integrated Performance Value (IPV).
5. ***Select the best measure:*** When the improvement in IPV has been calculated for each candidate adaptation separately, the government exits this hypothetical world. The adaptation measure that generated the largest improvement in the IPV compared to the IPV in the current timestep (ΔIPV) is then implemented in the 'real world'.
6. ***Continue the simulation:*** The implemented measure persists through the following time steps. The model continues until the following first of January, when the hypothetical world is initiated again.

As this decision-making is done annually, an adaptation pathway emerges over time from within the model that is driven by the government agent that has multiple objectives. 

**Objectives and indicators**

The government considers 3 objectives: flood risk, ecosystem health and equity. The indicators for them are as follows: 
- ***Flood risk:*** The expected annual damage (EAD, €), calculated from the integration over the damage–exceedance probability curve.
- ***Ecosystem health:*** Represented through the capacity of certain land uses to supply ecosystem services. The indicator calculates an area-weighted average score across the catchment. Grid cells with a higher capacity to supply ecosystem services, have a higher score. The scores are based on expert judgement derived from case studies in Europe [@burkhard2012].
- ***Equity:*** the indicator for equity is two-fold, made up by 1) how much of the total EAD, relative to household income, is borne by low-income households, compared to all households in the catchment and 2) the average fraction of forest low-income households have within their local neighbourhood. They are normalised and then summed and divided by 2.

**Normalisation and IPV**

To be able to combine and compare these indicators they are normalised on a 0-1 scale. This normalisation uses bounds representing the theoretically worst- and best-case scenarios. These bounds are derived from the minimum and maximum value for the indicators found within a set of reference runs. These reference runs can, for example, include one run with no adaptation under a high-end climate scenario and one run under baseline climate in which all adaptation measures are implemented with an unconstrained budget. However, these reference runs can be tailored to the experiment that is being simulated. For example, if there are clear quantitative goals available for these objectives in the studied catchment these can also be used in the normalisation. This allows for measuring how effective the simulated adaptation pathway is on achieving these goals.

By summing the normalised values of these indicators, combined with weights indicating government priorities, the IPV is calculated. This IPV can then drive the decision making of the government but can also help in assessing the effectiveness of pathways in contributing to the objectives. In default setting the government weights are equal. To experiment how different priorities can change the adaptation pathway decision making these values can be changed. 

**Applying to your catchment**

This method for generating adaptation pathways can be tailored to the studied catchment. To do this, a few things need to be specified in the configuration: the annual available budget for climate adaptation, the costs for the adaptation measure and the normalisation bounds. For the normalisation run, the reference runs need to be defined and redone. Furthermore, the adaptation measures and objectives that are considered can also be adjusted. To add or change objectives, quantitative indicators will need to be developed. 

**Configuration**

```yaml
  government:
    adaptation:
      enabled: true
      mode: pathway
    priority_weights:
      risk_reduction: 0.3333
      equity: 0.3333
      ecosystem_health: 0.3333
    adaptation_costs:
      budget: 3650000 #for the Geul catchment
      floodproofing_cost_per_household: 27384 #for the Geul catchment
      reforestation_cost_per_m2: 1.66 #for the Geul catchment
      communication_cost_per_household: 35 #for the Geul catchment
    normalisation_values:
      ead_best_value: 112729510 #for the Geul catchment
      ead_worst_value: 157183895 #for the Geul catchment
      flooddamageburden_best_value: 0.763942849 #for the Geul catchment
      flooddamageburden_worst_value: 0.747929522 #for the Geul catchment
      ecosystemhealth_best_value: 0.726538458 #for the Geul catchment
      ecosystemhealth_worst_value: 0.402870868 #for the Geul catchment
      forestaccesslowincome_best_value: 0.974048587 #for the Geul catchment
      forestaccesslowincome_worst_value: 0.310953705 #for the Geul catchment
```

**Configuration reference**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `adaptation.enabled` | `bool` | `false` | Enable government adaptation. |
| `adaptation.mode` | `str` | `pathway` | Decision mode. `pathway` runs the annual decision loop. |
| `priority_weights.risk_reduction` | `float` | `0.3333` | Weight of the flood risk objective in the IPV. |
| `priority_weights.equity` | `float` | `0.3333` | Weight of the equity objective in the IPV. |
| `priority_weights.ecosystem_health` | `float` | `0.3333` | Weight of the ecosystem health objective in the IPV. |
| `adaptation_costs.budget` | `float` | `3650000` | Annual adaptation budget (€). |
| `adaptation_costs.floodproofing_cost_per_household` | `float` | `27384` | Cost of floodproofing one household (€). |
| `adaptation_costs.reforestation_cost_per_m2` | `float` | `1.66` | Cost of reforestation (€/m²). |
| `adaptation_costs.communication_cost_per_household` | `float` | `35` | Cost of risk communication per household (€). |
| `normalisation_values.ead_best_value` | `float` | `112729510` | EAD (€) mapped to the best normalised score. |
| `normalisation_values.ead_worst_value` | `float` | `157183895` | EAD (€) mapped to the worst normalised score. |
| `normalisation_values.flooddamageburden_best_value` | `float` | `0.763942849` | Flood damage burden mapped to the best normalised score. |
| `normalisation_values.flooddamageburden_worst_value` | `float` | `0.747929522` | Flood damage burden mapped to the worst normalised score. |
| `normalisation_values.ecosystemhealth_best_value` | `float` | `0.726538458` | Ecosystem health mapped to the best normalised score. |
| `normalisation_values.ecosystemhealth_worst_value` | `float` | `0.402870868` | Ecosystem health mapped to the worst normalised score. |
| `normalisation_values.forestaccesslowincome_best_value` | `float` | `0.974048587` | Forest access for low-income households mapped to the best normalised score. |
| `normalisation_values.forestaccesslowincome_worst_value` | `float` | `0.310953705` | Forest access for low-income households mapped to the worst normalised score. |

---

### Policy

*Coming soon.*

---

## Flood protection standards

The government can raise the flood protection standard of individual subbasins. The initial standard of each subbasin (a return period in years, e.g. 100 for protection against the 1-in-100-year flood) is derived from the FLOPROS database[@scussolini2016flopros] during the model build, or set manually through `hazards.floods.flood_protection_standard` (`mode: manual`). Households in a protected subbasin are assumed to suffer no damage from floods with a return period lower than the standard.

This mechanism is part of the adaptation pathway and is activated with `adaptation.mode: cba` (cost-benefit analysis) together with `adaptation.enabled: true`. In this mode the threshold-based indicators described above are not used. On each January 1st, for every subbasin with a defined standard, the government evaluates raising the standard by one step to the next return period in the list of simulated return periods:

1. **Benefit** — The reduction in expected annual damage (EAD) for the households in the subbasin when moving from the current to the next standard. Damages for return periods below the new standard are set to zero before integrating over the exceedance probability curve. The reduction is multiplied by an indirect damage factor of 1.6 to account for damages not captured by the direct damage model.
2. **Investment cost** — The dike height is sampled along the river segments of the subbasin from the flood maps of the current and the next return period. The difference in height is multiplied by the segment length (approximately 100 m) and the unit elevation cost, and doubled to account for dikes on both river banks.
3. **Maintenance cost** — The yearly maintenance cost per meter of dike times the length of all dike segments, also doubled for both banks.
4. **Decision** — The standard is raised when the discounted benefit exceeds the investment cost plus the discounted maintenance cost. Benefits and maintenance costs are accumulated over 35 years using a discount rate of 10%. Subbasins where no dike needs to be raised, or that already have the highest return period as standard, are skipped.

In the final year of the simulation the resulting standards are exported to `flood_protection_standards.parquet` in the output folder, with one row per subbasin (`COMID`) and its `flood_protection_standard` (years).

```yaml
agent_settings:
  government:
    adaptation:
      enabled: true
      mode: cba
      dike_elevation_cost_per_meter_usd: 6800   # USD per meter of height per meter of dike length
      dike_maintenance_cost_per_year_usd: 80    # USD per year per meter of dike length
```

**Configuration reference**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `adaptation.enabled` | `bool` | `false` | Enable the adaptation pathway. |
| `adaptation.mode` | `str` | `"threshold"` | Set to `"cba"` to raise flood protection standards based on a cost-benefit analysis. |
| `adaptation.dike_elevation_cost_per_meter_usd` | `float` | `6800` | Cost of raising a dike by 1 m over 1 m of length (USD). |
| `adaptation.dike_maintenance_cost_per_year_usd` | `float` | `80` | Yearly maintenance cost of 1 m of dike (USD/year). |

## Code

::: geb.agents.government
