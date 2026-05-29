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

The adaptation pathway is an optional annual decision loop in which the government monitors three indicators and triggers adaptation measures when any threshold is crossed. It runs every January 1st during the simulation.

**How it works**

On each January 1st the government calculates:

- **EAD** — Expected Annual Damage from flooding (€), integrated over the damage–exceedance probability curve.
- **Equity indicator** — A measure of how evenly flood exposure is distributed across households (0–1, where 1 = perfect equality).
- **Ecosystem indicator** — A measure of ecosystem health (0–1, where 1 = healthy).

If any indicator exceeds its threshold, the corresponding adaptation measure is applied:

| Indicator crossed | Measure applied |
|---|---|
| EAD > `EAD_threshold` | Floodproof buildings of a fraction of at-risk households |
| Equity < `equity_indicator_threshold` | *(planned — subsidies)* |
| Ecosystem < `ecosystem_indicator_threshold` | Plant one increment of forest (`prepare_modified_soil_maps_for_forest`) |

**Configuration**

```yaml
agent_settings:
  government:
    adaptation:
      enabled: true
      EAD_threshold: 1000000          # in euros
      equity_indicator_threshold: 0.5  # 0–1
      ecosystem_indicator_threshold: 0.5  # 0–1
      adaptation_fraction: 0.1        # fraction of at-risk households to floodproof per year
```

**Configuration reference**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `adaptation.enabled` | `bool` | `false` | Enable the adaptation pathway. |
| `adaptation.EAD_threshold` | `float` | `1000000` | EAD (€) above which flood adaptation is triggered. |
| `adaptation.equity_indicator_threshold` | `float` | `0.5` | Equity value below which equity adaptation is triggered. |
| `adaptation.ecosystem_indicator_threshold` | `float` | `0.5` | Ecosystem health value below which reforestation is triggered. |
| `adaptation.adaptation_fraction` | `float` | `0.1` | Fraction of at-risk households to floodproof per trigger (0–1). |

---

### Policy

*Coming soon.*

---

## Code

::: geb.agents.government
