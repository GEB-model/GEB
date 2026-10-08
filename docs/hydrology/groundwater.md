# Groundwater

## Introduction

The groundwater module in GEB simulates the storage and movement of water in the subsurface, the interaction between the deep soil layers, the river network (baseflow), and human water abstractions. It is uses [MODFLOW 6](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model), and is connected to the other hydrological model using [xmipy](https://github.com/Deltares/xmipy) allowing in-memory and efficient exchange of data between the models.

The groundwater domain is discretized into one or more layers. The horizontal resolution matches the GEB grid. Vertical discretization is defined by `layer_boundary_elevation` and `elevation` (topography).

The module is implemented in the `GroundWater` class, which acts as a wrapper around a `ModFlowSimulation` instance. This creates a tight coupling between GEB and MODFLOW, allowing them to exchange fluxes at every timestep.

The module manages the following key state variables:

*   **Heads**: The hydraulic head in each cell and layer.
*   **Storage**: The volume of water stored in the aquifer.

In each timestep, the groundwater module interacts with the rest of the model through:

*   **Recharge**: Water entering the groundwater table from the lowest soil layer (provided by the [Land Surface](landsurface.md) module).
*   **Abstraction**: Water pumped from the aquifer for irrigation, industry, or domestic use (provided by the [Water Demand](water_demand.md) module).
*   **Baseflow**: Water flowing from the aquifer into the river channels, maintaining flow during dry periods (passed to the [Runoff Concentration](runoff_concentration.md) and [Routing](routing.md) modules).
*   **Capillary Rise**: Upward movement of water from the water table to the soil zone (passed back to the [Land Surface](landsurface.md) module for the *next* timestep).

## Model step

The groundwater simulation proceeds in the following steps during each model timestep:

1.  **Input Processing**:
    *   **Recharge**: The total depth of water percolating from the soil columns ($m$) is received from the land surface module, which will be added to the top layer of the aquifer..
    *   **Abstraction**: Total groundwater demand is received from the water demand module. This abstraction is distributed across the aquifer layers based on water availability and well depth.

2.  **MODFLOW Update**:
    *   The fluxes are applied to the MODFLOW model via the Basic Model Interface (BMI).
    *   MODFLOW executes a single time step, solving the groundwater flow equation to calculate new heads and flows.

3.  **Output Calculation**:
    *   **Drainage/Baseflow**: The exchange between the aquifer and the surface (specifically rivers/drains) is retrieved. In GEB, this drainage is treated as baseflow.
    *   **Capillary Rise**: A portion of the drainage flux is partitioned into capillary rise (water moving back up to the unsaturated zone), though currently, the implementation assigns most drainage to river baseflow.
    *   **State Update**: The updated hydraulic heads are synchronized back to the GEB grid state.

## Simple karst recharge

GEB can send part of the surface runoff and interflow (water flowing sideways
through the soil) to groundwater in karst areas. This represents water entering
cracks and sinkholes. MODFLOW then calculates groundwater storage and flow.

The build method `setup_karst` downloads the [World Karst Aquifer Map (WOKAM)](https://download.bgr.de/bgr/grundwasser/whymap/shp/WHYMAP_WOKAM_v1.zip)
and saves `groundwater/karst_fraction` (0–1). It includes rocks that cover only
part of a grid cell. WOKAM shows surface rocks that can form karst. It does not
give an exact karst fraction. We use 0.825 for continuous carbonate or evaporite
rocks, 0.4 for discontinuous rocks, and 0.65 for mixed rocks. These are estimates
from the middle of the coverage ranges. See
[Goldscheider et al. (2020)](https://doi.org/10.1007/s10040-020-02139-5).
Data source: WHYMAP WOKAM, BGR Berlin, IAH Reading, KIT Karlsruhe, UNESCO Paris, 2017.

Enable it in `model.yml`:

```yaml
hydrology:
  karst:
    enabled: true
    capture_fraction: 1.0
```

Karst recharge is off by default. `capture_fraction` must be between 0 and 1.
The default is 1.0, based on the full-capture assumption used by
[Wan, Döll and Müller Schmied (2024)](https://doi.org/10.1029/2023WR036182).
They found that WaterGAP land runoff, excluding urban runoff and soil overflow,
best represented groundwater recharge on the karst area, using estimates from
64 karst grid cells. This supports a simple model assumption, not a measured
capture fraction for the Geul. GEB applies it to surface runoff and interflow;
these are not the same components as WaterGAP's runoff. In particular, GEB's
surface runoff includes soil overflow, which this simple scheme also captures.
The earlier value of 0.5 was a trial value with no literature basis.

With a karst fraction of 0.6 and capture set to 1.0, 60% of the surface runoff
and interflow goes to groundwater. Each HRU uses the karst fraction of its grid
cell. Sealed areas and open water are excluded. The same amount is removed from runoff and added to
recharge, so no water is lost. This happens after the soil calculations.
Flood runoff files and erosion calculations use the remaining surface runoff.
Land-surface reports show runoff before karst capture. To report the added
recharge, use `.karst_recharge_m` in the `hydrology` module (HRU, meters per day).

For existing inputs, first run `uv run geb update-version -b build.yml` to update
the input version. It prints the karst update instructions once. Then build the
map before running with karst enabled:

```bash
uv run geb update -b build.yml::setup_karst
uv run geb spinup
uv run geb run
```

`setup_karst` is included in the default build. Custom build files must include
it after `setup_region`. Rerun spinup when turning karst on or changing the
capture fraction, so groundwater levels adjust to the new recharge.
For the Geul at Meerssen, compare runs with karst off and on. Start with 1.0
and check peak flows, how quickly flows fall after rain, low flows and annual
discharge. The effect on peaks depends on how quickly MODFLOW returns the water
to the river. This simple approach does not model individual springs or
underground river connections. It may not reduce the annual discharge.

## Code

::: geb.hydrology.groundwater
