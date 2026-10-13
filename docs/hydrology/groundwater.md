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

## Karst

One store holds the mapped karst share of soil drainage, plus captured runoff
and interflow. It releases water to groundwater over time. Sealed areas and
open water are excluded. This delays water; it does not remove it.

```yaml
hydrology:
  karst:
    capture_fraction: 1.0
    release_time_days: 10.0
```

Karst is included in every hydrology run. Cells with zero mapped coverage keep
their existing flow paths. There is no enable/disable setting. `capture_fraction` applies to runoff and interflow; soil
drainage uses mapped coverage. The release time is in days; after that many dry
days, 37% of stored water remains. Ten days is a starting value.

The default build includes `setup_karst`; custom builds must include it too.
When upgrading, remove `hydrology.karst.enabled` from `model.yml`. Build a
missing map with `geb update -b build.yml::setup_karst`, then rerun spinup and
simulation to initialize karst storage.
Rerun spinup after changing karst settings or updating its routing. Existing
maps can be reused unless they contain only zeros. Storage is included in
checkpoints and water balances.

## Code

::: geb.hydrology.groundwater
