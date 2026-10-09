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

Karst redirects part of the surface runoff and interflow directly to groundwater
recharge in the same daily timestep. Sealed areas and open water are excluded.
Karst is off by default, and no separate karst storage bucket is used.

`setup_karst` uses [WOKAM](https://download.bgr.de/bgr/grundwasser/whymap/shp/WHYMAP_WOKAM_v1.zip)
to estimate coverage per cell: 0.825 for continuous rocks, 0.4 for discontinuous
rocks, and 0.65 for mixed rocks. These are estimates from the mapped ranges;
see [Goldscheider et al. (2020)](https://doi.org/10.1007/s10040-020-02139-5).

Enable it in `model.yml`:

```yaml
hydrology:
  karst:
    enabled: true
    capture_fraction: 1.0
```

Capture equals karst coverage times `capture_fraction` (0–1). The default of 1.0
follows [Wan et al. (2024)](https://doi.org/10.1029/2023WR036182), but GEB also
captures soil overflow, which that study excludes.

Build the map and rerun spinup:

```bash
uv run geb update-version -b build.yml
uv run geb update -b build.yml::setup_karst
uv run geb spinup
uv run geb run
```

Custom builds need `setup_karst` after `setup_region`. Rebuild older karst maps,
which may contain only zeros. Rerun spinup after enabling karst or changing its
settings. Existing `release_time_days` settings are obsolete and should be
removed from `model.yml`.

In `hydrology`, report `.karst_capture_m` on HRUs and `.karst_recharge_m` on the
grid (m/day). Land-surface runoff reports show runoff before capture; captured
water is removed from routed runoff and added to groundwater recharge in the
same timestep.

## Code

::: geb.hydrology.groundwater
