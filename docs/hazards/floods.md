# Floods

## Introduction

The floods module in GEB uses the Super-Fast INundation of CoastS (SFINCS) hydrodynamic model to simulate floods. SFINCS is a fully automated, 2D reduced-complexity hydrodynamic model that solves simplified Saint-Venant equations of mass and momentum (Leijnse et al., 2021). It balances computational speed with physical realism, making it practical to simulate many flood scenarios across large regions. For detailed description of the model equations we refer to https://sfincs.readthedocs.io/en/latest/.

SFINCS works by dividing the area of interest into a grid of cells. For each cell, it calculates water depth and flow at successive time steps based on the elevation (topography), surface roughness (manning's), and incoming water (forcing) from rain (precipitation), rivers (discharge), or the coast (surge and storm tide).

Multiple flood types can be simulated:

- **Fluvial (riverine)**: Flooding from river overflow when discharge exceeds river channel capacity
- **Pluvial (precipitation)**: Surface (overland) flooding from intense rainfall overcoming local infiltration capacity
- **Coastal**: Inundation from elevated sea levels due to storm surge and tides
- **Return period**: Probability flood maps showing expected flooding for specific return periods (e.g., 1-in-100 year event)

## Model building

The SFINCS model in GEB is built in two stages: creating the base model structure (required input maps) and then adding forcing data for specific flood events.

When building a SFINCS model, GEB first creates a region of interest (eg., catchment boundary). This region by default is divided into grid cells (regular grid), with each cell storing information about elevation, land roughness properties, etc. The model can optionally use "subgrid", which captures fine-scale elevation details within each cell. This allows for faster simulations while still representing important features like small river channels.

Rivers are represented in the model in one of two ways:

- **With subgrid**: River channels are "burned" into a high-resolution subgrid, preserving their width and depth.
- **Without subgrid**: Rivers are directly carved into the main computational grid, modifying the elevation and roughness values of affected cells.

The model automatically identifies flood-prone areas inside the region using Height Above Nearest Drainage (HAND) analysis (*REF). This method calculates how high each location sits above the nearest stream or drainage channel, helping to define which areas are prione to flooding.

### Static input data

The static components of a SFINCS model remain constant across different flood simulations and include:

- **Digital Elevation Model (DEM)**: Multiple DEMs from different sources can be merged, with priority given to user defined 1st and subsequent source. For example in a riverine flood, the priority by default is given to inland elevation (FABDEM V1-2) and then if needed sometimes the outflows reach a part where topobathy is needed (2nd source: GEBCO version ?) 
- **Manning's roughness coefficient**: Represents surface friction that slows down water flow. Different land cover types (forests, urban developed areas, cropland etc.,) have different roughness values. By default the ESA Landcover 2021 is used.
- **Model domain (mask)**: Defines which grid cells are active in the simulation. This is determined based on the subbasins being modeled (delineated via the hydrological part) and made faster using the aforementioned HAND method.
- **River network**: The geometry (centerlines) by default use the MERIT-BASINS global product based on 90-m MERIT-HYDRO DEM. The width is derived in two parts, firstly satellite observed widths (resolution = 30m or larger) are given priority which comes from the MERIT-SWORD dataset (latest version 0.4)(*add ref). Secondly, whereever there is no satellite data available (<30m tributaries) a gap-filling method via the power-law equation is used to derive widths (*add ref). The depth of rivers are derived from discharge estimates and using the Manning's open channel flow equation.

### (Dynamic) forcing data

Dynamic forcing data varies between flood types and drives the actual inundation simulation. GEB supports multiple forcing methods that can be combined depending on the type of flood.

#### Riverine forcing

Riverine (fluvial) forcing represents water entering the model domain through rivers and streams. GEB provides two approaches:

- **Accumulated Runoff forcing**: The term "accumulated" refers to the fact that all rainfall-runoff from the upstream catchment area has been collected and concentrated at these locations.
- **All inflow point forcing**: Discharge is applied at multiple start points (headwater points) throughout the river network, including tributaries.

Discharge values comes from GEB hydrological module, which simulates rainfall-runoff processes across the region. For return period mapping, synthetic design hydrographs are generated based on extreme value analysis of long-term discharge records.

#### Precipitation Forcing

Direct precipitation forcing adds rainfall directly onto the SFINCS model grid. This is particularly important for pluvial (rainfall-induced) flooding, where surface runoff and local flooding cause inundation independently of river overflow.

Precipitation data can be taken from observed rainfall records, climate model outputs, or synthetic design storms. The SFINCS model routes this rainfall across the landscape based on topography and surface properties. This is an external input coming from sources outside GEB.

#### Coastal forcing

For coastal flood simulations, water level boundary conditions are applied along the coastline. These represent sea level variations due to tides and storm surges. To create storm surge hydrographs we follow the HGRAPHER method as outlined by Dullaart[@dullaart2023enabling]. To generate a hydrograph of the surge, this approach starts with extracting independent extremes from the surge time series based on the peaks-over-threshold (POT) method. For each selected surge event, the time series from 36 h before, until 36 h after the peak is extracted. Second, each 72 h surge event is normalized (i.e. dividing each surge level by the peak) such that the maximum surge value is equal to 1 (unitless). Third, the selected surge events are combined to calculate the average surge hydrograph. This is done by determining the time (relative to the peak) at which a specific surge height (from 0 to 1 with increments of 0.01) is exceeded. As an example, the figure below shows that for one surge event the exceedance time at a normalized surge height of 0.25 is 14.0 h before and 26.0 h (16.0 + 10.0) after the surge maximum occurred, as indicated by the black arrows. Then, for each normalized surge height the average exceedance time is computed, resulting in an average curve. Because the shape of the rising and falling limb of the surge can differ, the exceedance time is calculated separately for each, and they are subsequently merged into the final average surge hydrograph.


<figure markdown="span">
  ![HGRAPHER method](../images/hgrapher_method.png)
  <figcaption>Overview of the HGRAPHER method pipeline. Figure reproduced from Dullaart et al.[@dullaart2023enabling]</figcaption>
</figure>

The model identifies coastal boundary cells based on topography and closeness to the ocean. Water levels at these boundaries can vary through time, allowing simulation of storm surge events.

### Rebuilding

We distinguish two parts of a SFINCS model: the static data and dynamic data:

- The static data created in the [`SFINCS root model`][geb.hazards.floods.sfincs.SFINCSRootModel] is saved in the root model and often identical between SFINCS runs. For example, when there are two flood events, the DEM usually remains stable. Since building a SFINCS model can take quite some time dependent on the size and configuration (primarily the grid size), the static data can be re-used between runs. In the model [configuration](../getting_started/configuration.md) `hazards.floods.overwrite` you can find the setting for floods:

    - `auto`: automatically detect whether the model must be overwritten. If there are any changes in the data that is provided to the [`SFINCS root model`][geb.hazards.floods.sfincs.SFINCSRootModel] or there are any changes in the code in `geb.hazards.floods`, the model is overwritten. Otherwise, the existing model is used if it exists.
    - `true`: the model is always overwritten
    - `false`: the model is always read, unless it doesn't exist yet

- The dynamic data (e.g., forcing data) created in the [`SFINCS simulation`][geb.hazards.floods.sfincs.SFINCSSimulation] is different for each SFINCS simulation (and usually much smaller) and never saved between SFINCS runs.

## Model runs

Once the SFINCS model is built and forcing data is prepared, the simulation is executed to calculate how water moves and accumulates across the region domain. The model solves equations of water moving, tracking water depth at each time step, flow velocity, and direction throughout the domain.

Simulations can run on either CPU (default) or GPU (optional) hardware. GPU execution provides significant speed improvements (caution: GPU is untested at larger scales and can have instabilities) for large model domains, making it practical to simulate many flood scenarios or long time periods.

The model includes a spinup period (typically 24 hours) before the main simulation begins. During spinup, the model reaches a balanced initial state, ensuring that results are not affected by artificial conditions (too extreme amounts of water entering) at the start of the simulation.

### Flood events

Flood event simulations model specific historical or synthetic flood scenarios over a defined time period (e.g., a major storm lasting several days). These simulations use time-varying forcing data:

- Rivers discharge varies according to the hydrograph for that event
- Precipitation falls according to the rainfall pattern
- Coastal water levels vary following observed or modeled sea level conditions


### Return period maps

To generate return period flood maps (e.g., for a 1-in-100 year event), GEB simulates each subbasin individually.

#### The Paired Basin Approach

For each subbasin in the routing network, GEB constructs a SFINCS model domain that consists of the "subbasin of interest" and its immediate "downstream subbasin". This pairing ensures that:

1.  Flood waves travelling from the focus subbasin are properly routed through its downstream neighbor.
2.  Backwater effects or downstream water level constraints are better represented than if the model stopped exactly at a subbasin boundary.

<figure markdown="span">
  ![Paired subbasins and forcing points](../images/paired_basins.svg)
  <figcaption>**The paired subbasin approach.** For each segment, the local SFINCS model includes the focus subbasin and its immediate downstream neighbor. Design hydrographs are applied as discharge forcing at inflow nodes.</figcaption>
</figure>

#### Forcing and Simulation

The return period mapping process follows these steps:

1.  **Discharge Estimation**: GEB uses discharge time series from a long-term spinup or routing simulation to estimate peak flows for specific return periods (e.g., 10, 50, 100 years).
2.  **Hydrograph Generation**: For each subbasin of interest, a design hydrograph is generated for the estimated return period peak.
3.  **Boundary Conditions**: These hydrographs are applied as discharge forcing at the "inflow nodes" (upstream points) of the focused subbasin.
4.  **Local Hydrodynamic Modeling**: A separate SFINCS simulation is executed for each pairing.
5.  **Mosaicking**: The maximum flood depth maps from all individual simulations are then combined into a single, consistent flood visibility map for the entire region.

<figure markdown="span">
  ![Flood map mosaicking](../images/paired_basins_floods.svg)
  <figcaption>**Flood map mosaicking.** Individual localized flood depth maps are merged into a continuous mosaicked output.</figcaption>
</figure>

## Model output

SFINCS flood simulations produce both static outputs and time-varying dynamic outputs.

Common outputs include:

- **Maximum flood depth map**  
  Stored as `.zarr` file representing the maximum water depth over the entire simulation time period.

- **Time-varying dynamic output**  
  Stored as NetCDF (`.nc`) file containing water level, velocity, and other variables at each timestep.

- **Auxiliary outputs**  
  Depending on configuration, additional outputs such as figures for diagnostics may be produced.

## Performance metrics

Metrics may include:
- Comparison against observed flood extents
- Event-based skill scores (binary class statistics)

## Code

::: geb.hazards.floods.sfincs
