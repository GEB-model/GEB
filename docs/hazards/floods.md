# Floods

## Introduction

Lorem ipsum

## Model building

Lorem ipsum

### Static input data

Lorem ipsum

### (Dynamic) forcing data

Lorem ipsum

#### Riverine forcing

Lorem ipsum

#### Coastal forcing

Lorem ipsum

### Rebuilding

We distinguish two parts of a SFINCS model: the static data and dynamic data:

- The static data created in the [`SFINCS root model`][geb.hazards.floods.sfincs.SFINCSRootModel] is saved in the root model and often identical between SFINCS runs. For example, when there are two flood events, the DEM usually remains stable. Since building a SFINCS model can take quite some time dependent on the size and configuration (primarily the grid size), the static data can be re-used between runs. In the model [configuration](../getting_started/configuration.md) `hazards.floods.force_overwrite` you can find the setting for floods:

    - `auto`: automatically detect whether the model must be overwritten. If there are any changes in the data that is provided to the [`SFINCS root model`][geb.hazards.floods.sfincs.SFINCSRootModel] or there are any changes in the code in `geb.hazards.floods`, the model is overwritten. Otherwise, the existing model is used if it exists.
    - `true`: the model is always overwritten
    - `false`: the model is always read, unless it doesn't exist yet

- The dynamic data (e.g., forcing data) created in the [`SFINCS simulation`][geb.hazards.floods.sfincs.SFINCSSimulation] is different for each SFINCS simulation (and usually much smaller) and never saved between SFINCS runs.

## Model runs

Lorem ipsum

### Flood events

Lorem ipsum

### Return period maps

Lorem ipsum

## Code

::: geb.hazards.floods.sfincs
