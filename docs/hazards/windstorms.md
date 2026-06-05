# Windstorms

## Introduction

The windstorm hazard is integrated into the model in the form of return period maps that associate spatially explicit wind-speeds with event probabilities. The return period maps are calculated based on the peaks-over threshold method. For a detailed description of this method refer to https://docs.geb.sh/hazards/extreme_value

## Model Building / Map generation


## (Dynamic) Data
Wind speed is a commonly used variable to asses windstorm risk[@fonseca2025empirical].

**Copernicus ERA5 Data**
This dataset provides different climate indicators of windsorm derived form the fifth generation of the European Centre for Medium-Range Weather Forecasts (ECMWF) atmosprheric reanalyses (ERA5)[@C3S_Windstorm_2025]. This data has a temporal coverage of 60 years, from 1940 to present.

- **Windstorm footprint**  
  The maximum 10m wind gust over a 72-hour time window. The windstorm footprints are identified on the original ERA5 grid (0.25° x 0.25°) at roughly 31km resolution.

## Windstorm events
Based on the externaly created windstorm return maps, the model randomly simulates an event over a defined time period. These events can have different return periods (e.g, 1-in-100 year event or 50-in-100 year event). 


Since the maps are generated outside the GEB evironment, the maps go directly into the household script. For a detailed explanation of the household agent refer to https://docs.geb.sh/agents/households/

## Code













