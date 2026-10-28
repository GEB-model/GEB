# Future socioeconomic scenarios

GEB supports simulations into the future with some evolving socioeconomic elements, which can be used to analyze impacts of climate to change in evolving climate and socioeconomic scenarios.

## Overview

### Climate change scenarios

Future climate scenarios are standardized in community efforts to enhace comparability, reusability and reproducibility of climate science. Bear in mind this are **scenarios** and not *forecasts*, that is socioeconomic projections into the future are plausible and internally consistent narratives of the future, and they don't intend to place probabilities into each possible outcome as their obective is to explore the consequences of the possible evolution in the main driving forces for climate change. This driving forces are deeply uncertain as they depend on the evolution of humans systems with agency.

In the sixth report of the IPCC was released in 2021, the current scenario paradigm consists of Socioeconomic Development Pathways (SSPs) which contain narratives of future societies and their challenges to mitigate and adapt to climate change (e.g. SSP1 is called *'Sustainability: Taking the Green Road'*, while SSP3 is called: *'Regional Rivalry: A Rocky Road'*). 

Each SSP is compatible with different levels of climate action, which are called the Representative Concentration Pathways (RCPs). RCPs are called by the additional energy imbalance caused by a certain level of greenhouse gas (GHG) concentrations by 2100, that is, the anthropogenic radiative forcing in Watts per squared meter (e.g. RCP-1.9 is a very low emissions scenario roughly consistent with the 1.5°C threshold of the Paris agreeement, while the RCP-2.6 is roughly consistent with the 2°C threshold). ***Bear in mind that in the forthcoming IPCC's AR7 there are new emission scenarios called by their level of emissions, g.g. High, Low, etc, and that works using climate scenarios will need to eventually transition to these new emission scenarios.***

### Macroeconomic climate damage functions

Climate change is expected to have a significant impact on human systems and the non-human components of the Earth System on which our wellbeing depends. Some of this impacts can be **chronic**, meaning they are slow-evolving (like having less available land over decades due to sea level rise), or **acute** due to extreme events (like increasingly strong coastal floods). 

Climate economists build different kinds of macroeconomic models to measure the impacts of climate change in terms of lost economic output, that is, a reduction in Gross Domestic Product (GDP). Derived relationships between changes in temperature (or some other manifestation of climate change, like precipitation) and economic losses (mostly always GDP) are called **damage functions**. Most damage functions give damages at a global, regional, or country-level.

**Climrisk** is an integrated assessment model developed in the National Autonomous University of Mexico (UNAM) and the Vrije Universiteit Amsterdam (VU) by Francisco Estrada and collaborators. Climrisk emulates climate models to derive local temperature increases, and runs a spatialization kernel considering local population, GDP, and human development indexes to obtain a wide array global gridded damage functions at a 50x50 km resolution. This resolution is particularly useful when working with model with very high spatial resolutions such as GEB, where spatial differentiations of income growth and climate damages affect the spatial distribution of household decisions.

Most damage functions center on **chronic** damages (e.g. the *K* or *Kompas* damage function considers slow-evolving sea level rise as a land supply shock for agricultural productivity), while GEB models adaptation and expected annual damages to extreme events, that is, **acute** damages. Coupling spatial chronic economic damage functions with agent-level responses to local acute climate impacts allows for a more complete consideration of how individuals could response to combined acute and chronic impacts and how this could regionally redistribute adaption uptake and relocation decisions.

## Build Configuration

Building the model data for socioeconomic scenarios is specified in the `build.yml` file under the `setup_climrisk` method. Many combinations of parameters can be built at once. For further guidance, look at the `reasonable_default_build.yml` file.

The build method is automated to fetch the region mask and extract and crop data from the global Climrisk maps. In this way, socioeconomic scenarios can be built for any region in the world without any additional work on the part of the GEB user.

### Configuration Parameters

**ssp** *(default: '2')*:

List of Shared Socioeconomic Pathways to build data for. Choose from: '1', '2', '3', '4', '5'. Example:
 
    ssp:
      - '2'
      - '5'

**rcp** *(default: '70')*:

List of Representative Concentration Pathways to build data for. Choose from: '19', '26', '34', '45', '60', '70', '85'. Example:
 
    rcp:
      - '45'
      - '70'

**damage_function** *(required)*:

List of macroeconomic climate damages function to build data for. Choose from: 'K' (Kompas), 'KU' (Kompas + Urban Heat Island), 'KPU' (Kompas + U + persistent damages), 'RU' (RICE + U), 'RPU' (RICE + PU). Choose 'GDP' to project growth without damages. Example:

    damage_function:
      - 'GDP'
      - 'K'

**warming_percentile** *(default: '90')*

List of global warming percentiles to build data for. Choose from: '10', '50', '90'. Example:

    damage_function:
      - '10'
      - '50'

### Source data for model build

Currently, data for Climrisk spatial socioeconomic scenarios and damage functions is not public. If you need the data to build a model using the GEB Climrisk module, please contact Rodrigo Muñoz (<rodrigo.munos@atmosfera.unam.mx>)

GEB will look for the Climrisk data within the local cache of your model, so make sure the `climrisk` folder is located within the `your_model_directory/cache` folder.

## Run Configuration

Configuration for running the model data with socioeconomic scenarios is specified in the `config.yml` file under the `setup_climrisk` field, which is itself under the `agent_settings` field. Only a single combination of parameters can be used per run. For further guidance, look at the `reasonable_default_config.yml` file.

### Configuration Parameters


**project_income** *(required)*:

Turn on household income growth, calculating it with GDP per capita growth. Example:
 
    project_income: false

**project_costs** *(required)*:

Turn on adaptation, relocation and asset damages costs growth, calculating it with the growth of the sum of the GDP for all countries that include the modelled region. Example:
 
    project_costs: true

**ssp** *(required)*:

Shared Socioeconomic Pathway to use. Choose from: '1', '2', '3', '4', '5'. Example:
 
    ssp: '2'

**rcp** *(required)*:

Representative Concentration Pathway. Choose from: '19', '26', '34', '45', '60', '70', '85'. Example:
 
    rcp: '45'

**damage_function** *(required)*:

Macroeconomic climate damages function to build data for. Choose from: 'K' (Kompas), 'KU' (Kompas + Urban Heat Island), 'KPU' (Kompas + U + persistent damages), 'RU' (RICE + U), 'RPU' (RICE + PU). Choose 'GDP' to project growth without damages. Example:

    damage_function: 'KU'

**warming_percentile** *(required)*

Global warming percentile. Choose from: '10', '50', '90'. Example:

    damage_function: '90'

**country_wide** *(required)*

If turned on it will calculate income growth based on the growth of the average GDP per capita for all the region; if false it will calculate GDP per capita growth for each 50x50 km grid cell and assing the growth rate to households located within that cell. For households that appear to be in a Climrisk ocean cell (due to the lower Climrisk spatial resolution), the closest Climrisk land cell growth rate is chosen.  Example:

    country_wide: false

## References

- Riahi, K., Van Vuuren, D. P., Kriegler, E., Edmonds, J., O’Neill, B. C., Fujimori, S., Bauer, N., Calvin, K., Dellink, R., Fricko, O., Lutz, W., Popp, A., Cuaresma, J. C., Kc, S., Leimbach, M., Jiang, L., Kram, T., Rao, S., Emmerling, J., … Tavoni, M. (2017). The Shared Socioeconomic Pathways and their energy, land use, and greenhouse gas emissions implications: An overview. Global Environmental Change, 42, 153–168. [https://doi.org/10.1016/j.gloenvcha.2016.05.009](https://doi.org/10.1016/j.gloenvcha.2016.05.009)
- Estrada, F., & Botzen, W. J. W. (2021). Economic impacts and risks of climate change under failure and success of the Paris Agreement. Annals of the New York Academy of Sciences, 1504(1), 95–115. [https://doi.org/10.1111/nyas.14652](https://doi.org/10.1111/nyas.14652)
