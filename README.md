Hello! Welcome to GEB! You can find full documentation here: [jensdebruijn.github.io/GEB](https://jensdebruijn.github.io/GEB/index.html)

## Overview
GEB stands for Geographic Environmental and Behavioural model and is named after Geb, the personification of Earth in Egyptian mythology.

GEB aims to simulate both environment, for now the hydrological system, the individual behaviour of people and their interactions at large scale. The model does so by coupling an agent-based model which simulates millions individual people or households and a hydrological model. While the model can be expanded to other agents and environmental interactions, we focus on farmers, high-level agents, irrigation behaviour and land management for now.

The figure below shows a schematic overview of the model. The lower part of the figure shows the hydrological model, CWatM, while the upper part shows the agent-based model. Both models are imported in a single file and run iteratively at a daily timestep.

![Schematic model overview of GEB.](/docs/images/schematic_overview.svg "Schematic model overview")

## Creating model input
We are currently working on making the model globally applicable using hydroMT.

    hydromt build --force-overwrite geb ..\DataDrive\GEB_Sanctuary\input\ -d ..\DataDrive\GEB\original_data\data_catalog.yml -i .\hydromt.yml --region "{'subbasin': [[73.98727], [19.00465]], 'bounds': [66.55, 4.3, 93.17, 35.28]}"

## Cite as
de Bruijn, J. A., Smilovic, M., Burek, P., Guillaumot, L., Wada, Y., and Aerts, J. C. J. H.: GEB v0.1: a large-scale agent-based socio-hydrological model – simulating 10 million individual farming households in a fully distributed hydrological model, Geosci. Model Dev., 16, 2437–2454, [https://doi.org/10.5194/gmd-16-2437-2023](https://doi.org/10.5194/gmd-16-2437-2023), 2023.