## Overview
GEB (Geographical Environmental and Behavioural model) simulates the environment (e.g., hydrology, floods), the individual people, households and orginizations as well as their interactions at both small and large scale. The model does so through a "deep" coupling of an agent-based model a hydrological model, a vegetation model and a hydrodynamic model. You can find full documentation [here](https://geb-model.github.io/GEB/).

The figure below shows a schematic overview of the model agent-based and hydrological model.

![Schematic model overview of GEB.](https://raw.githubusercontent.com/GEB-model/GEB/refs/heads/main/docs/images/schematic_overview.svg "Schematic model overview")


## Installation (not for development)

GEB can be installed with pip, including all dependencies on Windows, Linux and Mac OS X.

```bash
pip install geb
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install geb --prerelease=allow
```

## Development installation and setup

To contribute to GEB, we recommend first cloning the repository from this repo using `git clone`, and then use uv to install the dependencies. Therefore, first install [uv](https://docs.astral.sh/uv/) and [git](https://git-scm.com/).

After installation, open a **new** shell and execute the following within a folder where you want to store the GEB repository and models:

```bash
git clone git@github.com:GEB-model/GEB.git
cd GEB
git switch main  # use main by default, but may be changed to another branch
git update-index --skip-worktree .vscode/extensions.json  .vscode/launch.json  .vscode/settings.json  # we don't want to commit local changes to the vscode settings
uv sync --dev
```

You will now have a virtual environment (`.venv`) in the GEB folder with the right Python installation and all packages you need.

Now open Visual Studio Code in the GEB folder (or use the "File -> Open Folder" dialog in Visual Studio Code).

```bash
code .
```

Visual Studio code should now prompts you to install the recommended extensions, which we recommend you do. After installing the Python extension VS Code should also automatically use the environment you created earlier. To test this, open a terminal in Visusal Studio Code (`Terminal -> New Terminal`) and run:

```bash
geb --help
```

We have also prepared a configuration for the debugger in `.vscode/launch.json`. The debugger assumes that you have the data files for the model located in `../model` (i.e., your `model.yml` is in `..model/`).

You may need to adjust the paths in  `.vscode/launch.json` to match your setup. In case you do this, please make sure to not commit these changes to the repository. To tell git to ignore any local changes here, you can run:

```bash
git update-index --skip-worktree .vscode/extensions.json  .vscode/launch.json  .vscode/settings.json  # we don't want to commit local changes to the vscode settings
```

Happy gebbing! Let us know when you run into issues, and any contributions to GEB are more than welcome. You can find a list of active and past contributors at the bottom of this file.

## Cite as

### Model framework

> de Bruijn, J. A., Smilovic, M., Burek, P., Guillaumot, L., Wada, Y., and Aerts, J. C. J. H.: GEB v0.1: a large-scale agent-based socio-hydrological model – simulating 10 million individual farming households in a fully distributed hydrological model, Geosci. Model Dev., 16, 2437–2454, [https://doi.org/10.5194/gmd-16-2437-2023](https://doi.org/10.5194/gmd-16-2437-2023), 2023.

### Applications

> Kalthof, M. W. M. L., de Bruijn, J., de Moel, H., Kreibich, H., and Aerts, J. C. J. H.: Adaptive Behavior of Over a Million Individual Farmers Under Consecutive Droughts: A Large-Scale Agent-Based Modeling Analysis in the Bhima Basin, India, EGUsphere preprint, [https://doi.org/10.5194/egusphere-2024-1588](https://doi.org/10.5194/egusphere-2024-1588), 2024.

## Building on the shoulders of giants

GEB builds on, couples and extends several models, depicted in the figure below.

![Model components of GEB.](https://raw.githubusercontent.com/GEB-model/GEB/refs/heads/main/docs/images/models_overview.svg "Schematic model overview")

1. Burek, Peter, et al. "Development of the Community Water Model (CWatM v1.04) A high-resolution hydrological model for global and regional assessment of integrated water resources management." (2019).
2. Langevin, Christian D., et al. Documentation for the MODFLOW 6 groundwater flow model. No. 6-A55. US Geological Survey, 2017.
3. Tierolf, Lars, et al. "A coupled agent-based model for France for simulating adaptation and migration decisions under future coastal flood risk." Scientific Reports 13.1 (2023): 4176.
4. Streefkerk, Ileen N., et al. "A coupled agent-based model to analyse human-drought feedbacks for agropastoralists in dryland regions." Frontiers in Water 4 (2023): 1037971.
5. Joshi, Jaideep, et al. "Plant-FATE-Predicting the adaptive responses of biodiverse plant communities using functional-trait evolution." EGU General Assembly Conference Abstracts. 2022.
6. Leijnse, Tim, et al. "Modeling compound flooding in coastal systems using a computationally efficient reduced-physics solver: Including fluvial, pluvial, tidal, wind-and wave-driven processes." Coastal Engineering 163 (2021): 103796.

## Developers (ordered by full-time equivalent working time on model)
- [Jens de Bruijn](https://research.vu.nl/en/persons/jens-de-bruijn)
- [Maurice Kalthof](https://research.vu.nl/en/persons/maurice-kalthof)
- [Veerle Bril](https://research.vu.nl/en/persons/veerle-bril)
- [Lars Tierolf](https://research.vu.nl/en/persons/lars-tierolf)
- [Tarun Sadana](https://research.vu.nl/en/persons/tarun-sadana)
- [Tim Busker](https://research.vu.nl/en/persons/tim-busker)
- [Rafaella Oliveira](https://research.vu.nl/en/persons/rafaella-gouveia-loureiro-oliveira)

## Current or past contributors (in order of first to last contribution)
- [Mikhail Smilovic](https://iiasa.ac.at/staff/mikhail-smilovic)
- Luca Guillaumot
- Romijn Servaas
- Thomas van Eldik