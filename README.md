GEB (Geographical Environmental and Behavioural model) simulates the environment (e.g., hydrology, floods), the individual people, households and orginizations as well as their interactions at both small and large scale. The model does so through a "deep" coupling of an agent-based model a hydrological model, a vegetation model and a hydrodynamic model. You can find full documentation [here](https://geb-model.github.io/GEB/).

The figure below shows a schematic overview of GEB.

![Schematic model overview of GEB.](https://raw.githubusercontent.com/GEB-model/GEB/refs/heads/main/docs/images/schematic_overview.svg "Schematic model overview")


## Installation (not for development)

GEB can be installed with pip, including all dependencies on Windows, Linux and Mac OS X.

```bash
pip install geb
```

or with [uv](https://docs.astral.sh/uv/), which first needs to be installed by running:


on Linux and Mac OS X:
  
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

on Windows: 

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After, GEB can be installed with: 

```bash
uv pip install geb --prerelease=allow
```

To run SFINCS (the hydrodynamic model), you also need to install Docker (on Windows) or Singularity (on Linux and Mac OS X). To install Docker you need to obtain and install Docker from their website (https://www.docker.com/get-started) and make sure Docker or Singularity is running.

## Development installation and setup

To contribute to GEB, we recommend first cloning the repository from this repo using `git clone`, and then use uv to install the dependencies. Therefore, first install [git](https://git-scm.com/) and uv (see above)

Also create a folder where you would like to store the code and model, we call this the *working directory*. Note that this folder should *NOT* be placed into a cloud synchonized folder (e.g., OneDrive). In this *working directory*, create a folder called *model*, and place the model input files in this folder. The directory structure should look like this:

```
working directory
|   model
|   |   model.yml
|   |   build.yml│       
|   |   input
|   |   (potential other files and folders)
```

Then, in the *working directory*, open a **new** terminal and run the following command to *clone* (download) all the code from the repository:

```bash
git clone git@github.com:GEB-model/GEB.git
```

Now the directory structure should look like this:

```
working directory
|   model
|   |   model.yml
|   |   build.yml     
|   |   input
|   |   (potential other files and folders)
|   GEB
|   |   README.md
|   |   (all files and folders from the repository)
```

Then proceed with the following commands:

```bash
cd GEB  # switch the terminal to GEB code folder
git switch main  # switch to the main development branch by default, but may be changed to another branch
uv sync --dev  # install all dependencies using uv
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

We have also prepared a configuration for the debugger in `.vscode/launch.json.sample` and a settings file in `.vscode/settings.json.sample` with some useful default settings. To activate these files, copy (i.e., not remove or rename) the files to `.vscode/launch.json` and `.vscode/settings.json` respectively. 

The debugger assumes that you have the data files for the model located in `../model` (i.e., your `model.yml` is in `..model/`). You may need to adjust the paths in  `.vscode/launch.json` to match your setup.

Happy gebbing! Let us know when you run into issues, and any contributions to GEB are more than welcome. You can find a list of active and past contributors at the bottom of this file.

## Cite as

### Model framework

> de Bruijn, J. A., Smilovic, M., Burek, P., Guillaumot, L., Wada, Y., and Aerts, J. C. J. H.: GEB v0.1: a large-scale agent-based socio-hydrological model – simulating 10 million individual farming households in a fully distributed hydrological model, Geosci. Model Dev., 16, 2437–2454, [https://doi.org/10.5194/gmd-16-2437-2023](https://doi.org/10.5194/gmd-16-2437-2023), 2023.

### Applications

> Kalthof, M. W. M. L., de Bruijn, J., de Moel, H., Kreibich, H., and Aerts, J. C. J. H.: Adaptive behavior of farmers under consecutive droughts results in more vulnerable farmers: a large-scale agent-based modeling analysis in the Bhima basin, India, NHESS, [https://doi.org/10.5194/nhess-25-1013-2025](https://doi.org/10.5194/nhess-25-1013-2025), 2025.

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
