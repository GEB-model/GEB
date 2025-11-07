GEB (Geographical Environmental and Behavioural model) simulates the environment (e.g., hydrology, floods), the individual people, households and orginizations as well as their interactions at both small and large scale. The model does so through a "deep" coupling of an agent-based model a hydrological model, a vegetation model and a hydrodynamic model. You can find full documentation [here](https://docs.geb.sh/).

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

To run SFINCS (the hydrodynamic model), you also need to install Docker (on Windows) or Apptainer (on Linux and Mac OS X). To install Docker you need to obtain and install Docker from their website (https://www.docker.com/get-started) and make sure Docker or Apptainer is running.

## Development installation and setup

To contribute to GEB, we recommend first cloning the repository from this repo using git clone. A couple of steps are necessary before you can clone. First, first install [git](https://git-scm.com/). Make sure you are a [member of the GEB-model](https://github.com/orgs/GEB-model/people/) and put the right user credentials in your git, by pasting the following in your git bash shell or VS code terminal:

```bash
git config --global user.name "USERNAME"
git config --global user.email "GITHUB EMAIL"
```

We need to connect to Github through SSH before we can clone the repo. For this, carefully follow all the steps to generate a SSH key and to [add this SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=windows). Select the right operator system (Max, Windows or Linux). See “Notes when installing on a HPC cluster” for instructions on how to do this on a remote machine. 
After this, you are ready to clone the GEB repository! 

Create a main GEB folder on your machine. Within this folder, create a folder where you would like to store the code and model, we call this the *working directory*. Note that this folder should NOT be placed into a cloud synchronized folder (e.g., OneDrive). In this *working directory*, create a folder called *model*, and place the model input files in this folder. The directory structure should look like this:

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
cd GEB  # switch the terminal to GEB code folder (../GEB/GEB)
git switch main  # switch to the main development branch by default, but may be changed to another branch
```

Now, install uv using the command as listed above, in “Installation (not for development)”. Then, execute:  

```bash
uv sync --dev  # install all dependencies using uv
```

You will now have a virtual environment (.venv) in the GEB folder with the right Python installation and all packages you need.

Now open a new Visual Studio Code window in the GEB code folder, “../GEB/GEB” (or use the "File -> Open Folder" dialog in Visual Studio Code). 

```bash
code .
```

Visual Studio code should now prompts you to install the recommended extensions, which we recommend you do. After installing the Python extension VS Code should also automatically use the environment you created earlier. To test this, open a terminal in VS Code (Terminal -> New Terminal) and run:

```bash
geb --help
```

If this doesn’t work, press "Ctrl+Shift+P", search for “Select Interpreter”, and choose the .venv environment (probably “./.venv/bin/python”). 

We have also prepared a configuration for the debugger in `.vscode/launch.json.sample` and a settings file in `.vscode/settings.json.sample` with some useful default settings. To activate these files, duplicate (i.e., not remove or rename) the files and rename to `.vscode/launch.json` and `.vscode/settings.json` respectively. 

The debugger assumes that you have the data files for the model located in `../model` (i.e., your `model.yml` is in `..model/`). You may need to adjust the paths in  `.vscode/launch.json` to match your setup. If the debugger doesn’t work, ensure that your VS Code is opened in the `../GEB/GEB` folder (not in the parent folder `../GEB`).

Happy gebbing! Explore the GEB documentation to [setup a model](https://docs.geb.sh/ ).  Let us know when you run into issues, and any contributions to GEB are more than welcome. You can find a list of active and past contributors at the bottom of this file.

## Installation on a remote High Performance Computer (HPC) 
When working with GEB on a High Performance Computing (HPC) cluster, you can follow the same steps as above, but instead on a local terminal you need to use a terminal connected to the HPC Cluster. You need to connect VS Code (or another code editor) to the cluster, for example with a tunnel or SSH connection (see these instructions for the [VU ADA HPC](https://ada-hpc.readthedocs.io/en/latest/login/#connecting-with-ssh-vs-code-editor). Moreover, you need to take some additional aspects into account.

- We recommend [WinSCP](https://winscp.net/eng/index.php) as a file manager. Ensure to show hidden files (which include the .ssh folder) by going to `preferences`,`panels`, and then `show hidden files`.
- To ensure VS Code can find git software, put the following in your .bashrc file: 

```bash
module load shared 2025 git/2.45.1-GCCcore-13.3.0
```

- To connect to Github with a SSH from the cluster, follow [this same link as above](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux), but choose “Linux” and execute the steps on a terminal connected to the cluster (for example, from an SSH connected VS code terminal). As a result, the .ssh folder and the ssh key will be made inside your cluster folder. 

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
