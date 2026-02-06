# Installation

## Installation using an LLM agent

To install GEB using an LLM agent, you can use the following prompt:

```
Please set up the GEB model environment for me. You can following the instructions here for the installation: https://docs.geb.sh/getting_started/installation/index.md. Ask the user whether they want to install it as a developer or the normal installation. When you install something, first check if it is not already installed. For the developer mode, note that you should clone the repository to the CURRENT folder using `git clone https://github.com/GEB-model/GEB.git .` including the `.`. For the developer mode, also include the configuration of VSCode, but do so using commands rather than using the visual interface. At the end, check if the ssh keys and the connection to GitHub are set up already. You can use "ssh -T git@github.com". If this is not working, ask the user whether they want to set this up to allow the user to contribute via GitHub. Note that the cloning itself can be done without the keys as it is a public repository.
```

## Installation with Python environment (not for development)

GEB can be installed with pip, including all dependencies on Windows, Linux and Mac OS X.

```bash
pip install geb
```

or with [uv](https://docs.astral.sh/uv/getting-started/installation/). If you have uv already, we recommend te update it first using `uv self update`.

```bash
uv pip install geb --prerelease=allow
```

To run SFINCS (the hydrodynamic model), you also need to install Docker (on Windows) or Apptainer (on Linux and Mac OS X). To install Docker you need to obtain and install Docker from their website (https://www.docker.com/get-started) and make sure Docker or Apptainer is running.

## Installation as a tool

GEB can also be installed without setting up a Python environment (thanks to [uvx.sh](https://uvx.sh/).

On Mac OS X and Linux:

```bash
curl -LsSf uvx.sh/geb/install.sh | sh
```

On Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://uvx.sh/geb/install.ps1 | iex"
```

## Development installation and setup

### Requirements

- git: if not installed, you can obtain it for example [here](https://git-scm.com/).
- uv: if not installed, you can find it [here](https://docs.astral.sh/uv/getting-started/installation/). If it is installed, ensure that it is updated using `uv self update`.
- docker/apptainer: To run SFINCS (the hydrodynamic flood model), you also need to install [Docker](https://www.docker.com) (on Windows) or [Apptainer](https://apptainer.org) (on Linux and Mac OS X).

### Installation

In this guide, we assume that you already created a folder where you would like the code to be created and that you have this folder open in the terminal.

First, *clone* (download) all the code from the repository into the current working directory (that's what the `.` at the end of the command is for):

```bash
git clone https://github.com/GEB-model/GEB.git .
```

Then to install GEB and its dependencies:

```bash
uv sync --dev  # install all dependencies including dev dependencies using uv
```

You will now have a virtual environment (.venv) in the GEB folder with the right Python installation and all packages you need.

Now open a new Visual Studio Code window in the GEB code folder, “GEB” (or use the "File -> Open Folder" dialog in Visual Studio Code). 

```bash
code .
```

Visual Studio code should now prompt you to install the recommended extensions, which we recommend you do. After installing the Python extension VS Code should also automatically use the environment you created earlier. To test this, open a terminal in VS Code (Terminal -> New Terminal) and run:

```bash
uv run geb --help
```

If this doesn’t work, press "Ctrl+Shift+P", search for “Select Interpreter”, and choose the .venv environment (probably “./.venv/bin/python”). 

We have also prepared a configuration for the debugger in `.vscode/launch.json.sample` and a settings file in `.vscode/settings.json.sample` with some useful default settings. To activate these files, duplicate (i.e., not remove or rename) the files and rename to `.vscode/launch.json` and `.vscode/settings.json` respectively. 

The debugger assumes that you have the data files for the model located in `../model` (i.e., your `model.yml` is in `..model/`). You may need to adjust the paths in  `.vscode/launch.json` to match your setup.

Happy gebbing! Let us know when you run into issues, and any contributions to GEB are more than welcome. If you make a contribution and push it to GitHub, you will also need to perform the steps in the next section.

## Contributing to GEB

To allow contributions to the GEB model, first make sure you are a [member of the GEB-model](https://github.com/orgs/GEB-model/people/) and put your GitHub user credentials in your git, by pasting the following in your git bash shell or VS code terminal replacing "GITHUB-USERNAME" and "GITHUB-EMAIL":

```bash
git config --global user.name "GITHUB-USERNAME"
git config --global user.email "GITHUB-EMAIL"
```

We need to connect to Github through SSH before we can clone the repo. For this, carefully follow all the steps to generate a SSH key and to [add this SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Select the right operating system (Mac, Windows or Linux).
