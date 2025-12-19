# Installation

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
cd GEB  # switch the terminal to GEB code folder
```

Now, install uv using the command as listed above, in “Installation (not for development)”. Then, execute:  

```bash
uv sync --dev  # install all dependencies including dev dependencies using uv
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

The debugger assumes that you have the data files for the model located in `../model` (i.e., your `model.yml` is in `..model/`). You may need to adjust the paths in  `.vscode/launch.json` to match your setup.

Happy gebbing! Explore the GEB documentation to [setup a model](https://docs.geb.sh/).  Let us know when you run into issues, and any contributions to GEB are more than welcome.