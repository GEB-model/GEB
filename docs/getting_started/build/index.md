# Creating a model

## GEB init

To initialize a new model, we first need to create a new folder for the model. This folder will contain all the files and data required for the model. Then in the folder, run the following command:

``` python
geb init
```

This will copy the default model configuration files from the examples folder in the GEB repository to the current folder. You can pass a custom subbasin ID or geom to the command to create a model for a specific region. See the section below for more information on how to specify the study region. If you do not specify a subbasin ID or geom, the default model configuration files will be copied.

``` python
geb init --basin-id 23011134
```

For more options, you can call the command with the `--help` flag:

``` python
geb init --help
```

### Study region

The `model.yml`-file specifies the configuration of the model, including the location of the model. An example of the `model.yml`-file is given in the examples folder in the GEB repository. Please refer to the yaml-section `general:region`. Examples are given below.

#### Subbasin

The subbasin option allows you to define your study region based on a hydrological basin. When using this option, the following parameter is required:

- `subbasin`: The subbasin ID of the model. This is the ID of the subbasin in the [MERIT-BASINS dataset](https://www.reachhydro.org/home/params/merit-basins) (version MERIT-Hydro v0.7/v1.0). This can be either a single subbasin or list of subbasins. All upstream basins are automatically included in the model, so only the most downstream subbasin of a specific catchments needs to be specified.

There is a very nice viewer to select the right basin [here](https://cw3e.ucsd.edu/hydro/merit_rivers/merit_rivers_carto.html). You will need the "COMID".

``` yaml
general:
  region:
    subbasin: 23011134
```

or

``` yaml
general:
  region:
    subbasin:
    - 23011134
    - 23011135
    - 23011136
```

#### geom

The name of a dataset specified in the `data_catalog.yml` (e.g., GADM_level0) or any other region or path that can be loaded in geopandas. Using the column and key parameters, a subset of data can be specified, for example:

``` yaml
general:
  region:
    geom: GADM_level0
    column: GID_0
    key: FRA
```

#### outflow

The outflow option allows you to define your study region based on a specific outflow point using lat, lon coordinates:

``` yaml
general:
  region:
    outflow:
      lat: 48.8566
      lon: 2.3522
```

## GEB build

GEB has a build module to preprocess all input data for the specified region. This command creates a new folder "input" with all input files for the model.

### Obtaining the raw input data

Most of the data that the build module uses to create the input data for GEB is downloaded by the tool itself when it is run. However, some data needs to be aquired seperately. To obtain this data, please send an email to Jens de Bruijn (<jens.de.bruijn@vu.nl>).

### Configuration

Some of the data that is obtained from online sources and APIs requires keys. You should take the following steps:

1.  Request access to MERIT Hydro dataset [MERIT Hydro](https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/), and create a ".env"-file in the GEB repository with the following content:

``` text
MERIT_USERNAME=<your_username>
MERIT_PASSWORD=<your_password>
```

2.  To set up the model with ERA5-Land forcing data using the build-method `setup_forcing_era5`, create an account on [Destination Earth](https://earthdatahub.destine.eu/). Then, create a personal access token and add the content to the ".env"-file in the GEB repository:

``` text
DESTINATION_EARTH_KEY=edh_pat_de<remainder_of_your_personal_access_token>
```

You can find the personal access token [here](https://earthdatahub.destine.eu/account-settings).

3.  To use forecasts in GEB, unflag the build-method "setup_forecasts" in the build.yml file. This will trigger the downloading and processing of ECMWF ensemble forecasts from the ECMWF archive. To access ECMWF forecasts, request access to the [ECMWF MARS archive](https://confluence.ecmwf.int/display/WEBAPI/Access+MARS). Afterwards, find your [API key](https://api.ecmwf.int/v1/key/) and add the content to the ".env"-file in the GEB repository.

4.  To set up the Global Tide and Surge Model using the build-method `setup_forcing_era5` you first need to create an account on [ECMF]([https://earthdatahub.destine.eu/](https://accounts.ecmwf.int/auth/realms/ecmwf/protocol/openid-connect/auth?client_id=cds&scope=openid%20email&response_type=code&redirect_uri=https%3A%2F%2Fcds.climate.copernicus.eu%2Fapi%2Fauth%2Fcallback%2Fkeycloak&state=IA76J5TAf7ZAgZ3YBCPSjsC1b4LKiENc3SozoQ5hbWA&code_challenge=LZdj2TMGRZZ4Aei7DFEKlht_kLHs7EInxqL3qax9oIE&code_challenge_method=S256).Afterwards, you will find your CDS API key and further instructions [here] (https://cds.climate.copernicus.eu/how-to-api). You should store the API url and key in your home folder as "$HOME/.cdsapirc".

### Building to model

The `build.yml`-file contains the name of functions that should be run to preprocess the data. The processed data will be stored in the "input" folder in the working directory. The data is stored in a format that is compatible with the GEB model. You can build the model using the following command, assuming you are in the working directory of the model which contains the `model.yml`-file and `build.yml`-file:

```bash
geb build
```

Optionally, you can include `--continue` to continue a previously interrupted build.

```bash
geb build --continue
```

Optionally, you can specify the path to the `build.yml`-file using the `-b/--build-config` flag, and the path to the `model.yml`-file using the `-c/--config` flag. You can find more information about this and other options by running:

```bash
geb build --help
```

### Updating the model

It is also possible to update an already existing model by running the following command.

```bash
geb update
```

This assumes you have a "update.yml"-file in the working directory. The `update.yml`-file contains the name of functions that should be run to update the data. The functions are defined in the "geb" plugin of HydroMT. The data will be updated in the "input" folder in the working directory. The data is stored in a format that is compatible with the GEB model.

For example to update the forcing data of the model, your "update.yml"-file could look like this, essentially a subset of the build.yml-file:

```yaml
setup_forcing:
```

Optionally, you can specify the path to the "update.yml"-file using the `-b/--build-config` flag, and the path to the `model.yml`-file using the `-c/--config` flag. 

You can also run update with the build.yml-file by using the following syntax, running the setup_forcing step from the build.yml (or any other) file.

```bash
geb update -b build.yml::setup_forcing
```

To do the same, and also run all subsequent methods, you can add a `+` at the end:

```bash
geb update -b build.yml::setup_forcing+
```

You can find more information about these and other options by running:

```bash
geb update --help
```
