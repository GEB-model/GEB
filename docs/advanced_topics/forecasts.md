# Forecasts

GEB supports the integration of weather forecasts to enable short-range hydrological forecasting and forecast reponse modelling. This functionality supports ensemble forecasts, in which each ensemble is run through the model chain in \"multiverse mode\".

## Overview

The forecast system in GEB enables:

- **Short-range hydrological forecasting**: Run hydrological simulations using weather forecast data
- **Ensemble forecasting**: Use multiple forecast members to quantify uncertainty
- **Trigger early warnings**: Provide flood warnings based on forecast data
- **Forecast-informed decision making**: Support agent decisions based on forecast scenarios

The system currently supports ECMWF (European Centre for Medium-Range Weather Forecasts) operational forecasts, with plans to extend support to hindcast data.

## Configuration

Forecast configuration is specified in the `build.yml` file under the `setup_forecasts` method.

### Configuration Parameters

**forecast_start** *(required)*

:   Start date for forecast data downloads. Format: `YYYY-MM-DD`

    Example: `"2024-01-01"`

**forecast_end** *(required)*

:   End date for forecast data downloads. Format: `YYYY-MM-DD`

    Example: `"2024-01-31"`

**forecast_provider** *(required)*

:   The forecast data provider. Currently only `"ECMWF"` is supported.

    Options: `"ECMWF"`

**forecast_model** *(required)*

:   The type of ECMWF forecast model to use:

    - `"probabilistic_forecast"`: Ensemble forecasts with 50 members (perturbed forecasts)
    - `"control_forecast"`: Deterministic forecast (single member)

    Options: `"probabilistic_forecast"` \| `"control_forecast"`

**forecast_resolution** *(required)*

:   Spatial resolution of the forecast data in degrees.

**forecast_horizon** *(required)*

:   Maximum forecast lead time in hours. The forecast will extend this many hours from each initialization time.

**forecast_timestep_hours** *(required)*

:   Temporal resolution of the forecast data in hours.

    Note: ECMWF has different temporal resolutions available depending on the forecast date:

    - Before 2016-11-23: 3-hourly from 0-144h, 6-hourly from 144-360h
    - From 2016-11-23 onwards: hourly from 0-90h, 3-hourly from 90-144h, 6-hourly from 144-360h

## ECMWF Forecast Support

### Data Source

GEB downloads forecast data from the ECMWF MARS (Meteorological Archival and Retrieval System) archive using the ECMWF Web API. This provides access to:

- **Operational archive**: Real-time and historical operational forecasts from 2010 onwards
- **High-resolution forecasts**: Spatial resolutions from 0.25° to 1.0°
- **Ensemble forecasts**: 50-member ensemble for uncertainty quantification
- **Control forecasts**: Single deterministic forecast runs

### Supported Variables

GEB automatically downloads the following meteorological variables from ECMWF:

  --------------------------------------------------------------------------------------
  Variable Name   MARS Code   Description                                   Units
  --------------- ----------- --------------------------------------------- ------------
  tp              228.128     Total precipitation                           kg m⁻² s⁻¹

  t2m             167.128     2-metre temperature                           K

  d2m             168.128     2-metre dewpoint temperature                  K

  ssrd            169.128     Surface shortwave solar radiation downwards   W m⁻²

  strd            175.128     Surface longwave radiation downwards          W m⁻²

  sp              134.128     Surface pressure                              Pa

  u10             165.128     10-metre u-component of wind                  m s⁻¹

  v10             166.128     10-metre v-component of wind                  m s⁻¹
  --------------------------------------------------------------------------------------

  : ECMWF Variables

### Forecast Types

**Ensemble Forecasts (probabilistic_forecast)**

:   - **Members**: 50 perturbed forecast members
    - **Purpose**: Quantify forecast uncertainty and provide probabilistic information
    - **Use case**: Risk assessment, early warning systems, ensemble-based decision making
    - **Output**: Multiple possible scenarios for each forecast initialization
    - **File naming**: `ENS_YYYYMMDDTHHMMSS.grb`

**Control Forecasts (control_forecast)**

:   - **Members**: 1 deterministic forecast
    - **Purpose**: Single \"best guess\" forecast without uncertainty information
    - **Use case**: Deterministic forecasting, computational efficiency
    - **Output**: Single scenario for each forecast initialization
    - **File naming**: `CTRL_YYYYMMDDTHHMMSS.grb`

## Setup Requirements

### API Access

To use ECMWF forecasts, you need:

1.  **ECMWF API Key**: Register at <https://api.ecmwf.int/v1/key/>

2.  

    **Environment Variable**: Set variables in your environment or `.env` file located in the GEB repository.

    :   Request access to the ECMWF MARS database \<https://confluence.ecmwf.int/display/WEBAPI/Access+to+MARS+data\>, and extend the \".env\"-file in the GEB repository with the following content:

> ``` text
> ECMWF_API_KEY=<your_API_KEY>
> ECMWF_API_URL="https://api.ecmwf.int/v1"
> ECMWF_API_EMAIL=<your_email>
> ```

3.  **ECMWF Python API**: Install the ECMWF API client

### Multiverse Mode

When forecasts are enabled in the model configuration (`model.yml`), GEB automatically enters \"multiverse mode\" when forecast data is available for the current simulation date:

``` yaml
# model.yml
general:
  forecasts:
    use: true  # Enable forecast-based multiverse mode
```

During multiverse mode:

1.  **State Saving**: Current model state is saved before forecast processing
2.  **Member Processing**: Each forecast member is processed sequentially:
    - Original forcing data is replaced with forecast data
    - Model runs to the end of the forecast period
    - Results are calculated and stored
3.  **State Restoration**: Model state is restored to the original condition
4.  **Continuation**: Normal simulation continues with historical data

## References

- [ECMWF Documentation](https://www.ecmwf.int/en/forecasts/documentation)
- [MARS Archive Documentation](https://apps.ecmwf.int/mars-catalogue/)
- [ECMWF Web API package](https://github.com/ecmwf/ecmwf-api-client)
- [GRIB Parameter Database](https://codes.ecmwf.int/grib/param-db/)
