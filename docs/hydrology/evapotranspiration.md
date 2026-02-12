# Evapotranspiration

## Wind speed adjustment

Wind speeds measured at different heights above the soil surface are different. Surface friction tends to slow down wind passing over it. Wind speed is slowest at the surface and increases with height. For this reason anemometers are placed at a chosen standard height, i.e., 10 m in meteorology and 2 or 3 m in agrometeorology. For the calculation of evapotranspiration, wind speed measured at 2 m above the surface is required. Therefore, we use a logarithmic wind speed profile to adjust wind speed[@allen1998crop]:

$$ u_2 = u_z \frac{\ln(\frac{z_{2} - d}{z_0})}{\ln(\frac{z_{meas} - d}{z_0})} $$

where:

* $u_2$ wind speed at 2 m above ground surface [m s-1],
* $u_z$ measured wind speed at $z_{meas}$ m above ground surface [m s-1],
* $z_{meas}$ height of measurement above ground surface [m],
* $d$ zero plane displacement height, $d = 0.08$ m,
* $z_0$ roughness length, $z_0 = 0.01476$ m.

The displacement height $d$ and roughness length $z_0$ are calculated based on a reference crop height $h = 0.12$ m, where:

$d = \frac{2}{3} h$ and $z_0 = 0.123 h$.

## Crop factors

Crop factors (also called crop coefficients, $K_c$) scale the reference evapotranspiration ($ET_0$) to the potential evapotranspiration for a given vegetation or crop type. The model computes crop factors differently depending on the land use:

- Vegetation / per-crop crops: crop factors for agricultural crops are computed from crop-specific growth-stage tables (basal crop coefficients per stage) and the crop age/progress.
- Forest: forest crop factors are estimated empirically from Leaf Area Index (LAI) following methods similar to Allen & Pereira (2009)[@allen2009estimating] and implemented in the model as an exponential response to LAI (see formula below).
- Grassland-like: grassland and natural grass-dominated pixels have their crop factor estimated from LAI using the same empirical exponential mapping used for forests.
- Irrigated / paddy: irrigated land uses are treated as crops or as paddy depending on land-use class; paddy and irrigated areas typically follow fixed small interception values but their crop factors are computed via the crop-stage logic when they are modelled as crops.

For forests and grassland-like pixels the model uses an empirical formulation that maps LAI to a crop factor[@allen2009estimating] in the range roughly from 0.2 (very low canopy) to 1.2 (dense canopy). The implemented relation is:

$$
K_c = 0.2 + (1.2 - 0.2) \cdot \left(1 - e^{-0.7 \cdot LAI}\right)
$$

Where:
- $K_c$ is the crop factor (dimensionless).
- $LAI$ is the leaf area index ($m^2 m^{-2}$).

## Vegetation Response and Crop Group Numbers

Transpiration is limited by soil moisture availability. The onset of water stress is determined by the **critical soil moisture content** ($\theta_{crit}$), which depends on the fraction of easily available soil water ($p$).

### Fraction of Easily Available Soil Water ($p$)

The parameter $p$ defines the fraction of total available water that a plant can extract without stress. It is calculated based on the crop group number[@supit1994system] and potential evapotranspiration ($PET$):

$$
p = \frac{1}{0.76 + 1.5 \cdot PET_{cm}} - 0.1 \cdot (5 - \text{CropGroupNumber})
$$

Where:
- $PET_{cm}$ is potential evapotranspiration in cm/day.
- $\text{CropGroupNumber}$ indicates the adaptation to dry climates (1 = sensitive, 5 = resistant).

**Relationship:**
- **Higher Crop Group Number** (e.g., 5) $\rightarrow$ Higher $p$ $\rightarrow$ Stress starts at lower soil moisture (more resistant).
- **Lower Crop Group Number** (e.g., 1) $\rightarrow$ Lower $p$ $\rightarrow$ Stress starts at higher soil moisture (more sensitive).

This calculated $p$ is then used to determine the critical soil moisture:

$$
\theta_{crit} = (1 - p) \cdot (\theta_{fc} - \theta_{wp}) + \theta_{wp}
$$

## Code

::: geb.hydrology.evapotranspiration