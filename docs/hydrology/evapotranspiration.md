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

## Code

::: geb.hydrology.evapotranspiration
