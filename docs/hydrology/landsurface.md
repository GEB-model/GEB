# Land Surface

## Land use classes

Within GEB there are 6 land use classes in GEB, following the [CWatM](https://cwatm.iiasa.ac.at) convention. Note that due to other parameterisations within a land-use class, two cells that are both simulated as `grassland-like` can respond very differently. For example, high grass and desert are both in the `grassland-like` class but due the absence of vegetation in the desert these cells still respond differently given the same atmospheric inputs. 

| Number | Name | Definition |
| :--- | :--- | :--- |
| 0 | Forest | Forests and other wooded land. |
| 1 | Grassland-like | Grasslands, scrublands, and other non-forest natural vegetation. |
| 2 | Paddy Irrigated | Irrigated land used for paddy cultivation. |
| 3 | Non-Paddy Irrigated | Irrigated land used for other crops than paddy cultivation. |
| 4 | Sealed | Urban areas, paved roads, and other sealed surfaces. |
| 5 | Open Water | Lakes, rivers, and other open water bodies. |

Land use classes `forest`, `grassland-like`, `sealed` and `open water` are set from the land `land_use_classes` input dataset. Land use classes `paddy irrigated` and `non-paddy irrigated` are set dynamically during the model run through the farmer agents. These lands are initially modelled as `grassland-like`, but when a farmer decides to irrigate their land, the land use type is switched dynamically to either `paddy irrigated` and `non-paddy irrigated`. After the growing season, the land is switched back to `grassland-like`.

## Vegetation water demand

`parameters.crop_factor_multiplier` scales vegetation factors used to calculate
potential evapotranspiration. Despite its name, it also affects forest and
grassland vegetation, not only cultivated crops. The default is `1.0`, which
preserves the vegetation factors derived by the model. A value of `1.5` increases
potential demand by 50%; actual evaporation and transpiration also depend on
available water and other model constraints.

The neutral default restores the value used before 21 October 2026. Use explicit
regional overrides for calibrated values. After changing this multiplier, rerun
spinup and simulation so soil moisture, groundwater heads, and channel geometry
reflect the chosen demand. No input-file migration is required.

## Code

::: geb.hydrology.landsurface
