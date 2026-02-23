# Interception

## Interception Capacity

The maximum amount of water that can be stored on the canopy is the **interception capacity** ($S_{max}$). For forests, this is estimated based on the Leaf Area Index ($LAI$) using the Von Hoyningen-Huene formula[@von1983interzeption], following its implementation in LISFLOOD[@vanderknijff2010lisflood].

$$
S_{max} = 0.935 + 0.498 \cdot LAI - 0.00575 \cdot LAI^2 \quad (\text{for } LAI > 0.1)
$$

$$
S_{max} = 0 \quad (\text{for } LAI \le 0.1)
$$

Where:
- $S_{max}$ is interception capacity in mm.
- $LAI$ is the Leaf Area Index ($m^2/m^2$).

For other land use types, fixed interception capacities are used (e.g., 1mm for irrigated crops).

::: geb.hydrology.interception
