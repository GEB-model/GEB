# Soil

## Intiltration

We implement a hybrid infiltration scheme that accounts for both saturation excess and infiltration excess runoff mechanisms by integrating the sub-grid heterogeneity logic of the Variable Infiltration Capacity (VIC) and Xinanjiang models with a physically based Green-Ampt rate constraint. Infiltration is calculated with sub-hourly timesteps of 10 minutes.

### Saturation excess

This component addresses storage-driven runoff generation, or Dunne runoff. Following the logic of the Xinanjiang model [@zhao1992xinanjiang] and VIC [@liang1994simple], the scheme assumes that the soil storage capacity varies spatially within a grid cell according to a power-law distribution. 

As the mean soil moisture ($W$) increases, an increasing fraction of the cell area ($A_s$) becomes fully saturated. Any precipitation falling on these saturated fractions is immediately converted to runoff. The fraction of the basin area ($A_s$) with infiltration capacity less than or equal to $i$ is assumed to be:

$$A_s = 1 - \left( 1 - \frac{i}{i_{max}} \right)^b$$

Where:
* $i$ is the point infiltration capacity.
* $i_{max}$ is the maximum point infiltration capacity in the basin.
* $b$ is the shape parameter of the distribution.

### Infiltration excess

While storage-based models often assume the soil can accept water at any rate until it is full, this scheme imposes a maximum infiltration capacity ($f_p$) based on the Green-Ampt equation [@green1911studies]. 

The infiltration capacity is dynamically scaled using a suction ratio derived from the soil's bubbling pressure ($\psi_b$) and the layer depth ($D$). The capacity is calculated as:

$$f_p = K_{sat} \left( 1 + \frac{\psi_f}{D} \frac{1 - S}{S} \right)$$

Where:
* $f_p$ is the infiltration capacity [$L/T$].
* $K_{sat}$ is the saturated hydraulic conductivity [$L/T$].
* $\psi_f$ is the effective suction head at the wetting front [$L$].
* $D$ is the soil layer depth [$L$].
* $S$ is the relative saturation (current storage / maximum capacity) $[-]$.

When the soil is dry, the matric suction gradient is at its peak, significantly increasing the infiltration rate above $K_{sat}$. As relative saturation ($S$) increases, this suction term decays. If the rainfall intensity exceeds $f_p$, Hortonian runoff (infiltration excess) is generated, even if the total soil storage is not yet exhausted.

### Runoff generation

The saturation excess and infiltration excess processes described above provide two independent constraints on potential infiltration. In this hybrid scheme, the actual infiltration is determined by the minimum of the two methods. Any precipitation that cannot infiltrate due to either the volumetric or flux limit is partitioned as surface runoff. This dual-constraint approach ensures the model remains robust across different soil moisture states and rainfall intensities.

::: geb.hydrology.soil
