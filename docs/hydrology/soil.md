# Soil

## Infiltration

We implement an infiltration scheme based on the Green-Ampt equation [@green1911studies], solving for infiltration capacity physically based on soil properties and moisture deficit.

### Infiltration Capacity (Green-Ampt)

The model determines the maximum infiltration capacity ($f_{cap}$) for the current timestep using the explicit Green-Ampt approximation derived by **Sadeghi et al. (2024)** [@sadeghi2024simple]. This avoids the need for iterative solutions or sub-stepping typically required for the implicit Green-Ampt formulation.

The cumulative infiltration $I(t)$ is calculated as:

$$ I(t) = K_{sat} t \left( 0.70635 + 0.32415 \sqrt{1 + 9.43456 \frac{S^2}{K_{sat}^2 t}} \right) $$

Where:
* $I(t)$ is the cumulative infiltration [$L$].
* $K_{sat}$ is the saturated hydraulic conductivity [$L/T$].
* $S^2 = 2 K_{sat} \psi_f \Delta \theta$ is the square of sorptivity [$L^2/T$].
* $\psi_f$ is the wetting front suction head [$L$].
* $\Delta \theta$ is the moisture deficit [$ - $].
* $t$ is the time since the start of the infiltration event [$T$].

The model tracks the wetting front depth ($L$) as a state variable. At the beginning of each timestep, the "effective time" ($t_{eff}$) corresponding to the current wetting front is calculated by inverting the standard Green-Ampt equation (exact analytical inversion). The potential cumulative infiltration at $t_{eff} + \Delta t$ is then calculated using the explicit formula above. The difference determines the maximum infiltration capacity for the timestep.

$$f_{cap} = I(t_{eff} + \Delta t) - I(t_{eff})$$

If the rainfall intensity exceeds this capacity, Hortonian (infiltration excess) runoff is generated.

### Saturation Excess

Infiltration is also limited by the available pore space in the soil column. The model calculates the available storage in the active soil layers (layers reached by the wetting front). If the soil becomes saturated (reaches $W_s$), no further infiltration can occur, and any additional water serves as saturation excess runoff (Dunne runoff).

### Runoff Generation

The actual infiltration for a timestep is determined by the minimum of:
1. The available water on the surface (precipitation + accumulated topwater) [$L$].
2. The Green-Ampt infiltration capacity ($f_{cap}$) [$L$].
3. The available storage in the soil column (up to the wetting front depth or bottom of soil) [$L$].

Any water that does not infiltrate contributes to surface runoff.

::: geb.hydrology.soil
