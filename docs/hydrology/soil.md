# Soil

## Interflow

Interflow, or lateral subsurface flow, is the movement of water within the soil profile parallel to the land surface. In GEB, interflow is calculated for each soil layer when the soil moisture content exceeds the field capacity. This "free water" is available to move laterally driven by gravity and the slope of the terrain, and then added to the channel in each grid cell.

The interflow calculation conceptualizes the hillslope as a draining reservoir. The rate of drainage is determined by:

1.  **Free Water**: The amount of water in excess of the soil's field capacity.
2.  **Drainable Porosity**: The difference between saturated water content and field capacity per unit of soil depth.
3.  **Physical Properties**:
    *   **Slope**: Steeper slopes result in faster drainage.
    *   **Hillslope Length**: Longer slopes provide more resistance/storage.
    *   **Lateral Hydraulic Conductivity**: Assumed to be 10 times the vertical saturated hydraulic conductivity to account for soil anisotropy.

The fraction of free water that becomes interflow is controlled by a `storage_coefficient`.

$$
\text{Interflow} = \text{Free Water} \times \text{Storage Coefficient}
$$

where the Storage Coefficient is derived as:

$$
\text{Storage Coeff.} = \left( \frac{K_{lat} \times \text{Slope}}{\phi_d \times L_{hill}} \right) \times \text{Multiplier}
$$

*   $K_{lat}$: Lateral saturated hydraulic conductivity [m/h]
*   $\phi_d$: Drainable porosity [-]
*   $L_{hill}$: Hillslope length [m]

::: geb.hydrology.soil
