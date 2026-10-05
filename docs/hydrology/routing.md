# Routing

The drainage network is derived at the target resolution (30'' by default) using Iterative Hydrography Upscaling (IHU;[@eilander2021hydrography]), from the original MERIT grid[@yamazaki2019merit] at 3'' resolution. To model channel flow, GEB uses a hybrid 1D routing scheme that couples a 1D local inertial formulation for low-gradient mainstem river channels with a kinematic wave approximation for steep upstream reaches. Both schemes receive lateral inflow ($q_{lat}$), which varies hourly and combines surface runoff, interflow, channel evaporation, and $1/24$ of daily baseflow and return flow.

## Local Inertial Routing

For all rivers, as defined in MERIT-Basins[@lin2019global], the model uses a 1D local inertial formulation[@yamazaki2013improving], using a simplified 1D Saint-Venant momentum equation with the local acceleration, gravity, and friction terms:

$$
\frac{\partial Q}{\partial t} + g A \frac{\partial \eta}{\partial x} + \frac{g n^2 |Q| Q}{A h^{4/3}} = 0
$$

where $Q$ is the discharge ($\text{m}^3/\text{s}$), $g$ is the gravitational acceleration ($\text{m}/\text{s}^2$), $\eta = z + h$ is the water surface elevation ($\text{m}$), $z$ is riverbed elevation ($\text{m}$), $h$ is water depth ($\text{m}$), $A$ is cross-sectional area ($\text{m}^2$), $n$ is Manning's roughness, and $x$ is reach length ($\text{m}$). Lateral inflow is included as a volume added to cell storage per sub-time step ($\Delta t$).

For full river basins, the boundary condition is defined by the ocean water level, which we keep stable at 0 m above sea level. If only part of a basin is selected, we approximate the conditions at the boundary by setting the water surface slope of the boundary grid cell equal to that of the grid cell immediately upstream.

## Kinematic Wave Routing

For all other grid cells, channel flow is modelled using the kinematic wave approximation of the Saint-Venant equations[@chow1988applied] as implemented in CWatM[@burek2020development]. The governing equations are the continuity equation and a simplified momentum power law linking channel area ($A$) to discharge ($Q$):

$$
\frac{\partial Q}{\partial x} + \frac{\partial A}{\partial t} = q_{lat}
$$

$$
A = \alpha Q^{\beta}
$$

where $A$ is wetted cross-sectional area ($\text{m}^2$), and $q_{lat}$ enters as a continuous source term representing lateral inflow per unit length ($\text{m}^2/\text{s}$) along the channel reach. The parameters $\alpha$ and $\beta$ are channel geometry terms derived from Manning's equation; for wide rectangular channels, $\beta = 0.6$ and $\alpha = (n \cdot P^{2/3} / \sqrt{S})^{0.6}$, where $P$ is wetted perimeter and $S$ is channel bed slope. Combining these yields an equation for the unknown discharge ($Q_{\text{new}}$), solved via Newton-Raphson iteration[@chow1988applied]:

$$
\frac{\Delta t}{\Delta x} Q_{\text{new}} + \alpha Q_{\text{new}}^{\beta} = \frac{\Delta t}{\Delta x} Q_{\text{in}} + \alpha Q_{\text{old}}^{\beta} + \Delta t \cdot q_{lat}
$$

## Channel Geometry and Waterbodies

River width $W$ is obtained from SWORD[@altenau2025swot] and translated to MERIT rivers using MERIT-SWORD[@wade2025bidirectional]. For channels not included in SWORD (i.e., small rivers), we estimate river width using the downstream hydraulic geometry relation[@moody2002characterization]:

$$
W = a \cdot Q^b
$$

with $a$ defaulting to 7.2 and $b$ to 0.5.

Mean river depth ($D$) is estimated using a similar equation, with default parameters $c=0.27$ and $d=0.30$[@moody2002characterization]:

$$
D = c \cdot Q^d
$$

For reaches with width observed from SWORD ($W_{\text{obs}}$), we adjust the bankfull depth to conserve bankfull cross-sectional area:

$$
h_{\text{bf}} = (r + 1) \, D \left( \frac{a \cdot Q^b}{W_{\text{obs}}} \right)
$$

where $r$ is the channel shape exponent ($r = 0.5$ for a parabolic cross-section), and $a \cdot Q^b$ is the expected hydraulic geometry width.

Lakes and reservoirs are integrated using HydroLAKES[@messager2016estimating]. Natural lake outflows follow a parabolic head-discharge relationship[@aigner2008dresdner; @burek2020development], whereas controlled reservoirs are managed via agent-based operational rules (see [Reservoir Operators](../agents/reservoir_operators.md)).

## Code

::: geb.hydrology.routing
