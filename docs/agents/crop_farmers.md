# Crop farmers

Crop farmers represent individual crop-farming agents in GEB. They link agricultural production and farmer behaviour directly to the hydrological model. Each farmer owns one or more fields, follows a crop calendar, may irrigate crops, receives income from harvests, experiences drought impacts, and can adapt over time.

The `crop_farmers` module manages both the **physical farming cycle** (planting, crop growth, irrigation and harvesting) and **farmer behaviour** (income, drought experience, finances and adaptation decisions). This creates two-way feedbacks between farmers and hydrology. For example, drought can make a groundwater well more attractive, while widespread well adoption can lower groundwater levels and change the profitability of wells for neighbouring farmers.

The figure below illustrates this feedback: hydrological conditions affect crop production, income and drought experience; these outcomes influence risk perception and adaptation; and changes in crops, insurance and irrigation feed back into the water system.

![Example of two-way feedbacks between hydrology and farmer adaptation in GEB.](../images/adaptation_flow_chart.png)

*Figure: Example of coupled feedbacks between climate and hydrology, crop production, drought experience, farmer characteristics and adaptation decisions in GEB. Source: Kalthof et al. (2026), [*Beyond profit: Modelling the socio-hydrological impacts of agricultural insurance and drought adaptation*](https://doi.org/10.1016/j.ijdrr.2026.106159).*

## Farmer representation

Each farmer owns one or more fields represented as Hydrological Response Units (HRUs). Physical variables such as soil moisture, evapotranspiration and irrigation demand are calculated at field level, while income, risk perception and adaptation are generally tracked per farmer. The module therefore maintains efficient mappings between farmers and their fields.

Farmers can differ in location, crop rotation, irrigation access and technology, risk aversion, discount rate, drought risk perception, adaptations, loans and other financial commitments. These characteristics are initialized during model setup or spin-up and are updated as the simulation progresses.

## Crop production cycle

Crop production follows farmer-specific crop calendars, including multi-year rotations. The module checks daily whether crops should be planted or harvested. Crop switching changes future planting through an updated crop calendar rather than replacing crops that are already growing.

During the growing season, the hydrological model tracks actual and potential evapotranspiration. At harvest, water stress is translated into a **yield ratio**: achieved yield relative to potential yield without water limitation. Crop prices then convert production into actual and potential income.

Harvest also updates the farmer's experience. Yield, income and drought conditions during the growing season are stored and later used to estimate drought risk and the benefits of adaptation.

## Irrigation and water abstraction

Irrigation responds dynamically to crop water status and water availability. Farmers begin irrigating when soil moisture falls below a crop-specific trigger between the critical soil-moisture level and field capacity. **Gross irrigation demand** is based on the root-zone deficit to field capacity and is limited by soil infiltration capacity.

Water can be abstracted from reservoirs, channels or other surface water, and groundwater. Actual abstraction may be lower than demand because each source has its own constraints. Reservoir abstraction depends on releases and command-area access, channel abstraction on local surface-water storage, and groundwater abstraction on well access, groundwater conditions and pumping capacity. Optional yearly irrigation limits can further restrict abstraction.

When water is abstracted, it is removed directly from the corresponding hydrological storage and distributed over the farmer's irrigated fields. Irrigation efficiency determines how much abstracted water reaches the soil, while the **return fraction** determines how much of the remaining water returns to the hydrological system instead of being lost through evaporation.

Channel, reservoir and groundwater withdrawals are accumulated separately for each farmer. These histories are used for reporting, water-cost calculations and adaptation decisions. The distinction between withdrawals, consumption and return flows is also important for representing basin-scale feedbacks such as the irrigation efficiency paradox ([Kalthof et al., 2026](https://doi.org/10.1088/1748-9326/ae7fa0)).

## Drought experience, learning and adaptation

### Drought experience and risk perception

Farmers keep a history of precipitation, SPEI, yields and income. Drought experience affects **subjective risk perception**: shortly after a drought, farmers tend to place more weight on drought risk, while this perceived risk declines during longer periods without drought.

This creates the behavioural feedback:

**drought → yield and income losses → changed risk perception → adaptation decisions**

Financial losses can also affect later decisions through loans and other financial commitments.

### Estimating adaptation benefits

GEB estimates adaptation benefits from the simulated history of farmers rather than using a fixed drought-damage curve. After harvest, yield ratios are linked to SPEI conditions during the growing season. A location-specific Generalized Extreme Value (GEV) distribution converts SPEI extremes into annual drought probabilities, allowing yields and income to be estimated for drought events with different return periods.

The resulting drought-probability–yield relationship is the farmer's **objective drought risk experience**. It describes expected physical outcomes under drought and is kept separate from **subjective risk perception**, which determines how strongly the farmer weights those drought probabilities in the decision.

Individual farmers often have too few observations to estimate this relationship robustly. GEB therefore pools information across comparable **farmer groups**, while keeping every farmer as an individual agent:

1. **Base groups** represent similar production systems, primarily using crop calendars or crop rotations.
2. **Meta groups** add characteristics relevant to a specific comparison, such as irrigation access or source, precipitation class, irrigation-efficiency group, location or elevation class, insurance status, or prior adaptations.
3. Historical yield ratios and drought probabilities are averaged within groups before fitting the drought-yield relationship, which is then assigned back to the individual farmers.

Meta groups are also used for counterfactual comparisons. To estimate the benefit of an adaptation, farmers are grouped by relevant characteristics **except the adaptation being tested**. For example, farmers with and without wells can be compared within otherwise similar groups. For crop switching, farmers with similar environmental and irrigation conditions but different crop rotations provide the alternative outcomes.

Conceptually:

**past droughts and yields → objective drought risk experience → expected yield and income under different drought probabilities → comparison of adaptation options**

See [Kalthof et al. (2026)](https://doi.org/10.1016/j.ijdrr.2026.106159) for an application of this meta-group approach.

### Adaptation decisions

Longer-term adaptation is generally evaluated at the start of a new hydrological year. Depending on the configuration, farmers can install or renew a well, change irrigation technology, switch crop rotations, or take agricultural insurance.

Decisions follow GEB's **Subjective Expected Utility Theory (SEUT)** framework. Farmers compare the expected utility of their current strategy with available alternatives across drought events of different probabilities and a no-drought situation. Objective drought-risk experience provides expected income under each event, while subjective risk perception modifies how strongly those events are weighted.

The comparison can also include risk aversion, discounting, adaptation and maintenance costs, existing loans, budget constraints and physical feasibility. An adaptation is therefore adopted only when it provides higher subjective expected utility and satisfies the relevant constraints.

Because adaptation benefits are estimated from simulated farmer histories and peer groups, interactions between adaptations can emerge endogenously. A well, insurance policy, irrigation technology or crop change can alter future income, hydrology and the profitability of other adaptations.

### Crop switching

Crop switching changes the farmer's crop rotation to an alternative already represented by comparable farmers. Candidate rotations are compared within meta groups that keep environmental and management characteristics similar while allowing crop rotation to vary.

For each candidate, GEB estimates expected profits across drought scenarios, including cultivation costs, and compares its SEUT with the current rotation. A configurable switching cap can limit annual uptake. The implementation also keeps at least one representative of a crop rotation within a meta group so an alternative does not disappear permanently from future comparisons.

### Irrigation adaptations

Farmers can invest in groundwater wells or more efficient irrigation technologies. Well profitability depends on expected income gains, investment and pumping costs, and groundwater conditions; wells must also remain deep enough to reach groundwater.

Irrigation technology is represented using two parameters: **irrigation efficiency**, the fraction of abstracted water effectively delivered for crop use, and **return fraction**, the fraction of non-applied water that returns to the hydrological system. This separation allows GEB to represent cases where field-scale efficiency gains reduce return flows or lead to behavioural rebound rather than equivalent basin-scale water savings.

### Agricultural insurance

Insurance is evaluated as another drought-risk adaptation. Premiums and insured outcomes are supplied by insurer agents, while the farmer decides whether insurance increases expected utility. Insurance can therefore change financial vulnerability and interact with other adaptations such as wells and crop switching.

## Annual model cycle

A simplified farmer cycle is:

1. **Growing season:** crops grow, water stress is tracked and irrigation water is abstracted where needed.
2. **Harvest:** yields and income are calculated and drought experience is stored.
3. **Hydrological-year boundary:** yearly drought, income, water-use and cost histories are updated.
4. **Adaptation and new crop cycle:** farmers evaluate adaptations, crop rotations advance and subsequent planting follows the updated strategy.

Planting and harvesting remain crop-calendar specific, while longer-term behavioural updates occur at the annual boundary.

## Computational design

GEB can simulate very large farmer populations, so `crop_farmers` is designed to minimize Python-level overhead. Farmer attributes are stored mainly in numerical arrays rather than separate Python objects, and arrays are preallocated where possible to avoid repeated memory allocation.

Mappings between farmers and fields are stored as indices so the model does not need to repeatedly search the spatial grid. Many operations are vectorized with NumPy, while computationally intensive loops are compiled with Numba.

The module also contains consistency and water-balance checks, for example to verify that irrigation withdrawals match changes in hydrological storage and that withdrawn water is correctly divided between consumption and return flows.

## Interaction with other GEB modules

`crop_farmers` coordinates information from several parts of GEB. The hydrological and land-surface models provide water availability, soil conditions and crop evapotranspiration; reservoir operators determine reservoir water availability; insurer agents provide insurance products; crop prices and cultivation costs determine economic outcomes; and the decision module calculates expected utilities.

The farmer module combines this information at the individual-agent level and feeds farmer decisions back into the physical system.

## References

Kalthof, M. W. M. L., di Baldassarre, G., Aerts, J. C. J. H., de Moel, H., & de Bruijn, J. (2026). Beyond profit: Modelling the socio-hydrological impacts of agricultural insurance and drought adaptation. *International Journal of Disaster Risk Reduction, 141*, 106159. [https://doi.org/10.1016/j.ijdrr.2026.106159](https://doi.org/10.1016/j.ijdrr.2026.106159)

Kalthof, M. W. M. L., de Moel, H., Aerts, J. C. J. H., & de Bruijn, J. (2026). Quantifying the irrigation efficiency paradox. *Environmental Research Letters, 21*, 134021. [https://doi.org/10.1088/1748-9326/ae7fa0](https://doi.org/10.1088/1748-9326/ae7fa0)

## Code

::: geb.agents.crop_farmers
