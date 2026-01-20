# Agents

## Introduction

The agent module in GEB contains the classes and methods for the agent implementations used to simulate actors in the model.
Agents could represent individual households, cropfarmers, livestock farmers, industry, reservoir operators, governments, and markets

## General Decision Framework
GEB simulates adaptation behavior using on a decision framework based on the DYNAMO agent-based model[@tierolf2023coupled; @de2022usa; @haer2020safe].
In this model, households make bounded rational decisions based on subjective (time discounted) utility theory[@fishburn1981subjective]. This theory has been applied in various ABMs simulating population mobility and household adaptation to flooding and drought[@kalthof_2024_11071746; @bell2021migration; @de2022usa; @haer2020safe]. One key benefit of DEU is that it enables direct weighing of households’ adaptation options, while accounting for different risk perceptions and risk preference changes over time based on experiences with flooding and trust in authorities’ protection[@di2013socio; @di2015debates;@haer2017integrating;@wachinger2013risk].

In each time step representing 1 year, each household agent calculates and compares the DEU of the following:

- Doing nothing (Eq. 1)
- Implementing dry flood-proofing measures (Eq. 2)


The agent executes the strategy yielding the highest utility within its budget constraints. The formulas for calculating the DEU of each strategy are as follows:

$$
DEU_1 = \int_{p_i}^{p_I} \beta_t \, p_i \, U \left(
\frac{
\sum_{t=0}^{T} \left( W_x + A_x + Inc_x - D_{x,t,i} \right)
}{
(1 + r)^t
}
\right) \, dp
$$

$$
DEU_2 = \int_{p_i}^{p_I} \beta_t \, p_i \, U \left(
\frac{
\sum_{t=0}^{T} \left( W_x + A_x + Inc_x - D_{x,t,i}^{adapt} - C_t^{adapt} \right)
}{
(1 + r)^t
}
\right) \, dp
$$

Utility is a function of household wealth $W_x$, the amenity value of the current household location, $A_x$, current household income $I_x$, expected damage $D$ per event $i$, and adaptation costs $C^{adapt}$.

Discounting and risk aversion: A time discounting factor $r$ of 3.2% specific to France is applied over a time horizon $T$ of 15 years, representing the number of years a homeowner on average stays in his or her home. We assume a general utility function as a function of relative risk aversion. The model is run with slightly risk-averse households, for which the following function is used:

$$
U(x) = ln(x)
$$

To derive the utility of staying with and without implementing dry flood-proofing measures, we take the integral of the summed and time discounted utility under all possible events $i$. These events have a probability $p_i$ of (no) flooding (see section x for a description of the modeling of the return-period based flood maps). Each household is assigned a position in the income distribution as taken from a global synthetic population database [@ton2024global]. This procedure is further described under section X. Household income ($Inc$) for each household is sampled from a national lognormal distribution constructed using OECD income distribution data [@oecd_data_inequality]. We calculate the wealth $W$ of each agent at time $t$ using the income-to-wealth ratios per income quintile (???).




::: geb.agents
