# Agents

## Introduction

The agent module in GEB contains the classes and methods for the agent implementations used to simulate actors in the model.
Agents could represent individual households, cropfarmers, livestock farmers, industry, reservoir operators, governments, and markets

## General Decision Framework
GEB simulates adaptation behavior using on a decision framework based on the DYNAMO agent-based model[@tierolf2023coupled; @de2022usa; @haer2020safe].
In this model, households make bounded rational decisions based on subjective (time discounted) utility theory[@fishburn1981subjective]. This theory has been applied in various ABMs simulating population mobility and household adaptation to flooding and drought[@kalthof_2024_11071746; @bell2021migration; @de2022usa; @haer2020safe]. One key benefit of calculating discounted expected utility (DEU) is that it enables direct weighing of households’ adaptation options, while accounting for different risk perceptions and risk preference changes over time based on experiences with flooding and trust in authorities’ protection[@di2013socio; @di2015debates;@haer2017integrating;@wachinger2013risk].

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

Calculating utility: To derive the utility of implementing dry flood-proofing measures or not, we take the integral of the summed and time discounted utility under all possible events $i$. These events have a probability $p_i$ of (no) flooding (see section x for a description of the modeling of the return-period based flood maps). Each household assesses their expected wealth state applying a time discounting factor $r$ of 3%[@tol2008climate] over a time horizon $T$ of 15 years, representing the number of years a homeowner on average stays in his or her home. Therefore each household is assigned an income based on their position in the income distribution as taken from a global synthetic population database [@ton2024global]. This household income ($Inc$) for is sampled from a national lognormal distribution constructed using OECD income distribution data [@oecd_data_inequality]. Each household is also assigned a wealth $W$ using income-to-wealth ratios per income quintile (#TODO). The household calculates the expected flood damages for their property $D$ based on depth damage curves as described in Huizinga et al.[@huizinga2017global]. Dry flood-proofing measures reduce flood damage by preventing floodwaters from entering a building. For this, we alter the depth-damage functions such that damage for inundation levels below 1 meter is reduced by 85% following Aerts et al.[@aerts2011flood]. Inundation above 1 meter overtops the dry flood proofing, resulting in full damage.


We assume a general utility function as a function of relative risk aversion. The model is run with slightly risk-averse households, for which the following function is used:

$$
U(x) = ln(x)
$$


Bounded rationality: Bounded rationality is captured by risk perception factor $β$. This perception factor results in both underestimations of flood hazard during periods of no flooding ($β$ < 1) and overestimations of flood hazard immediately after a flood event ($β$ > 1). We follow the DYNAMO setup by de Ruig et al.[@de2023agent] and Haer et al.[@haer2020safe] and define risk perception as a function of the number of years after the most recent flood event, shown here in the equation below:

$$
β_t = c * 1.6^{-d*t} + 0.01
$$

The function describes the evolution of risk perception factor $β$ over $t$ years after a flood event. The maximum overestimation of risk $c$ was calibrated on survey data on the implementation of dry flood-proofing measures[@poussin2013stimulating], see Tierolf et al.[@tierolf2023coupled].

Cost of flood proofing: In determining the costs of dry flood-proofing measures we use an average adaptation cost of 10,800 euros per building based on Hudson[@hudson2020affordability]. This cost includes installing pumps and water barriers. Annual payments for the dry flood-proofing measures ($C_{annual}^{building}$) are calculated using the formula presented below and depend on the dry flood-proofing cost per building ($C_{0}^{building}$), a fixed interest rate ($r$), and loan duration ($n$).

$$
C_{\text{annual}}^{\text{building}} = 
C_{0}^{\text{building}} * 
\frac{r(1+r)^n}{(1+r)^n - 1}
$$


Budget constraints: In defining the budget constraint for dry flood-proofing investments, we follow an expenditure cap definition of affordability by Hudson[@hudson2018comparison] and assume households can invest a fixed percentage of their disposable income. Hudson[@hudson2018comparison] distinguishes between investment affordability, entailing a household’s ability to pay for adaptation in a single upfront payment, and payment affordability, which applies when a series of annualized payments are made for a single measure. We apply a payment affordability definition and assume households obtain personal loans to finance dry flood proofing. If the annual loan payment exceeds the expenditure cap of the household, the household cannot afford to invest in dry flood proofing. We calibrate the loan duration, interest rate, and expenditure cap on the observed implementation rate of dry flood-proofing measures.

## GLOPOP-S
To initiate the household and farmer agent populations we make of the global synthetic population database GLOPOP-S[@ton2024global]. GLOPOS-S contains the following attributes: age, education, gender, income/wealth, settlement type (urban/rural), household size, household type, and for selected countries in the Global South, ownership of agricultural land and dwelling characteristics. GLOPOP-S is generated using microdata from the Luxembourg Income Study and Demographic and Health Surveys applying synthetic reconstruction techniques to fit national survey data to regional statistics, thereby accounting for spatial differences within and across countries. Additionally, it containes generated data for countries without available microdata. The data is further downscaled to grid cells with a resolution of approximately 1 kilometer. Both household and farmer agent attributes are initialized based on GLOPOP-S. Households use the location of their assigned building of residence to sample characteristics from the corresponding grid cell, whereas farmers use the location of their farm (see `setup_household_characteristics` and `setup_farmer_household_characteristics`).
The dataset can be downloaded per region or country, see the repository on GitHub[@GLOPOP-S_GitHub]. GLOPOP-S is open source and can be extended with other attributes. 
## Code

::: geb.agents
