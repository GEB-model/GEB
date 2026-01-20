# Agents

## Introduction

The agent module in GEB contains the classes and methods for the agent implementations used to simulate actors in the model.
Agents could represent individual households, cropfarmers, livestock farmers, industry, reservoir operators, governments, and markets

## General Decision Framework
GEB simulates adaptation behavior using on a decision framework based on the DYNAMO agent-based model[@tierolf2023coupled; @de2022usa; @haer2020safe].
In this model, households make bounded rational decisions based on subjective (time discounted) utility theory[@fishburn1981subjective]. This theory has been applied in various ABMs simulating population mobility and household adaptation to flooding and drought[@kalthof_2024_11071746; @bell2021migration; @de2022usa; @haer2020safe]. One key benefit of DEU is that it enables direct weighing of households’ adaptation options, while accounting for different risk perceptions and risk preference changes over time based on experiences with flooding and trust in authorities’ protection[@di2013socio; @di2015debates;@haer2017integrating;@wachinger2013risk].


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



::: geb.agents
