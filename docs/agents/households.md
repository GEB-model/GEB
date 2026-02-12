# Households

# Flood damage model 

Currently, flood damage calculations are performed in the households script. When a flood is simulated using the SFINCS model, the damage is automatically calculated. We use exposure data from several sources. Buildings are taken from the OpenBuildingMap dataset. Roads and railways are taken from OpenStreetMap. Agriculture and forest areas are taken from the ESA WorldCover database. The exposure data is combined with vulnerability curves to calculate the damage per object. The vulnerability curve shows the relationship between inundation depth and the damage factor (i.e. the fraction of maximum damage for a certain water depth). For buildings, we use the curves derived by [^endendijk2023flood]. For roads, we use the curves derived by [^van2021flood]. For railways, we use the curves developed by [^kellermann2015estimating]. For nature and agriculture, we use the curves developed by [^de2014evaluating]. The maximum damages are also taken from these sources. The figure below shows all curves and maximum damages used in the damage model. Damages are reported for every exposure category individually and as a combined total damage value. 

<img width="1280" height="720" alt="Vulnerability Curves and their corresponding maximum damages" src="https://github.com/user-attachments/assets/3a4dbc03-3a45-49c6-b1f6-028eac385d7d" />

## Code

::: geb.agents.households
