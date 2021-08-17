GEB
######################

:Copyright: .. include:: copyright.rst
:Authors: .. include:: authors.rst
:Version: 0.01
:Version Date: |today|

Welcome to GEB's documentation! GEB stands for Geographic Environmental and Behavioural model and is named after Geb, the personification of Earth in Egyptian mythology.

GEB aims to simulate both environment, for now the hydrological system, the behaviour of people and their interactions at large scale without sacrificing too much detail. The model does so by coupling an agent-based model which simulates millions individual people or households and a hydrological model. While the model can be expanded to other agents and environmental interactions, we focus on farmers, high-level agents, irrigation behaviour and land management for now.

The figure below shows a schematic overview of the model. The lower part of the figure shows the hydrological model, CWatM, while the upper part shows the agent-based model. Both models are imported in a single file and run iteratively at a daily timestep.

.. image:: images/schematic_overview.svg

.. toctree::
  :maxdepth: 1
  :caption: Getting Started

  Installation <installation>
  Preprocessing <preprocessing/preprocessing>
  Configuration <configuration>
  Running the model <running>

.. toctree::
  :maxdepth: 1
  :caption: Agents
  
  Agents <agents/__init__>
  Farmers <agents/farmers>
  NGO <agents/ngo>
  Government <agents/government>

.. toctree::
  :maxdepth: 2
  :caption: Reference

  Land units <landunits>
  Model <model>
  CWatM Model <cwatm_model>
  Report <report>
  Artists <artists>

.. toctree::
  :maxdepth: 2
  :caption: About

  Authors <authors_page>