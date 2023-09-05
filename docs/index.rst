GEB
######################

:Copyright: .. include:: copyright.rst
:Authors: .. include:: authors.rst
:Version: 0.2
:Version Date: |today|

Welcome to GEB's documentation! GEB stands for Geographic Environmental and Behavioural model and is named after Geb, the personification of Earth in Egyptian mythology.

GEB aims to simulate both environment, for now the hydrological system, the individual behaviour of people and their interactions at large scale. The model does so by coupling an agent-based model which simulates millions individual people or households and a hydrological model. While the model can be expanded to other agents and environmental interactions, we focus on farmers, high-level agents, irrigation behaviour and land management for now.

Building on the shoulders of giants
----------------------------------------
GEB builds on, couples and extends several models.
- The hydrological component of GEB is mainly build on the `CWatM model <https://cwatm.iiasa.ac.at/>`_.
- The agent-based component of GEB extends the `ADOPT model <https://vu-ivm.github.io/WCR-models-and-data/models/ADOPT/index.html>`_

The figure below shows a schematic overview of the model. The lower part of the figure shows the hydrological model, CWatM, while the upper part shows the agent-based model. Both models are imported in a single file and run iteratively at a daily timestep.

.. image:: images/schematic_overview.svg

.. toctree::
  :maxdepth: 1
  :caption: Getting Started

  Installation <installation>
  Updates <updates>
  Preprocessing <preprocessing>
  Configuration <configuration>
  Running the model <running>
  Farm-level HRUs <HRUs>

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

  Model <model>
  CWatM Model <cwatm_model>
  Report <report>
  Artists <artists>

.. toctree::
  :maxdepth: 2
  :caption: About

  Authors <authors_page>