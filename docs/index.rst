GEB
######################

:Copyright: .. include:: copyright.rst
:Authors: .. include:: authors.rst
:Version: |release|
:Version Date: |today|

Welcome to GEB's documentation! GEB stands for Geographic Environmental and Behavioural model and is named after Geb, the personification of Earth in Egyptian mythology.

GEB aims to simulate both environment, the individual behaviour of people and their interactions at small and large scale. The model does so by coupling an agent-based model which simulates millions individual people or households, a hydrological model, a vegetation model and a hydrodynamic model.

Building on the shoulders of giants
----------------------------------------
GEB builds on, couples and extends several models, depected in the figure below.

.. image:: images/models_overview.svg

The figure below shows a schematic overview of some parts of the model, showing the hydrological model and agent-based model. Both models are imported in a single file and run iteratively at a daily timestep.

.. image:: images/schematic_overview.svg

.. toctree::
  :maxdepth: 1
  :caption: Getting Started

  Installation <installation>
  Configuration <configuration>
  Preprocessing <preprocessing>
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

  Model <model>
  CWatM Model <cwatm_model>
  Report <report>
  Farm-level HRUs <HRUs>
  Artists <artists>

.. toctree::
  :maxdepth: 2
  :caption: About

  Authors <authors_page>
  Updates <updates>