# Modules

## Introduction

GEB is build from a collection of modules following a specified template in a Python class (using an Abstract Base Class) with the following structure.

Methods:

- The \_\_init\_\_ method contains the parts of the module that are always run when the module is loaded. The first argument to \_\_init\_\_ is always the GEBModel itself (model). Additional parameters can be included as well, but this depends on the specific module.
- The spinup method is run only when the model is run in spinup mode. Here, we create the state variables of the model on the var-variable. These state variables should persist across different timesteps, and are restored during a normal run. See [here](variables.md) for more details.
- The step method is run every day for each module. This method is used to implement the dynamic logic in the model, usually updating some of the state variables, but logic can be more complicated depending on the type of module.

## Code

::: geb.module
