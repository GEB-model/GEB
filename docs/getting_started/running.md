# Running the model

The GEB model is run in two steps:

* **spinup**: In the beginning of the spinup we use somewhat arbitrary starting conditions such as lake levels, risk perception and soil moisture conditions. The spinup period is then used to transition the model from these arbitrary initial conditions to a dynamic equilibrium consistent with internal physics, socio-economic processes and boundary conditions. All data that was created during the spinup period is discarded and not part of the results. To find the dynamic equilibrium for all processes we recommend a minimum spinup period of 30 years.
* **run**: The run then starts immediately after the spinup, loading the state of the model at the end of the spinup. Data during the run can be saved and is part of the model results.

Assuming the model is build (i.e., all input data is obtained, downloaded and processed), and all default configuration files are in place (see examples), you run the model using the following commands:

```bash
geb spinup
```

and 

```bash
geb run
```

This runs the model in the current working directory. To run the model using files that are in another directory, you can use

```bash
geb run -wd path/to/other/directory
```

More options, such as running custom configuration files, running in optimized mode (turning off all asserts) are available. To find out more about these options use:

```bash
geb spinup --help
```