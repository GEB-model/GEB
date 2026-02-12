# Runoff concentration

The runoff concentration module slows down discharge over several timesteps (i.e. hours). We do this because the runoff has to travel through the grid cell and cannot leave in the same timestep in which it was generated. We use a triangular weighting method to redistribute the discharge over 6 timesteps (i.e. hours), with the peak discharge being at 3 timesteps (i.e. hours). Only runoff is changed using this approach, the interflow and baseflow remain unchanged. This approach is based on the original CWatM model. The figure below shows an example of how the input runoff is redistributed using the runoff concentration process.

<img width="1920" height="1440" alt="runoff_concentration" src="https://github.com/user-attachments/assets/486ef5d7-c9e6-4ddb-8905-5f750ed366a4" />

## Code

::: geb.hydrology.runoff_concentration
