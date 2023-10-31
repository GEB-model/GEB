import os
from pathlib import Path
from collections import deque
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd

from sfincs_river_flood_simulator import build_sfincs, update_sfincs_model_forcing, run_sfincs_simulation, read_flood_map

class SFINCS:
    def __init__(self, model, config, n_timesteps=1):
        self.model = model
        self.config = config
        self.n_timesteps = n_timesteps
        self.data_folder = Path(os.environ.get('GEB_DATA_CATALOG')).parent / 'SFINCS'
        
        self.discharge_per_timestep = deque(maxlen=self.n_timesteps)

    def setup(self, basin_id, force_overwrite=False):
        config_fn = self.data_folder / 'sfincs_cli_build.yml'
        # force_overwrite = True
        if force_overwrite or not os.path.exists(self.data_folder / 'models' / str(basin_id) / 'sfincs.inp'):
            build_sfincs(
                basin_id=basin_id,
                config_fn=str(config_fn),
                basins_fn=str(self.data_folder / 'basins.gpkg'),
                model_root=self.data_folder / 'models' / str(basin_id),
                data_dir=self.data_folder,
                data_catalogs=[str(self.data_folder / 'global_data' / 'data_catalog.yml')],
                mask=self.model.area.geoms['region']
            )

    def to_sfincs_datetime(self, dt: datetime):
        return dt.strftime('%Y%m%d %H%M%S')

    def set_forcing(self, basin_id, event_name):
        n_timesteps = min(self.n_timesteps, len(self.discharge_per_timestep))
        substeps = self.discharge_per_timestep[0].shape[0]
        discharge_grid = self.model.data.grid.decompress(np.vstack(self.discharge_per_timestep))
        
        # when SFINCS starts with high values, this leads to numerical instabilities. Therefore, we first start with very low discharge and then build up slowly to timestep 0
        # TODO: Check if this is a right approach
        discharge_grid = np.vstack([np.full_like(discharge_grid[:substeps,:,:], fill_value=np.nan), discharge_grid])  # prepend zeros
        for i in range(substeps - 1, -1, -1):
            discharge_grid[i] = discharge_grid[i+1] * 0.9
        
        # convert the discharge grid to an xarray DataArray
        discharge_grid = xr.DataArray(
            data=discharge_grid,
            coords={
                'time': pd.date_range(
                    start=self.model.current_time,
                    periods=(n_timesteps + 1) * substeps,  # +1 because we prepend the discharge
                    freq=self.model.timestep_length / substeps, inclusive='left'
                ),
                'y': self.model.data.grid.lat,
                'x': self.model.data.grid.lon,
            },
            dims=['time', 'y', 'x'],
            name='discharge',
        )
        discharge_grid = xr.Dataset({'discharge': discharge_grid})
        discharge_grid.raster.set_crs(self.model.data.grid.crs)
        update_sfincs_model_forcing(
            event_name=event_name,
            current_event={
                'tstart': self.to_sfincs_datetime(self.model.current_time - self.model.timestep_length * n_timesteps),
                'tend': self.to_sfincs_datetime(self.model.current_time)
            },
            discharge_grid=discharge_grid,
            model_root=self.data_folder / 'models' / str(basin_id),
            data_catalogs=[str(self.data_folder / 'global_data' / 'data_catalog.yml')],
        )
        return None

    def run(self, basin_id):
        event_name = f"{self.model.current_time.strftime('%Y%m%dT%H%M%S')}_{basin_id}"
        self.set_forcing(basin_id, event_name)
        self.model.logger.info(f"Running SFINCS for {self.model.current_time}...")
        simulation_root = self.data_folder / 'models' / str(basin_id) / 'simulations' / event_name
        run_sfincs_simulation(
            simulation_root=simulation_root,
        )
        flood_map = read_flood_map(
            model_root=self.data_folder / 'models' / str(basin_id),
            simulation_root=simulation_root,
        )  # xc, yc is for x and y in rotated grid
        self.flood(flood_map)

    def flood(self, flood_map):
        self.model.agents.households.flood(flood_map)
    
    def save_discharge(self):
        self.discharge_per_timestep.append(self.model.data.grid.discharge_substep)  # this is a deque, so it will automatically remove the oldest discharge