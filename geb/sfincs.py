import os
from pathlib import Path
from collections import deque
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd

from sfincs_river_flood_simulator import build_sfincs, update_sfincs_model_forcing

SFINCS_EXE = os.path.abspath(r'../SFINCS/sfincs.exe')

class SFINCS:
    def __init__(self, model, config, n_timesteps=5):
        self.model = model
        self.config = config
        self.n_timesteps = n_timesteps
        self.data_folder = Path(os.environ.get('GEB_DATA_CATALOG')).parent / 'SFINCS'
        
        self.discharge_per_timestep = deque(maxlen=self.n_timesteps)

    def setup(self, basin_id, force_overwrite=False):
        config_fn = self.data_folder / 'sfincs_cli_build.yml'
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

    def run(self, basin_id):
        n_timesteps = min(self.n_timesteps, len(self.discharge_per_timestep))
        substeps = self.discharge_per_timestep[0].shape[0]
        discharge_grid = xr.DataArray(
            data=self.model.data.grid.decompress(np.vstack(self.discharge_per_timestep)),
            coords={
                'time': pd.date_range(
                    start=self.model.current_time,
                    periods=n_timesteps * substeps,
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
            event_name=f"{self.model.current_time.strftime('%Y%m%dT%H%M%S')}_{basin_id}",
            current_event={
                'tstart': self.to_sfincs_datetime(self.model.current_time - self.model.timestep_length * n_timesteps),
                'tend': self.to_sfincs_datetime(self.model.current_time)
            },
            discharge_grid=discharge_grid,
            model_root=self.data_folder / 'models' / str(basin_id),
            data_catalogs=[str(self.data_folder / 'global_data' / 'data_catalog.yml')],
        )
        return None
    
    def save_discharge(self):
        self.discharge_per_timestep.append(self.model.data.grid.discharge_substep)  # this is a deque, so it will automatically remove the oldest discharge

# class SFINCS:
#     def __init__(self, model, config: dict, bbox: list[float, float, float, float]):
#         self.model = model
#         # create sfincs model
#         subfolder = "_".join([str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])])
#         self.sfincs_folder = os.path.join(config['general']['report_folder'], 'sfincs', subfolder)
#         if hasattr(self, 'logger'):
#             self.logger = None

#         self.logger = logging.getLogger(__name__)
#         # set log level to debug
#         self.logger.setLevel(logging.DEBUG)
#         # set handler to print to console
#         self.logger.addHandler(logging.StreamHandler())
#         if not os.path.exists(self.sfincs_folder):
#             build_model = True
#         else:
#             build_model = False
#         self.mod = SfincsModel(root=self.sfincs_folder, data_libs=[
#             os.path.join(config['general']['original_data'], 'data_catalog.yml')
#         ], logger=self.logger, mode='w+' if build_model else 'r+')
#         if build_model:
#             self.mod.build(region={'bbox': bbox}, opt=configread(r'sfincs.ini'))

#     def plot_max_flood_depth(self, flood_depth, minimum_flood_depth=.2):
#         da_hmax_masked = flood_depth.where(flood_depth > minimum_flood_depth)
#         # update attributes for colorbar label later
#         da_hmax_masked.attrs.update(long_name="flood depth", unit="m")

#         # create hmax plot and save to mod.root/figs/hmax.png
#         fig, ax = self.mod.plot_basemap(
#             fn_out=None,
#             variable=None,
#             bmap="sat",
#             geoms=["src", "obs"],
#             plot_bounds=False,
#             figsize=(11, 7),
#         )
#         # plot overland flooding based on gswo mask and mimum flood depth
#         cbar_kwargs = {"shrink": 0.6, "anchor": (0, 0)}
#         cax_fld = da_hmax_masked.plot(
#             ax=ax, vmin=0, vmax=3.0, cmap=plt.cm.viridis, cbar_kwargs=cbar_kwargs
#         )

#         ax.set_title(f"SFINCS maximum water depth")
#         plt.show()
#         # plt.savefig(join(mod.root, 'figs', 'hmax.png'), dpi=225, bbox_inches="tight")

#     def plot_basemap(self):
#         self.mod.plot_basemap(
#             geoms=["src", "obs", "rivers"],
#             figsize=(14, 14 * 0.65),
#         )
#         plt.show()

#     def plot_forcing(self):
#         self.mod.plot_forcing()
#         plt.show()

#     def run(self, discharges: pd.DataFrame, lons: list, lats: list, plot: tuple=()):
#         timedelta = discharges.index[1] - discharges.index[0]
#         tstart = self.to_sfincs_datetime(discharges.index[0])
#         tend = self.to_sfincs_datetime(discharges.index[-1] + timedelta)
        
#         self.mod.set_config("tref", tstart)
#         self.mod.set_config("tstart", tstart)
#         self.mod.set_config("tstop", tend)
#         self.mod.write_config()

#         pnts = gpd.points_from_xy(lons, lats, crs=4326).to_crs(self.mod.crs)
#         src = gpd.GeoDataFrame(index=range(1, len(pnts) + 1), geometry=pnts)  # NOTE that the index should start at one
        
#         self.mod.set_forcing_1d(name="discharge", ts=discharges, xy=src)
#         self.mod.write_forcing()

#         if 'basemap' in plot:
#             self.plot_basemap()
#         if 'forcing' in plot:
#             self.plot_forcing()

#         cur_dir = os.getcwd()
#         os.chdir(self.sfincs_folder)
#         subprocess.run([SFINCS_EXE], check=True)
#         os.chdir(cur_dir)

#         # we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
#         self.mod.read_results()
#         # the following variables have been found
#         # print(list(mod.results.keys()))
#         # self.mod.write_raster("results.hmax", compress="LZW")

#         max_flood_depth = self.mod.results["hmax"]  # hmax is computed from zsmax - zb
#         if 'max_flood_depth' in plot:
#             self.plot_max_flood_depth(max_flood_depth)

#         crs = self.mod.crs
#         gt = self.mod.transform.to_gdal()
#         assert gt[5] < 0
#         if max_flood_depth.coords.variables['y'][0] < max_flood_depth.coords.variables['y'][-1]:
#             max_flood_depth = np.flipud(max_flood_depth.to_numpy())
#         else:
#             max_flood_depth = max_flood_depth.to_numpy()
#         max_flood_depth[max_flood_depth < 0] = np.nan
#         return max_flood_depth, crs, gt

#     def to_sfincs_datetime(self, dt: datetime):
#         return dt.strftime('%Y%m%d %H%M%S')