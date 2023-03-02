import logging
import os
import subprocess
import shutil
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from hydromt_sfincs import SfincsModel, utils
from hydromt.config import configread

SFINCS_EXE = r'../SFINCS/sfincs.exe'
# get absolute path for sfincs executable
SFINCS_EXE = os.path.abspath(SFINCS_EXE)

class SFINCS:

    def __init__(self, config, discharges, remove_existing=True):
        # create sfincs model
        sfincs_folder = os.path.join(config['general']['report_folder'], 'sfincs')
        if remove_existing and os.path.exists(sfincs_folder):
            shutil.rmtree(sfincs_folder)
        assert not os.path.exists(sfincs_folder)
        logger = logging.getLogger(__name__)
        # set log level to debug
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )

        # read model config and build model
        mod = SfincsModel(root=sfincs_folder, data_libs=None, logger=logger, mode='w')
        mod.read_config(config_fn=r'sfincs.ini')
        mod.build(region={'bbox': [11.97,45.78,12.28,45.94]}, opt=configread(r'sfincs.ini'))

        tstart = self.to_sfincs_datetime(discharges.index[0])
        tend = self.to_sfincs_datetime(discharges.index[-1])
        mod.set_config("tref", tstart)
        mod.set_config("tstart", tstart)
        mod.set_config("tstop", tend)
        
        x = [264891.02]
        y = [5083000.61]
        pnts = gpd.points_from_xy(x, y)
        index = [1]  # NOTE that the index should start at one
        src = gpd.GeoDataFrame(index=index, geometry=pnts, crs=mod.crs)
        
        mod.set_forcing_1d(name="discharge", ts=discharges, xy=src)
        mod.forcing["dis"]

        mod.write()

        mod = SfincsModel(root=sfincs_folder, data_libs=None, logger=logger, mode='r')
        mod.read()

        cur_dir = os.getcwd()
        os.chdir(sfincs_folder)
        subprocess.run([SFINCS_EXE], check=True)
        os.chdir(cur_dir)

        # mod.plot_basemap(
        #     geoms=["src", "obs", "rivers"],
        #     figsize=(14, 14 * 0.65),
        # )
        # plt.show()
        # mod.plot_forcing()
        # plt.show()

        mod = SfincsModel(sfincs_folder, mode="r")
        # we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
        mod.read_results()
        # the following variables have been found
        # print(list(mod.results.keys()))
        mod.write_raster("results.hmax", compress="LZW")

        hmin = .2 # minimum flood depth in meters

        da_hmax = mod.results["hmax"]  # hmax is computed from zsmax - zb
        da_hmax_masked = da_hmax.where(da_hmax > hmin)
        # update attributes for colorbar label later
        da_hmax_masked.attrs.update(long_name="flood depth", unit="m")

        # create hmax plot and save to mod.root/figs/hmax.png
        fig, ax = mod.plot_basemap(
            fn_out=None,
            variable=None,
            bmap="sat",
            geoms=["src", "obs"],
            plot_bounds=False,
            figsize=(11, 7),
        )
        # plot overland flooding based on gswo mask and mimum flood depth
        cbar_kwargs = {"shrink": 0.6, "anchor": (0, 0)}
        cax_fld = da_hmax_masked.plot(
            ax=ax, vmin=0, vmax=3.0, cmap=plt.cm.viridis, cbar_kwargs=cbar_kwargs
        )

        ax.set_title(f"SFINCS maximum water depth")
        plt.show()
        # plt.savefig(join(mod.root, 'figs', 'hmax.png'), dpi=225, bbox_inches="tight")

    def to_sfincs_datetime(self, datetime):
        return datetime.strftime('%Y%m%d %H%M%S')


if __name__ == '__main__':
    from config import config
    sfincs = SFINCS(config)