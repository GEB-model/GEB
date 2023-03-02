import logging
import os
import subprocess
import contextlib

import matplotlib.pyplot as plt

from hydromt_sfincs import SfincsModel
from hydromt.config import configread

from config import config

SFINCS_EXE = r'../SFINCS/sfincs.exe'
# get absolute path for sfincs executable
SFINCS_EXE = os.path.abspath(SFINCS_EXE)
sfincs_folder = os.path.join(config['general']['report_folder'], 'sfincs')


logger = logging.getLogger(__name__)
# set log level to debug
logger.setLevel(logging.DEBUG)

# read the model with hydromt sfincs methods
mod = SfincsModel(root=sfincs_folder, data_libs=None, logger=logger, mode='r')
opt = configread(r'sfincs.ini')  # parse .ini configuration

if not os.path.exists(sfincs_folder):
    mod.build(region={'bbox': [11.97,45.78,12.28,45.94]}, opt=opt)

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
print(list(mod.results.keys()))
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