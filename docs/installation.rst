Installation
#############

GEB runs on Python 3.11+ and can be installed using pip. In the future, GEB will be available on PyPI, but for now, you can install it from GitHub:

We recommend to install GEB in a conda environment. If you don't have conda installed, you can install it from https://docs.conda.io/en/latest/miniconda.html. Then create a conda environment with Python 3.11 and activate it:

.. code-block:: bash

    conda config --add channels conda-forge
    conda create -n geb python=3.11
    conda activate geb

Then install the required packages:

.. code-block:: bash

    conda config --set solver libmamba
    conda install rasterio numba tbb pandas 'geopandas>=0.14' numpy deap  pyyaml xarray 'dask>=2023.3.0' 'rioxarray>=0.15' pybind11 scipy netCDF4 flopy bmipy xmipy xlrd pyflow s3fs xesmf 'hydromt>=0.9.1' tqdm 'openpyxl>=3.1.2' xclim xesmf

Then install GEB from GitHub. You need to install the full version, which includes all dependencies:

.. code-block:: bash

    pip install geb[full]@git+https://github.com/jensdebruijn/GEB

Installation in development mode
--------------------------------

If you want to contribute to GEB, you can install it in development mode. This will install the package in editable mode, so that changes to the source code are immediately available. In that case, you need to clone the following repositories, and install in editable mode. Please note that some of these packages are private, so you need to have access to them. For that you can contact the authors.

.. code-block:: bash

    git clone https://github.com/jensdebruijn/ABCWatM_private
    pip install -e ABCWatM_private

    git clone https://github.com/jensdebruijn/hydromt_geb
    pip install -e hydromt_geb

Then install GEB in editable mode:

.. code-block:: bash

    git clone https://github.com/jensdebruijn/GEB_private
    pip install -e GEB_private