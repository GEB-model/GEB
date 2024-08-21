Installation
#############

GEB runs on Python 3.12+ and can be installed using pip.

.. code-block:: bash

    pip install geb-model

Hydromt-geb which is used for setting up the model domain and data preparation can be installed using:

.. code-block:: bash

    pip install hydromt-geb

Installation in development mode
--------------------------------

If you want to contribute to GEB, you can install it in development mode. This will install the package in editable mode, so that changes to the source code are immediately available. In that case, you need to clone the following repositories, and install in editable mode.

.. code-block:: bash

    git clone https://github.com/GEB-model/hydromt_geb
    pip install -e hydromt_geb

Then install GEB in editable mode:

.. code-block:: bash

    git clone https://github.com/GEB-model/GEB
    pip install -e GEB

Installing SFINCS (flood model)
--------------------------------

If you want to simulate floods in GEB, the flood model SFINCS is required. SFINCS runs in Docker, which needs to be installed. To install Docker you need to obtain and install Docker from their website (https://www.docker.com/get-started). The rest should work automatically.