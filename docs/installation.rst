Installation
#############

GEB runs on Python 3.11+ and can be installed using pip. In the future, GEB will be available on PyPI, but for now, you can install it from GitHub:

.. code-block:: bash

    pip install git+https://github.com/jensdebruijn/GEB

Installation in development mode
--------------------------------

If you want to contribute to GEB, you can install it in development mode. This will install the package in editable mode, so that changes to the source code are immediately available. In that case, you need to clone the following repositories, and install in editable mode. Please note that some of these packages are private, so you need to have access to them. For that you can contact the authors.

.. code-block:: bash

    git clone https://github.com/jensdebruijn/honeybees
    pip install -e honeybees

    git clone https://github.com/jensdebruijn/ABCWatM_private
    pip install -e ABCWatM_private

    git clone https://github.com/jensdebruijn/hydromt_geb
    pip install -e hydromt_geb

Then you can install GEB in editable mode, however, first make sure to comment out the following line in the pyproject.toml file to ensure that previously installed packages are not overruled.

    "honeybees@git+https://github.com/jensdebruijn/honeybees",
    "abcwatm@git+https://github.com/jensdebruijn/ABCWatM",
    "hydromt_geb@git+https://github.com/jensdebruijn/hydromt_geb",

to

    # "honeybees@git+https://github.com/jensdebruijn/honeybees",
    # "abcwatm@git+https://github.com/jensdebruijn/ABCWatM",
    # "hydromt_geb@git+https://github.com/jensdebruijn/hydromt_geb",

Then install GEB in editable mode:

.. code-block:: bash

    git clone https://github.com/jensdebruijn/GEB_private
    pip install -e GEB_private