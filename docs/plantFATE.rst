Download Packages
===============

```bash

git clone git@github.com:jaideep777/Plant-FATE.git
git switch feature_python_pckg  # switch to the feature branch
cd Plant-FATE

```

Install optional packages from plantfate extras in GEB:

```bash

uv sync --extra plantfate

```


Compiling PlantFATE
=====================

```bash

module load shared 2024 Eigen/3.4.0-GCCcore-12.3.0 GSL/2.7-GCC-12.3.0
cd Plant-FATE
make all
uv pip install ../Plant-FATE/

```