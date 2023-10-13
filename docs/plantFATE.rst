1) DOWNLOAD PACKAGES

So PlantFATE runs with the use of 2 other packages and with the coupling package you need to checkout 4 different github repositories. 

The packages are:

1) Coupling package: https://github.com/jensdebruijn/couple-plantFATE-CWatM (branch: develop)
2) PlantFATE package: https://github.com/jaideep777/Plant-FATE (branch: feature_stepByStepSim)
3) libpspm package: https://github.com/jaideep777/libpspm (branch: develop)
4) Phydro package: https://github.com/jaideep777/phydro (branch: master)

For ease of running and set up put these in one big project root folder (not necessary for the coupling package):

  root
  |---- phydro
  |     |--- inst/include
  |
  |---- libpspm
  |     |--- include
  |     |--- lib
  |
  |---- Plant-FATE
  |     |--- inst/include

2) INSTALL PLANTFATE and PYBIND

- module load cm-eigen3/3.3.7
- module load shared
- module load 2022
- module load GSL/2.7-GCC-11.3.0



As per instructions in PlantFATE readme, install natively in C++.

You may need to install additional libraries, specifically pybind: 

pip install pybind11
pip install cppimport
libpspm
cd libpspm
make clean testclean
make
make check
Phydro
cd phydro
make clean testclean
make check
PlantFATE
cd Plant-FATE
make clean testclean
make python

3) RUN PLANTFATE COUPLING FROM PYTHON
You can run the PLANTFATE model using the coupling.py file. 
The crucial elements to have are 2 files: parameters file (p_daily.ini) and plant traits data (Amz_trait_filled_HD.csv). 
The parameter file is in the repository but I am not sure about the plant trait data yet so it’s attached to this email instead (simply download and put in a data/Amz_trait_filled_HD.csv in the repository).
Then you can run coupling.py. The code for running plantFATE is in the __main__() function and I’ve tried to comment it. 
