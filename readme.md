## Description

Development version of code for inversion of atomic and atmospheric parameters
simultaneously. We employ the global inversion algorithm where atomic parameters are coupled between different pixels providing the unifrom value across the field-of-view. Atmospheric parameters are retrieved independently for each pixel.

## Installation

Create a new conda virtual environment using the Python3.10.

Due to some required packages (`emcee`) we need to add the `conda-forge` in the package search channels:

`conda config --add channels conda-forge`

Now, install all the required packages:

`conda install --file requirements.txt`

To install the package in the editable mode system-wide type:

`pip3 install -e /path/to/package`

Now you should have functional `globin` module. To test if everything is right, go to the `globin/tests` directory and type:

`python run.py`

which will start an inversion of a test sample.

To use the `globin` for forward modelling (and inversions), it relies on the cythonized version of RH called `pyrh` that is found [here](https://github.com/dvukadinovic/pyrh#).

## Synthesis

For simple synthesis we requiere two parameters: atmosphere and number of
threads for computations to use. Also, we are requiered and to write
wavelength grid data on which we want to synthesise spectrum.

## Inversion

In inversion case we need observations, nodes and values of parameters in
nodes. Wavelength grid is determined from observations (we assume that
wavelength grid is stored with observations; old).

## Input files

First of all, we use same input as RH code since the forward modeling is based
on RH code. For inversion we make use of 'params.input' file and here give
description of it.

## ToDo

* temperature initialization based on relative continuum intensity between pixels ?
* add covolution with PSF profile
* add weights for each Stokes that are pixel dependent ?

* make unit test
* extend Atmosphere() to hold and AtomicLine() class (?)

## Rewritings

* RH keywords commanded through Python env

## Comments

Higher noise will lower the impact of lower signals (like weak polarization ones). Can we
utilize this during inversion runs putting more constraints to Stokes I (temperature, v_LOS)
and later we lower tha noise and let the code see more polarization and adjust magnetic field 
vector. Hm?

## RH changes

1. Fixed problem with SOLVE_NE in ITERATION mode (check mail to Sowmya where it is explained what is changed).
2. HSE is done consistently inside RH (separate routine call)

## Requierments

subprocess>=
multiprocessing>=
astropy>=
os>=
sys>=
time>=
copy>=
numpy>=
matplotlib>=
time>=
rh == io, xdrlib
