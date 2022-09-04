## Description

Development version of code for inversion of atomic and atmospheric parameters
simultaneously. We employ the global inversion algorithm where atomic parameters are coupled between different pixels providing the unifrom value across the field-of-view. Atmospheric parameters are retrieved independently for each pixel.

## Installation

Install package in editable mode system-wide:

'pip3 install -e /path/to/package'

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

* add covolution with PSF profile
* add weights for each Stokes that are pixel dependent ?
* temperature initialization based on relative continuum intensity between pixels ?

* make unit test
* extend Atmosphere() to hold and AtomicLine() class (?)

## Rewritings


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