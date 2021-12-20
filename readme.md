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

* add inversion for OF parameters
* make unit test
* extend Atmosphere() to hold and AtomicLine() class (?)

## Rewritings

Rewrite output routines from RH of output spectrum? To write out only spectrum
at highest mu?

## RH changes

1. Fixed problem with SOLVE_NE in ITERATION mode (check mail to Sowmya where it is explained what is changed).
2. Hydrostatic() is done even if H atom is not in ACTIVE state (iterate.c) --> nope, changes have been reverted.

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