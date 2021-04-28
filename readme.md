## Description

Development version of code for inversion of atomic and atmospheric parameters
simultaneously.

## Installation

Install package in editable mode:

'pip3 install -e /path/to/package'

## Synthesis

For simple synthesis we requiere two parameters: atmosphere and number of
threads for computations to use. Also, we are requiered and to write
wavelength grid data on which we want to synthesise spectrum (old).

## Inversion

In inversion case we need observations, nodes and values of parameters in
nodes. Wavelength grid is determined from observations (we assume that
wavelength grid is stored with observations; old).

## Input files

First of all, we use same input as RH code since the forward modeling is based
on RH code. For inversion we make use of 'params.input' file and here give
description of it.

File is structured in following sections:

	1. MCMC parameters
	2. fitting parameters
	3. input data files (spectra, atmosphere)

(old)

## ToDo

* add stop criterion in inversion based on Chi2 values
* add inversion for OF parameters
* make test folder:
	* for synthesis
	* inversion of atmospheric parameters
	* inversion of atmospheric + OF parameters
* read atomic parameter line list
* extend Atmosphere() to hold and AtomicLine() class
* calculate RF for atomic parameters (log(gf) and wavelength)
* invert for one line for log(gf)

## Rewritings

Rewrite output routines from RH of output spectrum? To write out only spectrum
at highest mu?

## RH changes

1. Fixed problem with SOLVE_NE in ITERATION mode (check mail to Sowmya where it is explained what is changed)
2. Hydrostatic() is done even if H atom is not in ACTIVE state (iterate.c)

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