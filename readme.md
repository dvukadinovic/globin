## Description

Development version of code for inversion of atomic and atmospheric parameters
simultaneously.

## Synthesis

For simple synthesis we requiere two parameters: atmosphere and number of
threads for computations to use. Also, we are requiered and to write
wavelength grid data on which we want to synthesise spectrum.

## Inversion

In inversion case we need observations, nodes and values of parameters in
nodes. Wavelength grid is determined from observations (we assume that
wavelength grid is stored with observations).

## Input files

First of all, we use same input as RH code since the forward modeling is based
on RH code. For inversion we make use of 'params.input' file and here give
description of it.

File is structured in following sections:

	1. MCMC parameters
	2. fitting parameters
	3. input data files (spectra, atmosphere)

## Rewritings

Rewrite output routines from RH of output spectrum? To write out only spectrum
at highest mu?

## Buggs

When I am doing synthesis I am grabbing 'atm' object for synthesis. Rewrite
function for synthesis to load only neccessary data. 

## Requierments

subprocess>=
multiprocessing>=
emcee>=
astropy>=
os>=
sys>=
time>=
copy>=
rh...