## Description

Development version of code for inversion of atomic and atmospheric parameters
simultaneously.

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

Many to come...

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