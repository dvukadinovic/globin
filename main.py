"""
Inversion of atomic parameters.

Method functions:
  -- read input file ('params.input')
  -- forward solution --> calling RH (distribute load into threads)
  -- minimization rootine --> calling MCMC routines

Contributors:
  Dusan Vukadinovic (DV)

Diary: 
  17/09/2020 --- started writing the code (readme file, structuring)

Last update: 17/09/2020
"""

import subprocess as sp
import multiprocessing as mp
import emcee

def ReadInputFile(fname="params.input"):
	lines = open(fname, "r").readlines()
	pass

def ReadAtmosphere():
	# extract each atmosphere into single .atmos file
	# change path to atmos in 'keyword.input' file
	pass

def ComputeSpectra():
	# creat separate folders for each atmosphere which will be
	# synthesized
	# copy *.input files to appropriate folders
	# distribute spectrum synthesis to different threads
	
	# p = sp.run("../rhf1d", 
	# 		shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

ReadInputFile()
ComputeSpectra()