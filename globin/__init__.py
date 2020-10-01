import os
import numpy as np

from globin import rh
from globin import atmos
from globin import invert

COMMENT_CHAR = "#"

#--- curent working directory 
cwd = os.getcwd()

print()

class Input(object):
	"""
	Class for storing input data parameters.
	"""

	def __init__(self):
		# atmosphere
		atm = None
		# number of threads to use
		n_thread = 1
		# wavelength grid parameters
		lmin = None
		lmax = None
		step = None
		wavs = None
		# spectrum file name
		spec_name = None

def ReadInputFile(fname="params.input"):
	"""
	Function which opens 'params.input' file for reading
	input data.

	We assume that parameters are given in format:
		key = value

	Parameters:
	---------------
	fname : str
		File name in which are stored input parameters.

	Returns:
	---------------
	Input : struct
		Input class in which we store all the informations
	"""

	lines = open(fname, "r").readlines()

	input_data = Input()

	for line in lines:
		if line[0]!=COMMENT_CHAR:
			line = line.rstrip("\n").split("=")
			keyword = line[0].replace(" ", "")
			if keyword=="atmosphere":
				atm_path = line[1].replace(" ", "")
				input_data.atm = atmos.Atmosphere(atm_path)
			if keyword=="n_threads":
				input_data.n_thread = int(line[1].replace(" ", ""))
			if keyword=="lmin":
				input_data.lmin = float(line[1].replace(" ",""))
			if keyword=="lmax":
				input_data.lmax = float(line[1].replace(" ",""))
			if keyword=="step":
				input_data.step = float(line[1].replace(" ",""))

	make_wave_file(input_data.lmin, input_data.lmax, input_data.step, input_data)

	return input_data

def make_wave_file(lmin, lmax, step, init):
	step /= 1e4
	wavelength = np.arange(lmin, lmax+step, step)
	init.wavs = wavelength

	lines = open("keyword.input", "r").readlines()
	for line in lines:
		if line.replace(" ","")[0]!=COMMENT_CHAR:
			line = line.rstrip("\n").split("=")
			keyword = line[0].replace(" ", "")
			if keyword=="WAVETABLE":
				fname = line[1].replace(" ","")
			if keyword=="SPECTRUM_OUTPUT":
				init.spec_name = line[1].replace(" ","")

	rh.write_wavs(wavelength, fname)