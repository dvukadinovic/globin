import os
import numpy as np
import multiprocessing as mp

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
		self.atm = None
		# number of threads to use
		self.n_thread = 1
		# wavelength grid parameters
		self.lmin = None
		self.lmax = None
		self.step = None
		self.wavs = None
		# spectrum file name
		self.spec_name = None
		# parameter node position
		self.nodes = {"temp" : [1],
					  "Bx"   : [],
					  "By"   : [],
					  "Bz"   : [],
					  "vz"   : [1],
					  "vmic" : [1]}

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
				input_data.pool = mp.Pool(input_data.n_thread)
			if keyword=="lmin":
				input_data.lmin = float(line[1].replace(" ",""))
			if keyword=="lmax":
				input_data.lmax = float(line[1].replace(" ",""))
			if keyword=="step":
				input_data.step = float(line[1].replace(" ",""))
			if keyword=="temp_nodes":
				if len(line[1].replace(" ",""))==1:
					num_of_nodes = int(line[1].replace(" ",""))
					try:
						logtau = input_data.atm.data[0,0,0]
					except AttributeError:
						print("Error: Must read first atmosphere to make nodes for temperature\n")
					idx = np.round(np.linspace(0, len(logtau)-1, num_of_nodes)).astype(int)
					input_data.nodes["temp"] = logtau[idx]
				else:
					aux = [float(item) for item in line[1].split(",")]
	

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