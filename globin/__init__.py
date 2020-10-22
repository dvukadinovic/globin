"""
Contributors:
	Dusan Vukadinovic (DV)

13/10/2020 : rewriten class 'InputData'; we leave to user to read from given input
			 files; 
"""

import os
import numpy as np
import multiprocessing as mp

from . import rh
from . import atmos
from . import spec
from . import invert
from . import tools

__all__ = ["rh", "atmos", "invert"]
__name__ = "globin"
__path__ = os.path.dirname(__file__)

#--- comment character in files read by wrapper
COMMENT_CHAR = "#"

#--- curent working directory: one from which we imported 'globin'
cwd = os.getcwd()

#--- FAL C model (ref.): reference model if not given otherwise
falc = atmos.Atmosphere(__path__ + "/data/falc.dat")

class InputData(object):
	"""
	Class for storing input data parameters.
	"""

	def __init__(self):
		# atmosphere (constructed from nodes) --> one which we invert
		self.atm = atmos.Atmosphere()
		# reference atmosphere
		self.ref_atm = None
		# number of threads to use
		self.n_thread = 1
		# wavelength grid parameters
		self.lmin = None
		self.lmax = None
		self.step = None
		self.wavs = None
		# Pool object from multithread run
		self.pool = None
		# spectrum file name
		self.spec_name = None

	# def __str__(self):
	# 	pass

	def read_input_files(self, globin_input="params.input", rh_input="keyword.input"):
		"""
		Function which opens 'globin_input' file for reading
		input data. Also, we read and input for RH given in file 'rh_input'.

		We assume that parameters are given in format:
			key = value

		Parameters:
		---------------
		globin_input : str (optional)
			File name in which are stored input parameters. By default we read
			from 'params.input' file.

		rh_input : str (optional)	
			File name for RH main input file. Default value is 'keyword.input'.
		"""
		self.globin_input = globin_input
		self.rh_input = rh_input

		#--- read 'parameters.input' file
		lines = open(globin_input, "r").readlines()

		for line in lines:
			line = line.rstrip("\n").replace(" ","")
			# skip blank lines
			if len(line)>0:
				# skip commented lines
				if line[0]!=COMMENT_CHAR:
					line = line.split("=")
					keyword, value = line
					if keyword=="observation":
						self.obs = spec.Observation(value)
						# set dimensions for atmosphere same as dimension of observations
						self.atm.nx = self.obs.nx
						self.atm.ny = self.obs.ny
					elif keyword=="atmosphere":
						atm_path = value
						self.ref_atm = atmos.Atmosphere(atm_path)
					elif keyword=="n_threads":
						self.n_thread = int(value)
						self.pool = mp.Pool(self.n_thread)
					elif keyword=="wave_min":
						self.lmin = float(value) / 10 # from Angstroms to nm
					elif keyword=="wave_max":
						self.lmax = float(value) / 10 # from Angstroms to nm
					elif keyword=="wave_step":
						self.step = float(value) / 10 # from Angstroms to nm
					elif keyword=="nodes_temp":
						self.atm.nodes["temp"] = read_nodes_and_values(line)
					elif keyword=="nodes_temp_values":
						values = read_nodes_and_values(line)
						try:	
							matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["temp"])))
							matrix[:,:] = values
							self.atm.values["temp"] = matrix
						except:
							print("Can not store node values for parameter 'temp'.")
							print("  Must read first observation file.")
							sys.exit()
					elif keyword=="nodes_vz":
						self.atm.nodes["vz"] = read_nodes_and_values(line)
					elif keyword=="nodes_vz_values":
						values = read_nodes_and_values(line)
						try:	
							matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["vz"])))
							matrix[:,:] = values
							self.atm.values["vz"] = matrix
						except:
							print("Can not store node values for parameter 'vz'.")
							print("  Must read first observation file.")
							sys.exit()
					else:
						# block for not supported keywords
						print(f"Currently not supported keyword '{keyword}'")

		# if user have not provided reference atmosphere we will assume FAL C model
		if self.ref_atm is None:
			self.ref_atm = falc

		#--- read 'keyword.input' file
		lines = open(rh_input, "r").readlines()

		for line in lines:
			if line.replace(" ","")[0]!=COMMENT_CHAR:
				line = line.rstrip("\n").split("=")
				keyword = line[0].replace(" ", "")
				if keyword=="WAVETABLE":
					wave_file_path = line[1].replace(" ","")
				if keyword=="SPECTRUM_OUTPUT":
					self.spec_name = line[1].replace(" ","")

		if self.obs.wavelength is None:
			self.wavelength = np.arange(self.lmin, self.lmax+self.step, self.step)
		else:
			self.wavelength = self.obs.wavelength
		rh.write_wavs(self.wavelength, wave_file_path)

def read_nodes_and_values(line, param=None):
	if len(line[1].replace(" ",""))==1:
		pass
		num_of_nodes = int(line[1].replace(" ",""))
		try:
			logtau = self.atm.data[0,0,0]
		except AttributeError:
			print(f"Error: Must read first atmosphere to make nodes for {param}\n")
		idx = np.round(np.linspace(0, len(logtau)-1, num_of_nodes)).astype(int)
		self.nodes[param] = logtau[idx]
	else:
		return [float(item) for item in line[1].split(",")]