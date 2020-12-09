import os
import sys
import numpy as np
import multiprocessing as mp
import re
import copy

from .atmos import Atmosphere
from .spec import Observation
from .rh import write_wavs

import globin

#--- pattern search with regular expressions
pattern = lambda keyword: re.compile(f"^[^#\n]*({keyword})\s*=\s*(.*)", re.MULTILINE)

def find_value_by_key(key, text, key_type, default_val=None, conversion=str):
	match = pattern(key).search(text)
	if match:
		value = match.group(2)
		return conversion(value)
	else:
		if key_type=="required":
			sys.exit(f"We are missing keyword '{key}' in input file.")
		elif key_type=="default":
			return default_val
		elif key_type=="optional":
			return None

class InputData(object):
	"""
	Class for storing input data parameters.
	"""

	def __init__(self):
		# atmosphere (constructed from nodes) --> one which we invert
		self.atm = Atmosphere()
		# reference atmosphere
		self.ref_atm = None
		self.spectrum_path = "spectrum.fits"

	# def __str__(self):
	# 	pass

	def read_input_files(self, globin_input_name="params.input", rh_input_name="keyword.input"):
		""" 

		Read files 'globin_input_name' and 'rh_input_name' for input data.

		We assume that parameters are given in format:
			key = value

		Before every comment we have symbol '#', except in line with 'key = value'
		statement.

		Parameters:
		---------------
		globin_input_name : str (optional)
			File name in which are stored input parameters. By default we read
			from 'params.input' file.

		rh_input_name : str (optional)	
			File name for RH main input file. Default value is 'keyword.input'.
		"""

		self.globin_input_name = globin_input_name
		self.rh_input_name = rh_input_name
		
		#--- get parameters from RH input file
		text = open(rh_input_name, "r").read()
		self.rh_input = text

		wave_file_path = find_value_by_key("WAVETABLE", text, "required")
		self.spec_name = find_value_by_key("SPECTRUM_OUTPUT", text, "default", "spectrum.out")
		self.solve_ne = find_value_by_key("SOLVE_NE", text, "optional")

		#--- get parameters from globin input file
		text = open(globin_input_name, "r").read()
		self.params_input = text

		#--- find first mode of operation
		self.mode = find_value_by_key("mode",text,"required", conversion=int)
		
		#--- find number of threads
		self.n_thread = find_value_by_key("n_threads",text, "default", 1, conversion=int)
		self.pool = mp.Pool(self.n_thread)

		#--- get parameters for synthesis
		if self.mode==0:
			#--- required parameters
			path_to_atmosphere = find_value_by_key("atmosphere", text, "required")
			self.ref_atm = Atmosphere(path_to_atmosphere)
			
			#--- default parameters
			self.spectrum_path = find_value_by_key("spectrum", text, "default", "spectrum.fits")
			
			#--- optional parameters
			self.lmin = find_value_by_key("wave_min", text, "optional", conversion=float) / 10  # [nm]
			self.lmax = find_value_by_key("wave_max", text, "optional", conversion=float) / 10  # [nm]
			self.step = find_value_by_key("wave_step", text, "optional", conversion=float) / 10 # [nm]
			if (self.step is None) or (self.lmin is None) or (self.lmax is None):
				self.wave_grid_path = find_value_by_key("wave_grid", text, "required")
				self.wavelength = np.loadtxt(self.wave_grid_path)
			else:
				self.wavelength = np.arange(self.lmin, self.lmax+self.step, self.step)
			write_wavs(self.wavelength, wave_file_path)

		#--- get parameters for inversion
		if self.mode>=1:
			#--- required parameters
			path_to_observations = find_value_by_key("observation", text, "required")
			self.obs = Observation(path_to_observations)
			# set dimensions for atmosphere same as dimension of observations
			self.atm.nx = self.obs.nx
			self.atm.ny = self.obs.ny
			for idx in range(self.atm.nx):
				for idy in range(self.atm.ny):
					self.atm.atm_name_list.append(f"atmospheres/atm_{idx}_{idy}")
			
			#--- default parameters
			self.interp_degree = find_value_by_key("interp_degree", text, "default", 3, int)
			self.noise = find_value_by_key("noise", text, "default", 1e-3, float)
			self.marq_lambda = find_value_by_key("marq_lambda", text, "default", 1e-3, float)
			self.max_iter = find_value_by_key("max_iter", text, "default", 30, int)
			self.chi2_tolerance = find_value_by_key("chi2_tolerance", text, "default", 1e-2, float)
			values = find_value_by_key("weights", text, "default", [1,1,1,1])
			if type(values)==str:
				values = values.split(",")
				self.weights = np.array([float(item) for item in values], dtype=np.float64)
			else:
				self.weights = np.array(values, dtype=np.float64)

			#--- optional parameters
			path_to_atmosphere = find_value_by_key("atmosphere", text, "optional")
			if path_to_atmosphere is not None:
				self.ref_atm = Atmosphere(path_to_atmosphere)
			# if user have not provided reference atmosphere we will assume FAL C model
			else:
				self.ref_atm = Atmosphere(globin.__path__ + "/data/falc.dat")
			self.lmin = find_value_by_key("wave_min", text, "optional", conversion=float) / 10  # [nm]
			self.lmax = find_value_by_key("wave_max", text, "optional", conversion=float) / 10  # [nm]
			self.step = find_value_by_key("wave_step", text, "optional", conversion=float) / 10 # [nm]
			if (self.step is None) or (self.lmin is None) or (self.lmax is None):
				self.wave_grid_path = find_value_by_key("wave_grid", text, "required")
				self.wavelength = np.loadtxt(self.wave_grid_path)
			else:
				self.wavelength = np.arange(self.lmin, self.lmax+self.step, self.step)
			write_wavs(self.wavelength, wave_file_path)
			fpath = find_value_by_key("rf_weights", text, "optional")
			self.wavs_weight = np.ones((len(self.wavelength),4))
			if fpath is not None:
				lam, wI, wQ, wU, wV = np.loadtxt(fpath, unpack=True)
				if len(lam)==len(self.wavelength):
					self.wavs_weight[:,0] = wI
					self.wavs_weight[:,1] = wQ
					self.wavs_weight[:,2] = wU
					self.wavs_weight[:,3] = wV

			#--- nodes
			nodes = find_value_by_key("nodes_temp", text, "optional")
			values = find_value_by_key("nodes_temp_values", text, "optional")
			if (nodes is not None) and (values is not None):
				self.atm.nodes["temp"] = [float(item) for item in nodes.split(",")]
				self.atm.free_par += len(self.atm.nodes["temp"])

				values = [float(item) for item in values.split(",")]
				if len(values)!=len(self.atm.nodes["temp"]):
					sys.exit("Number of nodes and values for temperature are not the same!")
				
				try:	
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["temp"])), dtype=np.float64)
					matrix[:,:] = copy.deepcopy(values)
					self.atm.values["temp"] = copy.deepcopy(matrix)
				except:
					print("Can not store node values for parameter 'temp'.")
					print("  Must read first observation file.")
					sys.exit()

			nodes = find_value_by_key("nodes_vz", text, "optional")
			values = find_value_by_key("nodes_vz_values", text, "optional")
			if (nodes is not None) and (values is not None):
				self.atm.nodes["vz"] = [float(item) for item in nodes.split(",")]
				self.atm.free_par += len(self.atm.nodes["vz"])

				values = [float(item) for item in values.split(",")]
				if len(values)!=len(self.atm.nodes["vz"]):
					sys.exit("Number of nodes and values for vertical velocity are not the same!")

				try:	
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["vz"])), dtype=np.float64)
					matrix[:,:] = copy.deepcopy(values)
					self.atm.values["vz"] = copy.deepcopy(matrix)
				except:
					print("Can not store node values for parameter 'vz'.")
					print("  Must read first observation file.")
					sys.exit()
			
			nodes = find_value_by_key("nodes_mag", text, "optional")
			values = find_value_by_key("nodes_mag_values", text, "optional")
			if (nodes is not None) and (values is not None):
				self.atm.nodes["mag"] = [float(item) for item in nodes.split(",")]
				self.atm.free_par += len(self.atm.nodes["mag"])

				values = [float(item) for item in values.split(",")]
				if len(values)!=len(self.atm.nodes["mag"]):
					sys.exit("Number of nodes and values for magnetic field are not the same!")

				try:
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["mag"])), dtype=np.float64)
					matrix[:,:] = copy.deepcopy(values)
					self.atm.values["mag"] = copy.deepcopy(matrix) / 1e4 # Gauss --> Tesla
				except:
					print("Can not store node values for parameter 'mag'.")
					print("  Must read first observation file.")
					sys.exit()

			nodes = find_value_by_key("nodes_gamma", text, "optional")
			values = find_value_by_key("nodes_gamma_values", text, "optional")
			if (nodes is not None) and (values is not None):
				self.atm.nodes["gamma"] = [float(item) for item in nodes.split(",")]
				self.atm.free_par += len(self.atm.nodes["gamma"])

				values = [float(item) for item in values.split(",")]
				if len(values)!=len(self.atm.nodes["gamma"]):
					sys.exit("Number of nodes and values for magnetic field inclintion are not the same!")

				try:	
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["gamma"])), dtype=np.float64)
					matrix[:,:] = np.deg2rad(values) # degree --> radians
					self.atm.values["gamma"] = copy.deepcopy(matrix)
				except:
					print("Can not store node values for parameter 'gamma'.")
					print("  Must read first observation file.")
					sys.exit()

			nodes = find_value_by_key("nodes_chi", text, "optional")
			values = find_value_by_key("nodes_chi_values", text, "optional")
			if (nodes is not None) and (values is not None):
				self.atm.nodes["chi"] = [float(item) for item in nodes.split(",")]
				self.atm.free_par += len(self.atm.nodes["chi"])

				values = [float(item) for item in values.split(",")]
				if len(values)!=len(self.atm.nodes["chi"]):
					sys.exit("Number of nodes and values for magnetic field azimuth are not the same!")

				try:	
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["chi"])), dtype=np.float64)
					matrix[:,:] = np.deg2rad(values) # degree --> radians
					self.atm.values["chi"] = copy.deepcopy(matrix)
				except:
					print("Can not store node values for parameter 'chi'.")
					print("  Must read first observation file.")
					sys.exit()

			nodes = find_value_by_key("nodes_vmic", text, "optional")
			values = find_value_by_key("nodes_vmic_values", text, "optional")
			if (nodes is not None) and (values is not None):
				self.atm.nodes["vmic"] = [float(item) for item in nodes.split(",")]
				self.atm.free_par += len(self.atm.nodes["vmic"])

				values = [float(item) for item in values.split(",")]
				if len(values)!=len(self.atm.nodes["vmic"]):
					sys.exit("Number of nodes and values for vertical velocity are not the same!")

				try:	
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["vmic"])), dtype=np.float64)
					matrix[:,:] = copy.deepcopy(values)
					self.atm.values["vmic"] = copy.deepcopy(matrix)
				except:
					print("Can not store node values for parameter 'vmic'.")
					print("  Must read first observation file.")
					sys.exit()

			# missing nodes for micro-turbulent velocity
			# macro-turbulent broadening (can be fit)
			# instrument broadening
			# strailight contribution
