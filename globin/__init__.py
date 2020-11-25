"""
Contributors:
	Dusan Vukadinovic (DV)

13/10/2020 : rewriten class 'InputData'; we leave to user to read from given input
			 files;
18/10/2020 : read input parameters from 'params.input' using regular expressions

"""

import os
import sys
import numpy as np
import multiprocessing as mp
import re

from .rh import write_wavs, Rhout
from .atmos import Atmosphere, compute_rfs, compute_spectra, write_multi_atmosphere
from .spec import Observation
from .inversion import invert
from . import tools

__all__ = ["rh", "atmos", "invert", "spec", "tools"]
__name__ = "globin"
__path__ = os.path.dirname(__file__)

#--- comment character in files read by wrapper
COMMENT_CHAR = "#"

#--- limit values for atmospheric parameters
limit_values = {"temp"  : [3000,10000], 
				"vz"    : [-10, 10],
				"mag"   : [1, 5000],
				"gamma" : [-np.pi/2, np.pi/2],
				"chi"   : [-np.pi, np.pi],
				"vmic"  : [-10,10]}

#--- curent working directory: one from which we imported 'globin'
cwd = os.getcwd()

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

#===--- element abundances ---===#
from scipy.constants import k as K_BOLTZMAN
from scipy.interpolate import splrep, splev

abundance = np.loadtxt("../../Atoms/abundance.input", usecols=(1,), unpack=True)

def hydrogen_lvl_pops(logtau, temp, pg, pe, nlvl=6):
	"""
	Redistribute hydrogen atoms in first nlvl-1. Last column
	is reserved for proton number density.

	Parameters:
	---------------
	logtau : ndarray
		logarithm of optical depth.
	temp : ndarray
		temperature stratification in atmosphere.
	pg : ndarray
		gas pressure in CGS units (1D or 3D array).
	pe : ndarray
		electron pressure in CGS units (1D or 3D array).
	nlvl : int (optional)
		number of levels in hydrogen atom for which to calculate populations.
		last index stores proton numbers.

	Return:
	---------------
	popos : ndarray
		populations of hydrogen levels + protons. Dimension is (nvlv, len(temp)).
	tck : ndarray
		Spline knots for spline evaulation in given depth points.
	"""
	nH = (pg-pe)/10 / K_BOLTZMAN/temp / np.sum(10**(abundance-12)) / 1e6
	nH0 = nH / (1 + saha_phi(temp)/pe)
	nprot = nH - nH0
	
	pops = np.zeros((nlvl, *nH.shape))
	tcks = []

	for lvl in range(nlvl-1):
		e_lvl = 13.6*(1-1/(lvl+1)**2)
		pops[lvl] = nH/2 * 2*(lvl+1)**2 * np.exp(-5040/temp * e_lvl)
		tcks.append(splrep(logtau, pops[lvl]))
	pops[-1] = nprot
	tcks.append(splrep(logtau, nprot))

	return pops, tcks

def saha_phi(temp, u0=2, u1=1, Ej=13.6):
	"""
	Calculate Phi(T) function for Saha's equation in form:

	n+/n0 = Phi(T)/Pe

	All units are in cgs system.

	Parameters:
	---------------
	temp : ndarray
		temperature for which to calculate Phi(T) function
	u0 : float (optional)
		partition function of lower ionization stage. Default 2 (for H atom).
	u1 : float (optional)
		partition function of higher ionization stage.Default 1 (for H atom).
	Ej : float (optional)
		ionization energy of state in [eV]. Default 13.6 (for H atom).

	Return:
	---------------
	Phi(T) : ndarray
		value of Phi(T) function at every temperature
	"""
	return 0.6665 * u1/u0 * temp**(5/2) * 10**(-5040/temp*Ej)

#--- FAL C model (ref.): reference model if not given otherwise
falc = Atmosphere(__path__ + "/data/falc.dat")

# Hydrogen level population + interpolation
falc_hydrogen_pops, falc_hydrogen_lvls_tcks = hydrogen_lvl_pops(falc.data[0], falc.data[2], falc.data[3], falc.data[4])

# electron concentration [m-3] + interpolation
falc_ne = falc.data[4]/10/K_BOLTZMAN/falc.data[2] / 1e6
ne_tck = splrep(falc.data[0], falc_ne)

# temperature interpolation
temp_tck = splrep(falc.data[0],falc.data[2])

#===--- end ---====#

#--- polynomial degree for interpolation
interp_degree = None

#--- name of RH input file
rh_input = None

class InputData(object):
	"""
	Class for storing input data parameters.
	"""

	def __init__(self):
		# atmosphere (constructed from nodes) --> one which we invert
		self.atm = Atmosphere()
		# reference atmosphere
		self.ref_atm = None
		self.spectrum_path = "spec.fits"

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

		global interp_degree
		global rh_input

		self.globin_input_name = globin_input_name
		self.rh_input_name = rh_input_name
		rh_input = rh_input_name

		text = open(globin_input_name, "r").read()

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
			self.spectrum_path = find_value_by_key("spectrum", text, "default", "spec.fits")
			
			#--- optional parameters
			self.lmin = find_value_by_key("wave_min", text, "optional", conversion=float) / 10  # [nm]
			self.lmax = find_value_by_key("wave_max", text, "optional", conversion=float) / 10  # [nm]
			self.step = find_value_by_key("wave_step", text, "optional", conversion=float) / 10 # [nm]
			if (self.step is None) or (self.lmin is None) or (self.lmax is None):
				self.wave_grid_path = find_value_by_key("wave_grid", text, "required")
			else:
				self.wave_grid_path = None

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
			interp_degree = find_value_by_key("interp_degree", text, "default", 3, int)
			self.noise = find_value_by_key("noise", text, "default", 1e-3, float)
			self.marq_lambda = find_value_by_key("marq_lambda", text, "default", 1e-3, float)
			self.max_iter = find_value_by_key("max_iter", text, "default", 30, int)

			#--- optional parameters
			path_to_atmosphere = find_value_by_key("atmosphere", text, "optional")
			self.ref_atm = Atmosphere(path_to_atmosphere)
			# if user have not provided reference atmosphere we will assume FAL C model
			if self.ref_atm is None:
				self.ref_atm = falc
			self.lmin = find_value_by_key("wave_min", text, "optional", conversion=float) / 10  # [nm]
			self.lmax = find_value_by_key("wave_max", text, "optional", conversion=float) / 10  # [nm]
			self.step = find_value_by_key("wave_step", text, "optional", conversion=float) / 10 # [nm]
			if (self.step is None) or (self.lmin is None) or (self.lmax is None):
				self.wave_grid_path = find_value_by_key("wave_grid", text, "required")
			else:
				self.wave_grid_path = None

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
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["temp"])))
					matrix[:,:] = values
					self.atm.values["temp"] = matrix
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
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["vz"])))
					matrix[:,:] = values
					self.atm.values["vz"] = matrix
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
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["mag"])))
					matrix[:,:] = values
					self.atm.values["mag"] = matrix / 1e4 # Gauss --> Tesla
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
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["gamma"])))
					matrix[:,:] = np.deg2rad(values) # degree --> radians
					self.atm.values["gamma"] = matrix
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
					matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["chi"])))
					matrix[:,:] = np.deg2rad(values) # degree --> radians
					self.atm.values["chi"] = matrix
				except:
					print("Can not store node values for parameter 'chi'.")
					print("  Must read first observation file.")
					sys.exit()

			# missing nodes for micro-turbulent velocity
			# macro-turbulent broadening (can be fit)
			# instrument broadening
			# strailight contribution

		#--- get parameters from RH input file
		text = open(rh_input_name, "r").read()

		wave_file_path = find_value_by_key("WAVETABLE", text, "required")
		self.spec_name = find_value_by_key("SPECTRUM_OUTPUT", text, "default", "spectrum.out")

		if self.wave_grid_path is None:
			self.wavelength = np.arange(self.lmin, self.lmax+self.step, self.step)
		else:
			self.wavelength = np.loadtxt(self.wave_grid_path)
		write_wavs(self.wavelength, wave_file_path)

def read_nodes_and_values(line, param=None):
	if len(line[1].replace(" ",""))==1:
		num_of_nodes = int(line[1].replace(" ",""))
		try:
			logtau = self.atm.data[0,0,0]
		except AttributeError:
			print(f"Error: Must read first atmosphere to make nodes for {param}\n")
		idx = np.round(np.linspace(0, len(logtau)-1, num_of_nodes)).astype(int)
		self.nodes[param] = logtau[idx]
	else:
		return [float(item) for item in line[1].split(",")]