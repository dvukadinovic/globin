"""
Contributors:
	Dusan Vukadinovic (DV)

13/10/2020 : rewriten class 'InputData'; we leave to user to read from given input
			 files;
16/10/2020 : read input parameters from 'params.input' using regular expressions
	
	pattern = re.compile(f"^[^#\n]*({input_par})\s*=\s*(.*)", re.MULTILINE)
	res = pattern.search(text)

"""

import os
import sys
import numpy as np
import multiprocessing as mp
import re

from . import rh
from . import atmos
from . import spec
from . import invert
from . import tools

__all__ = ["rh", "atmos", "invert", "spec", "tools"]
__name__ = "globin"
__path__ = os.path.dirname(__file__)

#--- comment character in files read by wrapper
COMMENT_CHAR = "#"

#--- curent working directory: one from which we imported 'globin'
cwd = os.getcwd()

#--- pattern search with regular expressions
pattern = lambda keyword: re.compile(f"^[^#\n]*({keyword})\s*=\s*(.*)", re.MULTILINE)
def find_value_by_key(key,text):
	value = pattern(key).search(text).group(2)
	return value

#--- element abundances
from scipy.constants import k
from scipy.interpolate import splrep, splev

K_BOLTZMAN = k
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
falc = atmos.Atmosphere(__path__ + "/data/falc.dat")

# Hydrogen level population + interpolation
falc_hydrogen_pops, falc_hydrogen_lvls_tcks = hydrogen_lvl_pops(falc.data[0], falc.data[2], falc.data[3], falc.data[4])

# electron concentration [m-3] + interpolation
falc_ne = falc.data[4]/10/K_BOLTZMAN/falc.data[2] / 1e6
ne_tck = splrep(falc.data[0], falc_ne)

# temperature interpolation
temp_tck = splrep(falc.data[0],falc.data[2])

#--- polynomial degree for interpolation
interp_degree = 3

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
		self.wave_grid = None
		# Pool object from multithread run
		self.pool = None
		# spectrum file name
		self.spec_name = None
		# noise
		self.noise = 1e-3

	# def __str__(self):
	# 	pass

	def read_input_files(self, globin_input="params.input", rh_input="keyword.input"):
		"""
		Function which opens 'globin_input' file for reading
		input data. Also, we read and input for RH given in file 'rh_input'.

		We assume that parameters are given in format:
			key = value

		Before every comment we have symbol '#', except in line with 'key = value'
		statement.

		Parameters:
		---------------
		globin_input : str (optional)
			File name in which are stored input parameters. By default we read
			from 'params.input' file.

		rh_input : str (optional)	
			File name for RH main input file. Default value is 'keyword.input'.
		"""

		global interp_degree

		self.globin_input = globin_input
		self.rh_input = rh_input

		# text = open(globin_input, "r").read()

		# #--- find first mode of operation
		# value = find_value_by_key("mode",text)
		# mode = int(value)
		
		# #--- find number of threads
		# value = find_value_by_key("n_threads",text)
		# self.n_thread = int(value)
		# self.pool = mp.Pool(self.n_thread)

		# #--- define list of parameters for which to search through input file
		# req_parameters_for_synthesis = ["atmosphere","wave_min","wave_max","wave_step"]
		# opt_parameters_for_synthesis = ["noise","wave_grid"]
		# parameters_for_inversion_px_by_px = None
		# parameters_for_inversion_global = None

		# #--- get parameters for synthesis
		# if mode==0:
		# 	for par_name in req_parameters_for_synthesis:
		# 		value = find_value_by_key(par_name, text)

		# return 0

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
					if keyword=="interp_degree":
						interp_degree = int(value)
						if (interp_degree!=2) and (interp_degree!=3):
							sys.exit("Error: polynomial degree for interpolation is incorrect.\n  Valid values are 2 and 3.")
					elif keyword=="mode":
						self.mode = int(value)
					elif keyword=="observation" and self.mode>0:
						self.obs = spec.Observation(value)
						# set dimensions for atmosphere same as dimension of observations
						self.atm.nx = self.obs.nx
						self.atm.ny = self.obs.ny
						for idx in range(self.atm.nx):
							for idy in range(self.atm.ny):
								self.atm.atm_name_list.append(f"atmospheres/atm_{idx}_{idy}")
					elif keyword=="noise":
						self.noise = float(value)
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
					elif keyword=="wave_grid":
						self.wave_grid = value
					elif keyword=="nodes_temp" and self.mode>0:
						self.atm.nodes["temp"] = read_nodes_and_values(line)
						self.atm.free_par += len(self.atm.nodes["temp"])
					elif keyword=="nodes_temp_values" and self.mode>0:
						values = read_nodes_and_values(line)
						try:	
							matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["temp"])))
							matrix[:,:] = values
							self.atm.values["temp"] = matrix
						except:
							print("Can not store node values for parameter 'temp'.")
							print("  Must read first observation file.")
							sys.exit()
					elif keyword=="nodes_vz" and self.mode>0:
						self.atm.nodes["vz"] = read_nodes_and_values(line)
						self.atm.free_par += len(self.atm.nodes["vz"])
					elif keyword=="nodes_vz_values" and self.mode>0:
						values = read_nodes_and_values(line)
						try:	
							matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes["vz"])))
							matrix[:,:] = values
							self.atm.values["vz"] = matrix
						except:
							print("Can not store node values for parameter 'vz'.")
							print("  Must read first observation file.")
							sys.exit()
					elif keyword=="marq_lambda":
						self.marq_lambda = float(value)
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

		# if self.obs.wavelength is None:
		if self.wave_grid is None:
			self.wavelength = np.arange(self.lmin, self.lmax+self.step, self.step)
		else:
			self.wavelength = np.loadtxt(self.wave_file)
		# else:
			# self.wavelength = self.obs.wavelength
		aux = rh.write_wavs(self.wavelength, wave_file_path)

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