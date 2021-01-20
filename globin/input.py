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
		pass

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
		globin.rh_input_name = rh_input_name

		#--- get parameters from RH input file
		text = open(rh_input_name, "r").read()
		self.rh_input = text

		wave_file_path = find_value_by_key("WAVETABLE", text, "required")
		self.spec_name = find_value_by_key("SPECTRUM_OUTPUT", text, "default", "spectrum.out")
		self.solve_ne = find_value_by_key("SOLVE_NE", text, "optional")
		RLK_linelist_path = find_value_by_key("KURUCZ_DATA", text, "optional")

		#--- get parameters from globin input file
		text = open(globin_input_name, "r").read()
		self.params_input = text

		#--- find first mode of operation
		self.mode = find_value_by_key("mode",text,"required", conversion=int)
		
		#--- find number of threads
		self.n_thread = find_value_by_key("n_threads",text, "default", 1, conversion=int)

		#--- get parameters for synthesis
		if self.mode==0:
			#--- required parameters
			path_to_atmosphere = find_value_by_key("atmosphere", text, "required")
			self.atm = Atmosphere(path_to_atmosphere)
			
			#--- default parameters
			globin.spectrum_path = find_value_by_key("spectrum", text, "default", "spectrum.fits")
			vmac = abs(find_value_by_key("vmac", text, "default", default_val=0, conversion=float))
			self.atm.vmac = vmac*1e3 # [m/s]

			#--- optional parameters
			self.lmin = find_value_by_key("wave_min", text, "optional", conversion=float) / 10  # [nm]
			self.lmax = find_value_by_key("wave_max", text, "optional", conversion=float) / 10  # [nm]
			self.step = find_value_by_key("wave_step", text, "optional", conversion=float) / 10 # [nm]
			if (self.step is None) or (self.lmin is None) or (self.lmax is None):
				self.wave_grid_path = find_value_by_key("wave_grid", text, "required")
				self.wavelength = np.loadtxt(self.wave_grid_path)
				self.lmin = min(self.wavelength)
				self.lmax = max(self.wavelength)
				self.step = self.wavelength[1] - self.wavelength[0]
			else:
				self.wavelength = np.arange(self.lmin, self.lmax+self.step, self.step)
			write_wavs(self.wavelength, wave_file_path)
			# standard deviation of Gaussian kernel for macro broadening
			self.atm.sigma = lambda vmac: vmac / globin.LIGHT_SPEED * (self.lmin + self.lmax)*0.5 / self.step

		#--- get parameters for inversion
		if self.mode>=1:
			# initialize container for atmosphere which we invert
			self.atm = Atmosphere()

			# determine which observations from cube to take into consideration
			aux = find_value_by_key("range", text, "default", [1,None,1,None])
			self.atm_range = []
			if type(aux)==str:
				for item in aux.split(","):
					if item is None or int(item)==-1:
						self.atm_range.append(None)
					elif item is not None:
						self.atm_range.append(int(item))
			else:
				self.atm_range = aux
			# we count from zero, but let user count from 1
			self.atm_range[0] -= 1
			self.atm_range[2] -= 1
		
			#--- required parameters
			path_to_observations = find_value_by_key("observation", text, "required")
			self.obs = Observation(path_to_observations, self.atm_range)
			# set dimensions for atmosphere same as dimension of observations
			self.atm.nx = self.obs.nx
			self.atm.ny = self.obs.ny
			for idx in range(self.atm.nx):
				for idy in range(self.atm.ny):
					self.atm.atm_name_list.append(f"atmospheres/atm_{idx}_{idy}")
			
			#--- default parameters
			globin.interp_degree = find_value_by_key("interp_degree", text, "default", 3, int)
			self.noise = find_value_by_key("noise", text, "default", 1e-3, float)
			self.marq_lambda = find_value_by_key("marq_lambda", text, "default", 1e-3, float)
			self.max_iter = find_value_by_key("max_iter", text, "default", 30, int)
			self.chi2_tolerance = find_value_by_key("chi2_tolerance", text, "default", 1e-2, float)
			values = find_value_by_key("weights", text, "default", np.array([1,1,1,1], dtype=np.float64))
			if type(values)==str:
				values = values.split(",")
				self.weights = np.array([float(item) for item in values], dtype=np.float64)
			vmac = find_value_by_key("vmac", text, "default", default_val=0, conversion=float)
			self.atm.vmac = vmac*1e3 # [m/s]
			# if macro-turbulent velocity is negative, we fit for it
			if self.atm.vmac<0:
				self.atm.vmac = abs(self.atm.vmac)
				self.atm.global_pars["vmac"] = np.array([self.atm.vmac])

			#--- optional parameters
			path_to_atmosphere = find_value_by_key("atmosphere", text, "optional")
			if path_to_atmosphere is not None:
				self.ref_atm = Atmosphere(path_to_atmosphere, atm_range=self.atm_range)
			# if user have not provided reference atmosphere we will assume FAL C model
			else:
				self.ref_atm = Atmosphere(globin.__path__ + "/data/falc.dat")
			self.lmin = find_value_by_key("wave_min", text, "optional", conversion=float) / 10  # [nm]
			self.lmax = find_value_by_key("wave_max", text, "optional", conversion=float) / 10  # [nm]
			self.step = find_value_by_key("wave_step", text, "optional", conversion=float) / 10 # [nm]
			if (self.step is None) and (self.lmin is None) and (self.lmax is None):
				self.wave_grid_path = find_value_by_key("wave_grid", text, "required")
				self.wavelength = np.loadtxt(self.wave_grid_path)
				self.lmin = min(self.wavelength)
				self.lmax = max(self.wavelength)
				self.step = self.wavelength[1] - self.wavelength[0]
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
			# standard deviation of Gaussian kernel for macro broadening
			self.atm.sigma = lambda vmac: vmac / globin.LIGHT_SPEED * (self.lmin + self.lmax)*0.5 / self.step

			#--- nodes
			nodes = find_value_by_key("nodes_temp", text, "optional")
			values = find_value_by_key("nodes_temp_values", text, "optional")
			if (nodes is not None) and (values is not None):
				self.atm.nodes["temp"] = [float(item) for item in nodes.split(",")]
				
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

			nodes = find_value_by_key("nodes_vmic", text, "optional")
			values = find_value_by_key("nodes_vmic_values", text, "optional")
			if (nodes is not None) and (values is not None):
				self.atm.nodes["vmic"] = [float(item) for item in nodes.split(",")]
				
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
			
			nodes = find_value_by_key("nodes_mag", text, "optional")
			values = find_value_by_key("nodes_mag_values", text, "optional")
			if (nodes is not None) and (values is not None):
				self.atm.nodes["mag"] = [float(item) for item in nodes.split(",")]
				
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

			if self.mode==3:
				#--- line parameters to be fit
				line_par_path = find_value_by_key("line_parameters", text, "optional")

				if line_par_path:
					# if we provided line parameters for fit, read those parameters
					loggf_lineNo, loggf_init, loggf_min_max, dlam_lineNo, dlam_init, dlam_min_max = read_line_parameters(line_par_path)
					# if we have log(gf) init values read, set them in global_pars variable
					self.atm.line_no = {}
					if len(loggf_init)>0:
						self.atm.global_pars["loggf"] = np.array(loggf_init)
						self.atm.line_no["loggf"] = np.array(loggf_lineNo)
						globin.limit_values["loggf"] = loggf_min_max
					else:
						self.atm.global_pars["loggf"] = []
						self.atm.line_no["loggf"] = []
					# if we have dlam init values read, set them in global_pars variable
					if len(dlam_init)>0:
						self.atm.global_pars["dlam"] = np.array(dlam_init)
						self.atm.line_no["dlam"] = np.array(dlam_lineNo)
						globin.limit_values["dlam"] = dlam_min_max
					else:
						self.atm.global_pars["dlam"] = []
						self.atm.line_no["dlam"] = []

					#--- Kurucz line list for given spectral region
					if RLK_linelist_path:
						linelist_path = find_value_by_key("linelist", text, "required")
						self.RLK_text_lines, self.RLK_lines = read_RLK_lines(linelist_path)

						# go through RLK file and find the uncommented line
						# with path to atomic line files of Kurucz format
						lines = open(RLK_linelist_path, "r").readlines()
						for line in lines:
							line = line.rstrip("\n").strip(" ")
							# find the first uncommented line and break
							if line[0]!=globin.COMMENT_CHAR:
								self.RLK_path = line
								break

						# write down perturbed values
						self.write_line_parameters(self.atm.global_pars["loggf"], self.atm.line_no["loggf"],
												   self.atm.global_pars["dlam"], self.atm.line_no["dlam"])
						# sys.exit()
					else:
						print("No path to kurucz.input file.")
						# print("There is no Kurucz line list file to write to.")
						# print("If you want to invert for line parameters, you need to set")
						# print("path to file where Kurucz line lists are (kurucz.input file).")
						sys.exit()

			#--- if we have more threads than atmospheres, reduce the number of used threads
			if self.n_thread > self.atm.nx*self.atm.ny:
				self.n_thread = self.atm.nx*self.atm.ny
				print(f"\n\nWarning: reduced the number of threads to {self.n_thread}.\n\n")
			self.pool = mp.Pool(self.n_thread)

			#--- determine number of local and global parameters
			self.atm.n_local_pars = 0
			for pID in self.atm.nodes:
				self.atm.n_local_pars += len(self.atm.nodes[pID])

			self.atm.n_global_pars = 0
			for pID in self.atm.global_pars:
				self.atm.n_global_pars += len(self.atm.global_pars[pID])

			#--- missing parameters
			# instrument broadening: R or instrument profile provided
			# strailight contribution

	def write_line_parameters(self, loggf_val, loggf_no, dlam_val, dlam_no):
		out = open(self.RLK_path, "w")
		# out = open("test_RLK_file", "w")
		linelist = self.RLK_text_lines

		for no,val in zip(loggf_no, loggf_val):
			character_list = list(linelist[no])
			character_list[10:17] = "{: 7.3f}".format(val)
			linelist[no] = ''.join(character_list)
		for no,val in zip(dlam_no, dlam_val):
			character_list = list(linelist[no])
			# Note! Formating from Kurucz is F11.4, but here I used 10.4 since
			# this one gives correct number of characters as input array. RH needs
			# 160 character line, and I toke out one character from the 
			# beginning of the line. That is why we need here 10.4 format.
			character_list[0:10] = "{: 10.4f}".format(val/1e4 + self.RLK_lines[no].lam0)
			linelist[no] = ''.join(character_list)

		out.writelines(linelist)
		out.close()

	def write_line_par(self, par_val, par_no, parameter):
		out = open(self.RLK_path, "w")
		linelist = self.RLK_text_lines

		if parameter=="loggf":
			character_list = list(linelist[par_no])
			character_list[10:17] = "{: 7.3f}".format(par_val)
			linelist[par_no] = ''.join(character_list)
		if parameter=="dlam":
			character_list = list(linelist[par_no])
			# Note! Formating from Kurucz is F11.4, but here I used 10.4 since
			# this one gives correct number of characters as input array. RH needs
			# 160 character line, and I toke out one character from the 
			# beginning of the line. That is why we need here 10.4 format.
			character_list[0:10] = "{: 10.4f}".format(par_val/1e4 + self.RLK_lines[par_no].lam0)
			linelist[par_no] = ''.join(character_list)

		out.writelines(linelist)
		out.close()

class Line(object):

	def __init__(self, lineNo=None, lam0=None, loggf=None):
		self.lineNo = lineNo
		self.lam0 = lam0
		self.loggf = loggf

	def __str__(self):
		return "<LineNo: {}, lam0: {}, loggf: {}>".format(self.lineNo, self.lam0, self.loggf)

def read_RLK_lines(fpath):
	lines = open(fpath, "r").readlines()
	
	RLK_lines = []

	for i_, line in enumerate(lines):
		lam0 = float(line[0:10])
		loggf = float(line[10:17])

		RLK_lines.append(Line(i_+1, lam0, loggf))

	return lines, RLK_lines

def read_line_parameters(fpath):
		lines = open(fpath, "r").readlines()

		loggf_lineNo = []
		dlam_lineNo = []

		loggf_init = []
		dlam_init = []

		loggf_min_max = np.array([], dtype=np.float64)
		dlam_min_max = np.array([], dtype=np.float64)

		for line in lines:
			line = list(filter(None,line.rstrip("\n").split(" ")))
			if line[0]=="loggf":
				# substract 1 since we are counting from 0 here
				loggf_lineNo.append(int(line[1])-1)
				loggf_init.append(float(line[2]))
				loggf_min_max = np.append( loggf_min_max, np.array( [float(line[3]), float(line[4])] ) )
			elif line[0]=="dlam":
				# substract 1 since we are counting from 0 here
				dlam_lineNo.append(int(line[1])-1)
				dlam_init.append(float(line[2]))
				dlam_min_max = np.append( dlam_min_max, np.array( [float(line[3]), float(line[4])] ) )

		# reshape into dimension aprpropriate for min/max check later
		n_loggf = len(loggf_init)
		loggf_min_max = np.reshape(loggf_min_max, (n_loggf,2))
		n_dlam = len(dlam_init)
		dlam_min_max = np.reshape(dlam_min_max, (n_dlam,2))
		
		return loggf_lineNo, loggf_init, loggf_min_max, dlam_lineNo, dlam_init, dlam_min_max