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
	"""
	Regexp search of 'key' in given 'text'.

	Parameters:
	---------------
	key : str
		keyword whose value we are searching in 'text'.
	text : str
		string in which we look for the keyword 'key'.
	key_type : str
		parameter which determines the importance of the key. We have 'requiered',
		'default' and 'optional' parameter. In case of failed search for 'requiered'
		parameter error is raised. For failed 'default' key search returned value is
		one given with 'default_val' parameter. 'optional' type variable returns 'None'
		if there is no 'key' in 'text'.
	default_val : {str,float,int}
		value which will be returned in case of failed 'key' search for default
		'key_type'.
	conversion : function (default 'str')
		function which translate given string into given type ('float' or 'int'). By
		default we assume that return value will be string.

	Return:
	---------------
	value : {str, float, int}
		value of 'key' in variable type given by 'conversion' parameter.

	Errors:
	---------------
	In failed search for 'key_type=requiered' execution stops.
	"""
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

	def __init__(self, globin_input_name="params.input", rh_input_name="keyword.input"):
		
		if (rh_input_name is not None) and (globin_input_name is not None):
			self.read_input_files(globin_input_name, rh_input_name)
		else:
			if rh_input_name is None:
				print(f"  There is no path for globin input file path.")
			if globin_input_name is None:
				print(f"  There is no path for RH input file path.")
			sys.exit()

	def __str__(self):
		return "<InputData:\n  globin = {0}\n  RH = {1}\n>".format(self.globin_input_name, self.rh_input_name)

	def read_input_files(self, globin_input_name, rh_input_name):
		"""
		Read input files for globin ('globin_input_name') and RH ('rh_input_name').

		We assume that parameters (in both files) are given in format:
			key = value

		Commented lines begin with symbol '#'.

		Parameters:
		---------------
		globin_input_name : str
			file name in which are stored input parameters for globin. By default
			we read from 'params.input' file.

		rh_input_name : str
			file name for RH input file. Default value is 'keyword.input'.
		"""
		self.globin_input_name = globin_input_name
		self.rh_input_name = rh_input_name
		globin.rh_input_name = rh_input_name

		#--- get parameters from RH input file
		text = open(rh_input_name, "r").read()
		self.rh_input = text

		wave_file_path = find_value_by_key("WAVETABLE", text, "required")
		wave_file_path = wave_file_path.split("/")[-1]
		self.rh_spec_name = find_value_by_key("SPECTRUM_OUTPUT", text, "default", "spectrum.out")
		# self.solve_ne = find_value_by_key("SOLVE_NE", text, "optional")
		RLK_linelist_path = find_value_by_key("KURUCZ_DATA", text, "optional")
		globin.rf_file_path = find_value_by_key("RF_OUTPUT", text, "default", "rfs.out")

		#--- get parameters from globin input file
		text = open(globin_input_name, "r").read()
		self.globin_input = text

		#--- find first mode of operation
		globin.mode = find_value_by_key("mode", text, "required", conversion=int)
		
		#--- path to RH main folder
		rh_path = find_value_by_key("rh_path", text, "required")
		if rh_path.rstrip("\n")[-1]=="/":
			rh_path = rh_path.rstrip("/")
		globin.rh_path = rh_path
		
		#--- find number of threads
		globin.n_thread = find_value_by_key("n_thread",text, "default", 1, conversion=int)

		#--- interpolation degree
		globin.interp_degree = find_value_by_key("interp_degree", text, "default", 3, int)

		#--- get parameters for synthesis
		if globin.mode==0:
			# determine which observations from cube to take into consideration
			aux = find_value_by_key("range", text, "default", [1,None,1,None])
			if type(aux)==str:
				self.atm_range = []
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
			path_to_atmosphere = find_value_by_key("cube_atmosphere", text, "optional")
			if path_to_atmosphere is None:
				node_atmosphere_path = find_value_by_key("node_atmosphere", text, "required")
				self.atm = globin.construct_atmosphere_from_nodes(node_atmosphere_path, self.atm_range)
			else:
				self.atm = Atmosphere(path_to_atmosphere, atm_range=self.atm_range)
				self.atm.split_cube()

			#--- default parameters
			globin.spectrum_path = find_value_by_key("spectrum", text, "default", "spectrum.fits")
			self.noise = find_value_by_key("noise", text, "default", 1e-3, float)
			vmac = find_value_by_key("vmac", text, "default", default_val=0, conversion=float)
			self.atm.vmac = np.abs(vmac) # [km/s]

			#--- optional parameters
			self.lmin = find_value_by_key("wave_min", text, "optional", conversion=float) / 10  # [nm]
			self.lmax = find_value_by_key("wave_max", text, "optional", conversion=float) / 10  # [nm]
			self.step = find_value_by_key("wave_step", text, "optional", conversion=float) / 10 # [nm]
			if (self.step is None) or (self.lmin is None) or (self.lmax is None):
				wave_grid_path = find_value_by_key("wave_grid", text, "required")
				self.wavelength = np.loadtxt(wave_grid_path)
				self.lmin = min(self.wavelength)
				self.lmax = max(self.wavelength)
				self.step = self.wavelength[1] - self.wavelength[0]
			else:
				self.wavelength = np.arange(self.lmin, self.lmax+self.step, self.step)
			write_wavs(self.wavelength, globin.rh_path + "/Atoms/wave_files/" + wave_file_path)
			# standard deviation of Gaussian kernel for macro broadening
			self.atm.sigma = lambda vmac: vmac / globin.LIGHT_SPEED * (self.lmin + self.lmax)*0.5 / self.step

			# reference atmosphere is the same as input one in mode of synthesis
			self.ref_atm = copy.deepcopy(self.atm)

		#--- get parameters for inversion
		elif globin.mode>=1:
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
			self.noise = find_value_by_key("noise", text, "default", 1e-3, float)
			self.marq_lambda = find_value_by_key("marq_lambda", text, "default", 1e-3, float)
			self.max_iter = find_value_by_key("max_iter", text, "default", 30, int)
			self.chi2_tolerance = find_value_by_key("chi2_tolerance", text, "default", 1e-2, float)
			values = find_value_by_key("weights", text, "default", np.array([1,1,1,1], dtype=np.float64))
			if type(values)==str:
				values = values.split(",")
				self.weights = np.array([float(item) for item in values], dtype=np.float64)
			vmac = find_value_by_key("vmac", text, "default", default_val=0, conversion=float)

			#--- optional parameters
			path_to_atmosphere = find_value_by_key("cube_atmosphere", text, "optional")
			if path_to_atmosphere is not None:
				self.ref_atm = Atmosphere(path_to_atmosphere, atm_range=self.atm_range)
			# if user have not provided reference atmosphere try fidning node atmosphere
			else:
				path_to_node_atmosphere = find_value_by_key("node_atmosphere", text, "optional")
				if path_to_node_atmosphere is not None:
					self.ref_atm = globin.construct_atmosphere_from_nodes(path_to_node_atmosphere, self.atm_range)
				# if node atmosphere not given, set FAL C model as reference atmosphere
				else:
					self.ref_atm = Atmosphere(globin.__path__ + "/data/falc.dat")

			self.lmin = find_value_by_key("wave_min", text, "optional", conversion=float) / 10  # [nm]
			self.lmax = find_value_by_key("wave_max", text, "optional", conversion=float) / 10  # [nm]
			self.step = find_value_by_key("wave_step", text, "optional", conversion=float) / 10 # [nm]
			if (self.step is None) and (self.lmin is None) and (self.lmax is None):
				wave_grid_path = find_value_by_key("wave_grid", text, "required")
				self.wavelength = np.loadtxt(wave_grid_path)
				self.lmin = min(self.wavelength)
				self.lmax = max(self.wavelength)
				self.step = self.wavelength[1] - self.wavelength[0]
			else:
				self.wavelength = np.arange(self.lmin, self.lmax+self.step, self.step)
			write_wavs(self.wavelength, globin.rh_path + "/Atoms/wave_files/" + wave_file_path)

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
			self.atm.vmac = vmac # [km/s]
			self.atm.sigma = lambda vmac: vmac*1e3 / globin.LIGHT_SPEED * (self.lmin + self.lmax)*0.5 / self.step

			# if macro-turbulent velocity is negative, we fit it
			if self.atm.vmac<0:
				# check if initial macro veclocity is larger than the step size in wavelength
				vmac = np.abs(vmac)
				kernel_sigma = self.atm.sigma(vmac)
				if kernel_sigma<0.5:
					vmac = 0.5 * globin.LIGHT_SPEED / ((self.lmin + self.lmax)*0.5) * self.step
					vmac /= 1e3
					globin.limit_values["vmac"][0] = vmac
				
				self.atm.vmac = abs(vmac)
				self.atm.global_pars["vmac"] = np.array([self.atm.vmac])

			#--- read node parameters
			for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi"]:
				self.read_node_parameters(parameter, text)

			if globin.mode==3:
				#--- line parameters to be fit
				line_pars_path = find_value_by_key("line_parameters", text, "optional")

				if line_pars_path:
					# if we provided line parameters for fit, read those parameters
					lines_to_fit = globin.read_init_line_parameters(line_pars_path)

					# get log(gf) parameters from line list
					self.atm.global_pars["loggf"] = [line.loggf for line in lines_to_fit if line.loggf is not None]
					self.atm.line_no["loggf"] = [line.lineNo for line in lines_to_fit if line.loggf is not None]
					loggf_min = [line.loggf_min for line in lines_to_fit if line.loggf is not None]
					loggf_max = [line.loggf_max for line in lines_to_fit if line.loggf is not None]
					globin.limit_values["loggf"] = np.vstack((loggf_min, loggf_max)).T

					# get dlam parameters from lines list
					self.atm.global_pars["dlam"] = [line.dlam for line in lines_to_fit if line.dlam is not None]
					self.atm.line_no["dlam"] = [line.lineNo for line in lines_to_fit if line.dlam is not None]
					dlam_min = [line.dlam_min for line in lines_to_fit if line.dlam is not None]
					dlam_max = [line.dlam_max for line in lines_to_fit if line.dlam is not None]
					globin.limit_values["dlam"] = np.vstack((dlam_min, dlam_max)).T

					#--- Kurucz line list for given spectral region
					if RLK_linelist_path:
						# get path to line list which has original / expected values (will not be changed during execution)
						linelist_path = find_value_by_key("linelist", text, "required")
						# RLK_text_lines --> list of lines with Kurucz line format (needed for outputing atomic line list later)
						# RLK_lines --> Kurucz lines found in given line list (we use them to simply output log(gf) or dlam parameter during inversion)
						self.RLK_text_lines, self.RLK_lines = globin.read_RLK_lines(linelist_path)

						# go through RLK file and find the uncommented line
						# with path to atomic line files of Kurucz format
						lines = open(RLK_linelist_path, "r").readlines()
						for line in lines:
							line = line.rstrip("\n").strip(" ")
							# find the first uncommented line and break
							if line[0]!=globin.COMMENT_CHAR:
								self.RLK_path = line
								break

						# write down initial atomic lines values
						self.write_line_parameters(self.atm.global_pars["loggf"], self.atm.line_no["loggf"],
												   self.atm.global_pars["dlam"], self.atm.line_no["dlam"])
					else:
						print("No path to kurucz.input file.")
						# print("There is no Kurucz line list file to write to.")
						# print("If you want to invert for line parameters, you need to set")
						# print("path to file where Kurucz line lists are (kurucz.input file).")
						sys.exit()

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
			# opacity fudge coefficients

		#--- if we have more threads than atmospheres, reduce the number of used threads
		if globin.n_thread > self.atm.nx*self.atm.ny:
			globin.n_thread = self.atm.nx*self.atm.ny
			print(f"\nWarning: reduced the number of threads to {globin.n_thread}.\n")
		globin.pool = mp.Pool(globin.n_thread)

	def write_line_parameters(self, loggf_val, loggf_no, dlam_val, dlam_no):
		"""
		Write out full Kurucz line list for all parameters.
		"""
		out = open(self.RLK_path, "w")
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
		"""
		Write out parameter for one given line and parameter.

		Used when we are computing RFs.
		"""
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

	def read_node_parameters(self, parameter, text):
		"""
		For a given parameter read from input file node positions, values and 
		parameter mask.

		Parameters:
		---------------
		parameter : str
			name of the parameter for which we are loading in node parameters.
		text : str
			loaded input file string from which we are searching for the node keywords.
		"""
		nodes = find_value_by_key(f"nodes_{parameter}", text, "optional")
		values = find_value_by_key(f"nodes_{parameter}_values", text, "optional")
		mask = find_value_by_key(f"nodes_{parameter}_mask", text, "optional")
		
		if (nodes is not None) and (values is not None):
			self.atm.nodes[parameter] = [float(item) for item in nodes.split(",")]
			
			values = [float(item) for item in values.split(",")]
			if len(values)!=len(self.atm.nodes[parameter]):
				sys.exit(f"Number of nodes and values for {parameter} are not the same!")

			try:	
				matrix = np.zeros((self.atm.nx, self.atm.ny, len(self.atm.nodes[parameter])), dtype=np.float64)
				matrix[:,:] = copy.deepcopy(values)
				if parameter=="mag":
					self.atm.values[parameter] = copy.deepcopy(matrix) / 1e4
				elif parameter=="gamma":
					matrix *= np.pi/180
					self.atm.values[parameter] = copy.deepcopy(matrix)
				elif parameter=="chi":
					matrix *= np.pi/180
					self.atm.values[parameter] = copy.deepcopy(matrix)
				else:
					self.atm.values[parameter] = copy.deepcopy(matrix)
				
				if mask is None:
					self.atm.mask[parameter] = np.ones(len(self.atm.nodes[parameter]))
				else:
					mask = [float(item) for item in mask.split(",")]
					self.atm.mask[parameter] = np.array(mask)
			except:
				print(f"Can not store node values for parameter '{parameter}'.")
				print("  Must read first observation file.")
				sys.exit()

def slice_line(line, dtype=float):
    # remove 'new line' character
    line = line.rstrip("\n")
    # split line data based on 'space' separation
    line = line.split(" ")
    # filter out empty entries and convert to list
    lista = list(filter(None, line))
    # map read values into given data type
    lista = map(dtype, lista)
    # return list of values
    return list(lista)

def read_node_atmosphere(fpath):
    parameters = ["temp", "vz", "vmic", "mag", "gamma", "chi"]

    lines = open(fpath, "r").readlines()

    nx, ny = slice_line(lines[0], int)

    # number of data for given parameter
    # we have nx*ny data points
    # and 1 line for the name of the variable
    # and 1 line for the node positions
    nlines = nx*ny + 2

    atmos = Atmosphere(nx=nx, ny=ny)

    i_ = 2
    for parID in range(6):
        parameter = parameters[parID]
        if i_<len(lines):
            if lines[i_].rstrip("\n")==parameter:
                nodes = slice_line(lines[i_+1])
                atmos.nodes[parameter] = np.array(nodes)
                num_nodes = len(nodes)
                atmos.values[parameter] = np.zeros((nx, ny, num_nodes))

                for j_ in range(nx*ny):
                    idx = j_//ny
                    idy = j_%ny

                    temps = slice_line(lines[i_+2+j_])
                    atmos.values[parameter][idx, idy] = np.array(temps)

                if parameter=="mag":
                    atmos.values[parameter] /= 1e4 # [G --> T]
                if parameter=="gamma" or parameter=="chi":
                    atmos.values[parameter] *= np.pi/180 # [deg --> rad]

                i_ += nlines+1

    return atmos