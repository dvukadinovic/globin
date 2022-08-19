import os
import sys
import numpy as np
import multiprocessing as mp
import re
import copy
import subprocess as sp
from scipy.interpolate import interp1d
from astropy.io import fits
from scipy.interpolate import splrep, splev

import matplotlib.pyplot as plt

from .atoms import read_RLK_lines, read_init_line_parameters
from .atmos import Atmosphere
from .spec import Observation
from .rh import write_wavs
from .utils import _set_keyword, _slice_line, construct_atmosphere_from_nodes
from .container import Globin

import globin

class RHInput(object):
	"""
	Container for RH input fields and methods.
	"""
	def __init__(self):
		pass

	def set_keyword(self, key):
		pass

class InputData(object):
	def __init__(self):
		self.n_thread = 1

	def read_input_files(self, globin_input_name, rh_input_name):
		"""
		Read input files for globin and RH.

		We assume that parameters (in both files) are given in format:
			key = value

		Commented lines begin with symbol '#'.

		Parameters:
		---------------
		globin_input_name : str
			path to file that contains input parameters for globin
		rh_input_name : str
			path to 'keyword.input' file for RH parameters
		"""
		path = os.path.dirname(__file__)
		self.falc = Atmosphere(fpath=f"{path}/data/falc_multi.atmos", atm_type="multi")

		# temperature interpolation
		self.falc.temp_tck = splrep(self.falc.data[0,0,0],self.falc.data[0,0,1])
		self.globin_input_name = globin_input_name
		self.rh_input_name = rh_input_name

		# make runs directory if not existing
		# here we store all runs with different run_name
		if not os.path.exists("runs"):
			os.mkdir("runs")

		# make directory for specified run with provided 'run_name'
		if not os.path.exists(f"runs/{self.run_name}"):
			os.mkdir(f"runs/{self.run_name}")

		# copy all RH input files into run_name directory
		sp.run(f"cp *.input runs/{self.run_name}",
			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

		#--- get parameters from RH input file
		text = open(self.rh_input_name, "r").read()
		self.keyword_input = text

		wave_file_path = _find_value_by_key("WAVETABLE", self.keyword_input, "required")
		wave_file_path = wave_file_path.split("/")[-1]
		# obj.rh_spec_name = _find_value_by_key("SPECTRUM_OUTPUT", obj.keyword_input, "default", "spectrum.out")
		self.solve_ne = _find_value_by_key("SOLVE_NE", self.keyword_input, "optional")
		# obj.hydrostatic = _find_value_by_key("HYDROSTATIC", obj.keyword_input, "optional")
		# obj.kurucz_input_fname = _find_value_by_key("KURUCZ_DATA", obj.keyword_input, "required")
		# obj.rf_file_path = _find_value_by_key("RF_OUTPUT", obj.keyword_input, "default", "rfs.out")
		# obj.stokes_mode = _find_value_by_key("STOKES_MODE", obj.keyword_input, "default", "NO_STOKES")
		# self.of_mode = _find_value_by_key("OPACITY_FUDGE", self.keyword_input, "default", False)
		# if self.of_mode:
		# 	self.of_mode = True

		#--- get parameters from globin input file
		text = open(self.globin_input_name, "r").read()
		self.parameters_input = text

		# key used to make debug files/checks during synthesis/inversion
		debug = _find_value_by_key("debug", self.parameters_input, "optional")
		if debug is not None:
			if debug.lower()=="true":
				self.debug = True
			elif debug.lower()=="false":
				self.debug = False
		else:
			self.debug = False

		self.n_thread = _find_value_by_key("n_thread", self.parameters_input, "default", 1, conversion=int)
		self.mode = _find_value_by_key("mode", self.parameters_input, "required", conversion=int)
		norm = _find_value_by_key("norm", self.parameters_input, "optional")
		if norm is not None:
			if norm.lower()=="true":
				self.norm = True
			elif norm.lower()=="false":
				self.norm = False
		else:
			self.norm = False

		mean = _find_value_by_key("mean", self.parameters_input, "optional")
		if mean is not None:
			if mean.lower()=="true":
				self.mean = True
			elif mean.lower()=="false":
				self.mean = False
		else:
			self.mean = False

		if self.mean:
			mac_vel = _find_value_by_key("mac_vel", self.parameters_input, "required")
			self.mac_vel = [float(item) for item in mac_vel.split(",")]

			items = _find_value_by_key("filling_factor", self.parameters_input, "required")
			self.filling_factor = [float(item) for item in items.split(",")]

		# path to RH main folder
		rh_path = _find_value_by_key("rh_path", self.parameters_input, "required")
		if rh_path.rstrip("\n")[-1]=="/":
			rh_path = rh_path.rstrip("/")
		self.rh_path = rh_path

		# flag for HSE computation; by default we do HSE
		#obj.hydrostatic = _find_value_by_key("hydrostatic", obj.parameters_input, "default", 1, conversion=int)
		
		#--- get wavelength range and save it to file ('wave_file_path')
		self.lmin = _find_value_by_key("wave_min", self.parameters_input, "optional", conversion=float)
		self.lmax = _find_value_by_key("wave_max", self.parameters_input, "optional", conversion=float)
		self.step = _find_value_by_key("wave_step", self.parameters_input, "optional", conversion=float)
		self.interpolate_obs = False
		if (self.step is None) or (self.lmin is None) or (self.lmax is None):
			wave_grid_path = _find_value_by_key("wave_grid", self.parameters_input, "required")
			wavetable = np.loadtxt(wave_grid_path)
			self.lmin = min(wavetable)
			self.lmax = max(wavetable)
			self.step = wavetable[1] - wavetable[0]
			# self.interpolate_obs = True
		else:
			self.lmin /= 10
			self.lmax /= 10
			self.step /= 10
			wavetable = np.arange(self.lmin, self.lmax+self.step, self.step)
		
		self.wavelength_air = copy.deepcopy(wavetable)
		self.wavelength_vacuum = write_wavs(wavetable, wave_file_path)
		# self.Globin.wavelength_vacuum = wavetable
		# self.Globin.RH.set_wavetable(self.Globin.wavelength_vacuum)

		# common parameters for all modes
		atm_range = get_atmosphere_range(self.parameters_input)
		logtau_top = _find_value_by_key("logtau_top", self.parameters_input, "default", -6,float)
		logtau_bot = _find_value_by_key("logtau_bot", self.parameters_input, "default", 1, float)
		logtau_step = _find_value_by_key("logtau_step", self.parameters_input, "default", 0.1, float)
		self.noise = _find_value_by_key("noise", self.parameters_input, "default", 1e-3, float)
		atm_type = _find_value_by_key("atm_type", self.parameters_input, "default", "multi", str)
		atm_type = atm_type.lower()
		self.atm_scale = _find_value_by_key("atm_scale", self.parameters_input, "default", "tau", str)

		#--- read Opacity Fudge (OF) data
		self.of_mode = _find_value_by_key("of_mode", self.parameters_input, "default", -1, int)
		# of_mode:
		#   0    -- use if only for synthesis
		#   1    -- invert for it in pixel-by-pixel manner
		#   else -- OF is not applyed
		self.do_fudge = False
		if (self.of_mode==0) or (self.of_mode==1):
			self.do_fudge = True
			of_file_path = _find_value_by_key("of_file", self.parameters_input, "default", None, str)
			self.of_scatter = _find_value_by_key("of_scatter", self.parameters_input, "default", 0, int)
			if of_file_path:
				of_num, of_wave, of_value = read_OF_data(of_file_path)

		# get the name of the input line list
		self.linelist_name = _find_value_by_key("linelist", self.parameters_input, "required")
		# obj.linelist_name = linelist_path.split("/")[-1]
		# out = sp.run(f"cp {linelist_path} runs/{obj.wd}/{obj.linelist_name}",
		# 			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
		# if out.returncode!=0:
		# 	print(str(out.stdout, "utf-8"))

		#--- read data for different modus operandi
		if self.mode<=0:
			self.read_mode_0(atm_range, atm_type, logtau_top, logtau_bot, logtau_step)
		elif self.mode>=1:
			self.read_inversion_base(atm_range, atm_type, logtau_top, logtau_bot, logtau_step)			
			if self.mode==2:
				self.read_mode_2()
			elif self.mode==3:
				self.read_mode_3()

			#--- determine number of local and global parameters
			self.atmosphere.n_local_pars = 0
			for parameter in self.atmosphere.nodes:
				self.atmosphere.n_local_pars += len(self.atmosphere.nodes[parameter])

			if (self.do_fudge):
				self.atmosphere.n_local_pars += of_num

			self.atmosphere.n_global_pars = 0
			for parameter in self.atmosphere.global_pars:
				self.atmosphere.n_global_pars += self.atmosphere.global_pars[parameter].shape[-1]
		else:
			print("  Negative mode not supported. Soon to be RF calculation.")
			sys.exit()

		if not "loggf" in self.atmosphere.global_pars:
			self.atmosphere.global_pars["loggf"] = np.array([], dtype=np.float64)
			self.atmosphere.line_no["loggf"] = np.array([], dtype=np.int32)
		if not "dlam" in self.atmosphere.global_pars:
			self.atmosphere.global_pars["dlam"] = np.array([], dtype=np.float64)
			self.atmosphere.line_no["dlam"] = np.array([], dtype=np.int32)

		#--- if we have more threads than atmospheres, reduce the number of used threads
		if self.n_thread > self.atmosphere.nx*self.atmosphere.ny:
			self.n_thread = self.atmosphere.nx*self.atmosphere.ny
			print(f"  Warning: reduced the number of threads to {self.n_thread}.")

		
		#--- set OF data in atmosphere
		if self.do_fudge:
			# if we have 1D OF values, set equal values in all pixels
			if of_value.ndim==1:
				of_value = np.repeat(of_value[np.newaxis, :], self.atmosphere.nx, axis=0)
				of_value = np.repeat(of_value[:, np.newaxis, :], self.atmosphere.ny, axis=1)
			
			self.atmosphere.do_fudge = 1
			self.atmosphere.of_num = of_num
			self.atmosphere.nodes["of"] = of_wave
			self.atmosphere.values["of"] = of_value
			self.atmosphere.parameter_scale["of"] = np.ones((self.atmosphere.nx, self.atmosphere.ny, self.atmosphere.of_num))

			# create arrays to be passed to RH for synthesis
			self.atmosphere.of_scatter = self.of_scatter
			self.atmosphere.make_OF_table(self.wavelength_vacuum)
		else:
			self.atmosphere.do_fudge = 0
			self.atmosphere.fudge_lam = np.array([], dtype=np.float64)
			self.atmosphere.fudge = np.ones((self.atmosphere.nx, self.atmosphere.ny, 3, 0), 
											dtype=np.float64)

		if self.mean:
			if len(self.mac_vel)==1:
				vmac = self.mac_vel[0]
				self.mac_vel = np.ones(self.atmosphere.nx * self.atmosphere.ny) * vmac

				ff = self.filling_factor[0]
				self.filling_factor = np.ones(self.atmosphere.nx * self.atmosphere.ny) * ff

		# #--- for each thread make working directory inside rh/rhf1d directory
		# for pid in range(self.n_thread):
		# 	if not os.path.exists(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid+1}"):
		# 		os.mkdir(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid+1}")

		idx,idy = np.meshgrid(np.arange(self.atmosphere.nx), np.arange(self.atmosphere.ny))
		self.atmosphere.idx_meshgrid = idx.flatten()
		self.atmosphere.idy_meshgrid = idy.flatten()

		self.atmosphere.ids_tuple = list(zip(self.atmosphere.idx_meshgrid, self.atmosphere.idy_meshgrid))
		# for idx in range(obj.atmosphere.nx):
		# 	for idy in range(globin.atm.ny):
		# 		fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
		# 		globin.atm.atm_name_list.append(fpath)

		if self.mode>=1:
			#--- debugging variables initialization
			if self.debug:
				Npar = self.atmosphere.n_local_pars + self.atmosphere.n_global_pars
				self.rf_debug = np.zeros((self.atmosphere.nx, self.atmosphere.ny, self.max_iter, Npar, len(self.wavelength_air), 4))

				elements = []
				for parameter in self.atmosphere.nodes:
					aux = np.zeros((self.max_iter, self.atmosphere.nx, self.atmosphere.ny, len(self.atmosphere.nodes[parameter])))
					elements.append((parameter, aux))
				self.atmos_debug = dict(elements)

		#--- missing parameters
		# instrument broadening: R or instrument profile provided
		# strailight contribution

	def read_mode_0(self, atm_range, atm_type, logtau_top, logtau_bot, logtau_step):
		""" 
		Get parameters for synthesis.
		"""

		#--- default parameters
		self.output_spectra_path = _find_value_by_key("spectrum", self.parameters_input, "default", "spectrum.fits")
		vmac = _find_value_by_key("vmac", self.parameters_input, "default", 0, float)

		#--- required parameters
		path_to_atmosphere = _find_value_by_key("cube_atmosphere", self.parameters_input, "optional")
		if path_to_atmosphere is None:
			node_atmosphere_path = _find_value_by_key("node_atmosphere", self.parameters_input, "optional")
			if node_atmosphere_path is None:
				self.atmosphere = globin.falc
			else:
				self.atmosphere = construct_atmosphere_from_nodes(node_atmosphere_path, atm_range)
		else:
			self.atmosphere = Atmosphere(fpath=path_to_atmosphere, atm_type=atm_type, atm_range=atm_range,
							logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)
		self.atmosphere.vmac = np.abs(vmac) # [km/s]

		# reference atmosphere is the same as input one in synthesis mode
		# self.reference_atmosphere = copy.deepcopy(self.atmosphere)

	def read_inversion_base(self, atm_range, atm_type, logtau_top, logtau_bot, logtau_step):
		# interpolation degree for Bezier polynomial
		self.interp_degree = _find_value_by_key("interp_degree", self.parameters_input, "default", 3, int)
		self.svd_tolerance = _find_value_by_key("svd_tolerance", self.parameters_input, "default", 1e-4, float)

		#--- default parameters
		self.marq_lambda = _find_value_by_key("marq_lambda", self.parameters_input, "default", 1e-3, float)
		self.max_iter = _find_value_by_key("max_iter", self.parameters_input, "default", 30, int)
		self.chi2_tolerance = _find_value_by_key("chi2_tolerance", self.parameters_input, "default", 1e-2, float)
		self.ncycle = _find_value_by_key("ncycle", self.parameters_input, "default", 1, int)
		self.rf_type = _find_value_by_key("rf_type", self.parameters_input, "default", "node", str)
		self.weight_type = _find_value_by_key("weight_type", self.parameters_input, "default", None, str)
		values = _find_value_by_key("weights", self.parameters_input, "default", np.array([1,1,1,1], dtype=np.float64))
		if type(values)==str:
			values = values.split(",")
			self.weights = np.array([float(item) for item in values], dtype=np.float64)
		vmac = _find_value_by_key("vmac", self.parameters_input, "default", default_val=0, conversion=float)

		#--- required parameters
		path_to_observations = _find_value_by_key("observation", self.parameters_input, "required")
		self.observation = Observation(path_to_observations, obs_range=atm_range)
		if self.interpolate_obs or (not np.array_equal(self.observation.wavelength, self.wavelength_air)):
			self.observation.interpolate(self.wavelength_air)

		# initialize container for atmosphere which we invert
		# self.atmosphere = Atmosphere(nx=self.observation.nx, ny=self.observation.ny, 
		# 	logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)# atm_range=atm_range)
		self.atmosphere = Atmosphere(nx=self.observation.nx, ny=self.observation.ny)# atm_range=atm_range)

		#--- optional parameters
		"""
		cube_atmosphere -- refrence atmosphere (usually 3D) used for testing purpose; it is sliced as observations

		reference_atmosphere -- 1D reference atmosphere (FALC, HSRASP, HOLMUL,...) that is used as a reference one
								for inversion; if 'norm' is True, we use this atmosphere to compute the continuum
								intensity.

		If none of the above is given, we assume that FALC is reference atmosphere.
		"""
		path_to_atmosphere = _find_value_by_key("cube_atmosphere", self.parameters_input, "optional")
		if path_to_atmosphere is not None:
			self.reference_atmosphere = Atmosphere(path_to_atmosphere, atm_type=atm_type, atm_range=atm_range,
						logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)
		else:
			path_to_atmosphere = _find_value_by_key("reference_atmosphere", self.parameters_input, "optional")
			if path_to_atmosphere is not None:
				self.reference_atmosphere = Atmosphere(path_to_atmosphere, atm_type=atm_type, atm_range=[0,None,0,None],
							logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)
			else:
				self.reference_atmosphere = globin.falc

		#--- initialize invert atmosphere data from reference atmosphere
		self.atmosphere.interpolate_atmosphere(self.reference_atmosphere.data[0,0,0], self.reference_atmosphere.data, atm_range)

		fpath = _find_value_by_key("rf_weights", self.parameters_input, "optional")
		self.wavs_weight = np.ones((self.atmosphere.nx, self.atmosphere.ny, len(self.wavelength_air),4))
		if fpath is not None:
			lam, wI, wQ, wU, wV = np.loadtxt(fpath, unpack=True)
			# !!! Lenghts can be the same, but not the values in arrays. Needs to be changed.
			if len(lam)==len(self.wavelength_air):
				self.wavs_weight[...,0] = wI
				self.wavs_weight[...,1] = wQ
				self.wavs_weight[...,2] = wU
				self.wavs_weight[...,3] = wV
			else:
				self.wavs_weight[...,0] = interp1d(lam, wI)(self.wavelength)
				self.wavs_weight[...,1] = interp1d(lam, wQ)(self.wavelength)
				self.wavs_weight[...,2] = interp1d(lam, wU)(self.wavelength)
				self.wavs_weight[...,3] = interp1d(lam, wV)(self.wavelength)
		
		# standard deviation of Gaussian kernel for macro broadening
		self.atmosphere.vmac = vmac # [km/s]

		# if macro-turbulent velocity is negative, we fit it
		if self.atmosphere.vmac<0:
			# check if initial macro veclocity is larger than the step size in wavelength
			vmac = np.abs(vmac)
			kernel_sigma = vmac*1e3 / globin.LIGHT_SPEED * (self.lmin + self.lmax)*0.5 / self.step
			if kernel_sigma<0.5:
				vmac = 0.5 * globin.LIGHT_SPEED / ((self.lmin + self.lmax)*0.5) * self.step
				vmac /= 1e3
				self.limit_values["vmac"][0] = vmac
			
			self.atmosphere.vmac = abs(vmac)
			self.atmosphere.global_pars["vmac"] = np.array([self.atmosphere.vmac])
			self.atmosphere.parameter_scale["vmac"] = 1
		self.reference_atmosphere.vmac = abs(vmac)

		#--- read initial node parameter values	
		fpath = _find_value_by_key("initial_atmosphere", self.parameters_input, "optional")
		if fpath is not None:
			# read node parameters from .fits file that is inverted atmosphere
			# from older inversion run
			init_atmosphere = read_inverted_atmosphere(fpath, atm_range)
			self.atmosphere.nodes = init_atmosphere.nodes
			self.atmosphere.values = init_atmosphere.values
			self.atmosphere.mask = init_atmosphere.mask
			if (self.atmosphere.nx!=self.observation.nx) or (self.atmosphere.ny!=self.observation.ny):
				print("--> Error in input.read_inverted_atmosphere()")
				print("    initial atmosphere does not have same dimensions")
				print("    as observations:")
				print(f"    -- atm = ({self.atmosphere.nx},{self.atmosphere.ny})")
				print(f"    -- obs = ({self.observation.nx},{self.observation.ny})")
				sys.exit()
		else:
			# read node parameters from .input file
			for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi"]:
				self.read_node_parameters(parameter, self.parameters_input)

		self.atmosphere.hydrostatic = False
		if "temp" in self.atmosphere.nodes:
			self.atmosphere.hydrostatic = True

		#--- setup ne and nH
		fun = interp1d(self.reference_atmosphere.logtau, self.reference_atmosphere.data[0,0,2], kind=3)
		self.atmosphere.data[:,:,2,:] = fun(self.atmosphere.logtau)

		for idp in range(6):
			fun = interp1d(self.reference_atmosphere.logtau, self.reference_atmosphere.data[0,0,idp+8], kind=3)
			self.atmosphere.data[:,:,idp+8,:] = fun(self.atmosphere.logtau)

		#--- initialize the vz, mag and azimuth based on CoG and WFA methods (optional)
		fpath = _find_value_by_key("lines2atm", self.parameters_input, "optional")
		if fpath:
			initialize_atmos_pars(self.atmosphere, self.observation, fpath, norm=False)

	def read_mode_2(self):
		#--- Kurucz line list for given spectral region
		self.RLK_lines_text, self.RLK_lines = read_RLK_lines(self.linelist_name)

		#--- line parameters to be fit
		line_pars_path = _find_value_by_key("line_parameters", self.parameters_input, "optional")

		if line_pars_path:
			# if we provided line parameters for fit, read those parameters
			lines_to_fit = read_init_line_parameters(line_pars_path)

			# get log(gf) parameters from line list
			aux_values = [line.loggf for line in lines_to_fit if line.loggf is not None]
			aux_lineNo = [line.lineNo for line in lines_to_fit if line.loggf is not None]
			loggf_min = [line.loggf_min for line in lines_to_fit if line.loggf is not None]
			loggf_max = [line.loggf_max for line in lines_to_fit if line.loggf is not None]
			self.atmosphere.limit_values["loggf"] = np.vstack((loggf_min, loggf_max)).T
			self.atmosphere.parameter_scale["loggf"] = np.ones((self.atmosphere.nx, self.atmosphere.ny, len(aux_values)))

			self.atmosphere.global_pars["loggf"] = np.zeros((self.atmosphere.nx, self.atmosphere.ny, len(aux_values)), dtype=np.float64)
			self.atmosphere.line_no["loggf"] = np.zeros((len(aux_lineNo)), dtype=np.int32)

			self.atmosphere.global_pars["loggf"][:,:] = aux_values
			self.atmosphere.line_no["loggf"][:] = aux_lineNo

			# get dlam parameters from lines list
			aux_values = [line.dlam for line in lines_to_fit if line.dlam is not None]
			aux_lineNo = [line.lineNo for line in lines_to_fit if line.dlam is not None]
			dlam_min = [line.dlam_min for line in lines_to_fit if line.dlam is not None]
			dlam_max = [line.dlam_max for line in lines_to_fit if line.dlam is not None]
			self.atmosphere.limit_values["dlam"] = np.vstack((dlam_min, dlam_max)).T
			self.atmosphere.parameter_scale["dlam"] = np.ones((self.atmosphere.nx, self.atmosphere.ny, len(aux_values)))

			self.atmosphere.global_pars["dlam"] = np.zeros((self.atmosphere.nx, self.atmosphere.ny, len(aux_values)), dtype=np.float64)
			self.atmosphere.line_no["dlam"] = np.zeros((len(aux_lineNo)), dtype=np.int32)

			self.atmosphere.global_pars["dlam"][:,:] = aux_values
			self.atmosphere.line_no["dlam"][:] = aux_lineNo

			# write these data into files

			# make list of line lists paths (aka names)
			# obj.atmosphere.line_lists_path = []
			# for idx in range(globin.atm.nx):
			# 	for idy in range(globin.atm.ny):
			# 		fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
			# 		globin.atm.line_lists_path.append(fpath)

			# 		write_line_parameters(fpath,
			# 							   globin.atm.global_pars["loggf"][idx,idy], globin.atm.line_no["loggf"],
			# 							   globin.atm.global_pars["dlam"][idx,idy], globin.atm.line_no["dlam"])
		else:
			print("No atomic parameters to fit. You sure?\n")

	def read_mode_3(self):
		#--- Kurucz line list for given spectral region
		self.RLK_lines_text, self.RLK_lines = read_RLK_lines(self.linelist_name)

		#--- line parameters to be fit
		line_pars_path = _find_value_by_key("line_parameters", self.parameters_input, "optional")

		if line_pars_path:
			# if we provided line parameters for fit, read those parameters
			lines_to_fit = read_init_line_parameters(line_pars_path)

			# get log(gf) parameters from line list
			aux_values = [line.loggf for line in lines_to_fit if line.loggf is not None]
			aux_lineNo = [line.lineNo for line in lines_to_fit if line.loggf is not None]
			loggf_min = [line.loggf_min for line in lines_to_fit if line.loggf is not None]
			loggf_max = [line.loggf_max for line in lines_to_fit if line.loggf is not None]
			self.atmosphere.limit_values["loggf"] = np.vstack((loggf_min, loggf_max)).T
			self.atmosphere.parameter_scale["loggf"] = np.ones((1,1,len(aux_values)))

			self.atmosphere.global_pars["loggf"] = np.zeros((1,1,len(aux_values)), dtype=np.float64)
			self.atmosphere.line_no["loggf"] = np.zeros((len(aux_lineNo)), dtype=np.int32)

			self.atmosphere.global_pars["loggf"][0,0] = aux_values
			self.atmosphere.line_no["loggf"][:] = aux_lineNo

			# get dlam parameters from lines list
			aux_values = [line.dlam for line in lines_to_fit if line.dlam is not None]
			aux_lineNo = [line.lineNo for line in lines_to_fit if line.dlam is not None]
			dlam_min = [line.dlam_min for line in lines_to_fit if line.dlam is not None]
			dlam_max = [line.dlam_max for line in lines_to_fit if line.dlam is not None]
			self.atmosphere.limit_values["dlam"] = np.vstack((dlam_min, dlam_max)).T
			self.atmosphere.parameter_scale["dlam"] = np.ones((1,1,len(aux_values)))

			self.atmosphere.global_pars["dlam"] = np.zeros((1,1,len(aux_values)), dtype=np.float64)
			self.atmosphere.line_no["dlam"] = np.zeros((len(aux_lineNo)), dtype=np.int32)

			self.atmosphere.global_pars["dlam"][0,0] = aux_values
			self.atmosphere.line_no["dlam"][:] = aux_lineNo

			# write down initial atomic lines values
			# globin.write_line_parameters(obj.atmosphere.line_lists_path[0],
			# 						   globin.atm.global_pars["loggf"][0,0], globin.atm.line_no["loggf"],
			# 						   globin.atm.global_pars["dlam"][0,0], globin.atm.line_no["dlam"])
		else:
			print("No atomic parameters to fit. You sure?\n")

	def read_node_parameters(self, parameter, text):
		"""
		For a given parameter read from input file node positions, values and 
		parameter mask (optional).

		Parameters:
		---------------
		parameter : str
			name of the parameter for which we are loading in node parameters.
		text : str
			loaded input file string from which we are searching for the node keywords.
		"""
		atmosphere = self.atmosphere

		nodes = _find_value_by_key(f"nodes_{parameter}", text, "optional")
		values = _find_value_by_key(f"nodes_{parameter}_values", text, "optional")
		mask = _find_value_by_key(f"nodes_{parameter}_mask", text, "optional")
		
		if (nodes is not None) and (values is not None):
			atmosphere.nodes[parameter] = [float(item) for item in nodes.split(",")]
			
			values = [float(item) for item in values.split(",")]
			if len(values)!=len(atmosphere.nodes[parameter]):
				sys.exit(f"Number of nodes and values for {parameter} are not the same!")

			matrix = np.zeros((atmosphere.nx, atmosphere.ny, len(atmosphere.nodes[parameter])), dtype=np.float64)
			matrix[:,:] = copy.deepcopy(values)
			if parameter=="gamma":
				matrix *= np.pi/180
				# atmosphere.values[parameter] = np.tan(matrix/2)
				# atmosphere.values[parameter] = np.cos(matrix)
				atmosphere.values[parameter] = matrix
			elif parameter=="chi":
				matrix *= np.pi/180
				# atmosphere.values[parameter] = np.tan(matrix/4)
				# atmosphere.values[parameter] = np.cos(matrix)
				atmosphere.values[parameter] = matrix
			else:
				atmosphere.values[parameter] = matrix
			
			if mask is None:
				atmosphere.mask[parameter] = np.ones(len(atmosphere.nodes[parameter]))
			else:
				mask = [float(item) for item in mask.split(",")]
				atmosphere.mask[parameter] = np.array(mask)

			atmosphere.parameter_scale[parameter] = np.ones((atmosphere.nx, atmosphere.ny, len(atmosphere.nodes[parameter])))

#--- pattern search with regular expressions
pattern = lambda keyword: re.compile(f"^[^#\n]*({keyword})\s*=\s*(.*)", re.MULTILINE)

def _find_value_by_key(key, text, key_type, default_val=None, conversion=str):
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
		value = value.replace(" ", "")
		return conversion(value)
	else:
		if key_type=="required":
			print("--> Error in input.read_input_files()")
			print(f"    We are missing keyword '{key}' in input file.")
			sys.exit()
		elif key_type=="default":
			return default_val
		elif key_type=="optional":
			return None

def get_atmosphere_range(parameters_input):
	#--- determine which observations from cube to take into consideration
	aux = _find_value_by_key("range", parameters_input, "default", [1,None,1,None])
	atm_range = []
	if type(aux)==str:
		for item in aux.split(","):
			if item is None or int(item)==-1:
				atm_range.append(None)
			elif item is not None:
				atm_range.append(int(item))
		if atm_range[1] is not None:
			if atm_range[1]<atm_range[0]:
				print("--> Error in input.read_input_files()")
				print("    xmax smaller than xmin.")
				sys.exit()
		if atm_range[3] is not None:
			if atm_range[3]<atm_range[2]:
				print("--> Error in input.read_input_files()")
				print("    ymax smaller than ymin.")
				sys.exit()
		if atm_range[0]<1:
			print("--> Error in input.read_input_files()")
			print("    xmin is lower than 1.")
			sys.exit()
		if atm_range[2]<1:
			print("--> Error in input.read_input_files()")
			print("    ymin is lower than 1.")
			sys.exit()
	else:
		atm_range = aux
	# we count from zero, but let user count from 1
	atm_range[0] -= 1
	atm_range[2] -= 1

	return atm_range

def write_line_parameters(fpath, loggf_val, loggf_no, dlam_val, dlam_no):
	"""
	Write out full Kurucz line list for all parameters.
	"""
	out = open(fpath, "w")

	# because of Python memory handling
	# these two variables will be the same all the time!
	linelist = globin.RLK_lines_text

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
		character_list[0:10] = "{: 10.4f}".format(val/1e4 + globin.RLK_lines[no].lam0)
		linelist[no] = ''.join(character_list)

	out.writelines(linelist)
	out.close()

def write_line_par(fpath, par_val, par_no, parameter):
	"""
	Write out parameter for one given line and parameter.

	Used when we are computing RFs.
	"""
	out = open(fpath, "w")

	# because of Python memory handling
	# these two variables will be the same all the time!
	linelist = globin.RLK_lines_text

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
		character_list[0:10] = "{: 10.4f}".format(par_val/1e4 + globin.RLK_lines[par_no].lam0)
		linelist[par_no] = ''.join(character_list)

	out.writelines(linelist)
	out.close()

def read_node_atmosphere(fpath):
	"""
	Read atmosphere from file having the following structure:

	________________________________________________________
	nx, ny

	lgtau_top, lgtau_bot, lgtau_inc

	parameter_1
	logtau_node_1 logtau_node_2 ...
	T1 T2 ...
	T1 T2 ...
	.
	.
	.

	parameter_2
	logtau_node_1 logtau_node_2 ...
	T1 T2 ...
	T1 T2 ...
	.
	.
	.
	________________________________________________________

	Blank lines are mandatory!

	velocity is assumed to be in km/s, 
	magnetic field strength in G and angles in deg.

	Parameters:
	-----------
	fpath : string
		file path to atmosphere

	Return:
	-------
	atmos : globin.Atmosphere object
		atmosphere with nodes and values set from input file.

	"""
	parameters = ["temp", "vz", "vmic", "mag", "gamma", "chi"]

	lines = open(fpath, "r").readlines()

	nx, ny = _slice_line(lines[0], int)

	lgtop, lgbot, lginc = _slice_line(lines[2], float)

	# number of data for given parameter
	# 1 line for the name of the parameter
	# 1 line for the node positions
	# nx*ny lines for the values in the nodes
	nlines = nx*ny + 2

	logtau = np.arange(lgtop, lgbot+lginc, lginc)
	nz = len(logtau)

	atmos = Atmosphere(nx=nx, ny=ny, nz=nz)
	atmos.logtau = logtau
	atmos.data[:,:,0] = logtau

	idx, idy = np.meshgrid(np.arange(atmos.nx), np.arange(atmos.ny))
	atmos.idx_meshgrid = idx.flatten()
	atmos.idy_meshgrid = idy.flatten()
	atmos.ids_tuple = list(zip(atmos.idx_meshgrid, atmos.idy_meshgrid))

	path = os.path.dirname(__file__)
	falc = Atmosphere(fpath=f"{path}/data/falc_multi.atmos", atm_type="multi")

	# temperature interpolation
	atmos.temp_tck = splrep(falc.data[0,0,0], falc.data[0,0,1])
	atmos.interp_degree = 3
	
	# tck = splrep(falc.data[0,0,0], falc.data[0,0,2])
	# atmos.data[:,:,2] = splev(atmos.logtau, tck)
	# for i_ in range(6):
	# 	idp = 8 + i_
	# 	tck = splrep(falc.data[0,0,0], falc.data[0,0,idp])
	# 	atmos.data[:,:,idp] = splev(atmos.logtau, tck)

	# !!! added by hand..
	# globin.pool = mp.Pool(2)

	i_ = 4
	for parID in range(6):
		parameter = parameters[parID]
		if i_<len(lines):
			if lines[i_].rstrip("\n")==parameter:
				nodes = _slice_line(lines[i_+1])
				atmos.nodes[parameter] = np.array(nodes)
				num_nodes = len(nodes)
				atmos.values[parameter] = np.zeros((nx, ny, num_nodes))

				for j_ in range(nx*ny):
					idx = j_//ny
					idy = j_%ny

					temps = _slice_line(lines[i_+2+j_])
					atmos.values[parameter][idx, idy] = np.array(temps)

				if parameter=="gamma":
					# atmos.values[parameter] = np.tan(atmos.values[parameter]/2 * np.pi/180)
					# atmos.values[parameter] = np.cos(atmos.values[parameter] * np.pi/180)
					atmos.values[parameter] *= np.pi/180
				elif parameter=="chi":
					# atmos.values[parameter] = np.tan(atmos.values[parameter]/4 * np.pi/180)
					# atmos.values[parameter] = np.cos(atmos.values[parameter] * np.pi/180)
					atmos.values[parameter] *= np.pi/180 # [deg --> rad]

				i_ += nlines+1

	return atmos

def initialize_atmos_pars(atmos, obs_in, fpath, norm=True):
	"""
	We use Centre-of-Gravity method to initialize line-of-sight velocity and
	line-of-sight magnetic field strength.

	Weak-field approximation is used for computing the azimuth as: tan(2 chi) = U/Q.
	
	Parameters:
	-----------
	atmos : globin.Atmosphere()
		initial atmosphere which will contain the retrieved profiles. Before 
		entering here, we must have read which atmospheric parameters are to be
		inverted (atmos must contain .nodes and .values instances).
	obs : globin.Observation()
		the observed Stokes profiles which are to be inverted.
	fpath : string
		path to the file containing the information about spectral lines which
		we use to initialize the atmospheric parameters.
	"""
	from scipy.signal import argrelextrema
	from scipy.interpolate import splev, splrep
	import matplotlib.pyplot as plt

	def find_line_positions(x):
		inds = np.empty((atmos.nx, atmos.ny), dtype=np.int32)
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				try:
					inds[idx,idy] = argrelextrema(x[idx,idy], np.less, order=3)[0][0]
				except:
					# plt.plot(x[idx,idy])
					# plt.show()
					# print(idx,idy)
					# sys.exit()
					inds[idx,idy] = np.argmin(x[idx,idy])
		return inds

	obs = copy.deepcopy(obs_in)
	dlam = obs.wavelength[1] - obs.wavelength[0]
	wavs = obs.wavelength

	if norm:
		icont = obs.spec[:,:,0,0]
		obs.spec = np.einsum("ijkl,ij->ijkl", obs.spec, 1/icont)
		# if atmos.norm:
		# 	obs_in.norm()
		# else:
		# 	atmos.norm = True
		# 	obs_in.norm()
		# 	atmos.norm = False

	lines = open(fpath).readlines()
	lines = [line.rstrip("\n") for line in lines if (len(line.rstrip("\n"))>0) and ("#" not in line)]
	nl = len(lines)
	nl_mag = 0

	vlos = 0
	blos = 0
	azimuth = 0
	inclination = 0
	for line in lines:
		line = list(filter(None,line.split(" ")))

		init_mag = False
		if len(line)==2:
		   lam0, line_dlam = map(float,line)
		elif len(line)==7:
			init_mag = True
			lam0, line_dlam, geff, gl, gu, Jl, Ju = map(float, line)
			gs = gu + gl
			gd = gu - gl
			_s = Ju*(Ju+1) + Jl*(Jl+1)
			_d = Ju*(Ju+1) - Jl*(Jl+1)
			delta = 1/80*gd**2 * (16*_s - 7*_d**2 -4)
			if geff<0:
				geff = 1/2*gs + 1/4*gd*_d
			Geff = geff**2 - delta
		else:
			print("Error: input.initialize_atmos_pars():")
			print("  Wrong number of parameters for initializing")
			print("  the vertical velocity and magnetic field vector.")
			sys.exit()

		line_dlam /= 1e4
		lmin = lam0 - line_dlam
		lmax = lam0 + line_dlam

		ind_min = np.argmin(np.abs(wavs - lmin))
		ind_max = np.argmin(np.abs(wavs - lmax))

		# x = obs.wavelength[ind_min:ind_max]
		# si = obs.spec[0,5,ind_min:ind_max,0]
		# # lam0 = 401.6
		# lcog = np.sum(x*(1-si), axis=-1) / np.sum(1-si, axis=-1)
		# vlos += globin.LIGHT_SPEED * (1 - lcog/lam0) / 1e3
		# print(vlos)
		# sys.exit()

		# plt.plot(globin.ref_atm.logtau, globin.ref_atm.data[0,5,3])
		# plt.show()

		# # plt.plot(obs.spec[0,13,ind_min:ind_max,0])
		# plt.plot(obs.wavelength, obs.spec[0,5,:,0])
		# plt.show()
		# sys.exit()

		inds = find_line_positions(obs.spec[:,:,ind_min:ind_max,0])
		dd = int(line_dlam // dlam)
		ind_min += inds - dd
		ind_max = ind_min + 2*dd

		x = np.empty((atmos.nx, atmos.ny, 2*dd),dtype=np.float64)
		si = np.empty((atmos.nx, atmos.ny, 2*dd),dtype=np.float64)
		sq = np.empty((atmos.nx, atmos.ny, 2*dd),dtype=np.float64)
		su = np.empty((atmos.nx, atmos.ny, 2*dd),dtype=np.float64)
		sv = np.empty((atmos.nx, atmos.ny, 2*dd),dtype=np.float64)

		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				mmin = ind_min[idx,idy]
				mmax = ind_max[idx,idy]
				if mmin!=np.nan and mmax!=np.nan:
					x[idx,idy] = obs.wavelength[mmin:mmax]
					si[idx,idy] = obs.spec[idx,idy,mmin:mmax,0]
					sq[idx,idy] = obs.spec[idx,idy,mmin:mmax,1]
					su[idx,idy] = obs.spec[idx,idy,mmin:mmax,2]
					sv[idx,idy] = obs.spec[idx,idy,mmin:mmax,3]
				else:
					x[idx,idy] = np.nan
					si[idx,idy] = np.nan
					sq[idx,idy] = np.nan
					su[idx,idy] = np.nan
					sv[idx,idy] = np.nan
					# print(idx,idy)
					# print(mmin, mmax)
					
					# sys.exit()

		# for idx in range(atmos.nx):
		# 	for idy in range(atmos.ny):
		# 		lin_pol = np.sqrt(sq[idx,idy]**2 + su[idx,idy]**2)
		# 		plt.plot(x[idx,idy]-lam0, lin_pol)
		# 		# plt.plot(obs.wavelength, obs.spec[idx,idy,:,3])
		# 		# plt.axvline(obs.wavelength[ind_min[idx,idy]], color="black")
		# 		# plt.axvline(obs.wavelength[ind_max[idx,idy]], color="black")
		# 		plt.show()
		# sys.exit()

		#--- v_LOS initialization (CoG)
		if "vz" in atmos.nodes:
			lcog = np.sum(x*(1-si), axis=-1) / np.sum(1-si, axis=-1)
			vlos += globin.LIGHT_SPEED * (1 - lcog/lam0) / 1e3
			# vlos = np.repeat(vlos[..., np.newaxis], len(atmos.nodes["vz"]), axis=-1)
			# atmos.values["vz"] = vlos

		if init_mag:
			#--- azimuth initialization
			if "chi" in atmos.nodes:	
				_azimuth = np.arctan2(np.sum(su, axis=-1), np.sum(sq, axis=-1))# * 180/np.pi / 2
				_azimuth %= 2*np.pi
				azimuth += _azimuth
				# azimuth = np.mean(azimuth, axis=-1)
				# print(azimuth)
			
			#--- B + gamma initialization (CoG + WF)
			if "mag" in atmos.nodes:
				lamp = np.sum(x*(1-si-sv), axis=-1) / np.sum(1-si-sv, axis=-1)
				lamm = np.sum(x*(1-si+sv), axis=-1) / np.sum(1-si+sv, axis=-1)
				
				# idx, idy = 1,0
				# plt.plot(1-si[idx,idy]-sv[idx,idy])
				# plt.plot(1-si[idx,idy]+sv[idx,idy])
				# print(lamp[idx,idy], lamm[idx,idy])
				# plt.show()
				# sys.exit()

				if geff!=0:
					# C = 4*np.pi*globin.LIGHT_SPEED*globin.ELECTRON_MASS/globin.ELECTRON_CHARGE
					# _blos = (lamp - lamm)/2/lam0 / C / geff / (lam0*1e-9)
					C = 4.67e-13*(lam0*10)**2*geff
					_blos = (lamp - lamm)*10/2 / C
					blos += np.abs(_blos)
					nl_mag += 1

				# tck = splrep(x[0,0]*10, si[0,0])
				# si_der = splev(x[0,0]*10, tck, der=1)
				
				# blos_wf = -np.sum(sv[0,0]*si_der)/np.sum(si_der**2)/C
				# print(blos_wf, blos[0,0])

			#--- inclination initialization
			if "gamma" in atmos.nodes:
				ind_lam_wing = dd//2-1
				L = np.sqrt(sq**2 + su**2)

				gamma = np.zeros((atmos.nx, atmos.ny))
				for idx in range(atmos.nx):
					for idy in range(atmos.ny):
						tck = splrep(x[idx,idy], si[idx,idy])
						si_der = splev(x[idx,idy,ind_lam_wing], tck, der=1)
						
						_L = L[idx,idy,ind_lam_wing]
						delta_lam = x[idx,idy,dd] - x[idx,idy,ind_lam_wing]

						denom = -4*geff*delta_lam*si_der * _L
						denom = np.sqrt(np.abs(denom))
						nom = 3*Geff*sv[idx,idy,ind_lam_wing]**2
						nom = np.sqrt(np.abs(nom))
						gamma[idx,idy] = np.arctan2(denom, nom)

				inclination += gamma

	if "vz" in atmos.nodes:
		atmos.values["vz"] = np.repeat(vlos[..., np.newaxis]/nl, len(atmos.nodes["vz"]), axis=-1)

	if init_mag:
		if "gamma" in atmos.nodes:
			#--- check for the bounds in inclination
			#--- inclination can not be closer than 5 degrees to 90 degrees
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					if np.abs(inclination[idx,idy]-np.pi/2) < 5*np.pi/180:
						inclination[idx,idy] = np.pi/2 + 5*np.pi/180 * np.sign(inclination[idx,idy] - np.pi/2)
			# atmos.values["gamma"] = np.repeat(np.tan(inclination[..., np.newaxis]/nl_mag/2), len(atmos.nodes["gamma"]), axis=-1)
			# atmos.values["gamma"] = np.repeat(np.cos(inclination[..., np.newaxis]/nl_mag), len(atmos.nodes["gamma"]), axis=-1)
			atmos.values["gamma"] = np.repeat(inclination[..., np.newaxis]/nl_mag, len(atmos.nodes["gamma"]), axis=-1)
			y = np.cos(atmos.values["gamma"])
			atmos.values["gamma"] = np.arccos(y)
		if "mag" in atmos.nodes:
			blos /= nl_mag
			inclination /= nl_mag
			if "gamma" in atmos.nodes:
				mag = blos / np.cos(inclination)
			else:
				mag = blos / np.cos(np.pi/3)
			#--- check for the bounds in magnetic field strength
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					if mag[idx,idy] > atmos.limit_values["mag"][1]:
						mag[idx,idy] = atmos.limit_values["mag"][1]
			atmos.values["mag"] = np.repeat(mag[..., np.newaxis], len(atmos.nodes["mag"]), axis=-1)
		if "chi" in atmos.nodes:
			# atmos.values["chi"] = np.repeat(np.tan(azimuth[..., np.newaxis]/nl/4), len(atmos.nodes["chi"]), axis=-1)
			# atmos.values["chi"] = np.repeat(np.cos(azimuth[..., np.newaxis]/nl), len(atmos.nodes["chi"]), axis=-1)
			atmos.values["chi"] = np.repeat(azimuth[..., np.newaxis]/nl, len(atmos.nodes["chi"]), axis=-1)
			y = np.cos(atmos.values["chi"])
			atmos.values["chi"] = np.arccos(y)

		# print("-----")
		# # print(atmos.values["mag"])
		# # print(atmos.values["gamma"])
		# print(atmos.values["chi"] * 180/np.pi)
		# sys.exit()

def read_OF_data(fpath):
	try:
		hdu = fits.open(fpath)
		of_wave = hdu[0].data
		of_value = hdu[1].data
		of_num = len(globin.of_wave)
	except:
		lines = open(fpath, "r").readlines()

		of_wave, of_value = [], []
		of_num = 0

		for line in lines:
			if "#" not in line:
				lam, fudge = _slice_line(line)
				of_wave.append(lam)
				of_value.append(fudge)
				of_num += 1

		of_wave = np.array(of_wave)
		of_value = np.array(of_value)

	of_wave = write_wavs(of_wave, fname=None)

	return of_num, of_wave, of_value