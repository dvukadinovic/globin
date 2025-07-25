import os
import sys
import numpy as np
import re
import copy
import subprocess as sp
from astropy.io import fits
from scipy.interpolate import interp1d, splev
from scipy.integrate import simps
from scipy.signal import find_peaks

from .atoms import read_RLK_lines, read_init_line_parameters
from .atmos import Atmosphere
from .spec import Observation
from .utils import _slice_line, construct_atmosphere_from_nodes, air_to_vacuum
from .utils import compute_wavelength_grid

import globin

class InputData(object):
	def __init__(self):
		# number of threads to be used for parallel computing
		self.n_thread = 1

		# in each iteration of inversion we save:
		#   -- the LM parameter
		#   -- the inversion parameters
		#   -- spectra
		#   -- RFs
		self.debug = False

		# normalization flag (True/False)
		# self.norm = False

		# continuum value ('hsra', 1, float)
		# self.norm_level = None

		# flag for computing the mean spectrum
		self.mean = False

	def read_input_files(self, globin_input_name, rh_input_name):
		"""
		Read input files for globin and RH.

		We assume that parameters (in both files) are given in format:
			key = value

		Commented lines begin with symbol '#'.

		Parameters:
		-----------
		globin_input_name : str
			path to file that contains input parameters for globin
		rh_input_name : str
			path to 'keyword.input' file for RH parameters
		"""
		self.cwd = f"./runs/{self.run_name}"
		
		self.globin_input_name = globin_input_name
		self.rh_input_name = rh_input_name

		# make runs directory if not existing
		# here we store all runs with different 'run_name'
		if not os.path.exists("runs"):
			os.mkdir("runs")

		# make directory for specified run with provided 'run_name'
		if not os.path.exists(self.cwd):
			os.mkdir(self.cwd)
		# else:
			# sp.run(f"rm {self.cwd}/*.input", shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

		# copy input files into 'run_name' directory
		sp.run(f"cp {globin_input_name} {rh_input_name} {self.cwd}",
			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

		# create atoms and molecules RH input files 
		globin.rh.atmols.create_atoms_list(f"{self.cwd}/atoms.input")
		globin.rh.atmols.create_molecules_list(f"{self.cwd}/molecules.input")

		# create keywords.input file
		keywords = globin.rh.RHKeywords()
		keywords.create_input_file(f"{self.cwd}/keyword.input")

		#--- get parameters from globin input file
		text = open(self.globin_input_name, "r").read()
		self.parameters_input = text

		line_list = _find_value_by_key("line_list", self.parameters_input, "required")
		globin.rh.create_kurucz_input(line_list, f"{self.cwd}/kurucz.input")


		self.mode = _find_value_by_key("mode", self.parameters_input, "required", conversion=int)

		self.n_thread = _find_value_by_key("n_thread", self.parameters_input, "default", 1, conversion=int)

		# macro-turbulent velocity
		vmac = _find_value_by_key("vmac", self.parameters_input, "default", 0, conversion=float)
		if vmac<0 and self.mode!=3:
			raise ValueError(f"Cannot invert for 'vmac' in mode={self.mode}. Please use mode=3.")
		
		# the FOV of the observation that we want to invert
		# or the FOV of the cube atmosphere that we want to synthesize spectra for
		atm_range = get_atmosphere_range(self.parameters_input)
		
		# log(tau) scale for inversion atmosphere
		logtau_top = _find_value_by_key("logtau_top", self.parameters_input, "default", -6, conversion=float)
		logtau_bot = _find_value_by_key("logtau_bot", self.parameters_input, "default", 1, conversion=float)
		logtau_step = _find_value_by_key("logtau_step", self.parameters_input, "default", 0.1, conversion=float)
		
		# assumed noise for the synthetic spectrum/observations
		self.noise = _find_value_by_key("noise", self.parameters_input, "default", 1e-3, conversion=float)

		# type of the input atmosphere ('{reference, cube}_atmosphere')
		atm_type = _find_value_by_key("atm_type", self.parameters_input, "default", "multi", conversion=str)
		atm_type = atm_type.lower()

		# depth scale used to stratify atmospheric parameters
		atm_scale = _find_value_by_key("atm_scale", self.parameters_input, "default", "tau", conversion=str)

		# read Opacity Fudge (OF) data
		opacity_fudge_mode = _find_value_by_key("of_mode", self.parameters_input, "default", -1, conversion=int)
		add_opacity_fudge = False
		if opacity_fudge_mode in [0,1]:
			add_opacity_fudge = True
			file_path = _find_value_by_key("of_file", self.parameters_input, "default", None, conversion=str)
			if file_path:
				opacity_fudge_wavelength, opacity_fudge = read_OF_data(file_path)
			opacity_fudge_scatter = _find_value_by_key("of_scatt_flag", self.parameters_input, "default", 0, conversion=int)

		init_temp = _find_value_by_key("init_temp", self.parameters_input, "default", "false", conversion=str)
		self.init_temp = False
		if init_temp.lower()=="true":
			self.init_temp = True

		#----------------------------------------------------------------------
		# Optional parameters
		#----------------------------------------------------------------------

		# key used to make debug files/checks during synthesis/inversion
		debug = _find_value_by_key("debug", self.parameters_input, "optional")
		if debug is not None:
			self.debug = False
			if debug.lower()=="true":
				self.debug = True

		# if we want to compute the mean spectrum (from multi-component atmosphere)
		# mean = _find_value_by_key("mean", self.parameters_input, "optional")
		# if mean is not None:
		# 	if mean.lower()=="true":
		# 		self.mean = True
		# 	elif mean.lower()=="false":
		# 		self.mean = False
		# 	else:
		# 		raise ValueError(f"The 'mean' key contains unrecognized value {mean.lower()}.")

		# if we are computing the mean spectrum, we need also macro velocity and filling factor
		# for each atmosphere component. Usefull only when we have 2-3 components, not for whole cube...
		# if self.mean:
		# 	mac_vel = _find_value_by_key("mac_vel", self.parameters_input, "required")
		# 	self.mac_vel = [float(item) for item in mac_vel.split(",")]

		# 	items = _find_value_by_key("filling_factor", self.parameters_input, "required")
		# 	self.filling_factor = [float(item) for item in items.split(",")]

		#--- read data for different modus operandi
		if self.mode==0:
			self.read_mode_0(atm_range, atm_type, logtau_top, logtau_bot, logtau_step)
			self.atmosphere.vmac = np.abs(vmac) # [km/s]
		elif self.mode==-1:
			# compute the response functions for given parameters in every depth level
			raise NotImplemented(f"The mode={self.mode} is not yet supported.")
		elif self.mode>=1:
			self.read_inversion_base(vmac, atm_range, atm_type, logtau_top, logtau_bot, logtau_step)			
			if self.mode==2:
				self.read_mode_2()
			elif self.mode==3:
				self.read_mode_3()

			# determine the number of local and global parameters
			self.atmosphere.n_local_pars = 0
			for parameter in self.atmosphere.nodes:
				self.atmosphere.n_local_pars += np.sum(self.atmosphere.mask[parameter], dtype=np.int32)

			if add_opacity_fudge and opacity_fudge_mode==1:
				self.atmosphere.n_local_pars += len(opacity_fudge_wavelength)

			self.atmosphere.n_global_pars = 0
			for parameter in self.atmosphere.global_pars:
				self.atmosphere.n_global_pars += self.atmosphere.global_pars[parameter].shape[-1]

			# add the scale type (of the inversion atmosphere)
			self.atmosphere.scale_id = globin.scale_id[atm_scale]
		else:
			raise ValueError(f"Mode {self.mode} is unsupported.")

		# set wordking directory path (necessary for RH calls for reading runs/run_name/*.input files)
		self.atmosphere.set_cwd(self.cwd)

		# get the mu angle
		mu = load_mu_angle(self.parameters_input)
		self.atmosphere.set_mu(mu)

		# get the mode of operandi
		self.atmosphere.set_mode(self.mode)

		# add spectra normalization parmaeters
		norm, norm_level = load_spectrum_normalization(self.parameters_input)
		self.atmosphere.set_spectrum_normalization(norm, norm_level)

		# set the wavelength grid in the atmosphere
		wavelength_air = load_wavelength_grid(self.parameters_input)
		self.atmosphere.set_wavelength_grid(wavelength_air)
		if self.mode>=1:
			self.atmosphere.wavelength_obs = self.observation.wavelength

		# add the wavelength in the continuum used to normalize synthetic spectra
		if norm:
			continuum_wavelength = _find_value_by_key("continuum_wavelength", self.parameters_input, "optional", conversion=float)
			if continuum_wavelength is not None:
				idl = np.argmin(np.abs(self.atmosphere.wavelength_obs-continuum_wavelength/10))
				self.atmosphere.continuum_idl = idl

		# compare the wavelength sampling in the observations and in the synthetic spectrum
		if self.mode>=1:
			dlam_obs = self.observation.wavelength[1:] - self.observation.wavelength[:-1]
			dlam_synth = wavelength_air[1:] - wavelength_air[:-1]
			if np.mean(dlam_obs)<np.mean(dlam_synth):
				raise ValueError(f"Requested wavelength sampling {np.mean(dlam_synth):.4f} is smaller than the observed one {np.mean(dlam_obs):.4f}. Increase the wavelength sampling to improve the spectrum synthesis accuracy.")

		# if we have more threads than atmospheres, reduce the number of used threads
		if self.n_thread > self.atmosphere.nx*self.atmosphere.ny:
			self.n_thread = self.atmosphere.nx*self.atmosphere.ny
			print(f"[Warning] Reduced the number of threads to {self.n_thread}.")
		self.atmosphere.set_n_thread(self.n_thread)
		self.atmosphere.set_chunk_size()

		# set the OF values in atmosphere
		if add_opacity_fudge:
			# if we have 1D OF values, set equal values in all pixels
			if opacity_fudge.ndim==1:
				opacity_fudge = np.ones((self.atmosphere.nx, self.atmosphere.ny)) * opacity_fudge[0]

			self.atmosphere.set_opacity_fudge(opacity_fudge_mode, opacity_fudge_wavelength, opacity_fudge, opacity_fudge_scatter)

		#--- set the vmac_vel and ff for every atmosphere
		# [19.12.2022] DV: this does not make sense...
		# if self.mean:
		# 	if len(self.mac_vel)==1:
		# 		_vmac = self.mac_vel[0]
		# 		self.mac_vel = np.ones(self.atmosphere.nx * self.atmosphere.ny) * _vmac

		# 		ff = self.filling_factor[0]
		# 		self.filling_factor = np.ones(self.atmosphere.nx * self.atmosphere.ny) * ff

		# load stray light parameters
		sl_data = load_stray_light_parameters(self.parameters_input)
		self.atmosphere.set_stray_light_parameters(sl_data)

		if sl_data is not None:
			sl_type = sl_data[2]

			if sl_type=="2nd_component":
				if self.atmosphere.sl_atmos is not None:
					self.atmosphere.sl_atmos.global_pars = copy.deepcopy(self.atmosphere.global_pars)
					self.atmosphere.sl_atmos.line_no = self.atmosphere.line_no
				else:
					self.atmosphere.init_2nd_component()

					for parameter in ["sl_temp", "sl_vz", "sl_vmic"]:
						values = load_2nd_component_parameters(parameter, self.parameters_input)
						self.atmosphere.set_2nd_component_parameter(parameter, values)

					self.atmosphere.sl_atmos.makeHSE()

			elif sl_type=="atmos":
				raise NotImplemented()
			elif sl_type=="spec":
				raise NotImplemented()

		#--- meshgrid of pixels for computation optimization
		idx,idy = np.meshgrid(np.arange(self.atmosphere.nx), np.arange(self.atmosphere.ny))
		self.atmosphere.idx_meshgrid = idx.flatten()
		self.atmosphere.idy_meshgrid = idy.flatten()

		#--- debugging variables initialization
		if self.mode>=1:
			if self.debug:
				Npar = self.atmosphere.n_local_pars + self.atmosphere.n_global_pars
				self.rf_debug = np.zeros((self.atmosphere.nx, self.atmosphere.ny, self.max_iter[0], Npar, len(self.atmosphere.wavelength_air), 4))

				elements = []
				for parameter in self.atmosphere.nodes:
					aux = np.zeros((self.max_iter[0], self.atmosphere.nx, self.atmosphere.ny, len(self.atmosphere.nodes[parameter])))
					elements.append((parameter, aux))
				self.atmos_debug = dict(elements)

		#--- resample and normalize the instrumental profile to specified wavelength grid
		file_path = _find_value_by_key("instrumental_profile", self.parameters_input, "optional", None, str)
		if file_path is not None:
			instrumental_wave, instrumental_profile = np.loadtxt(file_path, unpack=True)
			instrumental_wave /= 10 # [A to nm]

			self.atmosphere.set_instrumental_profile(instrumental_wave, instrumental_profile)

		decreasing_temperature = _find_value_by_key("decreasing_temperature", self.parameters_input, "optional")
		if decreasing_temperature is not None:
			if decreasing_temperature.lower()=="false":
				self.atmosphere.decreasing_temperature = False
			if decreasing_temperature.lower()=="true":
				self.atmosphere.decreasing_temperature = True

	def read_mode_0(self, atm_range, atm_type, logtau_top, logtau_bot, logtau_step):
		""" 
		Get parameters for synthesis: spectra output path and the atmosphere.
		"""
		# spectrum file name to be saved after synthesis
		self.output_spectra_path = _find_value_by_key("spectrum", self.parameters_input, "default", "spectrum.fits")

		# read in the atmosphere for which we are computing the spectrum
		path_to_atmosphere = _find_value_by_key("cube_atmosphere", self.parameters_input, "optional")
		if path_to_atmosphere is None:
			path_to_atmosphere = _find_value_by_key("reference_atmosphere", self.parameters_input, "optional")
			if path_to_atmosphere is None:
				node_atmosphere_path = _find_value_by_key("node_atmosphere", self.parameters_input, "optional")
				if node_atmosphere_path is None:
					self.atmosphere = globin.falc
				else:
					self.atmosphere = construct_atmosphere_from_nodes(node_atmosphere_path, atm_range)
			else:
				self.atmosphere = Atmosphere(fpath=path_to_atmosphere, atm_type=atm_type, atm_range=atm_range,
							logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)

		else:
			self.atmosphere = Atmosphere(fpath=path_to_atmosphere, atm_type=atm_type, atm_range=atm_range,
							logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)

	def read_inversion_base(self, vmac, atm_range, atm_type, logtau_top, logtau_bot, logtau_step):
		reset_atomic_values = _find_value_by_key("reset_atomic_values", self.parameters_input, "optional", conversion=str)
		self.reset_atomic_values = False
		if reset_atomic_values is not None:
			if reset_atomic_values.lower()=="true":
				self.reset_atomic_values = True

		# parameters for atmosphere interpolation between nodes
		interp_degree = _find_value_by_key("interp_degree", self.parameters_input, "default", 3, int)
		interpolation_method = _find_value_by_key("interp_method", self.parameters_input, "default", "bezier", str)
		if interpolation_method.lower() not in ["bezier", "spline"]:
			raise ValueError(f"Interpolation method {interpolation_method.lower()} is not supported. Change it to 'bezier' or 'spline'.")
		if interpolation_method.lower()=="spline":
			spline_tension = _find_value_by_key("spline_tension", self.parameters_input, "default", 0, float)
		
		# inversion algorithm specific parameters
		self.ncycle = _find_value_by_key("ncycle", self.parameters_input, "default", 1, int)
		window = _find_value_by_key("smooth_window", self.parameters_input, "default", "5", str)
		self.gaussian_smooth_window = list(map(int, window.split(",")))
		std = _find_value_by_key("smooth_std", self.parameters_input, "default", "2.5", str)
		self.gaussian_smooth_std = list(map(float, std.split(",")))
		self.svd_tolerance = _find_value_by_key("svd_tolerance", self.parameters_input, "default", 1e-5, float)
		marq_lambda = _find_value_by_key("marq_lambda", self.parameters_input, "default", 1e1, str)
		tmp = marq_lambda.split(",")
		self.marq_lambda = np.array([float(item) for item in tmp])
		max_iter = _find_value_by_key("max_iter", self.parameters_input, "default", 30, str)
		tmp = max_iter.split(",")
		self.max_iter = np.array([int(item) for item in tmp])
		self.chi2_tolerance = _find_value_by_key("chi2_tolerance", self.parameters_input, "default", 1e-2, float)

		# type of RFs:
		#   -- node: compute the perturbations only in node
		#   -- snapi: compute the perturbations in every atmosphere level (Milic and van Noort 2019) [obsolete]
		self.rf_type = _find_value_by_key("rf_type", self.parameters_input, "default", "node", str)

		
		# type of wavelength-dependent weighting:
		#   -- StokesI: use 1/StokesI as a weighting
		self.weight_type = _find_value_by_key("weight_type", self.parameters_input, "default", None, str)

		# weights for Stokes components
		values = _find_value_by_key("weights", self.parameters_input, "default", np.array([1,1,1,1], dtype=np.float64))
		if type(values)==str:
			values = values.split(",")
			self.weights = np.array([float(item) for item in values], dtype=np.float64)

		# input observation format:
		#   -- globin: defult one where the input cube as dimension (nx, ny, nw, ns+1) where [...,0] is wavelength
		#   -- hinode: must have 3 extensions; 1st is Stokes, 2nd wavelength, 3rd continuum normalization value
		obs_fmt = _find_value_by_key("obs_format", self.parameters_input, "default", "globin", str)
		
		path_to_observations = _find_value_by_key("observation", self.parameters_input, "required")
		self.observation = Observation(path_to_observations, obs_range=atm_range, spec_type=obs_fmt)

		# import matplotlib.pyplot as plt

		# plt.imshow(self.observation.V[...,83])
		# plt.colorbar()
		# plt.show()

		# plt.plot(self.observation.I[10,10])
		# plt.show()

		# initialize container for atmosphere which we invert
		# self.atmosphere = Atmosphere(nx=self.observation.nx, ny=self.observation.ny, 
		# 	logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)# atm_range=atm_range)
		self.atmosphere = Atmosphere(nx=self.observation.nx, ny=self.observation.ny)# atm_range=atm_range)
		
		self.atmosphere.interp_degree = interp_degree
		self.atmosphere.interpolation_method = interpolation_method.lower()
		if self.atmosphere.interpolation_method=="spline":
			self.atmosphere.spline_tension = spline_tension

		# type of RF derivative
		#   -- central: central derivative 
		#   -- forward: forward derivative
		rf_der_type = _find_value_by_key("rf_der_type", self.parameters_input, "default", "central", str)
		self.atmosphere.rf_der_type = rf_der_type

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
				# self.reference_atmosphere = globin.falc
				self.reference_atmosphere = globin.hsra

		#--- initialize invert atmosphere data from reference atmosphere
		logtau = np.arange(logtau_top, logtau_bot + logtau_step, logtau_step)
		logtau = np.round(logtau, decimals=2)
		self.atmosphere.interpolate_atmosphere(logtau, self.reference_atmosphere.data)
		self.atmosphere.shape = self.atmosphere.data.shape

		# wavelength dependent weights that are generally defined by the RF's.
		fpath = _find_value_by_key("rf_weights", self.parameters_input, "optional")
		self.wavs_weight = None
		if fpath is not None:
			self.wavs_weight = np.zeros((self.atmosphere.nx, self.atmosphere.ny, len(self.observation.wavelength),4))
			lam, wI, wQ, wU, wV = np.loadtxt(fpath, unpack=True)
			# !!! Lenghts can be the same, but not the values in arrays. Needs to be changed.
			if len(lam)==len(self.observation.wavelength):
				self.wavs_weight[...,0] = wI
				self.wavs_weight[...,1] = wQ
				self.wavs_weight[...,2] = wU
				self.wavs_weight[...,3] = wV
			else:
				self.wavs_weight[...,0] = interp1d(lam, wI)(self.observation.wavelength)
				self.wavs_weight[...,1] = interp1d(lam, wQ)(self.observation.wavelength)
				self.wavs_weight[...,2] = interp1d(lam, wU)(self.observation.wavelength)
				self.wavs_weight[...,3] = interp1d(lam, wV)(self.observation.wavelength)
			# import matplotlib.pyplot as plt
			# plt.plot(self.wavs_weight[0,0,:,0])
			# plt.show()

		# if macro-turbulent velocity is negative, we fit it
		if vmac<0:
			# check if initial macro veclocity is larger than the step size in wavelength
			vmac = np.abs(vmac)
			# kernel_sigma = vmac*1e3 / globin.LIGHT_SPEED * (self.lmin + self.lmax)*0.5 / self.step
			kernel_sigma = globin.utils.get_kernel_sigma(vmac, self.observation.wavelength)
			if kernel_sigma<0.5:
				mean_wavelength = (self.observation.wavelength[0] + self.observation.wavelength[-1])/2
				step = self.observation.wavelength[1] - self.observation.wavelength[0]
				vmac = 0.5 * globin.LIGHT_SPEED / (mean_wavelength) * step
				vmac /= 1e3
				self.limit_values["vmac"][0] = vmac

			self.atmosphere.global_pars["vmac"] = np.array([np.abs(vmac)])
			self.atmosphere.parameter_scale["vmac"] = 1
		self.atmosphere.vmac = np.abs(vmac)
		self.reference_atmosphere.vmac = np.abs(vmac)

		#--- read initial node parameter values	
		fpath = _find_value_by_key("initial_atmosphere", self.parameters_input, "optional")
		if fpath is not None:
			# read node parameters from .fits file that is inverted atmosphere
			# from older inversion run
			self.atmosphere.read_multi_cube(fpath)

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
				self.read_node_parameters(parameter)

		for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi", "stray_factor", "sl_temp", "sl_vz", "sl_vmic"]:
			self.read_node_values_limits(parameter)

		#--- check for spatial regularization of atmospheric parameters
		tmp = _find_value_by_key("spatial_regularization_weight", self.parameters_input, "optional")
		if tmp is not None:
			self.atmosphere.spatial_regularization = True
			self.atmosphere.spatial_regularization_weight = float(tmp)

			if self.spatial_regularization_weight>10:
				print("[Warning] Spatial regularization weight larger than 10!")

			# if self.spatial_regularization_weight<1e-6:
			# 	print("[Info] Spatial regularization weight smaller than 1e-6. We will turn off the spatial regularization.")
			# 	self.atmosphere.spatial_regularization = False

			if self.atmosphere.spatial_regularization_weight<=0:
				print("[Info] Spatial regularization weight is 0 or negative. We will turn off the spatial regularization.")
				self.atmosphere.spatial_regularization = False

		# calculate the regularization weights for each parameter based on a given global value and relative weighting
		if self.atmosphere.spatial_regularization:
			for parameter in self.atmosphere.nodes:
				self.atmosphere.regularization_weight[parameter] *= self.atmosphere.spatial_regularization_weight

		# if we are doing a spatial regularization, we MUST go into mode 3 inversion!
		if self.mode!=3 and self.atmosphere.spatial_regularization:
			raise ValueError(f"Cannot perform spatial regularization in the mode={self.mode}. Change to mode=3.")

		#--- if temperature is inverted, we recompute the HSE
		if "temp" in self.atmosphere.nodes:
			self.atmosphere.hydrostatic = True

		#--- initialize the vz, mag and azimuth based on CoG and WFA methods (optional)
		fpath = _find_value_by_key("lines2atm", self.parameters_input, "optional")
		if fpath:
			initialize_atmos_pars(self.atmosphere, self.observation, fpath, norm=False)

	def read_mode_2(self):
		#--- line parameters to be fit
		line_pars_path = _find_value_by_key("line_parameters", self.parameters_input, "optional")

		if line_pars_path is None:
			print("[Warning] No atomic parameters to fit. You sure?\n")
			return

		# read line parameters
		if line_pars_path.split(".")[-1]=="fits":
			self.atmosphere.load_atomic_parameters(line_pars_path)
		else:
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

		#---
		if self.atmosphere.sl_atmos is not None:
			self.atmosphere.sl_atmos.global_pars = copy.deepcopy(self.atmosphere.global_pars)
			self.atmosphere.sl_atmos.line_no = self.atmosphere.line_no

	def read_mode_3(self):
		self.output_frequency = _find_value_by_key("output_frequency", self.parameters_input, "default", self.max_iter[0], int)
		
		#--- line parameters to be fit
		line_pars_path = _find_value_by_key("line_parameters", self.parameters_input, "optional")
		if line_pars_path is None:
			print("[Warning] No atomic parameters to fit. You sure?\n")
			return

		# if we provided line parameters for fit, read those parameters
		if line_pars_path.split(".")[-1]=="fits":
			self.atmosphere.load_atomic_parameters(line_pars_path)
		else:
			lines_to_fit = read_init_line_parameters(line_pars_path)

			if len(lines_to_fit)==0:
				print("[Warning] Did not find atomic parameters to fit. You sure?\n")
				return

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

		#---
		if self.atmosphere.sl_atmos is not None:
			self.atmosphere.sl_atmos.global_pars = copy.deepcopy(self.atmosphere.global_pars)
			self.atmosphere.sl_atmos.line_no = self.atmosphere.line_no

	def read_node_parameters(self, parameter):
		"""
		For a given parameter read from the input file the node positions, the values,
		the parameter mask (optional), the regularization type(s) and the limits for each node.

		Parameters:
		---------------
		parameter : str
			name of the parameter for which we are loading in node parameters.
		text : str
			loaded input file string from which we are searching for the node keywords.
		"""
		atmosphere = self.atmosphere
		text = self.parameters_input

		#--- get inversino parameters
		nodes = _find_value_by_key(f"nodes_{parameter}", text, "optional")
		if nodes is None:
			return

		values = _find_value_by_key(f"nodes_{parameter}_values", text, "required")
		mask = _find_value_by_key(f"nodes_{parameter}_mask", text, "optional")
		# relative weighting for spatial regularization
		reg_weight = _find_value_by_key(f"nodes_{parameter}_reg_weight", text, "optional", conversion=float)
		# weighting for the depth-dependent regularization
		# considered to be a tuple of weight and type (take a look at the header of Atmosphere() object)
		dd_reg_weight = _find_value_by_key(f"nodes_{parameter}_dd_reg", text, "optional")
		
		#--- allocte nodes to atmosphere
		atmosphere.nodes[parameter] = np.array([float(item) for item in nodes.split(",")])
		nnodes = len(atmosphere.nodes[parameter])
		
		#--- allocate values to atmosphere
		values = np.array([float(item) for item in values.split(",")])
		if len(values)!=len(atmosphere.nodes[parameter]):
			raise ValueError(f"Number of nodes and values for {parameter} are not the same.")

		matrix = np.zeros((atmosphere.nx, atmosphere.ny, nnodes), dtype=np.float64)
		# in 1D computation, one of the angles to obtain the J is 60deg, and with the gamma=60 gives
		# nan/inf in projection of B.
		if parameter=="gamma":
			for i_, val in enumerate(values):
				if val==60:
					values[i_] += 1

		if parameter=="temp" and self.init_temp:
			values = splev(atmosphere.nodes[parameter], globin.temp_tck)

		matrix[:,:] = copy.deepcopy(values)
		if parameter=="gamma":
			matrix *= np.pi/180
			atmosphere.values[parameter] = matrix
		elif parameter=="chi":
			matrix *= np.pi/180
			atmosphere.values[parameter] = matrix
		else:
			atmosphere.values[parameter] = matrix
		
		#--- assign the mask values
		atmosphere.mask[parameter] = np.ones(nnodes)
		if mask is not None:
			mask = [float(item) for item in mask.split(",")]
			atmosphere.mask[parameter] = np.array(mask)

		#--- assign the relative regularization weight for each parameter
		if reg_weight is not None:
			atmosphere.regularization_weight[parameter] = reg_weight

		#--- assign the depth-dependent regularization weight and type
		if dd_reg_weight is not None:
			values = [item for item in dd_reg_weight.split(",")]
			if len(values)!=2:
				print(f"[Error] Wrong number of parameters for the depth-dependent regularization for {parameter}.")
				print(f"        It has to consist of two number specifying weight and type.")
				sys.exit()

			atmosphere.dd_regularization_weight[parameter] = float(values[0])
			atmosphere.dd_regularization_function[parameter] = int(values[1])
			
			if atmosphere.dd_regularization_weight[parameter]<=0:
				print(f"[Warning] Depth-dependent regularization for {parameter} has 0 or negative weight. We will turn-off the regularization.")
			
			if atmosphere.dd_regularization_function[parameter]==0:
				print(f"[Warning] Depth-dependent regularization for {parameter} is turned-off. Type is set to 0.")
				atmosphere.dd_regularization_function[parameter] = 0

			if atmosphere.dd_regularization_function[parameter]<0 or \
			   atmosphere.dd_regularization_function[parameter]>4:
				print(f"[Warning] Depth-dependent regularization for {parameter} is of wrong type.")
				print(f"          It should be between 0 and 4 (int). We will turn it off now.")
				atmosphere.dd_regularization_function[parameter] = 0

		# set the parameter scale
		atmosphere.parameter_scale[parameter] = np.ones((atmosphere.nx, atmosphere.ny, len(atmosphere.nodes[parameter])))

	def read_node_values_limits(self, parameter):
		"""
		Get the lmits for each parameter (and for each node for a parameter if specified).
		"""
		atmosphere = self.atmosphere
		text = self.parameters_input

		if parameter not in atmosphere.nodes:
			return

		nnodes = atmosphere.nodes[parameter].size

		min_limits = _find_value_by_key(f"nodes_{parameter}_vmin", text, "optional")
		max_limits = _find_value_by_key(f"nodes_{parameter}_vmax", text, "optional")

		#--- assign the lower limit values for each node
		if min_limits is not None:
			vmin = np.array([float(item) for item in min_limits.split(",")])
			if parameter in ["gamma", "chi"]:
				vmin  = np.deg2rad(vmin)
			atmosphere.limit_values[parameter].vmin = vmin
			atmosphere.limit_values[parameter].vmin_dim = len(vmin)

			if len(vmin)!=1 and len(vmin)!=nnodes:
				raise ValueError(f"Incompatible number of minimum limits for {parameter} and given number of nodes.")

		#--- assign the upper limit values for each node
		if max_limits is not None:
			vmax = np.array([float(item) for item in max_limits.split(",")])
			if parameter in ["gamma", "chi"]:
				vmax  = np.deg2rad(vmax)
			atmosphere.limit_values[parameter].vmax = vmax
			atmosphere.limit_values[parameter].vmax_dim = len(vmax)

			if len(vmax)!=1 and len(vmax)!=nnodes:
					raise ValueError(f"Incompatible number of maximum limits for {parameter} and given number of nodes.")

def load_wavelength_grid(input_text):
	"""
	Create the wavelength grid from provided keywords. We first search
	for 'wave_min', 'wave_max', 'wave_step' in units of Angstrom.
	If these are not found, we expect that the wavelength grid is
	provided in a single file in units of nanometers.

	Parameter:
	----------
	input_text : str
		text string of the input file.

	Return:
	-------
	wavelength_air : ndarray
		wavelength grid in units of nanometer in the air.
	"""
	lmin = _find_value_by_key("wave_min", input_text, "optional", conversion=float)
	lmax = _find_value_by_key("wave_max", input_text, "optional", conversion=float)
	step = _find_value_by_key("wave_step", input_text, "optional", conversion=float)
	
	if (step is None) or (lmin is None) or (lmax is None):
		wave_grid_path = _find_value_by_key("wave_grid", input_text, "required")
		wavelength_air = np.loadtxt(wave_grid_path)
	else:
		wavelength_air = compute_wavelength_grid(lmin=lmin, lmax=lmax, dlam=step, unit="A")

	return wavelength_air

def load_mu_angle(input_text):
	"""
	Load the mu angle from input file

	Parameter:
	----------
	input_text : str
		text string of the input file.

	Return:
	-------
	mu : list
		list of mu angles for which we need to compute spectra
	"""
	# heliocentric angle for the outgoing radiation; forwarded to RH
	mu = _find_value_by_key("mu", input_text, "default", 1.0)
	mu = list(map(float, list(mu.split(","))))
	if any(mu)>1 or any(mu)<0:
		raise ValueError(f"Angle mu={mu} for computing the outgoing radiation is out of bounds [0,1].")

	return mu

def load_spectrum_normalization(input_text):
	# normalization type/factor for synthetic spectra
	_norm = _find_value_by_key("norm", input_text, "optional")
	
	norm = False
	norm_level = "absolute"
	
	if _norm is not None:	
		_norm = _norm.lower()
		if _norm in ["hsra", "true"]:
			norm = True
			norm_level = "hsra"
		elif _norm=="1":
			norm = True
			norm_level = 1
		elif _norm=="false":
			pass
		else:
			norm = True
			norm_level = float(_norm)

	return norm, norm_level

def load_stray_light_parameters(input_text):
	stray_mode = _find_value_by_key("stray_mode", input_text, "default", 0, int)
	if stray_mode==0:
		print("[Warning] We are ignoring the stray light contribution.")
		return

	if stray_mode not in [1,2,3]:
		raise ValueError(f"Stray mode {stray_mode} is not supported.")

	stray_factor = _find_value_by_key("stray_factor", input_text, "default", "0.0", conversion=str)
	if ".fits" in stray_factor:
		stray_factor = fits.open(stray_factor)[0].data
	else:
		stray_factor = float(stray_factor)
		if np.abs(stray_factor)>1:
			raise ValueError("Stray light factor value above 1.")

		if np.abs(stray_factor)==0:
			return
	
	stray_type = _find_value_by_key("stray_type", input_text, "default", "gray", str)
	stray_type = stray_type.lower()
	# if stray_type not in ["gray", "2nd_component", "hsra", "atmos", "spec"]:
	if stray_type not in ["gray", "2nd_component", "hsra"]:
		raise ValueError(f"Stray light type '{stray_type}' is not supported.")

	stray_min = _find_value_by_key("stray_factor_vmin", input_text, "optional", conversion=float)
	stray_max = _find_value_by_key("stray_factor_vmax", input_text, "optional", conversion=float)

	return stray_mode, stray_factor, stray_type, stray_min, stray_max

def load_2nd_component_parameters(parameter, input_text):
	"""
	Read in the parameters for the second component.
	"""
	value = _find_value_by_key(f"{parameter}", input_text, "optional", None, str)
	if value is not None:
		if ".fits" in value:
			value = fits.open(value)[0].data
		else:
			value = float(value)

	fit_flag = _find_value_by_key(f"{parameter}_fit", input_text, "optional", "false", str)
	if fit_flag is not None:
		if fit_flag.lower()=="true":
			fit_flag = True
		else:
			fit_flag = False
	vmin = _find_value_by_key(f"{parameter}_vmin", input_text, "optional", conversion=float)
	vmax = _find_value_by_key(f"{parameter}_vmax", input_text, "optional", conversion=float)

	return value, fit_flag, vmin, vmax

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
			raise ValueError(f"Keyword '{key}' is not found in the input file.")
			sys.exit()
		elif key_type=="default":
			return default_val
		elif key_type=="optional":
			return None

def get_atmosphere_range(parameters_input):
	#--- determine which observations from cube to take into consideration
	aux = _find_value_by_key("range", parameters_input, "default", [0,None,0,None])
	
	atm_range = aux

	if type(aux)==str:
		split = aux.split(",")
		if len(split)==1:
			atm_range = np.loadtxt(split[0], dtype=np.int32).T
		elif len(split)==4:
			atm_range = []
			for item in split:
				if item is None or int(item)==-1:
					atm_range.append(None)
				elif item is not None:
					atm_range.append(int(item))

			if atm_range[1] is not None:
				if atm_range[1]<atm_range[0]:
					raise ValueError("'xmax' is smaller than 'xmin' in 'range'.")
			if atm_range[3] is not None:
				if atm_range[3]<atm_range[2]:
					raise ValueError("'ymax' is smaller than 'ymin' in 'range'.")
			if atm_range[0]<1:
				raise ValueError("'xmin' in 'range' is lower than 1.")
			if atm_range[2]<1:
				raise ValueError("'ymin' in 'range' is lower than 1.")
			
			# we count from zero, but let user count from 1
			atm_range[0] -= 1
			atm_range[2] -= 1
		else:
			raise ValueError("Unsupported format of 'range'.")

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

	# temperature interpolation
	atmos.temp_tck = globin.temp_tck
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

def initialize_atmos_pars(atmos, obs, fpath, norm=True):
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
	from scipy.optimize import curve_fit
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

	def gaussian(x, mu, std, A):
		return A * np.exp(-(x-mu)**2/std**2)

	dlam = obs.wavelength[1] - obs.wavelength[0]
	wavs = obs.wavelength
	Ic = obs.I[:,:,0]

	# number of interpolation points for each spectral line
	N = 101

	lines = open(fpath).readlines()
	lines = [line.rstrip("\n") for line in lines if (len(line.rstrip("\n"))>0) and ("#" not in line)]
	nl = len(lines)
	nl_mag = 0
	
	# if nl>=3:
	# 	raise ValueError("We currently do not support atmosphere initialization for mroe than one spectral line.")

	indx = np.arange(obs.nx)
	indy = np.arange(obs.ny)
	indx, indy = np.meshgrid(indx, indy)
	indx = indx.ravel()
	indy = indy.ravel()

	vlos = 0
	blos = 0
	azimuth = 0
	inclination = 0

	init_lines = [None]*nl
	mag_line = [None]*nl

	for idl, line in enumerate(lines):
		line = list(filter(None,line.split(" ")))

		init_mag = False
		if len(line)==2:
			lam0, line_dlam = map(float,line)
			init_lines[idl] = [lam0, line_dlam/1e4]
			mag_line[idl] = False
		elif len(line)==8:
			# raise ValueError("We currently do not support intialization of magnetic field vector.")
			init_mag = True
			mag_line[idl] = True
			lam0, line_dlam, geff, gl, gu, Jl, Ju, Bexp = map(float, line)
			gs = gu + gl
			gd = gu - gl
			_s = Ju*(Ju+1) + Jl*(Jl+1)
			_d = Ju*(Ju+1) - Jl*(Jl+1)
			delta = 1/80*gd**2 * (16*_s - 7*_d**2 -4)
			if geff<0:
				geff = 1/2*gs + 1/4*gd*_d
			Geff = geff**2 - delta
			init_lines[idl] = [lam0, line_dlam/1e4, geff, Geff, Bexp]
		else:
			print("[Error] input.initialize_atmos_pars():")
			print("  Wrong number of parameters for initializing")
			print("  the LOS velocity and magnetic field vector.")
			sys.exit()


	# if we are using only one line, set the distance between lines to very high value
	# that it will not identify other lines in the spectral window
	D = obs.nw//2
	# if nl==1:
	# else:
	# 	# D = (0.1)/ dlam - 6
	# 	D = np.abs(init_lines[0][0] - init_lines[0][1]) / dlam

	# initialize the arrays to be used
	x = np.empty((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	si = np.empty((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	sq = np.empty((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	su = np.empty((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	sv = np.empty((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	dIdlam = np.empty((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	_lam0 = np.ones((atmos.nx, atmos.ny, nl), dtype=np.float64)

	for idl in range(nl):
		_lam0[...,idl] *= init_lines[idl][0]

	for idx in range(obs.nx):
		for idy in range(obs.ny):
			for idl, line in enumerate(init_lines):
				if not mag_line[idl]:
					lam0, line_dlam = line
				if mag_line[idl]:
					lam0, line_dlam, geff, Geff, Bexp = line

				lmin = lam0 - line_dlam
				lmax = lam0 + line_dlam

				ind_min = np.argmin(np.abs(wavs - lmin))
				ind_max = np.argmin(np.abs(wavs - lmax))+1
				
				# plt.plot(obs.I[idx,idy].max()/Ic[idx,idy] - obs.I[idx,idy,ind_min:ind_max]/Ic[idx,idy])
				# plt.show()

				# find spectral lines
				peaks, properties = find_peaks(obs.I[idx,idy].max()/Ic[idx,idy] - obs.I[idx,idy,ind_min:ind_max]/Ic[idx,idy],
					height=(0.05, None), 
					width=(1, None),
					distance=D)
				
				# plt.plot(obs.I[idx,idy].max()/Ic[idx,idy] - obs.I[idx,idy,ind_min:ind_max]/Ic[idx,idy])
				# plt.axvline(x=peaks[0], c="k", lw=0.75)
				# plt.show()

				try:
					peaks[0] += ind_min
				except:
					print(idx, idy)

					plt.plot(obs.I[idx,idy].max()/Ic[idx,idy] - obs.I[idx,idy,ind_min:ind_max]/Ic[idx,idy])
					plt.show()

					raise ValueError("Dumb")

				# top = 0.8
				# bot = -0.01

				# colors = ["tab:orange", "tab:red"]
				# for peak, w, lb, rb, c in zip(peaks, properties["widths"], properties["left_bases"], properties["right_bases"], colors):
				# 	peak += ind_min
				# 	plt.axvline(x=peak, c=c)
				# 	plt.fill_between(x=np.linspace(peak-w, peak+w, num=101), y1=bot, y2=top, color=c, alpha=0.7)
				
				# plt.show()

				# 'width' of the line; we add 20% to be sure that we can catch wings in Stokes V profile
				w = properties["widths"][0] * 3
				w = int(w)
				# dlamB = 4.6686e-13*(lam0*10)**2 * geff * Bexp * 0.3
				# print(w)
				# print(dlamB//dlam)
				ind_min = peaks[0] - w
				ind_max = peaks[0] + w
				if ind_min<0:
					ind_min = 0

				# get only the line of interest and interpolate it on finner grid
				tmp_x = obs.wavelength[ind_min:ind_max]
				tmp_si = obs.I[idx,idy,ind_min:ind_max]/Ic[idx,idy]
				tmp_sq = obs.Q[idx,idy,ind_min:ind_max]/Ic[idx,idy]
				tmp_su = obs.U[idx,idy,ind_min:ind_max]/Ic[idx,idy]
				tmp_sv = obs.V[idx,idy,ind_min:ind_max]/Ic[idx,idy]

				x[idx,idy,:,idl] = np.linspace(tmp_x.min(), tmp_x.max(), num=N)
				si[idx,idy,:,idl] = interp1d(tmp_x, tmp_si, kind=3)(x[idx,idy,:,idl])
				sq[idx,idy,:,idl] = interp1d(tmp_x, tmp_sq, kind=3)(x[idx,idy,:,idl])
				su[idx,idy,:,idl] = interp1d(tmp_x, tmp_su, kind=3)(x[idx,idy,:,idl])
				sv[idx,idy,:,idl] = interp1d(tmp_x, tmp_sv, kind=3)(x[idx,idy,:,idl])
				dIdlam[idx,idy,:,idl] = splev(x[idx,idy,:,idl], splrep(x[idx,idy,:,idl], si[idx,idy,:,idl]), der=1)

				# mean = x[idx,idy,:,idl].mean()
				# par, _ = curve_fit(gaussian, x[idx,idy,:,idl], 1-si[idx,idy,:,idl], 
				# 	p0=[mean, line_dlam, 1],
				# 	bounds=([mean-line_dlam/2, 0, 0],[mean+line_dlam/2, line_dlam, 2]))
				# _lam0[idx,idy,idl] = par[0]

				# plt.plot(x[idx,idy,:,idl]-par[0], 1-si[idx,idy,:,idl])
				# plt.plot(x[idx,idy,:,idl]-par[0], gaussian(x[idx,idy,:,idl], *par))
				# plt.show()

	#--- v_LOS initialization (CoG)
	if "vz" in atmos.nodes:
		lcog = simps((1-si)*x, x, axis=-2) / simps(1-si, x, axis=-2)
		vlos = globin.LIGHT_SPEED * (1 - lcog/_lam0) / 1e3
		vlos = np.sum(vlos, axis=-1)/nl

		atmos.values["vz"] = np.repeat(vlos[..., np.newaxis], len(atmos.nodes["vz"]), axis=-1)

		# plt.imshow(atmos.values["vz"][...,0].T)
		# plt.colorbar()
		# plt.show()

	if init_mag:
		#--- azimuth initialization
		# if "chi" in atmos.nodes:	
		# 	_azimuth = np.arctan2(np.sum(su, axis=-1), np.sum(sq, axis=-1))# * 180/np.pi / 2
		# 	# _azimuth %= 2*np.pi
		# 	azimuth += _azimuth
			# azimuth = np.mean(azimuth, axis=-1)
			# print(azimuth)
		
		#--- B (CoG method)
		if "mag" in atmos.nodes:
			lamp = simps((1-si-sv)*x, axis=-2) / simps(1-si-sv, axis=-2)
			lamm = simps((1-si+sv)*x, axis=-2) / simps(1-si+sv, axis=-2)

			# WF approximation
			# C = 4.67e-13 * (_lam0*10)**2 * geff
			# print(C.shape)
			# aux_1 = np.sum(dIdlam*sv, axis=(2))
			# aux_2 = np.sum(dIdlam**2, axis=(2))
			# blos = - 1/C * aux_1/aux_2
			# blos = (lamp - lamm)*10/2 / C
			
			# COG method
			C = 4*np.pi*globin.ELECTRON_MASS*globin.LIGHT_SPEED/globin.ELECTRON_CHARGE/geff
			blos = (lamp - lamm)/2 / _lam0**2 * 1e9 * C
			blos *= 1e4 # [T --> G]
			blos = np.sum(blos, axis=-1)

			# count how many magnetic lines do we have
			nl_mag = np.sum(np.ones(nl)[mag_line])
			blos /= nl_mag

			# convert LOS B to B strength assumin inclination of 60 degrees
			b = np.abs(blos) / np.cos(np.pi/3)

			atmos.values["mag"] = np.repeat(b[..., np.newaxis], len(atmos.nodes["mag"]), axis=-1)

			# plt.imshow(atmos.values["mag"][...,0].T)
			# plt.colorbar()
			# plt.show()

		#--- inclination initialization
		# if "gamma" in atmos.nodes:
		# 	ind_lam_wing = dd//2-1
		# 	L = np.sqrt(sq**2 + su**2)

		# 	gamma = np.zeros((atmos.nx, atmos.ny))
		# 	for idx in range(atmos.nx):
		# 		for idy in range(atmos.ny):
		# 			tck = splrep(x[idx,idy], si[idx,idy])
		# 			si_der = splev(x[idx,idy,ind_lam_wing], tck, der=1)
					
		# 			_L = L[idx,idy,ind_lam_wing]
		# 			delta_lam = x[idx,idy,dd] - x[idx,idy,ind_lam_wing]

		# 			denom = -4*geff*delta_lam*si_der * _L
		# 			denom = np.sqrt(np.abs(denom))
		# 			nom = 3*Geff*sv[idx,idy,ind_lam_wing]**2
		# 			nom = np.sqrt(np.abs(nom))
		# 			gamma[idx,idy] = np.arctan2(denom, nom)

		# 	inclination += gamma

def read_OF_data(fpath):
	try:
		hdu = fits.open(fpath)
		of_wave = hdu[0].data
		of_value = hdu[1].data
	except:
		of_wave, of_value = np.loadtxt(fpath, unpack=True)
		# convert single number to array
		if not isinstance(of_wave, np.ndarray):
			of_wave = np.array([of_wave])
			of_value = np.array([of_value])

	# RH assumes that the wavelength for OF coefficients is in vacuum units
	of_wave = globin.utils.air_to_vacuum(of_wave)

	return of_wave, of_value

class RF(object):
	def __init__(self, fpath=None):
		self.rf_local = None
		self.local_pars = {}
		
		self.rf_global = None
		self.global_pars = {}

		self.normed_spec = None

		if fpath is not None:
			self.read(fpath)

	def read(self, fpath):
		hdulist = fits.open(fpath)
		nheaders = len(hdulist)
		header_names = [hdulist[i].name for i in range(nheaders)]
		
		# read atmospheric (local) RFs
		if "RF_LOCAL" in header_names:
			index = hdulist.index_of("RF_LOCAL")
			self.read_atmospheric_RFs(hdulist[index])
		if "RF_GLOBAL" in header_names:
			index = hdulist.index_of("RF_GLOBAL")
			self.read_global_pars_RFs(hdulist[index])

		self._get_norm_flag(hdulist[0])

		# get the wavelength grid
		self.wavelength = hdulist[-2].data
		# get the optical depth scale
		self.logtau = hdulist[-1].data

	def read_atmospheric_RFs(self, hdu):
		self.rf_local = hdu.data

		shape = self.rf_local.shape
		self.nx, self.ny, self.nz = shape[0], shape[1], shape[3]

		npar = shape[2]
		nread = 0
		i_ = 1
		while nread<npar:
			parameter = hdu.header[f"PAR{i_}"]
			idp = hdu.header[f"PARID{i_}"] - 1
			nread += 1
			self.local_pars[parameter] = idp
			i_ += 1

	def read_global_pars_RFs(self, hdu):
		self.rf_global = hdu.data

		shape = self.rf_global.shape
		self.nx, self.ny, self.nw = shape[0], shape[1], shape[3]

		npar = shape[2]
		nread = 0
		i_ = 1
		while nread<npar:
			parameter = hdu.header[f"PAR{i_}"]
			idp_min = hdu.header[f"PAR{i_}S"]-1
			idp_max = hdu.header[f"PAR{i_}E"]-1
			idp = np.arange(idp_min, idp_max+1, 1, dtype=np.int32)
			nread += len(idp)
			self.global_pars[parameter] = idp
			i_ += 1

	def _get_norm_flag(self, hdu):
		norm = hdu.header["NORMED"]
		if norm.lower()=="true":
			self.normed_spec = True
		if norm.lower()=="false":
			self.normed_spec = False

	def norm(self):
		"""
		Normalize the RF's by summing through the wavelength and Stokes vector (for global parameters) 
		and through depth points (for local parameters. By normalizatio RFs become unitless values.
		"""
		for parameter in self.local_pars:
			idp = self.local_pars[parameter]
			# sum over wavelength, Stokes components and height
			norm = np.sqrt(np.sum(self.rf_local[:,:,idp]**2, axis=(2,3,4)))
			self.rf_local[:,:,idp] /= norm[..., np.newaxis, np.newaxis, np.newaxis]
		for parameter in self.global_pars:
			idp = self.global_pars[parameter]
			# sum over wavelength and Stokes components
			norm = np.sqrt(np.sum(self.rf_global[:,:,idp]**2, axis=(3,4)))
			self.rf_global[:,:,idp] /= norm[..., np.newaxis, np.newaxis]

	@property	
	def T(self):
		return self.get_par_rf("temp")

	@property	
	def B(self):
		return self.get_par_rf("mag")

	@property	
	def vz(self):
		return self.get_par_rf("vz")

	@property	
	def vmic(self):
		return self.get_par_rf("vmic")

	@property	
	def theta(self):
		return self.get_par_rf("gamma")

	@property	
	def phi(self):
		return self.get_par_rf("chi")

	@property
	def loggf(self):
		return self.get_par_rf("loggf")

	def get_par_rf(self, parameter):
		if parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi"]:
			idp = self.local_pars[parameter]
			return self.rf_local[:,:,idp]
		if parameter in ["loggf", "dlam"]:
			idp = self.global_pars[parameter]
			return self.rf_global[:,:,idp]
		
		raise ValueError(f"We do not have RF for {parameter}.")

	def get_stokes_rf(self, stokes):
		return None
		if stokes.lower()=="i":
			ids = 0
		if stokes.lower()=="q":
			ids = 1
		if stokes.lower()=="u":
			ids = 2
		if stokes.lower()=="v":
			ids = 3
		return self.rf[...,ids]