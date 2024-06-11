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
		# number of threads to be used for parallel computing
		self.n_thread = 1

		# in each iteration of inversion we save:
		#   -- the LM parameter
		#   -- the inversion parameters
		#   -- spectra
		#   -- RFs
		self.debug = False

		# normalization flag (True/False)
		self.norm = False
		# continuum value ('hsra', 1, float)
		self.norm_level = None

		# flag for computing the mean spectrum
		self.mean = False	

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

		# copy all input files into 'run_name' directory
		sp.run(f"cp *.input {self.cwd}",
			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

		#--- get parameters from RH input file
		text = open(self.rh_input_name, "r").read()
		self.keyword_input = text

		# wave_file_path = _find_value_by_key("WAVETABLE", self.keyword_input, "required")
		# wave_file_path = wave_file_path.split("/")[-1]
		# # obj.rh_spec_name = _find_value_by_key("SPECTRUM_OUTPUT", obj.keyword_input, "default", "spectrum.out")
		# self.solve_ne = _find_value_by_key("SOLVE_NE", self.keyword_input, "optional")
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

		#----------------------------------------------------------------------
		# Required parameters
		#----------------------------------------------------------------------

		# mode of operation
		#   -- 0: synthesis
		#   -- 1: pixel-by-pixel inversion of atmospheric parameters (+ stray light)
		#   -- 2: as 1 + pixel-by-pixel inversion of atomic parameters (log(gf) and dlam)
		#   -- 3: global inversion of atmospheric (+ stray light) and atomic parameters (and macro-velocity)
		self.mode = _find_value_by_key("mode", self.parameters_input, "required", conversion=int)
		
		# path to pyrh/ installation; sent to RH to correct that paths of 
		# atoms/molecules/partition functions/barklem data
		# self.pyrh_path = _find_value_by_key("pyrh_path", self.parameters_input, "required")

		# get wavelength range for computing the spectrum
		self.lmin = _find_value_by_key("wave_min", self.parameters_input, "optional", conversion=float)
		self.lmax = _find_value_by_key("wave_max", self.parameters_input, "optional", conversion=float)
		self.step = _find_value_by_key("wave_step", self.parameters_input, "optional", conversion=float)
		if (self.step is None) or (self.lmin is None) or (self.lmax is None):
			wave_grid_path = _find_value_by_key("wave_grid", self.parameters_input, "required")
			wavetable = np.loadtxt(wave_grid_path)
			self.lmin = min(wavetable)
			self.lmax = max(wavetable)
			self.step = wavetable[1] - wavetable[0]
		else:
			self.lmin /= 10
			self.lmax /= 10
			self.step /= 10
			Nw = int((self.lmax - self.lmin)/self.step) + 1
			wavetable = np.linspace(self.lmin, self.lmax, num=Nw)

		self.wavelength_air = wavetable
		self.wavelength_vacuum = air_to_vacuum(wavetable)

		#----------------------------------------------------------------------
		# Default parameters
		#----------------------------------------------------------------------

		self.n_thread = _find_value_by_key("n_thread", self.parameters_input, "default", 1, conversion=int)

		# angle for the outgoing radiation; forwarded to RH
		mu = _find_value_by_key("mu", self.parameters_input, "default", 1.0)
		mu = list(map(float, list(mu.split(","))))
		if any(mu)>1 or any(mu)<0:
			raise ValueError(f"Angle mu={mu} for computing the outgoing radiation is out of bounds [0,1].")

		# macro-turbulent velocity
		vmac = _find_value_by_key("vmac", self.parameters_input, "default", 0, conversion=float)
		if vmac<0 and self.mode!=3:
			raise ValueError(f"Cannot invert for 'vmac' in mode={self.mode}. Please use mode=3.")
		
		# FOV of atmosphere/observation that we want to synthesize/invert
		atm_range = get_atmosphere_range(self.parameters_input)
		
		# log(tau) scale for inversion atmosphere
		logtau_top = _find_value_by_key("logtau_top", self.parameters_input, "default", -6, conversion=float)
		logtau_bot = _find_value_by_key("logtau_bot", self.parameters_input, "default", 1, conversion=float)
		logtau_step = _find_value_by_key("logtau_step", self.parameters_input, "default", 0.1, conversion=float)
		
		# assumed noise for synthetic spectrum/observations
		self.noise = _find_value_by_key("noise", self.parameters_input, "default", 1e-3, conversion=float)

		# type of the input atmosphere ('{reference, cube}_atmosphere')
		atm_type = _find_value_by_key("atm_type", self.parameters_input, "default", "multi", conversion=str)
		atm_type = atm_type.lower()

		# scale used to define atmospheric parameters
		atm_scale = _find_value_by_key("atm_scale", self.parameters_input, "default", "tau", conversion=str)

		# filling-factor for stray light correction
		stray_factor = _find_value_by_key("stray_factor", self.parameters_input, "default", 0.0, conversion=float)
		if np.abs(stray_factor)>1:
			raise ValueError("Stray light factor value above 1.")
		#if stray_factor<0:
		#	raise ValueError("Stray light factor value below 0.")

		#--- read Opacity Fudge (OF) data
		self.of_mode = _find_value_by_key("of_fit_mode", self.parameters_input, "default", -1, conversion=int)
		# of_mode:
		#   0    -- use it only for synthesis
		#   1    -- invert for it in pixel-by-pixel manner
		#   else -- OF is not applyed
		self.do_fudge = False
		if (self.of_mode==0) or (self.of_mode==1):
			self.do_fudge = True
			of_file_path = _find_value_by_key("of_file", self.parameters_input, "default", None, conversion=str)
			self.of_scatter = _find_value_by_key("of_scatt_flag", self.parameters_input, "default", 0, conversion=int)
			if of_file_path:
				of_num, of_wave, of_value = read_OF_data(of_file_path)

		self.init_temp = _find_value_by_key("init_temp", self.parameters_input, "default", "false", conversion=str)
		if self.init_temp.lower()=="true":
			self.init_temp = True
		else:
			self.init_temp = False

		#----------------------------------------------------------------------
		# Optional parameters
		#----------------------------------------------------------------------

		# key used to make debug files/checks during synthesis/inversion
		debug = _find_value_by_key("debug", self.parameters_input, "optional")
		if debug is not None:
			if debug.lower()=="true":
				self.debug = True
			elif debug.lower()=="false":
				self.debug = False
		
		# normalization type/factor for synthetic spectra
		norm = _find_value_by_key("norm", self.parameters_input, "optional")
		if norm is not None:
			norm = norm.lower()
			if norm=="hsra" or norm=="true":
				self.norm = True
				self.norm_level = "hsra"
			elif norm=="1":
				self.norm = True
				self.norm_level = 1
			elif norm=="false":
				pass
			else:
				self.norm = True
				self.norm_level = float(norm)

		# if we want to compute the mean spectrum (from multi-component atmosphere)
		mean = _find_value_by_key("mean", self.parameters_input, "optional")
		if mean is not None:
			if mean.lower()=="true":
				self.mean = True
			elif mean.lower()=="false":
				self.mean = False
			else:
				raise ValueError(f"The 'mean' key contains unrecognized value {mean.lower()}.")

		# if we are computing the mean spectrum, we need also macro velocity and filling factor
		# for each atmosphere component. Usefull only when we have 2-3 components, not for whole cube...
		if self.mean:
			mac_vel = _find_value_by_key("mac_vel", self.parameters_input, "required")
			self.mac_vel = [float(item) for item in mac_vel.split(",")]

			items = _find_value_by_key("filling_factor", self.parameters_input, "required")
			self.filling_factor = [float(item) for item in items.split(",")]

		#----------------------------------------------------------------------

		# get the name of the input line list
		self.linelist_name = _find_value_by_key("linelist", self.parameters_input, "required")

		#--- read data for different modus operandi
		if self.mode==0:
			self.read_mode_0(atm_range, atm_type, logtau_top, logtau_bot, logtau_step)
			self.atmosphere.vmac = np.abs(vmac) # [km/s]
		elif self.mode==-1:
			# compute the response functions for given parameters in every depth level
			pass
		elif self.mode>=1:
			self.read_inversion_base(vmac, atm_range, atm_type, logtau_top, logtau_bot, logtau_step)			
			if self.mode==2:
				self.read_mode_2()
			elif self.mode==3:
				self.read_mode_3()

			#--- determine number of local and global parameters
			self.atmosphere.n_local_pars = 0
			for parameter in self.atmosphere.nodes:
				self.atmosphere.n_local_pars += np.sum(self.atmosphere.mask[parameter], dtype=np.int32)
				# self.atmosphere.n_local_pars += len(self.atmosphere.nodes[parameter])

			if self.do_fudge and self.of_mode==1:
				self.atmosphere.n_local_pars += of_num

			self.atmosphere.n_global_pars = 0
			for parameter in self.atmosphere.global_pars:
				self.atmosphere.n_global_pars += self.atmosphere.global_pars[parameter].shape[-1]

			# add the scale type (of the inversion atmosphere)
			self.atmosphere.scale_id = globin.scale_id[atm_scale]
		else:
			raise ValueError(f"Mode {self.mode} is unsupported.")

		# add angle for which we need to compute spectrum
		if len(mu)==1:
			self.atmosphere.mu = mu[0]
		else:
			mu = np.array(mu)
			self.atmosphere.mu = mu.reshape(self.atmosphere.nx, self.atmosphere.ny)

		# add the mode
		self.atmosphere.mode = self.mode

		# add spectra normalization parmaeters
		self.atmosphere.norm = self.norm
		self.atmosphere.norm_level = self.norm_level

		# add the step size for spectral synthesis
		# self.atmosphere.step = self.step

		# allocate wavelength grids to atmosphere (for spectrum synthesis)
		if self.mode>=1:
			self.atmosphere.wavelength_obs = self.observation.wavelength
		elif self.mode==0:
			self.atmosphere.wavelength_obs = self.wavelength_air
		
		self.atmosphere.wavelength_air = self.wavelength_air
		self.atmosphere.wavelength_vacuum = self.wavelength_vacuum

		# we do not want to have sparse synthetic spectrum from which we will scale up
		# we want to have the same or higher number of wavelength points than there
		# are in the observations.
		if self.mode>=1:
			if len(self.observation.wavelength)>len(self.wavelength_air):
				msg = "  Specified wavelength grid has lower number of points than\n"
				msg+= "  the observation's wavelength grid. Increase the number of\n"
				msg+= "  wavelength points to improve the sampling.\n"
				sys.exit(msg)

		# get the Pg at top of the atmosphere
		if self.atmosphere.pg_top is None:
			self.atmosphere.get_pg()

		#--- if we have more threads than atmospheres, reduce the number of used threads
		if self.n_thread > self.atmosphere.nx*self.atmosphere.ny:
			self.n_thread = self.atmosphere.nx*self.atmosphere.ny
			print(f"[Warning] Reduced the number of threads to {self.n_thread}.")
		self.atmosphere.n_thread = self.n_thread
		self.atmosphere.chunk_size = (self.atmosphere.nx * self.atmosphere.ny) // self.n_thread + 1

		#--- set OF data in atmosphere
		if self.do_fudge:
			# if we have 1D OF values, set equal values in all pixels
			if of_value.ndim==1:
				of_value = np.repeat(of_value[np.newaxis, :], self.atmosphere.nx, axis=0)
				of_value = np.repeat(of_value[:, np.newaxis, :], self.atmosphere.ny, axis=1)

			self.atmosphere.do_fudge = 1
			self.atmosphere.of_num = of_num
			self.atmosphere.of_mode = self.of_mode
			if self.of_mode==1:
				self.atmosphere.nodes["of"] = of_wave
				self.atmosphere.values["of"] = of_value
				self.atmosphere.parameter_scale["of"] = np.ones((self.atmosphere.nx, self.atmosphere.ny, self.atmosphere.of_num))
				self.atmosphere.mask["of"] = np.ones(self.atmosphere.of_num)
			else:
				self.atmosphere.of_wave = of_wave
				self.atmosphere.of_values = of_value

			# create arrays to be passed to RH for synthesis
			self.atmosphere.of_scatter = self.of_scatter
			self.atmosphere.make_OF_table(self.wavelength_vacuum)

		#--- set the vmac_vel and ff for every atmosphere
		# [19.12.2022] DV: this does not make sense...
		if self.mean:
			if len(self.mac_vel)==1:
				_vmac = self.mac_vel[0]
				self.mac_vel = np.ones(self.atmosphere.nx * self.atmosphere.ny) * _vmac

				ff = self.filling_factor[0]
				self.filling_factor = np.ones(self.atmosphere.nx * self.atmosphere.ny) * ff

		#--- check the status of stray light factor and if to be inverted; add it to atmosphere
		if np.abs(stray_factor)!=0:
			# get the mode of stray light
			self.stray_type = _find_value_by_key("stray_type", self.parameters_input, "default", "gray", str)
			self.stray_type = self.stray_type.lower()
			if self.stray_type not in ["gray", "hsra", "atmos", "spec"]:
				raise ValueError(f"stray_type '{self.stray_type}' is not supported. Only 'gray', 'hsra' or 'atmos'.")

			# get the mode for stray light (synthesis/inversion)			
			self.stray_mode = _find_value_by_key("stray_mode", self.parameters_input, "default", 3, int)

			self.atmosphere.add_stray_light = True
			
			if "stray" in self.atmosphere.global_pars:
				ones = np.ones((self.atmosphere.nx, self.atmosphere.ny, 1))
				self.atmosphere.stray_light = ones * np.abs(self.atmosphere.global_pars["stray"][0])
			else:
				ones = np.ones((self.atmosphere.nx, self.atmosphere.ny, 1))
				self.atmosphere.stray_light = ones * np.abs(stray_factor)
				if stray_factor<0:
					self.atmosphere.invert_stray = True
					if self.stray_mode==1 or self.stray_mode==2:
						# we are inverting for stray light factor (pixel-by-pixel mode)
						self.atmosphere.n_local_pars += 1
						self.atmosphere.nodes["stray"] = np.array([0])
						self.atmosphere.values["stray"] = self.atmosphere.stray_light
						self.atmosphere.parameter_scale["stray"] = ones
						self.atmosphere.mask["stray"] = np.ones(1)
					elif self.stray_mode==3:
						# stray light inversion in global mode
						self.atmosphere.n_global_pars += 1
						self.atmosphere.global_pars["stray"] = np.array([np.abs(stray_factor)], dtype=np.float64)
						self.atmosphere.parameter_scale["stray"] = 1.0
					else:
						raise ValueError(f"Stray light set to be fit, but the mode {self.stray_mode} is not supported.")

					# if self.stray_mode!=self.mode:
					# 	raise ValueError("Inversion mode of stray light factor is not the same as the main inversino mode.")

			# allocate parameters to atmosphere
			self.atmosphere.stray_mode = self.stray_mode
			self.atmosphere.stray_type = self.stray_type
			if self.atmosphere.stray_type=="atmos":
				fpath = _find_value_by_key("stray_atmosphere", self.parameters_input, "required")
				sl_atmosphere = globin.Atmosphere(fpath)
				sl_atmosphere.wavelength_air = self.atmosphere.wavelength_air
				sl_atmosphere.wavelength_obs = self.atmosphere.wavelength_obs
				sl_atmosphere.wavelength_vacuum = self.atmosphere.wavelength_vacuum
				self.atmosphere.stray_light_spectrum = sl_atmosphere.compute_spectra()
			if self.atmosphere.stray_type=="spec":
				fpath = _find_value_by_key("stray_spectrum", self.parameters_input, "required")
				self.atmosphere.stray_light_spectrum = globin.Observation(fpath, spec_type="hinode")
				self.atmosphere.stray_light_spectrum.interpolate(self.atmosphere.wavelength_air, n_thread=1, fill_value="extrapolate")

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
		instrumental_profile_path = _find_value_by_key("instrumental_profile", self.parameters_input, "optional", None, str)
		if instrumental_profile_path is not None:
			instrumental_wave, instrumental_profile = np.loadtxt(instrumental_profile_path, unpack=True)
			fun = interp1d(instrumental_wave/10, instrumental_profile, kind=3, fill_value=0)
			# if self.mode>=1:
			# 	dlam = self.observation.wavelength[1] - self.observation.wavelength[0]
			# else:
			dlam = self.step
			N = (instrumental_wave.max()/10 - instrumental_wave.min()/10) / (dlam)
			M = np.ceil(N)//2
			M = int(M)
			xnew = np.linspace(0, (M-1)*dlam, num=M)
			aux = np.linspace(-(M-1)*dlam, -dlam, num=M-1)
			xnew = np.append(aux, xnew)
			aux = fun(xnew)
			self.atmosphere.instrumental_profile = aux/np.sum(aux)

			# plt.plot(instrumental_wave/10, instrumental_profile)
			# plt.plot(xnew, aux)
			# plt.show()
			# sys.exit()

		# current working directory (path that is appended to RH input files before submitting to synthesis)
		self.atmosphere.cwd = self.cwd

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
		self.gaussian_smooth_window = list(map(float, window.split(",")))
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
				self.reference_atmosphere = globin.falc

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
			kernel_sigma = vmac*1e3 / globin.LIGHT_SPEED * (self.lmin + self.lmax)*0.5 / self.step
			if kernel_sigma<0.5:
				vmac = 0.5 * globin.LIGHT_SPEED / ((self.lmin + self.lmax)*0.5) * self.step
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
			# self.atmosphere.read_multi_cube(fpath, atm_range=atm_range)
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

		for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi"]:
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
		#--- Kurucz line list for given spectral region
		self.RLK_lines_text, self.RLK_lines = read_RLK_lines(self.linelist_name)
		Nlines = len(self.RLK_lines)

		#--- line parameters to be fit
		line_pars_path = _find_value_by_key("line_parameters", self.parameters_input, "optional")

		if line_pars_path is None:
			print("[Warning] No atomic parameters to fit. You sure?\n")
			return

		# read line parameters
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

		try:
			id_max = np.max(self.atmosphere.line_no["loggf"])
			if id_max>Nlines:
				raise ValueError(f"Read less spectral lines than the max line number from '{line_pars_path}'")
		except:
			pass

		try:
			id_max = np.max(self.atmosphere.line_no["dlam"])
			if id_max>Nlines:
				raise ValueError(f"Read less spectral lines than the max line number from '{line_pars_path}'")
		except:
			pass

	def read_mode_3(self):
		self.output_frequency = _find_value_by_key("output_frequency", self.parameters_input, "default", self.max_iter[0], int)
		
		#--- line parameters to be fit
		line_pars_path = _find_value_by_key("line_parameters", self.parameters_input, "optional")
		if line_pars_path is None:
			print("[Warning] No atomic parameters to fit. You sure?\n")
			return
		
		#--- Kurucz line list for given spectral region
		self.RLK_lines_text, self.RLK_lines = read_RLK_lines(self.linelist_name)
		Nlines = len(self.RLK_lines)

		# if we provided line parameters for fit, read those parameters
		lines_to_fit = read_init_line_parameters(line_pars_path)

		if len(lines_to_fit)==0:
			raise ValueError(f"'{line_pars_path}' does not contain line parameters.")

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

		try:
			id_max = np.max(self.atmosphere.line_no["loggf"])
			if id_max>Nlines:
				raise ValueError(f"Read less spectral lines than the max line number from '{line_pars_path}'")
		except:
			pass

		try:
			id_max = np.max(self.atmosphere.line_no["dlam"])
			if id_max>Nlines:
				raise ValueError(f"Read less spectral lines than the max line number from '{line_pars_path}'")
		except:
			pass

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
			atmosphere.limit_values[parameter].vmin = vmin
			atmosphere.limit_values[parameter].vmin_dim = len(vmin)

			if len(vmin)!=1 and len(vmin)!=nnodes:
				raise ValueError(f"Incompatible number of minimum limits for {parameter} and given number of nodes.")

		#--- assign the upper limit values for each node
		if max_limits is not None:
			vmax = np.array([float(item) for item in max_limits.split(",")])
			atmosphere.limit_values[parameter].vmax = vmax
			atmosphere.limit_values[parameter].vmax_dim = len(vmax)

			if len(vmax)!=1 and len(vmax)!=nnodes:
					raise ValueError(f"Incompatible number of maximum limits for {parameter} and given number of nodes.")

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
		elif len(line)==7:
			# raise ValueError("We currently do not support intialization of magnetic field vector.")
			init_mag = True
			mag_line[idl] = True
			lam0, line_dlam, geff, gl, gu, Jl, Ju = map(float, line)
			gs = gu + gl
			gd = gu - gl
			_s = Ju*(Ju+1) + Jl*(Jl+1)
			_d = Ju*(Ju+1) - Jl*(Jl+1)
			delta = 1/80*gd**2 * (16*_s - 7*_d**2 -4)
			if geff<0:
				geff = 1/2*gs + 1/4*gd*_d
			Geff = geff**2 - delta
			init_lines[idl] = [lam0, line_dlam/1e4, geff, Geff]
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
	x = np.zeros((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	si = np.zeros((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	sq = np.zeros((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	su = np.zeros((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	sv = np.zeros((atmos.nx, atmos.ny, N, nl),dtype=np.float64)
	_lam0 = np.ones((atmos.nx, atmos.ny, nl), dtype=np.float64)

	for idl in range(nl):
		_lam0[...,idl] *= init_lines[idl][0]

	for idx in range(obs.nx):
		for idy in range(obs.ny):
			for idl, line in enumerate(init_lines):
				if not mag_line[idl]:
					lam0, line_dlam = line
				if mag_line[idl]:
					lam0, line_dlam, geff, Geff = line

				lmin = lam0 - line_dlam
				lmax = lam0 + line_dlam

				ind_min = np.argmin(np.abs(wavs - lmin))
				ind_max = np.argmin(np.abs(wavs - lmax))+1
				
				# plt.plot(obs.I[idx,idy].max()/Ic[idx,idy] - obs.I[idx,idy,ind_min:ind_max]/Ic[idx,idy])
				# plt.show()

				# find spectral lines
				peaks, properties = find_peaks(obs.I[idx,idy].max()/Ic[idx,idy] - obs.I[idx,idy,ind_min:ind_max]/Ic[idx,idy],
					height=(0.2, None), 
					width=(1, None),
					distance=D)
				
				# plt.plot(obs.I[idx,idy].max()/Ic[idx,idy] - obs.I[idx,idy,ind_min:ind_max]/Ic[idx,idy])
				# plt.axvline(x=peaks[0], c="k", lw=0.75)
				# plt.show()

				peaks[0] += ind_min

				# top = 0.8
				# bot = -0.01

				# colors = ["tab:orange", "tab:red"]
				# for peak, w, lb, rb, c in zip(peaks, properties["widths"], properties["left_bases"], properties["right_bases"], colors):
				# 	peak += ind_min
				# 	plt.axvline(x=peak, c=c)
				# 	plt.fill_between(x=np.linspace(peak-w, peak+w, num=101), y1=bot, y2=top, color=c, alpha=0.7)
				
				# plt.show()

				# 'width' of the line; we add 20% to be sure that we can catch wings in Stokes V profile
				w = properties["widths"][0] * 1.2
				w = int(w)
				ind_min = peaks[0] - w
				ind_max = peaks[0] + w

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

	of_wave = globin.utils.write_wavs(of_wave, fname=None)

	return of_num, of_wave, of_value

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