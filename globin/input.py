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
from scipy.signal import resample

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
		# path = os.path.dirname(__file__)
		# path = os.getcwd()
		self.cwd = f"./runs/{self.run_name}"
		
		self.globin_input_name = globin_input_name
		self.rh_input_name = rh_input_name

		# make runs directory if not existing
		# here we store all runs with different run_name
		if not os.path.exists("runs"):
			os.mkdir("runs")

		# make directory for specified run with provided 'run_name'
		if not os.path.exists(self.cwd):
			os.mkdir(self.cwd)


		# copy all RH input files into run_name directory
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

		# key used to make debug files/checks during synthesis/inversion
		debug = _find_value_by_key("debug", self.parameters_input, "optional")
		self.debug = False
		if debug is not None:
			if debug.lower()=="true":
				self.debug = True
			elif debug.lower()=="false":
				self.debug = False

		# self.pyrh_path = _find_value_by_key("pyrh_path", self.parameters_input, "required")
		self.n_thread = _find_value_by_key("n_thread", self.parameters_input, "default", 1, conversion=int)
		self.mode = _find_value_by_key("mode", self.parameters_input, "required", conversion=int)
		norm = _find_value_by_key("norm", self.parameters_input, "optional")
		self.norm = False
		self.norm_level = None
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

		# filling-factor for stray light correction
		stray_factor = _find_value_by_key("stray_factor", self.parameters_input, "default", 0.0, float)
		if stray_factor>1:
			raise ValueError("Stray light factor value above 1.")
		#if stray_factor<0:
		#	raise ValueError("Stray light factor value below 0.")

		# flag for computing the mean spectrum
		mean = _find_value_by_key("mean", self.parameters_input, "optional")
		if mean is not None:
			if mean.lower()=="true":
				self.mean = True
			elif mean.lower()=="false":
				self.mean = False
		else:
			self.mean = False

		# if we are computing the mean spectrum, we need also macro velocity and filling factor
		# for each atmosphere component. Usefull only when we have 2-3 components, not for whole cube.
		if self.mean:
			mac_vel = _find_value_by_key("mac_vel", self.parameters_input, "required")
			self.mac_vel = [float(item) for item in mac_vel.split(",")]

			items = _find_value_by_key("filling_factor", self.parameters_input, "required")
			self.filling_factor = [float(item) for item in items.split(",")]
		
		#--- get wavelength range and save it to file ('wave_file_path')
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
			wavetable = np.arange(self.lmin, self.lmax+self.step, self.step)
		
		self.wavelength_air = copy.deepcopy(wavetable)
		self.wavelength_vacuum = write_wavs(wavetable, fname=None)
		
		# common parameters for all modes
		atm_range = get_atmosphere_range(self.parameters_input)
		logtau_top = _find_value_by_key("logtau_top", self.parameters_input, "default", -6,float)
		logtau_bot = _find_value_by_key("logtau_bot", self.parameters_input, "default", 1, float)
		logtau_step = _find_value_by_key("logtau_step", self.parameters_input, "default", 0.1, float)
		self.noise = _find_value_by_key("noise", self.parameters_input, "default", 1e-3, float)
		atm_type = _find_value_by_key("atm_type", self.parameters_input, "default", "multi", str)
		atm_type = atm_type.lower()
		self.atm_scale = _find_value_by_key("atm_scale", self.parameters_input, "default", "tau", str)
		
		# load the telescope instrumental profile
		self.instrumental_profile = None
		instrumental_profile_path = _find_value_by_key("instrumental_profile", self.parameters_input, "optional", None, str)
		if instrumental_profile_path is not None:	
			self.instrumental_wave, self.instrumental_profile = np.loadtxt(instrumental_profile_path, unpack=True)

		# angle for the outgoing radiation; forwarded to RH
		mu = _find_value_by_key("mu", self.parameters_input, "default", 1.0, float)
		if mu>1 or mu<0:
			raise ValueError(f"Angle mu={mu} for computing the outgoing radiation is out of bounds [0,1].")

		#--- read Opacity Fudge (OF) data
		self.of_mode = _find_value_by_key("of_mode", self.parameters_input, "default", -1, int)
		# of_mode:
		#   0    -- use it only for synthesis
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
		if self.mode==0:
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
				self.atmosphere.n_local_pars += np.sum(self.atmosphere.mask[parameter], dtype=np.int32)
				# self.atmosphere.n_local_pars += len(self.atmosphere.nodes[parameter])

			if (self.do_fudge):
				self.atmosphere.n_local_pars += of_num

			self.atmosphere.n_global_pars = 0
			for parameter in self.atmosphere.global_pars:
				self.atmosphere.n_global_pars += self.atmosphere.global_pars[parameter].shape[-1]
		else:
			print("[Error] Negative mode not supported. Soon to be RF calculation.")
			sys.exit()

		# add angle for which we need to compute spectrum to atmosphere
		self.atmosphere.mu = mu

		if not "loggf" in self.atmosphere.global_pars:
			self.atmosphere.global_pars["loggf"] = np.array([], dtype=np.float64)
			self.atmosphere.line_no["loggf"] = np.array([], dtype=np.int32)
		if not "dlam" in self.atmosphere.global_pars:
			self.atmosphere.global_pars["dlam"] = np.array([], dtype=np.float64)
			self.atmosphere.line_no["dlam"] = np.array([], dtype=np.int32)

		#--- if we have more threads than atmospheres, reduce the number of used threads
		if self.n_thread > self.atmosphere.nx*self.atmosphere.ny:
			self.n_thread = self.atmosphere.nx*self.atmosphere.ny
			print(f"[Warning] Reduced the number of threads to {self.n_thread}.")

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
			self.atmosphere.mask["of"] = np.ones(self.atmosphere.of_num)

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

		#--- check the status of stray light factor and if to be inverted; add it to atmosphere
		if np.abs(stray_factor)!=0:
			# get the mode of stray light
			self.stray_type = _find_value_by_key("stray_type", self.parameters_input, "default", "gray", str)
			self.stray_type = self.stray_type.lower()
			if self.stray_type!="gray" and self.stray_type!="hsra":
				raise ValueError(f"stray_type '{self.stray_type}' is not supported. Only 'gray' or 'hsra'.")

			# get the mode for stray light (synthesis/inversion)			
			self.stray_mode = _find_value_by_key("stray_mode", self.parameters_input, "default", 3, int)

			self.atmosphere.add_stray_light = True
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

		#--- meshgrid of pixels for computation optimization
		idx,idy = np.meshgrid(np.arange(self.atmosphere.nx), np.arange(self.atmosphere.ny))
		self.atmosphere.idx_meshgrid = idx.flatten()
		self.atmosphere.idy_meshgrid = idy.flatten()

		self.atmosphere.ids_tuple = list(zip(self.atmosphere.idx_meshgrid, self.atmosphere.idy_meshgrid))

		#--- debugging variables initialization
		if self.mode>=1:
			if self.debug:
				Npar = self.atmosphere.n_local_pars + self.atmosphere.n_global_pars
				self.rf_debug = np.zeros((self.atmosphere.nx, self.atmosphere.ny, self.max_iter, Npar, len(self.wavelength_air), 4))

				elements = []
				for parameter in self.atmosphere.nodes:
					aux = np.zeros((self.max_iter, self.atmosphere.nx, self.atmosphere.ny, len(self.atmosphere.nodes[parameter])))
					elements.append((parameter, aux))
				self.atmos_debug = dict(elements)

		#--- resample and normalize the instrumental profile to specified wavelength grid
		if self.instrumental_profile is not None:
			fun = interp1d(self.instrumental_wave, self.instrumental_profile, fill_value=0)
			dlam = self.step*10
			N = (self.instrumental_wave.max() - self.instrumental_wave.min()) / (dlam)
			M = np.ceil(N)//2
			M = int(M)
			xnew = np.linspace(0, (M-1)*dlam, num=M)
			aux = np.linspace(-(M-1)*dlam, -dlam, num=M-1)
			xnew = np.append(aux, xnew)
			aux = fun(xnew)
			self.instrumental_profile = aux/np.sum(aux)

			self.atmosphere.instrumental_profile = self.instrumental_profile

		# current working directory (path that is appended to RH input files before submitting to synthesis)
		self.atmosphere.cwd = self.cwd

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
		interpolation_method = _find_value_by_key("interp_method", self.parameters_input, "default", "bezier", str)
		if interpolation_method.lower() not in ["bezier", "spline"]:
			raise ValueError(f"Interpolation method {interpolation_method.lower()} is not supported. Change it to 'bezier' or 'spline'.")
		if interpolation_method.lower()=="spline":
			spline_tension = _find_value_by_key("spline_tension", self.parameters_input, "default", 0, float)
		self.svd_tolerance = _find_value_by_key("svd_tolerance", self.parameters_input, "default", 1e-8, float)

		#--- default parameters
		marq_lambda = _find_value_by_key("marq_lambda", self.parameters_input, "default", 1e-2, str)
		tmp = marq_lambda.split(",")
		self.marq_lambda = np.array([float(item) for item in tmp])

		max_iter = _find_value_by_key("max_iter", self.parameters_input, "default", 30, str)
		tmp = max_iter.split(",")
		self.max_iter = np.array([int(item) for item in tmp])

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
		obs_fmt = _find_value_by_key("obs_format", self.parameters_input, "default", "globin", str)
		path_to_observations = _find_value_by_key("observation", self.parameters_input, "required")
		self.observation = Observation(path_to_observations, obs_range=atm_range, spec_type=obs_fmt)
		# icont = _find_value_by_key("icont", self.parameters_input, "default", 1, float)
		# self.observation.spec /= icont
		# self.observation.icont = icont

		# initialize container for atmosphere which we invert
		# self.atmosphere = Atmosphere(nx=self.observation.nx, ny=self.observation.ny, 
		# 	logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)# atm_range=atm_range)
		self.atmosphere = Atmosphere(nx=self.observation.nx, ny=self.observation.ny)# atm_range=atm_range)
		self.atmosphere.interpolation_method = interpolation_method.lower()
		if self.atmosphere.interpolation_method=="spline":
			self.atmosphere.spline_tension = spline_tension

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
		logtau = np.arange(logtau_top, logtau_bot + logtau_step, logtau_step)
		logtau = np.round(logtau, decimals=2)
		self.atmosphere.interpolate_atmosphere(logtau, self.reference_atmosphere.data)

		fpath = _find_value_by_key("rf_weights", self.parameters_input, "optional")
		self.wavs_weight = None
		if fpath is not None:
			self.wavs_weight = np.ones((self.atmosphere.nx, self.atmosphere.ny, len(self.wavelength_air),4))
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
			init_atmosphere = globin.atmos.read_inverted_atmosphere(fpath, atm_range=[0,None,0,None])
			self.atmosphere.nodes = init_atmosphere.nodes
			self.atmosphere.values = init_atmosphere.values
			self.atmosphere.mask = init_atmosphere.mask
			self.atmosphere.parameter_scale = init_atmosphere.parameter_scale
			# copy regularization weights and flag
			self.atmosphere.spatial_regularization = init_atmosphere.spatial_regularization
			self.atmosphere.spatial_regularization_weight = init_atmosphere.spatial_regularization_weight
			for parameter in init_atmosphere.regularization_weight:
				self.atmosphere.regularization_weight[parameter] = init_atmosphere.regularization_weight[parameter]

			# copt depth-dependent regularization weight and type
			self.atmosphere.dd_regularization_function = init_atmosphere.dd_regularization_function
			self.atmosphere.dd_regularization_weight = init_atmosphere.dd_regularization_weight

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

		# check for spatial regularization of atmospheric parameters
		tmp = _find_value_by_key("spatial_regularization_weight", self.parameters_input, "optional")
		if tmp is not None:
			self.atmosphere.spatial_regularization = True
			self.atmosphere.spatial_regularization_weight = float(tmp)

			if self.spatial_regularization_weight>10:
				print("[Warning] Spatial regularization weight larger than 10!")

			# if self.spatial_regularization_weight<1e-6:
			# 	print("[Info] Spatial regularization weight smaller than 1e-6. We will turn off the spatial regularization.")
			# 	self.atmosphere.spatial_regularization = False

			if self.atmosphere.spatial_regularization_weight==0:
				print("[Info] Spatial regularization weight is 0. We will turn off the spatial regularization.")
				self.atmosphere.spatial_regularization = False

		#--- calculate the regularization weights for each parameter based on a given global value and relative weighting
		if self.atmosphere.spatial_regularization:
			for parameter in self.atmosphere.nodes:
				self.atmosphere.regularization_weight[parameter] *= self.atmosphere.spatial_regularization_weight

		#--- if we are doing a spatial regularization, we MUST go into mode 3 inversion!
		if self.mode!=3 and self.atmosphere.spatial_regularization:
			raise ValueError(f"Cannot perform spatial regularization in the mode={self.mode}. Change to mode=3.")

		# [18.11.2022] Depreciated? Not yet...
		self.atmosphere.hydrostatic = False
		if "temp" in self.atmosphere.nodes:
			self.atmosphere.hydrostatic = True

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
		else:
			print("No atomic parameters to fit. You sure?\n")

	def read_node_parameters(self, parameter, text):
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

		nodes = _find_value_by_key(f"nodes_{parameter}", text, "optional")
		values = _find_value_by_key(f"nodes_{parameter}_values", text, "optional")
		mask = _find_value_by_key(f"nodes_{parameter}_mask", text, "optional")
		min_limits = _find_value_by_key(f"nodes_{parameter}_vmin", text, "optional")
		max_limits = _find_value_by_key(f"nodes_{parameter}_vmax", text, "optional")
		# relative weighting for spatial regularization
		reg_weight = _find_value_by_key(f"nodes_{parameter}_reg_weight", text, "optional", conversion=float)
		# weighting for the depth-dependent regularization
		# considered to be a tuple of weight and type (take a look at the header of Atmosphere() object)
		dd_reg_weight = _find_value_by_key(f"nodes_{parameter}_dd_reg", text, "optional")
		
		if (nodes is not None) and (values is not None):
			atmosphere.nodes[parameter] = np.array([float(item) for item in nodes.split(",")])
			nnodes = len(atmosphere.nodes[parameter])
			
			values = np.array([float(item) for item in values.split(",")])
			if len(values)!=len(atmosphere.nodes[parameter]):
				raise ValueError(f"Number of nodes and values for {parameter} are not the same.")

			matrix = np.zeros((atmosphere.nx, atmosphere.ny, nnodes), dtype=np.float64)
			# in 1D computation, one of the angles to obtain the J is 60deg, and with the gamma=60 gives
			# nan/inf in projection of B.
			for i_,val in enumerate(values):
				if val==60:
					values[i_] += 1
			matrix[:,:] = copy.deepcopy(values)
			if parameter=="gamma":
				matrix *= np.pi/180
				atmosphere.values[parameter] = matrix
			elif parameter=="chi":
				matrix *= np.pi/180
				atmosphere.values[parameter] = matrix
			else:
				atmosphere.values[parameter] = matrix
			
			# assign the mask values
			if mask is None:
				atmosphere.mask[parameter] = np.ones(nnodes)
			else:
				mask = [float(item) for item in mask.split(",")]
				atmosphere.mask[parameter] = np.array(mask)

			# assign the lower limit values for each node
			if min_limits is not None:
				vmin = np.array([float(item) for item in min_limits.split(",")])
				atmosphere.limit_values[parameter].vmin = vmin
				atmosphere.limit_values[parameter].vmin_dim = len(vmin)

				if len(vmin)!=1 and len(vmin)!=nnodes:
					raise ValueError(f"Incompatible number of minimum limits for {parameter} and given number of nodes.")

			# assign the upper limit values for each node
			if max_limits is not None:
				vmax = np.array([float(item) for item in max_limits.split(",")])
				atmosphere.limit_values[parameter].vmax = vmax
				atmosphere.limit_values[parameter].vmax_dim = len(vmax)

				if len(vmax)!=1 and len(vmax)!=nnodes:
						raise ValueError(f"Incompatible number of maximum limits for {parameter} and given number of nodes.")

			# assign the relative regularization weight for each parameter
			if reg_weight is not None:
				atmosphere.regularization_weight[parameter] = reg_weight

			# assign the depth-dependent regularization weight and type
			if dd_reg_weight is not None:
				values = [item for item in dd_reg_weight.split(",")]
				if len(values)!=2:
					print(f"[Warning] Wrong number of parameters for the depth-dependent regularization for {parameter}.")
					print(f"  It has to consist of two number specifying weight and type.")
				else:
					atmosphere.dd_regularization_weight[parameter] = float(values[0])
					atmosphere.dd_regularization_function[parameter] = int(values[1])
					if int(values[1])==0:
						print(f"[Warning] Depth-dependent regularization for {parameter} is turned-off. Type is set to 0.")
					if int(values[1])<0 or int(values[1])>4:
						print(f"[Warning] Depth-dependent regularization for {parameter} if of wrong type.")
						print(f"  It should be between 0 and 4 (int). We will turn it off now.")
						atmosphere.dd_regularization_function[parameter] = 0
					if float(values[0])==0:
						print(f"[Warning] Depth-dependent regularization for {parameter} has 0 weight. We will turn-off the regularization.")

			# set the parameter scale
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

	dlam = obs.wavelength[1] - obs.wavelength[0]
	wavs = obs.wavelength

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
			print("[Error] input.initialize_atmos_pars():")
			print("  Wrong number of parameters for initializing")
			print("  the LOS velocity and magnetic field vector.")
			sys.exit()

		line_dlam /= 1e4 # [mA --> nm]
		lmin = lam0 - line_dlam
		lmax = lam0 + line_dlam

		ind_min = np.argmin(np.abs(wavs - lmin))
		ind_max = np.argmin(np.abs(wavs - lmax))+1

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

		inds = find_line_positions(obs.I[...,ind_min:ind_max])
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
					si[idx,idy] = obs.I[idx,idy,mmin:mmax]
					sq[idx,idy] = obs.Q[idx,idy,mmin:mmax]
					su[idx,idy] = obs.U[idx,idy,mmin:mmax]
					sv[idx,idy] = obs.V[idx,idy,mmin:mmax]
				else:
					x[idx,idy] = np.nan
					si[idx,idy] = np.nan
					sq[idx,idy] = np.nan
					su[idx,idy] = np.nan
					sv[idx,idy] = np.nan

		#--- v_LOS initialization (CoG)
		if "vz" in atmos.nodes:
			lcog = np.sum(x*(1-si), axis=-1) / np.sum(1-si, axis=-1)
			vlos += globin.LIGHT_SPEED * (1 - lcog/lam0) / 1e3
			# vlos = np.repeat(vlos[..., np.newaxis], len(atmos.nodes["vz"]), axis=-1)
			# atmos.values["vz"] = vlos

		if init_mag:
			#--- azimuth initialization
			# if "chi" in atmos.nodes:	
			# 	_azimuth = np.arctan2(np.sum(su, axis=-1), np.sum(sq, axis=-1))# * 180/np.pi / 2
			# 	# _azimuth %= 2*np.pi
			# 	azimuth += _azimuth
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

	if "vz" in atmos.nodes:
		atmos.values["vz"] = np.repeat(vlos[..., np.newaxis]/nl, len(atmos.nodes["vz"]), axis=-1)

	if init_mag:
		# if "gamma" in atmos.nodes:
		# 	#--- check for the bounds in inclination
		# 	#--- inclination can not be closer than 5 degrees to 90 degrees
		# 	for idx in range(atmos.nx):
		# 		for idy in range(atmos.ny):
		# 			if np.abs(inclination[idx,idy]-np.pi/2) < 5*np.pi/180:
		# 				inclination[idx,idy] = np.pi/2 + 5*np.pi/180 * np.sign(inclination[idx,idy] - np.pi/2)
		# 	# atmos.values["gamma"] = np.repeat(np.tan(inclination[..., np.newaxis]/nl_mag/2), len(atmos.nodes["gamma"]), axis=-1)
		# 	# atmos.values["gamma"] = np.repeat(np.cos(inclination[..., np.newaxis]/nl_mag), len(atmos.nodes["gamma"]), axis=-1)
		# 	atmos.values["gamma"] = np.repeat(inclination[..., np.newaxis]/nl_mag, len(atmos.nodes["gamma"]), axis=-1)
		# 	y = np.cos(atmos.values["gamma"])
		# 	atmos.values["gamma"] = np.arccos(y)
		if "mag" in atmos.nodes:
			blos /= nl_mag
			# inclination /= nl_mag
			# if "gamma" in atmos.nodes:
			# 	# mag = blos / np.cos(inclination)
			# 	mag = blos / np.cos(atmos.values["gamma"])
			# else:
			mag = blos / np.cos(np.pi/3)
			#--- check for the bounds in magnetic field strength
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					if mag[idx,idy] > atmos.limit_values["mag"].max:
						mag[idx,idy] = atmos.limit_values["mag"].max
			atmos.values["mag"] = np.repeat(mag[..., np.newaxis], len(atmos.nodes["mag"]), axis=-1)
		# if "chi" in atmos.nodes:
		# 	# atmos.values["chi"] = np.repeat(np.tan(azimuth[..., np.newaxis]/nl/4), len(atmos.nodes["chi"]), axis=-1)
		# 	# atmos.values["chi"] = np.repeat(np.cos(azimuth[..., np.newaxis]/nl), len(atmos.nodes["chi"]), axis=-1)
		# 	atmos.values["chi"] = np.repeat(azimuth[..., np.newaxis]/nl, len(atmos.nodes["chi"]), axis=-1)
		# 	y = np.cos(atmos.values["chi"])
		# 	atmos.values["chi"] = np.arccos(y)

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

class RF(object):
	def __init__(self, fpath=None):
		if fpath is not None:
			self.read(fpath)

	def read(self, fpath):
		hdu = fits.open(fpath)

		# print(repr(hdu[0].header))

		self.rf = hdu[0].data
		self.wavelength = hdu[1].data
		self.logtau = hdu[2].data

		self.nx = self.rf.shape[0]
		self.ny = self.rf.shape[1]
		self.nz = self.rf.shape[3]

		norm = hdu[0].header["NORMED"]
		if norm.lower()=="true":
			self.normed_spec = True
		elif norm.lower()=="false":
			self.normed_spec = False
		else:
			self.normed_spec = None

		npar = self.rf.shape[2]
		self.pars = {}
		nread = 0
		i_ = 0
		while nread < npar:
			parameter = hdu[0].header[f"PAR{i_+1}"]
			if parameter in ["loggf", "dlam"]:
				idp_min = hdu[0].header["PARIDMIN"]-1
				idp_max = hdu[0].header["PARIDMAX"]-1
				idp = np.arange(idp_min, idp_max+1)
				nread += len(idp)
			else:
				idp = hdu[0].header[f"PARID{i_+1}"] - 1
				nread += 1
			self.pars[parameter] = idp
			i_ += 1

	def norm(self):
		for parameter in self.pars:
			idp = self.pars[parameter]
			if parameter in ["loggf", "dlam"]:
				# norm = np.sqrt(np.sum(self.rf[:,:,idp]**2, axis=(4,5)))
				continue
			else:
			# sum over wavelength and Stokes components and depth
				norm = np.sqrt(np.sum(self.rf[:,:,idp]**2, axis=(2,3,4)))
			# print(norm[0,0])
			for idx in range(self.nx):
				for idy in range(self.ny):
					# self.rf[idx,idy,idp] = np.einsum("ijk,i->ijk", self.rf[idx,idy,idp], 1/norm[idx,idy])
					self.rf[idx,idy,idp] /= norm[idx,idy]

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

	def get_par_rf(self, parameter):
		idp = self.pars[parameter]
		return self.rf[:,:,idp]

	def get_stokes_rf(self, stokes):
		if stokes.lower()=="i":
			ids = 0
		if stokes.lower()=="q":
			ids = 1
		if stokes.lower()=="u":
			ids = 2
		if stokes.lower()=="v":
			ids = 3
		return self.rf[...,ids]

class Chi2(object):
	def __init__(self, fpath=None, nx=None, ny=None, niter=None, chi2=None):
		if fpath is not None:
			self.read(fpath)
		elif (nx is not None) and (ny is not None) and (niter is not None):
			self.chi2 = np.zeros((nx, ny, niter), dtype=np.float64)
			self.nx, self.ny, self.niter = nx, ny, niter

		self.mode = -1
		self.Nlocal_par = -1
		self.Nglobal_par = -1
		self.Nw = -1

		# regularization weight
		self.regularization_weight = 0
		# value of regularization functional
		self.regularization = 0

		if chi2 is not None:
			self.chi2 = chi2

	def read(self, fpath):
		hdu = fits.open(fpath)[0]
		header = hdu.header
		self.chi2 = hdu.data

		try:
			self.mode = header["MODE"]
			self.Nlocal_par = header["NLOCALP"]
			self.Nglobal_par = header["NGLOBALP"]
			self.Nw = header["NW"]
		except:
			# for the older outputs
			self.nx, self.ny,_ = self.chi2.shape
			self.chi2, _ = self.get_final_chi2()

		self.nx, self.ny = self.chi2.shape

	def get_final_chi2(self):
		last_iter = np.zeros((self.nx, self.ny))
		best_chi2 = np.zeros((self.nx, self.ny))
		for idx in range(self.nx):
			for idy in range(self.ny):
				inds_non_zero = np.nonzero(self.chi2[idx,idy])[0]
				last_iter[idx,idy] = inds_non_zero[-1]
				best_chi2[idx,idy] = self.chi2[idx,idy,inds_non_zero[-1]]

		return best_chi2, last_iter

	def per_pixel(self, best_chi2, copy=False):
		if self.mode==1 or self.mode==2:
			return best_chi2

		Natm = self.nx*self.ny
		if self.mode==3:
			Ndof = self.Nw*Natm - self.Nlocal_par*Natm - self.Nglobal_par

		best_chi2 *= Ndof
		best_chi2 /= (self.Nw - self.Nlocal_par - self.Nglobal_par)

		if copy:
			self.chi2 = best_chi2
		else:
			return best_chi2

	def save(self, fpath="chi2.fits"):
		best_chi2, last_iter = self.get_final_chi2()
		best_chi2 = self.per_pixel(best_chi2)

		primary = fits.PrimaryHDU(best_chi2)
		hdulist = fits.HDUList([primary])

		primary.name = "best_chi2"
		primary.header["NX"] = (self.nx, "number of x atmospheres")
		primary.header["NY"] = (self.ny, "number of y atmospheres")
		primary.header["MODE"] = (self.mode, "inversion mode")
		primary.header["NLOCALP"] = (self.Nlocal_par, "num. of local parameters")
		primary.header["NGLOBALP"] = (self.Nlocal_par, "num. of global parameters")
		primary.header["NW"] = (self.Nw, "number of wavelenghts (for full Stokes")

		# contianer for last iteration number for each pixel
		iter_hdu = fits.ImageHDU(last_iter)
		iter_hdu.name = "iteration_num"
		hdulist.append(iter_hdu)

		# save
		hdulist.writeto(fpath, overwrite=True)
