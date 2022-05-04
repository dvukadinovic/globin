import os
import sys
import numpy as np
import multiprocessing as mp
import re
import copy
import subprocess as sp
from scipy.interpolate import interp1d
from astropy.io import fits

import matplotlib.pyplot as plt

from .atmos import Atmosphere
from .spec import Observation
from .rh import write_wavs
from .utils import _set_keyword, _slice_line

import globin

class RHInput(object):
	"""
	Container for RH input fields and methods.
	"""
	def __init__(self):
		pass

	def set_keyword(self, key):
		pass

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

def read_input(run_name, globin_input_name="params.input", rh_input_name="keyword.input", obj=None):
	if (rh_input_name is not None) and (globin_input_name is not None):
		read_input_files(run_name, globin_input_name, rh_input_name, obj)
	else:
		if rh_input_name is None:
			print(f"  There is no path for globin input file.")
		if globin_input_name is None:
			print(f"  There is no path for RH input file.")
		sys.exit()

def read_input_files(obj):
	"""
	Read input files for globin and RH.

	We assume that parameters (in both files) are given in format:
		key = value

	Commented lines begin with symbol '#'.

	Parameters:
	---------------
	obj : class Globin
		container for all globin methods
	"""
	# make runs directory if not existing
	# here we store all runs with different run_name
	if not os.path.exists("runs"):
		os.mkdir("runs")

	# make directory for specified run with provided 'run_name'
	if not os.path.exists(f"runs/{obj.run_name}"):
		os.mkdir(f"runs/{obj.run_name}")

	# copy all RH input files into run_name directory
	sp.run(f"cp *.input runs/{obj.run_name}",
		shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

	#--- get parameters from RH input file
	text = open(obj.rh_input_name, "r").read()
	obj.keyword_input = text

	# wave_file_path = _find_value_by_key("WAVETABLE", obj.keyword_input, "required")
	# wave_file_path = wave_file_path.split("/")[-1]
	# obj.rh_spec_name = _find_value_by_key("SPECTRUM_OUTPUT", obj.keyword_input, "default", "spectrum.out")
	obj.solve_ne = _find_value_by_key("SOLVE_NE", obj.keyword_input, "optional")
	# obj.hydrostatic = _find_value_by_key("HYDROSTATIC", obj.keyword_input, "optional")
	# obj.kurucz_input_fname = _find_value_by_key("KURUCZ_DATA", obj.keyword_input, "required")
	# obj.rf_file_path = _find_value_by_key("RF_OUTPUT", obj.keyword_input, "default", "rfs.out")
	# obj.stokes_mode = _find_value_by_key("STOKES_MODE", obj.keyword_input, "default", "NO_STOKES")
	obj.of_mode = _find_value_by_key("OPACITY_FUDGE", obj.keyword_input, "default", False)
	if obj.of_mode:
		obj.of_mode = True

	#--- get parameters from globin input file
	text = open(obj.globin_input_name, "r").read()
	obj.parameters_input = text

	# key used to make debug files/checks during synthesis/inversion
	debug = _find_value_by_key("debug", obj.parameters_input, "optional")
	if debug is not None:
		if debug.lower()=="true":
			obj.debug = True
		elif debug.lower()=="false":
			obj.debug = False
	else:
		obj.debug = False

	obj.n_thread = _find_value_by_key("n_thread",obj.parameters_input, "default", 1, conversion=int)
	obj.mode = _find_value_by_key("mode", obj.parameters_input, "required", conversion=int)
	norm = _find_value_by_key("norm", obj.parameters_input, "optional")
	if norm is not None:
		if norm.lower()=="true":
			obj.norm = True
		elif norm.lower()=="false":
			obj.norm = False
	else:
		obj.norm = False

	mean = _find_value_by_key("mean", obj.parameters_input, "optional")
	if mean is not None:
		if mean.lower()=="true":
			obj.mean = True
		elif mean.lower()=="false":
			obj.mean = False
	else:
		obj.mean = False

	if obj.mean:
		mac_vel = _find_value_by_key("mac_vel", obj.parameters_input, "required")
		obj.mac_vel = [float(item) for item in mac_vel.split(",")]

		items = _find_value_by_key("filling_factor", obj.parameters_input, "required")
		obj.filling_factor = [float(item) for item in items.split(",")]

	# path to RH main folder
	rh_path = _find_value_by_key("rh_path", obj.parameters_input, "required")
	if rh_path.rstrip("\n")[-1]=="/":
		rh_path = rh_path.rstrip("/")
	obj.rh_path = rh_path

	# flag for HSE computation; by default we do HSE
	#obj.hydrostatic = _find_value_by_key("hydrostatic", obj.parameters_input, "default", 1, conversion=int)
	
	#--- get wavelength range and save it to file ('wave_file_path')
	obj.lmin = _find_value_by_key("wave_min", obj.parameters_input, "optional", conversion=float)
	obj.lmax = _find_value_by_key("wave_max", obj.parameters_input, "optional", conversion=float)
	obj.step = _find_value_by_key("wave_step", obj.parameters_input, "optional", conversion=float)
	obj.interpolate_obs = False
	if (obj.step is None) or (obj.lmin is None) or (obj.lmax is None):
		wave_grid_path = _find_value_by_key("wave_grid", obj.parameters_input, "required")
		wavetable = np.loadtxt(wave_grid_path)
		obj.lmin = min(wavetable)
		obj.lmax = max(wavetable)
		obj.step = wavetable[1] - wavetable[0]
		obj.interpolate_obs = True
	else:
		obj.lmin /= 10
		obj.lmax /= 10
		obj.step /= 10
		wavetable = np.arange(obj.lmin, obj.lmax+obj.step, obj.step)
	
	obj.wavelength_air = copy.deepcopy(wavetable)
	obj.wavelength_vacuum = wavetable
	obj.RH.set_wavetable(obj.wavelength_vacuum)

	# common parameters for all modes
	atm_range = get_atmosphere_range(obj.parameters_input)
	logtau_top = _find_value_by_key("logtau_top", obj.parameters_input, "default", -6,float)
	logtau_bot = _find_value_by_key("logtau_bot", obj.parameters_input, "default", 1, float)
	logtau_step = _find_value_by_key("logtau_step", obj.parameters_input, "default", 0.1, float)
	obj.noise = _find_value_by_key("noise", obj.parameters_input, "default", 1e-3, float)
	atm_type = _find_value_by_key("atm_type", obj.parameters_input, "default", "multi", str)
	atm_type = atm_type.lower()
	obj.atm_scale = _find_value_by_key("atm_scale", obj.parameters_input, "default", "tau", str)

	# read Opacity Fudge (OF) data
	if obj.of_mode:
		obj.of_fit_mode = _find_value_by_key("of_fit_mode", obj.parameters_input, "default", -1, float)
		
		if obj.of_fit_mode==-1:
			obj.of_mode = False

		of_file_path = _find_value_by_key("of_file", obj.parameters_input, "default", None, str)
		obj.of_scatt_flag = _find_value_by_key("of_scatt_flag", obj.parameters_input, "default", 0, int)
		if (obj.of_fit_mode>=0) or of_file_path:
			of_num, of_wave, of_value = read_OF_data(of_file_path)

	# get the name of the input line list
	obj.linelist_name = _find_value_by_key("linelist", obj.parameters_input, "required")
	# obj.linelist_name = linelist_path.split("/")[-1]
	# out = sp.run(f"cp {linelist_path} runs/{obj.wd}/{obj.linelist_name}",
	# 			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	# if out.returncode!=0:
	# 	print(str(out.stdout, "utf-8"))
	# 	sys.exit()

	#--- read data for different modus operandi
	if obj.mode<=0:
		read_mode_0(atm_range, atm_type, logtau_top, logtau_bot, logtau_step, obj)
	elif obj.mode>=1:
		read_inversion_base(atm_range, atm_type, logtau_top, logtau_bot, logtau_step, obj)
		if obj.mode==2:
			read_mode_2(obj)
		elif obj.mode==3:
			read_mode_3(obj)

		#--- determine number of local and global parameters
		obj.atmosphere.n_local_pars = 0
		for parameter in obj.atmosphere.nodes:
			obj.atmosphere.n_local_pars += len(obj.atmosphere.nodes[parameter])

		if obj.of_mode:
			obj.atmosphere.n_local_pars += of_num

		obj.atmosphere.n_global_pars = 0
		for parameter in obj.atmosphere.global_pars:
			obj.atmosphere.n_global_pars += obj.atmosphere.global_pars[parameter].shape[-1]
	else:
		print("--> Negative mode not supported. Soon to be RF calculation.")
		sys.exit()

	#--- if we have more threads than atmospheres, reduce the number of used threads
	if obj.n_thread > obj.atmosphere.nx*obj.atmosphere.ny:
		obj.n_thread = obj.atmosphere.nx*obj.atmosphere.ny
		print(f"Warning: reduced the number of threads to {obj.n_thread}.\n")

	#--- initialize Pool() object
	obj.pool = mp.Pool(obj.n_thread)

	#--- write OFs (to parallelize?)
	# if obj.of_mode:
	# 	obj.atmosphere.of_paths = []
	# 	for idx in range(obj.atmosphere.nx):
	# 		for idy in range(obj.atmosphere.ny):
	# 			fpath = f"{globin.cwd}/runs/{globin.wd}/ofs/of_{idx}_{idy}"
	# 			obj.atmosphere.of_paths.append(fpath)

	# 	# if we have 1D OF values, set equal values in all pixels
	# 	if of_value.ndim==1:
	# 		of_value = np.repeat(of_value[np.newaxis, :], obj.atmosphere.nx, axis=0)
	# 		of_value = np.repeat(of_value[:, np.newaxis, :], obj.atmosphere.ny, axis=1)
		
	# 	obj.atmosphere.of_num = of_num
	# 	obj.atmosphere.nodes["of"] = of_wave
	# 	obj.atmosphere.values["of"] = of_value

	# 	obj.parameter_scale["of"] = np.ones((obj.atmosphere.nx, obj.atmosphere.ny, of_num))

	# 	make_RH_OF_files(obj.atmosphere)

	if obj.mean:
		if len(obj.mac_vel)==1:
			vmac = obj.mac_vel[0]
			obj.mac_vel = np.ones(obj.atmosphere.nx * obj.atmosphere.ny) * vmac

			ff = obj.filling_factor[0]
			obj.filling_factor = np.ones(obj.atmosphere.nx * obj.atmosphere.ny) * ff

	# #--- for each thread make working directory inside rh/rhf1d directory
	# for pid in range(obj.n_thread):
	# 	if not os.path.exists(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid+1}"):
	# 		os.mkdir(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid+1}")

	idx,idy = np.meshgrid(np.arange(obj.atmosphere.nx), np.arange(obj.atmosphere.ny))
	obj.atmosphere.idx_meshgrid = idx.flatten()
	obj.atmosphere.idy_meshgrid = idy.flatten()

	# for idx in range(obj.atmosphere.nx):
	# 	for idy in range(globin.atm.ny):
	# 		fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
	# 		globin.atm.atm_name_list.append(fpath)

	if obj.mode>=1:
		#--- debugging variables initialization
		if obj.debug:
			Npar = obj.atmosphere.n_local_pars + obj.atmosphere.n_global_pars
			obj.rf_debug = np.zeros((obj.atmosphere.nx, obj.atmosphere.ny, obj.max_iter, Npar, len(obj.wavelength_air), 4))

			elements = []
			for parameter in obj.atmosphere.nodes:
				aux = np.zeros((obj.max_iter, obj.atmosphere.nx, obj.atmosphere.ny, len(obj.atmosphere.nodes[parameter])))
				elements.append((parameter, aux))
			obj.atmos_debug = dict(elements)

	#--- missing parameters
	# instrument broadening: R or instrument profile provided
	# strailight contribution

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

def read_mode_0(atm_range, atm_type, logtau_top, logtau_bot, logtau_step, obj):
	""" 
	Get parameters for synthesis.
	"""

	#--- default parameters
	obj.output_spectra_path = _find_value_by_key("spectrum", obj.parameters_input, "default", "spectrum.fits")
	vmac = _find_value_by_key("vmac", obj.parameters_input, "default", 0, float)

	#--- required parameters
	path_to_atmosphere = _find_value_by_key("cube_atmosphere", obj.parameters_input, "optional")
	if path_to_atmosphere is None:
		node_atmosphere_path = _find_value_by_key("node_atmosphere", obj.parameters_input, "optional")
		if node_atmosphere_path is None:
			obj.atmosphere = globin.falc
		else:
			obj.atmosphere = globin.construct_atmosphere_from_nodes(node_atmosphere_path, atm_range)
	else:
		obj.atmosphere = Atmosphere(fpath=path_to_atmosphere, atm_type=atm_type, atm_range=atm_range,
						logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)
	obj.atmosphere.vmac = np.abs(vmac) # [km/s]

	# reference atmosphere is the same as input one in synthesis mode
	obj.reference_atmosphere = copy.deepcopy(obj.atmosphere)

def read_inversion_base(atm_range, atm_type, logtau_top, logtau_bot, logtau_step, obj):
	# interpolation degree for Bezier polynomial
	obj.interp_degree = _find_value_by_key("interp_degree", obj.parameters_input, "default", 3, int)
	obj.svd_tolerance = _find_value_by_key("svd_tolerance", obj.parameters_input, "default", 1e-4, float)

	#--- default parameters
	obj.marq_lambda = _find_value_by_key("marq_lambda", obj.parameters_input, "default", 1e-3, float)
	obj.max_iter = _find_value_by_key("max_iter", obj.parameters_input, "default", 30, int)
	obj.chi2_tolerance = _find_value_by_key("chi2_tolerance", obj.parameters_input, "default", 1e-2, float)
	obj.ncycle = _find_value_by_key("ncycle", obj.parameters_input, "default", 1, int)
	obj.rf_type = _find_value_by_key("rf_type", obj.parameters_input, "default", "node", str)
	obj.weight_type = _find_value_by_key("weight_type", obj.parameters_input, "default", None, str)
	values = _find_value_by_key("weights", obj.parameters_input, "default", np.array([1,1,1,1], dtype=np.float64))
	if type(values)==str:
		values = values.split(",")
		obj.weights = np.array([float(item) for item in values], dtype=np.float64)
	vmac = _find_value_by_key("vmac", obj.parameters_input, "default", default_val=0, conversion=float)

	# initialize container for atmosphere which we invert
	obj.atmosphere = Atmosphere(logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step, atm_range=atm_range)

	#--- required parameters
	path_to_observations = _find_value_by_key("observation", obj.parameters_input, "required")
	obj.observation = Observation(path_to_observations, obs_range=atm_range)
	if obj.interpolate_obs or (not np.array_equal(obj.observation.wavelength, obj.wavelength_air)):
		obj.observation.interpolate(obj.wavelength_air)

	#--- optional parameters
	path_to_atmosphere = _find_value_by_key("cube_atmosphere", obj.parameters_input, "optional")
	if path_to_atmosphere is not None:
		obj.reference_atmosphere = Atmosphere(path_to_atmosphere, atm_type=atm_type, atm_range=atm_range,
					logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)
	# if user have not provided reference atmosphere try fidning node atmosphere
	else:
		path_to_node_atmosphere = _find_value_by_key("node_atmosphere", obj.parameters_input, "optional")
		if path_to_node_atmosphere is not None:
			obj.reference_atmosphere = globin.construct_atmosphere_from_nodes(path_to_node_atmosphere, atm_range)
		# if node atmosphere not given, set FAL C model as reference atmosphere
		else:
			obj.reference_atmosphere = obj.falc

	#--- initialize invert atmosphere data from reference atmosphere
	obj.atmosphere.interpolate_atmosphere(obj.reference_atmosphere.data[0,0,0], obj.reference_atmosphere.data)

	fpath = _find_value_by_key("rf_weights", obj.parameters_input, "optional")
	obj.wavs_weight = np.ones((obj.atmosphere.nx, obj.atmosphere.ny, len(obj.wavelength_air),4))
	if fpath is not None:
		lam, wI, wQ, wU, wV = np.loadtxt(fpath, unpack=True)
		# !!! Lenghts can be the same, but not the values in arrays. Needs to be changed.
		if len(lam)==len(obj.wavelength_air):
			obj.wavs_weight[...,0] = wI
			obj.wavs_weight[...,1] = wQ
			obj.wavs_weight[...,2] = wU
			obj.wavs_weight[...,3] = wV
		else:
			obj.wavs_weight[...,0] = interp1d(lam, wI)(obj.wavelength)
			obj.wavs_weight[...,1] = interp1d(lam, wQ)(obj.wavelength)
			obj.wavs_weight[...,2] = interp1d(lam, wU)(obj.wavelength)
			obj.wavs_weight[...,3] = interp1d(lam, wV)(obj.wavelength)
	
	# standard deviation of Gaussian kernel for macro broadening
	obj.atmosphere.vmac = vmac # [km/s]

	# if macro-turbulent velocity is negative, we fit it
	if obj.atmosphere.vmac<0:
		# check if initial macro veclocity is larger than the step size in wavelength
		vmac = np.abs(vmac)
		kernel_sigma = vmac*1e3 / globin.LIGHT_SPEED * (obj.lmin + obj.lmax)*0.5 / obj.step
		if kernel_sigma<0.5:
			vmac = 0.5 * globin.LIGHT_SPEED / ((obj.lmin + obj.lmax)*0.5) * obj.step
			vmac /= 1e3
			obj.limit_values["vmac"][0] = vmac
		
		obj.atmosphere.vmac = abs(vmac)
		obj.atmosphere.global_pars["vmac"] = np.array([obj.atmosphere.vmac])
		obj.parameter_scale["vmac"] = 1
	obj.reference_atmosphere.vmac = abs(vmac)

	#--- read initial node parameter values	
	fpath = _find_value_by_key("initial_atmosphere", obj.parameters_input, "optional")
	if fpath is not None:
		# read node parameters from .fits file that is inverted atmosphere
		# from older inversion run
		init_atmosphere = read_inverted_atmosphere(fpath, atm_range)
		obj.atmosphere.nodes = init_atmosphere.nodes
		obj.atmosphere.values = init_atmosphere.values
		obj.atmosphere.mask = init_atmosphere.mask
		if (obj.atmosphere.nx!=globin.observation.nx) or (obj.atmosphere.ny!=globin.observation.ny):
			print("--> Error in input.read_inverted_atmosphere()")
			print("    initial atmosphere does not have same dimensions")
			print("    as observations:")
			print(f"    -- atm = ({obj.atmosphere.nx},{obj.atmosphere.ny})")
			print(f"    -- obs = ({obj.observation.nx},{obj.observation.ny})")
			sys.exit()
	else:
		# read node parameters from .input file
		for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi"]:
			read_node_parameters(parameter, obj.parameters_input, obj)

	obj.atmosphere.hydrostatic = False
	if "temp" in obj.atmosphere.nodes:
		obj.atmosphere.hydrostatic = True

	#--- initialize the vz, mag and azimuth based on CoG and WFA methods (optional)
	# fpath = _find_value_by_key("lines2atm", obj.parameters_input, "optional")
	# if fpath:
	# 	initialize_atmos_pars(obj.atmosphere, obj.observation, fpath)

def read_mode_2(obj):
	#--- Kurucz line list for given spectral region
	obj.RLK_lines_text, obj.RLK_lines = globin.read_RLK_lines(obj.linelist_name)

	#--- line parameters to be fit
	line_pars_path = _find_value_by_key("line_parameters", obj.parameters_input, "optional")

	if line_pars_path:
		# if we provided line parameters for fit, read those parameters
		lines_to_fit = globin.read_init_line_parameters(line_pars_path)

		# get log(gf) parameters from line list
		aux_values = [line.loggf for line in lines_to_fit if line.loggf is not None]
		aux_lineNo = [line.lineNo for line in lines_to_fit if line.loggf is not None]
		loggf_min = [line.loggf_min for line in lines_to_fit if line.loggf is not None]
		loggf_max = [line.loggf_max for line in lines_to_fit if line.loggf is not None]
		globin.limit_values["loggf"] = np.vstack((loggf_min, loggf_max)).T
		obj.parameter_scale["loggf"] = np.ones((obj.atmosphere.nx, obj.atmosphere.ny, len(aux_values)))

		obj.atmosphere.global_pars["loggf"] = np.zeros((obj.atmosphere.nx, obj.atmosphere.ny, len(aux_values)))
		obj.atmosphere.line_no["loggf"] = np.zeros((len(aux_lineNo)), dtype=np.int)

		obj.atmosphere.global_pars["loggf"][:,:] = aux_values
		obj.atmosphere.line_no["loggf"][:] = aux_lineNo

		# get dlam parameters from lines list
		aux_values = [line.dlam for line in lines_to_fit if line.dlam is not None]
		aux_lineNo = [line.lineNo for line in lines_to_fit if line.dlam is not None]
		dlam_min = [line.dlam_min for line in lines_to_fit if line.dlam is not None]
		dlam_max = [line.dlam_max for line in lines_to_fit if line.dlam is not None]
		globin.limit_values["dlam"] = np.vstack((dlam_min, dlam_max)).T
		obj.parameter_scale["dlam"] = np.ones((obj.atmosphere.nx, obj.atmosphere.ny, len(aux_values)))

		obj.atmosphere.global_pars["dlam"] = np.zeros((obj.atmosphere.nx, obj.atmosphere.ny, len(aux_values)))
		obj.atmosphere.line_no["dlam"] = np.zeros((len(aux_lineNo)), dtype=np.int)

		obj.atmosphere.global_pars["dlam"][:,:] = aux_values
		obj.atmosphere.line_no["dlam"][:] = aux_lineNo

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

def read_mode_3(obj):
	#--- Kurucz line list for given spectral region
	obj.RLK_lines_text, obj.RLK_lines = globin.read_RLK_lines(obj.linelist_name)

	#--- line parameters to be fit
	line_pars_path = _find_value_by_key("line_parameters", obj.parameters_input, "optional")

	if line_pars_path:
		# if we provided line parameters for fit, read those parameters
		lines_to_fit = globin.read_init_line_parameters(line_pars_path)

		# get log(gf) parameters from line list
		aux_values = [line.loggf for line in lines_to_fit if line.loggf is not None]
		aux_lineNo = [line.lineNo for line in lines_to_fit if line.loggf is not None]
		loggf_min = [line.loggf_min for line in lines_to_fit if line.loggf is not None]
		loggf_max = [line.loggf_max for line in lines_to_fit if line.loggf is not None]
		globin.limit_values["loggf"] = np.vstack((loggf_min, loggf_max)).T
		obj.parameter_scale["loggf"] = np.ones((1,1,len(aux_values)))

		obj.atmosphere.global_pars["loggf"] = np.zeros((1,1,len(aux_values)))
		obj.atmosphere.line_no["loggf"] = np.zeros((len(aux_lineNo)), dtype=np.int)

		obj.atmosphere.global_pars["loggf"][0,0] = aux_values
		obj.atmosphere.line_no["loggf"][:] = aux_lineNo

		# get dlam parameters from lines list
		aux_values = [line.dlam for line in lines_to_fit if line.dlam is not None]
		aux_lineNo = [line.lineNo for line in lines_to_fit if line.dlam is not None]
		dlam_min = [line.dlam_min for line in lines_to_fit if line.dlam is not None]
		dlam_max = [line.dlam_max for line in lines_to_fit if line.dlam is not None]
		globin.limit_values["dlam"] = np.vstack((dlam_min, dlam_max)).T
		obj.parameter_scale["dlam"] = np.ones((1,1,len(aux_values)))

		obj.atmosphere.global_pars["dlam"] = np.zeros((1,1,len(aux_values)))
		obj.atmosphere.line_no["dlam"] = np.zeros((len(aux_lineNo)), dtype=np.int)

		obj.atmosphere.global_pars["dlam"][0,0] = aux_values
		obj.atmosphere.line_no["dlam"][:] = aux_lineNo

		# write down initial atomic lines values
		# globin.write_line_parameters(obj.atmosphere.line_lists_path[0],
		# 						   globin.atm.global_pars["loggf"][0,0], globin.atm.line_no["loggf"],
		# 						   globin.atm.global_pars["dlam"][0,0], globin.atm.line_no["dlam"])
	else:
		print("No atomic parameters to fit. You sure?\n")

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

def read_node_parameters(parameter, text, obj):
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
	atmosphere = obj.atmosphere

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

		obj.parameter_scale[parameter] = np.ones((atmosphere.nx, atmosphere.ny, len(atmosphere.nodes[parameter])))

def read_inverted_atmosphere(fpath, atm_range=[0,None,0,None]):
	"""
	Read atmosphere retrieved after inversion and store it in
	Atmosphere() object. We load fully stratified atmosphere and
	node position and values in nodes for all inverted 
	atmospheric parameters.

	Parameters:
	-----------
	fpath : string
		file path to inverted atmosphere. It should be .fits file.
	atm_range : list
		list containing [xmin,xmax,ymin,ymax] that define part of the
		cube to be read.

	Return:
	-------
	atmos : globin.atmos.Atmosphere() object
	"""
	try:
		hdu_list = fits.open(fpath)
	except:
		print("--> Error in input.read_inverted_atmosphere()")
		print(f"    Atmosphere file with path '{fpath}' does not exist.")
		sys.exit()

	xmin, xmax, ymin, ymax = atm_range

	data = hdu_list[0].data[xmin:xmax, ymin:ymax]
	nx, ny, npar, nz = data.shape

	atmos = globin.Atmosphere(nx=nx, ny=ny, nz=nz)
	aux = data.astype(np.float64, order="C", copy=True) # because of the pyrh module
	atmos.data = aux
	atmos.logtau = data[0,0,0]
	atmos.header = hdu_list[0].header

	for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi", "of"]:
		try:
			ind = hdu_list.index_of(parameter)
			data = hdu_list[ind].data[:, xmin:xmax, ymin:ymax, :]
			_, nx, ny, nnodes = data.shape

			atmos.nodes[parameter] = data[0,0,0]
			# angles are saved in radians, no need to convert them here
			if parameter=="gamma":
				# atmos.values[parameter] = np.tan(data[1]/2)
				atmos.values[parameter] = np.cos(data[1])
				atmos.values[parameter] = data[1]
			elif parameter=="chi":
				# atmos.values[parameter] = np.tan(data[1]/4)
				# atmos.values[parameter] = np.cos(data[1])
				atmos.values[parameter] = data[1]
			else:
				atmos.values[parameter] = data[1]
			atmos.mask[parameter] = np.ones(len(atmos.nodes[parameter]))

			globin.parameter_scale[parameter] = np.ones((atmos.nx, atmos.ny, nnodes))
		except:
			pass

	try:
		ind = hdu_list.index_of("Continuum_Opacity")
		atmos.chi_c = hdu_list[ind].data
	except:
		atmos.chi_c = None

	return atmos

def read_multi(fpath):
	"""
	Read MULTI type atmosphere data and store it in
	Atmosphere() object.

	Parameter:
	----------
	fpath : string
		path to the MULTI type atmosphere.

	Return:
	-------
	atmos : globin.atmos.Atmosphere() object
	"""
	lines = open(fpath, "r").readlines()

	# remove commented lines
	lines = [line.rstrip("\n") for line in lines if "*" not in line]

	# get number of depth points
	ndpth = int(lines[3].replace(" ", ""))

	nz = ndpth
	nx, ny = 1, 1

	atmos = globin.Atmosphere(nx=nx, ny=ny, nz=nz)

	for i_ in range(ndpth):
		# read first part of the atmosphere
		lista = list(filter(None,lines[4+i_].split(" ")))
		atmos.data[0,0,0,i_], \
		atmos.data[0,0,1,i_], \
		atmos.data[0,0,2,i_], \
		atmos.data[0,0,3,i_], \
		atmos.data[0,0,4,i_] = [float(element) for element in lista]

		# read H populations
		lista = list(filter(None,lines[4+ndpth+i_].split(" ")))
		atmos.data[0,0,8,i_], \
		atmos.data[0,0,9,i_], \
		atmos.data[0,0,10,i_], \
		atmos.data[0,0,11,i_], \
		atmos.data[0,0,12,i_], \
		atmos.data[0,0,13,i_] = [float(element) for element in lista]

	atmos.logtau = atmos.data[0,0,0]

	return atmos

def read_spinor(fpath):
	atmos_data = np.loadtxt(fpath, skiprows=1, dtype=np.float64).T
	# nz = atmos_data.shape[1]
	
	# atmos = globin.Atmosphere(nx=1, ny=1, nz=nz)
	# atmos.logtau = atmos_data[0]
	# atmos.data = atmos_data
	
	atmos = globin.atmos.convert_atmosphere(atmos_data[0], atmos_data, "spinor")

	return atmos

def read_sir(self):
	pass

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

	idx,idy = np.meshgrid(np.arange(atmos.nx), np.arange(atmos.ny))
	globin.idx = idx.flatten()
	globin.idy = idy.flatten()

	# !!! added by hand..
	globin.pool = mp.Pool(2)

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
		inds = np.empty((atmos.nx, atmos.ny), dtype=np.int64)
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				try:
					inds[idx,idy] = argrelextrema(x[idx,idy], np.less, order=3)[0][0]
				except:
					print(idx,idy)
					sys.exit()
		return inds

	obs = copy.deepcopy(obs_in)
	dlam = obs.wavelength[1] - obs.wavelength[0]
	wavs = obs.wavelength
	if norm:
		if globin.norm:
			obs.norm()
		else:
			globin.norm = True
			obs.norm()
			globin.norm = False

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
				try:
					x[idx,idy] = obs.wavelength[mmin:mmax]
					si[idx,idy] = obs.spec[idx,idy,mmin:mmax,0]
					sq[idx,idy] = obs.spec[idx,idy,mmin:mmax,1]
					su[idx,idy] = obs.spec[idx,idy,mmin:mmax,2]
					sv[idx,idy] = obs.spec[idx,idy,mmin:mmax,3]
				except:
					print(idx,idy)
					print(mmin, mmax)
					
					sys.exit()

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
					if mag[idx,idy] > globin.limit_values["mag"][1]:
						mag[idx,idy] = globin.limit_values["mag"][1]
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

def make_RH_OF_files(atmos):
	"""
	[12.04.2022.]
	This is not going to work in general OF correction :) Or at least
	for wavelengths belowe 210nm, for metal correction.
	"""
	for fpath in atmos.of_paths:
		lista = fpath.split("_")
		idx = int(lista[-2])
		idy = int(lista[-1])

		out = open(fpath, "w")
		
		#--- for constant OF correction
		if atmos.of_num==1:
			out.write("{:4d}\n".format(4))

			fudge = atmos.values["of"][idx,idy,0]
			
			# start point with 0 fudge factor
			out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(globin.wavelength_vacuum[0]-0.0002, 0, 0, 0))

			# we correct begining and end wavelength for -/+ 0.0001 because of interpolation inside RH;
			# if not corrected, these wavelengths would be extrapolated and not interpolated
			if atmos.nodes["of"][0]>=210:
				if globin.of_scatt_flag!=0:
					out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(globin.wavelength_vacuum[0]-0.0001, fudge, fudge, 0))
					out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(globin.wavelength_vacuum[-1]+0.0001, fudge, fudge, 0))
				else:
					out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(globin.wavelength_vacuum[0]-0.0001, fudge, 0, 0))
					out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(globin.wavelength_vacuum[-1]+0.0001, fudge, 0, 0))
			else:
				out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(globin.wavelength_vacuum[0]-0.0001, 0, 0, fudge))
				out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(globin.wavelength_vacuum[-1]+0.0001, 0, 0, fudge))

			# end point with 0 fudge factor
			out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(globin.wavelength_vacuum[-1]+0.0002, 0, 0, 0))
		#--- for multi-wavelength OF correction
		else:
			# two more points are added (one at the begining and one at the end of interval)
			out.write("{:4d}\n".format(atmos.of_num+2))
			
			out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(atmos.nodes["of"][0]-0.0002, 0, 0, 0))
			if globin.of_scatt_flag!=0:
				out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(atmos.nodes["of"][0]-0.0001, atmos.values["of"][idx,idy,0], atmos.values["of"][idx,idy,0], 0))
			else:
				out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(atmos.nodes["of"][0]-0.0001, atmos.values["of"][idx,idy,0], 0, 0))

			for i_ in range(1,atmos.of_num-1):
				wave = atmos.nodes["of"][i_]
				fudge = atmos.values["of"][idx,idy,i_]
				if wave>=210:
					if globin.of_scatt_flag!=0:
						out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(wave, fudge, fudge, 0))
					else:
						out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(wave, fudge, 0, 0))
				else:
					out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(wave, 0, 0, fudge))
			
			if globin.of_scatt_flag!=0:
				out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(atmos.nodes["of"][-1]+0.0001, atmos.values["of"][idx,idy,-1], atmos.values["of"][idx,idy,-1], 0))	
			else:
				out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(atmos.nodes["of"][-1]+0.0001, atmos.values["of"][idx,idy,-1], 0, 0))
			
			out.write("{:9.4f}  {:5.4f}  {:5.4f}  {:5.4f}\n".format(atmos.nodes["of"][-1]+0.0002, 0, 0, 0))

		out.close()