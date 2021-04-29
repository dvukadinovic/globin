import os
import sys
import numpy as np
import multiprocessing as mp
import re
import copy
import subprocess as sp
from scipy.interpolate import interp1d
from astropy.io import fits

from .atmos import Atmosphere
from .spec import Observation
from .rh import write_wavs

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

def read_input(run_name, globin_input_name="params.input", rh_input_name="keyword.input"):
	if (rh_input_name is not None) and (globin_input_name is not None):
		read_input_files(run_name, globin_input_name, rh_input_name)
	else:
		if rh_input_name is None:
			print(f"  There is no path for globin input file.")
		if globin_input_name is None:
			print(f"  There is no path for RH input file.")
		sys.exit()

def read_input_files(run_name, globin_input_name, rh_input_name):
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
	globin.wd = run_name # --> to change into globin.run_name? There are a lot cases and in other files! Be careful!

	# make runs directory if not existing
	# here we store all runs with different run_name
	if not os.path.exists("runs"):
		os.mkdir("runs")

	# make directory for specified run with provided 'run_name'
	if not os.path.exists(f"runs/{globin.wd}"):
		os.mkdir(f"runs/{globin.wd}")

	# make directory in which atmospheres will be extracted for given run
	if not os.path.exists(f"runs/{globin.wd}/atmospheres"):
		os.mkdir(f"runs/{globin.wd}/atmospheres")
	else:
		# clean directory if it exists (maybe we have atmospheres extracted
		# from some other cube); it takes few miliseconds, so not a big deal
		sp.run(f"rm runs/{globin.wd}/atmospheres/*",
			shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)

	# copy all RH input files into run directory
	# keyword.input and kurucz.input are changed accordingly during input reading
	# and saved back into 'runs/{run_name}' directory
	sp.run(f"cp *.input runs/{run_name}/",
		shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

	globin.rh_input_name = rh_input_name

	#--- get parameters from RH input file
	text = open(globin.rh_input_name, "r").read()
	globin.keyword_input = text

	wave_file_path = _find_value_by_key("WAVETABLE", globin.keyword_input, "required")
	wave_file_path = wave_file_path.split("/")[-1]
	globin.rh_spec_name = _find_value_by_key("SPECTRUM_OUTPUT", globin.keyword_input, "default", "spectrum.out")
	globin.solve_ne = _find_value_by_key("SOLVE_NE", globin.keyword_input, "optional")
	globin.hydrostatic = _find_value_by_key("HYDROSTATIC", globin.keyword_input, "optional")
	globin.kurucz_input_fname = _find_value_by_key("KURUCZ_DATA", globin.keyword_input, "required")
	globin.rf_file_path = _find_value_by_key("RF_OUTPUT", globin.keyword_input, "default", "rfs.out")

	#--- get parameters from globin input file
	text = open(globin_input_name, "r").read()
	globin.parameters_input = text

	globin.n_thread = _find_value_by_key("n_thread",globin.parameters_input, "default", 1, conversion=int)
	globin.mode = _find_value_by_key("mode", globin.parameters_input, "required", conversion=int)
	
	# path to RH main folder
	rh_path = _find_value_by_key("rh_path", globin.parameters_input, "required")
	if rh_path.rstrip("\n")[-1]=="/":
		rh_path = rh_path.rstrip("/")
	globin.rh_path = rh_path
	
	#--- get wavelength range and save it to file ('wave_file_path')
	globin.lmin = _find_value_by_key("wave_min", globin.parameters_input, "optional", conversion=float)
	globin.lmax = _find_value_by_key("wave_max", globin.parameters_input, "optional", conversion=float)
	globin.step = _find_value_by_key("wave_step", globin.parameters_input, "optional", conversion=float)
	if (globin.step is None) and (globin.lmin is None) and (globin.lmax is None):
		wave_grid_path = _find_value_by_key("wave_grid", globin.parameters_input, "required")
		globin.wavelength = np.loadtxt(wave_grid_path)/10
		globin.lmin = min(globin.wavelength)
		globin.lmax = max(globin.wavelength)
		globin.step = globin.wavelength[1] - globin.wavelength[0]
	else:
		globin.lmin /= 10
		globin.lmax /= 10
		globin.step /= 10
		globin.wavelength = np.arange(globin.lmin, globin.lmax+globin.step, globin.step)
	write_wavs(globin.wavelength, f"runs/{globin.wd}/" + wave_file_path)

	# set value of WAVETABLE in 'keyword.input' file
	globin.keyword_input = set_keyword(globin.keyword_input, "WAVETABLE", f"{globin.cwd}/runs/{globin.wd}/{wave_file_path}", f"runs/{globin.wd}/{globin.rh_input_name}")

	# common parameters for all modes
	atm_range = get_atmosphere_range()
	logtau_top = _find_value_by_key("logtau_top", globin.parameters_input, "default", -6,float)
	logtau_bot = _find_value_by_key("logtau_bot", globin.parameters_input, "default", 1, float)
	logtau_step = _find_value_by_key("logtau_step", globin.parameters_input, "default", 0.1, float)
	globin.noise = _find_value_by_key("noise", globin.parameters_input, "default", 1e-3, float)
	atm_type = _find_value_by_key("atm_type", globin.parameters_input, "default", "multi", str)
	atm_type = atm_type.lower()

	# get the name of the input line list
	linelist_path = _find_value_by_key("linelist", globin.parameters_input, "required")
	globin.linelist_name = linelist_path.split("/")[-1]
	out = sp.run(f"cp {linelist_path} runs/{globin.wd}/{globin.linelist_name}",
				shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	if out.returncode!=0:
		print(str(out.stdout, "utf-8"))
		sys.exit()

	#--- read data for different modus operandi
	if globin.mode==0:
		read_mode_0(atm_range, atm_type, logtau_top, logtau_bot, logtau_step)
		globin.atm.line_lists_path = [f"runs/{globin.wd}/{globin.linelist_name}"]
	elif globin.mode>=1:
		read_inversion_base(atm_range, atm_type, logtau_top, logtau_bot, logtau_step)
		globin.atm.line_lists_path = [f"runs/{globin.wd}/{globin.linelist_name}"]
		if globin.mode==2:
			read_mode_2()
		elif globin.mode==3:
			read_mode_3()

		#--- determine number of local and global parameters
		globin.atm.n_local_pars = 0
		for parameter in globin.atm.nodes:
			globin.atm.n_local_pars += len(globin.atm.nodes[parameter])

		globin.atm.n_global_pars = 0
		for parameter in globin.atm.global_pars:
			globin.atm.n_global_pars += globin.atm.global_pars[parameter].shape[-1]
	else:
		print("--> Negative mode not supported. Soon to be RF calculation.")
		sys.exit()

	#--- if we have more threads than atmospheres, reduce the number of used threads
	if globin.mode>=1:
		if globin.n_thread > globin.atm.nx*globin.atm.ny:
			globin.n_thread = globin.atm.nx*globin.atm.ny
			print(f"\nWarning: reduced the number of threads to {globin.n_thread}.\n")

	#--- initialize Pool() object
	globin.pool = mp.Pool(globin.n_thread)

	#--- for each thread make working directory inside rh/rhf1d directory
	for pid in range(globin.n_thread):
		if not os.path.exists(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid+1}"):
			os.mkdir(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid+1}")

	#--- missing parameters
	# instrument broadening: R or instrument profile provided
	# strailight contribution
	# opacity fudge coefficients
	# norm --> flag for normalized spectra

def get_atmosphere_range():
	#--- determine which observations from cube to take into consideration
	aux = _find_value_by_key("range", globin.parameters_input, "default", [1,None,1,None])
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

def read_mode_0(atm_range, atm_type, logtau_top, logtau_bot, logtau_step):
	""" 
	Get parameters for synthesis.
	"""

	#--- default parameters
	globin.output_spectra_path = _find_value_by_key("spectrum", globin.parameters_input, "default", "spectrum.fits")
	vmac = _find_value_by_key("vmac", globin.parameters_input, "default", 0, float)

	#--- required parameters
	path_to_atmosphere = _find_value_by_key("cube_atmosphere", globin.parameters_input, "optional")
	if path_to_atmosphere is None:
		node_atmosphere_path = _find_value_by_key("node_atmosphere", globin.parameters_input, "required")
		globin.atm = globin.construct_atmosphere_from_nodes(node_atmosphere_path, atm_range)
		globin.atm.split_cube()
	else:
		globin.atm = Atmosphere(fpath=path_to_atmosphere, atm_type=atm_type, atm_range=atm_range,
						logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)
		globin.atm.split_cube()
	globin.atm.vmac = np.abs(vmac) # [km/s]

	# standard deviation of Gaussian kernel for macro broadening
	globin.atm.sigma = lambda vmac: vmac / globin.LIGHT_SPEED * (globin.lmin + globin.lmax)*0.5 / globin.step

	# reference atmosphere is the same as input one in synthesis mode
	globin.ref_atm = copy.deepcopy(globin.atm)

def read_inversion_base(atm_range, atm_type, logtau_top, logtau_bot, logtau_step):
	# integration degree for Bezier interpolation
	globin.interp_degree = _find_value_by_key("interp_degree", globin.parameters_input, "default", 3, int)

	#--- default parameters
	globin.marq_lambda = _find_value_by_key("marq_lambda", globin.parameters_input, "default", 1e-3, float)
	globin.max_iter = _find_value_by_key("max_iter", globin.parameters_input, "default", 30, int)
	globin.chi2_tolerance = _find_value_by_key("chi2_tolerance", globin.parameters_input, "default", 1e-2, float)
	globin.ncycle = _find_value_by_key("ncycle", globin.parameters_input, "default", 1, int)
	globin.rf_type = _find_value_by_key("rf_type", globin.parameters_input, "default", "node", str)
	values = _find_value_by_key("weights", globin.parameters_input, "default", np.array([1,1,1,1], dtype=np.float64))
	if type(values)==str:
		values = values.split(",")
		globin.weights = np.array([float(item) for item in values], dtype=np.float64)
	vmac = _find_value_by_key("vmac", globin.parameters_input, "default", default_val=0, conversion=float)

	# initialize container for atmosphere which we invert
	globin.atm = Atmosphere(logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)

	#--- required parameters
	path_to_observations = _find_value_by_key("observation", globin.parameters_input, "required")
	globin.obs = Observation(path_to_observations, atm_range)
	# set dimensions for atmosphere same as dimension of observations
	globin.atm.nx = globin.obs.nx
	globin.atm.ny = globin.obs.ny
	for idx in range(globin.atm.nx):
		for idy in range(globin.atm.ny):
			globin.atm.atm_name_list.append(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
	
	#--- optional parameters
	path_to_atmosphere = _find_value_by_key("cube_atmosphere", globin.parameters_input, "optional")
	if path_to_atmosphere is not None:
		globin.ref_atm = Atmosphere(path_to_atmosphere, atm_type=atm_type, atm_range=atm_range,
					logtau_top=logtau_top, logtau_bot=logtau_bot, logtau_step=logtau_step)
	# if user have not provided reference atmosphere try fidning node atmosphere
	else:
		path_to_node_atmosphere = _find_value_by_key("node_atmosphere", globin.parameters_input, "optional")
		if path_to_node_atmosphere is not None:
			globin.ref_atm = globin.construct_atmosphere_from_nodes(path_to_node_atmosphere, atm_range)
		# if node atmosphere not given, set FAL C model as reference atmosphere
		else:
			globin.ref_atm = globin.falc

	#--- initialize invert atmosphere data from reference atmosphere
	globin.atm.interpolate_atmosphere(globin.ref_atm.data[0,0,0], globin.ref_atm.data)

	fpath = _find_value_by_key("rf_weights", globin.parameters_input, "optional")
	globin.wavs_weight = np.ones((globin.atm.nx, globin.atm.ny, len(globin.wavelength),4))
	if fpath is not None:
		lam, wI, wQ, wU, wV = np.loadtxt(fpath, unpack=True)
		if len(lam)==len(globin.wavelength):
			globin.wavs_weight[...,0] = wI
			globin.wavs_weight[...,1] = wQ
			globin.wavs_weight[...,2] = wU
			globin.wavs_weight[...,3] = wV
		else:
			globin.wavs_weight[...,0] = interp1d(lam, wI)(globin.wavelength)
			globin.wavs_weight[...,1] = interp1d(lam, wQ)(globin.wavelength)
			globin.wavs_weight[...,2] = interp1d(lam, wU)(globin.wavelength)
			globin.wavs_weight[...,3] = interp1d(lam, wV)(globin.wavelength)
	
	# standard deviation of Gaussian kernel for macro broadening
	globin.atm.vmac = vmac # [km/s]
	globin.atm.sigma = lambda vmac: vmac*1e3 / globin.LIGHT_SPEED * (globin.lmin + globin.lmax)*0.5 / globin.step

	# if macro-turbulent velocity is negative, we fit it
	if globin.atm.vmac<0:
		# check if initial macro veclocity is larger than the step size in wavelength
		vmac = np.abs(vmac)
		kernel_sigma = globin.atm.sigma(vmac)
		if kernel_sigma<0.5:
			vmac = 0.5 * globin.LIGHT_SPEED / ((globin.lmin + globin.lmax)*0.5) * globin.step
			vmac /= 1e3
			globin.limit_values["vmac"][0] = vmac
		
		globin.atm.vmac = abs(vmac)
		globin.atm.global_pars["vmac"] = np.array([globin.atm.vmac])

	globin.ref_atm.vmac = abs(vmac)

	#--- read initial node parameter values	
	fpath = _find_value_by_key("initial_atmosphere", globin.parameters_input, "optional")
	if fpath is not None:
		# read node parameters from .fits file
		load_node_data(fpath, atm_range)
	else:
		# read node parameters from .input file
		for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi"]:
			read_node_parameters(parameter, globin.parameters_input)

def read_mode_2():
	# this is pixel-by-pixel inversion and for atomic parameters
	# make directory in which line list will be saved for each atmosphere
	if not os.path.exists(f"runs/{globin.wd}/line_lists"):
		os.mkdir(f"runs/{globin.wd}/line_lists")
	else:
		# clean directory if it exists (maybe we have line lists saved
		# from some other run); it takes few miliseconds, so not a big deal
		sp.run(f"rm runs/{globin.wd}/line_lists/*",
			shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)

	#--- Kurucz line list for given spectral region
	globin.RLK_lines_text, globin.RLK_lines = globin.read_RLK_lines(globin.linelist_name)

	#--- line parameters to be fit
	line_pars_path = _find_value_by_key("line_parameters", globin.parameters_input, "optional")

	if line_pars_path:
		# if we provided line parameters for fit, read those parameters
		lines_to_fit = globin.read_init_line_parameters(line_pars_path)

		# get log(gf) parameters from line list
		aux_values = [line.loggf for line in lines_to_fit if line.loggf is not None]
		aux_lineNo = [line.lineNo for line in lines_to_fit if line.loggf is not None]
		loggf_min = [line.loggf_min for line in lines_to_fit if line.loggf is not None]
		loggf_max = [line.loggf_max for line in lines_to_fit if line.loggf is not None]
		globin.limit_values["loggf"] = np.vstack((loggf_min, loggf_max)).T
		globin.parameter_scale["loggf"] = np.ones((globin.atm.nx, globin.atm.ny, len(aux_values)))

		globin.atm.global_pars["loggf"] = np.zeros((globin.atm.nx, globin.atm.ny, len(aux_values)))
		globin.atm.line_no["loggf"] = np.zeros((len(aux_lineNo)), dtype=np.int)

		globin.atm.global_pars["loggf"][:,:] = aux_values
		globin.atm.line_no["loggf"][:] = aux_lineNo

		# get dlam parameters from lines list
		aux_values = [line.dlam for line in lines_to_fit if line.dlam is not None]
		aux_lineNo = [line.lineNo for line in lines_to_fit if line.dlam is not None]
		dlam_min = [line.dlam_min for line in lines_to_fit if line.dlam is not None]
		dlam_max = [line.dlam_max for line in lines_to_fit if line.dlam is not None]
		globin.limit_values["dlam"] = np.vstack((dlam_min, dlam_max)).T
		globin.parameter_scale["dlam"] = np.ones((globin.atm.nx, globin.atm.ny, len(aux_values)))

		globin.atm.global_pars["dlam"] = np.zeros((globin.atm.nx, globin.atm.ny, len(aux_values)))
		globin.atm.line_no["dlam"] = np.zeros((len(aux_lineNo)), dtype=np.int)

		globin.atm.global_pars["dlam"][:,:] = aux_values
		globin.atm.line_no["dlam"][:] = aux_lineNo

		# write these data into files

		# make list of line lists paths (aka names)
		globin.atm.line_lists_path = []
		for idx in range(globin.atm.nx):
			for idy in range(globin.atm.ny):
				fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
				globin.atm.line_lists_path.append(fpath)

				write_line_parameters(fpath,
									   globin.atm.global_pars["loggf"][idx,idy], globin.atm.line_no["loggf"],
									   globin.atm.global_pars["dlam"][idx,idy], globin.atm.line_no["dlam"])
	else:
		print("No atomic parameters to fit. You sure?\n")

def read_mode_3():
	#--- Kurucz line list for given spectral region
	globin.RLK_lines_text, globin.RLK_lines = globin.read_RLK_lines(globin.linelist_name)

	#--- line parameters to be fit
	line_pars_path = _find_value_by_key("line_parameters", globin.parameters_input, "optional")

	if line_pars_path:
		# if we provided line parameters for fit, read those parameters
		lines_to_fit = globin.read_init_line_parameters(line_pars_path)

		# get log(gf) parameters from line list
		aux_values = [line.loggf for line in lines_to_fit if line.loggf is not None]
		aux_lineNo = [line.lineNo for line in lines_to_fit if line.loggf is not None]
		loggf_min = [line.loggf_min for line in lines_to_fit if line.loggf is not None]
		loggf_max = [line.loggf_max for line in lines_to_fit if line.loggf is not None]
		globin.limit_values["loggf"] = np.vstack((loggf_min, loggf_max)).T
		globin.parameter_scale["loggf"] = np.ones((1,1,len(aux_values)))

		globin.atm.global_pars["loggf"] = np.zeros((1,1,len(aux_values)))
		globin.atm.line_no["loggf"] = np.zeros((len(aux_lineNo)), dtype=np.int)

		globin.atm.global_pars["loggf"][0,0] = aux_values
		globin.atm.line_no["loggf"][:] = aux_lineNo

		# get dlam parameters from lines list
		aux_values = [line.dlam for line in lines_to_fit if line.dlam is not None]
		aux_lineNo = [line.lineNo for line in lines_to_fit if line.dlam is not None]
		dlam_min = [line.dlam_min for line in lines_to_fit if line.dlam is not None]
		dlam_max = [line.dlam_max for line in lines_to_fit if line.dlam is not None]
		globin.limit_values["dlam"] = np.vstack((dlam_min, dlam_max)).T
		globin.parameter_scale["dlam"] = np.ones((1,1,len(aux_values)))

		globin.atm.global_pars["dlam"] = np.zeros((1,1,len(aux_values)))
		globin.atm.line_no["dlam"] = np.zeros((len(aux_lineNo)), dtype=np.int)

		globin.atm.global_pars["dlam"][0,0] = aux_values
		globin.atm.line_no["dlam"][:] = aux_lineNo

		# globin.atm.line_lists_path = [f"runs/{globin.wd}/{linelist_path.split('/')[-1]}"]

		# write down initial atomic lines values
		globin.write_line_parameters(globin.atm.line_lists_path[0],
								   globin.atm.global_pars["loggf"][0,0], globin.atm.line_no["loggf"],
								   globin.atm.global_pars["dlam"][0,0], globin.atm.line_no["dlam"])
	else:
		print("No atomic parameters to fit. You sure?\n")

def write_line_parameters(fpath, loggf_val, loggf_no, dlam_val, dlam_no):
	"""
	Write out full Kurucz line list for all parameters.
	"""
	# out = open(self.RLK_path, "w")
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
	# out = open(self.RLK_path, "w")
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

def read_node_parameters(parameter, text):
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
	nodes = _find_value_by_key(f"nodes_{parameter}", text, "optional")
	values = _find_value_by_key(f"nodes_{parameter}_values", text, "optional")
	mask = _find_value_by_key(f"nodes_{parameter}_mask", text, "optional")
	
	if (nodes is not None) and (values is not None):
		globin.atm.nodes[parameter] = [float(item) for item in nodes.split(",")]
		
		values = [float(item) for item in values.split(",")]
		if len(values)!=len(globin.atm.nodes[parameter]):
			sys.exit(f"Number of nodes and values for {parameter} are not the same!")

		matrix = np.zeros((globin.atm.nx, globin.atm.ny, len(globin.atm.nodes[parameter])), dtype=np.float64)
		matrix[:,:] = copy.deepcopy(values)
		if parameter=="mag":
			globin.atm.values[parameter] = copy.deepcopy(matrix) / 1e4
		elif parameter=="gamma" or parameter=="chi":
			matrix *= np.pi/180
			globin.atm.values[parameter] = copy.deepcopy(matrix)
		else:
			globin.atm.values[parameter] = copy.deepcopy(matrix)
		
		if mask is None:
			globin.atm.mask[parameter] = np.ones(len(globin.atm.nodes[parameter]))
		else:
			mask = [float(item) for item in mask.split(",")]
			globin.atm.mask[parameter] = np.array(mask)

		globin.parameter_scale[parameter] = np.ones((globin.atm.nx, globin.atm.ny, len(globin.atm.nodes[parameter])))

def load_node_data(fpath, atm_range):
	hdu_list = fits.open(fpath)

	xmin, xmax, ymin, ymax = atm_range

	for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi"]:
		try:
			ind = hdu_list.index_of(parameter)
			data = hdu_list[ind].data[:, xmin:xmax, ymin:ymax, :]
			_, nx, ny, nnodes = data.shape

			globin.atm.nx, globin.atm.ny = nx, ny

			if globin.atm.nx!=globin.obs.nx or globin.atm.ny!=globin.obs.ny:
				print("--> Error in input.load_node_data()")
				print("    initial atmosphere does not have same dimensions")
				print("    as observations.")
				print(f"    -- atm = ({globin.atm.nx},{globin.atm.ny})")
				print(f"    -- obs = ({globin.obs.nx},{globin.obs.ny})")
				sys.exit()

			globin.atm.nodes[parameter] = data[0,0,0]
			globin.atm.values[parameter] = data[1]
			globin.atm.mask[parameter] = np.ones(len(globin.atm.nodes[parameter]))

			globin.parameter_scale[parameter] = np.ones((globin.atm.nx, globin.atm.ny, nnodes))
		except:
			pass

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

def set_keyword(text, key, value, fpath=None):
	lines = text.split("\n")
		
	line_num = None
	for num, line in enumerate(lines):
		line = line.replace(" ","")
		if len(line)>0:
			if line[0]!="#":
				if key in line:
					line_num = num
					break

	if line_num is not None:
		lines[num] = "  " + key + " = " + value
	else:
		line = "  " + key + " = " + value
		lines.insert(0, line)
		pass

	lines = [line + "\n" for line in lines]
	
	if fpath is not None:
		out = open(fpath, "w")
		out.writelines(lines)
		out.close()
		return "".join(lines)
	else:
		return "".join(lines)