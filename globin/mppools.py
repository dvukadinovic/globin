"""
Script for pool functions passed to multiprocessing module
for on thread distribution of workload.
"""

from .atmos import write_multi_atmosphere, extract_spectra_and_atmospheres
import time
import multiprocessing as mp
import subprocess as sp
from scipy.interpolate import splev, splrep
import numpy as np

from .utils import _set_keyword

import globin

def pool_write_atmosphere(args):
	atmos, idx, idy = args
	fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
	write_multi_atmosphere(atmos.data[idx,idy], fpath)

def pool_build_from_nodes(args):
	atmos, idx, idy, save_atmos = args

	for parameter in atmos.nodes:
		# K0, Kn by default; True for vmic, mag, gamma and chi
		K0, Kn = 0, 0

		x = atmos.nodes[parameter]
		y = atmos.values[parameter][idx,idy]

		if parameter=="temp":
			if len(x)>=2:
				K0 = (y[1]-y[0]) / (x[1]-x[0])
				# check if extrapolation at the top atmosphere point goes below the minimum
				# if does, change the slopte so that at top point we have Tmin (globin.limit_values["temp"][0])
				if globin.limit_values["temp"][0]>(y[0] + K0 * (atmos.logtau[0]-x[0])):
					K0 = (globin.limit_values["temp"][0] - y[0]) / (atmos.logtau[0] - x[0])
			# bottom node slope for extrapolation based on temperature gradient from FAL C model
			Kn = splev(x[-1], globin.temp_tck, der=1)
			# Kn = (y[-1] - y[-2]) / (x[-1] - x[-2])
			# print(Kn, Knp)
		# we do tan(gamma/2) interpolation (as in inversion)
		elif (parameter=="gamma"):
			# y = 2*np.arctan(y)
			if len(x)>=2:
				K0 = (y[1]-y[0]) / (x[1]-x[0])
				Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
		# we do tan(chi/4) interpolation (as in inversion)
		elif (parameter=="chi"):
			# y = 4*np.arctan(y)
			if len(x)>=2:
				K0 = (y[1]-y[0]) / (x[1]-x[0])
				Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
		elif (parameter=="vz") or (parameter=="vmic") or (parameter=="mag"):
			if len(x)>=2:
				K0 = (y[1]-y[0]) / (x[1]-x[0])
				Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
				# check if extrapolation at the top atmosphere point goes below the minimum
				# if does, change the slopte so that at top point we have parameter_min (globin.limit_values[parameter][0])
				if globin.limit_values[parameter][0]>(y[0] + K0 * (atmos.logtau[0]-x[0])):
					K0 = (globin.limit_values[parameter][0] - y[0]) / (atmos.logtau[0] - x[0])
				elif globin.limit_values[parameter][1]<(y[0] + K0 * (atmos.logtau[0]-x[0])):
					K0 = (globin.limit_values[parameter][1] - y[0]) / (atmos.logtau[0] - x[0])
				# similar for the bottom for maximum/min values
				if globin.limit_values[parameter][1]<(y[-1] + Kn * (atmos.logtau[-1]-x[-1])):
					Kn = (globin.limit_values[parameter][1] - y[-1]) / (atmos.logtau[-1] - x[-1])
				elif globin.limit_values[parameter][0]>(y[-1] + Kn * (atmos.logtau[-1]-x[-1])):
					Kn = (globin.limit_values[parameter][0] - y[-1]) / (atmos.logtau[-1] - x[-1])

		y_new = globin.bezier_spline(x, y, atmos.logtau, K0=K0, Kn=Kn, degree=globin.interp_degree)
		# if parameter=="gamma":
		# 	atmos.data[idx,idy,atmos.par_id[parameter],:] = np.tan(y_new/2)
		# elif parameter=="chi":
		# 	atmos.data[idx,idy,atmos.par_id[parameter],:] = np.tan(y_new/4)
		# 	atmos.values[parameter][idx,idy] = y
		# else:
		atmos.data[idx,idy,atmos.par_id[parameter],:] = y_new

	if globin.hydrostatic:
		atmos.makeHSE(idx, idy)

	if save_atmos:
		fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
		write_multi_atmosphere(atmos.data[idx,idy], fpath)

	return atmos

def pool_rf(args):
	start = time.time()

	atm_path, rh_spec_name = args

	#--- for each thread process create separate directory
	pid = mp.current_process()._identity[0]
	
	#--- copy *.input files
	sp.run(f"cp runs/{globin.wd}/*.input {globin.rh_path}/rhf1d/{globin.wd}_{pid}",
		shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)

	# re-read 'keyword.input' file for given pID
	globin.keyword_input = open(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.rh_input_name}", "r").read()

	keyword_path = f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.rh_input_name}"
	globin.keyword_input = _set_keyword(globin.keyword_input, "ATMOS_FILE", f"{globin.cwd}/{atm_path}")
	globin.keyword_input = _set_keyword(globin.keyword_input, "STOKES_INPUT", f"{globin.cwd}/{atm_path}.B", keyword_path)
	
	aux = atm_path.split("_")
	idx, idy = aux[-2], aux[-1]
	log_file = open(f"{globin.cwd}/runs/{globin.wd}/logs/log_{idx}_{idy}", "w")
	out = sp.run(f"cd {globin.rh_path}/rhf1d/{globin.wd}_{pid}; ../rf_ray -i {globin.rh_input_name}",
			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	log_file.writelines(str(out.stdout, "utf-8"))
	log_file.close()

	stdout = str(out.stdout,"utf-8").split("\n")

	if out.returncode!=0:
		print("*** RH error (pool_rf)")
		print(f"    Failed to compute RF for pixel ({idx},{idy}).\n")
		for line in stdout[-5:]:
			print("   ", line)
		return None

	rh_obj = globin.rh.Rhout(fdir=f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}", verbose=False)
	rh_obj.read_spectrum(rh_spec_name)
	rh_obj.read_ray()

	rf = np.loadtxt(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.rf_file_path}")
	
	dt = time.time() - start
	# print("Finished synthesis of '{:}' in {:4.2f} s".format(atm_path, dt))

	return {"rf" : rf, "wave" : rh_obj.wave, "idx" : idx, "idy" : idy}

def pool_synth(args):
	"""
	Function which executes what to be done on single thread in multicore
    mashine.

	Here we check if directory for given process exits ('pid_##') and copy all
	input files there for smooth run of RH code. We change the atmosphere path to
	path to pixel atmosphere for which we want to syntesise spectrum (and same
	for magnetic field).

	Also, if the directory has files from old runs, then too speed calculation we
	use old J.

	After the successful synthesis, we read and store spectrum in variable
	'spec'.

	Parameters:
	---------------
	atm_path : string
		Atmosphere path located in directory 'atmospheres'.
	rh_spec_name : string
		File name in which spectrum is written on a disk (read from keyword.input
        file).
    """
	start = time.time()

	atm_path, line_list_path = args

	# get process ID number
	pid = mp.current_process()._identity[0]
	set_old_J = True

	#--- copy *.input files from 'runs/globin.wd' directory
	sp.run(f"cp runs/{globin.wd}/*.input {globin.rh_path}/rhf1d/{globin.wd}_{pid}",
		shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)
	set_old_J = False

	# re-read 'keyword.input' file for given pID
	globin.keyword_input = open(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.rh_input_name}", "r").read()

	keyword_path = f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.rh_input_name}"
	globin.keyword_input = _set_keyword(globin.keyword_input, "ATMOS_FILE", f"{globin.cwd}/{atm_path}")
	globin.keyword_input = _set_keyword(globin.keyword_input, "STOKES_INPUT", f"{globin.cwd}/{atm_path}.B", keyword_path)

	# make kurucz.input file in rhf1d/globin.wd_pid and give the line list path
	out = open(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.kurucz_input_fname}", "w")
	out.write(f"{globin.cwd}/{line_list_path}\n")
	out.close()

	# make log file
	aux = atm_path.split("_")
	idx, idy = aux[-2], aux[-1]
	log_file = open(f"{globin.cwd}/runs/{globin.wd}/logs/log_{idx}_{idy}", "w")
	
	# run rhf1d executable
	out = sp.run(f"cd {globin.rh_path}/rhf1d/{globin.wd}_{pid}; ../rhf1d -i {globin.rh_input_name}",
			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	
	# store log file
	log_file.writelines(str(out.stdout, "utf-8"))
	log_file.close()

	stdout = str(out.stdout,"utf-8").split("\n")

	# check if rhf1d executed normaly
	if out.returncode!=0:
		print("*** RH error (pool_synth)")
		print(f"    Failed to synthesize spectra for pixel ({idx},{idy}).\n")
		for line in stdout[-5:]:
			print("   ", line)
		return None
	else:
		# if everything was fine, run solverray executable
		out = sp.run(f"cd {globin.rh_path}/rhf1d/{globin.wd}_{pid}; ../solveray",
			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
		# stdout = str(out.stdout,"utf-8").split("\n")
		# if out.returncode!=0:
		# 	print(f"Could not synthesize the spectrum for the ray! --> ({idx},{idy})\n")
		# 	return None

	# read output spectra and spectrum ray from RH
	rh_obj = globin.rh.Rhout(fdir=f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}", verbose=False)
	rh_obj.read_spectrum(globin.rh_spec_name)
	rh_obj.read_ray()
	# rh_obj.read_opacity()

	dt = time.time() - start
	if globin.mode==0:	
		print("Finished synthesis of '{:}' in {:4.2f} s".format(atm_path, dt))

	return {"rh_obj":rh_obj, "idx":int(idx), "idy":int(idy)}

def pool_spinor2multi(args):
	# start = time.time()

	do_HSE, atmos_data = args

	_, nz = atmos_data.shape
	data = np.zeros((14, nz))

	# log(tau)
	data[0] = atmos_data[0]
	# Temperature [K]
	data[1] = atmos_data[2]
	# Electron density [1/m3]
	data[2] = atmos_data[4]/10 / 1.380649e-23 / atmos_data[2] / 1e6
	# Vertical velocity [cm/s] --> [km/s]
	data[3] = atmos_data[9]/1e5
	# Microturbulent velocitu [cm/s] --> [km/s]
	data[4] = atmos_data[8]/1e5
	# Magnetic field strength [G]
	data[5] = atmos_data[7]
	# Inclination [rad]
	data[6] = atmos_data[-2]# * np.pi/180
	# Azimuth [rad]
	data[7] = atmos_data[-1]# * np.pi/180

	if do_HSE and globin.hydrostatic:
		press, pel, kappa, rho = globin.makeHSE(5000, data[0], data[1])
		
		# electron density [1/cm3]
		data[2] = pel/10/globin.K_BOLTZMAN/data[1]/1e6

		# Hydrogen populations [1/cm3]
		data[8:] = globin.atmos.distribute_hydrogen(data[1], press, pel)
	else:
		# Hydrogen population
		data[8:] = globin.atmos.distribute_hydrogen(atmos_data[2], atmos_data[3], atmos_data[4])

	# print(f"Finished converting atmosphere ({idx+1},{idy+1}) in {time.time() - start}s")

	return data