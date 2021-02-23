import numpy as np
import sys
import os
import copy
import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import globin

def invert(init, output_path="results", verbose=True):
	if globin.mode==0:
		print("Parameters for synthesis mode are read. We can not run inversion.\n  Change mode before running again.\n")
		return None, None
	elif globin.mode==1:
		atm, spec = invert_pxl_by_pxl(init, output_path, verbose)
		return atm, spec
	elif globin.mode==3:
		atm, spec = invert_global(init, output_path, verbose)
		return atm, spec
	else:
		print(f"Not supported mode {globin.mode} currently.")
		return None, None

def invert_pxl_by_pxl(init, output_path, verbose):
	"""
	As input we expect all data to be present :)

	Pixel-by-pixel inversion of atmospheric parameters.

	Parameters:
	---------------
	init : InputData
		InputData object in which we have everything stored.
	"""
	obs = init.obs
	atmos = init.atm

	if verbose:
		print("Initial parameters:")
		print(atmos.values)
		print()

	LM_parameter = np.ones((obs.nx, obs.ny), dtype=np.float64) * init.marq_lambda
	# flags those pixels whose chi2 converged:
	#   1 -> we do inversion
	#   0 -> we converged
	# with flag we multiply the proposed steps, in that case for those pixles
	# in which we converged we will not change parameters, but, the calculations
	# will be done, as well as RFs... Find smarter way around it.
	stop_flag = np.ones((obs.nx, obs.ny), dtype=np.float64)

	Nw = len(init.wavelength)
	# this is number of local parameters only (we are doing pxl-by-pxl)
	Npar = atmos.n_local_pars

	# indices of diagonal elements of Hessian matrix
	x = np.arange(atmos.nx)
	y = np.arange(atmos.ny)
	p = np.arange(Npar)
	X,Y,P = np.meshgrid(x,y,p, indexing="ij")

	if Npar==0:
		sys.exit("There is no parameters to fit.\n   We exit.\n")

	# indices for wavelengths min/max for which we are fiting; based on input
	ind_min = np.argmin(abs(obs.wavelength - init.wavelength[0]))
	ind_max = np.argmin(abs(obs.wavelength - init.wavelength[-1]))+1

	if init.noise!=0:
		StokesI_cont = obs.spec[:,:,ind_min,0]
		noise_lvl = init.noise * StokesI_cont
		# noise_wavelength = (nx, ny, nw)
		noise_wavelength = np.sqrt(obs.spec[:,:,ind_min:ind_max,0].T / StokesI_cont.T).T
		# noise = (nx, ny, nw)
		noise = np.einsum("...,...w", noise_lvl, noise_wavelength)
		# noise_stokes_scale = (nx, ny, nw, 4)
		noise_stokes_scale = np.repeat(noise_wavelength[..., np.newaxis], 4, axis=3)
		# noise_stokes = (nx, ny, nw, 4)
		noise_stokes = np.repeat(noise[..., np.newaxis], 4, axis=3)
		# noies_scale_rf = (nx, ny, npar, nw, 4)
		noise_scale_rf = np.repeat(noise_stokes_scale[:,:, np.newaxis ,:,:], Npar, axis=2)
	else:
		noise_scale_rf = np.ones((obs.nx, obs.ny, Npar, Nw, 4), dtype=np.float64)
		noise_stokes = np.ones((obs.nx, obs.ny, Nw, 4), dtype=np.float64)
		noise_stokes_scale = np.ones((obs.nx, obs.ny, Nw, 4), dtype=np.float64)

	chi2 = np.zeros((atmos.nx, atmos.ny, init.max_iter))
	N_search_for_lambda = 5
	dof = np.count_nonzero(init.weights) * Nw - Npar

	start = time.time()

	updated_pars = True
	itter = np.zeros((atmos.nx, atmos.ny), dtype=np.int)
	# for i_ in range(init.max_iter):
	# we iterate until one of the pixels reach maximum numbre of iterations
	# other pixels will be blocked at max itteration earlier than or 
	# will stop due to convergence criterium
	while np.min(itter) <= init.max_iter:
		#--- if we updated parameters, recaluclate RF and referent spectra
		if updated_pars:
			if verbose:
				print("Iteration (min): {:2}\n".format(np.min(itter)+1))
			
			# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
			#               spec.shape = (nx, ny, Nw, 5)
			rf, spec = globin.compute_rfs(init, atmos, itter[0,0])

			# rf = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))
			# diff = np.zeros((atmos.nx, atmos.ny, Nw, 4))
			# for idx in range(atmos.nx):
			# 	for idy in range(atmos.ny):
			# 		for pID in range(Npar):
			# 			for sID in range(4):
			# 				rf[idx,idy,pID,:,sID] = np.ones(Nw)*(1+sID) + 10*pID + 100*idy + 1000*idx
			# 		for sID in range(4):
			# 			diff[idx,idy,:,sID] = np.ones(Nw)*(1+sID) + 10*idy + 100*idx
			
			#--- scale RFs with weights and noise scale
			rf *= init.weights
			rf /= noise_scale_rf

			diff = obs.spec - spec.spec
			diff *= init.weights
			chi2_old = np.sum(diff**2 / noise_stokes**2 * init.wavs_weight**2, axis=(2,3)) / dof
			diff /= noise_stokes_scale

			# globin.plot_spectra(obs, 0, 0)
			# globin.plot_spectra(spec, 0, 0)
			# plt.show()

			"""
			Gymnastics with indices for solving LM equations for
			next step parameters.
			"""
			J = rf.reshape(atmos.nx, atmos.ny, Npar, 4*Nw, order="F")
			# J = (nx, ny, 4*nw, npar)
			J = np.moveaxis(J, 2, 3)
			# JT = (nx, ny, npar, 4*nw)
			JT = np.einsum("ijlk", J)
			# JTJ = (nx, ny, npar, npar)
			JTJ = np.einsum("...ij,...jk", JT, J)
			# get diagonal elements from hessian matrix
			diagonal_elements = np.einsum("...kk->...k", JTJ)
			# reshaped array of differences between computed and observed spectra
			# flatted_diff = (nx, ny, 4*Nw)
			flatted_diff = diff.reshape(atmos.nx, atmos.ny, 4*Nw, order="F")

			# This was tested with arrays filled with hand and 
			# checked if the array manipulations return what we expect
			# and it does

		# hessian = (nx, ny, npar, npar)
		H = copy.deepcopy(JTJ)
		# multiply with LM parameter
		H[X,Y,P,P] = np.einsum("...i,...", diagonal_elements, 1+LM_parameter)
		# delta = (nx, ny, npar)
		delta = np.einsum("...pw,...w", JT, flatted_diff)
		# proposed_steps = (nx, ny, npar)
		proposed_steps = np.linalg.solve(H, delta)

		old_parameters = copy.deepcopy(atmos.values)
		atmos.update_parameters(proposed_steps, stop_flag)
		atmos.check_parameter_bounds()

		atmos.build_from_nodes(init.ref_atm)
		corrected_spec,_,_ = globin.compute_spectra(atmos, init.rh_spec_name, init.wavelength)
		corrected_spec.broaden_spectra(atmos.vmac)

		new_diff = obs.spec - corrected_spec.spec
		new_diff *= init.weights
		chi2_new = np.sum(new_diff**2 / noise_stokes**2 * init.wavs_weight**2, axis=(2,3)) / dof

		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				if stop_flag[idx,idy]==1:
					if chi2_new[idx,idy] > chi2_old[idx,idy]:
						LM_parameter[idx,idy] *= 10
						for parID in old_parameters:
							atmos.values[parID][idx,idy] = old_parameters[parID][idx,idy]
						updated_pars = False
					else:
						chi2[idx,idy,itter[idx,idy]] = chi2_new[idx,idy]
						LM_parameter[idx,idy] /= 10
						itter[idx,idy] += 1
						updated_pars = True

		# if Marquardt parameter is to large, we break
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				if LM_parameter[idx,idy]<=1e-5:
					LM_parameter[idx,idy] = 1e-5
				if LM_parameter[idx,idy]>=1e8:
					stop_flag[idx,idy] = 0
					print("Large LM parameter. We break.")

		if updated_pars and verbose:
			print(atmos.values)
			print(LM_parameter)
			# print("{:4.3e}".format(chi2[0,0,-1]))
			print("\n--------------------------------------------------\n")

		# we check if chi2 has converged for each pixel
		# if yes, we set stop_flag to 1 (True)
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				if stop_flag[idx,idy]==1:
					it_no = itter[idx,idy]
					if it_no>=2:
						# need to get -2 and -1 because I already rised itter by 1 
						# when chi2 list was updated.
						relative_change = abs(chi2[idx,idy,it_no-1]/chi2[idx,idy,it_no-2] - 1)
						if chi2[itter-1]<1e-32:
							print("chi2 is way low!\n")
							break_flag = True
						elif relative_change<init.chi2_tolerance:
							print(f"--> [{idx},{idy}] : chi2 relative change is smaller than given value.")
							stop_flag[idx,idy] = 0
						elif chi2[idx,idy,it_no-1] < 1 and init.noise!=0:
							# print(chi2[idx,idy,it_no-1])
							print(f"--> [{idx},{idy}] : chi2 smaller than 1")
							stop_flag[idx,idy] = 0
					# if given pixel iteration number has reached the maximum number of iterations
					# we stop the convergence for given pixel
					if it_no==init.max_iter-1:
						stop_flag[idx,idy] = 0
						print("Maximum number of iterations reached. We break.")
		
		# if all pixels have converged, we stop inversion
		if np.sum(stop_flag)==0:
			break

	atmos.build_from_nodes(init.ref_atm)
	inverted_spectra,_,_ = globin.compute_spectra(atmos, init.rh_spec_name, init.wavelength, )
	inverted_spectra.broaden_spectra(atmos.vmac)
	
	if output_path is not None:
		# check if there is result folder; if not, make it
		if not os.path.exists(f"{output_path}"):
			os.mkdir(f"{output_path}")

		globin.spectrum_path = f"{output_path}/inverted_spectra.fits"

		atmos.save_atmosphere(f"{output_path}/inverted_atmos.fits")
		inverted_spectra.save(globin.spectrum_path, init.wavelength)
		globin.save_chi2(chi2, f"{output_path}/chi2.fits")
		
		end = time.time() - start
		print("\nFinished in: {0}\n".format(end))

		#--- inverted params comparison with expected values
		out_file = open("{:s}/output.log".format(output_path), "w")

		out_file.write("Run time: {:10.1f}\n\n".format(end))

		out_file.write("\n\n     #===--- globin input file ---===#\n\n")
		out_file.write(init.globin_input)
		out_file.write("\n\n     #===--- RH input file ---===#\n\n")
		out_file.write(init.rh_input)

		out_file.close()

	return atmos, inverted_spectra

def invert_global(init, output_path, verbose):
	"""
	As input we expect all data to be present :)

	Glonal inversion of atmospheric and atomic parameters.

	Parameters:
	---------------
	init : InputData
		InputData object in which we have everything stored.
	"""
	obs = init.obs
	atmos = init.atm

	if verbose:
		print("Initial parameters:")
		print(atmos.values)
		print(atmos.global_pars)
		print()

	Nw = len(init.wavelength)
	Npar = atmos.n_local_pars + atmos.n_global_pars

	if Npar==0:
		sys.exit("There are no parameters to fit.\n   We exit.\n")

	# indices for wavelengths min/max for which we are fiting; based on input
	ind_min = np.argmin(abs(obs.wavelength - init.wavelength[0]))
	ind_max = np.argmin(abs(obs.wavelength - init.wavelength[-1]))+1

	if init.noise!=0:
		StokesI_cont = obs.spec[:,:,ind_min,0]
		noise_lvl = init.noise * StokesI_cont
		# noise_wavelength = (nx, ny, nw)
		noise_wavelength = np.sqrt(obs.spec[:,:,ind_min:ind_max,0].T / StokesI_cont.T).T
		# noise = (nx, ny, nw)
		noise = np.einsum("...,...w", noise_lvl, noise_wavelength)
		# noise_stokes_scale = (nx, ny, nw, 4)
		noise_stokes_scale = np.repeat(noise_wavelength[..., np.newaxis], 4, axis=3)
		# noise_stokes = (nx, ny, nw, 4)
		noise_stokes = np.repeat(noise[..., np.newaxis], 4, axis=3)
		# noies_scale_rf = (nx, ny, npar, nw, 4)
		noise_scale_rf = np.repeat(noise_stokes_scale[:,:, np.newaxis ,:,:], Npar, axis=2)
	else:
		noise_scale_rf = np.ones((obs.nx, obs.ny, Npar, Nw, 4), dtype=np.float64)
		noise_stokes = np.ones((obs.nx, obs.ny, Nw, 4), dtype=np.float64)
		noise_stokes_scale = np.ones((obs.nx, obs.ny, Nw, 4), dtype=np.float64)

	chi2 = np.zeros(init.max_iter, dtype=np.float64)
	LM_parameter = init.marq_lambda
	dof = np.count_nonzero(init.weights) * Nw - Npar

	start = time.time()

	break_flag = False
	updated_parameters = True
	num_failed = 0

	itter = 0
	while itter<init.max_iter:
		#--- if we updated parameters, recaluclate RF and referent spectra
		if updated_parameters:
			if verbose:
				print("Iteration: {:2}\n".format(itter+1))
			
			# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
			#               spec.shape = (nx, ny, Nw, 5)
			rf, spec = globin.compute_rfs(init, atmos)

			# globin.plot_spectra(obs, 0, 0)
			# globin.plot_spectra(spec, 0, 0)
			# plt.show()

			# sys.exit()

			# rf = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))
			# diff = np.zeros((atmos.nx, atmos.ny, Nw, 4))
			# for idx in range(atmos.nx):
			# 	for idy in range(atmos.ny):
			# 		for pID in range(Npar):
			# 			for sID in range(4):
			# 				rf[idx,idy,pID,:,sID] = np.ones(Nw)*(1+sID) + 10*pID + 100*idy + 1000*idx
			# 		for sID in range(4):
			# 			diff[idx,idy,:,sID] = np.ones(Nw)*(1+sID) + 10*idy + 100*idx

			# plt.imshow(rf[0,1,-1,:,:], aspect="auto")
			# plt.show()
			# sys.exit()
			
			# scale RFs with weights and noise scale
			rf *= init.weights
			rf /= noise_scale_rf

			# calculate difference between observation and synthesis
			diff = obs.spec - spec.spec
			diff *= init.weights

			# calculate chi2
			chi2_old = np.sum(diff**2 / noise_stokes**2 * init.wavs_weight**2) / dof
			diff /= noise_stokes_scale

			# make Jacobian matrix and fill with RF values
			aux = rf.reshape(atmos.nx, atmos.ny, Npar, 4*Nw, order="F")
			
			J = np.zeros((4*Nw*(atmos.nx*atmos.ny), atmos.n_local_pars*(atmos.nx*atmos.ny) + atmos.n_global_pars))
			flatted_diff = np.zeros(atmos.nx*atmos.ny*Nw*4)

			l = 4*Nw
			n_atmosphere = 0
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					low = n_atmosphere*l
					up = low + l 
					ll = n_atmosphere*atmos.n_local_pars
					uu = ll + atmos.n_local_pars
					J[low:up,ll:uu] = aux[idx,idy,:atmos.n_local_pars].T
					flatted_diff[low:up] = diff[idx,idy].flatten(order="F")
					n_atmosphere += 1

			n_atmosphere = 0
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					low = n_atmosphere*l
					up = low+l
					for gID in range(atmos.n_global_pars):
						J[low:up,uu+gID] = aux[idx,idy,atmos.n_local_pars+gID].T
					n_atmosphere += 1

			JT = J.T
			JTJ = np.dot(JT,J)
			delta = np.dot(JT, flatted_diff)

			# This was heavily(?) tested with simple filled 'rf' and 'diff' ndarrays.
			# It produces expected results. Checked with (1,1), (1,2) and (2,2) FoV sizes.

		H = copy.deepcopy(JTJ)
		diagonal_elements = np.diag(JTJ) * (1 + LM_parameter)
		np.fill_diagonal(H, diagonal_elements)
		proposed_steps = np.linalg.solve(H, delta)

		# print(np.linalg.eigvals(H))

		# plt.imshow(np.linalg.inv(H), aspect="auto")
		# plt.colorbar()
		# plt.show()

		# sys.exit()
		
		old_parameters = copy.deepcopy(atmos.values)
		old_global_pars = copy.deepcopy(atmos.global_pars)
		atmos.update_parameters(proposed_steps)
		atmos.check_parameter_bounds()
		if ("loggf" in atmos.global_pars) or ("dlam" in atmos.global_pars):
			init.write_line_parameters(atmos.global_pars["loggf"], atmos.line_no["loggf"],
									   atmos.global_pars["dlam"], atmos.line_no["dlam"])

		atmos.build_from_nodes(init.ref_atm)
		corrected_spec,_,_ = globin.compute_spectra(atmos, init.rh_spec_name, init.wavelength)
		corrected_spec.broaden_spectra(atmos.vmac)

		new_diff = obs.spec - corrected_spec.spec
		new_diff *= init.weights
		chi2_new = np.sum(new_diff**2 / noise_stokes**2 * init.wavs_weight**2) / dof

		if chi2_new > chi2_old:
			LM_parameter *= 10
			atmos.values = old_parameters
			atmos.global_pars = old_global_pars
			updated_parameters = False
			num_failed += 1
		else:
			chi2[itter] = chi2_new
			LM_parameter /= 10
			updated_parameters = True
			itter += 1
			num_failed = 0

		# if Marquardt parameter is to large, we break
		if LM_parameter<=1e-5:
			LM_parameter = 1e-5
		if LM_parameter>=1e8:
			print("Upper limit in LM_parameter. We break\n")
			break_flag = True

		if updated_parameters and verbose:
			print(atmos.values)
			print(atmos.global_pars)
			print(LM_parameter)
			# print(np.log10(chi2_new))
			print("\n--------------------------------------------------\n")

		# we check if chi2 has converged for each pixel
		# if yes, we set break_flag to True
		# we do not check for chi2 convergence until 3rd iteration
		if (itter)>=3:
			# need to get -2 and -1 because I already rised itter by 1 
			# when chi2 list was updated.
			relative_change = abs(chi2[itter-1]/chi2[itter-2] - 1)
			if chi2[itter-1]<1e-32:
				print("chi2 is way low!\n")
				break_flag = True
			elif relative_change<init.chi2_tolerance:
				print("chi2 relative change is smaller than given value.\n")
				break_flag = True
			elif chi2[itter-1] < 1 and init.noise!=0:
				print("chi2 smaller than 1\n")
				break_flag = True

		# if all pixels have converged, we stop inversion
		if break_flag:
			break

		if (num_failed==10 and itter>=3):
			print("Failed 10 times to fix the LM parameter. We break.\n")
			break

	atmos.build_from_nodes(init.ref_atm)
	
	inverted_spectra,_,_ = globin.compute_spectra(atmos, init.rh_spec_name, init.wavelength)
	inverted_spectra.broaden_spectra(atmos.vmac)

	if output_path is not None:
		# check if there is result folder; if not, make it
		if not os.path.exists(f"{output_path}"):
			os.mkdir(f"{output_path}")

		globin.spectrum_path = f"{output_path}/inverted_spectra.fits"

		atmos.save_atmosphere(f"{output_path}/inverted_atmos.fits")
		inverted_spectra.save(globin.spectrum_path, init.wavelength)
		globin.save_chi2(chi2, f"{output_path}/chi2.fits")
	
		end = time.time() - start
		print("Finished in: {0}\n".format(end))

		#--- make a log file (copy input files + inverted global parameters)
		out_file = open("{:s}/output.log".format(output_path), "w")

		out_file.write("Run time: {:10.1f}\n\n".format(end))

		out_file.write("\n\n     #===--- Global parameters ---===#\n\n")

		for par in atmos.global_pars:
			if par=="vmac":
				for i_ in range(len(atmos.global_pars[par])):
					out_file.write("{:s}    {: 4.3f}\n".format(par, atmos.global_pars[par][i_]))
			else:
				for i_ in range(len(atmos.global_pars[par])):
					out_file.write("{:s}    {: 4d}    {: 5.4f}\n".format(par, atmos.line_no[par][i_]+1, atmos.global_pars[par][i_]))

		out_file.write("\n\n     #===--- globin input file ---===#\n\n")
		out_file.write(init.globin_input)
		out_file.write("\n\n     #===--- RH input file ---===#\n\n")
		out_file.write(init.rh_input)

		out_file.close()

	return atmos, inverted_spectra

