import numpy as np
import sys
import os
import copy
import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import globin

def invert(save_output=True, verbose=True):
	if globin.mode==0:
		print("Parameters for synthesis mode are read. We can not run inversion.\n  Change mode before running again.\n")
		return None, None
	elif globin.mode>=1:
		for cycle in range(globin.ncycle):
			# double number of iterations in last cycle
			if cycle==globin.ncycle-1 and globin.ncycle!=1:
				globin.max_iter *= 2

			if globin.mode==1 or globin.mode==2:
				atm, spec = invert_pxl_by_pxl(save_output, verbose)
			elif globin.mode==3:
				atm, spec = invert_global(save_output, verbose)
			elif globin.mode==4:
				atm, spec = invert_mcmc(save_output, verbose)
			else:
				print(f"Not supported mode {globin.mode} currently.")
				return None, None
			
			# in last cycle we do not smooth atmospheric parameters
			if (cycle+1)<globin.ncycle:
				globin.atm.smooth_parameters()

		# globin.remove_dirs()

		return atm, spec

def invert_pxl_by_pxl(save_output, verbose):
	"""
	As input we expect all data to be present :)

	Pixel-by-pixel inversion of atmospheric parameters.

	Parameters:
	---------------
	init : InputData
		InputData object in which we have everything stored.
	"""
	obs = globin.obs
	atmos = globin.atm

	if verbose:
		print("Initial parameters:")
		print(atmos.values)
		if globin.mode==2:
			print(atmos.global_pars)
		print()

	LM_parameter = np.ones((obs.nx, obs.ny), dtype=np.float64) * globin.marq_lambda
	# flags those pixels whose chi2 converged:
	#   1 --> we do inversion
	#   0 --> we converged
	# with flag we multiply the proposed steps, in that case for those pixles
	# in which we converged we will not change parameters, but, the calculations
	# will be done, as well as RFs... Find smarter way around it.
	stop_flag = np.ones((obs.nx, obs.ny), dtype=np.float64)

	Nw = len(globin.wavelength)
	# this is number of local parameters only (we are doing pxl-by-pxl)
	if globin.mode==1:	
		Npar = atmos.n_local_pars
	elif globin.mode==2:
		Npar = atmos.n_local_pars + atmos.n_global_pars
	
	if Npar==0:
		print("There is no parameters to fit.\n   We exit.\n")
		globin.remove_dirs()
		sys.exit()

	# indices of diagonal elements of Hessian matrix
	x = np.arange(atmos.nx)
	y = np.arange(atmos.ny)
	p = np.arange(Npar)
	X,Y,P = np.meshgrid(x,y,p, indexing="ij")

	# indices for wavelengths min/max for which we are fiting; based on input
	ind_min = np.argmin(abs(obs.wavelength - globin.wavelength[0]))
	ind_max = np.argmin(abs(obs.wavelength - globin.wavelength[-1]))+1

	if globin.noise!=0:
		StokesI_cont = obs.spec[:,:,ind_min,0]
		noise_lvl = globin.noise * StokesI_cont
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

	# weights on Stokes vector based on dI over dlam (from observations)
	from scipy.interpolate import splev, splrep	
	weights = np.empty((obs.nx, obs.ny, Nw))
	for idx in range(obs.nx):
		for idy in range(obs.ny):
			tck = splrep(obs.wavelength, obs.spec[idx,idy,:,0])
			dIdlam = splev(obs.wavelength, tck, der=1)

			norm = np.sum(np.abs(dIdlam))

			weights[idx,idy,:] = (np.abs(dIdlam) / norm)[ind_min:ind_max]

	weights = np.repeat(weights[:,:,:,np.newaxis], 4, axis=3)

	# weights on Stokes vector based on observed Stokes I
	# aux = 1/obs.spec[...,0]
	# weights = np.repeat(aux[..., np.newaxis], 4, axis=3)
	# norm = np.sum(weights, axis=2)
	# weights = weights / np.repeat(norm[:,:, np.newaxis, :], Nw, axis=2)
	weights = 1

	chi2 = np.zeros((atmos.nx, atmos.ny, globin.max_iter))
	dof = np.count_nonzero(globin.weights) * Nw - Npar

	start = time.time()

	itter = np.zeros((atmos.nx, atmos.ny), dtype=np.int)
	# we iterate until one of the pixels reach maximum numbre of iterations
	# other pixels will be blocked at max itteration earlier than or 
	# will stop due to convergence criterium
	full_rf, old_atmos_parameters = None, None

	rf = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))
	spec = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))

	original_atm_name_list = copy.deepcopy(atmos.atm_name_list)
	original_line_lists_path = copy.deepcopy(atmos.line_lists_path)
	atm_name_list = copy.deepcopy(atmos.atm_name_list)
	line_lists_path = copy.deepcopy(atmos.line_lists_path)
	old_inds = []

	while np.min(itter) <= globin.max_iter:
		#--- if we updated parameters, recaluclate RF and referent spectra
		# if len(old_inds)!=atmos.nx*atmos.ny:
		if len(atm_name_list)>0:
			if verbose:
				print("Iteration (min): {:2}\n".format(np.min(itter)+1))

			atmos.atm_name_list = copy.deepcopy(atm_name_list)
			if globin.mode==2:
				atmos.line_lists_path = copy.deepcopy(line_lists_path)
			
			# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
			#             spec.shape = (nx, ny, Nw, 5)
			old_rf = copy.deepcopy(rf)
			old_spec = copy.deepcopy(spec)
			if globin.rf_type=="snapi":	
				rf, spec, full_rf = globin.compute_rfs(atmos, full_rf, old_atmos_parameters)
			elif globin.rf_type=="node":
				rf, spec, _ = globin.compute_rfs(atmos)

			# copy old RF into new for new itteration inversion
			if len(old_inds)>0:
				for ind in old_inds:
					idx, idy = ind
					rf[idx,idy] = old_rf[idx,idy]
					spec.spec[idx,idy] = old_spec.spec[idx,idy]

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
			_rf = rf*globin.weights
			_rf /= noise_scale_rf

			diff = obs.spec - spec.spec
			diff *= globin.weights
			chi2_old = np.sum(diff**2 / noise_stokes**2 * globin.wavs_weight**2 * weights**2, axis=(2,3)) / dof
			diff /= noise_stokes_scale

			"""
			Gymnastics with indices for solving LM equations for
			next step parameters.
			"""
			J = _rf.reshape(atmos.nx, atmos.ny, Npar, 4*Nw, order="F")
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
			# and it does!
			updated_pars = np.ones((atmos.nx, atmos.ny))

		atmos.atm_name_list = copy.deepcopy(original_atm_name_list)
		atm_name_list = copy.deepcopy(original_atm_name_list)
		if globin.mode==2:
			atmos.line_lists_path = copy.deepcopy(original_line_lists_path)
			line_lists_path = copy.deepcopy(original_line_lists_path)
		old_inds = []

		# hessian = (nx, ny, npar, npar)
		H = copy.deepcopy(JTJ)
		# multiply with LM parameter
		H[X,Y,P,P] = np.einsum("...i,...", diagonal_elements, 1+LM_parameter)
		# delta = (nx, ny, npar)
		delta = np.einsum("...pw,...w", JT, flatted_diff)
		# proposed_steps = (nx, ny, npar)
		proposed_steps = np.linalg.solve(H, delta)

		old_atmos_parameters = copy.deepcopy(atmos.values)
		if globin.mode==2:
			old_atomic_parameters = copy.deepcopy(atmos.global_pars)
		atmos.update_parameters(proposed_steps, stop_flag)
		atmos.check_parameter_bounds()

		if ("loggf" in atmos.global_pars) or ("dlam" in atmos.global_pars):
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
					globin.write_line_parameters(fpath,
											   atmos.global_pars["loggf"][idx,idy], atmos.line_no["loggf"],
											   atmos.global_pars["dlam"][idx,idy], atmos.line_no["dlam"])

		atmos.build_from_nodes()
		corrected_spec,_,_ = globin.compute_spectra(atmos)
		corrected_spec.broaden_spectra(atmos.vmac)

		new_diff = obs.spec - corrected_spec.spec
		new_diff *= globin.weights
		chi2_new = np.sum(new_diff**2 / noise_stokes**2 * globin.wavs_weight**2 * weights**2, axis=(2,3)) / dof

		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				if stop_flag[idx,idy]==1:
					if chi2_new[idx,idy] > chi2_old[idx,idy]:
						LM_parameter[idx,idy] *= 10
						for parameter in old_atmos_parameters:
							atmos.values[parameter][idx,idy] = copy.deepcopy(old_atmos_parameters[parameter][idx,idy])
						if globin.mode==2:
							for parameter in old_atomic_parameters:
								atmos.global_pars[parameter][idx,idy] = copy.deepcopy(old_atomic_parameters[parameter][idx,idy])
							fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
							line_lists_path.remove(fpath)
						fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
						atm_name_list.remove(fpath)
						old_inds.append((idx,idy))
						updated_pars[idx,idy] = 0
					else:
						chi2[idx,idy,itter[idx,idy]] = chi2_new[idx,idy]
						LM_parameter[idx,idy] /= 10
						itter[idx,idy] += 1
						updated_pars[idx,idy] = 1
				else:
					for parameter in old_atmos_parameters:
						atmos.values[parameter][idx,idy] = copy.deepcopy(old_atmos_parameters[parameter][idx,idy])
					if globin.mode==2:
						for parameter in old_atomic_parameters:
							atmos.global_pars[parameter][idx,idy] = copy.deepcopy(old_atomic_parameters[parameter][idx,idy])

		# we write down those atomic parameters which are not updated (rewerting back to old ones)
		if ("loggf" in atmos.global_pars) or ("dlam" in atmos.global_pars):
			for ind in old_inds:
				idx, idy = ind
				fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
				globin.write_line_parameters(fpath,
										   atmos.global_pars["loggf"][idx,idy], atmos.line_no["loggf"],
										   atmos.global_pars["dlam"][idx,idy], atmos.line_no["dlam"])

		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				if stop_flag[idx,idy]==1:
					if LM_parameter[idx,idy]<=1e-5:
						LM_parameter[idx,idy] = 1e-5
					# if Marquardt parameter is to large, we break
					if LM_parameter[idx,idy]>=1e8:
						stop_flag[idx,idy] = 0
						itter[idx,idy] = globin.max_iter
						original_atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
						if globin.mode==2:	
							original_line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
						print(f"[{idx},{idy}] --> Large LM parameter. We break.")

		if verbose:
			print(atmos.values)
			if globin.mode==2:
				print(atmos.global_pars)
			print(LM_parameter)
			print(old_inds)

		# we check if chi2 has converged for each pixel
		# if yes, we set stop_flag to 0 (True)
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				if stop_flag[idx,idy]==1 and updated_pars[idx,idy]==1:
					it_no = itter[idx,idy]
					if it_no>=2:
						# need to get -2 and -1 because I already rised itter by 1 
						# when chi2 list was updated.
						relative_change = abs(chi2[idx,idy,it_no-1]/chi2[idx,idy,it_no-2] - 1)
						if chi2[idx,idy,it_no-1]<1e-32:
							print(f"--> [{idx},{idy}] : chi2 is way low!\n")
							stop_flag[idx,idy] = 0
							itter[idx,idy] = globin.max_iter
							original_atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
							if globin.mode==2:	
								original_line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
						elif relative_change<globin.chi2_tolerance:
							print(f"--> [{idx},{idy}] : chi2 relative change is smaller than given value.\n")
							stop_flag[idx,idy] = 0
							itter[idx,idy] = globin.max_iter
							original_atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
							if globin.mode==2:	
								original_line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
						elif chi2[idx,idy,it_no-1] < 1 and globin.noise!=0:
							print(f"--> [{idx},{idy}] : chi2 smaller than 1\n")
							stop_flag[idx,idy] = 0
							itter[idx,idy] = globin.max_iter
							original_atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
							if globin.mode==2:
								original_line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
				# if given pixel iteration number has reached the maximum number of iterations
				# we stop the convergence for given pixel
				if itter[idx,idy]==globin.max_iter-1:
					stop_flag[idx,idy] = 0
					print(f"--> [{idx},{idy}] : Maximum number of iterations reached. We break.\n")

		if verbose:
			print("\n--------------------------------------------------\n")

		# if all pixels have converged, we stop inversion
		if np.sum(stop_flag)==0:
			break
	
	# return all original paths for final output
	atmos.atm_name_list = []
	atmos.line_lists_path = []
	for idx in range(atmos.nx):
		for idy in range(atmos.ny):
			fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
			atmos.atm_name_list.append(fpath)
			if globin.mode==2:
				fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
				atmos.line_lists_path.append(fpath)
	if globin.mode==1:
		fpath = f"runs/{globin.wd}/{globin.linelist_name}"
		atmos.line_lists_path.append(fpath)

	atmos.build_from_nodes(False)
	inverted_spectra, atm, _ = globin.compute_spectra(atmos)
	inverted_spectra.broaden_spectra(atmos.vmac)

	try:
		atmos.compute_errors(JTJ, chi2_old)
	except:
		print("Failed to compute parameter errors\n")
	
	if save_output is not None:
		output_path = f"runs/{globin.wd}"

		atm.save_atmosphere(f"{output_path}/inverted_RH_atmos.fits")
		
		atmos.save_atmosphere(f"{output_path}/inverted_atmos.fits")
		if atmos.n_global_pars>0:
			atmos.save_atomic_parameters(f"{output_path}/inverted_atoms.fits", kwargs={"RLK_LIST" : (f"{globin.cwd}/{atmos.line_lists_path[0].split('/')[-1]}", "reference line list")})
		inverted_spectra.save(f"{output_path}/inverted_spectra.fits", globin.wavelength)
		globin.save_chi2(chi2, f"{output_path}/chi2.fits")
		
		end = time.time() - start
		print("\nFinished in: {0}\n".format(end))

		out_file = open("{:s}/output.log".format(output_path), "w")
		
		out_file.write("Run time: {:10.1f}\n\n".format(end))
		out_file.write("\n\n     #===--- globin input file ---===#\n\n")
		out_file.write(globin.parameters_input)
		out_file.write("\n\n     #===--- RH input file ---===#\n\n")
		out_file.write(globin.keyword_input)

		out_file.close()

	return atmos, inverted_spectra

def invert_global(save_output, verbose):
	"""
	As input we expect all data to be present :)

	Glonal inversion of atmospheric and atomic parameters.

	Parameters:
	---------------
	init : InputData
		InputData object in which we have everything stored.
	"""
	from scipy.interpolate import interp1d

	obs = globin.obs
	atmos = globin.atm

	# new_spec = np.zeros((obs.nx, obs.ny, len(globin.wavelength), 4))
	# for idx in range(obs.nx):
	# 	for idy in range(obs.ny):
	# 		for ids in range(4):
	# 			new_spec[idx,idy,:,ids] = interp1d(obs.wavelength, obs.spec[idx,idy,:,ids], fill_value="extrapolate")(globin.wavelength)

	# obs.spec = new_spec
	# obs.wavelength = globin.wavelength
	# obs.wave = globin.wavelength

	if verbose:
		print("Initial parameters:")
		print(atmos.values)
		print(atmos.global_pars)
		print()

	Nw = len(globin.wavelength)
	Npar = atmos.n_local_pars + atmos.n_global_pars

	if Npar==0:
		print("There is no parameters to fit.\n   We exit.\n")
		globin.remove_dirs()
		sys.exit()

	# indices for wavelengths min/max for which we are fiting; based on input
	ind_min = np.argmin(abs(obs.wavelength - globin.wavelength[0]))
	ind_max = np.argmin(abs(obs.wavelength - globin.wavelength[-1]))+1

	if globin.noise!=0:
		StokesI_cont = obs.spec[:,:,ind_min,0]
		noise_lvl = globin.noise * StokesI_cont
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

	# weights on Stokes vector based on observed Stokes I
	aux = 1/obs.spec[...,0]
	weights = np.repeat(aux[..., np.newaxis], 4, axis=3)
	norm = np.sum(weights, axis=2)
	weights = weights / np.repeat(norm[:,:, np.newaxis, :], Nw, axis=2)
	weights = 1

	chi2 = np.zeros((atmos.nx, atmos.ny, globin.max_iter), dtype=np.float64)
	LM_parameter = globin.marq_lambda
	dof = np.count_nonzero(globin.weights) * Nw - Npar

	start = time.time()

	break_flag = False
	updated_parameters = True
	num_failed = 0

	itter = 0
	full_rf, old_local_parameters = None, None
	while itter<globin.max_iter:
		#--- if we updated parameters, recaluclate RF and referent spectra
		if updated_parameters:
			if verbose:
				print("Iteration: {:2}\n".format(itter+1))
			
			# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
			#               spec.shape = (nx, ny, Nw, 5)
			rf, spec, full_rf = globin.compute_rfs(atmos, full_rf, old_local_parameters)

			# globin.plot_spectra(obs, inv=spec, idx=0, idy=0)
			# plt.show()

			# rf = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))
			# diff = np.zeros((atmos.nx, atmos.ny, Nw, 4))
			# for idx in range(atmos.nx):
			# 	for idy in range(atmos.ny):
			# 		for pID in range(Npar):
			# 			for sID in range(4):
			# 				rf[idx,idy,pID,:,sID] = np.ones(Nw)*(1+sID) + 10*pID + 100*idy + 1000*idx
			# 		for sID in range(4):
			# 			diff[idx,idy,:,sID] = np.ones(Nw)*(1+sID) + 10*idy + 100*idx
			
			# scale RFs with weights and noise scale
			rf *= globin.weights
			rf /= noise_scale_rf

			# calculate difference between observation and synthesis
			diff = obs.spec - spec.spec
			diff *= globin.weights

			# calculate chi2
			chi2_old = np.sum(diff**2 / noise_stokes**2 * globin.wavs_weight**2 * weights**2, axis=(2,3)) / dof
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
			# It produces expected results.

		H = copy.deepcopy(JTJ)
		diagonal_elements = np.diag(JTJ) * (1 + LM_parameter)
		np.fill_diagonal(H, diagonal_elements)
		proposed_steps = np.linalg.solve(H, delta)

		old_local_parameters = copy.deepcopy(atmos.values)
		old_global_pars = copy.deepcopy(atmos.global_pars)
		atmos.update_parameters(proposed_steps)
		atmos.check_parameter_bounds()

		if ("loggf" in atmos.global_pars) or ("dlam" in atmos.global_pars):
			globin.write_line_parameters(atmos.line_lists_path[0],
									   atmos.global_pars["loggf"][0,0], atmos.line_no["loggf"],
									   atmos.global_pars["dlam"][0,0], atmos.line_no["dlam"])

		atmos.build_from_nodes()
		corrected_spec,_,_ = globin.compute_spectra(atmos)
		corrected_spec.broaden_spectra(atmos.vmac)

		new_diff = obs.spec - corrected_spec.spec
		new_diff *= globin.weights
		chi2_new = np.sum(new_diff**2 / noise_stokes**2 * globin.wavs_weight**2 * weights**2, axis=(2,3)) / dof

		if np.sum(chi2_new) > np.sum(chi2_old):
			LM_parameter *= 10
			atmos.values = old_local_parameters
			atmos.global_pars = old_global_pars
			updated_parameters = False
			num_failed += 1
		else:
			chi2[...,itter] = chi2_new
			LM_parameter /= 10
			updated_parameters = True
			itter += 1
			num_failed = 0

		if ("loggf" in atmos.global_pars) or ("dlam" in atmos.global_pars):
			globin.write_line_parameters(atmos.line_lists_path[0],
									   atmos.global_pars["loggf"][0,0], atmos.line_no["loggf"],
									   atmos.global_pars["dlam"][0,0], atmos.line_no["dlam"])

		if LM_parameter<=1e-5:
			LM_parameter = 1e-5
		# if Marquardt parameter is to large, we break
		if LM_parameter>=1e8:
			print("Upper limit in LM_parameter. We break\n")
			break_flag = True

		# we check if chi2 has converged for each pixel
		# if yes, we set break_flag to True
		# we do not check for chi2 convergence until 3rd iteration
		if (itter)>=3 and updated_parameters:
			# need to get -2 and -1 because I already rised itter by 1 
			# when chi2 list was updated.
			relative_change = abs(np.sum(chi2[...,itter-1])/np.sum(chi2[...,itter-2]) - 1)
			if np.sum(chi2[...,itter-1])<1e-32:
				print("chi2 is way low!\n")
				break_flag = True
			elif relative_change<globin.chi2_tolerance:
				print("chi2 relative change is smaller than given value.\n")
				break_flag = True
			elif np.sum(chi2[...,itter-1]) < 1 and globin.noise!=0:
				print("chi2 smaller than 1\n")
				break_flag = True
		
		if updated_parameters and verbose:
			print(atmos.values)
			print(atmos.global_pars)
			print(LM_parameter)
			print("\n--------------------------------------------------\n")

		# if all pixels have converged, we stop inversion
		if break_flag:
			break

		if (num_failed==10 and itter>=3):
			print("Failed 10 times to fix the LM parameter. We break.\n")
			break

	atmos.build_from_nodes(False)
	
	inverted_spectra,_,_ = globin.compute_spectra(atmos)
	inverted_spectra.broaden_spectra(atmos.vmac)

	# diff = obs.spec - inverted_spectra.spec
	# chi2 = np.sum(diff**2 / noise_stokes**2 * globin.wavs_weight**2 * weights**2, axis=(2,3)) / dof

	try:
		atmos.compute_errors(JTJ, chi2_old)
	except:
		print("Failed to compute parameter errors\n")

	if save_output is not None:
		output_path = f"runs/{globin.wd}"

		atmos.save_atmosphere(f"{output_path}/inverted_atmos.fits")
		atmos.save_atomic_parameters(f"{output_path}/inverted_atoms.fits", kwargs={"RLK_LIST" : (f"{globin.cwd}/{atmos.line_lists_path[0].split('/')[-1]}", "reference line list")})
		inverted_spectra.save(f"{output_path}/inverted_spectra.fits", globin.wavelength)
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
			# else:
			# 	for i_ in range(len(atmos.global_pars[par])):
			# 		out_file.write("{:s}    {: 4d}    {: 5.4f}\n".format(par, atmos.line_no[par][i_]+1, atmos.global_pars[par][i_]))

		out_file.write("\n\n     #===--- globin input file ---===#\n\n")
		out_file.write(globin.parameters_input)
		out_file.write("\n\n     #===--- RH input file ---===#\n\n")
		out_file.write(globin.keyword_input)

		out_file.close()

	return atmos, inverted_spectra

def invert_mcmc(init, save_output, verbose):
	obs = init.obs
	atmos = init.atm

	atmos.build_from_nodes()
	spec, _, _ = globin.compute_spectra(atmos, init.rh_spec_name, init.wavelength)

	diff = obs.spec - spec.spec
	chi2 = np.sum(diff**2 / noise_stokes**2 * init.wavs_weight**2, axis=(2,3)) / dof

	return atmos, spec

def lnprior(pars):
	"""
	Check if each parameter is in its respective bounds given by globin.limit_values.

	If one fails, return -np.inf, else return 0.
	"""
	blos = theta
	if abs(blos) < 3000:
		return 0.0
	return -np.inf

def lnlike(theta, x, y, yp, yerr):
	"""
	Compute chi2.

	We need:
	  -- observations
	  -- noise
	  -- parameters to compute spectra
	"""
	return -0.5 * np.sum( (y-fn(theta, x, yp))**2 / yerr**2)

def lnprob(theta, x, y, yp, yerr):
	"""
	Compute product of prior and likelihood.

	We need what is needed for prior and likelihood
	"""
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yp, yerr)

