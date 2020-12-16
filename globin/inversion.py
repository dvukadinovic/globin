import numpy as np
import sys
import copy
import time

import globin

def invert(init):
	if init.mode==0:
		print("Parameters for synthesis mode are read. We can not run inversion.\nChange mode before running again.\n")
	elif init.mode==1:
		invert_pxl_by_pxl(init)
	elif init.mode==3:
		# if we have global parameters
		# if not, redirect to invert_pxl_by_pxl
		invert_global(init)
	else:
		print(f"Not supported mode {init.mode} currently.")

def invert_pxl_by_pxl(init):
	"""
	As input we expect all data to be present :)

	Pixel-by-pixel inversion of atmospheric parameters.

	Parameters:
	---------------
	init : InputData
		InputData object in which we have everything stored.
	"""
	import matplotlib.pyplot as plt

	obs = init.obs
	atmos = init.atm

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
	ind_min = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[0]))
	ind_max = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[-1]))+1
	
	if init.noise!=0:
		StokesI_cont = obs.data[:,:,ind_min,1]
		noise_lvl = init.noise * StokesI_cont
		# noise_wavelength = (nx, ny, nw)
		noise_wavelength = np.sqrt(obs.data[:,:,ind_min:ind_max,1].T / StokesI_cont.T).T
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

	itter = np.zeros((atmos.nx, atmos.ny), dtype=np.int)
	for i_ in range(init.max_iter):
		print("Iteration: {:2}\n".format(i_+1))
		
		# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
		#               spec.shape = (nx, ny, Nw, 5)
		rf, spec, atm = globin.compute_rfs(init, local=True)
		
		#--- scale RFs with weights and noise scale
		rf *= init.weights
		rf /= noise_scale_rf

		diff = obs.spec - spec[:,:,:,1:]
		diff *= init.weights
		chi2_old = np.sum(diff**2 / noise_stokes**2 * init.wavs_weight**2, axis=(2,3)) / dof
		diff /= noise_stokes_scale

		"""
		Gymnastics with indices for solving LM equations for
		next step parameters.
		"""
		J = rf.reshape(atmos.nx, atmos.ny, Npar, 4*Nw)
		# J = (nx, ny, 4*nw, npar)
		J = np.moveaxis(J, 2, 3)
		# JT = (nx, ny, npar, 4*nw)
		JT = np.einsum("ijlk", J)
		# JTJ = (nx, ny, npar, npar)
		JTJ_new = np.einsum("...ij,...jk", JT, J)
		# get diagonal elements from hessian matrix
		diagonal_elements = np.einsum("...kk->...k", JTJ_new)
		# hessian = (nx, ny, npar, npar)
		H_new = copy.deepcopy(JTJ_new)
		# multiply with LM parameter
		H_new[X,Y,P,P] = np.einsum("...i,...", diagonal_elements, 1+LM_parameter)
		# delta = (nx, ny, npar)
		delta_new = np.einsum("...pw,...w", JT, diff.reshape(atmos.nx, atmos.ny, 4*Nw))
		# proposed_steps = (nx, ny, npar)
		proposed_steps_new = np.linalg.solve(H_new, delta_new)
		
		break_loop = True
		for j_ in range(N_search_for_lambda):
			old_parameters = copy.deepcopy(atmos.values)
			low_ind, up_ind = 0, 0
			for parID in atmos.values:
				low_ind = up_ind
				up_ind += len(atmos.nodes[parID])
				step = proposed_steps_new[:,:,low_ind:up_ind] * globin.parameter_scale[parID]
				# step = np.around(step, decimals=8)
				# we do not perturb parameters of those pixels which converged
				step = np.einsum("...i,...->...i", step, stop_flag)
				# print(step)
				atmos.values[parID] += step
			atmos.check_parameter_bounds()

			atmos.build_from_nodes(init.ref_atm)
			corrected_spec,_ = globin.compute_spectra(init, atmos, False, True)

			new_diff = obs.spec - corrected_spec[:,:,:,1:]
			new_diff *= init.weights
			chi2_new = np.sum(new_diff**2 / noise_stokes**2 * init.wavs_weight**2, axis=(2,3)) / dof

			# print(np.log10(chi2_new), np.log10(chi2_old))

			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					if stop_flag[idx,idy]==1:
						# print(f"  [{idx},{idy}] --> {np.log10(chi2_new[idx,idy])} ? {np.log10(chi2_old[idx,idy])}")
						if chi2_new[idx,idy] > chi2_old[idx,idy]:
							LM_parameter[idx,idy] *= 10
							for parID in old_parameters:
								atmos.values[parID][idx,idy] = old_parameters[parID][idx,idy]
						else:
							chi2[idx,idy,itter[idx,idy]] = chi2_new[idx,idy]
							LM_parameter[idx,idy] /= 10
							itter[idx,idy] += 1
							# break_loop = True
			
			# we do searching for best LM parameter only in first iteration
			# aftwerwards, we only go one time through this loop
			N_search_for_lambda = 1

			init.atm = atmos

			# if we have changed Marquardt parameter, we go for new RF estimation
			if break_loop:
				break

		print(atmos.values)
		print(LM_parameter)

		# if Marquardt parameter is to large, we break
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				if LM_parameter[idx,idy]<=1e-5:
					LM_parameter[idx,idy] = 1e-5
				if LM_parameter[idx,idy]>=1e8:
					stop_flag[idx,idy] = 0

		# we check if chi2 has converged for each pixel
		# if yes, we set stop_flag to 1 (True)
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				if stop_flag[idx,idy]==1:
					it_no = itter[idx,idy]
					if it_no>=5:
						# need to get -2 and -1 because I already rised itter by 1 
						# when chi2 list was updated.
						relative_change = abs(chi2[idx,idy,it_no-2]/chi2[idx,idy,it_no-1] - 1)
						if relative_change<init.chi2_tolerance:
							# print(relative_change)
							# print(chi2[idx,idy,it_no-2])
							# print(chi2[idx,idy,it_no-1])
							print(f"--> [{idx},{idy}] : chi2 relative change is smaller than given value.")
							stop_flag[idx,idy] = 0
						if chi2[idx,idy,it_no-1] < 1 and init.noise!=0:
							# print(chi2[idx,idy,it_no-1])
							print(f"--> [{idx},{idy}] : chi2 smaller than 1")
							stop_flag[idx,idy] = 0

		print("\n--------------------------------------------------\n")
		
		# if all pixels have converged, we stop inversion
		if np.sum(stop_flag)==0:
			break

	fname = "results"

	atmos.build_from_nodes(init.ref_atm)
	atmos.save_atmosphere(f"{fname}/inverted_atmos.fits")

	globin.spectrum_path = f"{fname}/inverted_spectra.fits"
	inverted_spectra = globin.compute_spectra(init, atmos, True, True)

	globin.save_chi2(chi2, f"{fname}/chi2.fits")
	
	end = time.time() - start
	print("\nFinished in: {0}\n".format(end))

	#--- inverted params comparison with expected values
	out_file = open("{:s}/output.log".format(fname), "w")

	out_file.write("Run time: {:10.1f}\n\n".format(end))

	out_file.write("\n\n     #===--- globin input file ---===#\n\n")
	out_file.write(init.params_input)
	out_file.write("\n\n     #===--- RH input file ---===#\n\n")
	out_file.write(init.rh_input)

	out_file.close()

def invert_global(init):
	"""
	As input we expect all data to be present :)

	Pixel-by-pixel inversion of atmospheric parameters.

	Parameters:
	---------------
	init : InputData
		InputData object in which we have everything stored.
	"""
	import matplotlib.pyplot as plt

	obs = init.obs
	atmos = init.atm

	print("Initial parameters:")
	print(atmos.values)
	print(atmos.global_pars)
	print()

	Nw = len(init.wavelength)
	Npar = atmos.n_local_pars + atmos.n_global_pars

	if Npar==0:
		sys.exit("There are no parameters to fit.\n   We exit.\n")

	# indices for wavelengths min/max for which we are fiting; based on input
	ind_min = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[0]))
	ind_max = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[-1]))+1
	
	if init.noise!=0:
		StokesI_cont = obs.data[:,:,ind_min,1]
		noise_lvl = init.noise * StokesI_cont
		# noise_wavelength = (nx, ny, nw)
		noise_wavelength = np.sqrt(obs.data[:,:,ind_min:ind_max,1].T / StokesI_cont.T).T
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

	LM_parameter = init.marq_lambda
	chi2 = np.zeros(init.max_iter, dtype=np.float64)
	N_search_for_lambda = 5
	dof = np.count_nonzero(init.weights) * Nw - Npar

	start = time.time()

	itter = 0
	stop_flag = False
	for i_ in range(init.max_iter):
		print("Iteration: {:2}\n".format(i_+1))
		
		# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
		#               spec.shape = (nx, ny, Nw, 5)
		rf, spec, atm = globin.compute_rfs(init, local=False)

		# for idx in range(atmos.nx):
		# 	for idy in range(atmos.ny):
		# 		plt.plot(rf[idx,idy,1,:,0])
		# 		plt.show()
		
		#--- scale RFs with weights and noise scale
		rf *= init.weights
		rf /= noise_scale_rf

		diff = obs.spec - spec[:,:,:,1:]
		diff *= init.weights
		chi2_old = np.sum(diff**2 / noise_stokes**2 * init.wavs_weight**2) / dof
		diff /= noise_stokes_scale
		
		# print(diff.shape)

		aux = rf.reshape(atmos.nx, atmos.ny, Npar, 4*Nw)
		aux = np.moveaxis(aux, 2, 3)
		
		J = np.zeros((4*Nw*(atmos.nx*atmos.ny), atmos.n_local_pars*(atmos.nx*atmos.ny) + atmos.n_global_pars))

		l = 4*Nw
		n_atmosphere = 0
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				low = n_atmosphere*l
				up = low + l 
				ll = n_atmosphere*atmos.n_local_pars
				uu = ll + atmos.n_local_pars
				J[low:up,ll:uu] = aux[idx,idy,:,:atmos.n_local_pars]
				n_atmosphere += 1

		for gID in range(atmos.n_global_pars):
			J[:,uu+gID] = aux[:,:,:,atmos.n_local_pars+gID].flatten()

		JT = J.T
		JTJ = np.dot(JT,J)
		H = copy.deepcopy(JTJ)
		diagonal_elements = np.diag(JTJ) * (1 + LM_parameter)
		np.fill_diagonal(H, diagonal_elements)
		plt.imshow(np.log10(H), aspect="auto")
		plt.colorbar()
		plt.show()
		delta = np.dot(JT, diff.flatten())
		proposed_steps = np.linalg.solve(H, delta)
		
		break_loop = False
		for j_ in range(N_search_for_lambda):
			old_parameters = copy.deepcopy(atmos.values)
			old_global_pars = copy.deepcopy(atmos.global_pars)
			low_ind, up_ind = 0, 0
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					for parID in atmos.values:
						low_ind = up_ind
						up_ind += len(atmos.nodes[parID])
						step = proposed_steps[low_ind:up_ind] * globin.parameter_scale[parID]
						step = np.around(step, decimals=8)
						# we do not perturb parameters of those pixels which converged
						atmos.values[parID][idx,idy] += step
			for parID in atmos.global_pars:
				low_ind = up_ind
				up_ind += 1
				step = proposed_steps[low_ind:up_ind] * globin.parameter_scale[parID]
				step = np.around(step, decimals=8)
				# we do not perturb parameters of those pixels which converged
				atmos.global_pars[parID] += step
			atmos.check_parameter_bounds()

			atmos.build_from_nodes(init.ref_atm)
			corrected_spec,_ = globin.compute_spectra(init, atmos, False, True)

			new_diff = obs.spec - corrected_spec[:,:,:,1:]
			new_diff *= init.weights
			chi2_new = np.sum(new_diff**2 / noise_stokes**2 * init.wavs_weight**2) / dof

			if chi2_new > chi2_old:
				LM_parameter *= 10
				atmos.values = old_parameters
				atmos.global_pars = old_global_pars
			else:
				chi2[itter] = chi2_new
				itter += 1
				LM_parameter /= 10
				break_loop = True
			
			# we do searching for best LM parameter only in first iteration
			# aftwerwards, we only go one time through this loop
			N_search_for_lambda = 1

			init.atm = atmos

			# if we have changed Marquardt parameter, we go for new RF estimation
			if break_loop:
				break

		print(atmos.values)
		print(atmos.global_pars)
		print(LM_parameter)

		# if Marquardt parameter is to large, we break
		if LM_parameter<=1e-5:
			LM_parameter = 1e-5
		if LM_parameter>=1e8:
			stop_flag = True

		# we check if chi2 has converged for each pixel
		# if yes, we set stop_flag to True
		if (itter)>=3:
			# need to get -2 and -1 because I already rised itter by 1 
			# when chi2 list was updated.
			relative_change = abs(chi2[itter-2]/chi2[itter-1] - 1)
			if relative_change<init.chi2_tolerance:
				print(relative_change)
				print(chi2[itter-2])
				print(chi2[itter-1])
				print("chi2 relative change is smaller than given value.")
				stop_flag = True
			if chi2[itter-1] < 1 and init.noise!=0:
				print(chi2[itter])
				print("chi2 smaller than 1")
				stop_flag = True

		# if all pixels have converged, we stop inversion
		if stop_flag:
			break

		print("\n--------------------------------------------------\n")

	fname = "results"

	atmos.build_from_nodes(init.ref_atm)
	atmos.save_atmosphere(f"{fname}/inverted_atmos.fits")

	globin.spectrum_path = f"{fname}/inverted_spectra.fits"
	inverted_spectra = globin.compute_spectra(init, atmos, True, True)

	globin.save_chi2(chi2, f"{fname}/chi2.fits")
	
	end = time.time() - start
	print("Finished in: {0}\n".format(end))

	#--- inverted params comparison with expected values
	out_file = open("{:s}/output.log".format(fname), "w")

	out_file.write("Run time: {:10.1f}\n\n".format(end))

	out_file.write("\n\n     #===--- Global parameters ---===#\n\n")

	for par in atmos.global_pars:
		for i_ in range(len(atmos.global_pars[par])):
			out_file.write("{:s}    {:4.3f}\n".format(par, atmos.global_pars[par][i_]))

	out_file.write("\n\n     #===--- globin input file ---===#\n\n")
	out_file.write(init.params_input)
	out_file.write("\n\n     #===--- RH input file ---===#\n\n")
	out_file.write(init.rh_input)

	out_file.close()
