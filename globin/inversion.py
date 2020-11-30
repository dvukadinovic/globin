import numpy as np
import sys
import copy

import globin

def search_best_lambda(init, chi2_old, obs, atmos, diff, jacobian, jacobian_t, LM_parameter, noise, N_search_for_lambda, ind_min, ind_max):
	stop_flag = np.zeros((obs.nx, obs.ny))
	proposed_steps = np.zeros((obs.nx, obs.ny))
	dof = 4 * (ind_max-ind_min)

	for j_ in range(N_search_for_lambda):
		# solve LM equation for new parameter steps
		for idx in range(obs.nx):
			for idy in range(obs.ny):
				if stop_flag[idx,idy]==0:
					JTJ = np.dot(jacobian_t[:,:,idx,idy], jacobian[idx,idy])
					hessian = JTJ
					diagonal_elements = np.diag(JTJ[idx,idy]) * (1 + LM_parameter[idx,idy])
					np.fill_diagonal(hessian, diagonal_elements)
					delta = np.dot(jacobian_t[:,:,idx,idy], diff[idx,idy].flatten())
					proposed_steps[idx,idy] = np.dot(np.linalg.inv(hessian), delta)
		# print(proposed_steps[idx,idy])

		old_parameters = copy.deepcopy(atmos.values)
		low_ind, up_ind = 0, 0
		for parID in atmos.values:
			low_ind = up_ind
			up_ind += len(atmos.nodes[parID])
			atmos.values[parID] += proposed_steps[:,:,low_ind:up_ind]
		atmos.check_parameter_bounds()

		atmos.build_from_nodes(init.ref_atm)
		corrected_spec = globin.compute_spectra(init, atmos)

		new_diff = obs.spec[:,:,ind_min:ind_max] - corrected_spec[:,:,ind_min:ind_max,1:]
		new_diff[:,:,:] *= init.weights
		for s in range(4):
			new_diff[:,:,:,s] /= noise
		chi2_new = np.sum(new_diff*new_diff, axis=(2,3)) / dof

		for idx in range(obs.nx):
			for idy in range(obs.ny):
				if stop_flag[idx,idy]==0:
					if chi2_new[idx,idy] > chi2_old[idx,idy]:
						LM_parameter[idx,idy] *= 10
						atmos.values = old_parameters
					else:
						chi2[i_,idx,idy] = chi2_new[idx,idy]
						LM_parameter[idx,idy] /= 10
						break_loop = True

		# if we have changed Marquardt parameter, we go for new RF estimation
		if break_loop:
			break

def multiply_matrices(A,B):
	"""
	Multiply matrices elemnt-wise.

	Here A is differnece between observed and computed spectra or response
	functions.

	Matrix B is weights (4 values for 4 Stokes components) or wavelength
	dependent noise (nx, ny, nw).
	"""
	shape = A.shape


def invert(init):
	"""
	As input we expect all data to be present :)

	Parameters:
	---------------
	init : InputData
		InputData object in which we have everything stored.
	"""
	import matplotlib.pyplot as plt

	if init.mode==0:
		sys.exit("Parameters for synthesis mode are read. We can not run inversion.\nChange mode before running again.\n")

	obs = init.obs
	atmos = init.atm

	# print("Initial parameters: ", atmos.values)
	# print()

	LM_parameter = np.ones((obs.nx, obs.ny)) * init.marq_lambda
	svd_tolerance = 1e-4 # not used currently

	# flags those pixels whose chi2 converged; not used currently
	stop_flag = np.zeros((obs.nx, obs.ny))

	Nw = len(init.wavelength)
	Npar = atmos.free_par

	# indices for wavelengths min/max for which we are fiting; based on input
	ind_min = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[0]))
	ind_max = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[-1]))+1
	Nw = ind_max - ind_min

	# matrices for solveing LS equation
	jacobian = np.zeros((obs.nx, obs.ny, Nw*4, Npar))
	JTJ = np.zeros((obs.nx, obs.ny, Npar, Npar))
	hessian = np.zeros((obs.nx, obs.ny, Npar, Npar))
	
	# casting noise and weights in appropriate dimensions
	noise_for_diff = np.zeros((obs.nx, obs.ny, ind_max-ind_min, 4))
	noise_for_rf = np.zeros((obs.nx, obs.ny, Npar, ind_max-ind_min, 4))
	weights_for_diff = np.ones((obs.nx, obs.ny, ind_max-ind_min, 4))
	weights_for_rf = np.ones((obs.nx, obs.ny, Npar, ind_max-ind_min, 4))

	if init.noise!=0:
		StokesI_cont = obs.data[:,:,ind_min,1]
		noise_lvl = init.noise * StokesI_cont
		noise_scaling = np.sqrt(obs.data[:,:,ind_min:ind_max,1] / StokesI_cont)
		noise = noise_lvl * noise_scaling
	else:
		noise = np.ones((obs.nx, obs.ny, ind_max-ind_min))

	for i_ in range(4):
		noise_for_diff[:,:,:,i_] = noise
		weights_for_diff[:,:,:,i_] = init.weights[i_]
		for j_ in range(Npar):
			noise_for_rf[:,:,j_,:,i_] = noise
			weights_for_rf[:,:,j_,:,i_] = init.weights[i_]

	chi2 = np.zeros((init.max_iter, obs.nx, obs.ny))
	N_search_for_lambda = 5
	dof = 4 * (ind_max-ind_min)

	itter = 0
	for i_ in range(init.max_iter):
		print("Iteration: {:2}\n".format(i_+1))
		
		# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
		#               spec.shape = (nx, ny, Nw, 5)
		rf, spec = globin.compute_rfs(init)
		rf = rf[:,:,:,ind_min:ind_max,:]

		diff = obs.spec[:,:,ind_min:ind_max] - spec[:,:,ind_min:ind_max,1:]
		diff *= weights_for_diff / noise_for_diff
		rf *= weights_for_rf / noise_for_rf
		chi2_old = np.sum(diff*diff, axis=(2,3)) / dof

		# Jacobian matrix
		for idx in range(obs.nx):
			for idy in range(obs.ny):
				if stop_flag[idx,idy]==0:
					for parID in range(Npar):
						jacobian[idx,idy,:,parID] = rf[idx,idy,parID].flatten()

		# is this correct? 
		jacobian_t = jacobian.T

		# if (i_+1)==1:
		# 	search_best_lambda()
		
		# loop for Marquardt lambda correction
		break_loop = False
		proposed_steps = np.zeros((obs.nx, obs.ny, Npar))
		for j_ in range(N_search_for_lambda):
			# solve LM equation for new parameter steps
			for idx in range(obs.nx):
				for idy in range(obs.ny):
					if stop_flag[idx,idy]==0:
						JTJ[idx,idy] = np.dot(jacobian_t[:,:,idx,idy], jacobian[idx,idy])
						hessian[idx,idy] = JTJ[idx,idy]
						diagonal_elements = np.diag(JTJ[idx,idy]) * (1 + LM_parameter[idx,idy])
						np.fill_diagonal(hessian[idx,idy], diagonal_elements)
						delta = np.dot(jacobian_t[:,:,idx,idy], diff[idx,idy].flatten())
						proposed_steps[idx,idy] = np.dot(np.linalg.inv(hessian[idx,idy]), delta)
			# print(proposed_steps[idx,idy])

			old_parameters = copy.deepcopy(atmos.values)
			low_ind, up_ind = 0, 0
			for parID in atmos.values:
				low_ind = up_ind
				up_ind += len(atmos.nodes[parID])
				atmos.values[parID] += proposed_steps[:,:,low_ind:up_ind]
			atmos.check_parameter_bounds()

			atmos.build_from_nodes(init.ref_atm)
			corrected_spec = globin.compute_spectra(init, atmos)

			new_diff = obs.spec[:,:,ind_min:ind_max] - corrected_spec[:,:,ind_min:ind_max,1:]
			new_diff *= weights_for_diff / noise_for_diff
			chi2_new = np.sum(new_diff*new_diff, axis=(2,3)) / dof

			for idx in range(obs.nx):
				for idy in range(obs.ny):
					if stop_flag[idx,idy]==0:
						if chi2_new[idx,idy] > chi2_old[idx,idy]:
							LM_parameter[idx,idy] *= 10
							atmos.values = old_parameters
						else:
							chi2[itter,idx,idy] = chi2_new[idx,idy]
							itter += 1
							LM_parameter[idx,idy] /= 10
							break_loop = True
			
			# we do searching for best LM parameter only in first iteration
			# aftwerwards, we only go one time through this loop
			N_search_for_lambda = 1

			# if we have changed Marquardt parameter, we go for new RF estimation
			if break_loop:
				break

		print(atmos.values)
		print(LM_parameter)

		# if Marquardt parameter is to large, we break
		for idx in range(obs.nx):
			for idy in range(obs.ny):
				if LM_parameter[idx,idy]<=1e-5:
					LM_parameter[idx,idy] = 1e-5
				if LM_parameter[idx,idy]>=1e8:
					stop_flag[idx,idy] = 1

		# we check if chi2 has converged for each pixel
		# if yes, we set stop_flag to 1 (True)
		if (itter)>=3:
			for idx in range(obs.nx):
				for idy in range(obs.ny):
					relative_change = abs(chi2[itter-2,idx,idy]/chi2[itter-1,idx,idy] - 1)
					# print(relative_change)
					# print(chi2[itter-1,idx,idy])
					if relative_change<init.chi2_tolerance:
						print(relative_change)
						print(chi2[itter-2,idx,idy])
						print(chi2[itter-1,idx,idy])
						print("chi2 relative change is smaller than given value.")
						stop_flag[idx,idy] = 1
					if chi2[itter-1,idx,idy] < 1 and init.noise!=0:
						print(chi2[itter, idx, idy])
						print("chi2 smaller than 1")
						stop_flag[idx,idy] = 1

			# if all pixels have converged, we stop inversion
			if np.sum(stop_flag)==(obs.nx*obs.ny):
				break

		print("\n--------------------------------------------------\n")

	fname = "invert_temp_vz"

	if init.noise!=0:
		noise = int(abs(np.log10(init.noise)))
	else:
		noise = 0

	#--- chi2 plot
	itt_num = np.arange(0,itter)
	plt.plot(itt_num+1, np.log10(chi2[:itter,0,0]), c="k")
	plt.xlabel("Iteration")
	plt.ylabel(r"$\log (\chi^2)$")
	plt.savefig("{:s}/chi2_n{:1d}.png".format(fname,noise))
	plt.close()
	# plt.show()

	#--- Stokes vector plot
	fix, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

	for i in range(atmos.nx):
		for j in range(atmos.ny):
			# Stokes I
			axs[0,0].set_title("Stokes I")
			axs[0,0].plot((obs.data[0,0,:,0] - 401.6)*10, obs.spec[0,0,:,0])
			axs[0,0].plot((corrected_spec[i,j,:,0] - 401.6)*10, corrected_spec[i,j,:,1])
			# Stokes Q
			axs[0,1].set_title("Stokes Q")
			axs[0,1].plot((obs.data[0,0,:,0] - 401.6)*10, obs.spec[0,0,:,1])
			axs[0,1].plot((corrected_spec[i,j,:,0] - 401.6)*10, corrected_spec[i,j,:,2])
			# Stokes U
			axs[1,0].set_title("Stokes U")
			axs[1,0].plot((obs.data[0,0,:,0] - 401.6)*10, obs.spec[0,0,:,2])
			axs[1,0].plot((corrected_spec[i,j,:,0] - 401.6)*10, corrected_spec[i,j,:,3])
			# Stokes V
			axs[1,1].set_title("Stokes V")
			axs[1,1].plot((obs.data[0,0,:,0] - 401.6)*10, obs.spec[0,0,:,3])
			axs[1,1].plot((corrected_spec[i,j,:,0] - 401.6)*10, corrected_spec[i,j,:,4])

	axs[1,0].set_xlabel(r"$\Delta \lambda$ [$\AA$]")
	axs[1,1].set_xlabel(r"$\Delta \lambda$ [$\AA$]")
	axs[0,0].set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
	axs[1,0].set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")

	axs[0,0].set_xlim([-1, 1])
	axs[0,1].set_xlim([-1, 1])
	axs[1,0].set_xlim([-1, 1])
	axs[1,1].set_xlim([-1, 1])
	plt.savefig("{:s}/stokes_vector_n{:1d}.png".format(fname,noise))
	
	#--- inverted params comparison with expected values
	out_file = open("{:s}/inverted_parameters_n{:1d}".format(fname,noise), "w")
	
	idx, idy = 0,0
	i_ = 0
	fact = 1
	labels = {"temp"  : r"T [K]",
			  "vz"    : r"$v_z$ [km/s]",
			  "mag"   : r"B [G]",
			  "gamma" : r"$\gamma$ [deg]",
			  "chi"   : r"$\chi$ [deg]"}
	for parameter in atmos.nodes:
		if parameter=="mag":
			fact = 1e4
		if parameter=="gamma" or parameter=="chi":
			fact = 180/np.pi

		plt.figure(2+i_)
		x = atmos.nodes[parameter]
		y = atmos.values[parameter][idx,idy]
		
		atmos.build_from_nodes(init.ref_atm)

		parID = atmos.par_id[parameter]
		plt.plot(x, y*fact, "ro")
		plt.plot(atmos.logtau, atmos.data[0,0,parID]*fact, color="tab:blue")
		plt.plot(init.ref_atm.data[idx,idy,0], init.ref_atm.data[idx,idy,parID]*fact, "k-")
		plt.xlabel(r"$\log \tau$")
		plt.ylabel(labels[parameter])
		plt.savefig("{:s}/{:s}_n{:1d}.png".format(fname,parameter,noise))
		plt.close()
		i_ += 1

		out_file.writelines([f"# {parameter} [km/s]\n"])
		for i_ in range(len(x)):
			out_file.write("{:2.1f}    {:5.4f}\n".format(x[i_], y[i_]*fact))
	
	out_file.close()
	# plt.show()

	#--- save inverted atmos
	# atmos.save_cube()