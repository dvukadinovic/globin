import numpy as np
import sys
import copy

import globin

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

	print("Initial parameters: ", atmos.values)
	print()

	LM_parameter = np.ones((obs.nx, obs.ny)) * init.marq_lambda
	svd_tolerance = 1e-4 # not used currently

	# flags those pixels whose chi2 converged; not used currently
	stop_flag = np.zeros((obs.nx, obs.ny))

	Nw = len(init.wavelength)
	Npar = atmos.free_par

	if Npar==0:
		sys.exit("We do not have any parameters to fit.\n   We exit.\n")

	# indices for wavelengths min/max for which we are fiting; based on input
	ind_min = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[0]))
	ind_max = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[-1]))+1

	# matrices for solveing LS equation
	jacobian = np.zeros((obs.nx, obs.ny, Nw*4, Npar))
	JTJ = np.zeros((obs.nx, obs.ny, Npar, Npar))
	hessian = np.zeros((obs.nx, obs.ny, Npar, Npar))
	
	# casting noise and weights in appropriate dimensions
	noise_for_chi2 = np.zeros((obs.nx, obs.ny, Nw, 4))
	noise_scale_for_diff = np.zeros((obs.nx, obs.ny, Nw, 4))
	noise_scale_for_rf = np.zeros((obs.nx, obs.ny, Npar, Nw, 4))
	
	weights_for_diff = np.ones((obs.nx, obs.ny, Nw, 4))
	weights_for_rf = np.ones((obs.nx, obs.ny, Npar, Nw, 4))

	noise = np.ones((obs.nx, obs.ny, Nw))
	noise_scaling = np.ones((obs.nx, obs.ny, Nw))
	if init.noise!=0:
		StokesI_cont = obs.data[:,:,ind_min,1]
		noise_lvl = init.noise * StokesI_cont
		for idx in range(obs.nx):
			for idy in range(obs.ny):
				noise_scaling[idx,idy] = np.sqrt(obs.data[idx,idy,ind_min:ind_max,1] / StokesI_cont[idx,idy])
				noise[idx,idy] = noise_lvl[idx,idy] * noise_scaling[idx,idy]

	for i_ in range(4):
		noise_for_chi2[:,:,:,i_] = noise
		noise_scale_for_diff[:,:,:,i_] = noise_scaling
		weights_for_diff[:,:,:,i_] = init.weights[i_]
		for j_ in range(Npar):
			noise_scale_for_rf[:,:,j_,:,i_] = noise_scaling
			weights_for_rf[:,:,j_,:,i_] = init.weights[i_]

	chi2 = np.zeros((init.max_iter, obs.nx, obs.ny))
	N_search_for_lambda = 5
	dof = np.count_nonzero(init.weights) * (Nw)

	itter = 0
	for i_ in range(init.max_iter):
		print("Iteration: {:2}\n".format(i_+1))
		
		# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
		#               spec.shape = (nx, ny, Nw, 5)
		rf, spec, atm = globin.compute_rfs(init)

		# atmos.build_from_nodes(init.ref_atm)
		# plt.figure(1)
		# plt.plot(atm[0,0,0], atm[0,0,1]-atmos.data[0,0,1])
		
		# plt.figure(2)
		# plt.plot(atmos.data[0,0,0], np.log10(atm[0,0,2]) - np.log10(atmos.data[0,0,2]))

		# plt.figure(3)
		# plt.plot(atm[0,0,0], atm[0,0,3]-atmos.data[0,0,3])

		# plt.figure(4)
		# plt.plot(atm[0,0,0], atm[0,0,4]-atmos.data[0,0,4])

		# plt.figure(5)
		# plt.plot(atm[0,0,0], atm[0,0,6]-atmos.data[0,0,6])

		# plt.figure(6)
		# plt.plot(spec[0,0,:,0], spec[0,0,:,1])
		# plt.plot(obs.data[0,0,:,0], obs.data[0,0,:,1])
		
		# plt.plot(rf[0,0,:,:,0].T)

		# plt.show()

		diff = obs.spec - spec[:,:,:,1:]
		diff *= weights_for_diff
		chi2_old = np.sum(diff*diff / noise_for_chi2 / noise_for_chi2 * init.wav_weight*init.wav_weight, axis=(2,3)) / dof
		diff *= 1/noise_scale_for_diff
		
		rf *= weights_for_rf / noise_scale_for_rf

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
			corrected_spec,_ = globin.compute_spectra(init, atmos, False, True)

			new_diff = obs.spec - corrected_spec[:,:,:,1:]
			new_diff *= weights_for_diff
			chi2_new = np.sum(new_diff*new_diff / noise_for_chi2 / noise_for_chi2 * init.wav_weight*init.wav_weight, axis=(2,3)) / dof

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

	fname = "results/invert_temp_vz_mag_gamma"

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