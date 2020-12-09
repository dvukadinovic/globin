import numpy as np
import sys
import copy
import time

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

	LM_parameter = np.ones((obs.nx, obs.ny), dtype=np.float64) * init.marq_lambda
	# flags those pixels whose chi2 converged
	stop_flag = np.zeros((obs.nx, obs.ny), dtype=np.float64)

	Nw = len(init.wavelength)
	Npar = atmos.free_par

	if Npar==0:
		sys.exit("There is no parameters to fit.\n   We exit.\n")

	# indices for wavelengths min/max for which we are fiting; based on input
	ind_min = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[0]))
	ind_max = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[-1]))+1

	# matrices
	jacobian = np.zeros((obs.nx, obs.ny, 4*Nw, Npar), dtype=np.float64)
	JTJ = np.zeros((obs.nx, obs.ny, Npar, Npar), dtype=np.float64)
	hessian = np.zeros((obs.nx, obs.ny, Npar, Npar), dtype=np.float64)
	
	# casting noise in appropriate dimensions
	noise_scale_rf = np.ones((obs.nx, obs.ny, Npar, Nw, 4), dtype=np.float64)
	noise_stokes = np.ones((obs.nx, obs.ny, Nw, 4), dtype=np.float64)
	noise_stokes_scale = np.ones((obs.nx, obs.ny, Nw, 4), dtype=np.float64)

	if init.noise!=0:
		StokesI_cont = obs.data[:,:,ind_min,1]
		noise_lvl = init.noise * StokesI_cont
		# noise_wavelength = (nx,ny,nw)
		noise_wavelength = np.sqrt(obs.data[:,:,ind_min:ind_max,1].T / StokesI_cont.T).T
		# noise = (nx,ny,nw)
		noise = (noise_lvl * noise_wavelength.T).T
		for sID in range(4):
			noise_stokes_scale[:,:,:,sID] = noise_wavelength
			noise_stokes[:,:,:,sID] = noise
			for pID in range(Npar):
				noise_scale_rf[:,:,pID,:,sID] = noise_stokes_scale[:,:,:,sID]

	chi2 = np.zeros((init.max_iter, obs.nx, obs.ny), dtype=np.float64)
	N_search_for_lambda = 5
	dof = np.count_nonzero(init.weights) * Nw

	start = time.time()

	itter = 0
	for i_ in range(init.max_iter):
		print("Iteration: {:2}\n".format(i_+1))
		
		# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
		#               spec.shape = (nx, ny, Nw, 5)
		rf, spec, atm = globin.compute_rfs(init)
		
		#--- scale RFs with weights and noise scale
		rf *= init.weights
		rf /= noise_scale_rf
		
		diff = obs.spec - spec[:,:,:,1:]
		diff *= init.weights
		chi2_old = np.sum(diff**2 / noise_stokes**2 * init.wavs_weight**2, axis=(2,3)) / dof
		diff /= noise_stokes_scale

		"""
		Gymnastics with indices and solving equation for
		next step parameters.
		"""
		J = rf.reshape(obs.nx, obs.ny, Npar, 4*Nw)
		J = np.moveaxis(J, 2, 3)
		# JT = (nx,ny,npar,4*nw)
		JT = np.einsum("ijlk", J)

		# JTJ = (nx,ny,npar,npar)
		JTJ_new = np.einsum("...ij,...jk", JT, J)
		# get diagonal elements from hessian matrix
		diagonal_elements = np.einsum("...kk->...k", JTJ_new)
		# diagonal elements indices for each axis
		indx, indy, indp1, indp2 = np.where(JTJ_new==diagonal_elements)
		# hessian = (nx,ny,npar,npar)
		hessian_new = copy.deepcopy(JTJ_new)
		# multiply with LM parameter
		hessian_new[indx,indy,indp1,indp2] = diagonal_elements * (1 + LM_parameter)
		# delta = (nx,ny,npar)
		delta_new = np.einsum("...pw,...w", JT, diff.reshape(obs.nx, obs.ny, 4*Nw))
		# proposed_steps = (nx,ny,npar)
		proposed_steps_new = np.linalg.solve(hessian_new, delta_new)

		#--- old pixel-by-pixel calculation using for loops
		# # loop for Marquardt lambda correction
		# for idx in range(obs.nx):
		# 	for idy in range(obs.ny):
		# 		if stop_flag[idx,idy]==0:
		# 			for parID in range(Npar):
		# 				jacobian[idx,idy,:,parID] = rf[idx,idy,parID].flatten()
		# jacobian_t = jacobian.T
		
		# break_loop = False
		# proposed_steps = np.zeros((obs.nx, obs.ny, Npar), dtype=np.float64)
		for j_ in range(N_search_for_lambda):
		# 	# solve LM equation for new parameter steps
		# 	for idx in range(obs.nx):
		# 		for idy in range(obs.ny):
		# 			if stop_flag[idx,idy]==0:
		# 				JTJ[idx,idy] = np.dot(jacobian_t[:,:,idy,idx], jacobian[idx,idy])
		# 				hessian[idx,idy] = JTJ[idx,idy]
		# 				diagonal_elements = np.diag(JTJ[idx,idy]) * (1 + LM_parameter[idx,idy])
		# 				np.fill_diagonal(hessian[idx,idy], diagonal_elements)
		# 				delta = np.dot(jacobian_t[:,:,idy,idx], diff[idx,idy].flatten())
		# 				proposed_steps[idx,idy] = np.dot(np.linalg.inv(hessian[idx,idy]), delta)

			old_parameters = copy.deepcopy(atmos.values)
			low_ind, up_ind = 0, 0
			for parID in atmos.values:
				low_ind = up_ind
				up_ind += len(atmos.nodes[parID])
				step = proposed_steps_new[:,:,low_ind:up_ind] * globin.parameter_scale[parID]
				step = np.around(step, decimals=8)
				atmos.values[parID] += step
			atmos.check_parameter_bounds()

			atmos.build_from_nodes(init.ref_atm, init.interp_degree)
			corrected_spec,_ = globin.compute_spectra(init, atmos, False, True)

			new_diff = obs.spec - corrected_spec[:,:,:,1:]
			new_diff *= init.weights
			chi2_new = np.sum(new_diff**2 / noise_stokes**2 * init.wavs_weight**2, axis=(2,3)) / dof

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

			init.atm = atmos

			# if we have changed Marquardt parameter, we go for new RF estimation
			if break_loop:
				break

		print()
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
					# need to get -2 and -1 because I already rised itter by 1 
					# when chi2 list was updated.
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

	fname = "results"

	atmos.build_from_nodes(init.ref_atm, init.interp_degree)
	atmos.save_cube(f"{fname}/inverted_atmos.fits")

	init.spectrum_path = f"{fname}/inverted_spectra.fits"
	inverted_spectra = globin.compute_spectra(init, atmos, True, True)

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
	out_file = open("{:s}/output.log".format(fname,noise), "w")

	end = time.time() - start
	print("Finished in: {0}\n".format(end))

	out_file.write("Run time: {:10.1f}\n\n".format(end))
	
	idx, idy = 0,0
	i_ = 0
	fact = 1
	labels = {"temp"  : r"T [K]",
			  "vz"    : r"$v_z$ [km/s]",
			  "mag"   : r"B [G]",
			  "gamma" : r"$\gamma$ [deg]",
			  "chi"   : r"$\chi$ [deg]"}
	out_file.write("\n     #===--- inverted parameters ---===#\n\n")
	for parameter in atmos.nodes:
		if parameter=="mag":
			fact = 1e4
		if parameter=="gamma" or parameter=="chi":
			fact = 180/np.pi

		plt.figure(2+i_)
		x = atmos.nodes[parameter]
		y = atmos.values[parameter][idx,idy]

		parID = atmos.par_id[parameter]
		plt.plot(x, y*fact, "ro")
		plt.plot(atmos.logtau, atmos.data[0,0,parID]*fact, color="tab:blue")
		plt.plot(init.ref_atm.data[idx,idy,0], init.ref_atm.data[idx,idy,parID]*fact, "k-")
		plt.xlabel(r"$\log \tau$")
		plt.ylabel(labels[parameter])
		plt.savefig("{:s}/{:s}_n{:1d}.png".format(fname,parameter,noise))
		plt.close()
		i_ += 1

		out_file.writelines("# " + labels[parameter] + "\n")
		for i_ in range(len(x)):
			out_file.write("{:2.1f}    {:5.4f}\n".format(x[i_], y[i_]*fact))
	
	out_file.write("\n\n     #===--- globin input file ---===#\n\n")
	out_file.write(init.params_input)
	out_file.write("\n\n     #===--- RH input file ---===#\n\n")
	out_file.write(init.rh_input)

	out_file.close()
