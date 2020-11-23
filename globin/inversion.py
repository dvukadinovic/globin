import globin
import numpy as np
import sys

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

	# parameters = np.array([])
	# for parID in atmos.values:
	# 	parameters = np.append(parameters, atmos.values[parID])
	parameters = atmos.values["vz"]
	print("Initial parameters (vz): ", parameters)
	print()

	marquardt_lambda = np.ones((obs.nx, obs.ny)) * init.marq_lambda
	svd_tolerance = 1e-4

	stop_flag = np.zeros((obs.nx, obs.ny))

	Nw = len(init.wavelength)
	Npar = atmos.free_par

	ind_min = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[0]))
	ind_max = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[-1]))+1

	jacobian = np.zeros((obs.nx, obs.ny, Nw*4, Npar))
	JTJ = np.zeros((obs.nx, obs.ny, Npar, Npar))
	hessian = np.zeros((obs.nx, obs.ny, Npar, Npar))
	
	chi2 = np.zeros(init.max_iter)

	for i_ in range(init.max_iter):
		print("Iteration: {:2}\n".format(i_+1))
		
		# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
		rf, spec = globin.compute_rfs(init)

		diff = obs.spec[:,:,ind_min:ind_max] - spec[:,:,ind_min:ind_max,1:]

		chi2_old = np.sum(diff*diff, axis=(2,3))

		# Jacobian matrix
		for idx in range(obs.nx):
			for idy in range(obs.ny):
				if stop_flag[idx,idy]==0:
					for parID in range(Npar):
						jacobian[idx,idy,:,parID] = rf[idx,idy,parID,ind_min:ind_max].flatten()

		# is this correct? 
		jacobian_t = jacobian.T
		
		# loop for Marquardt lambda correction
		break_loop = False
		proposed_steps = np.zeros((obs.nx, obs.ny, Npar))
		for j_ in range(5):
			for idx in range(obs.nx):
				for idy in range(obs.ny):
					if stop_flag[idx,idy]==0:
						JTJ[idx,idy] = np.dot(jacobian_t[:,:,idx,idy], jacobian[idx,idy])
						hessian[idx,idy] = JTJ[idx,idy]# + marquardt_lambda[idx,idy] * np.diag(JTJ[idx,idy])
						diagonal_elements = np.diag(JTJ[idx,idy]) * (1 + marquardt_lambda[idx,idy])
						np.fill_diagonal(hessian[idx,idy], diagonal_elements)
						delta = np.dot(jacobian_t[:,:,idx,idy], diff[idx,idy].flatten())
						proposed_steps[idx,idy] = np.dot(np.linalg.inv(hessian[idx,idy]), delta)

			# print(np.log10(JTJ[0,0]))
			# print(np.log10(hessian[0,0]))
			# print(proposed_steps)

			# low_ind, up_ind = 0, 0
			# for parID in atmos.values:
			# 	low_ind += up_ind
			# 	up_ind += len(atmos.nodes[parID])
			# 	atmos.values[parID] += parameters[low_ind:up_ind]

			atmos.values["vz"] += proposed_steps
			atmos.build_from_nodes(init.ref_atm)
			
			new_spec = globin.compute_spectra(init, atmos)

			diff = obs.spec[:,:,ind_min:ind_max] - new_spec[:,:,ind_min:ind_max,1:]		
			chi2_new = np.sum(diff*diff, axis=(2,3))

			# print(chi2_new, chi2_old)

			for idx in range(obs.nx):
				for idy in range(obs.ny):
					if stop_flag[idx,idy]==0:
						if chi2_new[idx,idy] > chi2_old[idx,idy]:
							marquardt_lambda[idx,idy] *= 10
							atmos.values["vz"][idx,idy] -= proposed_steps[idx,idy]
							# low_ind, up_ind = 0, 0
							# for parID in atmos.values:
							# 	low_ind += up_ind
							# 	up_ind += len(atmos.nodes[parID])
							# 	atmos.values[parID] -= parameters[low_ind:up_ind]
						else:
							chi2[i_] = chi2_new[idx,idy]
							marquardt_lambda[idx,idy] /= 10
							break_loop = True
			# if we have changed Marquardt parameter, we go for new RF estimation
			if break_loop:
				break

		# if abs(chi2_new[idx,idy]-chi2_old[idx,idy])<1e-3:
		# 	stop_flag[idx,idy] = 1

		print(atmos.values["vz"])
		print(marquardt_lambda[0,0])

		# if i_%5==0:
		# 	plt.plot(init.wavelength, obs.spec[0,0,ind_min:ind_max,0])
		# 	plt.plot(init.wavelength, new_spec[0,0,ind_min:ind_max,1])
		# 	plt.show()


		if marquardt_lambda[0,0]>1e7 or marquardt_lambda<1e-5:
			break

		# if np.sum(stop_flag)==(obs.nx*obs.ny):
		# 	print("I am out!")
		# 	break

		print()

	plt.plot(init.wavelength-401.6, obs.spec[0,0,ind_min:ind_max,0])
	plt.plot(init.wavelength-401.6, new_spec[0,0,ind_min:ind_max,1])
	plt.xlim([-0.1, 0.1])
	plt.xlabel(r"$\Delta \lambda$ [nm]")
	plt.ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
	plt.show()

	#--- save inverted atmos
	# atmos.save_cube()