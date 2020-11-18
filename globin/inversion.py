import globin
import numpy as np

def invert(init):
	"""
	As input we expect all data to be present :)

	Parameters:
	---------------
	init : InputData
		InputData object in which we have everything stored.
	"""
	import matplotlib.pyplot as plt

	obs = init.obs
	atmos = init.atm

	# parameters = np.array([])
	# for parameter in atmos.values:
	# 	parameters = np.append(parameters, atmos.values)
	parameters = atmos.values["vz"]

	marquardt_lambda = np.ones((obs.nx, obs.ny)) * init.marq_lambda
	svd_tolerance = 1e-4

	stop_flag = np.zeros((obs.nx, obs.ny))

	Nw = len(init.wavelength)
	Npar = atmos.free_par

	ind_min = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[0]))
	ind_max = np.argmin(abs(obs.data[0,0,:,0] - init.wavelength[-1]))

	jacobian = np.zeros((obs.nx, obs.ny, Nw*4, Npar))
	JTJ = np.zeros((obs.nx, obs.ny, Npar, Npar))
	hessian = np.zeros((obs.nx, obs.ny, Npar, Npar))
	
	chi2 = np.zeros(30)

	for i_ in range(30):
		print("Iteration: {:2}".format(i_+1))
		
		# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
		rf, synth = globin.atmos.compute_rfs(init)

		print(rf.shape)
		
		# if i_==0:
		# 	plt.plot(init.wavelength, synth[0,0,ind_min:ind_max,1])

		aux = rf[0,0,:,:,0]

		plt.plot(synth[0,0,:,0], aux[0])
		# plt.plot(synth[0,0,:,0], aux[1])
		# plt.plot(synth[0,0,:,0], aux[2])
		plt.show()
		break

		diff = obs.spec[:,:,ind_min:ind_max] - synth[:,:,ind_min:ind_max,1:]

		chi2_old = np.sum(diff*diff, axis=(2,3))

		# Jacobian matrix
		for idx in range(obs.nx):
			for idy in range(obs.ny):
				if stop_flag[idx,idy]==0:
					for parID in range(Npar):
						jacobian[idx,idy,:,parID] = rf[idx,idy,parID,ind_min:ind_max].flatten()

		jacobian_t = jacobian.T
		
		# loop for Marquardt lambda correction
		break_loop = False
		for j_ in range(5):
			proposed_steps = np.zeros((obs.nx, obs.ny, Npar))
			for idx in range(obs.nx):
				for idy in range(obs.ny):
					if stop_flag[idx,idy]==0:
						JTJ[idx,idy] = np.dot(jacobian_t[:,:,idx,idy], jacobian[idx,idy])
						hessian[idx,idy] = JTJ[idx,idy] + marquardt_lambda[idx,idy] * np.diag(JTJ[idx,idy])
						# min_hes = np.max(hessian[idx,idy]) * svd_tolerance

						delta = np.dot(jacobian_t[:,:,idx,idy], diff[idx,idy].flatten())
						proposed_steps[idx,idy] = np.dot(np.linalg.inv(hessian[idx,idy]), delta)

			# print(np.max(hessian))
			# print(hessian)
			proposed_steps = np.array(proposed_steps)
			print(proposed_steps)
			
			atmos.values["vz"] += proposed_steps
			# parameters += proposed_steps
			atmos.build_from_nodes(init.ref_atm)
			
			new_spec = globin.atmos.compute_spectra(init, atmos)

			diff = obs.spec[:,:,ind_min:ind_max] - new_spec[:,:,ind_min:ind_max,1:]		
			chi2_new = np.sum(diff*diff, axis=(2,3))

			for idx in range(obs.nx):
				for idy in range(obs.ny):
					if stop_flag[idx,idy]==0:
						if chi2_new[idx,idy] > chi2_old[idx,idy]:
							marquardt_lambda[idx,idy] *= 10
							atmos.values["vz"][idx,idy] -= proposed_steps[idx,idy]
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
		print(marquardt_lambda)

		if marquardt_lambda[0,0]>1e7 or marquardt_lambda<1e-5:
			break

		# if np.sum(stop_flag)==(obs.nx*obs.ny):
		# 	print("I am out!")
		# 	break

		print()

	plt.plot(obs.data[0,0,ind_min:ind_max,0] - 401.6, obs.spec[0,0,ind_min:ind_max,0], label="observations")
	plt.plot(new_spec[0,0,ind_min:ind_max,0] - 401.6, new_spec[0,0,ind_min:ind_max,1], label="inverted")
	plt.xlabel(r"$\Delta \lambda$ [nm]")
	plt.ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
	plt.legend()
	# plt.savefig("obs_vs_inverted.png")
	plt.show()

	# plt.plot(np.log10(chi2))
	# plt.show()

	#--- save inverted atmos
	# atmos.save_cube()