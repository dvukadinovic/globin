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

	print("Initial parameters: ", atmos.values)
	print()

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

			# print(proposed_steps)

			low_ind, up_ind = 0, 0
			for parID in atmos.values:
				low_ind += up_ind
				up_ind += len(atmos.nodes[parID])
				atmos.values[parID] += proposed_steps[:,:,low_ind:up_ind]
				atmos.check_parameters()
			print(atmos.values)

			atmos.build_from_nodes(init.ref_atm)
			new_spec = globin.compute_spectra(init, atmos)

			diff = obs.spec[:,:,ind_min:ind_max] - new_spec[:,:,ind_min:ind_max,1:]		
			chi2_new = np.sum(diff*diff, axis=(2,3))

			for idx in range(obs.nx):
				for idy in range(obs.ny):
					if stop_flag[idx,idy]==0:
						if chi2_new[idx,idy] > chi2_old[idx,idy]:
							marquardt_lambda[idx,idy] *= 10
							low_ind, up_ind = 0, 0
							for parID in atmos.values:
								low_ind += up_ind
								up_ind += len(atmos.nodes[parID])
								atmos.values[parID] -= proposed_steps[:,:,low_ind:up_ind]
								atmos.check_parameters()
						else:
							chi2[i_] = chi2_new[idx,idy]
							marquardt_lambda[idx,idy] /= 10
							break_loop = True
			# if we have changed Marquardt parameter, we go for new RF estimation
			if break_loop:
				break

		# if abs(chi2_new[idx,idy]-chi2_old[idx,idy])<1e-3:
		# 	stop_flag[idx,idy] = 1

		print(atmos.values)
		print(marquardt_lambda[0,0])

		if marquardt_lambda[0,0]<=1e-5:
			marquardt_lambda[0,0] = 1e-5

		if marquardt_lambda[0,0]>=1e8:
			break

		# if np.sum(stop_flag)==(obs.nx*obs.ny):
		# 	print("I am out!")
		# 	break

		print("--------------------------------------------------\n")

	fix, axs = plt.subplots(nrows=2, ncols=2)

	for i in range(atmos.nx):
		for j in range(atmos.ny):
			# Stokes I
			axs[0,0].set_title("Stokes I")
			axs[0,0].plot(obs.data[0,0,:,0] - 401.6, obs.spec[0,0,:,0])
			axs[0,0].plot(new_spec[i,j,:,0] - 401.6, new_spec[i,j,:,1])
			# Stokes Q
			axs[0,1].set_title("Stokes Q")
			axs[0,1].plot(obs.data[0,0,:,0] - 401.6, obs.spec[0,0,:,1])
			axs[0,1].plot(new_spec[i,j,:,0] - 401.6, new_spec[i,j,:,2])
			# Stokes U
			axs[1,0].set_title("Stokes U")
			axs[1,0].plot(obs.data[0,0,:,0] - 401.6, obs.spec[0,0,:,2])
			axs[1,0].plot(new_spec[i,j,:,0] - 401.6, new_spec[i,j,:,3])
			# Stokes V
			axs[1,1].set_title("Stokes V")
			axs[1,1].plot(obs.data[0,0,:,0] - 401.6, obs.spec[0,0,:,3])
			axs[1,1].plot(new_spec[i,j,:,0] - 401.6, new_spec[i,j,:,4])

	axs[1,0].set_xlabel(r"$\Delta \lambda$ [nm]")
	axs[1,1].set_xlabel(r"$\Delta \lambda$ [nm]")
	axs[0,0].set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
	axs[1,0].set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")

	axs[0,0].set_xlim([-0.1, 0.1])
	axs[0,1].set_xlim([-0.1, 0.1])
	axs[1,0].set_xlim([-0.1, 0.1])
	axs[1,1].set_xlim([-0.1, 0.1])

	plt.show()

	idx, idy = 0,0
	parameter = "vz"
	x = atmos.nodes[parameter]
	y = atmos.values[parameter][idx,idy]
	if parameter=="temp":
		Kn = splev(x[-1], globin.temp_tck, der=1)
	else:
		Kn = 0
	y_new = globin.tools.bezier_spline(x, y, atmos.logtau, Kn=Kn, degree=globin.interp_degree)

	plt.plot(x, y, "ro")
	plt.plot(atmos.logtau, y_new, color="tab:blue")
	plt.plot([min(atmos.logtau),max(atmos.logtau)], [0,0], "k-")
	plt.xlabel(r"$\log \tau$")
	plt.ylabel(r"$v_z$ [km/s]")
	plt.show()

	#--- save inverted atmos
	# atmos.save_cube()