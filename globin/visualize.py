import matplotlib.pyplot as plt
import numpy as np

def plot_atmosphere(atmos, idx=0, idy=0):
	logtau = atmos.logtau
	cube = atmos.data[idx,idy]

	nrows, ncols = 4, 2

	plt.subplot(nrows, ncols, 1)
	plt.plot(logtau, cube[1])
	plt.ylabel("T [K]")

	plt.subplot(nrows, ncols, 2)
	plt.plot(logtau, np.log10(cube[2]))
	plt.ylabel(r"$\log (n_e)$ [m$^{-3}$]")
	
	plt.subplot(nrows, ncols, 3)
	plt.plot(logtau, cube[3])
	plt.ylabel("$v_z$ [km/s]")

	plt.subplot(nrows, ncols, 4)
	plt.plot(logtau, cube[5]*1e4)
	plt.ylabel("B [G]")

	plt.subplot(nrows, ncols, 5)
	plt.plot(logtau, cube[6] * 180/np.pi)
	plt.ylabel(r"$\gamma$ [$^\circ$]")
	plt.xlabel(r"$\log \tau$")

	plt.subplot(nrows, ncols, 6)
	plt.plot(logtau, cube[7] * 180/np.pi)
	plt.ylabel(r"$\chi$ [$^\circ$]")
	plt.xlabel(r"$\log \tau$")

	plt.subplot(nrows, ncols, 7)
	plt.plot(logtau, cube[4])
	plt.ylabel(r"$v_{mic}$ [km/s]")
	plt.xlabel(r"$\log \tau$")

	# plt.savefig("atmosphere_from_nodes.png")

def plot_spectra(inv, axs, idx=0, idy=0):

	# Stokes I
	axs[0,0].set_title("Stokes I")
	# axs[0,0].plot((obs.data[idx,idy,:,0] - 401.6)*10, obs.spec[idx,idy,:,0])
	axs[0,0].plot((inv.data[idx,idy,:,0] - 401.6)*10, inv.spec[idx,idy,:,0])
	# Stokes Q
	axs[0,1].set_title("Stokes Q")
	# axs[0,1].plot((obs.data[idx,idy,:,0] - 401.6)*10, obs.spec[idx,idy,:,1])
	axs[0,1].plot((inv.data[idx,idy,:,0] - 401.6)*10, inv.spec[idx,idy,:,1])
	# Stokes U
	axs[1,0].set_title("Stokes U")
	# axs[1,0].plot((obs.data[idx,idy,:,0] - 401.6)*10, obs.spec[idx,idy,:,2])
	axs[1,0].plot((inv.data[idx,idy,:,0] - 401.6)*10, inv.spec[idx,idy,:,2])
	# Stokes V
	axs[1,1].set_title("Stokes V")
	# axs[1,1].plot((obs.data[idx,idy,:,0] - 401.6)*10, obs.spec[idx,idy,:,3])
	axs[1,1].plot((inv.data[idx,idy,:,0] - 401.6)*10, inv.spec[idx,idy,:,3])

	axs[1,0].set_xlabel(r"$\Delta \lambda$ [$\AA$]")
	axs[1,1].set_xlabel(r"$\Delta \lambda$ [$\AA$]")
	axs[0,0].set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
	axs[1,0].set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")

	axs[0,0].set_xlim([-1, 1])
	axs[0,1].set_xlim([-1, 1])
	axs[1,0].set_xlim([-1, 1])
	axs[1,1].set_xlim([-1, 1])
	
	# plt.savefig("{:s}/stokes_vector_n{:1d}.png".format(fname,noise))

def show():	
	plt.show()
