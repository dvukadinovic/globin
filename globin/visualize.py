import matplotlib.pyplot as plt
import numpy as np

def plot_atmosphere(atmos, idx=0, idy=0):
	logtau = atmos.logtau
	cube = atmos.data[idx,idy]

	fig = plt.figure(figsize=(12,14))

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

def show():	
	plt.show()
