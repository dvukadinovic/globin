import matplotlib.pyplot as plt
import numpy as np

import globin

def plot_atmosphere(atmos, idx=0, idy=0):
	logtau = atmos.logtau
	cube = atmos.data[idx,idy]

	fig = plt.figure(figsize=(12,10))

	plt.subplot(3,2,1)
	plt.plot(logtau, cube[1])
	plt.ylabel("T [K]")

	plt.subplot(3,2,2)
	plt.plot(logtau, np.log10(cube[2]))
	plt.ylabel(r"$\log (n_e)$ [cm$^{-3}$]")
	
	plt.subplot(3,2,3)
	plt.plot(logtau, cube[3])
	plt.ylabel("$v_z$ [km/s]")

	plt.subplot(3,2,4)
	plt.plot(logtau, cube[5])
	plt.ylabel("B [T]")

	plt.subplot(3,2,5)
	plt.plot(logtau, cube[6])
	plt.ylabel(r"$\gamma$ [rad]")

	plt.subplot(3,2,6)
	plt.plot(logtau, cube[7])
	plt.ylabel(r"$\chi$ [rad]")
	
	plt.show()
