import matplotlib.pyplot as plt
import numpy as np

import globin

fact = {"temp"  : 1,
		"vz"    : 1,
		"vmic"  : 1,
		"mag"   : 1e4,
		"gamma" : 180/np.pi,
		"chi"   : 180/np.pi}

unit = {"temp"  : "K",
		"vz"    : "km/s",
		"vmic"  : "km/s",
		"mag"   : "G",
		"gamma" : "deg",
		"chi"   : "deg"}

def plot_atmosphere(atmos, parameters, idx=0, idy=0):
	logtau = atmos.logtau
	cube = atmos.data[idx,idy]

	n_plots = len(parameters)
	if n_plots==1:
		ncols = 1
		nrows = 1
	else:
		ncols = 2
		nrows = int(np.ceil(n_plots/ncols))

	for k_ in range(n_plots):	
		parID = atmos.par_id[parameters[k_]]

		# if parameters[k_]=="gamma":
		# 	cube[parID] = np.arccos(cube[parID])
		# elif parameters[k_]=="chi":
		# 	cube[parID] = np.arcsin(cube[parID])

		plt.subplot(nrows, ncols, k_+1)

		plt.plot(logtau, cube[parID]*fact[parameters[k_]])
		plt.xlabel(r"$\log \tau$")
		plt.ylabel(f"{globin.parameter_name[parameters[k_]]} [{unit[parameters[k_]]}]")

def plot_spectra(spec, idx=0, idy=0):
	lmin = np.min(spec.wavelength)
	lmax = np.max(spec.wavelength)
	dlam = (lmax - lmin) * 10

	# Stokes I
	plt.subplot(2,2,1)
	plt.title("Stokes I")
	plt.plot((spec.wavelength - lmin)*10, spec.spec[idx,idy,:,0])
	plt.ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
	plt.xlim([0, dlam])
	# Stokes Q
	plt.subplot(2,2,2)
	plt.title("Stokes Q")
	plt.plot((spec.wavelength - lmin)*10, spec.spec[idx,idy,:,1])
	plt.xlim([0, dlam])
	# Stokes U
	plt.subplot(2,2,3)
	plt.title("Stokes U")
	plt.plot((spec.wavelength - lmin)*10, spec.spec[idx,idy,:,2])
	plt.xlim([0, dlam])
	plt.xlabel(r"$\Delta \lambda$ [$\AA$]")
	plt.ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
	# Stokes V
	plt.subplot(2,2,4)
	plt.title("Stokes V")
	plt.plot((spec.wavelength - lmin)*10, spec.spec[idx,idy,:,3])
	plt.xlim([0, dlam])
	plt.xlabel(r"$\Delta \lambda$ [$\AA$]")

def plot_chi2(chi2, fpath="chi2.png", log_scale=False):
	fig = plt.figure(figsize=(12,10))
	
	if chi2.ndim==3:	
		nx, ny, niter = chi2.shape
		for idx in range(nx):
			for idy in range(ny):
				inds_non_zero = np.nonzero(chi2[idx,idy])[0]
				plt.plot(chi2[idx,idy,inds_non_zero], label=f"[{idx+1},{idy+1}]")
	elif chi2.ndim==1:
		niter = chi2.shape
		inds_non_zero = np.nonzero(chi2)[0]
		plt.plot(chi2[inds_non_zero])

	plt.xlabel("Iteration", fontsize=14)
	plt.ylabel(r"$\chi^2$", fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	if log_scale:
		plt.yscale("log")
	plt.legend(fontsize=12)
	plt.savefig(fpath)

def plot_rf(rf, fpath=None):
	"""
	rf.shape = (nx, ny, nz, npar, nw, 4) ???
	"""
	pass
