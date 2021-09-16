import matplotlib.pyplot as plt
import numpy as np
import copy
from astropy.io import fits
import sys

import globin

fact = {"temp"  : 1,
		"ne"    : 1,
		"vz"    : 1,
		"vmic"  : 1,
		"mag"   : 1,
		"gamma" : 180/np.pi,
		"chi"   : 180/np.pi}

unit = {"temp"  : "K",
		"ne"    : "1/3",
		"vz"    : "km/s",
		"vmic"  : "km/s",
		"mag"   : "G",
		"gamma" : "deg",
		"chi"   : "deg"}

def plot_atmosphere(atmos, parameters, idx=0, idy=0, ls="-", lw=2, color="tab:blue"):
	logtau = atmos.data[idx,idy,0]
	cube = atmos.data[idx,idy]

	n_plots = len(parameters)
	if n_plots==1:
		ncols = 1
		nrows = 1
	else:
		ncols = 2
		nrows = int(np.ceil(n_plots/ncols))

	for k_ in range(n_plots):	
		parameter = parameters[k_]
		parID = atmos.par_id[parameter]

		# if parameter=="gamma":
		# 	cube[parID] = np.arccos(cube[parID])
		# elif parameter=="chi":
		# 	cube[parID] = np.arcsin(cube[parID])

		plt.subplot(nrows, ncols, k_+1)

		try:
			x = atmos.nodes[parameter]
			y = atmos.values[parameter][idx,idy] * fact[parameter]
			plt.scatter(x, y, s=20, color=color)
		except:
			# print(f"globin::visualize --> no nodes for parameter {parameter}")
			pass

		plt.plot(logtau, cube[parID]*fact[parameter], ls=ls, lw=lw, color=color)
		if parameter=="ne":
			plt.yscale("log")
		plt.xlabel(r"$\log \tau$")
		plt.ylabel(f"{globin.parameter_name[parameter]} [{unit[parameter]}]")

def plot_spectra(obs, wavelength, inv=None, title=None):
	"""
	Plot spectra.

	Parameters
	----------
	obs : ndarray
		Observed spectrum. Assumed shape is (nw, 4).
	wavelength : ndarray
		Array of wavelngths in nm for which the obs is taken.
	inv : ndarray (optional)
		Inverted spectra. Used to compare with observed one.
		Same shape as 'obs'.
	title : string (optional)
		Figure title.
	"""
	lmin = np.min(wavelength)
	lmax = np.max(wavelength)
	dlam = (lmax - lmin) * 10

	# Icont = np.max(obs.spec[idx,idy,:,0])

	# obs.spec /= Icont
	# if inv is not None:
	# 	print("Normed inverted spec")
	# 	inv.spec /= Icont
	# for ids in range(1,4):
	# 	obs.spec[:,:,:,ids] *= 100
	# 	if inv is not None:	
	# 		inv.spec[:,:,:,ids] *= 100

	if title:
		fig.suptitle(title, fontsize=16)

	if inv is None:
		# Stokes I
		plt.subplot(2,2,1)
		plt.title("Stokes I")
		plt.plot((wavelength - lmin)*10, obs[:,0])
		plt.ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		plt.xlim([0, dlam])
		# Stokes Q
		plt.subplot(2,2,2)
		plt.title("Stokes Q")
		plt.plot((wavelength - lmin)*10, obs[:,1])
		plt.xlim([0, dlam])
		# Stokes U
		plt.subplot(2,2,3)
		plt.title("Stokes U")
		plt.plot((wavelength - lmin)*10, obs[:,2])
		plt.xlim([0, dlam])
		plt.xlabel(r"$\Delta \lambda$ [$\AA$]")
		plt.ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		# Stokes V
		plt.subplot(2,2,4)
		plt.title("Stokes V")
		plt.plot((wavelength - lmin)*10, obs[:,3])
		plt.xlim([0, dlam])
		plt.xlabel(r"$\Delta \lambda$ [$\AA$]")
	else:
		fig = plt.figure(figsize=(9,9), dpi=150)
		gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.35, hspace=0.5)
		
		#--- Stokes I
		gsSI = gs[0,0].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)

		ax0 = fig.add_subplot(gsSI[0,0])
		ax0.set_title("Stokes I")
		ax0.plot((wavelength - lmin)*10, obs[:,0], "k-", markersize=2)
		ax0.plot((wavelength - lmin)*10, inv[:,0], color="tab:red", lw=1.5)
		# ax0.set_ylabel(r"I [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0.set_ylabel("I")
		ax0.set_xticks([], [])
		ax0.set_xlim([0, dlam])

		ax1 = fig.add_subplot((gsSI[1,0]))
		difference = obs[:,0] - inv[:,0]
		# difference /= obs[:,0] / 100
		ax1.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1.plot((wavelength - lmin)*10, difference, color="tab:blue", lw=1.5)
		ax1.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1.set_ylabel(r"$\Delta I$")
		ax1.set_xlim([0, dlam])

		#--- Stokes Q
		gsSI = gs[0,1].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)

		ax0 = fig.add_subplot(gsSI[0,0])
		ax0.set_title("Stokes Q")
		ax0.plot((wavelength - lmin)*10, obs[:,1], "k-", markersize=2)
		ax0.plot((wavelength - lmin)*10, inv[:,1], color="tab:red", lw=1.5)
		# ax0.set_ylabel(r"Q [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0.set_ylabel("Q")
		ax0.set_xticks([], [])
		ax0.set_xlim([0, dlam])

		ax1 = fig.add_subplot((gsSI[1,0]))
		difference = obs[:,1] - inv[:,1]
		# difference /= obs[:,1] / 100
		ax1.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1.plot((wavelength - lmin)*10, difference, color="tab:blue", lw=1.5)
		ax1.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1.set_ylabel(r"$\Delta Q$")
		ax1.set_xlim([0, dlam])

		#--- Stokes U
		gsSI = gs[1,0].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)

		ax0 = fig.add_subplot(gsSI[0,0])
		ax0.set_title("Stokes U")
		ax0.plot((wavelength - lmin)*10, obs[:,2], "k-", markersize=2)
		ax0.plot((wavelength - lmin)*10, inv[:,2], color="tab:red", lw=1.5)
		# ax0.set_ylabel(r"U [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0.set_ylabel("U")
		ax0.set_xticks([], [])
		ax0.set_xlim([0, dlam])

		ax1 = fig.add_subplot((gsSI[1,0]))
		difference = obs[:,2] - inv[:,2]
		# difference /= obs[:,2] / 100
		ax1.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1.plot((wavelength - lmin)*10, difference, color="tab:blue", lw=1.5)
		ax1.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1.set_ylabel(r"$\Delta U$")
		ax1.set_xlim([0, dlam])
		
		#--- Stokes V
		gsSI = gs[1,1].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)

		ax0 = fig.add_subplot(gsSI[0,0])
		ax0.set_title("Stokes V")
		ax0.plot((wavelength - lmin)*10, obs[:,3], "k-", markersize=2)
		ax0.plot((wavelength - lmin)*10, inv[:,3], color="tab:red", lw=1.5)
		# ax0.set_ylabel(r"V [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0.set_ylabel("V")
		ax0.set_xticks([], [])
		ax0.set_xlim([0, dlam])

		ax1 = fig.add_subplot((gsSI[1,0]))
		difference = obs[:,3] - inv[:,3]
		# difference /= obs[ :, 3] / 100
		ax1.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1.plot((wavelength - lmin)*10, difference, color="tab:blue", lw=1.5)
		ax1.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1.set_ylabel(r"$\Delta V$")
		ax1.set_xlim([0, dlam])

def plot_chi2(chi2, fpath="chi2.png", log_scale=False):
	fig = plt.figure(figsize=(12,10))
	
	nx, ny, niter = chi2.shape
	for idx in range(nx):
		for idy in range(ny):
			inds_non_zero = np.nonzero(chi2[idx,idy])[0]
			x = np.arange(inds_non_zero[-1]+1) + 1
			plt.plot(x, chi2[idx,idy,inds_non_zero], label=f"[{idx+1},{idy+1}]")

	plt.xlabel("Iteration", fontsize=14)
	plt.ylabel(r"$\chi^2$", fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	if log_scale:
		plt.yscale("log")
	plt.legend(fontsize=12)
	plt.savefig(fpath)
	plt.close()

def plot_rf(fpath=None, params=["temp", "mag", "vz"], idx=0, idy=0, Stokes="I", norm=True):
	cmap = {"temp"  : "YlOrRd", 
			"vmic"  : "bwr",
			"vz"    : "bwr", 
			"mag"   : "bwr", 
			"gamma" : "bwr",
			"chi"   : "bwr"}
	cbar_label = {"temp"   : "T [K]", 
				  "vmic"   : r"$v_\mathrm{mic}$",
				  "vz"     : r"$v_z$ [km/s]", 
				  "mag"    : "B [G]", 
				  "gamma"  : r"$\gamma$", 
				  "chi"    : r"$\chi$"}
	stokesV = {"I" : 0, "Q" : 1, "U" : 2, "V" : 3}

	if fpath is not None:
		hdulist = fits.open(fpath)
		print(repr(hdulist[0].header))
	else:
		sys.exit("No RF file to open.")

	logtau = hdulist["depths"].data
	wavs = hdulist["wavelength"].data
	lam_min, lam_max = wavs[0], wavs[-1]
	lam0 = (lam_max + lam_min)/2

	# rf.shape = (nx, ny, 6, nz, nw, 4)
	rf = hdulist["RF"].data
	nx, ny, npar, nz, nw, ns = rf.shape

	pars = [hdulist["RF"].header[f"PAR{i_+1}"] for i_ in range(npar)]
	parIDs = [hdulist["RF"].header[f"PARID{i_+1}"]-1 for i_ in range(npar)]
	pars = dict(zip(pars, parIDs))

	stokes_range = []
	stokes_labels = []
	for item in Stokes:
		item = item.upper()
		stokes_range.append(stokesV[item])
		stokes_labels.append(f"Stokes {item}")

	NZ = np.int((logtau[-1] - logtau[0]))

	nrows = len(params)
	ncols = len(stokes_range)
	NW = 11
	if ncols!=1:
		NW = 5

	xpos = np.linspace(0, nw, num=NW)
	xvals = np.round(np.linspace(lam_min-lam0, lam_max-lam0, num=NW), decimals=2)
	ypos = np.linspace(0, nz, num=NZ)
	yvals = np.round(np.linspace(logtau[0], logtau[-1], num=NZ), decimals=2)
	
	fig = plt.figure(figsize=(12,10), dpi=150)
	gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.35, hspace=0.5)

	for i_, parameter in enumerate(params):
		try:
			idp = pars[parameter]
		except:
			sys.exit(f"No RF for parameter {parameter}")

		for j_, ids in enumerate(stokes_range):
			ax = fig.add_subplot(gs[i_,j_])
			if i_==0:
				ax.set_title(stokes_labels[j_])
			matrix = rf[idx, idy, idp, :, :, ids]
			if norm:
				norm = np.sqrt(np.sum(matrix**2))
				matrix /= norm
			vmax = np.max(np.abs(matrix))
			vmin = -vmax
			par_cmap = cmap[parameter]
			if parameter=="temp":
				if ids!=0:
					par_cmap = "bwr"
				else:
					vmin = 0
			im = ax.imshow(matrix, aspect="auto", cmap=par_cmap, vmin=vmin, vmax=vmax)
			add_colorbar(fig, ax, im, label=cbar_label[parameter])
			ax.set_xticks(xpos)
			ax.set_xticklabels(xvals)
			ax.set_yticks(ypos)
			ax.set_yticklabels(yvals)
			ax.grid(b=True, which="major", axis="y", lw=0.5)

	plt.show()

def add_colorbar(fig, ax, im, label=None):
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	axins = inset_axes(ax,
	                   width="2%",
	                   height="100%",
	                   loc='lower left',
	                   bbox_to_anchor=(1.02, 0., 1, 1),
	                   bbox_transform=ax.transAxes,
	                   borderpad=0)
	cbar = fig.colorbar(im, cax=axins)
	if label is not None:
		cbar.set_label(label)