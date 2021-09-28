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

def plot_spectra(obs, wavelength, inv=None, axes=None, norm=False, color="tab:blue", lw=1, title=None, subtitles=False):
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

	if norm:
		coeff = (obs[0,0] - obs[-1,0]) / (wavelength[0] - wavelength[-1])
		n = obs[0,0] - coeff * wavelength[0]
		Icont = coeff * wavelength + n
		for ids in range(4):
			obs[:,ids] /= Icont

	if title:
		fig.suptitle(title)

	if inv is None:
		if axes is None:
			fig = plt.figure(figsize=(8,6))
			gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.3, hspace=0.3)
			axI = fig.add_subplot(gs[0,0])
			axQ = fig.add_subplot(gs[0,1])
			axU = fig.add_subplot(gs[1,0])
			axV = fig.add_subplot(gs[1,1])
		else:
			axI, axQ, axU, axV = axes

		#--- Stokes I
		if subtitles:
			axI.set_title("Stokes I")
			axQ.set_title("Stokes Q")
			axU.set_title("Stokes U")
			axV.set_title("Stokes V")
		
		axI.plot((wavelength - lmin)*10, obs[:,0], lw=lw, color=color)
		axI.set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		axI.set_xlim([0, dlam])

		#--- Stokes Q
		axQ.plot((wavelength - lmin)*10, obs[:,1]*100, lw=lw, color=color)
		axQ.set_ylabel(r"Stokes Q/I$_\mathrm{c}$ [%]")
		axQ.set_xlim([0, dlam])
		
		#--- Stokes U
		axU.plot((wavelength - lmin)*10, obs[:,2]*100, lw=lw, color=color)
		axU.set_xlim([0, dlam])
		axU.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		axU.set_ylabel(r"Stokes U/I$_\mathrm{c}$ [%]")
		
		#--- Stokes V
		axV.plot((wavelength - lmin)*10, obs[:,3]*100, lw=lw, color=color)
		axV.set_xlim([0, dlam])
		axV.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		axV.set_ylabel(r"Stokes V/I$_\mathrm{c}$ [%]")

		return axI, axQ, axU, axV
	else:
		if axes is None:
			fig = plt.figure(figsize=(9,9))
			gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.3, hspace=0.3)
			gsSI = gs[0,0].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)
			ax0_SI = fig.add_subplot(gsSI[0,0])
			ax1_SI = fig.add_subplot((gsSI[1,0]))
			gsSQ = gs[0,1].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)
			ax0_SQ = fig.add_subplot(gsSQ[0,0])
			ax1_SQ = fig.add_subplot((gsSQ[1,0]))
			gsSU = gs[1,0].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)
			ax0_SU = fig.add_subplot(gsSU[0,0])
			ax1_SU = fig.add_subplot((gsSU[1,0]))
			gsSV = gs[1,1].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)
			ax0_SV = fig.add_subplot(gsSV[0,0])
			ax1_SV = fig.add_subplot((gsSV[1,0]))
		else:
			ax0_SI, ax1_SI, ax0_SQ, ax1_SQ, ax0_SU, ax1_SU, ax0_SV, ax1_SV = axes	
		
		#--- Stokes I
		ax0_SI.set_title("Stokes I")
		ax0_SI.plot((wavelength - lmin)*10, obs[:,0], "k-", markersize=2)
		ax0_SI.plot((wavelength - lmin)*10, inv[:,0], color="tab:red", lw=1.5)
		# ax0_SI.set_ylabel(r"I [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0_SI.set_ylabel("I")
		ax0_SI.set_xticks([], [])
		ax0_SI.set_xlim([0, dlam])

		difference = obs[:,0] - inv[:,0]
		# difference /= obs[:,0] / 100
		ax1_SI.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1_SI.plot((wavelength - lmin)*10, difference, color="tab:blue", lw=1.5)
		ax1_SI.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1_SI.set_ylabel(r"$\Delta I$")
		ax1_SI.set_xlim([0, dlam])

		#--- Stokes Q
		ax0_SQ.set_title("Stokes Q")
		ax0_SQ.plot((wavelength - lmin)*10, obs[:,1], "k-", markersize=2)
		ax0_SQ.plot((wavelength - lmin)*10, inv[:,1], color="tab:red", lw=1.5)
		# ax0_SQ.set_ylabel(r"Q [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0_SQ.set_ylabel("Q")
		ax0_SQ.set_xticks([], [])
		ax0_SQ.set_xlim([0, dlam])

		difference = obs[:,1] - inv[:,1]
		# difference /= obs[:,1] / 100
		ax1_SQ.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1_SQ.plot((wavelength - lmin)*10, difference, color="tab:blue", lw=1.5)
		ax1_SQ.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1_SQ.set_ylabel(r"$\Delta Q$")
		ax1_SQ.set_xlim([0, dlam])

		#--- Stokes U
		ax0_SU.set_title("Stokes U")
		ax0_SU.plot((wavelength - lmin)*10, obs[:,2], "k-", markersize=2)
		ax0_SU.plot((wavelength - lmin)*10, inv[:,2], color="tab:red", lw=1.5)
		# ax0_SU.set_ylabel(r"U [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0_SU.set_ylabel("U")
		ax0_SU.set_xticks([], [])
		ax0_SU.set_xlim([0, dlam])

		difference = obs[:,2] - inv[:,2]
		# difference /= obs[:,2] / 100
		ax1_SU.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1_SU.plot((wavelength - lmin)*10, difference, color="tab:blue", lw=1.5)
		ax1_SU.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1_SU.set_ylabel(r"$\Delta U$")
		ax1_SU.set_xlim([0, dlam])
		
		#--- Stokes V
		ax0_SV.set_title("Stokes V")
		ax0_SV.plot((wavelength - lmin)*10, obs[:,3], "k-", markersize=2)
		ax0_SV.plot((wavelength - lmin)*10, inv[:,3], color="tab:red", lw=1.5)
		# ax0_SV.set_ylabel(r"V [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0_SV.set_ylabel("V")
		ax0_SV.set_xticks([], [])
		ax0_SV.set_xlim([0, dlam])

		difference = obs[:,3] - inv[:,3]
		# difference /= obs[ :, 3] / 100
		ax1_SV.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1_SV.plot((wavelength - lmin)*10, difference, color="tab:blue", lw=1.5)
		ax1_SV.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1_SV.set_ylabel(r"$\Delta V$")
		ax1_SV.set_xlim([0, dlam])

		return ax0_SI, ax1_SI, ax0_SQ, ax1_SQ, ax0_SU, ax1_SU, ax0_SV, ax1_SV

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

def plot_rf(fpath=None, params=["temp", "mag", "vz"], idx=0, idy=0, logtau_top=-5, logtau_bot=1, Stokes="I", norm=True):
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
	else:
		sys.exit("No RF file to open.")

	logtau = hdulist["depths"].data
	wavs = hdulist["wavelength"].data
	lam_min, lam_max = wavs[0], wavs[-1]
	lam0 = (lam_max + lam_min)/2

	ind_top = np.argmin(np.abs(logtau - logtau_top))
	ind_bot = np.argmin(np.abs(logtau - logtau_bot))+1

	# rf.shape = (nx, ny, 6, nz, nw, 4)
	rf = hdulist["RF"].data[:,:,:, ind_top:ind_bot, :, :]
	nx, ny, npar, nz, nw, ns = rf.shape
	logtau = logtau[ind_top:ind_bot]

	pars = [hdulist["RF"].header[f"PAR{i_+1}"] for i_ in range(npar)]
	parIDs = [hdulist["RF"].header[f"PARID{i_+1}"]-1 for i_ in range(npar)]
	pars = dict(zip(pars, parIDs))

	stokes_range = []
	stokes_labels = []
	for item in Stokes:
		item = item.upper()
		stokes_range.append(stokesV[item])
		stokes_labels.append(f"Stokes {item}")

	NZ = int(np.int((logtau[-1] - logtau[0])) / 0.5) + 1

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
			print(repr(hdulist[0].header))
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