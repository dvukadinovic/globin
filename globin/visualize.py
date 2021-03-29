import matplotlib.pyplot as plt
import numpy as np
import  copy
from astropy.io import fits
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

def plot_atmosphere(atmos, parameters, idx=0, idy=0, ls="-", lw=1, color="tab:blue"):
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
		parID = atmos.par_id[parameters[k_]]

		# if parameters[k_]=="gamma":
		# 	cube[parID] = np.arccos(cube[parID])
		# elif parameters[k_]=="chi":
		# 	cube[parID] = np.arcsin(cube[parID])

		plt.subplot(nrows, ncols, k_+1)

		plt.plot(logtau, cube[parID]*fact[parameters[k_]], ls=ls, lw=lw, color=color)
		plt.xlabel(r"$\log \tau$")
		plt.ylabel(f"{globin.parameter_name[parameters[k_]]} [{unit[parameters[k_]]}]")

def plot_spectra(obs, idx=0, idy=0, inv=None, title=None):
	obs = copy.deepcopy(obs)

	lmin = np.min(obs.wavelength)
	lmax = np.max(obs.wavelength)
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
		plt.plot((obs.wavelength - lmin)*10, obs.spec[idx,idy,:,0])
		plt.ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		plt.xlim([0, dlam])
		# Stokes Q
		plt.subplot(2,2,2)
		plt.title("Stokes Q")
		plt.plot((obs.wavelength - lmin)*10, obs.spec[idx,idy,:,1])
		plt.xlim([0, dlam])
		# Stokes U
		plt.subplot(2,2,3)
		plt.title("Stokes U")
		plt.plot((obs.wavelength - lmin)*10, obs.spec[idx,idy,:,2])
		plt.xlim([0, dlam])
		plt.xlabel(r"$\Delta \lambda$ [$\AA$]")
		plt.ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		# Stokes V
		plt.subplot(2,2,4)
		plt.title("Stokes V")
		plt.plot((obs.wavelength - lmin)*10, obs.spec[idx,idy,:,3])
		plt.xlim([0, dlam])
		plt.xlabel(r"$\Delta \lambda$ [$\AA$]")
	else:
		fig = plt.figure(figsize=(9,9), dpi=150)
		gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.35, hspace=0.5)
		
		#--- Stokes I
		gsSI = gs[0,0].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)

		ax0 = fig.add_subplot(gsSI[0,0])
		ax0.set_title("Stokes I")
		ax0.plot((obs.wavelength - lmin)*10, obs.spec[idx,idy,:,0], "ko", markersize=2)
		ax0.plot((inv.wavelength - lmin)*10, inv.spec[idx,idy,:,0], color="tab:red", lw=0.75)
		# ax0.set_ylabel(r"I [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0.set_ylabel("I")
		ax0.set_xticks([], [])
		ax0.set_xlim([0, dlam])

		ax1 = fig.add_subplot((gsSI[1,0]))
		difference = obs.spec[idx, idy, :, 0] - inv.spec[idx,idy,:,0]
		# difference /= obs.spec[idx,idy, :, 0] / 100
		ax1.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1.plot((inv.wavelength - lmin)*10, difference, color="tab:blue", lw=0.75)
		ax1.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1.set_ylabel(r"$\Delta I$")
		ax1.set_xlim([0, dlam])

		#--- Stokes Q
		gsSI = gs[0,1].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)

		ax0 = fig.add_subplot(gsSI[0,0])
		ax0.set_title("Stokes Q")
		ax0.plot((obs.wavelength - lmin)*10, obs.spec[idx,idy,:,1], "ko", markersize=2)
		ax0.plot((inv.wavelength - lmin)*10, inv.spec[idx,idy,:,1], color="tab:red", lw=0.75)
		# ax0.set_ylabel(r"Q [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0.set_ylabel("Q")
		ax0.set_xticks([], [])
		ax0.set_xlim([0, dlam])

		ax1 = fig.add_subplot((gsSI[1,0]))
		difference = obs.spec[idx, idy, :, 1] - inv.spec[idx,idy,:,1]
		# difference /= obs.spec[idx,idy, :, 1] / 100
		ax1.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1.plot((inv.wavelength - lmin)*10, difference, color="tab:blue", lw=0.75)
		ax1.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1.set_ylabel(r"$\Delta Q$")
		ax1.set_xlim([0, dlam])

		#--- Stokes U
		gsSI = gs[1,0].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)

		ax0 = fig.add_subplot(gsSI[0,0])
		ax0.set_title("Stokes U")
		ax0.plot((obs.wavelength - lmin)*10, obs.spec[idx,idy,:,2], "ko", markersize=2)
		ax0.plot((inv.wavelength - lmin)*10, inv.spec[idx,idy,:,2], color="tab:red", lw=0.75)
		# ax0.set_ylabel(r"U [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0.set_ylabel("U")
		ax0.set_xticks([], [])
		ax0.set_xlim([0, dlam])

		ax1 = fig.add_subplot((gsSI[1,0]))
		difference = obs.spec[idx, idy, :, 2] - inv.spec[idx,idy,:,2]
		# difference /= obs.spec[idx,idy, :, 2] / 100
		ax1.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1.plot((inv.wavelength - lmin)*10, difference, color="tab:blue", lw=0.75)
		ax1.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1.set_ylabel(r"$\Delta U$")
		ax1.set_xlim([0, dlam])
		
		#--- Stokes V
		gsSI = gs[1,1].subgridspec(nrows=2, ncols=1, height_ratios=[4,1], hspace=0)

		ax0 = fig.add_subplot(gsSI[0,0])
		ax0.set_title("Stokes V")
		ax0.plot((obs.wavelength - lmin)*10, obs.spec[idx,idy,:,3], "ko", markersize=2)
		ax0.plot((inv.wavelength - lmin)*10, inv.spec[idx,idy,:,3], color="tab:red", lw=0.75)
		# ax0.set_ylabel(r"V [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0.set_ylabel("V")
		ax0.set_xticks([], [])
		ax0.set_xlim([0, dlam])

		ax1 = fig.add_subplot((gsSI[1,0]))
		difference = obs.spec[idx, idy, :, 3] - inv.spec[idx,idy,:,3]
		# difference /= obs.spec[idx,idy, :, 3] / 100
		ax1.plot([0, dlam], [0,0], color="k", lw=0.5)
		ax1.plot((inv.wavelength - lmin)*10, difference, color="tab:blue", lw=0.75)
		ax1.set_xlabel(r"$\Delta \lambda$ [$\AA$]")
		ax1.set_ylabel(r"$\Delta V$")
		ax1.set_xlim([0, dlam])

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
	plt.close()

def plot_rf(rf=None, fpath=None):
	"""
	rf.shape = (nx, ny, 6, nz, nw, 4))
	"""
	if fpath is not None:
		rf = fits.open(fpath)[0].data
	else:
		if rf is None:
			print("No data to plot! Provide path to RF or give me the cube!")
			sys.exit()

	idx, idy = 0,0
	idp = 0

	xpos = np.linspace(0, 200, num=11)
	xvals = np.round(np.linspace(-1, 1, num=11), decimals=1)

	ypos = np.linspace(0, 70, num=6)
	yvals = np.round(np.linspace(-5, 1, num=6), decimals=1)

	aux = np.sum(rf[idx, idy, idp, :, :, 0], axis=1)
	norm = np.sqrt(np.sum(aux**2))
	plt.plot(np.linspace(-6, 1, num=71), aux/norm)
	plt.show()
	return 

	fig = plt.figure(figsize=(9,9), dpi=150)
	gs = fig.add_gridspec(nrows=4, ncols=1, wspace=0.35, hspace=0.5)

	ax = fig.add_subplot(gs[0,0])
	matrix = rf[idx, idy, idp, :, :, 0]
	norm = np.sqrt(np.sum(matrix**2))
	matrix /= norm
	vmax = np.max(np.abs(matrix))
	im = ax.imshow(rf[idx, idy, idp, :, :, 0], aspect="auto", cmap="OrRd", vmin=0, vmax=vmax)
	add_colorbar(fig, ax, im)
	ax.set_xticks(xpos)
	ax.set_xticklabels(xvals)
	ax.set_yticks(ypos)
	ax.set_yticklabels(yvals)
	ax.grid(b=True, which="major", axis="y", lw=0.5)

	ax = fig.add_subplot(gs[1,0])
	matrix = rf[idx, idy, idp, :, :, 1]
	norm = np.sqrt(np.sum(matrix**2))
	matrix /= norm
	vmax = np.max(np.abs(matrix))
	im = ax.imshow(matrix, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
	add_colorbar(fig, ax, im)
	ax.set_xticks(xpos)
	ax.set_xticklabels(xvals)
	ax.set_yticks(ypos)
	ax.set_yticklabels(yvals)
	ax.grid(b=True, which="major", axis="y", lw=0.5)

	ax = fig.add_subplot(gs[2,0])
	matrix = rf[idx, idy, idp, :, :, 2]
	norm = np.sqrt(np.sum(matrix**2))
	matrix /= norm
	vmax = np.max(np.abs(matrix))
	im = ax.imshow(matrix, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
	add_colorbar(fig, ax, im)
	ax.set_xticks(xpos)
	ax.set_xticklabels(xvals)
	ax.set_yticks(ypos)
	ax.set_yticklabels(yvals)
	ax.grid(b=True, which="major", axis="y", lw=0.5)

	ax = fig.add_subplot(gs[3,0])
	matrix = rf[idx, idy, idp, :, :, 3]
	norm = np.sqrt(np.sum(matrix**2))
	matrix /= norm
	vmax = np.max(np.abs(matrix))
	im = ax.imshow(matrix, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
	add_colorbar(fig, ax, im)
	ax.set_xticks(xpos)
	ax.set_xticklabels(xvals)
	ax.set_yticks(ypos)
	ax.set_yticklabels(yvals)
	ax.grid(b=True, which="major", axis="y", lw=0.5)

	plt.show()

def add_colorbar(fig, ax, im, label=None):
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