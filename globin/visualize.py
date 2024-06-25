import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import copy
from astropy.io import fits
import sys

fact = {"temp"  : 1,
		"ne"    : 1,
		"vz"    : 1,
		"vmic"  : 1,
		"mag"   : 1,
		"gamma" : 180/np.pi,
		"chi"   : 180/np.pi,
		"nH"    : 1}

unit = {"temp"  : "K",
		"ne"    : r"$1/m^3$",
		"vz"    : "km/s",
		"vmic"  : "km/s",
		"mag"   : "G",
		"gamma" : "deg",
		"chi"   : "deg",
		"nH"    : r"$1/m^3$"}

pars_symbol = {"temp"  : "T",
			   "ne"	   : r"$n_\mathrm{e}$",
			   "vz"    : r"$v_\mathrm{LOS}$",
			   "vmic"  : r"$v_\mathrm{mic}$",
			   "mag"   : "B",
			   "gamma" : r"$\gamma$",
			   "chi"   : r"$\phi$",
			   "nH"    : r"$n_\mathrm{H}^0$"}

parameter_norm = {"temp"  : 5000,		# [K]
				  "vz" 	  : 6,			# [km/s]
				  "vmic"  : 6,			# [km/s]
				  "mag"   : 1000,		# [G]
				  "gamma" : np.pi,		# [rad]
				  "chi"   : np.pi,		# [rad]
				  "of"    : 2,			#
				  "stray" : 0.1,
				  "vmac"  : 2,			# [km/s]
				  "dlam"  : 10,			# [mA]
				  "loggf" : -1.0}		#

def show():
	"""
	Just to show up the plot from 'plot_atmosphere' and/or
	'plot_spectra' withouth the need to import matplotlib.
	"""
	plt.show()

def plot_atmosphere(atmos, parameters, idx=0, idy=0, ls="-", lw=2, color="tab:red", show_errors=False, labels=None, reference=None):
	colors = ["tab:red", "tab:green", "tab:orange"]
	Ncolors = len(colors)

	logtau = atmos.logtau
	cube = atmos.data[idx,idy]

	n_plots = len(parameters)
	if n_plots==1:
		ncols = 1
		nrows = 1
	else:
		ncols = 3
		nrows = int(np.ceil(n_plots/ncols))

	width, height = 3, 2 + 2/3
	fig = plt.figure(figsize=(width*ncols, height*nrows))
	gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

	is_list = type(reference)==list
	Nref = 0
	if is_list:
		Nref = len(reference)
	else:
		if reference is not None:
			Nref = 1
			reference = [reference]

	if Nref<=3:
		legend_ncols = Nref+1
	else:
		legend_ncols = 3

	if labels is None:
		labels = ["inverted"]
		for idr in range(Nref):
			labels.append(f"ref{idr+1}")

	k_ = 0
	for i_ in range(nrows):
		for j_ in range(ncols):
			if k_+1>len(parameters):
				continue

			parameter = parameters[k_]
			parID = atmos.par_id[parameter]
			# if parameter=="chi":
			# 	y = np.sin(atmos.values[parameter])
			# 	atmos.values[parameter] = np.arcsin(y)

			# 	y = np.sin(cube[parID])
			# 	cube[parID] = np.arcsin(y)

			ax = fig.add_subplot(gs[i_,j_])

			try:
				x = atmos.nodes[parameter]
				y = atmos.values[parameter][idx,idy].copy() * fact[parameter]
				if show_errors and parameter in ["temp", "vz"]:
					yerr = atmos.errors[parameter][idx,idy].copy() * fact[parameter]
					ax.autoscale(False)
					ax.errorbar(x, y, yerr=yerr, 
						fmt=".",
						elinewidth=0.75,
						color=colors[0])
				else:
					ax.scatter(x, y, s=20, color=colors[0])
			except:
				pass

			ax.autoscale(True)
			ax.plot(atmos.data[idx,idy,0], cube[parID]*fact[parameter], ls=ls, lw=lw, color=colors[0], label=labels[0])
			if parameter=="ne" or parameter=="nH":
				ax.set_yscale("log")

			ax.set_xlim(atmos.data[idx,idy,0,0], atmos.data[idx,idy,0,-1])

			ax.set_xlabel(r"log$\tau$")
			ax.set_ylabel(f"{pars_symbol[parameter]} [{unit[parameter]}]")

			if reference is not None:
				for idr, ref in enumerate(reference):
					ax.plot(ref.data[idx,idy,0], ref.data[idx,idy,parID]*fact[parameter], ls=ls, lw=lw/2, color=colors[(idr+1)%Ncolors], label=labels[idr+1])
					try:
						x = ref.nodes[parameter]
						y = ref.values[parameter][idx,idy].copy() * fact[parameter]
						if show_errors and parameter in ["temp", "vz"]:
							yerr = ref.errors[parameter][idx,idy].copy() * fact[parameter]
							ax.autoscale(False)
							ax.errorbar(x, y, yerr=yerr, 
								fmt=".",
								elinewidth=0.75,
								color=colors[(idr+1)%Ncolors])
						else:
							ax.scatter(x, y, s=20, color=colors[(idr+1)%Ncolors])
					except:
						pass

			# set legend
			if i_+j_==0:
				ax.legend(loc="lower left", 
						  bbox_to_anchor=(0, 1.01, 1, 0.2),
						  ncol=legend_ncols, 
						  fontsize="x-small", 
						  frameon=True)
			k_ += 1
	
	fig.tight_layout()

	return fig

def plot_spectra(obs, wavelength, inv=None, axes=None, aspect=1, shift=None, norm=False, 
	color="tab:blue", inv_colors=None, lw=1, title=None, subtitles_flag=False, center_wavelength_grid=True, 
	labels=None):
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
	norm : bool (optional)
		Flag if the spectra are already normalized or not. It applies x100 to
		polarization signals if the spectra are normalized.
	center_wavelength_grid : bool (optional)
		Flag for setting the wavelength grid relative to the central wavelength.
	"""
	colors = ["tab:red", "tab:green", "tab:orange"]
	if inv_colors is not None:
		colors = inv_colors

	Ncolors = len(colors)

	lmin = np.min(wavelength)
	lmax = np.max(wavelength)
	lam0 = (lmax + lmin) / 2
	dlam = (lmax - lam0) * 10
	lmin *= 10
	lmax *= 10
	if not center_wavelength_grid:
		lam0 = 0
	else:
		lmin, lmax = -dlam, dlam

	fact = 1
	if norm:
		fact = 100
		# coeff = (obs[0,0] - obs[-1,0]) / (wavelength[0] - wavelength[-1])
		# n = obs[0,0] - coeff * wavelength[0]
		# Icont = coeff * wavelength + n
		# for ids in range(4):
		# 	obs[:,ids] /= Icont

	if title:
		fig.suptitle(title)

	if inv is None:
		if axes is None:
			height = 6
			width = 8*aspect
			fig = plt.figure(figsize=(8,6))
			gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.4, hspace=0.3)
			axI = fig.add_subplot(gs[0,0])
			axQ = fig.add_subplot(gs[0,1])
			axU = fig.add_subplot(gs[1,0])
			axV = fig.add_subplot(gs[1,1])
		else:
			axI, axQ, axU, axV = axes

		if subtitles_flag:
			axI.set_title("Stokes I")
			axQ.set_title("Stokes Q")
			axU.set_title("Stokes U")
			axV.set_title("Stokes V")
		
		#--- Stokes I
		if shift:
			if norm:
				obs[:,0] += shift/Icont[0]
			else:
				obs[:,0] += shift

		axI.plot((wavelength - lam0)*10, obs[:,0], lw=lw, color="k")
		# if norm:
		axI.set_ylabel(r"Stokes I/I$_\mathrm{c}$")
		# else:
		# 	axI.set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		axI.set_xlim([lmin, lmax])

		#--- Stokes Q
		axQ.plot((wavelength - lam0)*10, obs[:,1]*fact, lw=lw, color="k")
		axQ.set_ylabel(r"Stokes Q/I$_\mathrm{c}$ [\%]")
		axQ.set_xlim([lmin, lmax])
		
		#--- Stokes U
		axU.plot((wavelength - lam0)*10, obs[:,2]*fact, lw=lw, color="k")
		axU.set_xlim([lmin, lmax])
		axU.set_xlabel(r"$\Delta \lambda$ [$\mathrm{\AA}$]")
		if not center_wavelength_grid:
			axU.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
		axU.set_ylabel(r"Stokes U/I$_\mathrm{c}$ [\%]")
		
		#--- Stokes V
		axV.plot((wavelength - lam0)*10, obs[:,3]*fact, lw=lw, color="k")
		axV.set_xlim([lmin, lmax])
		axV.set_xlabel(r"$\Delta \lambda$ [$\mathrm{\AA}$]")
		if not center_wavelength_grid:
			axV.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
		axV.set_ylabel(r"Stokes V/I$_\mathrm{c}$ [\%]")
	else:
		if axes is None:
			width, height = 3, 2 + 2/3
			width *= 1.25
			height *= 1.25
			width *= aspect
			fig = plt.figure(figsize=(2*width,2*height))
			gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.40, hspace=0.15)
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

		Ninv = len(inv)

		set_labels = False
		if labels is not None:
			set_labels = True
		else:
			labels = [None]*(Ninv+1)

		#--- Stokes I
		ax0_SI.plot((wavelength - lam0)*10, obs[:,0], "k-", markersize=2, lw=lw, label=labels[0])
		for idn in range(Ninv):
			ax0_SI.plot((wavelength - lam0)*10, inv[idn][:,0], color=colors[(idn)%Ncolors], lw=1, label=labels[idn+1])
		ax0_SI.set_ylabel(r"Stokes $I$")
		ax0_SI.set_xlim([lmin, lmax])
		ax0_SI.set_xticklabels([])

		# plot legend
		if set_labels:
			if Ninv<=3:
				legend_ncols = Ninv+1
			else:
				legend_ncols = 3
			ax0_SI.legend(loc="lower left", 
						  bbox_to_anchor=(0, 1.01, 1, 0.2),
						  ncol=legend_ncols, 
						  fontsize=12, 
						  frameon=True)

		ax1_SI.plot([lmin, lmax], [0,0], color="k", lw=0.5)
		for idn in range(Ninv):
			difference = obs[:,0] - inv[idn][:,0]
			ax1_SI.plot((wavelength - lam0)*10, difference, color=colors[(idn)%Ncolors], lw=1)
		ax1_SI.set_ylabel(r"$\Delta I$")
		ax1_SI.set_xlim([lmin, lmax])
		ax1_SI.minorticks_off()

		#--- Stokes Q
		ax0_SQ.plot((wavelength - lam0)*10, obs[:,1]*100, "k-", markersize=2, lw=lw)
		for idn in range(Ninv):	
			ax0_SQ.plot((wavelength - lam0)*10, inv[idn][:,1]*100, color=colors[(idn)%Ncolors], lw=1)
		ax0_SQ.set_ylabel(r"Stokes $Q/I_c$ [\%]")
		ax0_SQ.set_xlim([lmin, lmax])
		ax0_SQ.set_xticklabels([])

		ax1_SQ.plot([lmin, lmax], [0,0], color="k", lw=0.5)
		for idn in range(Ninv):
			difference = obs[:,1] - inv[idn][:,1]
			ax1_SQ.plot((wavelength - lam0)*10, difference*100, color=colors[(idn)%Ncolors], lw=1)
		ax1_SQ.set_ylabel(r"$\Delta Q$")
		ax1_SQ.set_xlim([lmin, lmax])
		ax1_SQ.minorticks_off()

		#--- Stokes U
		ax0_SU.plot((wavelength - lam0)*10, obs[:,2]*100, "k-", markersize=2, lw=lw)
		for idn in range(Ninv):	
			ax0_SU.plot((wavelength - lam0)*10, inv[idn][:,2]*100, color=colors[(idn)%Ncolors], lw=1)
		ax0_SU.set_ylabel(r"Stokes $U/I_c$ [\%]")
		ax0_SU.set_xlim([lmin, lmax])
		ax0_SU.set_xticklabels([])

		ax1_SU.plot([lmin, lmax], [0,0], color="k", lw=0.5)
		for idn in range(Ninv):
			difference = obs[:,2] - inv[idn][:,2]
			ax1_SU.plot((wavelength - lam0)*10, difference*100, color=colors[(idn)%Ncolors], lw=1)
		ax1_SU.set_xlabel(r"$\Delta \lambda$ [$\mathrm{\AA}$]")
		if not center_wavelength_grid:
			ax1_SU.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
		ax1_SU.set_ylabel(r"$\Delta U$")
		ax1_SU.set_xlim([lmin, lmax])
		ax1_SU.minorticks_off()
		
		#--- Stokes V
		ax0_SV.plot((wavelength - lam0)*10, obs[:,3]*100, "k-", markersize=2, lw=lw)
		for idn in range(Ninv):	
			ax0_SV.plot((wavelength - lam0)*10, inv[idn][:,3]*100, color=colors[(idn)%Ncolors], lw=1)
		ax0_SV.set_ylabel(r"Stokes $V/I_c$ [\%]")
		ax0_SV.set_xlim([lmin, lmax])
		ax0_SV.set_xticklabels([])

		ax1_SV.plot([lmin, lmax], [0,0], color="k", lw=0.5)
		for idn in range(Ninv):	
			difference = obs[:,3] - inv[idn][:,3]
			ax1_SV.plot((wavelength - lam0)*10, difference*100, color=colors[(idn)%Ncolors], lw=1)
		ax1_SV.set_xlabel(r"$\Delta \lambda$ [$\mathrm{\AA}$]")
		if not center_wavelength_grid:
			ax1_SV.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
		ax1_SV.set_ylabel(r"$\Delta V$")
		ax1_SV.set_xlim([lmin, lmax])
		ax1_SV.minorticks_off()

	return fig

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
	# plt.legend(fontsize=12)
	plt.savefig(fpath)
	plt.close()

def plot_rf(_rf, local_parameters=[], global_parameters=[], idx=0, idy=0, Stokes="I", 
			 logtau_top=-6, logtau_bot=1,
			 lmin=None, lmax=None,
	    	 rf_wave_integrate=False, rf_tau_integrate=False):
	cmap = {"temp"  : "bwr",
			"vmic"  : "bwr",
			"vz"    : "bwr", 
			"mag"   : "bwr", 
			"gamma" : "bwr",
			"chi"   : "bwr",
			"loggf" : "plasma"}
	cbar_label = {"temp"   : "Temperature", 
				  "vmic"   : r"Micro-velocity",
				  "vz"     : r"LOS velocity", 
				  "mag"    : "Mag. field", 
				  "gamma"  : r"Inclination", 
				  "chi"    : r"Azimuth"}
	stokesV = {"I" : 0, "Q" : 1, "U" : 2, "V" : 3}
	global_pars_colors = ["black", "tab:red", "tab:blue", "tab:orange"]

	fontsize = "large"

	if not _rf.normed_spec:
		cmap["temp"] = "YlOrRd"

	wavs = _rf.wavelength * 10
	lam_min, lam_max = wavs[0], wavs[-1]
	if lmin is not None:
		lam_min = lmin
	if lmax is not None:
		lam_max = lmax
	ind_lmin = np.argmin(np.abs(wavs-lam_min))
	ind_lmax = np.argmin(np.abs(wavs-lam_max))+1
	wavs = wavs[ind_lmin:ind_lmax]
	lam0 = (lam_max + lam_min)/2

	ind_top = np.argmin(np.abs(_rf.logtau - logtau_top))
	ind_bot = np.argmin(np.abs(_rf.logtau - logtau_bot))+1

	logtau = _rf.logtau[ind_top:ind_bot]
	dtau = _rf.logtau[1] - _rf.logtau[0]

	local_pars, global_pars = _rf.local_pars, _rf.global_pars

	# rf_local.shape = (nx, ny, n_local_par, nz, nw, 4)
	# rf_global.shape = (nx, ny, n_global_par, nw, 4)
	try:
		rf_local = _rf.rf_local[idx,idy,:, ind_top:ind_bot, ind_lmin:ind_lmax]
	except:
		rf_local = None
	try:
		rf_global = _rf.rf_global[idx,idy,:, ind_lmin:ind_lmax]
	except:
		pass

	if rf_local is not None:
		npar, nz, nw, ns = rf_local.shape
	elif rf_global is not None:
		npar, nw, ns = rf_global.shape
	else:
		raise ValueError("There is not RF to be plotted.")

	stokes_range = []
	stokes_labels = []
	for item in Stokes:
		item = item.upper()
		stokes_range.append(stokesV[item])
		stokes_labels.append(f"Stokes {item}")

	NZ = int((logtau[-1] - logtau[0]) / dtau) + 1

	nrows = len(local_parameters) + len(global_parameters)
	ncols = len(stokes_range)
	NW = 11
	if ncols!=1:
		NW = 5

	width, height = 3, 2+2/3
	fig = plt.figure(figsize=(width*ncols, height*nrows))
	gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.5, hspace=0.4)

	if rf_local is not None:
		for i_, parameter in enumerate(local_parameters):
			try:
				idp = local_pars[parameter]
			except:
				print(f"No RF for parameter {parameter}")
				continue

			if not rf_wave_integrate:
				for j_, ids in enumerate(stokes_range):
					ax = fig.add_subplot(gs[i_,j_])
					if i_==0:
						ax.set_title(stokes_labels[j_], fontsize=fontsize)
					if j_==0:
						ax.set_ylabel(r"$\log(\tau)$", fontsize=fontsize)
					
					matrix = rf_local[idp,:,:, ids]
					vmax = np.max(np.abs(matrix))
					vmin = -vmax
					par_cmap = cmap[parameter]
					if parameter=="temp":
						if ids!=0:
							par_cmap = "bwr"
					# 	else:
					# 		vmin = 0
					
					im = ax.imshow(matrix, aspect="auto", origin="upper",
						cmap=par_cmap, vmin=vmin, vmax=vmax,
						extent=[wavs[0], wavs[-1], logtau[-1], logtau[0]])
					if j_+1==ncols:
						add_colorbar(fig, ax, im, label=cbar_label[parameter], fontsize=fontsize)
					else:
						add_colorbar(fig, ax, im)
					if j_>0:
						ax.set_yticklabels([])
					if i_+1<nrows:
						ax.set_xticklabels([])

					ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
					ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
					ax.grid(which="both", axis="y", lw=0.5)

			if rf_wave_integrate:	
				RF = rf_local[idp]
				# RF *= parameter_norm[parameter]
				integratedRF = np.sum(np.abs(RF), axis=1)
				vmax = np.max(integratedRF)*1.05
				vmin = np.min(integratedRF)*1.05

				for j_, ids in enumerate(stokes_range):
					ax = fig.add_subplot(gs[i_,j_])
					if i_==0:
						ax.set_title(stokes_labels[j_])

					ax.plot(logtau, integratedRF[...,ids], c="k")
					ax.axhline(y=0, c="k", lw=0.5, alpha=0.5)
					ax.set_xlim([logtau[0], logtau[-1]])
					ax.set_ylim([vmin, vmax])
					if j_==0:
						ax.set_ylabel(f"{cbar_label[parameter]}")

					# ax.grid(b=True, which="major", axis="both", lw=0.5)

	i_ = len(local_parameters)
	for ii, parameter in enumerate(global_parameters):
		i_ += ii
		try:
			idp = global_pars[parameter]
		except:
			print(f"No RF for parameter {parameter}.")
			continue

		for j_, ids in enumerate(stokes_range):
			ax = fig.add_subplot(gs[i_,j_])
			ax.set_xlabel(r"$\lambda [\AA{}]$", fontsize=fontsize)
			for idl in idp:
				ax.plot(wavs, rf_global[idl,:,ids], c=global_pars_colors[idl])#*parameter_norm[parameter])
				ax.set_xlim([wavs[0], wavs[-1]])

		ax.yaxis.set_label_position("right")
		ax.set_ylabel(r"$\log(gf)$", fontsize=fontsize)

def add_colorbar(fig, ax, im, label=None, fontsize="normal"):
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	axins = inset_axes(ax,
	                   width="5%",
	                   height="100%",
	                   loc='lower left',
	                   bbox_to_anchor=(1.02, 0., 1, 1),
	                   bbox_transform=ax.transAxes,
	                   borderpad=0)
	cbar = fig.colorbar(im, cax=axins)
	if label is not None:
		cbar.set_label(label, fontsize=fontsize)