import matplotlib.pyplot as plt
import numpy as np
import copy
from astropy.io import fits
import sys

# import globin

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

def show():
	"""
	Just to show up the plot from 'plot_atmosphere' and/or
	'plot_spectra' withouth the need to import matplotlib.
	"""
	plt.show()

def plot_atmosphere(atmos, parameters, idx=0, idy=0, ls="-", lw=2, color="tab:red", labels=None, reference=None):
	colors = ["tab:red", "tab:orange", "tab:green"]
	Ncolors = len(colors)

	logtau = atmos.logtau
	cube = atmos.data[idx,idy]

	n_plots = len(parameters)
	if n_plots==1:
		ncols = 1
		nrows = 1
	else:
		ncols = 2
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

			ax = fig.add_subplot(gs[i_,j_])

			try:
				x = atmos.nodes[parameter]
				y = atmos.values[parameter][idx,idy] * fact[parameter]
				ax.scatter(x, y, s=20, color=colors[0])
			except:
				pass

			ax.plot(logtau, cube[parID]*fact[parameter], ls=ls, lw=lw, color=colors[0], label=labels[0])
			if parameter=="ne" or parameter=="nH":
				ax.set_yscale("log")

			ax.set_xlabel(r"log$\tau$")
			ax.set_ylabel(f"{pars_symbol[parameter]} [{unit[parameter]}]")

			if reference is not None:
				for idr, ref in enumerate(reference):
					ax.plot(ref.logtau, ref.data[idx,idy,parID]*fact[parameter], ls=ls, lw=lw/2, color=colors[(idr+1)%Ncolors], label=labels[idr+1])
					try:
						x = ref.nodes[parameter]
						y = ref.values[parameter][idx,idy] * fact[parameter]
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

def plot_spectra(obs, wavelength, inv=None, axes=None, shift=None, norm=False, 
	color="tab:blue", lw=1, title=None, subtitles_flag=False, relative=True, 
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
	"""
	colors = ["tab:red", "tab:orange", "tab:green"]
	Ncolors = len(colors)

	lmin = np.min(wavelength)
	lmax = np.max(wavelength)
	lam0 = (lmax + lmin) / 2
	dlam = (lmax - lam0) * 10
	lmin *= 10
	lmax *= 10
	if not relative:
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

		axI.plot((wavelength - lam0)*10, obs[:,0], lw=lw, color=color)
		if norm:
			axI.set_ylabel(r"Stokes I/I$_\mathrm{c}$")
		else:
			axI.set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		axI.set_xlim([lmin, lmax])

		#--- Stokes Q
		axQ.plot((wavelength - lam0)*10, obs[:,1]*fact, lw=lw, color=color)
		axQ.set_ylabel(r"Stokes Q/I$_\mathrm{c}$ [\%]")
		axI.set_xlim([lmin, lmax])
		
		#--- Stokes U
		axU.plot((wavelength - lam0)*10, obs[:,2]*fact, lw=lw, color=color)
		axI.set_xlim([lmin, lmax])
		axU.set_xlabel(r"$\Delta \lambda$ [$\mathrm{\AA}$]")
		if not relative:
			axU.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
		axU.set_ylabel(r"Stokes U/I$_\mathrm{c}$ [\%]")
		
		#--- Stokes V
		axV.plot((wavelength - lam0)*10, obs[:,3]*fact, lw=lw, color=color)
		axI.set_xlim([lmin, lmax])
		axV.set_xlabel(r"$\Delta \lambda$ [$\mathrm{\AA}$]")
		if not relative:
			axV.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
		axV.set_ylabel(r"Stokes V/I$_\mathrm{c}$ [\%]")
		
		return axI, axQ, axU, axV
	else:
		if axes is None:
			fig = plt.figure(figsize=(9,9))
			gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.35, hspace=0.15)
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
		
		is_list = type(inv)==list
		if is_list:
			Ninv = len(inv)
		else:
			Ninv = 1
			inv = [inv]

		set_labels = False
		if labels is not None:
			set_labels = True
		else:
			labels = [None]*Ninv

		#--- Stokes I
		# ax0_SI.set_title("Stokes I")
		ax0_SI.plot((wavelength - lam0)*10, obs[:,0], "k-", markersize=2, label="Observation")
		for idn in range(Ninv):
			ax0_SI.plot((wavelength - lam0)*10, inv[idn][:,0], color=colors[idn%Ncolors], lw=1.5, label=labels[idn])
		# ax0_SI.set_ylabel(r"I [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0_SI.set_ylabel(r"Stokes $I$")
		ax0_SI.set_xlim([-dlam, dlam])

		# plot legend
		if Ninv<=3:
			legend_ncols = Ninv+1
		else:
			legend_ncols = 3
		ax0_SI.legend(loc="lower left", 
					  bbox_to_anchor=(0, 1.01, 1, 0.2),
					  ncol=legend_ncols, 
					  fontsize="x-small", 
					  frameon=True)

		ax1_SI.plot([-dlam, dlam], [0,0], color="k", lw=0.5)
		for idn in range(Ninv):
			difference = obs[:,0] - inv[idn][:,0]
			ax1_SI.plot((wavelength - lam0)*10, difference, color=colors[idn%Ncolors], lw=1.5)
		# ax1_SI.set_xlabel(r"$\Delta \lambda$ [$\mathrm{\AA}$]")
		# if not relative:
		# 	ax1_SI.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
		ax1_SI.set_ylabel(r"$\Delta I$")
		ax1_SI.set_xlim([-dlam, dlam])

		#--- Stokes Q
		# ax0_SQ.set_title("Stokes Q")
		ax0_SQ.plot((wavelength - lam0)*10, obs[:,1]*100, "k-", markersize=2)
		for idn in range(Ninv):	
			ax0_SQ.plot((wavelength - lam0)*10, inv[idn][:,1]*100, color=colors[idn%Ncolors], lw=1.5)
		# ax0_SQ.set_ylabel(r"Q [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0_SQ.set_ylabel(r"Stokes $Q/I_c$ [\%]")
		ax0_SQ.set_xlim([-dlam, dlam])

		ax1_SQ.plot([-dlam, dlam], [0,0], color="k", lw=0.5)
		for idn in range(Ninv):
			difference = obs[:,1] - inv[idn][:,1]
			ax1_SQ.plot((wavelength - lam0)*10, difference*100, color=colors[idn%Ncolors], lw=1.5)
		# ax1_SQ.set_xlabel(r"$\Delta \lambda$ [$\mathrm{\AA}$]")
		# if not relative:
		# 	ax1_SQ.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
		ax1_SQ.set_ylabel(r"$\Delta Q$")
		ax1_SQ.set_xlim([-dlam, dlam])

		#--- Stokes U
		# ax0_SU.set_title("Stokes U")
		ax0_SU.plot((wavelength - lam0)*10, obs[:,2]*100, "k-", markersize=2)
		for idn in range(Ninv):	
			ax0_SU.plot((wavelength - lam0)*10, inv[idn][:,2]*100, color=colors[idn%Ncolors], lw=1.5)
		# ax0_SU.set_ylabel(r"U [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0_SU.set_ylabel(r"Stokes $U/I_c$ [\%]")
		ax0_SU.set_xlim([-dlam, dlam])

		ax1_SU.plot([-dlam, dlam], [0,0], color="k", lw=0.5)
		for idn in range(Ninv):
			difference = obs[:,2] - inv[idn][:,2]
			ax1_SU.plot((wavelength - lam0)*10, difference*100, color=colors[idn%Ncolors], lw=1.5)
		ax1_SU.set_xlabel(r"$\Delta \lambda$ [$\mathrm{\AA}$]")
		if not relative:
			ax1_SU.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
		ax1_SU.set_ylabel(r"$\Delta U$")
		ax1_SU.set_xlim([-dlam, dlam])
		
		#--- Stokes V
		# ax0_SV.set_title("Stokes V")
		ax0_SV.plot((wavelength - lam0)*10, obs[:,3]*100, "k-", markersize=2)
		for idn in range(Ninv):	
			ax0_SV.plot((wavelength - lam0)*10, inv[idn][:,3]*100, color=colors[idn%Ncolors], lw=1.5)
		# ax0_SV.set_ylabel(r"V [10$^8$ W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
		ax0_SV.set_ylabel(r"Stokes $V/I_c$ [\%]")
		ax0_SV.set_xlim([-dlam, dlam])

		ax1_SV.plot([-dlam, dlam], [0,0], color="k", lw=0.5)
		for idn in range(Ninv):	
			difference = obs[:,3] - inv[idn][:,3]
			ax1_SV.plot((wavelength - lam0)*10, difference*100, color=colors[idn%Ncolors], lw=1.5)
		ax1_SV.set_xlabel(r"$\Delta \lambda$ [$\mathrm{\AA}$]")
		if not relative:
			ax1_SV.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
		ax1_SV.set_ylabel(r"$\Delta V$")
		ax1_SV.set_xlim([-dlam, dlam])

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
	# plt.legend(fontsize=12)
	plt.savefig(fpath)
	plt.close()

def plot_rf(_rf, parameters=["temp"], idx=0, idy=0, Stokes="I", logtau_top=-6, logtau_bot=1, norm=False):
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

	if not _rf.normed_spec:
		cmap["temp"] = "YlOrRd"

	wavs = _rf.wavelength
	lam_min, lam_max = wavs[0], wavs[-1]
	lam0 = (lam_max + lam_min)/2

	ind_top = np.argmin(np.abs(_rf.logtau - logtau_top))
	ind_bot = np.argmin(np.abs(_rf.logtau - logtau_bot))+1

	# rf.shape = (nx, ny, 6, nz, nw, 4)
	if norm:
		_rf.norm()
	rf = _rf.rf
	nx, ny, npar, nz, nw, ns = rf.shape
	logtau = _rf.logtau[ind_top:ind_bot]
	dtau = _rf.logtau[1] - _rf.logtau[0]

	pars = _rf.pars

	stokes_range = []
	stokes_labels = []
	for item in Stokes:
		item = item.upper()
		stokes_range.append(stokesV[item])
		stokes_labels.append(f"Stokes {item}")

	NZ = int(np.int((logtau[-1] - logtau[0])) / dtau) + 1

	nrows = len(parameters)
	ncols = len(stokes_range)
	NW = 11
	if ncols!=1:
		NW = 5

	fig = plt.figure(figsize=(12,10), dpi=90)
	gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.35, hspace=0.5)

	for i_, parameter in enumerate(parameters):
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
				if norm and ids==0:
					par_cmap = "bwr"
			# 	else:
			# 		vmin = 0
			im = ax.imshow(matrix, aspect="auto", origin="upper",
				cmap=par_cmap, vmin=vmin, vmax=vmax,
				extent=[wavs[0], wavs[-1], logtau[-1], logtau[0]])
			if j_+1==ncols:
				add_colorbar(fig, ax, im, label=cbar_label[parameter])
			else:
				add_colorbar(fig, ax, im)

			if j_>0:
				ax.set_yticklabels([])
			if i_+1<nrows:
				ax.set_xticklabels([])

			ax.grid(b=True, which="major", axis="y", lw=0.5)

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