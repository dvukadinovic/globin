import numpy as np
import emcee
from astropy.io import fits
import matplotlib.pyplot as plt
from time import time

import globin
from globin.inversion import Inverter

RNG = np.random.default_rng()
scales = {"temp"  : 1,			# [K]
		  "vz"    : 1e-3,		# [km/s]
		  "vmic"  : 1e-3,		# [km/s]
		  "mag"   : 1,			# [G]
		  "gamma" : 0.01,#1e-2*np.pi/360,	# [rad]
		  "chi"   : 0.01,#1e-2*np.pi/360,	# [rad]
		  "of"    : 1e-3,		# 
		  "stray" : 1e-3,		#
		  "vmac"  : 1e-3,		# [km/s]
		  "loggf" : 0.001,		#
		  "dlam"  : 0.1}		#

def invert_mcmc(obs, atmos, move, backend, reset_backend=True, weights=np.array([1,1,1,1]), noise=1e-3, nsteps=100, nwalkers=2, pool=None, sequential=True, progress_frequency=100):

	print("\n{:{char}{align}{width}}\n".format(f" Entering MCMC inversion mode ", char="-", align="^", width=globin.NCHAR))

	# if atmos.sl_atmos is not None:
	# 	# start = time()
	# 	obs.sl_spec = atmos.sl_atmos.compute_spectra(pool=pool)
	# 	# print(time() - start)

	atmos.limit_values["gamma"] = globin.atmos.MinMax(-1,1)
	atmos.limit_values["chi"] = globin.atmos.MinMax(-1,1)

	if atmos.stray_type=="hsra" or atmos.norm_level=="hsra":
		atmos.get_hsra_cont()
		if atmos.stray_type=="hsra":
			atmos.hsra_spec.broaden_spectra(atmos.vmac)

	Natmos = atmos.nx*atmos.ny

	atmos.n_local_pars = 0
	if not atmos.skip_local_pars:
		for parameter in atmos.nodes:
			atmos.n_local_pars += len(atmos.nodes[parameter])

	ndim = Natmos*atmos.n_local_pars
	if not atmos.skip_global_pars:
		ndim += atmos.n_global_pars
	
	obs.Ndof = np.count_nonzero(obs.weights)*obs.nw*Natmos - ndim

	print("\n{:{char}{align}{width}}\n".format(f" Info ", char="-", align="^", width=globin.NCHAR))
	print("atmos.shape {:{char}{align}{width}}".format(f" {atmos.shape}", char=".", align=">", width=20))
	if not atmos.skip_local_pars:
		print("N_local_pars {:{char}{align}{width}}".format(f" {atmos.n_local_pars}", char=".", align=">", width=20))
	if not atmos.skip_global_pars:
		print("N_global_pars {:{char}{align}{width}}".format(f" {atmos.n_global_pars}", char=".", align=">", width=20))
	print("Nwalkers {:{char}{align}{width}}".format(f" {nwalkers}", char=".", align=">", width=20))
	print("Nsteps {:{char}{align}{width}}\n".format(f" {nsteps}", char=".", align=">", width=20))
	
	if reset_backend:
		backend.reset(nwalkers, ndim)
		p0 = initialize_walker_states(nwalkers, ndim, atmos)
	else:
		p0 = backend.get_last_sample()

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, 
		args=[obs, atmos, None if sequential else pool], 
		moves=move, 
		pool=pool if sequential else None,
		backend=backend)

	old_tau = np.inf
	for sample in sampler.sample(initial_state=p0, iterations=nsteps, progress=True, store=True):
		if sampler.iteration%progress_frequency:
			continue

		tau = sampler.get_autocorr_time(tol=0, has_walkers=False, quiet=True)

		print(f"\nAR: {np.mean(sampler.acceptance_fraction):.3f} | ACT = {np.mean(tau):.2f}")

		# check convergence
		converged = np.all(tau * 100 < sampler.iteration)
		converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
		if converged:
			break
		old_tau = tau

def lnprior(local_pars, global_pars, limits):
	"""
	Check if each parameter is in its respective bounds given by globin.limit_values.

	If one fails, return -np.inf, else return 0.
	"""	
	if local_pars is not None:
		for parameter in local_pars:
			nnodes = local_pars[parameter].shape[-1]
			for idn in range(nnodes):
				#--- check lower boundary condition
				vmin = limits[parameter].min[0]
				if limits[parameter].vmin_dim!=1:
					vmin = limits[parameter].min[idn]
				indx, indy = np.where(local_pars[parameter][...,idn]<vmin)
				if len(indx)>0:
					return -np.inf

				#--- check upper boundary condition
				vmax = limits[parameter].max[0]
				if limits[parameter].vmax_dim!=1:
					vmax = limits[parameter].max[idn]
				indx, indy = np.where(local_pars[parameter][...,idn]>vmax)
				if len(indx)>0:
					return -np.inf
	
	if global_pars is not None:
		for parameter in global_pars:
			if parameter=="vmac":
				if (global_pars[parameter]<limits[parameter][0]) or \
				   (global_pars[parameter]>limits[parameter][1]):
					return -np.inf
			elif parameter=="stray":
				if (global_pars[parameter]<limits[parameter].min[0]) or \
				   (global_pars[parameter]>limits[parameter].max[0]):
					return -np.inf
			else:
				Npar = global_pars[parameter].shape[-1]
				limit = limits[parameter]
				if Npar>0:
					for idl in range(Npar):
						# if limits[parameter].ndim==2:
						# 	limit = limits[parameter][idl]
						#--- check lower boundary condition

						indx, indy = np.where(global_pars[parameter][...,idl]<limit[0])
						if len(indx)>0:
							return -np.inf

						#--- check upper boundary condition
						indx, indy = np.where(global_pars[parameter][...,idl]>limit[1])
						if len(indx)>0:
							return -np.inf					

	return 0.0

def lnlike(obs, atmos, pool):
	if pool is None:
		if not atmos.skip_local_pars:
			params = list(atmos.nodes.keys())
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					args = atmos, 1, idx, idy, params
					result = atmos._build_from_nodes(args)
					atmos.data[idx,idy] = result

		spec = sequential_synthesize(obs, atmos)
	else:
		spec = mpi_synthesize(obs, atmos, pool)

	# plt.plot(obs.I[0,0])
	# plt.plot(spec.I[0,0])
	# plt.show()

	diff = obs.spec - spec.spec
	diff *= obs.weights
	diff *= obs.wavs_weight
	diff /= obs.noise_stokes
	chi2 = np.sum(diff**2)
	chi2 /= obs.Ndof

	return chi2 * (-0.5)

def log_prob(theta, obs, atmos, pool):
	"""
	Compute product of prior and likelihood.

	We need what is needed for prior and likelihood
	"""
	Natmos = atmos.nx*atmos.ny

	#--- update parameters
	up = 0
	if not atmos.skip_local_pars:
		for parameter in atmos.nodes:
			nnodes = len(atmos.nodes[parameter])
			low = up
			up += Natmos*nnodes
			atmos.values[parameter][:,:,:] = theta[low:up].reshape(atmos.nx, atmos.ny, nnodes, order="C")

	for parameter in atmos.global_pars:
		npars = atmos.global_pars[parameter].shape[-1]
		if npars>0 and not atmos.skip_global_pars:
			low = up
			up += npars
			if parameter in ["loggf", "dlam"]:
				atmos.global_pars[parameter][0,0] = theta[low:up]
				if atmos.sl_atmos is not None:
					atmos.sl_atmos.global_pars[parameter][0,0] = theta[low:up]
			if parameter=="stray":
				atmos.global_pars[parameter] = theta[low:up]

	#--- compute posterior
	if atmos.skip_local_pars:
		lp = lnprior(None, atmos.global_pars, atmos.limit_values)
	if atmos.skip_global_pars:
		lp = lnprior(atmos.values, None, atmos.limit_values)

	if not np.isfinite(lp):
		return -np.inf
	
	# get back values into the atmosphere structure
	if not atmos.skip_global_pars:
		if "vmac" in atmos.global_pars:
			atmos.vmac = atmos.global_pars["vmac"][0]
		if "stray" in atmos.global_pars:
			atmos.stray_light = atmos.global_pars["stray"]

	# compute the azimuth from the sin^2(chi)
	if not atmos.skip_local_pars:
		if "chi" in atmos.nodes:
			# proposal = np.arcsin(np.sqrt(atmos.values["chi"]))
			proposal = np.arcsin(atmos.values["chi"])
			atmos.values["chi"][:,:,:] = proposal
		
		# compute the inclination from the cos(gamma)
		if "gamma" in atmos.nodes:
			proposal = np.arccos(atmos.values["gamma"])
			atmos.values["gamma"][:,:,:] = proposal

	return lp + lnlike(obs, atmos, pool)

def sequential_synthesize(obs, atmos):
	# compute spectra
	nw = len(atmos.wavelength_vacuum)
	spec = globin.spec.Spectrum(nx=atmos.nx, ny=atmos.ny, nw=nw)
	spec.wavelength = atmos.wavelength_air

	if atmos.vmac!=0:
		kernel = spec.get_kernel(atmos.vmac, order=0)
	
	for idx in range(atmos.nx):
		for idy in range(atmos.ny):
			stokes_vector = atmos._compute_spectra_sequential((idx,idy)).T
			if atmos.vmac!=0:	
				stokes_vector = globin.spec._broaden_spectra((stokes_vector, kernel))
			if atmos.instrumental_profile is not None:
				stokes_vector = globin.spec._broaden_spectra((stokes_vector, atmos.instrumental_profile))
			spec.spec[idx,idy] = stokes_vector

	if atmos.add_stray_light:
		# get the stray light factor(s)
		if "stray" in atmos.global_pars:
			stray_light = atmos.global_pars["stray"]
			stray_light = np.ones((atmos.nx, atmos.ny, 1)) * stray_light
		elif "stray" in atmos.values:
				stray_light = atmos.values["stray"]
		else:
			stray_light = atmos.stray_light

		# check for HSRA spectrum if we are using the 'hsra' stray light contamination
		sl_spectrum = None
		if atmos.stray_type=="hsra":
			sl_spectrum = atmos.hsra_spec.spec
		if atmos.stray_type=="2nd_component":
			sl_spectrum = obs.sl_spec.spec
		if atmos.stray_type in ["atmos", "spec"]:
			sl_spectrum = atmos.stray_light_spectrum.spec

		spec.add_stray_light(atmos.stray_mode, atmos.stray_type, stray_light, sl_spectrum=sl_spectrum)

	if atmos.norm:
		if atmos.norm_level==1:
			Ic = spec.I[...,atmos.continuum_idl]
			spec.spec = np.einsum("ij...,ij->ij...", spec.spec, 1/Ic)
		elif atmos.norm_level=="hsra":
			spec.spec /= atmos.icont
		else:
			spec.spec /= atmos.norm_level

	#--- downsample the synthetic spectrum to observed wavelength grid
	if not np.array_equal(atmos.wavelength_obs, atmos.wavelength_air):
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				args = spec.spec[idx,idy], atmos.wavelength_obs, "extrapolate"
				spec.spec[idx,idy] = spec._interpolate(args)

	return spec

def mpi_synthesize(obs, atmos, pool):
	atmos.build_from_nodes(pool=pool)
	
	spec = atmos.compute_spectra(pool=pool)
	if atmos.sl_atmos is not None:
		sl_spec = atmos.sl_atmos.compute_spectra(pool=pool)

	spec.broaden_spectra(atmos.vmac, pool=pool)
	if atmos.sl_atmos is not None:
		sl_spec.broaden_spectra(atmos.vmac, pool=pool)

	#--- add instrument broadening (if applicable)
	if atmos.instrumental_profile is not None:
		spec.instrumental_broadening(kernel=atmos.instrumental_profile, pool=pool)
		if atmos.sl_atmos is not None:
			sl_spec.instrumental_broadening(kernel=atmos.instrumental_profile, pool=pool)

	#--- add the stray light component:
	if atmos.add_stray_light:
		# get the stray light factor(s)
		if "stray" in atmos.global_pars:
			stray_light = atmos.global_pars["stray"]
			stray_light = np.ones((atmos.nx, atmos.ny, 1)) * stray_light
		elif "stray" in atmos.values:
			stray_light = atmos.values["stray"]
		else:
			stray_light = atmos.stray_light

		# check for HSRA spectrum if we are using the 'hsra' stray light contamination
		sl_spectrum = None
		if atmos.stray_type=="hsra":
			sl_spectrum = atmos.hsra_spec.spec
		if atmos.stray_type=="2nd_component":
			sl_spectrum = sl_spec.spec
		if atmos.stray_type in ["atmos", "spec"]:
			sl_spectrum = atmos.stray_light_spectrum.spec

		spec.add_stray_light(atmos.stray_mode, atmos.stray_type, stray_light, sl_spectrum=sl_spectrum)

	#--- norm spectra
	if atmos.norm:
		if atmos.norm_level==1:
			Ic = spec.I[...,atmos.continuum_idl]
			spec.spec = np.einsum("ij...,ij->ij...", spec.spec, 1/Ic)
		elif atmos.norm_level=="hsra":
			spec.spec /= atmos.icont
		else:
			spec.spec /= atmos.norm_level

	return spec

def initialize_walker_states(nwalkers, ndim, atmos):
	#--- get parameter vector
	p0 = np.empty((nwalkers, ndim))

	# get local parameters
	up = 0
	if not atmos.skip_local_pars:
		Natmos = atmos.nx*atmos.ny
		for parameter in atmos.nodes:
			nnodes = len(atmos.nodes[parameter])
			for idn in range(nnodes):
				low = up
				up += Natmos
				proposal = RNG.normal(
						loc=atmos.values[parameter][...,idn].ravel(order="C"), 
						scale=scales[parameter], 
						size=(nwalkers,Natmos))
				
				# check for lower boundary
				vmin = atmos.limit_values[parameter].min[0]
				if atmos.limit_values[parameter].vmin_dim!=1:
					vmin = atmos.limit_values[parameter].min[idn]
				if parameter not in ["gamma", "chi"]:
					proposal[proposal<vmin] = vmin
				
				# check for upper boundary
				vmax = atmos.limit_values[parameter].max[0]
				if atmos.limit_values[parameter].vmax_dim!=1:
					vmax = atmos.limit_values[parameter].max[idn]
				if parameter not in ["gamma", "chi"]:
					proposal[proposal>vmax] = vmax
				
				if parameter=="gamma":
					proposal = np.cos(proposal)
				if parameter=="chi":
					proposal = np.sin(proposal)#* np.sin(proposal)
				p0[:, low:up] = proposal

	# get global parameters
	for parameter in atmos.global_pars:
		npars = atmos.global_pars[parameter].shape[-1]
		if npars>0 and not atmos.skip_global_pars:
			low = up
			up += npars
			proposal = RNG.normal(
								loc=atmos.global_pars[parameter][0,0].ravel(order="C"),
								scale=scales[parameter],
								size=(nwalkers,npars))
			p0[:, low:up] = proposal

	return p0

def save_mcmc_results(chains, acceptance_fraction, log_probabilities, fpath="mcmc_results.fits", **kwargs):
	primary = fits.PrimaryHDU(chains)
	primary.name = "chains"
	total_n_steps, ndim = chains.shape
	primary.header["NAXIS1"] = (ndim, "number of free parameters")
	primary.header["NAXIS2"] = (total_n_steps, "total number of samples")
	primary.header["MEAN_AF"] = (acceptance_fraction, "average acceptance fraction over all walkers")
	
	for key, value in kwargs.items():
		primary.header[key.upper()] = value

	hdulist = fits.HDUList([primary])

	hdu = fits.ImageHDU(log_probabilities)
	hdu.name = "probabilities"
	primary.header["NAXIS1"] = (ndim, "number of free parameters")
	primary.header["NAXIS2"] = (total_n_steps, "total number of samples")
	hdulist.append(hdu)
	
	hdulist.writeto(fpath, overwrite=True)

class Chains(object):
	def __init__(self, fpath):
		self.read(fpath)

	def read(self, fpath):
		hdu = fits.open(fpath)

		self.chains = hdu[0].data
		self.shape = self.chains.shape
		self.nwalkers = hdu[0].header["NWALKERS"]
		self.nsteps = hdu[0].header["NSTEPS"]
		self.move = hdu[0].header["MOVE"]
		self.ntotal_samples = self.shape[0]

	def burn_chains(self, burn):
		idi = int(burn*self.ntotal_samples)
		return self.chains[idi:]
