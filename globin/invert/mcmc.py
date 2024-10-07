import numpy as np
import emcee
from astropy.io import fits
import matplotlib.pyplot as plt

import globin
from globin.inversion import Inverter

RNG = np.random.default_rng()
scales = {"temp"  : 10,			# [K]
		  "vz"    : 0.05,		# [km/s]
		  "vmic"  : 0.05,		# [km/s]
		  "mag"   : 10,			# [G]
		  "gamma" : np.pi/360,	# [rad]
		  "chi"   : np.pi/360,	# [rad]
		  "of"    : 0.01,		# 
		  "stray" : 0.01,		#
		  "vmac"  : 0.05,		# [km/s]
		  "loggf" : 0.01,		#
		  "dlam"  : 2}			#

def invert_mcmc(run_name, nsteps=100, nwalkers=2, pool=None, skip_local_pars=False, skip_global_pars=True, move=None, a=2):

	print("\n{:{char}{align}{width}}\n".format(f" Entering MCMC inversion mode ", char="-", align="^", width=globin.NCHAR))

	inverter = Inverter(verbose=False)
	inverter.read_input(run_name=run_name)

	obs = inverter.observation
	atmos = inverter.atmosphere

	if atmos.sl_atmos is not None:
		obs.sl_spec = atmos.sl_atmos.compute_spectra()

	atmos.limit_values["gamma"] = globin.atmos.MinMax(-1,1)
	atmos.limit_values["chi"] = globin.atmos.MinMax(0,1)

	atmos.skip_local_pars = skip_local_pars
	atmos.skip_global_pars = skip_global_pars

	if atmos.skip_local_pars and atmos.skip_global_pars:
		raise ValueError("Niether local nor global parameters will be inferred. Change one of the flags.")

	if atmos.stray_type=="hsra" or atmos.norm_level=="hsra":
		atmos.get_hsra_cont()
		if atmos.stray_type=="hsra":
			atmos.hsra_spec.broaden_spectra(atmos.vmac)

	Natmos = atmos.nx*atmos.ny
	obs.Ndof = 4*obs.nw*Natmos - 1

	atmos.n_local_pars = 0
	if not atmos.skip_local_pars:
		for parameter in atmos.nodes:
			atmos.n_local_pars += len(atmos.nodes[parameter])

	ndim = Natmos*atmos.n_local_pars
	if not atmos.skip_global_pars:
		ndim += atmos.n_global_pars
	if nwalkers<2*ndim:
		raise ValueError("Number of walkers is less than 2*number_of_parameters. StretchMove() will throw an error.")
	
	p0 = initialize_walker_states(nwalkers, ndim, atmos)

	#--- create the move
	if move is None:
		move = emcee.moves.StretchMove(a=a)

	print("\n{:{char}{align}{width}}\n".format(f" Info ", char="-", align="^", width=globin.NCHAR))
	print("run_name {:{char}{align}{width}}".format(f" {run_name}", char=".", align=">", width=20))
	print("atmos.shape {:{char}{align}{width}}".format(f" {atmos.shape}", char=".", align=">", width=20))
	if not atmos.skip_local_pars:
		print("N_local_pars {:{char}{align}{width}}".format(f" {atmos.n_local_pars}", char=".", align=">", width=20))
	if not atmos.skip_global_pars:
		print("N_global_pars {:{char}{align}{width}}".format(f" {atmos.n_global_pars}", char=".", align=">", width=20))
	print("Nwalkers {:{char}{align}{width}}".format(f" {nwalkers}", char=".", align=">", width=20))
	print("Nsteps {:{char}{align}{width}}\n".format(f" {nsteps}", char=".", align=">", width=20))

	noise = 1e-3
	noise_stokes = np.ones((obs.nx, obs.ny, obs.nw, 4))
	StokesI_cont = np.quantile(obs.I, 0.9, axis=2)
	noise_stokes = np.einsum("ijkl,ij->ijkl", noise_stokes, noise*StokesI_cont)
	if inverter.wavs_weight is not None:
		obs.wavs_weight = inverter.wavs_weight
	else:
		obs.wavs_weight = 1
	obs.noise_stokes = noise_stokes
	obs.weights = inverter.weights

	filename = f"runs/{run_name}/MCMC_sampler_results.h5"
	backend = emcee.backends.HDFBackend(filename)
	backend.reset(nwalkers, ndim)
	
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, 
		args=[obs, atmos], 
		moves=move, 
		pool=pool,
		backend=backend)

	check_every_nth = 100

	old_tau = np.inf
	for sample in sampler.sample(initial_state=p0, iterations=nsteps, progress=True, store=True):
		if sampler.iteration%check_every_nth:
			continue

		print(f"\nAR: {np.mean(sampler.acceptance_fraction):.3f}")	

		if sampler.iteration%(5*check_every_nth):
			tau = sampler.get_autocorr_time(has_walkers=False, quiet=True)
		
		print()

		# check convergence
		converged = np.all(tau * 100 < sampler.iteration)
		converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
		if converged:
			break
		old_tau = tau
	
	burnin = 0
	thin = 1
	chains = sampler.get_chain(discard=burnin, flat=True, thin=thin)
	log_prob0 = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
	log_prob0[np.isinf(log_prob0)] = np.nan
	ind_max = np.nanargmax(log_prob0)
	theta_best = chains[ind_max]

	#--- update parameters
	up = 0
	if not atmos.skip_local_pars:
		for parameter in atmos.nodes:
			nnodes = len(atmos.nodes[parameter])
			low = up
			up += Natmos*nnodes
			atmos.values[parameter][:,:,:] = theta_best[low:up].reshape(atmos.nx, atmos.ny, nnodes, order="C")

	for parameter in atmos.global_pars:
		npars = atmos.global_pars[parameter].shape[-1]
		if npars>0 and not atmos.skip_global_pars:
			low = up
			up += npars
			if parameter in ["loggf", "dlam"]:
				atmos.global_pars[parameter][0,0] = theta_best[low:up]
			if parameter=="stray":
				atmos.global_pars[parameter] = theta_best[low:up]

	spec = atmos.compute_spectra()
	spec.broaden_spectra(vmac=atmos.vmac, n_thread=atmos.n_thread)
	if atmos.instrumental_profile is not None:
		inverted_spectra.instrumental_broadening(kernel=atmos.instrumental_profile, n_thread=atmos.n_thread)
	if not np.array_equal(atmos.wavelength_obs, atmos.wavelength_air):
		spec.interpolate(atmos.wavelength_obs, atmos.n_thread)
	spec.save(f"runs/{run_name}/inverted_spectra_cmcmc.fits")

	atmos.save_atmosphere(f"runs/{run_name}/inverted_atmos_mcmc.fits")
	atmos.save_atomic_parameters(f"runs/{run_name}/inverted_atoms_mcmc.fits")

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
				if Npar>0:
					for idl in range(Npar):
						#--- check lower boundary condition
						indx, indy = np.where(global_pars[parameter][...,idl]<limits[parameter][idl,0])
						if len(indx)>0:
							return -np.inf

						#--- check upper boundary condition
						indx, indy = np.where(global_pars[parameter][...,idl]>limits[parameter][idl,1])
						if len(indx)>0:
							return -np.inf					

	return 0.0

def lnlike(obs, atmos):
	atmos.build_from_nodes()
	spec = atmos.compute_spectra()

	spec.broaden_spectra(vmac=atmos.vmac, n_thread=atmos.n_thread)

	if atmos.instrumental_profile is not None:
		inverted_spectra.instrumental_broadening(kernel=atmos.instrumental_profile, n_thread=atmos.n_thread)

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
		spec.interpolate(atmos.wavelength_obs, atmos.n_thread)

	# if atmos.values["vmic"][0,0,0]>0.5:
	# plt.plot(obs.I[0,0])
	# plt.plot(spec.I[0,0])
	# # 	plt.plot(diff[0,0,:,0])
	# plt.show()

	diff = obs.spec - spec.spec
	diff *= obs.weights
	diff *= obs.wavs_weight
	diff /= obs.noise_stokes
	chi2 = np.sum(diff**2)
	chi2 /= obs.Ndof

	return chi2 * (-0.5)

def log_prob(theta, obs, atmos):
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
			proposal = np.arcsin(np.sqrt(atmos.values["chi"]))
			atmos.values["chi"][:,:,:] = proposal
		
		# compute the inclination from the cos(gamma)
		if "gamma" in atmos.nodes:
			proposal = np.arccos(atmos.values["gamma"])
			atmos.values["gamma"][:,:,:] = proposal

	return lp + lnlike(obs, atmos)

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
					proposal = np.sin(proposal) * np.sin(proposal)
				p0[:, low:up] = proposal

	# get global parameters
	for parameter in atmos.global_pars:
		npars = atmos.global_pars[parameter].shape[-1]
		if npars>0 and not atmos.skip_global_pars:
			low = up
			up += npars
			p0[:, low:up] = RNG.normal(
								loc=atmos.global_pars[parameter].ravel(),
								scale=scales[parameter],
								size=(nwalkers,npars))

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
