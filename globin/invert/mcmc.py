import numpy as np
import emcee
from astropy.io import fits
import matplotlib.pyplot as plt
from time import time
import logging

logger = logging.getLogger(__name__)

import globin
from ..parallel_methods import _build_from_nodes, _makeHSE
from ..constants import LIGHT_SPEED

RNG = np.random.default_rng()
scales = {"temp"  : 50,   # [K]
		  "vz"    : 0.1,	   # [km/s]
		  "vmic"  : 0.1,	   # [km/s]
		  "mag"   : 50,   # [G]
		  "gamma" : 1,	   # [rad]
		  "chi"   : 1,	   # [rad]
		  "of"    : 1e-3,		# 
		  "stray" : 1e-2,		#
		  "sl_vz" : 0.1,       # [km/s]
		  "sl_vmic": 0.1,      # [km/s]
		  "vmac"  : 0.1,		# [km/s]
		  "loggf" : 0.01,		#
		  "dlam"  : 2}			#


def lnprior_gaussian(x, loc, std):
	return -((x - loc) / std) ** 2 - np.log(std * np.sqrt(np.pi))

def lnprior_uniform(x, low, high):
	if x < low or x > high:
		return -np.inf
	return 0

def invert_mcmc(obs, atmos, move, backend, reset_backend=True, weights=np.array([1,1,1,1]), noise=1e-3, nsteps=100, nwalkers=2, pool=None, sequential=True, progress_frequency=100):
	logger.info("\n{:{char}{align}{width}}\n".format(f" Entering MCMC inversion mode ", char="-", align="^", width=globin.constants.NCHAR))

	atmos.limit_values["gamma"] = globin.atmos.MinMax(-1,1)
	atmos.limit_values["chi"] = globin.atmos.MinMax(-1,1)

	Natmos = atmos.nx*atmos.ny

	atmos.n_local_pars = 0
	if not atmos.skip_local_pars:
		for parameter in atmos.nodes:
			atmos.n_local_pars += len(atmos.nodes[parameter])

	ndim = Natmos*atmos.n_local_pars
	if not atmos.skip_global_pars:
		ndim += atmos.n_global_pars
	
	logger.info("\n{:{char}{align}{width}}\n".format(f" Info ", char="-", align="^", width=globin.constants.NCHAR))
	logger.info("atmos.shape {:{char}{align}{width}}".format(f" {atmos.shape}", char=".", align=">", width=20))
	if not atmos.skip_local_pars:
		logger.info("N_local_pars {:{char}{align}{width}}".format(f" {atmos.n_local_pars}", char=".", align=">", width=20))
	if not atmos.skip_global_pars:
		logger.info("N_global_pars {:{char}{align}{width}}".format(f" {atmos.n_global_pars}", char=".", align=">", width=20))
	logger.info("Nwalkers {:{char}{align}{width}}".format(f" {nwalkers}", char=".", align=">", width=20))
	logger.info("Nsteps {:{char}{align}{width}}\n".format(f" {nsteps}", char=".", align=">", width=20))
	
	if reset_backend:
		# ndim += 1
		backend.reset(nwalkers, ndim)
		p0 = initialize_walker_states(nwalkers, ndim, atmos)
		# p0[:, -1] = np.random.normal(loc=noise, scale=noise/10, size=nwalkers)
		# p0[:, -1][p0[:, -1]<0] = np.min(p0[:, -1][p0[:, -1]>0])
		# p0[:, -1] = np.log10(p0[:, -1])
	else:
		p0 = backend.get_last_sample()

	# set some attributes that can be useful later for analysis
	attributes = {"noise" : noise, 
				  "local_parameters" : [],
				  "number_of_nodes" : [],
				  "global_parameters" : [],
				  "loggf_ID" : [],
				  "dlam_ID" : [],
				  "truths" : []}
	for parameter in atmos.nodes:
		attributes["local_parameters"].append(parameter)
		attributes["number_of_nodes"].append(len(atmos.nodes[parameter]))
		attributes["truths"] += atmos.values[parameter][0,0].tolist()
	for parameter in atmos.global_pars:
		if len(atmos.line_no[parameter])==0:
			continue

		attributes["global_parameters"].append(parameter)
		if parameter=="loggf":
			attributes["loggf_ID"] = atmos.line_no[parameter].tolist()
		if parameter=="dlam":
			attributes["dlam_ID"] = atmos.line_no[parameter].tolist()
		attributes["truths"] += atmos.global_pars[parameter][0,0].tolist()
	
	backend.set_attributes(attributes)

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, 
		args=[obs, atmos, None if sequential else pool], 
		moves=move, 
		pool=pool if sequential else None,
		backend=backend)

	for sample in sampler.sample(initial_state=p0, thin_by=1, iterations=nsteps, progress=True, store=True):
		if sampler.iteration%progress_frequency:
			continue

		# plt.close()
		# globin.plot_spectra(obs.spec[0,0], np.arange(obs.nw), 
		# 			  inv=atmos.spectrum.spec[0,0])
		# globin.show()

		tau = sampler.get_autocorr_time(tol=0, has_walkers=False, quiet=True)

		Neff = sampler.iteration / np.mean(tau)

		logger.info(f"\nAR: {np.mean(sampler.acceptance_fraction):.3f} | ACT = {np.mean(tau):.2f} | Neff = {Neff:.1f} \n")

		if Neff>50:
			break

def lnprior(local_pars, global_pars, limits, priors):
	"""
	Check if each parameter is in its respective bounds given by globin.limit_values.

	If one fails, return -np.inf, else return 0.
	"""	
	_lnprior = 0.0
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
						if limits[parameter].ndim==2:
							limit = limits[parameter][idl]
						
						if priors[parameter][idl]=="uniform":
							_lnprior += lnprior_uniform(global_pars[parameter][0,0,idl], limit[0], limit[1])
						if priors[parameter][idl]=="gaussian":
							_lnprior += lnprior_gaussian(global_pars[parameter][0,0,idl], limit[0], limit[1])

	return _lnprior

def lnlike(obs, atmos, pool, noise_factor):
	if pool is None:
		if not atmos.skip_local_pars:
			params = list(atmos.nodes.keys())
			params = [p for p in params if p not in ["stray", "sl_temp", "sl_vz", "sl_vmic"]]
			args = atmos.prepare_build_from_nodes_arguments(params)
			if "temp" in params:
				args_HSE = atmos.prepare_HSE_arguments()
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					ida = idx*atmos.ny + idy
					result = _build_from_nodes(args[ida])
					for idp in range(len(params)):
						# if params[idp] in ["stray", "sl_temp", "sl_vz", "sl_vmic"]:
						# 	continue
						atmos.data[idx,idy,atmos.par_id[params[idp]]] = result[idp]
					if "temp" in params:
						ne, nH = _makeHSE(args_HSE[ida])
						atmos.data[idx,idy,atmos.par_id["ne"]] = ne
						atmos.data[idx,idy,atmos.par_id["nH"]] = nH

		spec = sequential_synthesize(atmos)
	else:
		spec = mpi_synthesize(atmos, pool)

	# spec.spec += np.random.normal(loc=0, scale=10**noise_factor, size=spec.spec.shape)

	# spec.add_noise(np.sqrt(obs.noise**2))# + obs.noise_parameter**2))

	# fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
	# axs[0].set_title(f"{atmos.global_pars['loggf'][0,0,0]:.3f}")
	# axs[0].set_title(f"{atmos.values['vmic'][0,0,0]:.3f}")
	# axs[0].plot(obs.I[0,0])
	# axs[0].plot(spec.I[0,0])
	# axs[1].plot(obs.I[0,0] - spec.I[0,0])
	# plt.show()

	diff = obs.spec - spec.spec
	diff *= obs.weights
	diff *= obs.wavs_weight
	diff2 = diff**2

	noise = obs.noise_stokes
	noise2 = noise**2
	
	diff2 /= noise2
	diff2 -= np.log(2*np.pi*noise2)

	chi2 = np.sum(diff2)
	
	return -chi2/2

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
				correction = theta[low:up]
				atmos.global_pars[parameter][:,:] = correction
				if atmos.sl_atmos is not None:
					atmos.sl_atmos.global_pars[parameter][0,0] = correction
			if parameter=="stray":
				atmos.global_pars[parameter] = theta[low:up]

	#--- compute posterior
	if atmos.skip_local_pars:
		lp = lnprior(None, atmos.global_pars, atmos.limit_values, atmos.priors)
	elif atmos.skip_global_pars:
		lp = lnprior(atmos.values, None, atmos.limit_values, atmos.priors)
	else:
		lp = lnprior(atmos.values, atmos.global_pars, atmos.limit_values, atmos.priors)

	# prior for log10(noise)
	# if theta[-1] < -6 or theta[-1] > -2:
	# 	lp = -np.inf

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

	return lp + lnlike(obs, atmos, pool, theta[-1])

def sequential_synthesize(atmos):
	# compute spectra
	if atmos.vmac!=0:
		kernel = globin.utils.get_kernel(atmos.vmac, atmos.wavelength_obs, order=0)

	args = atmos.prepare_synthesis_args(get_atomic_rfs=False)
	if atmos.sl_atmos is not None:
		args_sl = atmos.sl_atmos.prepare_synthesis_args(get_atomic_rfs=False)
	
	for idx in range(atmos.nx):
		for idy in range(atmos.ny):
			stokes_vector = globin.parallel_methods._compute_spectra_sequential(args[idx*atmos.ny+idy])
			if atmos.sl_atmos is not None:
				sl_spec = globin.parallel_methods._compute_spectra_sequential(args_sl[idx*atmos.ny+idy])
			if atmos.vmac!=0:
				stokes_vector = globin.spec._broaden_spectra((stokes_vector, kernel))
				if atmos.sl_atmos is not None:
					sl_spec = globin.spec._broaden_spectra((sl_spec, kernel))
			if atmos.instrumental_profile is not None:
				stokes_vector = globin.spec._broaden_spectra((stokes_vector, atmos.instrumental_profile))
				if atmos.sl_atmos is not None:
					sl_spec = globin.spec._broaden_spectra((sl_spec, atmos.instrumental_profile))
			atmos.spectrum.spec[idx,idy] = stokes_vector
			if atmos.sl_atmos is not None:
				atmos.sl_atmos.spectrum.spec[idx,idy] = sl_spec

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
			sl_spectrum = atmos.sl_atmos.spectrum.spec
		if atmos.stray_type in ["atmos", "spec"]:
			sl_spectrum = atmos.stray_light_spectrum.spec

		atmos.spectrum.add_stray_light(atmos.stray_mode, atmos.stray_type, stray_light, sl_spectrum=sl_spectrum)

	if atmos.norm:
		if atmos.norm_level==1:
			Ic = atmos.spectrum.I[...,atmos.continuum_idl]
			atmos.spectrum.spec = np.einsum("ij...,ij->ij...", atmos.spectrum.spec, 1/Ic)
		elif atmos.norm_level=="hsra":
			atmos.spectrum.spec /= atmos.icont
		else:
			atmos.spectrum.spec /= atmos.norm_level

	#--- downsample the synthetic spectrum to observed wavelength grid
	# This will fail when I have to use it...
	if not np.array_equal(atmos.wavelength_obs, atmos.wavelength_air):
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				args = atmos.spectrum.spec[idx,idy], atmos.wavelength_obs, "extrapolate"
				atmos.spectrum.spec[idx,idy] = atmos.spectrum._interpolate(args)

	return atmos.spectrum

def mpi_synthesize(atmos, pool):
	atmos.build_from_nodes(pool=pool)
	atmos.makeHSE(pool=pool)
	
	atmos.compute_spectra(pool=pool)
	if atmos.sl_atmos is not None:
		atmos.sl_atmos.compute_spectra(pool=pool)

	atmos.degrade_spectra(pool=pool)

	return atmos.spectrum

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
			#if parameter=="dlam":
			#	p0[:, low:up] *= LIGHT_SPEED/1e3 / np.mean(atmos.wavelength_obs*1e4) # [mA] -> [km/s]

	return p0

class Storage(emcee.backends.HDFBackend):
	def __init__(self, filename, name="mcmc", read_only=False):
		super().__init__(filename, name=name, read_only=read_only)
		self.loaded_attrs = False

	@property
	def noise(self):
		with self.open(mode="r") as f:
			return f[self.name].attrs["noise"]

	@property
	def local_parameters(self):
		with self.open(mode="r") as f:
			return f[self.name].attrs["local_parameters"]

	@property
	def global_parameters(self):
		with self.open(mode="r") as f:
			return f[self.name].attrs["global_parameters"]

	@property
	def number_of_nodes(self):
		with self.open(mode="r") as f:
			return f[self.name].attrs["number_of_nodes"]

	@property
	def loggf_ID(self):
		with self.open(mode="r") as f:
			return f[self.name].attrs["loggf_ID"]

	@property
	def dlam_ID(self):
		with self.open(mode="r") as f:
			return f[self.name].attrs["dlam_ID"]

	@property
	def truths(self):
		with self.open(mode="r") as f:
			return f[self.name].attrs["truths"]

	def load_attributes(self):
		with self.open(mode="r") as f:
			self.attrs = f[self.name].attrs
			attributes = {}
			for key in self.attrs.keys():
				attributes[key] = self.attrs[key]

		self.loaded_attrs = True

		return attributes

	def set_attributes(self, attributes):
		with self.open(mode="a") as f:
			self.attrs = f[self.name].attrs
			for key, value in attributes.items():
				self.attrs[key] = value
