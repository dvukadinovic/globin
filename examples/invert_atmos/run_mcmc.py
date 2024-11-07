import globin
from schwimmbad import MPIPool, MultiPool
import emcee
import copy
import sys

with MPIPool() as pool:
	run_name = "mcmc"

	inverter = globin.Inverter(verbose=False)
	inverter.read_input(run_name=run_name, globin_input_name="params.input.mcmc")

	obs = inverter.observation.extract([0,1], [0,1])
	obs.weights = inverter.weights
	obs.noise_stokes = inverter._estimate_noise_level(obs.nx, obs.ny, obs.nw)
	if inverter.wavs_weight is not None:
		obs.wavs_weight = inverter.wavs_weight
	else:
		obs.wavs_weight = 1

	atmos = inverter.atmosphere.extract([0,1], [0,1], [0,None])
	atmos.skip_local_pars = False
	atmos.skip_global_pars = True

	if atmos.skip_local_pars and atmos.skip_global_pars:
		raise ValueError("Niether local nor global parameters will be inferred. Change one of the flags.")

	move = emcee.moves.StretchMove(a=2)
	nwalkers = 10

	filename = f"runs/{run_name}/sampler.h5"
	backend = emcee.backends.HDFBackend(filename)#, name="second_1000it")

	nodes = copy.copy(atmos.nodes)
	values = copy.copy(atmos.values)
	atmos.nodes = {}
	atmos.values = {}
	for parameter in ["gamma", "chi"]:
		atmos.nodes[parameter] = nodes[parameter]
		atmos.values[parameter] = values[parameter]

	atmos, spec = globin.invert.invert_mcmc(obs, atmos, move, backend,
											weights=inverter.weights,
											reset_backend=True,
											nsteps=3000,
											nwalkers=nwalkers,
											pool=pool,
											progress_frequency=100,
											)

# atmos.save_atmosphere(f"runs/{run_name}/inverted_atmos_mcmc.fits")
# atmos.save_atomic_parameters(f"runs/{run_name}/inverted_atoms_mcmc.fits")
# spec.save(f"runs/{run_name}/inverted_spectra_cmcmc.fits")
