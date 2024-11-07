import globin
from schwimmbad import MPIPool
import emcee

with MPIPool() as pool:
	inverter = globin.Inverter(verbose=False)
	inverter.read_input(run_name="mcmc")

	moves = emcee.moves.StretchMove(a=2)
	backend = emcee.backends.HDFBackend("runs/mcmc/test_w.h5")

	nsteps = 2000
	nwalkers = 12

	atmos = inverter.atmosphere
	atmos.skip_local_pars = False
	atmos.skip_global_pars = True

	obs = inverter.observation
	obs.weights = inverter.weights
	obs.wavs_weight = 1
	obs.noise_stokes = 1e-3

	globin.invert.invert_mcmc(obs, 
				atmos, 
				move=moves, 
				backend=backend, 
				reset_backend=True, 
				weights=inverter.weights, 
				noise=1e-3, 
				nsteps=nsteps, 
				nwalkers=nwalkers, 
				pool=pool, 
				progress_frequency=100)
