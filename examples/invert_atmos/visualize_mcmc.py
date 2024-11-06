import globin
import matplotlib.pyplot as plt
import numpy as np
import corner
import emcee
import sys

reader = emcee.backends.HDFBackend("runs/mcmc/sampler.h5")
print(f"Total iterations: {reader.iteration}")

tau = reader.get_autocorr_time(tol=0, quiet=True)
print(f"Autocorrelation time: {tau}")

burnin = int(2 * np.max(tau))
thin = 1#int(0.5 * np.min(tau))
chains = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
print(f"Chain shape: {chains.shape}")

chains[...,0] = np.arccos(chains[...,0])*180/np.pi
chains[...,1] = np.arcsin(chains[...,1])*180/np.pi

# fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
# axs[0].plot(chains[:,:,0], c="k", alpha=0.3)
# axs[1].plot(chains[:,:,1], c="k", alpha=0.3)
# plt.show()

# sys.exit()

# print(log_prob.shape)
log_prob[np.isinf(log_prob)] = np.nan
ind_min = np.nanargmin(log_prob)
ind_max = np.nanargmax(log_prob)
# print(chains[ind_min])
# print(chains[ind_max])
# plt.hist(log_prob, bins=14, histtype="step")
# plt.show()

for idp in range(2):
	mcmc = np.percentile(chains[:, idp], [16, 50, 84])
	q = np.diff(mcmc)
	print(f"{mcmc[1]:.3f}  {q[0]:.3f}  {q[1]:.3f} | {np.std(chains[:,idp]):.3f}")

# plt.hist(chains[...,3], bins=15, histtype="step")
# plt.plot(chains[...,0], c="k", alpha=0.3)
corner.corner(chains, 
	# labels=[r"$v_\mathrm{mic}$"],
	labels=["gamma", "phi"],
	quantiles=[0.16, 0.5, 0.8],
	shot_titles=True,
	# truths=chains[ind_max]
	)
plt.show()

sys.exit()

wavelength_mask = np.loadtxt("obs/wavelength_mask", unpack=True, usecols=(1,2,3,4))
_range = [0,1,0,1]
obs = globin.Observation("runs/m3_20x88_spec_sl_HS/obs_50p_best.fits", obs_range=_range, spec_type="hinode")
obs.mask(wavelength_mask)
print(obs.shape)

inv = globin.Observation("runs/errors_estimate_atmos/inverted_spectra_cmcmc.fits")
inv_LM = globin.Observation("runs/m3_20x88_spec_sl_QS/inverted_spectra_c1.fits", obs_range=_range)

globin.plot_spectra(inv_LM.spec[0,0], inv_LM.wavelength, 
	inv=[inv.spec[0,0]],
	labels=["obs", "mcmc", "LM"])
globin.show()