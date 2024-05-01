import globin
import matplotlib.pyplot as plt
import numpy as np
import corner
import emcee
import sys
from scipy.stats import skewnorm

reader = emcee.backends.HDFBackend("runs/dummy_all_new_angles/MCMC_sampler_results.h5", read_only=True)
tau = reader.get_autocorr_time(quiet=True)
print(tau)
burnin = int(2 * np.max(tau))
thin = 1#int(0.5 * np.min(tau))
chains = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
# print(log_prob)
print(chains.shape)

chains[:,-2] = np.arccos(chains[:,-2])
chains[:,-1] = np.arcsin(np.sqrt(chains[:,-1]))
chains[:, -2:] *= 180/np.pi

# print(log_prob.shape)
log_prob[np.isinf(log_prob)] = np.nan
ind_min = np.nanargmin(log_prob)
ind_max = np.nanargmax(log_prob)
print(chains[ind_max])
# plt.hist(log_prob, bins=14, histtype="step")
# plt.show()

for idp in range(chains.shape[1]):
	mcmc = np.percentile(chains[:, idp], [16, 50, 84])
	q = np.diff(mcmc)
	# print(mcmc[1], q[0], q[1], np.mean(chains[:,idp]), np.std(chains[:,idp]))
	#print(skewnorm.fit(chains[:,idp]))
	

# plt.hist(chains[...,3], bins=15, histtype="step")
# plt.plot(chains[...,0], c="k", alpha=0.3)
corner.corner(chains[...,4:], 
	show_titles=True,
	# labels=["B1", "B2", "Theta", "Phi"],
	quantiles=[0.16, 0.5, 0.8],
	# truths=[4500, 5000, 6150, 7200]#, 0.75, -0.1, 1.00, 300, 350, 30, 60]
	)
plt.show()

sys.exit()

obs = globin.Observation("obs_630_mu1_abs.fits")
print(obs.shape)

inv = globin.Observation("runs/dummy_vz/inverted_spectra_cmcmc.fits")
inv_LM = globin.Observation("runs/m3_20x88_spec_sl_QS/inverted_spectra_c1.fits", obs_range=_range)

globin.plot_spectra(inv_LM.spec[0,0], inv_LM.wavelength, 
	inv=[inv.spec[0,0]],
	labels=["obs", "mcmc", "LM"])
globin.show()
