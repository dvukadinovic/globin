import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys
import copy

import globin

from schwimmbad import MPIPool

with MPIPool() as pool:
    globin.inversion.invert_mcmc(run_name="dummy_all", 
            nsteps=8000,
            nwalkers=70,
            pool=pool, 
            skip_global_pars=True)

sys.exit()

# obs = globin.Observation("obs_630.fits")
# for idx in range(obs.nx):
#     for idy in range(obs.ny):
#         obs.spec[idx,idy] /= obs.I[idx,idy,0]
#         obs.spec[idx,idy,:,1:] *= 100
#         globin.plot_spectra(obs.spec[idx,idy], obs.wavelength, labels=[f"({idx},{idy})"])
#         plt.show()

# sys.exit()

#===--- Estimate the spatial regularization relative weighting
#inverter = globin.Inverter(verbose=True)
#inverter.read_input(run_name="reg_test")
# alpha, chi2, chi2_reg = inverter.estimate_regularization_weight(-6, 2, num=9, fpath="reg_weight")

# alpha, chi2, chi2_reg = np.loadtxt("reg_weight", unpack=True)

# plt.plot(alpha, chi2)
# plt.plot(alpha, chi2_reg)
# plt.xscale("log")
# plt.yscale("log")
# plt.show()

# sys.exit()

#===--- Synthesis/Inversion
# obs = globin.Observation("obs_bezier_Fe630_mu1_abs.fits")
# inv = globin.Observation("runs/dummy/inverted_spectra_c1.fits")

# for idx in range(obs.nx):
#     for idy in range(obs.ny):
#         globin.visualize.plot_spectra(obs.spec[idx,idy], obs.wavelength, inv=inv.spec[idx,idy])
#         globin.show()
# sys.exit()

inverter = globin.Inverter(verbose=True)
inverter.read_input(run_name="dummy")
inv_atmos, inv, chi2 = inverter.run()

# sys.exit()

idx, idy = 0,0
# inv = globin.Observation("runs/dummy/inverted_spectra_c1.fits")
globin.visualize.plot_spectra(inverter.observation.spec[idx,idy], inverter.observation.wavelength, 
    inv=[inv.spec[idx,idy]], 
    labels=["obs", "inv"])

atmos = globin.Atmosphere("atmos_bezier.fits", atm_range=[1,2,2,None])
# inv_atmos = globin.Atmosphere("runs/dummy/inverted_atmos_c1.fits")
globin.visualize.plot_atmosphere(inv_atmos, parameters=["temp", "vz", "mag", "gamma", "chi"], reference=atmos)
globin.show()
