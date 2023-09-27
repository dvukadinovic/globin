import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys
import copy

import globin

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
inverter.read_input(run_name="m3")
inv_atmos, inv_spec, chi2 = inverter.run()
sys.exit()

idx, idy = 0,0
inv = None
obs = inv_spec
if inverter.mode>=1:
    obs = inverter.observation
    inv = inv_spec.spec[idx,idy]
globin.visualize.plot_spectra(obs.spec[idx,idy], obs.wavelength, inv=inv)

# atmos = globin.Atmosphere("atmos_bezier.fits", atm_range=[0,1,0,1])
# globin.visualize.plot_atmosphere(inv_atmos, parameters=["temp", "ne", "nH"], reference=atmos)
globin.show()
