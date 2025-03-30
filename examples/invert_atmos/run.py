import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys
import copy
import globin

# atmos = globin.falc

# atmos.load_populations("populations_Fe52_FALC.fits")
# # atmos.load_populations("populations_FeEdvarda_FALC.fits")

# nI = np.sum(atmos.n_pops["FE"][0,0,:-2], axis=0)
# nII = atmos.n_pops["FE"][0,0,-2]
# nIII = atmos.n_pops["FE"][0,0,-1]
# n = nI + nII + nIII

# dc = atmos.n_pops["FE"][0,0]/atmos.nstar_pops["FE"][0,0]

# x = atmos.height[0,0]/1e3

# # plt.plot(x, dc[0])
# # plt.plot(x, dc[22])

# # plt.plot(x, dc[8], c="C0", label="6015.18")
# # plt.plot(x, dc[27], c="C0", ls="--")

# # plt.plot(x, dc[2], c="C0", label="5247.06")
# # plt.plot(x, dc[11], c="C0", ls="--")

# plt.plot(x, dc[4], c="C1", label="5250.11")
# plt.plot(x, dc[13], c="C1", ls="--")

# plt.plot(x, dc[7], c="C2", label="5250.64")
# plt.plot(x, dc[29], c="C2", ls="--")

# plt.ylim([1e-1, 5e2])
# plt.yscale("log")

# # plt.plot(x, nI/n*100)

# nI = np.sum(atmos.nstar_pops["FE"][0,0,:-2], axis=0)
# nII = atmos.nstar_pops["FE"][0,0,-2]
# nIII = atmos.nstar_pops["FE"][0,0,-1]
# n = nI + nII + nIII

# # plt.plot(x, nI/n*100)

# ax = plt.gca().twinx()
# ax.plot(x, atmos.T[0,0], c="k")
# ax.set_ylabel("Temperature [K]")

# plt.xlabel("Height [Mm]")
# plt.ylabel("Departure coefficients")
# # plt.xlim([x.min(), 1.5])
# # plt.gca().invert_xaxis()

# plt.legend(frameon=True)

# plt.show()

# sys.exit()

inverter = globin.Inverter(verbose=True)
inverter.read_input(run_name="dummy")
if inverter.mode==0:
    atmos = inverter.atmosphere
    atmos.get_populations = False

    spec = inverter.run()

    # obs = globin.Observation("obs_630_mu1_abs_LTE_20240918.fits")

    print(atmos.n_pops)

    # atmos.save_populations("populations_FeEdvarda_FALC.fits")

    # n = atmos.n_pops["FE"]
    # print(n.shape)
    # nstar = atmos.nstar_pops["FE"]
    # dc = n/nstar
    # dc = dc[0,0]

    # plt.figure(1)
    # plt.plot(atmos.logtau, dc[4])
    # plt.plot(atmos.logtau, dc[11])
    # plt.yscale("log")
    # plt.gca().invert_xaxis()

    globin.plot_spectra(spec.spec[0,0], spec.wavelength)
    #     inv=[obs.spec[0,0]], 
    #     labels=["new", "old"])
    globin.show()

    sys.exit()
else:
    inv_atmos, inv, chi2 = inverter.run()

idx, idy = 0,0
# inv = globin.Observation("runs/dummy/inverted_spectra_c1.fits")
globin.visualize.plot_spectra(inverter.observation.spec[idx,idy], inverter.observation.wavelength, 
    inv=[inv.spec[idx,idy]], 
    labels=["obs", "inv"])

atmos = globin.Atmosphere("atmos_bezier.fits", atm_range=[0,1,0,1])
# inv_atmos = globin.Atmosphere("runs/dummy/inverted_atmos_c1.fits")
globin.visualize.plot_atmosphere(inv_atmos, parameters=["temp", "vz", "mag", "vmic", "gamma", "chi"], reference=atmos)
globin.show()
