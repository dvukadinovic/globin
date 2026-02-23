import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys
import copy
import globin

inverter = globin.Inverter(verbose=False)
inverter.read_input(run_name="dummy")

inverter.atmosphere.atomic_number = np.array([26], dtype=np.int32)
inverter.atmosphere.atomic_abundance = np.array([7.44], dtype=np.float64)

if inverter.mode==0:
    # atmos = inverter.atmosphere
    # atmos.get_populations = False

    spec = inverter.run()

    # obs = globin.Observation("obs_630_mu1_abs_LTE_20240918.fits")

    # print(atmos.n_pops)

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
    print(inv_atmos.global_pars["loggf"][0,0])
    print(chi2.get_final_chi2()[0])
    print(inv_atmos.values)

sys.exit()

print(inv_atmos.global_pars["loggf"][0,0])

idx, idy = 1, 0

for parameter in inverter.atmosphere.nodes:
    print(inverter.reference_atmosphere.values[parameter])
    print(inv_atmos.values[parameter])

# inv = globin.Observation("runs/dummy/inverted_spectra_c1.fits")
globin.visualize.plot_spectra(inverter.observation.spec[idx,idy], 
                              inverter.observation.wavelength, 
                              inv=[inv.spec[idx,idy]], 
                              labels=["obs", "inv"])

# inv_atmos = globin.Atmosphere("runs/dummy/inverted_atmos_c1.fits")
globin.visualize.plot_atmosphere(inv_atmos, 
                                 parameters=["temp", "vz", "mag", "vmic", "gamma", "chi"], 
                                 idx=idx, 
                                 idy=idy, 
                                 reference=inverter.reference_atmosphere)

globin.show()
