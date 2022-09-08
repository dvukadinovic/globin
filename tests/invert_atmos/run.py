import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys

import globin

# obs = globin.Observation("obs.fits")
# atmos = globin.Atmosphere("atmos_HSE_pyrh.fits")

# inv = globin.Atmosphere("runs/mode3_test/inverted_atmos.fits")
# # inv_spec = globin.Observation("runs/mode3_test/inverted_spectra.fits")
# inv_spec = globin.Observation("spectrum.fits")

# params = ["temp", "vz", "vmic", "mag", "gamma", "chi"]
# # params = ["ne", "nH"]
# for idx in range(atmos.nx):
# 	for idy in range(atmos.ny):
# 		# globin.visualize.plot_atmosphere(atmos, params, idx=idx, idy=idy)
# 		# globin.visualize.plot_atmosphere(inv, params, idx=idx, idy=idy, color="tab:red")
# 		# plt.show()

# 		globin.visualize.plot_spectra(obs.spec[idx,idy], obs.wavelength, inv=inv_spec.spec[idx,idy])
# 		plt.show()

# sys.exit()

inverter = globin.Inverter(verbose=True)
inverter.read_input(run_name="dummy")
inv_atmos, inv_spec = inverter.run()

# inv_atmos.compare(atmos.data[0,0], idx=0, idy=0)
# inv_atmos.compare(atmos.data[0,1], idx=0, idy=1)
# inv_atmos.compare(atmos.data[0,2], idx=0, idy=2)

# plt.plot(inv_atmos.data[0,0,2])

# plt.plot(inv_atmos.data[0,1,2])
# # plt.plot(inv_atmos.data[0,1,2])
# # plt.show()

# plt.plot(inv_atmos.data[0,2,2])
# # plt.plot(inv_atmos.data[0,2,2])

# plt.yscale("log")
# plt.show()

# for idx in range(inv_atmos.nx):
# 	for idy in range(inv_atmos.ny):
# 		atmos.compare(inv_atmos.data[idx,idy], idx=idx, idy=idy)
# 		print("-----=-----=-----")

# for idx in range(inv_spec.nx):
# 	for idy in range(inv_spec.ny):
# 		globin.visualize.plot_spectra(inverter.observation.spec[idx,idy], inverter.observation.wavelength, inv=inv_spec.spec[idx,idy])
# 		plt.show()