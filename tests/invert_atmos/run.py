import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys
import copy

import globin

obs = globin.Observation("obs_uniform.fits")
#atm = globin.Atmosphere("atmos_uniform.fits")

#inv = globin.Atmosphere("runs/dummy/inverted_atmos_c1.fits")

# print(inv.values["temp"][0,0])
# print(atm.values["temp"][0,0])
# print(inv.pg_top, atm.pg_top)

# plt.plot(inv.data[0,0,2] - atm.data[0,0,2])
# plt.show()

# sys.exit()

# obs2 = globin.Observation("obs_uniform_norm_sl3p.fits")

# for idx in range(obs.nx):
# 	for idy in range(obs.ny):
# 		globin.visualize.plot_spectra(obs.spec[idx,idy], obs.wavelength,
# 			inv=obs2.spec[idx,idy])
# 		plt.show()
# sys.exit()

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
sys.exit()

for parameter in inv_atmos.nodes:
	print(parameter)
	if parameter in ["gamma", "chi"]:
		print(inv_atmos.values[parameter] * 180/np.pi)
	else:
		print(inv_atmos.values[parameter])
	print("-----")

# print(inv_atmos.global_pars)

for idx in range(inv_atmos.nx):
	for idy in range(inv_atmos.ny):
		globin.visualize.plot_spectra(obs.spec[idx,idy], obs.wavelength, inv=inv_spec.spec[idx,idy])
		plt.show()
