import globin
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# atm = globin.Atmosphere("atmos_uniform.fits")
# atm.vmin["vz"].vmin = np.array([-10, -8])
# print(atm.vmin["vz"].min)

# sys.exit()

# atm.wavelength_air = np.linspace(401.5, 401.7, num=201)
# atm.wavelength_obs = atm.wavelength_air
# atm.wavelength_vacuum = globin.rh.air_to_vacuum(atm.wavelength_air)
# spec = atm.compute_spectra(np.ones((1,1)))

# out = globin.rh.Rhout()
# out.read_spectrum()
# out.read_ray()

# plt.plot(out.wave, out.int, label="RH")
# plt.plot(spec.wavelength, spec.spec[0,0,:,0], label="globin")
# plt.legend()
# plt.show()

# # globin.visualize.plot_spectra(spec.spec[0,0], spec.wavelength, relative=False)
# # globin.show()

# sys.exit()

atmos = globin.utils.construct_atmosphere_from_nodes("node_atmosphere", 
	vmac=0, output_atmos_path=None)
# globin.falc.wavelength_air = np.array([451.0])
# globin.falc.wavelength_obs = np.array([451.0])
# globin.falc.wavelength_vacuum = np.array([451.0])
# spec = globin.falc.compute_spectra()

idx, idy = 1, 1

atmos.interpolation_method = "spinor"
atmos.spline_tension = 0
atmos.build_from_nodes()
plt.scatter(atmos.nodes["temp"], atmos.values["temp"][idx,idy], c="k")
plt.plot(atmos.logtau, atmos.data[idx,idy,1], label="SPINOR")

atmos.interpolation_method = "spinor"
atmos.spline_tension = 5
atmos.build_from_nodes()
plt.plot(atmos.logtau, atmos.data[idx,idy,1], label="SPINOR tension")

atmos.interpolation_method = "bezier"
atmos.build_from_nodes()
plt.plot(atmos.logtau, atmos.data[idx,idy,1], label="Bezier")

plt.legend()
globin.show()