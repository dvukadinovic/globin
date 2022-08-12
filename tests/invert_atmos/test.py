import globin
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy
import pyrh

atmos = globin.Atmosphere("atmos_dummy.fits")
# atmos = globin.Atmosphere("runs/dummy/inverted_atmos.fits")
atmos.RH = pyrh.RH()
atmos.wavelength_vacuum = np.linspace(401.6, 401.8, num=201)
atmos.ids_tuple = [(0,0)]
# atmos.data[:,:,1] += 100

# fig = plt.figure()
# gs = fig.add_gridspec(nrows=2, ncols=1)
# ax1, ax2 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0])

atmos.makeHSE()
# spec = atmos.compute_spectra()
# atmos.save_atmosphere("atmos_HSE_pyrh.fits")
# print(spec.wavelength)
# spec.save("obs.fits", spec.wavelength)
# sys.exit()

# ax1.plot(atmos.logtau, atmos.rho[0,0])
# ax2.plot(atmos.logtau, atmos.data[0,0,8])
# print()
# atmos.makeHSE()
# ax1.plot(atmos.logtau, atmos.rho[0,0])
# ax2.plot(atmos.logtau, atmos.data[0,0,8])

# ax1.set_yscale("log")
# ax2.set_yscale("log")

# plt.show()
# # atmos.save_atmosphere("atmos_HSE_pyrh.fits")

# sys.exit()

# spec = atmos.compute_spectra()

ne = copy.deepcopy(atmos.data[0,0,2])
temp = copy.deepcopy(atmos.data[0,0,1])

plt.scatter(atmos.logtau, atmos.data[0,0,2], label="orig")
# plt.scatter(spec.spec[0,0,:,0])

for idi in range(1):
	atmos.makeHSE()
	# spec = atmos.compute_spectra()

	new = atmos.data[0,0,2]

	diff = new - ne
	rmsd = np.sqrt(np.sum(diff**2)/atmos.nz)
	
	plt.scatter(atmos.logtau, atmos.data[0,0,2], label=idi)
	# plt.scatter(spec.spec[0,0,:,0])

plt.yscale("log")
plt.legend()
plt.show()