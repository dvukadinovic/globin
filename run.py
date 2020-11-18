import globin

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys

#--- initialize input object and then read input files
in_data = globin.InputData()
in_data.read_input_files()
# sys.exit()

# in_data.atm.build_from_nodes(in_data.ref_atm)
# rf, spec = globin.atmos.compute_full_rf(in_data)
# print(rf.shape)
# rf = rf[0,0,0,:,:,0]

# plt.imshow(rf, aspect="auto")
# plt.show()

globin.invert(in_data)
sys.exit()

#--- atmos compare
# inverted = fits.open("inverted_atmos.fits")[0].data[0,0]
# original = fits.open("../../Atmos/falc_1x1.fits")[0].data[0,0]
# print(inverted.shape)
# print(original.shape)

# plt.plot(original[0], original[10])
# plt.plot(inverted[0], inverted[10])
# plt.show()

# sys.exit()
#--- spectrum synthesis example
in_data.ref_atm.data[:,:,3,:] = -0.00068885
in_data.ref_atm.write_atmosphere()
spec = globin.atmos.compute_spectra(in_data, in_data.ref_atm, save=False, clean_dirs=True)

fix, axs = plt.subplots(nrows=2, ncols=2)

for i in range(in_data.atm.nx):
	for j in range(in_data.atm.ny):
		# Stokes I
		axs[0,0].set_title("Stokes I")
		axs[0,0].plot(in_data.obs.data[0,0,:,0] - 401.6, in_data.obs.spec[0,0,:,0])
		axs[0,0].plot(spec[i,j,:,0] - 401.6, spec[i,j,:,1])
		# Stokes Q
		axs[0,1].set_title("Stokes Q")
		axs[0,1].plot(in_data.obs.data[0,0,:,0] - 401.6, in_data.obs.spec[0,0,:,1])
		axs[0,1].plot(spec[i,j,:,0] - 401.6, spec[i,j,:,2])
		# Stokes U
		axs[1,0].set_title("Stokes U")
		axs[1,0].plot(in_data.obs.data[0,0,:,0] - 401.6, in_data.obs.spec[0,0,:,2])
		axs[1,0].plot(spec[i,j,:,0] - 401.6, spec[i,j,:,3])
		# Stokes V
		axs[1,1].set_title("Stokes V")
		axs[1,1].plot(in_data.obs.data[0,0,:,0] - 401.6, in_data.obs.spec[0,0,:,3])
		axs[1,1].plot(spec[i,j,:,0] - 401.6, spec[i,j,:,4])

axs[1,0].set_xlabel(r"$\Delta \lambda$ [nm]")
axs[1,1].set_xlabel(r"$\Delta \lambda$ [nm]")
axs[0,0].set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
axs[1,0].set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")

axs[0,0].set_xlim([-0.1, 0.1])
axs[0,1].set_xlim([-0.1, 0.1])
axs[1,0].set_xlim([-0.1, 0.1])
axs[1,1].set_xlim([-0.1, 0.1])

plt.show()

sys.exit()

#--- RF clauclation test
rf = globin.atmos.compute_full_rf(in_data)

# rf = fits.open("rf_temp_falc_2x2.fits")[0].data
logtau = np.round(in_data.ref_atm.data[0,0,0], 2)
wavs = np.round((in_data.wavelength - 401.6)*10, 2)

xpos = np.arange(rf.shape[4])
ypos = np.arange(rf.shape[3])

J = rf[0,0,0,:,:,0].T
JT = J.T
JTJ = np.dot(JT,J)

lam = 1e-3
H = JTJ + lam * np.diag(JTJ)

plt.imshow(H)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1)

plt.imshow(J.T, aspect="auto")
plt.setp(ax, xticks=xpos[::25], xticklabels=wavs[::25],
    yticks=ypos[::3], yticklabels=logtau[::3])
# plt.setp(ax, yticks=ypos[::3], yticklabels=logtau[::3])
plt.colorbar()
plt.show()