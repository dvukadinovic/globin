import globin

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys

#--- initialize input object and then read input files
in_data = globin.InputData()
in_data.read_input_files()

# list of all class variables
# var = vars(in_data)

#--- inversion
globin.invert(in_data); sys.exit()

#--- spectrum synthesis example
# spec = globin.compute_spectra(in_data, in_data.ref_atm, True, False)

# fix, axs = plt.subplots(nrows=2, ncols=2)

# for i in range(in_data.ref_atm.nx):
# 	for j in range(in_data.ref_atm.ny):
# 		# Stokes I
# 		axs[0,0].set_title("Stokes I")
# 		# axs[0,0].plot(in_data.obs.data[0,0,:,0] - 401.6, in_data.obs.spec[0,0,:,0])
# 		axs[0,0].plot(spec[i,j,:,0] - 401.6, spec[i,j,:,1])
# 		# Stokes Q
# 		axs[0,1].set_title("Stokes Q")
# 		# axs[0,1].plot(in_data.obs.data[0,0,:,0] - 401.6, in_data.obs.spec[0,0,:,1])
# 		axs[0,1].plot(spec[i,j,:,0] - 401.6, spec[i,j,:,2])
# 		# Stokes U
# 		axs[1,0].set_title("Stokes U")
# 		# axs[1,0].plot(in_data.obs.data[0,0,:,0] - 401.6, in_data.obs.spec[0,0,:,2])
# 		axs[1,0].plot(spec[i,j,:,0] - 401.6, spec[i,j,:,3])
# 		# Stokes V
# 		axs[1,1].set_title("Stokes V")
# 		# axs[1,1].plot(in_data.obs.data[0,0,:,0] - 401.6, in_data.obs.spec[0,0,:,3])
# 		axs[1,1].plot(spec[i,j,:,0] - 401.6, spec[i,j,:,4])

# axs[1,0].set_xlabel(r"$\Delta \lambda$ [nm]")
# axs[1,1].set_xlabel(r"$\Delta \lambda$ [nm]")
# axs[0,0].set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")
# axs[1,0].set_ylabel(r"Intensity [W sr$^{-1}$ Hz$^{-1}$ m$^{-2}$]")

# axs[0,0].set_xlim([-0.1, 0.1])
# axs[0,1].set_xlim([-0.1, 0.1])
# axs[1,0].set_xlim([-0.1, 0.1])
# axs[1,1].set_xlim([-0.1, 0.1])

# plt.show()

# sys.exit()

#--- RF clauclation test
# rf, _ = globin.atmos.compute_full_rf(in_data)

rf = fits.open("rf.fits")[0].data
logtau = np.round(in_data.ref_atm.data[0,0,0], 2)
wavs = np.round((in_data.wavelength - 401.6)*10, 2)

xpos = np.arange(rf.shape[4])
ypos = np.arange(rf.shape[3])

J = rf[0,0,0,:,:,0].T
JT = J.T
JTJ = np.dot(JT,J)

lam = 1e-3
H = JTJ
np.fill_diagonal(H, np.diag(JTJ)*(1+lam))

# plt.imshow(np.linalg.inv(H))
# plt.imshow(np.log10(H))
# plt.show()
# sys.exit()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

plt.setp(axs, xticks=xpos[::25], xticklabels=wavs[::25],
    yticks=ypos[::3], yticklabels=logtau[::3])

vmax = np.max(np.abs(rf[0,0,2,:,:,3]))
rf_temp = axs[0].imshow(rf[0,0,2,:,:,3], aspect="auto", cmap="bwr", vmax=vmax, vmin=-vmax)
plt.colorbar(rf_temp, ax=axs[0])

vmax = np.max(np.abs(rf[0,0,4,:,:,1]))
rf_vz = axs[1].imshow(rf[0,0,4,:,:,1], aspect="auto", cmap="bwr", vmax=vmax, vmin=-vmax)
plt.colorbar(rf_vz, ax=axs[1])

plt.show()