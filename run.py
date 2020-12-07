import globin

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys

#--- initialize input object and then read input files
sys.exit()
in_data = globin.InputData()
in_data.read_input_files()

#--- create atmos from nodes
# atmos = in_data.atm
# obs = in_data.obs
# atmos.build_from_nodes(in_data.ref_atm)

# spec, atm = globin.compute_spectra(in_data, atmos, True, True)

# new_atmos = globin.Atmosphere()
# new_atmos.data = atm
# globin.visualize.plot_atmosphere(new_atmos)
# new_atmos.save_cube("atmosphere_from_nodes.fits")

# plt.plot(obs.data[0,0,:-1,0], obs.data[0,0,:-1,1])
# plt.plot(spec[0,0,:-1,0], spec[0,0,:-1,1])
# plt.show()

# sys.exit()

# list of all class variables
# var = vars(in_data)

# --- inversion
globin.invert(in_data); sys.exit()

#--- spectrum synthesis example
# spec = globin.compute_spectra(in_data, in_data.ref_atm, False, False)

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
# rf,_,_ = globin.atmos.compute_full_rf(in_data)

# globin.visualize.plot_atmosphere(in_data.ref_atm)

# RFs shape : (nx, ny, npar, nz, nw, 4)
rf = fits.open("rf.fits")[0].data
logtau = np.round(in_data.ref_atm.data[0,0,0], 2)
wavs = np.round((in_data.wavelength - 401.6)*10, 2)

xpos = np.arange(rf.shape[4])
ypos = np.arange(rf.shape[3])

rf = rf[0,0]

par_scale = [5000, 1, 0.1, 1, 1]

weight = np.zeros((len(in_data.wavelength),5))
weight[:,0] = in_data.wavelength

for i_ in range(5):
	rf[i_] *= par_scale[i_]

par_rf = np.abs(rf)
weight[:,1:] = np.sum(par_rf, axis=(0,1))

from scipy.integrate import simps

for i_ in range(1,5):
	const = simps(weight[:,i_], in_data.wavelength)
	weight[:,i_] /= const

# plt.plot(weight[:,1:])
# plt.show()

np.savetxt("rf_weights", weight, fmt="%6.3f\t%5.2f\t%5.2f\t%5.2f\t%5.2f")

sys.exit()

fig = plt.figure(figsize=(20,14))

nrows, ncols = 4, 5

sid = 0
for sid in range(4):
	#--- RF plot for Temperature
	matrix = rf[0,:,:,sid] * par_scale[0]
	vmax = np.max(np.abs(matrix))

	if sid>0:
		cmap, vmax, vmin = "seismic", vmax, -vmax
	else:
		cmap, vmax, vmin = "gnuplot", vmax, 0

	plt.subplot(nrows,ncols,sid*ncols+1)
	if sid==0:
		plt.title("T [K]")
	plt.imshow(matrix, aspect="auto", cmap=cmap, vmax=vmax, vmin=vmin)
	plt.xticks(xpos[::50], wavs[::50])
	plt.yticks(ypos[::10], logtau[::10])
	plt.ylim([65,21])
	plt.colorbar()

	#--- RF plot for vertical velocity
	matrix = rf[1,:,:,sid] * par_scale[1]
	vmax = np.max(np.abs(matrix))

	plt.subplot(nrows,ncols,sid*ncols+2)
	if sid==0:
		plt.title(r"$v_z$ [km/s]")
	plt.imshow(matrix, aspect="auto", cmap="seismic", vmax=vmax, vmin=-vmax)
	plt.xticks(xpos[::50], wavs[::50])
	plt.yticks(ypos[::10], logtau[::10])
	plt.ylim([65,21])
	plt.colorbar()

	#--- RF plot for magnetic field
	matrix = rf[2,:,:,sid] * par_scale[2]
	vmax = np.max(np.abs(matrix))
	
	plt.subplot(nrows,ncols,sid*ncols+3)
	if sid==0:
		plt.title("B [T]")
	plt.imshow(matrix, aspect="auto", cmap="seismic", vmax=vmax, vmin=-vmax)
	plt.xticks(xpos[::50], wavs[::50])
	plt.yticks(ypos[::10], logtau[::10])
	plt.ylim([65,21])
	plt.colorbar()

	#--- RF plot for inclination
	matrix = rf[3,:,:,sid] * par_scale[3]
	vmax = np.max(np.abs(matrix))
	
	plt.subplot(nrows,ncols,sid*ncols+4)
	if sid==0:
		plt.title(r"$\gamma$ [rad]")
	plt.imshow(matrix, aspect="auto", cmap="seismic", vmax=vmax, vmin=-vmax)
	plt.xticks(xpos[::50], wavs[::50])
	plt.yticks(ypos[::10], logtau[::10])
	plt.ylim([65,21])
	plt.colorbar()

	#--- RF plot for azimuth
	matrix = rf[4,:,:,sid] * par_scale[4]
	vmax = np.max(np.abs(matrix))
	
	plt.subplot(nrows,ncols,sid*ncols+5)
	if sid==0:
		plt.title(r"$\chi$ [rad]")
	plt.imshow(matrix, aspect="auto", cmap="seismic", vmax=vmax, vmin=-vmax)
	plt.xticks(xpos[::50], wavs[::50])
	plt.yticks(ypos[::10], logtau[::10])
	plt.ylim([65,21])
	plt.colorbar()

plt.savefig("rf_from_nodes.png")
# plt.show()