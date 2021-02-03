import globin

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys

#--- initialize input object and then read input files
in_data = globin.InputData()

# vmac = np.linspace(0, 1, num=21) * 1000
# temp = np.linspace(4000, 5000, num=11)
# pars = {"vmac" : [None,vmac], "temp" : [0, temp]}

# globin.chi2_hypersurface(pars, in_data)

# sys.exit()

#--- compute test spectra
# in_data.atm.build_from_nodes(in_data.ref_atm)
# spec, atm, h = globin.compute_spectra(in_data.atm, in_data.rh_spec_name, in_data.wavelength)
# plt.plot(in_data.obs.wavelength, in_data.obs.spec[0,0,:,0])
# plt.plot(spec.wavelength, spec.spec[0,0,:,0])
# plt.show()

# spec.save("original_loggf.fits", in_data.wavelength)

# sys.exit()

#--- make synthetic observations
# globin.make_synthetic_observations(None, in_data.rh_spec_name, in_data.wavelength, in_data.atm.vmac, in_data.noise)
# sys.exit()

#--- inversion
globin.invert(in_data)
# sys.exit()

#--- analysis of the inverted data
inv_atm = globin.Atmosphere("results/inverted_atmos.fits")
atm = in_data.ref_atm

inv = globin.Observation("results/inverted_spectra.fits")
obs = in_data.obs

chi2 = fits.open("results/chi2.fits")[0].data
globin.plot_chi2(chi2, "results/chi2.png", True)

lista = list(in_data.atm.nodes)

for idx in range(inv_atm.nx):
	for idy in range(inv_atm.ny):
		fig = plt.figure(figsize=(12,10))
		# globin.plot_atmosphere(atm, parameters=lista, idx=idx+xmin, idy=idy+ymin)
		globin.plot_atmosphere(atm, parameters=lista, idx=idx, idy=idy)
		globin.plot_atmosphere(inv_atm, parameters=lista, idx=idx, idy=idy)
		plt.savefig(f"results/atm_vs_inv_{idx}_{idy}.png")
		plt.close()
		# globin.show()

		fig = plt.figure(figsize=(12,10))
		globin.plot_spectra(obs, idx=idx, idy=idy)
		globin.plot_spectra(inv, idx=idx, idy=idy)
		plt.savefig(f"results/obs_vs_inv_{idx}_{idy}.png")
		plt.close()
		# globin.show()

sys.exit()

#--- RF clauclation test
# rf,_,_ = globin.atmos.compute_full_rf(in_data, 
# 			local_params=["temp", "vz", "mag"], 
# 			global_params=["vmac"], 
# 			fpath="rf_new.fits")

# rf = rf[0,0]

# # print(rf[1,0,:,0])

# plt.figure(1)
# plt.plot(rf[1,0,:,0])

# plt.figure(2)
# plt.imshow(rf[0,:,:,0], aspect="auto", cmap="gnuplot")
# plt.colorbar()

# plt.show()

# sys.exit()

# RFs shape : (nx, ny, npar, nz, nw, 4)
rf = fits.open("rf_normed.fits")[0].data
logtau = np.round(in_data.ref_atm.logtau, 2)
wavs = np.round((in_data.wavelength - 401.6)*10, 2)

xpos = np.arange(rf.shape[4])
ypos = np.arange(rf.shape[3])

rf = rf[0,0]

par_scale = [1, 1, 1, 1, 1]

#===--- Make weights from RFs ---===#


# weight = np.zeros((len(in_data.wavelength),5))
# weight[:,0] = in_data.wavelength

# for i_ in range(5):
# 	rf[i_] *= par_scale[i_]

# par_rf = np.abs(rf)
# weight[:,1:] = np.sum(par_rf, axis=(0,1))

# from scipy.integrate import simps

# for i_ in range(1,5):
# 	const = simps(weight[:,i_], in_data.wavelength)
# 	weight[:,i_] /= const

# plt.plot(weight[:,1:])
# plt.show()

# np.savetxt("rf_weights", weight, fmt="%6.3f\t%5.2f\t%5.2f\t%5.2f\t%5.2f")

# sys.exit()

#===--- Plot RFs ---===#

fig = plt.figure(figsize=(20,14))

nrows, ncols = 4, 3

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

	# #--- RF plot for inclination
	# matrix = rf[3,:,:,sid] * par_scale[3]
	# vmax = np.max(np.abs(matrix))
	
	# plt.subplot(nrows,ncols,sid*ncols+4)
	# if sid==0:
	# 	plt.title(r"$\gamma$ [rad]")
	# plt.imshow(matrix, aspect="auto", cmap="seismic", vmax=vmax, vmin=-vmax)
	# plt.xticks(xpos[::50], wavs[::50])
	# plt.yticks(ypos[::10], logtau[::10])
	# plt.ylim([65,21])
	# plt.colorbar()

	# #--- RF plot for azimuth
	# matrix = rf[4,:,:,sid] * par_scale[4]
	# vmax = np.max(np.abs(matrix))
	
	# plt.subplot(nrows,ncols,sid*ncols+5)
	# if sid==0:
	# 	plt.title(r"$\chi$ [rad]")
	# plt.imshow(matrix, aspect="auto", cmap="seismic", vmax=vmax, vmin=-vmax)
	# plt.xticks(xpos[::50], wavs[::50])
	# plt.yticks(ypos[::10], logtau[::10])
	# plt.ylim([65,21])
	# plt.colorbar()

plt.figure(2)
plt.subplot(2,2,1)
plt.plot(rf[-1,0,:,0])
plt.subplot(2,2,2)
plt.plot(rf[-1,0,:,1])
plt.subplot(2,2,3)
plt.plot(rf[-1,0,:,2])
plt.subplot(2,2,4)
plt.plot(rf[-1,0,:,3])

# plt.savefig("rf_from_nodes.png")
plt.show()