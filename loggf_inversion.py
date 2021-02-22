import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys

import globin

Nrepeat = 10
# lineNo = np.arange(0,18)
#lineNo = [5,10]
lineNo = [15,16,17,18]

loggf = np.zeros((Nrepeat, len(lineNo)))

out_atmos = np.zeros((Nrepeat, 1, 3, 14, 71))
out_spec = np.zeros((Nrepeat, 1, 3, 201, 4))

for i_ in range(Nrepeat):
	print("/===========================/")
	print("           {:>3d}          ".format(i_+1))
	print("/===========================/\n")

	#--- perturb atomic parameters and save them to be read with globin
	globin.init_line_pars(lineNo, "../../Atoms/Kurucz/spinor_window_original", "test_line_pars")

	#--- initialize input object and read input files
	in_data = globin.InputData()
	
	#--- run inversion
	atmos, inv_spec = globin.invert(in_data, save_results=True, verbose=True)

	out_atmos[i_] = atmos.data
	out_spec[i_] = inv_spec.spec
	loggf[i_] = atmos.global_pars["loggf"]

	inv = globin.Observation("results/inverted_spectra.fits")
	obs = in_data.obs

	lista = list(in_data.atm.nodes)

	for idx in range(obs.nx):
		for idy in range(obs.ny):
			fig = plt.figure(figsize=(12,10))
			globin.plot_spectra(obs, idx=idx, idy=idy)
			globin.plot_spectra(inv, idx=idx, idy=idy)
			plt.savefig("results/loggf_stat_data/test_run/obs_vs_inv_{:2d}_{:2d}_{:03d}.png".format(idx, idy, i_+1))
			plt.close()

	globin.pool.terminate()

primary = fits.PrimaryHDU(out_atmos)
primary.writeto("results/loggf_stat_data/test_run/out_atmos.fits", overwrite=True)

primary = fits.PrimaryHDU(out_spec)
primary.writeto("results/loggf_stat_data/test_run/out_spec.fits", overwrite=True)

primary = fits.PrimaryHDU(loggf)
hdu_list = fits.HDUList([primary])
par_hdu = fits.ImageHDU(np.array(lineNo))
hdu_list.append(par_hdu)
hdu_list.writeto("results/loggf_stat_data/test_run/loggf.fits", overwrite=True)