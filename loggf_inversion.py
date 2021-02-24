import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys

import globin

Nrepeat = 20
lineNo = np.arange(0,18) + 1
lineNo = [10]

loggf = np.zeros((Nrepeat, len(lineNo)))
dlam = np.zeros((Nrepeat, len(lineNo)))

fpath = "results/loggf_stat_data/test_run"

in_data = globin.InputData()
obs = in_data.obs

nx, ny, nz = in_data.ref_atm.nx, in_data.ref_atm.ny, in_data.ref_atm.nz
nw = len(in_data.wavelength)

out_atmos = np.zeros((Nrepeat, nx, ny, 14, nz))
out_spec = np.zeros((Nrepeat, nx, ny, nw, 4))

for i_ in range(Nrepeat):
	print("/===========================/")
	print("           {:>3d}          ".format(i_+1))
	print("/===========================/\n")

	#--- perturb atomic parameters and save them to be read with globin
	globin.init_line_pars(lineNo, "../../Atoms/Kurucz/spinor_window_original", "test_line_pars")

	#--- initialize input object and read input files
	in_data = globin.InputData(init_pool=False)
	
	#--- run inversion
	atmos, inv_spec = globin.invert(in_data)

	out_atmos[i_] = atmos.data
	out_spec[i_] = inv_spec.spec
	loggf[i_] = atmos.global_pars["loggf"]
	dlam[i_] = atmos.global_pars["dlam"]

	# lista = list(in_data.atm.nodes)

	for idx in range(obs.nx):
		for idy in range(obs.ny):
			fig = plt.figure(figsize=(12,10))
			globin.plot_spectra(obs, idx=idx, idy=idy)
			globin.plot_spectra(inv_spec, idx=idx, idy=idy)
			plt.savefig("{:s}/obs_vs_inv_{:02d}_{:02d}_{:03d}.png".format(fpath, idx, idy, i_+1))
			plt.close()

primary = fits.PrimaryHDU(out_atmos)
primary.writeto(f"{fpath}/out_atmos.fits", overwrite=True)

primary = fits.PrimaryHDU(out_spec)
primary.writeto(f"{fpath}/out_spec.fits", overwrite=True)

primary = fits.PrimaryHDU(loggf)
hdu_list = fits.HDUList([primary])
par_hdu = fits.ImageHDU(np.array(lineNo))
hdu_list.append(par_hdu)
hdu_list.writeto(f"{fpath}/loggf.fits", overwrite=True)

primary = fits.PrimaryHDU(dlam)
hdu_list = fits.HDUList([primary])
par_hdu = fits.ImageHDU(np.array(lineNo))
hdu_list.append(par_hdu)
hdu_list.writeto(f"{fpath}/dlam.fits", overwrite=True)