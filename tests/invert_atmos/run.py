import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys

import globin

#--- initialize input object and read input files
run_name = "dummy_m2"
globin.read_input(run_name=run_name)

#--- make synthetic observations from input atmosphere
if globin.mode==0:
	globin.make_synthetic_observations(globin.atm, globin.noise, 
		atm_fpath=None)
	sys.exit()

#--- do inversion
inv_atm, inv = globin.invert()

#--- analysis of the inverted data
atm = globin.ref_atm
obs = globin.obs

chi2 = fits.open(f"runs/{run_name}/chi2.fits")[0].data
globin.plot_chi2(chi2, f"runs/{run_name}/chi2.png", True)

lista = list(globin.atm.nodes)

diff = (inv_atm.data[:,:,1] - atm.data[:,:,1])
rmsd = np.sqrt( np.sum(diff**2, axis=(2)) / atm.nz)
print(rmsd)

for idx in range(inv_atm.nx):
	for idy in range(inv_atm.ny):
		fig = plt.figure(figsize=(12,10))

		globin.plot_atmosphere(atm, parameters=lista, idx=idx, idy=idy)
		globin.plot_atmosphere(inv_atm, parameters=lista, idx=idx, idy=idy, color="tab:red")
		# plt.plot(atm.data[idx,idy,1])
		# plt.plot(inv_atm.data[idx,idy,1])
		plt.savefig(f"runs/{run_name}/inv_vs_atm_{idx}_{idy}.png")
		plt.close()

		globin.plot_spectra(obs, inv=inv, idx=idx, idy=idy)
		plt.savefig(f"runs/{run_name}/obs_vs_inv_{idx}_{idy}.png")
		plt.close()