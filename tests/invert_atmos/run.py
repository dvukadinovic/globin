import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys

import globin

# run_name = "m1"
# chi2 = fits.open(f"runs/{run_name}/chi2.fits")[0].data
# globin.plot_chi2(chi2, f"runs/{run_name}/chi2.png", True)
# sys.exit()

#--- initialize input object and read input files
run_name = "dummy"
globin.read_input(run_name=run_name)

#--- make synthetic observations from input atmosphere
if globin.mode==0:
	spec = globin.make_synthetic_observations(globin.atm, globin.noise, 
		atm_fpath=None)
	globin.plot_spectra(spec.spec[0,0], spec.wavelength)
	plt.show()
	sys.exit()

#--- RFs
# globin.atmos.compute_full_rf(local_params=["temp"], global_params=None)
# sys.exit()

#--- do inversion
inv_atm, inv = globin.invert(verbose=True)

# if "gamma" in inv_atm.nodes:
# 	print("gamma")
# 	print(2*np.arctan(inv_atm.values["gamma"]) * 180/np.pi)
# 	# print(2*np.arctan(inv_atm.data[0,0,6]) * 180/np.pi)
# 	print(2*np.arctan(globin.ref_atm.data[0,0,6,0]) * 180/np.pi)
# 	print("--------")

# if "chi" in inv_atm.nodes:
# 	print("chi")
# 	print(4*np.arctan(inv_atm.values["chi"]) * 180/np.pi)
# 	# print(inv_atm.data[0,0,7] * 180/np.pi)
# 	print(4*np.arctan(globin.ref_atm.data[0,0,7]) * 180/np.pi)

#--- analysis of the inverted data
atm = globin.ref_atm
obs = globin.obs

chi2 = fits.open(f"runs/{run_name}/chi2.fits")[0].data
globin.plot_chi2(chi2, f"runs/{run_name}/chi2.png", True)

sys.exit()

lista = list(globin.atm.nodes)

# for par in lista:
# 	print(par)
# 	idp = inv_atm.par_id[par]
# 	diff = (inv_atm.data[:,:,idp] - atm.data[:,:,idp])
# 	rmsd = np.sqrt( np.sum(diff**2, axis=(2)) / atm.nz)
# 	print(rmsd)
# 	print("-----------------------")

pars = ["temp", "vz", "mag", "gamma", "chi"]

for idx in range(atm.nx):
	for idy in range(atm.ny):
		ida = idx * atm.ny + idy + 1
		globin.plot_spectra(obs.spec[idx,idy], obs.wavelength, inv=inv.spec[idx,idy])
		# plt.savefig(f"runs/{run_name}/inv_vs_obs_{ida}.png")
		plt.show()

		globin.plot_atmosphere(atm, pars, idx=idx, idy=idy, color="black")
		globin.plot_atmosphere(inv_atm, pars, idx=idx, idy=idy, color="tab:blue")
		plt.show()
		# plt.savefig(f"runs/{run_name}/atmos_compare_{ida}.png")