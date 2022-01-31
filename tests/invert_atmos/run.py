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
	globin.make_synthetic_observations(globin.atm, globin.noise, 
		atm_fpath=None)
	sys.exit()

#--- RFs
# globin.atmos.compute_full_rf(local_params=["temp"], global_params=None)
# sys.exit()

#--- do inversion
inv_atm, inv = globin.invert()

# sys.exit()

#--- analysis of the inverted data
atm = globin.ref_atm
obs = globin.obs

#chi2 = fits.open(f"runs/{run_name}/chi2.fits")[0].data
#globin.plot_chi2(chi2, f"runs/{run_name}/chi2.png", True)

lista = list(globin.atm.nodes)

for par in lista:
	print(par)
	idp = inv_atm.par_id[par]
	diff = (inv_atm.data[:,:,idp] - atm.data[:,:,idp])
	rmsd = np.sqrt( np.sum(diff**2, axis=(2)) / atm.nz)
	print(rmsd)
	print("-----------------------")