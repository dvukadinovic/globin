import globin
import numpy as np

atm = globin.Atmosphere("atmos.fits")
inv = globin.Atmosphere("runs/dummy/inverted_atmos.fits")

lista = list(inv.nodes)

for par in lista:
	print(par)
	idp = atm.par_id[par]
	diff = (inv.data[:,:,idp] - atm.data[:,:,idp])
	rmsd = np.sqrt( np.sum(diff**2, axis=(2)) / atm.nz)

	print(rmsd)
	print("------------------")
