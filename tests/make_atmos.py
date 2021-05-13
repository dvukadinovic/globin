import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import globin

def get_H_pops():
	fpath = f"{globin.__path__}/data/falc.dat"
	falc = np.loadtxt(fpath, skiprows=1).T

	temp = interp1d(falc[0], falc[2])(atmos.logtau)
	pe = interp1d(falc[0], falc[4])(atmos.logtau)
	pg = interp1d(falc[0], falc[3])(atmos.logtau)

	atmos.data[:,:,8:,:] = globin.atmos.distribute_hydrogen(temp, pg, pe)

nx = 6
ny = 8
atmos = globin.Atmosphere(nx=nx, ny=ny, logtau_top=-4)
atmos.vmac = 0

atmos.data = np.zeros((nx,ny, atmos.npar, atmos.nz))
atmos.data[:,:,0,:] = atmos.logtau
atmos.data[:,:,2,:] = interp1d(globin.falc_logt, globin.falc_ne)(atmos.logtau)

Tnodes = [-2.5, -1.6, -0.9, 0]
NTnodes = len(Tnodes)

atmos.nodes["temp"] = np.array(Tnodes)
atmos.values["temp"] = np.zeros((nx,ny,NTnodes))

atmos.values["temp"][:,:,3] = np.linspace(6500, 7900, num=nx*ny).reshape(nx,ny)
atmos.values["temp"][:,:,2] = np.linspace(4500, 6500, num=nx*ny).reshape(nx,ny)
atmos.values["temp"][:,:,1] = np.linspace(4000, 5200, num=nx*ny).reshape(nx,ny)
atmos.values["temp"][:,:,0] = np.linspace(3800, 4300, num=nx*ny).reshape(nx,ny)

# build atmosphere from nodes
atmos.build_from_nodes(False)

# calculate Hydrogen populations based based on FAL C input model
get_H_pops()

for node in Tnodes:
	plt.axvline(node, color="k")
# plot atmosphere
for idx in range(nx):
	for idy in range(ny):
		plt.plot(atmos.logtau, atmos.data[idx,idy,1])

plt.show()

# save atmosphere
atmos.save_atmosphere("test_atmosphere.fits")