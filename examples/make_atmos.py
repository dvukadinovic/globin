import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import globin
from scipy.constants import k as k_b
from scipy.constants import h as hPlanck
from scipy.constants import m_e
import sys

fpath = f"{globin.__path__}/data/falc.dat"
falc = np.loadtxt(fpath, skiprows=1).T

# plt.plot(falc[0], falc[6])
# plt.plot(falc[0], falc[4]/10/k_b/falc[2])
# plt.yscale("log")
# plt.show()
# sys.exit()

def interpolate_density(parameter):
	top = falc[0,0]
	bot = falc[0,-1]

	a = 1 / (bot - top)
	b = -top / (bot - top)

	x = a * falc[0] + b
	y = parameter
	
	function = interp1d(x, y)

	top = atmos.logtau[0]
	bot = atmos.logtau[-1]
	a = 1 / (bot - top)
	b = -top / (bot - top)

	xx = a*atmos.logtau + b
	yy = function(xx)

	return yy

def distribute_nH(nH, temp, ne):
	Ej = 13.59844
	Ediss = 0.75
	u0_coeffs=[2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00001e+00, 2.00003e+00, 2.00015e+00], 
	u1_coeffs=[1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00],
	
	u1 = interp1d(np.linspace(3000,10000,num=8), u1_coeffs, fill_value="extrapolate")(temp)
	u0 = interp1d(np.linspace(3000,10000,num=8), u0_coeffs, fill_value="extrapolate")(temp)
	
	phi_t = 0.6665 * u1/u0 * temp**(5/2) * 10**(-5040/temp*Ej)
	phi_Hminus_t = 0.6665 * u0/1 * temp**(5/2) * 10**(-5040/temp*Ediss)
	
	pe = ne * k_b * temp * 1e6 * 10
	nH0 = nH / (1 + phi_t/pe)# + pe/phi_Hminus_t)
	nprot = phi_t/pe * nH0

	pops = np.zeros((6, len(temp)))

	pops[-1] = nprot

	for lvl in range(5):
		e_lvl = 13.59844*(1-1/(lvl+1)**2)
		g = 2*(lvl+1)**2
		pops[lvl] = nH0/u0 * g * np.exp(-5040/temp * e_lvl)
	
	return pops

nx = 6
ny = 8
atmos = globin.Atmosphere(nx=nx, ny=ny, logtau_top=-4, logtau_bot=1)
atmos.vmac = 0

atmos.data = np.zeros((nx,ny, atmos.npar, atmos.nz))
atmos.data[:,:,0,:] = atmos.logtau

# get electron density
# atmos.data[:,:,2,:] = interpolate_density(falc[4]/10/k_b/falc[2]/1e6)

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

# for pid in range(6):
# 	if pid==5:
# 		plt.plot(atmos.logtau, atmos.data[0,0,8+pid], label=f"prot")
# 	else:
# 		plt.plot(atmos.logtau, atmos.data[0,0,8+pid], label=f"{pid+1}")
# plt.yscale("log")
# plt.legend()
# plt.show()

for node in Tnodes:
	plt.axvline(node, color="k")
# plot atmosphere
for idx in range(nx):
	for idy in range(ny):
		plt.plot(atmos.logtau, atmos.data[idx,idy,1])

plt.show()

# save atmosphere
atmos.save_atmosphere("test_atmosphere_HSE.fits")

# atm = atmos.get_atmos(0,0)
# globin.atmos.write_multi_atmosphere(atm, "sample)nonHSE.atmos")