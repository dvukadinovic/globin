"""
Contributors:
	Dusan Vukadinovic (DV)

13/10/2020 : rewriten class 'InputData'; we leave to user to read from given input
			 files;
18/10/2020 : read input parameters from 'params.input' using regular expressions

"""

import os
import sys
import numpy as np
import multiprocessing as mp
import re

from .input import InputData
from .rh import write_wavs, Rhout, write_B
from .atmos import Atmosphere, compute_rfs, compute_spectra, write_multi_atmosphere
from .spec import Observation
from .inversion import invert
from . import tools
from . import visualize

__all__ = ["rh", "atmos", "invert", "spec", "tools"]
__name__ = "globin"
__path__ = os.path.dirname(__file__)

#--- comment character in files read by wrapper
COMMENT_CHAR = "#"

#--- limit values for atmospheric parameters
limit_values = {"temp"  : [3000,10000], 		# [K]
				"vz"    : [-10, 10],			# [km/s]
				"vmic"  : [-10,10],				# [km/s]
				"mag"   : [1/1e4, 5000/1e4],	# [T]
				"gamma" : [-np.pi/2, np.pi/2],	# [rad]
				"chi"   : [-np.pi, np.pi]}		# [rad]

#--- parameter scale for RFs
parameter_scale = {"temp"   : 5000,	# [K]
				   "vz"     : 1,	# [km/s]
				   "vmic"   : 1,	# [km/s]
				   "mag"    : 0.1,	# [T]
				   "gamma"  : 1,	# [rad]
				   "chi"    : 1}	# [rad]

#--- parameter perturbations for calculating RFs
delta = {"temp"  : 1,      # K
		 "vz"    : 10/1e3, # m/s --> km/s
		 "vmic"  : 10/1e3, # m/s --> km/s
		 "mag"   : 25/1e4, # G --> T
		 "gamma" : 0.001,  # rad
		 "chi"   : 0.001}  # rad


#--- curent working directory: one from which we imported 'globin'
cwd = os.getcwd()

#===--- element abundances ---===#
from scipy.constants import k as K_BOLTZMAN
from scipy.interpolate import splrep, splev

# abundance = np.loadtxt("../../Atoms/abundance.input", usecols=(1,), unpack=True)

# def hydrogen_lvl_pops(logtau, temp, pg, pe, nlvl=6):
# 	"""
# 	Redistribute hydrogen atoms in first nlvl-1. Last column
# 	is reserved for proton number density.

# 	Parameters:
# 	---------------
# 	logtau : ndarray
# 		logarithm of optical depth.
# 	temp : ndarray
# 		temperature stratification in atmosphere.
# 	pg : ndarray
# 		gas pressure in CGS units (1D or 3D array).
# 	pe : ndarray
# 		electron pressure in CGS units (1D or 3D array).
# 	nlvl : int (optional)
# 		number of levels in hydrogen atom for which to calculate populations.
# 		last index stores proton numbers.

# 	Return:
# 	---------------
# 	popos : ndarray
# 		populations of hydrogen levels + protons. Dimension is (nvlv, len(temp)).
# 	tck : ndarray
# 		Spline knots for spline evaulation in given depth points.
# 	"""
# 	nH = (pg-pe)/10 / K_BOLTZMAN/temp / np.sum(10**(abundance-12)) / 1e6
# 	nH0 = nH / (1 + saha_phi(temp)/pe)
# 	nprot = nH - nH0
	
# 	pops = np.zeros((nlvl, *nH.shape))
# 	tcks = []

# 	for lvl in range(nlvl-1):
# 		e_lvl = 13.6*(1-1/(lvl+1)**2)
# 		pops[lvl] = nH/2 * 2*(lvl+1)**2 * np.exp(-5040/temp * e_lvl)
# 		tcks.append(splrep(logtau, pops[lvl]))
# 	pops[-1] = nprot
# 	tcks.append(splrep(logtau, nprot))

# 	return pops, tcks

# def saha_phi(temp, u0=2, u1=1, Ej=13.6):
# 	"""
# 	Calculate Phi(T) function for Saha's equation in form:

# 	n+/n0 = Phi(T)/Pe

# 	All units are in cgs system.

# 	Parameters:
# 	---------------
# 	temp : ndarray
# 		temperature for which to calculate Phi(T) function
# 	u0 : float (optional)
# 		partition function of lower ionization stage. Default 2 (for H atom).
# 	u1 : float (optional)
# 		partition function of higher ionization stage.Default 1 (for H atom).
# 	Ej : float (optional)
# 		ionization energy of state in [eV]. Default 13.6 (for H atom).

# 	Return:
# 	---------------
# 	Phi(T) : ndarray
# 		value of Phi(T) function at every temperature
# 	"""
# 	return 0.6665 * u1/u0 * temp**(5/2) * 10**(-5040/temp*Ej)

#--- FAL C model (ref.): reference model if not given otherwise
falc = Atmosphere(__path__ + "/data/falc.dat")

# Hydrogen level population + interpolation
# falc_hydrogen_pops, falc_hydrogen_lvls_tcks = hydrogen_lvl_pops(falc.data[0], falc.data[2], falc.data[3], falc.data[4])

# electron concentration [m-3] + interpolation
# falc_ne = falc.data[4]/10/K_BOLTZMAN/falc.data[2] / 1e6
# ne_tck = splrep(falc.data[0], falc_ne)

# temperature interpolation
temp_tck = splrep(falc.data[0],falc.data[2])

#===--- end ---====#

#--- polynomial degree for interpolation
# interp_degree = None

#--- name of RH input file
# rh_input = None