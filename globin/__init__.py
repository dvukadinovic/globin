"""
Contributors:
	Dusan Vukadinovic (DV)

13/10/2020 : rewriten class 'InputData'; we leave to user to read from given input
			 files;
18/10/2020 : read input parameters from 'params.input' using regular expressions
07/12/2020 : moved 'InputData' class into 'input.py' file

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
from .visualize import plot_atmosphere, show

__all__ = ["rh", "atmos", "invert", "spec", "tools"]
__name__ = "globin"
__path__ = os.path.dirname(__file__)

#--- comment character in files read by wrapper
COMMENT_CHAR = "#"

#--- limit values for atmospheric parameters
limit_values = {"temp"  : [3000,10000], 		# [K]
				"vz"    : [-10, 10],			# [km/s]
				"vmic"  : [0,10],				# [km/s]
				"mag"   : [1/1e4, 5000/1e4],	# [T]
				"gamma" : [0, np.pi],			# [rad]
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

#--- FAL C model (ref.): reference model if not given otherwise
falc = Atmosphere(__path__ + "/data/falc.dat")

# Hydrogen level population + interpolation
# falc_hydrogen_pops, falc_hydrogen_lvls_tcks = hydrogen_lvl_pops(falc.data[0], falc.data[2], falc.data[3], falc.data[4])

# electron concentration [m-3] + interpolation
# falc_ne = falc.data[4]/10/K_BOLTZMAN/falc.data[2] / 1e6
# ne_tck = splrep(falc.data[0], falc_ne)

# temperature interpolation
temp_tck = splrep(falc.data[0],falc.data[2])
# ynew = splev(np.arange(0,1.5, 0.1), temp_tck, der=1)
# print(ynew)

#===--- end ---====#

#--- polynomial degree for interpolation
# interp_degree = None

#--- name of RH input file
# rh_input = None