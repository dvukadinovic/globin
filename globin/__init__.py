"""
Contributors:
	Dusan Vukadinovic (DV)

13/10/2020 : rewriten class 'InputData'; we leave to user to read from given input
			 files;
18/10/2020 : read input parameters from 'params.input' using regular expressions
07/12/2020 : moved 'InputData' class into 'input.py' file
23/02/2021 : a lot in mean time, but this is important; InputData() class has a parameter
			 which takes into account if we need to run mp.Pool(). Used in many iteration
			 run with different initial conditions. By default we initialize mp.Pool(), but
			 in this situation we do not need it to be initialized many times.
24/02/2021 : overwritten last thing (23/02/2021). Now we have function fot initializing mp.Pool()
			 and also take as argument working directory for RH computation so that we can have
			 many different executions from the same PC. Otherwise we have conflict in names.
"""

import os
import sys
import numpy as np

from .input import \
	InputData, find_value_by_key, read_node_atmosphere, set_keyword

from .atoms import \
	Line, read_RLK_lines, read_init_line_parameters, init_line_pars

from .rh import \
	write_wavs, Rhout, write_B

from .atmos import \
	Atmosphere, compute_rfs, compute_spectra, write_multi_atmosphere

from .spec import \
	Observation, Spectrum

from .inversion import \
	invert

from .tools import \
	save_chi2, bezier_spline

from .visualize import \
	plot_atmosphere, plot_spectra, plot_chi2

from .utils import \
	construct_atmosphere_from_nodes, RHatm2Spinor, make_synthetic_observations, \
	chi2_hypersurface, calculate_chi2

__all__ = ["rh", "atmos", "atoms", "inversion", "spec", "tools", "input", "visualize", "utils"]
__name__ = "globin"
__path__ = os.path.dirname(__file__)

#--- comment character in files read by wrapper
COMMENT_CHAR = "#"

#--- moduse operandi
# 0 --> synthesis
# 1 --> regular (pixle-by-pixel) inversion
# 2 --> PSF inversion (not implemented yet)
# 3 --> global inversion
mode = None

#--- name of RH input file
rh_input_name = None

#--- path to RH main folder
rh_path = None

#--- limit values for atmospheric parameters
limit_values = {"temp"  : [3000, 10000], 		# [K]
				"vz"    : [-10, 10],			# [km/s]
				"vmic"  : [1e-3, 10],			# [km/s]
				"vmac"  : [1e-3, 30],			# [km/s]
				"mag"   : [0, 5000/1e4],		# [T]
				"gamma" : [0, np.pi],			# [rad]
				"chi"   : [-np.pi/2, np.pi/2]}	# [rad]

#--- parameter scale for RFs
parameter_scale = {"temp"   : 5000,	# [K]
				   "vz"     : 1,	# [km/s]
				   "vmic"   : 1,	# [km/s]
				   "vmac"   : 1e3,	# [m/s]
				   "mag"    : 0.1,	# [T]
				   "gamma"  : 1,	# [rad]
				   "chi"    : 1,	# [rad]
				   "loggf"  : 1,	# 
				   "dlam"   : 1}	# [mA]

#--- parameter perturbations for calculating RFs
delta = {"temp"  : 1,		# K
		 "vz"    : 1/1e3,	# m/s --> km/s
		 "vmic"  : 1/1e3,	# m/s --> km/s
		 "mag"   : 1/1e4,	# G --> T
		 "gamma" : 0.001,	# rad
		 "chi"   : 0.001,	# rad
		 "loggf" : 0.001,	# 
		 "dlam"  : 1}		# mA

#--- full names of parameters (for FITS header)
parameter_name = {"temp"   : "Temperature",
				  "vz"     : "Vertical_velocity",
				  "vmic"   : "Microturbulent_velocity",
				  "vmac"   : "Macroturbulent_velocity",
				  "mag"    : "Magnetic_field_strength",
				  "gamma"  : "Inclination",
				  "chi"    : "Azimuth"}

#--- parameter units (for FITS header)
parameter_unit = {"temp"   : "K",
				  "vz"     : "km/s",
				  "vmic"   : "km/s",
				  "vmac"   : "km/s",
				  "mag"    : "T",
				  "gamma"  : "rad",
				  "chi"    : "rad"}

#--- curent working directory: one from which we imported 'globin'
cwd = os.getcwd()

from scipy.constants import k as K_BOLTZMAN
from scipy.constants import c as LIGHT_SPEED
from scipy.interpolate import splrep, splev

#--- FAL C model (ref.): reference model if not given otherwise
falc = Atmosphere(__path__ + "/data/falc.dat")

# temperature interpolation
temp_tck = splrep(falc.data[0],falc.data[2])
falc_logt = falc.data[0]
falc_ne = falc.data[4] / 10 / K_BOLTZMAN / falc.data[2]
# ynew = splev(np.arange(0,1.5, 0.1), temp_tck, der=1)
# print(ynew)