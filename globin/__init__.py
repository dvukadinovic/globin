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
import numpy as np

from .atmos import \
	Atmosphere, compute_rfs, compute_spectra, write_multi_atmosphere, compute_full_rf, convert_atmosphere

from .input import \
	read_input, read_node_atmosphere, \
	write_line_parameters, write_line_par, \
	read_inverted_atmosphere

from .atoms import \
	Line, read_RLK_lines, read_init_line_parameters, init_line_pars, write_line_pars

from .rh import \
	write_wavs, Rhout, write_B

from .spec import \
	Observation, Spectrum

from .inversion import \
	invert

from .tools import \
	save_chi2, bezier_spline

from .visualize import \
	plot_atmosphere, plot_spectra, plot_rf, plot_chi2

from .utils import \
	construct_atmosphere_from_nodes, RHatm2Spinor, make_synthetic_observations, \
	calculate_chi2, remove_dirs

from .makeHSE import makeHSE

from .mppools import pool_write_atmosphere, pool_build_from_nodes, pool_rf, pool_synth, pool_spinor2multi

__all__ = ["rh", "atmos", "atoms", "inversion", "spec", "tools", "input", "visualize", "utils"]
__name__ = "globin"
__path__ = os.path.dirname(__file__)

#--- moduse operandi
# 0 --> synthesis
# 1 --> regular (pixle-by-pixel) inversion
# 2 --> PSF inversion (not implemented yet)
# 3 --> global inversion
# mode = None

#--- interpolation degree
interp_degree = 3

abundance = np.array([12.0, 10.99, 1.16, 1.15, 2.6, 8.39, 8.0, 8.66, 4.4,
            		  8.09, 6.33, 7.58, 6.47, 7.55, 5.45, 7.21, 5.5, 6.56,
            		  5.12, 6.36, 3.1, 4.99, 4.0, 5.67, 5.39, 7.44, 4.92,
            		  6.25, 4.21, 4.6, 2.88, 3.41, 2.37, 3.35, 2.63, 3.23,
            		  2.6, 2.9, 2.24, 2.6, 1.42, 1.92, -7.96, 1.84, 1.12,
            		  1.69, 0.94, 1.86, 1.66, 2.0, 1.0, 2.24, 1.51, 2.23,
            		  1.12, 2.13, 1.22, 1.55, 0.71, 1.5, -7.96, 1.0, 0.51,
            		  1.12, -0.1, 1.1, 0.26, 0.93, 0.0, 1.08, 0.76, 0.88,
            		  0.13, 1.11, 0.27, 1.45, 1.35, 1.8, 1.01, 1.09, 0.9,
            		  1.85, 0.71, -7.96, -7.96, -7.96, -7.96, -7.96, -7.96,
            		  0.12, -7.96, -0.47, -7.96, -7.96, -7.96, -7.96, -7.96,
            		  -7.96, -7.96])

atom_mass = np.array([1.00797, 4.00260, 6.941, 9.01218, 10.81, 12.011, 14.0067,
					  15.9994, 18.998403, 20.179, 22.98977, 24.305, 26.98154,
					  28.0855, 30.97376, 32.06, 35.453, 39.948, 39.0983, 40.08,
					  44.9559, 47.90, 50.9415, 51.996, 54.9380, 55.847, 58.9332,
					  58.70, 63.546, 65.38, 69.72, 72.59, 74.9216, 78.96, 79.904,
					  83.80, 85.4678, 87.62, 88.9059, 91.22, 92.9064, 95.94, 98,
					  101.07, 102.9055, 106.4, 107.868, 112.41, 114.82, 118.69,
					  121.75, 127.60, 126.9045, 131.30, 132.9054, 137.33, 138.9055,
					  140.12, 140.9077, 144.24, 145, 150.4, 151.96, 157.25, 158.9254,
					  162.50, 164.9304, 167.26, 168.9342, 173.04, 174.967, 178.49,
					  180.9479, 183.85, 186.207, 190.2, 192.22, 195.09, 196.9665,
					  200.59, 204.37, 207.2, 208.9804, 209, 210, 222, 223, 226.0254,
					  227.0278, 232.0381, 231.0359, 238.029, 237.0482, 242, 243, 
					  247, 247, 251, 252])


#--- parameter scale (calcualted from RFs based on
#    Cristopher Frutigers' thesis, p.42)
parameter_scale = {}

#--- limit values for atmospheric parameters
limit_values = {"temp"  : [3000, 10000], 	# [K]
				"vz"    : [-10, 10],		# [km/s]
				"vmic"  : [0, 10],			# [km/s]
				"vmac"  : [0, 5],			# [km/s]
				"mag"   : [0, 10000],		# [G]
				"gamma" : [0, np.pi],		# [rad]
				"chi"   : [0, 2*np.pi]}		# [rad]

#--- standard deviations for smoothing resulting parameters in many cycle inversion run
smooth_std = {"temp"   : 50, 	# [K]
			  "vz"     : 0.1,	# [km/s]
			  "vmic"   : 0.1,	# [km/s]
			  "mag"    : 25,	# [G]
			  "gamma"  : 0.087, # [rad == 5deg]
			  "chi"    : 0.087,	# [rad == 5deg]
			  "vmac"   : 0.1,	# [km/s]
			  "loggf"  : 0.010,	#
			  "dlam"   : 5}		# [mA]

#--- parameter perturbations for calculating RFs (must be the same as in rf_ray.c)
delta = {"temp"  : 1,		# K
		 "vz"    : 1/1e3,	# m/s --> km/s
		 "vmic"  : 1/1e3,	# m/s --> km/s
		 "mag"   : 1,		# G
		 "gamma" : 0.01,	# rad
		 "chi"   : 0.01,	# rad
		 "loggf" : 0.001,	# 
		 "dlam"  : 1}		# mA

#--- parameter differences for flaging if RF needs to be computed
diff = {"temp"   : 10,		# K
		"vz"     : 0.01,	# km/s
		"vmic"   : 0.01,	# km/s
		"mag"    : 10,		# G
		"gamma"  : 0.001,	# rad = 0.057 deg
		"chi"    : 0.001}	# rad = 0.057 deg

#--- full names of parameters (for FITS header)
parameter_name = {"temp"   : "Temperature",
				  "ne"     : "Electron density",
				  "vz"     : "Vertical velocity",
				  "vmic"   : "Microturbulent_velocity",
				  "vmac"   : "Macroturbulent_velocity",
				  "mag"    : "Magnetic_field_strength",
				  "gamma"  : "Inclination",
				  "chi"    : "Azimuth"}

#--- parameter units (for FITS header)
parameter_unit = {"temp"   : "K",
				  "ne"     : "1/3",
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
from scipy.constants import h as PLANCK
from scipy.constants import m_e as ELECTRON_MASS
from scipy.interpolate import splrep

#--- FAL C model (ref.): reference model if not given otherwise
falc = Atmosphere(fpath=f"{__path__}/data/falc_multi.atmos", atm_type="multi")

# temperature interpolation
temp_tck = splrep(falc.data[0,0,0],falc.data[0,0,1])
falc_logt = falc.data[0,0,0]
falc_ne = falc.data[0,0,2]

#--- flag for HSE computation (here defined when we do not read params.input)
hydrostatic = 1

#--- flag for normalizing spectra
norm = True

# Atmosphere(fpath=f"{__path__}/data/falc.dat", atm_type="spinor")

#--- axes for atmosphere plot (duumy idea?)
atmos_axs = None