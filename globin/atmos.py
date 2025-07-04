from astropy.io import fits
import numpy as np
import os
import sys
import copy
import time
from scipy.ndimage import median_filter, correlate1d
from scipy.interpolate import splev, splrep, interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import multiprocessing as mp
import scipy.sparse as sp

try:
	import pyrh
except:
	# raise ImportError("No 'pyrh' module. Install it before using 'globin'")
	pass

import globin
from .spec import Spectrum
from .tools import bezier_spline, spline_interpolation, get_K0_Kn, get_control_point
from .utils import extend, Planck, azismooth, mygsmooth, air_to_vacuum
from .makeHSE import makeHSE
from .rh import atmols, create_kurucz_input

class MinMax(object):
	"""
	Container for minimum and maximum values of a parameter.
	"""
	def __init__(self, vmin=None, vmax=None):
		self.vmin = np.asarray([vmin])
		self.vmax = np.asarray([vmax])
		self.vmin_dim = len(self.vmin)
		self.vmax_dim = len(self.vmax)

	@property
	def min(self):
		return self.vmin

	@property
	def max(self):
		return self.vmax

	def __str__(self):
		return "<vmin: {}, vmax: {}>".format(self.vmin, self.vmax)

class Atmosphere(object):
	"""
	Object class for atmospheric models.

	We can read .fits file like cube with assumed shape of (nx, ny, npar, nz)
	where npar=14. Order of parameters are like for RH (MULTI atmos type) where
	after velocity we have magnetic field strength, inclination and azimuth. Last
	6 parameters are Hydrogen number density for first six levels.

	We distinguish different type of 'Atmopshere' based on the input mode. For .fits
	file we call it 'cube' while for rest we call it 'single'.

	The class is now able to hande direct conversion from SPINOR cube/single and
	SIR single type atmospheres to MULTI type adopted for 'globin'.
	"""
	# index of parameters in self.data ndarray
	par_id = {"logtau" : 0,
					  "temp"   : 1,
					  "ne"     : 2,
					  "vz"     : 3,
					  "vmic"   : 4,
					  "mag"    : 5,
					  "gamma"  : 6,
					  "chi"    : 7,
					  "nH"     : 8}

	#--- temperature limits in the atmosphere (used to limit the extrapolation to a top of the atmosphere)
	Tmin = 2800
	Tmax = 10000

	#--- parameter perturbations for calculating RFs
	delta = {"temp"  : 1,			# [K]
					 "vz"    : 1e-3,	# [km/s]
					 "vmic"  : 1e-3,	# [km/s]
					 "mag"   : 1,			# [G]
					 "gamma" : 0.01,	# [rad]
					 "chi"   : 0.01,	# [rad]
					 "loggf" : 0.001,	#
					 "dlam"  : 1,			# [mA]
					 "of"    : 0.001,
					 "stray" : 0,		# no perturbations (done analytically)
					 "sl_temp" : 1,		# [K]
					 "sl_vz" : 1e-3,	# [km/s]
					 "sl_vmic" : 1e-3	# [km/s]
					 }

	#--- full names of parameters (for FITS header)
	# parameter_name = {"temp"   : "Temperature",
	# 								  "ne"     : "Electron density",
	# 								  "vz"     : "Vertical velocity",
	# 								  "vmic"   : "Microturbulent velocity",
	# 								  "vmac"   : "Macroturbulent velocity",
	# 								  "mag"    : "Magnetic field strength",
	# 								  "gamma"  : "Inclination",
	# 								  "chi"    : "Azimuth",
	# 								  "of"     : "Opacity fudge",
	# 								  "nH"     : "Hydrogen density"}

	#--- normalization values for atmospheric parameters
	#    use them to scale RFs first and regularization functions;
	#    RFs are later scaled together with regularization 
	#    contributions to have 1s on the Hessian diagonal (before
	#    adding the Marquardt parameter).
	parameter_norm = {"temp"  : 5000,			# [K]
					  "vz" 	  : 6,				# [km/s]
					  "vmic"  : 6,				# [km/s]
					  "mag"   : 1000,			# [G]
					  "gamma" : np.pi,			# [rad]
					  "chi"   : np.pi,			# [rad]
					  "of"    : 2,				#
					  "stray" : 0.1,
					  "vmac"  : 2,				# [km/s]
					  "dlam"  : 10,				# [mA]
					  "loggf" : -1.0,			#
					  "sl_temp" : 5000,			# [K]
					  "sl_vz" : 6,				# [km/s]
					  "sl_vmic" : 6				# [km/s]
					  }

	#--- relative weighting for spatial regularization for each parameter
	regularization_weight = {"temp"  : 1,
													 "vz"    : 1,
													 "vmic"  : 1,
													 "mag"   : 1,
													 "gamma" : 1,
													 "chi"   : 1}

	#--- weighting for depth-dependen regularization of parameters
	dd_regularization_weight = {"temp"  : 1,
													 		"vz"    : 1,
													 		"vmic"  : 1,
													 		"mag"   : 1,
													 		"gamma" : 1,
													 		"chi"   : 1}

	#--- depth-dependent regularization functions
	# 0 -- no regularization
	# 1 -- penalize large deviations: (pi - pj)
	# 2 -- penalize oscillations (changes in the gradient) 
	#			 (A*p_i+1  + B*p_i + C*p_i-1)
	# 3 -- penalize diff. from a constant value: pi - const
	# 4 -- penalize diff. from a mean values: pi - np.mean(pi)
	dd_regularization_function = {"temp"  : 0,
																"vz"    : 0,
																"vmic"  : 0,
																"mag"   : 0,
																"gamma" : 0,
																"chi"   : 0}

	def __init__(self, fpath=None, atm_type="multi", atm_range=[0,None,0,None],
			nx=None, ny=None, nz=None, logtau_top=-6, logtau_bot=1, logtau_step=0.1):
		self.type = atm_type
		self.fpath = fpath

		# parameter scaling for inversion
		self.parameter_scale = {}

		# flag and weight for spatial regularization
		# the weight is multiplying individual weights for each parameter
		self.spatial_regularization = False
		self.spatial_regularization_weight = 0

		# current working directory is appended to paths sent to RH
		# (keyword, atmos, molecules, kurucz)
		self.cwd = "."

		# flag for HSE computation; if temperature is to be inverted, this is
		# switched to 'True'
		self.hydrostatic = False

		# gas pressure at the top (used for HSE computation from RH)
		self.pg_top = 0.03 # [N/m2]

		# interpolation degree, method for building atmosphere from node values
		# and (optional) tension value in case of "spinor" method
		self.interp_degree = 3
		self.interpolation_method = "bezier"
		self.spline_tension = 0

		# nodes: each atmosphere has the same nodes for given parameter
		self.nodes = {}
		# parameters values in nodes: shape = (nx, ny, nnodes)
		self.values = {}
		# node mask: we can specify only which nodes to invert (mask==1)
		# structure and ordering same as self.nodes
		self.mask = {}

		# mode of operation
		# self.mode = 0

		self.chi_c = None
		self.pg = None
		self.vmac = 0

		# flag if temperature should only decrease with the height
		self.decreasing_temperature = False

		# cos(theta) for which we are computing the spectrum
		self.mu = 1.0
		
		self.line_list = None
		# line number in list of lines for which we are inverting atomic data
		self.line_no = {"loggf" : np.array([], dtype=np.int32), 
										"dlam"  : np.array([], dtype=np.int32)}
		# global parameters: each is given in a list size equal to number of parameters
		self.global_pars = {"loggf" : np.array([]), 
							"dlam"  : np.array([])}

		# fudge parameters: H-, scatt, metals for each wavelength
		self.add_fudge = False
		self.fudge_lam = None
		self.fudge = None
		self.of_scatter = 1

		# flag for computing the atomic parameters RFs (for those in self.global_pars)
		self.get_atomic_rfs = False

		# containers for updating atomic abundances
		self.atomic_number = np.zeros(0, dtype=np.int32)
		self.atomic_abundance = np.zeros(0, dtype=np.float64)

		# flag to retrieve the level populations for ACTIVE atoms
		self.get_populations = False
		self.n_pops = {}
		self.nstar_pops = {}

		# type of RFs caluclation
		self.rf_der_type = "central"

		# number of threads to be used for parallel execution of different functions
		self.n_thread = 1

		# size of the chunks that are submitted to each process (for multiprocessing map())
		self.chunk_size = 1

		# 2nd component atmosphere treated as stray light source
		self.sl_atmos = None

		# continuum intensity
		self.icont = None
		# flag for normalizing the spectrum		
		self.norm = False
		# flag for normalization type
		self.norm_level = None
		# index of wavelength in the continuum
		self.continuum_idl = 0

		# stray-light contamination parameters
		self.add_stray_light = False
		self.invert_stray = False
		self.stray_mode = -1
		self.stray_type = None

		# instrumental profile
		self.instrumental_profile = None

		# slices dimension from inputed cube
		# self.xmin = atm_range[0]
		# self.xmax = atm_range[1]
		# self.ymin = atm_range[2]
		# self.ymax = atm_range[3]

		# atmosphere dimensions
		self.nx = nx
		self.ny = ny
		self.nz = nz
		self.npar = 14

		self.logtau_top = logtau_top
		self.logtau_bot = logtau_bot
		self.logtau_step = logtau_step
		
		# allocate logtau scale and nz
		if self.nz is None:
			self.logtau = np.arange(logtau_top, logtau_bot+logtau_step, logtau_step)
			self.nz = len(self.logtau)

		# read in the atmosphere if the fpath has been provided
		if fpath is not None:
			if atm_type=="spinor":
				self.read_spinor(fpath, atm_range)
			elif atm_type=="sir":
				self.read_sir(fpath)
			elif atm_type=="multi":
				if "fits" in fpath:
					self.read_multi_cube(fpath, atm_range)
				else:
					self.read_multi(fpath)
			else:
				raise ValueError(f"  Unsupported atmosphere type '{atm_type}'.")

			# valid, inds = self.is_valid()
			# if not valid:
			# 	raise ValueError(f"Found NaN in atmosphere {fpath} at ({inds[0]},{inds[1]}) for parameter {inds[2]}.")

			self.nHtot = np.sum(self.data[:,:,8:,:], axis=2)
			self.idx_meshgrid, self.idy_meshgrid = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
			self.idx_meshgrid = self.idx_meshgrid.ravel()
			self.idy_meshgrid = self.idy_meshgrid.ravel()
		else:
			self.data = None
			self.header = None
			if (self.nx is not None) and (self.ny is not None) and (self.nz is not None):
				self.data = np.empty((self.nx, self.ny, self.npar, self.nz), dtype=np.float64)
				self.height = np.empty((self.nx, self.ny, self.nz), dtype=np.float64)
				self.rho = np.empty((self.nx, self.ny, self.nz), dtype=np.float64)
				self.pg = np.empty((self.nx, self.ny, self.nz), dtype=np.float64)
				self.nHtot = np.empty((self.nx, self.ny, self.nz), dtype=np.float64)
				self.idx_meshgrid, self.idy_meshgrid = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
				self.idx_meshgrid = self.idx_meshgrid.ravel()
				self.idy_meshgrid = self.idy_meshgrid.ravel()

		#--- limit values for local and global parameters
		self.limit_values = {"temp"  : MinMax(3000, 10000), # [K]
							 "vz"    : MinMax(-10, 10),								# [km/s]
							 "vmic"  : MinMax(1e-3, 10),							# [km/s]
							 "mag"   : MinMax(10, 10000),							# [G]
							 "gamma" : MinMax(-np.pi, 2*np.pi),						# [rad]
							 # "gamma" : MinMax(-1,1),									# cos(gamma)
							 "chi"   : MinMax(-2*np.pi, 2*np.pi),		# [rad]
							 # "chi"   : MinMax(0, 1),									# sin^2(chi)
							 "of"    : [0, 20],												#
							 "stray" : MinMax(1e-3, 0.99),						#
							 "vmac"  : [1e-3, 5],												# [km/s]
							 "loggf" : [-10,4],												#
							 "dlam"  : [-50,50],					# [mA]
							 "sl_temp"  : MinMax(3000, 10000), # [K]
							 "sl_vz"    : MinMax(-10, 10),								# [km/s]
							 "sl_vmic"  : MinMax(1e-3, 10),							# [km/s]
							 }

	def __str__(self):
		_str = "<globin.atmos.Atmosphere():\n"
		_str += "  fpath = {}\n".format(self.fpath)
		_str += "  (nx,ny,npar,nz) = ({},{},{},{})>".format(*self.shape)
		return _str

	# def __deepcopy__(self, new):
	# 	new.data = self.data

	def read_multi(self, fpath):
		"""
		Read 1D MULTI type atmosphere data.

		Parameters:
		-----------
		fpath : str
			path to the MULTI type atmosphere.
		"""
		lines = open(fpath, "r").readlines()

		# remove commented lines
		lines = [line.rstrip("\n") for line in lines if "*" not in line]

		# get the scale of the atmosphere
		scale = lines[1].replace(" ", "")[0]
		if (scale.upper()=="M"):
			self.scale_id = globin.scale_id["cmass"]
		if (scale.upper()=="T"):
			self.scale_id = globin.scale_id["tau"]
		if (scale.upper()=="H"):
			self.scale_id = globin.scale_id["height"]

		# get number of depth points
		nz = int(lines[3].replace(" ", ""))

		self.shape = (1, 1, 14, nz)
		self.data = np.zeros(self.shape)
		self.nx, self.ny, self.npar, self.nz = self.shape

		for i_ in range(nz):
			# read first part of the atmosphere
			lista = list(filter(None,lines[4+i_].split(" ")))
			self.data[0,0,0,i_], \
			self.data[0,0,1,i_], \
			self.data[0,0,2,i_], \
			self.data[0,0,3,i_], \
			self.data[0,0,4,i_] = [float(element) for element in lista]

			# read H populations
			lista = list(filter(None,lines[4+nz+i_].split(" ")))
			self.data[0,0,8,i_], \
			self.data[0,0,9,i_], \
			self.data[0,0,10,i_], \
			self.data[0,0,11,i_], \
			self.data[0,0,12,i_], \
			self.data[0,0,13,i_] = [float(element) for element in lista]

		self.logtau = self.data[0,0,0]
		self.logtau_top = self.logtau[0]
		self.logtau_bot = self.logtau[-1]
		self.logtau_step = self.logtau[1] - self.logtau[0]

		self.pg = np.zeros((1,1,nz))
		self.rho = np.zeros((1,1,nz))

	def read_spinor(self, fpath, atm_range=[0,None,0,None]):
		"""
		Read in the SPINOR type atmosphere (fits, 3D, or text, 1D, format).

		Parameters:
		-----------
		fpath : str
			path to the SPINOR type atmosphere.
		"""
		if "fits" in fpath:
			hdu = fits.open(fpath)
			xmin, xmax, ymin, ymax = atm_range
			atmos_data = hdu[0].data[:, xmin:xmax, ymin:ymax]
			ndim = atmos_data.ndim
			_, self.nx, self.ny, self.nz = atmos_data.shape
		else:
			atmos_data = np.loadtxt(fpath, skiprows=1, dtype=np.float64).T
			ndim = atmos_data.ndim
			_, self.nz = atmos_data.shape
			self.nx, self.ny = 1, 1
			atmos_data = np.repeat(atmos_data[:, np.newaxis, :], self.ny, axis=1)
			atmos_data = np.repeat(atmos_data[:, np.newaxis, :, :], self.nx, axis=1)

		self.shape = (self.nx, self.ny, 14, self.nz)
		self.data = np.empty(self.shape)

		# log(tau)
		self.data[:,:,0] = atmos_data[0]
		self.logtau = atmos_data[0,0,0]
		self.logtau_top = self.logtau[0]
		self.logtau_bot = self.logtau[-1]
		self.logtau_step = self.logtau[1] - self.logtau[0]
		# Temperature [K]
		self.data[:,:,1] = atmos_data[2]
		# Electron density [1/cm3]
		if np.count_nonzero(atmos_data[4])!=0:
			self.data[:,:,2] = atmos_data[4]/10 / globin.K_BOLTZMAN / atmos_data[2] / 1e6
		# Vertical velocity [cm/s] --> [km/s]
		self.data[:,:,3] = atmos_data[9]/1e5
		# Microturbulent velocitu [cm/s] --> [km/s]
		self.data[:,:,4] = atmos_data[8]/1e5
		# Magnetic field strength [G]
		self.data[:,:,5] = atmos_data[7]
		# Inclination [rad]
		self.data[:,:,6] = atmos_data[-2]
		# Azimuth [rad]
		self.data[:,:,7] = atmos_data[-1]

		# gas pressure at the top of the atmosphere (in SI units)
		self.pg_top = atmos_data[3,0,0,0]/10

		# container for the gas pressure [dyn/cm2]
		self.pg = np.empty((self.nx, self.ny, self.nz))
		self.pg[:,:,:] = atmos_data[3]

		# rho --> total Hydrogen density [1/cm3] (in RH it is atmos.nHtot)
		m0 = globin.AMU*1e3 # [g]
		avg_mass = np.sum(10**(globin.abundance-12) * globin.atom_mass)
		self.data[:,:,8] = atmos_data[6]/m0/avg_mass

		# container for the height [km]
		height_flag = False
		if np.count_nonzero(atmos_data[1])!=0:
			self.height = np.empty((self.nx, self.ny, self.nz))
			self.height[:,:,:] = atmos_data[1]/1e5 # [km]
			height_flag = True

		# check for the scale
		if np.count_nonzero(self.logtau)!=0:
			self.scale_id = globin.scale_id["tau"]
			return
		elif height_flag:
			self.scale_id = globin.scale_id["height"]
			self.data[:,:,0] = self.height
			return
		elif np.count_nonzero(atmos_data[5])!=0:
			self.scale_id = globin.scale_id["cmass"]
			self.data[:,:,0] = atmos_data[5]
			return
		else:
			raise ValueError("There is none scaling of the atmosphere.")

	def read_sir(self, fpath):
		"""
		Read in the 1D SIR type atmosphere.

		Parameters:
		-----------
		fpath : str
			path to the SIR atmosphere.
		"""
		data = np.loadtxt(fpath, skiprows=1).T
		# we are reversing the order because SIR atmosphere starts from the bottom and
		# here we assume that the top is the first point in the atmosphere
		data = data[:,::-1]

		_, nz = data.shape

		self.shape = (1,1,14,nz)
		self.data = np.zeros(self.shape)
		self.nx, self.ny, self.npar, self.nz = self.shape

		# log(tau) @ 500nm
		self.data[0,0,0] = data[0]
		self.logtau = data[0]
		self.logtau_top = self.logtau[0]
		self.logtau_bot = self.logtau[-1]
		self.logtau_step = self.logtau[1] - self.logtau[0]
		# temperature [K]
		self.data[0,0,1] = data[1]
		# electron concentration [1/cm3]
		self.data[0,0,2] = data[2]/10/globin.K_BOLTZMAN/data[1]/1e6
		# LOS velocity [km/2]
		self.data[0,0,3] = data[5]*1e5
		# vmic [km/s]
		self.data[0,0,4] = data[3]*1e5
		# B [G]
		self.data[0,0,5] = data[4]
		# gamma [rad]
		self.data[0,0,6] = data[6] * np.pi/180
		# theta [rad]
		self.data[0,0,7] = data[7] * np.pi/180
		# Hydrogen populations [1/cm3]
		self.data[0,0,8:] = distribute_hydrogen(data[1], data[-2], data[2])

		self.pg = np.zeros((1,1,nz))
		self.pg[0,0] = data[-2]
		self.rho = np.zeros((1,1,nz))
		self.rho[0,0] = data[-3] # CHECK THIS! MAYBE IT IS HEIGHT and NOT DENSITY

		# check for the scale
		if np.count_nonzero(self.logtau)!=0:
			self.scale_id = globin.scale_id["tau"]
		# 	return
		# elif height_flag:
		# 	self.scale_id = globin.scale_id["height"]
		# 	self.data[:,:,0] = self.height
		# 	return
		# elif np.count_nonzero(atmos_data[5])!=0:
		# 	self.scale_id = globin.scale_id["cmass"]
		# 	self.data[:,:,0] = atmos_data[5]
		# 	return
		else:
			raise ValueError("There is none scaling of the atmosphere.")

	def read_multi_cube(self, fpath, atm_range=[0,None,0,None]):
		"""
		Read the 3D atmosphere of MULTI type in fits format.

		Parameters:
		-----------
		fpath : string
			path to the atmosphere in fits format.
		atm_range : list, optional
			list containing [xmin, xmax, ymin, ymax] defining the patch from the cube
			to be read into the memory.
		"""
		hdu_list = fits.open(fpath)

		if isinstance(atm_range, list):
			xmin, xmax, ymin, ymax = atm_range
			data = hdu_list[0].data[xmin:xmax, ymin:ymax]
		elif isinstance(atm_range, np.ndarray):
			data = hdu_list[0].data[atm_range[0],atm_range[1]][np.newaxis,...]
		else:
			raise ValueError("Unrecognized type of 'atm_range'.")
		
		self.data = data.astype(np.float64, order="C", copy=True) # because of the pyrh module
		self.nx, self.ny, self.npar, self.nz = self.data.shape
		self.shape = self.data.shape
		self.logtau = data[0,0,0]
		self.logtau_top = self.logtau[0]
		self.logtau_bot = self.logtau[-1]
		self.logtau_step = self.logtau[1] - self.logtau[0]
		self.header = hdu_list[0].header
		
		if self.npar!=14:
			raise ValueError(f"MULTI atmosphere is not compatible with globin. It has {self.npar} parameters instead of 14.")

		try:
			self.vmac = float(self.header["VMAC"])
		except:
			pass

		try:
			self.stray_type = self.header["SL_TYPE"]
			self.stray_mode = int(self.header["SL_MODE"])
		except:
			pass

		if self.stray_mode==3:
			# self.global_pars["stray"] = np.asarray([self.header["STRAY"]])
			for parameter in ["stray", "sl_temp", "sl_vz", "sl_vmic"]:
				try:
					self.global_pars[parameter] = np.asarray([self.header[parameter]])
					if parameter=="sl_temp":
						self.hydrostatic = True
				except:
					pass
		
		try:
			self.scale_id = globin.scale_id[self.header["SCALE"].lower()]
		except:
			self.scale_id = globin.scale_id["tau"]

		self.pg = np.empty((self.nx, self.ny, self.nz))
		self.rho = np.empty((self.nx, self.ny, self.nz))

		try:
			self.pg_top = hdu_list[0].header["PGTOP"]
		except:
			pass

		#--- check for the local parameters from inversion
		for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi", "of", "stray", "sl_temp", "sl_vz", "sl_vmic"]:
			try:
				ind = hdu_list.index_of(parameter)
				if isinstance(atm_range, list):
					data = hdu_list[ind].data[xmin:xmax, ymin:ymax, :]
				if isinstance(atm_range, np.ndarray):
					data = hdu_list[ind].data[atm_range[0], atm_range[1]][np.newaxis,...]
				nx, ny, nnodes = data.shape

				self.nodes[parameter] = np.zeros(nnodes)
				for idn in range(nnodes):
					node = hdu_list[ind].header[f"NODE{idn+1}"]
					self.nodes[parameter][idn] = node

				self.values[parameter] = data
				if parameter=="stray":
					self.stray_light = data
					self.add_stray_light = True
				self.mask[parameter] = np.ones(len(self.nodes[parameter]))

				self.parameter_scale[parameter] = np.ones((self.nx, self.ny, nnodes))
			except:
				pass

		#--- try to get the second atmosphere
		try:
			ind = hdu_list.index_of("2nd_Atmosphere")
			
			self.stray_type = "2nd_component"
			self.init_2nd_component()

			if "sl_temp" in self.nodes:
				self.sl_atmos.nodes["sl_temp"] = self.nodes["sl_temp"]
				self.sl_atmos.values["sl_temp"] = self.values["sl_temp"]
				delta = self.values["sl_temp"] - globin.T0_HSRA
				HSRA_temp = interp1d(globin.hsra.logtau, globin.hsra.T[0,0])(self.sl_atmos.logtau)
				self.sl_atmos.data[:,:,self.par_id["temp"]] = HSRA_temp + delta
				self.sl_atmos.hydrostatic = True
			if "sl_vz" in self.nodes:
				self.sl_atmos.nodes["sl_vz"] = self.nodes["sl_vz"]
				self.sl_atmos.values["sl_vz"] = self.values["sl_vz"]
				self.sl_atmos.data[:,:,self.par_id["vz"]] = self.values["sl_vz"]
			if "sl_vmic" in self.nodes:
				self.sl_atmos.nodes["sl_vmic"] = self.nodes["sl_vmic"]
				self.sl_atmos.values["sl_vmic"] = self.values["sl_vmic"]
				self.sl_atmos.data[:,:,self.par_id["vmic"]] = self.values["sl_vmic"]
		except:
			pass

		if self.sl_atmos is not None:
			for parameter in ["sl_temp", "sl_vz", "sl_vmic"]:
					self.build_from_nodes(params=parameter)

			if self.sl_atmos.hydrostatic:
				self.sl_atmos.makeHSE()

		#--- get the number of local parameters
		self.n_local_pars = 0
		for parameter in self.nodes:
			nnodes = len(self.nodes[parameter])
			self.n_local_pars += nnodes

		self.n_global_pars = 0

		#--- check for the spatial regularization weighting
		try:
			self.spatial_regularization_weight = self.header["REGW"]
			self.spatial_regularization = True
		except:
			pass

		if self.spatial_regularization:
			for parameter in self.nodes:
				reg_weight = self.header[f"{parameter}W"]
				self.regularization_weight[parameter] = reg_weight *self.spatial_regularization_weight

		#--- check for the depth-dependent reglarization weights and types
		for parameter in self.dd_regularization_function:
			try:
				self.dd_regularization_function[parameter] = self.header[f"{parameter}DDF"]
				self.dd_regularization_weight[parameter] = self.header[f"{parameter}DDW"]
			except:
				pass

		#--- check for continuum opacity values
		try:
			ind = hdu_list.index_of("Continuum_Opacity")
			self.chi_c = hdu_list[ind].data
		except:
			self.chi_c = None

		#--- check for the height
		try:
			ind = hdu_list.index_of("Height")
			
			unit = hdu_list[ind].header["UNIT"]
			fact = 1
			if unit.lower()=="cm":
				fact = 1e-5 # [cm --> km]
			if unit.lower()=="m":
				fact = 1e-3 # [m --> km]

			self.height = hdu_list[ind].data * fact
		except:
			self.height = np.empty((self.nx, self.ny, self.nz), dtype=np.float64)

	@property
	def T(self):
		return self.data[:,:,1]

	@property
	def vz(self):
		return self.data[:,:,3]

	@property
	def vmic(self):
		return self.data[:,:,4]

	@property
	def B(self):
		return self.data[:,:,5]

	@property
	def gamma(self):
		return self.data[:,:,6]

	@property
	def phi(self):
		return self.data[:,:,7]

	@property
	def ne(self):
		return self.data[:,:,2]

	@property
	def nH(self):
		return self.data[:,:,8]

	@property
	def scale(self):
		return list(globin.scale_id.keys())[self.scale_id].upper()

	def _remove_local_pars(self):
		self.nodes = {}
		self.values = {}
		self.n_local_pars = 0
		self.hydrostatic = False

	def is_valid(self):
		for idx in range(self.nx):
			for idy in range(self.ny):
				for idp in [1,2,3,4,5,6,7,8]:
					if np.isnan(self.data[idx,idy,idp]).any():
						return False, (idx,idy,idp)
		return True, None

	def split(self, size, fpath):
		"""
		Split the atmosphere in workable chunks that will be Scattered to
		each process/thread.
		"""
		self.indx = np.array_split(self.idx_meshgrid, size)
		self.indy = np.array_split(self.idy_meshgrid, size)

		# atm_chunks = [None]*size
		
		dic = self.__dict__
		keys = dic.keys()
		

		for idt in range(size):
			print(f"{idt+1}/{size}")
			idx = self.indx[idt]
			idy = self.indy[idt]
			
			atm_chunk = Atmosphere(nx=1, ny=len(idy), nz=self.nz)

			for key in keys:

				if key=="data":
					atm_chunk.data[0,:,:,:] = self.data[idx,idy]
					atm_chunk.shape = atm_chunk.data.shape
				elif key in ["nx", "ny", "npar", "nz", "shape"]:
					pass
				elif key in ["pg", "height", "rho", "nHtot"]:
					pass
				elif key in ["idx_meshgrid", "idy_meshgrid"]:
					pass
				elif key in ["rank", "size", "use_mpi"]:
					pass
				else:
					setattr(atm_chunk, key, dic[key])

			atm_chunk.save_atmosphere(f"{fpath}/atm_c{idt+1}.fits")

			# atm_chunks[idt].rank = idt

		# return atm_chunks

	def rebin(self, new_shape):
		nx, ny = new_shape

		assert self.nx%nx==0
		assert self.ny%ny==0

		dx = self.nx//nx
		dy = self.ny//ny

		new_atmos = np.empty((nx, ny, self.npar, self.nz))

		for idx in range(1,nx+1):
			for idy in range(1,ny+1):
				for idp in range(1,9):
					new_atmos[idx-1,idy-1,idp] = np.mean(self.data[(idx-1)*dx:idx*dx, (idy-1)*dy:idy*dy, idp], axis=(0,1))

		self.data = new_atmos
		self.nx = self.nx
		self.ny = self.ny
		self.shape = self.data.shape

		return self

	def extract(self, slice_x, slice_y, slice_z):
		dic = self.__dict__
		keys = dic.keys()

		idx_min, idx_max = slice_x
		idy_min, idy_max = slice_y
		idz_min, idz_max = slice_z

		new_atmos = globin.Atmosphere()

		for key in keys:

			if key=="data":
				new_atmos.data = self.data[idx_min:idx_max, idy_min:idy_max, :, idz_min:idz_max]
				new_atmos.shape = new_atmos.data.shape
				new_atmos.nx, new_atmos.ny, _, new_atmos.nz = new_atmos.data.shape
				new_atmos.logtau = self.logtau[idz_min:idz_max]
				new_atmos.height = self.height[idx_min:idx_max, idy_min:idy_max, idz_min:idz_max]
				idx, idy = np.meshgrid(np.arange(new_atmos.nx), np.arange(new_atmos.ny))
				new_atmos.idx_meshgrid = idx.ravel()
				new_atmos.idy_meshgrid = idy.ravel()
				for parameter in self.nodes:
					new_atmos.nodes[parameter] = self.nodes[parameter]
					new_atmos.values[parameter] = self.values[parameter][idx_min:idx_max, idy_min:idy_max, idz_min:idz_max]
			elif key in ["nx", "ny", "npar", "nz", "shape", "logtau", "height"]:
				pass
			elif key in ["pg", "height", "rho", "nHtot"]:
				pass
			elif key in ["idx_meshgrid", "idy_meshgrid"]:
				pass
			elif key in ["rank", "size", "use_mpi"]:
				pass
			else:
				setattr(new_atmos, key, dic[key])

		return new_atmos

	def get_atmos(self, indx, indy):
		nx = 1
		ny = len(indy)

		dtau = self.logtau[1] - self.logtau[0]
		new = Atmosphere(nx=nx, ny=ny, nz=self.nz, logtau_top=self.logtau[0], logtau_bot=self.logtau[-1], logtau_step=dtau)

		# this needs to be cleaned and optimized...
		new.cwd = self.cwd
		new.scale_id = self.scale_id
		# new.fudge_lam = self.fudge_lam
		# new.fudge = self.fudge[indx,indy]
		# new.global_pars = self.global_pars
		#-------------------------------------------

		new.data[0,:,:,:] = self.data[indx,indy]
		new.logtau = self.logtau

		for parameter in self.nodes:
			new.nodes[parameter] = self.nodes[parameter]
			new.values[parameter] = self.values[parameter][indx,indy]
			# nnodes = len(self.nodes[parameter])
			# new.nodes[parameter] = np.zeros((1,1,nnodes))
			# new.nodes[parameter][0,0] = self.nodes[parameter]
			# new.values[parameter] = np.zeros((1,1,nnodes))
			# new.values[parameter][0,0] = self.values[parameter][idx,idy]
			# new.parameter_scale[parameter][0,0] = self.parameter_scale[parameter][idx,idy]

		return new

	def build_from_nodes(self, flag=None, params=None, pool=None):
		"""
		Construct the atmosphere from node values for given parameters.

		It interpolates eighter using Bezier 3rd/2nd degree polynomial or using
		the spline interpolation of 3rd/2nd degree without tensions. Bezier
		interpolation is implemented from de la Cruz Rodriguez & Piskunov(2013)
		[implemented in the STiC code].

		Extrapolation to top/bottom of the atmosphere is assumed to be linear in
		all parameters.

		Temperature: 
			Extrapolation to the bottom assumes the slope that is
			identical	to the temeprature gradient in FAL C atmosphere model at the
			depth of the deepest node in the atmosphere. Extrapolation to the top is
			limited by self.Tmin and self.Tmax. Minimum check is there to limit the
			temprature going below certain value where RH is unable to compute the
			spectrum (partition functions or collisional coefficients are not
			define). Maximum check is currently there to limit strong gradients
			during inversion. For NLTE inversions it should be removed afterwards.
			Idea around this is based on Milic & van Noort (2018) [SNAPI code].

		vz, vmic, mag, gamma, chi: 
			Extrapolation to the top and the bottoms has
			the gradient that is defined by the consecutive nodes at the highest and
			lowest position in the atmosphere. We only check the extrapolation
			limits for the 'vmic' and 'mag'. RH can accept negative values for
			magnetic field, but it assumes that it is reversed polarity. Here, we
			control the polarity with 'gamma' parameter, and therefore the
			interpolation will artificially change the polarity, which we do not
			want to.

		Parameters:
		-----------
		flag : ndarray, optional
			flag for building the atmosphere (==1) or not (==0) for each pixel. 
			It has dimension of (self.nx, self.ny). Default is None (interpolate
			all pixels).
		params : str, optional
			parameter which we want to interpolate. If it is None, we assume that 
			all parameters need to be interpolated. Default is None.
		
		"""
		# interpolate atmosphere which is used as the stray light source (2nd component)
		if self.sl_atmos is not None:
			if self.stray_mode in [1,2]:
				sl_parameters = list(self.sl_atmos.nodes.keys())
				sl_values = self.values
			if self.stray_mode==3:
				sl_parameters = list(self.sl_atmos.global_pars.keys())
				sl_values = self.global_pars
				# sl_parameters = [par for par in sl_parameters if "sl_" in par]

			for parameter in sl_parameters:
				# skip non-fit parameters; we already assigned them on the setup
				if self.stray_mode in [1,2]:	
					if parameter not in self.nodes:
						continue

				if parameter in ["loggf", "dlam", "stray"]:
					continue

				if parameter=="sl_temp":
					delta = sl_values[parameter] - globin.T0_HSRA
					HSRA_temp = interp1d(globin.hsra.logtau, globin.hsra.T[0,0])(self.sl_atmos.logtau)
					self.sl_atmos.data[:,:,self.par_id[parameter.split("_")[1]]] = HSRA_temp + delta
				else:
					self.sl_atmos.data[:,:,self.par_id[parameter.split("_")[1]]] = sl_values[parameter]

			# stop futher call if we only needed to build the parameter of the 2nd component atmosphere
			if params in ["sl_temp", "sl_vz", "sl_vmic"]:
				return			

		if flag is None:
			flag = np.ones((self.nx, self.ny))

		flag = np.asarray(flag)

		atmos = [self]*(self.nx*self.ny)
		params = [params]*(self.nx*self.ny)
		args = zip(atmos, flag.ravel(), self.idx_meshgrid, self.idy_meshgrid, params)

		if pool is None:
			with mp.Pool(self.n_thread) as pool:
				results = pool.map(func=self._build_from_nodes, iterable=args, chunksize=self.chunk_size)
		else:
			results = pool.map(self._build_from_nodes, args)

		results = np.asarray(results)
		self.data = results.reshape(self.nx, self.ny, self.npar, self.nz, order="F")

	def _build_from_nodes(self, args):
		"""
		Parallelized call from self.build_from_nodes() function.
		"""
		def add_node(new_node, x, y, ymin, ymax, where="beginning"):
			"""
			Add the 'new_node' point to the atmosphere as a node by linear extrapolation
			to the top.

			Before adding the value, check if it is in the boundaries.

			Key 'where' regulates where we add this node: at the top ('beginning') or
			at the bottom ('end').
			"""
			if where=="beginning":
				K0 = (y[1]-y[0])/(x[1]-x[0])
				n = y[0] - K0*x[0]
			if where=="end":
				K0 = (y[-1]-y[-2])/(x[-1]-x[-2])
				n = y[-1] - K0*x[-1]
			y0 = K0*new_node + n
			if ymin is not None:
				if y0<ymin:
					y0 = ymin
			if ymax is not None:
				if y0>ymax:
					y0 = ymax
			if where=="beginning":
				y = np.append(y0, y)
				x = np.append(new_node, x)
			if where=="end":
				y = np.append(y, y0)
				x = np.append(x, new_node)

			return x, y
		
		atmos, flag, idx, idy, params = args

		parameters = self.nodes
		if params is not None:
			if not isinstance(params, list):
				parameters = [params]
			else:
				parameters = params

		if flag==0:
			return atmos.data[idx,idy]
		
		for parameter in parameters:
			# skip over OF and stray light parameters
			if parameter in ["stray", "of", "sl_temp", "sl_vz", "sl_vmic"]:
				continue

			# K0, Kn by default; True for vmic, mag, gamma and chi
			# K0, Kn = None, None
			K0, Kn = 0, 0

			x = self.nodes[parameter]
			y = self.values[parameter][idx,idy]

			# if we have a single node
			if len(x)==1:
				y_new = np.ones(self.nz) * y
				atmos.data[idx,idy,atmos.par_id[parameter],:] = y_new
				continue

			# for 2+ number of nodes
			if parameter=="temp":
				if self.interpolation_method=="bezier":	
					K0 = (y[1]-y[0]) / (x[1]-x[0])
					# bottom node slope for extrapolation based on temperature gradient from FAL C model
					Kn = splev(x[-1], globin.temp_tck, der=1)
				if self.interpolation_method=="spline":
					# add top of the atmosphere as a node (ask SPPINOR devs why ...)
					# to the bottom we assume that the gradient is based only on the node positions;
					# this is not fully reallistic thing to do, but... I do not wanna implement extrapolation
					# using adiabatic and HSE assumption like in SPINOR for now to show similarities between them
					x, y = add_node(self.logtau[0], x, y, self.Tmin, self.Tmax)
					K0, Kn = get_K0_Kn(x, y, tension=self.spline_tension)
				
				# check if extrapolation at the top atmosphere point goes below the minimum
				# if does, change the slope so that at top point we have Tmin (globin.limit_values["temp"][0])
				if self.Tmin>(y[0] + K0 * (atmos.logtau[0]-x[0])):
					K0 = (self.Tmin - y[0]) / (atmos.logtau[0] - x[0])
				# temperature can not go below 1900 K because the RH will not compute spectrum (dunno why)
				# if self.Tmax<(y[0] + K0 * (atmos.logtau[0]-x[0])):
				# 	K0 = (self.Tmax - y[0]) / (atmos.logtau[0] - x[0])
				
			elif parameter in ["gamma", "chi"]:
				if self.interpolation_method=="bezier":
					K0 = (y[1]-y[0]) / (x[1]-x[0])
					Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
				if self.interpolation_method=="spline":
					x, y = add_node(self.logtau[0], x, y, None, None)
					K0, Kn = get_K0_Kn(x, y, tension=self.spline_tension)
			
			elif parameter in ["vz", "mag", "vmic"]:
				if self.interpolation_method=="bezier":
					K0 = (y[1]-y[0]) / (x[1]-x[0])
					Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
				if self.interpolation_method=="spline":
					x, y = add_node(self.logtau[0], x, y, self.limit_values[parameter].min[0], self.limit_values[parameter].max[0])
					K0, Kn = get_K0_Kn(x, y, tension=self.spline_tension)
				
				if parameter in ["mag", "vmic"]:
					# check if extrapolation at the top atmosphere point goes below the minimum
					# if does, change the slopte so that at top point we have parameter_min (globin.limit_values[parameter][0])
					if self.limit_values[parameter].min[0]>(y[0] + K0 * (atmos.logtau[0]-x[0])):
						K0 = (self.limit_values[parameter].min[0] - y[0]) / (atmos.logtau[0] - x[0])
					# if self.limit_values[parameter].max[0]<(y[0] + K0 * (atmos.logtau[0]-x[0])):
					# 	K0 = (self.limit_values[parameter].max[0] - y[0]) / (atmos.logtau[0] - x[0])
					# similar for the bottom for maximum/min values
					# if self.limit_values[parameter].max[0]<(y[-1] + Kn * (atmos.logtau[-1]-x[-1])):
					# 	Kn = (self.limit_values[parameter].max[0] - y[-1]) / (atmos.logtau[-1] - x[-1])
					if self.limit_values[parameter].min[0]>(y[-1] + Kn * (atmos.logtau[-1]-x[-1])):
						Kn = (self.limit_values[parameter].min[0] - y[-1]) / (atmos.logtau[-1] - x[-1])

			if self.interpolation_method=="bezier":
				y_new = bezier_spline(x, y, atmos.logtau, K0=K0, Kn=Kn, degree=self.interp_degree, extrapolate=True)
			if self.interpolation_method=="spline":
				y_new = spline_interpolation(x, y, atmos.logtau, tension=self.spline_tension, K0=K0, Kn=Kn)

			atmos.data[idx,idy,atmos.par_id[parameter],:] = y_new

		return atmos.data[idx,idy]

	def makeHSE(self, flag=None, pool=None):
		"""
		For given atmosphere structure (logtau and temperature) compute electron
		and hydrogen populations assuming hydrostatic equilibrium, LTE and ideal
		gas equation of state.

		Parameters:
		-----------
		flag : ndarray, optional
			flag for computing the populations (==1) or not (==0) for each pixel. 
			It has dimension of (self.nx, self.ny). Default is None (compute HSE in
			all pixels).

		"""
		if flag is None:
			flag = np.ones((self.nx, self.ny))
		indx, indy = np.where(flag==1)
		args = zip(indx, indy)

		# obtain new Pg and use it as initial value for the HSE at the top
		self.get_pg()

		if pool is None:
			with mp.Pool(self.n_thread) as pool:
				results = pool.map(func=self._makeHSE, iterable=args, chunksize=self.chunk_size)
			pool = None
		else:
			results = pool.map(self._makeHSE, args)

		results = np.array(results)

		self.data[indx,indy,2] = results[:,0,:]
		self.data[indx,indy,8] = results[:,1,:]

		# solve HSE also for the 2nd atmospheric model
		if self.sl_atmos is not None:
			if self.stray_mode==3:
				flag = None
			self.sl_atmos.makeHSE(flag, pool=pool)

		# self.data[indx,indy,2] = results[:,0,:]
		# self.data[indx,indy,8:] = results[:,1:7,:]
		# self.rho[indx,indy] = results[:,7,:]
		# self.pg[indx,indy] = results[:,8,:]

	def _makeHSE(self, arg):
		"""
		Parallelized call from makeHSE() function.
		"""
		idx, idy = arg

		fudge_lam = None
		fudge_value = None
		if self.fudge_lam is not None:
			fudge_lam = self.fudge_lam
			fudge_value = self.fudge[idx,idy]

		ne, nHtot = pyrh.hse(cwd=self.cwd, 
					   		 atm_scale=self.scale_id,
							 scale=self.data[idx, idy, 0], 
							 temp=self.data[idx, idy, 1], 
							 pg_top=self.pg[idx,idy,0]/10, 
							 fudge_wave=fudge_lam, 
							 fudge_value=fudge_value,
							 atomic_number=self.atomic_number, 
							 atomic_abundance=self.atomic_abundance,
							 full_output=False)

		return np.vstack((ne/1e6, nHtot/1e6))

	def get_pg(self):
		"""
		Compute the gas pressure from total hydrogen density and electron density.

		Units dyn/cm2 (CGS).
		"""
		nH = self.nH * 1e6 # [m3]
		ne = self.ne * 1e6 # [m3]
		self.pg = (nH*globin.totalAbundance + ne) * globin.K_BOLTZMAN * self.data[...,1,:] * 10 # [dyn/cm2]

	def get_ne_from_nH(self, scale="tau"):
		"""
		Compute the electron concentration using RH from temperature and 
		total hydrogen density.
		"""
		if scale=="tau":
			self.scale_type = 0
			self.scale = self.data[:,:,0]
		if scale=="height":
			self.scale_type = 2
			self.scale = self.height[:,:]

		for idy in range(3):
			# print(f"{idy+1}/{self.ny}")
			for idx in range(self.nx):
				# print(f"{idx+1}/{self.nx}")
				pyrh.get_ne_from_nH(self.cwd, self.scale_type, self.scale[idx,idy], self.data[idx,idy])

		return

		args = zip(self.idx_meshgrid, self.idy_meshgrid)

		with mp.Pool(self.n_thread) as pool:
			results = pool.map(self._get_ne_from_nH, iterable=args, chunksize=self.chunk_size)

		results = np.array(results)
		print(results.shape)
		# self.data[:,:,2] = results

		del self.scale_type
		del self.scale

	def _get_ne_from_nH(self, args):
		idx, idy = args

		pyrh.get_ne_from_nH(self.cwd, self.scale_type, self.scale[idx,idy], self.data[idx,idy])

		return self.data[idx,idy,2]

	def get_scales(self, referent_wavelength):
		tau = np.empty((self.nx, self.ny, self.nz))
		height = np.empty((self.nx, self.ny, self.nz))
		cmass = np.empty((self.nx, self.ny, self.nz))

		for idx in range(self.nx):
			for idy in range(self.ny):
				tau[idx,idy], height[idx,idy], cmass[idx,idy] = pyrh.get_scales(self.cwd,
																					  						 self.scale_id, self.data[idx,idy,0],
																					  						 self.data[idx,idy],
																					  						 referent_wavelength)

		if self.scale_id==0:
			tau = 10**self.data[:,:,0]
		if self.scale_id==1:
			cmass = 10**self.data[:,:,0]
		if self.scale_id==2:
			height = self.data[:,:,0]*1e3

		return np.log10(tau), height/1e3, np.log10(cmass)
		# return tau, height, cmass

	def interpolate_atmosphere(self, z_new, ref_atm):
		"""
		
		Interpolate the structure of reference atmosphere into the new atmosphere.
		Used in inversions to make sure that our atmosphere has all the
		parameters it needs to compute the spectrum. It is most usefull when
		doing test with MHD atmospheres so that we can invert only one of the
		parameters while the rest has the same values as in the cube.

		Parameters:
		-----------
		z_new : array
			optical depth scale on which we want to interpolate the 'ref_atm'.
		ref_atm : ndarray
			reference atmosphere in the format (nx, ny, npar, nz) of MULTI type.

		Errors:
		-------
		If the top of the new atmosphere is higher than the reference atmosphere,
		we will have to extrapolate. In order to stop that, the function will
		throw an error and ask the user to change the top of the new atmosphere
		or to change the reference atmosphere that goes higher, if needed.
		
		"""
		z_new = np.round(z_new, decimals=2)
		ref_atm[:,:,0] = np.round(ref_atm[:,:,0], decimals=2)

		if (z_new[0]<ref_atm[0,0,0,0]) or (z_new[-1]>ref_atm[0,0,0,-1]):
			print("[Warning] The atmosphere will be extrapolated from {} to {} in the optical depth.\n".format(ref_atm[0,0,0,0], z_new[0]))

		self.nz = len(z_new)

		# check if reference atmosphere is 1D
		ref_atm_nx, ref_atm_ny, _, _ = ref_atm.shape
		oneD = (ref_atm_nx*ref_atm_ny==1)

		self.data = np.empty((self.nx, self.ny, self.npar, self.nz), dtype=np.float64)
		self.nHtot = np.empty((self.nx, self.ny, self.nz), dtype=np.float64)
		self.pg = np.empty((self.nx, self.ny, self.nz), dtype=np.float64)
		self.rho = np.empty((self.nx, self.ny, self.nz), dtype=np.float64)
		self.data[:,:,0,:] = z_new
		self.logtau = z_new

		if oneD:
			for idp in range(1, self.npar):
				if idp==2 or idp==8:
					fun = interp1d(ref_atm[0,0,0], np.log10(ref_atm[0,0,idp]), kind=3, fill_value="extrapolate")
					self.data[:,:,idp,:] = 10**(fun(z_new))
				else:
					fun = interp1d(ref_atm[0,0,0], ref_atm[0,0,idp], kind=3, fill_value="extrapolate")
					self.data[...,idp,:] = fun(z_new)
			# self.nHtot = np.sum(self.data[...,8:,:], axis=2)
		else:
			for idx in range(self.nx):
				for idy in range(self.ny):
					for idp in range(1, self.npar):
						if idp==2 or idp==8:
							fun = interp1d(ref_atm[idx,idy,0], np.log10(ref_atm[idx,idy,idp]), kind=3, fill_value="extrapolate")
							self.data[idx,idy,idp] = 10**(fun(z_new))
						else:
							fun = interp1d(ref_atm[idx,idy,0], ref_atm[idx,idy,idp], kind=3, fill_value="extrapolate")
							self.data[idx,idy,idp] = fun(z_new)
					# self.nHtot[idx,idy] = np.sum(self.data[idx,idy,8:,:], axis=0)
	
	def reinterpolate_atmosphere(self, z_new):
		"""
		Interpolate the atmosphere onto a new 'z_new' grid.
		"""
		if z_new[0]<np.max(self.data[...,0,0]):
			raise ValueError("The top value of the new scale is outside of the atmosphere scale.")
		if z_new[-1]>np.min(self.data[...,0,-1]):
			raise ValueError("The bottom value of the new scale is outside of the atmosphere scale.")

		new = Atmosphere(nx=self.nx, ny=self.ny, nz=len(z_new))
		new.data[...,0,:] = z_new

		for idp in range(1,9):
			for idx in range(self.nx):
				for idy in range(self.ny):
					new.data[idx,idy,idp] = interp1d(self.data[idx,idy,0], self.data[idx,idy,idp], kind=3)(z_new)

		self.data = new.data
		self.logtau = z_new
		self.logtau_top = z_new[0]
		self.logtau_bot = z_new[-1]
		self.logtau_step = z_new[-1] - z_new[-2]

		return self

	def resample(self, Nz):
		xnew = np.linspace(self.logtau[0], self.logtau[-1], num=Nz)

		cube = np.ones((self.nx, self.ny, self.npar, Nz))
	
		for idx in range(self.nx):
			for idy in range(self.ny):
				for idp in range(1, 8):
					if idp!=2:
						# tck = splrep(self.logtau, self.data[idx,idy,idp])
						# cube[idx,idy,idp] = splev(xnew, tck)
						cube[idx,idy,idp] = interp1d(self.logtau, self.data[idx,idy,idp])(xnew)

		self.data = cube
		self.data[:,:,0] = xnew
		self.logtau = xnew
		self.shape = self.data.shape
		self.nz = Nz
		self.rho = np.zeros((self.nx, self.ny, self.nz))
		self.pg = np.zeros((self.nx, self.ny, self.nz))

		self.makeHSE()

	def save_atmosphere(self, fpath="inverted_atmos.fits", kwargs=None):
		"""
		Save the atmosphere to a fits file with all the parameters.

		Primary header is reserved for the full atmosphere of shape (nx, ny, npar,
		nz). The following headers contain the values for each atmosphere
		parameter (if we have used the atmosphere in inversion mode). 

		Primary header also contains the values that are parameter independent
		(spatial regularization weight), gas pressure at the top,
		macro-turbulent velocity, as well as weights for each parameter and its
		function for regularization.

		Parameters:
		-----------
		fpath : str, optional
			file name for the atmosphere to be saved. Default value is 'inverted_atmos.fits'.
		kwargs : dict, optional
			additional values that we want to save into the atmosphere.
		"""
		primary = fits.PrimaryHDU(self.data, do_not_scale_image_data=True)
		primary.name = "Atmosphere"

		primary.header.comments["NAXIS1"] = "depth points"
		primary.header.comments["NAXIS2"] = "number of parameters"
		primary.header.comments["NAXIS3"] = "y-axis atmospheres"
		primary.header.comments["NAXIS4"] = "x-axis atmospheres"

		# primary.header["XMIN"] = f"{self.xmin+1}"
		# if self.xmax is None:
		# 	self.xmax = self.nx
		# primary.header["XMAX"] = f"{self.xmax}"
		# primary.header["YMIN"] = f"{self.ymin+1}"
		# if self.ymax is None:
		# 	self.ymax = self.ny
		# primary.header["YMAX"] = f"{self.ymax}"

		primary.header["NX"] = self.nx
		primary.header["NY"] = self.ny
		primary.header["NZ"] = self.nz

		# set the value for atmosphere scale type
		try:
			scale = list(globin.scale_id.keys())[self.scale_id]
			primary.header["SCALE"] = (scale.upper(), "scale type")
			if self.scale_id==1:
				primary.header["SCALEU"] = ("cm^2/g", "scale unit")
			if self.scale_id==2:
				primary.header["SCALEU"] = ("km", "scale unit")
		except:
			pass

		# save spatial regularization weights
		if self.spatial_regularization:
			primary.header["REGW"] = (self.spatial_regularization_weight, "regularization weight")
			for parameter in self.regularization_weight:
				weight = self.regularization_weight[parameter] / self.spatial_regularization_weight
				primary.header[f"{parameter}W"] = (weight, "relative spatial reg. weight")

		# save depth-dependent regularization
		for parameter in self.dd_regularization_function:
			if self.dd_regularization_function[parameter]!=0:
				primary.header[f"{parameter}DDF"] = (self.dd_regularization_function[parameter], "depth-dependent regularization type")
				primary.header[f"{parameter}DDW"] = (self.dd_regularization_weight[parameter], "depth-dependent regularization weight")

		# save the gass pressure at the top of the atmosphere
		if self.pg_top is not None:
			primary.header["PGTOP"] = (float(self.pg_top), "gas pressure at top (in SI units)")

		# save the stray light factor and mode of application
		if self.add_stray_light:
			primary.header["SL_TYPE"] = (self.stray_type, "stray light type")
			primary.header["SL_MODE"] = (self.stray_mode, " -1 no inversion; 3 global inversion")
			if self.invert_stray:
				primary.header["SL_FIT"] = ("True", "flag for fitting stray factor")
			else:
				primary.header["SL_FIT"] = ("False", "flag for fitting stray factor")
			if self.stray_mode==3:
				if self.invert_stray:
					primary.header["STRAY"] = (self.global_pars["stray"][0], "stray light factor")
				else:
					primary.header["STRAY"] = (self.stray_light[0,0,0], "stray light factor")

		if "vmac" in self.global_pars:
			primary.header["VMAC"] = ("{:5.3f}".format(self.vmac[0]), "macro-turbulen velocity [km/s]")
			primary.header["VMAC_FIT"] = ("True", "flag for fitting macro velocity")
		else:
			primary.header["VMAC"] = ("{:5.3f}".format(self.vmac), "macro-turbulen velocity [km/s]")
			primary.header["VMAC_FIT"] = ("False", "flag for fitting macro velocity")

		for parameter in ["sl_temp", "sl_vz", "sl_vmic"]:
			if parameter not in self.global_pars:
				continue

			primary.header[parameter.upper()] = self.global_pars[parameter][0]

		# add keys from kwargs (as dict)
		if kwargs:
			for key in kwargs:
				primary.header[key] = kwargs[key]

		hdulist = fits.HDUList([primary])

		# make separate HDU for each parameter (node values only)
		for parameter in self.nodes:
			par_hdu = fits.ImageHDU(self.values[parameter])
			par_hdu.name = parameter

			# par_hdu.header["unit"] = globin.parameter_unit[parameter]
			par_hdu.header.comments["NAXIS1"] = "number of nodes"
			par_hdu.header.comments["NAXIS2"] = "y-axis atmospheres"
			par_hdu.header.comments["NAXIS3"] = "x-axis atmospheres"

			# write down the node positions
			for idn in range(len(self.nodes[parameter])):
				par_hdu.header[f"NODE{idn+1}"] = self.nodes[parameter][idn]

			hdulist.append(par_hdu)

		if self.sl_atmos is not None:
			par_hdu = fits.ImageHDU(self.sl_atmos.data)
			par_hdu.name = "2nd_Atmosphere"

			par_hdu.header.comments["NAXIS1"] = "depth points"
			par_hdu.header.comments["NAXIS2"] = "number of parameters"
			par_hdu.header.comments["NAXIS3"] = "y-axis atmospheres"
			par_hdu.header.comments["NAXIS4"] = "x-axis atmospheres"

			hdulist.append(par_hdu)

		# save the continuum opacity (if we have it computed)
		# [22.11.2022.] Obsolete? Maybe we will need it later...
		if self.chi_c is not None:
			par_hdu = fits.ImageHDU(self.chi_c)
			par_hdu.name = "Continuum_Opacity"

			par_hdu.header.comments["NAXIS1"] = "number of wavelength points"
			par_hdu.header.comments["NAXIS2"] = "y-axis atmospheres"
			par_hdu.header.comments["NAXIS3"] = "x-axis atmospheres"

			hdulist.append(par_hdu)

		# height HDU
		par_hdu = fits.ImageHDU(self.height)
		par_hdu.name = "Height"

		par_hdu.header.comments["NAXIS1"] = "number of depths"
		par_hdu.header.comments["NAXIS2"] = "y-axis atmospheres"
		par_hdu.header.comments["NAXIS3"] = "x-axis atmospheres"
		par_hdu.header["UNIT"] = "KM"

		hdulist.append(par_hdu)

		# if globin.of_mode:
		# 	par_hdu = fits.ImageHDU(self.of_value)
		# 	par_hdu.name = "opacity_fudge"

		# 	par_hdu.header.comments["NAXIS1"] = "number of wavelength points"
		# 	par_hdu.header.comments["NAXIS2"] = "y-axis atmospheres"
		# 	par_hdu.header.comments["NAXIS3"] = "x-axis atmospheres"

		# 	hdulist.append(par_hdu)

		hdulist.writeto(fpath, overwrite=True)

	def save_atomic_parameters(self, fpath="inverted_atoms.fits", kwargs=None):
		"""
		Save the global parameters (vmac, stray-light, log(gf) and dlam) into fits
		file.

		Parameters:
		-----------
		fpath : str, optional
			name of the fits file to which we are saving atomic parameters. Default 
			value is 'inverted_atoms.fits'.
		kwargs : dict, optional
			additional values that we want to save in header.
		"""
		primary = fits.PrimaryHDU()
		hdulist = fits.HDUList([primary])
		
		for parameter in self.global_pars:
			if (parameter=="vmac") or (parameter=="stray") or ("sl_" in parameter):
				primary.header[parameter] = self.global_pars[parameter][0]
				primary.header["MODE"] = self.mode
				continue

			if self.mode==2:
				nx, ny = self.nx, self.ny
			elif self.mode==3:
				nx, ny = 1,1

			matrix = np.zeros((self.nx, self.ny, 2, self.line_no[parameter].size))
			matrix[:,:,0] = self.line_no[parameter]
			matrix[:,:,1] = self.global_pars[parameter]

			par_hdu = fits.ImageHDU(matrix)
			par_hdu.name = parameter

			par_hdu.header.comments["NAXIS1"] = "number of lines"
			par_hdu.header.comments["NAXIS2"] = "1 - line IDs | 2 - line values"
			par_hdu.header.comments["NAXIS3"] = "y-axis number of pixels"
			par_hdu.header.comments["NAXIS4"] = "x-axis number of pixels"
			
			if kwargs:
				for key in kwargs:
					par_hdu.header[key] = kwargs[key]

			hdulist.append(par_hdu)

			# boundary values for each line
			par_hdu = fits.ImageHDU(self.limit_values[parameter])
			par_hdu.name = f"MINMAX_{parameter}"

			hdulist.append(par_hdu)

		hdulist.writeto(fpath, overwrite=True)

	def save_populations(self, fpath="populations.fits", kwargs=None):
		primary = fits.PrimaryHDU()
		hdulist = fits.HDUList([primary])

		for element in self.n_pops:
			par_hdu = fits.ImageHDU(self.n_pops[element])
			par_hdu.name = f"{element}_N"
			hdulist.append(par_hdu)

			par_hdu = fits.ImageHDU(self.nstar_pops[element])
			par_hdu.name = f"{element}_Nstar"
			hdulist.append(par_hdu)

		hdulist.writeto(fpath, overwrite=True)

	def load_populations(self, fpath):
		hdulist = fits.open(fpath)

		self.n_pops = {}
		self.nstar_pops = {}
		for idh in range(1, len(hdulist)-1, 2):
			element = hdulist[idh].name.split("_")[0]

			self.n_pops[element] = hdulist[idh].data
			self.nstar_pops[element] = hdulist[idh+1].data

	def update_parameters(self, proposed_steps):
		"""
		Change the inversion parameters (local and global) based on the given steps.

		Parameters:
		-----------
		proposed_steps : ndarray
			step for each parameter in inversion. In case of mode=1 and mode=2 
			inversions it has a shape (self.nx, self.ny, num_of_parameters). 
			In case of mode=3 inversion, its shape is 
			(self.nx * self.ny * n_local_parameters + n_global_parameters).
		"""
		if self.mode==1 or self.mode==2:
			
			#--- update atmospheric parameters
			idp = 0
			for parameter in self.nodes:
				for idn in range(len(self.nodes[parameter])):
					if self.mask[parameter][idn]==0:
						continue
					step = proposed_steps[...,idp] / self.parameter_scale[parameter][...,idn]
					self.values[parameter][...,idn] += step
					idp += 1

			#--- update atomic parameters
			low_ind, up_ind = self.n_local_pars, self.n_local_pars
			for parameter in self.global_pars:
				if self.line_no[parameter].size > 0:
					low_ind = up_ind
					up_ind += self.line_no[parameter].size
					step = proposed_steps[:,:,low_ind:up_ind] / self.parameter_scale[parameter]
					self.global_pars[parameter] += step
					if self.sl_atmos is not None:
						self.sl_atmos.global_pars[parameter] += step

		if self.mode==3:
			Natmos = self.nx*self.ny
			NparLocal = self.n_local_pars
			NparGlobal = self.n_global_pars

			#--- update local parameters (atmospheric)			
			local_pars = proposed_steps[:NparLocal*Natmos]
			local_pars = local_pars.reshape(self.nx, self.ny, NparLocal, order="C")

			low_ind, up_ind = 0, 0
			idp = 0
			for parameter in self.nodes:
				for idn in range(len(self.nodes[parameter])):
					if self.mask[parameter][idn]==0:
						continue
					step = local_pars[...,idp] / self.parameter_scale[parameter][...,idn]
					self.values[parameter][...,idn] += step
					idp += 1

			#--- update global (atomic) parameters and macro-turbulent velocity
			global_pars = proposed_steps[NparLocal*Natmos:]
			low_ind, up_ind = 0, 0
			for parameter in self.global_pars:
				if (parameter=="vmac") or (parameter=="stray") or ("sl_" in parameter):
					low_ind = up_ind
					up_ind += 1
					step = global_pars[low_ind:up_ind] / self.parameter_scale[parameter]
					self.global_pars[parameter] += step
				if parameter=="loggf" or parameter=="dlam":
					Nlines = self.global_pars[parameter].size
					if Nlines>0:
						low_ind = up_ind
						up_ind += Nlines
						step = global_pars[low_ind:up_ind] / self.parameter_scale[parameter]
						self.global_pars[parameter] += step
						if self.sl_atmos is not None:
							self.sl_atmos.global_pars[parameter] += step

	def check_parameter_bounds(self, mode):
		"""
		After updating the inversion parameters, check if they are inside the
		defined bounds. If they are not, set the values to the closess boundary
		(min/max value).

		Magnetic field angles are boundless and are always wrapped. The
		inclination is wrapped in [0,180] degrees while the azimuth is wrapped in
		[-90, 90] degrees interval.
		"""
		for parameter in self.values:
			# inclination is wrapped around [0, 180] interval
			# if parameter=="gamma":
			# 	y = np.cos(self.values[parameter])
			# 	self.values[parameter] = np.arccos(y)
			# # azimuth is wrapped around [-90, 90] interval
			# elif parameter=="chi":
			# 	# y = np.sin(self.values[parameter])
			# 	# self.values[parameter] = np.arcsin(y)
			# 	y = np.where(self.values[parameter]<0, self.values[parameter]+360, self.values[parameter])
			# 	self.values[parameter] = y%180
			# else:
			for idn in range(len(self.nodes[parameter])-1, -1, -1):
				# check lower boundary condition
				vmin = self.limit_values[parameter].min[0]
				if self.limit_values[parameter].vmin_dim!=1:
					vmin = self.limit_values[parameter].min[idn]
				indx, indy = np.where(self.values[parameter][...,idn]<vmin)
				self.values[parameter][indx,indy,idn] = vmin

				# check upper boundary condition
				vmax = self.limit_values[parameter].max[0]
				if self.limit_values[parameter].vmax_dim!=1:
					vmax = self.limit_values[parameter].max[idn]
				if parameter=="temp" and self.decreasing_temperature and (idn+1)!=len(self.nodes[parameter]):
					vmax = self.values[parameter][...,idn+1]
					indx, indy = np.where(self.values[parameter][...,idn]>vmax)
					self.values[parameter][indx,indy,idn] = vmax[indx,indy]
				else:
					indx, indy = np.where(self.values[parameter][...,idn]>vmax)
					self.values[parameter][indx,indy,idn] = vmax

		for parameter in self.global_pars:
			if (parameter=="vmac"):
				# minimum check
				if self.global_pars[parameter]<self.limit_values[parameter][0]:
					self.global_pars[parameter] = np.array([self.limit_values[parameter][0]], dtype=np.float64)
				# maximum check
				if self.global_pars[parameter]>self.limit_values[parameter][1]:
					self.global_pars[parameter] = np.array([self.limit_values[parameter][1]], dtype=np.float64)
				# get back values into the atmosphere structure
				self.vmac = self.global_pars["vmac"]
			elif (parameter=="stray") or ("sl_" in parameter):
				if self.global_pars[parameter]<self.limit_values[parameter].min[0]:
					self.global_pars[parameter] = np.array([self.limit_values[parameter].min[0]], dtype=np.float64)
				# maximum check
				if self.global_pars[parameter]>self.limit_values[parameter].max[0]:
					self.global_pars[parameter] = np.array([self.limit_values[parameter].max[0]], dtype=np.float64)
				# get back values into the atmosphere structure
				if parameter=="stray":
					self.stray_light = self.global_pars["stray"]
			else:
				Npar = self.line_no[parameter].size
				if Npar > 0:
					for idl in range(Npar):
						# check lower boundary condition
						indx, indy = np.where(self.global_pars[parameter][...,idl]<self.limit_values[parameter][idl,0])
						self.global_pars[parameter][...,idl][indx,indy] = self.limit_values[parameter][idl,0]
						if self.sl_atmos is not None:
							self.sl_atmos.global_pars[parameter][...,idl][indx,indy] = self.limit_values[parameter][idl,0]

						# check upper boundary condition
						indx, indy = np.where(self.global_pars[parameter][...,idl]>self.limit_values[parameter][idl,1])
						self.global_pars[parameter][...,idl][indx,indy] = self.limit_values[parameter][idl,1]
						if self.sl_atmos is not None:
							self.sl_atmos.global_pars[parameter][...,idl][indx,indy] = self.limit_values[parameter][idl,1]

	def smooth_parameters(self, cycle, num=5, std=2.5):
		"""
		Run Gaussian kernel on 'num' x 'num' pixels with stadard deviation
		of 'std'. This is done after every inversion cycle, except on the last
		one. The inversion parameters are saved before smoothing is done.

		If the inversion patch is smaller than 'num' x 'num', we just apply
		median_filter() to each inversion parameter.
		"""
		if self.nx>=num and self.ny>=num:
			for parameter in self.nodes:
				for idn in range(len(self.nodes[parameter])):
					if parameter=="chi":
						aux = azismooth(self.values[parameter][...,idn]*180/np.pi, num)
						aux *= np.pi/180
					elif parameter=="gamma":
						aux = mygsmooth(self.values[parameter][...,idn]*180/np.pi, num, std)
						aux *= np.pi/180
					elif parameter=="of":
						aux = self.values[parameter][...,idn]
					else:
						aux = mygsmooth(self.values[parameter][...,idn], num, std)

					self.values[parameter][...,idn] = median_filter(aux, size=4)
		else:
			for parameter in self.nodes:
				for idn in range(len(self.nodes[parameter])):
					tmp = median_filter(self.values[parameter][...,idn], size=num)
					self.values[parameter][...,idn] = tmp

		for parameter in self.global_pars:
			if parameter=="loggf" and len(self.line_no["loggf"])!=0 and self.mode==3:
				Niter, Nloggf = self.loggf_history.shape
				weights = np.exp(np.arange(Niter)+1 - Niter)
				weights /= np.sum(weights)
				mean_values = np.average(self.loggf_history, axis=0, weights=weights)
				self.global_pars[parameter][0,0] = mean_values
				if self.sl_atmos is not None:
					self.sl_atmos.global_pars[parameter][0,0] = copy.deepcopy(mean_values)
		# 		size = self.line_no[parameter].size
		# 		nx, ny = 1, 1	
		# 		if self.mode==2:
		# 			nx = self.nx
		# 			ny = self.ny
		# 		# delta = 0.0413 --> 10% relative error in oscillator strength (f)
		# 		self.global_pars[parameter] += np.random.normal(loc=0, scale=0.0413, size=size*nx*ny).reshape(nx, ny, size)
			if parameter=="loggf" and self.mode==2:
				for idl in range(len(self.line_no[parameter])):
					aux = mygsmooth(self.global_pars[parameter][...,idl], num, std)
					self.global_pars[parameter][...,idl] = median_filter(aux, size=4)
					if self.sl_atmos is not None:
						self.sl_atmos.global_pars[parameter][...,idl] = copy.deepcopy(self.global_pars[parameter][...,idl])
			if parameter=="dlam" and len(self.line_no["dlam"])!=0 and self.mode==3:
				median = np.median(self.global_pars[parameter])
				Ndlam = len(self.line_no[parameter])
				self.global_pars[parameter][0,0] = median
				if self.sl_atmos is not None:
					self.sl_atmos.global_pars[parameter][0,0] = copy.deepcopy(median)

	def compute_errors(self, H, chi2, Ndof):
		"""
		Computing parameters error. Based on equations from Iniesta (2003) which
		are originally presented in Sanchez Almeida J. 1997, ApJ, 419.
		"""
		if self.mode==1 or self.mode==2:
			invH = np.linalg.pinv(H, rcond=1e-5, hermitian=True)
			diag = np.diagonal(invH, offset=0, axis1=2, axis2=3)

			chi2, _ = chi2.get_final_chi2()
			chi2 *= Ndof

			Nlocal = self.nx*self.ny*self.n_local_pars
			Nglobal = self.n_global_pars
			Npars = Nlocal + Nglobal

			parameter_error = diag * chi2 / Npars
			parameter_error = np.sqrt(parameter_error)

			self.local_pars_errors = parameter_error[:Nlocal].reshape(self.nx, self.ny, self.n_local_pars)
			self.global_pars_errors = parameter_error[Nlocal:]

		if self.mode==3:
			invH = sp.linalg.inv(H)
			diag = invH.diagonal(k=0)
			diag = np.array(diag)

			total_chi2 = chi2.total()
			total_chi2 *= Ndof

			Nlocal = self.nx*self.ny*self.n_local_pars
			Nglobal = self.n_global_pars
			Npars = Nlocal + Nglobal

			parameter_error = diag * total_chi2 / Npars
			parameter_error = np.sqrt(parameter_error)

			self.local_pars_errors = parameter_error[:Nlocal].reshape(self.nx, self.ny, self.n_local_pars)
			self.global_pars_errors = parameter_error[Nlocal:]

	def load_atomic_parameters(self, fpath):
		atoms = globin.atoms.AtomPars(fpath)

		for parameter in ["loggf", "dlam"]:
			if atoms.nl[parameter] is None:
				continue

			self.global_pars[parameter] = atoms.data[parameter]
			self.line_no[parameter] = atoms.data[f"{parameter}IDs"]
			self.parameter_scale[parameter] = np.ones((1, 1, self.global_pars[parameter].shape[-1]))
			if atoms.limit_values[parameter] is not None:
				self.limit_values[parameter] = atoms.limit_values[parameter]

			if self.sl_atmos is not None:
				self.sl_atmos.global_pars[parameter] = self.global_pars[parameter]
				self.sl_atmos.line_no[parameter] = self.line_no[parameter]
				self.sl_atmos.parameter_scale[parameter] = self.parameter_scale[parameter]
				self.sl_atmos.limit_values[parameter] = self.limit_values[parameter]

	def load_errors(self, local_pars=None, global_pars=None):
		local_errors = np.load(local_pars)

		nx, ny, n_local_pars = local_errors.shape

		if nx!=self.nx or ny!=self.ny:
			raise ValueError("Incompatible dimension of the atmosphere model and errors array.")

		if n_local_pars!=self.n_local_pars:
			raise ValueError("Incompatible number of parameters in the atmosphere and in the error arrays.")

		self.errors = {}
		start = 0
		end = 0
		for parameter in self.nodes:
			Nnodes = len(self.nodes[parameter])
			start = end
			end += Nnodes
			self.errors[parameter] = local_errors[..., start:end]

		if global_pars is None:
			return

		global_errors = np.load(global_pars)

		start = 0
		end = 0
		for parameter in self.global_pars:
			Npars = self.global_pars[parameter].size
			if Npars==0:
				continue

			start = end
			end += Npars
			self.errors[parameter] = global_errors[start:end]

	def load_errors_fits(self, fpath):
		hdu = fits.open(fpath)
		local_errors = hdu[0].data

		self.errors = {}
		end = 0
		for parameter in self.nodes:
			start = end
			end += len(self.nodes[parameter])
			self.errors[parameter] = local_errors[..., start:end]

		try:
			global_errors = hdu[1].data
		except:
			return

		end = 0
		for parameter in self.global_pars:
			npars = len(self.global_pars[parameter])
			if npars==0:
				continue

			start = end
			end += npars
			self.errors[parameter] = global_errors[start:end]

	def get_hsra_cont(self):
		"""
		Compute the HSRA spectrum for input wavelength grid.

		If we have to normalize spectra by HSRA continuum, allocate the value for the
		Icont and retrieve the HSRA spectrum that is normalized.

		Returnes:
		---------
		icont : float
			continuum intensity by which the spectrum should be normalized. If the 
			'self.norm' is False we return 1 and do not allocate self.icont.
		spec : globin.spec.Spectrum()
			HSRA full Stokes spectrum. If the 'self.norm' is True, the returned 
			spectrum is normalized to the local continuum point (@ first wavelength).
		"""
		hsra = Atmosphere(f"{globin.__path__}/data/hsrasp.dat", atm_type="spinor")
		hsra.mode = 0
		hsra.wavelength_air = self.wavelength_air
		hsra.wavelength_vacuum = air_to_vacuum(hsra.wavelength_air)
		hsra.cwd = self.cwd
		hsra.mu = self.mu
		# just to be sure that we are not normalizing...
		hsra.norm = False
		self.hsra_spec = hsra.compute_spectra()
		self.icont = self.hsra_spec.I[0,0,self.continuum_idl]
		if self.norm and self.norm_level=="hsra":
			self.hsra_spec.spec /= self.icont

			print("[Info] HSRA continuum level {:5.4e} @ {:8.4f}nm\n".format(self.icont, self.wavelength_air[self.continuum_idl]))

		if self.sl_atmos is not None:
			self.sl_atmos.icont = self.icont

		return self.icont, self.hsra_spec

	def get_tau(self, wlref):
		tau_wlref = pyrh.get_tau(self.cwd, self.mu, 0, self.data[0,0], np.array([wlref]))

		return tau_wlref

	def compute_spectra(self, synthesize=None, pool=None):
		"""
		Parameters:
		-----------
		synthesize : ndarray
		  flag for computing the pixels spectrum. If synthesize[idx,idy]==1 we compute spectrum,
		  otherwise, we ignore it (populate with nans/zeros).

		Return:
		-------
		spectra : globin.Spectrum() object
			structure containgin all the info regarding the spectrum.
		"""
		if synthesize is None:
			synthesize = np.ones((self.nx, self.ny))
		indx, indy = np.where(synthesize==1)
		args = zip(indx, indy)

		if pool is None:
			with mp.Pool(self.n_thread) as pool:
				spectra_list = pool.map(func=self._compute_spectra_sequential, iterable=args, chunksize=self.chunk_size)
		else:
			spectra_list = pool.map(self._compute_spectra_sequential, args)

		# print(spectra_list)

		spectra_list = np.array(spectra_list)[:,0,:4]
		natm, ns, nw = spectra_list.shape
		spectra_list = np.swapaxes(spectra_list, 1, 2)

		spectra = Spectrum(nx=self.nx, ny=self.ny, nw=nw)
		spectra.wavelength = self.wavelength_air
		spectra.spec[indx,indy] = spectra_list[:,:,:4]

		if self.get_atomic_rfs:
			self.atomic_rfs = spectra_list[:,:,4]

		if self.get_populations:
			pops = [item[1] for item in spectra_list]

			elements = [item.ID.decode("utf-8") for item in pops[0]]
			
			for ide, element in enumerate(elements):
				Nlevel = pops[0][ide].nlevel
				self.n_pops[element] = np.empty((self.nx*self.ny, Nlevel, self.nz))
				self.nstar_pops[element] = np.empty((self.nx*self.ny, Nlevel, self.nz))

			for ida in range(len(pops)):
				for ide in range(len(pops[ida])):
					self.n_pops[elements[ide]][ida] = pops[ida][ide].n
					self.nstar_pops[elements[ide]][ida] = pops[ida][ide].nstar

			for ide, element in enumerate(elements):
				Nlevel = pops[0][ide].nlevel
				self.n_pops[element] = self.n_pops[element].reshape(self.nx, self.ny, Nlevel, self.nz)
				self.nstar_pops[element] = self.nstar_pops[element].reshape(self.nx, self.ny, Nlevel, self.nz)

		# 	_spectra = [item[0] for item in spectra_list]
		# 	spectra_list = _spectra

		# print(self.atomic_rfs)

		return spectra

	def _compute_spectra_sequential(self, args):
		idx, idy = args

		got_populations = False

		start = time.time()

		try:
			mu = self.mu[idx,idy]
		except:
			mu = self.mu

		fudge_lam = None
		fudge_value = None
		if self.fudge_lam is not None:
			fudge_lam = self.fudge_lam
			fudge_value = self.fudge[idx,idy]

		if (self.line_no["loggf"].size>0) or (self.line_no["dlam"].size>0):
			if self.mode==2:
				_idx, _idy = idx, idy
			elif self.mode==3:
				_idx, _idy = 0, 0

			pyrh_output = pyrh.compute1d(cwd=self.cwd, 
										 mu=mu, 
										 atm_scale=self.scale_id, 
										 atmosphere=self.data[idx,idy], 
										 wave=self.wavelength_vacuum,
								  		 loggf_ids=self.line_no["loggf"], 
										 loggf_values=self.global_pars["loggf"][_idx, _idy],
								  		 lam_ids=self.line_no["dlam"], 
										 lam_values=self.global_pars["dlam"][_idx, _idy]/1e4,
								  		 fudge_wave=fudge_lam, 
										 fudge_value=fudge_value, 
										 atomic_number=self.atomic_number,
										 atomic_abundance=self.atomic_abundance,
										 get_atomic_rfs=self.get_atomic_rfs, 
										 get_populations=self.get_populations)
			# if self.get_atomic_rfs:
			# 	sI, sQ, sU, sV, rh_wave_vac, rf = pyrh_output
			# else:
			# 	sI, sQ, sU, sV, rh_wave_vac = pyrh_output
			# 	rf = None
		else:
			pyrh_output = pyrh.compute1d(cwd=self.cwd, 
										 mu=mu, 
										 atm_scale=self.scale_id, 
										 atmosphere=self.data[idx,idy],
										 wave=self.wavelength_vacuum,
								  		 fudge_wave=fudge_lam, 
										 fudge_value=fudge_value, 
										 atomic_number=self.atomic_number,
										 atomic_abundance=self.atomic_abundance,
										 get_atomic_rfs=False, 
										 get_populations=self.get_populations)
			# if len(pyrh_output)==5:
			# 	sI, sQ, sU, sV, rh_wave_vac = pyrh_output
			# if len(pyrh_output)==6:
			# 	sI, sQ, sU, sV, rh_wave_vac, populations = pyrh_output

			# rf = None

		sI, sQ, sU, sV, rh_wave_vac = pyrh_output[:5]

		tck = splrep(rh_wave_vac, sI, k=3)
		sI = splev(self.wavelength_vacuum, tck, der=0)
		tck = splrep(rh_wave_vac, sQ, k=3)
		sQ = splev(self.wavelength_vacuum, tck, der=0)
		tck = splrep(rh_wave_vac, sU, k=3)
		sU = splev(self.wavelength_vacuum, tck, der=0)
		tck = splrep(rh_wave_vac, sV, k=3)
		sV = splev(self.wavelength_vacuum, tck, der=0)

		stokes = np.vstack((sI, sQ, sU, sV))

		output = (stokes,)

		if self.get_atomic_rfs:
			rf = pyrh_output[6]

			for idp in range(rf.shape[1]):
				tck = splrep(rh_wave_vac, rf[:,idp], k=3)
				rf[:,idp] = splev(self.wavelength_vacuum, tck, der=0)

			output += rf

			# return np.vstack((np.vstack((sI, sQ, sU, sV)), rf.T))

		if self.get_populations:
			populations = pyrh_output[-1]
			output += populations
			# return np.vstack((sI, sQ, sU, sV)), populations

		return output
		#return np.vstack((sI, sQ, sU, sV))

	def compute_rfs(self, rf_noise_scale, weights=1, synthesize=None, rf_type="node", mean=False, old_rf=None, old_pars=None, pool=None):
		"""
		Compute response functions for atmospheric parameters at given nodes and
		specified global parameters (atomic line, vmac, stray light).

		Parameters:
		-----------
		rf_noise_scale : int or ndarray
			noise estimate in observed spectrum

		skip : tuple list
			list of tuples (idx, idy) for which to skip spectrum synthesis

		old_rf : ndarray
		"""
		if synthesize is None:
			synthesize = np.ones((self.nx, self.ny))

		self.build_from_nodes(synthesize, pool=pool)
		if self.hydrostatic:
			self.makeHSE(synthesize, pool=pool)
		spec = self.compute_spectra(synthesize, pool=pool)
		if self.sl_atmos is not None:
			flag = synthesize
			if self.stray_mode==3:
				flag = None
			sl_spec = self.sl_atmos.compute_spectra(flag, pool=pool)

		Nw = len(self.wavelength_air)

		if self.mode==1:
			Npar = self.n_local_pars
		elif (self.mode==2) or (self.mode==3):
			Npar = self.n_local_pars + self.n_global_pars

		rf = np.zeros((self.nx, self.ny, Npar, Nw, 4))

		active_indx, active_indy = np.where(synthesize==1)

		node_RF = np.zeros((self.nx, self.ny, Nw, 4))
		if self.spatial_regularization:
			self.scale_LT = np.zeros((self.nx*self.ny*self.n_local_pars + self.n_global_pars))
			addition = np.arange(0, self.nx*self.ny*self.n_local_pars, self.n_local_pars)
			shift = 0

		#--- loop through local (atmospheric) parameters and calculate RFs
		free_par_ID = 0
		free_par_ID_stray = -1
		free_par_ID_sl_pars = []
		for parameter in self.nodes:
			nodes = self.nodes[parameter]
			values = self.values[parameter]
			perturbation = self.delta[parameter]

			for nodeID in range(len(nodes)):
				if self.mask[parameter][nodeID]==0:
					continue

				if "sl_" in parameter:
					free_par_ID_sl_pars.append(free_par_ID)
				
				#--- positive perturbation
				self.values[parameter][:,:,nodeID] += perturbation
				if parameter=="of":
					self.make_OF_table()
				else:
					self.build_from_nodes(synthesize, params=parameter, pool=pool)

				if parameter in ["sl_temp", "sl_vz", "sl_vmic"]:
					spectra_plus = self.sl_atmos.compute_spectra(synthesize, pool=pool)
				elif parameter!="stray":
					spectra_plus = self.compute_spectra(synthesize, pool=pool)

				#--- negative perturbation (except for inclination and azimuth)
				if parameter=="gamma" or parameter=="chi" or self.rf_der_type=="forward":
					node_RF = (spectra_plus.spec - spec.spec ) / perturbation
				elif parameter=="stray":
					free_par_ID_stray = free_par_ID
					if self.stray_type=="hsra":
						node_RF = self.hsra_spec.spec - spec.spec
					if self.stray_type=="2nd_component":
						# node_RF = sl_spec.spec * brightness_fraction2 - spec.spec * brightness_fraction2
						node_RF = sl_spec.spec - spec.spec
					if self.stray_type in ["atmos", "spec"]:
						node_RF = self.stray_light_spectrum.spec - spec.spec
					if self.stray_type=="gray":
						node_RF = -spec.spec1				
				else:
					self.values[parameter][:,:,nodeID] -= 2*perturbation
					if parameter=="of":	
						self.make_OF_table()
					else:
						self.build_from_nodes(synthesize, params=parameter, pool=pool)

					if parameter in ["sl_temp", "sl_vz", "sl_vmic"]:
						spectra_minus = self.sl_atmos.compute_spectra(synthesize, pool=pool)
					else:
						spectra_minus = self.compute_spectra(synthesize, pool=pool)

					node_RF = (spectra_plus.spec - spectra_minus.spec ) / 2 / perturbation

				#--- compute parameter scale				
				node_RF *= weights
				node_RF /= rf_noise_scale
				node_RF *= np.sqrt(2)

				#--- set RFs value
				rf[active_indx,active_indy,free_par_ID] = node_RF[active_indx, active_indy] / self.parameter_norm[parameter]
				free_par_ID += 1

				#--- return back perturbations (node way)
				if parameter=="of":	
					self.values[parameter][:,:,nodeID] += perturbation
					self.make_OF_table()
				elif parameter=="stray":
					pass
				else:
					if parameter=="gamma" or parameter=="chi" or self.rf_der_type=="forward":
						self.values[parameter][:,:,nodeID] -= perturbation
					else:
						self.values[parameter][:,:,nodeID] += perturbation
					self.build_from_nodes(synthesize, params=parameter, pool=pool)

		#--- loop through global parameters and calculate RFs
		skip_par = -1
		free_par_ID_atomic = []
		if self.n_global_pars>0:
			#--- loop through global parameters and calculate RFs
			for parameter in self.global_pars:
				if parameter=="vmac":
					kernel_sigma = spec.get_kernel_sigma(self.vmac)
					kernel = spec.get_kernel(self.vmac, order=1)

					args = zip(spec.spec.reshape(self.nx*self.ny, Nw, 4), [kernel]*self.nx*self.ny)

					with mp.Pool(self.n_thread) as pool:
						results = pool.map(func=_compute_vmac_RF, iterable=args)

					dlam = spec.wavelength[1] - spec.wavelength[0]

					results = np.array(results)
					rf[:,:,free_par_ID] = results.reshape(self.nx, self.ny, Nw, 4)
					rf[:,:,free_par_ID] *= kernel_sigma * dlam / self.global_pars["vmac"]
					rf[:,:,free_par_ID] *= weights
					rf[:,:,free_par_ID] /= rf_noise_scale
					rf[:,:,free_par_ID] *= np.sqrt(2)

					rf[:,:,free_par_ID,:,:] /= self.parameter_norm[parameter]

					skip_par = free_par_ID
					free_par_ID += 1

				elif parameter=="stray":
					free_par_ID_stray = free_par_ID

					if self.stray_type=="2nd_component":
						diff = sl_spec.spec - spec.spec
					if self.stray_type=="hsra":
						diff = self.hsra_spec.spec - spec.spec
					if self.stray_type in ["atmos", "spec"]:
						diff = self.stray_light_spectrum.spec - spec.spec
					if self.stray_type=="gray":
						diff = -spec.spec

					diff *= weights
					diff /= rf_noise_scale
					diff *= np.sqrt(2)

					rf[:,:,free_par_ID,:,:] = diff / self.parameter_norm[parameter]
					free_par_ID += 1

				elif parameter in ["sl_temp", "sl_vz", "sl_vmic"]:
					free_par_ID_sl_pars.append(free_par_ID)

					perturbation = self.delta[parameter]
					
					self.global_pars[parameter] += perturbation
					self.build_from_nodes(params=parameter, pool=pool)
					spec_plus = self.sl_atmos.compute_spectra(synthesize=None, pool=pool)

					if self.rf_der_type=="central":
						self.global_pars[parameter] -= 2*perturbation
						self.build_from_nodes(params=parameter, pool=pool)
						spec_minus = self.sl_atmos.compute_spectra(synthesize=None, pool=pool)
						diff = (spec_plus.spec - spec_minus.spec) / 2 / perturbation
					if self.rf_der_type=="forward":
						diff = (spec_plus.spec - sl_spec.spec) / perturbation

					diff = np.repeat(diff, self.nx, axis=0)
					diff = np.repeat(diff, self.ny, axis=1)
					diff *= weights
					diff /= rf_noise_scale
					diff *= np.sqrt(2)
					rf[:,:,free_par_ID,:,:] = diff / self.parameter_norm[parameter]
					free_par_ID += 1

					if self.rf_der_type=="central":
						self.global_pars[parameter] += perturbation
					if self.rf_der_type=="forward":
						self.global_pars[parameter] -= perturbation
					self.build_from_nodes(flag=synthesize, params=parameter, pool=pool)

				elif parameter=="loggf" or parameter=="dlam":
					if self.line_no[parameter].size > 0:
						perturbation = self.delta[parameter]

						# get the stray light factor
						if self.sl_atmos is not None:
							if "stray" in self.global_pars:
								stray_light = self.global_pars["stray"]
								stray_light = np.ones((self.nx, self.ny, 1)) * stray_light
							else:
								stray_light = self.stray_light

						for idp in range(self.line_no[parameter].size):
							self.global_pars[parameter][...,idp] += perturbation
							spec_plus = self.compute_spectra(synthesize, pool=pool)
							if self.sl_atmos is not None:
								self.sl_atmos.global_pars[parameter][...,idp] += perturbation
								flag = synthesize
								if self.stray_mode==3:
									flag = None
								sl_plus = self.sl_atmos.compute_spectra(flag, pool=pool)

							if self.rf_der_type=="central":
								self.global_pars[parameter][...,idp] -= 2*perturbation
								spec_minus = self.compute_spectra(synthesize, pool=pool)
								diff = (spec_plus.spec - spec_minus.spec) / 2 / perturbation
								if self.sl_atmos is not None:
									self.sl_atmos.global_pars[parameter][...,idp] -= 2*perturbation
									flag = synthesize
									if self.stray_mode==3:
										flag = None
									sl_minus = self.sl_atmos.compute_spectra(flag, pool=pool)
									sl_diff = (sl_plus.spec - sl_minus.spec) / 2 / perturbation
							if self.rf_der_type=="forward":
								diff = (spec_plus.spec - spec.spec) / perturbation
								if self.sl_atmos is not None:
									sl_diff = (sl_plus.spec - sl_spec.spec) / perturbation

							diff *= weights
							diff /= rf_noise_scale
							diff *= np.sqrt(2)
							rf[:,:,free_par_ID,:,:] = diff / self.parameter_norm[parameter]

							if self.sl_atmos is not None:
								if self.stray_mode==3:
									sl_diff = np.repeat(sl_diff, self.nx, axis=0)
									sl_diff = np.repeat(sl_diff, self.ny, axis=1)
								sl_diff *= weights
								sl_diff /= rf_noise_scale
								sl_diff *= np.sqrt(2)
								rf_sl_atom = sl_diff / self.parameter_norm[parameter]
								rf[:,:,free_par_ID] = np.einsum("ij,ij...->ij...", 1 - stray_light[...,0], rf[:,:,free_par_ID])
								aux = np.einsum("ij...,ij->ij...", rf_sl_atom, stray_light[...,0])
								rf[:,:,free_par_ID] += aux

							free_par_ID_atomic.append(free_par_ID)
							free_par_ID += 1
							
							self.global_pars[parameter][...,idp] += perturbation
							if self.sl_atmos is not None:
								self.sl_atmos.global_pars[parameter][...,idp] += perturbation

				else:
					raise ValueError(f"Unsupported global parameter {parameter}")

		#--- broaden the spectra
		if not mean:
			spec.broaden_spectra(self.vmac, synthesize, self.n_thread, pool=pool)
			if self.sl_atmos is not None:
				flag = synthesize
				if self.stray_mode==3:
					flag = None
				sl_spec.broaden_spectra(self.vmac, flag, self.n_thread, pool=pool)
			if self.vmac!=0:
				kernel = spec.get_kernel(self.vmac, order=0)
				rf = broaden_rfs(rf, kernel, synthesize, skip_par, self.n_thread, pool=pool)
		
		#--- add instrumental broadening
		if self.instrumental_profile is not None:
			spec.instrumental_broadening(kernel=self.instrumental_profile, flag=synthesize, n_thread=self.n_thread, pool=pool)
			if self.sl_atmos is not None:
				flag = synthesize
				if self.stray_mode==3:
					flag = None
				sl_spec.instrumental_broadening(kernel=self.instrumental_profile, flag=flag, n_thread=self.n_thread, pool=pool)
			rf = broaden_rfs(rf, self.instrumental_profile, synthesize, -1, self.n_thread, pool=pool)

		#--- add the stray light component:
		if self.add_stray_light:
			# get the stray light factor(s)
			if "stray" in self.global_pars:
				stray_light = self.global_pars["stray"]
				stray_light = np.ones((self.nx, self.ny, 1)) * stray_light
			elif "stray" in self.values:
					stray_light = self.values["stray"]
			else:
				stray_light = self.stray_light

			# check for HSRA spectrum if we are using the 'hsra' stray light contamination
			sl_spectrum = None
			if self.stray_type=="hsra":
				sl_spectrum = self.hsra_spec.spec
			if self.stray_type=="2nd_component":
				sl_spectrum = sl_spec.spec
				if self.stray_mode==3:
					sl_spectrum = np.repeat(sl_spectrum, self.nx, axis=0)
					sl_spectrum = np.repeat(sl_spectrum, self.ny, axis=1)
			if self.stray_type in ["atmos", "spec"]:
				sl_spectrum = self.stray_light_spectrum.spec

			spec.add_stray_light(self.stray_mode, self.stray_type, stray_light, sl_spectrum=sl_spectrum, flag=synthesize)

			# skip adding the stray light to the RF for stray light
			for idp in range(Npar):
				# skip the RF to stray light factor
				if idp==free_par_ID_stray:
					continue
				# skip the RF to atomic parameters (log(gf) and dlam)
				if idp in free_par_ID_atomic:
					continue

				if idp in free_par_ID_sl_pars:
					rf[:,:,idp] = np.einsum("ij...,ij->ij...", rf[:,:,idp], stray_light[...,0])
				else:
					rf[:,:,idp] = np.einsum("ij...,ij->ij...", rf[:,:,idp], 1 - stray_light[...,0])

		#--- downsample the synthetic spectrum to observed wavelength grid
		if not np.array_equal(self.wavelength_obs, self.wavelength_air):
			spec.interpolate(self.wavelength_obs, self.n_thread, pool=pool)
			rf = interpolate_rf(rf, self.wavelength_air, self.wavelength_obs, self.n_thread, pool=pool)

		if self.norm:
			if self.norm_level==1:
				Ic = spec.I[...,self.continuum_idl]
				spec.spec = np.einsum("ij...,ij->ij...", spec.spec, 1/Ic)
				rf = np.einsum("ij...,ij->ij...", rf, 1/Ic)
			elif self.norm_level=="hsra":
				spec.spec /= self.icont
				rf /= self.icont
			else:
				spec.spec /= self.norm_level
				rf /= self.norm_level

		# update the RFs for those pixels that have updated parameters
		self.rf[active_indx, active_indy] = rf[active_indx, active_indy]

		return spec

	def get_regularization_gamma(self):
		"""
		Gamma matrix containing the regularization function values for each pixel.
		We already summed the contributions from each atmospheric parameter.
		"""
		npar = self.n_local_pars
		
		# number of regularization functions (per pixel)
		# we regularize spatialy in x- and y-axis
		nreg = 2

		#--- get regularization function values
		gamma = np.zeros((self.nx, self.ny, nreg*npar))
		
		idp = 0
		for parameter in self.nodes:
			for idn in range(len(self.nodes[parameter])):
				# p@x - p@x-1 (difference to the upper pixel)
				gamma[1:,:, 2*idp] = self.values[parameter][1:,:,idn] - self.values[parameter][:-1,:,idn]
				gamma[1:,:, 2*idp] /= self.parameter_norm[parameter]
				gamma[1:,:, 2*idp] *= np.sqrt(self.regularization_weight[parameter])
				# p@y - p@y-1 (difference to the left pixel)
				gamma[:,1:, 2*idp+1] = self.values[parameter][:,1:,idn] - self.values[parameter][:,:-1,idn]
				gamma[:,1:, 2*idp+1] /= self.parameter_norm[parameter]
				gamma[:,1:, 2*idp+1] *= np.sqrt(self.regularization_weight[parameter])
				idp += 1

		return gamma

	def get_regularization_der(self):
		"""
		Get the Jacobian matrix (transposed by default) for the spatial
		regularization functions. We assume that the spatial regularizations
		are linear function over atmospheric parameters and that we impose
		regularization to the upper and left pixel in respect to the target
		pixel.

		This has to be done only once, before iterative procedure since the
		matrix is filled with 1's and -1's and it does not depend on 
		the atmospheric parameters.

		[Dusan] it needs to be optimized; could be slow for big FOVs. 
		"""
		npar = self.n_local_pars
		
		# number of regularization functions (per pixel)
		# we regularize spatialy in x- and y-axis
		nreg = 2

		#--- compute derivative of regularization functions
		rows, cols = np.array([], dtype=np.int32), np.array([], dtype=np.int32)
		values = np.array([], dtype=np.float64)

		norm = np.ones(npar)
		weight = np.ones(npar)
		low, up = 0, 0
		for i_, parameter in enumerate(self.nodes):
			low = up
			up = low + len(self.nodes[parameter])
			norm[low:up] = self.parameter_norm[parameter]
			weight[low:up] = np.sqrt(self.regularization_weight[parameter])

		# norm /= weight

		ida = 0
		idp = np.arange(npar)
		for idx in range(self.nx):
			for idy in range(self.ny):
				ida = idx*self.ny + idy
				
				# Gamma_0 contribution
				if idx>0:
					ind = ida*npar + idp
					rows = np.append(rows, ind)

					ind = ida*npar*nreg + idp*2
					cols = np.append(cols, ind)
					
					values = np.append(values, np.ones(npar)/norm * weight)

					# derivative in respect to neighbouring pixel (up)
					tmp = (idx-1)*self.ny + idy
					ind = tmp*npar + idp
					rows = np.append(rows, ind)

					ind = ida*npar*nreg + idp*2
					cols = np.append(cols, ind)

					values = np.append(values, -1*np.ones(npar)/norm * weight)
				
				# Gamma_1 contribution
				if idy>0:
					ind = ida*npar + idp
					rows = np.append(rows, ind)
					
					ind = ida*npar*nreg + idp*2 + 1
					cols = np.append(cols, ind)
					
					values = np.append(values, np.ones(npar)/norm * weight)

					# derivative in respect to neighbouring pixel (left)
					tmp = idx*self.ny + idy-1
					ind = tmp*npar + idp
					rows = np.append(rows, ind)

					ind = ida*npar*nreg + idp*2 + 1
					cols = np.append(cols, ind)

					values = np.append(values, -1*np.ones(npar)/norm * weight)

		shape = (self.nx*self.ny*npar + self.n_global_pars, 
						 self.nx*self.ny*npar*nreg)

		LT = sp.csr_matrix((values, (rows, cols)), shape=shape, dtype=np.float64)
		
		# LTL = LT.dot(LT.transpose())
		# plt.imshow(LTL.toarray().T, origin="upper", cmap="bwr")
		# plt.colorbar()
		# plt.show()
		# sys.exit()

		return LT

	def get_dd_regularization(self):
		Nlocalpar = self.n_local_pars

		L = np.zeros((self.nx, self.ny, Nlocalpar, Nlocalpar))

		for parameter in self.dd_regularization_function:
			function = self.dd_regularization_function[parameter]
			if function==0:
				continue

			weight = self.dd_regularization_weight[parameter]
			if weight==0:
				continue

			nnodes = len(self.nodes[parameter])

			if function==1:
				_L = np.diag(np.ones(nnodes))
				tmp = np.diag(-1*np.ones(nnodes-1), k=-1)
				_L += tmp
				_L /= self.parameter_norm[parameter]
				_L *= np.sqrt(self.dd_regularization_weight[parameter])
				print(_L)

	def make_OF_table(self):
		if self.of_mode==0:
			nodes = self.of_wave
			values = self.of_values
		elif self.of_mode==1:
			nodes = self.nodes["of"]
			values = self.values["of"]
		else:
			raise ValueError("Unsupported of_mode value. Choose eighter 0 or 1 to apply the fudging.")

		if self.wavelength_vacuum[0]<=210:
			print("  Sorry, but OF is not properly implemented for wavelengths")
			print("    belowe 210 nm. You have to wait for newer version of globin.")
			sys.exit()

		of_num = len(nodes)
		
		if of_num==1:
			print("This is not checked yet... Do not trust results...")
			self.fudge_lam = np.zeros(4, dtype=np.float64)
			self.fudge = np.ones((self.nx, self.ny, 3,4), dtype=np.float64)

			# first point outside of interval (must be =1)
			self.fudge_lam[0] = self.wavelength_vacuum[0] - 0.0002
			self.fudge[...,0] = 1

			# left edge of wavelength interval
			self.fudge_lam[1] = self.wavelength_vacuum[0] - 0.0001
			self.fudge[...,0,1] = self.values["of"][...,0]
			if self.of_scatter:	
				self.fudge[...,1,1] = self.values["of"][...,0]
			self.fudge[...,2,1] = 1

			# right edge of wavelength interval
			self.fudge_lam[2] = self.wavelength_vacuum[-1] + 0.0001
			self.fudge[...,0,2] = self.values["of"][...,0]
			if self.of_scatter:
				self.fudge[...,1,2] = self.values["of"][...,0]
			self.fudge[...,2,1] = 1		

			# last point outside of interval (must be =1)
			self.fudge_lam[3] = self.wavelength_vacuum[-1] + 0.0002
			self.fudge[...,3] = 1

		#--- for multi-wavelength OF correction
		else:
			Nof = of_num + 2

			self.fudge_lam = np.zeros(Nof, dtype=np.float64)
			self.fudge = np.ones((self.nx, self.ny, 3, Nof), dtype=np.float64)
			
			# first point outside of interval (must be =1)
			self.fudge_lam[0] = nodes[0] - 0.0002
			self.fudge[...,0] = 1

			for idf in range(of_num):
				# mid points
				shift = 0
				if idf==0:
					shift = -0.0001
				if idf==of_num-1:
					shift = 0.0001
				self.fudge_lam[idf+1] = nodes[idf] + shift
				self.fudge[...,0,idf+1] = values[...,idf]
				if self.of_scatter:
					self.fudge[...,1,idf+1] = values[...,idf]
				self.fudge[...,2,idf+1] = 1	

			# last point outside of interval (must be =1)
			self.fudge_lam[-1] = nodes[-1] + 0.0002
			self.fudge[...,-1] = 1

	def compare(self, atmos, idx=0, idy=0):
		print("--------------------------------------")
		print("Atmosphere compare for [{:d},{:d}]:".format(idx,idy))
		for idp in range(1,self.npar):
			diff = np.abs(self.data[idx,idy,idp] - atmos[idp])
			delta = diff / np.abs(self.data[idx,idy,idp])
			delta = np.sum(delta) / self.nz
			rmsd = np.sqrt(np.sum(diff**2) / self.nz)
			print("{:2d}  {:4.3e}  {:4.3e}".format(idp, delta, rmsd))
		print("--------------------------------------")

	def read_spectral_lines(self, fpath):
		_, RLK_lines = globin.atoms.read_RLK_lines(fpath)

		nlines = len(RLK_lines)

		self.global_pars["loggf"] = np.zeros((self.nx, self.ny, nlines))
		self.global_pars["dlam"] = np.zeros((self.nx, self.ny, nlines))
		self.line_no["loggf"] = np.zeros(nlines, dtype=np.int32)
		self.line_no["dlam"] = np.zeros(nlines, dtype=np.int32)

		for idl in range(nlines):
			self.global_pars["loggf"][...,idl] = RLK_lines[idl].loggf
			self.line_no["loggf"][idl] = RLK_lines[idl].lineNo - 1 # we count from 0
			self.global_pars["dlam"][...,idl] = 0
			self.line_no["dlam"][idl] = RLK_lines[idl].lineNo - 1 # we count from 0

	def set_wavelength_grid(self, wavelength):
		self.wavelength_air = wavelength
		self.wavelength_obs = wavelength
		self.wavelength_vacuum = air_to_vacuum(wavelength)

		if self.sl_atmos is not None:
			self.sl_atmos.set_wavelength_grid(wavelength)

	def set_mu(self, mu):
		"""
		Assign the mu angle for which we want to synthesize spectra.
		"""
		if not isinstance(mu, list):
			mu = [mu]

		if len(mu)==1:
			self.mu = mu[0]
		else:
			mu = np.array(mu)
			self.mu = mu.reshape(self.nx, self.ny)

		if self.sl_atmos is not None:
			self.sl_atmos.mu = self.mu

	def set_vmac(self, vmac):
		self.vmac = vmac
		if self.sl_atmos is not None:
			self.sl_atmos.set_vmac(vmac)

	def set_n_thread(self, n_thread):
		"""
		Set the number of threads used for parallelized to call to methods (uses multiprocessing module)
		"""
		self.n_thread = n_thread
		if self.sl_atmos is not None:
			self.sl_atmos.n_thread = n_thread

	def set_chunk_size(self):
		"""
		Sets the amount of pixels sent to each thread for parallelised computation.
		"""
		self.chunk_size = (self.nx * self.ny) // self.n_thread + 1

	def set_mode(self, mode):
		if mode not in [0, 1, 2, 3]:
			raise ValueError(f"Cannot set the mode to {mode}.")
		
		self.mode = mode
		if self.sl_atmos is not None:
			self.sl_atmos.mode = mode

	def set_cwd(self, cwd):
		"""
		Working directory of the inversion run located in "./runs/run_name".

		Its passed to pyrh.compute1d() to update the path to *.input files
		necessary for the RH call.
		"""
		self.cwd = cwd
		if self.sl_atmos is not None:
			self.sl_atmos.cwd = cwd

	def set_spectrum_normalization(self, norm, norm_level):
		self.norm = norm
		self.norm_level = norm_level

	def set_opacity_fudge(self, mode, wavelength, value, scatter_flag):
		self.add_fudge = True
		self.of_mode = mode
		if self.of_mode==1:
			self.nodes["of"] = wavelength
			self.values["of"] = value
			self.parameter_scale["of"] = np.ones((self.nx, self.ny, len(wavelength)))
			self.mask["of"] = np.ones(len(wavelength))
		else:
			self.of_wave = wavelength
			self.of_values = value

		# create arrays to be passed to RH for synthesis
		self.of_scatter = scatter_flag
		self.make_OF_table()

	def set_instrumental_profile(self, wave, profile):
		dlam = self.wavelength_air[1] - self.wavelength_air[0]
		N = (wave.max() - wave.min()) / (dlam)
		M = np.ceil(N)//2
		M = int(M)
		xnew = np.linspace(0, (M-1)*dlam, num=M)
		aux = np.linspace(-(M-1)*dlam, -dlam, num=M-1)
		xnew = np.append(aux, xnew)
		
		aux = interp1d(wave, profile, kind=3, fill_value=0)(xnew)
		profile = aux/np.sum(aux)

		self.instrumental_profile = profile

	def set_stray_light_parameters(self, sl_data):
		if sl_data is None:
			return

		# we skip adding stray light parameter if its already present in the atmosphere;
		# this is the case when we initialize inversion from the previous run
		if "stray" in self.nodes:
			return

		self.add_stray_light = True

		stray_mode, stray_factor, stray_type, stray_min, stray_max = sl_data

		self.stray_mode = stray_mode
		self.stray_type = stray_type
		self.stray_light = np.ones((self.nx, self.ny, 1))
		if np.asarray(stray_factor).ndim==2:
			self.stray_light[...,0] = stray_factor
			return
		elif np.asarray(stray_factor).ndim==0:
			self.stray_light *= np.abs(stray_factor)

		# if the factor is inverted
		if stray_factor<0:
			self.invert_stray = True

			if stray_min is not None:
				self.limit_values["stray"].vmin = [stray_min]
				self.limit_values["stray"].vmin_dim = 1

			if stray_max is not None:
				self.limit_values["stray"].vmax = [stray_max]
				self.limit_values["stray"].vmax_dim = 1

			if self.stray_mode in [1,2]:
				self.n_local_pars += 1
				self.nodes["stray"] = np.array([0])
				self.values["stray"] = self.stray_light
				self.parameter_scale["stray"] = np.ones((self.nx, self.ny, 1))
				self.mask["stray"] = np.ones(1)
			elif self.stray_mode==3:
				self.n_global_pars += 1
				self.global_pars["stray"] = np.array([np.abs(stray_factor)], dtype=np.float64)
				self.parameter_scale["stray"] = 1.0

	def set_2nd_component_parameter(self, parameter, values):
		if len(values)==0:
			return

		if parameter in self.nodes:
			return

		value, fit_it, vmin, vmax = values

		if value is None:
			return

		if fit_it:
			if self.stray_mode in [1,2]:
				self.nodes[parameter] = np.array([0])
				self.values[parameter] = np.ones((self.nx, self.ny, 1)) * value
				self.parameter_scale[parameter] = np.ones((self.nx, self.ny, 1))
				self.mask[parameter] = np.ones(1)
				self.n_local_pars += 1
				self.sl_atmos.nodes[parameter] = np.array([0])
			if self.stray_mode==3:
				self.global_pars[parameter] = np.array([value], dtype=np.float64)
				self.sl_atmos.global_pars[parameter] = np.array([value], dtype=np.float64)
				self.parameter_scale[parameter] = 1.0
				self.n_global_pars += 1

			if parameter=="sl_temp":
				self.hydrostatic = True
				self.sl_atmos.hydrostatic = True

			if vmin is not None:
				self.limit_values[parameter].vmin = [vmin]
				self.limit_values[parameter].vmin_dim = 1

			if vmax is not None:
				self.limit_values[parameter].vmax = [vmax]
				self.limit_values[parameter].vmax_dim = 1

		if parameter=="sl_temp":
			delta = value - globin.T0_HSRA
			HSRA_temp = interp1d(globin.hsra.logtau, globin.hsra.T[0,0])(self.sl_atmos.logtau)
			value = HSRA_temp + delta

		if np.asarray(value).ndim==0:
			self.sl_atmos.data[:,:,self.par_id[parameter.split("_")[1]]] = value
		else:
			value = np.repeat(value[:,:,np.newaxis], self.nz).reshape(self.nx, self.ny, self.nz)
			self.sl_atmos.data[:,:,self.par_id[parameter.split("_")[1]]] = value

	def init_2nd_component(self):
		if self.stray_mode in [1,2]:
			nx = self.nx
			ny = self.ny
		elif self.stray_mode==3:
			nx = 1
			ny = 1
		else:
			raise ValueError("Unknown 'strey_mode' value.")

		self.sl_atmos = globin.Atmosphere(nx=nx, 
										 ny=ny,
										 nz=self.nz,
										 logtau_bot=self.logtau_bot,
										 logtau_top=self.logtau_top,
										 logtau_step=self.logtau_step)
		self.sl_atmos.interpolate_atmosphere(self.logtau, globin.hsra.data)
		self.sl_atmos.scale_id = self.scale_id

		self.sl_atmos.shape = self.sl_atmos.data.shape
		
		self.sl_atmos.set_mu(self.mu)
		self.sl_atmos.set_n_thread(self.n_thread)
		self.sl_atmos.set_cwd(self.cwd)

		try:
			self.sl_atmos.set_mode(self.mode)
			
			self.sl_atmos.wavelength_air = self.wavelength_air
			self.sl_atmos.wavelength_obs = self.wavelength_obs
			self.sl_atmos.wavelength_vacuum = self.wavelength_vacuum
		except:
			pass

		self.sl_atmos.norm = self.norm
		self.sl_atmos.norm_level = self.norm_level

		self.sl_atmos.fudge_lam = self.fudge_lam
		self.sl_atmos.fudge = self.fudge

		self.sl_atmos.atomic_number = self.atomic_number
		self.sl_atmos.atomic_abundance = self.atomic_abundance

		self.sl_atmos.continuum_idl = self.continuum_idl

		# when we invert global parameters in one, we need them also in the second one to compute RFs
		self.sl_atmos.global_pars = copy.deepcopy(self.global_pars)
		self.sl_atmos.line_no = self.line_no

	def add_magnetic_vector(self, B, gamma, chi):
		"""
		Add by hand a magnetic field vector to the atmosphere.

		Parameters:
		-----------
		B : float or ndarray
			magnetic field strength in G.
		gamma : float or ndarray
			magnetic field inclination in degrees.
		chi : float or ndarray
			magnetic field azimuth in degrees.
		"""
		gamma = np.deg2rad(gamma)
		chi = np.deg2rad(chi)

		self.data[:,:,5] = B
		self.data[:,:,6] = gamma
		self.data[:,:,7] = chi

	def set_node_values(self, nodes):
		"""
		If the atmosphere does not have the nodes/values, we set them based on
		stratified values (self.data) at given nodes position.

		Parameters:
		-----------
		nodes : dict
			dictionary containing the node positions for each parameter that
			we want.
		"""
		for parameter in nodes:
			self.nodes[parameter] = nodes[parameter]
			nnodes = len(nodes[parameter])
			self.values[parameter] = np.zeros((self.nx, self.ny, nnodes))
			if parameter in ["sl_temp", "sl_vz", "sl_vmic"]:
				self.values[parameter] = self.sl_atmos.data[...,self.par_id[parameter.split("_")[1]],:]
			elif parameter=="stray":
				self.values[parameter] = self.stray_light
			else:
				for idn in range(nnodes):
					idz = np.argmin(np.abs(self.logtau - nodes[parameter][idn]))
					self.values[parameter][...,idn] = self.data[...,self.par_id[parameter],idz]

	def reshape(self, shape, indx, indy):
		"""
		Transform the atmosphere shape. Used to convert the atmosphere structure
		back to the original shape after performing the inversion of X% of best
		pixels.

		Parameters:
		-----------
		shape : tuple
			original shape of the atmosphere to which we are casting the atmosphere
		indx : ndarray
			x-axis indices of original atmosphere which were selected for inversion
		indt : ndarray
			y-axis indices of original atmosphere which were selected for inversion

		"""
		data = np.empty((shape[0], shape[1], self.npar, self.nz), dtype=np.float64)
		data.fill(np.nan)

		data[indx,indy] = self.data

		values = {}
		for parameter in self.nodes:
			nnodes = self.nodes[parameter].size
			values[parameter] = np.empty((shape[0], shape[1], nnodes), dtype=np.float64)
			values[parameter].fill(np.nan)
			values[parameter][indx,indy] = self.values[parameter]

		self.data = data
		self.values = values

	def create_input_files(self, keys={}):
		atmols.create_atoms_list(f"{self.cwd}/atoms.input")
		atmols.create_molecules_list(f"{self.cwd}/molecules.input")
		if self.line_list is not None:
			create_kurucz_input(self.line_list, f"{self.cwd}/kurucz.input")

		keywords = globin.rh.RHKeywords()
		keywords.set_keywords(keys)
		keywords.create_input_file(f"{self.cwd}/keyword.input")

def broaden_rfs(rf, kernel, flag, skip_par, n_thread, pool=None):
	try:
		full = False
		nx, ny, npar, nw, ns = rf.shape
	except:
		full = True
		nx, ny, npar, nz, nw, ns = rf.shape

	indp = np.arange(npar, dtype=np.int32)
	if skip_par!=-1:
		npar -= 1
		indp = np.delete(indp, skip_par)

	# this happens when vmac is the only parameter in inversion;
	# we do not need to broaden RF for vmac
	if len(indp)==0:
		return rf

	if not full:
		_rf = rf[:,:,indp,:,:].reshape(nx*ny, npar, nw, ns)
		_rf = _rf.reshape(nx*ny*npar, nw, ns)
	else:
		_rf = rf[:,:,indp,:,:].reshape(nx*ny, npar, nz, nw, ns)
		_rf = _rf.reshape(nx*ny*npar, nz, nw, ns)
		_rf = _rf.reshape(nx*ny*npar*nz, nw, ns)
	
	_flag = flag.reshape(nx*ny)
	if not full:
		_flag = np.repeat(_flag, npar)
		args = zip(_rf, [kernel]*nx*ny*npar, _flag)
	else:
		_flag = np.repeat(_flag, npar*nz)
		args = zip(_rf, [kernel]*nx*ny*npar*nz, _flag)

	if pool is None:
		with mp.Pool(n_thread) as pool:
			results = pool.map(func=_broaden_rfs, iterable=args)
	else:
		results = pool.map(_broaden_rfs, args)

	results = np.array(results)
	if not full:
		results = results.reshape(nx*ny, npar, nw, ns)
		rf[:,:,indp,:,:] = results.reshape(nx, ny, npar, nw, ns)
	else:
		results = results.reshape(nx*ny*npar, nz, nw, ns)
		results = results.reshape(nx*ny, npar, nz, nw, ns)
		rf[:,:,indp] = results.reshape(nx, ny, npar, nz, nw, ns)

	return rf

def _broaden_rfs(args):
	rf, kernel, flag = args

	if flag==0:
		return rf

	N = len(kernel)
	for ids in range(4):
		aux = extend(rf[:,ids], N)
		rf[:,ids] = np.convolve(aux, kernel, mode="same")[N:-N]

	return rf

def _compute_vmac_RF(args):
	spec, kernel = args

	N = len(kernel)
	for ids in range(4):
		aux = extend(spec[:,ids], N)
		spec[:,ids] = np.convolve(aux, kernel, mode="same")[N:-N]

	return spec

def interpolate_rf(rf_in, wave_in, wave_out, n_thread, pool=None):
	nx, ny, npar, nw, ns = rf_in.shape

	_rf = rf_in.reshape(nx*ny, npar, nw, ns)

	args = zip(_rf, [wave_in]*(nx*ny), [wave_out]*(nx*ny))

	if pool is None:
		with mp.Pool(n_thread) as pool:
			results = pool.map(func=_interpolate_rf, iterable=args)
	else:
		results = pool.map(_interpolate_rf, args)

	results = np.array(results)
	rf_out = results.reshape(nx, ny, npar, len(wave_out), ns)

	return rf_out

def _interpolate_rf(args):
	rf_in, wave_in, wave_out = args

	npar,_,_ = rf_in.shape

	rf_out = np.zeros((npar, len(wave_out), 4))

	for idp in range(npar):
		for ids in range(4):
			rf_out[idp,:,ids] = interp1d(wave_in, rf_in[idp,:,ids], kind=3)(wave_out)

	return rf_out

def distribute_hydrogen(temp, pg, pe, vtr=0):
	Ej = 13.59844

	ne = pe/10 / globin.K_BOLTZMAN/temp # [1/m3]
	CC = 2*np.pi*globin.ELECTRON_MASS*globin.K_BOLTZMAN
	C1 = ne/2 * (globin.PLANCK / (CC*temp)**(1/2))**3

	nH = (pg-pe)/10 / globin.K_BOLTZMAN / temp / np.sum(10**(globin.abundance-12)) / 1e6 # [1/cm3]

	pops = np.zeros((6, *temp.shape))

	suma = np.ones(temp.shape)
	fact = np.ones((6, *temp.shape))
	for lvl in range(1,6):
		e_lvl = Ej*(1-1/(lvl+1)**2)
		g = 2*(lvl+1)**2
		fact[lvl] = g/2 * np.exp(-e_lvl*1.60218e-19/globin.K_BOLTZMAN/temp)
		if lvl==5:
			e_lvl = Ej
			fact[lvl] = 1/2 * np.exp(-e_lvl*1.60218e-19/globin.K_BOLTZMAN/temp)
			fact[lvl] /= C1
		suma += fact[lvl]

	pops[0] = nH/suma

	for lvl in range(1,6):
		pops[lvl] = fact[lvl] * pops[0]
	
	return pops

def distribute_H(nHtot, temp, ne):
	Ej = 13.59844
	eV = globin.ELECTRON_CHARGE

	lth = globin.PLANCK / np.sqrt(2*np.pi*globin.ELECTRON_MASS*globin.K_BOLTZMAN*temp)
	Saha = 2/ne/1e6 * 1/lth**3
	Saha *= np.exp(-Ej*eV/globin.K_BOLTZMAN/temp)
	Saha *= 2/1

	nH0 = nHtot / (1 + Saha)

	pops = np.zeros((6, len(temp)))
	
	# proton number
	pops[-1] = nHtot / (1 + 1/Saha)

	for idl in range(5):
		E = Ej*(1-1/(idl+1)**2)
		g = 2*(idl+1)**2
		pops[idl] = nH0 * g/1 * np.exp(-E*globin.ELECTRON_CHARGE/globin.K_BOLTZMAN/temp)

	return pops

def write_multi_atmosphere(atm, fpath, atm_scale="tau"):
	# write atmosphere 'atm' of MULTI type
	# into separate file and store them at 'fpath'.

	# atmosphere name (extracted from full path given)
	fname = fpath.split("/")[-1]

	out = open(fpath,"w")
	npar, nz = atm.shape

	out.write("* Model file\n")
	out.write("*\n")
	out.write(f"  {fname}\n")
	if atm_scale=="tau":	
		out.write("  Tau scale\n")
	elif atm_scale=="cmass":
		out.write("  Mass scale\n")
	else:
		print(f"Error: Not supported {atm_scale} atmosphere scale.")
		sys.exit()
	out.write("*\n")
	out.write("* log(g) [cm s^-2]\n")
	out.write("  4.44\n")
	out.write("*\n")
	out.write("* Ndep\n")
	out.write(f"  {nz}\n")
	out.write("*\n")
	if atm_scale=="tau":
		out.write("* log tau    Temp[K]    n_e[cm-3]    v_z[km/s]   v_turb[km/s]\n")
		for i_ in range(nz):
			out.write("  {:+5.4f}    {:12.6f}   {:8.6e}   {:8.6e}   {:8.6e}\n".format(atm[0,i_], atm[1,i_], atm[2,i_], atm[3,i_], atm[4,i_]))
	elif atm_scale=="cmass":
		out.write("* log cmass      Temp[K]    n_e[cm-3]    v_z[km/s]   v_turb[km/s]\n")
		for i_ in range(nz):
			out.write("  {:+8.6f}    {:12.6f}   {:8.6e}   {:8.6e}   {:8.6e}\n".format(atm[0,i_], atm[1,i_], atm[2,i_], atm[3,i_], atm[4,i_]))

	out.write("*\n")
	out.write("* Hydrogen populations [cm-3]\n")
	out.write("*     nh(1)        nh(2)        nh(3)        nh(4)        nh(5)        nh(6)\n")

	for i_ in range(nz):
		out.write("   {:8.6e}   {:8.6e}   {:8.6e}   {:8.6e}   {:8.6e}   {:8.6e}\n".format(atm[8,i_], atm[9,i_], atm[10,i_], atm[11,i_], atm[12,i_], atm[13,i_]))

	out.close()

	# store magnetic field vector
	write_B(f"{fpath}.B", atm[5]/1e4, atm[6], atm[7])

	if np.isnan(np.sum(atm)):
		print(fpath)
		print("We have NaN in atomic structure!\n")
		sys.exit()

def write_B(outfile, B, gamma_B, chi_B):
    ''' Writes a RH magnetic field file. Input B arrays can be any rank, as
        they will be flattened before write. Bx, By, Bz units should be T.'''
    if (B.shape != gamma_B.shape) or (gamma_B.shape != chi_B.shape):
        raise TypeError('writeB: B arrays have different shapes!')
    n = np.prod(B.shape)
    
    import xdrlib

    # Pack as double
    p = xdrlib.Packer()
    p.pack_farray(n, B.ravel().astype('d'), p.pack_double)
    p.pack_farray(n, gamma_B.ravel().astype('d'), p.pack_double)
    p.pack_farray(n, chi_B.ravel().astype('d'), p.pack_double)
    # Write to file
    f = open(outfile, 'wb')
    f.write(p.get_buffer())
    f.close()
    return

def compute_full_rf(atmos, local_pars=None, global_pars=None, norm=False, fpath=None):
	if (local_pars is None) and (global_pars is None):
		raise ValueError("None of atmospheric or atomic parameters are given for RF computation.")

	dlogtau = atmos.logtau[1] - atmos.logtau[0]

	if norm:
		atmos.norm = True
		atmos.norm_level = 1

	# reference spectrum for forward derivative computation
	if atmos.rf_der_type=="forward":
		spec = atmos.compute_spectra()
	
	# compute the total number of free parameters (have to sum atomic for each line)
	n_local = 0
	if local_pars is not None:
		n_local = len(local_pars)
		rf_local = np.zeros((atmos.nx, atmos.ny, n_local, atmos.nz, len(atmos.wavelength_air), 4), dtype=np.float64)
	
	n_global = 0
	if global_pars is not None:
		for parameter in global_pars:
			n_global += atmos.line_no[parameter].size
		rf_global = np.zeros((atmos.nx, atmos.ny, n_global, len(atmos.wavelength_air), 4), dtype=np.float64)

	free_par_ID = 0
	if local_pars is not None:
		for idp, parameter in enumerate(local_pars):
			perturbation = atmos.delta[parameter]
			parID = atmos.par_id[parameter]

			for idz in tqdm(range(atmos.nz), desc=parameter):
				atmos.data[:,:,parID,idz] += perturbation
				spec_plus = atmos.compute_spectra()
				# if atmos.sl_atmos is not None:
				# 	spec_plus_sl = atmos.sl_atmos.compute_spectra()

				if atmos.rf_der_type=="central":
					atmos.data[:,:,parID,idz] -= 2*perturbation
					spec_minus = atmos.compute_spectra()
					# if atmos.sl_atmos is not None:
					# 	spec_minus_sl = atmos.sl_atmos.compute_spectra()

					diff = spec_plus.spec - spec_minus.spec
					rf_local[:,:,free_par_ID,idz] = diff / 2 / perturbation
				elif atmos.rf_der_type=="forward":
					diff = spec_plus.spec - spec.spec
					rf_local[:,:,free_par_ID,idz] = diff / perturbation
				else:
					raise ValueError("Unsupported type of the RF derivative.")

				# remove perturbation from data
				if atmos.rf_der_type=="central":
					atmos.data[:,:,parID,idz] += perturbation
				if atmos.rf_der_type=="forward":
					atmos.data[:,:,parID,idz] -= perturbation

			free_par_ID += 1

	#--- loop through global parameters and calculate RFs
	if global_pars is not None:
		free_par_ID = 0
		for parameter in global_pars:
			if parameter=="vmac":
				pass
				# radius = int(4*kernel_sigma + 0.5)
				# x = np.arange(-radius, radius+1)
				# phi = np.exp(-x**2/kernel_sigma**2)
				# # normalaizing the profile
				# phi *= 1/(np.sqrt(np.pi)*kernel_sigma)
				# kernel = phi*(2*x**2/kernel_sigma**2 - 1)
				# # since we are correlating, we need to reverse the order of data
				# kernel = kernel[::-1]

				# spec, _ = compute_spectra(atmos)
				# if not globin.mean:
				# 	spec.broaden_spectra(atmos.vmac)

				# for idx in range(atmos.nx):
				# 	for idy in range(atmos.ny):
				# 		for sID in range(1,5):
				# 			rf_global[idx,idy,free_par_ID,0,:,sID-1] = correlate1d(spec[idx,idy,:,sID], kernel)
				# 			rf_global[idx,idy,free_par_ID,0,:,sID-1] *= 1/atmos.vmac * atmos.parameter_scale["vmac"]
				# free_par_ID += 1
			elif parameter in ["loggf", "dlam"]:
				perturbation = atmos.delta[parameter]

				for idp in tqdm(range(atmos.line_no[parameter].size), desc=parameter):
					atmos.global_pars[parameter][...,idp] += perturbation
					spec_plus = atmos.compute_spectra()

					if atmos.rf_der_type=="central":
						atmos.global_pars[parameter][...,idp] -= 2*perturbation
						spec_minus = atmos.compute_spectra()
						diff = spec_plus.spec - spec_minus.spec
						rf_global[:,:,free_par_ID] = diff / 2 / perturbation
					elif atmos.rf_der_type=="forward":
						diff = spec_plus.spec - spec.spec
						rf_global[:,:,free_par_ID] = diff / 2 / perturbation
					else:
						raise ValueError("Unsupported type of the RF derivative.")

					# if atmos.mode==3:
					# 	rf[:,:,free_par_ID,:,:] = np.repeat(diff[:,:,np.newaxis,:,:], atmos.nz , axis=2)

					free_par_ID += 1
					
					if atmos.rf_der_type=="central":
						atmos.global_pars[parameter][...,idp] += perturbation
					if atmos.rf_der_type=="forward":
						atmos.global_pars[parameter][...,idp] -= perturbation

			else:
				print(f"Parameter {parameter} not yet Supported.\n")

	# macro-turbulent broadening
	if atmos.vmac!=0:
		kernel = globin.utils.get_kernel(atmos.vmac, atmos.wavelength_air, order=0)
		if n_local>0:
			rf_local = broaden_rfs(rf_local, kernel, np.ones((atmos.nx, atmos.ny)), -1, atmos.n_thread)
		if n_global>0:
			rf_global = broaden_rfs(rf_global, kernel, np.ones((atmos.nx, atmos.ny)), -1, atmos.n_thread)

	# stray light contamination
	if atmos.add_stray_light:
		# get the stray light factor(s)
		if "stray" in atmos.global_pars:
			stray_light = atmos.global_pars["stray"]
			stray_light = np.ones((atmos.nx, atmos.ny, 1)) * stray_light
		elif "stray" in atmos.values:
				stray_light = atmos.values["stray"]
		else:
			stray_light = atmos.stray_light

		for idp  in range(n_local):
			rf_local[:,:,idp] = np.einsum("ij...,ij->ij...", rf_local[:,:,idp], 1 - stray_light[...,0])
		for idp  in range(n_global):
			rf_global[:,:,idp] = np.einsum("ij...,ij->ij...", rf_global[:,:,idp], 1 - stray_light[...,0])

	# if we have provided the path, save the RFs
	if fpath is not None:
		hdulist = []#fits.HDUList([])
		# save atmospheric RFs
		if n_local>0:
			atmos_hdu = fits.PrimaryHDU(rf_local)
			atmos_hdu.name = "RF_LOCAL"

			atmos_hdu.header.comments["NAXIS1"] = "stokes components"
			atmos_hdu.header.comments["NAXIS2"] = "number of wavelengths"
			atmos_hdu.header.comments["NAXIS3"] = "depth points"
			atmos_hdu.header.comments["NAXIS4"] = "number of parameters"
			atmos_hdu.header.comments["NAXIS5"] = "y-axis atmospheres"
			atmos_hdu.header.comments["NAXIS6"] = "x-axis atmospheres"

			atmos_hdu.header["STOKES"] = ("IQUV", "the Stokes vector order")

			if norm:
				atmos_hdu.header["NORMED"] = ("TRUE", "flag for spectrum normalization")
			else:
				atmos_hdu.header["NORMED"] = ("FALSE", "flag for spectrum normalization")

			i_ = 1
			for par in local_pars:
				atmos_hdu.header[f"PAR{i_}"] = (par, "parameter")
				atmos_hdu.header[f"PARID{i_}"] = (i_, "parameter ID")
				i_ += 1

			hdulist.append(atmos_hdu)
		if n_global>0:
			if n_local>0:
				global_hdu = fits.ImageHDU(rf_global)
			else:
				global_hdu = fits.PrimaryHDU(rf_global)
			global_hdu.name = "RF_GLOBAL"

			global_hdu.header.comments["NAXIS1"] = "stokes components"
			global_hdu.header.comments["NAXIS2"] = "number of wavelengths"
			global_hdu.header.comments["NAXIS3"] = "depth points"
			global_hdu.header.comments["NAXIS4"] = "y-axis atmospheres"
			global_hdu.header.comments["NAXIS5"] = "x-axis atmospheres"

			global_hdu.header["STOKES"] = ("IQUV", "the Stokes vector order")

			# if norm:
			global_hdu.header["NORMED"] = (f"{str(norm).upper()}", "flag for spectrum normalization")
			# else:
			# 	global_hdu.header["NORMED"] = ("FALSE", "flag for spectrum normalization")

			i_ = 1
			start_index = 1
			for par in global_pars:
				global_hdu.header[f"PAR{i_}"] = (par, "parameter")
				global_hdu.header[f"PAR{i_}S"] = (start_index, "ID of parameter RF begining")
				global_hdu.header[f"PAR{i_}E"] = (start_index + atmos.line_no[par].size -1, "ID of parameter RF end")
				i_ += 1
				start_index += atmos.line_no[par].size

			hdulist.append(global_hdu)

		#--- wavelength list
		par_hdu = fits.ImageHDU(atmos.wavelength_air)
		par_hdu.name = "wavelength"
		par_hdu.header["UNIT"] = "Angstrom"
		par_hdu.header.comments["NAXIS1"] = "wavelengths"
		hdulist.append(par_hdu)

		#--- depth list
		par_hdu = fits.ImageHDU(atmos.logtau)
		par_hdu.name = "depths"
		par_hdu.header["UNIT"] = "optical depth"
		par_hdu.header.comments["NAXIS1"] = "depth stratification"
		hdulist.append(par_hdu)

		#--- save HUDs to fits file
		hdulist = fits.HDUList(hdulist)
		hdulist.writeto(fpath, overwrite=True)

	if n_local>0 and n_global>0:
		return rf_local, rf_global
	if n_local>0:
		return rf_local
	if n_global>0:
		return rf_global

def convert_atmosphere(logtau, atmos_data, atm_type):
	"""
	Routine for general conversion of the input 'atmos_data' from 'atm_type' 
	to MULTI type atmosphere. We return globin.atmos.Atmosphere() object.

	Parameters:
	-----------
	atmos_data : ndarray
		atmosphere data to be converted. Can have dimensions of (npar, nz) or
		(nx, ny, npar, nz).
	atm_type : string
		type of input atmosphere to be converted

	Return:
	-------
	atmos : globin.atmos.Atmosphere() object
	"""
	if atm_type=="spinor":
		atmos = spinor2multi(atmos_data)
		# if not np.array_equal(multi_atmos[0,0,0], self.logtau):
		# 	self.interpolate_atmosphere(x_new=self.logtau, ref_atm=multi_atmos)
		# else:
		# 	self.logtau = multi_atmos[0,0,0]
		# 	self.data = multi_atmos
		# DV: removed interpolation when we convert from SPINOR;
		#     we use optical depth scale from input atmosphere
		# self.logtau = multi_atmos[0,0,0]
		# self.data = multi_atmos
	elif atm_type=="sir":
		atmos = sir2multi(atmos_data)
	else:
		print("--> Error in atmos.convert_atmosphere()")
		print(f"    Currently not recognized atmosphere type: {atm_type}")
		print("    Recognized ones are: spinor, sir.")
		sys.exit()

	return atmos

def spinor2multi(atmos_data):
	"""
	Routine for converting SPINOR inverted atmosphere to MULTI type.
	WARNING: the order of parameters in inverted and input SPINOR type atmosphere
	are not the same! Indexing here is only valid for inverted atmosphere.

	Parameters:
	-----------
	atmos_data : ndarray
		atmosphere data to be converted. Can have dimensions of (npar, nz) or
		(nx, ny, npar, nz).

	Return:
	-------
	atmos : globin.atmos.Atmosphere()
	"""
	###
	### DEBUG THIS
	###
	npar, nx, ny, nz = atmos_data.shape

	atmos = Atmosphere(nx=nx, ny=ny, nz=nz)
	atmos.shape = atmos.data.shape

	# logtau
	atmos.data[:,:,0] = atmos_data[0]
	atmos.logtau = atmos_data[0,0,0]
	atmos.logtau_top = atmos.logtau[0]
	atmos.logtau_bot = atmos.logtau[-1]
	atmos.logtau_step = atmos.logtau[1] - atmos.logtau[0]
	# temperature [K]
	atmos.data[:,:,1] = atmos_data[2]
	# electron density [1/cm3]
	atmos.data[:,:,2] = atmos_data[4]/10/globin.K_BOLTZMAN/atmos_data[2] / 1e6
	# LOS velocity [km/s]
	atmos.data[:,:,3] = atmos_data[8]/1e5*(-1)
	# micro-turbulent velocity [km/s]
	atmos.data[:,:,4] = atmos_data[7]/1e5
	# magnetic field strength [G]
	atmos.data[:,:,5] = atmos_data[9]
	# magnetic field inclination [rad]
	atmos.data[:,:,6] = atmos_data[-2] * np.pi/180
	# magnetic field zazimuth [rad]
	atmos.data[:,:,7] = atmos_data[-1] * np.pi/180
	# hydrogen density [1/cm3]
	tmp = distribute_hydrogen(atmos_data[2], atmos_data[3], atmos_data[4])
	for idp in range(6):
		atmos.data[:,:,8+idp] = tmp[idp]

	return atmos

def sir2multi(atmos_data):
	pass

def multi2sir(atmos, fpath):
	nx, ny, npar, nz = atmos.shape
	new = np.zeros((nx, ny, 11, nz))
	
	for idx in range(atmos.nx):
		for idy in range(atmos.ny):
			new[idx,idy,0] = atmos.data[idx,idy,0] 			# log(tau) @ 500nm
			new[idx,idy,1] = atmos.data[idx,idy,1] 			# T [K]
			new[idx,idy,2] = atmos.data[idx,idy,2]*1e6 	# ne [1/m3]
			new[idx,idy,2] *= globin.K_BOLTZMAN * new[idx,idy,1] * 10	# ne [dyn/cm2]
			new[idx,idy,3] = atmos.data[idx,idy,4]*1e5	# vmic [cm/s]
			new[idx,idy,4] = atmos.data[idx,idy,5]			# B [G]
			new[idx,idy,5] = atmos.data[idx,idy,3]*1e5	# vz [cm/s]
			new[idx,idy,6] = atmos.data[idx,idy,6]			# gamma [rad]
			new[idx,idy,6] *= 180/np.pi									# gamma [deg]
			new[idx,idy,7] = atmos.data[idx,idy,7]			# gamma [rad]
			new[idx,idy,7] *= 180/np.pi									# gamma [deg]

			new[idx,idy,9] = atmos.pg[idx,idy]
			new[idx,idy,10] = atmos.rho[idx,idy]

			np.savetxt(f"{fpath}_x{idx+1}_y{idy+1}.mod", new[idx,idy,:,::-1].T, header=" 1.0  1.0  0.0", comments="", fmt="%5.4e")

			# Column 9: Geometrical scale (km)
			# Column 10: Gas presure (dyn/cm^2)
			# Column 11: Gas density (gr/cm^3)
			# new[idx,idy,8] = atmos.height[idx,idy]

def multi2spinor(multi_atmosphere, fname=None):
	from .makeHSE import Axmu

	nx, ny, _, nz = multi_atmosphere.shape

	height = np.zeros(nz)
	spinor_atmosphere = np.zeros((nx,ny,nz,12))

	ind0 = np.argmin(np.abs(multi_atmosphere[0,0,0]))

	for idx in range(nx):
		for idy in range(ny):
			# pg, pe, kappa, rho = globin.makeHSE(5000, multi_atmosphere[idx,idy,0], multi_atmosphere[idx,idy,1])

			nHtot = np.sum(multi_atmosphere[idx,idy,8:,:], axis=0)
			pe = multi_atmosphere[idx,idy,2] * globin.K_BOLTZMAN * multi_atmosphere[idx,idy,1]# * 10 # [CGS]
			pg = pe + nHtot * globin.K_BOLTZMAN * multi_atmosphere[idx,idy,1]
			rho = nHtot * np.mean(Axmu)

			spinor_atmosphere[idx,idy,:,3] = pg*10 # [CGS unit]
			spinor_atmosphere[idx,idy,:,4] = pe*10 # [CGS unit]
			spinor_atmosphere[idx,idy,:,6] = rho

			# nHtot = np.sum(multi_atmosphere[idx,idy,8:], axis=0)
			# avg_mass = np.mean(Axmu)
			# rho = nHtot * avg_mass

			# for idz in range(nz-2,-1,-1):
			# 	height[idz] = height[idz+1] + 2*(kappa[idz+1] - kappa[idz]) / (rho[idz+1] + rho[idz])

			# height -= height[ind0]
			# spinor_atmosphere[idx,idy,:,1] = height

	spinor_atmosphere[:,:,:,0] = multi_atmosphere[:,:,0,:]
	spinor_atmosphere[:,:,:,2] = multi_atmosphere[:,:,1,:]
	spinor_atmosphere[:,:,:,7] = multi_atmosphere[:,:,4,:]*1e5
	spinor_atmosphere[:,:,:,8] = multi_atmosphere[:,:,3,:]*1e5
	# No magnetic field
	spinor_atmosphere[:,:,:,9] = multi_atmosphere[:,:,5,:]
	spinor_atmosphere[:,:,:,10] = multi_atmosphere[:,:,6,:]
	spinor_atmosphere[:,:,:,11] = multi_atmosphere[:,:,7,:]

	if ".fits" in fname:
		primary = fits.PrimaryHDU(spinor_atmosphere)
		primary.header["LTTOP"] = min(spinor_atmosphere[0,0,:,0])
		primary.header["LTINC"] = np.round(spinor_atmosphere[0,0,1,0] - spinor_atmosphere[0,0,0,0], decimals=2)
		primary.header["KCWLR"] = 5000.00

		primary.writeto(fname, overwrite=True)
	else:
		np.savetxt(fname, spinor_atmosphere[0,0], header=" {:2d}  1.0 atmosphere".format(nz), fmt="%5.4e", comments="")

	return spinor_atmosphere

def read_inverted_atmosphere(fpath, atm_range=[0,None,0,None]):
	"""
	Read atmosphere retrieved after inversion and store it in
	Atmosphere() object. We load fully stratified atmosphere and
	node position and values in nodes for all inverted 
	atmospheric parameters.

	Parameters:
	-----------
	fpath : string
		path to the inverted atmosphere. It must be fits file.
	atm_range : list
		list containing [xmin,xmax,ymin,ymax] that define part of the
		cube to be read.

	Return:
	-------
	atmos : globin.atmos.Atmosphere() object
	"""
	try:
		hdu_list = fits.open(fpath)
	except:
		print("--> Error in input.read_inverted_atmosphere()")
		print(f"    Atmosphere file with path '{fpath}' does not exist.")
		sys.exit()

	xmin, xmax, ymin, ymax = atm_range

	data = hdu_list[0].data[xmin:xmax, ymin:ymax]
	nx, ny, npar, nz = data.shape

	atmos = Atmosphere(nx=nx, ny=ny, nz=nz)
	aux = data.astype(np.float64, order="C", copy=True) # because of the pyrh module
	atmos.data = aux
	atmos.logtau = data[0,0,0]
	atmos.header = hdu_list[0].header

	try:
		atmos.pg_top = hdu_list[0].header["PGTOP"]
	except:
		pass

	for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi", "of", "stray"]:
		# if parameter=="stray":
		# 	stray = hdu_list[0].header
		try:
			ind = hdu_list.index_of(parameter)
			data = hdu_list[ind].data[xmin:xmax, ymin:ymax, :]
			nx, ny, nnodes = data.shape

			atmos.nodes[parameter] = np.zeros(nnodes)
			for idn in range(nnodes):
				node = hdu_list[ind].header[f"NODE{idn+1}"]
				atmos.nodes[parameter][idn] = node

			atmos.values[parameter] = data
			atmos.mask[parameter] = np.ones(len(atmos.nodes[parameter]))

			atmos.parameter_scale[parameter] = np.ones((atmos.nx, atmos.ny, nnodes))
		except:
			pass

	#--- check for the spatial regularization weighting
	try:
		atmos.spatial_regularization_weight = atmos.header["REGW"]
		atmos.spatial_regularization = True
	except:
		pass

	if atmos.spatial_regularization:
		for parameter in atmos.nodes:
			reg_weight = atmos.header[f"{parameter}W"]
			atmos.regularization_weight[parameter] = reg_weight

	#--- check for the depth-dependent reglarization weight and type
	for parameter in atmos.dd_regularization_function:
		try:
			atmos.dd_regularization_function[parameter] = atmos.header[f"{parameter}DDF"]
			atmos.dd_regularization_weight[parameter] = atmos.header[f"{parameter}DDW"]
		except:
			pass

	#--- check for the existance of continuum opacity
	try:
		ind = hdu_list.index_of("Continuum_Opacity")
		atmos.chi_c = hdu_list[ind].data
	except:
		atmos.chi_c = None

	return atmos

