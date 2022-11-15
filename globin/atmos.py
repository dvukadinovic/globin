"""
Contributors:
  Dusan Vukadinovic (DV)

17/09/2020 : started writing the code (readme file, structuring)
12/10/2020 : wrote down reading of atmos in .fits format and sent
			 calculation to different process
15/11/2020 : rewriten class Atmosphere
"""

import subprocess as sp
from astropy.io import fits
import numpy as np
import os
import sys
import copy
import time
from scipy.ndimage import median_filter, correlate1d
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import multiprocessing as mp
import scipy.sparse as sp

try:
	import pyrh
except:
	raise ImportError("No 'pyrh' module. Install it before using 'globin'")

import globin
from .spec import Spectrum
from .tools import bezier_spline, spline_interpolation
from .makeHSE import makeHSE

# order of parameters RFs in output file from 'rf_ray'
rf_id = {"temp"  : 0,
		 		 "vz"    : 1,
		 		 "vmic"  : 2,
		 		 "mag"   : 3,
		 		 "gamma" : 4,
		 		 "chi"   : 5}

class Atmosphere(object):
	"""
	Object class for atmospheric models.

	We can read .fits file like cube with assumed shape of (nx, ny, npar, nz)
	where npar=14. Order of parameters are like for RH (MULTI atmos type) where
	after velocity we have magnetic field strength, inclination and azimuth. Last
	6 parameters are Hydrogen number density for first six levels.

	We distinguish different type of 'Atmopshere' based on the input mode. For .fits
	file we call it 'cube' while for rest we call it 'single'.

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

	#--- limit values for atmospheric parameters
	limit_values = {"temp"  : [2800, 10000], 				# [K]
									"vz"    : [-10, 10],						# [km/s]
									"vmic"  : [1e-3, 10],						# [km/s]
									"mag"   : [10, 10000],					# [G]
									"gamma" : [-np.pi, 2*np.pi],		# [rad]
									"chi"   : [-2*np.pi, 2*np.pi],  # [rad]
									"of"    : [0, 20],							#
									"stray" : [0, 1],								#
									"vmac"  : [0, 5]}								# [km/s]

	#--- parameter perturbations for calculating RFs (must be the same as in rf_ray.c)
	delta = {"temp"  : 1,			# [K]
					 "vz"    : 1/1e3,	# [m/s --> km/s]
					 "vmic"  : 1/1e3,	# [m/s --> km/s]
					 "mag"   : 1,			# [G]
					 "gamma" : 0.01,	# [rad]
					 "chi"   : 0.01,	# [rad]
					 "loggf" : 0.001,	#
					 "dlam"  : 1,			# [mA]
					 "of"    : 0.001,
					 "stray" : 0}			# no perturbations (done analytically)

	#--- full names of parameters (for FITS header)
	parameter_name = {"temp"   : "Temperature",
									  "ne"     : "Electron density",
									  "vz"     : "Vertical velocity",
									  "vmic"   : "Microturbulent velocity",
									  "vmac"   : "Macroturbulent velocity",
									  "mag"    : "Magnetic field strength",
									  "gamma"  : "Inclination",
									  "chi"    : "Azimuth",
									  "of"     : "Opacity fudge",
									  "nH"     : "Hydrogen density"}

	#--- normalization values for atmospheric parameters (used in regularization)
	parameter_norm = {"temp"  : 5000,			# [K]
										"vz" 	  : 6,				# [km/s]
										"vmic"  : 6,				# [km/s]
										"mag"   : 1000,			# [G]
										"gamma" : np.pi/3,	# [rad]
										"chi"   : np.pi/3,	# [rad]
										"of"    : 2,				#
										"stray" : 0.1}			#

	def __init__(self, fpath=None, atm_type="multi", atm_range=[0,None,0,None], nx=None, ny=None, nz=None, logtau_top=-6, logtau_bot=1, logtau_step=0.1):
		self.type = atm_type
		self.fpath = fpath

		# parameter scaling for inversino
		self.parameter_scale = {}

		# container for regularization types for atmospheric parameters
		self.regularization = {}
		# number of regularization functions per parameter (not per node):
		#   -- depth regularization counts as 1
		#   -- spatial regularization counts as 2 (in x and y directions each)
		self.nreg = 0

		# current working directory -- appended to paths sent to RH (keyword, atmos, molecules, kurucz)
		self.cwd = "."

		self.hydrostatic = False
		# gas pressure at the top (used for HSE computation from RH)
		self.pg_top = None # [N/m2]

		self.interp_degree = 3

		# nodes: each atmosphere has the same nodes for given parameter
		self.nodes = {}
		# parameters in nodes: shape = (nx, ny, nnodes)
		self.values = {}
		# node mask: we can specify only which nodes to invert (mask==1)
		# structure and ordering same as self.nodes
		self.mask = {}

		# mode of operation
		# self.mode = 0

		self.chi_c = None
		self.pg = None

		self.mu = 1

		self.interpolation_method = "bezier"
		
		# line number in list of lines for which we are inverting atomic data
		self.line_no = {"loggf" : np.array([], dtype=np.int32), "dlam" : np.array([], dtype=np.int32)}
		# global parameters: each is given in a list size equal to number of parameters
		self.global_pars = {"loggf" : np.array([]), "dlam" : np.array([])}

		self.do_fudge = 0
		self.fudge_lam = np.linspace(400, 500, num=3)
		self.of_scatter = 1

		self.ids_tuple = [(0,0)]
		self.n_thread = 1

		# container for the RH
		self.RH = pyrh.RH()

		self.icont = None

		self.add_stray_light = False
		self.invert_stray = False
		self.stray_mode = -1

		self.xmin = atm_range[0]
		self.xmax = atm_range[1]
		self.ymin = atm_range[2]
		self.ymax = atm_range[3]

		self.norm = False

		self.nx = nx
		self.ny = ny
		self.npar = 14
		if nz is None:
			self.logtau = np.arange(logtau_top, logtau_bot+logtau_step, logtau_step)
			self.nz = len(self.logtau)
		else:
			self.nz = nz

		try:
			self.nHtot = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
		except:
			pass

		# if we provide path to atmosphere, read in data
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
				raise ValueError(f"  Unsupported atmosphere type {atm_type}.")

			self.nHtot = np.sum(self.data[:,:,8:,:], axis=-2)
			self.idx_meshgrid, self.idy_meshgrid = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
			self.idx_meshgrid = self.idx_meshgrid.flatten()
			self.idy_meshgrid = self.idy_meshgrid.flatten()
			self.ids_tuple = list(zip(self.idx_meshgrid, self.idy_meshgrid))
			self.fudge = np.ones((self.nx, self.ny, 3, 3))
		else:
			self.header = None
			if (self.nx is not None) and (self.ny is not None) and (self.nz is not None):
				self.data = np.zeros((self.nx, self.ny, self.npar, self.nz), dtype=np.float64)
				self.height = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
				self.rho = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
				self.pg = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
				self.nHtot = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
				if nz is None:
					self.data[:,:,0,:] = self.logtau
				self.idx_meshgrid, self.idy_meshgrid = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
				self.idx_meshgrid = self.idx_meshgrid.flatten()
				self.idy_meshgrid = self.idy_meshgrid.flatten()
				self.ids_tuple = list(zip(self.idx_meshgrid, self.idy_meshgrid))
			else:
				self.data = None

	def __str__(self):
		_str = "<globin.atmos.Atmosphere():\n"
		_str += "  fpath = {}\n".format(self.fpath)
		_str += "  (nx,ny,npar,nz) = ({},{},{},{})>".format(*self.shape)
		return _str

	def read_multi(self, fpath):
		"""
		Read MULTI type atmosphere data.

		Parameters:
		-----------
		fpath : str
			path to the MULTI type atmosphere.
		"""
		lines = open(fpath, "r").readlines()

		# remove commented lines
		lines = [line.rstrip("\n") for line in lines if "*" not in line]

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
		Read in the SPINOR type atmosphere (fits or txt format).

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
		self.data = np.zeros(self.shape)

		# log(tau)
		self.data[:,:,0] = atmos_data[0]
		self.logtau = atmos_data[0,0,0]
		self.logtau_top = self.logtau[0]
		self.logtau_bot = self.logtau[-1]
		self.logtau_step = self.logtau[1] - self.logtau[0]
		# Temperature [K]
		self.data[:,:,1] = atmos_data[2]
		# Electron density [1/cm3]
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
		# Hydrogen populations [1/cm3]
		nH = distribute_hydrogen(atmos_data[2], atmos_data[3], atmos_data[4])
		for idl in range(6):
			self.data[:,:,8+idl] = nH[idl]

		# gas pressure at the top of the atmosphere (in SI units)
		self.pg_top = atmos_data[3,0,0,0]/10

		# container for the gas pressure [dyn/cm2]
		self.pg = np.zeros((self.nx, self.ny, self.nz))
		self.pg = atmos_data[3]

		# container for the density [g/cm3]
		self.rho = np.zeros((self.nx, self.ny, self.nz))
		self.rho = atmos_data[6]

	def read_sir(self, fpath):
		"""
		Read in the SIR type atmosphere.

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

	def read_multi_cube(self, fpath, atm_range=[0,None,0,None]):
		"""
		Read the cube atmosphere of MULTI type in fits format.

		Parameters:
		-----------
		fpath : string
			path to the atmosphere in fits format.
		atm_range : list, optional
			list containing [xmin, xmax, ymin, ymax] defining the patch from the cube
			to be read into the memory.
		"""
		try:
			hdu_list = fits.open(fpath)
		except:
			print("--> Error in globin.atmos.read_cube()")
			print(f"    Atmosphere file with path '{fpath}' does not exist.")
			sys.exit()

		xmin, xmax, ymin, ymax = atm_range

		data = hdu_list[0].data[xmin:xmax, ymin:ymax]
		
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

		self.pg = np.zeros((self.nx, self.ny, self.nz))
		self.rho = np.zeros((self.nx, self.ny, self.nz))

		try:
			self.pg_top = hdu_list[0].header["PGTOP"]
		except:
			pass

		# check for the parameters from inversion for each parameter
		for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi", "of", "stray"]:
			# if parameter=="stray":
			# 	stray = hdu_list[0].header
			try:
				ind = hdu_list.index_of(parameter)
				data = hdu_list[ind].data[xmin:xmax, ymin:ymax, :]
				nx, ny, nnodes = data.shape

				self.nodes[parameter] = np.zeros(nnodes)
				for idn in range(nnodes):
					node = hdu_list[ind].header[f"NODE{idn+1}"]
					self.nodes[parameter][idn] = node

				self.values[parameter] = data
				self.mask[parameter] = np.ones(len(self.nodes[parameter]))

				self.parameter_scale[parameter] = np.ones((self.nx, self.ny, nnodes))
			except:
				pass

		try:
			ind = hdu_list.index_of("Continuum_Opacity")
			self.chi_c = hdu_list[ind].data
		except:
			self.chi_c = None

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
		return self.data[:,:,8:]

	def build_from_nodes(self, flag=None, params=None):
		"""
		Here we build our atmosphere from node values.

		We assume polynomial interpolation of quantities for given degree (given by
		the number of nodes and limited by 3rd degree). Velocity and magnetic field
		vectors are interpolated independantly of any other parameter. Extrapolation
		of these parameters to upper and lower boundary of atmosphere are taken as
		constant from highest / lowest node in atmosphere.

		Temeperature extrapolation in deeper atmosphere is assumed to have same
		slope as in FAL C model (read when loading a package), and to upper boundary
		it is extrapolated constantly as from highest node. From assumed temperature
		and initial run (from which we have continuum opacity) we are solving HE for
		electron and gas pressure iterativly. Few iterations should be enough (we
		stop when change in electron density is retrieved with given accuracy).

		Idea around this is based on Milic & van Noort (2018) [SNAPI code] and for
		interpolation look at de la Cruz Rodriguez & Piskunov (2013) [implemented in
		STiC].
		"""
		if flag is None:
			flag = np.ones((self.nx, self.ny))

		atmos = [self]*(self.nx*self.ny)
		params = [params]*(self.nx*self.ny)
		args = zip(atmos, flag[self.idx_meshgrid, self.idy_meshgrid], self.idx_meshgrid, self.idy_meshgrid, params)

		with mp.Pool(self.n_thread) as pool:
			results = pool.map(func=self._build_from_nodes, iterable=args)

		results = np.array(results)
		self.data = results.reshape(self.nx, self.ny, self.npar, self.nz, order="F")

		# # we need to assign built atmosphere structure to self atmosphere
		# # otherwise self.data would be only 0's.
		# for idl in range(self.nx*self.ny):
		# 	idx, idy = self.idx_meshgrid[idl], self.idy_meshgrid[idl]
		# 	self.data[idx,idy] = atm[idl].data[idx,idy]

	def _build_from_nodes(self, args):
		atmos, flag, idx, idy, params = args

		parameters = atmos.nodes
		if params is not None:
			parameters = [params]

		if flag==0:
			return atmos.data[idx,idy]

		for parameter in parameters:
			# skip over OF and stray light parameters
			if parameter=="stray" or parameter=="of":
				continue

			# K0, Kn by default; True for vmic, mag, gamma and chi
			# K0, Kn = None, None
			K0, Kn = 0, 0

			x = atmos.nodes[parameter]
			y = atmos.values[parameter][idx,idy]

			if parameter=="temp":
				if len(x)>=2:
					K0 = (y[1]-y[0]) / (x[1]-x[0])
					# check if extrapolation at the top atmosphere point goes below the minimum
					# if does, change the slopte so that at top point we have Tmin (globin.limit_values["temp"][0])
					if self.limit_values["temp"][0]>(y[0] + K0 * (atmos.logtau[0]-x[0])):
						K0 = (self.limit_values["temp"][0] - y[0]) / (atmos.logtau[0] - x[0])
					# temperature can not go below 1900 K because the RH will not compute spectrum (dunno why)
					# if 1900>(y[0] + K0 * (atmos.logtau[0]-x[0])):
					# 	K0 = (1900 - y[0]) / (atmos.logtau[0] - x[0])
				# bottom node slope for extrapolation based on temperature gradient from FAL C model
				Kn = splev(x[-1], globin.temp_tck, der=1)
				extrapolate = True
			elif parameter in ["vz", "gamma", "chi"]:
				extrapolate = True
				if len(x)>=2:
					K0 = (y[1]-y[0]) / (x[1]-x[0])
					Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
			elif parameter in ["vmic", "mag"]:
				extrapolate = True
				if len(x)>=2:
					K0 = (y[1]-y[0]) / (x[1]-x[0])
					Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
					# check if extrapolation at the top atmosphere point goes below the minimum
					# if does, change the slopte so that at top point we have parameter_min (globin.limit_values[parameter][0])
					if self.limit_values[parameter][0]>(y[0] + K0 * (atmos.logtau[0]-x[0])):
						K0 = (self.limit_values[parameter][0] - y[0]) / (atmos.logtau[0] - x[0])
					# elif self.limit_values[parameter][1]<(y[0] + K0 * (atmos.logtau[0]-x[0])):
					# 	K0 = (self.limit_values[parameter][1] - y[0]) / (atmos.logtau[0] - x[0])
					# similar for the bottom for maximum/min values
					# if self.limit_values[parameter][1]<(y[-1] + Kn * (atmos.logtau[-1]-x[-1])):
					# 	Kn = (self.limit_values[parameter][1] - y[-1]) / (atmos.logtau[-1] - x[-1])
					if self.limit_values[parameter][0]>(y[-1] + Kn * (atmos.logtau[-1]-x[-1])):
						Kn = (self.limit_values[parameter][0] - y[-1]) / (atmos.logtau[-1] - x[-1])

			if atmos.interpolation_method=="bezier":
				y_new = bezier_spline(x, y, atmos.logtau, K0=K0, Kn=Kn, degree=atmos.interp_degree, extrapolate=extrapolate)
			if atmos.interpolation_method=="spline":
				y_new = spline_interpolation(x, y, atmos.logtau, K0=K0, Kn=Kn, degree=atmos.interp_degree)
			atmos.data[idx,idy,atmos.par_id[parameter],:] = y_new

		return atmos.data[idx,idy]

	def makeHSE(self, flag=None):
		if flag is None:
			flag = np.ones((self.nx, self.ny))
		indx, indy = np.where(flag==1)
		args = zip(indx, indy)

		with mp.Pool(self.n_thread) as pool:
			results = pool.map(func=self._makeHSE, iterable=args)

		results = np.array(results)

		self.data[indx,indy,2] = results[:,0,:]
		self.data[indx,indy,8:] = results[:,1:7,:]
		self.rho[indx,indy] = results[:,7,:]
		self.pg[indx,indy] = results[:,8,:]

	def _makeHSE(self, arg):
		fudge_num = 2
		fudge_lam = np.linspace(401.5, 401.7, num=fudge_num, dtype=np.float64)
		fudge = np.ones((3, fudge_num), dtype=np.float64)

		idx, idy = arg
		
		ne, nH, nHtot, rho, pg = self.RH.hse(self.cwd, 0, self.pg_top,
														 self.data[idx, idy, 0], self.data[idx, idy, 1], 
														 self.data[idx, idy, 2],
					                   self.data[idx, idy, 3], self.data[idx, idy, 4],
					                   self.data[idx, idy, 5]/1e4, self.data[idx, idy, 6], self.data[idx, idy, 7],
					                   self.data[idx, idy, 8:], self.nHtot[idx,idy], 0, fudge_lam, fudge)
		
		result = np.vstack((ne/1e6, nH/1e6, rho/1e3, pg*10))
		return result

	def get_pg_top(self):
		top = self.logtau[0]
		if top<globin.falc.logtau[0]:
			pg_top = pg[0]
		elif top>=globin.falc.logtau[0] and top<=globin.falc.logtau[-1]:
			pg_top = splev(self.logtau[0], globin.pg_tck)
		else:
			sys.exit("Top of atmosphere not in range of FAL C log_tau scale.")

		# convert to SI unit
		self.pg_top = pg_top/10

	def get_pg(self):
		"""
		Compute the gas pressure from total Hydrogen density and electron density.

		Units dyn/cm2 (CGS).
		"""
		nH = np.sum(self.data[...,8:,:], axis=2) * 1e6 # [m3]
		ne = self.data[...,2,:] * 1e6 # [m3]
		self.pg = (nH + ne) * globin.K_BOLTZMAN * self.data[...,1,:] * 10

	def makeHSE_old(self):
		for idx in range(self.nx):
			for idy in range(self.ny):
				pg, pe, _, rho = makeHSE(5000, self.logtau, self.data[idx,idy,1], self.pg_top)
				self.data[idx,idy,2] = pe/10/globin.K_BOLTZMAN/self.data[idx,idy,1] / 1e6
				self.data[idx,idy,8:] = distribute_hydrogen(self.data[idx,idy,1], pg, pe)
				self.rho[idx,idy] = rho

	def _makeHSE_old(self, arg):
		idx, idy = arg
		pg, pe, _, rho = makeHSE(5000, self.logtau, self.data[idx,idy,1], self.pg_top*10)

		ne = pe/10/globin.K_BOLTZMAN/self.data[idx,idy,1] / 1e6 # [1/cm3]
		nH = distribute_hydrogen(self.data[idx,idy,1], pg, pe) # [1/cm3]

		return np.vstack((ne, nH, rho))

	def interpolate_atmosphere(self, x_new, ref_atm):
		new_top = np.round(x_new[0], decimals=2)
		old_top = np.round(ref_atm[0,0,0,0], decimals=2)
		new_bot = np.round(x_new[-1], decimals=2)
		old_bot = np.round(ref_atm[0,0,0,-1], decimals=2)
		if new_top<old_top or new_bot>old_bot:
			print("--> Warning: atmosphere will be extrapolated")
			print("    from {} to {} in optical depth.\n".format(ref_atm[0,0,0,0], x_new[0]))
			raise ValueError("Do not trust it... Just check your parameters for logtau scale in 'params.input' file.")
			self.logtau = ref_atm[0,0,0]
			self.data = ref_atm
			return

		self.nz = len(x_new)

		# check if reference atmosphere is 1D
		ref_atm_nx, ref_atm_ny, _, _ = ref_atm.shape
		oneD = (ref_atm_nx*ref_atm_ny==1)

		self.data = np.zeros((self.nx, self.ny, self.npar, self.nz))
		self.nHtot = np.zeros((self.nx, self.ny, self.nz))
		self.pg = np.zeros((self.nx, self.ny, self.nz))
		self.rho = np.zeros((self.nx, self.ny, self.nz))
		self.data[:,:,0,:] = x_new
		self.logtau = x_new

		if oneD:
			for idp in range(1, self.npar):
				tck = splrep(ref_atm[0,0,0], ref_atm[0,0,idp])
				self.data[...,idp,:] = splev(x_new, tck)
			self.nHtot = np.sum(self.data[...,8:,:], axis=2)
		else:
			for idx in range(self.nx):
				for idy in range(self.ny):
					for idp in range(1, self.npar):
						tck = splrep(ref_atm[0,0,0], ref_atm[idx,idy,idp])
						self.data[idx,idy,idp] = splev(x_new, tck)
					self.nHtot[idx,idy] = np.sum(self.data[idx,idy,8:,:], axis=0)
				
	def save_atmosphere(self, fpath="inverted_atmos.fits", kwargs=None):
		primary = fits.PrimaryHDU(self.data, do_not_scale_image_data=True)
		primary.name = "Atmosphere"

		primary.header.comments["NAXIS1"] = "depth points"
		primary.header.comments["NAXIS2"] = "number of parameters"
		primary.header.comments["NAXIS3"] = "y-axis atmospheres"
		primary.header.comments["NAXIS4"] = "x-axis atmospheres"

		primary.header["XMIN"] = f"{self.xmin+1}"
		if self.xmax is None:
			self.xmax = self.nx
		primary.header["XMAX"] = f"{self.xmax}"
		primary.header["YMIN"] = f"{self.ymin+1}"
		if self.ymax is None:
			self.ymax = self.ny
		primary.header["YMAX"] = f"{self.ymax}"

		primary.header["NX"] = self.nx
		primary.header["NY"] = self.ny
		primary.header["NZ"] = self.nz

		if self.pg_top is not None:
			primary.header["PGTOP"] = (float(self.pg_top), "gas pressure at top (in SI units)")

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

		#primary.header["VMAC"] = ("{:5.3f}".format(self.vmac[0]), "macro-turbulen velocity [km/s]")
		if "vmac" in self.global_pars:
			primary.header["VMAC_FIT"] = ("True", "flag for fitting macro velocity")
		else:
			primary.header["VMAC_FIT"] = ("False", "flag for fitting macro velocity")

		# add keys from kwargs (as dict)
		if kwargs:
			for key in kwargs:
				primary.header[key] = kwargs[key]

		hdulist = fits.HDUList([primary])

		for parameter in self.nodes:
			par_hdu = fits.ImageHDU(self.values[parameter])
			par_hdu.name = parameter

			# par_hdu.header["unit"] = globin.parameter_unit[parameter]
			par_hdu.header.comments["NAXIS1"] = "number of nodes"
			par_hdu.header.comments["NAXIS2"] = "y-axis atmospheres"
			par_hdu.header.comments["NAXIS3"] = "x-axis atmospheres"
			# par_hdu.header.comments["NAXIS4"] = "parameter values"

			for idn in range(len(self.nodes[parameter])):
				par_hdu.header[f"NODE{idn+1}"] = self.nodes[parameter][idn]

			hdulist.append(par_hdu)

		if self.chi_c is not None:
			par_hdu = fits.ImageHDU(self.chi_c)
			par_hdu.name = "Continuum_Opacity"

			par_hdu.header.comments["NAXIS1"] = "number of wavelength points"
			par_hdu.header.comments["NAXIS2"] = "y-axis atmospheres"
			par_hdu.header.comments["NAXIS3"] = "x-axis atmospheres"

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
		pars = list(self.global_pars.keys())

		primary = fits.PrimaryHDU()
		hdulist = fits.HDUList([primary])
		
		for parameter in pars:
			if parameter=="vmac" or parameter=="stray":
				primary.header[parameter] = self.global_pars[parameter][0]
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

		hdulist.writeto(fpath, overwrite=True)

	def update_parameters(self, proposed_steps):
		if self.mode==1 or self.mode==2:
			low_ind, up_ind = 0, 0
			
			#--- update atmospheric parameters
			idp = 0
			for parameter in self.values:
				for idn in range(len(self.nodes[parameter])):
					if self.mask[parameter][idn]==0:
						continue
				# low_ind = up_ind
				# up_ind += len(self.nodes[parameter])
				# step = proposed_steps[:,:,low_ind:up_ind] / self.parameter_scale[parameter]
				# self.values[parameter] += step
				# step *= self.mask[parameter]
					step = proposed_steps[...,idp] / self.parameter_scale[parameter][...,idn]
					self.values[parameter][...,idn] += step
					idp += 1

			#--- update atomic parameters
			for parameter in self.global_pars:
				if self.line_no[parameter].size > 0:
					low_ind = up_ind
					up_ind += self.line_no[parameter].size
					step = proposed_steps[:,:,low_ind:up_ind] / self.parameter_scale[parameter]
					self.global_pars[parameter] += step

		if self.mode==3:
			Natmos = self.nx*self.ny
			NparLocal = self.n_local_pars
			NparGlobal = self.n_global_pars

			#--- update local parameters (atmospheric)			
			local_pars = proposed_steps[:NparLocal*Natmos]
			local_pars = local_pars.reshape(self.nx, self.ny, NparLocal, order="C")

			low_ind, up_ind = 0, 0
			idp = 0
			for parameter in self.values:
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
				if parameter=="vmac" or parameter=="stray":
					low_ind = up_ind
					up_ind += 1
					step = global_pars[low_ind:up_ind] / self.parameter_scale[parameter]
					self.global_pars[parameter] += step
				else:
					Nlines = self.global_pars[parameter].size
					if Nlines>0:
						low_ind = up_ind
						up_ind += Nlines
						step = global_pars[low_ind:up_ind] / self.parameter_scale[parameter]
						self.global_pars[parameter] += step

	def check_parameter_bounds(self, mode):
		for parameter in self.values:
			# inclination is wrapped around [0, 180] interval
			if parameter=="gamma":
				y = np.cos(self.values[parameter])
				self.values[parameter] = np.arccos(y)
			# azimuth is wrapped around [-90, 90] interval
			elif parameter=="chi":
				y = np.sin(self.values[parameter])
				self.values[parameter] = np.arcsin(y)
			else:
				# if parameter=="vmic":
				# 	self.values[parameter] = np.abs(self.values[parameter])
				# check lower boundary condition
				indx, indy, indz = np.where(self.values[parameter]<self.limit_values[parameter][0])
				self.values[parameter][indx,indy,indz] = self.limit_values[parameter][0]

				# check upper boundary condition
				indx, indy, indz = np.where(self.values[parameter]>self.limit_values[parameter][1])
				self.values[parameter][indx,indy,indz] = self.limit_values[parameter][1]

		for parameter in self.global_pars:
			if parameter=="vmac" or parameter=="stray":
				# minimum check
				if self.global_pars[parameter]<self.limit_values[parameter][0]:
					self.global_pars[parameter] = np.array([self.limit_values[parameter][0]], dtype=np.float64)
				# maximum check
				if self.global_pars[parameter]>self.limit_values[parameter][1]:
					self.global_pars[parameter] = np.array([self.limit_values[parameter][1]], dtype=np.float64)
				# get back values into the atmosphere structure
				if parameter=="vmac":
					self.vmac = self.global_pars["vmac"]
				if parameter=="stray":
					self.stray_light = self.global_pars["stray"]
			else:
				Npar = self.line_no[parameter].size
				if Npar > 0:
					for idl in range(Npar):
						# check lower boundary condition
						indx, indy = np.where(self.global_pars[parameter][...,idl]<self.limit_values[parameter][idl,0])
						self.global_pars[parameter][...,idl][indx,indy] = self.limit_values[parameter][idl,0]

						# check upper boundary condition
						indx, indy = np.where(self.global_pars[parameter][...,idl]>self.limit_values[parameter][idl,1])
						self.global_pars[parameter][...,idl][indx,indy] = self.limit_values[parameter][idl,1]

	def smooth_parameters(self, cycle, num=5, std=2.5):
		if self.nx>=num and self.ny>=num:
			for parameter in self.nodes:
				for idn in range(len(self.nodes[parameter])):
					if parameter=="chi":
						aux = globin.utils.azismooth(self.values[parameter][...,idn]*180/np.pi, num)
						aux *= np.pi/180
					elif parameter=="gamma":
						aux = globin.utils.mygsmooth(self.values[parameter][...,idn]*180/np.pi, num, std)
						aux *= np.pi/180
					elif parameter=="of":
						aux = self.values[parameter][...,idn]
					else:
						aux = globin.utils.mygsmooth(self.values[parameter][...,idn], num, std)

					self.values[parameter][...,idn] = median_filter(aux, size=4)
		else:
			size = 11
			if cycle==1:
				size = 7
			if cycle>=2:
				size = 5

			for parameter in self.nodes:
				for idn in range(len(self.nodes[parameter])):
					tmp = median_filter(self.values[parameter][...,idn], size=size)
					self.values[parameter][...,idn] = tmp

		for parameter in self.global_pars:
			if parameter=="loggf" and len(self.line_no["loggf"])>0:
				size = self.line_no[parameter].size
				nx, ny = 1, 1	
				if self.mode==2:
					nx = self.nx
					ny = self.ny
				# delta = 0.0413 --> 10% relative error in oscillator strength (f)
				self.global_pars[parameter] += np.random.normal(loc=0, scale=0.0413, size=size*nx*ny).reshape(nx, ny, size)

	def compute_errors(self, H, chi2):
		invH = np.linalg.inv(H)
		diag = np.einsum("...kk->...k", invH)
		diag = diag.flatten(order="F")

		npar = self.n_local_pars + self.n_global_pars

		self.errors = np.zeros(npar)

		low, up = 0, 0
		for parameter in self.nodes:
			scale = globin.parameter_scale[parameter].flatten()
			up += len(self.nodes[parameter])
			self.errors[low:up] = np.sqrt(chi2/npar * diag[low:up] / scale**2)
			low = up
		for parameter in self.global_pars:
			scale = globin.parameter_scale[parameter]
			up += scale.size
			self.errors[low:up] = np.sqrt(chi2/npar * diag[low:up] / scale**2)
			low = up

	def get_hsra_cont(self):
		"""
		Compute the HSRA continuum intensity. Use the wavelength vector
		specified in the atmosphere (not the HSRA one).
		"""
		if self.norm and self.norm_level=="hsra":
			hsra = Atmosphere(f"{__path__}/data/hsrasp.dat", atm_type="spinor")
			hsra.wavelength_air = self.wavelength_obs
			hsra.mu = self.mu
			spec = hsra.compute_spectra()
			self.icont = spec.spec[0,0,0,0]

	def compute_spectra(self, synthesize=None):
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
		
		with mp.Pool(self.n_thread) as pool:
			spectra_list = pool.map(func=self._compute_spectra_sequential, iterable=args)

		spectra_list = np.array(spectra_list)
		natm, ns, nw = spectra_list.shape
		spectra_list = spectra_list.reshape(natm, ns, nw, order="C")
		spectra_list = np.swapaxes(spectra_list, 1, 2)

		_, nw, _ = spectra_list.shape

		# spectra = Spectrum(self.nx, self.ny, len(self.wavelength_obs), nz=self.nz)
		spectra = Spectrum(self.nx, self.ny, nw, nz=self.nz)
		# spectra.wavelength = self.wavelength_obs
		spectra.wavelength = spectra_list[0,:,-1]
		spectra.spec[indx,indy] = spectra_list[...,:-1]

		if self.norm:
			if self.norm_level=="hsra":
				spectra.spec /= self.icont
			if self.norm_level==1:
				Ic = spectra.spec[:,:,0,0]
				Ic = np.repeat(Ic[...,np.newaxis], spectra.spec.shape[2], axis=-1)
				Ic = np.repeat(Ic[...,np.newaxis], spectra.spec.shape[3], axis=-1)
				spectra.spec /= Ic

		return spectra

	def _compute_spectra_sequential(self, args):
		idx, idy = args

		if (self.line_no["loggf"].size>0) or (self.line_no["dlam"].size>0):
			if self.mode==2:
				_idx, _idy = idx, idy
			elif self.mode==3:
				_idx, _idy = 0, 0
			spec = self.RH.compute1d(self.cwd, self.mu, 0, self.data[idx, idy, 0], 
									self.data[idx, idy, 1], 
								  self.data[idx, idy, 2], self.data[idx, idy, 3], 
								  self.data[idx, idy, 4], self.data[idx, idy, 5]/1e4, 
								  self.data[idx, idy, 6], self.data[idx, idy, 7],
								  self.data[idx, idy, 8:], self.wavelength_vacuum,
								  self.do_fudge, self.fudge_lam, self.fudge[idx,idy],
								  self.line_no["loggf"], self.global_pars["loggf"][_idx, _idy],
								  self.line_no["dlam"], self.global_pars["dlam"][_idx, _idy]/1e4)
		else:
			spec = self.RH.compute1d(self.cwd, self.mu, 0, self.data[idx, idy, 0], 
									self.data[idx, idy, 1], 
								  self.data[idx, idy, 2], self.data[idx, idy, 3], 
								  self.data[idx, idy, 4], self.data[idx, idy, 5]/1e4, 
								  self.data[idx, idy, 6], self.data[idx, idy, 7],
								  self.data[idx, idy, 8:], self.wavelength_vacuum,
								  self.do_fudge, self.fudge_lam, self.fudge[idx,idy],
								  self.line_no["loggf"], self.global_pars["loggf"],
								  self.line_no["dlam"], self.global_pars["dlam"]/1e4)

		# interpolate the output spectrum to an observation wavelength 
		# grid because conversion air --> vacuum --> air has a significant
		# difference which hinders the vz inversion (~ 0.13 km/s for 
		# Hinode line par).

		if (not np.array_equal(self.wavelength_obs, self.wavelength_air)):
			tck = splrep(self.wavelength_air, spec.I)
			StokesI = splev(self.wavelength_obs, tck, ext=3)
			tck = splrep(self.wavelength_air, spec.Q)
			StokesQ = splev(self.wavelength_obs, tck, ext=1)
			tck = splrep(self.wavelength_air, spec.U)
			StokesU = splev(self.wavelength_obs, tck, ext=1)
			tck = splrep(self.wavelength_air, spec.V)
			StokesV = splev(self.wavelength_obs, tck, ext=1)

			return StokesI, StokesQ, StokesU, StokesV, spec.lam

		# when we are computing only the continuum intensity from HSRA for
		# spectrum normalization or in synthesis mode 
		# (wavelength_air == wavelength_obs)
		return spec.I, spec.Q, spec.U, spec.V, spec.lam

	def compute_rfs(self, rf_noise_scale, Ndof, weights=1, synthesize=[], rf_type="node", mean=False, old_rf=None, old_pars=None, instrumental_profile=None):
		"""
		Parameters:
		-----------
		atmos : Atmosphere() object
			atmosphere

		rf_noise_scale : ndarray
			noise

		skip : tuple list
			list of tuples (idx, idy) for which to skip spectrum synthesis

		old_rf : ndarray
		"""
		self.build_from_nodes(synthesize)
		if self.hydrostatic:
			self.makeHSE(synthesize)
		# globin.visualize.plot_atmosphere(self, ["temp", "vz", "vmic", "mag", "gamma", "chi"])
		# plt.show()
		spec = self.compute_spectra(synthesize)
		# globin.visualize.plot_spectra(spec.spec[0,0], spec.wavelength)
		# plt.show()
		# sys.exit()
		Nw = len(self.wavelength_obs)

		active_indx, active_indy = np.where(synthesize==1)

		node_RF = np.zeros((spec.nx, spec.ny, Nw, 4))
		if self.spatial_regularization:
			self.scale_LT = np.zeros((self.nx*self.ny*self.n_local_pars + self.n_global_pars))
			addition = np.arange(0, self.nx*self.ny*self.n_local_pars, self.n_local_pars)
			shift = 0

		#--- loop through local (atmospheric) parameters and calculate RFs
		free_par_ID = 0
		for parameter in self.nodes:
			nodes = self.nodes[parameter]
			values = self.values[parameter]
			perturbation = self.delta[parameter]

			# for nodeID in trange(len(nodes), desc=f"{parameter} nodes", leave=None, ncols=0):
			for nodeID in range(len(nodes)):
				if self.mask[parameter][nodeID]==0:
					continue			
				#--- positive perturbation
				self.values[parameter][:,:,nodeID] += perturbation
				if parameter=="of":
					self.make_OF_table(self.wavelength_vacuum)
				elif parameter=="stray":
					pass
				else:
					self.build_from_nodes(synthesize, params=parameter)
				spectra_plus = self.compute_spectra(synthesize)

				#--- negative perturbation (except for inclination and azimuth)
				if parameter=="gamma" or parameter=="chi":
					node_RF = (spectra_plus.spec - spec.spec ) / perturbation
				elif parameter=="stray":
					if self.stray_type=="hsra":
						node_RF = self.hsra_spec - spec.spec
					if self.stray_type=="gray":
						node_RF = -spec.spec
				else:
					self.values[parameter][:,:,nodeID] -= 2*perturbation
					if parameter=="of":	
						self.make_OF_table(self.wavelength_vacuum)
					else:
						self.build_from_nodes(synthesize, params=parameter)
					spectra_minus = self.compute_spectra(synthesize)

					node_RF = (spectra_plus.spec - spectra_minus.spec ) / 2 / perturbation

				#--- compute parameter scale				
				node_RF *= weights
				node_RF /= rf_noise_scale
				node_RF /= np.sqrt(Ndof)
				
				scale = np.sqrt(np.sum(node_RF**2, axis=(2,3)))
				
				# save parameter scales from previous iteration
				indx, indy = np.where(scale==0)
				scale[indx,indy] = self.parameter_scale[parameter][indx,indy,nodeID]

				# recompute the scales for only those pixels for which we computed spectra
				# (do not touch old ones)
				# indx, indy = np.where(synthesize==1)
				self.parameter_scale[parameter][active_indx,active_indy,nodeID] = scale[active_indx,active_indy]
				# self.parameter_scale[parameter][active_indx,active_indy,nodeID] = 1

				#--- set RFs value
				self.rf[active_indx,active_indy,free_par_ID] = np.einsum("ikl,i->ikl", node_RF[active_indx, active_indy], 1/self.parameter_scale[parameter][active_indx,active_indy,nodeID])
				free_par_ID += 1

				#--- return back perturbations (node way)
				if parameter=="of":	
					self.values[parameter][:,:,nodeID] += perturbation
					self.make_OF_table(self.wavelength_vacuum)
				elif parameter=="stray":
					pass
				else:
					if parameter=="gamma" or parameter=="chi":
						self.values[parameter][:,:,nodeID] -= perturbation
					else:
						self.values[parameter][:,:,nodeID] += perturbation
					self.build_from_nodes(synthesize, params=parameter)

			# reshape parameter scale for the scaling of regularization Jacobian matrix
			if self.spatial_regularization:
				tmp = self.parameter_scale[parameter].reshape(self.nx*self.ny, len(nodes))
				tmp = tmp.reshape(self.nx*self.ny*len(nodes))
				inds = np.arange(len(nodes), dtype=np.int32) + shift
				inds = np.repeat(inds[np.newaxis,:], self.nx*self.ny, axis=0)
				inds = (inds.T + addition).T.flatten()
				self.scale_LT[inds] = tmp
				shift += len(nodes)

		#--- loop through global parameters and calculate RFs
		skip_par = -1
		if self.n_global_pars>0:
			#--- loop through global parameters and calculate RFs
			for parameter in self.global_pars:
				if parameter=="vmac":
					kernel_sigma = spec.get_kernel_sigma(self.vmac)
					kernel = spec.get_kernel(self.vmac, order=1)

					args = zip(spec.spec.reshape(self.nx*self.ny, Nw, 4), [kernel]*self.nx*self.ny)

					with mp.Pool(self.n_thread) as pool:
						results = pool.map(func=_compute_vmac_RF, iterable=args)

					results = np.array(results)
					self.rf[:,:,free_par_ID] = results.reshape(self.nx, self.ny, Nw, 4)
					self.rf[:,:,free_par_ID] *= kernel_sigma * self.step / self.global_pars["vmac"]
					self.rf[:,:,free_par_ID] /= rf_noise_scale
					self.rf[:,:,free_par_ID] /= np.sqrt(Ndof)
					self.rf[:,:,free_par_ID] = np.einsum("ijkl,l->ijkl", self.rf[:,:,free_par_ID], weights)

					self.parameter_scale[parameter] = np.sqrt(np.sum(self.rf[:,:,free_par_ID,:,:]**2))
					self.rf[:,:,free_par_ID,:,:] /= self.parameter_scale[parameter]

					skip_par = free_par_ID
					free_par_ID += 1

				elif parameter=="stray":
					if self.stray_type=="hsra":
						diff = self.hsra_spec - spec.spec
					if self.stray_type=="gray":
						diff = -spec.spec
					diff *= weights
					diff /= rf_noise_scale
					diff /= np.sqrt(Ndof)

					scale = np.sqrt(np.sum(diff**2))
					self.parameter_scale[parameter] = scale

					self.rf[:,:,free_par_ID,:,:] = diff / self.parameter_scale[parameter]
					free_par_ID += 1

				elif parameter=="loggf" or parameter=="dlam":
					if self.line_no[parameter].size > 0:
						perturbation = self.delta[parameter]

						for idp in range(self.line_no[parameter].size):
							# model_plus.global_pars[parameter][...,idp] += perturbation
							self.global_pars[parameter][...,idp] += perturbation
							spec_plus = self.compute_spectra(synthesize)

							self.global_pars[parameter][...,idp] -= 2*perturbation
							spec_minus = self.compute_spectra(synthesize)

							diff = (spec_plus.spec - spec_minus.spec) / 2 / perturbation
							diff *= weights
							diff /= rf_noise_scale
							diff /= np.sqrt(Ndof)

							if self.mode==2:
								scale = np.sqrt(np.sum(diff**2, axis=(2,3)))
								for idx in range(self.nx):
									for idy in range(self.ny):
										if not np.isnan(np.sum(scale[idx,idy])):
											self.parameter_scale[parameter][idx,idy,idp] = scale[idx,idy]
										else:
											self.parameter_scale[parameter][idx,idy,idp] = 1
							elif self.mode==3:
								scale = np.sqrt(np.sum(diff**2))
								self.parameter_scale[parameter][...,idp] = scale

							if self.mode==2:
								for idx in range(self.nx):
									for idy in range(self.ny):
										self.rf[idx,idy,free_par_ID] = diff[idx,idy] / self.parameter_scale[parameter][idx,idy,idp]
							elif self.mode==3:
								self.rf[:,:,free_par_ID,:,:] = diff / self.parameter_scale[parameter][0,0,idp]
							free_par_ID += 1
							
							self.global_pars[parameter][...,idp] += perturbation

		#--- broaden the spectra
		if not mean:
			spec.broaden_spectra(self.vmac, synthesize, self.n_thread)
			# if self.vmac!=0:
			# 	kernel = spec.get_kernel(self.vmac, order=0)
				# self.rf = broaden_rfs(self.rf, kernel, synthesize, skip_par, self.n_thread)

		#--- add the stray light component:
		if self.add_stray_light:
			for idx in range(self.nx):
				for idy in range(self.ny):
					if self.stray_mode==1 or self.stray_mode==2:
						stray_factor = self.stray_light[idx,idy]
					if self.stray_mode==3:
						if self.invert_stray:
							stray_factor = self.global_pars["stray"]
						else:
							stray_factor = self.stray_light[idx,idy]
					if self.stray_type=="hsra":
						spec.spec[idx,idy] = stray_factor * self.hsra_spec + (1-stray_factor) * spec.spec[idx,idy]
					if self.stray_type=="gray":
						spec.spec[idx,idy,:,0] = stray_factor + (1-stray_factor) * spec.spec[idx,idy,:,0]
						spec.spec[idx,idy,:,1] = (1-stray_factor) * spec.spec[idx,idy,:,1]
						spec.spec[idx,idy,:,2] = (1-stray_factor) * spec.spec[idx,idy,:,2]
						spec.spec[idx,idy,:,3] = (1-stray_factor) * spec.spec[idx,idy,:,3]

		#--- add instrumental broadening
		if instrumental_profile is not None:
			spec.instrumental_broadening(kernel=instrumental_profile, flag=synthesize, n_thread=self.n_thread)
			self.rf = broaden_rfs(self.rf, instrumental_profile, synthesize, -1, self.n_thread)

		# for idp in range(3):
		# 	plt.plot(self.rf[0,0,idp,:,0], label=f"{idp+1}")
		# plt.legend()
		# plt.show()
		# sys.exit()

		# atmos.spec[active_indx, active_indy] = spec.spec[active_indx, active_indy]

		return spec

	def get_regularization_gamma(self):
		# number of regularization functions (per parameter; not per node)
		# if self.nreg==0:
		# 	return None

		"""
		Gamma matrix containing the regularization function values for each pixel.
		We already summed the contributions from each atmospheric parameter.
		"""
		npar = self.n_local_pars
		
		# number of regularization functions (per pixel)
		# we regularize spatialy in x- and y-axis
		nreg = 2

		#--- get regularization function values
		gamma = np.zeros((self.nx, self.ny, nreg, npar))
		
		idp = 0
		for parameter in self.nodes:
			for idn in range(len(self.nodes[parameter])):
				# p@x - p@x-1 (difference to the upper pixel)
				gamma[1:,:,0,idp] = self.values[parameter][1:,:,idn] - self.values[parameter][:-1,:,idn]
				gamma[1:,:,0,idp] /= self.parameter_norm[parameter]
				# p@y - p@y-1 (difference to the left pixel)
				gamma[:,1:,1,idp] = self.values[parameter][:,1:,idn] - self.values[parameter][:,:-1,idn]
				gamma[:,1:,1,idp] /= self.parameter_norm[parameter]
				idp += 1

		return gamma.reshape(self.nx, self.ny, nreg*npar, order="F")

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
		low, up = 0, 0
		for i_, parameter in enumerate(self.nodes):
			low = up
			up = low + len(self.nodes[parameter])
			norm[low:up] = self.parameter_norm[parameter]

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
					
					values = np.append(values, np.ones(npar)/norm)

					# derivative in respect to neighbouring pixel (up)
					tmp = (idx-1)*self.ny + idy
					ind = tmp*npar + idp
					rows = np.append(rows, ind)

					ind = ida*npar*nreg + idp*2
					cols = np.append(cols, ind)

					values = np.append(values, -1*np.ones(npar)/norm)
				
				# Gamma_1 contribution
				if idy>0:
					ind = ida*npar + idp
					rows = np.append(rows, ind)
					
					ind = ida*npar*nreg + idp*2 + 1
					cols = np.append(cols, ind)
					
					values = np.append(values, np.ones(npar)/norm)

					# derivative in respect to neighbouring pixel (left)
					tmp = idx*self.ny + idy-1
					ind = tmp*npar + idp
					rows = np.append(rows, ind)

					ind = ida*npar*nreg + idp*2 + 1
					cols = np.append(cols, ind)

					values = np.append(values, -1*np.ones(npar)/norm)

		shape = (self.nx*self.ny*npar + self.n_global_pars, 
						 self.nx*self.ny*npar*nreg)
		LT = sp.csr_matrix((values, (rows, cols)), shape=shape, dtype=np.float64)
		
		# LTL = LT.dot(LT.transpose())

		# plt.imshow(LTL.toarray().T, origin="upper", cmap="bwr")
		# plt.colorbar()
		# plt.show()

		return LT

	def make_OF_table(self, wavelength_vacuum):
		if wavelength_vacuum[0]<=210:
			print("  Sorry, but OF is not properly implemented for wavelengths")
			print("    belowe 210 nm. You have to wait for newer version of globin.")
			sys.exit()
		
		if self.of_num==1:
			self.fudge_lam = np.zeros(4, dtype=np.float64)
			self.fudge = np.ones((self.nx, self.ny, 3,4), dtype=np.float64)

			# first point outside of interval (must be =1)
			self.fudge_lam[0] = wavelength_vacuum[0] - 0.0002
			self.fudge[...,0] = 1

			# left edge of wavelength interval
			self.fudge_lam[1] = wavelength_vacuum[0] - 0.0001
			self.fudge[...,0,1] = self.values["of"][...,0]
			if self.of_scatter:	
				self.fudge[...,1,1] = self.values["of"][...,0]
			self.fudge[...,2,1] = 1

			# right edge of wavelength interval
			self.fudge_lam[2] = wavelength_vacuum[-1] + 0.0001
			self.fudge[...,0,2] = self.values["of"][...,0]
			if self.of_scatter:
				self.fudge[...,1,2] = self.values["of"][...,0]
			self.fudge[...,2,1] = 1		

			# last point outside of interval (must be =1)
			self.fudge_lam[3] = wavelength_vacuum[-1] + 0.0002
			self.fudge[...,3] = 1
		#--- for multi-wavelength OF correction
		else:
			Nof = self.of_num + 2

			self.fudge_lam = np.zeros(Nof, dtype=np.float64)
			self.fudge = np.ones((self.nx, self.ny, 3, Nof), dtype=np.float64)
			
			# first point outside of interval (must be =1)
			self.fudge_lam[0] = wavelength_vacuum[0] - 0.0002
			self.fudge[...,0] = 1

			for idf in range(self.of_num):
				# mid points
				shift = 0
				if idf==0:
					shift = -0.0001
				if idf==self.of_num-1:
					shift = 0.0001
				self.fudge_lam[idf+1] = self.nodes["of"][idf] + shift
				self.fudge[...,0,idf+1] = self.values["of"][...,idf]
				if self.of_scatter:
					self.fudge[...,1,idf+1] = self.values["of"][...,idf]
				self.fudge[...,2,idf+1] = 1	

			# last point outside of interval (must be =1)
			self.fudge_lam[-1] = wavelength_vacuum[-1] + 0.0002
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

	def set_wavelength(self, lmin=None, lmax=None, nwl=None, dlam=None, fpath=None, unit="A"):
		"""
		Create the wavelength vector in Angstroms. Transform the values to vacuume one for 
		synthesis in RH.

		If 'fpath' is provided, we first read the wavelengths from this file.

		One of 'nwl' and 'dlam' must be provided to be able to compute the wavelength 
		vector. Otherwise, an error is thrown.

		Parameters:
		-----------
		lmin : float (optional)
			lower limit of wavelength vector.
		lmax : float (optional)
			upper limit of wavelength vector.
		nwl : float (optional)
			number of wavelength points between 'lmin' and 'lmax'.
		dlam : float (optional)
			spacing in wavelength vector between 'lmin' and 'lamx'.
		fpath : str (optinoal)
			a path to a text file containing the wavelength vector. If it is 
			provided, if has priority over manual specification of wavelength
			vector.

		Error:
		------
		If neighter of 'nwl' and 'dlam' is provided, an error is thrown.
		"""

		if fpath is not None:
			self.wavelength = np.loadtxt(fpath)
		else:
			if nwl is not None:
				self.wavelength = np.linspace(lmin, lmax, num=nwl)
			elif dlam is not None:
				self.wavelength = np.arange(lmin, lmax+dlam, dlam)
			else:
				sys.exit("globin.atmos.Atmosphere.set_wavelength():\n  Neighter the number of wavelenths or spacing has been provided.")

		# transform values to nm and compute the wavelengths in vacuume
		if unit=="A":
			self.wavelength /= 10
		self.wavelength_vacuum = globin.rh.write_wavs(self.wavelength, fname=None)

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

def broaden_rfs(rf, kernel, flag, skip_par, n_thread):
	nx, ny, npar, nw, ns = rf.shape

	indx, indy = np.where(flag==1)
	indp = np.arange(npar, dtype=np.int32)
	if skip_par!=-1:
		npar -= 1
		indp = np.delete(indp, skip_par)

	# this happens when vmac is the only parameter in inversion;
	# we do not need to broaden RF for vmac
	if len(indp)==0:
		return rf

	_rf = rf[indx,indy][:,indp]
	_rf = _rf.reshape(len(indx)*npar, nw, ns)
	
	args = zip(_rf,[kernel]*len(indx)*npar)

	with mp.Pool(n_thread) as pool:
		results = pool.map(func=_broaden_rfs, iterable=args)

	results = np.array(results)
	results = results.reshape(len(indx), npar, nw, ns)
	rf[indx,indy][:,indp] = results

	return rf

def _broaden_rfs(args):
	rf, kernel = args

	for ids in range(4):
		rf[...,ids] = correlate1d(rf[...,ids], kernel)

	return rf

def _compute_vmac_RF(args):
	spec, kernel = args

	for ids in range(4):
		spec[...,ids] = correlate1d(spec[...,ids], kernel)

	return spec

def distribute_hydrogen(temp, pg, pe, vtr=0):
	Ej = 13.59844

	ne = pe/10 / globin.K_BOLTZMAN/temp # [1/m3]
	C1 = ne/2 * globin.PLANCK**3 / (2*np.pi*globin.ELECTRON_MASS*globin.K_BOLTZMAN*temp)**(3/2)
	
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
	
	gamma = atm[6]
	# if "gamma" in globin.atm.nodes:
		# gamma = 2*np.arctan(atm[6])
		# gamma = np.arccos(atm[6])

	chi = atm[7]
	# if "chi" in globin.atm.nodes:
	# 	# chi = 4*np.arctan(atm[7])
	# 	chi = np.arccos(atm[7])

	globin.rh.write_B(f"{fpath}.B", atm[5]/1e4, gamma, chi)

	if np.isnan(np.sum(atm)):
		print(fpath)
		print("We have NaN in atomic structure!\n")
		sys.exit()

def compute_full_rf(atmos, local_pars=None, global_pars=None, norm=False, fpath=None):
	if (local_pars is None) and (global_pars is None):
		raise ValueError("None of atmospheric or atomic are given for RF computation.")

	dlogtau = atmos.logtau[1] - atmos.logtau[0]

	if norm:
		atmos.norm = True
		atmos.norm_level = 1

	# compute the total number of free parameters (have to sum atomic for each line)
	n_global = 0
	if global_pars is not None:
		for parameter in global_pars:
			n_global += atmos.line_no[parameter].size
	
	n_local = 0
	if local_pars is not None:
		n_local = len(local_pars)
	
	n_pars = n_local + n_global
	
	rf = np.zeros((atmos.nx, atmos.ny, n_pars, atmos.nz, len(atmos.wavelength), 4), dtype=np.float64)

	i_ = -1
	free_par_ID = 0
	if local_pars is not None:
		for idp, parameter in enumerate(local_pars):
			perturbation = atmos.delta[parameter]
			parID = atmos.par_id[parameter]

			for idz in tqdm(range(atmos.nz), desc=parameter):
				atmos.data[:,:,parID,idz] += perturbation
				spec_plus = atmos.compute_spectra()

				atmos.data[:,:,parID,idz] -= 2*perturbation
				spec_minus = atmos.compute_spectra()
				
				diff = spec_plus.spec - spec_minus.spec
				rf[:,:,free_par_ID,idz] = diff / 2 / perturbation

				# remove perturbation from data
				atmos.data[:,:,parID,idz] += perturbation

			free_par_ID += 1

	#--- loop through global parameters and calculate RFs
	if global_pars is not None:
		for parameter in global_pars:
			if parameter=="vmac":
				radius = int(4*kernel_sigma + 0.5)
				x = np.arange(-radius, radius+1)
				phi = np.exp(-x**2/kernel_sigma**2)
				# normalaizing the profile
				phi *= 1/(np.sqrt(np.pi)*kernel_sigma)
				kernel = phi*(2*x**2/kernel_sigma**2 - 1)
				# since we are correlating, we need to reverse the order of data
				kernel = kernel[::-1]

				spec, _ = compute_spectra(atmos)
				if not globin.mean:
					spec.broaden_spectra(atmos.vmac)

				for idx in range(atmos.nx):
					for idy in range(atmos.ny):
						for sID in range(1,5):
							rf[idx,idy,free_par_ID,0,:,sID-1] = correlate1d(spec[idx,idy,:,sID], kernel)
							rf[idx,idy,free_par_ID,0,:,sID-1] *= 1/atmos.vmac * globin.parameter_scale["vmac"]
				free_par_ID += 1
			elif parameter in ["loggf", "dlam"]:
				perturbation = atmos.delta[parameter]

				for idp in tqdm(range(atmos.line_no[parameter].size), desc=parameter):
					atmos.global_pars[parameter][...,idp] += perturbation
					spec_plus = atmos.compute_spectra()

					atmos.global_pars[parameter][...,idp] -= 2*perturbation
					spec_minus = atmos.compute_spectra()

					diff = (spec_plus.spec - spec_minus.spec) / 2 / perturbation

					rf[:,:,free_par_ID] = diff
					if atmos.mode==3:
						rf[:,:,free_par_ID,:,:] = np.repeat(diff[:,:,np.newaxis,:,:], atmos.nz , axis=2)

					free_par_ID += 1
					
					atmos.global_pars[parameter][...,idp] += perturbation

			else:
				print(f"Parameter {parameter} not yet Supported.\n")

	# if we have provided the path, save the RFs
	if fpath is not None:
		primary = fits.PrimaryHDU(rf)
		primary.name = "RF"

		# np.zeros((atmos.nx, atmos.ny, n_pars, atmos.nz, len(globin.wavelength), 4), dtype=np.float64)

		primary.header.comments["NAXIS1"] = "stokes components"
		primary.header.comments["NAXIS2"] = "number of wavelengths"
		primary.header.comments["NAXIS3"] = "depth points"
		primary.header.comments["NAXIS4"] = "number of parameters"
		primary.header.comments["NAXIS5"] = "y-axis atmospheres"
		primary.header.comments["NAXIS6"] = "x-axis atmospheres"

		primary.header["STOKES"] = ("IQUV", "the Stokes vector order")

		if norm:
			primary.header["NORMED"] = ("TRUE", "flag for spectrum normalization")

		i_ = 1
		if local_pars:
			for par in local_pars:
				primary.header[f"PAR{i_}"] = (par, "parameter")
				primary.header[f"PARID{i_}"] = (i_, "parameter ID")
				i_ += 1
		if global_pars:
			for par in global_pars:
				primary.header[f"PAR{i_}"] = (par, "parameter")
				primary.header["PARIDMIN"] = (i_, "parameter ID min")
				primary.header["PARIDMAX"] = (i_ + atmos.line_no[par].size -1, "parameter ID max")
				i_ += atmos.line_no[par].size

		hdulist = fits.HDUList([primary])

		#--- wavelength list
		par_hdu = fits.ImageHDU(atmos.wavelength)
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
		hdulist.writeto(fpath, overwrite=True)

	return rf

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
	Routine for converting 'atmos_data' from SPINOR to MULTI type.

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

	# logtau
	atmos.data[:,:,0] = atmos_data[0]
	atmos.logtau = atmos_data[0,0,0]
	atmos.logtau_top = atmos.logtau[0]
	atmos.logtau_bot = atmos.logtau[-1]
	atmos.logtau_step = atmos.logtau[1] - atmos.logtau[0]
	# temperature [K]
	atmos.data[:,:,1] = atmos_data[2]
	# electron density [1/cm3]
	amtos.data[:,:,2] = atmos_data[4]/10/globin.K_BOLTZMAN/atmos_data[2] / 1e6
	# LOS velocity [km/s]
	atmos_data[:,:,3] = atmos_data[9]/1e5
	# micro-turbulent velocity [km/s]
	atmos_data[:,:,4] = atmos_data[8]/1e5
	# magnetic field strength [G]
	atmos_data[:,:,5] = atmos_data[7]
	# magnetic field inclination [rad]
	atmos.data[:,:,6] = atmos_data[-2]# * np.pi/180
	# magnetic field zazimuth [rad]
	atmos.data[:,:,7] = atmos_data[-1]# * np.pi/180
	# hydrogen density [1/cm3]
	atmos.data[:,:,8] = distribute_hydrogen(atmos_data[2], atmos_data[3], atmos_data[4])

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

	try:
		ind = hdu_list.index_of("Continuum_Opacity")
		atmos.chi_c = hdu_list[ind].data
	except:
		atmos.chi_c = None

	return atmos
