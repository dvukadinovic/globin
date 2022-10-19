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
	raise ImportError("No module 'pyrh'.")

# from .input import read_multi, read_spinor, read_inverted_atmosphere

import globin
from .mppools import pool_spinor2multi
from .spec import Spectrum
from .tools import bezier_spline
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

	We distinguish different type of 'Atmopshere' bsaed on input mode. For .fits
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
									"mag"   : [1, 10000],						# [G]
									"gamma" : [-np.pi, np.pi],			# [rad]
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
		self.fpath = fpath
		self.type = atm_type

		# list of fpath's to line lists (in mode=2)
		self.line_lists_path = []
		# list of fpath's to each atmosphere column
		self.atm_name_list = []
		# parameter scaling for inversino
		self.parameter_scale = {}

		# container for regularization types for atmospheric parameters
		self.regularization = {}
		# number of regularization functions per parameter (not per node):
		#   -- depth regularization counts as 1
		#   -- spatial regularization counts as 2 (in x and y directions each)
		self.nreg = 0

		self.hydrostatic = False
		# gas pressure at the top (used for HSE computation from RH)
		self.pg_top = 1 # [N/m2]

		# self.atmosphere.RH = pyrh.RH()
		# self.atmosphere.n_thread = self.n_thread
		# self.atmosphere.temp_tck = self.falc.temp_tck
		# self.atmosphere.interp_degree = self.interp_degree

		# nodes: each atmosphere has the same nodes for given parameter
		self.nodes = {}
		# parameters in nodes: shape = (nx, ny, nnodes)
		self.values = {}
		# node mask: we can specify only which nodes to invert (mask==1)
		# structure and ordering same as self.nodes
		self.mask = {}

		self.chi_c = None
		
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
		if self.fpath is not None:
			extension = self.fpath.split(".")[-1]

			if extension=="dat" or extension=="txt" or extension=="atmos":
				if self.type=="spinor":
					atmos = read_spinor(self.fpath)
				elif self.type=="multi":
					atmos = read_multi(self.fpath)
					print()
			elif (extension=="fits") or (extension=="fit") and (self.type=="multi"):
				atmos = read_inverted_atmosphere(self.fpath, atm_range)
			else:
				print("--> Error in atmos.Atmosphere()")
				print(f"    Unsupported extension '{extension}' of atmosphere file.")
				print("    Supported extensions are: .dat, .txt, .fit(s)")
				sys.exit()
			
			self.shape = atmos.data.shape
			self.nx, self.ny, self.npar, self.nz = self.shape
			self.data = atmos.data
			self.logtau = atmos.logtau
			self.nodes = atmos.nodes
			self.values = atmos.values
			self.mask = atmos.mask
			self.header = atmos.header
			self.height = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
			self.rho = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
			self.chi_c = atmos.chi_c
			self.nHtot = np.sum(atmos.data[:,:,8:,:], axis=-2)
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
				self.nHtot = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
				if nz is None:
					self.data[:,:,0,:] = self.logtau
				self.idx_meshgrid, self.idy_meshgrid = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
				self.idx_meshgrid = self.idx_meshgrid.flatten()
				self.idy_meshgrid = self.idy_meshgrid.flatten()
				self.ids_tuple = list(zip(self.idx_meshgrid, self.idy_meshgrid))
			else:
				self.data = None

	def __deepcopy__(self, memo):
		new = Atmosphere()
		new.data = copy.deepcopy(self.data)
		new.logtau = copy.deepcopy(self.logtau)
		new.nx = copy.deepcopy(self.nx)
		new.ny = copy.deepcopy(self.ny)
		new.npar = copy.deepcopy(self.npar)
		new.nz = copy.deepcopy(self.nz)
		new.atm_name_list = copy.deepcopy(self.atm_name_list)
		new.nodes = copy.deepcopy(self.nodes)
		new.values = copy.deepcopy(self.values)
		new.mode = copy.deepcopy(self.mode)
		new.norm = copy.deepcopy(self.norm)
		if self.norm:
			new.icont = copy.deepcopy(self.icont)
		# if (globin.of_mode) and (globin.mode>=1):
		try:
			new.of_num = copy.deepcopy(self.of_num)
			new.of_paths = copy.deepcopy(self.of_paths)
		except:
			pass
		try:
			new.do_fudge = copy.deepcopy(self.do_fudge)
			new.fudge_lam = copy.deepcopy(self.fudge_lam)
			new.fudge = copy.deepcopy(self.fudge)
			new.of_scatter = copy.deepcopy(self.of_scatter)
			new.wavelength_vacuum = copy.deepcopy(self.wavelength_vacuum)
		except:
			pass
		try:
			new.wavelength = copy.deepcopy(self.wavelength)
		except:
			pass
		new.global_pars = copy.deepcopy(self.global_pars)
		new.line_no = copy.deepcopy(self.line_no)
		new.par_id = copy.deepcopy(self.par_id)
		new.vmac = copy.deepcopy(self.vmac)
		new.line_lists_path = copy.deepcopy(self.line_lists_path)
		new.xmin = copy.deepcopy(self.xmin)
		new.xmax = copy.deepcopy(self.xmax)
		new.ymin = copy.deepcopy(self.ymin)
		new.ymax = copy.deepcopy(self.ymax)
		new.ids_tuple = copy.deepcopy(self.ids_tuple)
		new.idx_meshgrid = copy.deepcopy(self.idx_meshgrid)
		new.idy_meshgrid = copy.deepcopy(self.idy_meshgrid)
		try:
			new.RH = copy.deepcopy(self.RH)
		except:
			pass
		try:
			new.n_thread = copy.deepcopy(self.n_thread)
			new.interp_degree = copy.deepcopy(self.interp_degree)
			new.temp_tck = copy.deepcopy(self.temp_tck)
		except:
			pass
		try:
			new.n_local_pars = copy.deepcopy(self.n_local_pars)
			new.n_global_pars = copy.deepcopy(self.n_global_pars)
		except:
			pass

		return new

	def __str__(self):
		return "<Atmosphere: fpath = {}, (nx,ny,npar,nz) = ({},{},{},{})>".format(self.fpath, self.nx, self.ny, self.npar, self.nz)

	def get_atmos(self, idx, idy):
		# return atmosphere from cube with given indices 'idx' and 'idy'.
		try:
			return self.data[idx,idy]
		except:
			print(f"Error: Atmosphere index notation does not match")
			print(f"       it's dimension. No atmosphere (x,y) = ({idx},{idy})\n")
			sys.exit()

	def write_atmosphere(self):
		atmos = [self]*(self.nx*self.ny)
		args = zip(atmos, globin.idx, globin.idy)
		globin.pool.map(func=globin.pool_write_atmosphere, iterable=args)

	def build_from_nodes(self, flag, params=None):
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

		atmos = [self]*(self.nx*self.ny)
		args = zip(atmos, flag[self.idx_meshgrid, self.idy_meshgrid], self.idx_meshgrid, self.idy_meshgrid)

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
		atmos, flag, idx, idy = args

		if flag==0:
			return atmos.data[idx,idy]

		for parameter in atmos.nodes:
			# skip over OF and stray light parameters
			if parameter=="stray" or parameter=="of":
				continue

			# K0, Kn by default; True for vmic, mag, gamma and chi
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
				# bottom node slope for extrapolation based on temperature gradient from FAL C model
				Kn = splev(x[-1], self.temp_tck, der=1)
				# Kn = (y[-1] - y[-2]) / (x[-1] - x[-2])
			elif (parameter=="gamma") or (parameter=="chi") or (parameter=="vz") or (parameter=="vmic") or (parameter=="mag"):
				if len(x)>=2:
					K0 = (y[1]-y[0]) / (x[1]-x[0])
					Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
					# check if extrapolation at the top atmosphere point goes below the minimum
					# if does, change the slopte so that at top point we have parameter_min (globin.limit_values[parameter][0])
					if self.limit_values[parameter][0]>(y[0] + K0 * (atmos.logtau[0]-x[0])):
						K0 = (self.limit_values[parameter][0] - y[0]) / (atmos.logtau[0] - x[0])
					elif self.limit_values[parameter][1]<(y[0] + K0 * (atmos.logtau[0]-x[0])):
						K0 = (self.limit_values[parameter][1] - y[0]) / (atmos.logtau[0] - x[0])
					# similar for the bottom for maximum/min values
					if self.limit_values[parameter][1]<(y[-1] + Kn * (atmos.logtau[-1]-x[-1])):
						Kn = (self.limit_values[parameter][1] - y[-1]) / (atmos.logtau[-1] - x[-1])
					elif self.limit_values[parameter][0]>(y[-1] + Kn * (atmos.logtau[-1]-x[-1])):
						Kn = (self.limit_values[parameter][0] - y[-1]) / (atmos.logtau[-1] - x[-1])

			y_new = bezier_spline(x, y, atmos.logtau, K0=K0, Kn=Kn, degree=atmos.interp_degree)
			atmos.data[idx,idy,atmos.par_id[parameter],:] = y_new

		return atmos.data[idx,idy]

	def makeHSE(self, flag):
		indx, indy = np.where(flag==1)
		args = zip(indx, indy)

		with mp.Pool(self.n_thread) as pool:
			results = pool.map(func=self._makeHSE, iterable=args)

		results = np.array(results)

		self.data[indx,indy,2] = results[:,0,:]
		self.data[indx,indy,8:] = results[:,1:7,:]
		self.rho[indx,indy] = results[:,7,:]

	def _makeHSE(self, arg):
		fudge_num = 2
		fudge_lam = np.linspace(401.5, 401.7, num=fudge_num, dtype=np.float64)
		fudge = np.ones((3, fudge_num), dtype=np.float64)

		idx, idy = arg
		
		ne, nH, nHtot, rho = self.RH.hse(0, self.pg_top,
												 self.data[idx, idy, 0], self.data[idx, idy, 1], 
												 self.data[idx, idy, 2],
			                   self.data[idx, idy, 3], self.data[idx, idy, 4],
			                   self.data[idx, idy, 5]/1e4, self.data[idx, idy, 6], self.data[idx, idy, 7],
			                   self.data[idx, idy, 8:], self.nHtot[idx,idy], 0, fudge_lam, fudge)
		
		result = np.vstack((ne/1e6, nH/1e6, rho/1e3))
		return result

	def makeHSE_old(self):
		for idx in range(self.nx):
			for idy in range(self.ny):
				pg, pe, _, rho = makeHSE(5000, self.logtau, self.data[idx,idy,1], self.pg_top*10)

				self.data[idx,idy,2] = pe/10/globin.K_BOLTZMAN/self.data[idx,idy,1] / 1e6
				self.data[idx,idy,8:] = distribute_hydrogen(self.data[idx,idy,1], pg, pe)
				self.rho[idx,idy] = rho

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

		# primary.header["VMAC"] = ("{:5.3f}".format(self.vmac[0]), "macro-turbulen velocity")
		# if "vmac" in self.global_pars:
		# 	primary.header["VMAC_FIT"] = ("TRUE", "flag for fitting macro velocity")
		# else:
		# 	primary.header["VMAC_FIT"] = ("FALSE", "flag for fitting macro velocity")

		# add keys from kwargs (as dict)
		if kwargs:
			for key in kwargs:
				primary.header[key] = kwargs[key]

		hdulist = fits.HDUList([primary])

		for parameter in self.nodes:
			matrix = np.ones((2, self.nx, self.ny, len(self.nodes[parameter])))
			matrix[0] = self.nodes[parameter]
			matrix[1] = self.values[parameter]

			# if parameter=="gamma":
				# matrix[1] = 2*np.arctan(matrix[1])
				# matrix[1] = np.arccos(matrix[1])
			# if parameter=="chi":
				# matrix[1] = 4*np.arctan(matrix[1])
				# matrix[1] = np.arccos(matrix[1])

			par_hdu = fits.ImageHDU(matrix)
			par_hdu.name = parameter

			# par_hdu.header["unit"] = globin.parameter_unit[parameter]
			par_hdu.header.comments["NAXIS1"] = "number of nodes"
			par_hdu.header.comments["NAXIS2"] = "y-axis atmospheres"
			par_hdu.header.comments["NAXIS3"] = "x-axis atmospheres"
			par_hdu.header.comments["NAXIS4"] = "1 - node values | 2 - parameter values"

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
			for parameter in self.values:
				low_ind = up_ind
				up_ind += len(self.nodes[parameter])
				step = proposed_steps[:,:,low_ind:up_ind] / self.parameter_scale[parameter]
				self.values[parameter] += step

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
			for parameter in self.values:
				low_ind = up_ind
				up_ind += len(self.nodes[parameter])
				self.values[parameter] += local_pars[..., low_ind:up_ind] / self.parameter_scale[parameter]

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

	def compute_spectra(self, synthesize):
		indx, indy = np.where(synthesize==1)
		args = zip(indx, indy)
		
		with mp.Pool(self.n_thread) as pool:
			spectra_list = pool.map(func=self._compute_spectra_sequential, iterable=args)

		spectra_list = np.array(spectra_list)
		natm, ns, nw = spectra_list.shape
		spectra_list = spectra_list.reshape(natm, ns, nw, order="C")
		spectra_list = np.swapaxes(spectra_list, 1, 2)

		spectra = Spectrum(self.nx, self.ny, len(self.wavelength_vacuum), nz=self.nz)
		spectra.wavelength = self.wavelength_vacuum
		spectra.spec[indx,indy] = spectra_list

		if self.norm:
			if self.norm_level=="hsra":
				spectra.spec /= self.icont
			if self.norm_level==1:
				spectra.spec /= spectra.spec[:,:,0,0]

		return spectra

	def _compute_spectra_sequential(self, args):
		# start = time.time()
		idx, idy = args

		if (self.line_no["loggf"].size>0) or (self.line_no["dlam"].size>0):
			if self.mode==2:
				_idx, _idy = idx, idy
			elif self.mode==3:
				_idx, _idy = 0, 0
			spec = self.RH.compute1d(self.mu, 0, self.data[idx, idy, 0], 
									self.data[idx, idy, 1], 
								  self.data[idx, idy, 2], self.data[idx, idy, 3], 
								  self.data[idx, idy, 4], self.data[idx, idy, 5]/1e4, 
								  self.data[idx, idy, 6], self.data[idx, idy, 7],
								  self.data[idx, idy, 8:], self.wavelength_vacuum,
								  self.do_fudge, self.fudge_lam, self.fudge[idx,idy],
								  self.line_no["loggf"], self.global_pars["loggf"][_idx, _idy],
								  self.line_no["dlam"], self.global_pars["dlam"][_idx, _idy]/1e4)
		else:
			spec = self.RH.compute1d(self.mu, 0, self.data[idx, idy, 0], 
									self.data[idx, idy, 1], 
								  self.data[idx, idy, 2], self.data[idx, idy, 3], 
								  self.data[idx, idy, 4], self.data[idx, idy, 5]/1e4, 
								  self.data[idx, idy, 6], self.data[idx, idy, 7],
								  self.data[idx, idy, 8:], self.wavelength_vacuum,
								  self.do_fudge, self.fudge_lam, self.fudge[idx,idy],
								  self.line_no["loggf"], self.global_pars["loggf"],
								  self.line_no["dlam"], self.global_pars["dlam"]/1e4)
		
		# print(f"Finished [{idx},{idy}] in ", time.time() - start)

		return spec.I, spec.Q, spec.U, spec.V

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
		# print("  Get parameter RF...")
		if self.hydrostatic:
			# print("    Set the atmosphere in HSE...")
			self.makeHSE(synthesize)
		# print("    Compute spectra...")
		spec = self.compute_spectra(synthesize)
		Nw = spec.nw

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
				#--- positive perturbation
				self.values[parameter][:,:,nodeID] += perturbation
				if parameter=="of":
					self.make_OF_table(self.wavelength_vacuum)
				elif parameter=="stray":
					pass
				else:
					self.build_from_nodes(synthesize)
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
						self.build_from_nodes(synthesize)
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
					self.build_from_nodes(synthesize)

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
			if self.vmac!=0:
				kernel = spec.get_kernel(self.vmac, order=0)
				# self.rf = broaden_rfs(self.rf, kernel, synthesize, skip_par, self.n_thread)

		#--- add the stray light component:
		if self.add_stray_light:
			for idx in range(self.nx):
				for idy in range(self.ny):
					if self.stray_mode==1 or self.stray_mode==2:
						stray_factor = self.stray_light[idx,idy]
					if self.stray_mode==3:
						stray_factor = self.global_pars["stray"]
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
			# self.rf = broaden_rfs(self.rf, instrumental_profile, synthesize, -1, self.n_thread)

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
			print(idp, delta, rmsd)
		print("--------------------------------------")

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

	ne = pe/10 / globin.K_BOLTZMAN/temp
	C1 = ne/2 * globin.PLANCK**3 / (2*np.pi*globin.ELECTRON_MASS*globin.K_BOLTZMAN*temp)**(3/2)
	
	nH = (pg-pe)/10 / globin.K_BOLTZMAN / temp / np.sum(10**(globin.abundance-12)) / 1e6

	pops = np.zeros((6, len(temp)))

	suma = np.ones(len(temp))
	fact = np.ones((6, len(temp)))
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

def extract_spectra_and_atmospheres(lista, Nx, Ny, Nz):
	Nw = len(globin.wavelength)
	spectra = globin.Spectrum(Nx, Ny, Nw)
	spectra.noise = globin.noise
	spectra.wavelength = globin.wavelength
	spectra.lmin = globin.lmin
	spectra.lmax = globin.lmax
	spectra.step = globin.step

	# atmospheres = copy.deepcopy(globin.atm)
	atmospheres = globin.Atmosphere(nx=Nx, ny=Ny, nz=Nz)
	atmospheres.height = np.zeros((Nx, Ny, Nz))
	atmospheres.cmass = np.zeros((Nx, Ny, Nz))

	# this will work only for LTE synthesis...?
	# in NLTE we have an active wavelengths that can be
	# on eaither side of 500nm. Smarter way needed for this...
	if globin.lmin>500:
		ind_min, ind_max = 1, None
	if globin.lmax<500:
		ind_min, ind_max = 0, -1

	if globin.stokes_mode=="NO_STOKES":
		chi_c_shape = list(lista[0]["rh_obj"].chi_c.shape)
		chi_c_shape[0] -= 1 # we have one less wavelenght that we are not taking inot account (@500nm)
		atmospheres.chi_c = np.zeros((Nx,Ny,*chi_c_shape))

	for item in lista:
		if item is not None:
			rh_obj, idx, idy = item.values()

			# ind_min = np.argmin(abs(rh_obj.wave - globin.lmin))
			# ind_max = np.argmin(abs(rh_obj.wave - globin.lmax))+1

			# print(globin.wavelength)
			# print(rh_obj.wave)
			# print(ind_min, ind_max)
			# print(rh_obj.wave[ind_min:ind_max])

			# Stokes vector
			spectra.spec[idx,idy,:,0] = rh_obj.int[slice(ind_min,ind_max)]
			# if there is magnetic field, read the Stokes components
			if rh_obj.stokes:
				spectra.spec[idx,idy,:,1] = rh_obj.ray_stokes_Q[ind_min:ind_max]
				spectra.spec[idx,idy,:,2] = rh_obj.ray_stokes_U[ind_min:ind_max]
				spectra.spec[idx,idy,:,3] = rh_obj.ray_stokes_V[ind_min:ind_max]

			# Atmospheres
			# Atmopshere read here is one projected to local reference frame (from rhf1d) and
			# not from solveray!
			atmospheres.data[idx,idy,0] = np.log10(rh_obj.geometry["tau500"])
			atmospheres.data[idx,idy,1] = rh_obj.atmos["T"]
			atmospheres.data[idx,idy,2] = rh_obj.atmos["n_elec"] / 1e6 	# [1/m3 --> 1/cm3]
			atmospheres.data[idx,idy,3] = rh_obj.geometry["vz"] / 1e3  	# [m/s --> km/s]
			atmospheres.data[idx,idy,4] = rh_obj.atmos["vturb"] / 1e3  	# [m/s --> km/s]
			try:
				# magnetic field should be reprojected when we do for mu!=1 (???)
				atmospheres.data[idx,idy,5] = rh_obj.atmos["B"] * 1e4 # [T --> G]
				atmospheres.data[idx,idy,6] = rh_obj.atmos["gamma_B"] #	[rad]
				atmospheres.data[idx,idy,7] = rh_obj.atmos["chi_B"] 	# [rad]
			except:
				pass
			atmospheres.height[idx,idy] = rh_obj.geometry["height"]
			atmospheres.cmass[idx,idy] = rh_obj.geometry["cmass"]
			if globin.stokes_mode=="NO_STOKES":
				atmospheres.chi_c[idx,idy] = rh_obj.chi_c[ind_min:ind_max] + rh_obj.scatt[ind_min:ind_max]
			for i_ in range(rh_obj.atmos['nhydr']):
				atmospheres.data[idx,idy,8+i_] = rh_obj.atmos["nh"][:,i_] / 1e6 # [1/cm3 --> 1/m3]

	# tau = globin.rh.get_tau(atmospheres.height[0,0], 1, atmospheres.chi_c[0,0,-1])
	# tau = np.log10(tau)
	
	# plt.plot(atmospheres.height[0,0]/1e3, tau, label="average chi")
	
	# from scipy.integrate import simps

	# tau = np.zeros(len(rh_obj.atmos["T"]))
	# tau_local = np.zeros(len(rh_obj.atmos["T"]))
	# for idz in range(1, len(tau)):
	# 	tau[idz] = simps(-atmospheres.chi_c[0,0,-1,:idz], atmospheres.height[0,0,:idz])
	# 	dh = atmospheres.height[0,0,idz-1] - atmospheres.height[0,0,idz]
	# 	tau_local[idz] = tau_local[idz-1] + atmospheres.chi_c[0,0,-1,idz] * dh

	# plt.plot(atmospheres.height[0,0]/1e3, np.log10(tau_local), label="local chi")
	# plt.plot(atmospheres.height[0,0]/1e3, np.log10(tau), label="proper integral")

	# plt.xlabel("Height [km]")
	# plt.ylabel(r"$\log\tau$")

	# plt.legend()
	# plt.show()

	# sys.exit()

	spectra.wave = rh_obj.wave

	return spectra, atmospheres

def compute_spectra(atmos):
	"""
	Function which computes spectrum from input atmosphere. It will distribute
	the calculation to number of threads given in 'init' and store the spectrum
	in file 'spectra.fits'.

	For each run we make log files which are stored in 'logs' directory.

	Parameters:
	---------------
	init : Input object
		Input class object. It must contain number of threads 'n_thread', list
		of sliced atmospheres from cube 'atm_name_list' and name of the output
		spectrum 'rh_spec_name' read from 'keyword.input' file. Rest are dimension
		of the cube ('nx' and 'ny') which are initiated from reading atmosphere
		file. Also, when reading input we set Pool object for multi thread
		claculation of spectra.
	atmos : Atmosphere object
		Atmosphere object for which we compute spectra.
	clean_dirs : bool (optional)
		Flag which controls if we should delete wroking directories after
		finishing the synthesis. Default value is False.

	Returns:
	---------------
	spec_cube : ndarray
		Spectral cube with dimensions (nx, ny, nlam, 5) which stores
		the wavelength and Stokes vector for each pixel in atmosphere cube.
	"""
	if len(atmos.atm_name_list)==0:
		print("Empty list of atmosphere names.\n")
		globin.remove_dirs()
		sys.exit()

	if globin.of_mode:
		if globin.mode<=0 or globin.mode==1 or globin.mode==3:
			args = [ [atm_name, atmos.line_lists_path[0], of_path] for atm_name, of_path in zip(atmos.atm_name_list, atmos.of_paths)]
		elif globin.mode==2:
			args = [ [atm_name, line_list_path, of_path] for atm_name, line_list_path, of_path in zip(atmos.atm_name_list, atmos.line_lists_path, atmos.of_paths)]
	else:
		if globin.mode<=0 or globin.mode==1 or globin.mode==3:
			args = [ [atm_name, atmos.line_lists_path[0]] for atm_name in atmos.atm_name_list]
		elif globin.mode==2:
			args = [ [atm_name, line_list_path] for atm_name, line_list_path in zip(atmos.atm_name_list, atmos.line_lists_path)]
	# else:
	# 	print("--> Error in compute_spectra()")
	# 	print("    We can not make a list of arguments for computing spectra.")
	# 	globin.remove_dirs()
	# 	sys.exit()

	#--- make directory in which we will save logs of running RH
	if not os.path.exists(f"{globin.cwd}/runs/{globin.wd}/logs"):
		os.mkdir(f"{globin.cwd}/runs/{globin.wd}/logs")
	else:
		sp.run(f"rm {globin.cwd}/runs/{globin.wd}/logs/*",
			shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)

	#--- distribute the process to threads
	rh_obj_list = globin.pool.map(func=globin.pool_synth, iterable=args)

	#--- exit if all spectra returned from child process are None (failed synthesis)
	kill = True
	for item in rh_obj_list:
		kill = kill and (item is None)
		if not kill:
			break
	if kill:
		print("--> Spectrum synthesis on all pixels have failed!\n")
		# globin.remove_dirs()
		sys.exit()

	#--- extract data cubes of spectra and atmospheres from finished synthesis
	spectra, atmospheres = extract_spectra_and_atmospheres(rh_obj_list, atmos.nx, atmos.ny, atmos.nz)
	spectra.mean_spectrum()
	spectra.norm()

	return spectra, atmospheres

def RH_compute_RF(atmos, par_flag, rh_spec_name, wavelength):
	compute_spectra(atmos, rh_spec_name, wavelength)
	
	lmin, lmax = wavelength[0], wavelength[-1]

	#--- arguments for 'pool_rf' function
	args = [[name,rh_spec_name] for name in atmos.atm_name_list]

	#--- make directory in which we will save logs of running 'rf_ray'
	if not os.path.exists(f"{globin.cwd}/runs/{globin.wd}/logs"):
		os.mkdir(f"{globin.cwd}/runs/{globin.wd}/logs")
	else:
		sp.run(f"rm {globin.cwd}/runs/{globin.wd}/logs/*",
			shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)

	# rf.shape = (nx, ny, np=6, nz, nw, ns=4)
	rf = np.zeros((atmos.nx, atmos.ny, 6, atmos.nz, len(wavelength), 4))

	"""
	Rewrite 'keyword.input' file to compute appropriate RFs for
	parameters of inversion.
	"""

	keyword_path = f"runs/{globin.wd}/{globin.rh_input_name}"
	for i_, par in enumerate(atmos.nodes):
		if par_flag[i_]:
			rfID = rf_id[par]
			key = "RF_" + par.upper()
			# broken now
			globin.keyword_input = globin.utils._set_keyword(globin.keyword_input, key, "TRUE", keyword_path)
			
			pars = ["temp", "vz", "vmic", "mag", "gamma", "chi"]
			pars.remove(par)
			for item in pars:
				key = "RF_" + item.upper()
				# broken
				globin.keyword_input = globin.utils._set_keyword(globin.keyword_input, key, "FALSE", keyword_path)

			rf_list = globin.pool.map(func=globin.pool_rf, iterable=args)

			for item in rf_list:
				if item is not None:
					wave = item["wave"]
					
					#--- indices of min and max values for wavelength
					ind_min = np.argmin(np.abs(wave-lmin))
					ind_max = np.argmin(np.abs(wave-lmax))+1
					
					idx, idy = int(item["idx"]), int(item["idy"])
					aux = item["rf"].reshape(6, atmos.nz, len(wave), 4)[:, :, ind_min:ind_max, :]
					rf[idx,idy,rfID] = aux[rfID]

			print("Done RF for ", par)

	return rf

def compute_full_rf(local_params=["temp", "vz", "mag", "gamma", "chi"], global_params=["vmac"], fpath=None):
	if (local_params is None) and (global_params is None):
		raise ValueError("None of atmospheric or atomic are given for RF computation.")

	# set reference atmosphere
	atmos = copy.deepcopy(globin.atm)
	atmos.write_atmosphere()
	atmos.line_lists_path = globin.atm.line_lists_path

	atmos.line_no = globin.atm.line_no
	atmos.global_pars = globin.atm.global_pars

	#--- copy current atmosphere to new model atmosphere with +/- perturbation
	model_plus = copy.deepcopy(atmos)
	model_minus = copy.deepcopy(atmos)
	dlogtau = atmos.logtau[1] - atmos.logtau[0]

	if global_params is not None:
		n_global = 0
		for parameter in global_params:
			n_global += atmos.line_no[parameter].size
	else:
		n_global = 0
	if local_params is not None:
		n_local = len(local_params)
	else:
		n_local = 0
	n_pars = n_local + n_global
	
	rf = np.zeros((atmos.nx, atmos.ny, n_pars, atmos.nz, len(globin.wavelength), 4), dtype=np.float64)

	i_ = -1
	if local_params is not None:
		for i_, parameter in tqdm(enumerate(local_params)):
			print(parameter)
			perturbation = globin.delta[parameter]
			parID = atmos.par_id[parameter]

			for zID in tqdm(range(atmos.nz)):
				model_plus.data[:,:,parID,zID] += perturbation
				model_plus.write_atmosphere()
				spec_plus,_ = compute_spectra(model_plus)
				if not globin.mean:
					spec_plus.broaden_spectra(atmos.vmac)

				model_minus.data[:,:,parID,zID] -= perturbation
				model_minus.write_atmosphere()
				spec_minus,_ = compute_spectra(model_minus)
				if not globin.mean:
					spec_minus.broaden_spectra(atmos.vmac)

				diff = spec_plus.spec - spec_minus.spec

				rf[:,:,i_,zID,:,:] = diff / 2 / perturbation

				# remove perturbation from data
				model_plus.data[:,:,parID,zID] -= perturbation
				model_minus.data[:,:,parID,zID] += perturbation

	free_par_ID = i_+1

	#--- loop through global parameters and calculate RFs
	if global_params is not None:
		for parameter in global_params:
			print(parameter)
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
			elif parameter=="loggf" or parameter=="dlam":
				perturbation = globin.delta[parameter]

				for idp in range(atmos.line_no[parameter].size):
					line_no = atmos.line_no[parameter][idp]
					values = copy.deepcopy(atmos.global_pars[parameter][:,:,idp])

					# positive perturbation
					values += perturbation
					# write atomic parameters in files
					if globin.mode==2:
						for idx in range(atmos.nx):
							for idy in range(atmos.ny):	
								globin.write_line_par(atmos.line_lists_path[idx*atmos.ny + idy],
													values[idx,idy], line_no, parameter)
					elif globin.mode==3:
						globin.write_line_par(atmos.line_lists_path[0], values[0,0], line_no, parameter)
					
					spec_plus,_ = compute_spectra(atmos)
					if not globin.mean:
						spec_plus.broaden_spectra(atmos.vmac)

					# negative perturbation
					values -= 2*perturbation
					if globin.mode==2:
						for idx in range(atmos.nx):
							for idy in range(atmos.ny):	
								globin.write_line_par(atmos.line_lists_path[idx*atmos.ny + idy],
													values[idx,idy], line_no, parameter)
					elif globin.mode==3:
						globin.write_line_par(atmos.line_lists_path[0], values[0,0], line_no, parameter)
					
					spec_minus,_ = compute_spectra(atmos)
					if not globin.mean:
						spec_minus.broaden_spectra(atmos.vmac)

					diff = (spec_plus.spec - spec_minus.spec) / 2 / perturbation

					# if globin.mode==2:
					# 	scale = np.sqrt(np.sum(diff**2, axis=(2,3)))
					# elif globin.mode==3:
					# 	scale = np.sqrt(np.sum(diff**2))
					# globin.parameter_scale[parameter][...,idp] = scale

					if globin.mode==2:
						for idx in range(atmos.nx):
							for idy in range(atmos.ny):
								rf[idx,idy,free_par_ID] = diff[idx,idy] # / globin.parameter_scale[parameter][idx,idy,idp]
						free_par_ID += 1
					elif globin.mode==3:
						rf[:,:,free_par_ID,:,:] = np.repeat(diff[:,:,np.newaxis,:,:], atmos.nz , axis=2) # / globin.parameter_scale[parameter][0,0,idp]
						free_par_ID += 1
					
					# return perturbation back
					values += perturbation
					if globin.mode==2:
						for idx in range(atmos.nx):
							for idy in range(atmos.ny):	
								globin.write_line_par(atmos.line_lists_path[idx*atmos.ny + idy],
													values[idx,idy], line_no, parameter)
					elif globin.mode==3:
						globin.write_line_par(atmos.line_lists_path[0], values[0,0], line_no, parameter)

			else:
				print(f"Parameter {parameter} not yet Supported.\n")

	if fpath is not None:
		primary = fits.PrimaryHDU(rf)
		primary.name = "RF"

		np.zeros((atmos.nx, atmos.ny, n_pars, atmos.nz, len(globin.wavelength), 4), dtype=np.float64)

		primary.header.comments["NAXIS1"] = "stokes components"
		primary.header.comments["NAXIS2"] = "number of wavelengths"
		primary.header.comments["NAXIS3"] = "depth points"
		primary.header.comments["NAXIS4"] = "number of parameters"
		primary.header.comments["NAXIS5"] = "y-axis atmospheres"
		primary.header.comments["NAXIS6"] = "x-axis atmospheres"

		primary.header["STOKESV"] = ("IQUV", "ordering of Stokes vector")

		i_ = 1
		if local_params:
			for par in local_params:
				primary.header[f"PAR{i_}"] = (par, "parameter")
				primary.header[f"PARID{i_}"] = (i_, "parameter ID")
				i_ += 1
		if global_params:
			for par in global_params:
				primary.header[f"PAR{i_}"] = (par, "parameter")
				primary.header[f"PARID{i_}"] = (i_, "parameter ID")
				i_ += 1

		hdulist = fits.HDUList([primary])

		#--- wavelength list
		par_hdu = fits.ImageHDU(globin.wavelength)
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

def spinor2multi(atmos_data, do_HSE=False, nproc=1):
	"""
	Routine for converting 'atmos_data' from SPINOR to MULTI type.

	Parameters:
	-----------
	atmos_data : ndarray
		atmosphere data to be converted. Can have dimensions of (npar, nz) or
		(nx, ny, npar, nz).
	do_HSE : bool (optional)
		flag which defines if we are seting H pops from temperature of input atmosphere.
		Otherwise, we use Pe and Pg from input atmosphere (default).
	nproc : int (optional)
		number of proceses to be used for conversion. Default is 1.

	Return:
	-------
	atmos : globin.atmos.Atmosphere()
	"""
	dim = atmos_data.ndim
	if dim==2:
		_, nz = atmos_data.shape
		nx, ny = 1, 1
		idx, idy = [0], [0]
		atmos_data = [atmos_data]
	elif dim==4:
		nx, ny, _, nz = atmos_data.shape
		idx, idy = np.meshgrid(np.arange(nx), np.arange(ny))
		idx = idx.flatten()
		idy = idy.flatten()
		atmos_data = [atmos_data[IDx,IDy] for IDx,IDy in zip(idx,idy)]
	else:
		print("--> Error: unknown number of dimensions for atmosphere")
		print("    to be converted from SPINOR to MULTI format.\n")
		sys.exit()

	args = zip([do_HSE]*(nx*ny), atmos_data)

	print("Converting atmosphere...")
	try:
		items = globin.pool.map(func=globin.pool_spinor2multi, iterable=args)
	except:
		import multiprocessing as mp
		with mp.Pool(nproc) as pool:
			items = pool.map(func=pool_spinor2multi, iterable=args)
	print("Done!")

	atmos = Atmosphere(nx=nx, ny=ny, nz=nz)
	for i_, in_data in enumerate(items):
		IDx, IDy = idx[i_], idy[i_]
		atmos.data[IDx,IDy] = in_data
	atmos.logtau = atmos.data[0,0,0]

	return atmos

def sir2multi(atmos_data):
	pass

def multi2sir(atmos, fpath):
	from scipy.constants import k as kb

	nx, ny, npar, nz = atmos.shape
	new = np.zeros((nx, ny, 11, nz))
	
	for idx in range(atmos.nx):
		for idy in range(atmos.ny):
			new[idx,idy,0] = atmos.data[idx,idy,0] 			# log(tau) @ 500nm
			new[idx,idy,1] = atmos.data[idx,idy,1] 			# T [K]
			new[idx,idy,2] = atmos.data[idx,idy,2]*1e6 	# ne [1/m3]
			new[idx,idy,2] *= kb * new[idx,idy,1] * 10	# ne [dyn/cm2]
			new[idx,idy,3] = atmos.data[idx,idy,4]*1e5	# vmic [cm/s]
			new[idx,idy,4] = atmos.data[idx,idy,5]			# B [G]
			new[idx,idy,5] = atmos.data[idx,idy,3]*1e5	# vz [cm/s]
			new[idx,idy,6] = atmos.data[idx,idy,6]			# gamma [rad]
			new[idx,idy,6] *= 180/np.pi									# gamma [deg]
			new[idx,idy,7] = atmos.data[idx,idy,7]			# gamma [rad]
			new[idx,idy,7] *= 180/np.pi									# gamma [deg]


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
	# spinor_atmosphere[:,:,:,9] = multi_atmosphere[:,:,5,:]
	# spinor_atmosphere[:,:,:,10] = multi_atmosphere[:,:,6,:]
	# spinor_atmosphere[:,:,:,11] = multi_atmosphere[:,:,7,:]

	if fname:
		np.savetxt(fname, spinor_atmosphere[0,0], header=" {:2d}  1.0 ".format(nz))

	# if fname:
	# 	primary = fits.PrimaryHDU(spinor_atmosphere)
	# 	primary.header["LTTOP"] = min(spinor_atmosphere[0,0,0])
	# 	primary.header["LTBOT"] = max(spinor_atmosphere[0,0,0])
	# 	primary.header["LTINC"] = np.round(spinor_atmosphere[0,0,0,1] - spinor_atmosphere[0,0,0,0], decimals=2)
	# 	primary.header["LOGTAU"] = (1, "optical depth scale")
	# 	primary.header["HEIGHT"] = (2, "height scale (cm)")
	# 	primary.header["TEMP"] = (3, "temperature (K)")
	# 	primary.header["PGAS"] = (4, "gass pressure (dyn/cm2)")
	# 	primary.header["PEL"] = (5, "electron pressure (dyn/cm2)")
	# 	primary.header["KAPPA"] = (6, "mass density (g/cm2)")
	# 	primary.header["DENS"] = (7, "density (g/cm3)")
	# 	primary.header["VMIC"] = (8, "micro-turbulent velocity (cm/s)")
	# 	primary.header["VZ"] = (9, "vertical velocity (cm/s)")
	# 	primary.header["MAG"] = (10, "magnetic field strength (G)")
	# 	primary.header["GAMMA"] = (11, "magnetic field inclination (rad)")
	# 	primary.header["CHI"] = (12, "magnetic field azimuth (rad)")

	# 	primary.writeto(fname, overwrite=True)

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
		file path to inverted atmosphere. It should be .fits file.
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

	for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi", "of", "stray"]:
		try:
			ind = hdu_list.index_of(parameter)
			data = hdu_list[ind].data[:, xmin:xmax, ymin:ymax, :]
			_, nx, ny, nnodes = data.shape

			atmos.nodes[parameter] = data[0,0,0]
			# angles are saved in radians, no need to convert them here
			if parameter=="gamma":
				atmos.values[parameter] = data[1]
			elif parameter=="chi":
				atmos.values[parameter] = data[1]
			else:
				atmos.values[parameter] = data[1]
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

def read_multi(fpath):
	"""
	Read MULTI type atmosphere data and store it in
	Atmosphere() object.

	Parameter:
	----------
	fpath : string
		path to the MULTI type atmosphere.

	Return:
	-------
	atmos : globin.atmos.Atmosphere() object
	"""
	lines = open(fpath, "r").readlines()

	# remove commented lines
	lines = [line.rstrip("\n") for line in lines if "*" not in line]

	# get number of depth points
	ndpth = int(lines[3].replace(" ", ""))

	nz = ndpth
	nx, ny = 1, 1

	atmos = Atmosphere(nx=nx, ny=ny, nz=nz)
	atmos.shape = (nx, ny, 14, nz)

	for i_ in range(ndpth):
		# read first part of the atmosphere
		lista = list(filter(None,lines[4+i_].split(" ")))
		atmos.data[0,0,0,i_], \
		atmos.data[0,0,1,i_], \
		atmos.data[0,0,2,i_], \
		atmos.data[0,0,3,i_], \
		atmos.data[0,0,4,i_] = [float(element) for element in lista]

		# read H populations
		lista = list(filter(None,lines[4+ndpth+i_].split(" ")))
		atmos.data[0,0,8,i_], \
		atmos.data[0,0,9,i_], \
		atmos.data[0,0,10,i_], \
		atmos.data[0,0,11,i_], \
		atmos.data[0,0,12,i_], \
		atmos.data[0,0,13,i_] = [float(element) for element in lista]

	atmos.logtau = atmos.data[0,0,0]

	return atmos

def read_spinor(fpath):
	atmos_data = np.loadtxt(fpath, skiprows=1, dtype=np.float64).T
	# nz = atmos_data.shape[1]
	
	# atmos = globin.Atmosphere(nx=1, ny=1, nz=nz)
	# atmos.logtau = atmos_data[0]
	# atmos.data = atmos_data
	
	atmos = convert_atmosphere(atmos_data[0], atmos_data, "spinor")

	return atmos

def read_sir(fpath):
	from scipy.constants import k as kb

	data = np.loadtxt(fpath, skiprows=1).T
	data = data[:,::-1]
	npar, nz = data.shape

	atmos = Atmosphere(nx=1, ny=1, nz=nz)

	atmos.data[0,0,0] = data[0]			# log(tau) @ 500nm
	atmos.logtau = data[0]
	atmos.data[0,0,1] = data[1]			# T [K]
	atmos.data[0,0,2] = data[2]  		# ne [dyn/cm2]
	atmos.data[0,0,2] *= 1/10/kb/data[1]/1e6
	atmos.data[0,0,3] = data[5]*1e5	# vz [km/s]
	atmos.data[0,0,4] = data[3]*1e5 	# vmic [km/s]
	atmos.data[0,0,5] = data[4]			# B [G]
	atmos.data[0,0,6] = data[6] * np.pi/180 # gamma [rad]
	atmos.data[0,0,7] = data[7] * np.pi/180 # theta [rad]

	atmos.data[0,0,8:] = distribute_hydrogen(data[1], data[-2], data[2])

	return atmos