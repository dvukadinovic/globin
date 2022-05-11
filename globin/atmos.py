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
from scipy.ndimage import correlate1d
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

# from .input import read_multi, read_spinor, read_inverted_atmosphere

# import globin

from .tools import bezier_spline

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
					  "chi"    : 7}

	#--- limit values for atmospheric parameters
	limit_values = {"temp"  : [3000, 10000], 		# [K]
				"vz"    : [-10, 10],			# [km/s]
				"vmic"  : [1e-3, 10],			# [km/s]
				"vmac"  : [0, 5],				# [km/s]
				"mag"   : [1, 10000],			# [G]
				"of"    : [0, 20],
				"gamma" : [-np.pi, np.pi],	# [rad]
				# "gamma" : [-0.999999, 0.999999],
				"chi"   : [-2*np.pi, 2*np.pi]}	# [rad]
				# "chi"   : [-0.999999, 0.999999]}

	#--- parameter perturbations for calculating RFs (must be the same as in rf_ray.c)
	delta = {"temp"  : 1,		# K
			 "vz"    : 1/1e3,	# m/s --> km/s
			 "vmic"  : 1/1e3,	# m/s --> km/s
			 "mag"   : 1,		# G
			 "gamma" : 0.01,	# rad
			 "chi"   : 0.01,	# rad
			 "loggf" : 0.001,	# 
			 "dlam"  : 1,		# mA
			 "of"    : 0.001}

	def __init__(self, fpath=None, atm_type="multi", atm_range=[0,None,0,None], nx=None, ny=None, nz=None, logtau_top=-6, logtau_bot=1, logtau_step=0.1):
		self.fpath = fpath
		self.type = atm_type

		# list of fpath's to line lists (in mode=2)
		self.line_lists_path = []
		# list of fpath's to each atmosphere column
		self.atm_name_list = []
		# parameter scaling for inversino
		self.parameter_scale = {}

		# self.atmosphere.RH = pyrh.RH()
		# self.atmosphere.n_thread = self.n_thread
		# self.atmosphere.temp_tck = self.falc.temp_tck
		# self.atmosphere.interp_degree = self.interp_degree

		# container for spectrum
		self.spectra = None

		# nodes: each atmosphere has the same nodes for given parameter
		self.nodes = {}
		# parameters in nodes: shape = (nx, ny, nnodes)
		self.values = {}
		# node mask: we can specify only which nodes to invert (mask==1)
		# structure and ordering same as self.nodes
		self.mask = {}

		self.chi_c = None
		
		# global parameters: each is given in a list size equal to number of parameters
		self.global_pars = {}
		# line number in list of lines for which we are inverting atomic data
		self.line_no = {}

		self.xmin = atm_range[0]
		self.xmax = atm_range[1]
		self.ymin = atm_range[2]
		self.ymax = atm_range[3]

		self.npar = 14

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
			self.chi_c = atmos.chi_c
		else:
			self.header = None
			if (nx is not None) and (ny is not None) and (nz is not None):
				self.nx, self.ny, self.nz = nx, ny, nz
				self.data = np.zeros((self.nx, self.ny, self.npar, self.nz), dtype=np.float64)
				self.height = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
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
		new.global_pars = copy.deepcopy(self.global_pars)
		new.line_no = copy.deepcopy(self.line_no)
		new.par_id = copy.deepcopy(self.par_id)
		new.vmac = copy.deepcopy(self.vmac)
		new.line_lists_path = copy.deepcopy(self.line_lists_path)
		new.spectra = copy.deepcopy(self.spectra)
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

	def build_from_nodes(self):
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
		args = zip(atmos, self.idx_meshgrid, self.idy_meshgrid)

		with mp.Pool(self.n_thread) as pool:
			atm = pool.map(func=self._build_from_nodes, iterable=args)

		# we need to assign built atmosphere structure to self atmosphere
		# otherwise self.data would be only 0's.
		for idl in range(self.nx*self.ny):
			idx, idy = self.idx_meshgrid[idl], self.idy_meshgrid[idl]
			self.data[idx,idy] = atm[idl].data[idx,idy]

	def _build_from_nodes(self, args):
		atmos, idx, idy = args

		for parameter in atmos.nodes:
			if parameter!="of":
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

		# if globin.hydrostatic:
		# 	# atmos.makeHSE_old(idx, idy)
		# 	atmos.makeHSE(idx, idy)

		return atmos

	def makeHSE(self, idx, idy):
		press, pel, kappa, rho = globin.makeHSE(5000, self.data[idx,idy,0], self.data[idx,idy,1])
		
		# electron density [1/cm3]
		self.data[idx,idy,2] = pel/10/globin.K_BOLTZMAN/self.data[idx,idy,1]/1e6

		# # Hydrogen populations [1/cm3]
		self.data[idx,idy,8:] = distribute_hydrogen(self.data[idx,idy,1], press, pel)
	
	def makeHSE_new(self, idx, idy):
		aux = pyrh.RH(argc, argv)
		pops = aux.setHSE(self.nz)

		self.data[idx,idy,2] = pops.ne / 1e6 # [1/m3 --> 1/cm3] right?
		self.data[idx,idy,8:] = pops.nH / 1e6 # [1/m3 --> 1/cm3] right?

	def interpolate_atmosphere(self, x_new, ref_atm):
		if (x_new[0]<ref_atm[0,0,0,0]) or \
		   (x_new[-1]>ref_atm[0,0,0,-1]):
			# print("--> Warning: atmosphere will be extrapolated")
			# print("    from {} to {} in optical depth.\n".format(ref_atm[0,0,0,0], x_new[0]))
			self.logtau = ref_atm[0,0,0]
			self.data = ref_atm
			return

		self.nz = len(x_new)
		self.nx, self.ny, _, _ = ref_atm.shape

		oneD = (self.nx*self.ny==1)

		self.data = np.zeros((self.nx, self.ny, self.npar, self.nz))
		self.data[:,:,0,:] = x_new
		self.logtau = x_new

		for idx in range(self.nx):
			for idy in range(self.ny):
				for parID in range(1,self.npar):
					if oneD:
						tck = splrep(ref_atm[0,0,0], ref_atm[0,0,parID])
					else:
						tck = splrep(ref_atm[0,0,0], ref_atm[idx,idy,parID])
					self.data[idx,idy,parID] = splev(x_new, tck)
				
	def save_atmosphere(self, fpath="inverted_atmos.fits", kwargs=None):
		# reverting back angles into radians
		data = copy.deepcopy(self.data)
		# if "gamma" in self.nodes:
			# data[:,:,6] = 2*np.arctan(data[:,:,6])
			# data[:,:,6] = np.arccos(data[:,:,6])
		# if "chi" in self.nodes:
			# data[:,:,7] = 4*np.arctan(data[:,:,7])
			# data[:,:,7] = np.arccos(data[:,:,7])

		# wraping the angles into the interval 0-180 degrees and 0-360 degrees
		# data[:,:,6] %= np.pi
		# data[:,:,7] %= 2*np.pi

		primary = fits.PrimaryHDU(data, do_not_scale_image_data=True)
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
		try:
			pars.remove("vmac")
		except:
			pass

		primary = fits.PrimaryHDU()
		hdulist = fits.HDUList([primary])
		
		for parameter in pars:
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

	def update_parameters(self, proposed_steps, stop_flag=None):
		if stop_flag is not None:
			low_ind, up_ind = 0, 0
			# update atmospheric parameters
			for parameter in self.values:
				low_ind = up_ind
				up_ind += len(self.nodes[parameter])
				step = proposed_steps[:,:,low_ind:up_ind] / self.parameter_scale[parameter]
				# we do not perturb parameters of those pixels which converged
				step = np.einsum("...i,...->...i", step, stop_flag)
				# if globin.rf_type=="snapi":
				# 	# RH returns RFs in m/s and we are working with km/s in globin
				# 	# so we have to return the values from m/s to km/s
				# 	if parameter=="vz" or parameter=="vmic":
				# 		step /= 1e3
				np.nan_to_num(step, nan=0.0, copy=False)
				self.values[parameter] += step

			# update atomic parameters + vmac
			for parameter in self.global_pars:
				if self.line_no[parameter].size > 0:
					low_ind = up_ind
					up_ind += self.line_no[parameter].size
					step = proposed_steps[:,:,low_ind:up_ind] / self.parameter_scale[parameter]
					np.nan_to_num(step, nan=0.0, copy=False)
					self.global_pars[parameter] += step
		else:
			low_ind, up_ind = 0, 0
			# update atmospheric parameters
			for idx in range(self.nx):
				for idy in range(self.ny):
					for parameter in self.values:
						low_ind = up_ind
						up_ind += len(self.nodes[parameter])
						step = proposed_steps[low_ind:up_ind] / self.parameter_scale[parameter][idx,idy]
						# RH returns RFs in m/s and we are working with km/s in globin
						# so we have to return the values from m/s to km/s
						# if globin.rf_type=="snapi":
						# 	if parameter=="vz" or parameter=="vmic":
						# 		step /= 1e3
						np.nan_to_num(step, nan=0.0, copy=False)
						self.values[parameter][idx,idy] += step

			# update atomic parameters + vmac
			for parameter in self.global_pars:
				if self.line_no[parameter].size > 0:
					low_ind = up_ind
					up_ind += self.global_pars[parameter].size
					step = proposed_steps[low_ind:up_ind] / self.parameter_scale[parameter]
					np.nan_to_num(step, nan=0.0, copy=False)
					self.global_pars[parameter] += step

	def check_parameter_bounds(self, mode):
		for parameter in self.values:
			# inclination is wrapped around [0, 180] interval
			if parameter=="gamma":
				y = np.cos(self.values[parameter])
				self.values[parameter] = np.arccos(y)
				# gamma = 2*np.arctan(self.values[parameter])
				# gamma %= np.pi
				# self.values[parameter] = np.tan(gamma/2)
				pass
			# azimuth is wrapped around [0, 360] interval
			elif parameter=="chi":
				# chi = 4*np.arctan(self.values[parameter])
				# chi %= 2*np.pi
				# self.values[parameter] = np.tan(chi/4)
				y = np.cos(self.values[parameter])
				self.values[parameter] = np.arccos(y)
				pass
			else:
				for i_ in range(len(self.nodes[parameter])):
					for idx in range(self.nx):
						for idy in range(self.ny):
							# minimum check
							if self.values[parameter][idx,idy,i_]<self.limit_values[parameter][0]:
								self.values[parameter][idx,idy,i_] = self.limit_values[parameter][0]
							# maximum check
							if self.values[parameter][idx,idy,i_]>self.limit_values[parameter][1]:
								self.values[parameter][idx,idy,i_] = self.limit_values[parameter][1]

		for parameter in self.global_pars:
			if parameter=="vmac":
				# minimum check
				if self.global_pars[parameter]<self.limit_values[parameter][0]:
					self.global_pars[parameter] = np.array([self.limit_values[parameter][0]])
				# maximum check
				if self.global_pars[parameter]>self.limit_values[parameter][1]:
					self.global_pars[parameter] = np.array([self.limit_values[parameter][1]])
				self.vmac = self.global_pars["vmac"]
			else:
				if self.line_no[parameter].size > 0:
					if mode==2:
						nx, ny = self.nx, self.ny
					elif mode==3:
						nx, ny = 1,1
					for idx in range(nx):
						for idy in range(ny):
							for i_ in range(self.line_no[parameter].size):
								# minimum check
								if self.global_pars[parameter][idx,idy,i_]<self.limit_values[parameter][i_,0]:
									self.global_pars[parameter][idx,idy,i_] = self.limit_values[parameter][i_,0]
								# maximum check
								if self.global_pars[parameter][idx,idy,i_]>self.limit_values[parameter][i_,1]:
									self.global_pars[parameter][idx,idy,i_] = self.limit_values[parameter][i_,1]

	def smooth_parameters(self, cycleID):
		from scipy.ndimage import gaussian_filter

		stds = [5,3,1]
		std = stds[cycleID]

		for parameter in self.nodes:
			for nodeID in range(len(self.nodes[parameter])):
				self.values[parameter][...,nodeID] = gaussian_filter(self.values[parameter][...,nodeID], std)

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

	def compute_spectra(self, skip=[]):
		args = copy.deepcopy(self.ids_tuple)
		for item in skip:
			args.remove(item)

		with mp.Pool(self.n_thread) as pool:
			spectra = pool.map(func=self._compute_spectra_sequential, iterable=args)

		for i_ in range(len(spectra)):
			idx, idy = spectra[i_]["idx"], spectra[i_]["idy"]
			self.spectra.spec[idx,idy,:,0] = spectra[i_]["spec"].I
			self.spectra.spec[idx,idy,:,1] = spectra[i_]["spec"].Q
			self.spectra.spec[idx,idy,:,2] = spectra[i_]["spec"].U
			self.spectra.spec[idx,idy,:,3] = spectra[i_]["spec"].V

		return self.spectra

	def _compute_spectra_sequential(self, arg):
		# start = time.time()
		idx, idy = arg
		if (self.line_no["loggf"].size>0) or (self.line_no["dlam"].size>0):
			if self.mode==2:
				_idx, _idy = idx, idy
			elif self.mode==3:
				_idx, _idy = 0, 0
			spec = self.RH.compute1d(0, self.data[idx, idy, 0], self.data[idx, idy, 1], 
								  self.data[idx, idy, 2], self.data[idx, idy, 3], 
								  self.data[idx, idy, 4], self.data[idx, idy, 5]/1e4, 
								  self.data[idx, idy, 6], self.data[idx, idy, 7],
								  self.data[idx, idy, 8:], self.do_fudge, self.fudge_lam, self.fudge[idx,idy],
								  self.line_no["loggf"], self.global_pars["loggf"][_idx, _idy],
								  self.line_no["dlam"], self.global_pars["dlam"][_idx, _idy]/1e4)
		else:
			spec = self.RH.compute1d(0, self.data[idx, idy, 0], self.data[idx, idy, 1], 
								  self.data[idx, idy, 2], self.data[idx, idy, 3], 
								  self.data[idx, idy, 4], self.data[idx, idy, 5]/1e4, 
								  self.data[idx, idy, 6], self.data[idx, idy, 7],
								  self.data[idx, idy, 8:], self.do_fudge, self.fudge_lam, self.fudge[idx,idy],
								  self.line_no["loggf"], self.global_pars["loggf"],
								  self.line_no["dlam"], self.global_pars["dlam"]/1e4)
		# print(f"Finished [{idx},{idy}] in ", time.time() - start)

		return {"spec": spec, "idx" : idx, "idy" : idy}

	def compute_rfs(self, rf_noise_scale, weights=1, skip=[], rf_type="node", mean=False, old_rf=None, old_pars=None):
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
		#--- get inversion parameters for atmosphere and interpolate it on finner grid (original)
		self.build_from_nodes()
		spec = self.compute_spectra(skip)
		Nw = spec.nw

		if rf_type=="snapi":	
			# full_rf.shape = (nx, ny, np, nz, nw, 4)
			# check weather we need to recalculate RF for atmospheric parameters;
			# if change in parameters is less than 10 x perturbation we do not compute RF again
			if atmos.n_local_pars!=0:
				pars = list(atmos.nodes.keys())
				par_flag = [True]*len(pars)
				# if old_pars is not None:
				# 	for i_, par in enumerate(atmos.nodes):
				# 		delta = np.abs(atmos.values[par].flatten() - old_pars[par].flatten())
				# 		flag = [False if item<globin.diff[par] else True for item in delta]
				# 		if any(flag):
				# 			par_flag[i_] = True
				# 		else:
				# 			par_flag[i_] = False
				full_rf = RH_compute_RF(atmos, par_flag, globin.rh_spec_name, globin.wavelength)
				for i_,par in enumerate(pars):
					rfID = rf_id[par]
					if not par_flag[i_]:
						full_rf[:,:, rfID] = old_rf[:,:, rfID]
			else:
				full_rf = None
		elif rf_type=="node":
			full_rf = None
		else:
			print("\n  Not proper RF type calculation.")
			sys.exit()

		#--- get total number of parameters (local + global)
		Npar = self.n_local_pars + self.n_global_pars
		
		rf = np.zeros((spec.nx, spec.ny, Npar, Nw, 4), dtype=np.float64)
		node_RF = np.zeros((spec.nx, spec.ny, Nw, 4))

		model_plus = copy.deepcopy(self)
		model_minus = copy.deepcopy(self)

		#--- loop through local (atmospheric) parameters and calculate RFs
		free_par_ID = 0
		for parameter in self.nodes:
			nodes = self.nodes[parameter]
			values = self.values[parameter]
			perturbation = self.delta[parameter]

			for nodeID in range(len(nodes)):
				if rf_type=="snapi":
					#---
					# This is "broken" from the point we introduced key mean for the spectrum.
					#---
					#===--- computing RFs for given parameter (proper way as in SNAPI)
					# positive perturbation
					self.values[parameter][:,:,nodeID] += perturbation
					self.build_from_nodes(False)
					positive = copy.deepcopy(self.data[:,:,parID])

					# negative perturbation
					self.values[parameter][:,:,nodeID] -= 2*perturbation
					self.build_from_nodes(False)
					negative = copy.deepcopy(self.data[:,:,parID])

					# derivative of parameter distribution to node perturbation
					dy_dnode = (positive - negative) / 2 / perturbation

					for idx in range(spec.nx):
						for idy in range(spec.ny):
							node_RF[idx,idy] = np.einsum("i,ijk", dy_dnode[idx,idy], full_rf[idx,idy,rfID])
				elif rf_type=="node":
					#===--- Computing RFs in nodes
					
					model_plus.values[parameter][:,:,nodeID] += perturbation
					if parameter=="of":
						model_plus.make_OF_table(self.wavelength_vacuum)
					else:
						model_plus.build_from_nodes()
					spectra_plus = model_plus.compute_spectra(skip)

					# negative perturbation (except for inclination and azimuth)
					if parameter=="gamma" or parameter=="chi":
						node_RF = (spectra_plus.spec - spec.spec ) / perturbation
					else:
						model_minus.values[parameter][:,:,nodeID] -= perturbation
						if parameter=="of":	
							model_minus.make_OF_table(self.wavelength_vacuum)
						else:
							model_minus.build_from_nodes()
						spectra_minus = model_minus.compute_spectra(skip)

						node_RF = (spectra_plus.spec - spectra_minus.spec ) / 2 / perturbation

				node_RF *= weights
				node_RF /= rf_noise_scale
				scale = np.sqrt(np.sum(node_RF**2, axis=(2,3)))

				for idx in range(spec.nx):
					for idy in range(spec.ny):
						if scale[idx,idy]==0:
							# print("scale==0 for --> ", parameter)
							self.parameter_scale[parameter][idx,idy,nodeID] = 1
						elif not np.isnan(np.sum(scale[idx,idy])):
							self.parameter_scale[parameter][idx,idy,nodeID] = scale[idx,idy]
				
				for idx in range(spec.nx):
					for idy in range(spec.ny):
						rf[idx,idy,free_par_ID] = node_RF[idx,idy] / self.parameter_scale[parameter][idx,idy,nodeID]
				free_par_ID += 1

				if rf_type=="snapi":
					# return back perturbation (SNAPI way)
					self.values[parameter][:,:,nodeID] += perturbation
					self.build_from_nodes()
				elif rf_type=="node":
					# return back perturbations (node way)
					if parameter=="gamma" or parameter=="chi":
						model_plus.values[parameter][:,:,nodeID] = copy.deepcopy(self.values[parameter][:,:,nodeID])
					else:
						model_plus.values[parameter][:,:,nodeID] -= perturbation
						model_minus.values[parameter][:,:,nodeID] += perturbation

		#--- loop through global parameters and calculate RFs
		skip_par = -1
		if self.n_global_pars>0:
			#--- loop through global parameters and calculate RFs
			for parameter in self.global_pars:
				if parameter=="vmac":
					kernel_sigma = spec.get_kernel_sigma(self.vmac)
					radius = int(4*kernel_sigma + 0.5)
					x = np.arange(-radius, radius+1)
					phi = np.exp(-x**2/kernel_sigma**2)
					# normalaizing the profile
					phi *= 1/(np.sqrt(np.pi)*kernel_sigma)
					kernel = phi*(2*x**2/kernel_sigma**2 - 1) * 1 / kernel_sigma / globin.step
					# since we are correlating, we need to reverse the order of data
					kernel = kernel[::-1]

					for idx in range(spec.nx):
						for idy in range(spec.ny):
							for sID in range(4):
								rf[idx,idy,free_par_ID,:,sID] = correlate1d(spec.spec[idx,idy,:,sID], kernel)
								rf[idx,idy,free_par_ID,:,sID] *= kernel_sigma * globin.step / self.global_pars["vmac"]
								rf[idx,idy,free_par_ID,:,sID] *= globin.weights[sID]
								rf[idx,idy,free_par_ID,:,sID] /= rf_noise_scale[idx,idy,:,sID]

					globin.parameter_scale[parameter] = np.sqrt(np.sum(rf[:,:,free_par_ID,:,:]**2))
					rf[:,:,free_par_ID,:,:] /= globin.parameter_scale[parameter]

					skip_par = free_par_ID
					free_par_ID += 1

				elif parameter=="loggf" or parameter=="dlam":
					if self.line_no[parameter].size > 0:
						perturbation = self.delta[parameter]

						for idp in range(self.line_no[parameter].size):
							model_plus.global_pars[parameter][...,idp] += perturbation
							spec_plus = model_plus.compute_spectra(skip)

							model_minus.global_pars[parameter][...,idp] -= perturbation
							spec_minus = model_minus.compute_spectra(skip)

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
										rf[idx,idy,free_par_ID] = diff[idx,idy] / self.parameter_scale[parameter][idx,idy,idp]
							elif self.mode==3:
								rf[:,:,free_par_ID,:,:] = diff / self.parameter_scale[parameter][0,0,idp]
							free_par_ID += 1
							
							model_plus.global_pars[parameter][...,idp] -= perturbation
							model_minus.global_pars[parameter][...,idp] += perturbation

		#--- broaden the spectra
		if not mean:
			spec.broaden_spectra(self.vmac)
			rf = broaden_rfs(rf, self.vmac, skip_par)

		# for idy in range(atmos.ny):
		# 	plt.figure(1)
		# 	for parID in range(4):
		# 		plt.plot(rf[0, idy, parID, :, 0])
		# 	plt.figure(2)
		# 	for parID in range(4,Npar):
		# 		plt.plot(rf[0, idy, parID, :, 0])
		# plt.show()
		# sys.exit()

		# plt.plot(globin.obs.spec[0,0,:,0])
		# plt.plot(spec.spec[0,0,:,0])
		# plt.show()

		#--- compare RFs for single parameter
		# for idx in range(atmos.nx):
		# 	for idy in range(atmos.ny):
		# 		aux = rf[idx,idy, :, :, :]
				# plt.plot(aux[0,:,0], label="T-3")
				# plt.plot(aux[1,:,0], label="T-2")
				# plt.plot(aux[2,:,0], label="T-1")
		# 		plt.plot(aux[3,:,0], label="T0")
		# 		plt.plot(aux[4,:,0], label="OF 1")
		# 		plt.plot(aux[5,:,0], label="OF 2")
		# 		plt.plot(aux[6,:,0], label="OF 3")
		# 		plt.plot(aux[7,:,0], label="OF 4")
		# 		plt.plot(aux[8,:,0], label="OF 5")
		# plt.legend()
		# plt.show()
		# sys.exit()

		return rf, spec, full_rf

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

def write_multi_atmosphere(atm, fpath):
	# write atmosphere 'atm' of MULTI type
	# into separate file and store them at 'fpath'.

	# atmosphere name (extracted from full path given)
	fname = fpath.split("/")[-1]

	out = open(fpath,"w")
	npar, nz = atm.shape

	out.write("* Model file\n")
	out.write("*\n")
	out.write(f"  {fname}\n")
	if globin.atm_scale=="tau":	
		out.write("  Tau scale\n")
	elif globin.atm_scale=="cmass":
		out.write("  Mass scale\n")
	else:
		print(f"Error: Not supported {globin.atm_scale} atmosphere scale.")
		sys.exit()
	out.write("*\n")
	out.write("* log(g) [cm s^-2]\n")
	out.write("  4.44\n")
	out.write("*\n")
	out.write("* Ndep\n")
	out.write(f"  {nz}\n")
	out.write("*\n")
	if globin.atm_scale=="tau":
		out.write("* log tau    Temp[K]    n_e[cm-3]    v_z[km/s]   v_turb[km/s]\n")
		for i_ in range(nz):
			out.write("  {:+5.4f}    {:12.6f}   {:8.6e}   {:8.6e}   {:8.6e}\n".format(atm[0,i_], atm[1,i_], atm[2,i_], atm[3,i_], atm[4,i_]))
	elif globin.atm_scale=="cmass":
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

	globin.write_B(f"{fpath}.B", atm[5]/1e4, gamma, chi)

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

def broaden_rfs(rf, vmac, skip_par):
	if vmac==0:
		return rf

	nx, ny, npar, nw, ns = rf.shape

	step = globin.wavelength[1] - globin.wavelength[0]
	kernel_sigma = vmac*1e3 / globin.LIGHT_SPEED * (globin.wavelength[0] + globin.wavelength[-1])*0.5 / step

	# we assume equidistant seprataion
	radius = int(4*kernel_sigma + 0.5)
	x = np.arange(-radius, radius+1)
	phi = np.exp(-x**2/kernel_sigma**2)
	kernel = phi/phi.sum()
	# since we are correlating, we need to reverse the order of data
	kernel = kernel[::-1]

	for idx in range(nx):
		for idy in range(ny):
			for sID in range(ns):
				for pID in range(npar):
					if pID!=skip_par:
						rf[idx,idy,pID,:,sID] = correlate1d(rf[idx,idy,pID,:,sID], kernel)

	return rf

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
		return

	from tqdm import tqdm

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
			items = pool.map(func=globin.pool_spinor2multi, iterable=args)
	print("Done!")

	atmos = Atmosphere(nx=nx, ny=ny, nz=nz)
	for i_, in_data in enumerate(items):
		IDx, IDy = idx[i_], idy[i_]
		atmos.data[IDx,IDy] = in_data
	atmos.logtau = atmos.data[0,0,0]

	return atmos

def sir2multi(atmos_data):
	pass

def multi2spinor(multi_atmosphere, fname=None):
	from .makeHSE import Axmu

	nx, ny, _, nz = multi_atmosphere.shape

	height = np.zeros(nz)
	spinor_atmosphere = np.zeros((nx,ny,nz,12))

	ind0 = np.argmin(np.abs(multi_atmosphere[0,0,0]))

	for idx in range(nx):
		for idy in range(ny):
			pg, pe, kappa, rho = globin.makeHSE(5000, multi_atmosphere[idx,idy,0], multi_atmosphere[idx,idy,1])

			spinor_atmosphere[idx,idy,:,3] = pg
			spinor_atmosphere[idx,idy,:,4] = pe
			spinor_atmosphere[idx,idy,:,5] = kappa
			spinor_atmosphere[idx,idy,:,6] = rho

			# nHtot = np.sum(multi_atmosphere[idx,idy,8:], axis=0)
			# avg_mass = np.mean(Axmu)
			# rho = nHtot * avg_mass

			for idz in range(nz-2,-1,-1):
				height[idz] = height[idz+1] + 2*(kappa[idz+1] - kappa[idz]) / (rho[idz+1] + rho[idz])

			height -= height[ind0]
			spinor_atmosphere[idx,idy,:,1] = height

	spinor_atmosphere[:,:,:,0] = multi_atmosphere[:,:,0,:]
	spinor_atmosphere[:,:,:,2] = multi_atmosphere[:,:,1,:]
	spinor_atmosphere[:,:,:,7] = multi_atmosphere[:,:,4,:]*1e5
	spinor_atmosphere[:,:,:,8] = multi_atmosphere[:,:,3,:]*1e5
	spinor_atmosphere[:,:,:,9] = multi_atmosphere[:,:,5,:]
	spinor_atmosphere[:,:,:,10] = multi_atmosphere[:,:,6,:]
	spinor_atmosphere[:,:,:,11] = multi_atmosphere[:,:,7,:]

	if fname:
		primary = fits.PrimaryHDU(spinor_atmosphere)
		primary.header["LTTOP"] = min(spinor_atmosphere[0,0,0])
		primary.header["LTBOT"] = max(spinor_atmosphere[0,0,0])
		primary.header["LTINC"] = np.round(spinor_atmosphere[0,0,0,1] - spinor_atmosphere[0,0,0,0], decimals=2)
		primary.header["LOGTAU"] = (1, "optical depth scale")
		primary.header["HEIGHT"] = (2, "height scale (cm)")
		primary.header["TEMP"] = (3, "temperature (K)")
		primary.header["PGAS"] = (4, "gass pressure (dyn/cm2)")
		primary.header["PEL"] = (5, "electron pressure (dyn/cm2)")
		primary.header["KAPPA"] = (6, "mass density (g/cm2)")
		primary.header["DENS"] = (7, "density (g/cm3)")
		primary.header["VMIC"] = (8, "micro-turbulent velocity (cm/s)")
		primary.header["VZ"] = (9, "vertical velocity (cm/s)")
		primary.header["MAG"] = (10, "magnetic field strength (G)")
		primary.header["GAMMA"] = (11, "magnetic field inclination (rad)")
		primary.header["CHI"] = (12, "magnetic field azimuth (rad)")

		primary.writeto(fname, overwrite=True)

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

	for parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi", "of"]:
		try:
			ind = hdu_list.index_of(parameter)
			data = hdu_list[ind].data[:, xmin:xmax, ymin:ymax, :]
			_, nx, ny, nnodes = data.shape

			atmos.nodes[parameter] = data[0,0,0]
			# angles are saved in radians, no need to convert them here
			if parameter=="gamma":
				# atmos.values[parameter] = np.tan(data[1]/2)
				atmos.values[parameter] = np.cos(data[1])
				atmos.values[parameter] = data[1]
			elif parameter=="chi":
				# atmos.values[parameter] = np.tan(data[1]/4)
				# atmos.values[parameter] = np.cos(data[1])
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
	
	atmos = globin.atmos.convert_atmosphere(atmos_data[0], atmos_data, "spinor")

	return atmos

def read_sir():
	pass
