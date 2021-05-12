"""
Contributors:
  Dusan Vukadinovic (DV)

17/09/2020 : started writing the code (readme file, structuring)
12/10/2020 : wrote down reading of atmos in .fits format and sent
			 calculation to different process
15/11/2020 : rewriten class Atmosphere
"""

import subprocess as sp
import multiprocessing as mp
from astropy.io import fits
import numpy as np
import os
import sys
import time
import copy
from scipy.ndimage import gaussian_filter, gaussian_filter1d, correlate1d
from scipy.ndimage.filters import _gaussian_kernel1d
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt

import globin

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
	def __init__(self, fpath=None, atm_type="multi", verbose=False, atm_range=[0,None,0,None], nx=None, ny=None, logtau_top=-6, logtau_bot=1, logtau_step=0.1):
		self.verbose = verbose
		self.fpath = fpath
		self.type = atm_type

		# list of fpath's to line lists (in mode=2)
		self.line_lists_path = []

		# nodes: each atmosphere has the same nodes
		self.nodes = {}
		
		# node mask: we can specify only which nodes to invert (mask==1)
		# structure and ordering same as self.nodes
		self.mask = {}
		
		# parameters in nodes: shape = (nx, ny, nnodes)
		self.values = {}
		
		# global parameters: each is given in a list size equal to number of parameters
		self.global_pars = {}
		# line number in list of lines for which we are inverting atomic data
		self.line_no = {}

		# index of parameters in self.data ndarray
		self.par_id = {"logtau" : 0,
					   "temp"   : 1,
					   "ne"     : 2,
					   "vz"     : 3,
					   "vmic"   : 4,
					   "mag"    : 5,
					   "gamma"  : 6,
					   "chi"    : 7}

		self.nx, self.ny, self.npar = nx, ny, 14

		# define log(tau) scale
		self.nz = int((logtau_bot - logtau_top) / logtau_step) + 1
		self.logtau = np.linspace(logtau_top, logtau_bot, num=self.nz)

		# if we provided nx and ny, make empty atmosphere; otherwise set it to None
		# if (self.nx is not None) and (self.ny is not None):
		# 	self.data = np.zeros((self.nx, self.ny, self.npar, self.nz), dtype=np.float64)
		# 	self.data[:,:,0,:] = self.logtau
		# else:
		self.data = None

		self.path = None
		self.atm_name_list = []

		# if we provide path to atmosphere, read in data
		if fpath is not None:
			# by file extension determine atmosphere type / format
			extension = fpath.split(".")[-1]

			# read atmosphere by the type
			if extension=="dat" or extension=="txt":
				self.read_spinor(fpath)
			# read fits / fit / cube type atmosphere
			elif extension=="fits" or extension=="fit":
				self.read_fits(fpath, atm_range)
			else:
				print("--> Error in atmos.Atmosphere()")
				print(f"    Unsupported extension '{extension}' of atmosphere file.")
				print("    Supported extensions are: .dat, .txt, .fit(s)")
				sys.exit()

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
		new.global_pars = copy.deepcopy(self.global_pars)
		new.par_id = copy.deepcopy(self.par_id)
		new.vmac = copy.deepcopy(self.vmac)
		new.sigma = copy.deepcopy(self.sigma)
		new.line_lists_path = copy.deepcopy(self.line_lists_path)
		try:
			new.n_local_pars = copy.deepcopy(self.n_local_pars)
			new.n_global_pars = copy.deepcopy(self.n_global_pars)
		except:
			pass

		return new

	def __str__(self):
		return "<Atmosphere: fpath = {}, (nx,ny,npar,nz) = ({},{},{},{})>".format(self.fpath, self.nx, self.ny, self.npar, self.nz)

	def read_fits(self, fpath, atm_range):
		try:
			atmos = fits.open(fpath)[0]
		except:
			print("--> Error in atmos.read_fits()")
			print(f"    Atmosphere file with path '{fpath}' does not exist.")
			sys.exit()

		self.header = atmos.header
		atmos_data = np.array(atmos.data, dtype=np.float64)
		xmin, xmax, ymin, ymax = atm_range
		atmos_data = atmos_data[xmin:xmax, ymin:ymax]
		logtau = atmos_data[0,0,0]
		self.path = fpath
		self.nx, self.ny, self.nz = atmos_data.shape[0], atmos_data.shape[1], atmos_data.shape[3]

		self.convert_atmosphere(logtau, atmos_data)

		print(f"Read atmosphere '{self.path}' with dimensions:")
		print(f"  (nx, ny, npar, nz) = {self.data.shape}\n")

	def read_spinor(self, fpath):
		atmos_data = np.loadtxt(fpath, skiprows=1, dtype=np.float64).T
		logtau = atmos_data[0]
		self.nz = atmos_data.shape[1]
		self.nx = 1
		self.ny = 1
		
		aux = np.zeros((self.nx, self.ny, *atmos_data.shape))
		aux[0,0] = atmos_data

		self.convert_atmosphere(logtau, aux)

	def read_sir(self):
		pass

	def read_multi(self):
		pass

	def convert_atmosphere(self, logtau, atmos_data):
		if self.type=="spinor":
			multi_atmos = self.spinor2multi(atmos_data)
			if not np.array_equal(multi_atmos[0,0,0], self.logtau):
				self.interpolate_atmosphere(x_new=self.logtau, ref_atm=multi_atmos)
			else:
				self.data = np.zeros(multi_atmos.shape)
				self.data = multi_atmos
		elif self.type=="sir":
			self.sir2multi()
		elif self.type=="multi":
			if not np.array_equal(logtau, self.logtau):
				self.interpolate_atmosphere(x_new=logtau, ref_atm=atmos_data)
			else:
				self.logtau = logtau
				self.data = atmos_data
		else:
			print("--> Error in atmos.read_fits()")
			print(f"    Currently not recognized atmosphere type: {self.type}")
			print("    Recognized ones are: spinor, sir and multi.")
			sys.exit()

	def spinor2multi(self, atmos_data):
		nx, ny, _, nz = atmos_data.shape
		npar = 14
		data = np.zeros((nx, ny, npar, nz))

		for idx in range(nx):
			for idy in range(ny):	
				# log(tau)
				data[idx,idy,0] = atmos_data[idx,idy,0]
				# Temperature [K]
				data[idx,idy,1] = atmos_data[idx,idy,2]
				# Electron density [1/m3]
				data[idx,idy,2] = atmos_data[idx,idy,4]/10 / 1.380649e-23 / atmos_data[idx,idy,2] / 1e6
				# Vertical velocity [cm/s] --> [km/s]
				data[idx,idy,3] = atmos_data[idx,idy,9]/1e5
				# Microturbulent velocitu [cm/s] --> [km/s]
				data[idx,idy,4] = atmos_data[idx,idy,8]/1e5
				# Magnetic field strength [G] --> [T]
				data[idx,idy,5] = atmos_data[idx,idy,7]/1e4
				# Inclination [deg] --> [rad]
				data[idx,idy,6] = atmos_data[idx,idy,-2] * np.pi/180
				# Azimuth [deg] --> [rad]
				data[idx,idy,7] = atmos_data[idx,idy,-1] * np.pi/180

				# Hydrogen population
				data[idx,idy,8:] = distribute_hydrogen(atmos_data[idx,idy,2], atmos_data[idx,idy,3], atmos_data[idx,idy,4])

		return data

	def sir2multi(self):
		pass

	def get_atmos(self, idx, idy):
		# return atmosphere from cube with given indices 'idx' and 'idy'.
		try:
			return self.data[idx,idy]
		except:
			print(f"Error: Atmosphere index notation does not match")
			print(f"       it's dimension. No atmosphere (x,y) = ({idx},{idy})\n")
			sys.exit()

	def write_atmosphere(self):
		for idx in range(self.nx):
			for idy in range(self.ny):
				atm = self.get_atmos(idx, idy)
				fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
				write_multi_atmosphere(atm, fpath)

	def split_cube(self):
		# go through each atmosphere and extract it to separate file
		for idx in range(self.nx):
			for idy in range(self.ny):
				atm = self.get_atmos(idx, idy)
				fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
				self.atm_name_list.append(fpath)
				write_multi_atmosphere(atm, fpath)

		if self.verbose:
			print("Extracted all atmospheres into folder 'atmospheres'\n")

	def build_from_nodes(self, save_atmos=True):
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
		for idx in range(self.nx):
			for idy in range(self.ny):
				for parameter in self.nodes:
					# K0, Kn by default; True for vmic, gamma and chi
					K0, Kn = 0, 0

					x = self.nodes[parameter]
					y = self.values[parameter][idx,idy]

					if parameter=="temp":
						if len(x)>=2:
							K0 = (y[1]-y[0]) / (x[1]-x[0])
							# check if extrapolation at the top atmosphere point goes below the minimum
							# if does, change the slopte so that at top point we have Tmin (globin.limit_values["temp"][0])
							if globin.limit_values["temp"][0]>(y[0] + K0 * (self.logtau[0]-x[0])):
								K0 = (globin.limit_values["temp"][0] - y[0]) / (self.logtau[0] - x[0])
						# bottom node slope for extrapolation based on temperature gradient from FAL C model
						Kn = splev(x[-1], globin.temp_tck, der=1)
					elif parameter=="vz":
						if len(x)>=2:
							K0 = (y[1]-y[0]) / (x[1]-x[0])
							Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
							#--- this checks does not make any sense to me now (23.12.2020.) --> Recheck this later
							# check if extrapolation at the top atmosphere point goes below the minimum
							# if does, change the slopte so that at top point we have vzmin (globin.limit_values["vz"][0])
							if globin.limit_values["vz"][0]>(y[0] + K0 * (self.logtau[0]-x[0])):
								K0 = (globin.limit_values["vz"][0] - y[0]) / (self.logtau[0] - x[0])
							# similar for the bottom for maximum values
							if globin.limit_values["vz"][1]<(y[-1] + K0 * (self.logtau[-1]-x[-1])):
								Kn = (globin.limit_values["vz"][1] - y[-1]) / (self.logtau[-1] - x[-1])
					elif parameter=="mag":
						if len(x)>=2:
							Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
							#--- this checks does not make any sense to me now (23.12.2020.) --> Recheck this later
							if globin.limit_values["mag"][1]<(y[-1] + K0 * (self.logtau[-1]-x[-1])):
								Kn = (globin.limit_values["mag"][1] - y[-1]) / (self.logtau[-1] - x[-1])
							if globin.limit_values["mag"][0]>(y[-1] + K0 * (self.logtau[-1]-x[-1])):
								Kn = (globin.limit_values["mag"][1] - y[-1]) / (self.logtau[-1] - x[-1])

					y_new = globin.bezier_spline(x, y, self.logtau, K0=K0, Kn=Kn, degree=globin.interp_degree)
					self.data[idx,idy,self.par_id[parameter],:] = y_new

				#--- save interpolated atmosphere to appropriate file
				if save_atmos:
					fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
					write_multi_atmosphere(self.data[idx,idy], fpath)

	def interpolate_atmosphere(self, x_new, ref_atm):
		if (x_new[0]<ref_atm[0,0,0,0]) or \
		   (x_new[-1]>ref_atm[0,0,0,-1]):
			# print("--> Warning: atmosphere will be extrapolated")
			# print("    from {} to {} in optical depth.\n".format(ref_atm[0,0,0,0], x_new[0]))
			self.logtau = ref_atm[0,0,0]
			self.data = ref_atm
			return

		self.nz = len(x_new)
		nx, ny, _, _ = ref_atm.shape

		self.data = np.zeros((self.nx, self.ny, self.npar, self.nz))
		self.data[:,:,0,:] = x_new
		self.logtau = x_new

		for idx in range(self.nx):
			for idy in range(self.ny):
				for parID in range(1,self.npar):
					if nx*ny>1:
						tck = splrep(ref_atm[0,0,0], ref_atm[idx,idy,parID])
					elif nx*ny==1:
						tck = splrep(ref_atm[0,0,0], ref_atm[0,0,parID])
					self.data[idx,idy,parID] = splev(x_new, tck)

	def save_atmosphere(self, fpath="inverted_atmos.fits", kwargs=None):
		primary = fits.PrimaryHDU(self.data, do_not_scale_image_data=True)
		primary.name = "Atmosphere"

		primary.header.comments["NAXIS1"] = "depth points"
		primary.header.comments["NAXIS2"] = "number of parameters"
		primary.header.comments["NAXIS3"] = "y-axis atmospheres"
		primary.header.comments["NAXIS4"] = "x-axis atmospheres"

		primary.header["VMAC"] = ("{:5.3f}".format(self.vmac), "macro-turbulen velocity")
		if "vmac" in self.global_pars:
			primary.header["VMAC_FIT"] = ("TRUE", "flag for fitting macro velocity")
		else:
			primary.header["VMAC_FIT"] = ("FALSE", "flag for fitting macro velocity")

		# add keys from kwargs (as dict)
		if kwargs:
			for key in kwargs:
				primary.header[key] = kwargs[key]

		hdulist = fits.HDUList([primary])

		for parameter in self.nodes:
			matrix = np.ones((2, self.nx, self.ny, len(self.nodes[parameter])))
			matrix[0] *= self.nodes[parameter]
			matrix[1] = self.values[parameter]

			par_hdu = fits.ImageHDU(matrix)
			par_hdu.name = parameter

			par_hdu.header["unit"] = globin.parameter_unit[parameter]
			par_hdu.header.comments["NAXIS1"] = "number of nodes"
			par_hdu.header.comments["NAXIS2"] = "y-axis atmospheres"
			par_hdu.header.comments["NAXIS3"] = "x-axis atmospheres"
			par_hdu.header.comments["NAXIS4"] = "1 - node values | 2 - parameter values"

			hdulist.append(par_hdu)

		hdulist.writeto(fpath, overwrite=True)

	def save_atomic_parameters(self, fpath="inverted_atoms.fits", kwargs=None):
		pars = list(self.global_pars.keys())

		primary = fits.PrimaryHDU()
		hdulist = fits.HDUList([primary])
		
		for parameter in pars:
			if globin.mode==2:
				nx, ny = self.nx, self.ny
			elif globin.mode==3:
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

	def check_parameter_bounds(self):
		for parameter in self.values:
			# for every gamma, we are returnig the angle (in rad) 
			# which is in 1st and 2nd quadrant
			# there is no need for checking the bounds because
			# np.arccos() returns the angle in range [0, np.pi]
			if parameter=="gamma":# or parameter=="chi":
				cos_angle = np.cos(self.values[parameter])
				self.values[parameter] = np.arccos(cos_angle)
			elif parameter=="chi":
				self.values[parameter] %= 2*np.pi
			else:
				for i_ in range(len(self.nodes[parameter])):
					for idx in range(self.nx):
						for idy in range(self.ny):
							# minimum check
							if self.values[parameter][idx,idy,i_]<globin.limit_values[parameter][0]:
								self.values[parameter][idx,idy,i_] = globin.limit_values[parameter][0]
							# maximum check
							if self.values[parameter][idx,idy,i_]>globin.limit_values[parameter][1]:
								self.values[parameter][idx,idy,i_] = globin.limit_values[parameter][1]
		for parameter in self.global_pars:
			if parameter=="vmac":
				# minimum check
				if self.global_pars[parameter]<globin.limit_values[parameter][0]:
					self.global_pars[parameter] = np.array(globin.limit_values[parameter][0])
				# maximum check
				if self.global_pars[parameter]>globin.limit_values[parameter][1]:
					self.global_pars[parameter] = np.array(globin.limit_values[parameter][1])
				self.vmac = self.global_pars["vmac"]
			else:
				if globin.mode==2:
					nx, ny = self.nx, self.ny
				elif globin.mode==3:
					nx, ny = 1,1
				for idx in range(nx):
					for idy in range(ny):
						for i_ in range(self.line_no[parameter].size):
							# minimum check
							if self.global_pars[parameter][idx,idy,i_]<globin.limit_values[parameter][i_,0]:
								self.global_pars[parameter][idx,idy,i_] = globin.limit_values[parameter][i_,0]
							# maximum check
							if self.global_pars[parameter][idx,idy,i_]>globin.limit_values[parameter][i_,1]:
								self.global_pars[parameter][idx,idy,i_] = globin.limit_values[parameter][i_,1]

	def update_parameters(self, proposed_steps, stop_flag=None):
		if stop_flag is not None:
			low_ind, up_ind = 0, 0
			# update atmospheric parameters
			for parameter in self.values:
				low_ind = up_ind
				up_ind += len(self.nodes[parameter])
				step = proposed_steps[:,:,low_ind:up_ind] / globin.parameter_scale[parameter]
				# we do not perturb parameters of those pixels which converged
				step = np.einsum("...i,...->...i", step, stop_flag)
				# if parameter=="gamma":
				# 	aux = np.cos(self.values[parameter]) + step
				# 	self.values[parameter] = np.arccos(aux)
				# elif parameter=="chi":
				# 	aux = np.sin(self.values[parameter]) + step
				# 	self.values[parameter] = np.arcsin(aux)
				# else:
				if globin.rf_type=="snapi":
					# RH returns RFs in m/s and we are working with km/s in globin
					# so we have to return the values from m/s to km/s
					if parameter=="vz" or parameter=="vmic":
						step /= 1e3
				np.nan_to_num(step, nan=0.0, copy=False)
				self.values[parameter] += step * self.mask[parameter]

			# update atomic parameters
			for parameter in self.global_pars:
				low_ind = up_ind
				up_ind += self.line_no[parameter].size
				step = proposed_steps[:,:,low_ind:up_ind] / globin.parameter_scale[parameter]
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
						step = proposed_steps[low_ind:up_ind] / globin.parameter_scale[parameter][idx,idy]
						# RH returns RFs in m/s and we are working with km/s in globin
						# so we have to return the values from m/s to km/s
						if parameter=="vz" or parameter=="vmic":
							step /= 1e3
						np.nan_to_num(step, nan=0.0, copy=False)
						self.values[parameter][idx,idy] += step * self.mask[parameter]

			# update atomic parameters
			for parameter in self.global_pars:
				low_ind = up_ind
				up_ind += self.global_pars[parameter].size
				step = proposed_steps[low_ind:up_ind] / globin.parameter_scale[parameter]
				np.nan_to_num(step, nan=0.0, copy=False)
				self.global_pars[parameter] += step

	def smooth_parameters(self):
		#--- atmospheric parameters
		for parameter in self.nodes:
			new_values = np.random.normal(loc=self.values[parameter], 
										  scale=globin.smooth_std[parameter], 
										  size=self.values[parameter].shape)
			self.values[parameter] = new_values

		#--- global parameters
		for parameter in self.global_pars:
			new_values = np.random.normal(loc=self.global_pars[parameter],
										  scale=globin.smooth_std[parameter],
										  size=self.global_pars[parameter].shape)
			self.global_pars[parameter] = new_values

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

def distribute_hydrogen(temp, pg, pe):
	from scipy.interpolate import interp1d

	kb = 1.38064852e-23
	h = 6.62607004e-34
	me = 9.10938356e-31

	Ej = 13.59844
	Ediss = 0.75
	u0_coeffs=[2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00001e+00, 2.00003e+00, 2.00015e+00], 
	u1_coeffs=[1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00],
	
	u1 = interp1d(np.linspace(3000,10000,num=8), u1_coeffs, fill_value="extrapolate")(temp)
	u0 = interp1d(np.linspace(3000,10000,num=8), u0_coeffs, fill_value="extrapolate")(temp)
	
	phi_t = 0.6665 * u1/u0 * temp**(5/2) * 10**(-5040/temp*Ej)
	phi_Hminus_t = 0.6665 * u0/1 * temp**(5/2) * 10**(-5040/temp*Ediss)
	
	nH = (pg-pe)/10 / kb / temp / np.sum(10**(globin.abundance-12)) / 1e6 	
	nH0 = nH / (1 + phi_t/pe + pe/phi_Hminus_t)
	nprot = phi_t/pe * nH0

	pops = np.zeros((6, len(temp)))

	pops[-1] = nprot

	for lvl in range(5):
		e_lvl = 13.59844*(1-1/(lvl+1)**2)
		g = 2*(lvl+1)**2
		pops[lvl] = nH0/u0 * g * np.exp(-5040/temp * e_lvl)
	
	return pops

	nH = (pg-pe)/10 / kb / temp / np.sum(10**(globin.abundance-12)) / 1e6
	ne = pe/10 / kb / temp / 1e6

	Ej = 13.59844
	Ej *= 1.60218e-19 # [eV --> J]
	nstar = np.zeros((6, len(temp)))
	for i_ in range(1,5):
		n = i_+1
		dE = Ej * (n**2 - 1) / n**2
		gn = 2*n**2 / 2
		nstar[i_] = gn * np.exp(-dE/kb/temp)
		
	nstar[5] = 1/2 * 2/ne * (2*np.pi*me*kb*temp)**(3/2) / h**3 * np.exp(-Ej/kb/temp)
	
	suma = 1 + np.sum(nstar[1:], axis=0)
	nstar[0] = nH / suma

	# plt.plot(nstar[0], label="ground")
	for i_ in range(1,6):
		nstar[i_] *= nstar[0]
		# plt.plot(nstar[i_], label=f"{i_}")
	
	# plt.yscale("log")
	# plt.legend()
	# plt.show()

	return nstar

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
	out.write("  Tau scale\n")
	out.write("*\n")
	out.write("* log(g) [cm s^-2]\n")
	out.write("  4.44\n")
	out.write("*\n")
	out.write("* Ndep\n")
	out.write(f"  {nz}\n")
	out.write("*\n")
	out.write("* log tau    Temp[K]    n_e[cm-3]    v_z[km/s]   v_turb[km/s]\n")

	for i_ in range(nz):
		out.write("  {:+5.4f}    {:6.2f}   {:5.4e}   {:5.4e}   {:5.4e}\n".format(atm[0,i_], atm[1,i_], atm[2,i_], atm[3,i_], atm[4,i_]))

	out.write("*\n")
	out.write("* Hydrogen populations [cm-3]\n")
	out.write("*     nh(1)        nh(2)        nh(3)        nh(4)        nh(5)        nh(6)\n")

	for i_ in range(nz):
		out.write("   {:5.4e}   {:5.4e}   {:5.4e}   {:5.4e}   {:5.4e}   {:5.4e}\n".format(atm[8,i_], atm[9,i_], atm[10,i_], atm[11,i_], atm[12,i_], atm[13,i_]))

	out.close()

	# store now and magnetic field vector
	globin.write_B(f"{fpath}.B", atm[5], atm[6], atm[7])

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

	atmospheres = copy.deepcopy(globin.atm)
	height = np.zeros((Nx, Ny, Nz), dtype=np.float64)

	for item in lista:
		if item is not None:
			rh_obj, idx, idy = item.values()

			ind_min = np.argmin(abs(rh_obj.wave - globin.lmin))
			ind_max = np.argmin(abs(rh_obj.wave - globin.lmax))+1
			
			if globin.norm:
				sI_cont = rh_obj.int[ind_min]
			else:
				sI_cont = 1

			# Stokes vector
			spectra.spec[idx,idy,:,0] = rh_obj.int[ind_min:ind_max] / sI_cont
			# if there is magnetic field, read the Stokes components
			if rh_obj.stokes:
				spectra.spec[idx,idy,:,1] = rh_obj.ray_stokes_Q[ind_min:ind_max] / sI_cont
				spectra.spec[idx,idy,:,2] = rh_obj.ray_stokes_U[ind_min:ind_max] / sI_cont
				spectra.spec[idx,idy,:,3] = rh_obj.ray_stokes_V[ind_min:ind_max] / sI_cont

			# Atmospheres
			# Atmopshere read here is one projected to local reference frame (from rhf1d) and
			# not from solveray!
			atmospheres.data[idx,idy,0] = np.log10(rh_obj.geometry["tau500"])
			atmospheres.data[idx,idy,1] = rh_obj.atmos["T"]
			atmospheres.data[idx,idy,2] = rh_obj.atmos["n_elec"] / 1e6 	# [1/m3 --> 1/cm3]
			atmospheres.data[idx,idy,3] = rh_obj.geometry["vz"] / 1e3  	# [m/s --> km/s]
			atmospheres.data[idx,idy,4] = rh_obj.atmos["vturb"] / 1e3  	# [m/s --> km/s]
			try:
				atmospheres.data[idx,idy,5] = rh_obj.atmos["B"] #* 1e4    	# [T --> G]
				atmospheres.data[idx,idy,6] = rh_obj.atmos["gamma_B"] #* 180/np.pi	# [rad --> deg]
				atmospheres.data[idx,idy,7] = rh_obj.atmos["chi_B"] #* 180/np.pi    	# [rad --> deg]
			except:
				pass
			height[idx,idy] = rh_obj.geometry["height"]
			for i_ in range(rh_obj.atmos['nhydr']):
				atmospheres.data[idx,idy,8+i_] = rh_obj.atmos["nh"][:,i_] / 1e6 # [1/cm3 --> 1/m3]

	spectra.wave = rh_obj.wave

	return spectra, atmospheres, height

def synth_pool(args):
	"""
	Function which executes what to be done on single thread in multicore
    mashine.

	Here we check if directory for given process exits ('pid_##') and copy all
	input files there for smooth run of RH code. We change the atmosphere path to
	path to pixel atmosphere for which we want to syntesise spectrum (and same
	for magnetic field).

	Also, if the directory has files from old runs, then too speed calculation we
	use old J.

	After the successful synthesis, we read and store spectrum in variable
	'spec'.

	Parameters:
	---------------
	atm_path : string
		Atmosphere path located in directory 'atmospheres'.
	rh_spec_name : string
		File name in which spectrum is written on a disk (read from keyword.input
        file).
    """
	start = time.time()

	atm_path, line_list_path = args

	# get process ID number
	pid = mp.current_process()._identity[0]
	set_old_J = True

	#--- copy *.input files from 'runs/globin.wd' directory
	sp.run(f"cp runs/{globin.wd}/*.input {globin.rh_path}/rhf1d/{globin.wd}_{pid}",
		shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)
	set_old_J = False

	# re-read 'keyword.input' file for given pID
	globin.keyword_input = open(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.rh_input_name}", "r").read()

	keyword_path = f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.rh_input_name}"
	globin.keyword_input = globin.set_keyword(globin.keyword_input, "ATMOS_FILE", f"{globin.cwd}/{atm_path}")
	globin.keyword_input = globin.set_keyword(globin.keyword_input, "STOKES_INPUT", f"{globin.cwd}/{atm_path}.B", keyword_path)

	# make kurucz.input file in rhf1d/globin.wd_pid and give the line list path
	out = open(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.kurucz_input_fname}", "w")
	out.write(f"{globin.cwd}/{line_list_path}\n")
	out.close()

	# make log file
	aux = atm_path.split("_")
	idx, idy = aux[-2], aux[-1]
	log_file = open(f"{globin.cwd}/runs/{globin.wd}/logs/log_{idx}_{idy}", "w")
	
	# run rhf1d executable
	out = sp.run(f"cd {globin.rh_path}/rhf1d/{globin.wd}_{pid}; ../rhf1d -i {globin.rh_input_name}",
			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	
	# store log file
	log_file.writelines(str(out.stdout, "utf-8"))
	log_file.close()

	stdout = str(out.stdout,"utf-8").split("\n")

	# check if rhf1d executed normaly
	if out.returncode!=0:
		print("*** RH error (synth_pool)")
		print(f"    Failed to synthesize spectra for pixel ({idx},{idy}).\n")
		for line in stdout[-5:]:
			print("   ", line)
		return None
	else:
		# if everything was fine, run solverray executable
		out = sp.run(f"cd {globin.rh_path}/rhf1d/{globin.wd}_{pid}; ../solveray",
			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
		if out.returncode!=0:
			print(f"Could not synthesize the spectrum for the ray! --> ({idx},{idy})\n")
			return None

	# read output spectra and spectrum ray from RH
	rh_obj = globin.rh.Rhout(fdir=f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}", verbose=False)
	rh_obj.read_spectrum(globin.rh_spec_name)
	rh_obj.read_ray()

	dt = time.time() - start
	if globin.mode==0:	
		print("Finished synthesis of '{:}' in {:4.2f} s".format(atm_path, dt))

	return {"rh_obj":rh_obj, "idx":int(idx), "idy":int(idy)}

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

	if globin.mode==0 or globin.mode==1 or globin.mode==3:
		args = [ [atm_name, atmos.line_lists_path[0]] for atm_name in atmos.atm_name_list]
	elif globin.mode==2:
		if len(atmos.line_lists_path)>1:	
			args = [ [atm_name, line_list_path] for atm_name, line_list_path in zip(atmos.atm_name_list, atmos.line_lists_path)]
		else:
			args = [ [atm_name, atmos.line_lists_path[0]] for atm_name in atmos.atm_name_list]
	else:
		print("--> Error in compute_spectra()")
		print("    We can not make a list of arguments for computing spectra.")
		globin.remove_dirs()
		sys.exit()

	#--- make directory in which we will save logs of running RH
	if not os.path.exists(f"{globin.cwd}/runs/{globin.wd}/logs"):
		os.mkdir(f"{globin.cwd}/runs/{globin.wd}/logs")
	else:
		sp.run(f"rm {globin.cwd}/runs/{globin.wd}/logs/*",
			shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)

	#--- distribute the process to threads
	rh_obj_list = globin.pool.map(func=synth_pool, iterable=args)

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
	spectra, atmospheres, height = extract_spectra_and_atmospheres(rh_obj_list, atmos.nx, atmos.ny, atmos.nz)

	return spectra, atmospheres, height

def compute_rfs(atmos, rf_noise_scale, old_rf=None, old_pars=None):
	#--- get inversion parameters for atmosphere and interpolate it on finner grid (original)
	atmos.build_from_nodes()
	spec, _, _ = compute_spectra(atmos)

	# globin.plot_atmosphere(atmos, ["temp"])
	# globin.plot_atmosphere(globin.ref_atm, ["temp"], color="tab:red")
	# plt.show()

	if globin.rf_type=="snapi":	
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
			full_rf = RH_compute_RF(atmos, par_flag, init.rh_spec_name, init.wavelength)
			for i_,par in enumerate(pars):
				rfID = rf_id[par]
				if not par_flag[i_]:
					full_rf[:,:, rfID] = old_rf[:,:, rfID]
		else:
			full_rf = None
	elif globin.rf_type=="node":
		full_rf = None
	else:
		print("Not proper RF type calculation.\n")
		globin.remove_dirs()
		sys.exit()

	# for i_,par in enumerate(pars):
	# 	rfID = rf_id[par]
	# 	plt.figure(i_+1)
	# 	plt.imshow(full_rf[0,0,rfID,:,:,0], aspect="auto")
	# 	plt.colorbar()
	# plt.show()

	#--- get total number of parameters (local + global)
	Npar = atmos.n_local_pars + atmos.n_global_pars
	
	rf = np.zeros((atmos.nx, atmos.ny, Npar, len(globin.wavelength), 4), dtype=np.float64)
	node_RF = np.zeros((atmos.nx, atmos.ny, len(globin.wavelength), 4))

	model_plus = copy.deepcopy(atmos)
	model_minus = copy.deepcopy(atmos)

	#--- loop through local (atmospheric) parameters and calculate RFs
	free_par_ID = 0
	for parameter in atmos.nodes:
		parID = atmos.par_id[parameter]
		rfID = rf_id[parameter]

		nodes = atmos.nodes[parameter]
		perturbation = globin.delta[parameter]
		
		# if parameter=="gamma" or parameter=="chi":
		# 	parameter_scale = -1/np.sin(values) * parameter_scale
		# elif parameter=="chi":
		# 	parameter_scale = 1/np.cos(values) * parameter_scale

		for nodeID in range(len(nodes)):
			if globin.rf_type=="snapi":
				#===--- computing RFs for given parameter (proper way as in SNAPI)
				# positive perturbation
				atmos.values[parameter][:,:,nodeID] += perturbation
				atmos.build_from_nodes(False)
				positive = copy.deepcopy(atmos.data[:,:,parID])

				# negative perturbation
				atmos.values[parameter][:,:,nodeID] -= 2*perturbation
				atmos.build_from_nodes(False)
				negative = copy.deepcopy(atmos.data[:,:,parID])

				# derivative of parameter distribution to node perturbation
				dy_dnode = (positive - negative) / 2 / perturbation

				for idx in range(atmos.nx):
					for idy in range(atmos.ny):
						node_RF[idx,idy] = np.einsum("i,ijk", dy_dnode[idx,idy], full_rf[idx,idy,rfID])
			elif globin.rf_type=="node":
				#===--- Computing RFs in nodes
				# positive perturbation
				model_plus.values[parameter][:,:,nodeID] += perturbation
				model_plus.build_from_nodes()
				spectra_plus,_,_ = compute_spectra(model_plus)

				# negative perturbation (except for gamma and chi)
				if parameter=="gamma" or parameter=="chi":
					node_RF = (spectra_plus.spec - spec.spec ) / perturbation
				else:
					model_minus.values[parameter][:,:,nodeID] -= perturbation
					model_minus.build_from_nodes()
					spectra_minus,_,_ = compute_spectra(model_minus)

					node_RF = (spectra_plus.spec - spectra_minus.spec ) / 2 / perturbation

			node_RF *= globin.weights
			node_RF /= rf_noise_scale
			scale = np.sqrt(np.sum(node_RF**2, axis=(2,3)))

			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					if scale[idx,idy]==0:
						# print("scale==0 for --> ", parameter)
						globin.parameter_scale[parameter][idx,idy,nodeID] = 1
					elif not np.isnan(np.sum(scale[idx,idy])):
						globin.parameter_scale[parameter][idx,idy,nodeID] = scale[idx,idy]

			# if parameter=="gamma" or parameter=="chi":
			# 	rf[:,:,free_par_ID,:,:] = np.einsum("...ij,...", node_RF, parameter_scale[:,:,nodeID])
			# else:
			
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					rf[idx,idy,free_par_ID] = node_RF[idx,idy] / globin.parameter_scale[parameter][idx,idy,nodeID]
			free_par_ID += 1

			if globin.rf_type=="snapi":
				# return back perturbation (SNAPI way)
				atmos.values[parameter][:,:,nodeID] += perturbation
				atmos.build_from_nodes(False)
			elif globin.rf_type=="node":
				# return back perturbations (node way)
				model_plus.values[parameter][:,:,nodeID] -= perturbation
				if parameter!="gamma" or parameter!="chi":
					model_minus.values[parameter][:,:,nodeID] += perturbation

	# continuum intensity for each pixel (need to scale RFs for global parameters in mode=3)
	sIcont = spec.spec[:,:,0,0]
	fact = sIcont / sIcont[0,0]
	#--- loop through global parameters and calculate RFs
	if atmos.n_global_pars>0:
		#--- loop through global parameters and calculate RFs
		for parameter in atmos.global_pars:
			if parameter=="vmac":
				kernel_sigma = spec.get_kernel_sigma(atmos.vmac)
				radius = int(4*kernel_sigma + 0.5)
				x = np.arange(-radius, radius+1)
				phi = np.exp(-x**2/kernel_sigma**2)
				# normalaizing the profile
				phi *= 1/(np.sqrt(np.pi)*kernel_sigma)
				kernel = phi*(2*x**2/kernel_sigma**2 - 1) * 1 / kernel_sigma / init.step
				# since we are correlating, we need to reverse the order of data
				kernel = kernel[::-1]

				"""
				Miss here the good parameter scaling as is done for
				other parameters.
				"""

				for idx in range(atmos.nx):
					for idy in range(atmos.ny):
						for sID in range(4):
							rf[idx,idy,free_par_ID,:,sID] = correlate1d(spec.spec[idx,idy,:,sID], kernel)
							# rf[idx,idy,free_par_ID,:,sID] *= globin.parameter_scale["vmac"]
							rf[idx,idy,free_par_ID,:,sID] *= kernel_sigma * init.step / atmos.global_pars["vmac"]
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
								fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
								globin.write_line_par(fpath,
													values[idx,idy], line_no, parameter)
					elif globin.mode==3:
						globin.write_line_par(atmos.line_lists_path[0], values[0,0], line_no, parameter)
					
					spec_plus,_,_ = compute_spectra(atmos)
					spec_plus.broaden_spectra(atmos.vmac)

					# negative perturbation
					values -= 2*perturbation
					if globin.mode==2:
						for idx in range(atmos.nx):
							for idy in range(atmos.ny):	
								fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
								globin.write_line_par(fpath,
													values[idx,idy], line_no, parameter)
					elif globin.mode==3:
						globin.write_line_par(atmos.line_lists_path[0], values[0,0], line_no, parameter)
					
					spec_minus,_,_ = compute_spectra(atmos)
					spec_minus.broaden_spectra(atmos.vmac)

					# sIcont = spec_plus.spec[:,:,0,0]
					# print(sIcont)
					# spec_plus.spec[0,0] /= sIcont[0,0]
					# spec_plus.spec[0,1] /= sIcont[0,1]
					# spec_plus.spec[0,2] /= sIcont[0,2]
					# sIcont = spec_minus.spec[:,:,0,0]
					# print(sIcont)
					# spec_minus.spec[0,0] /= sIcont[0,0]
					# spec_minus.spec[0,1] /= sIcont[0,1]
					# spec_minus.spec[0,2] /= sIcont[0,2]

					diff = (spec_plus.spec - spec_minus.spec) / 2 / perturbation

					# scale2 = np.sqrt(np.sum(diff**2, axis=(2,3)))
					# scale3 = np.sqrt(np.sum(diff**2))

					# plt.subplot(2,1,1)
					# plt.plot(diff[0,0,:,:].flatten(order="F") / scale2[0,0])
					# plt.plot(diff[0,1,:,:].flatten(order="F") / scale2[0,1])
					# plt.plot(diff[0,2,:,:].flatten(order="F") / scale2[0,2])
					# plt.xlim([100,200])
					
					# plt.subplot(2,1,2)
					# plt.plot(diff[0,0,:,:].flatten(order="F") / scale3)# / fact[0,0])
					# plt.plot(diff[0,1,:,:].flatten(order="F") / scale3)# / fact[0,1])
					# plt.plot(diff[0,2,:,:].flatten(order="F") / scale3)# / fact[0,2])
					# plt.xlim([100,200])

					# plt.show()

					# sys.exit()

					diff *= globin.weights

					if globin.mode==2:
						scale = np.sqrt(np.sum(diff**2, axis=(2,3)))
						for idx in range(atmos.nx):
							for idy in range(atmos.ny):
								if not np.isnan(np.sum(scale[idx,idy])):
									globin.parameter_scale[parameter][idx,idy,idp] = scale[idx,idy]
					elif globin.mode==3:
						# print(fact)
						# diff = np.einsum("ijkl,ij->ijkl", diff, 1/fact)
						scale = np.sqrt(np.sum(diff**2))
						globin.parameter_scale[parameter][...,idp] = scale

					if globin.mode==2:
						for idx in range(atmos.nx):
							for idy in range(atmos.ny):
								rf[idx,idy,free_par_ID] = diff[idx,idy] / globin.parameter_scale[parameter][idx,idy,idp]
					elif globin.mode==3:
						rf[:,:,free_par_ID,:,:] = diff / globin.parameter_scale[parameter][0,0,idp]	
					free_par_ID += 1
					
					# return perturbation back
					values += perturbation
					if globin.mode==2:
						for idx in range(atmos.nx):
							for idy in range(atmos.ny):
								fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"	
								globin.write_line_par(fpath,
													values[idx,idy], line_no, parameter)
					elif globin.mode==3:
						globin.write_line_par(atmos.line_lists_path[0], values[0,0], line_no, parameter)

	# print(globin.parameter_scale["loggf"])
	# print(atmos.data[:,:,1])
	# sys.exit()

	#--- broaden the spectra
	spec.broaden_spectra(atmos.vmac)

	# for idy in range(atmos.ny):
	# 	for parID in range(Npar):
	# 		plt.figure(parID+1)
	# 		plt.plot(rf[0, idy, parID, :, 0] + 0.1*idy)
	# plt.show()
	# sys.exit()

	#--- compare RFs for single parameter
	# for idx in range(atmos.nx):
	# 	for idy in range(atmos.ny):
	# 		plt.plot(rf[idx,idy, 3, :, 0])
	# plt.show()
	# sys.exit()

	return rf, spec, full_rf

def rf_pool(args):
	start = time.time()

	atm_path, rh_spec_name = args

	#--- for each thread process create separate directory
	pid = mp.current_process()._identity[0]
	
	#--- copy *.input files
	sp.run(f"cp runs/{globin.wd}/*.input {globin.rh_path}/rhf1d/{globin.wd}_{pid}",
		shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)

	# re-read 'keyword.input' file for given pID
	globin.keyword_input = open(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.rh_input_name}", "r").read()

	keyword_path = f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.rh_input_name}"
	globin.keyword_input = globin.set_keyword(globin.keyword_input, "ATMOS_FILE", f"{globin.cwd}/{atm_path}")
	globin.keyword_input = globin.set_keyword(globin.keyword_input, "STOKES_INPUT", f"{globin.cwd}/{atm_path}.B", keyword_path)
	
	aux = atm_path.split("_")
	idx, idy = aux[-2], aux[-1]
	log_file = open(f"{globin.cwd}/runs/{globin.wd}/logs/log_{idx}_{idy}", "w")
	out = sp.run(f"cd {globin.rh_path}/rhf1d/{globin.wd}_{pid}; ../rf_ray -i {globin.rh_input_name}",
			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	log_file.writelines(str(out.stdout, "utf-8"))
	log_file.close()

	stdout = str(out.stdout,"utf-8").split("\n")

	if out.returncode!=0:
		print("*** RH error (rf_pool)")
		print(f"    Failed to compute RF for pixel ({idx},{idy}).\n")
		for line in stdout[-5:]:
			print("   ", line)
		return None

	rh_obj = globin.rh.Rhout(fdir=f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}", verbose=False)
	rh_obj.read_spectrum(rh_spec_name)
	rh_obj.read_ray()

	rf = np.loadtxt(f"{globin.rh_path}/rhf1d/{globin.wd}_{pid}/{globin.rf_file_path}")
	
	dt = time.time() - start
	# print("Finished synthesis of '{:}' in {:4.2f} s".format(atm_path, dt))

	return {"rf" : rf, "wave" : rh_obj.wave, "idx" : idx, "idy" : idy}

def RH_compute_RF(atmos, par_flag, rh_spec_name, wavelength):
	compute_spectra(atmos, rh_spec_name, wavelength)
	
	lmin, lmax = wavelength[0], wavelength[-1]

	#--- arguments for 'rf_pool' function
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
			globin.keyword_input = globin.set_keyword(globin.keyword_input, key, "TRUE", keyword_path)
			
			pars = ["temp", "vz", "vmic", "mag", "gamma", "chi"]
			pars.remove(par)
			for item in pars:
				key = "RF_" + item.upper()
				globin.keyword_input = globin.set_keyword(globin.keyword_input, key, "FALSE", keyword_path)

			rf_list = globin.pool.map(func=rf_pool, iterable=args)

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
	from tqdm import tqdm

	# set reference atmosphere
	atmos = globin.ref_atm
	atmos = copy.deepcopy(globin.atm)
	atmos.split_cube()
	atmos.line_lists_path = globin.atm.line_lists_path
	atmos.sigma = globin.atm.sigma

	atmos.line_no = globin.atm.line_no
	atmos.global_pars = globin.atm.global_pars

	spec, _,_ = compute_spectra(atmos)
	spec.broaden_spectra(atmos.vmac)

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
				spec_plus,_,_ = compute_spectra(model_plus)
				spec_plus.broaden_spectra(atmos.vmac)

				model_minus.data[:,:,parID,zID] -= perturbation
				model_minus.write_atmosphere()
				spec_minus,_,_ = compute_spectra(model_minus)
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
					
					spec_plus,_,_ = compute_spectra(atmos)
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
					
					spec_minus,_,_ = compute_spectra(atmos)
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
		fits.writeto(fpath, rf, overwrite=True)

	return rf, spec
