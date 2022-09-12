import sys
from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, correlate1d
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
import multiprocessing as mp

import globin

class Spectrum(object):
	"""
	Custom class for storing computed spectra object. It is rebuilt from RH
	class.
	"""
	def __init__(self, nx=None, ny=None, nw=None, spec=None, wave=None, fpath=None, nz=None):
		if fpath:
			self.read(fpath)

		self.nx = nx
		self.ny = ny
		self.nw = nw
		self.nz = nz
		self.noise = None
		# storage for full wavelength list from RH (used for full RF calculation)
		self.wave = wave
		self.wavelength = wave
		if spec is not None:
			self.spec = spec
			shape = self.spec.shape
			self.nx = shape[0]
			self.ny = shape[1]
			self.nw = shape[2]
		elif (nx is not None) and (ny is not None) and (nw is not None):
			self.spec = np.zeros((nx, ny, nw, 4))
			self.spec[:,:,:,:] = np.nan
			self.wavelength = np.zeros(nw)
			self.wavelength[:] = np.nan
			if self.nz is not None:
				self.J = np.zeros((nx, ny, nw, nz))

		self.xmin, self.xmax = 0, None
		self.ymin, self.ymax = 0, None

	def list_spectra(self):
		for idx in range(self.nx):
			for idy in range(self.ny):
				yield self.spec[idx,idy]

	def generate_list(self):
		self.spectrum_list = [spec for spec in self.list_spectra()]

	def add_noise(self, noise):
		self.noise = noise

		self.mean = np.nanmean(np.max(self.spec[...,0], axis=2))
		wavs_dependent_factor = 1 # np.sqrt(self.spec[...,0] / self.mean)
		
		gauss_noise = np.random.normal(0, self.noise, size=(self.nx, self.ny, self.nw, 4))
		SI_cont_err = gauss_noise * self.mean

		self.spec[...,0] += wavs_dependent_factor * SI_cont_err[...,0]
		self.spec[...,1] += wavs_dependent_factor * SI_cont_err[...,1]
		self.spec[...,2] += wavs_dependent_factor * SI_cont_err[...,2]
		self.spec[...,3] += wavs_dependent_factor * SI_cont_err[...,3]

	def get_kernel_sigma(self, vmac):
		"""
		Get Gaussian kernel standard deviation based on given macro-turbulent velocity (in km/s).
		"""
		step = self.wavelength[1] - self.wavelength[0]
		return vmac*1e3 / globin.LIGHT_SPEED * (self.wavelength[0] + self.wavelength[-1])*0.5 / step

	def get_kernel(self, vmac, order=0):
		# we assume equidistant seprataion in wavelength grid
		kernel_sigma = self.get_kernel_sigma(vmac)
		radius = int(4*kernel_sigma + 0.5)
		x = np.arange(-radius, radius+1)
		phi = np.exp(-x**2/kernel_sigma**2)
		kernel = phi/phi.sum()

		if order==0:
			# Gaussian kernel
			return kernel[::-1]
		elif order==1:
			# first derivative of Gaussian kernel with respect to standard deviation
			step = self.wavelength[1] - self.wavelength[0]
			kernel *= (2*x**2/kernel_sigma**2 - 1) * 1 / kernel_sigma / step
			return kernel[::-1]
		else:
			raise ValueError(f"Kernel order {order} not supported.")

	def broaden_spectra(self, vmac, flag, n_thread=1):
		if vmac==0:
			return

		# get Gaussian kernel
		kernel = self.get_kernel(vmac, order=0)

		# get only sample of spectra that we want to convolve
		# (no need to do it in every pixel during inversion if
		# we have not updated parameters)
		indx, indy = np.where(flag==1)
		args = zip(self.spec[indx,indy], [kernel]*len(indx))

		with mp.Pool(n_thread) as pool:
			results = pool.map(func=_broaden_spectra, iterable=args)

		results = np.array(results)
		self.spec[indx, indy] = results

	def instrumental_broadening(self, flag, n_thread, kernel=None, R=None):
		if R is not None:
			vinst = globin.LIGHT_SPEED/R/1e3 # [km/s]
			self.broaden_spectra(vinst, flag, n_thread)
		if kernel is not None:
			# get only sample of spectra that we want to convolve
			# (no need to do it in every pixel during inversion if
			# we have not updated parameters)
			indx, indy = np.where(flag==1)
			args = zip(self.spec[indx,indy], [kernel]*len(indx))

			with mp.Pool(n_thread) as pool:
				results = pool.map(func=_broaden_spectra, iterable=args)

			results = np.array(results)
			self.spec[indx,indy] = results

	def norm(self):
		if (globin.norm) and (globin.Icont is not None):
			for idx in range(self.nx):
				for idy in range(self.ny):
			# 		# sI_cont = rh_obj.int[ind_min]
			# 		# sI_cont = np.max(rh_obj.int[ind_min:ind_max])
			# 		# k = (self.spec[idx,idy,-1,0] - self.spec[idx,idy,0,0]) / (self.wavelength[-1] - self.wavelength[0])
			# 		# n = self.spec[idx,idy,-1,0] - k*self.wavelength[-1]
			# 		# sI_cont = k*self.wavelength + n
			# 		# sI_cont = np.repeat(sI_cont[..., np.newaxis], 4, axis=-1)
			# 		sI_cont = 1e-8#np.max(self.spec[idx,idy,:,0])
					# sI_cont = np.mean(self.spec[idx,idy,:,0])
					self.spec[idx,idy] /= globin.Icont

	def mean(self):
		"""
		Return an average spectrum over observed field-of-view.
		"""
		if globin.mean:
			return np.mean(self.spec, axis=(0,1))

	def mean_spectrum(self):
		if globin.mean:
			weights = np.zeros((self.nx, self.ny, self.nw, 4))
			for idx in range(self.nx):
				for idy in range(self.ny):
					vmac = globin.mac_vel[idx*self.ny + idy]
					if vmac!=0:
						kernel_sigma = self.get_kernel_sigma(vmac)
						radius = int(4*kernel_sigma + 0.5)
						x = np.arange(-radius, radius+1)
						phi = np.exp(-x**2/kernel_sigma**2)
						kernel = phi/phi.sum()
						# since we are correlating, we need to reverse the order of data
						kernel = kernel[::-1]

						for sID in range(4):
							self.spec[idx,idy,:,sID] = correlate1d(self.spec[idx,idy,:,sID], kernel)

					weights[idx,idy] = globin.filling_factor[idx*self.ny + idy]

			mean = np.average(self.spec, axis=(0,1), weights=weights)

			self.nx, self.ny = 1, 1
			self.spec = np.zeros((self.nx, self.ny, self.nw, 4))
			self.spec[0,0] = mean

	def save(self, fpath, wavelength):
		"""
		Get list of spectra computed for every pixel and store them in fits file.
		Spectra for pixels which are not computed, we set to 0.

		Parameters:
		---------------
		spectra : list
			List of dictionaries with fields 'spectra', 'idx' and 'idy'.
			Spectrum in dictionary is Rhout() class object and 'idx' and 'idy'
			are pixel positions in the atmosphere cube to which given spectrum
			coresponds.

		fpath : string (optional)
			Name of the output file. Default is "spectra.fits".

		Returns:
		---------------
		spectra : ndarray
			Spectral cube with dimensions (nx, ny, nlam, 5) which stores
			the wavelength and Stokes vector for each pixel in atmosphere cube.

		Changes:
		---------------
		make fits header with additional info
		"""
		data = np.zeros((self.nx, self.ny, len(wavelength), 5))
		data[...,0] = wavelength
		data[...,1:] = self.spec
		
		primary = fits.PrimaryHDU(data)

		primary.header["XMIN"] = self.xmin+1
		if self.xmax is None:
			self.xmax = self.nx
		primary.header["XMAX"] = self.xmax
		primary.header["YMIN"] = self.ymin+1
		if self.ymax is None:
			self.ymax = self.ny
		primary.header["YMAX"] = self.ymax

		primary.header["NX"] = self.nx
		primary.header["NY"] = self.ny
		primary.header["NW"] = len(wavelength)
		
		if self.noise is None:
			self.noise = -1
		primary.header["noise"] = ("{:4.3e}".format(self.noise), "assumed noise level; -1 when we do not know")

		hdulist = fits.HDUList([primary])
		hdulist.writeto(fpath, overwrite=True)

	def read(self, fpath):
		return

class Observation(Spectrum):
	"""
	Class object for storing observations.

	We assume currently that wavelength grid is given along with full Stokes
	observations. Assumed dimension of read data is (nx, ny, nw, 5) where first
	row in last axis is reserved for wavelength and rest 4 are for Stokes vector.
	"""

	def __init__(self, fpath, obs_range=[0,None,0,None], spec_type="globin"):
		super().__init__()
		ftype = fpath.split(".")[-1]

		self.xmin = obs_range[0]
		self.xmax = obs_range[1]
		self.ymin = obs_range[2]
		self.ymax = obs_range[3]

		if ftype=="txt" or ftype=="dat":
			print("  Currently unsupported type of observation file.")
			print("    Supported only is .fits/.fit file format.")
			sys.exit()

		if ftype=="fits" or ftype=="fit":
			if spec_type=="globin":
				self.read_fits(fpath, obs_range)
			if spec_type=="spinor":
				self.read_spinor(fpath, obs_range)

	def read_fits(self, fpath, obs_range):
		hdu = fits.open(fpath)[0]
		self.header = hdu.header

		xmin, xmax, ymin, ymax = obs_range
		data = np.array(hdu.data[xmin:xmax,ymin:ymax], dtype=np.float64)

		# we assume that wavelngth is same for every pixel in observation
		self.wavelength = data[0,0,:,0]
		self.spec = data[:,:,:,1:]
		self.nx, self.ny = self.spec.shape[0], self.spec.shape[1]
		self.nw = len(self.wavelength)
		self.shape = self.spec.shape

	def read_spinor(self, fpath, obs_range):
		hdu = fits.open(fpath)
		header = hdu[0].header

		self.spec = hdu[0].data
		self.spec = np.swapaxes(self.spec, 2, 3)
		self.nx, self.ny, self.nw, _ = self.spec.shape

		wave_ref = header["WLREF"]/10
		wave_min = header["WLMIN"]/10 + wave_ref
		wave_max = header["WLMAX"]/10 + wave_ref

		self.wavelength = np.linspace(wave_min, wave_max, num=self.nw)

		self.shape = self.spec.shape

	def interpolate(self, wavs):
		self.nw = len(wavs)
		spectra = np.zeros((self.nx, self.ny, self.nw, 4))

		for idx in range(self.nx):
			for idy in range(self.ny):
				for ids in range(4):
					tck = splrep(self.wavelength, self.spec[idx,idy,:,ids])
					spectra[idx,idy,:,ids] = splev(wavs, tck)

		self.spec = spectra
		self.wavelength = wavs

	def norm(self):
		Spectrum.norm(self)

def _broaden_spectra(args):
	spec, kernel = args

	for ids in range(4):
		spec[...,ids] = correlate1d(spec[...,ids], kernel)

	return spec

def get_Icont(wavelength=500, mu=1.0):
	"""
	Compute the continuum intensity in the given wavelength from FAL C model that
	will be used as a normalization factor for synthetic spectra.
	"""
	import pyrh

	hsrasp = globin.Atmosphere(f"{globin.__path__}/data/hsrasp_multi.fits")
	hsrasp.mu = mu
	hsrasp.norm = False
	hsrasp.RH = pyrh.RH()
	try:
		nw = len(wavelength)
		hsrasp.wavelength_vacuum = wavelength
	except:
		hsrasp.wavelength_vacuum = np.array([wavelength], dtype=np.float64)
	nw = len(hsrasp.wavelength_vacuum)
	
	hsrasp.do_fudge = 0

	# spec = hsrasp._compute_spectra_sequential(arg=(0,0))
	hsrasp.spectra = Spectrum(nx=1, ny=1, nw=nw)
	spec = hsrasp.compute_spectra(np.ones((1,1)))
	icont = spec.spec[:,:,-1,0]
	return icont, spec.spec/icont