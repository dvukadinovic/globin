import sys
from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, correlate1d
import matplotlib.pyplot as plt

import globin

class Spectrum(object):
	"""
	Custom class for storing computed spectra object. It is rebuilt from RH
	class.
	"""
	def __init__(self, nx=None, ny=None, nw=None, spec=None, wave=None, fpath=None):
		if fpath:
			self.read(fpath)

		self.nx = nx
		self.ny = ny
		self.nw = nw
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
			self.spec = np.empty((nx, ny, nw, 4))
			self.spec[:,:,:,:] = np.nan
			self.wavelength = np.empty(nw)
			self.wavelength[:] = np.nan

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
		step = self.wavelength[1] - self.wavelength[0]
		return vmac*1e3 / globin.LIGHT_SPEED * (self.wavelength[0] + self.wavelength[-1])*0.5 / step

	def broaden_spectra(self, vmac):
		if vmac==0:
			return

		# we assume equidistant seprataion
		kernel_sigma = self.get_kernel_sigma(vmac)
		radius = int(4*kernel_sigma + 0.5)
		x = np.arange(-radius, radius+1)
		phi = np.exp(-x**2/kernel_sigma**2)
		kernel = phi/phi.sum()
		# since we are correlating, we need to reverse the order of data
		kernel = kernel[::-1]

		# output = gaussian_filter(spectra, [0,0,kernel_sigma,0], mode="reflect")

		for idx in range(self.nx):
			for idy in range(self.ny):
				for sID in range(4):
					self.spec[idx,idy,:,sID] = correlate1d(self.spec[idx,idy,:,sID], kernel)
					
					# plt.plot(spectra[idx,idy,:,1+sID])
					# plt.plot(output[idx,idy,:,1+sID])
					# plt.show()

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

	def __init__(self, fpath, atm_range=[0,None,0,None]):
		super().__init__()
		ftype = fpath.split(".")[-1]

		self.xmin = atm_range[0]
		self.xmax = atm_range[1]
		self.ymin = atm_range[2]
		self.ymax = atm_range[3]

		if ftype=="txt" or ftype=="dat":
			print("  Currently unsupported type of observation file.")
			print("    Supported only is .fits/.fit file format.")
			sys.exit()

		if ftype=="fits" or ftype=="fit":
			self.read_fits(fpath, atm_range)

	def read_fits(self, fpath, atm_range):
		hdu = fits.open(fpath)[0]
		self.header = hdu.header

		xmin, xmax, ymin, ymax = atm_range
		data = np.array(hdu.data[xmin:xmax,ymin:ymax], dtype=np.float64)

		# we assume that wavelngth is same for every pixel in observation
		self.wavelength = data[0,0,:,0]
		self.spec = data[:,:,:,1:]
		self.nx, self.ny = self.spec.shape[0], self.spec.shape[1]
		self.nw = len(self.wavelength)