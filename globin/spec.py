import sys
from astropy.io import fits
import numpy as np

class Observation(object):
	"""
	Class object for storing observations.

	We assume currently that wavelength grid is given along with full Stokes
	observations. Assumed dimension of read data is (nx, ny, nw, 5) where first
	row in last axis is reserved for wavelength and rest 4 are for Stokes vector.
	"""

	def __init__(self, fpath, atm_range=[0,None,0,None]):
		ftype = fpath.split(".")[-1]
		self.wavelength = None
		
		if ftype=="txt" or ftype=="dat":
			sys.exit("Currently unsupported type of observation file.")

		if ftype=="fits" or ftype=="fit":
			self.read_fits(fpath, atm_range)

	def read_fits(self, fpath, atm_range):
		from scipy.constants import c as LIGHT_SPEED
		hdu = fits.open(fpath)[0]
		self.header = hdu.header
		xmin, xmax, ymin, ymax = atm_range
		self.data = np.array(hdu.data[xmin:xmax,ymin:ymax], dtype=np.float64)
		# we assume that wavelngth is same for every pixel in observation
		self.wavelength = hdu.data[0,0,:,0]
		fact = 1 # LIGHT_SPEED / (self.wavelength*1e-9)**2
		self.spec = self.data[:,:,:,1:]
		for sID in range(4):
			self.spec[:,:,:,sID] *= fact
			self.data[:,:,:,1+sID] *= fact
		self.nx, self.ny = self.spec.shape[0], self.spec.shape[1]

class Spectrum(object):
	"""

	Custom class for storing computed spectra object. It is rebuilt from RH
	class.

	"""
	def __init__(self):
		self.data = None

def plot_stokes():
	pass