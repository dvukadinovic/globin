import sys
from astropy.io import fits

class Observation(object):

	"""
	Class object for storing observations.

	We assume currently that wavelength grid is given along with full Stokes
	observations. Assumed dimension of read data is (nx, ny, nw, 5) where first
	row in last axis is reserved for wavelength and rest 4 are for Stokes vector.
	"""

	def __init__(self, fpath):
		ftype = fpath.split(".")[-1]
		self.wavelength = None
		
		if ftype=="txt" or ftype=="dat":
			sys.exit("Currently unsupported type of observation file.")

		if ftype=="fits" or ftype=="fit":
			self.read_fits(fpath)

	def read_fits(self, fpath):
		hdu = fits.open(fpath)[0]
		self.header = hdu.header
		self.data = hdu.data
		# we assume that wavelngth is same for every pixel in observation
		# self.wavelength = hdu.data[0,0,:,0]
		self.spec = hdu.data[:,:,:,1:]
		self.nx, self.ny = self.spec.shape[0], self.spec.shape[1]