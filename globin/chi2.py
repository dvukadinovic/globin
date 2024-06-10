import numpy as np
import scipy.special as sc
from astropy.io import fits

class Chi2(object):
	def __init__(self, fpath=None, nx=None, ny=None, niter=None, chi2=None):
		if fpath is not None:
			self.read(fpath)
		elif (nx is not None) and (ny is not None) and (niter is not None):
			self.chi2 = np.zeros((nx, ny, niter), dtype=np.float64)
			self.nx, self.ny, self.niter = nx, ny, niter

		self.shape = self.chi2.shape

		self.mode = -1
		self.Nlocal_par = -1
		self.Nglobal_par = -1
		self.Nw = -1

		# regularization weight
		self.regularization_weight = 0
		# value of regularization functional
		self.regularization = 0

		if chi2 is not None:
			self.chi2 = chi2

	def read(self, fpath):
		hdu_list = fits.open(fpath)
		hdu = hdu_list[0]
		header = hdu.header
		self.chi2 = hdu.data

		try:
			self.mode = header["MODE"]
			self.Nlocal_par = header["NLOCALP"]
			self.Nglobal_par = header["NGLOBALP"]
			self.Nw = header["NW"]
		except:
			# for the older outputs
			pass		

		try:
			self.nx, self.ny,_ = self.chi2.shape
			self.chi2, self.last_iter = self.get_final_chi2()
		except:
			self.last_iter = None

		try:
			self.full_chi2 = hdu_list[2].data
		except:
			self.full_chi2 = None

		self.nx, self.ny = self.chi2.shape

	def get_final_chi2(self):
		last_iter = np.zeros((self.nx, self.ny))
		best_chi2 = np.zeros((self.nx, self.ny))
		for idx in range(self.nx):
			for idy in range(self.ny):
				inds_non_zero = np.nonzero(self.chi2[idx,idy])[0]
				if inds_non_zero.size==0:
					continue
				last_iter[idx,idy] = inds_non_zero[-1]
				best_chi2[idx,idy] = self.chi2[idx,idy,inds_non_zero[-1]]

		return best_chi2, last_iter

	def per_pixel(self, best_chi2, copy=False):
		if self.mode==1 or self.mode==2:
			return best_chi2

		Natm = self.nx*self.ny
		if self.mode==3:
			Ndof = self.Nw*Natm - self.Nlocal_par*Natm - self.Nglobal_par

		best_chi2 *= Ndof
		best_chi2 /= (self.Nw - self.Nlocal_par - self.Nglobal_par)

		if copy:
			self.chi2 = best_chi2
		else:
			return best_chi2

	def total(self):
		chi2, _ = self.get_final_chi2()
		return np.sum(chi2)

	def save(self, fpath="chi2.fits"):
		# best_chi2, last_iter = self.get_final_chi2()
		# best_chi2 = self.per_pixel(best_chi2)

		primary = fits.PrimaryHDU(self.chi2)
		hdulist = fits.HDUList([primary])

		primary.name = "best_chi2"
		primary.header["NX"] = (self.nx, "number of x atmospheres")
		primary.header["NY"] = (self.ny, "number of y atmospheres")
		primary.header["MODE"] = (self.mode, "inversion mode")
		primary.header["NLOCALP"] = (self.Nlocal_par, "num. of local parameters")
		primary.header["NGLOBALP"] = (self.Nlocal_par, "num. of global parameters")
		primary.header["NW"] = (self.Nw, "number of wavelenghts (for full Stokes")

		# container for last iteration number for each pixel
		# iter_hdu = fits.ImageHDU(last_iter)
		# iter_hdu.name = "iteration_num"
		# hdulist.append(iter_hdu)

		# container for all the chi2 values (for every pixel in every iteration)
		all_hdu = fits.ImageHDU(self.chi2)
		all_hdu.name = "All chi2 values"
		hdulist.append(all_hdu)

		# save
		hdulist.writeto(fpath, overwrite=True)

def compute_chi2(obs, inv, weights=np.array([1,1,1,1]), noise=1e-3, npar=0, total=False):
	"""
	Get the chi^2 value between observed O_i and inverted S_i spectrum as:

	chi^2 = 1/Ndof * sum_i [(w_i/sig_i)^2 * (O_i - S_i)^2 ]

	Ndof = 4*Nw - Npar
	w_i --> wavelength dependent weight
	sig_i --> wavelength depedent noise

	Parameters:
	-----------
	obs : globin.spec.Spectrum()
		observed spectrum.
	inv : globin.spec.Spectrum()
		inverted spectrum.
	weights : ndarray
		array containing weights for each Stokes component.
	noise : float
		assumed noise level of observations.
	npar : int
		number of free parameters in the inference
	total : bool, False (default)
		sum chi^2 value through all the pixels.

	Return:
	-------
	chi2 : float/ndarray
		returns float if 'total=True', otherwise its ndarray
	"""
	noise = obs._get_weights_by_noise(noise)

	diff = obs.spec - inv.spec
	diff *= weights
	diff /= noise
	chi2 = np.sum(diff**2, axis=(2,3))

	Ndof = np.count_nonzero(weights)*obs.nw*obs.nx*obs.ny - npar

	if total:
		chi2 = np.sum(chi2)

	chi2 /= Ndof

	return chi2

def critical_chi2(Ndof, p):
	"""
	chi2 range for estimating confidence interval of inversion parameters

	returns chi2_crit for which p = P(chi2(ndof) < chi2_crit)

	https://stackoverflow.com/questions/60423364/how-to-calculate-the-critical-chi-square-value-using-python
	"""
	return 2 * sc.gammaincinv(Ndof/2, p)