import sys
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.ndimage import gaussian_filter, gaussian_filter1d, correlate1d
from scipy.interpolate import splev, splrep, interp1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

import globin

from .utils import extend
from .utils import congrid
from .utils import get_first_larger_divisor

class Spectrum(object):
	"""
	Custom class for storing computed spectra object. It is rebuilt from RH
	class.
	"""
	def __init__(self, nx=None, ny=None, nw=None, spec=None, wave=None, fpath=None, nz=None):
		self.Icont = None

		self.nx = nx
		self.ny = ny
		self.nw = nw
		self.nz = nz
		self.noise = None
		# storage for full wavelength list from RH (used for full RF calculation)
		self.wave = wave
		self.wavelength = wave

		if fpath:
			self.read(fpath)

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
			if self.nz is not None:
				self.J = np.empty((nx, ny, nw, nz))

		self.xmin, self.xmax = 0, None
		self.ymin, self.ymax = 0, None

	@property
	def I(self):
		return self.spec[...,0]

	@property
	def Q(self):
		return self.spec[...,1]

	@property
	def U(self):
		return self.spec[...,2]

	@property
	def V(self):
		return self.spec[...,3]

	@property
	def LinP(self):
		return np.sqrt(self.Q**2 + self.U**2)

	@property
	def TotalP(self):		
		return np.sqrt(self.Q**2 + self.U**2 + self.V**2)

	def integrate_V(self):
		StokesV_int = np.empty((self.nx, self.ny))
		ones = -1*np.ones(self.nw)
		for idx in range(self.nx):
			for idy in range(self.ny):
				StokesV = self.V[idx,idy]
				tck = splrep(self.wavelength, StokesV)
				der_StokesI = splev(self.wavelength, tck, der=1)
				flip = der_StokesI<0
				StokesV[flip] *= ones[flip]
				StokesV_int[idx,idy] = np.sum(StokesV) / self.nw
		return StokesV_int

	def list_spectra(self):
		for idx in range(self.nx):
			for idy in range(self.ny):
				yield self.spec[idx,idy]

	def is_array_valid(self):
		"""
		Check if there is NaN's in the spectrum.
		"""
		valid = True
		for idx in range(self.nx):
			for idy in range(self.ny):
				if np.isnan(self.spec[idx,idy]).any():
					print(idx, idy)
					valid &= False

		return valid

	def get_spectra(self, indx, indy):
		nx = 1
		ny = len(indy)

		new = Spectrum(nx=nx, ny=ny, nw=self.nw)

		new.spec[0,:,:,:] = self.spec[indx,indy]
		new.wavelength = self.wavelength

		return new

	def generate_list(self):
		self.spectrum_list = [spec for spec in self.list_spectra()]

	def add_noise(self, noise):
		self.noise = noise

		self.mean = np.mean(self.spec[...,0,0])
		wavs_dependent_factor = 1 # np.sqrt(self.spec[...,0] / self.mean)
		
		gauss_noise = np.random.normal(0, self.noise, size=(self.nx, self.ny, self.nw, 4))
		SI_cont_err = gauss_noise * self.mean

		self.spec[...,0] += wavs_dependent_factor * SI_cont_err[...,0]
		self.spec[...,1] += wavs_dependent_factor * SI_cont_err[...,1]
		self.spec[...,2] += wavs_dependent_factor * SI_cont_err[...,2]
		self.spec[...,3] += wavs_dependent_factor * SI_cont_err[...,3]

	def denoise(self, line_locations, width=20):
		# mean_profile = np.mean(self.I, axis=(0,1))

		# idx, idy = 0, 0
		# O = np.empty((4*len(line_locations), width*2 + 1))
		# wave_inds = np.empty(len(line_locations)*(2*width+1), dtype=np.int32)
		# low, up = 0, 2*width+1
		# for idl,line in enumerate(line_locations):
		# 	_ind0 = np.argmin(np.abs(self.wavelength - line))
		# 	ind0 = np.argmin(mean_profile[_ind0-width:_ind0+width]) + _ind0 - width
		# 	wave_inds[low:up] = np.linspace(ind0-width, ind0+width, num=2*width+1, dtype=np.int32)
		# 	low = up
		# 	up += 2*width+1
		# # 	O[idl*4+0] = self.spec[idx,idy, ind0-width:ind0+width+1, 0]
		# # 	O[idl*4+1] = self.spec[idx,idy, ind0-width:ind0+width+1, 1]
		# # 	O[idl*4+2] = self.spec[idx,idy, ind0-width:ind0+width+1, 2]
		# # 	O[idl*4+3] = self.spec[idx,idy, ind0-width:ind0+width+1, 3]

		# _ind0 = np.argmin(np.abs(self.wavelength - line_locations[4]))
		# ind0 = np.argmin(mean_profile[_ind0-width:_ind0+width]) + _ind0 - width
		# O = self.spec[:,:20, ind0-width:ind0+width+1,:]
		
		O = self.spec[:, :,:,1:]
		O = np.swapaxes(O, 2, 3)
		shape = O.shape
		O = O.reshape(shape[0]*shape[1]*shape[2], O.shape[3], order="F")

		pca = PCA()

		X = np.dot(O.T, O)
		pca.fit(X)
		eigen_vectors = pca.components_
		eigen_values = pca.singular_values_
		eigen_values_normed = np.cumsum(eigen_values) / np.sum(eigen_values)

		# inds = pca.explained_variance_ratio_*100 > 5e-3
		# print(pca.explained_variance_ratio_*100)

		# same for random noise		
		RNG = np.random.default_rng()
		O_noise = RNG.normal(loc=0, scale=2e-3, size=O.shape)
		X_noise = np.dot(O_noise.T, O_noise)

		pca.fit(X_noise)
		_eigen_values = pca.singular_values_
		_eigen_values_normed = np.cumsum(_eigen_values) / np.sum(_eigen_values)

		eigen_value_ratio = eigen_values_normed/_eigen_values_normed

		inds = eigen_value_ratio > 1.2

		plt.plot(eigen_values_normed)
		plt.plot(_eigen_values_normed*1.2)
		
		# plt.plot(eigen_value_ratio)
		# plt.axhline(y=1.2)
		plt.yscale("log")
		plt.show()

		# plt.plot(eigen_vectors[10:13].T)
		# plt.show()
		# return

		# reconstruct
		# flag = eigen_values_normed > 3*_eigen_values_normed
		
		C = np.dot(O, eigen_vectors[:15].T)
		O_denoised = np.dot(C, eigen_vectors[:15])
		O_denoised = O_denoised.reshape(*shape, order="F")
		O_denoised = np.swapaxes(O_denoised, 2, 3)
		spec_denoised = np.zeros(self.spec.shape)
		spec_denoised[...,1:] = O_denoised
		spec_denoised[...,:1] = self.spec[...,:1]
		# print(eigen_value_ratio[5])

		idx, idy = 200, 50
		globin.plot_spectra(spec_denoised[idx,idy], self.wavelength, 
			norm=True,
			aspect=3)
		# globin.plot_spectra(self.spec[idx,idy], self.wavelength, 
		# 	inv=[spec_denoised[idx,idy]],
		# 	aspect=3,
		# 	norm=True,
		# 	labels=["original", "denoised"])
		plt.legend()
		plt.show()

		# noise estimate
		# noise = 1e-3
		# noise_stokes = np.ones(shape)
		# StokesI_cont = np.quantile(self.I[:,:5], 0.95, axis=2)
		# noise_stokes = np.einsum("ijkl,ij->ijkl", noise_stokes, noise*StokesI_cont)
		# noise_stokes = np.swapaxes(noise_stokes, 2, 3)

		# weights = np.array([1,7,7,5])
		# weights = np.array([1,10,10,7])

		# # reconstruct
		# Nc = 20
		
		# O_denoised = np.empty((Nc, *shape))
		# O_denoised = np.swapaxes(O_denoised, 3, 4)
		# print(O_denoised.shape)

		# C = np.dot(O, eigen_vectors[:1].T)
		# _O_denoised = np.dot(C, eigen_vectors[:1])
		# _O_denoised = _O_denoised.reshape(*shape, order="F")
		# O_denoised[0] = np.swapaxes(_O_denoised, 2, 3)
		
		# chi2 = np.ones(Nc)
		# for idc in range(1,Nc):
		# 	C = np.dot(O, eigen_vectors[:idc].T)
		# 	_O_denoised = np.dot(C, eigen_vectors[:idc])
		# 	_O_denoised = _O_denoised.reshape(*shape, order="F")
		# 	O_denoised[idc] = np.swapaxes(_O_denoised, 2, 3)
			
		# 	diff = O_denoised[idc] - O_denoised[idc-1]
		# 	diff *= weights
		# 	diff /= noise_stokes
		# 	diff *= np.sqrt(2)
		# 	chi2[idc] = np.sum(diff**2)/len(O_denoised[idc])
		# 	if idc>=3:
		# 		rel = np.abs(chi2[idc]/chi2[idc-1] - 1)
		# 		print(f"{idc:>2d}  {chi2[idc]:.3e}  {rel:.3f}")
		# 	else:
		# 		print(f"{idc:>2d}  {chi2[idc]:.3e}")

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

		if order==0:
			# Gaussian kernel
			kernel = phi/phi.sum()
			return kernel
		elif order==1:
			# first derivative of Gaussian kernel with respect to standard deviation
			# kernel = phi/phi.sum() * 2 * x**2/kernel_sigma**3
			kernel = phi/phi.sum()
			step = self.wavelength[1] - self.wavelength[0]
			kernel *= (2*x**2/kernel_sigma**2 - 1) * 1 / kernel_sigma / step
			return kernel
		else:
			raise ValueError(f"Kernel order {order} not supported.")

	@globin.utils.timeit
	def broaden_spectra(self, vmac, flag=None, n_thread=1):
		if vmac==0:
			return

		# get Gaussian kernel
		kernel = self.get_kernel(vmac, order=0)

		# get only sample of spectra that we want to convolve
		# (no need to do it in every pixel during inversion if
		# we have not updated parameters)
		if flag is None:
			flag = np.ones((self.nx, self.ny), dtype=np.int32)
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

	def add_stray_light(self, mode, stray_light, sl_spectrum=None):
		"""
		Add the stray light contamination to the spectra.
		"""
		if sl_spectrum is not None:
			stray_type = "spectrum"

		for idx in range(self.nx):
			for idy in range(self.ny):
				if mode==1 or mode==2:
					stray_factor = stray_light[idx,idy]
				elif mode==3:
					stray_factor = stray_light
					# if self.invert_stray:
					# 	stray_factor = stray_light
					# else:
					# 	stray_factor = stray_light[idx,idy]
				else:
					raise ValueError(f"Unknown mode {mode} for stray light contribution. Choose one from 1,2 or 3.")
				
				if stray_type=="spectrum":
					self.spec[idx,idy] = stray_factor * sl_spectrum + (1-stray_factor) * self.spec[idx,idy]
				if stray_type=="gray":
					self.spec[idx,idy,:,0] = stray_factor + (1-stray_factor) * self.spec[idx,idy,:,0]
					self.spec[idx,idy,:,1] = (1-stray_factor) * self.spec[idx,idy,:,1]
					self.spec[idx,idy,:,2] = (1-stray_factor) * self.spec[idx,idy,:,2]
					self.spec[idx,idy,:,3] = (1-stray_factor) * self.spec[idx,idy,:,3]

	def norm(self, degree=7, roi=None):
		# if (globin.norm) and (globin.Icont is not None):
		# 	for idx in range(self.nx):
		# 		for idy in range(self.ny):
		# 	# 		# sI_cont = rh_obj.int[ind_min]
		# 	# 		# sI_cont = np.max(rh_obj.int[ind_min:ind_max])
		# 	# 		# k = (self.spec[idx,idy,-1,0] - self.spec[idx,idy,0,0]) / (self.wavelength[-1] - self.wavelength[0])
		# 	# 		# n = self.spec[idx,idy,-1,0] - k*self.wavelength[-1]
		# 	# 		# sI_cont = k*self.wavelength + n
		# 	# 		# sI_cont = np.repeat(sI_cont[..., np.newaxis], 4, axis=-1)
		# 	# 		sI_cont = 1e-8#np.max(self.spec[idx,idy,:,0])
		# 			# sI_cont = np.mean(self.spec[idx,idy,:,0])
		# 			self.spec[idx,idy] /= globin.Icont
		"""
		Fit a high order polynomial to the continuum level (divided in 'degree'+1 number of bands) and use this to normalize spectra.
		"""
		nbands = get_first_larger_divisor(self.nw, 2*degree)
		bands = np.split(self.I, nbands, axis=2)
		wbands = np.split(self.wavelength, nbands)

		x = np.empty(nbands)
		Ics = np.empty((nbands, self.nx, self.ny))
		
		for idb in range(nbands):
			Ics[idb] = np.quantile(bands[idb], 0.95, axis=2)
			x[idb] = np.mean(wbands[idb])

		self.Ic = np.empty((self.nx, self.ny))

		for idy in range(self.ny):
			coeffs = np.polyfit(x, Ics[...,idy], degree)
			for idx in range(self.nx):
				continuum = np.polyval(coeffs[...,idx], self.wavelength)
				self.Ic[idx,idy] = np.mean(continuum)
				self.spec[idx,idy] /= continuum[:, np.newaxis]
				self.spec[idx,idy] *= self.Ic[idx,idy]

		if roi is not None:
			Icont = np.mean(self.Ic[roi[0],roi[1]])
		else:
			Icont = np.quantile(self.Ic, 0.95)
		self.Icont = Icont
		self.spec /= Icont

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

	@globin.utils.timeit
	def save(self, fpath, wavelength=None, spec_type="globin"):
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
		if spec_type=="globin":
			data = np.empty((self.nx, self.ny, len(self.wavelength), 5))
			data[...,0] = self.wavelength
			data[...,1:] = self.spec
		elif spec_type=="hinode":
			data = np.swapaxes(self.spec, 2, 3) * self.Icont
		else:
			raise ValueError(f"'{spec_type}' as a spectra type is not supported. Choose one from 'globin' or 'hinode'.")

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
		primary.header["NW"] = len(self.wavelength)
		
		if self.noise is None:
			self.noise = -1
		primary.header["noise"] = ("{:4.3e}".format(self.noise), "assumed noise level; -1 when not specified")

		hdulist = fits.HDUList([primary])
		
		if spec_type=="hinode":
			#--- wavelengths
			par_hdu = fits.ImageHDU(self.wavelength*10)
			par_hdu.name = "wavelength"
			par_hdu.header["UNIT"] = "Angstrom"

			hdulist.append(par_hdu)

			#--- Ic
			try:
				par_hdu = fits.ImageHDU(self.Ic)
				par_hdu.name = "continuum_intensity"
				par_hdu.header["IC_HSRA"] = np.mean(self.Icont)

				hdulist.append(par_hdu)
			except:
				raise ValueError("No information regarding the continuum intensity. Unable to save the spectra of 'hinode' type.")

		hdulist.writeto(fpath, overwrite=True)

	def read(self, fpath):
		return

	@globin.utils.timeit
	def interpolate(self, wave_out, n_thread=1, fill_value="extrapolate"):
		"""
		Interpolate the spectrum on given 'wave_out' wavelength grid.
		"""
		# if wave_out[0]<self.wavelength[0] or wave_out[-1]>self.wavelength[-1]:
		# 	raise ValueError("Interpolation is outside of the wavelength range.")
		
		# spectra = np.zeros((self.nx, self.ny, len(wave_out), 4))
		_spec = self.spec.reshape(self.nx*self.ny, self.nw, 4)

		args = zip(_spec, [wave_out]*(self.nx*self.ny), [fill_value]*(self.nx*self.ny))

		with mp.Pool(n_thread) as pool:
			results = pool.map(func=self._interpolate, iterable=args)

		results = np.array(results)
		self.nw = len(wave_out)
		self.wavelength = wave_out
		self.spec = results.reshape(self.nx, self.ny, self.nw, 4)

	def _interpolate(self, args):
		spec_in, wave_out, fill_value = args

		spec_out = np.zeros((len(wave_out), 4))

		for ids in range(4):
			spec_out[:,ids] = interp1d(self.wavelength, spec_in[:,ids], kind=3, fill_value=fill_value)(wave_out)

		return spec_out

	def extract_wavelength_region(self, lmin, lmax):
		ind_min = np.argmin(np.abs(self.wavelength - lmin))
		ind_max = np.argmin(np.abs(self.wavelength - lmax))+1

		self.wavelength = self.wavelength[ind_min:ind_max]
		self.spec = self.spec[:,:,ind_min:ind_max,:]
		self.shape = self.spec.shape
		self.nx, self.ny, self.nw, _ = self.spec.shape

	def extract(self, slice_x, slice_y):
		dic = self.__dict__
		keys = dic.keys()

		idx_min, idx_max = slice_x
		idy_min, idy_max = slice_y

		new_spec = globin.Spectrum()

		for key in keys:
			if key=="spec":
				new_spec.spec = self.spec[idx_min:idx_max, idy_min:idy_max]
				new_spec.shape = new_spec.spec.shape
				new_spec.nx, new_spec.ny, _, _ = new_spec.shape
			elif key=="Ic":
				new_spec.Ic = self.Ic[idx_min:idx_max, idy_min:idy_max]
			elif key in ["nx", "ny", "shape"]:
				pass
			else:
				setattr(new_spec, key, dic[key])

		return new_spec

	def wavelength_rebinning(self, binning):
		if binning==1:
			return

		if binning<0:
			raise ValueError("globin.spec.wavelength_rebinning :: Binning factor must be larger than 0.")

		self.wavelength = congrid(self.wavelength[:, np.newaxis], [self.nw//binning, 1], method="neighbour", centre=True)[:,0]

		old_spec = np.copy(self.spec)
		self.spec = np.empty((self.nx, self.ny, len(self.wavelength), 4), dtype=np.float64)
		for idx in range(self.nx):
			for idy in range(self.ny):
				self.spec[idx,idy] = congrid(old_spec[idx,idy], [self.nw//binning, 4], method="linear", centre=True)

		self.shape = self.spec.shape
		self.nx, self.ny, self.nw, _ = self.spec.shape

	def spatial_rebinning(self, binning):
		if binning==1:
			return

		if binning<0:
			raise ValueError("globin.spec.wavelength_rebinning :: Binning factor must be larger than 0.")

		old_spec = np.copy(self.spec)
		self.spec = np.empty((self.nx//2, self.ny//2, self.nw, 4), dtype=np.float64)
		for idw in range(self.nw):
			for ids in range(4):
				self.spec[:,:,idw,ids] = congrid(old_spec[:,:,idw,ids], [self.nx//2, self.ny//2], method="linear", centre=True)

		self.shape = self.spec.shape
		self.nx, self.ny, _, _ = self.spec.shape

	def mask(self, mask):
		"""
		Mask the Stokes spectrum at different wavelengths by multiplying them
		with the mask.

		Preferably, mask should be binary: 0 if not considered and 1 if
		considered. The masked values are converted to np.nan values.

		Parameters:
		---------------
		mask : numpy.ndarray 
			array of shape (4, nw) containing values 0 and 1 for each Stokes
			and each wavelength.
		"""

		for ids in range(4):
			self.spec[...,ids] *= mask[ids]

		self.spec[self.spec==0] = np.nan

class Observation(Spectrum):
	"""
	Class object for storing observations.

	We assume currently that wavelength grid is given along with full Stokes
	observations. Assumed dimension of read data is (nx, ny, nw, 5) where first
	row in last axis is reserved for wavelength and rest 4 are for Stokes vector.
	"""

	def __init__(self, fpath, obs_range=[0,None,0,None], spec_type="globin", verify=True):
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

		if ftype=="spec":
			self.read_1d_spectrum(fpath)

		if ftype=="fits" or ftype=="fit":
			if spec_type=="globin":
				self.read_fits(fpath, obs_range)
			if spec_type=="spinor":
				self.read_spinor(fpath, obs_range)
			if spec_type=="hinode":
				self.read_hinode(fpath, obs_range)

			if verify:
				if not self.is_array_valid():
					raise ValueError(f"Spectrum {fpath} contains NaNs. Check the data.")
		else:
			raise IOError("We cannot recognize the observation file type.")

	@globin.utils.timeit
	def read_fits(self, fpath, obs_range):
		hdu = fits.open(fpath)[0]
		self.header = hdu.header

		xmin, xmax, ymin, ymax = obs_range
		data = hdu.data[xmin:xmax,ymin:ymax]

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

	def read_hinode(self, fpath, obs_range):
		hdu = fits.open(fpath)

		if len(hdu)!=3:
			raise IndexError("Wrong number of headers. Header list does not contain 3 extensions (including primary).")

		self.header = hdu[0].header

		xmin, xmax, ymin, ymax = obs_range

		self.Ic = hdu[2].data

		self.Icont = float(hdu[2].header["IC_HSRA"])
		
		# we assume that wavelength is same for every pixel in observation
		self.wavelength = hdu[1].data/10 # [A --> nm]
		# data = np.array(hdu[0].data[xmin:xmax,ymin:ymax], dtype=np.float64)
		data = hdu[0].data[xmin:xmax, ymin:ymax]
		
		# idx, idy = 120, 52
		# plt.plot(data[idx,idy,0]/self.icont)
		# plt.axhline(y=hdu[2].data[idx,idy]/self.icont)
		# plt.show()
		
		nx, ny, ns, nw = data.shape
		self.spec = np.swapaxes(data, 2, 3)
		self.spec /= self.Icont
		self.nx, self.ny = self.spec.shape[0], self.spec.shape[1]
		self.nw = len(self.wavelength)
		self.shape = self.spec.shape

	def read_1d_spectrum(self, fpath):
		spectrum = np.loadtxt(fpath)
		self.wavelength = spectrum[0]
		self.nw = len(self.wavelength)
		self.spec = np.zeros((1,1,self.nw,4))
		self.spec[0,0] = spectrum[1:]
		self.nx, self.ny = 1, 1
		self.Ic = self.spec[0,0,0,0]
		self.Icont = np.zeros((self.nx, self.ny))
		self.Icont[:,:] = self.Ic
		self.shape = (self.nx, self.ny, self.nw, 4)

def _broaden_spectra(args):
	spec, kernel = args

	N = len(kernel)
	for ids in range(4):
		aux = extend(spec[:,ids], N)
		spec[:,ids] = np.convolve(aux, kernel, mode="same")[N:-N]

	return spec