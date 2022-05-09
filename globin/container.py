from .input import InputData
import pyrh
import multiprocessing as mp
import time
from .spec import Spectrum

import sys

class Globin(object):
	"""
	Main container for all globin and pyrh methods and classes.
	"""
	def __init__(self, rh_logfile=None, rh_quiet=True):
		self.rh_logfile = rh_logfile
		self.rh_quiet = rh_quiet

		#--- parameter scale (calcualted from RFs based on
		#    Cristopher Frutigers' thesis, p.42)
		self.parameter_scale = {}

		self.atmosphere = None
		self.wavelength_vacuum = None
		self.input = InputData(_Globin=self)
		self.spectra = None

	def read_input(self, run_name, globin_input_name="params.input", rh_input_name="keyword.input"):
		self.run_name = run_name
		if (rh_input_name is not None) and (globin_input_name is not None):
			# initialize RH class (cythonized)
			# self.RH = pyrh.RH(input=rh_input_name, logfile=self.rh_logfile, quiet=self.rh_quiet)
			self.RH = pyrh.RH()
			
			# store input parameters in InputData() and Globin()
			self.input.read_input_files(globin_input_name, rh_input_name)
			# self.RH.read_RLK_lines()		
		else:
			if rh_input_name is None:
				print(f"  There is no path for globin input file.")
			if globin_input_name is None:
				print(f"  There is no path for RH input file.")
			sys.exit()

	def compute_spectra(self, parallel=False):
		if (self.atmosphere is not None) and (self.wavelength_vacuum is not None):
			if self.spectra is None:
				self.spectra = Spectrum(nx=self.atmosphere.nx, ny=self.atmosphere.ny, nw=len(self.wavelength_vacuum)+1)
			if parallel:
				spectra = self._compute_spectra_parallel()
				for i_ in range(len(spectra)):
					idx, idy = spectra[i_]["idx"], spectra[i_]["idy"]
					self.spectra.spec[idx,idy,:,0] = spectra[i_]["spec"].I
					self.spectra.spec[idx,idy,:,1] = spectra[i_]["spec"].Q
					self.spectra.spec[idx,idy,:,2] = spectra[i_]["spec"].U
					self.spectra.spec[idx,idy,:,3] = spectra[i_]["spec"].V
			else:
				for idx in range(self.atmosphere.nx):
					for idy in range(self.atmosphere.ny):
						spectrum = self._compute_spectra_sequential((idx, idy))
						self.spectra.spec[idx,idy,:,0] = spectrum["spec"].I
						self.spectra.spec[idx,idy,:,1] = spectrum["spec"].Q
						self.spectra.spec[idx,idy,:,2] = spectrum["spec"].U
						self.spectra.spec[idx,idy,:,3] = spectrum["spec"].V
		else:
			print("Unable to synthesize spectrum. No atmosphere or no wavelength points.")

		return self.spectra

	def _compute_spectra_sequential(self, arg):
		start = time.time()
		idx, idy = arg
		spec = self.RH.compute1d(self.atmosphere.data[idx, idy, 0], self.atmosphere.data[idx, idy, 1], 
							  self.atmosphere.data[idx, idy, 2], self.atmosphere.data[idx, idy, 3], 
							  self.atmosphere.data[idx, idy, 4], self.atmosphere.data[idx, idy, 5]/1e4, 
							  self.atmosphere.data[idx, idy, 6], self.atmosphere.data[idx, idy, 7],
							  self.atmosphere.data[idx, idy, 8:], 0)
		print(f"Finished [{idx},{idy}] in ", time.time() - start)

		return {"spec": spec, "idx" : idx, "idy" : idy}

	def _compute_spectra_parallel(self):
		args = zip(self.atmosphere.idx_meshgrid, self.atmosphere.idy_meshgrid)
		with mp.Pool(self.input.n_thread) as pool:
			spectra = pool.map(func=self._compute_spectra_sequential, iterable=args)

		return spectra

	def invert(save_output=True, verbose=True):
		invert(save_output, verbose)

class Errors:
	def __init__(self):
		pass