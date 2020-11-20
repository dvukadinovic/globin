"""
Contributors:
  Dusan Vukadinovic (DV)

17/09/2020 : started writing the code (readme file, structuring)
12/10/2020 : wrote down reading of atmos in .fits format and sent
			 calculation to different process
##/10/2020 : rewriten class Atmosphere
"""

import subprocess as sp
import multiprocessing as mp
from astropy.io import fits
import numpy as np
import os
import sys
import time
import copy

import globin

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
	def __init__(self, fpath=None, verbose=False):
		self.verbose = verbose
		# dictionary for nodes
		self.nodes = {}
		# dictionary for values of parameters in nodes; when we are inverting for
		# cube atmosphere, each parameters in dictionary will be a matrix with
		# dimensions (nx,ny,nnodes).
		self.values = {}
		
		self.par_id = {"logtau" : 0,
					   "temp"   : 1,
					   "Bx"     : -1,
					   "By"     : -1,
					   "Bz"     : -1,
					   "vz"     : 3,
					   "vmic"   : 4,}

		self.nx = None
		self.ny = None
		self.npar = 14
		self.free_par = 0

		self.logtau = np.linspace(-6, 1, num=71)
		self.nz = len(self.logtau)

		self.data = None
		self.header = None
		self.path = None
		self.atm_name_list = []

		if fpath is not None:
			# by file extension determine atmosphere type / format
			ftype = fpath.split(".")[-1]
			
			# read atmosphere by the type
			if ftype=="dat" or ftype=="txt":
				self.type = "spinor"
				self.read_spinor(fpath)
			# read fits / fit / cube type atmosphere
			elif ftype=="fits" or ftype=="fit":
				self.type = "multi"
				self.read_fits(fpath)
			else:
				print(f"Unsupported type '{ftype}' of atmosphere file.")
				print(" Supported types are: .dat, .txt, .fit(s)")
				sys.exit()

	def __deepcopy__(self, memo):
		new = Atmosphere()
		new.data = copy.deepcopy(self.data)
		new.nx, new.ny, new.npar, new.nz = new.data.shape
		new.atm_name_list = copy.deepcopy(self.atm_name_list)
		return new

	def read_fits(self, fpath):
		try:
			atmos = fits.open(fpath)[0]
		except:
			print(f"Error: Atmosphere file with path '{fpath}'")
			print("       does not exist.\n")
			sys.exit()
		
		self.header = atmos.header
		self.data = atmos.data
		self.nx, self.ny, self.npar, self.nz = self.data.shape
		self.path = fpath

		# type of atmosphere: are we reading MULTI, SPINOR or SIR format
		# self.atm_type = self.header["TYPE"]

		print(f"Read atmosphere '{self.path}' with dimensions:")
		print(f"  (nx, ny, npar, nz) = {self.data.shape}\n")

		self.split_cube()

	def read_spinor(self, fpath):
		self.data = np.loadtxt(fpath, skiprows=1).T
		self.npar = self.data.shape[0]
		self.nz = self.data.shape[1]

	def read_sir(self):
		pass

	def read_multi(self):
		pass

	def get_atmos(self, idx, idy):
		# return atmosphere from cube with given indices 'idx' and 'idy'.
		try:
			return self.data[idx,idy]
		except:
			print(f"Error: Atmosphere index notation does not match")
			print(f"       it's dimension. No atmosphere (x,y) = ({idx},{idy})\n")
			sys.exit()

	def split_cube(self):
		# go through the cube and save each pixel as separate atmosphere
		if not os.path.exists("atmospheres"):
			os.mkdir("atmospheres")
		else:
			# clean directory if it exists (maybe we have atmospheres extracted
			# from some other cube); it takes few miliseconds, so not a big deal
			sp.run("rm atmospheres/*",
				shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)

		# go through each atmosphere and extract it to separate file
		for idx in range(self.nx):
			for idy in range(self.ny):
				atm = self.get_atmos(idx, idy)
				fpath = f"atmospheres/atm_{idx}_{idy}"
				self.atm_name_list.append(fpath)
				write_multi_atmosphere(atm, fpath)

		if self.verbose:
			print("Extracted all atmospheres into folder 'atmospheres'\n")

	def build_from_nodes(self, ref_atm):
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

		import matplotlib.pyplot as plt
		from scipy.interpolate import splev

		# we are not fitting; write it in smarter way! using reference atmosphere
		if self.data is None:
		# we fill here atmosphere with data which will not be interpolated for which
			try:
				self.data = np.zeros((self.nx, self.ny, self.npar, self.nz))
				self.data[:,:,0,:] = self.logtau
				self.interpolate_atmosphere(ref_atm)
				# # electron concentration (interpolate from FAL C model as initial value)
				# self.data[:,:,2,:] = splev(self.logtau, globin.ne_tck)
				# # hydrogen levels population (interpolate from FAL C model as initial
				# # value); when H atom is ACTIVE in RH atoms.input file, exact values are
				# # irrelevant
				# for lvlid in range(6):
				# 	self.data[:,:,8+lvlid,:] = splev(self.logtau, globin.falc_hydrogen_lvls_tcks[lvlid])
			except:
				sys.exit("Could not allocate variable for storing atmosphere built from nodes.")

		for idx in range(self.nx):
			for idy in range(self.ny):
				for parameter in self.nodes:

					x = self.nodes[parameter]
					y = self.values[parameter][idx,idy]
					if parameter=="temp":
						Kn = splev(x[-1], globin.temp_tck, der=1)
					else:
						Kn = 0
					y_new = globin.tools.bezier_spline(x, y, self.logtau, Kn=Kn, degree=globin.interp_degree)
					self.data[idx,idy,self.par_id[parameter],:] = y_new

					# plt.scatter(x,y)
					# plt.plot(self.logtau, y_new)
					# plt.show()

				#--- save interpolated atmosphere to appropriate file
				write_multi_atmosphere(self.data[idx,idy], self.atm_name_list[idx*self.ny + idy])

	def write_atmosphere(self):
		for idx in range(self.nx):
			for idy in range(self.ny):
				write_multi_atmosphere(self.data[idx,idy], self.atm_name_list[idx*self.ny + idy])

	def interpolate_atmosphere(self, ref_atm):
		from scipy.interpolate import interp1d

		x_new = self.logtau
		x_old = ref_atm.data[0,0,0]

		for idx in range(self.nx):
			for idy in range(self.ny):
				for parID in range(1,self.npar):
					fun = interp1d(x_old, ref_atm.data[idx,idy,parID])
					self.data[idx,idy,parID] = fun(x_new)

	def save_cube(self):
		primary = fits.PrimaryHDU(self.data)
		hdulist = fits.HDUList([primary])
		hdulist.writeto("inverted_atmos.fits", overwrite=True)

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
	out.write("* log tau    Temp[K]    n_e[m-3]    v_z[km/s]   v_turb[km/s]\n")

	for i_ in range(nz):
		out.write("  {:+5.4f}    {:6.2f}   {:5.4e}   {:5.4e}   {:5.4e}\n".format(atm[0,i_], atm[1,i_], atm[2,i_], atm[3,i_], atm[4,i_]))

	out.write("*\n")
	out.write("* Hydrogen populations [m-3]\n")
	out.write("*     nh(1)        nh(2)        nh(3)        nh(4)        nh(5)        nh(6)\n")

	for i_ in range(nz):
		out.write("   {:5.4e}   {:5.4e}   {:5.4e}   {:5.4e}   {:5.4e}   {:5.4e}\n".format(atm[8,i_], atm[9,i_], atm[10,i_], atm[11,i_], atm[12,i_], atm[13,i_]))

	out.close()

	# store now and magnetic field vector
	globin.rh.write_B(f"{fpath}.B", atm[5], atm[6], atm[7])

def pool_distribute(arg):
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
	spec_name : string
		File name in which spectrum is written on a disk (read from keyword.input
        file). """
	start = time.time()

	atm_path, spec_name = arg

	#--- for each thread process create separate directory
	pid = mp.current_process()._identity[0]
	set_old_J = True
	if not os.path.exists(f"../pid_{pid}"):
		os.mkdir(f"../pid_{pid}")
		#--- copy *.input files
		sp.run(f"cp *.input ../pid_{pid}", 
			shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)
		set_old_J = False

	# lines = open(f"keyword.input", "r").readlines()
	lines = open(globin.rh_input, "r").readlines()
				
	for i_,line in enumerate(lines):
		line = line.rstrip("\n").replace(" ","")
		# skip blank lines
		if len(line)>0:
			# skip commented lines
			if line[0]!=globin.COMMENT_CHAR:
				line = line.split("=")
				keyword, value = line
				#--- change atmos path in 'keyword.input' file
				if keyword=="ATMOS_FILE":
					lines[i_] = f"  ATMOS_FILE = {globin.cwd}/{atm_path}\n"
				#--- change path for magnetic field
				elif keyword=="STOKES_INPUT":
					lines[i_] = f"  STOKES_INPUT = {globin.cwd}/{atm_path}.B\n"
				#--- set to read old J.dat file
				elif keyword=="STARTING_J":
					if set_old_J:
						lines[i_] = "  STARTING_J      = OLD_J\n"
					else:
						lines[i_] = "  STARTING_J      = NEW_J\n"

	out = open(f"../pid_{pid}/{globin.rh_input}","w")
	out.writelines(lines)
	out.close()

	aux = atm_path.split("_")
	idx, idy = aux[1], aux[2]
	out_file = open(f"{globin.cwd}/logs/log_{idx}_{idy}", "w")
	out = sp.run(f"cd ../pid_{pid}; ../rhf1d -i {globin.rh_input}",
			shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
	stdout = str(out.stdout,"utf-8").split("\n")
	out_file.writelines(str(out.stdout, "utf-8"))
	out_file.close()

	if out.returncode!=0:
		print("*** RH error")
		print(f"    Failed to synthesize spectra for pixel ({idx},{idy}).")
		print()
		for line in stdout[-5:]:
			print("   ", line)
		return None
	
	spec = globin.rh.Rhout(fdir=f"../pid_{pid}", verbose=False)
	spec.read_spectrum(spec_name)

	dt = time.time() - start
	# print("Finished synthesis of '{:}' in {:4.2f} s".format(atm_path, dt))
	
	return {"spectra":spec, "idx":int(idx), "idy":int(idy)}

def compute_spectra(init, atmos, save=False, clean_dirs=False):
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
		spectrum 'spec_name' read from 'keyword.input' file. Rest are dimension
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

	n_thread = init.n_thread
	atm_name_list = atmos.atm_name_list
	spec_name = init.spec_name

	args = [[atm_name,spec_name] for atm_name in atm_name_list]
	#--- make directory in which we will save logs of running RH
	if not os.path.exists("logs"):
		os.mkdir("logs")
	else:
		sp.run("rm logs/*",
			shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)

	#--- distribute the process to threads
	specs = init.pool.map(func=pool_distribute, iterable=args)
	
	#--- exit if all spectra returned from child process are None (failed synthesis)
	kill = True
	for item in specs:
		kill = kill and (item is None)
		if not kill:
			break
	if kill:
		sys.exit("--> Spectrum synthesis on all pixels have failed!")

	spec_cube = save_spectra(specs, init.ref_atm.nx, init.ref_atm.ny, init.spectrum_path, save=save)

	#--- delete thread directories (save them if you want to use previous run J)
	if clean_dirs:
		for threadID in range(n_thread):
			sp.run(f"rm -r ../pid_{threadID+1}",
				shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

	return spec_cube

def save_spectra(specs, nx, ny, fpath="spectra.fits", save=False):
	"""
	Get list of spectra computed for every pixel and store them in fits file.
	Spectra for pixels which are not computed, we set to 0.

	Parameters:
	---------------
	specs : list
		List of dictionaries with fields 'spectra', 'idx' and 'idy'.
		Spectrum in dictionary is Rhout() class object and 'idx' and 'idy'
		are pixel positions in the atmosphere cube to which given spectrum
		coresponds.

	nx : int
		Number of pixels along x-axis in atmosphere cube.

	ny : ind
		Number of pixels along y-axis in atmosphere cube.

	Returns:
	---------------
	spectra : ndarray
		Spectral cube with dimensions (nx, ny, nlam, 5) which stores
		the wavelength and Stokes vector for each pixel in atmosphere cube.

	Changes:
	---------------
	make fits header with additional info
	"""

	created_array = False
	for item in specs:
		if item is not None:
			if not created_array:
				wavs = item["spectra"].wave
				spectra = np.zeros((nx, ny, len(wavs), 5))
				spectra[:,:,:,0] = wavs
				created_array = True
			spec, idx, idy = item["spectra"], item["idx"], item["idy"]
			spectra[idx,idy,:,1] = spec.imu[-1]
			spectra[idx,idy,:,2] = spec.stokes_Q[-1]
			spectra[idx,idy,:,3] = spec.stokes_U[-1]
			spectra[idx,idy,:,4] = spec.stokes_V[-1]

	if save:
		primary = fits.PrimaryHDU(spectra)
		hdulist = fits.HDUList([primary])
		hdulist.writeto(fpath, overwrite=True)

	return spectra

def compute_rfs(init):
	import matplotlib.pyplot as plt

	#--- get inversion parameters for atmosphere and interpolate it on finner grid (original)
	atmos = init.atm
	atmos.build_from_nodes(init.ref_atm)

	spec = compute_spectra(init, atmos)

	# solve HS equation to get electron concentration
	# and store data in atmos.data

	#--- get current iteration atmosphere
	logtau = atmos.logtau # atmos.data[0,0,atmos.par_id["logtau"]]
	dtau = logtau[1] - logtau[0]

	#--- copy current atmosphere to new model atmosphere with +/- perturbation
	model_plus = copy.deepcopy(atmos)

	delta = {"temp" : 1,
			 "Bx"   : 25/1e4, # G --> T
			 "By"   : 25/1e4, # G --> T
			 "Bz"   : 25/1e4, # G --> T
			 "vz"   : 10/1e3, # m/s --> km/s
			 "vmic" : 10/1e3} # m/s --> km/s

	rf = np.zeros((atmos.nx, atmos.ny, atmos.free_par, len(spec[0,0,:,0]), 4))
	# nw = len(init.wavelength)
	# rf = np.zeros((atmos.nx, atmos.ny, atmos.free_par, nw, 4))

	free_par_ID = 0
	for i_,parameter in enumerate(atmos.nodes):
		parID = atmos.par_id[parameter]

		nodes = atmos.nodes[parameter]
		values = atmos.values[parameter]
		
		perturbation = delta[parameter]

		for nodeID in range(len(nodes)):
			zID = np.argmin(abs(logtau-nodes[nodeID]))

			model_plus.data[:,:,parID,zID] += perturbation
			model_plus.write_atmosphere()
			spec_plus = compute_spectra(init, model_plus, clean_dirs=False)

			# ind_min = np.argmin(abs(spec_plus[0,0,:,0] - init.wavelength[0]))
			# ind_max = np.argmin(abs(spec_plus[0,0,:,0] - init.wavelength[-1]))+1
			# ind_max = ind_min + nw
			
			# diff = spec_plus[:,:,ind_min:ind_max,1:] - spec[:,:,ind_min:ind_max,1:]
			diff = spec_plus[:,:,:,1:] - spec[:,:,:,1:]

			rf[:,:,free_par_ID,:,:] = diff / perturbation # / dtau
			free_par_ID += 1
			
			# remove perturbation from data
			model_plus.data[:,:,parID,zID] -= perturbation

	# fits.writeto("rf.fits", rf, overwrite=True)
	# print("RF done!\n\n")

	return rf, spec# [:,:,:,1:]

def compute_full_rf(init):
	import matplotlib.pyplot as plt

	#--- get inversion parameters for atmosphere and interpolate it on finner grid (original)
	atmos = init.ref_atm
	spec = compute_spectra(init, atmos)

	# solve HS equation to get electron concentration
	# and store data in atmos.data

	#--- get current iteration atmosphere
	logtau = atmos.logtau # atmos.data[0,0,atmos.par_id["logtau"]]
	dtau = logtau[1] - logtau[0]

	#--- copy current atmosphere to new model atmosphere with +/- perturbation
	model_plus = copy.deepcopy(atmos)
	model_minus = copy.deepcopy(atmos)

	delta = {"temp" : 1,
			 "Bx"   : 25/1e4, # G --> T
			 "By"   : 25/1e4, # G --> T
			 "Bz"   : 25/1e4, # G --> T
			 "vz"   : 10/1e3, # m/s --> km/s
			 "vmic" : 10/1e3} # m/s --> km/s

	rf = np.zeros((atmos.nx, atmos.ny, 1, atmos.nz, len(init.wavelength), 4))

	parameter = "vz"
	perturbation = delta[parameter]
	parID = atmos.par_id[parameter]

	for zID in range(atmos.nz):
		model_plus.data[:,:,parID,zID] += perturbation
		model_plus.write_atmosphere()
		spec_plus = compute_spectra(init, model_plus, clean_dirs=False)

		ind_min = np.argmin(abs(spec_plus[0,0,:,0] - init.wavelength[0]))
		ind_max = np.argmin(abs(spec_plus[0,0,:,0] - init.wavelength[-1]))
		
		# diff = spec_plus[:,:,ind_min:ind_max,1:] - spec_minus[:,:,ind_min:ind_max,1:]
		diff = spec_plus[:,:,ind_min:ind_max,1:] - spec[:,:,ind_min:ind_max,1:]
		rf[:,:,0,zID,:,:] = diff / perturbation / dtau
		
		# remove perturbation from data
		model_plus.data[:,:,parID,zID] -= perturbation
		# model_minus.data[:,:,parID,zID] += perturbation

	# fits.writeto("rf.fits", rf, overwrite=True)

	return rf, spec
