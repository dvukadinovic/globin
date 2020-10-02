"""
Inversion of atomic parameters.

Method functions:
  -- read input file ('params.input')
  -- forward solution --> calling RH (distribute load into threads)
  -- minimization rootine --> calling MCMC routines

Contributors:
  Dusan Vukadinovic (DV)

Diary: 
  17/09/2020 --- started writing the code (readme file, structuring)
  21/09/2020 --- wrote down reading of atmos in .fits format and sent
  				 calculation to different process

Last update: 17/09/2020
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
	def __init__(self, fpath):
		try:
			atmos = fits.open(fpath)[0]
		except:
			print(f"Error: Atmosphere file with path {fpath}")
			print("       does not exist.\n")
			sys.exit()
		
		self.header = atmos.header
		self.data = atmos.data
		self.shape = self.data.shape
		self.path = fpath
		self.atm_name_list = []
		self.par_id = {"logtau" : 0,
					   "temp"   : 1,
					   "Bx"     : -1,
					   "By"     : -1,
					   "Bz"     : -1,
					   "vz"     : 3,
					   "vmic"   : 4,}

		print(f"Read atmosphere {self.path} with dimensions:")
		print(f"  (nx, ny, npar, nz) = {self.shape}\n")
		
		for idx in range(self.shape[0]):
			for idy in range(self.shape[1]):
				self.atm_name_list.append(f"atmospheres/atm_{idx}_{idy}")

		self.WriteAtmospheres()

	def get_atmos(self, idx, idy):
		# return atmosphere from cube with given
		# indices 'idx' and 'idy'
		try:
			return self.data[idx,idy]
		except:
			print("Error: Atmosphere index notation does not match")
			print("       it's dimension.\n")
			sys.exit()

	def WriteAtmospheres(self):
		# write every atmosphere from cube into separate file and
		# store them in 'atmosphere' directory

		if not os.path.exists("atmospheres"):
			os.mkdir("atmospheres")
		else:
			# clean directory if it exists (maybe we have atmospheres extracted
			# from some other cube); it takes few seconds, so not a big deal
			sp.run("rm atmospheres/*",
				shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)

		# go through each atmosphere and extract it to separate file
		for idx in range(self.shape[0]):
			for idy in range(self.shape[1]):
				atm = self.get_atmos(idx, idy)
				out = open(f"atmospheres/atm_{idx}_{idy}","w")

				out.write("* Model file'\n")
				out.write("*\n")
				out.write(f"  atm_{idx}_{idy}\n")
				out.write("  Tau scale\n")
				out.write("*\n")
				out.write("* log(g) [cm s^-2]\n")
				out.write("  4.44\n")
				out.write("*\n")
				out.write("* Ndep\n")
				out.write(f"  {self.shape[3]}\n")
				out.write("*\n")
				out.write("* lot tau    Temp[K]    n_e[m-3]    v_z[km/s]   v_turb[km/s]\n")

				for i_ in range(self.shape[3]):
					out.write("  {:+5.4f}    {:6.2f}   {:5.4e}   {:5.4e}   {:5.4e}\n".format(atm[0,i_], atm[1,i_], atm[2,i_], atm[3,i_], atm[4,i_]))

				out.write("*\n")
				out.write("* Hydrogen populations [m-3]\n")
				out.write("*     nh(1)        nh(2)        nh(3)        nh(4)        nh(5)        nh(6)\n")

				for i_ in range(self.shape[3]):
					out.write("   {:5.4e}   {:5.4e}   {:5.4e}   {:5.4e}   {:5.4e}   {:5.4e}\n".format(atm[5,i_], atm[6,i_], atm[7,i_], atm[8,i_], atm[9,i_], atm[10,i_]))

				out.close()

		# print("Extracted all atmospheres into folder 'atmospheres'\n")

def pool_distribute(arg):
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

	lines = open(f"../pid_{pid}/keyword.input","r").readlines()
	for i_, line in enumerate(lines):
		if line.replace(" ","")[0]!=globin.COMMENT_CHAR:
			line = line.rstrip("\n").split("=")
			#--- change atmos path in 'keyword.input' file
			if line[0].replace(" ","")=="ATMOS_FILE":
				lines[i_] = f"  ATMOS_FILE = {globin.cwd}/{atm_path}\n"
			#--- set to read old J.dat file
			if line[0].replace(" ","")=="STARTING_J":
				if set_old_J:
					lines[i_] = "  STARTING_J      = OLD_J\n"
				else:
					lines[i_] = "  STARTING_J      = NEW_J\n"

	out = open(f"../pid_{pid}/keyword.input","w")
	out.writelines(lines)
	out.close()

	aux = atm_path.split("_")
	idx, idy = aux[1], aux[2]
	stdout = open(f"{globin.cwd}/logs/log_{idx}_{idy}", "w")
	sp.run(f"cd ../pid_{pid}; ../rhf1d",
			shell=True, stdout=stdout, stderr=sp.STDOUT)
	stdout.close()

	spec = globin.rh.Rhout(fdir=f"../pid_{pid}", verbose=False)
	spec.read_spectrum(spec_name)

	dt = time.time() - start
	# print("Finished synthesis of {:} in {:4.2f} s".format(atm_path, dt))
	
	return {"spectra":spec, "idx":int(idx), "idy":int(idy)}

def ComputeSpectra(init, clean_dirs=True):
	n_thread = init.n_thread
	atm_name_list = init.atm.atm_name_list
	spec_name = init.spec_name

	args = [[atm_name,spec_name] for atm_name in atm_name_list]
	# args = [for i_,item in enumerate(args)]

	#--- make directory in which we will save logs of running RH
	if not os.path.exists("logs"):
		os.mkdir("logs")
	else:
		sp.run("rm logs/*",
			shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)

	#--- distribute the process to threads
	specs = init.pool.map(func=pool_distribute, iterable=args)

	spec_cube = save_spectra(specs, init.atm.shape)

	#--- delete thread directories (save them if you want to use previous run J)
	if clean_dirs:
		for threadID in range(n_thread):
			sp.run(f"rm -r ../pid_{threadID+1}",
				shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)

	return spec_cube

def save_spectra(specs, shape):
	"""
	make fits header with additional info
	"""
	wavs = specs[0]["spectra"].wave
	spectra = np.zeros((shape[0], shape[1], len(wavs), 5))
	spectra[:,:,:,0] = wavs

	for item in specs:
		spec, idx, idy = item["spectra"], item["idx"], item["idy"]
		spectra[idx,idy,:,1] = spec.imu[-1]

	primary = fits.PrimaryHDU(spectra)
	# secondary = fits.ImageHDU(wavs)
	# hdulist = fits.HDUList([primary, secondary])
	hdulist = fits.HDUList([primary])
	hdulist.writeto("spectra.fits", overwrite=True)

	return spectra

def ComputeRF(init):
	#--- get inversion parameters for atmosphere and interpolate it on finner grid (original)
	# InterpolateAtmos(init)
	#--- get current iteration atmosphere
	atmos = init.atm
	logtau = atmos.data[0,0,atmos.par_id["logtau"]]
	dtau = logtau[1] - logtau[0]
	#--- copy current atmosphere to new model atmosphere with +/- perturbation
	model_plus = copy.deepcopy(atmos)
	model_minus = copy.deepcopy(atmos)

	nz = init.atm.shape[-1]
	delta = {"temp" : 1,
			 "Bx"   : 25/1e4, # G --> T
			 "By"   : 25/1e4, # G --> T
			 "Bz"   : 25/1e4, # G --> T
			 "vz"   : 10/1e3, # m/s --> km/s
			 "vmic" : 10/1e3} # m/s --> km/s
	
	rf = np.zeros((atmos.shape[0], atmos.shape[1], atmos.shape[-1], len(delta), len(init.wavs), 4))

	import matplotlib.pyplot as plt

	spec = ComputeSpectra(init)

	#-- compute RF for each atmospheric parameter for which we are inverting
	for parID,par in enumerate(init.nodes):
		nodes = init.nodes[par]
		par_id = init.atm.par_id[par]
		perturbation = delta[par]
		if nodes!=[]:
			print(par)
			for zID in range(nz):

				model_plus.data[:,:,par_id,zID] = np.copy(atmos.data[:,:,par_id,zID]) + perturbation
				model_plus.WriteAtmospheres()
				spec_plus = ComputeSpectra(init, clean_dirs=False)
				
				model_minus.data[:,:,par_id,zID] = atmos.data[:,:,par_id,zID] - perturbation
				model_minus.WriteAtmospheres()
				spec_minus = ComputeSpectra(init, clean_dirs=False)
				
				# print(model_plus.data[:,:,par_id,zID])
				# print(model_minus.data[:,:,par_id,zID])
				
				ind_min = np.argmin(abs(spec_plus[0,0,:,0] - init.wavs[0]))
				ind_max = np.argmin(abs(spec_plus[0,0,:,0] - init.wavs[-1]))+1
				
				diff = spec_plus[:,:,ind_min:ind_max,1:] - spec_minus[:,:,ind_min:ind_max,1:]

				rf[:,:,zID,parID,:,:] = diff / 2 / perturbation / dtau

	fits.writeto("rf.fits", rf, overwrite=True)

	print("RF done!\n\n")

	return rf