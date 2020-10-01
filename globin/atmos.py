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
import os
import sys
import rh
import time

class Atmosphere(object):
	def __init__(self, fpath):
		try:
			atmos = fits.open(fpath)[0]
		except:
			print(f"Error: Atmosphere file with path {fpath}")
			print("       does not exist.\n")
			sys.exit()
		
		self.header = atmos.header
		self.atmos = atmos.data
		self.shape = self.atmos.shape
		self.path = fpath
		self.WriteAtmospheres()
		print(f"Read atmosphere {self.path} with dimensions {self.shape}\n")

	def get_atmos(self, idx, idy):
		try:
			return self.atmos[idx,idy]
		except:
			print("Error: Atmosphere index notation does not match")
			print("       it's dimension.\n")
			sys.exit()

	def WriteAtmospheres(self):
		# write every atmosphere from cube into separate file and
		# store them in 'atmosphere' directory
		self.atm_name_list = []

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
				self.atm_name_list.append(f"atmospheres/atm_{idx}_{idy}")
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

		print("Extracted all atmospheres into folder 'atmospheres'\n")

def pool_distribute(atm_path):
	start = time.time()
	pid = mp.current_process()._identity[0]
	if not os.path.exists(f"../pid_{pid}"):
		os.mkdir(f"../pid_{pid}")

	# copy *.input files
	sp.run(f"cp *.input ../pid_{pid}", 
		shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)
	# change atmos name in 'keyword.input' file
	lines = open(f"../pid_{pid}/keyword.input","r").readlines()
	for i_, line in enumerate(lines):
		aux = line.rstrip("\n").split("=")
		if aux[0].replace(" ","")=="ATMOS_FILE":
			lines[i_] = f"  ATMOS_FILE = {os.getcwd()}/{atm_path}\n"
		if aux[0].replace(" ","")=="SPECTRUM_OUTPUT":
			spec_name = aux[1].replace(" ","")
	out = open(f"../pid_{pid}/keyword.input","w")
	out.writelines(lines)
	out.close()

	aux = atm_path.split("_")
	stdout = open(f"{os.getcwd()}/logs/log_{aux[1]}_{aux[2]}", "w")
	sp.run(f"cd ../pid_{pid}; ../rhf1d",
			shell=True, stdout=stdout, stderr=sp.STDOUT)
	stdout.close()

	dt = time.time() - start
	print("Finished synthesis of {:} in {:4.2f} s".format(atm_path, dt))

	spec = rh.Rhout(fdir=f"../pid_{pid}", verbose=False)
	spec.read_spectrum(spec_name)

	return spec

def ComputeSpectra(init):
	n_thread = init.n_thread
	atm = init.atm
	if not os.path.exists("logs"):
		os.mkdir("logs")
	else:
		sp.run("rm logs/*",
			shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)

	with mp.Pool(processes=n_thread) as pool:
		specs = pool.map(func=pool_distribute, iterable=atm.atm_name_list)
	return specs

def ComputeRF():
	pass