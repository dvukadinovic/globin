import numpy as np
import sys
import os
import copy
import time
from datetime import datetime
import matplotlib.pyplot as plt
from astropy.io import fits
import multiprocessing as mp

import pyrh

from .spec import get_Icont, Spectrum
from .container import Globin
from .input import InputData
from .visualize import plot_spectra
from .tools import save_chi2

import globin

def pretty_print_parameters(atmos, conv_flag, mode):
	# Npar = atmos.nx*atmos.ny
	# if Npar>40:
	# 	return
	# return

	for parameter in atmos.values:
		print(parameter)
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				if conv_flag[idx,idy]==1:
					if parameter=="gamma":
						print(f"[{idx+1},{idy+1}] --> ", atmos.values[parameter][idx,idy] * 180/np.pi)
					elif parameter=="chi":
						print(f"[{idx+1},{idy+1}] --> ", atmos.values[parameter][idx,idy] * 180/np.pi)
					else:
						print(f"[{idx+1},{idy+1}] --> ", atmos.values[parameter][idx,idy])

	if mode>=2:
		for parameter in atmos.global_pars:
			if atmos.line_no[parameter].size > 0:
				print(parameter)
				print(atmos.global_pars[parameter])

class Inverter(InputData):
	def __init__(self, save_output=True, verbose=True):
		super().__init__()
		self.save_output = save_output
		self.verbose = verbose

	def read_input(self, run_name, globin_input_name="params.input", rh_input_name="keyword.input"):
		self.run_name = run_name
		if (rh_input_name is not None) and (globin_input_name is not None):
			# initialize RH class (cythonized)
			# store input parameters in InputData()
			self.read_input_files(globin_input_name, rh_input_name)

			self.atmosphere.RH = pyrh.RH()
			self.atmosphere.n_thread = self.n_thread
			self.atmosphere.temp_tck = self.falc.temp_tck
			self.atmosphere.mode = self.mode
			self.atmosphere.norm = self.norm

			# fudge data
			if self.mode>=1:
				self.atmosphere.interp_degree = self.interp_degree
			
			self.atmosphere.wavelength_vacuum = self.wavelength_vacuum
			# self.atmosphere.wavelength = self.wavelength

		else:
			if rh_input_name is None:
				print(f"  There is no path for globin input file.")
			if globin_input_name is None:
				print(f"  There is no path for RH input file.")
			sys.exit()

	def run(self, skip=[]):
		# if self.atmosphere.spectra is None:
		# 		self.atmosphere.spectra = Spectrum(nx=self.atmosphere.nx, ny=self.atmosphere.ny, nw=len(self.wavelength_vacuum))
		
		if self.mode>=1:
			print("\n  --- Entering inversion mode ---\n")
			for cycle in range(self.ncycle):
				# double the number of iterations in the last cycle
				if cycle==self.ncycle-1 and self.ncycle!=1:
					self.max_iter *= 2

				if (self.mode==1) or (self.mode==2):
					atm, spec = self.invert_pxl_by_pxl()
				elif self.mode==3:
					atm, spec = self.invert_global()
				# elif self.mode==4:
				# 	atm, spec = invert_mcmc(save_output, verbose)
				else:
					print(f"Not supported mode {self.mode}, currently.")
					return None, None

				# in last cycle we do not smooth atmospheric parameters
				if (cycle+1)<self.ncycle:
					self.atmosphere.smooth_parameters(cycle)
					self.marq_lambda /= 10

				return atm, spec

		elif self.mode==0:
			print("\n  Entering synthesis mode.")
			spec = self.atmosphere.compute_spectra(skip=skip)
			spec.save(self.output_spectra_path, self.wavelength_air)
			return self.atmosphere, spec
			print("\n  Done!")
		else:
			print("\n  Unrecognized mode of operation. Check again and run the script.")
			sys.exit()

	def _get_Npar(self):
		# this is the number of local parameters only (we are doing pxl-by-pxl)
		if self.mode==1:
			Npar = self.atmosphere.n_local_pars
		elif (self.mode==2) or (self.mode==3):
			Npar = self.atmosphere.n_local_pars + self.atmosphere.n_global_pars

		if Npar==0:
			print("\n  There are no parameters to fit.")
			sys.exit()

		return Npar

	def invert_pxl_by_pxl(self):
		"""
		As input we expect all data to be present :)

		Pixel-by-pixel inversion of atmospheric parameters.

		Parameters:
		---------------
		init : InputData
			InputData object in which we have everything stored.
		"""
		obs = self.observation
		atmos = self.atmosphere

		LM_parameter = np.ones((obs.nx, obs.ny), dtype=np.float64) * self.marq_lambda
		if self.debug:
			LM_debug = np.zeros((self.max_iter, atmos.nx, atmos.ny))

		if self.norm:
			wavelength = np.mean(obs.wavelength)
			print(f"  Get the HSRA continuum intensity @ {wavelength}...")
			icont = get_Icont(wavelength=wavelength)
			nx, ny, = atmos.nx, atmos.ny
			nw = len(atmos.wavelength_vacuum)
			atmos.icont = np.ones((nx, ny, nw, 4))
			atmos.icont = np.einsum("ijkl,ij->ijkl", atmos.icont, icont)

		# flags those pixels whose chi2 converged:
		#   1 --> we do inversion
		#   0 --> we converged
		# with flag we multiply the proposed steps, in that case for those pixles
		# in which we converged we will not change parameters, but, the calculations
		# will be done, as well as RFs... Find smarter way around it.
		stop_flag = np.ones((obs.nx, obs.ny), dtype=np.float64)

		if self.verbose:
			print("\nInitial parameters:\n")
			pretty_print_parameters(atmos, stop_flag, atmos.mode)
			print()

		Nw = len(self.wavelength_air)
		Npar = self._get_Npar()

		# indices of diagonal elements of Hessian matrix
		x = np.arange(atmos.nx)
		y = np.arange(atmos.ny)
		p = np.arange(Npar)
		X,Y,P = np.meshgrid(x,y,p, indexing="ij")

		# indices for wavelengths min/max for which we are fiting; based on input
		ind_min = np.argmin(abs(obs.wavelength - self.wavelength_air[0]))
		ind_max = np.argmin(abs(obs.wavelength - self.wavelength_air[-1]))+1

		# print("  Get the noise estimate...")
		if self.noise==0:
			noise = 1e-4
		else:
			noise = self.noise
		StokesI_cont = obs.spec[...,ind_min,0]
		noise_lvl = noise * StokesI_cont
		# noise_wavelength = (nx, ny, nw)
		noise_wavelength = np.sqrt(obs.spec[...,ind_min:ind_max,0].T / StokesI_cont.T).T
		# noise_stokes_scale = (nx, ny, nw, 4)
		noise_stokes_scale = np.repeat(noise_wavelength[..., np.newaxis], 4, axis=3)
		# noise = (nx, ny, nw)
		noise = np.einsum("...,...w", noise_lvl, noise_wavelength)
		# noise_stokes = (nx, ny, nw, 4)
		noise_stokes = np.repeat(noise[..., np.newaxis], 4, axis=3)
		# noies_scale_rf = (nx, ny, npar, nw, 4)
		# noise_scale_rf = np.repeat(noise_stokes_scale[:,:, np.newaxis ,:,:], Npar, axis=2)
		# noise_scale_rf = 1
		# noise_stokes_scale = 1
		# noise_stokes = np.ones((obs.nx, obs.ny, Nw, 4))

		# weights on Stokes vector based on observed Stokes I
		# (nx, ny, nw, 4)
		if self.weight_type=="StokesI":
			print("  Set the weight based on Stokes I...")
			aux = 1/obs.spec[...,0]
			weights = np.repeat(aux[..., np.newaxis], 4, axis=3)
			norm = np.sum(weights, axis=2)
			weights = weights / np.repeat(norm[:,:, np.newaxis, :], Nw, axis=2)
		else:
			weights = 1

		noise_stokes /= weights

		chi2 = np.zeros((atmos.nx, atmos.ny, self.max_iter), dtype=np.float64)
		Ndof = np.count_nonzero(self.weights) * Nw - Npar

		start = time.time()

		itter = np.zeros((atmos.nx, atmos.ny), dtype=np.int)
		# we iterate until one of the pixels reach maximum numbre of iterations
		# other pixels will be blocked at max itteration earlier than or
		# will stop due to convergence criterium
		
		full_rf, old_atmos_parameters = None, None

		rf = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))
		spec = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))

		# original_atm_name_list = copy.deepcopy(atmos.atm_name_list)
		# original_line_lists_path = copy.deepcopy(atmos.line_lists_path)
		# atm_name_list = copy.deepcopy(atmos.atm_name_list)
		# line_lists_path = copy.deepcopy(atmos.line_lists_path)
		# if self.of_mode:
		# 	original_of_paths = copy.deepcopy(atmos.of_paths)
		# 	of_paths = copy.deepcopy(atmos.of_paths)

		old_inds = []

		print()
		while np.min(itter) <= self.max_iter:
			#--- if we updated parameters, recaluclate RF and referent spectra
			# if len(old_inds)!=atmos.nx*atmos.ny:
			# if len(atm_name_list)>0:
			if len(old_inds)!=(atmos.nx*atmos.ny):
				# print(2*np.arctan(atmos.values["gamma"]) * 180/np.pi)
				# print(2*np.arctan(atmos.data[0,0,6]) * 180/np.pi)
				# if self.verbose:
				t0 = datetime.now()
				t0 = t0.isoformat(sep=' ', timespec='seconds')
				print("[{:s}] Iteration (min): {:2}\n".format(t0, np.min(itter)+1))

				# atmos.atm_name_list = copy.deepcopy(atm_name_list)
				# if self.of_mode:	
				# 	atmos.of_paths = copy.deepcopy(of_paths)
				# if self.mode==2:
				# 	atmos.line_lists_path = copy.deepcopy(line_lists_path)

				# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
				#             spec.shape = (nx, ny, Nw, 5)
				old_rf = copy.deepcopy(rf)
				old_spec = copy.deepcopy(spec)
				# if self.rf_type=="snapi":
				# 	# rf, spec, full_rf = globin.compute_rfs(atmos, full_rf, old_atmos_parameters)
				# 	rf, spec, full_rf = self.compute_rfs(old_rf=full_rf, old_pars=old_atmos_parameters)
				# print(atmos.data[0,0,8])
				if self.rf_type=="node":
					# rf, spec, _ = globin.compute_rfs(atmos, skip, rf_noise_scale=noise_stokes)
					rf, spec, _ = atmos.compute_rfs(weights=self.weights, rf_noise_scale=noise_stokes, skip=old_inds, rf_type=self.rf_type)

				# plt.plot(spec.spec[0,0,:,0])
				# ax2.plot(atmos.data[0,0,1])

				# ax1.legend()
				# plt.legend()
				
				# plt.show()

				# copy old RF into new for new itteration inversion
				if len(old_inds)>0:
					for ind in old_inds:
						idx, idy = ind
						rf[idx,idy] = old_rf[idx,idy]
						spec.spec[idx,idy] = old_spec.spec[idx,idy]

				if self.debug:
					for idx in range(atmos.nx):
						for idy in range(atmos.ny):
							if stop_flag[idx,idy]==1:
								niter = itter[idx,idy]
								self.rf_debug[idx,idy,niter] = rf[idx,idy]

				# atmos.compare(self.reference_atmosphere.data[0,0])
				parameters = ["temp", "ne", "nH"]
				# globin.visualize.plot_atmosphere(atmos, parameters, color="tab:red", label="inv")
				# globin.visualize.plot_atmosphere(self.reference_atmosphere, parameters, label="ref")
				# plt.show()
				# sys.exit()

				# globin.visualize.plot_spectra(obs.spec[0,0], obs.wavelength, inv=spec.spec[0,0])
				# plt.show()
				# plt.close()
				# sys.exit()

				#--- scale RFs with weights and noise scale
				_rf = rf

				diff = obs.spec - spec.spec
				diff *= self.weights
				diff /= noise_stokes
				chi2_old = np.sum(diff**2, axis=(2,3))

				"""
				Gymnastics with indices for solving LM equations for
				next step parameters.
				"""
				# J = (nx, ny, npar, 4*nw)
				J = _rf.reshape(atmos.nx, atmos.ny, Npar, 4*Nw, order="F")
				# J = (nx, ny, 4*nw, npar)
				J = np.moveaxis(J, 2, 3)
				# JT = (nx, ny, npar, 4*nw)
				JT = np.einsum("ijlk", J)

				# JTJ = (nx, ny, npar, npar)
				JTJ = np.einsum("...ij,...jk", JT, J)
				# get diagonal elements from hessian matrix
				diagonal_elements = np.einsum("...kk->...k", JTJ)
				# reshaped array of differences between computed and observed spectra
				# flatted_diff = (nx, ny, 4*Nw)
				flatted_diff = diff.reshape(atmos.nx, atmos.ny, 4*Nw, order="F")

				# This was tested with arrays filled with hand and
				# checked if the array manipulations return what we expect
				# and it does!
				updated_pars = np.ones((atmos.nx, atmos.ny))

			# atmos.atm_name_list = copy.deepcopy(original_atm_name_list)
			# atm_name_list = copy.deepcopy(original_atm_name_list)
			# if self.of_mode:
			# 	atmos.of_paths = copy.deepcopy(original_of_paths)
			# 	of_paths = copy.deepcopy(original_of_paths)
			# if self.mode==2:
			# 	atmos.line_lists_path = copy.deepcopy(original_line_lists_path)
			# 	line_lists_path = copy.deepcopy(original_line_lists_path)
			
			old_inds = []

			# hessian = (nx, ny, npar, npar)
			H = copy.deepcopy(JTJ)
			# multiply with LM parameter
			H[X,Y,P,P] = np.einsum("...i,...", diagonal_elements, 1+LM_parameter)
			# delta = (nx, ny, npar)
			delta = np.einsum("...pw,...w", JT, flatted_diff)

			# proposed_steps = (nx, ny, npar)
			# proposed_steps = svd_invert(H, delta, stop_flag, self.svd_tolerance)
			try:
				proposed_steps = np.linalg.solve(H, delta)
			except np.linalg.LinAlgError:
				print(H)
				sys.exit()

			old_atmos_parameters = copy.deepcopy(atmos.values)
			if self.mode==2:
				old_atomic_parameters = copy.deepcopy(atmos.global_pars)
			atmos.update_parameters(proposed_steps, stop_flag)
			atmos.check_parameter_bounds(self.mode)

			# if ("loggf" in atmos.global_pars) or ("dlam" in atmos.global_pars):
			# 	for idx in range(atmos.nx):
			# 		for idy in range(atmos.ny):
			# 			fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
			# 			globin.write_line_parameters(fpath,
			# 									   atmos.global_pars["loggf"][idx,idy], atmos.line_no["loggf"],
			# 									   atmos.global_pars["dlam"][idx,idy], atmos.line_no["dlam"])
			if atmos.do_fudge==1:
				atmos.make_OF_table(self.wavelength_vacuum)

			atmos.build_from_nodes()
			if atmos.hydrostatic:
				atmos.makeHSE()

			corrected_spec = atmos.compute_spectra(old_inds)
			if not self.mean:
				corrected_spec.broaden_spectra(atmos.vmac)

			# globin.visualize.plot_spectra(obs.spec[0,0], obs.wavelength, inv=corrected_spec.spec[0,0])
			# plt.show()

			new_diff = obs.spec - corrected_spec.spec
			new_diff *= self.weights
			new_diff /= noise_stokes
			chi2_new = np.sum(new_diff**2, axis=(2,3))

			# if chi2_new[0,0]==np.nan:
			# globin.visualize.plot_atmosphere(atmos, ["temp", "ne", "nH"])
			# plt.show()
			
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					if self.debug and stop_flag[idx,idy]==1:
						niter = itter[idx,idy]
						LM_debug[niter] = LM_parameter[idx,idy]

					if stop_flag[idx,idy]==1:
						print(chi2_old[idx,idy], chi2_new[idx,idy])
						if chi2_new[idx,idy] > chi2_old[idx,idy]:
							LM_parameter[idx,idy] *= 10
							for parameter in old_atmos_parameters:
								atmos.values[parameter][idx,idy] = copy.deepcopy(old_atmos_parameters[parameter][idx,idy])
							if self.mode==2:
								for parameter in old_atomic_parameters:
									atmos.global_pars[parameter][idx,idy] = copy.deepcopy(old_atomic_parameters[parameter][idx,idy])
							# 	fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
							# 	line_lists_path.remove(fpath)
							# fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
							# atm_name_list.remove(fpath)
							# if self.of_mode:
							# 	fpath = f"{globin.cwd}/runs/{globin.wd}/ofs/of_{idx}_{idy}"
							# 	of_paths.remove(fpath)
							old_inds.append((idx,idy))
							updated_pars[idx,idy] = 0
						else:
							chi2[idx,idy,itter[idx,idy]] = chi2_new[idx,idy] / Ndof
							LM_parameter[idx,idy] /= 10
							itter[idx,idy] += 1
							updated_pars[idx,idy] = 1
					else:
						for parameter in old_atmos_parameters:
							atmos.values[parameter][idx,idy] = copy.deepcopy(old_atmos_parameters[parameter][idx,idy])
						if self.mode==2:
							for parameter in old_atomic_parameters:
								atmos.global_pars[parameter][idx,idy] = copy.deepcopy(old_atomic_parameters[parameter][idx,idy])

			# we write down those atomic parameters which are not updated (rewerting back to old ones)
			# if ("loggf" in atmos.global_pars) or ("dlam" in atmos.global_pars):
			# 	for ind in old_inds:
			# 		idx, idy = ind
			# 		fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
			# 		globin.write_line_parameters(fpath,
			# 								   atmos.global_pars["loggf"][idx,idy], atmos.line_no["loggf"],
			# 								   atmos.global_pars["dlam"][idx,idy], atmos.line_no["dlam"])

			if atmos.do_fudge==1:
				atmos.make_OF_table(self.wavelength_vacuum)

			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					if self.debug:
						for parameter in atmos.nodes:
							niter = itter[idx,idy]-1
							self.atmos_debug[parameter][niter,idx,idy] = atmos.values[parameter][idx,idy]
					if stop_flag[idx,idy]==1:
						if LM_parameter[idx,idy]<=1e-5:
							LM_parameter[idx,idy] = 1e-5
						# if Marquardt parameter is to large, we break
						if LM_parameter[idx,idy]>=1e8:
							stop_flag[idx,idy] = 0
							itter[idx,idy] = self.max_iter
							# original_atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
							# if self.mode==2:
							# 	original_line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
							print(f"[{idx},{idy}] --> Large LM parameter. We break.")

			if self.verbose:
				pretty_print_parameters(atmos, stop_flag, atmos.mode)
			print(LM_parameter)
			# print(old_inds)

			# we check if chi2 has converged for each pixel
			# if yes, we set stop_flag to 0 (True)
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					if stop_flag[idx,idy]==1 and updated_pars[idx,idy]==1:
						it_no = itter[idx,idy]
						if it_no>=2:
							# need to get -2 and -1 because I already rised itter by 1
							# when chi2 list was updated.
							relative_change = abs(chi2[idx,idy,it_no-1]/chi2[idx,idy,it_no-2] - 1)
							if chi2[idx,idy,it_no-1]<1e-32:
								print(f"--> [{idx+1},{idy+1}] : chi2 is way low!\n")
								stop_flag[idx,idy] = 0
								itter[idx,idy] = self.max_iter
								# original_atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
								# atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
								# if globin.of_mode:
								# 	original_of_paths.remove(f"{globin.cwd}/runs/{globin.wd}/ofs/of_{idx}_{idy}")
								# 	of_paths.remove(f"{globin.cwd}/runs/{globin.wd}/ofs/of_{idx}_{idy}")
								# if globin.mode==2:
								# 	original_line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
								# 	line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
							elif relative_change<self.chi2_tolerance:
								print(f"--> [{idx+1},{idy+1}] : chi2 relative change is smaller than given value.\n")
								stop_flag[idx,idy] = 0
								itter[idx,idy] = self.max_iter
								# original_atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
								# atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
								# if globin.of_mode:
								# 	original_of_paths.remove(f"{globin.cwd}/runs/{globin.wd}/ofs/of_{idx}_{idy}")
								# 	of_paths.remove(f"{globin.cwd}/runs/{globin.wd}/ofs/of_{idx}_{idy}")
								# if globin.mode==2:
								# 	original_line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
								# 	line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
							elif chi2[idx,idy,it_no-1] < 1:
								print(f"--> [{idx+1},{idy+1}] : chi2 smaller than 1\n")
								stop_flag[idx,idy] = 0
								itter[idx,idy] = self.max_iter
								# original_atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
								# atm_name_list.remove(f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}")
								# if globin.of_mode:
								# 	original_of_paths.remove(f"{globin.cwd}/runs/{globin.wd}/ofs/of_{idx}_{idy}")
								# 	of_paths.remove(f"{globin.cwd}/runs/{globin.wd}/ofs/of_{idx}_{idy}")
								# if globin.mode==2:
								# 	original_line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
								# 	line_lists_path.remove(f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}")
							# if given pixel iteration number has reached the maximum number of iterations
							# we stop the convergence for given pixel
							if it_no-1==self.max_iter-1:
								stop_flag[idx,idy] = 0
								print(f"--> [{idx+1},{idy+1}] : Maximum number of iterations reached. We break.\n")

			# if self.verbose:
			print("\n--------------------------------------------------\n")

			# if all pixels have converged, we stop inversion
			if np.sum(stop_flag)==0:
				break

		# return all original paths for final output
		# atmos.atm_name_list = []
		# atmos.line_lists_path = []
		# if globin.of_mode:
		# 	atmos.of_paths = []
		
		# for idx in range(atmos.nx):
		# 	for idy in range(atmos.ny):
				# fpath = f"runs/{globin.wd}/atmospheres/atm_{idx}_{idy}"
				# atmos.atm_name_list.append(fpath)
				# if globin.of_mode:
				# 	fpath = f"{globin.cwd}/runs/{globin.wd}/ofs/of_{idx}_{idy}"
				# 	atmos.of_paths.append(fpath)
				# if globin.mode==2:
				# 	fpath = f"runs/{globin.wd}/line_lists/rlk_list_x{idx}_y{idy}"
				# 	atmos.line_lists_path.append(fpath)
		
		# if self.mode==1:
		# 	fpath = f"runs/{globin.wd}/{globin.linelist_name}"
		# 	atmos.line_lists_path.append(fpath)

		# wrap the nodes of angles in 0-180 degrees and 0-360 degrees
		# if "gamma" in atmos.nodes:
		# 	atmos.values["gamma"] %= np.pi
		# if "chi" in atmos.nodes:
		# 	atmos.values["chi"] %= 2*np.pi

		atmos.build_from_nodes()
		if atmos.hydrostatic:
			atmos.makeHSE()
		inverted_spectra = atmos.compute_spectra()
		if not self.mean:
			inverted_spectra.broaden_spectra(atmos.vmac)

		try:
			atmos.compute_errors(JTJ, chi2_old)
		except:
			print("Failed to compute parameters error\n")

		if self.debug:

			output_path = f"runs/{self.run_name}"

			primary = fits.PrimaryHDU(self.rf_debug)
			primary.header.comments["NAXIS1"] = "Stokes components"
			primary.header.comments["NAXIS2"] = "wavelengths"
			primary.header.comments["NAXIS3"] = "parameters"
			primary.header.comments["NAXIS4"] = "iterations"
			primary.header.comments["NAXIS5"] = "y-axis atmospheres"
			primary.header.comments["NAXIS6"] = "x-axis atmospheres"
			primary.writeto(f"{output_path}/rf_pars_debug.fits", overwrite=True)

			hdulist = fits.HDUList([])

			for parameter in atmos.nodes:
				matrix = self.atmos_debug[parameter]

				par_hdu = fits.ImageHDU(matrix)
				par_hdu.name = parameter

				# par_hdu.header["unit"] = globin.parameter_unit[parameter]
				par_hdu.header.comments["NAXIS1"] = "number of nodes"
				par_hdu.header.comments["NAXIS2"] = "y-axis atmospheres"
				par_hdu.header.comments["NAXIS3"] = "x-axis atmospheres"
				par_hdu.header.comments["NAXIS4"] = "number of iterations"

				hdulist.append(par_hdu)

			hdulist.writeto(f"{output_path}/atmos_debug.fits", overwrite=True)

			primary = fits.PrimaryHDU(LM_debug)
			primary.writeto(f"{output_path}/marquardt_parameter.fits", overwrite=True)

		if self.save_output:
			output_path = f"runs/{self.run_name}"

			if self.mode==2:
				if atmos.line_no["loggf"].size>0:
					mean_loggf = np.mean(atmos.global_pars["loggf"], axis=(1,2))
				else:
					mean_loggf = None
				if atmos.line_no["dlam"].size>0:
					mean_dlam = np.mean(atmos.global_pars["dlam"], axis=(1,2))
				else:
					mean_dlam = None

				# globin.write_line_pars(f"{output_path}/line_pars_m3", mean_loggf, atmos.line_no["loggf"],
				# 													  mean_dlam, atmos.line_no["dlam"])

			if atmos.do_fudge==1:
				atmos.make_OF_table(self.wavelength_vacuum)

			inverted_spectra.xmin = obs.xmin
			inverted_spectra.xmax = obs.xmax
			inverted_spectra.ymin = obs.ymin
			inverted_spectra.ymax = obs.ymax

			atmos.save_atmosphere(f"{output_path}/inverted_atmos.fits")
			if self.mode==2:
				atmos.save_atomic_parameters(f"{output_path}/inverted_atoms.fits", kwargs={"RLK_LIST" : (f"{self.linelist_name}", "reference line list")})
			inverted_spectra.save(f"{output_path}/inverted_spectra.fits", self.wavelength_air)
			save_chi2(chi2, f"{output_path}/chi2.fits", obs.xmin, obs.xmax, obs.ymin, obs.ymax)

			end = time.time() - start
			print("\nFinished in: {0}\n".format(end))

		return atmos, inverted_spectra

	def invert_global(self):
		"""
		As input we expect all data to be present :)

		Glonal inversion of atmospheric and atomic parameters.

		Parameters:
		---------------
		init : InputData
			InputData object in which we have everything stored.
		"""
		obs = self.observation
		atmos = self.atmosphere

		if self.verbose:
			print("\nInitial parameters:\n")
			pretty_print_parameters(atmos, np.ones((atmos.nx, atmos.ny)), atmos.mode)
			print()

		Nw = len(self.wavelength_air)
		Npar = self._get_Npar()

		if self.norm:
			get_Icont()

		# if globin.norm:
			# globin.falc.write_atmosphere()
			# globin.falc.atm_name_list = [f"runs/{globin.wd}/atmospheres/atm_0_0"]
			# globin.falc.line_lists_path = atmos.line_lists_path
			
			# falc_spec, _ = globin.compute_spectra(obin.falc)
			# globin.Icont = np.max(falc_spec.spec[0,0,:,0])

		# indices for wavelengths min/max for which we are fiting; based on input
		ind_min = np.argmin(abs(obs.wavelength - self.wavelength_air[0]))
		ind_max = np.argmin(abs(obs.wavelength - self.wavelength_air[-1]))+1

		if self.noise==0:
			noise = 1e-4
		else:
			noise = self.noise

		StokesI_cont = obs.spec[...,ind_min,0]
		noise_lvl = noise * StokesI_cont
		# noise_wavelength = (nx, ny, nw)
		noise_wavelength = np.sqrt(obs.spec[...,ind_min:ind_max,0].T / StokesI_cont.T).T
		# noise = (nx, ny, nw)
		noise = np.einsum("...,...w", noise_lvl, noise_wavelength)
		# noise_stokes_scale = (nx, ny, nw, 4)
		noise_stokes_scale = np.repeat(noise_wavelength[..., np.newaxis], 4, axis=3)
		# noise_stokes = (nx, ny, nw, 4)
		noise_stokes = np.repeat(noise[..., np.newaxis], 4, axis=3)
		# noise_stokes_scale = 1
		# noise_stokes = np.ones((obs.nx, obs.ny, Nw, 4))

		# weights on Stokes vector based on observed Stokes I
		if self.weight_type=="StokesI":
			aux = 1/obs.spec[...,0]
			weights = np.repeat(aux[..., np.newaxis], 4, axis=3)
			# norm = np.sum(weights, axis=2)
			# weights = weights / np.repeat(norm[:,:, np.newaxis, :], Nw, axis=2)
		else:
			weights = 1

		noise_stokes /= weights

		chi2 = np.zeros((obs.nx, obs.ny, self.max_iter), dtype=np.float64)
		LM_parameter = self.marq_lambda
		dof = np.count_nonzero(self.weights)*Nw - Npar

		if self.debug:
			LM_debug = np.zeros((self.max_iter), dtype=np.float64)

		Natmos = atmos.nx * atmos.ny
		Ndof = np.count_nonzero(self.weights)*Nw - atmos.n_local_pars*Natmos - atmos.n_global_pars

		start = time.time()

		break_flag = False
		updated_parameters = True
		num_failed = 0

		itter = 0
		full_rf, old_local_parameters = None, None
		while itter<self.max_iter:
			if self.debug:
				LM_debug[itter] = LM_parameter
			#--- if we updated parameters, recaluclate RF and referent spectra
			if updated_parameters:
				if self.verbose:
					print("Iteration: {:2}\n".format(itter+1))

				# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
				#               spec.shape = (nx, ny, Nw, 5)
				rf, spec, full_rf = atmos.compute_rfs(rf_noise_scale=noise_stokes, weights=self.weights, mean=self.mean)

				# plt.plot(obs.spec[0,0,:,0])
				# plt.plot(spec.spec[0,0,:,0])
				# plt.show()

				# rf = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))
				# diff = np.zeros((atmos.nx, atmos.ny, Nw, 4))
				# for idx in range(atmos.nx):
				# 	for idy in range(atmos.ny):
				# 		for pID in range(Npar):
				# 			for sID in range(4):
				# 				rf[idx,idy,pID,:,sID] = np.ones(Nw)*(1+sID) + 10*pID + 100*idy + 1000*idx
				# 		for sID in range(4):
				# 			diff[idx,idy,:,sID] = np.ones(Nw)*(1+sID) + 10*idy + 100*idx

				# calculate difference between observation and synthesis
				diff = obs.spec - spec.spec
				diff *= self.weights

				if self.debug:
					for idx in range(atmos.nx):
						for idy in range(atmos.ny):
							self.rf_debug[idx,idy,itter] = rf[idx,idy]

				# calculate chi2
				# chi2_old = np.sum(diff**2 / noise_stokes**2 * globin.wavs_weight**2 * weights**2, axis=(2,3))
				diff /= noise_stokes
				chi2_old = np.sum(diff**2, axis=(2,3))

				# make Jacobian matrix and fill with RF values
				aux = rf.reshape(obs.nx, obs.ny, Npar, 4*Nw, order="F")

				J = np.zeros((4*Nw*(obs.nx*obs.ny), atmos.n_local_pars*(obs.nx*obs.ny) + atmos.n_global_pars), dtype=np.float64)
				flatted_diff = np.zeros(obs.nx*obs.ny*Nw*4, dtype=np.float64)

				l = 4*Nw
				n_atmosphere = 0
				for idx in range(obs.nx):
					for idy in range(obs.ny):
						low = n_atmosphere*l
						up = low + l
						ll = n_atmosphere*atmos.n_local_pars
						uu = ll + atmos.n_local_pars
						J[low:up,ll:uu] = aux[idx,idy,:atmos.n_local_pars].T
						flatted_diff[low:up] = diff[idx,idy].flatten(order="F")
						n_atmosphere += 1

				n_atmosphere = 0
				for idx in range(obs.nx):
					for idy in range(obs.ny):
						low = n_atmosphere*l
						up = low+l
						for gID in range(atmos.n_global_pars):
							J[low:up,uu+gID] = aux[idx,idy,atmos.n_local_pars+gID].T
						n_atmosphere += 1

				JT = J.T
				JTJ = np.dot(JT,J)
				delta = np.dot(JT, flatted_diff)

				# This was heavily(?) tested with simple filled 'rf' and 'diff' ndarrays.
				# It produces expected results.

			H = copy.deepcopy(JTJ)
			diagonal_elements = np.diag(JTJ) * (1 + LM_parameter)
			np.fill_diagonal(H, diagonal_elements)
			proposed_steps = np.linalg.solve(H, delta)

			old_local_parameters = copy.deepcopy(atmos.values)
			old_global_pars = copy.deepcopy(atmos.global_pars)
			atmos.update_parameters(proposed_steps)
			atmos.check_parameter_bounds(self.mode)

			if self.do_fudge==1:
				atmos.make_OF_table(self.wavelength_vacuum)

			atmos.build_from_nodes()
			if atmos.hydrostatic:
				atmos.makeHSE()
			corrected_spec = atmos.compute_spectra()
			if not self.mean:
				corrected_spec.broaden_spectra(atmos.vmac)

			new_diff = obs.spec - corrected_spec.spec
			new_diff *= self.weights
			new_diff /= noise_stokes
			chi2_new = np.sum(new_diff**2, axis=(2,3))

			if np.sum(chi2_new) > np.sum(chi2_old):
				LM_parameter *= 10
				atmos.values = old_local_parameters
				atmos.global_pars = old_global_pars
				updated_parameters = False
				num_failed += 1
			else:
				chi2[...,itter] = chi2_new / Ndof
				LM_parameter /= 10
				updated_parameters = True
				itter += 1
				num_failed = 0

			if self.debug:
				for parameter in atmos.nodes:
					self.atmos_debug[parameter][itter-1] = atmos.values[parameter]

			if self.do_fudge==1:
				atmos.make_OF_table(self.wavelength_vacuum)

			if LM_parameter<=1e-5:
				LM_parameter = 1e-5
			# if Marquardt parameter is to large, we break
			if LM_parameter>=1e8:
				print("Upper limit in LM_parameter. We break\n")
				break_flag = True

			# we check if chi2 has converged for each pixel
			# if yes, we set break_flag to True
			# we do not check for chi2 convergence until 3rd iteration
			if (itter)>=3 and updated_parameters:
				# need to get -2 and -1 because I already rised itter by 1
				# when chi2 list was updated.
				new_chi2 = np.sum(chi2[...,itter-1]) / Natmos
				old_chi2 = np.sum(chi2[...,itter-2]) / Natmos
				relative_change = abs(new_chi2/old_chi2 - 1)
				if new_chi2<1e-32:
					print("chi2 is way low!\n")
					break_flag = True
				elif relative_change<self.chi2_tolerance:
					print("chi2 relative change is smaller than given value.\n")
					break_flag = True
				elif new_chi2 < 1:
					print("chi2 smaller than 1\n")
					break_flag = True

			if updated_parameters and self.verbose:
				pretty_print_parameters(atmos, np.ones((atmos.nx, atmos.ny)), atmos.mode)
				print(LM_parameter)
				print("\n--------------------------------------------------\n")

			# if all pixels have converged, we stop inversion
			if break_flag:
				break

			if (num_failed==10 and itter>=3):
				print("Failed 10 times to fix the LM parameter. We break.\n")
				break

		atmos.build_from_nodes()
		if atmos.hydrostatic:
			atmos.makeHSE()
		inverted_spectra = atmos.compute_spectra()
		if not self.mean:
			inverted_spectra.broaden_spectra(atmos.vmac)

		try:
			atmos.compute_errors(JTJ, chi2_old)
		except:
			print("Failed to compute parameters error\n")

		if self.debug:
			output_path = f"runs/{self.run_name}"

			primary = fits.PrimaryHDU(self.rf_debug)
			primary.header.comments["NAXIS1"] = "Stokes components"
			primary.header.comments["NAXIS2"] = "wavelengths"
			primary.header.comments["NAXIS3"] = "parameters"
			primary.header.comments["NAXIS4"] = "iterations"
			primary.header.comments["NAXIS5"] = "y-axis atmospheres"
			primary.header.comments["NAXIS6"] = "x-axis atmospheres"
			primary.writeto(f"{output_path}/rf_pars_debug.fits", overwrite=True)

			hdulist = fits.HDUList([])

			for parameter in atmos.nodes:
				matrix = self.atmos_debug[parameter]

				par_hdu = fits.ImageHDU(matrix)
				par_hdu.name = parameter

				# par_hdu.header["unit"] = globin.parameter_unit[parameter]
				par_hdu.header.comments["NAXIS1"] = "number of nodes"
				par_hdu.header.comments["NAXIS2"] = "y-axis atmospheres"
				par_hdu.header.comments["NAXIS3"] = "x-axis atmospheres"
				par_hdu.header.comments["NAXIS4"] = "number of iterations"

				hdulist.append(par_hdu)

			hdulist.writeto(f"{output_path}/atmos_debug.fits", overwrite=True)

			primary = fits.PrimaryHDU(LM_debug)
			primary.writeto(f"{output_path}/marquardt_parameter.fits", overwrite=True)

		if self.save_output is not None:
			output_path = f"runs/{self.run_name}"

			# if ("loggf" in atmos.global_pars) or ("dlam" in atmos.global_pars):
			# 	globin.write_line_pars(f"{output_path}/line_pars_m3", atmos.global_pars["loggf"][0,0], atmos.line_no["loggf"],
			# 													  atmos.global_pars["dlam"][0,0], atmos.line_no["dlam"])

			inverted_spectra.xmin = obs.xmin
			inverted_spectra.xmax = obs.xmax
			inverted_spectra.ymin = obs.ymin
			inverted_spectra.ymax = obs.ymax

			atmos.save_atmosphere(f"{output_path}/inverted_atmos.fits")
			if atmos.n_global_pars!=0:	
				atmos.save_atomic_parameters(f"{output_path}/inverted_atoms.fits", kwargs={"RLK_LIST" : (f"{self.linelist_name}", "reference line list")})
			inverted_spectra.save(f"{output_path}/inverted_spectra.fits", self.wavelength_air)
			save_chi2(chi2, f"{output_path}/chi2.fits", obs.xmin, obs.xmax, obs.ymin, obs.ymax)

			end = time.time() - start
			print("Finished in: {0}\n".format(end))

		return atmos, inverted_spectra

def svd_invert(H, delta, stop_flag, svd_tolerance):
	nx, ny, npar, _ = H.shape

	one = np.ones(npar)

	np.nan_to_num(H, nan=0.0, copy=False)

	steps = np.zeros((nx, ny, npar))
	for idx in range(nx):
		for idy in range(ny):
			if stop_flag[idx,idy]==1:
				det = np.linalg.det(H[idx,idy])
				if det==0:
					u, eigen_vals, vh = np.linalg.svd(H[idx,idy], full_matrices=True, hermitian=True)
					vmax = svd_tolerance*np.max(eigen_vals)
					inv_eigen_vals = np.divide(one, eigen_vals, out=np.zeros_like(eigen_vals), where=eigen_vals>vmax)
					Gamma_inv = np.diag(inv_eigen_vals)
					invHess = np.dot(u, np.dot(Gamma_inv, vh))
					steps[idx,idy] = np.dot(invHess, delta[idx,idy])
				else:
					u, eigen_vals, vh = np.linalg.svd(H[idx,idy], full_matrices=True, hermitian=True)
					inv_eigen_vals = np.divide(one, eigen_vals)
					Gamma_inv = np.diag(inv_eigen_vals)
					invHess = np.dot(u, np.dot(Gamma_inv, vh))
					steps[idx,idy] = np.dot(invHess, delta[idx,idy])

	return steps

def invert_mcmc(init, save_output, verbose):
	obs = init.obs
	atmos = init.atm

	atmos.build_from_nodes()
	spec = atmos.compute_spectra()

	diff = obs.spec - spec.spec
	chi2 = np.sum(diff**2 / noise_stokes**2 * init.wavs_weight**2, axis=(2,3)) / dof

	return atmos, spec

def lnprior(pars):
	"""
	Check if each parameter is in its respective bounds given by globin.limit_values.

	If one fails, return -np.inf, else return 0.
	"""
	blos = theta
	if abs(blos) < 3000:
		return 0.0
	return -np.inf

def lnlike(theta, x, y, yp, yerr):
	"""
	Compute chi2.

	We need:
	  -- observations
	  -- noise
	  -- parameters to compute spectra
	"""
	return -0.5 * np.sum( (y-fn(theta, x, yp))**2 / yerr**2)

def lnprob(theta, x, y, yp, yerr):
	"""
	Compute product of prior and likelihood.

	We need what is needed for prior and likelihood
	"""
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yp, yerr)
