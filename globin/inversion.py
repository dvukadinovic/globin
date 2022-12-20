import numpy as np
import sys
import os
import copy
import time
from datetime import datetime
import matplotlib.pyplot as plt
from astropy.io import fits
import multiprocessing as mp
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.interpolate import splrep, splev

from tqdm import tqdm, trange

import pyrh

from .spec import Spectrum
from .container import Globin
from .input import InputData, Chi2
from .visualize import plot_spectra, add_colorbar
from .utils import pretty_print_parameters

import globin

class Inverter(InputData):
	def __init__(self, save_output=True, verbose=True):
		super().__init__()
		self.save_output = save_output
		self.verbose = verbose

	def read_input(self, run_name, globin_input_name="params.input", rh_input_name="keyword.input"):
		if rh_input_name is None:
			print(f"[Error] There is no path for globin input file.")
			sys.exit()
		if globin_input_name is None:
			print(f"[Error] There is no path for RH input file.")
			sys.exit()

		# name of the folder in which we will store all synthesis/inversion results
		self.run_name = run_name
		
		#--- read the input parameters from input files
		self.read_input_files(globin_input_name, rh_input_name)

	def run(self):
		# if self.atmosphere.spectra is None:
		# 		self.atmosphere.spectra = Spectrum(nx=self.atmosphere.nx, ny=self.atmosphere.ny, nw=len(self.wavelength_vacuum))
		
		if self.mode>=1:
			print("\n{:{char}{align}{width}}\n".format(f" Entering inversion mode {self.mode} ", char="-", align="^", width=globin.NCHAR))
			start = time.time()
			for cycle in range(self.ncycle):
				if self.ncycle>1:
					print("="*globin.NCHAR)
					print("{:{align}{width}}".format(f"Inversion cycle {cycle+1}", align="^", width=globin.NCHAR,))
					print("="*globin.NCHAR)
					print()

				# if the cycle number is larger than the number of given max iterations
				# take the last given number of max iterations
				if len(self.max_iter)<cycle+1:
					max_iter = self.max_iter[-1]
				else:
					max_iter = self.max_iter[cycle]

				# if the cycle number is larger than the number of given max iterations
				# take the last given number of max iterations
				if len(self.marq_lambda)<cycle+1:
					marq_lambda = self.marq_lambda[-1]
				else:
					marq_lambda = self.marq_lambda[cycle]

				if (self.mode==1) or (self.mode==2):
					atmos, spec, chi2 = self.invert_pxl_by_pxl(max_iter, marq_lambda)
				elif self.mode==3:
					atmos, spec, chi2 = self.invert_global(max_iter, marq_lambda)
				else:
					print(f"Not supported mode {self.mode}, currently.")
					return None, None

				if self.debug:
					self.save_debug()

				if self.save_output:
					self.save_cycle(chi2, self.observation, spec, atmos, cycle+1)

				# in last cycle we do not smooth atmospheric parameters after inversion
				if (cycle+1)<self.ncycle:
					self.atmosphere.smooth_parameters(cycle)

			t0 = datetime.now()
			t0 = t0.isoformat(sep=' ', timespec='seconds')
			end = time.time() - start
			print(f"[{t0:s}] Finished in: {end:.2f}s\n")

			return atmos, spec, chi2

		elif self.mode==0:
			if self.verbose:	
				print("\n{:{char}{align}{width}}\n".format(" Entering synthesis mode ", char="-", align="^", width=globin.NCHAR))

			atmos = self.atmosphere

			if atmos.add_stray_light or atmos.norm_level=="hsra":
				print("[Info] Computing the HSRA spectrum...\n")
				atmos.get_hsra_cont()

			ones = np.ones((atmos.nx, atmos.ny))
			spec = atmos.compute_spectra(ones)
			
			#--- add macro-turbulent broadening
			spec.broaden_spectra(atmos.vmac, ones, self.n_thread)

			#--- add stray light contribution
			if atmos.add_stray_light:
				for idx in range(atmos.nx):
					for idy in range(atmos.ny):
						stray_factor = atmos.stray_light[idx,idy]
						if self.stray_type=="hsra":
							spec.spec[idx,idy] = stray_factor * atmos.hsra_spec + (1-stray_factor) * spec.spec[idx,idy]
						if self.stray_type=="gray":
							spec.spec[idx,idy,:,0] = stray_factor + (1-stray_factor) * spec.spec[idx,idy,:,0]
							spec.spec[idx,idy,:,1] = (1-stray_factor) * spec.spec[idx,idy,:,1]
							spec.spec[idx,idy,:,2] = (1-stray_factor) * spec.spec[idx,idy,:,2]
							spec.spec[idx,idy,:,3] = (1-stray_factor) * spec.spec[idx,idy,:,3]
			
			#--- add instrument broadening (if applicable)
			if atmos.instrumental_profile is not None:
				spec.instrumental_broadening(kernel=atmos.instrumental_profile, flag=ones, n_thread=self.n_thread)

			#--- add noise
			if self.noise!=0:
				spec.add_noise(self.noise)

			#--- save spectra
			if self.save_output:
				spec.save(self.output_spectra_path, spec.wavelength)

			if self.verbose:
				print("\n{:{char}{align}{width}}\n".format("All done!", char="", align="^", width=globin.NCHAR))
				print("-"*globin.NCHAR)

			return atmos, spec, None
		else:
			raise ValueError(f"Unrecognized mode={self.mode} of operation. Check input parameters.")

	def _get_Npar(self):
		# this is the number of local parameters only (we are doing pxl-by-pxl)
		if self.mode==1:
			Npar = self.atmosphere.n_local_pars
		elif (self.mode==2) or (self.mode==3):
			Npar = self.atmosphere.n_local_pars + self.atmosphere.n_global_pars

		if Npar==0:
			print("  There are no parameters to fit.")
			sys.exit()

		return Npar

	def _estimate_noise_level(self, nx, ny, nw):
		if self.noise==0:
			# noise = 1e-4
			return np.ones((nx, ny, nw, 4))
		else:
			noise = self.noise
		
		noise_stokes = np.ones((nx, ny, nw, 4))
		StokesI_cont = self.observation.spec[...,0,0]
		noise_stokes = np.einsum("ijkl,ij->ijkl", noise_stokes, noise*StokesI_cont)

		# weights on Stokes vector based on observed Stokes I
		# (nx, ny, nw, 4)
		if self.weight_type=="StokesI":
			# print("  Set the weight based on Stokes I...")
			aux = 1/self.observation.spec[...,0]
			weights = np.repeat(aux[..., np.newaxis], 4, axis=3)
			norm = np.sum(weights, axis=2)
			weights = weights / np.repeat(norm[:,:, np.newaxis, :], nw, axis=2)
		else:
			weights = 1
		
		noise_stokes /= weights

		return noise_stokes

	def invert_pxl_by_pxl(self, max_iter, marq_lambda):
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

		if atmos.add_stray_light or atmos.norm_level=="hsra":
			print("[Info] Computing the HSRA spectrum...\n")
			atmos.get_hsra_cont()

		if self.verbose:
			print("Initial parameters:\n")
			pretty_print_parameters(atmos, np.ones((atmos.nx, atmos.ny)), atmos.mode)
			print()

		Nw = len(atmos.wavelength_obs)
		Npar = self._get_Npar()
		Natmos = atmos.nx*atmos.ny
		Ndof = np.count_nonzero(self.weights)*Nw - Npar

		LM_parameter = np.ones((atmos.nx, atmos.ny), dtype=np.float64) * marq_lambda
		if self.debug:
			LM_debug = np.zeros((self.max_iter, atmos.nx, atmos.ny))

		# flags those pixels whose chi2 converged:
		#   1 --> we do inversion
		#   0 --> we converged
		stop_flag = np.ones((atmos.nx, atmos.ny), dtype=np.int32)

		# flag those pixels whose parameter we have updated
		#   0 -- no parameters update --> we have not got good parameters or we converged
		#   1 -- we have updated parameters; we need new RF and spectrum
		updated_pars = np.ones((atmos.nx, atmos.ny), dtype=np.int32)

		"""
		'stop_flag' and 'updated_pars' do not contain the same info. We can have fail update in
		parameters in one pixel in given iteration, but in the next one, we can have successful. 
		While 'stop_flag' will always stay 0 after convergence and parameters will not change.
		"""

		# indices of diagonal elements of Hessian matrix
		x = np.arange(atmos.nx)
		y = np.arange(atmos.ny)
		p = np.arange(Npar)
		X,Y,P = np.meshgrid(x,y,p, indexing="ij")

		noise_stokes = self._estimate_noise_level(atmos.nx, atmos.ny, Nw)

		chi2 = Chi2(nx=atmos.nx, ny=atmos.ny, niter=max_iter)
		chi2.mode = self.mode
		chi2.Nlolcal_par = Npar
		chi2.Nglobal_par = 0
		chi2.Nw = np.count_nonzero(self.weights)*Nw
		itter = np.zeros((atmos.nx, atmos.ny), dtype=np.int)

		atmos.rf = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))
		spec = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))
		# atmos.spec = Spectrum(nx=atmos.nx, ny=atmos.ny, nw=Nw, nz=atmos.nz)

		proposed_steps = np.zeros((atmos.nx, atmos.ny, Npar))

		# atmos.get_dd_regularization()
		# sys.exit()

		iter_start = datetime.now()

		print(f"Observations: {obs.nx} x {obs.ny}")
		print(f"Number of parameters: {Npar}")
		print(f"Number of degrees of freedom: {Ndof}\n")

		# we iterate until one of the pixels reach maximum numbre of iterations
		# other pixels will be blocked at max itteration earlier than or
		# will stop due to convergence criterium
		while np.min(itter) <=max_iter:
			start_iter = time.time
			# counter for the progress bar
			old = np.sum(stop_flag)

			#--- if we updated parameters, recaluclate RF and referent spectra
			total = np.sum(updated_pars)
			if total!=0:
				# if self.verbose:
				t0 = datetime.now()
				t0 = t0.isoformat(sep=' ', timespec='seconds')
				if self.verbose:
					print(f"[{t0:s}] Iteration (min): {np.min(itter)+1:2}\n")
				else:
					n = Natmos - np.sum(stop_flag)
					dt = datetime.now() - iter_start
					dt = dt.total_seconds()/60
					dt = np.round(dt, decimals=1)
					print(f"[{t0:s}] Iteration (min): {np.min(itter)+1:2} | per. iter {dt} min | Finished {n}/{Natmos}")
					iter_start = datetime.now()

				# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
				#             spec.shape = (nx, ny, Nw, 5)
				if total!=atmos.nx*atmos.ny:
					old_spec = copy.deepcopy(spec)
				
				spec = atmos.compute_rfs(weights=self.weights, rf_noise_scale=noise_stokes, synthesize=updated_pars, rf_type=self.rf_type)

				# globin.visualize.plot_spectra(obs.spec[0,0], obs.wavelength, inv=[spec.spec[0,0]], labels=["Inverted"])
				# globin.show()

				# copy old RF into new for new itteration inversion
				if total!=atmos.nx*atmos.ny:
					indx, indy = np.where(updated_pars==0)
					spec.spec[indx,indy] = old_spec.spec[indx, indy]

				if self.debug:
					for idx in range(atmos.nx):
						for idy in range(atmos.ny):
							if stop_flag[idx,idy]==1:
								niter = itter[idx,idy]
								self.rf_debug[idx,idy,niter] = rf[idx,idy]

				#--- compute chi2
				diff = obs.spec - spec.spec
				diff *= self.weights
				diff /= noise_stokes
				diff *= np.sqrt(2)
				if self.wavs_weight is not None:
					diff *= self.wavs_weight
				chi2_old = np.sum(diff**2, axis=(2,3))
				chi2_old /= Ndof

				"""
				Gymnastics with indices for solving LM equations for
				next step parameters.
				"""

				# JT = (nx, ny, npar, 4*nw)
				JT = atmos.rf.reshape(atmos.nx, atmos.ny, Npar, 4*Nw, order="F")
				# J = (nx, ny, 4*nw, npar)
				J = np.moveaxis(JT, 2, 3)
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

			# hessian = (nx, ny, npar, npar)
			H = JTJ

			# get scaling
			H_scale, delta_scale = normalize_hessian(H, atmos, mode=1)

			H *= H_scale

			# multiply with LM parameter
			H[X,Y,P,P] = np.einsum("...i,...", diagonal_elements, 1+LM_parameter)
			
			# delta = (nx, ny, npar)
			delta = np.einsum("...pw,...w", JT, flatted_diff)
			delta *= delta_scale

			# for idx in range(atmos.nx):
			# 	for idy in range(atmos.ny):
			# 		for ids in range(4):
			# 			flag = np.isnan(spec.spec[idx,idy,:,ids])
			# 			if any(flag):
			# 				globin.visualize.plot_atmosphere(atmos, parameters=["temp", "mag", "vz", "vmic", "gamma", "chi", "ne", "nH"],
			# 					idx=idx, idy=idy)
			# 				globin.show()

			#--- invert Hessian matrix using SVD method with specified svd_tolerance
			proposed_steps = invert_Hessian(H, delta, self.svd_tolerance, stop_flag, Npar, atmos.nx, atmos.ny, self.n_thread)

			#--- save old parameters (atmospheric and atomic)
			old_atmos_parameters = copy.deepcopy(atmos.values)
			if self.mode==2:
				old_atomic_parameters = copy.deepcopy(atmos.global_pars)
			
			#--- update and check parameter boundaries
			atmos.update_parameters(proposed_steps)
			atmos.check_parameter_bounds(self.mode)

			#--- set OF table after parameters update
			if atmos.do_fudge==1:
				atmos.make_OF_table(self.wavelength_vacuum)

			#--- rebuild new atmosphere after parameters update
			atmos.build_from_nodes(stop_flag)
			if atmos.hydrostatic:
				atmos.makeHSE(stop_flag)

			#--- compute new spectrum after parameters update
			corrected_spec = atmos.compute_spectra(stop_flag)
			
			if not self.mean:
				corrected_spec.broaden_spectra(atmos.vmac, stop_flag, self.n_thread)
			
			if atmos.add_stray_light:
				for idx in range(atmos.nx):
					for idy in range(atmos.ny):
						if stop_flag[idx,idy]==1:
							if self.stray_mode==1 or self.stray_type==2:
								stray_factor = atmos.stray_light[idx,idy]
							if self.stray_mode==3:
								if atmos.invert_stray:
									stray_factor = atmos.global_pars["stray"]
								else:
									stray_factor = atmos.stray_light[idx,idy]
							if self.stray_type=="hsra":
								corrected_spec.spec[idx,idy] = stray_factor * atmos.hsra_spec + (1-stray_factor) * corrected_spec.spec[idx,idy]
							if self.stray_type=="gray":
								corrected_spec.spec[idx,idy,:,0] = stray_factor + (1-stray_factor) * corrected_spec.spec[idx,idy,:,0]
								corrected_spec.spec[idx,idy,:,1] = (1-stray_factor) * corrected_spec.spec[idx,idy,:,1]
								corrected_spec.spec[idx,idy,:,2] = (1-stray_factor) * corrected_spec.spec[idx,idy,:,2]
								corrected_spec.spec[idx,idy,:,3] = (1-stray_factor) * corrected_spec.spec[idx,idy,:,3]
			
			if atmos.instrumental_profile is not None:
				corrected_spec.instrumental_broadening(kernel=atmos.instrumental_profile, flag=stop_flag, n_thread=self.n_thread)

			#--- compute new chi2 after parameter correction
			new_diff = obs.spec - corrected_spec.spec
			new_diff *= self.weights
			new_diff /= noise_stokes
			new_diff *= np.sqrt(2)
			if self.wavs_weight is not None:
				diff *= self.wavs_weight
			chi2_new = np.sum(new_diff**2, axis=(2,3))
			chi2_new /= Ndof

			#--- if new chi2 is lower than the old chi2
			indx, indy = np.where(chi2_new<chi2_old)
			chi2.chi2[indx,indy,itter[indx,indy]] = chi2_new[indx,indy]
			LM_parameter[indx,indy] /= 10
			itter[indx,indy] += 1
			updated_pars[indx,indy] = 1
			
			#--- if new chi2 is worse than old chi2
			indx, indy = np.where(chi2_new>=chi2_old)
			LM_parameter[indx,indy] *= 10
			for parameter in old_atmos_parameters:
				atmos.values[parameter][indx,indy] = copy.deepcopy(old_atmos_parameters[parameter][indx,indy])
			if self.mode==2:
				for parameter in old_atomic_parameters:
					atmos.global_pars[parameter][indx,indy] = copy.deepcopy(old_atomic_parameters[parameter][indx,indy])
			updated_pars[indx,indy] = 0

			#--- remake OF table after check of chi2 convergance
			if atmos.do_fudge==1:
				atmos.make_OF_table(self.wavelength_vacuum)

			#--- check the Marquardt parameter value
			indx, indy = np.where(LM_parameter<1e-5)
			LM_parameter[indx,indy] = 1e-5

			indx, indy = np.where(LM_parameter>=1e5)
			stop_flag[indx,indy] = 0
			itter[indx,indy] = max_iter

			#--- print the current state of the inversion
			if self.verbose and np.sum(stop_flag)!=0:
				pretty_print_parameters(atmos, stop_flag, atmos.mode)
				idx, idy = np.where(stop_flag==1)
				print(LM_parameter[idx,idy])

			best_chi2,_ = chi2.get_final_chi2()
			print("  chi2 --> {:4.3e}".format(np.sum(best_chi2)/Natmos))

			#--- check the convergence only for pixels whose iteration number is larger than 2
			stop_flag, itter, updated_pars = chi2_convergence(chi2.chi2, itter, stop_flag, updated_pars, self.n_thread, max_iter, self.chi2_tolerance)

			if self.verbose:
				print("\n--------------------------------------------------\n")

			# if all pixels have converged, we stop inversion
			if np.sum(stop_flag)==0:
				break

		print()

		# all pixels will be synthesized when we finish everything (should we do this?)
		updated_pars[...] = 1

		atmos.build_from_nodes(updated_pars)
		if atmos.hydrostatic:
			atmos.makeHSE(updated_pars)
		inverted_spectra = atmos.compute_spectra(updated_pars)
		if not self.mean:
			inverted_spectra.broaden_spectra(atmos.vmac, updated_pars, self.n_thread)
		if atmos.add_stray_light:
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					if self.stray_mode==1 or self.stray_mode==2:
						stray_factor = atmos.stray_light[idx,idy]
					if self.stray_mode==3:
						if atmos.invert_stray:
							stray_factor = atmos.global_pars["stray"]
						else:
							stray_factor = atmos.stray_light[idx,idy]
					if self.stray_type=="hsra":
						inverted_spectra.spec[idx,idy] = stray_factor * atmos.hsra_spec + (1-stray_factor) * inverted_spectra.spec[idx,idy]
					if self.stray_type=="gray":
						inverted_spectra.spec[idx,idy,:,0] = stray_factor + (1-stray_factor) * inverted_spectra.spec[idx,idy,:,0]
						inverted_spectra.spec[idx,idy,:,1] = (1-stray_factor) * inverted_spectra.spec[idx,idy,:,1]
						inverted_spectra.spec[idx,idy,:,2] = (1-stray_factor) * inverted_spectra.spec[idx,idy,:,2]
						inverted_spectra.spec[idx,idy,:,3] = (1-stray_factor) * inverted_spectra.spec[idx,idy,:,3]
		#if self.instrumental_profile is not None:
		#	inverted_spectra.instrumental_broadening(kernel=self.instrumental_profile, flag=updated_pars, n_thread=self.n_thread)

		return atmos, inverted_spectra, chi2

	def invert_global(self, max_iter, marq_lambda):
		"""
		Glonal inversion of atmospheric and atomic parameters.
		"""
		obs = self.observation
		atmos = self.atmosphere

		if self.verbose:
			print("Initial parameters:\n")
			pretty_print_parameters(atmos, np.ones((atmos.nx, atmos.ny)), atmos.mode)
			print()

		if atmos.add_stray_light or atmos.norm_level=="hsra":
			print("[Info] Computing the HSRA spectrum...\n")
			atmos.get_hsra_cont()

		Nw = len(atmos.wavelength_obs)
		# number of total free parameters: local per pixel + global
		Npar = self._get_Npar()
		Natmos = atmos.nx * atmos.ny
		Nlocalpar = atmos.n_local_pars
		Nglobalpar = atmos.n_global_pars
		# number of degrees of freedom
		Ndof = np.count_nonzero(self.weights)*Nw*Natmos - Nlocalpar*Natmos - Nglobalpar

		#--- estimate of Stokes noise
		noise_stokes = self._estimate_noise_level(atmos.nx, atmos.ny, Nw)

		# chi2 = np.zeros((obs.nx, obs.ny, max_iter), dtype=np.float64)
		chi2 = Chi2(nx=obs.nx, ny=obs.ny, niter=max_iter)
		chi2.mode = self.mode
		chi2.Nlolcal_par = Nlocalpar
		chi2.Nglobal_par = Nglobalpar
		chi2.Nw = np.count_nonzero(self.weights)*Nw
		
		LM_parameter = marq_lambda
		if self.debug:
			LM_debug = np.zeros((max_iter), dtype=np.float64)

		#--- the regularization function Jacobian (only spatial)
		if atmos.spatial_regularization:
			# start = time.time()
			reg_weight = atmos.spatial_regularization_weight
			LT = atmos.get_regularization_der()
			LTL = LT.dot(LT.transpose())
			chi2.regularization_weight = reg_weight
			# print("Reg = ", time.time() - start)

		# in mode==3 we always have to compute spectrum for every pixel
		ones = np.ones((atmos.nx, atmos.ny))
		# indices for transforming 4D array of RFs into 2D Jacobian matrix [slicing through atmospheres]
		indx, indy = np.where(ones==1)		

		# create the RF array for atmosphere for each free parameter (saves copy/paste time)
		atmos.rf = np.zeros((atmos.nx, atmos.ny, Npar, Nw, 4))

		# eye matrix
		eye = sp.eye(Natmos*Nlocalpar + Nglobalpar, Natmos*Nlocalpar + Nglobalpar)

		# if we reach convergence, we break from the loop
		break_flag = False

		# if proposed steps gave better fit; used to skip computing RFs otherwise
		updated_parameters = True

		# proposed parameter steps; previous iteration values are used as
		# a starting solution for the current iteration
		proposed_steps = None

		# number of times we failed to adjust parameters (rise in Marquardt parameter)
		num_failed = 0

		print(f"Observations: {obs.nx} x {obs.ny}")
		print(f"Number of parameters/px: {Npar}")
		print(f"  local/global: {Nlocalpar}/{Nglobalpar}")
		print(f"Number of degrees of freedom: {Ndof}\n")

		iter_start = datetime.now()

		itter = 0
		while itter<max_iter:
			if self.debug:
				LM_debug[itter] = LM_parameter
			
			#--- if we updated parameters, recaluclate RF and referent spectra
			if updated_parameters:
				t0 = datetime.now()
				t0 = t0.isoformat(sep=' ', timespec='seconds')
				if self.verbose:
					print(f"[{t0:s}] Iteration: {np.min(itter)+1:2}\n")
				else:
					dt = datetime.now() - iter_start
					dt = dt.total_seconds()/60
					dt = np.round(dt, decimals=1)
					print(f"[{t0:s}] Iteration: {np.min(itter)+1:2} | per. iter {dt} min")
					iter_start = datetime.now()

				# calculate RF; RF.shape = (nx, ny, Npar, Nw, 4)
				#               spec.shape = (nx, ny, Nw, 5)
				spec = atmos.compute_rfs(rf_noise_scale=noise_stokes, weights=self.weights, synthesize=ones, mean=self.mean)

				if atmos.spatial_regularization:
					Gamma = atmos.get_regularization_gamma()
					# Gamma *= np.sqrt(reg_weight)

				# globin.visualize.plot_spectra(obs.spec[0,0], obs.wavelength, inv=spec.spec[0,0])
				# plt.show()

				if self.debug:
					# for idx in range(atmos.nx):
					# 	for idy in range(atmos.ny):
					# 		self.rf_debug[idx,idy,itter] = rf[idx,idy]
					self.rf_debug[:,:,itter] = rf

				#--- calculate chi2
				diff = obs.spec - spec.spec
				diff *= self.weights
				diff /= noise_stokes
				diff *= np.sqrt(2)
				# diff /= np.sqrt(Ndof)
				if self.wavs_weight is not None:
					diff *= self.wavs_weight
				
				chi2_old = np.sum(diff**2, axis=(2,3))
				chi2_old /= Ndof
				if atmos.spatial_regularization:
					chi2_reg = np.sum(Gamma**2, axis=-1)
					chi2_old += chi2_reg
					# print(chi2_reg/chi2_old)

				#--- create the global Jacobian matrix and fill it with RF values
				tmp = atmos.rf.reshape(atmos.nx, atmos.ny, Npar, 4*Nw, order="F")
				tmp = np.swapaxes(tmp, 2, 3)
				tmp = tmp.reshape(Natmos, 4*Nw, Npar)

				# local part
				Jl = sp.block_diag(tmp[...,:Nlocalpar].tolist())
				# global part
				Jg = sp.coo_matrix(tmp[...,Nlocalpar:].reshape(4*Nw*Natmos, Nglobalpar))
				
				del tmp

				# global Jacobian matrix
				Jglobal = sp.hstack([Jl,Jg])
				Jglobal = Jglobal.tocsr()
				
				# delete parts of Jacobian (to save some bits of memory)
				del Jl
				del Jg

				#---
				JglobalT = Jglobal.transpose()

				aux = diff.reshape(atmos.nx, atmos.ny, 4*Nw, order="F")
				aux = aux.reshape(Natmos, 4*Nw, order="C")
				aux = aux.reshape(4*Nw*Natmos, order="C")

				deltaSP = JglobalT.dot(aux)
				del aux

				if atmos.spatial_regularization:
					Gamma = Gamma.reshape(atmos.nx*atmos.ny, 2*Nlocalpar, order="C")
					Gamma = Gamma.reshape(atmos.nx*atmos.ny*2*Nlocalpar, order="C")

					deltaSP -= LT.dot(Gamma)
					# print(LT.dot(Gamma)/deltaSP)

				# This was heavily(?) tested with simple filled 'rf' and 'diff' ndarrays.
				# It produces expected results.
				# [17.11.2022.] This statement still holds also for regularization terms.

			#--- invert Hessian matrix

			# fig = plt.figure()
			# gs = fig.add_gridspec(nrows=1, ncols=2)
			# ax1 = fig.add_subplot(gs[0,0])
			# ax2 = fig.add_subplot(gs[0,1])

			H = JglobalT.dot(Jglobal)
			if atmos.spatial_regularization:
				H += LTL
				Hdiag = H.diagonal(k=0)
				LTLdiag = LTL.diagonal(k=0)
			
				eta = LTLdiag/Hdiag

				# plt.plot(eta[:Nlocalpar])
				# plt.show()
				# print(eta[:Nlocalpar])
				# sys.exit()

			# get the scaling for H, deltaSP and parameters
			sp_scales, scales = normalize_hessian(H, atmos, mode=3)

			H = H.multiply(sp_scales)
			RHS = deltaSP/scales

			# add Marquardt parameter
			diagonal = H.diagonal(k=0)
			diagonal *= LM_parameter
			diagonal = sp.diags(diagonal, offsets=0, format="csc")
			H += diagonal

			# ax1.set_title("Hessian")
			# im = ax1.imshow(H.toarray(), origin="upper")
			# add_colorbar(fig, ax1, im)

			# if atmos.spatial_regularization:
			# 	tmp = LTL.multiply(sp_scales)
			# 	# tmp = LTL
			# 	ax2.set_title("LTL")
			# 	im = ax2.imshow(tmp.toarray(), origin="upper")
			# 	add_colorbar(fig, ax2, im)

			# plt.show()
			# plt.close()

			start = time.time()
			# proposed_steps, info = sp.linalg.bicgstab(H, deltaSP, M=sp.block_diag(H.diagonal(k=0)))
			proposed_steps, info = sp.linalg.bicgstab(H, RHS)
			if info>0:
				print(f"[Warning] Did not converge the solution of Ax=b.")
			if info<0:
				print("[Error] Could not solve the system Ax=b.\n  Exiting now.\n")
				return atmos, spec, chi2
			# residual = deltaSP - H.dot(proposed_steps)
			# rel_err = residual/proposed_steps
			# residual = np.sqrt(np.sum(residual**2))
			# rel_err = np.sqrt(np.sum(rel_err)**2)
			# print(f"[Info] Residual = {residual}.")
			# print(f"[Info] Relative error = {rel_err}")
			end = time.time() - start
			# print(f"[Info] Convergence time {end}s.")

			#--- save the old parameters
			old_local_parameters = copy.deepcopy(atmos.values)
			old_global_pars = copy.deepcopy(atmos.global_pars)
			
			#--- update and check boundaries for new parameter values
			atmos.update_parameters(proposed_steps)
			atmos.check_parameter_bounds(self.mode)

			#--- set OF data (if we are inverting for them)
			if self.do_fudge==1:
				atmos.make_OF_table(self.wavelength_vacuum)

			#--- rebuild the atmosphere and compute new spectrum
			atmos.build_from_nodes(ones)
			if atmos.hydrostatic:
				atmos.makeHSE(ones)

			corrected_spec = atmos.compute_spectra(ones)
			
			# broaden the corrected spectra by macro velocity
			if not self.mean:
				corrected_spec.broaden_spectra(atmos.vmac, ones, self.n_thread)
			
			# add the stray light contamination
			if atmos.add_stray_light:
				for idx in range(atmos.nx):
					for idy in range(atmos.ny):
						if self.stray_mode==1 or self.stray_mode==2:
							stray_factor = atmos.stray_light[idx,idy]
						if self.stray_mode==3:
							if atmos.invert_stray:
								stray_factor = atmos.global_pars["stray"]
							else:
								stray_factor = atmos.stray_light[idx,idy]
						if self.stray_type=="hsra":
							corrected_spec.spec[idx,idy] = stray_factor * atmos.hsra_spec + (1-stray_factor) * corrected_spec.spec[idx,idy]
						if self.stray_type=="gray":
							corrected_spec.spec[idx,idy,:,0] = stray_factor + (1-stray_factor) * corrected_spec.spec[idx,idy,:,0]
							corrected_spec.spec[idx,idy,:,1] = (1-stray_factor) * corrected_spec.spec[idx,idy,:,1]
							corrected_spec.spec[idx,idy,:,2] = (1-stray_factor) * corrected_spec.spec[idx,idy,:,2]
							corrected_spec.spec[idx,idy,:,3] = (1-stray_factor) * corrected_spec.spec[idx,idy,:,3]

			# convolve profiles with instrumental profile
			if atmos.instrumental_profile is not None:
				corrected_spec.instrumental_broadening(kernel=atmos.instrumental_profile, flag=ones, n_thread=self.n_thread)

			#--- compute new chi2 value
			new_diff = obs.spec - corrected_spec.spec
			new_diff *= self.weights
			new_diff /= noise_stokes
			new_diff *= np.sqrt(2)
			if self.wavs_weight is not None:
				new_diff *= self.wavs_weight
			
			chi2_new = np.sum(new_diff**2, axis=(2,3))
			chi2_new /= Ndof
			if atmos.spatial_regularization:
				Gamma = atmos.get_regularization_gamma()
				# Gamma *= np.sqrt(reg_weight)
				chi2_reg = np.sum(Gamma**2, axis=-1)
				chi2_new += chi2_reg

			#--- check if chi2 is improved
			if np.sum(chi2_new) > np.sum(chi2_old):
				LM_parameter *= 10
				atmos.values = old_local_parameters
				atmos.global_pars = old_global_pars
				# when we fail, we must revert the OF parameters
				if self.do_fudge==1:
					atmos.make_OF_table(self.wavelength_vacuum)
				updated_parameters = False
				num_failed += 1
			else:
				chi2.chi2[...,itter] = chi2_new
				if atmos.spatial_regularization:
					chi2.regularization = chi2_reg
				LM_parameter /= 10
				updated_parameters = True
				itter += 1
				num_failed = 0

			if self.debug:
				for parameter in atmos.nodes:
					self.atmos_debug[parameter][itter-1] = atmos.values[parameter]

			#--- check the Marquardt parameter boundaries
			if LM_parameter<=1e-5:
				LM_parameter = 1e-5
			if LM_parameter>=1e5:
				print("Upper limit in LM_parameter. We break\n")
				break_flag = True

			if updated_parameters:
				print("  chi2 --> {:4.3e} | log10(LM) --> {:2.0f}".format(np.sum(chi2.chi2[...,itter-1]), np.log10(LM_parameter)))
			
			#--- print current parameters
			if updated_parameters and self.verbose:
				pretty_print_parameters(atmos, ones, atmos.mode)
				print(LM_parameter)
				print("-"*globin.NCHAR, "\n")


			# we check if chi2 has converged for each pixel
			# if yes, we set break_flag to True
			# we do not check for chi2 convergence until 2nd iteration
			if (itter)>=2 and updated_parameters:
				# need to get -2 and -1 because we already rised itter by 1
				# when chi2 was updated
				new_chi2 = np.sum(chi2.chi2[...,itter-1])
				old_chi2 = np.sum(chi2.chi2[...,itter-2])
				relative_change = np.abs(new_chi2/old_chi2 - 1)
				if relative_change<self.chi2_tolerance:
					print("\nchi2 relative change is smaller than given value.\n")
					break_flag = True
				elif new_chi2<1:
					print("\nchi2 smaller than 1\n")
					break_flag = True
				elif itter==max_iter:
					print("\nMaximum number of iteratinos reached.\n")
					break_flag = True
				else:
					pass

			#--- break if we could not adjust Marquardt parameter more than 6 times
			if (num_failed==6 and itter>=2):
				print("Failed 6 times to fix the LM parameter. We break.\n")
				break_flag = True

			#--- if all pixels have converged, we stop inversion
			if break_flag:
				break

		atmos.build_from_nodes(ones)
		if atmos.hydrostatic:
			atmos.makeHSE(ones)
		inverted_spectra = atmos.compute_spectra(ones)
		
		if not self.mean:
			inverted_spectra.broaden_spectra(atmos.vmac, ones, self.n_thread)
		
		if atmos.add_stray_light:
			for idx in range(atmos.nx):
				for idy in range(atmos.ny):
					if self.stray_mode==1 or self.stray_mode==2:
						stray_factor = atmos.stray_light[idx,idy]
					if self.stray_mode==3:	
						if atmos.invert_stray:
							stray_factor = atmos.global_pars["stray"]
						else:
							stray_factor = atmos.stray_light[idx,idy]
					if self.stray_type=="hsra":
						inverted_spectra.spec[idx,idy] = stray_factor * atmos.hsra_spec + (1-stray_factor) * inverted_spectra.spec[idx,idy]
					if self.stray_type=="gray":
						inverted_spectra.spec[idx,idy,:,0] = stray_factor + (1-stray_factor) * inverted_spectra.spec[idx,idy,:,0]
						inverted_spectra.spec[idx,idy,:,1] = (1-stray_factor) * inverted_spectra.spec[idx,idy,:,1]
						inverted_spectra.spec[idx,idy,:,2] = (1-stray_factor) * inverted_spectra.spec[idx,idy,:,2]
						inverted_spectra.spec[idx,idy,:,3] = (1-stray_factor) * inverted_spectra.spec[idx,idy,:,3]
		
		if atmos.instrumental_profile is not None:
			inverted_spectra.instrumental_broadening(kernel=atmos.instrumental_profile, flag=ones, n_thread=self.n_thread)

		return atmos, inverted_spectra, chi2

	def save_debug(self):
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

	def save_cycle(self, chi2, obs, spec, atmos, cycle):
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

		if atmos.do_fudge==1:
			atmos.make_OF_table(self.wavelength_vacuum)

		spec.xmin = obs.xmin
		spec.xmax = obs.xmax
		spec.ymin = obs.ymin
		spec.ymax = obs.ymax

		atmos.save_atmosphere(f"{output_path}/inverted_atmos_c{cycle}.fits")
		if self.mode==2 or self.mode==3:
			atmos.save_atomic_parameters(f"{output_path}/inverted_atoms_c{cycle}.fits")
		spec.save(f"{output_path}/inverted_spectra_c{cycle}.fits", spec.wavelength)
		chi2.save(fpath=f"{output_path}/chi2_c{cycle}.fits")

	def estimate_regularization_weight(self, alpha_min, alpha_max, num=11, fpath=None):
		# reset the original relative weighting for each parameter
		original_weight = self.atmosphere.spatial_regularization_weight
		for parameter in self.atmosphere.nodes:
			self.atmosphere.regularization_weight[parameter] /= original_weight

		relative_weights = copy.deepcopy(self.atmosphere.regularization_weight)

		reg_weight = np.logspace(alpha_min, alpha_max, num=num)
		init_atmos_values = self.atmosphere.values
		if self.mode==2 or self.mode==3:
			init_global_values = self.atmosphere.global_pars

		total_chi2 = np.zeros(num)
		regul_chi2 = np.zeros(num)

		for i_, alpha in enumerate(reg_weight):
			print("\n{:{char}{align}{width}}".format(f" Round {i_+1}/{num} ", char="*", align="^", width=globin.NCHAR))

			# recompute the regularization weights for each parameter
			for parameter in self.atmosphere.nodes:
				self.atmosphere.regularization_weight[parameter] = relative_weights[parameter] * alpha

			self.atmosphere.values = copy.deepcopy(init_atmos_values)
			if self.mode==2 or self.mode==3:
				self.atmosphere.global_pars = copy.deepcopy(init_global_values)
			atm, _, chi2 = self.run()
			regul_chi2[i_] = np.sum(chi2.regularization)
			total_chi2[i_] = np.sum(chi2.get_final_chi2()[0])
			# print(atm.values["temp"])

		if fpath:
			np.savetxt(fpath, np.vstack((reg_weight, total_chi2, regul_chi2)).T, fmt="%5.4e", header=" alpha  tot_chi2  reg_chi2")

		return reg_weight, total_chi2, regul_chi2

def invert_Hessian(H, delta, svd_tolerance, stop_flag, Npar, nx, ny, n_thread=1):
	indx, indy = np.where(stop_flag==1)
	args = zip(H[indx,indy], delta[indx,indy], [svd_tolerance]*nx*ny)

	with mp.Pool(n_thread) as pool:
		results = pool.map(func=_invert_Hessian, iterable=args)

	steps = np.zeros((nx, ny, Npar))

	results = np.array(results)
	natm, npar = results.shape
	steps[indx, indy] = results.reshape(natm, npar)

	return steps

def _invert_Hessian(args):
	hessian, delta, svd_tolerance = args
	
	Npar = delta.shape
	one = np.ones(Npar)

	# steps = np.linalg.solve(hessian, delta)

	invH = np.linalg.pinv(hessian, rcond=svd_tolerance, hermitian=True)
	steps = np.dot(invH, delta)

	# det = np.linalg.det(hessian)
	# if det==0:
	# u, eigen_vals, vh = np.linalg.svd(hessian, full_matrices=True, hermitian=True)
	# vmax = svd_tolerance*np.max(eigen_vals)
	# inv_eigen_vals = np.divide(one, eigen_vals, out=np.zeros_like(eigen_vals), where=eigen_vals>vmax)
	# Gamma_inv = np.diag(inv_eigen_vals)
	# invHess = np.dot(u, np.dot(Gamma_inv, vh))
	# steps = np.dot(invHess, delta)
	# else:
	# 	steps = np.linalg.solve(hessian, delta)

	return steps

#--- check the convergence of chi2
def chi2_convergence(chi2, itter, stop_flag, updated_pars, n_thread, max_iter, chi2_tolerance):
	nx, ny = stop_flag.shape

	_max_iter = [max_iter]*nx*ny
	_chi2_tolerance = [chi2_tolerance]*nx*ny

	args = zip(chi2.reshape(nx*ny, max_iter), stop_flag.flatten(), itter.flatten(), updated_pars.flatten(), _max_iter, _chi2_tolerance)

	with mp.Pool(n_thread) as pool:
		results = pool.map(func=_chi2_convergence, iterable=args)

	results = np.array(results)
	stop_flag = results[...,0].reshape(nx, ny)
	itter = results[...,1].reshape(nx, ny)
	updated_pars = results[...,2].reshape(nx, ny)

	return stop_flag.astype(np.int32), itter.astype(np.int32), updated_pars.astype(np.int32)

def _chi2_convergence(args):
	"""
	Return:
	----------
 	stop_flag    --> 0 for the end
 	max_iter     --> maximum number of iteration if we ended
	updated_pars --> flag if we do not need to update parameters 
					 anymore since we converged
	"""
	chi2, flag, itter, updated_pars, max_iter, chi2_tolerance = args

	# we do not check chi2 until the iteration=3
	if itter<2:
		return 1, itter, updated_pars
	
	# if we have already converged or max_iter reached
	if flag==0 or itter==max_iter:
		return 0, max_iter, 0

	# if chi2 is lower than 1
	if chi2[itter-1]<1:
		return 0, max_iter, 0

	relative_change = np.abs(chi2[itter-1]/chi2[itter-2] - 1)
	# if relative change of chi2 is lower than given tolerance level
	if relative_change<chi2_tolerance:
		return 0, max_iter, 0

	# if we still have not converged
	return 1, itter, updated_pars

def normalize_hessian(H, atmos, mode):
	"""
	Normalize Hessian matrix so that diagonal elements are =1.

	Recompute the paramters scale so that we can reconstruct proper values.

	Parameters:
	-----------
	Hessian : ndarray or scipy.sparse.matrix
		Hessian matrix (squared). It is assumed to be sparse matrix, but it 
		should work with dense matrices also.
	atmos : globin.atmos.Atmosphere()
		atmosphere structure containing the nodes and values of inversion 
		parameters as well as scaling for each parameter node.
	mode : int
		inversion mode. Defines the structure of the output.
	
	Returns:
	--------
	sp_scale : scipy.sparse.matrix
		sparse matrix containing the values with which we need to multiply 
		Hessian matrix in order to get 1s on a diagonal.
	scale : ndarray
		array containing the scale for each parameter that is needed to divide the
		RHS of LM equation.
	"""
	if mode==1:
		Npar = atmos.n_local_pars

		RHS_scales = np.zeros((atmos.nx, atmos.ny, Npar))
		H_scales = np.zeros((atmos.nx, atmos.ny, Npar, Npar))
		
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				diagonal = np.diagonal(H[idx,idy], offset=0)
				scales = np.sqrt(diagonal)

				l, u = 0, 0
				for parameter in atmos.nodes:
					l, u = u, u + len(atmos.nodes[parameter])
					atmos.parameter_scale[parameter][idx,idy,:] = scales[l:u]
					atmos.parameter_scale[parameter][idx,idy,:] *= atmos.parameter_norm[parameter]

				scales = 1/scales
				RHS_scales[idx,idy] = scales
				H_scales[idx,idy] = np.outer(scales, scales)

		return H_scales, RHS_scales

	if mode==2:
		raise ValueError("Not implemented yet.")

	if mode==3:
		Nlocalpar = atmos.n_local_pars
		Nglobalpar = atmos.n_global_pars
		Natmos = atmos.nx*atmos.ny

		# get scales (as sqrt of Hessian diagonal elements)
		diagonal = H.diagonal(k=0)
		scales = np.sqrt(diagonal)

		# create the indices of sub-jacobian matrix
		X, Y = np.meshgrid(np.arange(Nlocalpar), np.arange(Nlocalpar), indexing="ij")

		# get the local (atmospheric) parameters scale values
		shift = 0
		for parameter in atmos.nodes:
			nnodes = len(atmos.nodes[parameter])
			for idn in range(nnodes):
				ind = np.arange(0, Natmos*Nlocalpar, Nlocalpar) + shift
				atmos.parameter_scale[parameter][:,:,idn] = scales[ind].reshape(atmos.nx, atmos.ny, order="F")
				shift += 1
			# parameter scale must be multiplied by the normalization value in order to retrieve 
			# the proposed step in unit of a parameter. We scaled the Regularization function and
			# the RF with this normalization value.
			atmos.parameter_scale[parameter] *= atmos.parameter_norm[parameter]

		# get the global (atomic) parameters scale values
		start = Natmos*Nlocalpar
		for parameter in atmos.global_pars:
			if parameter=="vmac":
				atmos.parameter_scale[parameter] = scales[start]
				atmos.parameter_scale[parameter] *= atmos.parameter_norm[parameter]
				start += 1
			if parameter in ["loggf", "dlam"]:
				N = len(atmos.line_no[parameter])
				if N==0:
					continue

				for idl in range(N):
					ind = start + idl
					atmos.parameter_scale[parameter][...,idl] = scales[ind]

				atmos.parameter_scale[parameter] *= atmos.parameter_norm[parameter]

				start += N

		# create the sparse matrix of scales for each parameter combination for every atmosphere
		l, u = 0, 0
		ida = 0
		rows = np.array([], dtype=np.int32)
		cols = np.array([], dtype=np.int32)
		values = np.array([])
		for idx in range(atmos.nx):
			for idy in range(atmos.ny):
				l = u
				u += Nlocalpar
				division = np.outer(scales[l:u], scales[l:u])
				rows = np.append(rows, X.ravel() + ida*Nlocalpar)
				cols = np.append(cols, Y.ravel() + ida*Nlocalpar)
				values = np.append(values, 1/division.ravel())
				ida += 1

		start = Natmos*Nlocalpar
		Ng = 0
		for parameter in atmos.global_pars:
			if parameter in ["loggf", "dlam"]:
				N = len(atmos.line_no[parameter])
				if N==0:
					continue

			if parameter=="vmac":
				N = 1

			for idl in range(N):
				scale = scales[start + idl]
				val = scales[:start+idl+1] * scale
				# vertical part of Hessian
				values = np.append(values, 1/val)
				# horizontal part of Hessian
				# we exclude the last point because it is already present
				values = np.append(values, 1/val[:-1])

				# indices for vertical part of Hessian
				row = np.arange(start + idl+1)
				rows = np.append(rows, row)

				col = np.ones(start + idl+1) * (start + idl)
				cols = np.append(cols, col)

				# indices for horizontal part of Hessian
				# (we exclude last point because it would be doubled)
				rows = np.append(rows, col[:-1])
				cols = np.append(cols, row[:-1])

			start += N
			
		sp_scale = sp.csr_matrix((values, (rows, cols)), shape=H.shape, dtype=np.float64)

		# H = H.multiply(sp_scale)
		# plt.imshow(H.toarray(), origin="upper")
		# plt.colorbar()
		# plt.show()
		# sys.exit()

		return sp_scale, scales

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
