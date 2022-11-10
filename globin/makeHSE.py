import numpy as np
from scipy.interpolate import splev, splrep
from scipy.constants import Rydberg
from scipy.constants import c as LIGHT_SPEED
from scipy.constants import h as PLANCK
from scipy.constants import k as K_BOLTZMAN
import sys

Rydberg /= 1e10 # [1/m --> 1/A]
gravity = 10**(4.44)
eps = 1e-2

logtau0 = np.array([-7.00, -5.91, -5.05, -4.06, -3.00])
pg0 = np.array([6.32E0,1.06E2,3.18E2,1.01E3,3.46E3])
pg0_tck = splrep(logtau0, pg0)

class Ion(object):
	temp_pf = np.arange(3000, 11000, 1000)
	
	def __init__(self, pf0, pf1, ej, abund, mass):
		self.pf0_tck = splrep(self.temp_pf, pf0)
		self.pf1_tck = splrep(self.temp_pf, pf1)
		self.ej = ej
		self.abund = 10**(abund-12)
		self.mass = mass

	def get_pf0(self, temp):
		if temp>10000:
			return 1
		return splev(temp, self.pf0_tck)

	def get_pf1(self, temp):
		if temp>10000:
			return 1
		return splev(temp, self.pf1_tck)

	def get_phi(self, temp):
		theta = 5040.0/temp
		u0 = self.get_pf0(temp)
		u1 = self.get_pf1(temp)
		phi = 1.2020e9*u1/u0*theta**(-5/2)*10**(-theta*self.ej)
		return phi

H = Ion([2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00001e+00, 2.00003e+00, 2.00015e+00],
		[1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00],
		13.5984, 12, 1.00797)
He = Ion([1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00],
		 [2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00, 2.00000e+00],
		 24.5874, 10.99, 4.0026)
C = Ion([8.91124e+00, 9.03328e+00, 9.19239e+00, 9.37771e+00, 9.57770e+00, 9.78474e+00, 9.99517e+00, 1.02090e+01],
		[5.88018e+00, 5.90980e+00, 5.92772e+00, 5.94003e+00, 5.94994e+00, 5.95988e+00, 5.97207e+00, 5.98845e+00],
		11.2603, 8.39, 12.011)
Si = Ion([8.62816e+00, 9.05525e+00, 9.45211e+00, 9.81774e+00, 1.01594e+01, 1.04988e+01, 1.08773e+01, 1.13575e+01],
		 [5.48529e+00, 5.60740e+00, 5.68276e+00, 5.73421e+00, 5.77257e+00, 5.80440e+00, 5.83448e+00, 5.86668e+00],
		 8.1517, 7.55, 28.0855)
Fe = Ion([2.19554e+01, 2.46130e+01, 2.77940e+01, 3.17409e+01, 3.66787e+01, 4.28266e+01, 5.04100e+01, 5.96627e+01],
		 [3.43147e+01, 3.91534e+01, 4.34176e+01, 4.75631e+01, 5.18654e+01, 5.64784e+01, 6.14794e+01, 6.69023e+01],
		 7.9025, 7.44, 55.847)
Mg = Ion([1.00025e+00, 1.00344e+00, 1.01678e+00, 1.04919e+00, 1.11004e+00, 1.21285e+00, 1.37994e+00, 1.64434e+00],
		 [2.00000e+00, 2.00002e+00, 2.00021e+00, 2.00114e+00, 2.00389e+00, 2.00976e+00, 2.02002e+00, 2.03571e+00],
		 7.6462, 7.58, 24.305)
Ni = Ion([2.63546e+01, 2.89074e+01, 3.08870e+01, 3.26360e+01, 3.44033e+01, 3.63831e+01, 3.87326e+01, 4.15802e+01],
		 [8.29948e+00, 9.43082e+00, 1.08117e+01, 1.23743e+01, 1.40520e+01, 1.57985e+01, 1.75863e+01, 1.94018e+01],
		 7.6399, 6.25, 58.7)
Cr = Ion([7.65435e+00, 8.76169e+00, 1.03912e+01, 1.26680e+01, 1.58279e+01, 2.01376e+01, 2.58494e+01, 3.31787e+01],
		 [6.08747e+00, 6.41191e+00, 7.10899e+00, 8.26418e+00, 9.94024e+00, 1.21840e+01, 1.50263e+01, 1.84825e+01],
		 6.7665, 5.67, 51.996)
Ca = Ion([1.00701e+00, 1.04991e+00, 1.17173e+00, 1.41809e+00, 1.85725e+00, 2.60365e+00, 3.81954e+00, 5.69578e+00],
		 [2.01415e+00, 2.07348e+00, 2.19899e+00, 2.38950e+00, 2.63340e+00, 2.91713e+00, 3.22898e+00, 3.56027e+00],
		 6.1132, 6.36, 40.08)
Na = Ion([2.00178e+00, 2.01488e+00, 2.06447e+00, 2.21609e+00, 2.60177e+00, 3.40984e+00, 4.84529e+00, 7.08960e+00],
		 [1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00],
		 5.1391, 6.33, 22.98977)
Al = Ion([5.79075e+00, 5.84273e+00, 5.87971e+00, 5.92580e+00, 6.01456e+00, 6.19328e+00, 6.51886e+00, 7.05012e+00],
		 [1.00000e+00, 1.00001e+00, 1.00018e+00, 1.00112e+00, 1.00405e+00, 1.01064e+00, 1.02261e+00, 1.04138e+00],
		 5.9858, 6.47, 26.981539)
S = Ion([8.30016e+00, 8.59592e+00, 8.87684e+00, 9.15036e+00, 9.41390e+00, 9.66532e+00, 9.90537e+00, 1.01385e+01],
		[4.00804e+00, 4.04842e+00, 4.14369e+00, 4.29938e+00, 4.50914e+00, 4.76202e+00, 5.04656e+00, 5.35265e+00],
		10.3600, 7.21, 32.065)
K = Ion([2.01222e+00, 2.06702e+00, 2.22592e+00, 2.61039e+00, 3.39777e+00, 4.77353e+00, 6.88524e+00, 9.82105e+00],
		[1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00],
		4.3407, 5.12, 39.0983)

ions = [H,He,C,Si,Fe,Mg,Ni,Cr,Ca,Na,Al,S,K]

Axmu = 0
for ion in ions:
	Axmu += ion.abund * ion.mass
Axmu *= 1.6606e-24 # [g]

def Hminus_bf(wavelength, T, Pe):
	a0 = 1.99654
	a1 = -1.18267e-5
	a2 = 2.64243e-6
	a3 = -4.40524e-10
	a4 = 3.23992e-14
	a5 = -1.39568e-18
	a6 = 2.78701e-23
	alpha = a0 + a1*wavelength + \
				a2*wavelength**2 + \
				a3*wavelength**3 + \
				a4*wavelength**4 + \
				a5*wavelength**5 + \
				a6*wavelength**6

	theta = 5040/T
	kappa = 4.158e-28*alpha*Pe*theta**(5/2)*10**(0.754*theta)
	return kappa

def Hminus_ff(wavelength, T, Pe):
	log_lam = np.log10(wavelength)
	theta = 5040./T
	f0 = -2.2763 - 1.6850*log_lam + 0.76661*log_lam**2 - 0.053346*log_lam**3 
	f1 = 15.2827 - 9.2846*log_lam + 1.99381*log_lam**2 - 0.142631*log_lam**3
	f2 = -197.789 + 190.266*log_lam - 67.9775*log_lam**2 + 10.6913*log_lam**3 - 0.625151*log_lam**4
	fact = f0 + f1*np.log10(theta) + f2*np.log10(theta)**2
	return 1e-26 * Pe * 10**fact

def H0_bf(wavelength, T):
	theta = 5040/T
	alpha0 = 1.0449e-26
	suma = 0
	for n in range(1,8):
		ejon = 13.598 * 1/n**2
		lam = 6.6261e-34 * LIGHT_SPEED / ejon / 1.602e-19 * 1e10 # limit lam [A]
		if wavelength<lam:
			gbf = 1 - 0.3456/(wavelength*Rydberg)**(1/3) * (wavelength*Rydberg/n**2 - 1/2)
			energy = 13.598*(1-1/n**2)
			suma += gbf/n**3 * 10**(-theta*energy)
		else:
			pass
	op = alpha0 * wavelength**3 * suma
	return op

def H0_ff(wavelength, T):
	alpha0 = 1.0449e-26
	theta = 5040/T
	
	chi_lam = 1.2398e4/wavelength
	gff = 1 + 0.3456/(wavelength*Rydberg)**(1/3) * (np.log10(np.exp(1)/theta/chi_lam) + 1/2)
	I = PLANCK*LIGHT_SPEED*Rydberg*1e10 / 1.602e-19
	
	op = alpha0 * wavelength**3 * gff * np.log10(np.exp(1)) / 2 / theta / I * 10**(-theta*I)
	return op

def H2p_ff(wavelength, T, Pe, Phi_H):
	log_lam = np.log10(wavelength)
	sigma1 = -1040.54 + 1345.71*log_lam - 547.628*log_lam**2 + 71.9684*log_lam**3
	u1 = 54.0532 - 32.713*log_lam + 6.6699*log_lam**2 - 0.4574*log_lam**3

	theta = 5040/T
	op = 3.61e-30*sigma1*10**(-u1*theta) * theta * Pe

	suma = 0
	for ion in ions:
		phi = ion.get_phi(T)
		suma += ion.abund*phi / (1 + phi/Pe)
	op *= Phi_H / (1+Phi_H/Pe) / suma

	return op

def Thompson(Pe, Pg):
	suma = 0
	for ion in ions:
		suma += ion.abund
	op = 0.6648e-24 * Pe / (Pg - Pe) * suma
	return op

def Opacity(wavelength, T, Pe, Pg):
	"""
	Total mass opacity in cm2/g.

	Ref. Gray (2005).
	"""
	Phi_H = H.get_phi(T)

	Hm_bf = Hminus_bf(wavelength, T, Pe)
	Hm_ff = Hminus_ff(wavelength, T, Pe)
	Hn_bf = H0_bf(wavelength, T)
	Hn_ff = H0_ff(wavelength, T)
	H2p = H2p_ff(wavelength, T, Pe, Phi_H)
	electrons = Thompson(Pe, Pg)

	metals_op = 0
	Hen_ff = 0

	theta = 5040/T
	chi_lam = 1.2398e4/wavelength
	stim_factor = 1-10**(-chi_lam*theta)

	total_opacity = (Hn_bf + Hn_ff + Hm_bf + H2p) * stim_factor
	total_opacity += Hm_ff
	total_opacity *= 1/(1 + Phi_H/Pe)
	total_opacity += metals_op + Hen_ff + electrons

	return total_opacity/Axmu

def solve_ne(temp, pg, pe, eps=1e-2):
	"""
	Get electron pressure. We assume that ions
	can be ionized only once.

	We solve for Pe using Newton-Raphson method.
	"""
	niter_pe = 0
	dPe = 1e5
	while (dPe/pe)>eps:
		sum1 = 0
		sum2 = 0
		sum1_p = 0	
		for ion in ions:
			phi = ion.get_phi(temp)
			sum1 += ion.abund*phi/pe/(1+phi/pe)
			sum2 += ion.abund*(1 + phi/pe/(1+phi/pe))
			sum1_p += ion.abund * phi/pe / (1+phi/pe) * (-1) / (pe + phi)
		pe_old = pe
		pe_new = pg*sum1/sum2
		f = pe - pe_new
		fp = 1 - pe_new * sum1_p/sum1 + pe_new * sum1_p/sum2
		pe = pe - f/fp
		dPe = np.abs(pe - pe_old)

		niter_pe += 1
		if niter_pe==20:
			print("Max iter in Pe loop.")
			break

	return pe

def makeHSE(wave, logt, temp, pg_top=None):
	"""
	Routine from D. F. Gray. 

	Only one ionization stage of an atom is assumed.
	No molecules.

	CGS unit system.
	"""
	tau = 10**(logt)
	nz = len(tau)

	pg = np.zeros(nz)
	pe = np.zeros(nz)
	kappa = np.zeros(nz)

	#--- get Pg and Pe for the top point of atmosphere
	if pg_top is None:
		if (logt[0]<logtau0[-1]) and (logt[0]>logtau0[0]):
			pg[0] = splev(logt[0], pg0_tck)
		elif logt[0]<logtau0[0]:
			pg[0] = pg0[0]
	else:
		pg[0] = pg_top
	pe[0] = pg[0]*1e-3

	dP = 1e10
	niter_pe = 0
	while np.abs(dP/pe[0])>eps:
		
		pe_ = pe[0]
		pe[0] = solve_ne(temp[0], pg[0], pe[0], eps)
		kappa[0] = Opacity(wave, temp[0], pe[0], pg[0])

		dP = pe[0] - pe_

		if np.abs(dP/pe[0])<eps:
			break
		# pg_ = gravity/kappa[0]*tau[0]
		# dP = np.abs(pg_ - pg[0])
		# pg[0] = pg_

		niter_pe += 1
		if niter_pe==20:
			print("Max iter in Pe loop @ top.")
			break

	#--- iterate over atmosphere points for Pg, Pe and Kappa
	for i_ in range(1,nz):
		dtau = tau[i_] - tau[i_-1]
		pg[i_] = pg[i_-1] + gravity/kappa[i_-1] * dtau
		pe[i_] = 1
		
		niter_pg = 0
		dP = 1e10
		while (dP/pg[i_])>eps:
			pe[i_] = solve_ne(temp[i_], pg[i_], pe[i_], eps)
			kappa[i_] = Opacity(wave, temp[i_], pe[i_], pg[i_])

			mean_kappa = (kappa[i_] + kappa[i_-1])/2
			pg_ = pg[i_-1] + gravity/mean_kappa*dtau
			dP = np.abs(pg_ - pg[i_])
			pg[i_] = pg_

			niter_pg += 1
			if niter_pg==20:
				print(f"Max iter in Pg loop @ k={i_}.")
				break

	rho = (pg-pe)/K_BOLTZMAN/temp/10 * np.mean(Axmu) / 1e6 # [kg/m3]
	# ne = pe/10/K_BOLTZMAN/temp / 1e6 # [1/cm3]

	return pg, pe, kappa, rho

if __name__=="__main__":
	import globin

	atmos = globin.Atmosphere("./data/falc_multi.atmos")

	# print(atmos.data[0,0,2] * 1e7 * k * atmos.data[0,0,1])

	wave = 5000

	pg, pe, kappa, rho = makeHSE(wave, atmos.data[0,0,0], atmos.data[0,0,1])
	# print("electron pressure:")
	# print(pe)