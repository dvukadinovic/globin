import numpy as np

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

	Ndof = np.count_nonzero(weights)*obs.nw - npar

	if total:
		chi2 = np.sum(chi2)

	chi2 /= Ndof

	return chi2