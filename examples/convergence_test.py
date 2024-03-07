from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

import globin

def read_inverted_atmos(fpath):
	hdul = fits.open(fpath)
	parameters = ["TEMP", "VZ", "VMIC", "MAG", "GAMMA", "CHI"]
	inverted_parameters = []
	for parameter in parameters:
		try:
			ind = hdul.index_of(parameter)
			inverted_parameters.append(parameter.lower())
		except:
			pass

	atmos = globin.Atmosphere()
	atmos.data = hdul[0].data
	atmos.nx, atmos.ny, atmos.npar, atmos.nz = atmos.data.shape
	atmos.logtau = atmos.data[0,0,0]

	return atmos, inverted_parameters

atmos = globin.Atmosphere("invert_atoms/atmos.fits")
obs = globin.Observation("invert_atoms/obs.fits")

inverted_atmos, inverted_parameters = read_inverted_atmos("invert_atoms/runs/mode3/inverted_atmos.fits")
inverted_obs = globin.Observation("invert_atoms/runs/mode3/inverted_spectra.fits")
try:
	inverted_atoms = fits.open("invert_atoms/runs/mode3/inverted_atoms.fits")
	loggfID = inverted_atoms[1].data[:,:,0,:]
	loggf = inverted_atoms[1].data[:,:,1,:]

	_, lines = globin.read_RLK_lines("invert_atoms/lines_4016")
	loggf0 = [line.loggf for line in lines]
except Exception as e:
	print(e)
	print("No atomic data to read.")
	pass

for parameter in inverted_parameters:
	parID = atmos.par_id[parameter]

	diff = atmos.data[:,:,parID] - inverted_atmos.data[:,:,parID]
	rms = np.sum( diff**2 / atmos.nz, axis=-1)

	plt.imshow(rms)
	plt.colorbar()
	plt.show()