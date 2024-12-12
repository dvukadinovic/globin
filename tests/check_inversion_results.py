import globin
import numpy as np
import matplotlib.pyplot as plt

obs = globin.Observation("hinode4d.20130912.sample.fits", spec_type="hinode")

globin_inv = globin.Observation("runs/test/inverted_spectra_c1.fits")
globin_atm = globin.Atmosphere("runs/test/inverted_atmos_c1.fits")

spinor_atm, spinor_inv, _ = globin.utils.convert_spinor_inversion("spinor_inv")
spinor_inv.interpolate(obs.wavelength, 1)
spinor_atm.data[:,:,0] = spinor_atm.logtau

idx, idy = 0, 0
globin.plot_spectra(obs.spec[idx,idy], obs.wavelength,
	inv=[globin_inv.spec[idx,idy], spinor_inv.spec[idx,idy]],
	labels=["hinode", "globin", "spinor"])

globin.plot_atmosphere(globin_atm, ["temp", "vz", "mag", "gamma", "chi"], 
	reference=spinor_atm,
	labels=["globin", "spinor"],
	idx=idx,
	idy=idy)

globin.show()