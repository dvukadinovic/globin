import globin
import numpy as np
import matplotlib.pyplot as plt

obs = globin.Observation("hinode4d.20130912.sample.fits", spec_type="hinode")

globin_inv = globin.Observation("runs/m1_spline_SPINOR_compare_vmic0/inverted_spectra_c1.fits")
globin_atm = globin.Atmosphere("runs/m1_spline_SPINOR_compare_vmic0/inverted_atmos_c1.fits")

spinor_atm, spinor_inv, _ = globin.utils.convert_spinor_inversion("spinor_inv")
spinor_inv.interpolate(obs.wavelength, 4)
spinor_atm.data[:,:,0] = spinor_atm.logtau

idx, idy = 4, 3
globin.plot_spectra(obs.spec[idx,idy], obs.wavelength,
	inv=[globin_inv.spec[idx,idy], spinor_inv.spec[idx,idy]],
	labels=["hinode", "globin", "spinor"])

globin.plot_atmosphere(globin_atm, ["temp", "vz", "mag", "gamma", "chi"], 
	reference=spinor_atm,
	labels=["globin", "spinor"],
	idx=idx,
	idy=idy)

globin.show()