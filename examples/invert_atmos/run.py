import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys
import copy

import globin

inverter = globin.Inverter(verbose=True)
inverter.read_input(run_name="dummy")
if inverter.mode==0:
    inverter.run()
    sys.exit()
else:
    inv_atmos, inv, chi2 = inverter.run()

idx, idy = 0,0
# inv = globin.Observation("runs/dummy/inverted_spectra_c1.fits")
globin.visualize.plot_spectra(inverter.observation.spec[idx,idy], inverter.observation.wavelength, 
    inv=[inv.spec[idx,idy]], 
    labels=["obs", "inv"])

atmos = globin.Atmosphere("atmos_bezier.fits", atm_range=[1,2,0,1])
# inv_atmos = globin.Atmosphere("runs/dummy/inverted_atmos_c1.fits")
globin.visualize.plot_atmosphere(inv_atmos, parameters=["temp", "vz", "mag", "vmic", "gamma", "chi"], reference=atmos)
globin.show()
