import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys
import copy
import time

import globin

falc = globin.falc

nz = 51
x = np.linspace(-4, 1, num=nz)
atmos = globin.Atmosphere(nx=10, ny=10, nz=nz)
atmos.interpolate_atmosphere(x, falc.data)
atmos.scale_id = 0

# atmos.read_spectral_lines("lines_4016")
atmos.set_wavelength(lmin=4015, lmax=4017, nwl=101, unit="A")

atmos.mode = 3
atmos.n_thread = 4

start = time.time()

atmos.chunk_size = atmos.nx*atmos.ny//atmos.n_thread + 1
atmos.compute_spectra()

end = time.time()

print(end - start)