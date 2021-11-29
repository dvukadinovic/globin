import globin
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

# (nx, ny, nw, ns)
obs = globin.Observation("obs.fits")
obs.norm()
wavs = obs.wavelength
lam0 = np.mean(wavs)

Sp = obs.spec[:,:,:,0] + obs.spec[:,:,:,3]
Sm = obs.spec[:,:,:,0] - obs.spec[:,:,:,3]

dlamp = np.sum(Sp*(wavs-lam0), axis=-1) / np.sum(Sp, axis=-1)
dlamm = np.sum(Sm*(wavs-lam0), axis=-1) / np.sum(Sm, axis=-1)

vlos = c/lam0 * (dlamp + dlamm)/2
print(vlos/1e3)