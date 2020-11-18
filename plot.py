from astropy.io import fits
import matplotlib.pyplot as plt

# data = fits.open("test.fits")[0].data[0,0]

# plt.plot(data[:,0],data[:,1])
# plt.show()

import rh

spec = rh.Rhout(fdir="../pid_1")
spec.read_spectrum("spectrum.out")
spec.read_ray("spectrum_1.00")

plt.plot(spec.wave, spec.int)
plt.show()