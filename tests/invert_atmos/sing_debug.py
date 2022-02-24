from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

hdu = fits.open("hessian.fits")[0]
H = hdu.data

nx, ny, npar, _ = H.shape

# for idx in range(nx):
# 	for idy in range(ny):
# 		print(np.linalg.det(H[idx,idy]))

idx, idy = 0,0
print(np.linalg.det(H[idx,idy]))

plt.imshow(H[idx,idy], origin="upper")
plt.colorbar()
plt.show()