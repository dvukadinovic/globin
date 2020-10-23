import globin

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys
import os

#--- initialize input object and then read input files
in_data = globin.InputData()
in_data.read_input_files()

# in_data.atm.build_from_nodes()

spec = globin.atmos.compute_spectra(in_data)

for i in range(2):
	for j in range(2):
		plt.plot(spec[i,j,:,0], spec[i,j,:,1])
plt.show()

#--- compute spectra
# spec = globin.compute_spectra(atmosphere, n_thread=2)

# init = globin.ReadInputFile()
# specs = globin.atmos.ComputeSpectra(init, clean_dirs=False)
# rf = globin.atmos.ComputeRF(init)
sys.exit()

#--- RF clauclation test
# rf = fits.open("rf.fits")[0].data
# logtau = np.round(init.atm.data[0,0,0], 2)
# wavs = np.round((init.wavs - 401.6)*10, 2)

# xpos = np.arange(rf.shape[4])
# ypos = np.arange(rf.shape[2])
# # print(ypos)

# fig, ax = plt.subplots(nrows=1, ncols=1)

# plt.imshow(rf[0,0,:,0,:,0], aspect="auto")
# plt.setp(ax, xticks=xpos[::25], xticklabels=wavs[::25],
#     yticks=ypos[::3], yticklabels=logtau[::3])
# # plt.setp(ax, yticks=ypos[::3], yticklabels=logtau[::3])
# plt.colorbar()
# plt.show()