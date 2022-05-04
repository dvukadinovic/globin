import globin

cont = globin.Globin()
cont.read_input(run_name="test")

spec = cont.compute_spectra()

import matplotlib.pyplot as plt

obs = globin.Observation("obs.fits")

plt.plot(spec.I[:-1])
plt.plot(obs.spec[0,0,:,0])
plt.show()