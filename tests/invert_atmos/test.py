import globin
import matplotlib.pyplot as plt
import numpy as np

cont = globin.Globin()
cont.read_input(run_name="test")

spec = cont.compute_spectra(parallel=True)
print(spec.spec)

# obs = globin.Observation("obs_globin.fits")

plt.plot(spec.spec[0,2,:,0])
# plt.plot(obs.spec[0,0,:,0])
plt.show()

#-----
# import multiprocessing as mp

# dummy = globin.pyrh.Dummy()

# pool = mp.Pool(4)

# args = [np.arange(1000000, dtype=np.float64)]*10

# pool.map(dummy.add, args)

# for i_ in range(10000):
# 	dummy.add(args[i_])

#-----
