import globin
import matplotlib.pyplot as plt
import numpy as np
import sys

# cont = globin.Globin()
# cont.read_input(run_name="test")

inverter = globin.Inverter()
inverter.read_input(run_name="pryh_test")
spec = inverter.run()

# sys.exit()

# import timeit
# def fun():
# 	cont.compute_spectra(parallel=True)
# times = timeit.Timer(fun).repeat(repeat=3, number=10)
# print(times)

obs = globin.Observation("obs_globin.fits")

plt.plot(spec.spec[0,2,:,3] - obs.spec[0,2,:,3])
# plt.plot(obs.spec[0,2,:,3])
plt.show()