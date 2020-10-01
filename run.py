import globin
import matplotlib.pyplot as plt

init = globin.ReadInputFile()
specs = globin.atmos.ComputeSpectra(init)

# for spec in specs:
# 	plt.plot(spec.wave, spec.imu[-1])

# plt.show()

# # remove process folders
# sp.run("rm -r ../pid_*", 
# 	shell=True, stdout=sp.DEVNULL, stderr=sp.PIPE)