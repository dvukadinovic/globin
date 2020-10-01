import globin
import matplotlib.pyplot as plt

init = globin.ReadInputFile()
specs = globin.atmos.ComputeSpectra(init, new_run=True)
# globin.atmos.ComputeRF(init)