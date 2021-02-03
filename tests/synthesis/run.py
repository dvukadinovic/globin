import globin

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import sys

#--- initialize input object and then read input files
in_data = globin.InputData()
in_data.read_input_files()

globin.compute_spectra(in_data, in_data.ref_atm)
sys.exit()