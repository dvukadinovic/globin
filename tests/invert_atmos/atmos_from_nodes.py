import globin
import numpy as np

atmos = globin.utils.construct_atmosphere_from_nodes("node_atmosphere_uniform", 
	vmac=0, output_atmos_path=None)
globin.falc.wavelength_air = np.array([451.0])
globin.falc.wavelength_obs = np.array([451.0])
globin.falc.wavelength_vacuum = np.array([451.0])
spec = globin.falc.compute_spectra()