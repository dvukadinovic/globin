import globin

init_data = globin.InputData()
globin.make_synthetic_observations(init_data.atm, init_data.rh_spec_name, 
								   init_data.wavelength, init_data.noise)