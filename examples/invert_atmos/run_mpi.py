import globin

from schwimmbad import MPIPool

with MPIPool() as pool:
	inverter = globin.Inverter(verbose=True)
	inverter.read_input(run_name="dummy")
	spec = inverter.run(pool)

	globin.plot_spectra(spec.spec[0,0], spec.wavelength)
	globin.show()