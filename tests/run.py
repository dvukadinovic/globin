import globin

inverter = globin.Inverter(verbose=False)
inverter.read_input(run_name="m1_spline_SPINOR_compare_vmic0")
inv_atmos, inv_spec, chi2 = inverter.run()

# globin.plot_spectra(inverter.observation.spec[0,0], inverter.observation.wavelength,
# 	inv=[inv_spec.spec[0,0]])
# globin.plot_atmosphere(inv_atmos, ["temp", "vz", "mag"], idx=0, idy=0)
# globin.show()