import globin
import os

atmosphere = globin.falc

atmosphere.set_mu(1.00)
atmosphere.set_cwd(".")
atmosphere.set_spectrum_normalization(True, 1)
atmosphere.add_magnetic_vector(B=1500, gamma=60, chi=15)

atmosphere.line_list = "fe6300"
atmosphere.create_input_files()

wavelength = globin.utils.compute_wavelength_grid(6301.0, 6303.0, nwl=201, unit="A")
atmosphere.set_wavelength_grid(wavelength)

spec = globin.inversion.synthesize(atmosphere)

globin.plot_spectra(spec.spec[0,0], spec.wavelength, 
                    norm=True,
                    center_wavelength_grid=False)
globin.show()

# clear input files
os.system("rm *.input")