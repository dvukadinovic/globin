import globin
import matplotlib.pyplot as plt
import os

# get the FAL-C atmospheric model
atmosphere = globin.falc
# we add magnetic field vector to the atmosphere
atmosphere.add_magnetic_vector(B=1500, gamma=60, chi=15)

# set the viewing angle cos(theta) = mu = 1
atmosphere.set_mu(mu=1.00)
# set if we want to normalize the spectrum
atmosphere.set_spectrum_normalization(norm=True, norm_level=1)

# set the working directory (we will do everything locally here)
atmosphere.set_cwd(cwd=".")

# we specify the line list for which we wish to model the spectra
atmosphere.line_list = "fe6300"
# now we create RH input files based on all previous inputs
atmosphere.create_input_files()
# create the wavelength grid for which to compute Stokes vector
wavelength = globin.utils.compute_wavelength_grid(lmin=6301.0, lmax=6303.0, nwl=201, unit="A")
# set the wavelength grid
atmosphere.set_wavelength_grid(wavelength=wavelength)

# run the forward modelling
spec = globin.inversion.synthesize(atmosphere)

globin.plot_spectra(spec.spec[0,0], spec.wavelength, 
                    norm=True,
                    center_wavelength_grid=False)
plt.savefig("spec_fe6300_falc_1500_60_15.png", dpi=300)
globin.show()

# clear input files
os.system("rm *.input")