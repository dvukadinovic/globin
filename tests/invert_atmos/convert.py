import globin

atm = globin.Atmosphere("atmos.fits")

globin.multi2spinor(atm.data, "aux.fits")