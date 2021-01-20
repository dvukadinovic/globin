import sys
import numpy as np
import matplotlib.pyplot as plt
# from scipy.constants import k as k_b
from scipy.constants import m_e, m_p
from astropy.io import fits

k_b = 1.38064852e-16
m_p *= 1e3
m_e *= 1e3

import globin

atom_mass = np.array([1.00797,
            4.00260,
            6.941,
            9.01218,
            10.81,
            12.011,
            14.0067,
            15.9994,
            18.998403,
            20.179,
            22.98977,
            24.305,
            26.98154,
            28.0855,
            30.97376,
            32.06,
            35.453,
            39.948,
            39.0983,
            40.08,
            44.9559,
            47.90,
            50.9415,
            51.996,
            54.9380,
            55.847,
            58.9332,
            58.70,
            63.546,
            65.38,
            69.72,
            72.59,
            74.9216,
            78.96,
            79.904,
            83.80,
            85.4678,
            87.62,
            88.9059,
            91.22,
            92.9064,
            95.94,
            98,
            101.07,
            102.9055,
            106.4,
            107.868,
            112.41,
            114.82,
            118.69,
            121.75,
            127.60,
            126.9045,
            131.30,
            132.9054,
            137.33,
            138.9055,
            140.12,
            140.9077,
            144.24,
            145,
            150.4,
            151.96,
            157.25,
            158.9254,
            162.50,
            164.9304,
            167.26,
            168.9342,
            173.04,
            174.967,
            178.49,
            180.9479,
            183.85,
            186.207,
            190.2,
            192.22,
            195.09,
            196.9665,
            200.59,
            204.37,
            207.2,
            208.9804,
            209,
            210,
            222,
            223,
            226.0254,
            227.0278,
            232.0381,
            231.0359,
            238.029,
            237.0482,
            242,
            243,
            247,
            247,
            251,
            252])
atom_abundance = np.array([12.0,
            10.99,
            1.16,
            1.15,
            2.6,
            8.39,
            8.0,
            8.66,
            4.4,
            8.09,
            6.33,
            7.58,
            6.47,
            7.55,
            5.45,
            7.21,
            5.5,
            6.56,
            5.12,
            6.36,
            3.1,
            4.99,
            4.0,
            5.67,
            5.39,
            7.44,
            4.92,
            6.25,
            4.21,
            4.6,
            2.88,
            3.41,
            2.37,
            3.35,
            2.63,
            3.23,
            2.6,
            2.9,
            2.24,
            2.6,
            1.42,
            1.92,
            -7.96,
            1.84,
            1.12,
            1.69,
            0.94,
            1.86,
            1.66,
            2.0,
            1.0,
            2.24,
            1.51,
            2.23,
            1.12,
            2.13,
            1.22,
            1.55,
            0.71,
            1.5,
            -7.96,
            1.0,
            0.51,
            1.12,
            -0.1,
            1.1,
            0.26,
            0.93,
            0.0,
            1.08,
            0.76,
            0.88,
            0.13,
            1.11,
            0.27,
            1.45,
            1.35,
            1.8,
            1.01,
            1.09,
            0.9,
            1.85,
            0.71,
            -7.96,
            -7.96,
            -7.96,
            -7.96,
            -7.96,
            -7.96,
            0.12,
            -7.96,
            -0.47,
            -7.96,
            -7.96,
            -7.96,
            -7.96,
            -7.96,
            -7.96,
            -7.96])

def construct_atmosphere_from_nods(in_data):
    atmos = in_data.atm

    # print(atmos.values)

    atmos.values["temp"] = np.zeros((2,3,3))
    atmos.values["temp"][0,0] = [4500,5500,7300]
    atmos.values["temp"][0,1] = [4400,5400,7200]
    atmos.values["temp"][0,2] = [4000,5400,7500]
    atmos.values["temp"][1,0] = [4500,5600,7500]
    atmos.values["temp"][1,1] = [4800,5300,7600]
    atmos.values["temp"][1,2] = [4200,5200,7400]
    # atmos.values["temp"][0,0] = [4500,5500,7300]
    # atmos.values["temp"][0,1] = [4500,5500,7300]
    # atmos.values["temp"][0,2] = [4500,5500,7300]
    # atmos.values["temp"][1,0] = [4500,5500,7300]
    # atmos.values["temp"][1,1] = [4500,5500,7300]
    # atmos.values["temp"][1,2] = [4500,5500,7300]

    atmos.values["vz"] = np.zeros((2,3,2))
    # atmos.values["vz"][0,0] = [1,-0.5]
    # atmos.values["vz"][0,1] = [0.5,0.1]
    # atmos.values["vz"][0,2] = [0,0]
    # atmos.values["vz"][1,0] = [0.75,-0.1]
    # atmos.values["vz"][1,1] = [-1,0]
    # atmos.values["vz"][1,2] = [-1.2,0.5]
    # atmos.values["vz"][0,0] = [0,0]
    # atmos.values["vz"][0,1] = [0,0]
    # atmos.values["vz"][0,2] = [0,0]
    # atmos.values["vz"][1,0] = [0,0]
    # atmos.values["vz"][1,1] = [0,0]
    # atmos.values["vz"][1,2] = [0,0]

    atmos.values["vmic"] = np.zeros((2,3,1))
    # atmos.values["vmic"][0,0] = [0.0]
    # atmos.values["vmic"][0,1] = [0.25]
    # atmos.values["vmic"][0,2] = [0.50]
    # atmos.values["vmic"][1,0] = [0.75]
    # atmos.values["vmic"][1,1] = [1.0]
    # atmos.values["vmic"][1,2] = [1.25]
    # atmos.values["vmic"][0,0] = [0.0]
    # atmos.values["vmic"][0,1] = [0.0]
    # atmos.values["vmic"][0,2] = [0.0]
    # atmos.values["vmic"][1,0] = [0.0]
    # atmos.values["vmic"][1,1] = [0.0]
    # atmos.values["vmic"][1,2] = [0.0]

    atmos.values["mag"] = np.zeros((2,3,2))
    # atmos.values["mag"][0,0] = [100,250]
    # atmos.values["mag"][0,1] = [0,0]
    # atmos.values["mag"][0,2] = [500,1500]
    # atmos.values["mag"][1,0] = [300,350]
    # atmos.values["mag"][1,1] = [2000,2500]
    # atmos.values["mag"][1,2] = [250,500]

    # atmos.values["mag"][0,0] = [-500]
    # atmos.values["mag"][0,1] = [500]
    # atmos.values["mag"][0,2] = [500]
    # atmos.values["mag"][1,0] = [500]
    # atmos.values["mag"][1,1] = [500]
    # atmos.values["mag"][1,2] = [500]
    atmos.values["mag"] /= 1e4 # [G --> T]

    atmos.values["gamma"] = np.zeros((2,3,1))
    # atmos.values["gamma"][0,0] = [70]
    # atmos.values["gamma"][0,1] = [45]
    # atmos.values["gamma"][0,2] = [10]
    # atmos.values["gamma"][1,0] = [30]
    # atmos.values["gamma"][1,1] = [60]
    # atmos.values["gamma"][1,2] = [0]
    # atmos.values["gamma"][0,0] = [0]
    # atmos.values["gamma"][0,1] = [0]
    # atmos.values["gamma"][0,2] = [180]
    # atmos.values["gamma"][1,0] = [45]
    # atmos.values["gamma"][1,1] = [45]
    # atmos.values["gamma"][1,2] = [45]
    atmos.values["gamma"] *= np.pi/180 # [deg --> rad]

    atmos.values["chi"] = np.zeros((2,3,1))
    # atmos.values["chi"][0,0] = [0]
    # atmos.values["chi"][0,1] = [30]
    # atmos.values["chi"][0,2] = [45]
    # atmos.values["chi"][1,0] = [60]
    # atmos.values["chi"][1,1] = [90]
    # atmos.values["chi"][1,2] = [130]
    # atmos.values["chi"][0,0] = [0]
    # atmos.values["chi"][0,1] = [0]
    # atmos.values["chi"][0,2] = [0]
    # atmos.values["chi"][1,0] = [0]
    # atmos.values["chi"][1,1] = [45]
    # atmos.values["chi"][1,2] = [90]
    atmos.values["chi"] *= np.pi/180 # [deg --> rad]
 
    atmos.build_from_nodes(in_data.ref_atm)

    print("vmac: ", atmos.vmac/1e3)

    spec,_,_ = globin.compute_spectra(in_data, atmos, False, False)
    globin.atmos.broaden_spectra(spec, atmos)
    globin.spectrum_path = "obs_2x3_from_nodes_NO_MAG.fits"
    globin.atmos.save_spectra(spec, globin.spectrum_path)

    # fig = plt.figure(figsize=(12,14))
    for idx in range(atmos.nx):
        for idy in range(atmos.ny):
            globin.visualize.plot_atmosphere(atmos, ["temp", "vz", "mag", "gamma", "chi"], idx, idy)
    globin.visualize.show()
    atmos.save_atmosphere("atmosphere_2x3_from_nodes_NO_MAG.fits")

    # print(spec[0,0,:,0])

    spec = globin.spec.Observation(globin.spectrum_path)

    for idx in range(atmos.nx):
        for idy in range(atmos.ny):
            # fig = plt.figure(figsize=(12,10))
            globin.plot_spectra(spec, idx=idx, idy=idy)
            # plt.plot(spec[0,0,:,0], spec[0,0,:,1])
    plt.show()
            # plt.savefig("results/mag_field_test/stokes_m{:04.0f}_g{:03.0f}_a{:03.0f}.png".format(mag,gamma,chi))
    # plt.savefig(f"results/mag_field_test/stokes_gamma_compare.png")
    # plt.show()

def RHatm2Spinor(in_data, atmos, fpath="globin_node_atm_SPINOR.fits"):
    spinor_atm = np.zeros((12, atmos.nx, atmos.ny, atmos.nz))

    spec, atm, height = globin.compute_spectra(in_data, atmos, False, False)

    spinor_atm[0] = atmos.logtau
    spinor_atm[1] = height
    spinor_atm[2] = atmos.data[:,:,1,:] # temperature [K]

    nH = 0
    for i_ in range(6):
        nH += atmos.data[:,:,8+i_,:]
    ni = np.einsum("ijk,l->ijkl", nH, 10**(atom_abundance - 12))
    density = np.einsum("ijkl,l->ijk", ni, atom_mass*m_p)
    # density += atmos.data[:,:,2,:] * m_e
    density += atm[:,:,2,:] * m_e
    spinor_atm[6] = density # density [g/cm3]
    
    # gas_pressure = (np.sum(ni, axis=-1) + atmos.data[:,:,2,:]) * k_b * atmos.data[:,:,1,:] # [Pa]
    gas_pressure = (np.sum(ni, axis=-1) + atm[:,:,2,:]) * k_b * atmos.data[:,:,1,:] # [Pa]
    spinor_atm[3] = gas_pressure * 10 # gas pressure [dyn/cm2]

    # electron_pressure = atmos.data[:,:,2,:] * k_b * atmos.data[:,:,1,:] # [Pa]
    electron_pressure = atm[:,:,2,:] * k_b * atmos.data[:,:,1,:] # [Pa]
    spinor_atm[4] = electron_pressure # electron pressure [dyn/cm2]

    spinor_atm[7] = atmos.data[:,:,5,:] * 1e4 # magnetic field [G]
    spinor_atm[8] = atmos.data[:,:,4,:] * 1e5 # turbulent velocity [cm/s]
    spinor_atm[9] = atmos.data[:,:,3,:] * 1e5 # turbulent velocity [cm/s]
    spinor_atm[10] = atmos.data[:,:,6,:] # inclination [rad]
    spinor_atm[11] = atmos.data[:,:,7,:] # azimuth [rad]

    primary = fits.PrimaryHDU(spinor_atm)
    hdulist = fits.HDUList([primary])
    hdulist.writeto(fpath, overwrite=True)

    np.savetxt("globin_atm_0_0.dat", spinor_atm[:,0,0,:].T, header=f"{atmos.nz}\tdummy.dat", comments="", 
        fmt="%3.2f %5.4e %6.2f %5.4e %5.4e %5.4e %5.4e %5.4e %5.4e %5.4e %5.4f %5.4f")