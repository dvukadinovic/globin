import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import subprocess as sp
import multiprocessing as mp
import copy

from scipy.constants import m_e, m_p
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

def construct_atmosphere_from_nodes(node_atmosphere_path, atm_range, output_atmos_path=None):
    from scipy.interpolate import splev, splrep

    atmos = globin.read_node_atmosphere(node_atmosphere_path)
    # ref_atmos = globin.Atmosphere(ref_atmos_path)

    tck = splrep(globin.falc_logt, globin.falc_ne)
    ne = splev(atmos.logtau, tck)
    atmos.data[:,:,2,:] = ne
    ref_atmos = globin.Atmosphere(nx=atmos.nx, ny=atmos.ny)
    atmos.build_from_nodes(ref_atmos, False)
    atmos.distribute_hydrogen(globin.falc.data[0], globin.falc.data[2], globin.falc.data[3], globin.falc.data[4], atom_abundance)

    if output_atmos_path is not None:
        atmos.save_atmosphere(output_atmos_path)
    
    if atm_range is not None:
        xmin, xmax, ymin, ymax = atm_range
        atmos.data = atmos.data[xmin:xmax, ymin:ymax]
        atmos.nx, atmos.ny, atmos.npar, atmos.nz = atmos.data.shape
    
    atmos.split_cube()

    print("Constructed atmosphere from nodes: {}".format(node_atmosphere_path))
    print("  (nx, ny, nz, npar) = ({0}, {1}, {2}, {3})".format(atmos.nx, atmos.ny, atmos.nz, atmos.npar))

    return atmos

def make_synthetic_observations(atmos, rh_spec_name, wavelength, vmac, noise, node_atmosphere_path=None, atm_range=None):
    if atmos is None:
        atmos = construct_atmosphere_from_nodes(node_atmosphere_path, atm_range)
        atmos.vmac = vmac
        atmos.save_atmosphere("atmosphere_2x3_from_nodes.fits")

    if globin.mode!=0:
        print(f"  Current mode is {globin.mode}.")
        print("  We can make synthetic observations only in mode = 0.")
        print("  Change it before running the script again.")
        sys.exit()

    spec, _, _ = globin.compute_spectra(atmos, rh_spec_name, wavelength)
    spec.broaden_spectra(atmos.vmac)
    spec.add_noise(noise)
    spec.save(globin.spectrum_path, wavelength)

    for idx in range(atmos.nx):
        for idy in range(atmos.ny):
            globin.plot_atmosphere(atmos, ["temp", "vz", "mag", "gamma", "chi"], idx, idy)
    plt.show()

    for idx in range(atmos.nx):
        for idy in range(atmos.ny):
            globin.plot_spectra(spec, idx=idx, idy=idy)
    plt.show()

    return spec

def RHatm2Spinor(in_data, atmos, fpath="globin_node_atm_SPINOR.fits"):
    spinor_atm = np.zeros((12, atmos.nx, atmos.ny, atmos.nz))

    spec, atm, height = globin.compute_spectra(atmos, in_data.rh_spec_name, in_data.wavelength)

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

def chi2_hypersurface(pars, init):
    atmos = init.atm
    obs = init.obs
    weights = init.weights
    noise = init.noise

    shape = []

    par_names = list(pars.keys())

    # initialize parameters
    # for parameter in pars:
    #     ptype, pvalues = pars[parameter]
    #     # ptype for global parameters is None
    #     if ptype==None:
    #         atmos.global_pars[parameter] = pvalues[0]
    #         shape.append(len(pvalues))
    #         # if we have loggf or dlam parameter, we need to write those data
    #         # into the file first
    #         if parameter=="loggf" or parameter=="dlam":
    #             pass
    #     # for local parameters it is nodes index (starting from 0)
    #     else:
    #         # atmos.values[parameter][:,:,ptype] = pvalues[0]
    #         shape.append(len(pvalues))

    par = par_names[0]
    shape = len(pars[par][1])
    chi2 = np.ones(shape)
    atmos.build_from_nodes(init.ref_atm)

    loggf0 = [-2.172, -1.806, -0.084, -0.857, -0.781, -2.419,  0.080, -0.515, -0.087, -1.928, -1.870, -0.714, -1.426, -3.513, -1.160, -2.905, -2.576, -0.547]

    #--- vmac
    # for i_, vmac in enumerate(pars["vmac"][1]):
    #     atmos.vmac = pars["vmac"][1][i_]
    #     print(i_, atmos.vmac/1e3)
    #     spec,_,_ = globin.compute_spectra(atmos, init.rh_spec_name, init.wavelength)
    #     spec.broaden_spectra(atmos.vmac)

    #     diff = obs.spec - spec.spec
    #     diff *= weights

    #     chi2[i_,0] = np.sum(diff[0,0]**2) # / noise_stokes**2) / dof

    #--- loggf
        # ind = pars["loggf"][0]-1
    # for ind in range(0,17):
    # for ind in [0,1,2,3]:
    # for ind in [3,5,6]:
    # for ind in [0,1,2,3,4,5,6,7]:
    # for ind in [8,9,10,11,12,13]:
    for ind in range(18):
    # for ind in [8,9,10]:
    # for ind in [11,12,13]:
    # for ind in [14,15,16,17]:
        print(ind+1)
        for i_, val in enumerate(pars[par][1]):
            init.write_line_par(val, ind, par)
            spec,_,_ = globin.compute_spectra(atmos, init.rh_spec_name, init.wavelength)
            spec.broaden_spectra(atmos.vmac)

            # print(i_)

            diff = obs.spec - spec.spec
            diff *= weights
            chi2[i_] = np.sum(diff[0,0]**2)

            # plt.subplot(1,2,1)
            # globin.plot_spectra(obs, 0, 0)
            # globin.plot_spectra(spec, 0, 0)

            # plt.subplot(1,2,2)
            # diff = copy.deepcopy(obs)
            # diff.spec -= spec.spec
            # globin.plot_spectra(diff, 0, 0)

            # plt.show()
 
        if par=="loggf":
            init.write_line_par(loggf0[ind], ind, par)
        elif par=="dlam":
            init.write_line_par(0, ind, par)

        plt.plot(pars[par][1], chi2)
        plt.yscale("log")
        plt.xlabel(r"$\log (gf)$")
        plt.ylabel(r"$\chi ^2$")
        plt.show()

    # for i_, vmac in enumerate(pars["vmac"][1]):
    #     atmos.vmac = pars["vmac"][1][i_]
    #     print(i_, atmos.vmac/1e3)
    #     for j_, temp in enumerate(pars["temp"][1]):
    #         atmos.values["temp"][:,:,pars["temp"][0]] = pars["temp"][1][j_]

    #         atmos.build_from_nodes(init.ref_atm)

    #         spec,_,_ = globin.compute_spectra(atmos, init.rh_spec_name, init.wavelength)
    #         spec.broaden_spectra(atmos.vmac)

    #         diff = obs.spec - spec.spec
    #         diff *= weights

    #         chi2[i_,j_] = np.sum(diff[0,0]**2) # / noise_stokes**2) / dof

    #         plt.plot(obs.spec[0,0,:,0])
    #         plt.plot(spec.spec[0,0,:,0])
    #         plt.show()

    # plt.imshow(np.log10(chi2), aspect="auto", cmap="gnuplot")
    # plt.colorbar()
    # plt.xticks(list(range(shape[1])), np.round(pars[par_names[1]][1]/1e3, decimals=1))
    # plt.yticks(list(range(shape[0])), np.round(pars[par_names[0]][1]/1e3, decimals=1))
    # plt.xlabel(r"T [kK]")
    # plt.ylabel(r"$v_{mac}$ [km/s]")
    # pltobs, 0, 0.show()

def claculate_chi2(init):
    obs = init.obs
    atmos = init.atm
    pass
