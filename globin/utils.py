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

def remove_dirs():
    """
    We remove working dirs located in rh/rhf1d if we fail to run RH
    or if we have finished synthesis/inversion.

    In case of an error, logs of running RH will be saved and could be
    investigated further.
    """
    for threadID in range(globin.n_thread):
        out = sp.run(f"rm -r {globin.rh_path}/rhf1d/{globin.wd}_{threadID+1}", 
                shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
        if out.returncode!=0:
            print(out.stdout)
            sys.exit()

def construct_atmosphere_from_nodes(node_atmosphere_path, atm_range=None, vmac=0, output_atmos_path=None):
    atmos = globin.read_node_atmosphere(node_atmosphere_path)

    # atmos.data = np.zeros((atmos.nx, atmos.ny, atmos.npar, atmos.nz), dtype=np.float64)
    # atmos.data[:,:,0,:] = atmos.logtau
    atmos.vmac = vmac
    atmos.interpolate_atmosphere(atmos.logtau, globin.falc.data)
    atmos.build_from_nodes(False)

    if output_atmos_path is not None:
        atmos.save_atmosphere(output_atmos_path)
    
    if atm_range is not None:
        xmin, xmax, ymin, ymax = atm_range
        atmos.data = atmos.data[xmin:xmax, ymin:ymax]
        atmos.nx, atmos.ny, atmos.npar, atmos.nz = atmos.data.shape
    
    # atmos.split_cube()

    print("Constructed atmosphere from nodes: {}".format(node_atmosphere_path))
    print("  (nx, ny, nz, npar) = ({0}, {1}, {2}, {3})".format(atmos.nx, atmos.ny, atmos.nz, atmos.npar))

    return atmos

def make_synthetic_observations(atmos, noise, atm_fpath=None):
    if globin.mode!=0:
        print(f"  Current mode is {globin.mode}.")
        print("  We can make synthetic observations only in mode = 0.")
        print("  Change it before running the script again.")
        globin.remove_dirs()
        sys.exit()
    
    spec, atm, _ = globin.compute_spectra(atmos)
    spec.broaden_spectra(atmos.vmac)
    spec.add_noise(noise)
    spec.save(globin.output_spectra_path, globin.wavelength)

    if atm_fpath is not None:
        atm.save_atmosphere(atm_fpath)

    # for idx in range(atmos.nx):
    #     for idy in range(atmos.ny):
    #         # globin.plot_atmosphere(atmos, ["temp", "vz", "mag", "gamma", "chi"], idx, idy)
    #         globin.plot_atmosphere(atmos, ["ne"], idx, idy, color="tab:blue")
    #         globin.plot_atmosphere(atm, ["ne"], idx, idy, color="tab:orange")
    # plt.show()

    for idx in range(atmos.nx):
        for idy in range(atmos.ny):
            globin.plot_spectra(spec, idx=idx, idy=idy)
            plt.savefig(f"runs/{globin.wd}/spec_x{idx}y{idy}.png")
            plt.close()

    # globin.remove_dirs()

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

def calculate_chi2(init, pars, fname):
    # pars:
    #   atmospheric --> [node, values, idx, idy]
    #   atomic      --> [line, values]
    #   vmac        --> [None, values]
    
    noise_stokes = 1
    dof = 1

    # get parameter names
    par_names = [item[0] for item in pars]

    unique_names = list(set(par_names))
    atmos_par_names = [name for name in unique_names if name in ["temp", "vz", "vmic", "mag", "gamma", "chi"]]
    
    # set all node values to expected ones;
    # for that we use reference atmosphere
    for par in atmos_par_names:
        nodes = init.atm.nodes[par]

        par_id = init.ref_atm.par_id[par]
        value = init.ref_atm.data[:,:,par_id,:]

        for i_,node in enumerate(nodes):
            ind = np.argmin(np.abs(init.ref_atm.logtau - node))
            par_in_node = value[:,:,ind]
            init.atm.values[par][:,:,i_] = par_in_node

    # set shape of chi2
    shape = [len(item[2]) for item in pars]
    chi2 = np.zeros(shape)

    # number of parameter combinations
    N = np.prod(shape)

    # set a list if indice for each parameter combination
    ranges = [np.arange(size) for size in shape]
    inds = np.meshgrid(*ranges, indexing="ij")
    inds = [item.flatten() for item in inds]

    #--- for each combination compute spectra and calculate chi2
    for j_,ind in enumerate(zip(*inds)):
        print("{:3.1f} %".format( (j_+1)/N * 100 ))

        # set parameters
        for i_,par in enumerate(par_names):
            node = pars[i_][1]
            value_ind = ind[i_]
            value = pars[i_][2][value_ind]
            try:
                idx, idy = pars[i_][3], pars[i_][4]
            except:
                idx, idy = None, None
            args = par, node, value, idx, idy
            init = set_parameter(init, args)

        # build atmosphere and compute spectra
        init.atm.build_from_nodes()
        spec, _, _ = globin.compute_spectra(init.atm, init.rh_spec_name, init.wavelength)

        # compute chi2
        diff = init.obs.spec - spec.spec
        chi2[ind] = np.sum(diff**2 / noise_stokes**2 * init.wavs_weight**2) / dof

    #--- save chi2 into fits file
    primary = fits.PrimaryHDU(chi2)
    primary.name = "chi2_vals"

    hdulist = fits.HDUList([primary])

    for item in pars:
        parameter = item[0]
        node = item[1]
        values = item[2]
        try:
            idx, idy = item[3], item[4]
        except:
            idx, idy = -1, -1

        par_hdu = fits.ImageHDU(values)
        if parameter in ["temp", "vz", "vmic", "mag", "gamma", "chi"]:
            par_hdu.name = f"{parameter}_x{idx}_y{idy}_n{node}"
        else:
            par_hdu.name = f"{parameter}_line{node}"

        par_hdu.header["XPOS"] = (idx, " x position of atmosphere")
        par_hdu.header["YPOS"] = (idx, " y position of atmosphere")
        par_hdu.header["NODE"] = (node, " node index")

        hdulist.append(par_hdu)
    
    hdulist.writeto(fname, overwrite=True)

    return chi2

def set_parameter(init, args):
    # koji parametar --> parameters
    # ind / line_no
    # value
    # idx, idy = None, None
    # 
    atmos_pars = ["temp", "vz", "vmic", "mag", "gamma", "chi"]
    global_pars = ["vmac", "loggf", "dlam"]

    parameter, ind, value, idx, idy = args

    #--- set the initial parameters
    if parameter in atmos_pars:
        init.atm.values[parameter][idx, idy, ind] = value
    if parameter in global_pars:
        if ind is not None:
            init.atm.global_pars[parameter][ind] = value
            init.write_line_par(value, ind, parameter)
        else:
            init.atm.global_pars[parameter] = value

    return init
