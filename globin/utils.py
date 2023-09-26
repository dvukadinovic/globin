import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import subprocess as sp
import multiprocessing as mp
import copy
import pyrh
from scipy.interpolate import splrep, splev
from scipy.interpolate import interp1d

from .makeHSE import makeHSE

from scipy.constants import m_e, m_p
from scipy.constants import k as k_b

m_p *= 1e3
m_e *= 1e3

import globin

def convert_spinor_inversion(fpath, get_obs=False, inversion=True):
    parameter_relay = {"TEMPE" : "temp",
                       "VELOS" : "vz",
                       "VMICI" : "vmic",
                       "BFIEL" : "mag",
                       "GAMMA" : "gamma",
                       "AZIMU" : "chi",
                       "LOGGF" : "loggf"}
    #--- inverted profiles
    inv_spinor = fits.open(f"{fpath}/inverted_profs.1.fits")[0]
    wlref = inv_spinor.header["WLREF"]
    wlmin, wlmax = inv_spinor.header["WLMIN"], inv_spinor.header["WLMAX"]
    nwl = inv_spinor.header["NWL"]
    inv_spinor_lam = np.linspace(wlref+wlmin, wlref+wlmax, num=nwl)

    #--- observations seen by SPINOR (they are same as globin ones; have checked)
    # spinor_obs = fits.open(f"{fpath}/inverted_obs.1.fits")[0].data
    # spinor_lam_obs = fits.open(f"{fpath}/inverted_lam.1.fits")[0].data[0,0]
    # spinor_lam_obs += wlref

    #--- inverted atmosphere (node values)
    hdu = fits.open(f"{fpath}/inverted_atmos.fits")[0]
    par_header = hdu.header
    par_data = hdu.data

    # get the chi2 values
    chi2 = globin.input.Chi2(chi2=par_data[-1])

    read_atmos = False
    try:
        hdu = fits.open(f"{fpath}/inverted_atmos_maptau.1.fits")[0]
        _header = hdu.header
        spinor = hdu.data#[:,:-1,:,:]
        npar, nz, nx, ny = spinor.shape
        spinor_logtau = np.linspace(_header["LTTOP"], _header["LTBOT"], num=nz)
        tmp = np.zeros((12, nz, nx, ny))
        tmp[0,:,0,0] = spinor_logtau
        tmp[1:,:,:,:] = spinor
        tmp = np.swapaxes(tmp, 1, 2)
        tmp = np.swapaxes(tmp, 2, 3)
        atm = globin.atmos.spinor2multi(tmp)
        read_atmos = True
    except:
        pass

    nx, ny, ns, nw = inv_spinor.data.shape

    # create Atmosphere() structure if we have not done the conversion
    if not read_atmos:
        lttop = par_data[par_header["LTTOP"]-1,0,0]
        ltbot = 1
        ltinc = par_data[par_header["LTINC"]-1,0,0]
        nz = int((ltbot - lttop)/ltinc) + 1
        # nz = int(nz) + 1 # dunno why are there +1 more than it should be...
        atm = globin.Atmosphere(nx=nx, ny=ny, nz=nz)
        atm.logtau = np.linspace(lttop, ltbot, num=nz)
    
    # get the nodes for each parameter
    if inversion:
        max_nodes = len(par_header["LGTRF*"])
        if max_nodes!=0:
            start = par_header["LGTRF"]-1
            nodes = par_data[start:start+max_nodes,0,0]

            # add the node values into the atmosphere structure
            for parameter in ["TEMPE", "VELOS", "VMICI", "BFIEL", "GAMMA", "AZIMU"]:
                ind = par_header[f"{parameter}*"]
                nnodes = len(ind)
                if nnodes==0:
                    continue
                start = ind[0] - 1
                if nnodes==1:
                    atm.nodes[parameter_relay[parameter]] = np.array([0])
                else:
                    atm.nodes[parameter_relay[parameter]] = nodes

                fact = 1
                if parameter=="VELOS":
                    fact = -1
                if parameter in ["GAMMA", "AZIMU"]:
                    fact = np.pi/180
                
                atm.values[parameter_relay[parameter]] = np.zeros((nx,ny,nnodes))
                for idn in range(nnodes):
                    atm.values[parameter_relay[parameter]][:,:,idn] = par_data[start+idn] * fact

            for parameter in ["LOGGF"]:
                ind = par_header[f"{parameter}*"]
                nlines = len(ind)
                if nlines==0:
                    continue

                start = ind[0] - 1
                atm.global_pars[parameter_relay[parameter]] = np.empty((atm.nx, atm.ny, nlines))

                for idl in range(nlines):
                    atm.global_pars[parameter_relay[parameter]][:,:,idl] = par_data[start+idl]

    # create the Spectrum() structure
    spec = globin.Spectrum(nx=nx, ny=ny, nw=nw)
    # inv_spinor = np.swapaxes(inv_spinor.data, 2, 3)
    spec.spec[...,0] = inv_spinor.data[:,:,0,:]
    spec.spec[...,1] = inv_spinor.data[:,:,2,:]
    spec.spec[...,2] = inv_spinor.data[:,:,3,:]
    spec.spec[...,3] = inv_spinor.data[:,:,1,:]
    spec.wavelength = inv_spinor_lam/10

    if get_obs:
        data = fits.open(f"{fpath}/inverted_obs.1.fits")[0].data
        nw = data.shape[-1]
        obs = globin.Spectrum(nx=nx, ny=ny, nw=nw)
        obs.spec[...,0] = data[:,:,0]
        obs.spec[...,1] = data[:,:,2]
        obs.spec[...,2] = data[:,:,3]
        obs.spec[...,3] = data[:,:,1]

        return atm, spec, chi2, obs

    return atm, spec, chi2

def construct_atmosphere_from_nodes(node_atmosphere_path, atm_range=None, vmac=0, output_atmos_path=None, intp_method="bezier"):
    atmos = globin.input.read_node_atmosphere(node_atmosphere_path)

    atmos.vmac = vmac
    atmos.interpolation_method = intp_method
    atmos.get_pg_top()
    atmos.build_from_nodes(np.ones((atmos.nx, atmos.ny)))
    atmos.makeHSE()

    if output_atmos_path is not None:
        atmos.save_atmosphere(output_atmos_path)
    
    if atm_range is not None:
        xmin, xmax, ymin, ymax = atm_range
        atmos.data = atmos.data[xmin:xmax, ymin:ymax]
        atmos.nx, atmos.ny, atmos.npar, atmos.nz = atmos.data.shape

    print("Constructed atmosphere from nodes: {}".format(node_atmosphere_path))
    print("  (nx, ny, nz, npar) = ({0}, {1}, {2}, {3})".format(atmos.nx, atmos.ny, atmos.nz, atmos.npar))

    return atmos

def extend(array, N):
    ones = np.ones(N)
    array = np.append(ones*array[0], array)
    array = np.append(array, ones*array[-1])

    return array

def Planck(wave, T):
    wave *= 1e-9 # [nm --> m]
    C1 = 2*globin.PLANCK*globin.LIGHT_SPEED/(wave**3)
    C2 = globin.PLANCK*globin.LIGHT_SPEED/(wave*globin.K_BOLTZMAN*T)

    return C1 / (np.exp(C2) - 1)

def pretty_print_parameters(atmos, flag):
    for parameter in atmos.values:
        print(parameter)
        for idx in range(atmos.nx):
            for idy in range(atmos.ny):
                if flag[idx,idy]==1:
                    if parameter=="gamma":
                        print(f"[{idx+1},{idy+1}] --> ", atmos.values[parameter][idx,idy] * 180/np.pi)
                    elif parameter=="chi":
                        print(f"[{idx+1},{idy+1}] --> ", atmos.values[parameter][idx,idy] * 180/np.pi)
                    else:
                        print(f"[{idx+1},{idy+1}] --> ", atmos.values[parameter][idx,idy])

    if atmos.mode>=2:
        for parameter in atmos.global_pars:
            if parameter=="vmac" or parameter=="stray":
                print(parameter)
                print(atmos.global_pars[parameter])
            else:
                if atmos.line_no[parameter].size > 0:
                    indx = atmos.idx
                    indy = atmos.idy
                    if atmos.mode==3:
                        indx, indy = 0, 0
                    print(parameter)
                    print(atmos.global_pars[parameter][indx,indy])

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
    density = np.einsum("ijkl,l->ijk", ni, globin.atom_mass*m_p)
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

def calculate_chi2(pars, fname):
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
        nodes = globin.atm.nodes[par]

        par_id = globin.ref_atm.par_id[par]
        value = globin.ref_atm.data[:,:,par_id,:]

        for i_,node in enumerate(nodes):
            ind = np.argmin(np.abs(globin.ref_atm.logtau - node))
            par_in_node = value[:,:,ind]
            globin.atm.values[par][:,:,i_] = par_in_node

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
            set_parameter(args)

        # build atmosphere and compute spectra
        globin.atm.build_from_nodes()
        spec, _, _ = globin.compute_spectra(globin.atm)
        if not globin.mean:
            spec.broaden_spectra(globin.atm.vmac)

        # compute chi2
        diff = globin.obs.spec - spec.spec
        
        # plt.plot(globin.obs.spec[0,0,:,0])
        # plt.plot(spec.spec[0,0,:,0])
        # plt.show()

        chi2[ind] = np.sum(diff**2 / noise_stokes**2 * globin.wavs_weight**2) / dof

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

def set_parameter(args):
    # koji parametar --> parameters
    # ind / line_no / None
    # value
    # idx, idy = None, None
    # 
    atmos_pars = ["temp", "vz", "vmic", "mag", "gamma", "chi"]
    global_pars = ["vmac", "loggf", "dlam"]

    parameter, ind, value, idx, idy = args

    #--- set the initial parameters
    if parameter in atmos_pars:
        globin.atm.values[parameter][idx, idy, ind] = value
    if parameter in global_pars:
        if ind is not None:
            globin.atm.global_pars[parameter][ind] = value
            fpath = globin.atm.line_lists_path[0]
            line_no = int(globin.atm.line_no[parameter][ind])
            globin.write_line_par(fpath, value, line_no, parameter)
        else:
            globin.atm.global_pars[parameter] = value
            globin.atm.vmac = value

def _set_keyword(text, key, value, fpath=None):
    """
    Go through 'text' and set the value of 'key' to 'value'. 
    Optionaly, save the text into file 'fpath'

    Parameters:
    -----------
    text : string
        text to be processed
    key : string
        name of the key to be set
    value : string
        value to assign to 'key'
    fpath : string (optional)
        path to which 

    Return:
    -------
    text : string
        text with new 'value' for the 'key'
    -------
    """
    lines = text.split("\n")
        
    line_num = None
    for num, line in enumerate(lines):
        line = line.replace(" ","")
        if len(line)>0:
            if line[0]!="#":
                if key in line:
                    line_num = num
                    break

    if value is not None:
        # if we found a key in the text, change it's value
        if line_num is not None:
            lines[num] = "  " + key + " = " + value
        # if we have not found a key in the text, we add it at the begin
        else:
            line = "  " + key + " = " + value
            lines.insert(0, line)
            pass
    else:
        if line_num is not None:
            lines[num] = "#  " + key + " = " + "None"

    lines = [line + "\n" for line in lines]
    
    # concatanate all the lines into signle string
    # if fpath is not None:
    #     out = open(fpath, "w")
    #     out.writelines(lines)
    #     out.close()
    #     return "".join(lines)
    # else:
    return "".join(lines)

def _write_to_the_file(lines, fpath):
    out = open(fpath, "w")
    out.writelines(lines)
    out.close()

def _slice_line(line, dtype=float, separator=" "):
    # remove 'new line' character
    line = line.rstrip("\n")
    # split line data based on 'space' separation
    line = line.split(separator)
    # filter out empty entries and convert to list
    lista = list(filter(None, line))
    # map read values into given data type
    lista = map(dtype, lista)
    # return list of values
    return list(lista)

#--- routines for smoothing out the inversion parameters 
#    (used for SPINOR; got it from Sebas)
def sqr(x):
    return x*x

def mygsmooth(a, num, sd):
    d = int(num/2)
    dim = a.shape
    ny = dim[0]
    nx = dim[1]

    # make gaussian filter (cut to 1.0 std)
    x, y = np.meshgrid(np.arange(num), np.arange(num))
    r = ((x-d)**2 + (y-d)**2) / (2*sd)**2
    f = np.zeros((ny,nx))
    f[:num, :num] = np.exp(-r)
    nn = np.sum(f)
    f /= nn

    # smooth parameter
    rv = np.zeros((ny,nx))
    for x in range(nx):
        xl=np.max([x-d,0])
        xh=np.min([x+d,nx-1])
        for y in range(ny):
            yl=np.max([y-d,0])
            yh=np.min([y+d,ny-1])
            nn=np.sum(f[d+(yl-y):d+(yh-y),d+(xl-x):d+(xh-x)])
            rv[y,x]=np.sum(f[d+(yl-y):d+(yh-y),d+(xl-x):d+(xh-x)]*a[yl:yh,xl:xh])/nn
    
    return rv

def mysmooth(a, num):
    d = int(num/2)
    dim = a.shape
    ny = dim[0]
    nx = dim[1]
    rv = np.zeros((ny,nx))
    sd = np.zeros((ny,nx))
    
    for x in range(nx):
        xl = np.max([x-d,0])
        xh = np.min([x+d,nx-1])
        mm = xh-xl+1
        for y in range(ny):
            yl = np.max([y-d,0])
            yh = np.min([y+d,ny-1])
            nn = mm*(yh-yl+1)
            rv[y,x] = np.sum(a[yl:yh,xl:xh])/nn
            sd[y,x] = np.sum(sqr(a[yl:yh,xl:xh]-rv[y,x]))/nn

    # return average of the map inside num x num of pixels and
    # standard deviation of this patch
    return rv, sd
    
def azismooth(tmp,num):
    """
    Smoothing out the azimuth. Must be taken care separately 
    because of the ambiguity and switches between pixels.

    This has to be modified a bit for the usage in globin.
    """
    eps=0.01
    ma=360.0
    a1=((tmp+ma) % 180.0)             #   0 .. 180
    a2=((tmp+ma+90.0) % 180.0)-90.0   # -90 ..  90
  
    sa1, d1 = mysmooth(a1,num)
    sa2, d2 = mysmooth(a2,num)
  
    sa2b = ((sa2+180.0) % 180.0)           #   0 .. 180

    
    idx1 = np.where((np.abs(sa1-sa2b) >  eps) & (d2 > d1))  # problem points
    idx2 = np.where((np.abs(sa1-sa2b) >  eps) & (d1 >= d2)) # problem points

    az = (sa1+sa2b)/2.0
    
    nidx1 = len(idx1[0])
    nidx2 = len(idx2[0])

    if nidx1 > 0: 
        for x in range(nidx1):
            az[idx1[0][x],idx1[1][x]]=sa1[idx1[0][x],idx1[1][x]]
    if nidx2 > 0:         
        for x in range(nidx2):
            az[idx2[0][x],idx2[1][x]]=sa2[idx2[0][x],idx2[1][x]]

    return ((az+90.0) % 180.0) - 90.0

def congrid(a, newdims, method='neighbour', centre=True, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[np.float64](a)

    m1 = np.cast[np.int32](minusone)
    ofs = np.cast[np.int32](centre) * 0.5
    old = np.array( a.shape )
    ndims = len(a.shape)
    if len( newdims ) != ndims:
        print("[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray( newdims, dtype=np.int32)
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(np.int32)
        # newa = a[list(cd)]
        newa = a[cd[0],cd[1]]
        return newa

    elif method in ['nearest','linear','cubic']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype=np.float64) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + list(range( ndims - 1 ))
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (n.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None

def get_first_larger_divisor(n, vmin):
    for i in range(vmin+1, n//2 + 1):
        if n % i == 0:
            return i
    return n
