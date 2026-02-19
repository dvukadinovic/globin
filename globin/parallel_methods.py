import pyrh
import numpy as np
from scipy.interpolate import splev

from .tools import get_K0_Kn
from .tools import bezier_spline
from .tools import spline_interpolation
from .constants import Tmin
from .constants import Tmax

# from .models import FALC_temp_tck
import globin

def _makeHSE(args):
    """
    Parallelized call from makeHSE() function.
    """
    global_pars, fudge_lam, fudge_value, scale, temp, ne, nHtot, rho, pg = args

    pyrh.hse(cwd=global_pars["cwd"], 
            atm_scale=global_pars["atm_scale"],
            scale=scale,
            temp=temp,
            ne=ne,
            nHtot=nHtot,
            rho=rho,
            pg=pg,
            # pg_top=100,
            pg_top=pg[0]/10,
            fudge_wave=fudge_lam, 
            fudge_value=fudge_value,
            atomic_number=global_pars["atomic_number"], 
            atomic_abundance=global_pars["atomic_abundance"],
            full_output=False)

    return np.vstack((ne/1e6, nHtot/1e6))

def add_node(new_node, x, y, ymin, ymax, where="beginning"):
    """
    Add the 'new_node' point to the atmosphere as a node by linear extrapolation
    to the top.

    Before adding the value, check if it is in the boundaries.

    Key 'where' regulates where we add this node: at the top ('beginning') or
    at the bottom ('end').
    """
    if where=="beginning":
        K0 = (y[1]-y[0])/(x[1]-x[0])
        n = y[0] - K0*x[0]
    if where=="end":
        K0 = (y[-1]-y[-2])/(x[-1]-x[-2])
        n = y[-1] - K0*x[-1]
    y0 = K0*new_node + n
    if ymin is not None:
        if y0<ymin:
            y0 = ymin
    if ymax is not None:
        if y0>ymax:
            y0 = ymax
    if where=="beginning":
        y = np.append(y0, y)
        x = np.append(new_node, x)
    if where=="end":
        y = np.append(y, y0)
        x = np.append(x, new_node)

    return x, y

def _build_from_nodes(args):
    """
    Parallelized call from self.build_from_nodes() function.
    """
    FALC_temp_tck = globin.FALC_temp_tck

    global_pars = args[0]
    nodes = global_pars["nodes"]
    interpolation_method = global_pars["method"]
    logtau = global_pars["logtau"]
    spline_tension = global_pars["spline_tension"]
    limits = global_pars["limits"]
    degree = global_pars["degree"]

    values = args[1]
    parameters = list(values.keys())
    
    Npars = len(parameters)
    data = np.array(args[2:2+Npars])

    Nz = len(data[0])

    for idp, parameter in enumerate(parameters):
        # K0, Kn by default; True for vmic, mag, gamma and chi
        # K0, Kn = None, None
        K0, Kn = 0, 0

        x = nodes[parameter]
        y = values[parameter]

        # if we have a single node
        if len(x)==1:
            y_new = np.ones_like(Nz) * y
            data[idp] = y_new
            continue

        # for 2+ number of nodes
        if parameter=="temp":
            if interpolation_method=="bezier":	
                K0 = (y[1]-y[0]) / (x[1]-x[0])
                # bottom node slope for extrapolation based on temperature gradient from FAL C model
                # if Tmax<(y[0] + K0 * (logtau[0]-x[0])):
                # 	K0 = (Tmax - y[0]) / (logtau[0] - x[0])
                Kn = splev(x[-1], FALC_temp_tck, der=1)
            if interpolation_method=="spline":
                # add top of the atmosphere as a node (ask SPPINOR devs why ...)
                # to the bottom we assume that the gradient is based only on the node positions;
                # this is not fully reallistic thing to do, but... I do not wanna implement extrapolation
                # using adiabatic and HSE assumption like in SPINOR for now to show similarities between them
                x, y = add_node(logtau[0], x, y, Tmin, Tmax)
                K0, Kn = get_K0_Kn(x, y, tension=spline_tension)
            
            # check if extrapolation at the top atmosphere point goes below the minimum
            # if does, change the slope so that at top point we have Tmin (globin.limit_values["temp"][0])
            if Tmin>(y[0] + K0 * (logtau[0]-x[0])):
                K0 = (Tmin - y[0]) / (logtau[0] - x[0])
            # temperature can not go below 1900 K because the RH will not compute spectrum (dunno why)
            
        elif parameter in ["gamma", "chi"]:
            if interpolation_method=="bezier":
                K0 = (y[1]-y[0]) / (x[1]-x[0])
                Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
            if interpolation_method=="spline":
                x, y = add_node(logtau[0], x, y, None, None)
                K0, Kn = get_K0_Kn(x, y, tension=spline_tension)
        
        elif parameter in ["vz", "mag", "vmic"]:
            if interpolation_method=="bezier":
                K0 = (y[1]-y[0]) / (x[1]-x[0])
                Kn = (y[-1]-y[-2]) / (x[-1]-x[-2])
            if interpolation_method=="spline":
                x, y = add_node(logtau[0], x, y, limits[parameter].min[0], limits[parameter].max[0])
                K0, Kn = get_K0_Kn(x, y, tension=spline_tension)
            
            if parameter in ["mag", "vmic"]:
                # check if extrapolation at the top atmosphere point goes below the minimum
                # if does, change the slopte so that at top point we have parameter_min (globin.limit_values[parameter][0])
                if limits[parameter].min[0]>(y[0] + K0 * (logtau[0]-x[0])):
                    K0 = (limits[parameter].min[0] - y[0]) / (logtau[0] - x[0])
                # if limits[parameter].max[0]<(y[0] + K0 * (logtau[0]-x[0])):
                # 	K0 = (limits[parameter].max[0] - y[0]) / (logtau[0] - x[0])
                # similar for the bottom for maximum/min values
                # if limits[parameter].max[0]<(y[-1] + Kn * (logtau[-1]-x[-1])):
                # 	Kn = (limits[parameter].max[0] - y[-1]) / (logtau[-1] - x[-1])
                if limits[parameter].min[0]>(y[-1] + Kn * (logtau[-1]-x[-1])):
                    Kn = (limits[parameter].min[0] - y[-1]) / (logtau[-1] - x[-1])

        if interpolation_method=="bezier":
            y_new = bezier_spline(x, y, logtau, K0=K0, Kn=Kn, degree=degree, extrapolate=True)
        if interpolation_method=="spline":
            y_new = spline_interpolation(x, y, logtau, tension=spline_tension, K0=K0, Kn=Kn)

        data[idp] = y_new

    return data

def _compute_spectra_sequential(args):
    global_pars, data, spec, rfs, loggf, dlam, fudge_wave, fudge = args

    pyrh.compute1d(cwd=global_pars["cwd"], 
                    mu=global_pars["mu"],
                    spectrum=spec,
                    atm_scale=global_pars["atm_scale"],
                    atmosphere=data, 
                    wave=global_pars["wavelength"],
                    loggf_ids=global_pars["loggf_ids"], 
                    loggf_values=loggf,
                    lam_ids=global_pars["dlam_ids"],
                    lam_values=dlam,
                    fudge_wave=fudge_wave, 
                    fudge_value=fudge,
                    atomic_number=global_pars["atomic_number"],
                    atomic_abundance=global_pars["abundances"],
                    get_atomic_rfs=global_pars["get_atomic_rfs"],
                    rfs=rfs,
                    get_populations=global_pars["get_populations"])

    if global_pars["get_atomic_rfs"]:
        return np.concatenate((spec[...,np.newaxis], rfs), axis=-1)

    return spec