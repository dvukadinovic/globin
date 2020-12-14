import numpy as np
import matplotlib.pyplot as plt

import globin

def get_func3(a,b,c,d):
    return lambda t: (1-t)*(1-t)*(1-t)*a + 3*(1-t)*(1-t)*t*b + 3*(1-t)*t*t*c + t*t*t*d

def get_func2(a,b,c):
    return lambda t: (1-t)*(1-t)*a + t*t*c + 2*t*(1-t)*b

def bezier_spline(x, y, xintp, K0=0, Kn=0, degree=3):
    """

    Bezier spline interpolation based on paper by de la Cruz Rodirguez &
    Piskunov (2013).

    Parameters:
    ---------------
    x : ndarray
        x-axis values of data points.
    y : ndarray
        y-axis values of data points.
    xintp : ndarray
        x values for which to calculate spline interpolation.
    K0 : float, optional
        derivative of spline at first point. Default 0.
    Kn : float, optional
        derivative of spline at last point. Default 0.
    degree : int, optional
        spline degree (k=2 quadratic or k=3 cubic). Default 3 (cubic interpolation).

    Return:
    ---------------
    yintp : ndarray
        interpolated values at 'xintp' positions.
    """
    n = len(x)
    x = np.round(x, 2)
    # xintp = np.round(xintp, 2)

    # in single node case, we return constant value
    if n==1:
        return np.ones(len(xintp), dtype=np.float64) * y

    curves = [0]*(n-1)

    y_prim = [0]*n
    # derivative in first knot
    y_prim[0] = K0
    # derivative in last knot
    y_prim[-1] = Kn
    
    # derivative in inner knots
    for i in range(1,n-1):
        dx_0 = x[i+1] - x[i]
        dx_1 = x[i] - x[i-1]
        alpha = (1 + dx_0 / (dx_0 + dx_1))/3
        d_0 = (y[i+1] - y[i]) / dx_0
        d_1 = (y[i] - y[i-1]) / dx_1
        if d_0*d_1>0:
            eta = alpha*d_0 + (1-alpha)*d_1
            y_prim[i] = d_0*d_1 / eta

    # construct curves for each segment
    for i in range(n-1):
        dx_0 = x[i+1] - x[i]
        if degree==2:
            c0 = y[i] + dx_0/2 * y_prim[i]
            c1 = y[i+1] - dx_0/2 * y_prim[i+1]
            c = (c0 + c1)/2
            curves[i] = get_func2(y[i],c,y[i+1])
        if degree==3:
            e = y[i] + dx_0/3 * y_prim[i]
            f = y[i+1] - dx_0/3 * y_prim[i+1]
            curves[i] = get_func3(y[i], e, f, y[i+1])

    # evaluete spline at given points
    yintp = np.zeros(len(xintp), dtype=np.float64)
    for j in range(len(xintp)):
        for k in range(n-1):
            if (xintp[j]>=x[k]) and (xintp[j]<=x[k+1]):
                u = (xintp[j] - x[k] ) / (x[k+1] - x[k])
                yintp[j] = curves[k](u)
        if xintp[j]<x[0]:
            yintp[j] = y[0] + y_prim[0]*(xintp[j] - x[0])
        if xintp[j]>=x[-1]:
            yintp[j] = y[-1] + y_prim[-1]*(xintp[j] - x[-1])

    return yintp

def construct_atmosphere_from_nods(in_data):
    atmos = in_data.atm
    obs = in_data.obs
    print(atmos.values)

    atmos.values["temp"] = np.zeros((2,3,3))
    atmos.values["temp"][0,0] = [4500,5500,7300]
    atmos.values["temp"][0,1] = [4400,5400,7200]
    atmos.values["temp"][0,2] = [4000,5400,7500]
    atmos.values["temp"][1,0] = [4500,5600,7500]
    atmos.values["temp"][1,1] = [4800,5300,7600]
    atmos.values["temp"][1,2] = [4200,5200,7400]

    atmos.values["vz"] = np.zeros((2,3,2))
    atmos.values["vz"][0,0] = [1,-0.5]
    atmos.values["vz"][0,1] = [0.5,0.1]
    atmos.values["vz"][0,2] = [0,0]
    atmos.values["vz"][1,0] = [0.75,-0.1]
    atmos.values["vz"][1,1] = [-1,0]
    atmos.values["vz"][1,2] = [-1.2,0.5]

    atmos.values["vmic"] = np.zeros((2,3,1))
    atmos.values["vmic"][0,0] = [0.0]
    atmos.values["vmic"][0,1] = [0.25]
    atmos.values["vmic"][0,2] = [0.50]
    atmos.values["vmic"][1,0] = [0.75]
    atmos.values["vmic"][1,1] = [1.0]
    atmos.values["vmic"][1,2] = [1.25]

    atmos.values["mag"] = np.zeros((2,3,2))
    atmos.values["mag"][0,0] = [100,250]
    atmos.values["mag"][0,1] = [0,0]
    atmos.values["mag"][0,2] = [500,1500]
    atmos.values["mag"][1,0] = [300,350]
    atmos.values["mag"][1,1] = [2000,2500]
    atmos.values["mag"][1,2] = [250,500]
    atmos.values["mag"] /= 1e4 # [G --> T]

    atmos.values["gamma"] = np.zeros((2,3,1))
    atmos.values["gamma"][0,0] = [70]
    atmos.values["gamma"][0,1] = [45]
    atmos.values["gamma"][0,2] = [10]
    atmos.values["gamma"][1,0] = [30]
    atmos.values["gamma"][1,1] = [60]
    atmos.values["gamma"][1,2] = [0]
    atmos.values["gamma"] *= np.pi/180 # [deg --> rad]

    atmos.values["chi"] = np.zeros((2,3,1))
    atmos.values["chi"][0,0] = [0]
    atmos.values["chi"][0,1] = [30]
    atmos.values["chi"][0,2] = [45]
    atmos.values["chi"][1,0] = [60]
    atmos.values["chi"][1,1] = [90]
    atmos.values["chi"][1,2] = [130]
    atmos.values["chi"] *= np.pi/180 # [deg --> rad]

    atmos.nx = 2
    atmos.ny = 3

    atmos.atm_name_list = []
    for idx in range(atmos.nx):
        for idy in range(atmos.ny):
            name = f"atmospheres/atm_{idx}_{idy}"
            atmos.atm_name_list.append(name)

    atmos.build_from_nodes(in_data.ref_atm)

    globin.spectrum_path = "obs_2x3_from_nodes.fits"
    spec, atm = globin.compute_spectra(in_data, atmos, True, True)

    # fig = plt.figure(figsize=(12,14))
    new_atmos = globin.Atmosphere()
    new_atmos.data = atm
    # for idx in range(atmos.nx):
    #     for idy in range(atmos.ny):
    #         globin.visualize.plot_atmosphere(new_atmos, idx, idy)
    # globin.visualize.show()
    new_atmos.save_cube("atmosphere_2x3_from_nodes.fits")

    for idx in range(atmos.nx):
        for idy in range(atmos.ny):
            # plt.plot(obs.data[0,0,:-1,0], obs.data[0,0,:-1,1])
            plt.plot(spec[idx,idy,:-1,0], spec[idx,idy,:-1,1])
    plt.show()

def save_chi2(chi2, fpath="chi2.fits"):
    from astropy.io import fits

    primary = fits.PrimaryHDU(chi2)
    hdulist = fits.HDUList([primary])
    hdulist.writeto(fpath, overwrite=True)

if __name__=="__main__":
    # example from de la Cruz Rodriguez et al. (2019)
    x = np.array([-3,-2,-1.95, -1, 0.4, 2, 3.2])
    y = np.array([0.2, 0, 0.6, 0.55, 0.29, 0.21, 0.4])
    xintp = np.linspace(x[0],x[-1], num=601)

    yintp = bezier_spline(x,y,xintp, degree=3)

    plt.plot(x,y, "ro", label="knots")
    plt.plot(xintp,yintp,"k-", label="Bezier-3")

    yintp = bezier_spline(x,y,xintp, degree=2)
    plt.plot(xintp, yintp,"k--", label="Bezier-2")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend()

    plt.show()