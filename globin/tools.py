import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splrep, splev

def get_func3(a,b,c,d):
    return lambda t: (1-t)*(1-t)*(1-t)*a + 3*(1-t)*(1-t)*t*b + 3*(1-t)*t*t*c + t*t*t*d

def get_func2(a,b,c):
    return lambda t: (1-t)*(1-t)*a + t*t*c + 2*t*(1-t)*b

def bezier_spline(x, y, xintp, K0=None, Kn=None, degree=3, extrapolate=False):
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

    # in single node case, we return constant value
    if n==1:
        return np.ones(len(xintp), dtype=np.float64) * y

    curves = [0]*(n-1)

    y_prim = [0]*n
    # derivative in first knot
    y_prim[0] = (y[1] - y[0])/(x[1] - x[0])
    if K0 is not None:
        y_prim[0] = K0
    # derivative in last knot
    y_prim[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    if Kn is not None:
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

    # evalueate spline at given points
    yintp = np.zeros(len(xintp), dtype=np.float64)
    for j in range(len(xintp)):
        for k in range(n-1):
            if (xintp[j]>=x[k]) and (xintp[j]<=x[k+1]):
                u = (xintp[j] - x[k] ) / (x[k+1] - x[k])
                yintp[j] = curves[k](u)
        # if xintp[j]<x[0]:
        #     yintp[j] = y[0] + y_prim[0]*(xintp[j] - x[0])
        # if xintp[j]>=x[-1]:
        #     yintp[j] = y[-1] + y_prim[-1]*(xintp[j] - x[-1])
    
    if extrapolate:
        yintp[xintp<x[0]] = y_prim[0]*(xintp[xintp<x[0]] - x[0]) + y[0]
        yintp[xintp>x[-1]] = y_prim[-1]*(xintp[xintp>x[-1]] - x[-1]) + y[-1]
    else:
        # constant extrapolation to to top/bottom of atmosphere
        yintp[xintp<x[0]] = y[0]
        yintp[xintp>x[-1]] = y[-1]

    return yintp

def spline_interpolation(xknot, yknot, x, K0=0, Kn=0, degree=3):
    """
    Spline interpolation of node values.

    In the xknot and yknot, we added edges (top and bottom of the atmosphere)
    in order to get the correct smooth extrapolation.
    """

    # in a single node case, we return constant value
    if len(xknot)==3:
        return np.ones(len(x), dtype=np.float64) * yknot[1]

    if len(xknot)-3<degree:
        degree = 2

    tck = splrep(xknot, yknot, k=degree)
    y = splev(x, tck, der=0, ext=0)

    n = yknot[1] - K0*xknot[1]
    y[x<xknot[1]] = K0*x[x<xknot[1]] + n
    
    n = yknot[-2] - Kn*xknot[-2]
    y[x>xknot[-2]] = Kn*x[x>xknot[-2]] + n

    return y

if __name__=="__main__":
    # example from de la Cruz Rodriguez et al. (2019)
    # x = np.array([-3,-2,-1.95, -1, 0.4, 2, 3.2])
    # y = np.array([0.2, 0, 0.6, 0.55, 0.29, 0.21, 0.4])
    x = np.random.random(5)
    x.sort()
    y = np.random.random(5)
    xintp = np.linspace(x[0]-0.05,x[-1]+0.05, num=601)

    yintp = bezier_spline(x, y, xintp, degree=3)

    plt.plot(x, y, "ro", label="knots")
    plt.plot(xintp, yintp,"k-", label="Bezier-3")

    yintp = spline_interpolation(x, y, xintp, degree=3)

    plt.plot(xintp, yintp,"k--", label="spline")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend()

    plt.show()