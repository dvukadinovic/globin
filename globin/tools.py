"""
Stolen from: https://towardsdatascience.com/b%C3%A9zier-interpolation-8033e9a262c2

Thank you!
"""

import numpy as np
import matplotlib.pyplot as plt

def get_func3(a,b,c,d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

def get_func2(a,b,c):
    return lambda t: (1-t)*(1-t)*a + t*t*c + 2*t*(1-t)*b

def bezier_spline(x, y, xintp, K0=0, Kn=0, degree=3):
    """
    Po radu de la Cruz Rodirguez & Piskunov (2013) izracunavam izvode
    polinoma u nodovima (x). Potom trazim Cubic Hermite Spline sa datim
    cvornim tackama i sracunatim izvodima. Deluje pristojno.
    """
    n = len(x)

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

    u = np.linspace(0,1,11)
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

    yintp = np.zeros(len(xintp))
    for j in range(len(xintp)):
        for k in range(n-1):
            if (xintp[j]>=x[k]) and (xintp[j]<=x[k+1]):
                u = (xintp[j] - x[k] ) / (x[k+1] - x[k])
                yintp[j] = curves[k](u)
        if xintp[j]<x[0]:
            yintp[j] = y[0]
        if xintp[j]>x[-1]:
            yintp[j] = y[-1]

    return yintp


if __name__=="__main__":
    # example from de la Cruz Rodriguez et al. (2019)
    x = np.array([-3,-2,-1.95, -1, 0.4, 2, 3.2])
    y = np.array([0.2, 0, 0.6, 0.55, 0.29, 0.21, 0.4])
    xintp = np.linspace(x[0],x[-1], num=601)

    yintp = bezier_spline(x,y,xintp, degree=3)

    plt.plot(x,y, "ro")
    plt.plot(xintp,yintp,"k-")

    yintp = bezier_spline(x,y,xintp, degree=2)
    plt.plot(xintp, yintp,"k--")

    plt.show()