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

# def spline_interpolation(xknot, yknot, x, K0=0, Kn=0, degree=3):
#     """
#     Spline interpolation of node values.

#     In the xknot and yknot, we added edges (top and bottom of the atmosphere)
#     in order to get the correct smooth extrapolation.
#     """

#     # in a single node case, we return constant value
#     if len(xknot)==3:
#         return np.ones(len(x), dtype=np.float64) * yknot[1]

#     if len(xknot)-3<degree:
#         degree = 2

#     tck = splrep(xknot, yknot, k=degree)
#     y = splev(x, tck, der=0, ext=0)

#     n = yknot[1] - K0*xknot[1]
#     y[x<xknot[1]] = K0*x[x<xknot[1]] + n
    
#     n = yknot[-2] - Kn*xknot[-2]
#     y[x>xknot[-2]] = Kn*x[x>xknot[-2]] + n

#     return y

def get_K0_Kn(x, y, tension=0):
    def get_Cs(DEL1, DEL2, tension):
        """
        DEL1 = x[1] - x[0]
        DEL2 = x[2] - x[0]
        """
        COSHM1 = np.cosh(tension*DEL1) - 1
        COSHM2 = np.cosh(tension*DEL2) - 1

        DELP = tension/2*(DEL2+DEL1)
        DELM = tension/2*(DEL2-DEL1)

        SINHMP = np.sinh(DELP)/DELP - 1
        SINHMM = np.sinh(DELM)/DELM -  1

        DENOM = COSHM1*(DEL2-DEL1) - 2*DEL1*DELP*DELM*(1+SINHMP)*(1+SINHMM)

        C1 = 2*DELP*DELM*(1+SINHMP)*(1+SINHMM)/DENOM
        C2 = -COSHM2/DENOM
        C3 = COSHM1/DENOM

        return C1, C2, C3
    
    # based on node positions, determine the first derivative @ boundary points
    N = len(x)

    # constant value
    if N==1:
        return 0, 0

    # linear interpolation
    if N==2:
        K0 = (y[1]-y[0])/(x[1]-x[0])
        return K0, K0

    if tension==0:
        # derivative at the first node
        Y = np.array([y[0],y[1],y[2]])
        X = np.array([[x[0]**2, x[0], 1],
                      [x[1]**2, x[1], 1],
                      [x[2]**2, x[2], 1]])
        C = np.linalg.inv(X).dot(Y)
        K0 = 2*C[0]*x[0] + C[1]

        # derivative at the last node
        Y = np.array([y[-1],y[-2],y[-3]])
        X = np.array([[x[-1]**2, x[-1], 1],
                      [x[-2]**2, x[-2], 1],
                      [x[-3]**2, x[-3], 1]])
        C = np.linalg.inv(X).dot(Y)
        Kn = 2*C[0]*x[-1] + C[1]

        return K0, Kn

    tension *= (N-1)/(x[-1]-x[0])

    DEL1 = x[1]-x[0]
    DEL2 = x[2]-x[0]
    C1, C2, C3 = get_Cs(DEL1, DEL2, tension)
    K0 = C1*y[0] + C2*y[1] + C3*y[2]

    DEL1 = x[-1]-x[-2]
    DEL2 = x[-1]-x[-3]
    C1, C2, C3 = get_Cs(-DEL1, -DEL2, tension)
    Kn = C1*y[-1] + C2*y[-2] + C3*y[-3]

    return K0, Kn

def spline_interpolation(x, y, xintp, tension=0, K0=0, Kn=0):
    def get_ABCD(x1, x2, x):
            A = (x2 - x)/(x2 - x1)
            B = 1-A#(x - x1)/(x2 - x1)
            C = 1/6*(A**3 - A)*(x2 - x1)**2
            D = 1/6*(B**3 - B)*(x2 - x1)**2

            return A,B,C,D

    # if K0 and Kn are None we set y'' @ x0 and @ xn to 0 (natural cubic spline)
    N = len(x)

    # constant interpolation
    if N==1:
        return np.ones(len(xintp), dtype=np.float64) * y

    # linear interpolation
    if N==2:
        return K0*(xintp - x[0]) + y[0]

    if tension==0:
        Y = np.zeros(N)
        X = np.zeros((N,N))
        for i in range(1,N-1):
            Y[i] = (y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i] - x[i-1])
            X[i,i-1] = (x[i] - x[i-1])/6
            X[i,i] = (x[i+1] - x[i-1])/3
            X[i,i+1] = (x[i+1] - x[i])/6
        
        # fill with boundary values
        X[0,0] = -1/3*(x[1]-x[0])
        X[0,1] = -1/6*(x[1]-x[0])
        X[-1,-2] = 1/6*(x[-1] - x[-2])
        X[-1,-1] = 1/3*(x[-1] - x[-2])
        
        if (K0 is not None) and (Kn is not None):
            # get the RHS at boundaries
            Y[0] = K0 -(y[1]-y[0])/(x[1]-x[0])
            Y[-1] = Kn - (y[-1]-y[-2])/(x[-1]-x[-2])
            
            # get second derivatives of interpolation polynom
            Der = np.linalg.inv(X).dot(Y)
        else:
            Der = np.zeros(N)
            # get second derivatives of interpolation polynom (cut off boundaries since those are == 0)
            Der[1:-1] = np.linalg.inv(X[1:-1, 1:-1]).dot(Y[1:-1])

        # interpolate
        M = len(xintp)
        yintp = np.zeros(M)
        for i in range(N-1):
            for j in range(M):
                if (x[i]<=xintp[j]) and (x[i+1]>=xintp[j]):
                    A, B, C, D = get_ABCD(x[i], x[i+1], xintp[j])
                    yintp[j] = A*y[i] + B*y[i+1] + C*Der[i] + D*Der[i+1]

        # if (K0 is None) and (Kn is None):
        #     K0 = 1
        #     Kn = 1

        # top extrapolation
        yintp[xintp<x[0]] = K0*(xintp[xintp<x[0]] - x[0]) + y[0]

        # bottom extrapolation
        ind = np.argmin(np.abs(xintp-x[-1]))-1
        Kn = (yintp[ind]-yintp[ind-1])/(xintp[ind]-xintp[ind-1])
        n = yintp[ind] - Kn*xintp[ind]
        yintp[ind:] = Kn*xintp[ind:] + n
        
        # yintp[xintp>x[-1]] = Kn*(xintp[xintp>x[-1]] - x[-1]) + y[-1]

        return yintp

    tension *= (N-1)/(x[-1]-x[0])

    Y = np.zeros(N)
    X = np.zeros((N,N))
    for i in range(1,N-1):
        Y[i] = (y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1])
        X[i,i-1] = 1/(x[i]-x[i-1]) - tension/np.sinh(tension*(x[i]-x[i-1]))
        X[i,i] = tension*np.cosh(tension*(x[i]-x[i-1]))/np.sinh(tension*(x[i]-x[i-1])) - 1/(x[i] - x[i-1])
        X[i,i] += tension*np.cosh(tension*(x[i+1]-x[i]))/np.sinh(tension*(x[i+1]-x[i])) - 1/(x[i+1] - x[i])
        X[i,i+1] = 1/(x[i+1]-x[i]) - tension/np.sinh(tension*(x[i+1]-x[i]))
    X[0,0] = tension*np.cosh(tension*(x[1]-x[0]))/np.sinh(tension*(x[1]-x[0])) - 1/(x[1]-x[0])
    X[0,1] = 1/(x[1]-x[0]) - tension/np.sinh(tension*(x[1]-x[0]))
    X[-1,-2] = 1/(x[-1]-x[-2]) - tension/np.sinh(tension*(x[-1]-x[-2]))
    X[-1,-1] = tension*np.cosh(tension*(x[-1]-x[-2]))/np.sinh(tension*(x[-1]-x[-2])) - 1/(x[-1]-x[-2])

    Y[0] = (y[1]-y[0])/(x[1]-x[0]) - K0
    Y[-1] = Kn - (y[-1]-y[-2])/(x[-1]-x[-2])

    # get the second derivatives in nodes
    # these are acctually y''/tension^2
    Der = np.linalg.inv(X).dot(Y)

    # interpolate
    M = len(xintp)
    yintp = np.zeros(M)
    for i in range(N-1):
        for j in range(M):
            if (x[i]<=xintp[j]) and (x[i+1]>=xintp[j]):
                A = np.sinh(tension*(x[i+1]-xintp[j]))/np.sinh(tension*(x[i+1]-x[i]))
                B = (x[i+1]-xintp[j])/(x[i+1]-x[i])
                C = np.sinh(tension*(xintp[j]-x[i]))/np.sinh(tension*(x[i+1]-x[i]))
                D = (xintp[j] - x[i])/(x[i+1]-x[i])
                yintp[j] = Der[i]*A + (y[i] - Der[i])*B + Der[i+1]*C + (y[i+1] - Der[i+1])*D

    # top extrapolation
    yintp[xintp<x[0]] = K0*(xintp[xintp<x[0]] - x[0]) + y[0]

    # bottom extrapolation
    ind = np.argmin(np.abs(xintp-x[-1]))-1
    Kn = (yintp[ind]-yintp[ind-1])/(xintp[ind]-xintp[ind-1])
    n = yintp[ind] - Kn*xintp[ind]
    yintp[ind:] = Kn*xintp[ind:] + n

    # yintp[xintp>x[-1]] = Kn*(xintp[xintp>x[-1]] - x[-1]) + y[-1]

    return yintp

if __name__=="__main__":
    # example from de la Cruz Rodriguez et al. (2019)
    x = np.array([-3,-2,-1.95, -1, 0.4, 2, 3.2])
    y = np.array([0.2, 0, 0.6, 0.55, 0.29, 0.21, 0.4])
    x = np.array([-3,-2,-1, 0.4, 2, 3.2])
    y = np.array([0.2, 0, 0.55, 0.29, 0.21, 0.4])

    # x = np.random.random(5)
    # x.sort()
    # y = np.random.random(5)
    
    plt.scatter(x, y)

    xintp = np.linspace(x[0]-0.2,x[-1]+0.2, num=201)

    tension = 5
    K0, Kn = get_K0_Kn(x, y, tension=tension)
    yintp_ = spline_interpolation(x, y, xintp, tension=tension, K0=K0, Kn=Kn)
    plt.plot(xintp, yintp_, label="Cubic spline (s!=0)")

    tension = 0
    K0, Kn = get_K0_Kn(x, y, tension=tension)
    yintp_ = spline_interpolation(x, y, xintp, tension=tension, K0=K0, Kn=Kn)
    plt.plot(xintp, yintp_, label="Cubic spline (s=0)")

    yintp = bezier_spline(x, y, xintp, K0=K0, Kn=Kn, degree=3, extrapolate=True)
    plt.plot(xintp, yintp, label="Cubic Bezier")   
    
    plt.legend()

    plt.show()
    
    # plt.plot(x, y, "ro", label="knots")
    # plt.plot(xintp, yintp,"k-", label="Bezier-3")

    # yintp = spline_interpolation(x, y, xintp, degree=3)

    # plt.plot(xintp, yintp,"k--", label="spline")

    # plt.xlabel("x")
    # plt.ylabel("y")

    # plt.legend()

    # plt.show()