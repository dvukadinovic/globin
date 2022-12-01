import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .tools import add_colorbar

def lin(x, a, b):
    return x*a + b

def scatter_plots(atm1, atm2, parameters=["temp"], weight=None, labels=["referent", "inversion"], statistics=False):
    # print(list(atm1.nodes.keys()))
    # if parameters not in list(atm1.nodes.keys()):
    #     print("Not all parameters present in atm1.")
    #     return
    # if parameters not in list(atm2.nodes.keys()):
    #     print("Not all parameters present in atm2.")
    #     return
    
    nrows = 0
    for parameter in parameters:
        n1 = len(atm1.nodes[parameter])
        n2 = len(atm2.nodes[parameter])
        nrows = max([nrows, n1, n2])
    ncols = len(parameters)
    width, height = 3, 2 + 2/3
    fig = plt.figure(figsize=(width*ncols, height*nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

    # change the marker size based on the weights provided (chi2 values,...)
    ms = 9
    if weight is not None:
        k = (100 - 5) / (np.max(weight) - np.min(weight))
        n = 5 - k*np.min(weight)
        ms = k*weight + n

    for idc in range(ncols):
        parameter = parameters[idc]
        nnodes = len(atm1.nodes[parameter])
        for idr in range(nnodes):
            ax = fig.add_subplot(gs[idr,idc])
            x = atm1.values[parameter][:,:,idr].ravel()
            y = atm2.values[parameter][:,:,idr].ravel()

            # set titles
            if idr==0:
                ax.set_title("{:s} @ {:3.2f}".format(parameter, atm1.nodes[parameter][idr]))
            else:
                ax.set_title("@ {:3.2f}".format(atm2.nodes[parameter][idr]))

            # set axis labels
            if idr+1==nnodes:
                ax.set_xlabel(labels[0])
            if idc==0:
                ax.set_ylabel(labels[1])

            ax.scatter(x, y, s=ms, edgecolor="k", facecolor="none", alpha=0.7)
            mean = np.mean(x)
            std = np.std(x)
            vmin = mean - 3*std
            vmax = mean + 3*std
            if parameter=="mag":
                if vmin<10:
                    vmin = 9
            if parameter=="vmic":
                if vmin<1e-3:
                    vmin = 1e-3
    #         vmin = np.min([x.min(), y.min()])
    #         vmax = np.max([x.max(), y.max()])
            vmin *= 0.99
            vmax *= 1.01
            ax.plot([vmin, vmax], [vmin, vmax], c="tab:red")
            ax.set_xlim([vmin, vmax])
            ax.set_ylim([vmin, vmax])

            if statistics:
                par, cov = curve_fit(lin, x, y, p0=[1,0])
                print("{:6s}  {:5.4f}  {:>+5.1f}".format(parameter, *par))

    fig.tight_layout()
    plt.show()

def imshow_plots(atm1, atm2=None, parameters=["temp"], labels=["reference", "inversion"]):
    cmaps = {"temp" : "plasma", "mag" : "nipy_spectral", "vz" : "bwr_r", "vmic" : "plasma"}
    
    N = 1
    n2 = 0
    if atm2 is not None:
        N = 2

    nrows = 0
    for parameter in parameters:
        n1 = len(atm1.nodes[parameter])
        if atm2 is not None:
            n2 = len(atm2.nodes[parameter])
        nrows = max([nrows, n1, n2])
    ncols = len(parameters)
    width, height = 3, 2 + 2/3
    fig = plt.figure(figsize=(N*width*ncols, height*nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=N*ncols, wspace=0.5, hspace=0.1)

    for idc in range(ncols):
        parameter = parameters[idc]
        nnodes = len(atm1.nodes[parameter])
        for idr in range(nnodes):
            ax = fig.add_subplot(gs[idr,N*idc])
            
            # set titles
            if idr==0:
                ax.set_title("{:s} \n {:s} @ {:3.2f}".format(labels[0], parameter, atm1.nodes[parameter][idr]))
            else:
                ax.set_title("@ {:3.2f}".format(atm1.nodes[parameter][idr]))

            x = atm1.values[parameter][:,:,idr]
            mean = np.mean(x)
            std = np.std(x)
            vmin = mean - 3*std
            vmax = mean + 3*std
            if parameter=="mag":
                if vmin<10:
                    vmin = 9
            if parameter=="vmic":
                if vmin<1e-3:
                    vmin = 1e-3
            if parameter=="vz":
                vmax = np.max([np.abs(vmin), np.abs(vmax)])
                vmin = -vmax

            im = ax.imshow(x, origin="lower", vmin=vmin, vmax=vmax, cmap=cmaps[parameter])
            add_colorbar(fig, ax, im)
            if idr+1!=nnodes:
                ax.set_xticklabels([])
            if idc!=0:
                ax.set_yticklabels([])

            if atm2 is not None:
                ax = fig.add_subplot(gs[idr,2*idc+1])
                # set titles
                if idr==0:
                    ax.set_title(labels[1])
                y = atm2.values[parameter][:,:,idr]
                im = ax.imshow(y, origin="lower", vmin=vmin, vmax=vmax, cmap=cmaps[parameter])
                add_colorbar(fig, ax, im)
                if idr+1!=nnodes:
                    ax.set_xticklabels([])
                ax.set_yticklabels([])

    plt.show()