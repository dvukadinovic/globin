import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from .tools import add_colorbar

alphabet = list(map(chr, range(97, 123)))

def lin(x, a, b):
    return x*a + b

def scatter_plots(atm1, atm2, parameters=["temp"], weight=None, labels=["referent", "inversion"], statistics=False, subplot_markers=False):
    parameter_relay = {"temp"  : "Temperature [K]",
                       "vz"    : "LOS velocity [km/s]",
                       "vmic"  : "Micro-turbulent velocity [km/s]",
                       "mag"   : "Magnetic field [G]",
                       "gamma" : r"Inclination [$^\circ$]",
                       "chi"   : r"Azimuth [$^\circ$]",
                       "stray" : "stray light factor"}

    nrows = 0
    _parameters = []
    for parameter in parameters:
        n1 = len(atm1.nodes[parameter])
        n2 = len(atm2.nodes[parameter])
        nrows = max([nrows, n1, n2])
        _parameters.append(parameter)
    
    ncols = len(_parameters)
    width, height = 3, 2 + 2/3
    fig = plt.figure(figsize=(width*ncols, height*nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

    # change the marker size based on the weights provided (chi2 values,...)
    ms = 9
    if weight is not None:
        k = (100 - 5) / (np.max(weight) - np.min(weight))
        n = 5 - k*np.min(weight)
        ms = k*weight + n

    idsp = 0
    for idc in range(ncols):
        parameter = _parameters[idc]
        nnodes = len(atm1.nodes[parameter])
        for idr in range(nnodes):

            ax = fig.add_subplot(gs[idr,idc])
            x = atm1.values[parameter][:,:,idr].ravel()
            y = atm2.values[parameter][:,:,idr].ravel()

            if statistics:
                R = pearsonr(x, y)
                # par, cov = curve_fit(lin, x, y, p0=[1,0])
                t = ax.text(0.95, 0.05, r"$R={:3.2f}$".format(R.statistic),
                        fontsize="x-small", 
                        horizontalalignment="right",
                        verticalalignment="bottom",
                        transform=ax.transAxes)
                t.set_bbox(dict(facecolor="white", alpha=0.9, linewidth=0))

            if subplot_markers:
                t = ax.text(0.05, 0.95, alphabet[idsp],
                        weight="bold",
                        fontsize="large",
                        horizontalalignment="left",
                        verticalalignment="top",
                        transform=ax.transAxes)
                t.set_bbox(dict(facecolor="white", alpha=0.9, linewidth=0))

            # set parameters at top of the columns
            if idr==0:
                ax.set_title(f"{parameter_relay[parameter]}")

            # set log(tau) values as y-label on the right side of last column
            if idc==(ncols-1):
                ax.set_ylabel(r"$\log\tau = {:3.2f}$".format(atm2.nodes[parameter][idr]))
                ax.yaxis.set_label_position("right")

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

            # if statistics:
            #     x = np.array([vmin, vmax])
            #     ax.plot(x, lin(x, *par), c="tab:purple", lw=1)

            idsp += 1

    # set global labels
    fig.supxlabel(f"{labels[0]}", ha="center", fontsize="large")
    fig.supylabel(f"{labels[1]}", va="center", fontsize="large")

    fig.tight_layout()
    plt.show()

def imshow_plots(atm1, atm2=None, parameters=["temp"], labels=["reference", "inversion"], contrast=3, fontsize=15, aspect=4/3, grid=False):
    cmaps = {"temp"  : "plasma", 
             "vz"    : "bwr_r", 
             "vmic"  : "plasma",
             "mag"   : "nipy_spectral", 
             "gamma" : "nipy_spectral",
             "stray" : "nipy_spectral"}

    parameter_relay = {"temp"  : "Temperature [K]",
                       "vz"    : "LOS velocity [km/s]",
                       "vmic"  : "Micro-turbulent velocity [km/s]",
                       "mag"   : "Magnetic field [G]",
                       "gamma" : r"Inclination [$^\circ$]",
                       "chi"   : r"Azimuth [$^\circ$]",
                       "stray" : "stray light factor"}

    mpl.rcParams.update({"font.size" : fontsize})

    N = 1
    n2 = 0
    if atm2 is not None:
        N = 2

    nrows = 0
    _parameters = []
    for parameter in parameters:
        n1 = len(atm1.nodes[parameter])
        if atm2 is not None:
            n2 = len(atm2.nodes[parameter])
        nrows = max([nrows, n1, n2])
        _parameters.append(parameter)

    ncols = len(_parameters)
    height = 3
    width = height*aspect
    fig = plt.figure(figsize=(N*width*ncols, height*nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=N*ncols, wspace=0.5, hspace=0.1)

    for idc in range(ncols):
        parameter = _parameters[idc]
        nnodes = len(atm1.nodes[parameter])
        for idr in range(nnodes):
            ax = fig.add_subplot(gs[idr,N*idc])
            
            x = atm1.values[parameter][:,:,idr]
            mean = np.mean(x)
            std = np.std(x)
            vmin = mean - contrast*std
            vmax = mean + contrast*std
            if parameter=="mag":
                if vmin<10:
                    vmin = 9
            if parameter=="vmic":
                if vmin<1e-3:
                    vmin = 1e-3
            if parameter=="vz":
                vmax = np.max([np.abs(vmin), np.abs(vmax)])
                vmin = -vmax

            if idr==0:
                ax.set_title(labels[0], fontsize="large")

            # label for color bars
            cblabel_1 = r"$\log\tau = {:3.2f}$".format(atm1.nodes[parameter][idr])
            if atm2 is not None:
                cblabel_2 = cblabel_1
                cblabel_1 = None
            
            im = ax.imshow(x.T, origin="lower", vmin=vmin, vmax=vmax, cmap=cmaps[parameter])
            if idc==(ncols-1):
                add_colorbar(fig, ax, im, label=cblabel_1)
            else:
                add_colorbar(fig, ax, im)

            # if idr+1!=nnodes:
            #     ax.set_xticklabels([])
            # if idc!=0:
            #     ax.set_yticklabels([])

            # add the second axis if we have two atmospheres
            if atm2 is not None:
                ax2 = fig.add_subplot(gs[idr,2*idc+1])
                
                # set titles
                y = atm2.values[parameter][:,:,idr]
                
                if idr==0:
                    ax2.set_title(labels[1], fontsize="large")
                im = ax2.imshow(y.T, origin="lower", vmin=vmin, vmax=vmax, cmap=cmaps[parameter])
                if idc==(ncols-1):
                    add_colorbar(fig, ax2, im, label=cblabel_2)
                else:
                    add_colorbar(fig, ax2, im)
                
                # if idr+1!=nnodes:
                #     ax2.set_xticklabels([])
                # ax2.set_yticklabels([])

                # ax2.set_xticks([])
                # ax2.set_xticklabels([])
                # ax2.set_yticks([])
                # ax2.set_yticklabels([])

            if grid:
                ax.grid(which="major", axis="both", lw=0.75, color="gray")
                ax2.grid(which="major", axis="both", lw=0.75, color="gray")

            # ax.set_xticks([])
            # ax.set_xticklabels([])
            # ax.set_yticks([])
            # ax.set_yticklabels([])

        # add parameter wise titles (spanning N columns)
        y0 = 0.91
        x0 = (ax.get_position().x1 + ax.get_position().x0)/2
        if N==2:
            x0 = (ax2.get_position().x1 + ax.get_position().x0)/2

        fig.text(x0, y0, parameter_relay[parameter],
            fontsize="large",
            ha="center",
            va="center",
            transform=fig.transFigure)