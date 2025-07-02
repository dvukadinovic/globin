import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from .tools import add_colorbar
from .tools import MidpointNormalizeFair

alphabet = list(map(chr, range(97, 123)))

cmaps = {"temp"  : "plasma", 
         "vz"    : "bwr_r", 
         "vmic"  : "plasma",
         "mag"   : "nipy_spectral", 
         "gamma" : "nipy_spectral",
         "chi"   : "nipy_spectral",
         "stray" : "nipy_spectral",
         "sl_temp" : "plasma",
         "sl_vz" : "bwr_r",
         "sl_vmic" : "plasma"}

parameter_relay_text = {"temp"  : "Temperature [K]",
                        "vz"    : "LOS velocity [km/s]",
                        "vmic"  : "Micro-turbulent velocity [km/s]",
                        "mag"   : "Magnetic field [G]",
                        "gamma" : r"Inclination [$^\circ$]",
                        "chi"   : r"Azimuth [$^\circ$]",
                        "stray" : "stray light factor",
                        "sl_temp" : "magnetized Temperature [K]",
                        "sl_vz" : "magnetized LOS velocity [km/s]",
                        "sl_vmic" : "magnetized Micro-turbulent velocity [km/s]"}

parameter_relay_symbols = {"temp"    : r"$T [\mathrm{K}]$",
                           "vz"      : r"$v_\mathrm{LOS} [\mathrm{km/s}]$",
                           "vmic"    : r"$v_\mathrm{mic} [\mathrm{km/s}]$",
                           "mag"     : r"$B [\mathrm{G}]$",
                           "gamma"   : r"$\gamma [^\circ]$",
                           "chi"     : r"$\phi   [^\circ]$",
                           "stray"   : r"$\alpha$",
                           "sl_temp" : r"$T^{m} [\mathrm{K}]$",
                           "sl_vz"   : r"$v_\mathrm{LOS}^m [\mathrm{km/s}]$",
                           "sl_vmic" : r"$v_\mathrm{vmic}^m [\mathrm{km/s}]$"
                           }

bands = {"temp"  : [100, 200],
         "vz"    : [0.2, 0.5],
         "vmic"  : [0.2, 0.5],
         "mag"   : [100, 250],
         "stray" : [0.01, 0.05],
         "sl_temp" : [100, 200],
         "sl_vz" : [0.2, 0.5],
         "sl_vmic" : [0.2, 0.5]}

def lin(x, a, b):
    return x*a + b

def scatter_plots(atm1, atm2, parameters=["temp"], weight=None, labels=["referent", "inversion"], statistics=False, subplot_markers=False, show_bands=False, show_errors=False, mark=None):
    nrows = 0
    _parameters = []
    for parameter in parameters:
        n1 = len(atm1.nodes[parameter])
        n2 = len(atm2.nodes[parameter])
        nrows = max([nrows, n1, n2])
        _parameters.append(parameter)
   
    parameter_relay = parameter_relay_symbols

    ncols = len(_parameters)
    width, height = 3, 2 + 2/3
    fig = plt.figure(figsize=(width*ncols, height*nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

    mark_markers = False
    if mark is not None:
        mark_markers = True
        if not isinstance(mark, list):
            mark = [mark]
        nconditions = len(mark)
        inds = np.arange(atm1.nx*atm1.ny)
        mark_inds = [None]*len(mark)
        total_condition = True
        for idc, condition in enumerate(mark):
            mark_inds[idc] = inds[condition]
            total_condition &= condition
        donot_mark_inds = inds[np.logical_not(total_condition)] 
    
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
            if show_errors:
                try:
                    xerr = atm1.errors[parameter][...,idr].ravel()
                except:
                    xerr = None
                try:
                    yerr = atm2.errors[parameter][...,idr].ravel()
                except:
                    yerr = None

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

            if show_errors:
                ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", markeredgecolor="k", ecolor="k", alpha=0.7, fillstyle="none")
            else:
                if mark_markers:
                    ax.scatter(x[donot_mark_inds], y[donot_mark_inds], s=9, edgecolor="k", facecolor="none", alpha=0.7)
                    for idcon in range(nconditions):
                        ax.scatter(x[mark_inds[idcon]], y[mark_inds[idcon]], s=9, edgecolor=f"C{idcon}", facecolor="none", alpha=0.7)
                else:
                    ax.scatter(x, y, s=ms, edgecolor="k", facecolor="none", alpha=0.7)
            mean = np.nanmean(x)
            std = np.nanstd(x)
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
            if show_bands:
                x = np.array([vmin, vmax]) 
                ax.fill_between(x, x-bands[parameter][0], x+bands[parameter][0], alpha=0.3, color="tab:green")
                ax.fill_between(x, x-bands[parameter][1], x+bands[parameter][1], alpha=0.3, color="tab:orange")

            # if statistics:
            #     x = np.array([vmin, vmax])
            #     ax.plot(x, lin(x, *par), c="tab:purple", lw=1)

            idsp += 1

    # set global labels
    fig.supxlabel(f"{labels[0]}", ha="center", fontsize="large")
    fig.supylabel(f"{labels[1]}", va="center", fontsize="large")

    fig.tight_layout()
    plt.show()

def imshow_plots(atm1, atm2=None, parameters=["temp"], labels=["reference", "inversion"], boundaries={}, contrast=3, fontsize=15, parameters_titles="text", wspace=0.3, aspect=4/3, grid=False, show_errors=False, show_axis_ticks=True):
    mpl.rcParams.update({"font.size" : fontsize})

    if parameters_titles=="text":
        parameter_relay = parameter_relay_text
    elif parameters_titles=="symbols":
        parameter_relay = parameter_relay_symbols
    else:
        raise ValueError(f"Unrecognized title for parameters '{parameters_titles}'. Only 'text' and 'symbols' are supported.")

    N = 1
    n2 = 0
    if atm2 is not None:
        if not isinstance(atm2, list):
            atm2 = [atm2]
        N += len(atm2)

    nrows = 1
    _parameters = []
    for parameter in parameters:
        nmax = len(atm1.nodes[parameter])
        if atm2 is not None:
            for _atm in atm2:
                n2 = len(_atm.nodes[parameter])
                nmax = max([nmax, n2])
        nrows = max([nrows, nmax])
        _parameters.append(parameter)

    ncols = len(_parameters)
    height = 3
    width = height*aspect
    width, height = mpl.figure.figaspect(aspect)
    fig = plt.figure(figsize=(N*width*ncols, height*nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=N*ncols, wspace=wspace, hspace=0.1)

    for idc in range(ncols):
        parameter = _parameters[idc]
        nnodes = len(atm1.nodes[parameter])
        for idr in range(nnodes):
            ax = fig.add_subplot(gs[idr,N*idc])
            
            fact = 1
            if parameter in ["gamma", "chi"]:
                fact = 180/np.pi

            if show_errors:
                x = atm1.errors[parameter][...,idr].copy()
            else:
                x = atm1.values[parameter][:,:,idr].copy()
            x *= fact
            mean = np.nanmean(x)
            std = np.nanstd(x)
            vmin = mean - contrast*std
            vmax = mean + contrast*std
            norm = None
            if parameter=="mag":
                if vmin<10:
                    vmin = 9
            if parameter in ["vmic", "sl_vmic"]:
                if vmin<1e-3:
                    vmin = 1e-3
            if parameter in ["vz", "sl_vz"]:
                vmax = np.max([np.abs(vmin), np.abs(vmax)])
                vmin = -vmax
                # norm = MidpointNormalizeFair(vmin=vmin, vmax=vmax, midpoint=0)
                # vmin, vmax = None, None
            if show_errors:
                vmin = 0

            if idr==0:
                ax.set_title(labels[0], fontsize="large")

            # label for color bars
            # cblabel_1 = r"$\log\tau = {:3.2f}$".format(atm1.nodes[parameter][idr])
            # if atm2 is not None:
            #     cblabel_2 = cblabel_1
            #     cblabel_1 = None

            if parameter in boundaries:
                vmin = boundaries[parameter][0]
                vmax = boundaries[parameter][1]

            im = ax.imshow(x.T, origin="lower", vmin=vmin, vmax=vmax, norm=norm, cmap=cmaps[parameter])
            add_colorbar(fig, ax, im)
            if nnodes>1:
                ax.set_ylabel(r"$\log\tau = {:3.2f}$".format(atm1.nodes[parameter][idr]))
            # if idc==(ncols-1):
            #     add_colorbar(fig, ax, im, label=cblabel_1)
            # else:

            # if idr+1!=nnodes:
            #     ax.set_xticklabels([])
            # if idc!=0:
            #     ax.set_yticklabels([])

            # add the second axis if we have two atmospheres
            if atm2 is not None:
                for ida, _atm in enumerate(atm2):
                    ax2 = fig.add_subplot(gs[idr,N*idc+ida+1])
                    
                    # set titles
                    if show_errors:
                        y = _atm.errors[parameter][...,idr].copy()
                    else:
                        y = _atm.values[parameter][:,:,idr].copy()
                    y *= fact

                    if idr==0:
                        ax2.set_title(labels[1+ida], fontsize="large")
                    im = ax2.imshow(y.T, origin="lower", vmin=vmin, vmax=vmax, norm=norm, cmap=cmaps[parameter])
                    add_colorbar(fig, ax2, im)
                    # if idc==(ncols-1):
                    #     add_colorbar(fig, ax2, im, label=cblabel_2)
                    # else:
                    
                    # if idr+1!=nnodes:
                    #     ax2.set_xticklabels([])
                    # ax2.set_yticklabels([])

                    # ax2.set_xticks([])
                    # ax2.set_xticklabels([])
                    # ax2.set_yticks([])
                    # ax2.set_yticklabels([])

                    if not show_axis_ticks:
                        ax2.set_xticks([])
                        ax2.set_xticklabels([])
                        ax2.set_yticks([])
                        ax2.set_yticklabels([])

            if grid:
                ax.grid(which="major", axis="both", lw=0.75, color="gray")
                ax2.grid(which="major", axis="both", lw=0.75, color="gray")

            if not show_axis_ticks:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])
                
        # add parameter wise titles (spanning N columns)
        y0 = 0.91 if nrows>1 else 0.95
        x0 = (ax.get_position().x1 + ax.get_position().x0)/2
        if N==2:
            x0 = (ax2.get_position().x1 + ax.get_position().x0)/2

        fig.text(x0, y0, parameter_relay[parameter],
            fontsize="large",
            ha="center",
            va="center",
            transform=fig.transFigure)
