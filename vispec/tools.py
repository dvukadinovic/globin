import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

WIDTH = 3
HEIGHT = 2 + 2/3

def add_colorbar(fig, ax, im, label=None):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax,
                       width="5%",
                       height="100%",
                       loc="lower left",
                       bbox_to_anchor=(1.00, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins)
    # cbar.ax.set_yticklabels(cbar.ax.get_yticks(), fontsize="medium")
    if label is not None:
        cbar.set_label(label)
    return cbar

def create_fig(nrows=1, ncols=1, figsize=None, aspect="auto", hspace=0.3, wspace=0.3):
    if figsize is None:
        if aspect=="equal":
            figsize = (WIDTH*ncols, WIDTH*nrows)
        figsize = (WIDTH*ncols, HEIGHT*nrows)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, 
                          hspace=hspace, wspace=wspace)

    axs = []
    for idx in range(nrows):
        for idy in range(ncols):
            axs.append(fig.add_subplot(gs[idx,idy]))

    axs = np.asarray(axs)
    axs = axs.reshape(nrows, ncols)

    return fig, axs

class MidpointNormalizeFair(mpl.colors.Normalize):
    """ From: https://matplotlib.org/users/colormapnorms.html"""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...

        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)

        vlargest = max( abs( self.vmax - self.midpoint ), abs( self.vmin - self.midpoint ) )
        x, y = [ self.midpoint - vlargest, self.midpoint, self.midpoint + vlargest], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))