def add_colorbar(fig, ax, im, label=None):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax,
                       width="5%",
                       height="100%",
                       loc="lower left",
                       bbox_to_anchor=(1.02, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins)
    # cbar.ax.set_yticklabels(cbar.ax.get_yticks(), fontsize="medium")
    if label is not None:
        cbar.set_label(label)