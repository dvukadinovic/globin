import numpy as np
import matplotlib.pyplot as plt

from .tools import create_fig, add_colorbar

def parameter_correlation(RF, parameters, idx=0, idy=0, logtau_top=-6, logtau_bot=1, norm=False, nodes=None):
	"""
	RF : globin.input.RF() object

	"""
	npar = len(parameters)
	nx, ny, _, nz, nw, ns = RF.rf.shape

	if npar==1:
		raise ValueError("We can not correlate one parameter.")

	ind_top = np.argmin(np.abs(RF.logtau - logtau_top))
	ind_bot = np.argmin(np.abs(RF.logtau - logtau_bot))+1

	logtau = RF.logtau[ind_top:ind_bot]
	
	fig, axs = create_fig(nrows=npar-1, ncols=npar-1, hspace=0.35)

	for idr in range(npar-1):
		p1_RF = RF.get_par_rf(parameters[idr])[idx,idy]
		p1_RF = p1_RF.reshape(nz, nw*ns, order="F")
		p1_RF = p1_RF[ind_top:ind_bot]
		for idc in range(idr+1, npar):
			p2_RF = RF.get_par_rf(parameters[idc])[idx,idy]
			p2_RF = p2_RF.reshape(nz, nw*ns, order="F")
			p2_RF = p2_RF[ind_top:ind_bot]

			C = p1_RF.dot(p2_RF.T)
			C /= np.max(np.abs(C))

			vlim = np.max(np.abs(C))

			axs[idc-1,idr].set_title(f"{parameters[idr]} vs {parameters[idc]}")
			im = axs[idc-1,idr].imshow(C[:,:], origin="lower", 
				cmap="bwr",
				vmin=-vlim, vmax=vlim,
				extent=[logtau[-1], logtau[0], logtau[-1], logtau[0]])
			add_colorbar(fig, axs[idc-1,idr], im)

			if nodes is not None:
				for node in nodes:
					ind = np.argmin(np.abs(logtau - node))
					axs[idc-1,idr].axhline(y=logtau[ind], c="k", lw=0.7, alpha=0.5)
					axs[idc-1,idr].axvline(x=logtau[ind], c="k", lw=0.7, alpha=0.5)

		# kill not used axises
		for idc in range(idr):
			axs[idc,idr].axis("off")